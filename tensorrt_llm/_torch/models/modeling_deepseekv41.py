# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 modeling: the delta from DeepSeek-V4.

Everything V4.1 shares with V4 -- MLA with a compressed long-range branch, the
sparse indexer, the mHC residual stream, the MoE stack, Engram's n-gram lookup,
MTP/DSpark -- is inherited from ``modeling_deepseekv4``. This module holds only
what the two checkpoints genuinely disagree about. Each difference is stated
against the reference implementation shipped inside the checkpoint
(``<checkpoint>/inference/model.py``, cited by line below), because several of
them look like the V4 code would just work:

1. **RoPE per layer.** V4 keys the YaRN long-range branch on ``ratio > 1``. V4.1
   spends ratio 1 on a genuine *unpooled* long-range layer, so the test is
   ``ratio != 0``. Twenty of V4.1's forty layers are ratio 1; under the V4 rule
   every one of them would use base RoPE with the wrong theta.

2. **mHC constants.** V4.1's Sinkhorn kernel uses one ``hc_eps`` (1e-6) for the
   pre-sigmoid, the comb softmax and every Sinkhorn division, and its RMS
   statistic uses ``rms_norm_eps`` = 1e-20 rather than the module default.

3. **Lagged ``pre``.** V4's mHC sublayer consumes the ``pre`` coefficients it
   just computed. V4.1's consumes the *previous* sublayer's and hands its own
   forward (model.py:968-995). This is the reason V4.1 cannot use the fused mHC
   kernels: they collapse the residual with the ``pre`` they compute internally.

4. **No ``hc_head``.** V4.1 closes the stream with the last layer's own
   ``hc_pre`` (model.py:1268) and then a plain norm + head, so the model returns
   an already-collapsed ``[N, hidden]``.

5. **Engram.** V4.1's ``Engram`` is ``embed -> wkv -> gate`` with no depthwise
   convolution, splits ``wkv`` keys-first, computes its gate in fp32, and stores
   its 384-million-row tables as row-sharded fp8. V4's ``Engram`` has a
   ``ShortConv``, splits value-first, and keeps a replicated dense table -- which
   for these tables would be ~197 GiB per rank.

6. **Checkpoint layout.** V4.1's dense weights are fp8 with block-32 ue8m0
   scales, and its module names differ (``ffn.gate.bias_vl``, ``engram.embed``,
   ``engram.wkv``, ``engram.q_weight``/``k_weight``).

Deliberately *not* here, having been checked against the reference and found to
need no code:

- **The MoE gate.** ``DeepSeekV4MoeRoutingMethod`` ignores ``n_group`` /
  ``topk_group`` and delegates to ``torch.ops.trtllm.gate_forward``, whose kernel
  already computes ``sqrt(softplus(logit))``, selects top-k on ``score + bias``,
  gathers weights from the *raw* unbiased scores, sum-normalizes and applies
  ``routed_scaling_factor`` -- V4.1's gate exactly (model.py:793-827). It
  supports 384 experts and the compile-time top-6. V4.1's ``gate_temp`` is 1.0,
  so the kernel's missing temperature division is inert.
- **The Engram hash layout.** TRT-LLM's ``NgramHashMapping`` reproduces the
  reference's bucket primes and multipliers bit-for-bit given the config
  translation in ``configs/deepseek_v41.py``; verified by the derived prime sums
  matching the checkpoint's ``engram_num_embeddings`` (384006168 / 384016682) and
  by the compressed vocab coming out at ``engram_compressed_vocab_size`` (99092).
- **The logits path.** ``DeepseekV4LogitsProcessor`` already accepts a collapsed
  ``[N, hidden]`` when ``hc_head is None``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention.backends.sparse.deepseek_v4.candidate_prefilter import (
    candidate_prefilter_is_on,
    inert_up_to_positions,
)
from ..attention.backends.sparse.deepseek_v4.decoder_replay import (
    enter_decoder_replay,
    exit_decoder_replay,
    gather_replayed_rows,
    plan_decoder_replay,
    scatter_replayed_rows,
)
from ..attention.backends.sparse.deepseek_v4.params import (
    has_compressor_state,
    is_compress_layer,
    pool_factor,
)
from ..configs.deepseek_v41 import (
    DeepseekV41QuantLayout,
    assert_weight_layout,
    decoder_bounded_replay_enabled,
    dense_fp8_requant_enabled,
    engram_cpu_offload_enabled,
    quant_role_for_weight_key,
)
from ..distributed import AllReduce, AllReduceParams, AllReduceStrategy, allgather
from ..model_config import ModelConfig
from ..modules.engram import Engram, ShardedFp8MultiHeadEmbedding
from ..modules.mhc.hyper_connection import HCState, mHC
from ..speculative import SpecMetadata
from .modeling_deepseekv4 import (
    DeepseekV4Attention,
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4WeightLoader,
    _deepseek_v4_layer_compress_ratio,
    _deepseek_v4_pos_embd_params,
    _remap_deepseek_v4_checkpoint_keys,
    weight_dequant,
)
from .modeling_utils import register_auto_model

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# ---------------------------------------------------------------------------
# 1. Per-layer RoPE
# ---------------------------------------------------------------------------


def _deepseek_v41_pos_embd_params(
    config: PretrainedConfig,
    model_config: ModelConfig,
    layer_idx: Optional[int],
    predicted_tokens_per_seq: int = 1,
    *,
    long_range: Optional[bool] = None,
):
    """V4's RoPE selection with V4.1's ratio semantics.

    The reference gates the whole compressed branch -- and with it the
    ``compress_rope_theta`` frequencies -- on ``if self.compress_ratio:``
    (model.py:775), i.e. on the ratio being non-zero, not on it exceeding 1.
    V4's ``> 1`` test is right for V4 only because there a ratio of 1 can only
    arise from the LLM API normalizing a checkpoint 0, meaning "SWA-only". V4.1
    keeps 0 for that and uses 1 for a long-range layer that skips KV *pooling*
    but still attends over a compressed cache and still takes an indexer
    selection.
    """
    ratio = _deepseek_v4_layer_compress_ratio(config, model_config, layer_idx)
    return _deepseek_v4_pos_embd_params(
        config,
        model_config,
        layer_idx,
        predicted_tokens_per_seq,
        long_range=(ratio != 0) if long_range is None else long_range,
    )


class DeepseekV41Attention(DeepseekV4Attention):
    """V4 attention with V4.1's ratio -> RoPE mapping and no per-head Q norm.

    The geometry is shared: V4.1's ``head_dim`` 512, ``qk_rope_head_dim`` 64,
    ``q_lora_rank`` 1280, ``o_lora_rank`` 1024 and ``o_groups`` 8 satisfy the base
    class's asserts once ``kv_lora_rank`` and ``v_head_dim`` are derived as
    ``head_dim - qk_rope_head_dim`` and ``head_dim`` -- see the properties on
    ``DeepseekV41TextConfig``.

    Two things do differ. The ratio -> RoPE mapping, and the query normalization:
    V4 rescales each head of ``wq_b``'s output by its own RMS before RoPE, and
    V4.1 does not (compare ``model.py`` line 498 in DeepSeek-V4-Flash with lines
    770-772 in DeepSeek-V4.1-Flash -- the ``q *= rsqrt(...)`` is simply gone).

    That difference is not small and it is not self-announcing. Inheriting it cost
    a factor of ~0.52 on every layer-0 query, which reshapes the softmax rather
    than merely scaling the output: measured against a captured reference forward,
    the layer's attention output landed at rel 0.58 / cos 0.82 while every operand
    feeding the kernel was exact to 0.002, and the model produced fluent but wrong
    text. Removing the norm brings the query to rel 0.0025 of the reference.
    """

    _pos_embd_params = staticmethod(_deepseek_v41_pos_embd_params)
    q_b_norm_enabled = False


# ---------------------------------------------------------------------------
# 5. Engram
# ---------------------------------------------------------------------------


class DeepseekV41Engram(Engram):
    """V4.1's ``embed -> wkv -> gate``, with no short convolution.

    Three differences from the V4 ``Engram``, all load-bearing:

    * **No ``ShortConv``.** The reference returns the gated value directly
      (model.py:365); V4's returns ``value + short_conv(value)``. V4-Flash ships
      no engram weights at all, so the conv was never exercised against a real
      checkpoint -- and V4.1's checkpoint has no conv weights to load into it.
      ``_make_short_conv`` returns None rather than building a zeroed conv, so
      the parameters never exist and a stray conv key in a future checkpoint
      would surface as an unmatched key instead of being silently absorbed.

    * **Keys-first split.** The reference splits ``wkv``'s output as
      ``[hc_mult * dim, dim] -> (key, value)`` (model.py:354); the V4 module
      splits ``[dim, hc_mult * dim] -> (value, keys)``. Reading the checkpoint's
      fused projection with the wrong split silently swaps the value with the
      first stream's key -- no shape error, just garbage.

    * **fp32 gate and fp8 table.** The reference computes the gate and the write
      in fp32 and casts once at the end; and its embedding table is fp8 with
      per-32-lane ue8m0 scales, row-sharded over ranks (see
      ``ShardedFp8MultiHeadEmbedding`` for why that is mandatory rather than an
      optimization).

    The gate arithmetic itself is unchanged -- the reference folds
    ``q_weight * k_weight`` into one product and normalizes with the product of
    two rsqrts, which is algebraically what the V4 module's two separate RMS
    scalings compute.
    """

    def __init__(
        self,
        layer_id: int,
        config,
        vocab_sizes_flat: Optional[List[int]] = None,
        stream: Optional[torch.cuda.Stream] = None,
        mapping: Optional[Mapping] = None,
        fp8_block_size: int = 32,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ):
        # The Engram table is sharded over the TP group and put back together in
        # `_combine_embeddings` by an all-gather over the head dim, or an all-reduce
        # when the fallback row cut is used. Both need every rank to present the same
        # shape (`distributed/ops.py` all-gather with `sizes=None` requires it), and
        # attention DP gives each rank its own token count while `mapping.tp_size`
        # stays at the full width. So the collective is malformed rather than merely
        # slow, and it fails as a NCCL shape mismatch or as silently wrong rows.
        #
        # This refuses the configuration; it is not a claim that it cannot be built.
        # The shard is already the right one -- sharding heads over the whole rank set
        # is what vLLM calls `embedding_across_dp`, and under attention DP TP x DP is
        # the whole rank set -- so only the gather axis is wrong. Gathering on the
        # token dim instead stays equal-shape, and each rank then selects its own
        # token chunk with all heads out of the gathered buffer. What that needs and
        # this path does not have is a token count agreed across ranks to pad to.
        # Note that replication is not the alternative: the tables are ~40% of the
        # checkpoint, so any real support has to shard.
        if (
            mapping is not None
            and getattr(mapping, "enable_attention_dp", False)
            and mapping.tp_size > 1
        ):
            raise NotImplementedError(
                "DeepSeek-V4.1 Engram does not support attention data parallelism "
                "(enable_attention_dp=True) at tensor_parallel_size > 1: the Engram "
                "table is sharded over the TP group and recombined with a collective "
                "that requires every rank to hold the same number of tokens, which "
                "attention DP violates. Run with enable_attention_dp=False, or at "
                "tensor_parallel_size=1."
            )

        # Read by `_make_multi_head_embedding`, which the base __init__ calls,
        # so these have to be in place first. Plain attributes on a not-yet
        # initialized nn.Module are fine (nn.Module.__setattr__ only intercepts
        # Parameters, Modules and already-registered names).
        self.mapping = mapping
        self.tp_size = 1 if mapping is None else mapping.tp_size
        self.tp_rank = 0 if mapping is None else mapping.tp_rank
        self.fp8_block_size = fp8_block_size

        super().__init__(
            layer_id=layer_id,
            config=config,
            vocab_sizes_flat=vocab_sizes_flat,
            stream=stream,
        )

        # Sum all-reduce reassembles the row shards. Built here rather than
        # reaching for a global so a TP=1 run allocates nothing, and carrying the
        # model's configured strategy like every other collective in the model.
        # A head-sharded table needs a concatenation instead, so it allocates no
        # workspace at all (see `_combine_embeddings`).
        #
        # It carries a width nothing else does -- `engram_n_heads *
        # engram_head_dim` (6144) against `hidden_size` (5120) -- which made it
        # the prime suspect for V4.1's nondeterministic all-NaN decode logits,
        # since AUTO routes both through one shared Lamport workspace whose
        # receive-slot clearing is sized from the *previous* call. Forcing only
        # this collective onto NCCL was measured and did **not** cure the NaN, so
        # the width-mixing story is falsified and a local override here would be
        # cargo cult. The defect is somewhere in the custom all-reduce path as a
        # whole -- `allreduce_strategy="NCCL"` model-wide is the only arm observed
        # clean -- and it belongs to that path, not to Engram.
        self.embed_all_reduce = (
            AllReduce(mapping=mapping, strategy=allreduce_strategy)
            if mapping is not None
            and self.tp_size > 1
            and not self.multi_head_embedding.shard_heads
            else None
        )

    def _make_multi_head_embedding(self, list_of_N: List[int], D: int) -> nn.Module:
        """Sharded fp8 table instead of the dense replicated one.

        Same bucket layout (``list_of_N`` -> ``offsets``), so the indices the
        shared ``EngramHashProvider`` computes are unchanged; only the storage and
        the cut differ. Which cut it picks -- head or row -- follows from the head
        count and ``tp_size``, and ``_combine_embeddings`` reads it back off the
        module to choose the matching collective.

        ``TRTLLM_V41_ENGRAM_CPU_OFFLOAD`` additionally moves the shard off the GPU
        (see ``engram_cpu_offload_enabled``). It is read here rather than taken as
        a constructor argument because the offload changes nothing a caller can
        observe -- same shapes, same values, same checkpoint keys -- so there is
        nothing for a caller to coordinate with.
        """
        return ShardedFp8MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=D,
            block_size=self.fp8_block_size,
            dtype=self.config.dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            cpu_offload=engram_cpu_offload_enabled(),
        )

    def _make_short_conv(self) -> None:
        return None

    def _combine_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reassemble the per-rank contributions into the full ``[T, H * D]``.

        Which collective that takes is decided by how the table was cut, so it is
        read off the embedding rather than configured twice:

        * **head-sharded** -- each rank holds a distinct, complete slice of the
          head axis, so the pieces concatenate. ``flatten`` is head-major and the
          owned heads are a contiguous ascending run, so an all-gather on the last
          dimension lands every rank's columns in exactly the right place; no
          ``sizes`` is needed because an uneven split is what disables this cut.
        * **row-sharded** -- every rank produces a full-width partial sum with the
          rows it does not own zeroed, so the pieces add.

        The all-gather is the cheaper of the two -- it moves each rank's slice once
        rather than reducing full-width buffers -- but the real saving is upstream:
        a head-sharded rank gathers ``1/tp_size`` of the table rows.
        """
        if self.embed_all_reduce is not None:
            return self.embed_all_reduce(
                embeddings, all_reduce_params=AllReduceParams(enable_allreduce=True)
            )
        if self.multi_head_embedding.shard_heads:
            return allgather(embeddings, self.mapping, dim=-1)
        return embeddings

    def precompute(
        self,
        hash_indices: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Sharded lookup on the Engram side stream, collective on the caller's.

        Same contract as the base class (``[T, num_heads]`` in,
        ``[T, num_heads * D]`` out, ``sync_event`` recorded). Each rank holds only
        a slice of the table, so the flattened lookup is this rank's contribution
        and is not the return value until ``_combine_embeddings`` has run.

        The split is not a style choice, and it holds for either collective. Both
        share one communicator -- and ``AllReduce`` one IPC workspace -- with every
        other collective in the model, the MoE reduction and the attention TP
        reduction included, all of which are issued on the main stream. Two
        collectives in flight over the same workspace race on its flags, and eight
        ranks that disagree about ordering race on the communicator itself. The
        result is silently wrong embeddings rather than a crash, and it only
        reproduces when the streams actually overlap: it disappears under
        ``CUDA_LAUNCH_BLOCKING=1``, and it disappeared under an instrumentation
        pass whose per-layer ``isfinite`` checks happened to sync often enough to
        serialize the fork. So the lookup overlaps and the collective does not.
        """
        if self.stream is None:
            embeddings = self.multi_head_embedding(hash_indices)
            embeddings = self._combine_embeddings(embeddings.flatten(start_dim=-2))
            return embeddings.to(dtype) if dtype is not None else embeddings

        # See the base class: the two `record_stream` calls are what keep the
        # caching allocator from recycling a block across the fork, in either
        # direction. `wait_stream`/`sync_event` order the *work*, not the memory.
        caller_stream = torch.cuda.current_stream()
        self.stream.wait_stream(caller_stream)
        hash_indices.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            # `background`: the lookup is on the side stream precisely so it can
            # overlap the main one, which it cannot do while holding every SM.
            embeddings = self.multi_head_embedding(hash_indices, background=True)
            embeddings = embeddings.flatten(start_dim=-2)
            self.sync_event.record()
        embeddings.record_stream(caller_stream)

        if self.tp_size > 1:
            # Ordered against the lookup explicitly, since the collective is now
            # on a different stream than the contribution it consumes.
            self.sync_event.wait(caller_stream)
            embeddings = self._combine_embeddings(embeddings)
        if dtype is not None:
            embeddings = embeddings.to(dtype)
        # Re-recorded so `sync_event` keeps meaning "the returned tensor is
        # ready". The caller waits on it before the consuming layer; left
        # pointing at the pre-collective lookup, that wait would be satisfied
        # too early and the layer could read a partial sum.
        self.sync_event.record(caller_stream)
        return embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """``[T, HC, D]`` stream + precomputed embeddings -> the ``[T, HC, D]`` delta.

        Returns the *delta* (``gate * value``), matching the V4 module's contract
        and the caller's ``residual = residual + engram(...)``. The reference
        returns ``h + gate * value`` instead; same thing, one addition moved.
        """
        if use_cache or conv_state is not None:
            raise ValueError(
                "DeepSeek-V4.1's Engram has no convolution, so there is no conv "
                "state to carry across decode steps."
            )
        D = self.config.hidden_size
        HC = self.config.hc_mult

        kv = self.kv_proj(embeddings)
        # Keys first -- see the class docstring.
        keys, value_raw = kv.split([HC * D, D], dim=-1)
        keys = keys.float().unflatten(-1, (HC, D))

        # One product of the two per-stream norm weights, exactly as the
        # reference does it ("only ever used as a product", model.py:356).
        weight = self.query_norm_weight.float() * self.key_norm_weight.float()
        h = hidden_states.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.norm_eps) * torch.rsqrt(
            keys.square().mean(-1) + self.norm_eps
        )
        dot = (h * weight * keys).sum(-1) * rstd * D**-0.5
        # Signed sqrt before the sigmoid. `copysign` rather than `* sign()`
        # because `sign(0) == 0` would zero the clamp floor the reference keeps.
        # 1e-6 is the reference's own `self.clamp_value` (model.py:340), a literal
        # there too -- it is not derived from norm_eps and does not follow it.
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (gate.unsqueeze(-1) * value_raw.float().unsqueeze(-2)).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# 2 + 3. mHC constants and the lagged-`pre` wiring
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(DeepseekV4DecoderLayer):
    """V4's decoder layer with V4.1's mHC wiring.

    Everything structural is inherited -- the attention module, the MoE stack,
    the two RMSNorms, the fusion config, the engram slot. Only the residual
    plumbing differs, and it differs in a way that cannot be expressed by a flag:
    V4.1 collapses each sublayer's input with the *previous* sublayer's ``pre``
    coefficients (model.py:968-995), so the ``pre`` a sublayer computes has to
    survive past its own forward.

    That single fact rules out both mHC fusions:

    * ``fused_hc`` computes ``pre`` internally and consumes it in the same
      kernel, so ``enable_fused_hc`` is pinned False here rather than read from
      the config. ``post_load_weights`` derives ``defer_post_mapping`` from the
      two neighbours' ``enable_fused_hc``, so pinning this one flag also stops
      every layer from deferring -- no second override needed.
    * With deferral off, each layer resolves its own ``hc_ffn.post_mapping`` and
      hands the next layer a *resolved* ``HCState``, which is where the trailing
      ``pre`` rides along (``HCState.pre_mix``).

    The cost is real: two Sinkhorn tails per layer run in torch instead of one
    fused kernel. It is not optional, and the profile is not the reason V4 has
    the fused path -- correctness is the reason V4.1 cannot use it.
    """

    attention_cls = DeepseekV41Attention

    def _make_mhc(self) -> mHC:
        """V4.1's mHC constants, all four of which differ from the V4 defaults.

        ``hc_eps`` (1e-6) covers all three epsilon roles the reference kernel
        uses it for -- the ``pre`` sigmoid offset, the ``comb`` softmax floor and
        every Sinkhorn division -- while the RMS statistic in the mixer
        projection uses ``rms_norm_eps``, which for V4.1 is 1e-20 and not the
        module's 1e-6 default. At bf16 magnitudes 1e-20 is far below the
        mean-square of any real activation, so this is effectively "no epsilon";
        getting it wrong the other way (using 1e-6 where the reference uses
        1e-20) would perturb every token's mixer coefficients.
        """
        config = self.config
        hc_eps = config.hc_eps
        return mHC(
            config.hc_mult,
            config.hidden_size,
            config.hc_sinkhorn_iters,
            dtype=torch.float32,
            eps=hc_eps,
            norm_eps=config.rms_norm_eps,
            sinkhorn_eps=hc_eps,
            post_mult_value=2.0,
        )

    def _make_engram(self, layer_idx: int, engram_config, vocab_sizes_flat, stream):
        return DeepseekV41Engram(
            layer_id=layer_idx,
            config=engram_config,
            vocab_sizes_flat=vocab_sizes_flat,
            stream=stream,
            mapping=self.mapping,
            allreduce_strategy=self.model_config.allreduce_strategy,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Not a knob: the fused kernels cannot express a lagged `pre`. Set after
        # super().__init__ so it overrides _resolve_enable_fused_hc (including
        # the TRTLLM_MHC_ENABLE_FUSED_HC env override, which must not be able to
        # turn on a path that computes the wrong thing).
        self.enable_fused_hc = False
        self.defer_post_mapping = False

    def forward(
        self,
        position_ids: torch.IntTensor,
        hc_state: HCState,
        attn_metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        input_ids: Optional[torch.IntTensor] = None,
        engram_embeddings=None,
        **kwargs,
    ) -> HCState:
        """One V4.1 block: ``engram -> attn -> MoE``, both sublayers lagged.

        A transcription of the reference ``Block.forward`` (model.py:968-995),
        which per sublayer does::

            residual = x
            pre_own, post, comb = hc_mixes(x)  # coeffs from x, BEFORE collapse
            x = hc_pre(x, pre_external)  # collapse with the LAGGED pre
            x = sublayer(norm(x))
            x = hc_post(x, residual, post, comb)

        and returns ``(x, ffn_pre)`` -- the FFN sublayer's own ``pre``, for the
        next block. ``mHC.pre_mapping_lagged`` is exactly the first three lines,
        which is why this method reads as two calls per sublayer rather than
        reimplementing the coefficient math.

        The ordering detail worth stating: the coefficients come from the
        *uncollapsed* stream, so ``pre_mapping_lagged`` must be given the
        residual before any collapse, and ``post_mapping`` must be given that
        same residual afterwards. Passing the collapsed ``layer_input`` to
        either would typecheck and produce garbage.
        """
        residual = hc_state.residual
        pre_mix = hc_state.pre_mix

        # Engram fires at layer entry, before the mHC coefficients are computed,
        # so its delta is visible both to `hc_mixes` and to `hc_post`'s residual
        # (model.py:1256-1260). Same ordering as V4's `_entry_boundary`.
        if self.engram is not None and engram_embeddings is not None:
            residual = residual + self.engram(residual, engram_embeddings)

        capture_this_layer = spec_metadata is not None and spec_metadata.is_layer_capture(
            self.layer_idx
        )
        if capture_this_layer and spec_metadata.spec_dec_mode.is_dspark():
            # V4.1 captures at the layer ENTRY (after engram), not the exit:
            # `if i in target_layer_ids: main_hiddens.append(h.mean(dim=2))`
            # runs before `layer(...)` (model.py:1258-1262). V4 captures the
            # post-MoE resolved residual instead, so this cannot be inherited.
            # The shape contract is unchanged -- the flat [N, mult * hidden]
            # stream, which the DSpark metadata means over the mult axis -- and
            # `mtp.0.main_proj.weight`'s [5120, 3 * 5120] confirms the mean
            # rather than a flatten.
            spec_metadata.maybe_capture_hidden_states(
                self.layer_idx, residual.reshape(residual.shape[0], -1), None
            )

        # --- attention sublayer -------------------------------------------
        attn_pre, post_mix, comb_mix, layer_input = self.hc_attn.pre_mapping_lagged(
            residual, pre_mix
        )
        layer_input = self.input_layernorm(layer_input)
        x_attn = self.self_attn(
            position_ids=position_ids,
            hidden_states=layer_input,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )
        residual = self.hc_attn.post_mapping(
            x=x_attn,
            residual=residual,
            post_layer_mix=post_mix,
            comb_res_mix=comb_mix,
        )

        # --- MoE sublayer, collapsing with the attention sublayer's `pre` ---
        ffn_pre, post_mix, comb_mix, layer_input = self.hc_ffn.pre_mapping_lagged(
            residual, attn_pre
        )
        # No norm here: forward_MoE applies post_attention_layernorm itself
        # (either standalone or fused into the pre-MoE all-reduce).
        x_ffn = self.forward_MoE(
            hidden_states=layer_input,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            input_ids=input_ids,
        )
        residual = self.hc_ffn.post_mapping(
            x=x_ffn,
            residual=residual,
            post_layer_mix=post_mix,
            comb_res_mix=comb_mix,
        )

        # `ffn_pre` rides to the next block, which collapses with it.
        return HCState.resolved(residual, pre_mix=ffn_pre)


# ---------------------------------------------------------------------------
# 4. No hc_head: the stream is opened and closed differently
# ---------------------------------------------------------------------------


class DeepseekV41Model(DeepseekV4Model):
    """V4's model body with V4.1's two mHC stream boundaries.

    The loop between them is inherited verbatim -- engram precompute on its side
    stream, the per-layer event waits, the PP-rank branch. Only the two ends move.
    """

    decoder_layer_cls = DeepseekV41DecoderLayer
    uses_hc_head = False

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        self._decoder_replay_observed = False
        self.decoder_replay_split, self.decoder_replay_window = self._resolve_replay_policy(
            model_config
        )

    def _resolve_replay_policy(
        self, model_config: ModelConfig[PretrainedConfig]
    ) -> Tuple[Optional[int], int]:
        """``(first replayed layer, window)``, or ``(None, 0)`` if the model cannot replay.

        The split is derived rather than fixed at ``L/2``, because the invariant
        that makes §3.2.2 legal is not "half the layers" -- it is *no compressor at
        or after the split*. ``max(kv_source_layer_ids) + 1`` is the earliest layer
        that satisfies it, and on the released checkpoint that lands at 21 (sources
        ``[2, 8, 14, 20]``), one past the ``L/2 = 20`` the report names. Reading the
        list means a checkpoint that moves a source moves the split with it instead
        of silently replaying a layer that publishes truncated compressed KV.

        Two structural refusals, both checked once here rather than per forward:

        * **No sparse config.** Without ``kv_source_layer_ids`` there is no evidence
          any suffix is compressor-free, and assuming it is the corruption case.
        * **An Engram layer at or after the split.** Engram embeddings are
          precomputed for all ``N`` rows on a side stream before the loop, so a
          replayed Engram layer would need its cache re-indexed *and* its event
          waited on at the boundary. The released checkpoint puts Engram on layers
          1 and 14, both well inside the encoder half, so that code would be
          unreachable and untested -- refusing is honest where handling it is not.
        """
        if not decoder_bounded_replay_enabled():
            return None, 0
        sparse_config = model_config.sparse_attention_config
        kv_sources = getattr(sparse_config, "kv_source_layer_ids", None) or ()
        if not kv_sources:
            logger.warning(
                "DeepSeek-V4.1 decoder bounded replay was requested but this model has "
                "no kv_source_layer_ids, so no suffix of the stack is known to be free "
                "of compressors. Running the ordinary single-pass prefill."
            )
            return None, 0
        split = int(max(kv_sources)) + 1
        late_engram = [idx for idx in getattr(self, "engram_layer_ids", ()) or () if idx >= split]
        if late_engram:
            logger.warning(
                f"DeepSeek-V4.1 decoder bounded replay was requested but Engram layers "
                f"{late_engram} are at or after the replay split {split}; their "
                "embeddings are precomputed for the full prompt on a side stream and "
                "the replay boundary does not re-index them. Running the ordinary "
                "single-pass prefill."
            )
            return None, 0
        window = int(sparse_config.window_size)
        logger.info(
            f"DeepSeek-V4.1 decoder bounded replay enabled: layers {split}-"
            f"{self.num_hidden_layers - 1} replay over a {window}-token window "
            f"(kv sources {sorted(int(s) for s in kv_sources)})."
        )
        return split, window

    def _plan_bounded_replay(self, attn_metadata, all_token_states_required: bool):
        """Replay the decoder half over the last ``n_win`` rows, when that is legal.

        ``all_token_states_required`` is the caller's answer to "does anything
        downstream read a hidden state that is not the last of its sequence" --
        context logits, in practice. The replayed pass produces no such rows, so a
        caller that needs them gets the ordinary single pass.
        """
        if self.decoder_replay_split is None or all_token_states_required:
            return None, None
        plan = plan_decoder_replay(attn_metadata, self.decoder_replay_window)
        if plan is None:
            return None, None
        if not self._decoder_replay_observed:
            # Once per process, on the first batch that actually replays. The
            # construction-time line above says the policy resolved; this one says a
            # forward took it, which is the only thing an A/B can assert on -- a run
            # whose every batch is refused looks identical to a run with the feature
            # off, and that is exactly the failure that turns an A/B into an A/A.
            self._decoder_replay_observed = True
            logger.info(
                f"DeepSeek-V4.1 decoder bounded replay engaged: "
                f"{plan.num_replay_tokens} of {plan.num_encoder_tokens} rows re-run "
                f"from layer {self.decoder_replay_split}."
            )
        return self.decoder_replay_split, plan

    def _enter_bounded_replay(self, plan, attn_metadata, position_ids, input_ids, hc_state):
        """Narrow the per-token state to the replayed rows and rebuild the metadata.

        Three carriers cross this boundary, and they are exactly the arguments the
        remaining layers take per token: ``position_ids`` (RoPE), ``input_ids``
        (which ``forward_MoE`` forwards to the MoE for balancing) and the mHC
        stream. The stream is always ``resolved`` here -- V4.1 forces
        ``enable_fused_hc = False``, so ``post_mix`` / ``comb_mix`` / ``x_prev`` are
        all None and only ``residual`` and the lagged ``pre_mix`` carry state.
        """
        enter_decoder_replay(attn_metadata, plan)
        return (
            gather_replayed_rows(position_ids, plan),
            gather_replayed_rows(input_ids, plan),
            HCState.resolved(
                gather_replayed_rows(hc_state.residual, plan),
                pre_mix=gather_replayed_rows(hc_state.pre_mix, plan),
            ),
        )

    def _exit_bounded_replay(self, plan, attn_metadata, hidden_states):
        """Restore the encoder shape on both the metadata and the output."""
        exit_decoder_replay(attn_metadata, plan)
        return scatter_replayed_rows(hidden_states, plan)

    def _init_hc_state(self, hidden_states: torch.Tensor) -> HCState:
        """Seed the lagged ``pre`` with the reference's one-hot.

        ``make_identity_pre_mix`` (model.py:1159) is ``zeros(..., hc_mult)`` with
        ``[..., 0] = 1.0`` -- a one-hot on stream 0, *not* ones and not
        ``1 / hc_mult``. Since layer 0 receives four identical copies of the
        embedding (``h.unsqueeze(2).repeat(...)``), the resulting collapse returns
        the embedding unscaled; a ones seed would hand layer 0 four times the
        embedding, and a uniform ``1/mult`` seed would be right only by accident
        of the streams being identical -- and would diverge the moment engram
        fires on layer 0.
        """
        pre_mix = hidden_states.new_zeros(
            (hidden_states.shape[0], self.hc_mult, 1), dtype=torch.float32
        )
        pre_mix[:, 0, :] = 1.0
        return HCState.resolved(hidden_states, pre_mix=pre_mix)

    def _finalize_hc_state(self, hc_state: HCState) -> torch.Tensor:
        """Close the stream with the last layer's trailing ``pre``.

        The reference runs ``h = layer.hc_pre(h, pre_mix)`` after the loop
        (model.py:1268) using the last block's ``ffn_pre`` -- one more collapse
        with the same coefficients every other sublayer boundary uses, rather
        than V4's separately-parameterized ``hc_head``. Returns plain
        ``[N, hidden]``; ``DeepseekV4LogitsProcessor`` then applies ``norm`` and
        the head, matching ``head(norm(h))``.
        """
        return mHC.collapse(hc_state.residual, hc_state.pre_mix)

    # The descriptor fields a *constructed* layer observably owns. Everything
    # else on ``DeepseekV41LayerDescriptor`` is either bookkeeping only the
    # config knows (which layer sources which cache, ``kind``) or a statement
    # about weights that do not exist until load time (``owns_indexer_wk``,
    # ``compressor_wkv_dtype`` -- an index source that is not a kv source still
    # builds an ``Indexer`` whose ``wk`` gets zero-filled, so the module cannot
    # answer who owns the real ``wk``). Those are emitted but not cross-checked.
    @staticmethod
    def _observe_layer(layer, layer_idx: int) -> Dict[str, Any]:
        """Read the topology a constructed decoder layer actually got.

        Everything here is available immediately after ``__init__`` with no
        backend, KV-cache manager, or metadata: ``MLA.__init__`` stores
        ``sparse_params`` from ``ModelConfig.sparse_attention_config`` and then
        calls ``self.sparse_attn_hooks.initialize(self)`` (attention/mla.py:576),
        which is what binds ``indexer`` / ``compressor``.

        The point of going through the *modules* rather than re-reading the
        config is that this is the only thing that fails when the sparse config
        never reaches the attention layer. A config-only dump would agree with
        the reference-derived plan even with ``sparse_attention_config=None``,
        because both sides would be reading the same ``config.json``.
        """
        attn = layer.self_attn
        sparse_params = getattr(attn, "sparse_params", None)
        if sparse_params is None:
            raise ValueError(
                f"layer {layer_idx}: MLA.sparse_params is None, so no layer of "
                "this model is sparse -- ModelConfig.from_pretrained derived no "
                "DeepSeekV4SparseAttentionConfig for this architecture. The "
                "topology in the checkpoint config is then invisible to the "
                "attention stack."
            )
        variant = sparse_params.variant
        ratios = sparse_params.compress_ratios
        ratio = int(ratios[min(layer_idx, len(ratios) - 1)])
        kv_sources = sparse_params.kv_source_layer_ids or ()
        rope = attn.pos_embd_params
        return {
            "compress_ratio": ratio,
            # Straight from the backend's own predicates, so a variant
            # regression in params.py shows up here rather than at cache
            # allocation time.
            "has_long_range": is_compress_layer(ratio, variant),
            "pools_kv": has_compressor_state(ratio, variant),
            "pool_factor": pool_factor(ratio),
            "rope_theta": float(rope.rope.theta),
            "yarn_enabled": rope.type is PositionEmbeddingType.yarn,
            "window_size": sparse_params.window_size,
            "is_kv_source": layer_idx in kv_sources,
            # These two come off the constructed module tree rather than out of
            # the id lists again: ``initialize_sparse_attn`` aliases the
            # backend's modules onto the MLA (sparse/deepseek_v4/module.py:71),
            # and the backend only builds them for an owner
            # (``owns_index_topk`` / ``owns_compressed_kv``). So they check the
            # whole chain -- config -> sparse_params -> backend predicate ->
            # module -- against the checkpoint's own weight-key topology:
            # ``attn.compressor.wkv`` exists on layers [2, 8, 14, 20] and
            # ``attn.indexer.wq_b`` on [2, 8, 14, 20, 24, 28, 32, 36] in
            # model.safetensors.index.json.
            "is_index_source": attn.indexer is not None,
            "owns_compressor": attn.compressor is not None,
            "has_engram": layer.engram is not None,
        }

    def describe_layers(self) -> Dict[str, Any]:
        """Per-layer topology of *this* instance, for ``gates/layer_plan.py --diff``.

        One row per ``compress_ratios`` entry -- 43 for the released checkpoint,
        i.e. 40 decoder layers plus 3 MTP layers -- because that list is what
        both the reference and every ratio-keyed buffer in the backend are sized
        by. Each row is the full descriptor, plus:

        * ``constructed``: whether this instance actually built the layer. The 3
          MTP rows are ``False`` -- MTP construction needs a ``spec_config``, a
          ``DeepseekV41MTP``, and V4.1's DSpark cross-attention, none of which
          exist yet, and ``DeepseekV41ForCausalLM`` refuses a ``spec_config``
          outright. Their fields are config-derived and unverified.

        Every constructed layer is cross-checked against its descriptor row and
        a disagreement raises, so a passing dump means the modules and the
        reference-derived plan agree -- not merely that two readers of the same
        ``config.json`` agree.
        """
        pretrained_config = self.model_config.pretrained_config
        descriptors = pretrained_config.layer_descriptors
        # `self.layers` is extended with MTP layers on the spec path; V4.1 has no
        # spec path yet, but slice defensively rather than mis-attribute an MTP
        # module to a decoder descriptor.
        built = {idx: layer for idx, layer in enumerate(self.layers[: self.num_hidden_layers])}

        rows: List[Dict[str, Any]] = []
        verified: set = set()
        problems: List[str] = []
        for descriptor in descriptors:
            row = descriptor.to_dump_dict()
            layer = built.get(descriptor.layer_idx)
            row["constructed"] = layer is not None
            if layer is not None:
                observed = self._observe_layer(layer, descriptor.layer_idx)
                for field, value in observed.items():
                    verified.add(field)
                    if row[field] != value:
                        problems.append(
                            f"layer {descriptor.layer_idx}: {field} "
                            f"descriptor={row[field]!r} module={value!r}"
                        )
            rows.append(row)

        if problems:
            raise ValueError(
                "constructed layers disagree with their config descriptors:\n  "
                + "\n  ".join(problems)
            )

        not_constructed = [d.layer_idx for d in descriptors if d.layer_idx not in built]
        if not_constructed:
            logger.warning(
                f"DeepSeek-V4.1 describe_layers: layers {not_constructed} are "
                "described from the config only -- this instance did not build "
                "them (MTP is not implemented), so their fields are unverified."
            )
        return {
            "model": type(self).__name__,
            "variant": "v41",
            "num_hidden_layers": self.num_hidden_layers,
            "num_ratio_entries": len(descriptors),
            "num_constructed": len(built),
            "not_constructed": not_constructed,
            "verified_fields": sorted(verified),
            "layers": rows,
        }


# ---------------------------------------------------------------------------
# 6. Checkpoint layout
# ---------------------------------------------------------------------------

# Engram module renames. The checkpoint names the table `embed` and the fused
# projection `wkv`; TRT-LLM's Engram calls them `multi_head_embedding` and
# `kv_proj`. `q_weight`/`k_weight` are the two `[hc_mult, hidden]` norm weights,
# which the module holds as direct parameters.
_ENGRAM_SUBKEY_RENAME = {
    "embed": "multi_head_embedding",
    "wkv": "kv_proj",
    "q_weight": "query_norm_weight",
    "k_weight": "key_norm_weight",
}

# Text-only bring-up: the vision tower and its projector are dropped rather than
# left to fall through the passthrough branch, so a genuinely unexpected key
# still stands out. `image_*` are the three learned separator embeddings.
_V41_DROP_PREFIXES = ("vision.", "aligner.")
_V41_DROP_KEYS = ("image_start", "image_end", "image_newline")

# Sub-trees whose `.weight`/`.scale` pair must NOT be dequantized:
#   - routed experts are packed MXFP4 (int8 pairs + a ue8m0 scale per 32 lanes),
#     consumed quantized by the MoE backend;
#   - the engram table is 98 GiB of fp8 and stays fp8, row-sharded
#     (`ShardedFp8MultiHeadEmbedding` dequantizes per lookup).
_V41_NO_DEQUANT = (".ffn.experts.", ".engram.embed.")

_V41_FP8_BLOCK_SIZE = 32

# Under `dense_fp8_requant_enabled()`, the dense fp8 stems whose 32x32 scales are
# *requantized* to 128x128 instead of being dequantized away. Everything not named
# here keeps the bf16 fallback, which is deliberate: this is an allow-list, so a
# checkpoint key nobody has looked at falls back to the path GSM8K was measured on
# rather than into a block-scaled Linear that may not exist.
#
# Each entry is the checkpoint stem's tail, matched with `str.endswith`, so
# `.attn.wkv` does not also catch `.attn.wkv_b` and `.attn.wo_b` does not catch
# `.attn.wo_a`. The membership test that has to be right is *inclusion*, not
# exclusion: a stem left out here is dequantized to bf16 while its consumer, having
# seen `FP8_BLOCK_SCALES`, allocates fp8 and a scale buffer. `.attn.wo_a` is the
# instructive case -- `o_a_proj` looks like a bare `nn.Parameter` of the model dtype
# (`sparse/deepseek_v4/module.py:120`), so leaving it bf16 looks free, but
# `create_sparse_attn_weights` (`module.py:213-238`) *redeclares* it as fp8 on
# SM100 and adds `o_a_proj_scale`. Handing that a bf16 tensor makes
# `load_o_a_proj`'s `.to(module.o_a_proj.dtype)` an unscaled bf16 -> fp8 cast, which
# clips every value above 448 and reports nothing. The release census test is what
# caught it, via the 40 unfilled `o_a_proj_scale` parameters.
#
# `endswith` is also what made `.attn.indexer.wq_b` invisible here for a while, and
# that is the cautionary half of the same property: what keeps `.attn.wkv` from also
# matching `.attn.wkv_b` is exactly what stops `.attn.wq_b` from covering the *longer*
# `.attn.indexer.wq_b`. The indexer's Q projection is a `Linear` handed the
# `quant_config` object directly (`sparse/dsa/indexer.py:743`), so no name-based
# `exclude_modules` entry reaches it either, and its (4096, 1280) weight is
# 128-divisible both ways -- it becomes an fp8 block-scaled `Linear` the moment the
# global algo is set. Two independent silencers kept that quiet: `Linear`'s block-scale
# load is `if scale_name in weights[0]` (`modules/linear.py:1276`), so a missing
# `weight_scale_inv` leaves `weight_scale` at its `torch.empty`; and `copy_weight`
# casts, so the bf16 weight enters the fp8 parameter unscaled and clips above 448.
# Neither the load census nor a short-prompt smoke test can see it -- see
# `test_v41_every_fp8_consumer_receives_its_scale`, which asserts the pairing
# invariant that survives fusion.
#
# The two stems deliberately left out. These are the only two: the checkpoint's
# `.scale` stems reduce to five shapes -- `.attn.wq_a` / `.attn.wq_b` /
# `.attn.indexer.wq_b` / `.attn.wkv` / `.engram.wkv` under `layers.N` and `mtp.N`, plus
# `.ffn.*` and `.main_proj` -- so this list plus these two bullets accounts for all of
# them, which is the property that makes an allow-list auditable.
#
#   * `.engram.wkv`  -- `*engram*` is in `exclude_modules`, and the consumer is a
#     plain `nn.Linear` (`modules/engram/engram.py:1002`), which has no block-scale
#     path at all, so its shape and dtype do not move with the quant config.
#   * `.main_proj`   -- `mtp.0.main_proj` has no consumer in V4's remap; it reaches
#     no module, so quantizing it would only change what an unconsumed tensor looks
#     like.
_V41_REQUANT_STEM_TAILS = (
    ".attn.wq_a",
    ".attn.wq_b",
    ".attn.indexer.wq_b",
    ".attn.wkv",
    ".attn.wo_a",
    ".attn.wo_b",
    ".ffn.shared_experts.w1",
    ".ffn.shared_experts.w2",
    ".ffn.shared_experts.w3",
)

_E4M3_MAX = 448.0
# e4m3 keeps 3 mantissa bits, so the smallest normal step relative to a tile whose
# max sits at the e4m3 ceiling is 2**-9. Recorded as a constant because it is the
# error bound the unit tests assert against, not a tuning knob.
_E4M3_SUBNORMAL_FLOOR_EXP = -9
_V41_REQUANT_BLOCK = 128

# The flat mHC parameter names V4's checkpoints use (``hc_attn_fn``) against the
# structured ones the modules register (``hc_attn.fn``). ``load_flat_hc_weights``
# in V4's walk does the loading; the census only needs the mapping so it can tell
# a real orphan from a name it already knows how to place.
_V41_FLAT_HC_STEMS = ("hc_attn", "hc_ffn", "hc_head")
_V41_FLAT_HC_ATTRS = ("fn", "base", "scale")

# The reverse rewrite: keys V4's walk spells as a sub-module of a *plain
# ``nn.Parameter``*, whose destination parameter is a flat sibling instead.
# ``{structured suffix: flat parameter name}``.
#
# ``o_a_proj`` is an ``nn.Parameter`` on the attention module, not a ``Linear``, so
# there is no sub-module for a ``.weight_scale_inv`` to live under;
# ``load_o_a_proj`` (V4's walk, ``modeling_deepseekv4.py:805``) reads the key under
# that spelling and copies it into the sibling ``o_a_proj_scale``. Both spellings
# are V4's, not V4.1's -- but the pair only *exists* under
# ``TRTLLM_V41_DENSE_FP8=1``, because ``create_sparse_attn_weights``
# (``sparse/deepseek_v4/module.py:208``) allocates ``o_a_proj_scale`` only when the
# quant config reports fp8 block scales, and on the bf16 default the ``.scale``
# sibling is folded away by the remap and never forwarded. So the rewrite is
# unconditional and still fails closed: if the module did not allocate the
# parameter, the rewritten name is absent from ``targets`` and the key is reported
# as an orphan rather than quietly accepted.
_V41_FLAT_SCALE_PARAMS = {"o_a_proj.weight_scale_inv": "o_a_proj_scale"}

# Checkpoint sub-modules V4's walk fuses into a *differently named* model module,
# so their keys never match a parameter name and the census would read them as
# orphans. ``Linear(gate_up_proj)`` is loaded from ``gate_proj`` + ``up_proj``
# (the walk's ``params_map``), and MLA's a-projection is one fused
# ``kv_a_proj_with_mqa`` built from the checkpoint's ``q_a_proj`` +
# ``kv_a_proj_with_mqa`` (the walk's ``kv_a_proj_with_mqa`` branch).
_V41_FUSED_MODULES = {
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "q_a_proj": "kv_a_proj_with_mqa",
}

# Leaf parameters V4's walk *relocates* onto an inner module that is not reachable
# by name: ``{leaf: attribute holding the module it lands on}``.
#
# ``attn_sink`` is keyed on the attention module (``self_attn.attn_sink``) but the
# walk assigns it to ``self_attn.mqa``, and ``mqa`` is held as a plain Python
# attribute rather than a registered sub-module -- deliberately, so that the inner
# ``DeepseekV4TrtllmAttention`` stays off the generic weight-loading walk and out
# of ``state_dict``. The consequence for the audit is that *no* name reaches the
# sinks: they are absent from ``named_modules``, ``named_parameters`` and
# ``named_buffers`` alike, so rewriting the key to ``self_attn.mqa.attn_sink``
# would only move the mismatch rather than resolve it.
#
# So ``_v41_load_coverage`` resolves these two hops as attributes instead, which
# is also strictly stronger than a name match: it reports whether the tensor
# actually arrived, and the sinks are exactly the case where that matters, because
# they are created only when the checkpoint ships them (absent, the attribute
# stays ``None`` and the kernel is called without sinks -- silently wrong rather
# than uninitialized).
_V41_RELOCATED_PARAMS = {"attn_sink": "mqa"}


def _v41_ignore_reason(key: str, *, keep_mtp_layers: int = 0) -> Optional[Tuple[str, str]]:
    """``(name pattern, why)`` if this raw checkpoint key is deliberately not loaded.

    One function decides, so the remap and the census cannot disagree about what
    was dropped -- a census that recomputes the drop rules separately from the
    code that applies them audits nothing.

    ``keep_mtp_layers`` is how many MTP layers the *model* built, not how many the
    checkpoint ships: V4's remap routes ``mtp.0.`` to ``model.layers.<n>.`` and has
    no route for the rest, so anything at or past the built count has no consumer
    and is counted as ignored rather than forwarded to be silently dropped.
    """
    if key.startswith(_V41_DROP_PREFIXES):
        return (
            "vision.* | aligner.*",
            "vision tower and projector; text-only bring-up",
        )
    if key in _V41_DROP_KEYS:
        return (
            "image_start | image_end | image_newline",
            "learned image separator embeddings; text-only bring-up",
        )
    if key.endswith("ffn.gate.bias_vl"):
        # The router's second correction bias, selected per token in place of
        # `ffn.gate.bias` for image tokens only (`inference/model.py:816`). With
        # the vision tower dropped there are no image tokens, so `DeepseekV4Gate`
        # allocates no parameter for it and the key has no consumer. Dropped here
        # rather than forwarded so the census reports it as a deliberate omission
        # instead of an unclaimed key.
        return (
            "layers.<i>.ffn.gate.bias_vl",
            "per-token vision router bias; text-only bring-up selects `bias` at "
            "every position, so the resolved bias is bitwise unchanged",
        )
    if key.startswith("mtp."):
        parts = key.split(".", 2)
        index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else keep_mtp_layers
        if index >= keep_mtp_layers:
            return (
                "mtp.*" if keep_mtp_layers == 0 else f"mtp.<i>.* for i >= {keep_mtp_layers}",
                "MTP layers are not constructed (DeepseekV41ForCausalLM refuses a "
                "spec_config), so no layer claims them",
            )
    return None


def _v41_tensor_shape(value: Any) -> Sequence[int]:
    """Shape of a loaded tensor or of a lazy safetensors slice."""
    return value.shape if hasattr(value, "shape") else value.get_shape()


def _v41_tensor_dtype(value: Any) -> str:
    """Dtype spelling of a loaded tensor or of a lazy safetensors slice."""
    return str(value.dtype) if hasattr(value, "dtype") else str(value.get_dtype())


def _v41_structural_key(key: str) -> str:
    """A forwarded key under the name the *model* would know it by.

    Three rewrites, all of them things V4's walk does rather than things V4.1
    changed: flat mHC names become structured ones, a block scale spelled as a
    sub-module of a bare ``nn.Parameter`` becomes its flat sibling, and a fused
    sub-module's name becomes the fused module's. Everything else is already a
    model name.

    Relocated leaf parameters (``_V41_RELOCATED_PARAMS``) are deliberately *not*
    rewritten here: their destination is not reachable by name at all, so
    ``_v41_load_coverage`` resolves them as attributes instead.
    """
    for attr in _V41_FLAT_HC_ATTRS:
        suffix = f"_{attr}"
        if key.endswith(suffix):
            head = key[: -len(suffix)]
            if head.rsplit(".", 1)[-1] in _V41_FLAT_HC_STEMS:
                return f"{head}.{attr}"
    for suffix, flat in _V41_FLAT_SCALE_PARAMS.items():
        if key.endswith(f".{suffix}"):
            return f"{key[: -len(suffix)]}{flat}"
    parts = key.split(".")
    if len(parts) > 1 and parts[-2] in _V41_FUSED_MODULES:
        parts[-2] = _V41_FUSED_MODULES[parts[-2]]
        return ".".join(parts)
    return key


@dataclass
class DeepseekV41LoadCensus:
    """Where every tensor the checkpoint shipped went.

    ``consumed + ignored`` has to equal ``total``. The point is not the number: it
    is that a tensor cannot leave the accounting without someone naming a reason,
    which is the only way a renamed or newly-added checkpoint key shows up as a
    failure instead of as a zero-initialized parameter.
    """

    total: int = 0
    folded: int = 0
    forwarded: int = 0
    # Dense stems whose 32x32 fp8 pair was requantized to 128x128 rather than
    # dequantized to bf16. Not part of the closing identity -- both halves of a
    # requantized stem are forwarded, so they are already counted in `forwarded` --
    # but reported because it is the one line that says which of the two dense
    # paths a given run actually took.
    requantized: int = 0
    checked: Dict[str, Tuple[int, DeepseekV41QuantLayout]] = field(default_factory=dict)
    ignored: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def ignore(self, key: str, reason: Tuple[str, str]) -> None:
        self.ignored.setdefault(reason, []).append(key)

    @property
    def ignored_count(self) -> int:
        return sum(len(names) for names in self.ignored.values())

    def check(self, role: str, layout: DeepseekV41QuantLayout) -> None:
        """Record that one (weight, scale) pair passed ``assert_weight_layout``."""
        seen, _ = self.checked.get(role, (0, layout))
        self.checked[role] = (seen + 1, layout)

    @property
    def checked_pairs(self) -> int:
        return sum(count for count, _ in self.checked.values())

    @property
    def consumed(self) -> int:
        """Folded into a dequantized weight, or forwarded under a model name."""
        return self.folded + self.forwarded

    def render(self, examples: int = 2) -> str:
        closes = self.consumed + self.ignored_count == self.total
        lines = [
            "DeepSeek-V4.1 checkpoint census:",
            f"  raw checkpoint tensors                    {self.total:>7}",
            f"  consumed                                  {self.consumed:>7}",
            f"    .scale folded into its weight           {self.folded:>7}",
            f"    forwarded under a model name            {self.forwarded:>7}",
            f"  deliberately ignored                      {self.ignored_count:>7}",
        ]
        lines.append(
            "  dense fp8 path: "
            + (
                f"requantized to 128x128, {self.requantized} stem(s)"
                if self.requantized
                else "dequantized to bf16 (TRTLLM_V41_DENSE_FP8 unset)"
            )
        )
        for (pattern, reason), names in sorted(self.ignored.items()):
            lines.append(f"    {len(names):>7}  {pattern}")
            lines.append(f"              reason: {reason}")
            lines.append(f"              e.g.    {', '.join(sorted(names)[:examples])}")
        lines.append(
            f"  consumed + ignored == {self.total:<7}             {'OK' if closes else 'MISMATCH'}"
        )
        lines.append(f"  quantized (weight, scale) pairs checked    {self.checked_pairs:>7}")
        # Print the block each role was *held to*, because the number that decides
        # whether the experts were read as MXFP4 or as NVFP4 is the element extent,
        # and on disk it is invisible: MXFP4's 32-element block packs two nibbles
        # per byte and so derives 16 -- exactly NVFP4's block. A census that only
        # says "46412 pairs checked" cannot distinguish the two.
        for role, (count, layout) in sorted(self.checked.items()):
            lines.append(
                f"    {count:>7}  {role}: {layout.weight_dtype} in "
                f"{layout.element_block} element blocks, on disk "
                f"{layout.storage_dtype} {layout.stored_block}"
            )
        return "\n".join(lines)

    def verify(self) -> None:
        if self.consumed + self.ignored_count != self.total:
            raise ValueError(
                "DeepSeek-V4.1 load census does not close: "
                f"{self.consumed} consumed + {self.ignored_count} ignored != "
                f"{self.total} checkpoint tensors. Some tensor left the load path "
                "without a recorded reason.\n" + self.render()
            )


class _V41ForwardedWeights(dict):
    """The remapped weights, with their key namespace pinned at construction.

    ``DeepseekV4WeightLoader._load_weights_impl`` hands this dict to
    ``ConsumableWeightsDict.take_ownership``, which wraps it and then *deletes*
    from it as modules consume their weights -- so by the time the walk is over
    there is nothing left to audit. Snapshotting the keys here, and recording the
    defaults the loader synthesizes afterwards, is what lets the audit run after
    the walk instead of having to instrument it.
    """

    def __init__(self, mapping: Dict[str, Any], census: DeepseekV41LoadCensus):
        super().__init__(mapping)
        self.census = census
        self.forwarded_keys = frozenset(self)
        self.synthesized_keys: set = set()

    def _record(self, key: str) -> None:
        if key not in self.forwarded_keys:
            self.synthesized_keys.add(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._record(key)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs) -> None:
        other = dict(*args, **kwargs)
        for key in other:
            self._record(key)
        super().update(other)

    @property
    def all_keys(self) -> set:
        return set(self.forwarded_keys) | self.synthesized_keys


def _v41_load_coverage(model: nn.Module, keys) -> Tuple[List[str], List[str]]:
    """``(keys no module would load, parameters no key would fill)``.

    Both directions are decided by name, against the modules that were actually
    built. A module exposing ``load_weights`` fuses several checkpoint tensors
    into its own parameters (``Linear`` stacking a TP shard, ``ConfigurableMoE``
    stacking 384 experts into ``w3_w1_weight``), so its subtree is matched by
    prefix; everything else has to name a parameter or buffer exactly. The root
    module is excluded -- it also has ``load_weights``, and accepting its prefix
    would accept every key and audit nothing.

    The one exception to "decided by name" is ``_V41_RELOCATED_PARAMS``, whose
    destination no name reaches; those are resolved as attributes and checked for
    having actually been populated.
    """
    modules = dict(model.named_modules())
    consumers = {
        name for name, module in modules.items() if name and hasattr(module, "load_weights")
    }
    parameters = dict(model.named_parameters())
    targets = set(parameters) | set(dict(model.named_buffers()))

    def enclosing(name: str) -> List[str]:
        parts = name.split(".")
        return [
            prefix
            for prefix in (".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1))
            if prefix in consumers
        ]

    def relocated_and_populated(name: str) -> bool:
        """``name`` names a relocated leaf whose tensor is in place."""
        parts = name.split(".")
        child = _V41_RELOCATED_PARAMS.get(parts[-1])
        if child is None or len(parts) < 2:
            return False
        inner = getattr(modules.get(".".join(parts[:-1])), child, None)
        return getattr(inner, parts[-1], None) is not None

    unexpected: List[str] = []
    fed: set = set()
    for key in keys:
        structural = _v41_structural_key(key)
        owners = enclosing(structural)
        fed.update(owners)
        if relocated_and_populated(structural):
            continue
        if structural not in targets and not owners:
            unexpected.append(key)

    structural_keys = {_v41_structural_key(key) for key in keys}
    unfed = [
        name
        for name in parameters
        if name not in structural_keys and not any(owner in fed for owner in enclosing(name))
    ]
    return sorted(unexpected), sorted(unfed)


def _remap_deepseek_v41_checkpoint_keys(
    weights: Dict,
    num_hidden_layers: int,
    kv_lora_rank: int = 448,
    *,
    keep_mtp_layers: int = 0,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> _V41ForwardedWeights:
    """V4.1 checkpoint keys -> model parameter keys, with a census of both.

    A pre-pass over the raw keys followed by V4's remap, rather than a fork of
    it: the attention / FFN / MTP / compressor renames are identical, so the only
    V4.1-specific work is what the pre-pass does.

    1. **Re-layout the dense fp8 weights.** V4.1 stores them e4m3 with one ue8m0
       exponent per 32x32 block; V4, and every block-scaled path in ``Linear``,
       assumes 128x128 fp32 block scales. The block size is not a parameter anyone
       passes down, so the 32 cannot simply be threaded through the quantization
       stack. There are two ways out and both are implemented:

       * **bf16 (default).** Dequantize here and drop the ``.scale`` siblings.
         ``ModelConfig._build_deepseek_v41_quant_config`` then leaves the global algo
         unset, so the modules being loaded expect bf16. This is the configuration
         GSM8K was measured on (94.39 / 93.86 at n=1319) and it trades dense GEMM
         throughput for having no re-layout to be wrong about.
       * **fp8 128x128** (``TRTLLM_V41_DENSE_FP8=1``, ``dense_fp8_requant_enabled``).
         Requantize the stems in ``_V41_REQUANT_STEM_TAILS`` to the 128x128 layout
         and emit both halves, so ``_build_deepseek_v41_quant_config`` can declare
         ``FP8_BLOCK_SCALES``. Nearly lossless -- the scale is rounded *up* to a
         power of two, which moves exponents and not mantissas, so the only
         deviation is e4m3 subnormal flush more than 9 binades below a tile's max
         (see ``_pow2_tile_scale``).

       The two halves read the *same* switch on purpose. If the remap emitted fp8
       while the quant config said bf16 (or the reverse), the failure would be a
       dtype mismatch deep inside the module walk rather than anything a verdict
       line reports, which is why the switch lives in ``configs/deepseek_v41.py``
       and not in either module that consumes it.

       Detection is by *name* -- a ``<stem>.weight`` with a sibling
       ``<stem>.scale`` -- not by dtype, because the two exclusions in
       ``_V41_NO_DEQUANT`` are also fp8-with-scale and must survive.

    2. **Rename the engram sub-modules** (``embed``/``wkv``/``q_weight``/``k_weight``).
       V4's ``_rename_layer_subkey`` falls through with ``return rest`` for
       anything it does not recognize, so ``layers.N.engram.*`` reaches
       ``model.layers.N.engram.*`` unchanged once the leaf names are right --
       including ``.scale``, which the sharded table wants under that exact name.

    3. **Drop what this model does not implement**: the vision tower and MTP
       heads 1 and 2 -- V4's
       remap only routes ``mtp.0.``, and V4.1's three MTP layers are
       heterogeneous (only ``mtp.2`` carries the Markov and confidence heads),
       so speculative decoding is a separate piece of work. What is dropped is
       decided by ``_v41_ignore_reason`` alone, so every dropped tensor carries a
       name pattern and a reason into the returned census.

    4. **Check the quantized layouts while the scales are still here.** Every
       ``<stem>.weight`` with a ``<stem>.scale`` sibling goes through
       ``assert_weight_layout`` before anything is dequantized or dropped. This is
       the only point in the load where both halves of a quantized tensor are in
       hand, and it raises rather than warns: the routed experts are MXFP4 with an
       element block of 32, one scale per 16 *stored* int8 lanes, and reading that
       as NVFP4's block of 16 yields plausible weights and fluent, wrong text.

    Returns a ``_V41ForwardedWeights`` -- a plain dict of model-named tensors that
    also carries the census and remembers its own key namespace, so the loader can
    audit coverage after the module walk has drained it.
    """
    census = DeepseekV41LoadCensus(total=len(weights))

    # One pass to classify every raw key and to check every quantized layout.
    # Classification comes first so that an ignored subtree's `.scale` is never
    # checked against a role expectation that was never meant to cover it.
    requant = dense_fp8_requant_enabled()
    dequant_stems = set()
    requant_stems = set()
    for key in weights:
        reason = _v41_ignore_reason(key, keep_mtp_layers=keep_mtp_layers)
        if reason is not None:
            census.ignore(key, reason)
            continue
        if not key.endswith(".scale"):
            continue
        stem = key[: -len(".scale")]
        weight_key = f"{stem}.weight"
        if weight_key not in weights:
            continue
        layout = assert_weight_layout(
            weight_key,
            _v41_tensor_shape(weights[weight_key]),
            _v41_tensor_shape(weights[key]),
            storage_dtype=_v41_tensor_dtype(weights[weight_key]),
            quantization_config=quantization_config,
        )
        census.check(quant_role_for_weight_key(weight_key), layout)
        if any(marker in f".{key}" for marker in _V41_NO_DEQUANT):
            continue
        if requant and stem.endswith(_V41_REQUANT_STEM_TAILS):
            requant_stems.add(stem)
        else:
            dequant_stems.add(stem)

    # A requantized stem is asked for twice -- once for `.weight`, once for
    # `.scale` -- and the conversion round-trips through the GPU, so the pair is
    # computed on first touch and popped on second. Popping rather than
    # accumulating matters: holding all ~320 requantized dense weights at once is
    # the ~13 GiB `_dequantize_block32` explicitly refuses to hold.
    requant_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _requant_half(stem: str, want_scale: bool) -> torch.Tensor:
        pair = requant_pairs.pop(stem, None)
        if pair is None:
            pair = _requantize_dense_stem(weights[f"{stem}.weight"], weights[f"{stem}.scale"])
            requant_pairs[stem] = pair
            census.requantized += 1
        return pair[1] if want_scale else pair[0]

    pre: Dict[str, torch.Tensor] = {}
    for key, value in weights.items():
        if _v41_ignore_reason(key, keep_mtp_layers=keep_mtp_layers) is not None:
            continue

        stem = key.rsplit(".", 1)[0]
        if stem in dequant_stems:
            if key.endswith(".scale"):
                census.folded += 1
                continue
            value = _dequantize_block32(value, weights[f"{stem}.scale"])
        elif stem in requant_stems:
            # Both halves are replaced, so each raw key still maps to exactly one
            # `pre` entry and nothing is folded. The `.scale` is emitted under its
            # checkpoint name and V4's remap renames it to `weight_scale_inv`
            # (`_rename_deepseek_v4_attn_subkey`, `_rename_deepseek_v4_ffn_subkey`)
            # -- which is why the requant allow-list only names stems under `attn.`
            # and `ffn.`, the two subtrees that rename actually reaches.
            value = _requant_half(stem, want_scale=key.endswith(".scale"))

        pre[_rename_deepseek_v41_engram_key(key)] = value

    if requant_pairs:
        # Every requantized stem has exactly a `.weight` and a `.scale` in the
        # checkpoint (that pairing is what put it in `requant_stems`), so both
        # halves must have been drawn. A leftover means one half was ignored or
        # renamed out from under the other, which would leave a Linear holding a
        # scale for a weight it never received.
        raise ValueError(
            "DeepSeek-V4.1 requantization left one half of "
            f"{len(requant_pairs)} stem(s) unconsumed, e.g. "
            f"{', '.join(sorted(requant_pairs)[:3])}"
        )

    # `_rename_deepseek_v41_engram_key` renames one leaf, so it is injective and
    # `len(pre)` is exactly the number of raw keys that were forwarded.
    census.forwarded = len(pre)
    census.verify()
    logger.info(census.render())

    return _V41ForwardedWeights(
        _remap_deepseek_v4_checkpoint_keys(
            pre, num_hidden_layers=num_hidden_layers, kv_lora_rank=kv_lora_rank
        ),
        census=census,
    )


def _rename_deepseek_v41_engram_key(key: str) -> str:
    """``layers.N.engram.<leaf>...`` -> TRT-LLM's engram sub-module names."""
    marker = ".engram."
    idx = key.find(marker)
    if idx < 0:
        return key
    head = key[: idx + len(marker)]
    rest = key[idx + len(marker) :]
    leaf, sep, tail = rest.partition(".")
    new_leaf = _ENGRAM_SUBKEY_RENAME.get(leaf, leaf)
    return f"{head}{new_leaf}{sep}{tail}" if sep else f"{head}{new_leaf}"


def _dequantize_block32(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp8 e4m3 + ue8m0 block scales -> bf16, on the GPU, back on the host.

    ``weight_dequant`` is a Triton kernel and needs both operands resident, so
    the tensor makes a round trip. Returning to the host rather than leaving the
    result on the device is deliberate: the loader's own TP split copies each
    shard to the GPU anyway, and holding all ~13 GiB of dequantized dense
    weights on the device at once -- on top of the shard being built -- would
    compete with the model's own allocation for no benefit.
    """
    x = weight[:] if not isinstance(weight, torch.Tensor) else weight
    s = scale[:] if not isinstance(scale, torch.Tensor) else scale
    y = weight_dequant(
        x.contiguous().cuda(),
        s.float().contiguous().cuda(),
        block_size=_V41_FP8_BLOCK_SIZE,
    )
    return y.to(torch.bfloat16).cpu()


def _pow2_tile_scale(amax: torch.Tensor) -> torch.Tensor:
    """Per-tile multiplier ``S = 2**ceil(log2(amax / 448))``, one per 128x128 tile.

    Rounding the scale **up** to a power of two is what makes the requantization in
    ``_requantize_block128`` nearly lossless, and it buys two separate properties:

    * **No mantissa motion.** Dividing an e4m3 value by a power of two decrements
      its exponent field and leaves all three mantissa bits alone, so every value
      that stays in the e4m3 normal range survives ``fp8 -> bf16 -> fp8`` bit-exactly.
      The obvious alternative ``S = amax / 448`` is not a power of two, perturbs
      every mantissa, and reproduces only ~16-21% of elements exactly.
    * **No clamp needed.** ``ceil`` means ``S >= amax / 448``, hence
      ``max|w / S| <= 448`` by construction -- including on tiles where ``S`` lands
      below some constituent 32-block's own on-disk scale.
    """
    ratio = amax / _E4M3_MAX
    # An all-zero tile has amax == 0, and log2(0) -> -inf -> ceil -> -inf ->
    # exp2 -> 0.0: finite at every step, no NaN, but a zero scale would make the
    # reconstruction 0/0. Give it S = 1 instead: any positive scale reproduces zero
    # exactly, and 1.0 keeps the emitted scale tensor readable.
    #
    # Written with `where` rather than boolean-mask assignment because the census
    # tests audit the whole load path on `meta` tensors, and masked indexing needs
    # `nonzero()`, which meta has no data-independent implementation for.
    return torch.where(
        ratio > 0,
        torch.exp2(torch.ceil(torch.log2(ratio))),
        torch.ones_like(ratio),
    )


def _requantize_block128(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """A bf16 dense weight -> ``(fp8_e4m3, fp32 128x128 block scales)``.

    Paired with ``_dequantize_block32``: that reads V4.1's on-disk 32x32 ue8m0
    layout, this writes the 128x128 layout ``Linear``'s ``FP8_BLOCK_SCALES`` path
    already implements. The emitted scale is the **multiplier** -- ``w ~= q * S`` --
    matching what ``linear.py`` stores under the (misleadingly named)
    ``weight_scale_inv``.

    Deviation from the input is bounded by ``S * 2**-9`` per element and is entirely
    e4m3 subnormal flush: within a tile whose own max reaches the e4m3 ceiling the
    downshift is exact, and only values more than 9 binades below the tile max fall
    into the subnormal range where mantissa bits are lost. See
    ``pending_patches/FP8_DENSE_PATH_DESIGN.md``.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    block = _V41_REQUANT_BLOCK
    m, n = weight.shape
    tiles_m = (m + block - 1) // block
    tiles_n = (n + block - 1) // block
    w = weight.float()
    pad_m, pad_n = tiles_m * block - m, tiles_n * block - n
    if pad_m or pad_n:
        # Zero padding cannot raise a tile's amax, so the padded tile's scale is
        # the same one the real elements would have produced on their own.
        w = torch.nn.functional.pad(w, (0, pad_n, 0, pad_m))
    amax = w.reshape(tiles_m, block, tiles_n, block).abs().amax(dim=(1, 3))
    scale = _pow2_tile_scale(amax)
    expanded = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    q = (w / expanded)[:m, :n].to(torch.float8_e4m3fn)
    return q.contiguous(), scale.contiguous()


def _requantize_dense_stem(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4.1's on-disk 32x32 fp8 pair -> the 128x128 fp8 pair ``Linear`` expects.

    Dequantize-then-requantize rather than a direct 32 -> 128 scale merge: the merge
    would still have to re-round every element that its 32-block's scale placed above
    the coarser tile's ceiling, so it is the same arithmetic with a hand-rolled
    kernel in the middle. ``_dequantize_block32`` already runs on the GPU and returns
    to the host; the requantization runs on whatever device it is handed.
    """
    return _requantize_block128(_dequantize_block32(weight, scale))


class DeepseekV41WeightLoader(DeepseekV4WeightLoader):
    """V4's loader, plus the three load-time audits V4.1's checkpoint needs.

    * **Layout** -- every quantized tensor's on-disk block extent is checked
      against its role before anything is dequantized, and a mismatch raises.
      V4.1 ships three different quantized layouts under one
      ``quantization_config`` (routed experts MXFP4 element block 1x32 stored
      1x16, the Engram tables fp8 e4m3 1x32, everything else fp8 e4m3 32x32),
      and ``weight_block_size`` in the config describes only the third.
    * **Census** -- ``consumed + deliberately_ignored`` must equal the number of
      tensors the checkpoint shipped, every ignored group with a name pattern and
      a reason. See ``DeepseekV41LoadCensus``.
    * **Coverage** -- after the module walk, every forwarded key must have had a
      consumer and every parameter a source. V4's walk visits modules and asks
      for their weights, so a key nothing asks for is dropped in silence: a
      rename that misses becomes a zero-initialized projection, not an error.

    All three raise. A warning is worse than nothing at 475 GiB and 96085
    tensors, where the log has long scrolled past by the time the output is wrong.
    """

    def __init__(self, model, is_draft_model: bool = False):
        super().__init__(model, is_draft_model=is_draft_model)
        self.census: Optional[DeepseekV41LoadCensus] = None
        self._forwarded: Optional[_V41ForwardedWeights] = None

    def remap_checkpoint_keys(
        self, weights: Dict, *, num_hidden_layers: int, kv_lora_rank: int = 448
    ) -> _V41ForwardedWeights:
        """``DeepseekV4WeightLoader._load_weights_impl``'s remap hook.

        Bound rather than static because the census needs two things only the
        loader knows: the checkpoint's ``quantization_config`` (which layout the
        dense role expects) and how many MTP layers the model actually built.
        """
        pretrained = self.model_config.pretrained_config
        text = getattr(pretrained, "text_config", pretrained)
        # What the *model* built, not what the checkpoint ships: V4's remap only
        # routes `mtp.0.`, so one is the ceiling even if more were constructed.
        built = len(getattr(getattr(self.model, "model", None), "layers", ()))
        forwarded = _remap_deepseek_v41_checkpoint_keys(
            weights,
            num_hidden_layers=num_hidden_layers,
            kv_lora_rank=kv_lora_rank,
            keep_mtp_layers=min(1, max(0, built - int(num_hidden_layers))),
            quantization_config=getattr(text, "quantization_config", None),
        )
        self.census = forwarded.census
        self._forwarded = forwarded
        return forwarded

    def load_weights(self, weights: Dict, skip_modules: List[str] = []):
        result = super().load_weights(weights, skip_modules=skip_modules)
        self.assert_load_complete()
        return result

    def assert_load_complete(self) -> None:
        """Raise unless every forwarded key had a consumer and every parameter a source.

        A no-op when the checkpoint arrived in HF-style ``model.*`` names, which
        skip the remap entirely and so have no census to audit against.
        """
        if self._forwarded is None:
            return
        unexpected, unfed = _v41_load_coverage(self.model, self._forwarded.all_keys)
        if not unexpected and not unfed:
            logger.info(
                "DeepSeek-V4.1 load coverage: "
                f"{len(self._forwarded.forwarded_keys)} forwarded + "
                f"{len(self._forwarded.synthesized_keys)} synthesized keys all had a "
                "consumer; every model parameter had a source."
            )
            # The census is rendered on the way out as well as on the way to a
            # raise: "every tensor is accounted for" is a claim a reader has to be
            # able to check, and a 475 GiB load is not something anyone reruns to
            # find out where the count went.
            if self.census is not None:
                logger.info("\n" + self.census.render())
            return
        detail = []
        if unexpected:
            detail.append(
                f"{len(unexpected)} checkpoint tensor(s) no module would load, "
                f"e.g. {unexpected[:8]}"
            )
        if unfed:
            detail.append(
                f"{len(unfed)} model parameter(s) no checkpoint tensor would fill, e.g. {unfed[:8]}"
            )
        raise ValueError(
            "DeepSeek-V4.1 weight load is incomplete: "
            + "; ".join(detail)
            + ".\n"
            + (self.census.render() if self.census is not None else "")
        )


def _candidate_prefilter_is_wired(model_config: ModelConfig[PretrainedConfig]) -> bool:
    """Whether this build actually runs both prefilter levels for this config.

    Deliberately asked of the *lowered* :class:`DeepSeekV4Params` rather than of
    the checkpoint config, because that is the value the ``Indexer`` will read:
    the two can disagree, and the interesting direction is a checkpoint that
    carries the three fields against a ``to_sparse_params`` that does not forward
    them. Reading the config directly would call the feature wired and drop the
    length refusal while the runtime silently skipped level one.
    """
    sparse_config = model_config.sparse_attention_config
    if sparse_config is None:
        return False
    params = sparse_config.to_sparse_params(pretrained_config=model_config.pretrained_config)
    return candidate_prefilter_is_on(
        getattr(params, "candidate_source_layer_id", None),
        getattr(params, "candidate_topk_blocks", 0) or 0,
        getattr(params, "candidate_block_size", 0) or 0,
    )


def _candidate_source_ratio(text: PretrainedConfig, candidate_source: int) -> int:
    """The compress ratio of the layer that publishes level one, defaulting to 1.

    Level one counts *compressed* positions, so the length it stays inert up to
    scales with the source layer's pooling factor. The release's source is layer
    20, in the ratio-1 band, where compressed and raw lengths coincide -- so this
    returns 1 there and the bound is unchanged. A config that moved the source
    into the pooled band would otherwise be refused at half the length it is
    actually exact to.
    """
    ratios = getattr(text, "compress_ratios", None)
    if not ratios or not 0 <= candidate_source < len(ratios):
        return 1
    return max(1, int(ratios[candidate_source]))


def _assert_candidate_prefilter_inert(model_config: ModelConfig[PretrainedConfig]) -> None:
    """Refuse a sequence length at which the missing candidate prefilter would matter.

    V4.1 runs a two-level selection: layer ``candidate_source_layer_id`` (20 in the
    release) publishes the top ``candidate_topk_blocks`` blocks of
    ``candidate_block_size`` positions each, and every *later* index source masks
    its own scores with that set before taking its top-k.

    Skipping it is exact only while the mask covers everything a consumer could
    have selected anyway, i.e. while a sequence has no more candidate positions
    than ``candidate_topk_blocks * candidate_block_size`` (16384 in the release).
    Past that the reference scores a strict subset of what we score, so our top-k
    can contain positions the reference excluded -- a silent accuracy loss that
    grows with length and is invisible in any short-context test.

    Hence a refusal at construction rather than a warning at the first long
    request: the check is on the configured ceiling, so a run either cannot start
    or is provably in the inert regime for its whole life. ``max_seq_len`` unset
    means the caller has not committed to a ceiling, and the model's own
    ``max_position_embeddings`` (1 Mi) is far past the threshold, so that is
    refused too rather than assumed short.
    """
    if _candidate_prefilter_is_wired(model_config):
        # Both levels run, so the length ceiling this function exists to enforce
        # no longer binds. Kept as a function rather than deleted: the refusal is
        # still the correct behaviour for any configuration where the wiring is
        # off, and "is it wired" is a property of the build, not of the request.
        return

    sparse_config = model_config.sparse_attention_config
    pretrained = model_config.pretrained_config
    text = getattr(pretrained, "text_config", pretrained)
    candidate_source = getattr(text, "candidate_source_layer_id", None)
    if candidate_source is None or sparse_config is None:
        return

    blocks = int(getattr(text, "candidate_topk_blocks", 0) or 0)
    block_size = int(getattr(text, "candidate_block_size", 0) or 0)
    if blocks <= 0 or block_size <= 0:
        return
    # Not the bare `blocks * block_size` product: the bound is on *compressed*
    # positions, so a pooled candidate source is inert to a proportionally longer
    # raw sequence. Identity at the release's ratio-1 source (2048 * 8 = 16384).
    inert_up_to = inert_up_to_positions(
        blocks, block_size, _candidate_source_ratio(text, int(candidate_source))
    )

    max_seq_len = model_config.max_seq_len
    if max_seq_len is not None and int(max_seq_len) <= inert_up_to:
        return
    configured = "unset" if max_seq_len is None else f"{int(max_seq_len)}"
    raise ValueError(
        "DeepSeek-V4.1's two-level candidate prefilter is not wired up for this "
        f"configuration, so this model is only exact up to max_seq_len={inert_up_to} "
        f"(candidate_topk_blocks={blocks} x candidate_block_size={block_size}); "
        f"got max_seq_len={configured}. Above that length, layer "
        f"{int(candidate_source)} would restrict which positions the later index "
        "sources may select and skipping it silently widens their candidate set. "
        f"Set max_seq_len <= {inert_up_to} explicitly, or check that the sparse "
        "attention config forwards candidate_source_layer_id / "
        "candidate_topk_blocks / candidate_block_size into DeepSeekV4Params -- the "
        "Indexer reads them from there, and this refusal only fires when it will "
        "not see them."
    )


@register_auto_model("DeepseekV41ForCausalLM")
class DeepseekV41ForCausalLM(DeepseekV4ForCausalLM):
    """DeepSeek-V4.1.

    Known unimplemented, and deliberately so for the first correctness
    milestone -- each is inert unless the corresponding feature is switched on:

    * **Speculative decoding.** ``model_nextn`` must be 0. V4.1's three MTP
      layers are heterogeneous (only ``mtp.2`` carries the Markov draft head and
      the confidence head) and the remap drops all but the first. The DSpark
      capture *site* is already correct in
      ``DeepseekV41DecoderLayer.forward``, so the remaining work is the MTP
      construction, not the capture.
    * **Vision.** The tower and projector are dropped at remap; text-only.
    * **Dense fp8.** Dequantized to bf16 at load (see
      ``_remap_deepseek_v41_checkpoint_keys``).
    """

    model_cls = DeepseekV41Model
    weight_loader_cls = DeepseekV41WeightLoader

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        # Both of these are refusals rather than silent degradations: each one
        # would otherwise produce numerically wrong output that looks like a
        # model bug.
        #
        # Pipeline parallelism: the lagged `pre` is a second piece of
        # cross-layer state, carried on `HCState.pre_mix` alongside the residual.
        # `DeepseekV4Model.forward`'s PP branch passes hidden states between
        # ranks and has no channel for it, so a PP split would restart every
        # rank's stream from the identity seed. The release target is TP8/EP8 on
        # one node, so nothing needs it yet.
        if model_config.mapping is not None and model_config.mapping.has_pp():
            raise ValueError(
                "DeepSeek-V4.1 does not support pipeline parallelism: its mHC "
                "stream carries a lagged `pre` across layer boundaries "
                "(HCState.pre_mix) that the PP hidden-state exchange does not "
                "transport. Use tensor/expert parallelism instead "
                f"(got pp_size={model_config.mapping.pp_size})."
            )
        # Speculative decoding: V4.1's three MTP layers are heterogeneous, the
        # weight remap keeps only `mtp.0.`, and `modeling_speculative.py` has no
        # `deepseek_v41` draft case. The DSpark capture site in
        # `DeepseekV41DecoderLayer.forward` is already correct, so what is
        # missing is the draft-model construction, not the capture.
        if model_config.spec_config is not None:
            raise ValueError(
                "DeepSeek-V4.1 does not support speculative decoding yet: its "
                "MTP layers are not constructed (the checkpoint's mtp.1/mtp.2 "
                "tensors are dropped at load) and no draft model is registered "
                "for model_type 'deepseek_v41'. Run with speculative decoding "
                f"disabled (got spec_config={type(model_config.spec_config).__name__})."
            )
        _assert_candidate_prefilter_inert(model_config)
        super().__init__(model_config)

    @classmethod
    def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> dict:
        """V4's defaults, minus window-KV reuse when bounded replay is switched on.

        The two are the same policy read in opposite directions, so they cannot
        both hold: V4's ``enable_swa_scratch_reuse=True`` spends memory keeping
        sliding-window KV readable by a later request, and §3.2.2's replayed window
        KV is approximate for every token but the last and must therefore never be
        read by one. Flipping the default here rather than asking the deployment to
        set both is what makes the switch a single decision; ``plan_decoder_replay``
        still reads the constructed manager, so a manager that kept reuse anyway
        wins and replay is refused instead of corrupting a prefix.
        """
        defaults = super().get_model_defaults(llm_args)
        if decoder_bounded_replay_enabled():
            defaults.setdefault("kv_cache_config", {})["enable_swa_scratch_reuse"] = False
        return defaults

    def forward(
        self,
        *args,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Tell the model body whether anything downstream reads non-final rows.

        ``return_context_logits`` cannot simply be inspected inside the body:
        ``SpecDecOneEngineForCausalLM.forward`` takes it as a named parameter and
        consumes it itself, so it never appears in the ``**kwargs`` forwarded to
        ``self.model``. Hence a separate key, and hence it is *derived* here rather
        than read from the metadata -- the body must not have to guess.
        """
        kwargs["all_token_states_required"] = bool(return_context_logits)
        return super().forward(*args, return_context_logits=return_context_logits, **kwargs)
