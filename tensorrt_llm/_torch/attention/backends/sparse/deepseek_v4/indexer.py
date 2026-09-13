# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 indexer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.utils import fp8_utils

from ..dsa.indexer import HAS_FAST_HADAMARD, Indexer, rotate_activation
from .compressor import (
    Compressor,
    KVCacheDtype,
    postprocess_scatter_compressed,
    resolve_kv_cache_dtype,
)
from .params import DeepSeekV4Params

if TYPE_CHECKING:
    from ..dsa.metadata import DSAtrtllmAttentionMetadata
    from .metadata import DeepseekV4TrtllmAttentionMetadata


class DeepseekV4Indexer(Indexer):
    # Whether the key branch can run concurrently with the Q branch on the aux
    # stream. True for V4, whose keys come from an indexer-local compressor over
    # `hidden_states`; False for V4.1, whose keys depend on the main compressor.
    supports_overlapped_prepare = True

    def _decode_indexer_block_table(self, metadata) -> torch.Tensor:
        """This layer's INDEXER_COMPRESS page table, selected by compression ratio.

        V4 has one indexer pool, so this returns what the base class would. V4.1
        has one per ratio -- the pools differ in page scale -- so the ratio is the
        only thing that distinguishes them and a ratio-blind read would score
        against another pool's pages.
        """
        return metadata.indexer_block_table(self.compress_ratio)

    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_params: DeepSeekV4Params,
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__(
            quant_config,
            pos_embd_params,
            mla_params,
            skip_create_weights_in_init,
            sparse_params,
            dtype,
            compress_ratio,
            layer_idx,
            aux_stream,
        )
        # Keep the checkpoint's FP8 quantization while deriving the scale view
        # consumed by the fused CuTe DSL indexer-Q projection.
        self.wq_b.use_indexer_q_cutedsl_fusion = True
        # Override base Indexer.weights_proj to bf16 (matches V4 checkpoint).
        self.weights_proj = Linear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_dim,
            is_neox=False,
        )
        # `args.norm_eps` in the reference; 1e-6 for V4 but 1e-20 for V4.1.
        rms_norm_eps = sparse_params.rms_norm_eps
        index_head_dim = sparse_params.index_head_dim
        indexer_mla_params = MLAParams(
            hidden_size=mla_params.hidden_size,
            qk_rope_head_dim=mla_params.qk_rope_head_dim,
            qk_nope_head_dim=index_head_dim - mla_params.qk_rope_head_dim,
        )
        # Map the user-facing FP4 knob ("fp8" / "fp4") onto the Compressor's
        # cache layout preset string. The Compressor's preset namespace also
        # covers main-attention layouts (bf16 / fp8_pertensor), which the
        # indexer doesn't use, so the translation lives here at the
        # boundary instead of leaking through the user-facing config.
        self.indexer_k_dtype = sparse_params.indexer_k_dtype
        compressor_preset = "mxfp4" if self.indexer_k_dtype == "fp4" else "fp8_blockwise"
        self.indexer_cache_dtype = resolve_kv_cache_dtype(compressor_preset)
        self._build_key_path(
            indexer_mla_params,
            layer_idx,
            compress_ratio,
            rms_norm_eps,
            skip_create_weights_in_init,
            pos_embd_params,
            dtype,
            compressor_preset,
        )
        self.indexer_start_event = torch.cuda.Event()
        self.weights_proj_event = torch.cuda.Event()
        self.k_cache_update_event = torch.cuda.Event()

    def _build_key_path(
        self,
        indexer_mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        rms_norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams],
        dtype: Optional[torch.dtype],
        compressor_preset: KVCacheDtype | str,
    ) -> None:
        """Build whatever produces this indexer's keys.

        For V4 that is a ``Compressor`` of the indexer's own, pooling hidden
        states down to ``index_head_dim``: the checkpoint's ``indexer.wk`` /
        ``indexer.k_norm`` are vestigial there (the loader ones/zeros-fills them,
        ``modeling_deepseekv4.py:626``). V4.1 instead derives index keys from the
        *main* compressor's pre-RoPE latent with a real ``wk``, so it has no
        indexer compressor at all -- hence the hook rather than an unconditional
        construction. See :class:`DeepseekV41Indexer`.
        """
        self.compressor = Compressor(
            indexer_mla_params,
            layer_idx,
            compress_ratio,
            rms_norm_eps,
            skip_create_weights_in_init,
            pos_embd_params,
            dtype=dtype,
            kv_cache_dtype=compressor_preset,
            is_indexer=True,
            rotate_activation=HAS_FAST_HADAMARD,
        )

    def post_load_weights(self):
        # V4 does not use the V3 fused fp32 wk+weights_proj GEMM, and the
        # base concat would now hit an fp32/bf16 dtype mismatch.
        return

    def _qk_projection_and_rope(self, qr: torch.Tensor, position_ids: torch.Tensor):
        """Project Q and apply RoPE.

        Returns q with layout [num_tokens, n_heads, head_dim] where
        head_dim = nope_dim + rope_dim, RoPE already applied in-place.
        """
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)
        return self._apply_q_rope(q, position_ids)

    def _apply_q_rope(self, q: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply RoPE in-place to a projected indexer Q tensor."""
        # Fused in-place RoPE on the rope portion of each head
        nope_dim = self.head_dim - self.rope_dim
        torch.ops.trtllm.mla_rope_inplace(
            q,
            position_ids.view(-1),
            self.rotary_emb.rotary_cos_sin,
            self.n_heads,
            nope_dim,
            self.rope_dim,
            False,
            self.rotary_emb.is_neox,
        )
        return q

    def _project_and_quantize_q(
        self, qr: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project and quantize Q, using the fused MXFP4 path when supported."""
        use_fused_project_mxfp4 = (
            self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE
            and not HAS_FAST_HADAMARD
            and not self.rotary_emb.is_neox
            and self.head_dim == 128
            and self.rope_dim == 64
            and self.wq_b.has_fp8_block_scales
            and hasattr(self.wq_b, "indexer_q_weight_scale_cutedsl")
            and hasattr(
                torch.ops.trtllm,
                "cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell",
            )
            and qr.dtype == torch.bfloat16
            and is_sm_100f()
        )
        if use_fused_project_mxfp4:
            q_fp4, q_scale = torch.ops.trtllm.cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell(
                qr,
                self.wq_b.weight,
                self.wq_b.indexer_q_weight_scale_cutedsl,
                position_ids.view(-1),
                self.rotary_emb.rotary_cos_sin.view(-1, self.rope_dim),
                self.wq_b.indexer_q_alpha_cutedsl,
                use_tvm_ffi=True,
            )
            return q_fp4.view(-1, self.n_heads, self.head_dim // 2), q_scale.view(
                -1, self.n_heads, 1
            )

        q = self.wq_b(qr).view(-1, self.n_heads, self.head_dim)
        q = self._apply_q_rope(q, position_ids)
        return self._quantize_q(q)

    def _quantize_q(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rotate + quantize (layout matches compressor K: [nope|pe]). After
        # rotate_activation (Hadamard) the nope/rope split becomes a linear
        # mix; treating the row as a (head_dim - rope_dim, rope_dim) split
        # for fused_cat_fp4 is equivalent to a single per-token FP4 quantize
        # because the kernel only cares about the concatenated row. K goes
        # through the same rotation in compressor_postprocess_scatter so
        # layouts match.
        q = rotate_activation(q)
        q = q.view(-1, self.head_dim)
        if self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
            nope_dim = self.head_dim - self.rope_dim
            q_nope, q_pe = q.split([nope_dim, self.rope_dim], dim=-1)
            q_fp8, q_scale = torch.ops.trtllm.fused_cat_fp4(q_nope, q_pe)
            # Two FP4 codes pack into one byte: trailing dim is head_dim // 2.
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim // 2)
            q_scale = q_scale.view(-1, self.n_heads, 1)
        else:
            q_fp8, q_scale = fp8_utils.fp8_quantize_1x128_sf_transpose(
                q, use_ue8m0=self.scale_fmt == "ue8m0"
            )
            q_fp8 = q_fp8.view(-1, self.n_heads, self.head_dim)
            q_scale = q_scale.view(-1, self.n_heads, 1)
        return q_fp8, q_scale

    def _apply_weight_scale(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        # The DeepGEMM FP4 kernel applies per-block q_scale internally, so
        # weights only carry softmax_scale * n_heads^-0.5. `weights_proj` is
        # bf16 to match the V4 checkpoint, and the FP4 branch's only
        # post-projection multiplier is a Python float (`bf16 * float` stays
        # bf16). DeepGEMM's fp8_fp4_(paged_)mqa_logits asserts
        # `weights.scalar_type() == kFloat`, so cast explicitly here. The FP8
        # branch gets upcast for free via the fp32 `q_scale` multiply in
        # `_weight_scale`, so no cast is needed there.
        if self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
            return weights.float() * self.weight_scale_factor
        return self._weight_scale(weights, q_scale)

    def _update_k_cache_if_needed(
        self,
        k_fp8: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
    ) -> None:
        if k_fp8 is None:
            return

        # The MXFP4 compressor's postprocess_scatter performs rotation +
        # quantization + cache scatter in one fused op, so the cache row is
        # already up to date by the time we get here.
        if self.indexer_cache_dtype == KVCacheDtype.MXFP4_BLOCKWISE:
            return

        assert k_scale is not None, "FP8 blockwise indexer cache update requires scale tensor"
        self._update_k_cache(k_fp8, k_scale, metadata)

    def precompute_aux(
        self,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Pre-launch the qr-independent half of the indexer prepare phase.

        Runs weights_proj + internal compressor + k_cache_update on
        ``self.aux_stream`` and records ``weights_proj_event`` /
        ``k_cache_update_event``.  The caller hands the returned tuple back to
        ``forward()`` via the ``pre_aux`` kwarg, which makes the overlapped
        prepare path skip its own aux-stream launch and consume these results
        directly.

        Returns ``None`` when multi-stream mode is off (caller should fall
        back to the normal ``forward()`` call without ``pre_aux``).
        """
        if not (do_multi_stream() and self.aux_stream is not None):
            return None
        self.indexer_start_event.record()
        with torch.cuda.stream(self.aux_stream):
            self.indexer_start_event.wait()
            weights = self.weights_proj(hidden_states)
            self.weights_proj_event.record()
            k_fp8, k_scale = self.compressor(hidden_states, metadata)
            self._update_k_cache_if_needed(k_fp8, k_scale, metadata)
            self.k_cache_update_event.record()
        return (weights, k_fp8, k_scale)

    def _run_overlapped_indexer_prepare(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: torch.Tensor,
        pre_aux: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        """Prepare indexer inputs by splitting independent work across two streams.

        The current stream owns the Q path.  The auxiliary stream starts from
        the recorded launch point and owns weights projection, compressor, and
        the K-cache update.

        When ``pre_aux`` is provided the aux-stream work has already been
        launched (and ``weights_proj_event`` / ``k_cache_update_event``
        recorded) by ``precompute_aux``; the aux-stream block here is skipped
        and the precomputed tensors are consumed directly.

        Timeline (pre_aux=None):
            current stream:
                record indexer_start_event
                q_proj + RoPE -> quant_q
                wait weights_proj_event -> weight_scale
                wait k_cache_update_event -> return

            aux_stream:
                wait indexer_start_event
                weights_proj -> record weights_proj_event
                compressor -> update_k_cache -> record k_cache_update_event

        Dependency graph:
            q_proj + RoPE -> quant_q -- q_scale --.
                                                   v
            weights_proj --------------------> weight_scale
            compressor -> update_k_cache ----> final wait
        """
        if pre_aux is None:
            self.indexer_start_event.record()

            q_fp8, q_scale = self._project_and_quantize_q(qr, position_ids)

            with torch.cuda.stream(self.aux_stream):
                self.indexer_start_event.wait()

                weights = self.weights_proj(hidden_states)
                self.weights_proj_event.record()

                k_fp8, k_scale = self.compressor(hidden_states, metadata)
                self._update_k_cache_if_needed(k_fp8, k_scale, metadata)
                self.k_cache_update_event.record()
        else:
            weights, k_fp8, k_scale = pre_aux
            # pre_aux tensors were allocated on aux_stream; record on the
            # consuming stream so the caching allocator can't recycle them mid-use.
            cur_stream = torch.cuda.current_stream()
            weights.record_stream(cur_stream)
            if k_fp8 is not None:
                k_fp8.record_stream(cur_stream)
            if k_scale is not None:
                k_scale.record_stream(cur_stream)
            q_fp8, q_scale = self._project_and_quantize_q(qr, position_ids)

        self.weights_proj_event.wait()
        weights = self._apply_weight_scale(weights, q_scale)

        self.k_cache_update_event.wait()
        return q_fp8, q_scale, k_fp8, k_scale, weights

    def _run_serial_indexer_prepare(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        # V4 builds its index keys from `hidden_states` through its own compressor,
        # so it has no use for the main compressor's latent. Accepted and ignored
        # to keep one call signature across the generations.
        del latent
        q_fp8, q_scale = self._project_and_quantize_q(qr, position_ids)

        weights = self.weights_proj(hidden_states)

        weights = self._apply_weight_scale(weights, q_scale)

        k_fp8, k_scale = self.compressor(hidden_states, metadata)
        self._update_k_cache_if_needed(k_fp8, k_scale, metadata)
        return q_fp8, q_scale, k_fp8, k_scale, weights

    def _update_k_cache(
        self,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
    ) -> None:
        # DSV4's indexer compressor already scatters INDEXER_COMPRESS. The
        # shared DSA scatter would duplicate that write.
        return

    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: torch.Tensor,
        pre_aux: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = None,
        latent: Optional[torch.Tensor] = None,
    ):
        # The overlapped prepare exists to hide an *independent* key branch behind
        # the Q GEMM. V4.1's key branch is not independent -- it consumes the main
        # compressor's latent, produced on the caller's stream in the same layer --
        # so `supports_overlapped_prepare` turns it off there rather than having the
        # aux stream race a producer it cannot wait on.
        if self.supports_overlapped_prepare and do_multi_stream() and self.aux_stream is not None:
            q_fp8, q_scale, k_fp8, k_scale, weights = self._run_overlapped_indexer_prepare(
                qr,
                hidden_states,
                metadata,
                position_ids,
                pre_aux=pre_aux,
            )
        else:
            assert pre_aux is None, "pre_aux requires multi-stream mode"
            q_fp8, q_scale, k_fp8, k_scale, weights = self._run_serial_indexer_prepare(
                qr, hidden_states, metadata, position_ids, latent=latent
            )

        # If there are no compressed tokens, return an topk indices buffer with all -1s in the tensor.
        if k_fp8 is None:
            topk_indices = metadata.empty_topk_indices_buffer[: hidden_states.shape[0]]
        else:
            topk_indices = self.sparse_attn_indexer(
                metadata,
                hidden_states,
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                q_scale=q_scale,
            )
        return topk_indices


class DeepseekV41Indexer(DeepseekV4Indexer):
    """The V4.1 indexer: index keys projected from the main compressor's latent.

    V4 gives every indexer a ``Compressor`` of its own, pooling hidden states
    straight down to ``index_head_dim``; its checkpoint's ``indexer.wk`` and
    ``indexer.k_norm`` are vestigial and the loader fills them with ones/zeros.
    V4.1 inverts that. Its indexer has no compressor at all -- ``wk`` is a real
    ``Linear(head_dim -> index_head_dim)`` applied to the *main* compressor's
    RoPE-free latent, so the index keys are a cheap reprojection of the
    compressed KV rather than an independent compression of the hidden states
    (model.py:516-517, :533-537).

    Two consequences shape this class:

    * Only a layer that compresses its own KV can build index keys, so ``wk`` /
      ``k_norm`` exist exactly on ``kv_source_layer_ids`` (``owns_k`` at
      model.py:499). The other four index sources (24, 28, 32, 36) own queries
      and ``weights_proj`` but read keys from layer 20's cache, and their
      checkpoint carries no ``wk`` at all -- building one would make the loader
      demand a tensor that does not exist.
    * The reference quantizes both queries and index keys with
      ``fp4_act_quant`` (model.py:537, :549), so the faithful precision for this
      generation is FP4, not the FP8 that V4's indexer uses.
    """

    # The key branch consumes the main compressor's latent, produced on the
    # caller's stream in this same layer, so it cannot be hidden behind the Q
    # GEMM on the aux stream the way V4's independent key branch can.
    supports_overlapped_prepare = False

    def __init__(
        self,
        quant_config: Optional[QuantConfig],
        pos_embd_params: Optional[PositionalEmbeddingParams],
        mla_params: Optional[MLAParams],
        skip_create_weights_in_init: bool,
        sparse_params: DeepSeekV4Params,
        dtype: Optional[torch.dtype],
        compress_ratio: int = 1,
        layer_idx: int = 0,
        aux_stream: Optional[torch.cuda.Stream] = None,
        owns_index_keys: bool = True,
        kv_source_layer_idx: Optional[int] = None,
    ):
        # The compressed latent this indexer reprojects is `head_dim` wide, and
        # MLAParams spells that as its nope+rope split (the V4 family folds RoPE
        # into head_dim rather than appending it, so 448 + 64 = 512). Resolved
        # before super().__init__ because `_build_key_path` runs inside it.
        self.compress_latent_dim = mla_params.qk_nope_head_dim + mla_params.qk_rope_head_dim
        self.owns_index_keys = owns_index_keys
        # Which layer's index-key cache this indexer scores against. Equal to
        # `layer_idx` on an owner; on the four index sources that own no
        # compressed KV (24, 28, 32, 36) it names the kv source they alias.
        self.kv_source_layer_idx = layer_idx if owns_index_keys else kv_source_layer_idx
        assert self.kv_source_layer_idx is not None, (
            f"layer {layer_idx} builds a V4.1 indexer without owning index keys, so "
            "its kv source layer must be given"
        )
        super().__init__(
            quant_config,
            pos_embd_params,
            mla_params,
            skip_create_weights_in_init,
            sparse_params,
            dtype,
            compress_ratio,
            layer_idx,
            aux_stream,
        )

    def _build_key_path(
        self,
        indexer_mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        rms_norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams],
        dtype: Optional[torch.dtype],
        compressor_preset: KVCacheDtype | str,
    ) -> None:
        # No indexer compressor: the keys come from the main one. Bound to None
        # rather than left absent so the inherited helpers that reach for it fail
        # with an AttributeError on None instead of silently finding a V4 module.
        self.compressor = None

        # The base Indexer builds `wk` as Linear(hidden_size -> index_head_dim)
        # in fp32 and `k_norm` as a LayerNorm (weight *and* bias). V4.1 needs
        # Linear(head_dim -> index_head_dim) in bf16 and an RMSNorm, and needs
        # both absent on a non-owner. Dropping the base modules keeps
        # `named_parameters()` equal to the checkpoint's key set, which is what
        # the weight loader diffs against.
        del self.wk
        del self.k_norm
        if not self.owns_index_keys:
            return

        self.wk = Linear(
            self.compress_latent_dim,
            self.head_dim,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        # The reference uses `args.norm_eps`, which V4.1 sets to 1e-20. Keeping
        # the 1e-6 the rest of this backend uses is numerically indistinguishable
        # here: the norm runs over 128 activations whose mean square is O(1), so
        # the two epsilons differ by ~5e-7 relative, four orders of magnitude
        # under bf16's resolution -- and both send an all-zero row to zero. (mHC
        # is the opposite case and does honor 1e-20, because its normalized
        # quantities really do get that small.)
        self.k_norm = RMSNorm(hidden_size=self.head_dim, eps=rms_norm_eps, dtype=dtype)

    def _build_index_keys(
        self,
        latent: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Turn this layer's compressed latent into index keys and cache them.

        ``k = k_norm(wk(latent))``, RoPE on the trailing ``rope_dim``, then the
        same Hadamard-and-quantize the queries get, then a paged write into
        ``INDEXER_COMPRESS`` (model.py:536-540). The whole tail after ``wk`` is
        already a fused kernel on the compressor path, so it is reused verbatim:
        ``k_norm`` stands in for the compressor's ``norm`` and the compressed
        positions are the same ones the main compressor writes, because V4.1 gives
        the index keys and the compressed KV the same ratio by construction.

        One row per compressed position, so ``compressed_position_ids_cuda`` maps
        group ``j`` to token position ``j * ratio`` -- exactly the reference's
        ``freqs_cis[: seqlen - seqlen % ratio : ratio]``.

        A zero-row ``latent`` reaches here on every decode step of a ratio-2 layer
        that does not complete a pooling group, and ``wk`` cannot be asked to run
        on it: ``cublas_mm`` fails the ``cublasLtMatmul`` call outright at ``M ==
        0`` rather than returning the empty product. So the projection is
        synthesized instead. The rest of the tail is deliberately still executed:
        it allocates the zero-row ``(k, scale)`` pair the top-k expects and skips
        its own kernel internally, and the top-k must still run, because it scores
        this step's queries against the *cached* index keys -- returning no keys
        here is not the same as having no keys.
        """
        rows = latent.new_empty((0, self.head_dim)) if latent.shape[0] == 0 else self.wk(latent)
        return postprocess_scatter_compressed(
            rows,
            metadata,
            layer_idx=self.layer_idx,
            compress_ratio=self.compress_ratio,
            norm_weight=self.k_norm.weight,
            norm_eps=self.k_norm.variance_epsilon,
            rotary_cos_sin=self.rotary_emb.rotary_cos_sin,
            nope_head_dim=self.head_dim - self.rope_dim,
            rope_head_dim=self.rope_dim,
            kv_cache_dtype=self.indexer_cache_dtype,
            # Must match `_quantize_q`. The Hadamard is orthogonal, so it cancels in
            # the q.k dot product only if both sides get it, or neither.
            #
            # Note what is and is not shared here. The *decision* is shared -- this
            # flag and `_quantize_q` both read `HAS_FAST_HADAMARD`, so a
            # half-application cannot happen through this call site. The
            # *implementation* is not: Q is rotated in Python by
            # fast-hadamard-transform (`rotate_activation`, scale `head_dim**-0.5`),
            # while K is rotated inside `compressor_postprocess_scatter` by that
            # kernel's own Hadamard. Turning the package on therefore enables two
            # independent implementations that have never been cross-validated
            # against each other; if their normalization or butterfly ordering
            # differs, every indexer logit is silently wrong while both sides look
            # "rotated". The reference applies no Hadamard at all (there is no
            # hadamard in `inference/`), so off-on-both is the reference-faithful
            # setting and is what this bring-up has measured.
            rotate_activation=HAS_FAST_HADAMARD,
            is_indexer=True,
        )

    def _run_serial_indexer_prepare(self, qr, hidden_states, metadata, position_ids, latent=None):
        q_fp8, q_scale = self._project_and_quantize_q(qr, position_ids)
        weights = self._apply_weight_scale(self.weights_proj(hidden_states), q_scale)

        if self.owns_index_keys:
            assert latent is not None, (
                f"layer {self.layer_idx} owns V4.1 index keys, so its own compressor's "
                "pre-RoPE latent must be handed in; got None"
            )
            k_fp8, k_scale = self._build_index_keys(latent, metadata)
            # Publish for the index sources downstream that share this kv source:
            # they score with their own queries against these keys
            # (``shared_attn.index_k`` at model.py:541).
            metadata.v41_index_keys[self.layer_idx] = (k_fp8, k_scale)
        else:
            assert latent is None, (
                f"layer {self.layer_idx} does not own V4.1 index keys, so it has no "
                "compressor and should not have been handed a latent"
            )
            source = self.kv_source_layer_idx
            assert source is not None, (
                f"layer {self.layer_idx} reads index keys but has no kv source layer"
            )
            published = metadata.v41_index_keys.get(source)
            assert published is not None, (
                f"layer {self.layer_idx} needs layer {source}'s index keys, but that "
                "layer has not published them in this forward pass"
            )
            k_fp8, k_scale = published

        return q_fp8, q_scale, k_fp8, k_scale, weights

    def precompute_aux(self, hidden_states, metadata):
        # Nothing to overlap: V4.1's aux-stream work in the V4 path is the
        # indexer compressor, which does not exist here. Returning None routes
        # `forward` to the serial path.
        return None
