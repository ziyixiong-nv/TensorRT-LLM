# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

from .params import DeepseekV4AttentionType

if TYPE_CHECKING:
    from .metadata import DeepseekV4TrtllmAttentionMetadata


class KVCacheDtype(IntEnum):
    """KV cache dtype/layout preset (values match C++ cache_scale_type parameter).

    The store dtype and scale layout are implied by this value:
      - NONE:              keeps the input dtype (bf16/fp32, decided by the
                           caller's tensor element size).
      - FP8_PERTENSOR:     1 byte per value (FP8 E4M3) with implicit scale=1.
      - FP8_BLOCKWISE:     1 byte per value + 1 fp32 scale per 128 values.
      - MXFP4_BLOCKWISE:   packed FP4 (½ byte per value) + 1 UE8M0 byte per
                           32 values.
      - NVFP4_BLOCKWISE:   packed FP4 (½ byte per value) + 1 E4M3 byte per
                           16 values. The V4.1 tech report §2.4.4 main-KV
                           format; see :mod:`.fp4_kv` for the page layout and
                           the read side.

    Storage size in bytes per logical element is therefore::

        size_per_value = {
            NONE: elem_bytes,  # caller-side
            FP8_PERTENSOR: 1,
            FP8_BLOCKWISE: 1 + 4 / 128,  # data + fp32 scale
            MXFP4_BLOCKWISE: 0.5 + 1 / 32,  # nibble + ue8m0 byte
            NVFP4_BLOCKWISE: 0.5 + 1 / 16,  # nibble + e4m3 byte
        }[kv_cache_dtype]
    """

    NONE = 0
    FP8_PERTENSOR = 1  # FP8 E4M3 with implicit scale=1
    FP8_BLOCKWISE = 2  # FP8 E4M3 with per-128 fp32 scales
    MXFP4_BLOCKWISE = 3  # packed FP4 E2M1 with per-32 UE8M0 scales
    NVFP4_BLOCKWISE = 4  # packed FP4 E2M1 with per-16 E4M3 scales (§2.4.4)

    @property
    def fp4_scale_block(self) -> Optional[int]:
        """Channels per scale byte for the packed-FP4 presets, else ``None``.

        The two FP4 layouts are byte-identical apart from this number (and the
        encoding of the byte itself, which only the kernel and the dequant
        codebook care about), so every size computation can branch on it.
        """
        return {KVCacheDtype.MXFP4_BLOCKWISE: 32, KVCacheDtype.NVFP4_BLOCKWISE: 16}.get(self)


_KV_CACHE_DTYPE_MAP = {
    "default": KVCacheDtype.NONE,
    "bf16": KVCacheDtype.NONE,
    "fp8_pertensor": KVCacheDtype.FP8_PERTENSOR,
    "fp8_blockwise": KVCacheDtype.FP8_BLOCKWISE,
    "mxfp4": KVCacheDtype.MXFP4_BLOCKWISE,
    "nvfp4": KVCacheDtype.NVFP4_BLOCKWISE,
}


def resolve_kv_cache_dtype(kv_cache_dtype: Union[str, KVCacheDtype]) -> KVCacheDtype:
    if isinstance(kv_cache_dtype, str):
        return _KV_CACHE_DTYPE_MAP[kv_cache_dtype]
    return kv_cache_dtype


def postprocess_scatter_compressed(
    rows: torch.Tensor,
    metadata: "DeepseekV4TrtllmAttentionMetadata",
    *,
    layer_idx: int,
    compress_ratio: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
    rotary_cos_sin: torch.Tensor,
    nope_head_dim: int,
    rope_head_dim: int,
    kv_cache_dtype: KVCacheDtype,
    rotate_activation: bool,
    is_indexer: bool,
    want_unquantized_copy: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """RMSNorm + RoPE + Hadamard + quantize + paged write for one set of compressed rows.

    Everything downstream of the pooling reduction, in one place. Three callers
    reach it and they differ only in what produced ``rows``:

    * :meth:`Compressor.forward` -- the pooled output of the reduction kernels.
    * :meth:`DeepseekV41Compressor._forward_identity` -- a plain projection, one
      row per token, because ratio 1 has nothing to reduce.
    * :class:`~.indexer.DeepseekV41Indexer` -- ``wk`` applied to the main
      compressor's latent. V4.1's indexer has no compressor of its own, so
      without this split it would have to duplicate the tail.

    ``rows`` is pre-normalization: the fused kernel applies ``norm_weight`` /
    ``norm_eps`` itself. Positions, masks and page tables all come from
    ``metadata`` keyed by ``compress_ratio``, which is why the caller passes the
    ratio rather than a slice.

    Returns the ``(data, scale)`` pair the caller should hand to the indexer top-k:
    ``(fp4_or_fp8_rows, scale)`` when the cache is quantized, ``(copy, None)``
    when ``want_unquantized_copy`` is set, and ``(rows, None)`` otherwise.
    """
    head_dim = nope_head_dim + rope_head_dim
    bsz = metadata.num_contexts + metadata.num_generations
    total_tokens = rows.shape[0]

    if is_indexer:
        compress_type = DeepseekV4AttentionType.INDEXER_COMPRESS
        block_table = metadata.indexer_block_table(compress_ratio)
    else:
        compress_type = DeepseekV4AttentionType.COMPRESS
        block_table = metadata.compress_block_tables[compress_ratio]
    kv_cache = metadata.kv_cache_manager.get_buffers(layer_idx, compress_type)
    compress_tokens_per_block = metadata.kv_cache_manager.compressed_block_sizes[layer_idx]

    num_comp_tokens = metadata.new_comp_kv_lens_cuda[compress_ratio][:bsz]
    cu_new_comp_kv = metadata.cu_new_comp_kv_cuda[compress_ratio]
    start_pos = metadata.past_kv_lens_cuda[compress_ratio][:bsz]
    position_ids = metadata.compressed_position_ids_cuda[compress_ratio][:total_tokens]
    compressed_mask = metadata.compressed_mask_cuda[compress_ratio][:total_tokens]

    unquantized_copy = torch.empty_like(rows) if want_unquantized_copy else None
    quant_output = None
    scale_output = None
    # The kernel writes the quantized cache either way; these buffers are only the
    # extra copy the index top-k consumes, so a main compressor never pays for them.
    if is_indexer:
        if kv_cache_dtype == KVCacheDtype.FP8_BLOCKWISE:
            quant_output = torch.empty(
                total_tokens, head_dim, dtype=torch.uint8, device=rows.device
            )
            scale_output = torch.empty(
                total_tokens, head_dim // 128, dtype=torch.float32, device=rows.device
            )
        elif (scale_block := kv_cache_dtype.fp4_scale_block) is not None:
            quant_output = torch.empty(
                total_tokens, head_dim // 2, dtype=torch.uint8, device=rows.device
            )
            scale_output = torch.empty(
                total_tokens, head_dim // scale_block, dtype=torch.uint8, device=rows.device
            )

    torch.ops.trtllm.compressor_postprocess_scatter(
        rows,
        unquantized_copy,
        norm_weight,
        norm_eps,
        rotary_cos_sin,
        position_ids,
        nope_head_dim,
        rope_head_dim,
        kv_cache,
        num_comp_tokens,
        cu_new_comp_kv,
        start_pos,
        block_table,
        compressed_mask,
        compress_tokens_per_block,
        int(kv_cache_dtype),
        rotate_activation,
        quant_output,
        scale_output,
    )

    if quant_output is not None:
        if kv_cache_dtype.fp4_scale_block is not None:
            return quant_output.view(torch.float4_e2m1fn_x2), scale_output
        return quant_output.view(torch.float8_e4m3fn), scale_output
    if unquantized_copy is not None:
        return unquantized_copy, None
    return rows, None


class Compressor(nn.Module):
    """KV compressor using Triton kernels with paged memory management.

    Args:
        mla_params: MLA parameters containing hidden_size and head dimensions
        layer_idx: Layer index for cache management
        compress_ratio: Compression ratio (e.g., 4 compresses 4 tokens into 1)
        norm_eps: RMSNorm epsilon
        skip_create_weights_in_init: Whether to skip weight initialization
        pos_embd_params: Positional embedding parameters for RoPE
        dtype: Data type for computation
        kv_cache_dtype: Cache preset string or KVCacheDtype.
        rotate_activation: Whether to apply Hadamard transform in postprocessing (False to skip)
    """

    def __init__(
        self,
        mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: PositionalEmbeddingParams,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_dtype: Union[str, KVCacheDtype] = KVCacheDtype.NONE,
        is_indexer: bool = False,
        rotate_activation: bool = False,
    ):
        super().__init__()
        # Dimensions
        self.dim = mla_params.hidden_size
        self.head_dim = mla_params.qk_rope_head_dim + mla_params.qk_nope_head_dim
        self.rope_head_dim = mla_params.qk_rope_head_dim
        self.nope_head_dim = mla_params.qk_nope_head_dim

        # Compression config
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.state_dim = 2 * self.head_dim if self.overlap else self.head_dim

        # Cache config
        self.layer_idx = layer_idx
        self.kv_cache_dtype: KVCacheDtype = resolve_kv_cache_dtype(kv_cache_dtype)
        self.is_indexer = is_indexer
        self.rotate_activation = rotate_activation
        # The C++ scatter does not write FlashInfer's footer-scale layout.
        self.footer_scale_cache = False

        # Modules
        self.wkv_gate = Linear(
            self.dim,
            self.state_dim * 2,
            bias=False,
            dtype=dtype,
            quant_config=None,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
        self.norm = RMSNorm(hidden_size=self.head_dim, eps=norm_eps, dtype=dtype)
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=self.rope_head_dim,
            is_neox=pos_embd_params.is_neox,
        )

        # Learnable absolute positional encoding for compression
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.state_dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        metadata: "DeepseekV4TrtllmAttentionMetadata",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for paged KV compression.

        Args:
            x: Input tensor [num_tokens, dim]
            metadata: Attention metadata with cache info

        Returns:
            (kv_data, scale) tuple:
            - default / fp8_pertensor main compressor: (kv_comp, None)
            - default indexer:                         (kv_out, None)  bf16
            - fp8_blockwise indexer:                   (fp8_output, fp32 scale)
            - mxfp4 indexer:                           (fp4_output, ue8m0 scale)
            - no compressed tokens:                    (None, None)
        """
        # Extract metadata
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        bsz = num_contexts + num_generations

        # Determine attention types based on whether this is an indexer compressor
        if self.is_indexer:
            compress_type = DeepseekV4AttentionType.INDEXER_COMPRESS
            kv_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV
            score_type = DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE
        else:
            compress_type = DeepseekV4AttentionType.COMPRESS
            kv_type = DeepseekV4AttentionType.COMPRESSOR_KV
            score_type = DeepseekV4AttentionType.COMPRESSOR_SCORE

        # Get cache buffers
        kv_cache = metadata.kv_cache_manager.get_buffers(self.layer_idx, compress_type)
        paged_kv_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_type)
        paged_score_state = metadata.kv_cache_manager.get_buffers(self.layer_idx, score_type)

        # Get block tables
        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        if self.is_indexer:
            block_table = metadata.indexer_block_table(self.compress_ratio)
        else:
            block_table = metadata.compress_block_tables[self.compress_ratio]
        block_table_kv_state = metadata.sliding_block_tables[local_layer_idx, kv_type.value]
        block_table_score_state = metadata.sliding_block_tables[local_layer_idx, score_type.value]

        # Get tokens_per_block from cache manager
        # state_tokens_per_block: for compressor kv/score state caches (used in compress kernels)
        # compress_tokens_per_block: for compressed KV cache (used in scatter)
        state_tokens_per_block = metadata.kv_cache_manager.tokens_per_block
        compress_tokens_per_block = metadata.kv_cache_manager.compressed_block_sizes[self.layer_idx]

        # Get compression metadata
        cu_new_comp_kv = metadata.cu_new_comp_kv_cuda[self.compress_ratio]
        kv_lens = metadata.kv_lens_cuda_runtime
        total_num_comp_tokens = metadata.num_total_compressed_tokens[self.compress_ratio]
        num_comp_tokens = metadata.new_comp_kv_lens_cuda[self.compress_ratio][:bsz]
        max_ctx_comp_kv_lens = metadata.max_ctx_compressed_tokens[self.compress_ratio]

        # Project input to KV and score in the checkpoint dtype. The compressor
        # kernels accept bf16 or fp32 kv_score and convert values to fp32
        # internally for state updates and online-softmax accumulation.
        kv_score = F.linear(x.to(self.wkv_gate.weight.dtype), self.wkv_gate.weight)

        # Allocate output buffer
        kv_comp = torch.empty(total_num_comp_tokens, self.head_dim, device=x.device, dtype=x.dtype)

        # Run compression kernels
        if num_contexts > 0:
            torch.ops.trtllm.compressor_prefill_reduction(
                kv_score[:num_ctx_tokens],
                self.ape,
                paged_kv_state,
                paged_score_state,
                block_table_kv_state[:num_contexts],
                block_table_score_state[:num_contexts],
                kv_comp,
                kv_lens[:num_contexts],
                metadata.cached_token_lens_cuda[:num_contexts],
                metadata.cu_seq_lens_cuda,
                cu_new_comp_kv[: num_contexts + 1],
                num_contexts,
                state_tokens_per_block,
                self.head_dim,
                self.compress_ratio,
                max_ctx_comp_kv_lens,
            )

        if num_generations > 0:
            gen_kv_lens = kv_lens[num_contexts:]
            next_n = metadata.num_gen_tokens_per_seq
            # Pass full kv_score (not sliced) with the generation portion of
            # cu_seq_lens so the kernel reads at the correct absolute offsets.
            torch.ops.trtllm.compressor_paged_kv_compress(
                kv_score,
                self.ape,
                paged_kv_state,
                paged_score_state,
                block_table_kv_state[num_contexts:],
                block_table_score_state[num_contexts:],
                kv_comp,
                gen_kv_lens,
                metadata.cu_seq_lens_cuda[num_contexts:],
                cu_new_comp_kv[num_contexts:],
                num_generations,
                state_tokens_per_block,
                self.head_dim,
                self.compress_ratio,
                next_n,
            )

        if self.footer_scale_cache and not self.is_indexer:
            total_tokens = kv_comp.shape[0]
            self._footer_scale_postprocess_scatter(
                kv_comp,
                kv_cache,
                block_table,
                compress_tokens_per_block,
                num_comp_tokens,
                cu_new_comp_kv,
                metadata.past_kv_lens_cuda[self.compress_ratio][:bsz],
                metadata.compressed_position_ids_cuda[self.compress_ratio][:total_tokens],
                metadata.compressed_mask_cuda[self.compress_ratio][:total_tokens],
                bsz,
            )
            return kv_comp, None

        # Fused postprocess + scatter: RMSNorm + RoPE + Hadamard + paged cache
        # write. Only an indexer compressor wants the postprocessed rows handed
        # back -- a main compressor's consumer reads the cache.
        out = postprocess_scatter_compressed(
            kv_comp,
            metadata,
            layer_idx=self.layer_idx,
            compress_ratio=self.compress_ratio,
            norm_weight=self.norm.weight,
            norm_eps=self.norm.variance_epsilon,
            rotary_cos_sin=self.rotary_emb.rotary_cos_sin,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            kv_cache_dtype=self.kv_cache_dtype,
            rotate_activation=self.rotate_activation,
            is_indexer=self.is_indexer,
            want_unquantized_copy=self.is_indexer and self.kv_cache_dtype == KVCacheDtype.NONE,
        )
        return out

    def enable_footer_scale_cache(self) -> None:
        assert not self.is_indexer, "the indexer compressor keeps its native cache layout"
        assert not self.rotate_activation, (
            "the footer-scale postprocess does not apply the Hadamard rotation; "
            "footer-scale mode requires rotate_activation=False"
        )
        self.footer_scale_cache = True

    def _footer_scale_postprocess_scatter(
        self,
        kv_comp: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        num_comp_tokens: torch.Tensor,
        cu_new_comp_kv: torch.Tensor,
        start_pos: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_mask: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Apply RMSNorm/RoPE and scatter into the footer-scale cache."""
        from . import fp4_kv

        total_tokens = kv_comp.shape[0]
        if total_tokens == 0:
            return
        device = kv_comp.device

        x = kv_comp.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.norm.variance_epsilon)
        x = x * rms * self.norm.weight.to(torch.float32)

        rope_dim = self.rope_head_dim
        half = rope_dim // 2
        # Reserved padding slots may carry uninitialized position IDs and are
        # discarded by compressed_mask during scatter.
        safe_position_ids = torch.where(compressed_mask, position_ids, 0)
        cos_sin = self.rotary_emb.rotary_cos_sin.view(-1, rope_dim)[
            safe_position_ids.to(torch.long)
        ]
        cos, sin = cos_sin[:, :half], cos_sin[:, half:]
        rope = x[:, self.nope_head_dim :]
        even, odd = rope[:, 0::2], rope[:, 1::2]
        rope_rot = torch.empty_like(rope)
        rope_rot[:, 0::2] = even * cos - odd * sin
        rope_rot[:, 1::2] = odd * cos + even * sin
        rows = torch.cat([x[:, : self.nope_head_dim], rope_rot], dim=-1).to(torch.bfloat16)

        token_idx = torch.arange(total_tokens, dtype=torch.int32, device=device)
        cu = cu_new_comp_kv[: batch_size + 1].to(torch.int32)
        batch_idx = torch.searchsorted(cu[1:], token_idx, right=True).clamp(max=batch_size - 1)
        local_idx = token_idx - cu[batch_idx]
        valid = compressed_mask.to(torch.bool) & (local_idx < num_comp_tokens[batch_idx])

        cache_pos = start_pos[batch_idx] + local_idx
        logical_block = torch.div(cache_pos, tokens_per_block, rounding_mode="floor")
        token_in_block = cache_pos - logical_block * tokens_per_block
        max_blocks = block_table.shape[1]
        in_range = valid & (logical_block >= 0) & (logical_block < max_blocks)
        phys_block = block_table[
            batch_idx.to(torch.long),
            logical_block.clamp(min=0, max=max_blocks - 1).to(torch.long),
        ]
        slot = phys_block * tokens_per_block + token_in_block
        loc = torch.where(in_range & (phys_block >= 0), slot, torch.full_like(slot, -1))

        pool = kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1)
        # Which of the two footer-scale layouts this pool holds -- FP8 at 584 B/token
        # or §2.4.4's FP4 at 288 -- is written on the pool itself, so a compressor
        # feeding a long-range cache needs no extra flag to find out. `rows` is
        # post-RoPE BF16 either way, which is what both scatters want.
        layout = fp4_kv.resolve_pool_layout(pool.shape[1] // tokens_per_block)
        layout.quant_scatter(
            pool, loc.to(torch.int32).contiguous(), rows, page_size=tokens_per_block
        )


class DeepseekV41Compressor(Compressor):
    """V4.1's KV compressor: the same softmax pooling, minus the learned APE.

    Three deltas against V4, all visible in the reference ``Compressor``
    (model.py:437-485):

    * **No APE.** V4 adds a learned ``[compress_ratio, state_dim]`` bias inside
      the pooling softmax; V4.1 has no such parameter and its checkpoint carries
      none. The kernels compute ``sum_r kv[r,d] * softmax(score[r,d] + ape[r,d])``
      (compressorKernels.cu:1031), so a zero APE *is* the V4.1 math exactly --
      hence a zeros buffer rather than a parameter, and no kernel change.
    * **Ratio 1 is a plain projection.** ``norm(wkv(x))`` with no gate, no fp32,
      and no cross-step pooling state (model.py:461). The checkpoint has ``wkv``
      but no ``wgate``, so the fused ``wkv_gate`` of the base class would demand
      a tensor that does not exist; and there is nothing for the reduction
      kernels to reduce, so the whole pooling stage is skipped.
    * **fp32 above ratio 1.** The reference promotes ``wkv``/``wgate`` to fp32
      for the pooling path (model.py:447-449) while leaving ratio 1 in bf16. Only
      the projection is promoted -- ``norm`` stays in the model dtype, because
      the reference normalizes after casting the pooled result back
      (model.py:485).

    The other V4.1-specific need is a *pre-RoPE* view of the compressed latent:
    the indexer's ``wk`` consumes it (model.py:526), while the cache holds the
    RoPE'd form. :meth:`latent_pre_rope` reproduces it from the kernel's pooled
    output.
    """

    def __init__(
        self,
        mla_params: MLAParams,
        layer_idx: int,
        compress_ratio: int,
        norm_eps: float,
        skip_create_weights_in_init: bool,
        pos_embd_params: PositionalEmbeddingParams,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        kv_cache_dtype: Union[str, KVCacheDtype] = KVCacheDtype.NONE,
        is_indexer: bool = False,
        rotate_activation: bool = False,
    ):
        assert not is_indexer, (
            "V4.1's Indexer has no compressor of its own: its keys are a "
            "reprojection of this compressor's latent (model.py:516-517)."
        )
        self.is_pooling = compress_ratio > 1
        super().__init__(
            mla_params,
            layer_idx,
            compress_ratio,
            norm_eps,
            skip_create_weights_in_init,
            pos_embd_params,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            is_indexer=is_indexer,
            rotate_activation=rotate_activation,
        )
        assert not self.overlap, (
            "V4.1 keeps one pooling group in flight (model.py:451-456), so no "
            f"layer should be an overlap compressor (ratio {compress_ratio})"
        )

        # A zeros buffer where V4 has a learned parameter: still a valid fp32
        # pointer for the kernels, but nothing the loader has to find a tensor
        # for. Non-persistent so it never reaches a saved state dict either.
        del self.ape
        self.register_buffer(
            "ape",
            torch.zeros(compress_ratio, self.state_dim, dtype=torch.float32),
            persistent=False,
        )

        if self.is_pooling:
            # Only the pooling projection goes fp32; `norm` keeps the model dtype.
            self.wkv_gate = Linear(
                self.dim,
                self.state_dim * 2,
                bias=False,
                dtype=torch.float32,
                quant_config=None,
                skip_create_weights_in_init=skip_create_weights_in_init,
                use_custom_cublas_mm=True,
            )
        else:
            del self.wkv_gate
            self.wkv = Linear(
                self.dim,
                self.head_dim,
                bias=False,
                dtype=dtype,
                quant_config=None,
                skip_create_weights_in_init=skip_create_weights_in_init,
                use_custom_cublas_mm=True,
            )

    def latent_pre_rope(self, kv_comp: torch.Tensor) -> torch.Tensor:
        """The compressed latent as the indexer sees it: normalized, unrotated.

        ``compressor_postprocess_scatter`` fuses RMSNorm, RoPE and the cache
        write, so the intermediate the indexer needs is never materialized. It is
        cheap to recompute -- one RMSNorm over the compressed tokens, of which
        there are ``seqlen / compress_ratio`` -- and doing so keeps the fused
        kernel on the cache path untouched.

        An empty batch is routine, not exceptional: a ratio-2 layer completes a
        pooling group only every other decode step, so half of this layer's
        generation forwards produce no compressed row at all. It has to be
        short-circuited rather than left to the kernel because the unfused ops on
        this path do not tolerate zero rows the way the fused scatter does
        (``postProcessScatterLaunch`` returns early at ``total_tokens == 0``):
        flashinfer's RMSNorm launches ``grid.x == M`` and CUDA rejects a zero
        grid with ``cudaErrorInvalidValue`` before the kernel runs.
        """
        if kv_comp.shape[0] == 0:
            return kv_comp
        return self.norm(kv_comp)

    def forward(
        self, x: torch.Tensor, metadata
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.is_pooling:
            return super().forward(x, metadata)
        return self._forward_identity(x, metadata)

    def _forward_identity(
        self, x: torch.Tensor, metadata
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """The ratio-1 path: project every token, then norm + RoPE + scatter.

        With one token per group there is no reduction and no state, so the two
        compressor reduction kernels are skipped entirely -- which is also why
        this generation does not need them instantiated for ``COMPRESS_RATIO=1``.
        Token ``i`` of the batch becomes compressed row ``i``, so the pooled
        buffer the scatter expects is just the projection itself: the per-request
        row counts in ``cu_new_comp_kv`` equal the per-request token counts at
        this ratio, in the same order.
        """
        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        bsz = num_contexts + num_generations

        total_num_comp_tokens = metadata.num_total_compressed_tokens[self.compress_ratio]
        assert total_num_comp_tokens == x.shape[0], (
            "a ratio-1 compressor produces one compressed token per input token, "
            f"but metadata expects {total_num_comp_tokens} from {x.shape[0]} tokens"
        )
        kv_comp = self.wkv(x)
        total_tokens = kv_comp.shape[0]

        if self.footer_scale_cache:
            self._footer_scale_postprocess_scatter(
                kv_comp,
                metadata.kv_cache_manager.get_buffers(
                    self.layer_idx, DeepseekV4AttentionType.COMPRESS
                ),
                metadata.compress_block_tables[self.compress_ratio],
                metadata.kv_cache_manager.compressed_block_sizes[self.layer_idx],
                metadata.new_comp_kv_lens_cuda[self.compress_ratio][:bsz],
                metadata.cu_new_comp_kv_cuda[self.compress_ratio],
                metadata.past_kv_lens_cuda[self.compress_ratio][:bsz],
                metadata.compressed_position_ids_cuda[self.compress_ratio][:total_tokens],
                metadata.compressed_mask_cuda[self.compress_ratio][:total_tokens],
                bsz,
            )
            return kv_comp, None

        postprocess_scatter_compressed(
            kv_comp,
            metadata,
            layer_idx=self.layer_idx,
            compress_ratio=self.compress_ratio,
            norm_weight=self.norm.weight,
            norm_eps=self.norm.variance_epsilon,
            rotary_cos_sin=self.rotary_emb.rotary_cos_sin,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            kv_cache_dtype=self.kv_cache_dtype,
            rotate_activation=self.rotate_activation,
            is_indexer=False,
        )
        return kv_comp, None
