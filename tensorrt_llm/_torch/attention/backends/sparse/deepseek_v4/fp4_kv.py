# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP4 main-KV packing for DeepSeek-V4.1 sparse MLA (tech report §2.4.4).

Each token occupies 288 bytes: 512 E2M1 values packed two per byte (256 bytes)
followed by 32 E4M3 per-16-channel scales in the page footer.

The format is NVFP4 *without* its second-level per-tensor global scale, which
§2.4.4 omits deliberately -- the latent is bounded by ``sqrt(kv_lora_rank)``
(~22.6, observed ~10) against an E2M1xE4M3 reach of 448*6, so the second level
buys nothing and costs cache-layout complexity. Concretely that means every
``fp4_quantize`` call here passes a global scale of 1.0, and dequantization is
exactly ``codebook[nibble] * e4m3_scale``.

Two further §2.4.4 rules are the caller's responsibility, not this module's:

* **Quantize after RoPE.** RoPE must already be applied to the rows handed to
  :func:`quant_scatter`; the report measured pre-RoPE quantization as only
  marginally more accurate and not worth the decode-time cost.
* **SWA KV stays FP8.** §2.4.4 excludes the sliding-window cache from FP4 "due
  to its sensitivity to quantization", so this layout is for the global
  (compressed) cache only and :mod:`footer_scale_kv` still owns SWA.

Unlike :mod:`footer_scale_kv`, the RoPE channels are quantized in the same
format as the non-RoPE ones rather than kept at BF16 -- also §2.4.4 ("we adopt
the same quantization format for both components").

Numerically this module contributes nothing of its own: quantization is
delegated to ``torch.ops.trtllm.fp4_quantize`` and the codebook to
:mod:`...moe.fused_moe.triton_dequant_nvfp4`, so a scatter/gather round trip is
*bit-exact* against a direct dequantization of that op's output. All error is
the format's, which is what makes the round-trip test in
``test_deepseek_v41_fp4_kv.py`` an exact-equality assertion.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.moe.fused_moe.triton_dequant_nvfp4 import _get_e2m1_codebook

DIM_TOTAL = 512
SF_VEC_SIZE = 16
NUM_SF = DIM_TOTAL // SF_VEC_SIZE
DATA_ROW_BYTES = DIM_TOTAL // 2
FOOTER_ROW_BYTES = NUM_SF
TOKEN_BYTES = DATA_ROW_BYTES + FOOTER_ROW_BYTES
PAGE_SIZE = 64  # matches footer_scale_kv, so both pools page identically
PAGE_BYTES = PAGE_SIZE * TOKEN_BYTES


def resolve_pool_layout(token_bytes: int):
    """Return the module that owns a KV pool with ``token_bytes`` bytes per token.

    With FP4 on the global cache and FP8 on the sliding-window cache, one process
    holds pools in two layouts at once, and the pool's own row width is what tells
    them apart -- 288 here, 584 in :mod:`footer_scale_kv`. Asking the pool means a
    compressor writing either one needs no extra flag threaded down to it.
    """
    # Imported here rather than at module scope to keep this module a leaf: it is
    # the one `compressor` reaches for, and `footer_scale_kv` pulls in `.kernels`
    # and `.params`.
    from . import footer_scale_kv, fp4_kv

    if token_bytes == TOKEN_BYTES:
        return fp4_kv
    if token_bytes == footer_scale_kv.TOKEN_BYTES:
        return footer_scale_kv
    raise ValueError(
        f"KV pool row of {token_bytes} bytes matches no known layout "
        f"(fp4 {TOKEN_BYTES}, footer-scale fp8 {footer_scale_kv.TOKEN_BYTES})."
    )


# §2.4.4 omits NVFP4's second level; fp4_quantize still wants the argument.
_GLOBAL_SCALE_ONE: dict[torch.device, torch.Tensor] = {}


def _global_scale_one(device: torch.device) -> torch.Tensor:
    scale = _GLOBAL_SCALE_ONE.get(device)
    if scale is None:
        scale = torch.ones((), dtype=torch.float32, device=device)
        _GLOBAL_SCALE_ONE[device] = scale
    return scale


def _check_pool(pool_u8: torch.Tensor, page_bytes: int) -> None:
    """Accept any pool view whose pages are whole, as ``footer_scale_kv`` does.

    Callers pass both a flat ``[pages, page_bytes]`` buffer and the cosmetic
    ``[pages, page_size, TOKEN_BYTES]`` view that ``_prepare_bf16_pool`` builds;
    only the byte total per page is a layout claim, since the scale footer lives
    at the end of the page rather than beside each token.
    """
    assert pool_u8.dtype == torch.uint8 and pool_u8.is_contiguous()
    assert pool_u8.numel() % page_bytes == 0, (
        f"pool of {pool_u8.numel()} bytes is not a whole number of {page_bytes}-byte pages"
    )


@triton.jit
def _scatter_kernel(
    data_ptr,  # uint8 [num_tokens, DATA_ROW_BYTES] packed E2M1 nibbles
    sf_ptr,  # uint8 [num_tokens * NUM_SF] linear E4M3 scales
    loc_ptr,  # int32/int64 [num_tokens] global slot ids, < 0 skips
    pool_u8_ptr,
    data_stride0,
    PAGE_SIZE_C: tl.constexpr,
    PAGE_BYTES_C: tl.constexpr,
    DATA_ROW_BYTES_C: tl.constexpr,
    FOOTER_OFFSET: tl.constexpr,
    FOOTER_ROW_BYTES_C: tl.constexpr,
):
    token_id = tl.program_id(0)

    loc = tl.load(loc_ptr + token_id)
    if loc >= 0:
        # Pool byte offsets can exceed int32 even when slot ids do not.
        loc64 = loc.to(tl.int64)
        loc_page = loc64 // PAGE_SIZE_C
        loc_off = loc64 % PAGE_SIZE_C
        page_base = loc_page * PAGE_BYTES_C

        data_range = tl.arange(0, DATA_ROW_BYTES_C)
        packed = tl.load(data_ptr + token_id * data_stride0 + data_range)
        tl.store(pool_u8_ptr + page_base + loc_off * DATA_ROW_BYTES_C + data_range, packed)

        sf_range = tl.arange(0, FOOTER_ROW_BYTES_C)
        scales = tl.load(sf_ptr + token_id * FOOTER_ROW_BYTES_C + sf_range)
        tl.store(
            pool_u8_ptr + page_base + FOOTER_OFFSET + loc_off * FOOTER_ROW_BYTES_C + sf_range,
            scales,
        )


def quant_scatter(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    rows_bf16: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> None:
    """Quantize BF16 latent rows to §2.4.4 FP4 and scatter them into pages.

    ``rows_bf16`` must already carry RoPE (§2.4.4 quantizes after RoPE).
    Slots whose ``loc`` is negative are skipped, matching
    :func:`footer_scale_kv.quant_scatter`.
    """
    page_bytes = page_size * TOKEN_BYTES
    _check_pool(pool_u8, page_bytes)
    assert rows_bf16.dtype == torch.bfloat16
    assert rows_bf16.shape[-1] == DIM_TOTAL
    assert loc.dtype in (torch.int32, torch.int64) and loc.is_contiguous()
    num_tokens = rows_bf16.shape[0]
    assert loc.shape[0] == num_tokens
    if num_tokens == 0:
        return

    rows = rows_bf16.contiguous()
    # Linear (un-swizzled) scales, E4M3 rather than UE8M0, 16-channel blocks:
    # exactly the §2.4.4 format, and exactly NUM_SF bytes per token with no
    # row padding (verified for non-multiple-of-4 token counts).
    data, scales = torch.ops.trtllm.fp4_quantize(
        rows, _global_scale_one(rows.device), SF_VEC_SIZE, False, False
    )

    _scatter_kernel[(num_tokens,)](
        data,
        scales,
        loc,
        pool_u8,
        data.stride(0),
        PAGE_SIZE_C=page_size,
        PAGE_BYTES_C=page_bytes,
        DATA_ROW_BYTES_C=DATA_ROW_BYTES,
        FOOTER_OFFSET=page_size * DATA_ROW_BYTES,
        FOOTER_ROW_BYTES_C=FOOTER_ROW_BYTES,
    )


@triton.jit
def _dequant_gather_kernel(
    pool_u8_ptr,
    loc_ptr,
    lut_ptr,  # fp32 [16] E2M1 codebook
    out_ptr,  # bf16 [num_slots, DIM_TOTAL]
    out_stride0,
    PAGE_SIZE_C: tl.constexpr,
    PAGE_BYTES_C: tl.constexpr,
    DATA_ROW_BYTES_C: tl.constexpr,
    FOOTER_OFFSET: tl.constexpr,
    FOOTER_ROW_BYTES_C: tl.constexpr,
    DIM_C: tl.constexpr,
    SF_VEC_C: tl.constexpr,
):
    slot_id = tl.program_id(0)
    ch = tl.arange(0, DIM_C)
    out_offsets = slot_id * out_stride0 + ch

    loc = tl.load(loc_ptr + slot_id)
    if loc < 0:
        # Padding slots must read as zero rather than wrapping to slot -1.
        tl.store(out_ptr + out_offsets, tl.zeros([DIM_C], dtype=out_ptr.dtype.element_ty))
        return

    loc64 = loc.to(tl.int64)
    loc_page = loc64 // PAGE_SIZE_C
    loc_off = loc64 % PAGE_SIZE_C
    page_base = loc_page * PAGE_BYTES_C

    # Two channels share a byte; adjacent lanes reload it from cache.
    data_base = page_base + loc_off * DATA_ROW_BYTES_C
    packed = tl.load(pool_u8_ptr + data_base + ch // 2).to(tl.int32)
    nibble = (packed >> ((ch % 2) * 4)) & 0xF
    value = tl.load(lut_ptr + nibble)

    sf_base = page_base + FOOTER_OFFSET + loc_off * FOOTER_ROW_BYTES_C
    scale_byte = tl.load(pool_u8_ptr + sf_base + ch // SF_VEC_C)
    scale = scale_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    tl.store(out_ptr + out_offsets, (value * scale).to(out_ptr.dtype.element_ty))


def dequant_gather(
    pool_u8: torch.Tensor,
    loc: torch.Tensor,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    """Gather FP4 slots and dequantize them to BF16 latent rows.

    Mirrors :func:`footer_scale_kv.dequant_gather`, including its
    ``(*loc.shape, dim)`` output shape, so an FP4 pool is a drop-in for the
    dequantize-before-attention step §2.4.4 sanctions. Negative ``loc`` entries
    yield zero rows.
    """
    page_bytes = page_size * TOKEN_BYTES
    _check_pool(pool_u8, page_bytes)
    loc_flat = loc.reshape(-1).contiguous()
    num_slots = loc_flat.numel()
    out = torch.empty(num_slots, DIM_TOTAL, dtype=torch.bfloat16, device=pool_u8.device)
    if num_slots == 0:
        return out.reshape(*loc.shape, DIM_TOTAL)

    _dequant_gather_kernel[(num_slots,)](
        pool_u8,
        loc_flat,
        _get_e2m1_codebook(pool_u8.device),
        out,
        out.stride(0),
        PAGE_SIZE_C=page_size,
        PAGE_BYTES_C=page_bytes,
        DATA_ROW_BYTES_C=DATA_ROW_BYTES,
        FOOTER_OFFSET=page_size * DATA_ROW_BYTES,
        FOOTER_ROW_BYTES_C=FOOTER_ROW_BYTES,
        DIM_C=DIM_TOTAL,
        SF_VEC_C=SF_VEC_SIZE,
    )
    return out.reshape(*loc.shape, DIM_TOTAL)
