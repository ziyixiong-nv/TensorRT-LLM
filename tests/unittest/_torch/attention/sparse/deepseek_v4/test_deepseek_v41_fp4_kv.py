# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Layout tests for the DeepSeek-V4.1 FP4 main-KV cache (tech report §2.4.4).

These are deliberately *exact-equality* tests, not tolerance tests. The module
under test delegates all arithmetic -- quantization to
``torch.ops.trtllm.fp4_quantize``, dequantization to the E2M1 codebook shared
with the MoE path -- and contributes only page addressing. So the property to
assert is that a scatter/gather round trip is bit-identical to dequantizing the
same ``fp4_quantize`` output directly out of its dense buffers. Any drift is an
addressing bug, and format error never enters the comparison.

One tolerance test is kept at the end purely to confirm the format is being
used at all (a layout that returned zeros would pass every exactness check).
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import fp4_kv
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import get_token_bytes
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.moe.fused_moe.triton_dequant_nvfp4 import _get_e2m1_codebook

# Only the kernel tests need a GPU; the sizing tests below are pure arithmetic.
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# ---------------------------------------------------------------------------
# Cache sizing -- no GPU
# ---------------------------------------------------------------------------

# compress_ratio 1 is the V4.1 decoder ratio and carries both SWA and COMPRESS,
# so one ratio exercises the FP4-claims-COMPRESS-only split.
_SIZING = dict(index_head_dim=128, has_fp8_kv_cache=False, variant="v41")


def _bytes(attn_type, *, head_dim=576, **kwargs):
    return get_token_bytes(
        head_dim=head_dim,
        index_head_dim=_SIZING["index_head_dim"],
        compress_ratio=1,
        attn_type=attn_type,
        has_fp8_kv_cache=_SIZING["has_fp8_kv_cache"],
        variant=_SIZING["variant"],
        **kwargs,
    )


def test_sizing_fp4_claims_compress_only():
    """§2.4.4: FP4 on the global cache, SWA left at FP8."""
    compress = _bytes(
        DeepseekV4AttentionType.COMPRESS, head_dim=fp4_kv.DIM_TOTAL, main_kv_dtype="fp4"
    )
    assert compress == fp4_kv.TOKEN_BYTES == 288

    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import footer_scale_kv

    swa = _bytes(
        DeepseekV4AttentionType.SWA,
        head_dim=fp4_kv.DIM_TOTAL,
        main_kv_dtype="fp4",
        use_fp8_ds_mla=True,
    )
    assert swa == footer_scale_kv.TOKEN_BYTES, "FP4 must not claim the SWA cache"


def test_sizing_default_is_unchanged():
    """``main_kv_dtype='auto'`` must be byte-identical to not passing it."""
    for attn_type in (DeepseekV4AttentionType.COMPRESS, DeepseekV4AttentionType.SWA):
        for use_fp8_ds_mla in (False, True):
            head_dim = fp4_kv.DIM_TOTAL if use_fp8_ds_mla else 576
            assert _bytes(attn_type, head_dim=head_dim, use_fp8_ds_mla=use_fp8_ds_mla) == _bytes(
                attn_type,
                head_dim=head_dim,
                use_fp8_ds_mla=use_fp8_ds_mla,
                main_kv_dtype="auto",
            )


def test_sizing_rejects_unknown_dtype_and_wrong_head_dim():
    with pytest.raises(ValueError, match="main_kv_dtype"):
        _bytes(DeepseekV4AttentionType.COMPRESS, main_kv_dtype="fp6")
    with pytest.raises(ValueError, match="head_dim"):
        _bytes(DeepseekV4AttentionType.COMPRESS, head_dim=576, main_kv_dtype="fp4")


def test_sizing_reproduces_the_reports_2x():
    """2.03x over today's footer-scale FP8 on the same cache."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import footer_scale_kv

    today = _bytes(DeepseekV4AttentionType.COMPRESS, head_dim=fp4_kv.DIM_TOTAL, use_fp8_ds_mla=True)
    fp4 = _bytes(DeepseekV4AttentionType.COMPRESS, head_dim=fp4_kv.DIM_TOTAL, main_kv_dtype="fp4")
    assert today == footer_scale_kv.TOKEN_BYTES == 584
    assert round(today / fp4, 2) == 2.03
    # And 3.56x over BF16 at the same 512-dim latent.
    assert round(fp4_kv.DIM_TOTAL * 2 / fp4, 2) == 3.56


def _reference_dequant(rows_bf16: torch.Tensor) -> torch.Tensor:
    """Quantize with the same op, then dequantize straight from the dense buffers.

    Deliberately shares only the *op*, not the layout: this reads ``data`` and
    ``scales`` as ``fp4_quantize`` returned them, so it is independent of every
    page-offset computation in :mod:`fp4_kv`.
    """
    data, scales = torch.ops.trtllm.fp4_quantize(
        rows_bf16.contiguous(),
        torch.ones((), dtype=torch.float32, device=rows_bf16.device),
        fp4_kv.SF_VEC_SIZE,
        False,
        False,
    )
    n = rows_bf16.shape[0]
    packed = data.view(torch.uint8).reshape(n, fp4_kv.DATA_ROW_BYTES).to(torch.int32)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    nibbles = torch.stack((lo, hi), dim=-1).reshape(n, fp4_kv.DIM_TOTAL)
    values = _get_e2m1_codebook(rows_bf16.device)[nibbles]
    scale = (
        scales.view(torch.uint8)
        .reshape(n, fp4_kv.NUM_SF)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
        .repeat_interleave(fp4_kv.SF_VEC_SIZE, dim=-1)
    )
    return (values * scale).to(torch.bfloat16)


def _empty_pool(num_pages: int, page_size: int = fp4_kv.PAGE_SIZE) -> torch.Tensor:
    return torch.zeros(num_pages, page_size * fp4_kv.TOKEN_BYTES, dtype=torch.uint8, device="cuda")


def test_token_bytes_matches_report():
    """§2.4.4 arithmetic: 512/2 data + 512/16 scale bytes = 288 B/entry."""
    assert fp4_kv.DATA_ROW_BYTES == 256
    assert fp4_kv.FOOTER_ROW_BYTES == 32
    assert fp4_kv.TOKEN_BYTES == 288
    # The abstract's 890 B/token, at 2.5 global entries/token, is 720 + 170.
    assert fp4_kv.TOKEN_BYTES * 2.5 == 720


@requires_gpu
def test_fp4_quantize_emits_the_report_format():
    """Pin the op's contract, so a change in it fails here and not in a model run."""
    rows = torch.randn(5, fp4_kv.DIM_TOTAL, dtype=torch.bfloat16, device="cuda")
    data, scales = torch.ops.trtllm.fp4_quantize(
        rows, torch.ones((), dtype=torch.float32, device="cuda"), fp4_kv.SF_VEC_SIZE, False, False
    )
    assert data.view(torch.uint8).numel() == 5 * fp4_kv.DATA_ROW_BYTES
    # 5 rows, so a padded scale layout would not be a multiple of NUM_SF.
    assert scales.view(torch.uint8).numel() == 5 * fp4_kv.NUM_SF


@pytest.mark.parametrize("num_tokens", [1, 3, 5, 64, 65, 130])
@requires_gpu
def test_round_trip_is_bit_exact(num_tokens):
    """The whole point: addressing adds no error of its own."""
    torch.manual_seed(num_tokens)
    rows = torch.randn(num_tokens, fp4_kv.DIM_TOTAL, dtype=torch.bfloat16, device="cuda") * 3
    loc = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    pool = _empty_pool(num_tokens // fp4_kv.PAGE_SIZE + 1)

    fp4_kv.quant_scatter(pool, loc, rows)
    got = fp4_kv.dequant_gather(pool, loc)

    torch.testing.assert_close(got, _reference_dequant(rows), atol=0, rtol=0)


@requires_gpu
def test_scatter_and_gather_cross_page_boundaries():
    """Slots straddling pages must land in the right page's data and footer."""
    torch.manual_seed(0)
    page_size = fp4_kv.PAGE_SIZE
    # Last slot of page 0, first and last of page 1, first of page 2.
    loc = torch.tensor(
        [page_size - 1, page_size, 2 * page_size - 1, 2 * page_size],
        dtype=torch.int32,
        device="cuda",
    )
    rows = torch.randn(4, fp4_kv.DIM_TOTAL, dtype=torch.bfloat16, device="cuda") * 3
    pool = _empty_pool(3)

    fp4_kv.quant_scatter(pool, loc, rows)
    torch.testing.assert_close(
        fp4_kv.dequant_gather(pool, loc), _reference_dequant(rows), atol=0, rtol=0
    )

    # Every other slot is untouched: a footer write that overran into the next
    # page's data region, or a data write into this page's footer, shows up here.
    written = {int(x) for x in loc}
    others = torch.tensor(
        [s for s in range(3 * page_size) if s not in written], dtype=torch.int32, device="cuda"
    )
    assert not fp4_kv.dequant_gather(pool, others).any()


@requires_gpu
def test_negative_loc_is_skipped_on_scatter():
    torch.manual_seed(1)
    rows = torch.randn(4, fp4_kv.DIM_TOTAL, dtype=torch.bfloat16, device="cuda") * 3
    loc = torch.tensor([0, -1, 2, -1], dtype=torch.int32, device="cuda")
    pool = _empty_pool(1)

    fp4_kv.quant_scatter(pool, loc, rows)

    reference = _reference_dequant(rows)
    kept = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(fp4_kv.dequant_gather(pool, kept), reference[[0, 2]], atol=0, rtol=0)
    # Slots 1 and 3 were never addressed, so the pool there is still zero.
    untouched = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    assert not fp4_kv.dequant_gather(pool, untouched).any()


@requires_gpu
def test_negative_loc_gathers_zeros_not_slot_minus_one():
    """``footer_scale_kv.dequant_gather`` would wrap here; this one must not."""
    torch.manual_seed(2)
    rows = torch.randn(2, fp4_kv.DIM_TOTAL, dtype=torch.bfloat16, device="cuda") * 3
    pool = _empty_pool(1)
    fp4_kv.quant_scatter(pool, torch.tensor([0, 1], dtype=torch.int32, device="cuda"), rows)

    got = fp4_kv.dequant_gather(pool, torch.tensor([-1, 1, -1], dtype=torch.int32, device="cuda"))
    assert not got[0].any()
    assert not got[2].any()
    torch.testing.assert_close(got[1], _reference_dequant(rows)[1], atol=0, rtol=0)


@requires_gpu
def test_gather_preserves_loc_shape():
    """Mirrors ``footer_scale_kv.dequant_gather``'s ``(*loc.shape, dim)`` contract."""
    pool = _empty_pool(1)
    loc = torch.zeros(2, 3, dtype=torch.int32, device="cuda")
    assert fp4_kv.dequant_gather(pool, loc).shape == (2, 3, fp4_kv.DIM_TOTAL)
    empty = torch.zeros(0, dtype=torch.int32, device="cuda")
    assert fp4_kv.dequant_gather(pool, empty).shape == (0, fp4_kv.DIM_TOTAL)


@requires_gpu
def test_rope_channels_are_quantized_too():
    """§2.4.4 uses one format for both components, unlike ``footer_scale_kv``.

    Written as a layout assertion rather than a numeric one: the last 64 dims
    occupy 32 packed bytes, not 128 BF16 bytes, so a 288-byte token has no room
    for a BF16 RoPE tail.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import footer_scale_kv

    assert footer_scale_kv.TOKEN_BYTES == 584
    assert fp4_kv.TOKEN_BYTES * 2 < footer_scale_kv.TOKEN_BYTES  # the 2.03x claim
    rows = torch.randn(1, fp4_kv.DIM_TOTAL, dtype=torch.bfloat16, device="cuda") * 3
    pool = _empty_pool(1)
    fp4_kv.quant_scatter(pool, torch.zeros(1, dtype=torch.int32, device="cuda"), rows)
    # Data occupies exactly PAGE_SIZE * 256 bytes; the footer starts right after.
    assert pool[0, fp4_kv.PAGE_SIZE * fp4_kv.DATA_ROW_BYTES :].any()


@requires_gpu
def test_format_error_is_the_formats_own():
    """Sanity floor: the round trip must actually resemble the input.

    Every other test here is exact against a reference dequant, which a layout
    returning constant zeros would also satisfy. This is the one test that would
    catch that.
    """
    torch.manual_seed(3)
    rows = torch.randn(256, fp4_kv.DIM_TOTAL, dtype=torch.bfloat16, device="cuda") * 3
    loc = torch.arange(256, dtype=torch.int32, device="cuda")
    pool = _empty_pool(4)
    fp4_kv.quant_scatter(pool, loc, rows)
    got = fp4_kv.dequant_gather(pool, loc).to(torch.float32)
    ref = rows.to(torch.float32)

    cosine = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    assert cosine > 0.99, f"FP4 round trip lost the signal: cosine {cosine.item()}"
