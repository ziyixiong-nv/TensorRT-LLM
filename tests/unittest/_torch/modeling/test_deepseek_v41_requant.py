# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4.1's dense fp8 requantization: 32x32 ue8m0 on disk -> 128x128 fp32 blocks.

Under ``TRTLLM_V41_DENSE_FP8=1`` the loader re-lays-out every dense fp8 stem
instead of widening it to bf16 (``_remap_deepseek_v41_checkpoint_keys`` docstring
point 1). The claim that makes that acceptable is *near-losslessness*, and it
rests on one line: the tile scale is rounded **up to a power of two**, so the
``fp8 -> bf16 -> fp8`` round trip moves exponent fields and no mantissa bits.

Two layers of test, because they fail for different reasons:

* **Mechanism** (CPU, no checkpoint). ``torch.float8_e4m3fn`` casts work host-side,
  so the exactness argument is testable without a GPU and without 475 GiB. These
  pin ``_pow2_tile_scale`` and ``_requantize_block128`` in isolation, including a
  negative test that fails if the ``ceil(log2(...))`` is ever "simplified" away.
* **Load equivalence** (GPU + checkpoint). The mechanism tests would all still
  pass if the emitted scale were the *reciprocal* of what the consumer expects --
  they reconstruct with the same convention they produced. So the second layer
  reconstructs with the loader's own ``weight_dequant`` at block 128 and compares
  against ``_dequantize_block32``'s bf16 output on real tensors. An inverted
  convention shows up there as a gross mismatch; on a GSM8K score it would show
  up only as a vague accuracy loss.
"""

import json
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.configs.deepseek_v41 import dense_fp8_requant_enabled
from tensorrt_llm._torch.models import modeling_deepseekv41 as v41
from tensorrt_llm._torch.models.modeling_deepseekv4 import weight_dequant

_BLOCK = v41._V41_REQUANT_BLOCK
_E4M3_MAX = v41._E4M3_MAX
_SUBNORMAL_STEP = 2.0**v41._E4M3_SUBNORMAL_FLOOR_EXP

RELEASE_CHECKPOINT = Path("/code/llm-models/DeepSeek-V4.1-Flash")

requires_release_checkpoint = pytest.mark.skipif(
    not (RELEASE_CHECKPOINT / "config.json").is_file(),
    reason=f"release checkpoint not present at {RELEASE_CHECKPOINT}",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="weight_dequant is a Triton kernel and needs both operands resident",
)

# Loose regression guard, deliberately not a tight bound -- the subnormal-step
# assertion in the same test is what pins the mechanism. An earlier draft set this
# to 1e-3 and it failed at exponent spread 7 with 2.24e-03 while the bound did not
# fire. Probing the fraction across shapes shows why 1e-3 was the wrong kind of
# number: it swings 20x with the *tile count* at fixed spread, because the
# fixture's exponent pattern lays out a different set of downshifted blocks for
# each shape, so it measures the fixture and not the function. It is also harsher
# than reality -- synthetic ``randn`` has a heavier low-magnitude tail than trained
# weights, and the worst fraction measured on real V4.1 weights is 4.97e-04. 1e-2
# still catches a gross regression: the negative test below sits at 8.9e-01.
_MAX_INEXACT_FRACTION = 1e-2


def _reconstruct(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """What a 128x128 block-scale consumer reconstructs, for the mechanism tests.

    Mirrors the expansion ``Linear``'s FP8_BLOCK_SCALES path applies to
    ``weight_scale`` (``modules/linear.py:1188`` sizes it with ``ceil``, so a
    partial edge tile is expected). Kept local to this file and used only where a
    GPU is not available: the load-equivalence tests below reconstruct with the
    loader's real ``weight_dequant`` instead, which is the assertion that can
    catch an inverted convention.
    """
    m, n = q.shape
    expanded = scale.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    return (q.float() * expanded[:m, :n]).to(dtype)


def _fp8_32_weight(tiles_m: int, tiles_n: int, spread: float, seed: int = 0) -> torch.Tensor:
    """A bf16 weight built the way V4.1's checkpoint is: e4m3 values x ue8m0 32-block scales.

    Two properties of the construction are load-bearing, and getting either wrong
    makes the positive tests prove nothing:

    * **Each 32-block is normalized to its own amax**, because that is what a ue8m0
      quantizer does -- one exponent fitted to that block's largest magnitude. It
      bounds the dynamic range *inside* a block, which is what keeps a later
      downshift from underflowing e4m3's subnormal floor. Normalizing globally
      instead lets one block span the whole ``randn`` range and manufactures
      failures that say nothing about this function.
    * **The 0.93 factor** keeps each block's amax off 448 exactly, so a tile amax is
      a realistic e4m3 value like 416. Without it ``amax / 448`` would itself be a
      power of two and the negative test below would be vacuous.

    ``spread`` is the exponent range across the 32-blocks *within* one 128-tile --
    the quantity V4.1's checkpoint census bounds (worst measured anywhere: 7.00, at
    ``layers.{1,14}.engram.wkv``).
    """
    g = torch.Generator().manual_seed(seed)
    m, n = tiles_m * _BLOCK, tiles_n * _BLOCK
    base = torch.randn(m, n, generator=g)

    blocks = base.reshape(m // 32, 32, n // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True)
    base = (blocks / amax * _E4M3_MAX * 0.93).reshape(m, n)
    q = base.to(torch.float8_e4m3fn).float()

    nb_m, nb_n = m // 32, n // 32
    exps = torch.arange(nb_m * nb_n, dtype=torch.float32) % (spread + 1)
    exps = exps.reshape(nb_m, nb_n) - spread  # keep magnitudes modest
    s32 = torch.exp2(exps).repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    return (q * s32).to(torch.bfloat16)


def _residue(w: torch.Tensor):
    """``(inexact count, worst |err| / global max, per-element error)`` for the round trip."""
    q, s = v41._requantize_block128(w)
    back = _reconstruct(q, s)
    err = (back.float() - w.float()).abs()
    return int((back != w).sum()), err.max().item() / w.float().abs().max().item(), err


def test_roundtrip_is_bit_exact_at_spread_zero() -> None:
    """Spread 0 means no 32-block is downshifted, so there is no loss channel at all.

    The only case where strict equality is the right assertion, and the case that
    pins the mechanism: if a pure exponent shift were not exact, this fails.
    """
    w = _fp8_32_weight(2, 3, 0.0)
    q, s = v41._requantize_block128(w)
    back = _reconstruct(q, s)
    assert torch.equal(back, w), (
        f"{int((back != w).sum())} of {w.numel()} elements changed at spread 0, where the "
        f"round trip is a pure exponent shift and must be exact"
    )


@pytest.mark.parametrize("spread", [3.0, 7.0])
def test_roundtrip_deviation_is_only_subnormal_flush(spread: float) -> None:
    """At nonzero spread the only loss channel is the e4m3 subnormal floor. Bound it.

    Spread 3 is where most roles sit; 7 is the worst measured anywhere in the
    checkpoint. Two assertions, because the fraction alone would pass if a few
    elements were catastrophically wrong, and the bound alone would pass if
    everything drifted by half a step.
    """
    w = _fp8_32_weight(2, 3, spread)
    bad, _, err = _residue(w)
    assert bad / w.numel() <= _MAX_INEXACT_FRACTION, (
        f"spread={spread}: {bad} of {w.numel()} elements deviate "
        f"({bad / w.numel():.2e}), above the {_MAX_INEXACT_FRACTION:.0e} budget"
    )

    _, s = v41._requantize_block128(w)
    step = s.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    step = step[: w.shape[0], : w.shape[1]] * _SUBNORMAL_STEP
    assert torch.all(err <= step), (
        f"spread={spread}: a deviation exceeds one e4m3 subnormal step of its tile scale, so "
        f"the loss mechanism is NOT subnormal flush and the exactness argument behind "
        f"TRTLLM_V41_DENSE_FP8 needs rederiving rather than patching"
    )


def test_scale_is_a_power_of_two() -> None:
    """ue8m0 compliance, and the precondition for everything above."""
    _, s = v41._requantize_block128(_fp8_32_weight(2, 2, 5.0))
    log2s = torch.log2(s)
    assert torch.equal(log2s, torch.round(log2s)), f"non-power-of-two scale: {s}"


def test_no_overflow_without_a_clamp() -> None:
    """``S >= amax / 448`` by construction, so nothing can exceed e4m3's finite range."""
    q, _ = v41._requantize_block128(_fp8_32_weight(2, 2, 7.0))
    assert torch.isfinite(q.float()).all(), "e4m3 saturated to a non-finite value"
    assert q.float().abs().max() <= _E4M3_MAX


def test_pow2_scale_beats_amax_over_448_by_orders_of_magnitude() -> None:
    """The **negative** test, and the reason it has to exist.

    Every other test in this file still passes if someone deletes the
    ``ceil(log2(...))`` and leaves ``S = amax / 448`` -- they reconstruct with the
    same convention they produced, so they would be testing a round trip against
    itself. This pins the one line the exactness rests on.
    """
    w = _fp8_32_weight(2, 2, 3.0)
    tiles = w.float().reshape(2, _BLOCK, 2, _BLOCK)
    ratio = tiles.abs().amax(dim=(1, 3)) / _E4M3_MAX
    assert not torch.all(torch.log2(ratio) == torch.round(torch.log2(ratio))), (
        "amax/448 happens to be a power of two on this input, which makes this test vacuous -- "
        "check the 0.93 factor in _fp8_32_weight"
    )

    exp = ratio.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    naive = ((w.float() / exp).to(torch.float8_e4m3fn).float() * exp).to(torch.bfloat16)
    naive_bad = int((naive != w).sum())
    naive_err = (naive.float() - w.float()).abs().max().item()

    pow2_bad, pow2_rel, _ = _residue(w)
    tile_max = w.float().abs().max().item()

    assert naive_bad > 100 * max(pow2_bad, 1), (
        f"amax/448 was not dramatically worse ({naive_bad} vs {pow2_bad} inexact) -- either the "
        f"input is not a power-of-two-scaled e4m3 tensor, or _pow2_tile_scale leaked into here"
    )
    assert naive_err / tile_max > 100 * pow2_rel, (
        f"amax/448 worst error {naive_err / tile_max:.2e} is not far worse than the pow2 form's "
        f"{pow2_rel:.2e}"
    )


def test_zero_tile_is_handled() -> None:
    """An all-zero tile has amax == 0, where ``log2`` is ``-inf``."""
    w = _fp8_32_weight(2, 2, 0.0)
    w[:_BLOCK, :_BLOCK] = 0.0
    q, s = v41._requantize_block128(w)
    assert torch.isfinite(s).all(), f"non-finite scale from a zero tile: {s}"
    assert s[0, 0] == 1.0
    assert torch.equal(_reconstruct(q, s), w)


@pytest.mark.parametrize("shape", [(_BLOCK + 7, _BLOCK * 2), (_BLOCK * 2, _BLOCK + 1), (5, 3)])
def test_partial_edge_tiles(shape) -> None:
    """Dimensions need not be multiples of 128, even though this checkpoint's all are.

    Every dense fp8 stem V4.1 ships has both dims a multiple of 128, so the padding
    branch never fires in production -- which is exactly why it needs a test: an
    untaken branch is where a shape bug waits for the next checkpoint.
    """
    m, n = shape
    w = torch.randn(m, n).to(torch.float8_e4m3fn).to(torch.bfloat16)
    q, s = v41._requantize_block128(w)
    assert q.shape == (m, n)
    assert s.shape == ((m + _BLOCK - 1) // _BLOCK, (n + _BLOCK - 1) // _BLOCK)
    assert torch.isfinite(s).all()
    assert torch.equal(_reconstruct(q, s), w)


def test_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        v41._requantize_block128(torch.zeros(4, 4, 4))


def test_pow2_tile_scale_covers_amax() -> None:
    """The overflow-impossibility proof, tested directly rather than via its consequence."""
    amax = torch.tensor([[1e-30, 1.0, 447.9, 448.0, 448.1, 1e4]])
    s = v41._pow2_tile_scale(amax)
    assert torch.all(amax / s <= _E4M3_MAX)
    log2s = torch.log2(s)
    assert torch.equal(log2s, torch.round(log2s))


def test_pow2_tile_scale_runs_on_meta_tensors() -> None:
    """No data-dependent op, so the load-census fixtures can walk the requant path.

    ``modeling_deepseekv41``'s audits run the whole remap on ``meta`` tensors. A
    boolean-mask assignment here needs ``nonzero()``, which meta has no
    data-independent implementation for -- that is a real failure this file caught,
    and the reason ``_pow2_tile_scale`` is written with ``torch.where``.
    """
    s = v41._pow2_tile_scale(torch.zeros(4, 4, device="meta"))
    assert s.device.type == "meta" and s.shape == (4, 4)


# --------------------------------------------------------------------------------
# Load equivalence: the emitted pair, read back by the loader's own consumer.
# --------------------------------------------------------------------------------


def _safetensors_index() -> dict:
    with open(RELEASE_CHECKPOINT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _read_tensor(shard: str, name: str) -> torch.Tensor:
    """One tensor out of one shard, without a safetensors dependency in the test."""
    from safetensors.torch import load_file

    return load_file(RELEASE_CHECKPOINT / shard)[name]


# A handful of stems, one per distinct role and shape family in
# ``_V41_REQUANT_STEM_TAILS``. `attn.wo_a` is here deliberately: it is the stem the
# load census caught missing from the allow-list, because
# ``create_sparse_attn_weights`` redeclares ``o_a_proj`` as fp8 once the quant
# config reports block scales, so handing it bf16 would cast to fp8 *unscaled* and
# clip everything above 448 in silence.
_EQUIVALENCE_STEMS = (
    "layers.1.attn.wq_a",
    "layers.1.attn.wq_b",
    "layers.1.attn.wkv",
    "layers.1.attn.wo_a",
    "layers.1.attn.wo_b",
    "layers.1.ffn.shared_experts.w1",
    "layers.1.ffn.shared_experts.w2",
)


@requires_release_checkpoint
@requires_cuda
@pytest.mark.parametrize("stem", _EQUIVALENCE_STEMS)
def test_requantized_pair_reconstructs_to_the_bf16_fallback(stem: str) -> None:
    """The emitted ``(weight, weight_scale_inv)`` pair, read the way a consumer reads it.

    The reference is the *shipping default*: ``_dequantize_block32``, the same call
    the bf16 path makes, on the same on-disk tensors. The reconstruction is
    ``weight_dequant`` at block 128 -- the loader's own function, the one
    ``load_o_a_proj`` uses -- rather than a local expansion, so the multiplier
    convention is exercised on both sides by real code. Reconstructing with the
    same expansion the producer used would pass even if the scale were inverted.

    The assertion is exact equality, not a tolerance. Every deviation the design
    permits is an e4m3 subnormal flush of a value more than 9 binades below its
    tile's max, and on real trained weights the census measured that at 0 to
    4.97e-04 of elements. So a nonzero count here is reported with its worst
    element and checked against the subnormal bound: within it, the mechanism is
    the documented one; outside it, the convention or the layout is wrong.
    """
    index = _safetensors_index()
    weight = _read_tensor(index[f"{stem}.weight"], f"{stem}.weight")
    scale = _read_tensor(index[f"{stem}.scale"], f"{stem}.scale")

    reference = v41._dequantize_block32(weight, scale)
    q, s = v41._requantize_dense_stem(weight, scale)

    assert q.dtype == torch.float8_e4m3fn
    assert q.shape == reference.shape
    assert s.shape == (
        (q.shape[0] + _BLOCK - 1) // _BLOCK,
        (q.shape[1] + _BLOCK - 1) // _BLOCK,
    )

    back = (
        weight_dequant(q.contiguous().cuda(), s.float().contiguous().cuda(), block_size=_BLOCK)
        .to(torch.bfloat16)
        .cpu()
    )

    err = (back.float() - reference.float()).abs()
    bad = int((back != reference).sum())
    if bad:
        step = s.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
        step = step[: q.shape[0], : q.shape[1]] * _SUBNORMAL_STEP
        worst = err.max().item()
        assert torch.all(err <= step), (
            f"{stem}: {bad} of {reference.numel()} elements differ and the worst ({worst:.3e}) "
            f"exceeds one e4m3 subnormal step of its tile scale. That is not subnormal flush -- "
            f"suspect the scale convention (weight_scale_inv holds the multiplier, not its "
            f"reciprocal) or the 128 block extent"
        )
        assert bad / reference.numel() <= _MAX_INEXACT_FRACTION, (
            f"{stem}: {bad / reference.numel():.2e} of elements deviate, far above the "
            f"4.97e-04 worst case measured across the checkpoint"
        )


@requires_release_checkpoint
@requires_cuda
def test_an_inverted_scale_convention_would_be_caught() -> None:
    """The negative control for the test above, on a real tensor.

    ``weight_scale_inv`` holds the *multiplier* despite the name. If
    ``_requantize_block128`` ever returned ``1 / S``, the mechanism tests would all
    still pass -- so demonstrate here that the load-equivalence assertion is what
    separates the two, and that it fails by orders of magnitude rather than
    marginally.
    """
    index = _safetensors_index()
    stem = "layers.1.attn.wq_b"
    weight = _read_tensor(index[f"{stem}.weight"], f"{stem}.weight")
    scale = _read_tensor(index[f"{stem}.scale"], f"{stem}.scale")

    reference = v41._dequantize_block32(weight, scale)
    q, s = v41._requantize_dense_stem(weight, scale)

    def worst(scale_tensor: torch.Tensor) -> float:
        back = (
            weight_dequant(
                q.contiguous().cuda(),
                scale_tensor.float().contiguous().cuda(),
                block_size=_BLOCK,
            )
            .to(torch.bfloat16)
            .cpu()
        )
        return (back.float() - reference.float()).abs().max().item()

    correct, inverted = worst(s), worst(1.0 / s)
    assert inverted > 1e3 * max(correct, 1e-12), (
        f"an inverted scale was only {inverted:.3e} off against the correct form's "
        f"{correct:.3e}; the load-equivalence test cannot be relied on to catch the convention"
    )


def test_the_switch_is_off_by_default() -> None:
    """The bf16 path is what GSM8K was measured on (94.39 / 93.86 at n=1319).

    Asserted here rather than trusted, because both halves of the change read this
    one function and a default flip would silently move the shipping path off the
    configuration that has a banked score.

    Skipped only when the switch is actually *on* -- which is how the requant arm
    runs. Note that ``TRTLLM_V41_DENSE_FP8=0`` is set-but-off, so the assertion
    still runs there and still means something: an earlier draft skipped on
    ``os.environ.get(...)`` being truthy and therefore skipped on the string
    ``"0"`` too, silently covering the off arm of every paired run.
    """
    if dense_fp8_requant_enabled():
        pytest.skip("TRTLLM_V41_DENSE_FP8 turns the requant path on in this environment")
    assert dense_fp8_requant_enabled() is False
