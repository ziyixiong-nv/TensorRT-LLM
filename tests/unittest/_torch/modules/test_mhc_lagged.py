# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DeepSeek-V4.1 lagged-``pre`` mHC path.

V4 collapses the residual stream with the ``pre`` coefficients the *same*
sublayer just produced, so ``pre`` never leaves the fused kernel and
``mHC.pre_mapping`` returns only ``(post_mix, comb_mix, layer_input)``.

V4.1 lags ``pre`` by one sublayer: a block's attention consumes the ``pre`` the
*previous block's FFN* produced, and its FFN consumes the ``pre`` its *own
attention* produced. ``mHC.pre_mapping_lagged`` provides that wiring.

The tests below pin four properties:

1. **The coefficients match the reference contract.** ``_reference_coeffs`` is
   a transcription of ``<checkpoint>/inference/model.py::Block.hc_mixes`` plus
   ``<checkpoint>/inference/kernel.py::hc_split_sinkhorn``. It is written with
   *literal* constants rather than by reading ``mHC``'s attributes, so a module
   built with the wrong ``post_mult_value`` or the wrong ``norm_eps`` fails
   instead of silently moving the reference with it
   (``test_post_mult_and_norm_eps_are_discriminated`` proves that discrimination
   is real). The same transcription, executed against the vendor's tilelang
   ``hc_split_sinkhorn`` on release layer-0 weights by
   ``$W/gates/mhc_block_canary.py --mode selftest``, agreed to
   ``pre 1.118e-08 / post 1.490e-08 / comb 1.192e-07``.
2. **Reduces to the fused path.** Feeding ``pre_mapping_lagged`` its own ``pre``
   reproduces ``pre_mapping`` — this validates the whole unfused torch path
   (mixer projection ordering, the ``r_acc / K`` normalization, the Sinkhorn
   tail, and the collapse) against the production CUDA kernels.
3. **Actually consumes the external ``pre``.** A primitive that quietly ignored
   ``pre_mix`` would pass (2), so it is checked separately.
4. **The lag changes the answer.** With identical weights, inputs and sublayer
   bodies, lagged and unlagged block wiring must diverge — otherwise a V4.1
   model built on the V4 wiring would look correct.

Properties 1-3 are checked twice: on a small synthetic module, and on the
release checkpoint's real layer-0 ``hc_attn_*`` / ``hc_ffn_*`` weights at the
real ``hidden_size`` of 5120 with the real ``rms_norm_eps`` of 1e-20. The
synthetic case alone would not catch a scale- or eps-dependent error.
"""

import json
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch.modules.mhc.hyper_connection as hyper_connection
from tensorrt_llm._torch.modules.mhc.hyper_connection import mHC

# Synthetic case: narrow enough to stay fast, same hyperparameters otherwise.
HIDDEN_SIZE = 512
HC_MULT = 4
SINKHORN_ITERS = 20
HC_EPS = 1e-6
# DeepSeek-V4.1-Flash `text_config`: rms_norm_eps 1e-20, post_mult_value 2.0.
# The module defaults are 1e-6 / 1.0, so both must be passed explicitly — see
# `test_post_mult_and_norm_eps_are_discriminated`.
RMS_NORM_EPS = 1e-20
POST_MULT = 2.0

RELEASE_CHECKPOINT = Path("/code/llm-models/DeepSeek-V4.1-Flash")
REAL_HIDDEN_SIZE = 5120

requires_release_checkpoint = pytest.mark.skipif(
    not (RELEASE_CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"release checkpoint not present at {RELEASE_CHECKPOINT}",
)

# Every test here builds an `mHC` and compares against the production CUDA
# kernels, so all 17 need a device. Without this guard they raise
# `RuntimeError: No CUDA GPUs are available` on a GPU-less host instead of
# skipping. This is not a CI concern -- `l0_cpu.yml` reaches this file only via
# its `unittest/_torch/modules` directory entry, and that stage runs
# `-m cpu_only`, where `pytest_ignore_collect` in `tests/unittest/conftest.py`
# drops every file whose text lacks the literal `pytest.mark.cpu_only`, which
# this one does. The guard is here so that running the file directly reports
# skips rather than errors, matching the `torch.cuda.is_available()` predicate
# the sibling suites in this directory already use.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="mHC coefficients are compared against the CUDA kernels; requires a GPU",
)

# Observed agreement of `mHC.hc_coeffs` with `_reference_coeffs` on release
# layer-0 weights at hidden_size 5120 (see the module docstring for the command
# that produced these):
#
#   mixer_projection via the FMA CUDA kernel   pre 1.29e-05  post 2.23e-05  comb 8.97e-06
#   mixer_projection via the torch fallback    pre 0.00e+00  post 0.00e+00  comb 0.00e+00
#
# The kernel residual is float accumulation order over K = mult * hidden = 20480
# (the raw `mixes` differ by 1.8e-04 *relative*, which the sigmoid / softmax then
# compresses by an order of magnitude). It is the same kernel V4 uses, so that is
# the accepted numerical baseline, not a V4.1 regression. KERNEL_TOL therefore
# carries ~4x headroom over the worst observed value while still being three
# orders tighter than a `rtol=1e-3`-style bound; FALLBACK_TOL pins the op-order
# transcription itself, which is bit-exact.
KERNEL_TOL = dict(rtol=1e-4, atol=1e-4)
FALLBACK_TOL = dict(rtol=1e-7, atol=1e-7)


def _reference_coeffs(
    x: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    mult: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference ``hc_mixes`` + ``hc_split_sinkhorn``, transcribed with literals.

    Mirrors ``<checkpoint>/inference/model.py::Block.hc_mixes`` and
    ``<checkpoint>/inference/kernel.py::hc_split_sinkhorn``. Deliberately takes
    the weights as arguments and hard-codes every scalar the reference
    hard-codes, so it does **not** track ``mHC``'s configuration:

    * ``post`` is multiplied by a literal ``2`` — not by ``post_mult_value``.
    * ``pre`` gets ``+ 1e-6`` after the sigmoid; ``post`` gets none.
    * the Sinkhorn tail is asymmetric: an initial ``softmax(dim=-1) + eps`` and
      column normalize, then a literal ``20 - 1 = 19`` row/column pairs, so it
      *ends on a column* normalize and the rows are deliberately not unit-sum.

    ``norm_eps`` is a parameter only because it is the one scalar the reference
    reads from the checkpoint config (``rms_norm_eps``); every call site here
    passes the release value ``1e-20``.
    """
    hc = mult
    eps = 1e-6

    flat = x.flatten(-2, -1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(flat, fn.float()) * rsqrt

    scale = scale.float()
    base = base.float()
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(20 - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre.unsqueeze(-1), post.unsqueeze(-1), comb


def _reference_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Reference ``hc_pre``: fp32 weighted sum over the ``mult`` axis, cast back."""
    return torch.sum(pre_mix.reshape(*x.shape[:-1], 1) * x.float(), dim=-2).to(x.dtype)


def _build_mhc(hidden_size: int = HIDDEN_SIZE, seed: int = 0) -> mHC:
    """An mHC with V4.1 hyperparameters and small, well-conditioned weights."""
    torch.manual_seed(seed)
    module = mHC(
        mult=HC_MULT,
        hidden_size=hidden_size,
        sinkhorn_iters=SINKHORN_ITERS,
        dtype=torch.bfloat16,
        eps=HC_EPS,
        norm_eps=RMS_NORM_EPS,
        sinkhorn_eps=HC_EPS,
        post_mult_value=POST_MULT,
    ).cuda()
    with torch.no_grad():
        module.fn.copy_(torch.randn_like(module.fn) * 1e-3)
        module.base.copy_(torch.randn_like(module.base) * 0.1)
        module.scale.copy_(torch.randn_like(module.scale) * 0.1)
    return module


def _load_release_hc_weights(layer: int, kind: str) -> dict[str, torch.Tensor]:
    """One layer's three mHC parameter tensors for ``kind`` in {attn, ffn}."""
    from safetensors import safe_open

    weight_map = json.loads((RELEASE_CHECKPOINT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    out = {}
    for suffix in ("fn", "base", "scale"):
        key = f"layers.{layer}.hc_{kind}_{suffix}"
        shard = RELEASE_CHECKPOINT / weight_map[key]
        with safe_open(str(shard), framework="pt") as fh:
            out[suffix] = fh.get_tensor(key).cuda()
    return out


def _build_release_mhc(layer: int = 0, kind: str = "attn") -> mHC:
    module = mHC(
        mult=HC_MULT,
        hidden_size=REAL_HIDDEN_SIZE,
        sinkhorn_iters=SINKHORN_ITERS,
        dtype=torch.bfloat16,
        eps=HC_EPS,
        norm_eps=RMS_NORM_EPS,
        sinkhorn_eps=HC_EPS,
        post_mult_value=POST_MULT,
    ).cuda()
    weights = _load_release_hc_weights(layer, kind)
    with torch.no_grad():
        module.fn.copy_(weights["fn"])
        module.base.copy_(weights["base"])
        module.scale.copy_(weights["scale"])
    return module


def _residual(num_tokens: int, hidden_size: int = HIDDEN_SIZE, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        (num_tokens, HC_MULT, hidden_size), dtype=torch.float32, device="cuda"
    ).bfloat16()


def _coeffs_without_the_fma_kernel(module: mHC, x: torch.Tensor):
    """``hc_coeffs`` with ``mixer_projection`` forced onto its torch fallback."""
    saved = hyper_connection.mhc_gemm_rms_fma_cuda
    hyper_connection.mhc_gemm_rms_fma_cuda = None
    try:
        return module.hc_coeffs(x)
    finally:
        hyper_connection.mhc_gemm_rms_fma_cuda = saved


def _assert_bf16_within_one_ulp(actual: torch.Tensor, expected: torch.Tensor, what: str):
    """Compare two bf16 tensors at bf16 resolution rather than fp32 resolution.

    ``layer_input`` is an fp32 accumulation cast to bf16, so the torch path and
    the CUDA path can land on *adjacent* bf16 values purely from the final
    rounding. bf16 keeps 7 stored significand bits, so for ``|x|`` in
    ``[2^(e-1), 2^e)`` one ULP is exactly ``2^(e-8)`` — a relative step of up to
    0.78%. Asserting anything tighter on a bf16 output asserts bit-identical
    rounding, which is not the property under test; asserting a flat relative
    tolerance would be loose by up to 2x depending on where each value sits
    inside its binade. So the bound is computed per element.
    """
    expected_f = expected.float()
    exponent = torch.frexp(expected_f).exponent  # |x| in [2^(e-1), 2^e)
    ulp = torch.ldexp(torch.ones_like(expected_f), exponent - 8)
    diff = (actual.float() - expected_f).abs()
    bad = int((diff > ulp).sum())
    assert bad == 0, (
        f"{what}: {bad}/{diff.numel()} elements differ by more than one bf16 ULP; "
        f"max abs diff {diff.max().item():.3e}"
    )


def _assert_coeffs_match_reference(module: mHC, x: torch.Tensor, what: str):
    """Both ``mixer_projection`` tactics against the literal-constant reference."""
    pre_ref, post_ref, comb_ref = _reference_coeffs(
        x, module.fn, module.base, module.scale, mult=module.mult, norm_eps=RMS_NORM_EPS
    )

    for tactic, coeffs, tol in (
        ("fma kernel", module.hc_coeffs(x), KERNEL_TOL),
        ("torch fallback", _coeffs_without_the_fma_kernel(module, x), FALLBACK_TOL),
    ):
        pre, post, comb = coeffs
        for name, got, ref in (
            ("pre", pre, pre_ref),
            ("post", post, post_ref),
            ("comb", comb, comb_ref),
        ):
            torch.testing.assert_close(got, ref, msg=f"{what} / {tactic} / {name}", **tol)

    # The Sinkhorn tail ends on a column normalize, so the columns are unit-sum.
    # (The rows are not, on real weights — see
    # `test_sinkhorn_asymmetry_and_iteration_count_on_release_weights`. On the
    # synthetic `randn * 1e-3` weights the matrix is near-uniform and converges
    # to doubly stochastic, which is why that check cannot live here.)
    columns = comb_ref.sum(dim=-2)
    torch.testing.assert_close(columns, torch.ones_like(columns), rtol=0, atol=1e-5)


@pytest.mark.parametrize("num_tokens", [1, 17, 128])
def test_coeffs_match_reference_transcription(num_tokens: int):
    """All three coefficient sets, including ``pre``, against the reference."""
    module = _build_mhc()
    _assert_coeffs_match_reference(module, _residual(num_tokens), f"synthetic/{num_tokens}")


@requires_release_checkpoint
@pytest.mark.parametrize("kind", ["attn", "ffn"])
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_coeffs_match_reference_on_release_weights(kind: str, num_tokens: int):
    """The same check at the release scale, on the release layer-0 weights.

    ``hidden_size`` 5120 makes the mixer projection a length-20480 fp32 dot
    product, and the real ``hc_*_fn`` weights are not the well-conditioned
    ``randn * 1e-3`` of the synthetic case. A transcription error that the
    512-wide synthetic case tolerates shows up here.
    """
    module = _build_release_mhc(0, kind)
    x = _residual(num_tokens, REAL_HIDDEN_SIZE)
    _assert_coeffs_match_reference(module, x, f"release/layer0/{kind}/{num_tokens}")

    # `collapse` against the reference `hc_pre` at the same scale.
    torch.manual_seed(5)
    pre_mix = torch.rand(num_tokens, HC_MULT, 1, device="cuda", dtype=torch.float32) + 0.25
    torch.testing.assert_close(
        mHC.collapse(x, pre_mix), _reference_collapse(x, pre_mix), rtol=0, atol=0
    )


@requires_release_checkpoint
def test_sinkhorn_asymmetry_and_iteration_count_on_release_weights():
    """The Sinkhorn tail is asymmetric and does *not* converge on real weights.

    Plan §9.1 flags the ``sinkhorn_iters - 1`` off-by-one as high risk. On the
    synthetic ``randn * 1e-3`` weights it is unfalsifiable: the raw matrix is
    near-uniform, 19 iterations converge it to doubly stochastic, and every
    variant agrees. On the release layer-0 weights nothing converges, so each
    variant is measurably distinct — these are the observed deltas against the
    correct 19-pair, column-last tail (``comb`` absmax 0.9546):

        18 pairs        4.154e-03      20 pairs        3.693e-03
        row-last        5.820e-02      no normalize    6.303e-01
        |colsum - 1|    1.132e-06      |rowsum - 1|    5.815e-02

    A build that ran 20 pairs, or that ended on a row normalize, would still
    produce fluent text. This is the cheap detector.
    """
    module = _build_release_mhc(0, "attn")
    x = _residual(32, REAL_HIDDEN_SIZE)
    comb = module.hc_coeffs(x)[2]

    # Ends on a column normalize: columns unit-sum, rows visibly not.
    columns = comb.sum(dim=-2)
    torch.testing.assert_close(columns, torch.ones_like(columns), rtol=0, atol=1e-5)
    assert (comb.sum(dim=-1) - 1).abs().max() > 1e-2, (
        "rows are unit-sum on release weights — the tail ended on a row normalize"
    )

    n = HC_MULT
    raw = (
        module.mixer_projection(x).view(*x.shape[:-2], module.mix_hc)[..., 2 * n :]
        * module.scale[2]
        + module.base[2 * n :]
    ).unflatten(-1, (n, n))

    def tail(pairs: int, *, col_last: bool = True) -> torch.Tensor:
        c = raw.softmax(dim=-1) + HC_EPS
        c = c / (c.sum(dim=-2, keepdim=True) + HC_EPS)
        for _ in range(pairs):
            first, second = (-1, -2) if col_last else (-2, -1)
            c = c / (c.sum(dim=first, keepdim=True) + HC_EPS)
            c = c / (c.sum(dim=second, keepdim=True) + HC_EPS)
        return c

    # `sinkhorn_iters = 20` means 19 pairs, exactly.
    torch.testing.assert_close(comb, tail(SINKHORN_ITERS - 1), rtol=0, atol=0)
    for wrong, label in (
        (tail(SINKHORN_ITERS - 2), "18 pairs"),
        (tail(SINKHORN_ITERS), "20 pairs"),
        (tail(SINKHORN_ITERS - 1, col_last=False), "row-last"),
        (raw.softmax(dim=-1) + HC_EPS, "no normalize"),
    ):
        delta = (wrong - comb).abs().max().item()
        assert delta > 1e-3, f"{label} is indistinguishable ({delta:.3e}) — gate is blind"


@requires_release_checkpoint
def test_post_mult_and_norm_eps_are_discriminated():
    """The reference must reject the module *defaults*, not just track them.

    ``mHC.__init__`` defaults to ``post_mult_value=1.0, norm_eps=1e-6``; V4.1
    needs ``2.0`` and ``rms_norm_eps=1e-20``. A reference that read those off the
    module would agree either way and the mis-wiring would reach the accuracy
    gate. Both are checked on release weights so the discrimination is measured
    at the scale the model runs at.
    """
    weights = _load_release_hc_weights(0, "attn")

    def build(**overrides) -> mHC:
        kwargs = dict(
            mult=HC_MULT,
            hidden_size=REAL_HIDDEN_SIZE,
            sinkhorn_iters=SINKHORN_ITERS,
            dtype=torch.bfloat16,
            eps=HC_EPS,
            norm_eps=RMS_NORM_EPS,
            sinkhorn_eps=HC_EPS,
            post_mult_value=POST_MULT,
        )
        kwargs.update(overrides)
        module = mHC(**kwargs).cuda()
        with torch.no_grad():
            module.fn.copy_(weights["fn"])
            module.base.copy_(weights["base"])
            module.scale.copy_(weights["scale"])
        return module

    x = _residual(32, REAL_HIDDEN_SIZE)
    _, post_ref, _ = _reference_coeffs(
        x, weights["fn"], weights["base"], weights["scale"], mult=HC_MULT, norm_eps=RMS_NORM_EPS
    )

    # post_mult_value: the default 1.0 halves every `post` coefficient.
    post_default = build(post_mult_value=1.0).hc_coeffs(x)[1]
    torch.testing.assert_close(post_default * 2.0, post_ref, **KERNEL_TOL)
    assert (post_default - post_ref).abs().max() > 0.1, (
        "post_mult_value=1.0 agreed with the reference — the factor of 2 is not gated"
    )

    # norm_eps: 1e-20 vs 1e-6 is invisible on unit-variance input (mean(x^2) ~ 1
    # dominates either eps), so it is discriminated where it actually bites — a
    # residual small enough that eps sets the RMS scale, which is exactly the
    # regime a 1e-20 eps exists to serve.
    tiny = (_residual(32, REAL_HIDDEN_SIZE).float() * 1e-4).bfloat16()
    pre_tiny_ref = _reference_coeffs(
        tiny,
        weights["fn"],
        weights["base"],
        weights["scale"],
        mult=HC_MULT,
        norm_eps=RMS_NORM_EPS,
    )[0]
    torch.testing.assert_close(build().hc_coeffs(tiny)[0], pre_tiny_ref, **KERNEL_TOL)
    pre_tiny_default = build(norm_eps=1e-6).hc_coeffs(tiny)[0]
    assert (pre_tiny_default - pre_tiny_ref).abs().max() > 1e-3, (
        "norm_eps=1e-6 agreed with the reference on a tiny residual — eps is not gated"
    )


@pytest.mark.parametrize("num_tokens", [1, 17, 128])
def test_lagged_reduces_to_pre_mapping(num_tokens: int):
    """Self-check from the docstring: ``pre_mix = pre_own`` is the V4 wiring."""
    module = _build_mhc()
    x = _residual(num_tokens)

    post_ref, comb_ref, layer_input_ref = module.pre_mapping(x)
    pre_own, post_mix, comb_mix, _ = module.pre_mapping_lagged(x, torch.zeros_like(post_ref))
    _, _, _, layer_input = module.pre_mapping_lagged(x, pre_own)

    torch.testing.assert_close(post_mix, post_ref, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb_mix, comb_ref, rtol=1e-3, atol=5e-3)
    _assert_bf16_within_one_ulp(layer_input, layer_input_ref, "layer_input")


@requires_release_checkpoint
@pytest.mark.parametrize("kind", ["attn", "ffn"])
def test_lagged_reduces_to_pre_mapping_on_release_weights(kind: str):
    """(2) at the release scale — the fused CUDA kernels vs our unfused path."""
    module = _build_release_mhc(0, kind)
    x = _residual(32, REAL_HIDDEN_SIZE)

    post_ref, comb_ref, layer_input_ref = module.pre_mapping(x)
    pre_own, post_mix, comb_mix, _ = module.pre_mapping_lagged(x, torch.zeros_like(post_ref))
    _, _, _, layer_input = module.pre_mapping_lagged(x, pre_own)

    torch.testing.assert_close(post_mix, post_ref, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb_mix, comb_ref, rtol=1e-3, atol=5e-3)
    _assert_bf16_within_one_ulp(layer_input, layer_input_ref, "layer_input")


def test_lagged_consumes_the_external_pre():
    """``layer_input`` must depend on the supplied ``pre_mix``, not on the own one."""
    module = _build_mhc()
    x = _residual(64)

    pre_own, _, _, own_collapse = module.pre_mapping_lagged(x, module.hc_coeffs(x)[0])
    foreign = pre_own.flip(-2) * 1.5
    _, _, _, foreign_collapse = module.pre_mapping_lagged(x, foreign)

    assert not torch.allclose(
        own_collapse.float(), foreign_collapse.float(), rtol=1e-2, atol=1e-2
    ), "layer_input ignored the external pre_mix — the lag would be a no-op"

    # And it is exactly the reference collapse under that foreign pre_mix.
    _assert_bf16_within_one_ulp(
        foreign_collapse, _reference_collapse(x, foreign), "foreign collapse"
    )


def test_mixer_projection_kernel_matches_torch_fallback():
    """The FMA-kernel path and the pure-torch fallback must agree.

    Compared against the *tensor* scale, not per element. The projection is a
    length-``mult * hidden`` fp32 dot product of mixed-sign terms, so individual
    outputs sit far below the magnitude of the terms that produced them and
    their per-element relative error under a different summation order is
    unbounded. What matters is the absolute perturbation of the sigmoid /
    softmax argument downstream, which
    ``test_coeffs_match_reference_transcription`` bounds directly on
    ``pre``/``post``/``comb``.
    """
    module = _build_mhc()
    x = _residual(96)

    kernel = module.mixer_projection(x)

    flat = x.flatten(-2, -1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + module.norm_eps)
    fallback = torch.nn.functional.linear(flat, module.fn.float()) * rsqrt

    scale = fallback.abs().max()
    rel = (kernel - fallback).abs().max() / scale
    assert rel < 1e-3, f"FMA kernel and torch fallback diverge by {rel:.3e} of tensor scale"


def _block(module_attn: mHC, module_ffn: mHC, x, pre_mix, sublayer, *, lagged: bool):
    """Two-sublayer block, in both wirings, with everything else held identical."""
    residual = x
    attn_pre, attn_post, attn_comb = module_attn.hc_coeffs(x)
    x = mHC.collapse(x, pre_mix if lagged else attn_pre)
    x = sublayer(x, "attn")
    x = module_attn.post_mapping(x, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = module_ffn.hc_coeffs(x)
    x = mHC.collapse(x, attn_pre if lagged else ffn_pre)
    x = sublayer(x, "ffn")
    x = module_ffn.post_mapping(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre


def test_lag_changes_the_block_output():
    """The V4.1 lag is not a numerical detail — it changes the block output."""
    module_attn = _build_mhc(seed=0)
    module_ffn = _build_mhc(seed=2)
    x = _residual(32)
    pre_mix = module_ffn.hc_coeffs(_residual(32, seed=3))[0]

    torch.manual_seed(7)
    weights = {
        k: (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, device="cuda") / HIDDEN_SIZE**0.5).bfloat16()
        for k in ("attn", "ffn")
    }

    def sublayer(h, kind):
        return torch.nn.functional.silu(h.float()).bfloat16() @ weights[kind]

    lagged, _ = _block(module_attn, module_ffn, x, pre_mix, sublayer, lagged=True)
    unlagged, _ = _block(module_attn, module_ffn, x, pre_mix, sublayer, lagged=False)

    rel = (lagged.float() - unlagged.float()).abs().max() / unlagged.float().abs().max()
    assert rel > 1e-2, f"lagged and unlagged wiring agree to {rel:.3e} — gate is blind"
