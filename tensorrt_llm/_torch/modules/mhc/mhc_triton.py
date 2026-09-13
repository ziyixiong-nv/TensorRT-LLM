# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Triton kernel for the whole V4.1 mHC coefficient tail, and one for the collapse.

Replaces the ~129 eager kernels that ``mHC.split_sinkhorn`` plus the rsqrt tail
of ``mHC.mixer_projection`` launch per sublayer with a single launch. The tail
operates on ``[M, mult]`` and ``[M, mult, mult]`` fp32 tensors with ``mult == 4``,
so every intermediate fits in registers and the entire ``sinkhorn_iters``-long
alternating normalization runs without a round trip to HBM.

Why it matters: ``pre_mapping_lagged`` runs twice per V4.1 layer, so at 40 layers
the eager tail is ~10.8k launches per forward. Tech-report §3.2 budgets *11
kernels for an entire Reuse-Mode decode layer*.

Semantics are transcribed from ``mHC.split_sinkhorn``, which is itself a
transcription of the reference ``hc_split_sinkhorn`` tilelang kernel. Two details
are load-bearing and easy to lose:

* ``pre`` gets ``+ eps``, ``post`` does not.
* the loop runs ``sinkhorn_iters - 1`` row/column *pairs* after the initial
  softmax + column normalize, so it ends on a column normalize — rows are
  deliberately not unit-sum.

``comb`` is ``unflatten(-1, (n, n))`` of the trailing ``n * n`` mixes, so element
``[i, j]`` is ``mixes[2n + i * n + j]``; ``dim=-1`` sums over ``j`` (rows) and
``dim=-2`` over ``i`` (columns).
"""

import torch

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:  # pragma: no cover - exercised only on builds without triton
    _triton_available = False
    triton = None
    tl = None


if _triton_available:

    @triton.jit
    def _mhc_coeff_tail_kernel(
        y_ptr,  # [M, mix_hc] fp32 raw mixer GEMM accumulator
        r_ptr,  # [M]         fp32 row sum of squares
        scale_ptr,  # [3]     fp32
        base_ptr,  # [mix_hc] fp32
        pre_ptr,  # [M, N]    fp32 out
        post_ptr,  # [M, N]   fp32 out
        comb_ptr,  # [M, N, N] fp32 out
        M,
        hc_dim,  # mult * hidden_size, as fp32
        norm_eps,
        eps,
        sinkhorn_eps,
        post_mult_value,
        # `sinkhorn_iters - 1`. constexpr so the trip count is fully unrolled --
        # it is a fixed architecture constant (20 on the release), so there is
        # exactly one specialization, and unrolling keeps the whole alternating
        # normalization in registers with no loop-carried memory traffic.
        SINKHORN_REPEAT: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,  # next power of two >= N
    ):
        m = tl.program_id(0)
        if m >= M:
            return

        mix_hc = (2 + N) * N

        # --- the rsqrt that mixer_projection applies to the GEMM result --------
        r = tl.load(r_ptr + m).to(tl.float32)
        # Two deliberate choices, both to stay on the eager path's arithmetic:
        #
        # * divide by `hc_dim` rather than multiply by a folded `1 / hc_dim` --
        #   hc_dim (20480 on the release) is not a power of two, so the folded
        #   form would round differently from `r_acc / self.hc_dim` for no gain;
        # * `tl.rsqrt`, not `1 / tl.sqrt(...)` -- these are different CUDA
        #   operations (the approximate reciprocal square root against a
        #   correctly-rounded sqrt and divide), and the eager path calls
        #   `torch.rsqrt`. Bit-exactness is not claimed either way; the point is
        #   not to diverge from the reference on purpose.
        rsqrt = tl.rsqrt(r / hc_dim + norm_eps)

        j = tl.arange(0, BLOCK_N)
        keep = j < N
        row_base = m * mix_hc

        scale0 = tl.load(scale_ptr + 0).to(tl.float32)
        scale1 = tl.load(scale_ptr + 1).to(tl.float32)
        scale2 = tl.load(scale_ptr + 2).to(tl.float32)

        # --- pre: sigmoid(mixes * scale0 + base) + eps ------------------------
        y_pre = tl.load(y_ptr + row_base + j, mask=keep, other=0.0).to(tl.float32) * rsqrt
        b_pre = tl.load(base_ptr + j, mask=keep, other=0.0).to(tl.float32)
        pre = tl.sigmoid(y_pre * scale0 + b_pre) + eps
        tl.store(pre_ptr + m * N + j, pre, mask=keep)

        # --- post: post_mult_value * sigmoid(...), no eps ---------------------
        y_post = tl.load(y_ptr + row_base + N + j, mask=keep, other=0.0).to(tl.float32) * rsqrt
        b_post = tl.load(base_ptr + N + j, mask=keep, other=0.0).to(tl.float32)
        post = post_mult_value * tl.sigmoid(y_post * scale1 + b_post)
        tl.store(post_ptr + m * N + j, post, mask=keep)

        # --- comb: [N, N], softmax over the last dim then Sinkhorn ------------
        oi = tl.arange(0, BLOCK_N)[:, None]
        oj = tl.arange(0, BLOCK_N)[None, :]
        mask2 = (oi < N) & (oj < N)
        flat = 2 * N + oi * N + oj

        y_comb = tl.load(y_ptr + row_base + flat, mask=mask2, other=0.0).to(tl.float32) * rsqrt
        b_comb = tl.load(base_ptr + flat, mask=mask2, other=0.0).to(tl.float32)
        comb = y_comb * scale2 + b_comb

        # Max-subtracted to match torch.softmax; padding lanes must not win the
        # max, so push them to -inf first and zero them straight after.
        comb = tl.where(mask2, comb, float("-inf"))
        comb = comb - tl.max(comb, axis=1)[:, None]
        comb = tl.exp(comb)
        comb = tl.where(mask2, comb, 0.0)
        comb = comb / tl.sum(comb, axis=1)[:, None]
        comb = comb + sinkhorn_eps
        comb = tl.where(mask2, comb, 0.0)

        # Initial column normalize, then `sinkhorn_iters - 1` row/column pairs.
        comb = comb / (tl.sum(comb, axis=0)[None, :] + sinkhorn_eps)
        for _ in tl.static_range(SINKHORN_REPEAT):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + sinkhorn_eps)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + sinkhorn_eps)

        tl.store(comb_ptr + m * N * N + oi * N + oj, comb, mask=mask2)


if _triton_available:

    @triton.jit
    def _mhc_collapse_kernel(
        x_ptr,  # [M, N, H] bf16 residual stream
        pre_ptr,  # [M, N]  fp32 lagged `pre` coefficients
        out_ptr,  # [M, H]  bf16 out
        H,
        N: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        h0 = tl.program_id(1) * BLOCK_H
        h = h0 + tl.arange(0, BLOCK_H)
        keep = h < H

        # fp32 accumulation, matching ``mHC.collapse``: the reference sums the
        # weighted heads in fp32 and casts once at the end.
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for i in tl.static_range(N):
            xi = tl.load(x_ptr + m * N * H + i * H + h, mask=keep, other=0.0).to(tl.float32)
            wi = tl.load(pre_ptr + m * N + i).to(tl.float32)
            acc += xi * wi

        # No explicit cast: `tl.store` converts to the pointee dtype, so this
        # follows `x.dtype` instead of silently rounding an fp32 caller to bf16.
        tl.store(out_ptr + m * H + h, acc, mask=keep)


def mhc_collapse_triton(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """``mHC.collapse`` in one launch: ``[..., N, H] x [..., N] -> [..., H]``.

    The eager version slices one head at a time and needs ``3 * mult`` launches
    plus an fp32 upcast per head; here the whole ``mult``-way weighted sum stays
    in registers and each input element is read once.
    """
    if not _triton_available:
        raise RuntimeError("triton is unavailable")

    outer = x.shape[:-2]
    mult, hidden = x.shape[-2], x.shape[-1]
    x_flat = x.reshape(-1, mult, hidden).contiguous()
    m = x_flat.shape[0]
    pre_flat = pre_mix.reshape(m, mult).to(torch.float32).contiguous()

    out = torch.empty((m, hidden), dtype=x.dtype, device=x.device)
    block_h = 1024
    _mhc_collapse_kernel[(m, triton.cdiv(hidden, block_h))](
        x_flat,
        pre_flat,
        out,
        hidden,
        N=mult,
        BLOCK_H=block_h,
        # Keep the multiply and the add separate instead of letting the compiler
        # contract ``acc += xi * wi`` into an FMA. Both forms are correct -- and
        # measured against fp64 the FMA form is if anything the more accurate one
        # -- but the eager ``mHC.collapse`` does a separate fp32 multiply and add,
        # and matching it bit-for-bit is worth more here than the fraction of an
        # ULP the FMA would buy: it keeps the fused path a drop-in replacement, so
        # the tests can assert equality rather than a tolerance, and switching a
        # layer onto it cannot shift logits. vLLM disables it on the same kernel
        # for the same reason.
        enable_fp_fusion=False,
    )
    return out.reshape(*outer, hidden)


def mhc_coeff_tail_triton(
    y_acc: torch.Tensor,
    r_acc: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    mult: int,
    hc_dim: int,
    norm_eps: float,
    eps: float,
    sinkhorn_eps: float,
    post_mult_value: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y_acc, r_acc) -> (pre, post, comb)`` in one launch.

    ``y_acc``/``r_acc`` are exactly what ``mhc_gemm_rms_fma_cuda`` returns, i.e.
    the *raw* GEMM accumulator and row sum of squares — the per-token rsqrt is
    applied inside the kernel, so this subsumes the tail of
    ``mHC.mixer_projection`` as well as all of ``mHC.split_sinkhorn``.
    """
    if not _triton_available:
        raise RuntimeError("triton is unavailable")

    m = y_acc.shape[0]
    assert y_acc.shape[-1] == (2 + mult) * mult, y_acc.shape
    assert r_acc.numel() == m, r_acc.shape

    y_acc = y_acc.contiguous()
    r_acc = r_acc.reshape(m).contiguous()
    scale = scale.to(torch.float32).contiguous()
    base = base.to(torch.float32).contiguous()

    pre = torch.empty((m, mult), dtype=torch.float32, device=y_acc.device)
    post = torch.empty((m, mult), dtype=torch.float32, device=y_acc.device)
    comb = torch.empty((m, mult, mult), dtype=torch.float32, device=y_acc.device)

    _mhc_coeff_tail_kernel[(m,)](
        y_acc,
        r_acc,
        scale,
        base,
        pre,
        post,
        comb,
        m,
        float(hc_dim),
        float(norm_eps),
        float(eps),
        float(sinkhorn_eps),
        float(post_mult_value),
        SINKHORN_REPEAT=sinkhorn_iters - 1,
        N=mult,
        # ``mult`` is 4 on the release, so this is exact and there are no padding
        # lanes at all. The masking above only matters for a non-power-of-two
        # ``mult``, where the padded lanes transiently go NaN (``-inf - -inf`` in
        # the softmax, ``0 / 0`` in its normalize) and are re-zeroed before they
        # can reach any reduction.
        BLOCK_N=triton.next_power_of_2(mult),
        num_warps=1,
        # Same reason as the collapse kernel: the three ``mixes * scale + base``
        # affine maps are separate multiplies and adds in the reference, so do not
        # let them contract. The Sinkhorn loop itself is divides and adds, which
        # never contract, so this costs nothing there.
        enable_fp_fusion=False,
    )
    return pre, post, comb
