# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DFlash + Target-Side Saguaro Speculation Decoding (T-SSD).

Implements the design in
``tensorrt_llm/_torch/speculative/DFLASH_TARGET_SIDE_SSD_DESIGN.md``.

Phase 2 (production) entry point. See PHASE0_RESULTS.md and
phase1_reference.py for measurements and validation that motivated the
final spec.

Public surface:
    - ``DFlashTSSDWorker``: subclass of ``DFlashWorker`` adding the
      extended-verify + commit-on-hit / fallback-on-miss flow.
    - ``geometric_fanout``: Saguaro Theorem 12 fan-out solver.
    - ``build_tssd_mask``: §4.2 attention mask constructor (per request).
    - ``tssd_should_enable``: runtime gate (B, prefix_len, K, F_total).
    - ``sample_candidates``: residual / target-distribution sampler that
      builds Cand_k for k = 0..K (§3.3).

Hookup contract for the extended verify:
    The worker takes an injected ``extended_verify_fn`` callable (default
    ``None``, which makes the worker fall back to baseline DFlash). The
    callable signature is::

        extended_verify_fn(
            input_ids:   LongTensor [total_q],
            position_ids: LongTensor [total_q],
            qo_indptr:   IntTensor [B+1]      # offsets into input_ids
            custom_mask: BoolTensor [Σ q_b * kv_b]
            kv_indptr:   IntTensor [B+1]
            kv_seq_lens: IntTensor [B]
            ...                              # backend-specific extras
        ) -> dict with keys
            'logits':      [B, K, V]
            'cand_logits': [B, ΣF_k, V]      # not used in current commit
                                              # (we sample candidates from
                                              # the *previous* round)
            'cand_hidden_per_layer': dict[layer_id -> [B, ΣF_k, H]]

When the hook is None and ``spec_config.tssd_enabled`` is True we still
exercise candidate selection (so unit tests can verify the math) but the
forward delegates to the baseline DFlash forward — letting production
wire up a kernel without breaking the build.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .dflash import DFlashWorker

# ---------------------------------------------------------------------------
# Pure helpers (mirrors phase1_reference.py; kept self-contained here so the
# production path has no dependency on the reference module).
# ---------------------------------------------------------------------------


def geometric_fanout(K: int, B_budget: int, a_p: float = 0.78, r: float = 0.5) -> List[int]:
    """Saguaro Theorem 12 geometric fan-out solver.

    F_k    = F_0 · a_p^(k/(1+r))                  for k < K
    F_K    = F_0 · a_p^(K/(1+r)) · (1-a_p)^(-1/(1+r))
    Σ F_k = B_budget   →   solve F_0

    Returns integer ``F_k`` list (rounded), with min 1 per slot. Sum may
    drift from ``B_budget`` by at most ``K+1`` due to rounding; callers
    that need exact totals should re-tune ``B_budget``.
    """
    if K < 0:
        raise ValueError(f"K must be >= 0, got {K}")
    if B_budget == 0:
        return [0] * (K + 1)
    if B_budget < K + 1:
        # Tight budget: 1 slot for the first B_budget groups, 0 for the
        # rest. Sums exactly to B_budget. (Previously this returned
        # [1]*(K+1) which exceeded the budget and broke shape contracts.)
        return [1] * B_budget + [0] * (K + 1 - B_budget)
    rho = 1.0 / (1.0 + r)
    coeffs = [a_p ** (k * rho) for k in range(K)]
    coeffs.append(a_p ** (K * rho) * (1.0 - a_p) ** (-rho))
    F0 = B_budget / sum(coeffs)
    F = [max(1, int(round(F0 * c))) for c in coeffs]
    # Adjust to make sum exactly equal to B_budget (rounding can drift by ±K).
    drift = B_budget - sum(F)
    if drift > 0:
        # Add slots to the K-th (full-accept) group, which has highest mass.
        F[-1] += drift
    elif drift < 0:
        # Remove slots from groups with the largest count, never below 1.
        for _ in range(-drift):
            idx = max(range(len(F)), key=lambda i: F[i])
            if F[idx] > 1:
                F[idx] -= 1
            else:
                break
    return F


def build_tssd_mask(K: int, F: List[int], prefix_len: int) -> torch.Tensor:
    """Build the §4.2 mask for one request.

    Q             = K + Σ F_k
    KV-length     = prefix_len + K + Σ F_k
        [0, prefix_len)              prefix       (everyone attends)
        [prefix_len, prefix_len+K)   d_1..d_K     (main verify K/V)
        [prefix_len+K, prefix_len+Q) candidate    (own diagonal slot)

    Each query row attends to its own K/V (self) plus visible context per
    accept-length k. Returns a CPU boolean tensor of shape ``(Q, kv_len)``.
    """
    if len(F) != K + 1:
        raise ValueError(f"F must have length K+1={K + 1}, got {len(F)}")
    Q = K + sum(F)
    kv_len = prefix_len + Q
    mask = torch.zeros(Q, kv_len, dtype=torch.bool)

    mask[:, :prefix_len] = True

    # Main verify rows: row i attends to d_1..d_i AND self.
    for i in range(K):
        mask[i, prefix_len : prefix_len + i + 1] = True

    # Candidate rows: in group k, attend to d_1..d_k AND own diagonal slot.
    a = 0
    for k in range(K + 1):
        for _ in range(F[k]):
            row = K + a
            own_kv = prefix_len + K + a
            if k > 0:
                mask[row, prefix_len : prefix_len + k] = True
            mask[row, own_kv] = True
            a += 1

    return mask


def build_tssd_packed_mask(
    K: int,
    F: List[int],
    max_num_requests: int,
) -> torch.Tensor:
    """Build the §4.2 mask in TRTLLM bit-packed format.

    Local sequence (per gen request) has ``K + 1 + sum(F)`` positions:
        [0]                    bonus            (input = previous accepted)
        [1..K]                 main verify d_1..d_K
        [K+1..K+1+sum(F)-1]    candidates       (group order: k=0..K)

    Returned tensor shape: ``(max_num_requests, n, ceil(n/32))`` where
    ``n = K + 1 + sum(F)``. ``mask[req, q, blk]`` is a 32-bit int whose
    bit ``j`` (0-indexed) is 1 iff query position ``q`` may attend to
    LOCAL key position ``blk*32 + j``. Prefix is implicit (causal).

    Mask rules:
      - bonus (q=0): bit 0 set (self only).
      - main d_i (q=i, 1<=i<=K): bits 0..i set (causal: bonus + d_1..d_i).
      - candidate in group k (input c_{k,j} at q = K+1 + offset(k,j)):
        bits 0..k set (bonus + d_1..d_k) + own diagonal bit q.
        NOT bit for d_{k+1}..d_K, NOT bits for other candidates.

    The mask is identical for every request → one pattern, broadcast to
    all max_num_requests rows. Cached by ``functools.lru_cache``.
    """
    if len(F) != K + 1:
        raise ValueError(f"F must have length K+1={K + 1}, got {len(F)}")
    n = K + 1 + sum(F)
    num_blocks = math.ceil(n / 32)
    mask = torch.zeros((max_num_requests, n, num_blocks), dtype=torch.int32, device="cuda")

    # Build the pattern for one request as a python list, then copy.
    bits = [[False] * n for _ in range(n)]

    # Bonus row q=0: attends to self.
    bits[0][0] = True

    # Main verify rows q=1..K: causal over [0..q].
    for q in range(1, K + 1):
        for j in range(q + 1):
            bits[q][j] = True

    # Candidate rows.
    a = 0  # candidate index within all candidates
    for k in range(K + 1):
        for _ in range(F[k]):
            q = K + 1 + a
            # Bonus + d_1..d_k visible.
            for j in range(k + 1):  # j in [0..k] inclusive, includes bonus
                bits[q][j] = True
            # Own diagonal slot (the candidate's own K/V).
            bits[q][q] = True
            a += 1

    # Convert to packed int32 per row.
    for q in range(n):
        for blk in range(num_blocks):
            val = 0
            for j in range(32):
                col = blk * 32 + j
                if col < n and bits[q][col]:
                    val |= 1 << j
            mask[:, q, blk] = val

    return mask


def tssd_should_enable(
    B: int,
    prefix_len: int,
    K: int,
    F_total: int,
    max_batch: int = 2,
) -> bool:
    """Runtime gate matching PHASE0_RESULTS.md viable envelope.

    Piecewise rule fitted to the Phase 0 microbenchmark: T-SSD is viable
    only at low batch + bounded prefix·F_total product. Out-of-envelope
    configs disable T-SSD; the worker then runs the baseline DFlash
    verify path.
    """
    if B > max_batch:
        return False
    if F_total == 0:
        return False
    if F_total > 16:
        return False
    fp = F_total * prefix_len
    if B == 1:
        return fp <= 50_000
    if B == 2:
        return fp <= 12_000
    if B == 3:
        return fp <= 6_000
    return False


def sample_candidates(
    draft_logits: torch.Tensor,  # [B, K, V]
    target_logits: torch.Tensor,  # [B, K+1, V]
    F: List[int],  # length K+1
    mode: str = "topk",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample candidate token IDs Cand_k, k = 0..K (§3.3 + §6.2).

    Two modes:
      - ``"topk"``: deterministic top-F_k of the residual (k<K) / target
        (k=K) distribution. The reference path; what unit tests pin.
      - ``"target"``: top-F_k of target distribution at every k (ignores
        residual). Useful when draft_logits are not available (first
        iteration).

    Returns:
        candidate_tokens: LongTensor [B, K+1, max_F_k], padded with -1.
        candidate_log_p:  FloatTensor [B, K+1, max_F_k], padded with -inf.
    """
    B, K = target_logits.shape[0], target_logits.shape[1] - 1
    if len(F) != K + 1:
        raise ValueError(f"F must have length K+1={K + 1}, got {len(F)}")
    max_Fk = max(F)
    cand = torch.full(
        (B, K + 1, max_Fk), fill_value=-1, dtype=torch.long, device=target_logits.device
    )
    log_p = torch.full(
        (B, K + 1, max_Fk),
        fill_value=float("-inf"),
        dtype=torch.float32,
        device=target_logits.device,
    )

    target_log = torch.log_softmax(target_logits.float(), dim=-1)
    if mode == "topk":
        if draft_logits is not None:
            draft_log = torch.log_softmax(draft_logits.float(), dim=-1)
        else:
            draft_log = None

    for k in range(K + 1):
        Fk = F[k]
        if Fk == 0:
            continue
        if mode == "target" or k == K or draft_log is None:
            scores = target_log[:, k, :]
        else:
            # Residual r_k = max(p_target - p_draft, 0).
            # Operate in prob space, then drop back to log for top-k.
            p_t = target_log[:, k, :].exp()
            p_d = draft_log[:, k, :].exp()
            r = (p_t - p_d).clamp(min=1e-12)
            scores = r.log()
        top = torch.topk(scores, k=Fk, dim=-1)
        cand[:, k, :Fk] = top.indices
        log_p[:, k, :Fk] = top.values

    return cand, log_p


# ---------------------------------------------------------------------------
# Scratch state container
# ---------------------------------------------------------------------------


@dataclass
class TSSDState:
    """Per-worker scratch buffers for T-SSD candidate K/V and hidden states.

    All buffers are pre-allocated by ``_lazy_init_tssd_state`` and reused
    across iterations. Sized by ``max_batch * F_total`` so CUDA graphs see
    constant shapes.
    """

    K: int
    F_total: int
    F: List[int]
    max_batch: int

    # Saguaro fan-out as a CUDA tensor for kernel-side use.
    fan_out: torch.Tensor = field(default=None)  # [K+1] int32

    # Per-iteration: candidate token IDs at each k.
    candidate_tokens: torch.Tensor = field(default=None)  # [max_batch, K+1, max_F_k]

    # Per-iteration: candidate target hidden states (capture layers).
    # Shape: [num_capture_layers, max_batch, F_total, hidden]
    cand_target_hidden: torch.Tensor = field(default=None)

    # Per-iteration: per-candidate target main-KV scratch.
    # Shape: [num_target_layers, max_batch, F_total, num_kv_heads, head_dim]
    cand_target_k: torch.Tensor = field(default=None)
    cand_target_v: torch.Tensor = field(default=None)

    # Per-iteration: per-candidate dflash ctx K/V (already projected
    # through fc + hidden_norm + RoPE). Read on commit.
    # Shape: [num_draft_layers, max_batch, F_total, num_kv_heads_d, head_dim_d]
    cand_dflash_k: torch.Tensor = field(default=None)
    cand_dflash_v: torch.Tensor = field(default=None)

    @property
    def max_F_k(self) -> int:
        return max(self.F)

    def reset_for_iter(self, batch_size: int) -> None:
        """No-op when we always overwrite all batch slots; kept as an
        explicit hook for future debugging / sanity-zeroing."""
        pass


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


# Type alias for the injected extended-verify callable.  The contract is
# documented at the top of this module.
ExtendedVerifyFn = Callable[..., Dict[str, Any]]


class DFlashTSSDWorker(DFlashWorker):
    """DFlash worker with target-side multi-outcome speculation (T-SSD).

    Adds:
      - candidate selection from the previous round's logits (§3.3, §6.2)
      - extended verify with the §4.2 custom mask (delegated to an
        injected callable; defaults to None → fall back to baseline)
      - commit-on-hit copying candidate K/V into target main paged KV
        and pre-projected ctx K/V into ``_ctx_k_buf`` / ``_ctx_v_buf``
        (§5.2, §6.3)
      - per-batch fallback when the speculation cache misses (§6.4)

    Thin wrapper: when ``spec_config.tssd_enabled`` is False (default) or
    the runtime gate disables T-SSD for the current iteration, this class
    behaves identically to ``DFlashWorker``.
    """

    def __init__(
        self,
        spec_config,
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
        extended_verify_fn: Optional[ExtendedVerifyFn] = None,
    ):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        self.tssd_enabled: bool = bool(getattr(spec_config, "tssd_enabled", False))
        self.tssd_F_total: int = int(getattr(spec_config, "tssd_F_total", 8))
        self.tssd_a_p: float = float(getattr(spec_config, "tssd_a_p", 0.78))
        self.tssd_max_batch: int = int(getattr(spec_config, "tssd_max_batch", 2))
        self._extended_verify_fn: Optional[ExtendedVerifyFn] = extended_verify_fn

        if self.tssd_enabled and self.tssd_F_total > 0:
            F_list = geometric_fanout(
                K=self.max_draft_len,
                B_budget=self.tssd_F_total,
                a_p=self.tssd_a_p,
            )
            actual = sum(F_list)
            if actual != self.tssd_F_total:
                logger.info(
                    f"DFlashTSSD: requested F_total={self.tssd_F_total}, "
                    f"geometric fan-out rounds to {actual} ({F_list})"
                )
            self._tssd_F_list: List[int] = F_list
            self._tssd_active = True
        else:
            # F_total=0 (or tssd_enabled=False): all-zero list, scaffold no-op.
            self._tssd_F_list = [0] * (self.max_draft_len + 1)
            self._tssd_active = False
            # Force the worker's effective tssd_enabled=False when budget is 0
            # so DFlashTSSDWorker.forward immediately delegates to baseline.
            if self.tssd_F_total == 0:
                self.tssd_enabled = False

        self.tssd_state: Optional[TSSDState] = None

        # Counters surfaced to perf-bench / tests.
        self._tssd_iters_total = 0
        self._tssd_iters_gated_off = 0
        self._tssd_iters_gated_on = 0
        self._tssd_hits = 0
        self._tssd_misses = 0

        # Multi-stream infrastructure for parallel candidate forward.
        # Pre-allocated here so the stream + events are stable references
        # across CUDA graph capture/replay (TRTLLM's
        # ``maybe_execute_in_parallel`` records stream-switch + event ops
        # into the captured graph; needs the objects to exist beforehand).
        if torch.cuda.is_available():
            self._aux_stream = torch.cuda.Stream()
            self._event_main = torch.cuda.Event()
            self._event_aux = torch.cuda.Event()
        else:
            self._aux_stream = None
            self._event_main = None
            self._event_aux = None

        if self.tssd_enabled:
            logger.warning(
                f"DFlashTSSDWorker: tssd_enabled=True, F_total={self.tssd_F_total}, "
                f"F={self._tssd_F_list}, a_p={self.tssd_a_p}, "
                f"max_batch={self.tssd_max_batch}.\n"
                f"WARNING: T-SSD production wiring is incomplete. The math/state-machine "
                f"layer ({len(self._tssd_F_list)}-group fan-out, candidate selection, "
                f"hit/miss detection, scratch buffers) is implemented and unit-tested, "
                f"but candidate token injection, TRTLLM packed-mask hookup, and "
                f"commit-on-hit are pending. Setting tssd_enabled=True without these "
                f"will grow tokens_per_gen_step to K+1+F_total but feed garbage into "
                f"the unused slots. See PHASE2_PROGRESS.md for the remaining work."
            )
        else:
            logger.info("DFlashTSSDWorker: tssd_enabled=False, behaves as DFlashWorker")

    # ---- public attribute access for tests ----
    @property
    def F(self) -> List[int]:
        return list(self._tssd_F_list)

    @property
    def F_total(self) -> int:
        return sum(self._tssd_F_list)

    def set_extended_verify_fn(self, fn: Optional[ExtendedVerifyFn]) -> None:
        """Production wiring hook: install or replace the extended-verify
        callable. Tests use the default ``None`` and exercise math paths
        directly; production wires this to the FlashInfer custom-mask
        attention path."""
        self._extended_verify_fn = fn

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict[str, int]:
        out = dict(
            iters_total=self._tssd_iters_total,
            iters_gated_on=self._tssd_iters_gated_on,
            iters_gated_off=self._tssd_iters_gated_off,
            hits=self._tssd_hits,
            misses=self._tssd_misses,
        )
        if hasattr(self, "_tssd_hits_gpu"):
            h = int(self._tssd_hits_gpu.item())
            t = int(self._tssd_total_gpu.item())
            out["hits_gpu"] = h
            out["total_gpu"] = t
            out["hit_rate"] = h / t if t > 0 else 0.0
        return out

    def __del__(self):
        # Best-effort: print hit-rate stats at shutdown if collected.
        try:
            if hasattr(self, "_tssd_hits_gpu"):
                h = int(self._tssd_hits_gpu.item())
                t = int(self._tssd_total_gpu.item())
                if t > 0:
                    import sys

                    rate = h / t
                    print(
                        f"[T-SSD final] hits={h} / total={t} hit_rate={rate:.4f}",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception:
            pass

    def reset_stats(self) -> None:
        self._tssd_iters_total = 0
        self._tssd_iters_gated_off = 0
        self._tssd_iters_gated_on = 0
        self._tssd_hits = 0
        self._tssd_misses = 0

    # ------------------------------------------------------------------ scratch
    def _lazy_init_tssd_state(
        self,
        draft_model: nn.Module,
        target_num_layers: int,
        target_num_kv_heads: int,
        target_head_dim: int,
        target_hidden: int,
        num_capture_layers: int,
        max_batch: int,
        kv_dtype: torch.dtype = torch.bfloat16,
        hidden_dtype: torch.dtype = torch.bfloat16,
    ) -> TSSDState:
        if self.tssd_state is not None:
            return self.tssd_state

        F = self._tssd_F_list
        F_total = sum(F)
        K = self.max_draft_len
        max_F_k = max(F) if F else 0

        # Draft K/V geometry (mirrors what _lazy_init_ctx_buffers reads).
        L_d = getattr(draft_model, "_num_attn_layers", target_num_layers)
        nkv_d = getattr(draft_model, "_num_kv_heads", target_num_kv_heads)
        hd_d = getattr(draft_model, "_head_dim", target_head_dim)

        device = "cuda"
        cand_tok = torch.full(
            (max_batch, K + 1, max(1, max_F_k)),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        cand_h = torch.zeros(
            (num_capture_layers, max_batch, max(1, F_total), target_hidden),
            dtype=hidden_dtype,
            device=device,
        )
        cand_tk = torch.zeros(
            (target_num_layers, max_batch, max(1, F_total), target_num_kv_heads, target_head_dim),
            dtype=kv_dtype,
            device=device,
        )
        cand_tv = torch.zeros_like(cand_tk)
        cand_dk = torch.zeros(
            (L_d, max_batch, max(1, F_total), nkv_d, hd_d),
            dtype=kv_dtype,
            device=device,
        )
        cand_dv = torch.zeros_like(cand_dk)
        fanout = torch.tensor(F, dtype=torch.int32, device=device)

        self.tssd_state = TSSDState(
            K=K,
            F_total=F_total,
            F=list(F),
            max_batch=max_batch,
            fan_out=fanout,
            candidate_tokens=cand_tok,
            cand_target_hidden=cand_h,
            cand_target_k=cand_tk,
            cand_target_v=cand_tv,
            cand_dflash_k=cand_dk,
            cand_dflash_v=cand_dv,
        )
        logger.info(
            f"DFlashTSSD: scratch allocated "
            f"K={K}, F_total={F_total}, max_batch={max_batch}, "
            f"target_KV=[{target_num_layers}, *, {F_total}, {target_num_kv_heads}, {target_head_dim}], "
            f"draft_KV=[{L_d}, *, {F_total}, {nkv_d}, {hd_d}]"
        )
        return self.tssd_state

    # ------------------------------------------------------------------ candidate selection
    def select_candidates(
        self,
        draft_logits: Optional[torch.Tensor],
        target_logits: torch.Tensor,
        mode: str = "topk",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build Cand_k for k = 0..K from previous-round logits (§6.2).

        Returns ``(candidate_tokens, candidate_log_p)``.  The token tensor
        is *not* written into ``self.tssd_state.candidate_tokens`` — that
        copy is the caller's responsibility (so unit tests can inspect
        without touching scratch).
        """
        return sample_candidates(
            draft_logits=draft_logits,
            target_logits=target_logits,
            F=self._tssd_F_list,
            mode=mode,
        )

    # ------------------------------------------------------------------ commit
    def try_commit_candidate(
        self,
        num_accepted_tokens: torch.Tensor,  # [B] int
        accepted_tokens: torch.Tensor,  # [B, K+1] int (k* th column = x*)
        candidate_tokens: torch.Tensor,  # [B, K+1, max_F_k] int (-1 pad)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For each request: was x* in Cand_{k*}?

        Returns:
            hit_flag: BoolTensor [B]
            cand_idx: LongTensor [B], position in Cand_{k*} (or -1 on miss)

        Lossless: we accept x* unchanged regardless of cache hit. The
        cache is a *speedup*, not a sampling perturbation.
        """
        B = accepted_tokens.shape[0]
        K_plus_1 = candidate_tokens.shape[1]
        device = accepted_tokens.device

        # k_star = num_accepted_tokens - 1 if num_accepted_tokens >= 1
        # else 0 (no draft accepted; x* is sampled at position 0).
        # In DFlash partial-accept semantics, accepted_tokens[:, k_star]
        # is x*. num_accepted_tokens >= 1 always (the bonus token at
        # position 0 is always accepted), so k_star >= 0.
        k_star = (num_accepted_tokens - 1).clamp(min=0).long()  # [B]
        # Some workloads use num_accepted == 1 for "rejected at first";
        # cap at K to stay within Cand_0..Cand_K bounds.
        k_star = k_star.clamp(max=K_plus_1 - 1)

        x_star = accepted_tokens.gather(1, k_star.unsqueeze(1)).squeeze(1)  # [B]

        # Gather candidate row for each request at k_star.
        # candidate_tokens: [B, K+1, max_F_k] → row[b] = cand[b, k_star[b], :]
        b_idx = torch.arange(B, device=device)
        cand_row = candidate_tokens[b_idx, k_star, :]  # [B, max_F_k]

        # Compare each candidate vs x*. Padding = -1, which won't match
        # any valid token id. First-match index per row.
        match = cand_row == x_star.unsqueeze(1)  # [B, max_F_k]
        any_hit = match.any(dim=1)  # [B]

        # argmax on bool returns first True index, or 0 if all False.
        first_hit_idx = torch.argmax(match.int(), dim=1).long()
        cand_idx = torch.where(any_hit, first_hit_idx, torch.full_like(first_hit_idx, -1))

        return any_hit, cand_idx

    def commit_candidate_to_buffers(
        self,
        hit_flag: torch.Tensor,  # [B] bool
        cand_idx: torch.Tensor,  # [B] long
        num_accepted_tokens: torch.Tensor,  # [B] int
        slots: torch.Tensor,  # [B] long  (DFlash _batch_to_slot[gen])
        prefix_lens: torch.Tensor,  # [B] long  (target |P| at start of round)
        target_main_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """On hit: copy candidate K/V into target main paged KV and copy
        pre-projected dflash ctx K/V into ``_ctx_k_buf`` / ``_ctx_v_buf``
        at logical position ``|P| + k*``.  On miss: no-op (caller runs
        baseline DFlash fixup).

        ``target_main_kv`` is a ``(K, V)`` pair of paged-cache tensors.
        We commit to slot ``slots[b]`` at position ``prefix_lens[b] + k*``.
        Production may have to translate logical positions → page indices;
        this routine writes to a logical-flat tensor so the kernel side
        stays simple. When ``target_main_kv`` is None we skip the target
        commit (test path; the toy transformer has no main KV cache).
        """
        if self.tssd_state is None:
            return
        st = self.tssd_state

        # Number of hits — early out.
        if not hit_flag.any():
            return

        b_hit = torch.nonzero(hit_flag, as_tuple=False).flatten()  # [n_hit]
        slot_hit = slots[b_hit]
        idx_hit = cand_idx[b_hit]
        k_star = (num_accepted_tokens[b_hit] - 1).clamp(min=0).long()
        target_pos = prefix_lens[b_hit] + k_star  # [n_hit]

        # 1. Commit dflash ctx K/V at slot position |P|+k*.
        # st.cand_dflash_k: [L_d, max_batch, F_total, nkv_d, hd_d]
        #   → slice with (slot_hit, idx_hit) gives [L_d, n_hit, nkv_d, hd_d]
        # Permute for advanced indexing: we want [n_hit, L_d, nkv_d, hd_d].
        kvecs = st.cand_dflash_k[:, slot_hit, idx_hit, :, :]  # [L_d, n_hit, nkv_d, hd_d]
        vvecs = st.cand_dflash_v[:, slot_hit, idx_hit, :, :]
        # _ctx_k_buf: [max_batch, L, max_ctx+block, nkv, hd]
        # Write into [slot_hit, :, target_pos, :, :].
        # Broadcasting needs care: do per-hit assignment.
        if self._ctx_k_buf is not None:
            for i in range(b_hit.shape[0]):
                s = int(slot_hit[i].item())
                p = int(target_pos[i].item())
                # Stay within buffer bounds: max_ctx + block.
                p = min(p, self._ctx_k_buf.shape[2] - 1)
                self._ctx_k_buf[s, :, p, :, :] = kvecs[:, i, :, :]
                self._ctx_v_buf[s, :, p, :, :] = vvecs[:, i, :, :]
                # _ctx_len[s] = max(_ctx_len[s], p+1) — the position is now valid.
                self._ctx_len[s] = max(int(self._ctx_len[s].item()), p + 1)

        # 2. Optionally commit target main-KV scratch into the request's
        # paged target KV cache. In production, the kernel-side hookup
        # will pass an indexable view of the paged KV; here we accept a
        # logical-flat tensor for tests.
        if target_main_kv is not None:
            t_k, t_v = target_main_kv
            # t_k: [num_layers, total_logical_positions, nkv, hd]
            # st.cand_target_k: [num_layers, max_batch, F_total, nkv, hd]
            for i in range(b_hit.shape[0]):
                s = int(slot_hit[i].item())
                p = int(target_pos[i].item())
                t_k[:, s, p, :, :] = st.cand_target_k[:, s, idx_hit[i], :, :]
                t_v[:, s, p, :, :] = st.cand_target_v[:, s, idx_hit[i], :, :]

        self._tssd_hits += int(hit_flag.sum().item())
        self._tssd_misses += int((~hit_flag).sum().item())

    # ------------------------------------------------------------------ gating
    def gate(self, batch_size: int, prefix_len: int) -> bool:
        if not self.tssd_enabled:
            return False
        return tssd_should_enable(
            B=batch_size,
            prefix_len=int(prefix_len),
            K=self.max_draft_len,
            F_total=self.F_total,
            max_batch=self.tssd_max_batch,
        )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        """T-SSD forward (multi-stream scaffold).

        Main verify processes only K+1 tokens (baseline shape). Candidate
        Q processing — when integrated — runs on a separate CUDA stream
        in parallel with super().forward (which does accept + draft +
        samples). This way candidate cost overlaps with the rest of the
        iter rather than serializing. See §7.2 of the design doc.

        Current state: scaffold + multi-stream proxy compute. Full
        candidate forward integration (target.model on stream_B with
        §4.2 mask + kv_append_lens=0) is the next implementer's task.
        """
        self._tssd_iters_total += 1

        if not self.tssd_enabled:
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                resource_manager=resource_manager,
            )

        F_total = self.F_total
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        # Multi-stream candidate work via TRTLLM's
        # ``maybe_execute_in_parallel`` — capture-safe, replays correctly
        # under CUDA graphs. Pattern:
        #   * fn0 (default stream): super().forward (baseline DFlash)
        #   * fn1 (aux stream): candidate proxy compute (placeholder for
        #     real candidate target forward; will be replaced once the
        #     full integration lands)
        # The aux_stream and events are pre-allocated once (in __init__'s
        # parent process before capture); inside capture they're recorded
        # as graph ops and replayed every iter.
        from ..modules.multi_stream_utils import do_multi_stream, maybe_execute_in_parallel

        if (
            num_gens > 0
            and self._tssd_active
            and F_total > 0
            and getattr(self, "_aux_stream", None) is not None
            and do_multi_stream()
        ):
            # Closure for default-stream fn0 (the actual DFlash work).
            out_holder = {}

            def _fn_main():
                out_holder["out"] = super(DFlashTSSDWorker, self).forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    logits=logits,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    draft_model=draft_model,
                    resource_manager=resource_manager,
                )
                return out_holder["out"]

            def _fn_candidate():
                # Realistic candidate-forward proxy. Models per-token
                # cost of running through (last_capture_layer + 1)
                # target transformer layers. For gpt-oss-120b that's
                # 34 of 36 (last captured = layer 33). Per layer we
                # use one H×H GEMM (representing attention QKV proj;
                # MoE FFN is ~3% of dense at top-4/128 so we omit it).
                if not hasattr(self, "_proxy_in"):
                    H = hidden_states.shape[-1] if hidden_states is not None else 4096
                    proxy_n = max(1, num_gens * F_total)
                    self._proxy_in = torch.zeros(
                        (proxy_n, H), dtype=hidden_states.dtype, device=hidden_states.device
                    )
                    self._proxy_w = torch.randn(
                        (H, H), dtype=hidden_states.dtype, device=hidden_states.device
                    )
                    self._proxy_layers = 34
                x = self._proxy_in
                for _ in range(self._proxy_layers):
                    x = x @ self._proxy_w
                return x

            _, _ = maybe_execute_in_parallel(
                _fn_main,
                _fn_candidate,
                self._event_main,
                self._event_aux,
                aux_stream=self._aux_stream,
                disable_on_compile=True,
            )
            out = out_holder["out"]
            self._tssd_iters_gated_on += 1
            return out

        # Fallback (no multi-stream): plain super().forward.
        out = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
            resource_manager=resource_manager,
        )

        self._tssd_iters_gated_on += 1
        return out


__all__ = [
    "DFlashTSSDWorker",
    "TSSDState",
    "geometric_fanout",
    "build_tssd_mask",
    "build_tssd_packed_mask",
    "tssd_should_enable",
    "sample_candidates",
]
