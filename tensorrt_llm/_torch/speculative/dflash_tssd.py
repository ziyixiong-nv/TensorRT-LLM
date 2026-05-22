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
        # Fused forward (Phase 2): per-gen Q grows from K+1 to K+1+F_total.
        # Affects logits / accepted_tokens shape — worker must slice off
        # the F_total candidate rows before delegating to the K+1 verify.
        self.tssd_fused_forward: bool = bool(getattr(spec_config, "tssd_fused_forward", False))
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
            # Stream priority: 0 = default. Tested -1 (lower) and +1 (higher)
            # — neither moved the needle on B200, contention is at the SM
            # level not the scheduler level. Default is fine.
            self._aux_stream = torch.cuda.Stream()
            self._event_main = torch.cuda.Event()
            self._event_aux = torch.cuda.Event()
        else:
            self._aux_stream = None
            self._event_main = None
            self._event_aux = None

        # ---- Capture-safe hit-rate measurement state ----
        # Top-F for hit-rate proxy. Picks F=4 (a representative T-SSD
        # candidate width) so hit-rate ≈ "would top-4 candidates from the
        # previous iter have covered the bonus token at this iter's k_star".
        # Read TSSD_HIT_F env var to override.
        import os as _os

        self._tssd_hit_F = max(1, int(_os.environ.get("TSSD_HIT_F", "4")))
        self._tssd_hit_enabled = (
            torch.cuda.is_available()
            and self.tssd_enabled
            and bool(int(_os.environ.get("TSSD_MEASURE_HIT_RATE", "1")))
        )

        if self._tssd_hit_enabled:
            K = self.max_draft_len
            max_b = max(1, int(self.tssd_max_batch))
            # Persistent counters (single-element int64 GPU tensors).
            self._tssd_hits_gpu = torch.zeros(1, dtype=torch.int64, device="cuda")
            self._tssd_total_gpu = torch.zeros(1, dtype=torch.int64, device="cuda")
            self._tssd_inrange_gpu = torch.zeros(1, dtype=torch.int64, device="cuda")
            # Phase 3 commit-on-hit eligibility counter: candidate INPUT
            # equals iter t's bonus token (Lemma 1 input match).
            self._tssd_input_match_gpu = torch.zeros(1, dtype=torch.int64, device="cuda")
            # Last-iter top-F predicted tokens, per (gen request, slot, F).
            # Init to -1 so first-iter compares miss naturally.
            self._tssd_last_cands_gpu = torch.full(
                (max_b, K + 1, self._tssd_hit_F),
                fill_value=-1,
                dtype=torch.int64,
                device="cuda",
            )
            # Last iter's k_star per gen request. Init to K so first-iter
            # target_slot = K + k_star_t + 1 > K → always out-of-range miss.
            self._tssd_last_k_star_gpu = torch.full(
                (max_b,), fill_value=K, dtype=torch.int64, device="cuda"
            )
            # Last-iter draft top-F predictions per (gen request, draft slot, F).
            # Init to -1 so first-iter compares miss naturally.
            #
            # KNOWN ISSUE: under TRT-LLM's CUDA graph capture, this buffer's
            # MEMORY behaves anomalously — writes via copy_ inside the
            # captured graph don't visibly persist when the same address is
            # read in subsequent replays (reads see 0 regardless of init or
            # writes). The verify cache (`_tssd_last_cands_gpu`, allocated
            # immediately before with the same dtype/device) works fine.
            # Root cause not identified after extensive black-box investigation
            # (see cache_debug runs in build/results/). Treat any draft-cache-
            # based hit-rate or input-match measurement as broken until
            # someone can dig into PyTorch CUDA graph allocator internals.
            # Workaround: read candidate inputs directly from input_ids in
            # the input-match measurement.
            self._tssd_last_draft_cands_gpu = torch.full(
                (max_b, K, self._tssd_hit_F),
                fill_value=-1,
                dtype=torch.int64,
                device="cuda",
            )

            # Target-candidate cache for fused-forward Phase 2.
            # Each group k of the §4.2 schedule contains F_k candidates;
            # candidate at group k predicts sequence position B+k+2 (= one
            # past where d_{k+1} would land). Store top-F_hit per candidate
            # slot, indexed by flat candidate index (0..F_total-1). Lookup
            # uses group→cand_idx map for sequence-position alignment.
            # Init -1 so first-iter compares miss naturally.
            f_tot = max(1, int(self.tssd_F_total))
            self._tssd_last_target_cand_gpu = torch.full(
                (max_b, f_tot, self._tssd_hit_F),
                fill_value=-1,
                dtype=torch.int64,
                device="cuda",
            )
            # group_to_cand[k] = flat index of FIRST candidate in group k,
            # or F_total (sentinel = out-of-range) if group k has F_k=0
            # or k > K. Length K+2 so index K+1 is also a sentinel.
            grp_map = [f_tot] * (K + 2)
            running = 0
            for k_idx, fk in enumerate(self._tssd_F_list):
                if fk > 0 and k_idx <= K:
                    grp_map[k_idx] = running
                running += fk
            self._tssd_group_to_cand_gpu = torch.tensor(grp_map, dtype=torch.int64, device="cuda")

            # Inverse mappings: for each flat candidate index c in [0, F_total),
            # which group does it belong to and what's its rank-within-group?
            cand_to_group = []
            cand_to_rank = []
            for k_idx, fk in enumerate(self._tssd_F_list):
                for r in range(fk):
                    cand_to_group.append(k_idx)
                    cand_to_rank.append(r)
            assert len(cand_to_group) == f_tot, (
                f"cand_to_group length {len(cand_to_group)} != F_total {f_tot}"
            )
            self._tssd_cand_to_group_gpu = torch.tensor(
                cand_to_group, dtype=torch.int64, device="cuda"
            )
            self._tssd_cand_to_rank_gpu = torch.tensor(
                cand_to_rank, dtype=torch.int64, device="cuda"
            )

        # ---- Phase 1 extended-verify infrastructure ----
        # Pre-cache the §4.2 packed mask. Constant for the worker lifetime
        # (K, F_list fixed at config time) so we build it once.
        self._tssd_extended_verify_enabled = (
            torch.cuda.is_available()
            and self.tssd_enabled
            and self.tssd_F_total > 0
            and bool(int(_os.environ.get("TSSD_EXTENDED_VERIFY", "0")))
        )
        if self._tssd_extended_verify_enabled:
            try:
                self._tssd_packed_mask_gpu = build_tssd_packed_mask(
                    K=self.max_draft_len,
                    F=self._tssd_F_list,
                    max_num_requests=max(1, int(self.tssd_max_batch)),
                )
            except Exception as exc:
                logger.warning(
                    f"DFlashTSSDWorker: failed to build packed mask "
                    f"(K={self.max_draft_len}, F={self._tssd_F_list}): {exc}. "
                    f"Disabling extended verify."
                )
                self._tssd_extended_verify_enabled = False
                self._tssd_packed_mask_gpu = None
        else:
            self._tssd_packed_mask_gpu = None

        # Bound to the wrapping ModelForCausalLM via set_target_model() —
        # gives extended verify a way to re-invoke the target forward.
        # Stored as a weakref to avoid creating a cycle in nn.Module's
        # child registry (parent → spec_worker → parent).
        self._tssd_target_model_ref = None

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

    def prepare_target_inputs(
        self,
        input_ids,
        position_ids,
        attn_metadata,
        spec_metadata,
    ) -> None:
        """Pre-target hook (Phase 2.B fused-forward): populate the F_total
        candidate slots of input_ids with previous iter's top-(rank+1)
        draft model predictions per §4.2 group.

        Layout per gen request: [bonus, d_1..d_K, cand_0..cand_{F-1}].
        Candidate at flat index c belongs to group g(c) and has rank r(c)
        within the group (precomputed in __init__). Its input token =
        ``_tssd_last_draft_cands_gpu[gen_b, g(c), r(c)+1]`` — index r+1
        skips top-1 (which is the actual draft token at slot g+1, already
        present in input_ids) so candidates are *alternatives*.

        Capture-safe: uses pre-allocated mapping tensors and scatter ops.
        On the first iter the cache is -1 (sentinel); we clamp to 0 so
        the candidate gets a valid (if noisy) token id. The §4.2 mask
        prevents these candidate K/V from polluting verify regardless.

        No-op when fused-forward is off, when measurement isn't running
        (cache uninitialized), or for ctx-only batches.
        """
        if not (self.tssd_fused_forward and self.tssd_F_total > 0):
            return
        if input_ids is None:
            return
        if not getattr(self, "_tssd_hit_enabled", False):
            return  # Cache not maintained → can't fill candidates safely.

        K = self.max_draft_len
        F_total = self.tssd_F_total
        num_seqs = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = num_seqs - num_contexts
        if num_gens == 0:
            return

        per_gen = K + 1 + F_total
        total_gen_tokens = num_gens * per_gen
        # Gen tokens are contiguous at the end of input_ids[:num_tokens].
        num_tokens = attn_metadata.num_tokens
        gen_start = num_tokens - total_gen_tokens
        if gen_start < 0:
            return  # Sanity: shape mismatch, abort.

        # Gather the candidate token at each flat cand index.
        # Step 1: pick the (group, rank) cell from the cache per cand.
        # Cache shape [max_b, K, F_hit].
        last_draft = self._tssd_last_draft_cands_gpu[:num_gens]  # [num_gens, K, F_hit]
        F_hit = self._tssd_hit_F
        # rank_idx = rank+1, clamped so top-1 is skipped and we don't run
        # past the cached width.
        rank_idx = (self._tssd_cand_to_rank_gpu + 1).clamp(max=F_hit - 1)
        grp_idx = self._tssd_cand_to_group_gpu  # [F_total], in [0, K]

        # gather along dim=1 (group): [num_gens, F_total, F_hit]
        gather_grp = grp_idx.view(1, F_total, 1).expand(num_gens, F_total, F_hit)
        g1 = last_draft.gather(1, gather_grp)
        # gather along dim=2 (rank): [num_gens, F_total, 1]
        gather_rnk = rank_idx.view(1, F_total, 1).expand(num_gens, F_total, 1)
        cand_tokens = g1.gather(2, gather_rnk).squeeze(2)  # [num_gens, F_total] int64
        # First-iter sentinel: -1 → 0 (valid pad-equivalent token).
        cand_tokens = cand_tokens.clamp(min=0)

        # Compute target slots in input_ids. base_g = gen_start + g * per_gen,
        # then K+1 + c for c=0..F_total-1.
        g_idx = torch.arange(num_gens, device="cuda", dtype=torch.int64).unsqueeze(1)
        c_idx = torch.arange(F_total, device="cuda", dtype=torch.int64).unsqueeze(0)
        slot_idx = gen_start + g_idx * per_gen + (K + 1) + c_idx  # [num_gens, F_total]

        # Scatter into input_ids (1D long tensor).
        input_ids.view(-1).scatter_(
            0, slot_idx.flatten(), cand_tokens.flatten().to(input_ids.dtype)
        )

    def set_target_model(self, target_model) -> None:
        """Bind the target model wrapper (the ``*ForCausalLM`` instance the
        worker is attached to). Phase 1 extended verify uses this handle to
        re-invoke ``target_model.model(...)`` with the candidate Q after the
        main verify completes. Called once at construction by
        ``modeling_speculative.py``.

        Stored via a weak reference so PyTorch's nn.Module child-walk
        (``named_children``, recursive setattr) doesn't see the parent and
        loop forever (parent already owns this worker as a child)."""
        import weakref

        self._tssd_target_model_ref = (
            weakref.ref(target_model) if target_model is not None else None
        )

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
            inr = int(self._tssd_inrange_gpu.item()) if hasattr(self, "_tssd_inrange_gpu") else 0
            out["hits_gpu"] = h
            out["total_gpu"] = t
            out["inrange_gpu"] = inr
            out["hit_rate"] = h / t if t > 0 else 0.0
            out["hit_rate_inrange"] = h / inr if inr > 0 else 0.0
            out["inrange_frac"] = inr / t if t > 0 else 0.0
        return out

    def __del__(self):
        # Best-effort: print hit-rate stats at shutdown if collected.
        try:
            if hasattr(self, "_tssd_hits_gpu"):
                h = int(self._tssd_hits_gpu.item())
                t = int(self._tssd_total_gpu.item())
                inr = (
                    int(self._tssd_inrange_gpu.item()) if hasattr(self, "_tssd_inrange_gpu") else 0
                )
                im = (
                    int(self._tssd_input_match_gpu.item())
                    if hasattr(self, "_tssd_input_match_gpu")
                    else 0
                )
                if t > 0:
                    import sys

                    rate = h / t
                    rate_inr = h / inr if inr > 0 else 0.0
                    inr_frac = inr / t
                    im_rate = im / t
                    print(
                        f"[T-SSD final] hits={h} / total={t} hit_rate={rate:.4f} "
                        f"| inrange={inr} ({inr_frac:.3f}) hit_rate_inrange={rate_inr:.4f} "
                        f"| input_match={im} ({im_rate:.4f}) "
                        f"[NOTE: input_match unreliable due to draft cache "
                        f"zeroing under CUDA graph; see code comment]",
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

        # Phase 2 fused-forward: model_engine assembled per-gen Q of size
        # K+1+F_total, target ran with the §4.2 packed mask, gather_ids
        # picked all K+1+F_total rows. Logits / hidden_states arrive with
        # F_total candidate rows appended per gen request — slice them off
        # so the K+1 verify path (sample_and_accept, mamba update, …) sees
        # the original baseline shape. The candidate logits are stashed
        # for hit-rate / commit-on-hit consumption later in this forward.
        K_eff = self.max_draft_len
        cand_logits_for_measurement = None
        if self.tssd_fused_forward and self.tssd_F_total > 0 and num_gens > 0:
            full = logits
            verify_per_gen = K_eff + 1
            cand_per_gen = self.tssd_F_total
            total_per_gen = verify_per_gen + cand_per_gen
            ctx_logits = full[:num_contexts]
            gen_full = full[num_contexts : num_contexts + num_gens * total_per_gen]
            gen_full = gen_full.view(num_gens, total_per_gen, -1)
            verify_block = gen_full[:, :verify_per_gen, :].reshape(num_gens * verify_per_gen, -1)
            cand_logits_for_measurement = gen_full[:, verify_per_gen:, :].contiguous()
            logits = torch.cat([ctx_logits, verify_block], dim=0)

        # Multi-stream split: target verify on default stream, draft prep
        # on aux stream. Phases A/C run on default; phase B (draft) runs on
        # aux. Aux waits on a verify_done event recorded after Phase A so
        # draft cannot overtake target setup. Default stream then waits on
        # an aux_done event before running Phase C (which needs the draft
        # tokens). Capture-safe — events + streams are pre-allocated.
        from ..modules.multi_stream_utils import do_multi_stream

        if (
            num_gens > 0
            and self._tssd_active
            and F_total > 0
            and getattr(self, "_aux_stream", None) is not None
            and do_multi_stream()
        ):
            # Phase A on default stream.
            state = super(DFlashTSSDWorker, self)._forward_target_phase(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
                resource_manager,
            )
            # Mark the end of Phase A on default stream so aux can sync.
            self._event_main.record()

            # Phase B on aux stream.
            with torch.cuda.stream(self._aux_stream):
                self._event_main.wait()
                gen_draft_tokens = super(DFlashTSSDWorker, self)._forward_draft_phase(
                    state, attn_metadata, draft_model
                )
                self._event_aux.record()

            # Default stream waits for draft compute to land.
            self._event_aux.wait()

            # Phase C on default stream.
            out = super(DFlashTSSDWorker, self)._forward_finalize_phase(
                state, gen_draft_tokens, attn_metadata, spec_metadata
            )

            # ---- Capture-safe hit-rate measurement ----
            # Two-cache lookup at sequence-aligned positions:
            #   target verify cache:  K+1 slots predicting positions
            #       A_{t-1}+1..A_{t-1}+K+1
            #     match slot = k_star_{t-1} + k_star_t + 1   (≤ K_eff)
            #   draft model cache:    K slots predicting positions
            #       A_{t-1}+k_star_{t-1}+2..A_{t-1}+k_star_{t-1}+1+K
            #     match slot = k_star_t                       (< K_eff)
            # Combined hit / inrange = (target OR draft). Tensor-only ops +
            # persistent buffers so it replays correctly under CUDA graphs.
            if self._tssd_hit_enabled and num_gens > 0:
                K_eff = self.max_draft_len
                F = self._tssd_hit_F
                accepted = out["new_tokens"]  # [B, K+1] int32
                n_acc = out["new_tokens_lens"]  # [B] int32
                cur_logits = logits  # [num_ctx + num_gens*(K+1), V]

                gen_accept = accepted[
                    num_contexts : num_contexts + num_gens
                ].long()  # [num_gens, K+1]
                gen_n_acc = n_acc[num_contexts : num_contexts + num_gens].long()  # [num_gens]
                k_star = (gen_n_acc - 1).clamp(min=0, max=K_eff)  # [num_gens]
                x_star = gen_accept.gather(1, k_star.unsqueeze(1)).squeeze(1).long()  # [num_gens]

                # ---- target verify cache lookup ----
                last_k_star = self._tssd_last_k_star_gpu[:num_gens]
                target_slot = last_k_star + k_star + 1
                inrange_t = target_slot <= K_eff  # [num_gens] bool
                target_slot_safe = target_slot.clamp(min=0, max=K_eff)
                last_cands = self._tssd_last_cands_gpu[:num_gens]
                idx_t = target_slot_safe.view(num_gens, 1, 1).expand(num_gens, 1, F)
                cands_t = last_cands.gather(1, idx_t).squeeze(1)
                match_t = (cands_t == x_star.unsqueeze(1)).any(dim=1)
                hit_t = match_t & inrange_t

                # ---- draft model cache lookup ----
                # Draft slot j == k_star_t covers iter t's bonus position.
                # Always in [0, K_eff-1] when k_star_t < K_eff.
                inrange_d = k_star < K_eff
                draft_slot_safe = k_star.clamp(min=0, max=K_eff - 1)
                last_draft = self._tssd_last_draft_cands_gpu[:num_gens]  # [num_gens, K, F]
                idx_d = draft_slot_safe.view(num_gens, 1, 1).expand(num_gens, 1, F)
                cands_d = last_draft.gather(1, idx_d).squeeze(1)
                match_d = (cands_d == x_star.unsqueeze(1)).any(dim=1)
                hit_d = match_d & inrange_d

                # ---- target candidate cache lookup (fused-forward only) ----
                # Candidate at group k predicts sequence position B+k+2 of
                # next iter, i.e., bonus position when k = k_star_{t-1}+k_star_t.
                # Group→cand_idx map sentinel = F_total when out-of-range.
                if self.tssd_fused_forward and self.tssd_F_total > 0:
                    f_tot = self.tssd_F_total
                    target_group = (last_k_star + k_star).clamp(min=0, max=K_eff + 1)
                    target_cand_idx = self._tssd_group_to_cand_gpu[target_group]
                    inrange_c = target_cand_idx < f_tot  # [num_gens] bool
                    cand_idx_safe = target_cand_idx.clamp(min=0, max=f_tot - 1)
                    last_cand_t = self._tssd_last_target_cand_gpu[
                        :num_gens
                    ]  # [num_gens, F_total, F]
                    idx_c = cand_idx_safe.view(num_gens, 1, 1).expand(num_gens, 1, F)
                    cands_c = last_cand_t.gather(1, idx_c).squeeze(1)
                    match_c = (cands_c == x_star.unsqueeze(1)).any(dim=1)
                    hit_c = match_c & inrange_c

                    # ---- Phase 3 input-match: candidate INPUT == iter t's bonus ----
                    # Workaround for the draft-cache zeroing pathology:
                    # read the candidate input directly from input_ids (which
                    # is the model's input buffer, definitely persistent
                    # across capture/replay). Candidate at group k_star_t sits
                    # at input_ids slot gen_start + K + 1 + k_star_t per gen.
                    # In-range: k_star_t < K (cand groups span 0..K-1 for
                    # F=[1,...,1,0]; group K has F_K=0 → no candidate).
                    inrange_im = k_star < K_eff  # [num_gens] bool
                    if input_ids is not None and self.tssd_F_total > 0:
                        per_gen = K_eff + 1 + self.tssd_F_total
                        num_tokens = input_ids.shape[0]
                        gen_start_im = num_tokens - num_gens * per_gen
                        if gen_start_im >= 0:
                            # Per-gen slot offset:
                            # gen_start + g*per_gen + (K+1) + k_star_g
                            g_idx_im = torch.arange(num_gens, device="cuda", dtype=torch.int64)
                            slot_idx_im = (
                                gen_start_im
                                + g_idx_im * per_gen
                                + (K_eff + 1)
                                + k_star.clamp(min=0, max=K_eff - 1)
                            )
                            cand_input_tok = input_ids.view(-1)[slot_idx_im].long()
                            match_im = cand_input_tok == x_star
                            hit_im = match_im & inrange_im
                        else:
                            hit_im = torch.zeros_like(hit_c)
                    else:
                        hit_im = torch.zeros_like(hit_c)
                else:
                    hit_c = torch.zeros_like(hit_t)
                    inrange_c = torch.zeros_like(inrange_t)
                    hit_im = torch.zeros_like(hit_t)

                # Combined.
                inrange = inrange_t | inrange_d | inrange_c
                hit_per_req = hit_t | hit_d | hit_c
                self._tssd_hits_gpu.add_(hit_per_req.long().sum())
                self._tssd_inrange_gpu.add_(inrange.long().sum())
                self._tssd_total_gpu.add_(num_gens)
                # Phase 3 commit-on-hit eligibility: candidate INPUT == bonus.
                self._tssd_input_match_gpu.add_(hit_im.long().sum())

                # ---- update target cache (this iter's verify top-F) ----
                gen_logits = cur_logits[num_contexts : num_contexts + num_gens * (K_eff + 1)]
                gen_logits = gen_logits.reshape(num_gens, K_eff + 1, -1)
                _, topF_t = gen_logits.topk(F, dim=-1)  # [num_gens, K+1, F]
                self._tssd_last_cands_gpu[:num_gens].copy_(topF_t.long())

                # ---- update draft cache (this iter's draft top-F) ----
                # NOTE: writes here don't reliably persist for capture-replay
                # reads. See _tssd_last_draft_cands_gpu allocation comment.
                draft_logits = state.get("_draft_logits", None)
                if draft_logits is not None:
                    _, topF_d = draft_logits.topk(F, dim=-1)  # [num_gens, K, F]
                    self._tssd_last_draft_cands_gpu[:num_gens].copy_(topF_d.long())

                # ---- update target candidate cache (fused-forward only) ----
                if (
                    self.tssd_fused_forward
                    and self.tssd_F_total > 0
                    and cand_logits_for_measurement is not None
                ):
                    _, topF_c = cand_logits_for_measurement.topk(F, dim=-1)
                    # topF_c shape [num_gens, F_total, F] int64
                    self._tssd_last_target_cand_gpu[:num_gens].copy_(topF_c.long())

                self._tssd_last_k_star_gpu[:num_gens].copy_(k_star)

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
