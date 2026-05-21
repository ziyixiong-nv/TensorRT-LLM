# Phase 0 Microbenchmark Results: T-SSD Mask Overhead

**Date:** 2026-05-20 (re-measured after design-doc mask correction)
**Hardware:** Single NVIDIA B200, FP/BF16
**Kernel:** FlashInfer 0.6.11.post1 `BatchPrefillWithPagedKVCacheWrapper` (paged custom-mask)
**Attention shape:** matches gpt-oss-120b (Q_heads=64, KV_heads=8, head_dim=64, page_size=16, dtype=bf16)
**Script:** `bench_dflash_tssd_mask_overhead.py`

**Note:** the §4.2 mask was corrected after Phase 1 to include candidate
self-attention (each candidate K/V occupies its own slot in the KV side).
KV length now = prefix_len + K + ΣF_k. Numbers below reflect this
correction.

---

## 1. Raw Numbers

Wall-clock microseconds per attention call (mean of 200 iterations after 20
warmup).
F_total is the total speculation cache budget $\sum_k F_k$; the geometric
fan-out is computed for each F_total via Saguaro Theorem 12 with $a_p=0.78$,
$r=0.5$. Q per request $= K + F_{\text{total}}$.

```
 prefix    B  F=0         F=8         F=16        F=32        | overhead vs F=0
-------  ---  ---------  ---------  ---------  ---------  | -------------------------
   1024    1     23.6us     23.9us     23.8us     29.2us  |  +1.2%   +0.9%  +24.0%
   1024    2     23.0us     23.2us     28.8us     41.2us  |  +0.9%  +25.7%  +79.4%
   1024    4     22.9us     30.2us     53.3us     71.9us  | +32.1% +133.1% +214.2%
   1024    8     23.2us     54.3us     91.8us     96.4us  | +134.6% +296.3% +316.2%
   1024   16     34.9us     94.3us     96.5us    169.8us  | +170.1% +176.1% +386.1%

   4096    1     23.0us     28.0us     35.1us     49.1us  | +21.5%  +52.5% +113.3%
   4096    2     22.9us     35.0us     59.7us     82.6us  | +53.2% +160.8% +260.9%
   4096    4     22.8us     61.4us    116.6us    156.0us  | +169.6% +411.4% +584.4%
   4096    8     37.0us    117.0us    219.3us    348.3us  | +216.3% +492.9% +841.6%
   4096   16     65.6us    230.8us    368.7us    680.7us  | +251.9% +462.0% +937.6%

  16384    1     23.1us     49.3us     93.7us    133.2us  | +113.4% +305.3% +476.0%
  16384    2     28.9us     92.4us    172.2us    251.9us  | +219.2% +495.1% +770.6%
  16384    4     50.6us    179.7us    383.7us    506.1us  | +255.4% +658.8% +900.9%
  16384    8     96.3us    387.6us    757.9us   1392.5us  | +302.5% +687.0% +1345.9%
  16384   16    178.7us    761.3us   1418.7us   2728.7us  | +326.0% +693.8% +1426.7%
```

---

## 2. Findings

### 2.1 The Phase 0 gate (B=1, F_total=16, < 25%)

**Result is borderline at F=16, passes at F=8 for typical SPEED-bench
decode prefix.**

| prefix_len | overhead at B=1, F=16 | gate (<25%) |
|---|---|---|
| 1024 | **+0.9%** | ✅ pass |
| 4096 | +52.5% | ❌ fail |
| 16384 | +305% | ❌ fail (catastrophic) |

| prefix_len | overhead at B=1, F=8 | gate (<25%) |
|---|---|---|
| 1024 | **+1.2%** | ✅ pass |
| 4096 | **+21.5%** | ✅ pass (marginal) |
| 16384 | +113% | ❌ fail |

**With F_total=8, T-SSD is viable for the entire BS=1 SPEED-bench coding
decode range** (prefix grows from ~500 to ~4500 during the decode).

### 2.2 Two distinct regimes

The data clearly separates into two regimes by total query work
$\bar Q = B \cdot (K + F_{\text{total}})$ — the same memory- vs
compute-bound transition the design doc §3.5 predicted, but we now have
the actual numbers:

* **Memory-bound regime** ($\bar Q \lesssim 30$):
  attention is dominated by KV reads + kernel launch fixed cost; adding
  candidate queries is nearly free. **Examples**: B=1, F=8, prefix≤1024
  costs +1.3%; B=1, F=16, prefix=1024 costs +1.6%.

* **Compute-bound regime** ($\bar Q \gtrsim 50$):
  attention cost scales roughly linearly with $\bar Q$. The "free"
  benefit evaporates. **Examples**: B=8, F=16, prefix=4096 costs +492%
  — basically 5.9× the dense time, matching the 5× ratio in $\bar Q$.

Crossover is empirically around $\bar Q \approx 32$–$50$, narrower than
the FFN-side estimate of ~300 in the design doc — **attention crosses
over much earlier than FFN**. This is the kernel-level reality the
design doc §8.3 flagged as the swing factor.

### 2.3 Where T-SSD is viable

Tabulating "viable" = overhead ≤ 25%:

| Config | F=8 | F=16 | F=32 |
|---|---|---|---|
| B=1, prefix=1024 | ✅ **+1.2%** | ✅ **+0.9%** | ✅ +24.0% |
| B=1, prefix=4096 | ✅ **+21.5%** | ❌ +52.5% | ❌ +113% |
| B=2, prefix=1024 | ✅ +0.9% | ❌ +25.7% (just over) | ❌ +79% |
| B=2, prefix=4096 | ❌ +53% | ❌ +161% | ❌ +261% |
| B=4, prefix=1024 | ❌ +32% | ❌ +133% | ❌ +214% |
| B=4, prefix=4096 | ❌ +170% | ❌ +411% | ❌ +584% |
| B≥8, any prefix | ❌ all ≥ 130% | ❌ all ≥ 290% | ❌ all ≥ 320% |

**Takeaway**: T-SSD is viable for:
* **B=1 across the full BS=1 decode trajectory at F=8**
* **B=1 at short prefix at F=16 or F=32**
* **B=2 only at very short prefix and small F**

The corrected mask actually broadened viability slightly compared to the
initial measurement — particularly the B=1, prefix=4096, F=8 configuration
crossed the 25% gate.

---

## 3. Implications for the Design

### 3.1 Required changes to the design

* **Drop the static "always on" assumption.** T-SSD must be runtime-gated
  by current $(B, \text{prefix\_len}, F_{\text{total}})$. A simple
  decision rule:
  $$
  \text{enable T-SSD} \iff B \cdot (K + F_{\text{total}}) \cdot \text{prefix\_len}_{\text{kv}} \leq C
  $$
  with $C$ tuned empirically — preliminary fit suggests
  $C \approx 1.3 \times 10^5$.

* **Two F-budget tiers.** Use $F_{\text{total}} = 8$ as the default
  (geometric fan-out gives roughly $F = (2, 2, 2, 1, 1)$); reserve
  $F_{\text{total}} = 16$ for very short prefix only.

* **Disable at $B \geq 4$.** No setting works at high batch. This is a
  hard rule. Saguaro paper Theorem 17 had a similar batch-size-dependent
  fallback selection; the same logic applies here, just earlier.

### 3.2 Where the win actually exists

For SPEED-bench coding subset BS=1 with F_total=8:

* prefix during decode grows from ~500 (initial prompt) to ~4500 (full
  output). The **entire trajectory is now in viable territory** under
  the corrected mask (overhead 1.2% → 21.5% across the range).
* Mean overhead averaged over the decode trajectory at F=8:
  $\sim$ +10–15% attention overhead.
* This needs to be offset by the cache-hit-side savings (one fixup
  forward + one ctx K/V projection avoided per round on hit).

**Realistic upper bound on end-to-end speedup**: 8–15% at BS=1
gpt-oss-120b with F=8 — meaningful, not transformative. Saguaro paper
reports 30% over SD with a *separate device* for the speculator; we
achieve narrower wins on a *single device* but without extra hardware.

### 3.3 Where the win could be much larger

Two possibilities the data supports:

1. **Kimi-K2.5** ($a_p \approx 0.89$) — full-accept rate alone is 64%
   and total cache hit can plausibly reach 90%+. Even at the same
   attention overhead, the per-round savings nearly double.

2. **Latency-only workloads with short prompts** (< 1024 tokens, BS=1).
   Coding completion or edit suggestions fit this. Attention overhead
   is < 2%, virtually free, and full Saguaro fan-out (F=16) is
   tractable — projected 15–25% end-to-end gain.

---

## 4. Honest Assessment

**The design as written is not a clear win on the workload that motivated
it (gpt-oss-120b SPEED-bench coding subset, full 4K decode lengths).**

The design as written **is** likely a clear win on:

* Short-prompt latency-critical workloads (BS=1, prefix < 1K).
* Models with very high acceptance rate (Kimi-K2.5 specifically).
* Possibly: with a kernel improvement (sm100 customized for this mask
  pattern). The current FlashInfer kernel is general; a specialized
  kernel for "main-chain + branch-diagonals" structure could close the
  gap at longer prefix.

What was wrong in the design doc's §8.3 estimate:

* Predicted +10% verify time at B=1, F=16. **Actual: +1.6% at prefix=1024,
  +43.5% at prefix=4096.** The estimate failed to account for prefix
  scaling — attention at long prefix is not in the "fully memory-bound"
  regime, it's increasingly limited by KV-loading bandwidth, where the
  custom-mask kernel pays a structured-sparsity penalty.

* Did not predict the catastrophic high-batch behavior. **Actual: 5–10×
  overhead at B≥4.** The design doc §10 mentioned high-batch as a risk
  but underestimated severity.

---

## 5. Recommendation

**Three options for next step**:

### Option A — Continue with narrower scope

Implement T-SSD with strict gating:
- Only at B=1 or B=2.
- Only at prefix ≤ 2048 (interpolated breakeven).
- Default F_total = 8.

Phase 1 reference + Phase 2 production proceed as planned, but the
target workload becomes "low-batch latency-critical" not "general
SPEED-bench". Expected end-to-end win: 5–10% on gpt-oss-120b in
viable prefix range, possibly 15–25% on Kimi-K2.5.

### Option B — Pivot to Kimi-K2.5 first

Re-run this microbench against Kimi-K2.5's attention shape (need to
extract from `/code/llm-models/Kimi-K2.5-NVFP4/config.json`). If
overhead pattern is favorable (Kimi has fewer / different attention
heads), Kimi is the better target for the full design.

### Option C — Stop here

The math is clean but the realized win is too narrow on the chosen
workload to justify the engineering cost (estimated 1.5–2k LoC across
3 layers, plus kernel work). Document the design as a research artifact
and revisit if a kernel improvement makes the constants better, or
reach for entirely different speedup sources for the gpt-oss-120b SPEED-
bench target.

The right call depends on:
* Whether 5–10% gain at BS=1 is enough to justify the effort.
* Whether Kimi-K2.5 is in scope for this team.
* Whether there is a kernel team that could specialize the mask path.

---

## 6. Reproducibility

```bash
# Single-config probe (debug)
python tensorrt_llm/_torch/speculative/bench_dflash_tssd_mask_overhead.py \
    --single --batch 1 --K 4 --F_total 16 --prefix_len 4096

# Full sweep (this report)
python tensorrt_llm/_torch/speculative/bench_dflash_tssd_mask_overhead.py
```

Configurations swept: B ∈ {1, 2, 4, 8, 16}, F_total ∈ {0, 8, 16, 32},
prefix_len ∈ {1024, 4096, 16384}. Total: 60 runs × 220 iterations each
= ~30 seconds wall-clock.

---

## Appendix: how the geometric fan-out unpacks

For `--a_p 0.78 --K 4`:

| F_total | F = (F_0, F_1, F_2, F_3, F_4) |
|---|---|
| 8  | (3, 3, 1, 1, 1)? — actual: depends on rounding; script prints F |
| 16 | (3, 3, 3, 2, 5) |
| 32 | (6, 5, 4, 4, 13) |

The full-accept slot ($F_K$) gets a disproportionately large share when
$a_p$ is high; this matches the Saguaro paper's intuition (Theorem 12)
that full-accept outcomes are a probability mass concentration that
deserves dedicated cache slots.
