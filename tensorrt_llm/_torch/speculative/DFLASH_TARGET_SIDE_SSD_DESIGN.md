# DFlash + Target-Side Speculative Speculative Decoding (T-SSD): Design Document

**Status:** Draft v2 — design + Phase 0 (microbench) + Phase 1 (reference impl) complete
**Audience:** Engineers implementing the kernel + scheduler changes, and reviewers
of the math.
**Scope:** Lossless multi-outcome pre-speculation for DFlash, where the
multi-outcome work is shifted from the draft model (as in
[Saguaro / SSD, arXiv:2603.03251](https://arxiv.org/abs/2603.03251)) onto the
**target** model's verify pass. The result is a same-GPU, no-extra-device
implementation that is mathematically lossless and exploits the
memory-bound regime of large MoE targets at small batch.

**Companion artifacts (read alongside):**
* `bench_dflash_tssd_mask_overhead.py` — Phase 0 microbench (FlashInfer)
* `PHASE0_RESULTS.md` — measured attention overhead and viability envelope
* `phase1_reference.py` — pure PyTorch reference; runs 6 self-validating tests

**Revisions:**
* **v2 (post Phase 1):** Mask spec corrected — candidates now have their own
  K/V slots in the KV side and a self-attention diagonal entry in the mask.
  Without this, the candidate hidden state was *not* equivalent to a fixup
  forward (Test 3 in `phase1_reference.py` failed at v1 → fixed in v2).
  Position arithmetic standardized to 0-indexed: corrected/bonus token at
  position $|P|+k^*$, not $|P|+k^*+1$. KV scratch sizing formulas now
  include the factor 2 for K **and** V. gpt-oss-120b actual shape used for
  sanity checks (head_dim=64).
* v1: initial design.

---

## 1. Executive Summary

Standard SSD (Saguaro) parallelizes drafting and verification by having the
draft model pre-speculate for many possible verification outcomes
$(k, x^*)$ on a *separate device*, then looking up the actual outcome after
verify completes. This works cleanly for token-input drafts (vanilla SD,
EAGLE-3 with self-conditioning fallback) but does **not** translate directly
to DFlash:

* **DFlash drafts on top of target hidden states** captured during the
  previous round's target verify, fed in as cross-attention K/V
  (`ctx_k_cache` / `ctx_v_cache` in `dflash.py`).
* For any verification outcome $(k, x^*)$ with $k < K$, the corrected bonus
  token $x^*$ does **not** have a target hidden state computed for it — the
  hidden state target produced at that position was conditioned on the
  *rejected* draft token $d_{k+1}$, not on $x^*$.
* Without that hidden state, the round-$T{+}1$ DFlash draft cannot start.

The Saguaro paper acknowledges a related issue for EAGLE-3 (Appendix E,
Figure 9) and proposes substituting draft self-activations as a surrogate.
This is **lossy** and explicitly recommends retraining the draft.
DFlash is even harder — it has no autoregressive
self-conditioning chain to substitute into, and we do not want to retrain.

**This design proposes a different solution.** We exploit a structural fact
of causal transformers: replacing one input token at the last position of a
prefix changes hidden states only at that position. So the missing hidden
state for $(k, x^*)$ is exactly **one** position of target forward —
specifically, position $|P|+k$ with input $x^*$ (0-indexed: prefix occupies
$0..|P|{-}1$, accepted draft tokens $|P|..|P|{+}k{-}1$, corrected/bonus at
$|P|{+}k$).

We extend the target verify pass with $\sum_k F_k$ extra **candidate
positions** that pre-compute hidden states for each $(k, \text{candidate})$
combination. Because target on a 70B+ model at small batch is heavily
memory-bound, these extra positions are nearly free (estimated < 5%
verify-time overhead at $B{=}1$, $K{=}4$, $\sum F_k{=}20$ on B200). After
verify decides $(k, x^*)$, we look up the matching candidate's hidden state
and feed it into round-$T{+}1$'s DFlash draft.

The scheme is:

* **Lossless** — every candidate hidden state is a real target forward
  output, no approximation;
* **Compatible with frozen draft** — DFlash worker / model unchanged below
  the cross-attention K/V interface;
* **Same-GPU** — no extra device required;
* **Power-law cache hit rate** preserved from Saguaro (Theorem 12),
  enabling 80%+ hit rate at modest budgets.

The cost is real but localized: a custom attention mask for the target's
verify kernel, a scratch KV / hidden-state buffer with commit-or-discard
semantics, and a speculation cache on the DFlash worker side. This document
specifies all three precisely.

---

## 2. Background

### 2.1 DFlash, in one paragraph

DFlash is a one-shot block-diffusion-style draft: given prefix + accepted
tokens, the draft model runs a **single forward pass** with $K$ mask tokens
appended; the cross-attention layers attend to a per-request,
per-layer projected K/V cache built from target's captured hidden states
(`fc + hidden_norm` projection of `hidden_states` at `layers_to_capture`,
stored in `_ctx_k_buf` / `_ctx_v_buf` in
`tensorrt_llm/_torch/speculative/dflash.py`). The draft outputs $K$ logits
in parallel — one per mask position — and these are the round's draft
tokens. There is no autoregressive self-conditioning inside the draft.

Anchors in code:

* `DFlashWorker.forward` — `tensorrt_llm/_torch/speculative/dflash.py:370`
* `_store_prefill_context` (cross-attention K/V projection) —
  `tensorrt_llm/_torch/speculative/dflash.py:292`
* `prepare_1st_drafter_inputs` (assembles draft input from accepted tokens
  + mask tokens + ctx K/V slots) —
  `tensorrt_llm/_torch/speculative/dflash.py:532`

### 2.2 SSD / Saguaro, in one paragraph

Saguaro (arXiv:2603.03251) parallelizes drafting and verification by
launching round-$T{+}1$ draft pre-speculation **during** round-$T$
verification. Without knowing the verify outcome
$v^T = (k, x^*) \in \mathcal{V}^T$, the speculator pre-computes draft
sequences for the $B$ most likely outcomes (the *speculation cache*
$\mathcal{S}^{T+1}$). After verify completes, $\mathcal{S}^{T+1}[v^T]$ is
returned immediately if present (a *cache hit*); otherwise a fallback
strategy runs. The geometric fan-out theorem (Saguaro Thm 12) gives the
optimal allocation of cache entries across accept lengths $k$ for a given
budget $B = \sum_k F_k$. Saguaro runs the speculator on a **separate GPU**.

### 2.3 Why naive SSD doesn't apply to DFlash

Naive translation: launch DFlash for each candidate $(k, x^*)$ during
target verify, on the same or separate device. **The blocker is not
compute, it is the cross-attention input.** Each candidate needs:

$$
\text{KV}_{\text{xattn}}^{(k, x^*)} = \text{Project}\bigl(h^{\ell}_{\text{target}}(P \oplus d_1{:}d_k \oplus x^*)\bigr)_{\ell \in \mathcal{L}}
$$

where $\mathcal{L}$ is the captured-layer set and $P$ is the prefix.
Target's main verify pass produces $h^\ell$ at positions
$|P|, |P|{+}1, \ldots, |P|{+}K{-}1$ conditioned on inputs $d_1, \ldots, d_K$
(the drafted tokens). For candidate $(k, x^*)$ with $k < K$, we need
$h^\ell$ at position $|P|{+}k$ with input $x^*$, **not** $d_{k+1}$.
Target has not computed this. We cannot derive it from anything target *did*
compute without either (a) re-running a target forward (defeats SSD's
purpose) or (b) approximating it (lossy).

The Saguaro paper hits this same wall for EAGLE-3 and accepts the lossy
path. This document proposes a lossless path specific to DFlash.

---

## 3. Mathematical Foundation

### 3.1 Notation

| Symbol | Meaning |
|---|---|
| $L$ | Number of target layers |
| $\mathcal{L} \subseteq [L]$ | Captured layer indices (DFlash `layers_to_capture`) |
| $H$ | Hidden size |
| $V$ | Vocabulary size |
| $K$ | DFlash speculative lookahead (block size) |
| $B$ | Batch size |
| $P = (p_1, \ldots, p_{|P|})$ | Accepted prefix at start of round $T$ |
| $d_1, \ldots, d_K$ | Drafted tokens proposed for round $T$ |
| $h^\ell_t(\sigma)$ | Target hidden state at layer $\ell$, position $t$, given input sequence $\sigma$ |
| $\pi_{\text{tgt}}^{(t)}(\cdot \mid \sigma)$ | Target distribution over next token at position $t$ given $\sigma$ |
| $k \in \{0, 1, \ldots, K\}$ | Verify accept length (number of $d_i$'s accepted) |
| $x^* \in V$ | Verify bonus / corrected token |
| $v^T = (k, x^*)$ | Verify outcome |
| $F_k$ | Number of candidate $x^*$'s pre-computed for accept length $k$ |
| $\text{Cand}_k \subset V$, $\lvert \text{Cand}_k \rvert = F_k$ | Candidate set at $k$ |
| $\mathcal{S}^{T+1}$ | Speculation cache: $(k, c) \mapsto$ draft sequence + ctx K/V for round $T{+}1$ |
| $a_p \in (0, 1)$ | Per-token draft acceptance rate (DFlash data: $\approx 0.78$ on gpt-oss-120b, $\approx 0.89$ on Kimi-K2.5) |
| $r > 0$ | Power-law exponent for cache miss rate (Saguaro Defn 11) |

### 3.2 The Causal One-Token Swap Lemma

The core mathematical observation:

**Lemma 1 (Causal one-token swap).** Let $\sigma = (\sigma_1, \ldots, \sigma_n)$
and $\sigma' = (\sigma_1, \ldots, \sigma_{n-1}, \sigma'_n)$ be two input
sequences differing only at the final position. For any layer $\ell$ and
position $t$ in a strictly causal transformer:

$$
h^\ell_t(\sigma') = h^\ell_t(\sigma) \quad \forall t < n,
$$

and $h^\ell_n(\sigma')$ is in general different from $h^\ell_n(\sigma)$.

*Proof sketch.* Strict causality means $h^\ell_t$ depends only on inputs at
positions $\leq t$. Inputs $\sigma_{1:n-1} = \sigma'_{1:n-1}$, so
$h^\ell_t$ for $t < n$ is invariant under the swap. ∎

**Corollary 1.** For any DFlash verify outcome $(k, x^*)$, the cross-attention
context required for round $T{+}1$ differs from what target's main verify
already produced **only at position $|P|{+}k{+}1$**. All other positions
$0, \ldots, |P|{+}k$ are already valid in target's main verify outputs.

This is the entire mathematical leverage. We need to fill in **one position
per candidate**, not redo a forward pass per candidate.

### 3.3 Target-side multi-outcome formulation

Define an **extended verify pass** as the concatenation of:

* A *main* segment of $K$ query positions, with inputs $(d_1, \ldots, d_K)$;
  position $i$ attends to $P \oplus d_1{:}d_{i-1}$ (causal).
* A *candidate* segment of $\sum_{k=0}^{K} F_k$ query positions, partitioned
  into $K{+}1$ groups indexed by $k \in \{0, 1, \ldots, K\}$. Group $k$ has
  $F_k$ positions with inputs drawn from $\text{Cand}_k \subset V$
  (top-$F_k$ candidates for the bonus / corrected token at accept length
  $k$). Each candidate position in group $k$ attends to
  $P \oplus d_1{:}d_k$ — that is, **only the prefix and the first $k$
  draft tokens**, never to $d_{k+1}{:}d_K$ and never to other candidates.

Each candidate position $(k, c)$ outputs $h^\ell_{|P|+k}(P \oplus d_{1:k} \oplus c)$
at every captured layer $\ell$. By Corollary 1, when verify decides
$(k^{\!*}, x^{\!*})$ with $x^{\!*} \in \text{Cand}_{k^{\!*}}$, the
corresponding candidate's hidden state is *exactly* what round $T{+}1$'s
DFlash cross-attention K/V needs at position $|P|{+}k^*$.

The cross-attention K/V for round $T{+}1$ is then assembled as
(using 0-indexed positions, so the accepted sequence
$P \oplus d_{1:k^*} \oplus x^*$ has length $|P|+k^*+1$ and indices
$0, \ldots, |P|+k^*$):

$$
\text{KV}^{(k^*, x^*)}_{\text{xattn}, t} = \begin{cases}
\text{Project}\bigl(h^\ell_t \text{ from prefix store}\bigr) & 0 \leq t \leq |P|-1 \\
\text{Project}\bigl(h^\ell_t \text{ from main verify}\bigr) & |P| \leq t \leq |P|+k^*-1 \\
\text{Project}\bigl(h^\ell_t \text{ from candidate } (k^*, x^*)\bigr) & t = |P|+k^*
\end{cases}
$$

All three cases are exact — no approximation anywhere.

### 3.4 Verify outcome probability and geometric fan-out

The probability of outcome $(k, c)$ under draft acceptance rate $a_p$ is

$$
\Pr[(k, c)] = \begin{cases}
a_p^k (1 - a_p) \cdot r_k(c) & k < K \\
a_p^K \cdot \pi^{(K)}_{\text{tgt}}(c \mid P \oplus d_{1:K}) & k = K
\end{cases}
$$

where $r_k(\cdot) \propto \max(\pi^{(k)}_{\text{tgt}} - \pi^{(k)}_{\text{drf}}, 0)$
is the residual distribution at position $k$. Cache hit at fan-out $F_k$
selects the top-$F_k$ entries of $r_k$ (or $\pi^{(K)}_{\text{tgt}}$ for $k=K$);
miss probability scales (empirically, per Saguaro Fig 3) as a power-law
$1 - p_{\text{hit},k}(F_k) \approx F_k^{-r}$.

Subject to budget $\sum_k F_k = B$, Saguaro Theorem 12 gives the optimal
allocation:

$$
F_k = F_0 \cdot a_p^{k/(1+r)} \quad (k < K), \qquad
F_K = F_0 \cdot a_p^{K/(1+r)} \cdot (1 - a_p)^{-1/(1+r)},
$$

with $F_0$ solved from the budget. **This theorem transfers verbatim** —
its derivation is independent of where the multi-outcome speculation
runs (draft side or target side). Only the *cost* of growing $B$ changes.

For DFlash on gpt-oss-120b ($a_p \approx 0.78$, $r \approx 0.5$ from
Saguaro Fig 3), $K{=}4$, $B{=}20$:

| $k$ | $F_k$ |
|---|---|
| 0 | 6.0 |
| 1 | 5.2 |
| 2 | 4.6 |
| 3 | 4.0 |
| 4 (full accept) | 0.2 → rounded to 1 |

(Approximate, intended for sizing only — actual implementation should
solve the $F_0$ closed form from Saguaro Appendix A.2.)

### 3.5 Cost analysis: memory-bound headroom

Claim: for a 70B+ MoE target at $B \cdot K \ll N^*$, where $N^*$ is the
memory-to-compute crossover position count, adding $N$ candidate positions
to the verify pass increases verify wall-clock by approximately
$N / N^*$ (i.e., approaches zero as $N \ll N^*$).

**Estimate for gpt-oss-120b on B200:**

| Quantity | Approx. value |
|---|---|
| Active params (FP8/MXFP4) | $\sim 60$ GB |
| HBM bandwidth | $\sim 8$ TB/s |
| FFN dense FLOPs (FP8) | $\sim 5$ PFLOPs |
| Per-position FFN compute | $2 \cdot \text{params}_{\text{active}}$ |
| Memory-bound layer time | $\text{params}_{\text{active}} / \text{BW}$ |
| Compute-bound crossover $N^*$ | $\frac{\text{params/BW}}{2 \cdot \text{params/FLOPs}} = \frac{\text{FLOPs}}{2 \cdot \text{BW}}$ |
| | $\approx 5\text{e15} / (2 \cdot 8\text{e12}) \approx 300$ |

So at $B{=}1$, adding $\sum F_k = 20$ candidate positions to a $K{=}4$
verify keeps total positions = 24 ≪ $N^* \approx 300$ for FFN. **Verify
time is dominated by parameter loading, not by per-position compute.**
Marginal cost is $\sim 24/300 \approx 8\%$, almost all in attention
(custom mask sparsity hurts more than dense attention adding a few
positions).

For attention, the cost is more nuanced because of paged KV cache loading
patterns and custom mask sparsity. See §8.3 for kernel-level discussion;
empirically Saguaro reports custom-mask attention as the actual critical
path, which mirrors what we expect here.

At higher batch ($B \cdot K \gtrsim 50$) the regime changes — see §10
for the high-batch fallback design.

---

## 4. Attention Mask Specification

This section is the **kernel contract**. An implementer reading only §1,
§3, and §4 should be able to write the kernel.

### 4.1 Layout

For one request in the batch, the extended-verify query sequence is laid
out in the following order:

```
              Main verify           Candidate segments (one block per k)
              ┌───────────┐    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌─────┐
queries  =    │ q_1 … q_K │ ⊕  │ k=0   │ │ k=1   │ │ k=2   │ │k=K-1  │ │k=K  │
              └───────────┘    │ F_0   │ │ F_1   │ │ F_2   │ │F_{K-1}│ │ F_K │
                               └───────┘ └───────┘ └───────┘ └───────┘ └─────┘
input tokens: d_1 … d_K        c_{0,1} … c_{0,F_0}   …                  bonus
                                                                         cands
```

Total query positions per request: $Q = K + \sum_{k=0}^{K} F_k$.

The KV (keys/values) sequence **grows with the queries** — every attention
position contributes its own K and V, including candidates:

```
              Prefix       Main verify   Candidate K/V (same order as queries)
              ┌──────┐    ┌───────────┐ ┌────────────────────────────────────┐
KV     =      │ P    │ ⊕  │ d_1 … d_K │⊕│ c_{0,1} … c_{0,F_0} c_{1,1} …      │
              └──────┘    └───────────┘ └────────────────────────────────────┘
positions:    0 .. |P|-1  |P| .. |P|+K-1   |P|+K .. |P|+K+ΣF_k-1
```

Total KV positions: $|P| + K + \sum_k F_k = |P| + Q$.

**Why candidate K/V must be in the KV side.** A causal transformer at
position $p$ with input token $t$ outputs a hidden state where the
attention layer queries against positions $0, 1, \ldots, p$ — *including
position $p$ itself*. For the candidate's hidden state to be bit-identical
to a real fixup forward (Lemma 1), the candidate query must attend to its
own K and V. The K/V of *other* candidates is masked out (each candidate
is hypothetical, mutually exclusive); only its **own** K/V is visible.

**Persistence.** Main-verify K/V positions ($|P| \ldots |P|+K-1$) are
written to target's main paged KV cache normally. Candidate K/V positions
($|P|+K \ldots$) are written to a per-iteration scratch buffer; on cache
hit, the matching candidate's K/V is copied into the main cache (see §5).

### 4.2 Mask matrix definition

Let $M \in \{0, 1\}^{Q \times (|P| + Q)}$ be the attention mask, with
$M_{q, t} = 1$ iff query $q$ may attend to KV position $t$.

For convenience define $\text{cand}\_kv(q) = |P| + K + (q - K)$, the KV
position assigned to candidate query $q$ (well-defined for $q \geq K$).

**Main verify rows** ($q = i \in [0, K)$, input $d_{i+1}$):
$$M_{i, t} = \mathbb{1}[0 \leq t \leq |P| - 1] \;\vee\; \mathbb{1}[|P| \leq t \leq |P| + i]$$

i.e. attend to all of prefix, to the first $i$ draft tokens, and to *self*
($t = |P| + i$). Standard spec-dec causal mask.

**Candidate group $k$ rows** ($q = K + \sum_{k'<k} F_{k'} + j$,
$j \in [0, F_k)$, input $c_{k,j} \in \text{Cand}_k$):
$$M_{q, t} = \mathbb{1}[0 \leq t \leq |P| - 1] \;\vee\; \mathbb{1}[|P| \leq t \leq |P| + k - 1] \;\vee\; \mathbb{1}[t = \text{cand\_kv}(q)]$$

i.e. attend to prefix, to the first $k$ draft tokens, and to *its own*
candidate KV slot. **Not** to $d_{k+1}{:}d_K$, **not** to other
candidates.

### 4.3 Worked example

$K = 4$, $F = (F_0, F_1, F_2, F_3, F_4) = (1, 2, 3, 5, 1)$, $|P| = 10$.
Total $Q = 4 + 12 = 16$, $|P|+Q = 26$.

The mask `M[16, 26]`. KV columns $0\ldots 9$ are prefix, $10\ldots 13$ are
main verify ($d_1\ldots d_4$), $14\ldots 25$ are candidate K/V positions
in the same order as the candidate queries.

```
          KV positions →     0 1 2 3 4 5 6 7 8 9 | 10 11 12 13 | 14 15 16 17 18 19 20 21 22 23 24 25
                             ─ prefix ─────────  | d1 d2 d3 d4 | candidate K/V (one per cand query)
                                                 |             |  c01 c11 c12 c21 c22 c23 c31..c35 c41
query rows ↓
─ main verify ────────────────────────────────────────────────────────────────────────────────
q_0  (input d_1, self KV=10) 1 1 1 1 1 1 1 1 1 1 |  1  0  0  0 |  0  0  0  0  0  0  0  0  0  0  0  0
q_1  (input d_2, self KV=11) 1 1 1 1 1 1 1 1 1 1 |  1  1  0  0 |  0  0  0  0  0  0  0  0  0  0  0  0
q_2  (input d_3, self KV=12) 1 1 1 1 1 1 1 1 1 1 |  1  1  1  0 |  0  0  0  0  0  0  0  0  0  0  0  0
q_3  (input d_4, self KV=13) 1 1 1 1 1 1 1 1 1 1 |  1  1  1  1 |  0  0  0  0  0  0  0  0  0  0  0  0
─ candidate group k=0 (F_0=1) — attend prefix + self
q_4  (c_{0,1}, self KV=14)   1 1 1 1 1 1 1 1 1 1 |  0  0  0  0 |  1  0  0  0  0  0  0  0  0  0  0  0
─ candidate group k=1 (F_1=2) — attend prefix + d_1 + self
q_5  (c_{1,1}, self KV=15)   1 1 1 1 1 1 1 1 1 1 |  1  0  0  0 |  0  1  0  0  0  0  0  0  0  0  0  0
q_6  (c_{1,2}, self KV=16)   1 1 1 1 1 1 1 1 1 1 |  1  0  0  0 |  0  0  1  0  0  0  0  0  0  0  0  0
─ candidate group k=2 (F_2=3) — attend prefix + d_1..d_2 + self
q_7  (c_{2,1}, self KV=17)   1 1 1 1 1 1 1 1 1 1 |  1  1  0  0 |  0  0  0  1  0  0  0  0  0  0  0  0
q_8  (c_{2,2}, self KV=18)   1 1 1 1 1 1 1 1 1 1 |  1  1  0  0 |  0  0  0  0  1  0  0  0  0  0  0  0
q_9  (c_{2,3}, self KV=19)   1 1 1 1 1 1 1 1 1 1 |  1  1  0  0 |  0  0  0  0  0  1  0  0  0  0  0  0
─ candidate group k=3 (F_3=5) — attend prefix + d_1..d_3 + self
q_10 (c_{3,1}, self KV=20)   1 1 1 1 1 1 1 1 1 1 |  1  1  1  0 |  0  0  0  0  0  0  1  0  0  0  0  0
q_11 (c_{3,2}, self KV=21)   1 1 1 1 1 1 1 1 1 1 |  1  1  1  0 |  0  0  0  0  0  0  0  1  0  0  0  0
q_12 (c_{3,3}, self KV=22)   1 1 1 1 1 1 1 1 1 1 |  1  1  1  0 |  0  0  0  0  0  0  0  0  1  0  0  0
q_13 (c_{3,4}, self KV=23)   1 1 1 1 1 1 1 1 1 1 |  1  1  1  0 |  0  0  0  0  0  0  0  0  0  1  0  0
q_14 (c_{3,5}, self KV=24)   1 1 1 1 1 1 1 1 1 1 |  1  1  1  0 |  0  0  0  0  0  0  0  0  0  0  1  0
─ candidate group k=4 (F_4=1) — full-accept bonus candidate
q_15 (c_{4,1}, self KV=25)   1 1 1 1 1 1 1 1 1 1 |  1  1  1  1 |  0  0  0  0  0  0  0  0  0  0  0  1
```

The candidate-K/V block is **diagonal** — each candidate sees only its
own K/V slot, not other candidates'. Main verify is unchanged from a
standard spec-dec verify (causal among the K positions).

### 4.4 Variable batch and paged KV

For batch $B > 1$, each request has its own $|P|^{(b)}$, its own
$d_1^{(b)} {:} d_K^{(b)}$, and its own $\text{Cand}_k^{(b)}$. The
extended verify becomes a multi-query / variable-length attention, which
is the regime FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` already
supports if we feed it:

* `qo_indptr[B+1]` with $\text{qo\_indptr}[b+1] - \text{qo\_indptr}[b] = Q^{(b)} = K + \sum_k F_k^{(b)}$
* `paged_kv_indptr / paged_kv_indices / paged_kv_last_page_len` for the
  KV side (prefix + main verify d's, paged from main KV cache)
* A custom causal mask (FlashInfer supports `custom_mask` in
  `plan` / `run`).

For FlashAttention-3 the equivalent path is `flash_attn_varlen_func` with
a custom `mask_mod` callable (FA3 introduced this in 2024) — feasible but
mask sparsity hurts perf more than FlashInfer's paged path when the mask
is structured-sparse like ours.

**Recommendation:** start with FlashInfer paged custom-mask. The mask is
quite structured (block-diagonal in the candidate region) and sparsity
patterns are deterministic, so we should be able to match or improve over
Saguaro's reported attention overhead.

---

## 5. KV Cache State Machine

### 5.1 Buffers and lifetimes

We introduce three new buffer regions on top of DFlash's existing state:

| Buffer | Purpose | Lifetime | Size |
|---|---|---|---|
| `target_main_kv` | Target's primary paged KV (existing) | Persistent | unchanged |
| `target_cand_kv_scratch` | K AND V for candidate query positions, per layer | Per-iteration; written during extended verify, read at commit time only | $2 \cdot L \cdot \sum_k F_k \cdot \text{nkv} \cdot \text{hd}$ per request (factor 2 for K and V) |
| `target_cand_hidden` | Captured hidden states $h^\ell$ for candidate positions, captured layers only | Per-iteration; used to build round-$T{+}1$ ctx K/V on commit | $\lvert\mathcal{L}\rvert \cdot \sum_k F_k \cdot H$ per request |
| `dflash_ctx_scratch_kv` | Round-$T{+}1$ ctx K/V projected from `target_cand_hidden`, per candidate | Per-iteration; one of these is committed into `_ctx_k_buf` / `_ctx_v_buf` on hit | $2 \cdot L_{\text{draft}} \cdot \sum_k F_k \cdot \text{nkv}_{\text{draft}} \cdot \text{hd}_{\text{draft}}$ per request |

**Sizing sanity check** for $B{=}16$, $\sum F_k = 20$, gpt-oss-120b actual
shape from `config.json`: $L = 36$, $\text{nkv} = 8$, $\text{hd} = 64$,
$H = 2880$, captured-layer count $\lvert\mathcal{L}\rvert = 3$:

* `target_cand_hidden`: $16 \cdot 20 \cdot 3 \cdot 2880 \cdot 2\,\text{B}$
  (bf16) $\approx 5.6$ MiB. Negligible.
* `target_cand_kv_scratch`: $16 \cdot 2 \cdot 36 \cdot 20 \cdot 8 \cdot 64 \cdot 1\,\text{B}$
  (FP8 KV) $\approx 11.8$ MiB. Negligible.
* `dflash_ctx_scratch_kv`: smaller. Negligible.

Memory cost is not the binding constraint. Bookkeeping (paged page-table
entries, slot indirection) is the work.

**Note on K/V management.** Inside the extended-verify attention call, K
and V for candidate query positions are computed (every transformer layer
projects Q, K, V from input embeddings). FlashInfer's paged wrappers
typically write all Q positions' K/V into the paged KV cache via a
companion `append_paged_kv_cache` call — which we **do not want for
candidates**. Implementation must:
1. Compute Q/K/V projections for all $K + \sum_k F_k$ positions.
2. Run attention with the §4.2 mask using **paged main KV plus the new
   K/V positions in scratch** (one possible approach: make the candidate
   K/V live in pages that get logically discarded, or use the
   ragged / `BatchPrefillWithRaggedKVCache` variant where candidate K/V
   stays out of the paged cache).
3. After attention, call `append_paged_kv_cache` only for the $K$ main
   verify positions; candidate K/V stays in scratch.

Decision on which FlashInfer variant to use is part of Phase 2 kernel
work; both options are feasible.

### 5.2 Commit / rollback semantics

After verify completes and produces $(k^*, x^*)$:

1. **Cache lookup**: search $\text{Cand}_{k^*}$ for $x^*$.
2. **Hit case** ($x^* \in \text{Cand}_{k^*}$, with index $j$):
   * Copy `target_cand_kv_scratch[k^*, j]` (one position's K/V at every
     layer) into `target_main_kv` at logical position $|P| + k^*$ — i.e.
     immediately after the $k^*$ accepted draft tokens. (The accepted
     sequence has length $|P| + k^* + 1$, with $x^*$ at index $|P|+k^*$.)
   * Copy `dflash_ctx_scratch_kv[k^*, j]` into `_ctx_k_buf`/`_ctx_v_buf`
     at the request's slot, position $|P| + k^*$.
   * Update `_ctx_len[slot]` and DFlash's per-request bookkeeping.
   * Drop all other candidates' scratch.
3. **Miss case** ($x^* \notin \text{Cand}_{k^*}$):
   * Drop **all** candidate scratch.
   * Run a single-position target forward at logical position $|P| + k^*$
     with input $x^*$. (This is the "fixup forward" current DFlash does
     anyway when accept length $< K$.)
   * Project the resulting hidden state into `_ctx_k_buf`/`_ctx_v_buf`
     normally.

The lossless property holds because: hit case uses exact target hidden
states (Lemma 1); miss case is identical to baseline DFlash. The overall
output distribution is unchanged.

### 5.3 Page allocation

The candidate K/V scratch buffers do **not** participate in target's main
paged KV allocator. They are pre-allocated once (sized by `max_batch * B`
where `B` is the max budget) and indexed directly. This avoids interaction
with the `KVCacheManager`'s page table.

When a candidate is committed (hit), we write its K/V into the main paged
KV cache at the request's *next free position*, which is exactly the
position the standard DFlash fixup forward would have written to. The
allocator behavior is therefore unchanged from DFlash's perspective.

---

## 6. DFlash Worker Extensions

This section describes the changes inside `DFlashWorker` and
`DFlashSpecMetadata`.

### 6.1 Speculation cache structure

A new field on `DFlashSpecMetadata`:

```python
@dataclass
class DFlashSpecMetadata(SpecMetadata):
    # ... existing fields ...

    # Saguaro fan-out: F_k per accept length k, length K+1.
    # Set once at construction from spec_config; constant across iterations.
    fan_out: Optional[torch.Tensor] = None  # shape [K+1], dtype int32

    # Per-iteration: candidate token IDs at each k.
    # Shape: [batch, K+1, max_F_k], padded with -1.
    candidate_tokens: Optional[torch.Tensor] = None

    # Per-iteration scratch buffers (cleared / reused each iteration).
    cand_target_hidden: Optional[torch.Tensor] = None
    cand_dflash_ctx_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
```

### 6.2 Candidate selection and pre-projection

A new method on `DFlashWorker`:

```python
def select_candidates(self, draft_logits, prev_target_logits_K):
    """
    Choose Cand_k for k=0..K.

    For k < K: top-F_k from the residual distribution
        r_k = max(p_target - p_draft, 0) at position k.
    For k = K: top-F_K from p_target at position K (no residual; this is the
        full-accept bonus distribution).

    Both are computed from quantities already produced during the
    *previous* round (the draft and target logits we already have on hand
    at the moment we issue the next round's verify). No extra forward.
    """
```

A second new method that runs immediately after the extended verify
returns hidden states:

```python
def precompute_dflash_ctx_for_candidates(
    self, cand_target_hidden_layers, draft_model
):
    """
    Apply draft_model.fc + hidden_norm to each candidate's captured
    target hidden state, producing per-candidate ctx K/V to be parked
    in `cand_dflash_ctx_kv`. On commit (cache hit), one of these is
    copied into _ctx_k_buf / _ctx_v_buf.
    """
```

### 6.3 Cache lookup at acceptance time

The existing `forward()` flow (lines 370–530 of `dflash.py`) currently:

1. Accepts draft tokens against target logits.
2. Computes `num_accepted_tokens` and accepted-token IDs.
3. Calls `_store_prefill_context` (for newly-prefilled requests) which
   projects target hidden states into `_ctx_k_buf` / `_ctx_v_buf`.
4. Runs `dflash_forward` to produce next-round draft tokens.

We insert between steps 2 and 3 a `try_commit_candidate` step:

```python
def try_commit_candidate(self, num_accepted_tokens, accepted_tokens, spec_metadata):
    """
    For each batch element: check whether the actual (k*, x*) is in
    the speculation cache. On hit, commit candidate K/V and ctx K/V
    into main buffers; on miss, mark request for fixup forward.

    Returns:
        per_request_hit_flag: BoolTensor [B]
        per_request_cand_idx: IntTensor [B], or -1 on miss
    """
```

### 6.4 Fallback path

On per-request miss, the existing single-position fixup is what
DFlash already does in the partial-accept branch. We reuse that code path
unchanged, gated by the per-request hit flag.

If a CUDA Graph for the "all-hit" fast path was captured, it cannot
handle a mixed batch with some misses. There are two options:

* **A. Issue mode**: capture two graph variants — `all_hit_graph` and
  `fallback_graph`. Choose at runtime based on `per_request_hit_flag.all()`.
  Simpler, but loses overlap when *any* request misses.
* **B. Mask mode**: capture one graph that does fixup unconditionally for
  all positions but skips positions whose hit_flag is true. This wastes
  some work on hit-only batches but supports mixed batches without
  branching.

Recommendation: start with A (simpler), measure, switch to B if mixed
batches are common.

---

## 7. PyExecutor Integration

### 7.1 Schedule changes

Current schedule, simplified
(`tensorrt_llm/_torch/pyexecutor/py_executor.py:4508`):

```
target.verify(K positions)              ← stream main
    ↓
_accept_draft_tokens                    ← stream main
    ↓
drafter.generate_draft_tokens_with_overlap   ← stream main
    ↓ (next iter)
target.verify(K positions) ...
```

Proposed schedule with target-side SSD:

```
target.verify(K + ΣF_k positions)         ← stream main
    │
    ├── select_candidates (CPU-light) ─── on host between verify and accept
    │
    ↓
_accept_draft_tokens                      ← stream main
    ↓
try_commit_candidate                      ← stream main
    ↓
   ├─ all hit ─→ skip fixup, skip drafter.* invocation,
   │            use pre-projected ctx K/V; draft forward kicks off
   │            with already-valid context
   │
   └─ any miss ─→ run fixup forward(s) for missing requests; rest as before
    ↓
drafter.dflash_forward                    ← stream main
    ↓ (next iter)
```

The **draft forward itself** still runs after verify for the current round —
SSD does not pre-run the draft forward, it pre-builds the cross-attention
context. So the latency win is: the cross-attention K/V projection
(`_store_prefill_context`'s ctx K/V projection from candidate hidden states)
happens **inside the extended verify time window**, not after. On full hit,
we save the cost of one fixup forward + one ctx K/V projection per
request.

### 7.2 Stream layout

Two CUDA streams:

* **`stream_verify`** — extended verify on target. Owns target main KV
  writes and candidate scratch writes.
* **`stream_proj`** — runs `precompute_dflash_ctx_for_candidates` (the
  fc + hidden_norm projection of candidate hidden states into draft
  ctx K/V). Reads from candidate hidden state outputs of `stream_verify`,
  synchronized via CUDA event.

The draft `dflash_forward` runs on `stream_verify` after acceptance and
commit.

### 7.3 CUDA Graph variants

We need (conservatively):

| Graph | Inputs | Outputs |
|---|---|---|
| `extended_verify_graph(B, K, ΣF_k)` | tokens, KV pages, custom mask | logits[B,K,V], hidden[L_capt, B, K+ΣF_k, H] |
| `precompute_dflash_ctx_graph(B, ΣF_k)` | hidden[L_capt, B, ΣF_k, H] | ctx_kv_scratch |
| `commit_hit_graph(B, ΣF_k)` | hit_idx | (writes into _ctx_k_buf/_ctx_v_buf and target main KV) |
| `dflash_forward_graph(B, K)` | (existing) | (existing) |
| `fixup_forward_graph(B)` | x* per request | hidden_at_fixup_pos |

All shapes are determined by the static schedule (`max_batch`,
`spec_config.fan_out`, $K$). The number of distinct captured graphs is
bounded; replay overhead is minimal.

---

## 8. Kernel Implementation Guide

### 8.1 FlashInfer paged custom-mask path (recommended)

FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` (and its sm100 / sm120
variants in TRT-LLM bindings) accepts:

* `qo_indptr[B+1]`: cumulative query offsets per batch item.
* `paged_kv_indptr[B+1]`, `paged_kv_indices[*]`,
  `paged_kv_last_page_len[B]`: paged KV layout (use existing target KV).
* `custom_mask`: a flat bitmask over $(\text{q}, \text{kv})$ pairs.

Construction of `custom_mask` for our case is fully determined by §4.2.
The mask is **structured-sparse** (block-rectangular), which FlashInfer's
mask-bit-packed kernel handles without falling off the fast path.

Per-iteration cost of constructing the mask: $O(B \cdot Q \cdot (|P|+Q))$
bits. For typical $Q \approx 24$, $|P|+Q \approx 4120$, $B \le 16$:
$\sim 12$ KiB. Trivial.

### 8.2 FlashAttention-3 with mask_mod (alternative)

FA-3's `mask_mod` callable is more flexible but pays a per-tile overhead
when masks deviate from the canonical causal pattern. For our structured
mask the deviation is small (only the candidate rows differ), but
empirical benchmark is needed before committing. Keep this as a backup
path.

### 8.3 Performance considerations (UPDATED with measured data)

A Phase 0 microbenchmark has been run; full results in
`PHASE0_RESULTS.md`. Headline numbers below; supersede earlier estimates.

**Measured attention overhead** (B200, gpt-oss-120b shape, BF16,
FlashInfer 0.6.11.post1 paged custom-mask):

| Config | Overhead vs dense | Status |
|---|---|---|
| B=1, prefix=1024, F_total=8  | **+1.3%**  | ✅ free |
| B=1, prefix=1024, F_total=16 | **+1.6%**  | ✅ free |
| B=1, prefix=4096, F_total=8  | +29%       | ❌ marginal |
| B=1, prefix=4096, F_total=16 | +43%       | ❌ over budget |
| B=2, prefix=1024, F_total=16 | +15%       | ✅ ok |
| B=4, prefix=1024, F_total=16 | +109%      | ❌ catastrophic |
| B≥8, any prefix, any F>0     | +100% to +1400% | ❌ catastrophic |

**Two regimes empirically observed**, separated by $\bar Q = B \cdot (K + F_{\text{total}})$:

* **Memory-bound** ($\bar Q \lesssim 30$): attention dominated by KV
  reads + kernel launch fixed cost; candidate queries nearly free.
* **Compute-bound** ($\bar Q \gtrsim 50$): attention scales linearly
  with $\bar Q$, no "free" benefit.

The crossover at $\bar Q \approx 32$–$50$ is **much earlier than the
~300 estimate in §3.5** (which was for FFN, not attention). Attention
crosses over much earlier than FFN.

**Implications:**

1. **T-SSD must be runtime-gated** by $(B, \text{prefix\_len},
   F_{\text{total}})$. Static "always on" is not viable.

2. **Practical viability is narrow**: B ≤ 2, prefix ≤ ~2048. Beyond
   that, attention overhead exceeds any plausible cache-hit savings.

3. **Realistic end-to-end speedup**: 5–10% at BS=1 gpt-oss-120b in
   viable prefix range; possibly 15–25% on Kimi-K2.5 (higher $a_p$);
   below 0% (regression) at high batch or long prefix.

The original estimate of +10% verify time was directionally correct at
short prefix but blew up at typical decode lengths. **Anyone implementing
this must read PHASE0_RESULTS.md before starting Phase 1 / Phase 2.**

---

## 9. Validation Strategy

### 9.1 Correctness: lossless equivalence

**Test**: Run baseline DFlash and target-side SSD DFlash on the same
prompts, same sampling seed, same temperature. Output token sequences
must be **bit-identical** (or sampling-distribution-identical for
$T > 0$, validated by KL divergence of empirical token frequencies over
many runs).

**Why this should hold**: Lemma 1 + Corollary 1 say all candidate hidden
states are bitwise-identical to what target would produce in a fixup
forward at that position. The cache hit substitutes one for the other.
Cache miss falls back to baseline. No sampling change.

**Test infrastructure**: Add to `tests/unittest/_torch/speculative/` a
test parameterized on (model, T, F-budget) that runs ~64 prompts and
asserts equivalence to baseline.

### 9.2 Cache hit rate vs theoretical

Instrument `try_commit_candidate` to log per-iteration $(k, x^*)$ and
membership flag in `Cand_{k^*}`. Aggregate over a SPEED-bench coding
subset run; expected curve from Saguaro Fig 3 / Theorem 12 plus the
empirical $a_p$ from `acceptance_rate.json`. Plot and compare.

### 9.3 End-to-end perf

Run on SPEED-bench coding subset, gpt-oss-120b, T = 1.0, BS ∈ {1, 2, 4, 8, 16},
DL = 4 (matches existing baseline data). Compare:

* Baseline DFlash decode tok/s
* T-SSD with $B \in \{8, 16, 32\}$ candidate budget

Win threshold: ≥ 5% decode tok/s improvement at $B{=}1$ to justify the
implementation cost. If under 5%, revisit the design (see §10).

---

## 10. Open Questions and Risks

* **Attention overhead is the swing factor.** If FlashInfer's structured-
  mask kernel does not stay near the dense-attention performance for our
  mask shape, the design is a wash. **A dedicated microbenchmark in
  Phase 1 is mandatory.**

* **High-batch regime.** At $B \cdot K \gtrsim 50$ on B200, target moves
  toward compute-bound and adding candidate positions is no longer free.
  Analogous to Saguaro's `Theorem 17` (fast vs neural backup), at high
  batch we should either reduce $B$ aggressively or disable T-SSD and
  fall back to baseline DFlash.

* **CUDA Graph proliferation.** Each (batch_size, fan-out) combination
  needs its own graph. We may need a graph-pool with eviction.

* **Bonus token sampling for $T > 0$.** When the verify outcome is full-
  accept, $x^*$ is sampled from $\pi^{(K)}_{\text{tgt}}$. The candidate
  set $\text{Cand}_K$ should be its top-$F_K$. For correctness of the
  hit/miss decision, we sample $x^*$ deterministically given a seed
  (already standard in TRT-LLM's sampler).

* **Saguaro sampling extension.** Saguaro §4.2 introduces a residual-
  control sampling scheme that biases the draft's sampled token to make
  the residual distribution easier to predict, improving hit rate. We
  document it in Appendix A but defer implementation to phase 2 — it
  changes draft sampling and the user requested no changes to the
  draft model's behavior.

* **Multi-rank / TP.** Extended verify must shard the same way as main
  verify (TP across attention heads). The candidate KV scratch buffers
  must be allocated per rank, in the rank's FP/quant dtype.

---

## 11. Implementation Phases

### Phase 0 — Spec & feasibility (this document, plus benchmarks) ✅ COMPLETED

* This document. ✓
* Microbenchmark FlashInfer paged custom-mask attention with the §4.2
  mask shape on B200 + gpt-oss-120b weights (or a same-arch proxy).
  Measure overhead vs dense baseline at $B{=}1, 2, 4, 8, 16$, $K{=}4$,
  $\sum F_k \in \{8, 16, 32\}$. ✓ — see `bench_dflash_tssd_mask_overhead.py`
* **Gate**: proceed only if attention overhead at $B{=}1$, $\sum F_k = 16$
  is < 25% of dense verify-attention time.
* **Result** (see `PHASE0_RESULTS.md`):
  - $B{=}1$, prefix=1024, $\sum F_k{=}16$: **+0.9%** ✅ pass
  - $B{=}1$, prefix=1024, $\sum F_k{=}8$: +1.2% ✅
  - $B{=}1$, prefix=4096, $\sum F_k{=}8$: +21.5% ✅ (marginal)
  - $B{\geq}4$: catastrophic (≥130%) ❌
* **Decision**: proceed with narrow scope — gate T-SSD on
  $(B, \text{prefix}, F_{\text{total}})$, default $F_{\text{total}}{=}8$,
  disable at $B{\geq}4$.

### Phase 1 — Reference Python implementation ✅ COMPLETED

* Pure PyTorch reference (no CUDA Graph, no kernel fusion) of the entire
  T-SSD flow on a toy transformer to validate:
  - Mask construction matches §4.3 worked example bitwise. ✓
  - Lemma 1 (causal one-token swap) holds: prefix unchanged $\leq$ 1e-6. ✓
  - Extended verify == individual fixup forwards (within FP32 noise
    $\sim$ 2e-4). ✓
  - Cache hit rate matches Saguaro Theorem 12 prediction within 5%
    (measured: 0–3.4% relative error). ✓
  - End-to-end lossless simulation: 6 rounds hit/miss, final tokens
    identical to baseline. ✓
  - Gating policy matches Phase 0 viable configs. ✓
* All 6/6 tests pass. See `phase1_reference.py`.
* **Mask correction discovered during Phase 1**: original §4.2 mask shape
  $Q \times (|P|{+}K)$ did not include candidate self-attention, making
  extended verify NOT mathematically equivalent to per-candidate fixup
  forward. Corrected mask: KV grew to $|P|{+}K{+}\sum F_k$ with a
  diagonal self-attention slot per candidate. After correction the
  equivalence test passes within FP32 noise. The bench was re-run with
  the corrected mask shape (see `PHASE0_RESULTS.md`).

### Phase 2 — Production kernel + scheduler

* FlashInfer custom-mask integration for target verify.
* DFlash worker extensions (§6).
* PyExecutor schedule changes (§7).
* CUDA Graph capture for the new graphs (§7.3).
* End-to-end perf on gpt-oss-120b at SPEED-bench.

### Phase 3 — Saguaro sampling

* Residual-control sampling on draft (Saguaro §4.2 + Appendix A.3) to
  push hit rate higher. Behind a flag; off by default for users who want
  byte-exact equivalence to baseline DFlash sampling.

### Phase 4 — High-batch fallback

* Auto-disable / reduce $B$ at high batch (compute-bound regime).
* Optional: n-gram backup speculator at high batch (Saguaro §4.3).

---

## Appendix A: Saguaro sampling for residual control

(Documented for completeness; not in initial scope.)

Given draft logits $z \in \mathbb{R}^V$ and a downweight constant
$C \in [0, 1]$, the Saguaro sampling scheme is:

$$
\sigma_{F, C}(z)_t \propto \begin{cases}
C \cdot \exp(z_t) & t \in \text{top-}F(z) \\
\exp(z_t) & \text{otherwise}
\end{cases}
$$

This biases the draft distribution *away* from its top-$F$, which by
construction concentrates the residual distribution
$r \propto \max(\pi_{\text{tgt}} - \pi_{\text{drf}}, 0)$ *onto* those
top-$F$ tokens — exactly the candidates in $\text{Cand}_k$. As
$C \to 0$, hit rate increases; acceptance rate $a_p$ decreases. The
optimal $C$ trades off the two (Saguaro Fig 5).

For DFlash, this would apply to the draft's logits inside
`prepare_1st_drafter_inputs` / the draft model's mask-token output. Doing
so changes the *distribution* of draft tokens compared to baseline DFlash,
so token-level equivalence with baseline is lost (though full target-
distribution correctness is preserved, since Saguaro sampling is still
lossless in the rejection-sampling sense).

---

## Appendix B: Code anchors summary

| Concern | File:Line | What changes |
|---|---|---|
| DFlash worker forward | `tensorrt_llm/_torch/speculative/dflash.py:370` | Insert `try_commit_candidate` between accept and `_store_prefill_context`; route hits to skip fixup |
| DFlash ctx K/V projection | `tensorrt_llm/_torch/speculative/dflash.py:292` (`_store_prefill_context`) | Generalize to write into `_ctx_k_buf` / `_ctx_v_buf` either from a candidate scratch OR from a fresh fixup (existing behavior) |
| DFlash worker init | `tensorrt_llm/_torch/speculative/dflash.py:158` (`__init__`) | Allocate candidate scratch buffers (cand_target_hidden, cand_dflash_ctx_kv) sized by `max_batch * sum(F_k)` |
| Spec metadata | `tensorrt_llm/_torch/speculative/dflash.py:37` (`DFlashSpecMetadata`) | Add `fan_out`, `candidate_tokens`, scratch buffer references |
| Spec config | `tensorrt_llm/llmapi/llm_args.py` (`DFlashDecodingConfig`) | Add `tssd_fan_out_budget` (int, default 0 = disabled), `tssd_power_law_r` (float, default 0.5) |
| Drafter overlap | `tensorrt_llm/_torch/speculative/model_drafter.py:856` (`generate_draft_tokens_with_overlap`) | Detect T-SSD-on path; on full hit skip drafter forward (DFlash forward already kicked off in commit step) |
| PyExecutor | `tensorrt_llm/_torch/pyexecutor/py_executor.py:4508` (`_handle_speculative_decoding`) | Insert `select_candidates` call; commit candidates after acceptance; conditional drafter invocation |
| Target verify kernel | wherever target's attention runs for spec-dec verify (`attention_backend/`) | New code path that takes the §4.2 custom mask and computes the extended verify |
| Tests | new file `tests/unittest/_torch/speculative/test_dflash_tssd_lossless.py` | Equivalence test vs baseline DFlash |

---

## Appendix C: Glossary of acronyms

* **DFlash** — block-diffusion speculative decoder (this codebase).
* **SSD** — speculative speculative decoding (Saguaro paper, arXiv:2603.03251).
* **T-SSD** — target-side SSD; this design.
* **AR** — acceptance rate (mean number of accepted draft tokens per round).
* **TP** — tensor parallelism.
* **MQ attention** — multi-query attention pass with non-rectangular,
  custom mask.
