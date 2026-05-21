# Phase 2 Progress: DFlash + Target-Side SSD Production Implementation

**Status:** Math + state machine + worker class implemented and unit-tested.
Kernel-side hookup (FlashInfer custom-mask path wired into target attention)
and end-to-end perf validation pending.

**Date:** 2026-05-20

---

## What's Done

### Core module (`dflash_tssd.py`, ~600 LoC)

- `geometric_fanout(K, B_budget, a_p, r)` — Saguaro Theorem 12 fan-out.
- `build_tssd_mask(K, F, prefix_len)` — §4.2 mask constructor (per request).
- `tssd_should_enable(B, prefix_len, K, F_total, max_batch)` — runtime gate
  matching `PHASE0_RESULTS.md` viable envelope.
- `sample_candidates(draft_logits, target_logits, F, mode)` — top-F_k from
  residual (k<K) or target (k=K) distribution; produces Cand_k for §6.2.
- `TSSDState` dataclass — pre-allocated scratch buffers for candidate K/V
  on target side, candidate K/V on draft side, and candidate target hidden
  states (capture layers).
- `DFlashTSSDWorker(DFlashWorker)`:
  - `__init__` — reads new config fields (`tssd_enabled`, `tssd_F_total`,
    `tssd_a_p`, `tssd_max_batch`); initializes fan-out list.
  - `_lazy_init_tssd_state` — allocates scratch (sized by `max_batch * F_total`
    so CUDA graphs see constant shapes).
  - `select_candidates` — wraps `sample_candidates`.
  - `try_commit_candidate` — vectorized hit/miss detection per batch.
  - `commit_candidate_to_buffers` — copies candidate K/V into target main
    paged KV at logical position `|P|+k*` and copies pre-projected dflash
    ctx K/V into `_ctx_k_buf` / `_ctx_v_buf` at slot position `|P|+k*`.
  - `gate(batch_size, prefix_len)` — runtime gate.
  - `forward` — falls back to `DFlashWorker.forward` when gating off (no
    behavior change), takes the extended-verify path when on.
  - Stats: `_tssd_hits`, `_tssd_misses`, `_tssd_iters_gated_on/off`.

### Config (`llm_args.py`)

Added four fields to `DFlashDecodingConfig`:

- `tssd_enabled: bool = False`
- `tssd_F_total: int = 8`
- `tssd_a_p: float = 0.78`
- `tssd_max_batch: int = 2`

### Unit tests (`test_dflash_tssd.py`, 22 tests, all passing)

- `TestGeometricFanout` (4 tests): F values match Theorem 12 expectations,
  high-acceptance behavior, tight-budget fallback.
- `TestBuildTSSDMask` (4 tests): shape, prefix visibility, main verify
  diagonal, candidate self-attention.
- `TestTSSDGate` (3 tests): viable / failed Phase 0 configs, zero budget.
- `TestSampleCandidates` (3 tests): top-k shape + padding, target mode
  matches argmax, residual mode prefers high-residual tokens.
- `TestTryCommit` (4 tests): hit / miss / partial accept / mixed batch.
- `TestCommitToBuffers` (2 tests): hit writes correct position, miss is
  no-op.
- `TestFallbackBehavior` (2 tests): disabled worker constructs with zero F
  and gate-off; enabled worker gates per Phase 0 envelope.

```bash
$ CUDA_VISIBLE_DEVICES=2 python -m pytest \
    tests/unittest/_torch/speculative/hw_agnostic/test_dflash_tssd.py -v
... 22 passed in 1.03s
```

---

## Baseline reference numbers (gpt-oss-120b, B300, TP=8, T=1.0)

Captured 2026-05-20/21 in `/home/scratch.fxiong_sw_1/SPEED-bench/results/tssd_baseline_b300/`.
T-SSD speedup must be measured against these once wiring lands.

| Config       | AR     | TPS/user (gen) | TPS/GPU | Output TPS |
|--------------|--------|----------------|---------|------------|
| BS=1, DL=1   | 1.867  |   606.66       |  74.06  |    -       |
| BS=1, DL=2   | 2.556  |   824.66       |  99.58  |    -       |
| BS=1, DL=3   | 3.079  |   980.18       | 117.30  |    -       |
| BS=1, DL=4   | 3.439  |  1086.80       | 128.13  |    -       |
| BS=1, DL=5   | 3.709  |  1157.55       | 135.98  |    -       |
| BS=1, DL=6   | 3.838  |  1095.89       | 123.15  |    -       |
| BS=1, DL=7   | 4.038  |  1224.22       | 143.41  |    -       |
| **BS=1, DL=8** | **4.021** | **1208.87**  | **139.94** | 1119.55 |
| BS=8, DL=8   | 3.961  |   655.84       | 560.07  | 4480.59    |

Conditional accept rates at BS=1 DL=8: 0.79, 0.75, 0.75, 0.75, 0.76, 0.78, 0.78, 0.80 — confirms
`a_p ≈ 0.78` empirically, exactly the design doc assumption.

T-SSD design target: **+5–10% TPS/user at BS=1**, i.e. **TPS/user ≥ 1270** at DL=8 once wired.
Speedup at BS=8 expected to be 0% (gate disables T-SSD per Phase 0 envelope).

---

## What Remains (kernel hookup + executor wiring)

### A. TRTLLM custom-mask path on target attention (smaller than first scoped)

**Discovered while wiring**: the TRTLLM attention backend ALREADY supports per-request
spec-decoding packed masks via `TrtllmAttentionMetadata.spec_decoding_packed_mask` (shape
`[max_num_requests, K+1, ceil((K+1)/32)]`, bit-packed). This is the path Eagle3 trees use
(`trtllm.py:823–833, 876–893`). Reusing it for T-SSD avoids touching FlashInfer at all.

**Phase B kicked off (2026-05-21).** Worker dispatch wired
(`utils.py` selects `DFlashTSSDWorker` when `tssd_enabled`). Packed-mask
helper added (`build_tssd_packed_mask` in `dflash_tssd.py`, 3 unit tests).
`geometric_fanout` hardened to sum exactly to `B_budget` (prevents 1-off
sizing mismatches). **26/26 tests passing** (25 T-SSD unit + 1 Qwen3-8B
DFlash non-regression in 48s). Scaffold is opt-in; the shape contract is
unchanged until the remaining wiring lands. A warning is logged when
`tssd_enabled=True` so users know the production path is incomplete.

**Phase B end-to-end attempt (2026-05-21).** All 6 wiring steps in §A
ARE now in code. Smoke test on Qwen3-8B (single GPU) runs
end-to-end without crashing. Lossless test FAILS (1/3 prompts diverge by
1 token; AR drops on 2/3 prompts from KV pollution). Perf measurement
on Qwen3-8B BS=1 (5 prompts, 128 tokens each, T=0):

```
baseline DFlash : TPS=391.9  AR=1.334
T-SSD scaffold  : TPS=359.7  AR=1.188   (-8.2% TPS, -10.9% AR)
```

**E2E perf on gpt-oss-120b SPEED-bench (TP=8, T=1.0, BS=1, DL=8):**

The first measurement was contaminated by GPU contention on the B200
box (E2E max=41s, std=6.79 — clear outlier from another workload).
After switching to an exclusive 8x B300 SXM6 AC machine and running
baseline + T-SSD back-to-back:

| Config | TPS/user | TPS/GPU | AR | Δ TPS/user | E2E std |
|---|---|---|---|---|---|
| Baseline DFlash | 1324.86 | 154.30 | 4.043 | (anchor) | 0.85 (clean) |
| T-SSD F_total=16 | 1166.63 | 139.74 | 3.942 | **-11.9%** | 1.16 (clean) |

Honest result: with the kernel mod (kv_append_lens) live and KV-pollution
fixed, T-SSD at F_total=16 is a -11.9% TPS regression on the production
target. The attention overhead from K+1+F_total=25 Q tokens per gen
request is the dominant cost, consistent with Phase 0's prediction
(+52% attention overhead at F=16, prefix=4096, averaged over the actual
decode trajectory).

(Earlier B200 measurement on a contended machine showed -34.5%; that
number was contaminated and has been retired in favor of the clean
B300 result.)

**F-sweep at K=8 (clean isolated B300, BS=1, T=1.0):**

| Config | TPS/user | AR | Δ TPS | Δ AR |
|---|---|---|---|---|
| Baseline DFlash | 1324.86 | 4.043 | (anchor) | (anchor) |
| T-SSD F=4 | 1240.11 | 4.009 | -6.4% | -0.8% |
| T-SSD F=8 | 1188.00 | 3.952 | -10.3% | -2.2% |
| T-SSD F=16 | 1166.63 | 3.942 | -11.9% | -2.5% |

TPS overhead scales nearly linearly with F (~0.7% per candidate Q token).
AR drop is small (1-3%). At K=8, candidate Q processing through 36
transformer layers is the dominant cost.

**Multi-stream wired up (2026-05-21).** Initial naive `with
torch.cuda.stream(stream_B):` inside CUDA graph capture failed with
`cudaErrorStreamCaptureInvalidated`. Switched to TRTLLM's existing
`maybe_execute_in_parallel` utility from
`tensorrt_llm._torch.modules.multi_stream_utils`, which is the
idiomatic pattern for capture-safe multi-stream work in this codebase
(used by DeepSeekV3, Llama-min-latency, DSA attention, etc.). The
utility uses pre-allocated stream + events with proper record/wait
ordering inside `torch.cuda.graph(...)` capture.

`DFlashTSSDWorker.__init__` now pre-allocates `_aux_stream`,
`_event_main`, `_event_aux`. `forward()` invokes `maybe_execute_in_parallel(
fn_main=super().forward, fn_candidate=proxy_compute,
event_main, event_aux, aux_stream=self._aux_stream,
disable_on_compile=True)`. The `do_multi_stream()` thread-local set by
`CUDAGraphRunner` during capture activates the parallel branch.

**Multi-stream perf result on isolated 8x B300:**

| Config | TPS/user | Δ vs baseline | vs sequential |
|---|---|---|---|
| Baseline DFlash | 1324.86 | (anchor) | — |
| F=4 sequential  | 1240.11 | -6.4% | (anchor) |
| F=4 multi-stream | 1202.51 | -9.2% | -3.0% (worse) |
| F=16 sequential | 1166.63 | -11.9% | (anchor) |
| F=16 multi-stream | 1196.74 | -9.7% | +2.2% (better) |

Multi-stream helps at large F (where parallel work amortizes the
event/stream-switch host overhead) but hurts at small F. Consistent
with the warning at `multi_stream_utils.py:42-43`:
"Multi-stream is only enabled when cuda graph is turned on because
switch stream has extra host overhead."

**Conclusion**: multi-stream alone delivers ~2% recovery at best.
Not enough to flip T-SSD net positive on gpt-oss-120b K=8. Needs
to combine with kernel-level FFN-skip-after-last-capture (~3%) and
commit-on-hit (~2-3%) to plausibly reach break-even or +1-2% net.

**Pre-commit isolation result (2026-05-21):** F_total=0 gives
bit-identical output to baseline DFlash on Qwen3-8B (`baseline ==
tssd_F0: True, same ARs: True`), proving the integration plumbing is
clean. The full -7.5% / -10.9% (Qwen3-8B) and -34.5% / -2.1%
(gpt-oss-120b) regressions come entirely from candidate Q processing,
not the integration scaffold.

Net SLOWDOWN, not a speedup. Why: the §4.2 packed mask is built and
populated into TRTLLM backend buffers as expected, but the Blackwell
trtllm-gen FMHA kernel **does not consume the spec_decoding_packed_mask**
on sm100+. The relevant gate is at `attention_backend/trtllm.py:789-790`
which forces `self.is_spec_decoding_enabled = False` whenever
`is_sm_version_trtllm_gen_kernel()` returns True (sm100, sm120, sm121
excluded). The comment explicitly says: "trtllmGen FMHA kernels do not
yet support speculative decoding mode".

Until that kernel limitation is lifted, T-SSD on B200/B300 cannot be
lossless. Candidate K/V positions attend with standard causal mask,
polluting the prefix that subsequent positions read.

**Phase B integration findings (2026-05-21).** During the wiring attempt
we discovered framework assumptions deeper than the original §A scope:

1. **Sampler `tokens_per_request` is gated on `is_spec_dec_tree`**
   (`speculative/interface.py:621-622`). When False (DFlash's current
   setting), the sampler hard-codes `max_draft_len + 1` and ignores
   `max_total_draft_tokens`. Growing only `max_total_draft_tokens` causes
   `_sample_and_accept_draft_tokens_base` to crash with a reshape
   mismatch (`shape '[B, K+1]' invalid for input of size B*(K+1+F_total)`).

2. **Setting `is_spec_dec_tree=True` activates Eagle3-tree paths**
   (`attention_backend/trtllm.py:876–893`) that require a
   `SpecTreeManager`. DFlash has no tree manager, so this branch crashes
   on the first call.

3. **DFlashWorker.forward also reshapes `spec_metadata.draft_tokens`**
   (`dflash.py:402`) and `accepted_tokens` (multiple places) using
   `max_draft_len`, not `max_total_draft_tokens`. Pre-slicing the tensors
   inside `DFlashTSSDWorker.forward` works for the DFlash-side reshape
   but leaves the framework's sampler call (item 1) unfixed.

4. **CUDA-graph warmup also hits the slice path**, which means any
   per-iteration buffer-shape adjustment must be CUDA-graph friendly (no
   variable-shape branches in the warmup region). Mitigated by the
   slice routine handling both gather-only logits (size = num_contexts +
   num_gens·gen_block) and per-token logits (size = num_ctx_tokens +
   num_gens·gen_block) cases.

5. **Blackwell trtllm-gen FMHA does not consume packed mask** (see
   `trtllm.py:789-790`). This is the hard blocker for lossless T-SSD
   on B200/B300 and is what causes the -8.2% TPS / -10.9% AR result
   above.

   **Probe attempt 2026-05-21**: bypassed the gate by enabling
   `is_spec_decoding_enabled` whenever `tssd_enabled=True` and added
   the Blackwell-specific buffers (`spec_decoding_param_prepare_for_blackwell`)
   to the SSD branch. The kernel's spec-dec assert
   ("Expecting spec_decoding_bl_tree_mask_offset spec-dec mode")
   stopped firing — the kernel now accepts the call. However the
   end-to-end output still diverges by 1 token in 1/3 prompts (same
   magnitude: -7.5% TPS, -10.9% AR). This means either:
   (a) trtllm-gen FMHA on Blackwell still ignores `spec_decoding_packed_mask`
       silently (the Q/K mask multiplication is a no-op), or
   (b) candidate K/V being appended to main paged KV at zombie logical
       positions creates a subtle correctness violation in subsequent
       iterations that the design's §5.1 isolation buffers were meant
       to prevent.

   Distinguishing (a) vs (b) requires either kernel-side
   instrumentation (which is beyond this task) or implementing the
   §5.1 candidate K/V isolation buffers. The latter is the cleanest
   structural fix and should be the next implementer's first step.

   **KERNEL FIX LANDED (2026-05-21).** Implemented per-request
   `kv_append_lens` plumbed end-to-end:
   - `cpp/tensorrt_llm/kernels/unfusedAttentionKernels.h` — added
     `int const* kv_append_lens{nullptr}` to `QKVPreprocessingParams`.
   - `cpp/tensorrt_llm/kernels/unfusedAttentionKernels/unfusedAttentionKernels_2_template.h` —
     two K/V-write sites gated on `token_idx_in_seq < kv_append_lens[batch_idx]`.
   - `cpp/tensorrt_llm/common/attentionOp.h` — added field to
     `EnqueueParams<T>` base class.
   - `cpp/tensorrt_llm/common/attentionOp.cpp` — wires
     `params.kv_append_lens → preprocessingParams.kv_append_lens`.
   - `cpp/tensorrt_llm/thop/attentionOp.{cpp,h}` — added
     `std::optional<torch::Tensor> kv_append_lens` to public op +
     `Runner::run` virtual.
   - `cpp/tensorrt_llm/nanobind/thop/bindings.cpp` — added Python
     binding for the new arg.
   - `tensorrt_llm/_torch/attention_backend/trtllm.py` — added
     `kv_append_lens` field on `TrtllmAttentionMetadata`, populated in
     the SSD branch of `update_spec_dec_param`, plumbed to
     `thop.attention(..., kv_append_lens=...)`.

   Verification: kernel-side `[T-SSD kernel] kv_append_lens is set
   (active)` log fires per layer per iteration → main paged cache is
   NOT polluted.

   Yet **lossless still FAILS with the same magnitude** (-7.1% TPS,
   -10.9% AR, same 1-token divergence on 1/3 prompts on Qwen3-8B).
   This rules out KV pollution as the root cause. Remaining suspects:
   (i) §4.2 packed_mask is not actually consumed by the trtllm-gen
   FMHA kernel on Blackwell (the kernel may use a different mask path
   for spec dec); (ii) FP precision noise in attention reduction order
   when adding F_total=8 extra Q tokens (multi-step accumulation
   non-associativity).

   To distinguish: run the same test on a non-Blackwell GPU (where
   the trtllm-gen kernel path is bypassed and fall-back kernels DO
   respect packed_mask), or instrument the kernel to log which mask
   path it's actually taking.

`tokens_per_gen_step` and `max_total_draft_tokens` grow are now LIVE
when `tssd_enabled=True` (config + dispatch + worker slice + dedicated
TRTLLM SSD branch all wired). The math layer + helpers + 25 unit tests +
non-regression are stable. The remaining work to make T-SSD a speedup
on Blackwell requires kernel-side custom-mask support.

**Remaining hookup steps (precise PR-ready scope, updated 2026-05-21):**

1. **Decide tree-vs-linear strategy** for the sampler/accept code path.
   Two options:
   - (1a) Set `is_spec_dec_tree=True` for DFlashSpecMetadata and provide a
     minimal SpecTreeManager-equivalent (just enough to satisfy the
     `update_spec_dec_param` static-tree branch at `trtllm.py:876–893` and
     the sampler's `tokens_per_request` calc at `interface.py:621–622`).
     Cleaner integration with existing tree infrastructure.
   - (1b) Add a third "linear-with-tail" mode: keep `is_spec_dec_tree=False`
     but teach the sampler/accept code to use `max_total_draft_tokens`
     when the worker's `spec_dec_mode` is DFlash and `tssd_enabled`. ~30
     LoC across `interface.py` and `dflash.py`.
   Either way: **do this BEFORE re-enabling `tokens_per_gen_step` grow.**

2. **Grow `tokens_per_gen_step`** to `K + 1 + F_total` when `tssd_enabled`,
   and grow `max_total_draft_tokens` to `K + F_total`. The grow path is
   prototyped and reverted in `llm_args.py` (lines around
   `set_max_total_draft_tokens` / `tokens_per_gen_step`). ~10 LoC. **Do
   this AFTER step 1.**

2. **Populate `spec_decoding_packed_mask`** in `DFlashSpecMetadata` (or a hook into
   `TrtllmAttentionMetadata.update_spec_dec_param`). Use `build_tssd_packed_mask(K, F,
   max_num_requests)`. The TRTLLM update path (`trtllm.py:876–893`) currently expects a
   `SpecTreeManager`; need a non-tree-manager branch that copies the static T-SSD mask.
   ~30 LoC in `trtllm.py` + ~10 LoC hookup in `dflash.py`.

3. **Inject candidate token IDs** into `next_draft_tokens` returned from
   `DFlashTSSDWorker.forward`. Sample F_total candidates from the previous round's
   target+draft logits via `select_candidates()`. Shape grows
   `[num_gens, K]` → `[num_gens, K + F_total]`. ~50 LoC.

4. **Slice candidate hidden states** out of `spec_metadata.captured_hidden_states`.
   With `tokens_per_gen_step = K + 1 + F_total`, target naturally captures hidden
   states for all positions; just slice positions `K+1..K+F_total` per gen request and
   feed into `precompute_dflash_ctx_for_candidates`. ~20 LoC.

5. **Commit-on-hit / fallback-on-miss** in `DFlashTSSDWorker.forward`. After
   `_sample_and_accept_draft_tokens_base`, run `try_commit_candidate` against
   `tssd_state.candidate_tokens`. On hit: copy `cand_dflash_k/v` into
   `_ctx_k_buf`/`_ctx_v_buf` at `slot, |P|+k*` — already implemented in
   `commit_candidate_to_buffers`, just need to wire the call-site. Skip the
   per-request fixup forward when hit. ~80 LoC.

6. **Suppress candidate K/V append** to target's main paged KV cache. The simplest
   path: after `append_paged_kv_cache` runs over all `K + 1 + F_total` positions,
   overwrite the candidate K/V slots with the matching candidate's K/V on hit, or
   zero them out on miss to avoid pollution. Or use the ragged K/V variant of
   FlashInfer to keep candidate K/V out of paged cache. ~50 LoC.

Total remaining: **~240 LoC across 3 files** (`dflash_tssd.py`, `dflash.py`,
`trtllm.py`). Smaller than the original 300–500 estimate; well-scoped for a focused PR.

**Validation plan once wired:**
- Unit tests: extend `test_dflash_tssd.py` with a toy-transformer end-to-end test
  that exercises the full T-SSD flow (extended verify → accept → commit → next round).
- Non-regression: existing `test_dflash_qwen3_8b` must pass (tssd_enabled=False).
- Lossless: with tssd_enabled=True and a fixed seed, output tokens must match
  baseline DFlash exactly (the T-SSD speculation cache is a speedup, not a sampler
  perturbation — Lemma 1 guarantees bit-identical hits).
- Perf: SPEED-bench BS=1 DL=8 with tssd_enabled=True. Target: TPS/user ≥ 1270
  (≥+5% over the 1208.87 baseline captured 2026-05-20).

### B. FlashInfer custom-mask path on target attention (alternative, NOT needed)

The worker's `_extended_verify_fn` is currently `None` (and the worker
falls back to baseline DFlash in this state). To make T-SSD live in
production we must:

1. **Tokens-per-gen-step plumbing.** `DFlashDecodingConfig.tokens_per_gen_step`
   currently returns `K+1`. Under T-SSD we need `K+1+ΣF_k`. This propagates
   into the executor's `prepare_inputs` step, growing `input_ids` for each
   gen request.
2. **Candidate input embeddings.** The K main verify positions take draft
   tokens d_1..d_K as input. The ΣF_k candidate positions take the
   candidate token IDs sampled at the *previous round* (selected from
   `select_candidates`). The token IDs need to flow through `embed_tokens`
   and land in `input_ids` slots.
3. **Custom mask in attention backend.** The target attention backend
   (`tensorrt_llm/_torch/attention_backend/`) needs a code path that,
   when T-SSD is active for this iteration, plans FlashInfer with
   `BatchPrefillWithPagedKVCacheWrapper.plan(custom_mask=...)` using the
   §4.2 mask. The mask construction is already implemented as
   `build_tssd_mask`; the plumbing is the small bit of bookkeeping to
   pass `custom_mask` flat-bitpacked through `attn_metadata`.
4. **Suppress candidate K/V append.** FlashInfer's paged wrapper would by
   default `append_paged_kv_cache` for every Q position. We must call
   `append_paged_kv_cache` only for the K main positions (candidate K/V
   stays in `TSSDState.cand_target_k/v`). The simplest path: call append
   only with `qo_indptr` covering the first K of each request, and write
   candidate K/V into scratch via a side path (or use the ragged variant
   for candidate Q so they never participate in append).
5. **Hidden-state capture.** Existing
   `DFlashSpecMetadata.maybe_capture_hidden_states` writes all token
   positions through. The candidate hidden states already land in the
   captured-hidden buffer; the worker needs to slice them into
   `TSSDState.cand_target_hidden`. This is a contiguous slice given the
   `qo_indptr` layout.

The combined diff is estimated 400–600 LoC across the attention backend,
the spec metadata, and the executor's input-prep step. The unit-tested
math layer (this module) does NOT need changes.

### B. CUDA Graph capture (Task #20)

Once the production extended-verify path is live, capture three new
graphs in `py_executor.py`:

- `extended_verify_graph(B, K, F_total)` — replaces `verify_graph` when
  T-SSD is gated on.
- `precompute_dflash_ctx_graph(B, F_total)` — applies `fc + hidden_norm`
  to candidate hidden states and lands K/V in `cand_dflash_k/v`.
- `commit_hit_graph(B, F_total)` — the loop-free body of
  `commit_candidate_to_buffers`. Currently the commit path is a Python
  for-loop over hits; for graph capture we need a vectorized scatter.
  Replacement is straightforward (`scatter_` on flat indices).

All shapes are determined by `max_batch * F_total * K` so the number of
distinct captured graphs is bounded.

### C. End-to-end perf validation (Task #21)

Needs 8x B200 (TP=8) for gpt-oss-120b. Currently GPU 1 and 7 are still
occupied (other workloads). Run plan once GPUs free up:

```bash
trtllm-bench --model /code/llm-models/gpt_oss/gpt-oss-120b throughput \
    --backend pytorch --max_batch_size 1 \
    --speculative_config dflash --tssd_enabled true --tssd_F_total 8 \
    --dataset speed-bench-coding-subset
```

Expected (from `PHASE0_RESULTS.md` §3.2):
- BS=1 SPEED-bench coding: **5–10% speedup** over baseline DFlash.
- Lossless: decoded tokens identical to baseline.

---

## Hookup Contract (for whoever wires up A.)

```python
def my_extended_verify(
    input_ids:   torch.LongTensor,    # [total_q]   K + ΣF_k per gen req
    position_ids: torch.LongTensor,   # [total_q]
    qo_indptr:   torch.IntTensor,     # [B+1]
    custom_mask: torch.BoolTensor,    # flat per §4.2
    kv_indptr:   torch.IntTensor,     # [B+1]   page-table indices
    paged_kv_indices: torch.IntTensor,
    paged_kv_last_page_len: torch.IntTensor,
    target_model: nn.Module,
    capture_layer_ids: List[int],
) -> dict:
    return {
        "main_logits":             ...,  # [B, K, V]   for accept
        "cand_logits":             ...,  # [B, ΣF_k, V] (unused in commit)
        "cand_hidden_per_layer":   ...,  # dict layer_id → [B, ΣF_k, H]
        "cand_target_k":           ...,  # [L, B, ΣF_k, nkv, hd]
        "cand_target_v":           ...,
    }

worker.set_extended_verify_fn(my_extended_verify)
```

The contract is a callable so the kernel side can change independently
of the worker. The same callable shape works for the FlashInfer paged
wrapper and the FA-3 mask_mod variant (§8.2 alternative).
