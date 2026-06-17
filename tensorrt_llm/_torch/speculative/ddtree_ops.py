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
"""CUDA-graph-safe construction of the DDTree draft tree.

DFlash produces per-position marginal distributions {q_1..q_K} (K = max_draft_len)
in a single drafter forward. The linear DFlash path collapses these to one chain
via argmax. DDTree (https://arxiv.org/pdf/2604.12989) instead expands the K
independent marginals into a best-first draft tree under a node budget B, letting
the target verify many continuations in one forward pass.

This module implements DDTree Algorithm 1 (greedy / temperature 0). Heap
expansion AND tree finalize are fused into a single Triton kernel (one
persistent CTA per request) so the parent/token/depth row never lands in
global memory between the two phases. All outputs are statically shaped for
fixed ``(B, K, V)`` so the whole routine captures cleanly into an outer
CUDA graph.

The output contract matches ``DynamicTreeOpsConverter.build_dynamic_tree``:
``retrieve_index`` (identity, root at index 0), ``retrieve_next_token`` (first-child
pointer), ``retrieve_next_sibling`` (next-sibling pointer), ``positions`` (= depth),
a bit-packed ancestor mask, and the flat per-node draft tokens ``[num_gens, B]``.
"""

from typing import Dict

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import nvtx_range

# Sentinel score for inactive / invalid frontier slots. Large-negative but finite
# so arithmetic (score + log q) never produces NaN.
_NEG_INF = -1.0e30


@triton.jit
def _ddtree_topk_phase1_kernel(
    logits_ptr,  # [G*K, vocab]
    partial_max_ptr,  # [G*K, n_chunks] float32
    partial_sumexp_ptr,  # [G*K, n_chunks] float32
    partial_top_ptr,  # [G*K, n_chunks, V] int64 (packed)
    vocab: tl.int32,
    n_chunks: tl.int32,
    V: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
):
    """Phase 1: per-chunk reduction. Grid ``(G*K, n_chunks)``.

    Each CTA owns one (g, k, chunk) tile of the vocab and emits:
      * ``partial_max[gk, c]`` — chunk's max logit (for online LSE).
      * ``partial_sumexp[gk, c]`` — ``sum(exp(x - chunk_max))`` over the chunk.
      * ``partial_top[gk, c, :V]`` — packed top-V of the chunk (matching the
        old kernel's int64 layout: high bit XOR'd so signed-int descending
        order = descending logit, mid-32 = float-as-monotonic-uint32, low-32 =
        ``~vocab_id`` for tiebreak by smallest id).

    With grid ``(G*K, n_chunks)`` and ``BLOCK_VOCAB`` tuned to keep
    ``n_chunks`` large (e.g. ``BLOCK_VOCAB=4096``, vocab≈152k → n_chunks≈38),
    BS=1 fills ~600 CTAs vs the old ``G*K=16``-CTA serial-stream pass.
    """
    gk = tl.program_id(0)
    c = tl.program_id(1)

    NEG_INF_F32 = float("-inf")
    NEG_PACK = -0x8000000000000000  # int64 most-negative

    base = gk * vocab
    v_arange = tl.arange(0, BLOCK_VOCAB)
    v_off = c * BLOCK_VOCAB + v_arange
    v_mask = v_off < vocab

    x = tl.load(logits_ptr + base + v_off, mask=v_mask, other=NEG_INF_F32).to(tl.float32)

    # ---- Per-chunk max and sum_exp (relative to chunk_max) ----
    chunk_max = tl.max(x, axis=0)
    e = tl.exp(x - chunk_max)
    e = tl.where(v_mask, e, 0.0)
    chunk_sumexp = tl.sum(e, axis=0)

    # ---- Pack (logit, ~vocab_id) -> signed int64 ----
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 31) & 1
    ord_bits = tl.where(sign == 0, bits ^ 0x80000000, ~bits)
    tie_low = (~v_off.to(tl.uint32)).to(tl.uint64)
    packed_u = (ord_bits.to(tl.uint64) << 32) | tie_low
    packed_signed = (packed_u ^ 0x8000000000000000).to(tl.int64, bitcast=True)
    packed_signed = tl.where(v_mask, packed_signed, NEG_PACK)

    # ---- Top-V via V sequential mask-max reductions over BLOCK_VOCAB ----
    chunk_top = tl.full((V,), NEG_PACK, tl.int64)
    v_idx = tl.arange(0, V)
    ps = packed_signed
    for v in tl.static_range(V):
        cmax = tl.max(ps, axis=0)
        chunk_top = tl.where(v_idx == v, cmax, chunk_top)
        ps = tl.where(ps == cmax, NEG_PACK, ps)

    # ---- Store partials ----
    pm_off = gk * n_chunks + c
    tl.store(partial_max_ptr + pm_off, chunk_max)
    tl.store(partial_sumexp_ptr + pm_off, chunk_sumexp)

    pt_off = (gk * n_chunks + c) * V + v_idx
    tl.store(partial_top_ptr + pt_off, chunk_top)


@triton.jit
def _ddtree_topk_phase2_kernel(
    partial_max_ptr,  # [G*K, n_chunks] float32
    partial_sumexp_ptr,  # [G*K, n_chunks] float32
    partial_top_ptr,  # [G*K, n_chunks, V] int64 (packed)
    d2t_ptr,  # [vocab] int64 (or null if not used)
    topk_log_vals_ptr,  # [G*K, V] float32
    topk_ids_ptr,  # [G*K, V] int64
    n_chunks: tl.int32,
    V: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,  # next_pow2(n_chunks)
    BLOCK_MERGE: tl.constexpr,  # next_pow2(n_chunks * V)
    HAS_D2T: tl.constexpr,
):
    """Phase 2: merge ``n_chunks`` partials. Grid ``(G*K,)``.

    Per (g, k):
      1. Combine ``(chunk_max, chunk_sumexp)`` across chunks via a single
         parallel log-sum-exp:
         ``log_norm = global_max + log(sum_c sumexp_c * exp(chunk_max_c - global_max))``.
      2. Bitonic-style top-V over the ``n_chunks * V`` packed candidates via
         V sequential mask-max reductions over the ``BLOCK_MERGE`` register tile.
      3. Unpack to ``(log_prob, vocab_id)``, optionally apply ``d2t`` mapping,
         and write to outputs.

    Per-CTA work is tiny (≤512-element register tile, V≤16); the launch is
    bounded by the small-grid latency, but the wave is fully hidden under
    Phase 1's tail since the two phases run sequentially on the same stream.
    """
    gk = tl.program_id(0)

    NEG_INF_F32 = float("-inf")
    NEG_PACK = -0x8000000000000000

    # ---- Combined log-sum-exp over chunks ----
    c_off = tl.arange(0, BLOCK_CHUNKS)
    c_mask = c_off < n_chunks
    pmax = tl.load(partial_max_ptr + gk * n_chunks + c_off, mask=c_mask, other=NEG_INF_F32)
    psum = tl.load(partial_sumexp_ptr + gk * n_chunks + c_off, mask=c_mask, other=0.0)
    global_max = tl.max(pmax, axis=0)
    rescaled = psum * tl.exp(pmax - global_max)
    rescaled = tl.where(c_mask, rescaled, 0.0)
    total_sumexp = tl.sum(rescaled, axis=0)
    log_norm = global_max + tl.log(total_sumexp)

    # ---- Load all n_chunks*V packed partials into one tile ----
    m_off = tl.arange(0, BLOCK_MERGE)
    m_mask = m_off < (n_chunks * V)
    merge_base = gk * n_chunks * V
    merged = tl.load(partial_top_ptr + merge_base + m_off, mask=m_mask, other=NEG_PACK)

    # ---- Top-V via V mask-max reductions over the merge tile ----
    final_top = tl.full((V,), NEG_PACK, tl.int64)
    v_idx = tl.arange(0, V)
    ps = merged
    for v in tl.static_range(V):
        cmax = tl.max(ps, axis=0)
        final_top = tl.where(v_idx == v, cmax, final_top)
        ps = tl.where(ps == cmax, NEG_PACK, ps)

    # ---- Unpack final_top into (log_val, vocab_id). Guard against the
    # degenerate case where vocab < V (some lanes still hold NEG_PACK, whose
    # unpack would produce NaN and poison downstream argmax). ----
    valid_slot = final_top != NEG_PACK
    top_u = final_top.to(tl.uint64, bitcast=True) ^ 0x8000000000000000
    top_ord = (top_u >> 32).to(tl.uint32)
    top_tie = (top_u & 0xFFFFFFFF).to(tl.uint32)
    top_vid = (~top_tie).to(tl.int32)
    sm = (top_ord >> 31) & 1
    rec_bits = tl.where(sm == 1, top_ord ^ 0x80000000, ~top_ord)
    rec_logit = rec_bits.to(tl.float32, bitcast=True)
    log_prob = rec_logit - log_norm

    out_id_i64 = top_vid.to(tl.int64)
    if HAS_D2T:
        safe_vid = tl.where(valid_slot, top_vid, 0)
        d2t_offset = tl.load(d2t_ptr + safe_vid, mask=valid_slot, other=0).to(tl.int64)
        out_id_i64 = out_id_i64 + d2t_offset

    out_off = gk * V + v_idx
    tl.store(topk_log_vals_ptr + out_off, log_prob, mask=valid_slot)
    tl.store(topk_ids_ptr + out_off, out_id_i64, mask=valid_slot)


def _ddtree_topk(gen_logits: torch.Tensor, d2t: torch.Tensor | None, V: int):
    """Fused log-softmax + top-V over [G, K, vocab] with optional d2t mapping.

    Replaces ``F.log_softmax(gen_logits.float(), dim=-1) + torch.topk + d2t``
    with a 2-phase Triton pipeline:

      * Phase 1 (grid ``(G*K, n_chunks)``): one CTA per (g, k, vocab-chunk) does
        per-chunk max / sum_exp / top-V. With ``BLOCK_VOCAB=4096`` and
        vocab≈152k this produces ~38 chunks/row, fully saturating the GPU
        (~600 CTAs at BS=1 vs the previous 16-CTA serial-stream pass that left
        a B300 at ~1.5% wave occupancy).
      * Phase 2 (grid ``(G*K,)``): merge per-chunk partials → final top-V and
        log-prob via one parallel LSE + one V-step mask-max reduction over a
        ≤512-element register tile.

    Net effect on Qwen3-8B BS=1 b=16 (B300, vocab=151936, V=8): the topk
    path drops from ~142 µs to ~19 µs (phase-1 ~15.6 µs + phase-2 ~3.3 µs) —
    the dominant fixable contributor to the DDTree-vs-DFlash regression.
    """
    G, K, vocab = gen_logits.shape
    device = gen_logits.device
    topk_log_vals = torch.empty((G, K, V), dtype=torch.float32, device=device)
    topk_ids = torch.empty((G, K, V), dtype=torch.int64, device=device)

    # ``BLOCK_VOCAB`` chosen small enough that ``n_chunks`` saturates the GPU
    # at low batch (target ≥ ~600 phase-1 CTAs at BS=1 on B300, i.e.
    # ``G*K * n_chunks >= 600``). For tiny vocab (unit tests, vocab<=4096)
    # clamp so every CTA has at least one valid element.
    BLOCK_VOCAB = min(4096, max(1024, triton.next_power_of_2(vocab)))
    n_chunks = (vocab + BLOCK_VOCAB - 1) // BLOCK_VOCAB

    # Phase-1 partials. Small per-call workspace (≤ a few MB at typical
    # batch / vocab sizes). The allocator hits cache after warmup so this
    # adds ~3 dispatches but no GPU work; CUDA-graph-safe under the standard
    # caching allocator.
    GK = G * K
    partial_top = torch.empty((GK, n_chunks, V), dtype=torch.int64, device=device)
    partial_max = torch.empty((GK, n_chunks), dtype=torch.float32, device=device)
    partial_sumexp = torch.empty((GK, n_chunks), dtype=torch.float32, device=device)

    has_d2t = d2t is not None
    d2t_arg = d2t if has_d2t else gen_logits  # dummy ptr; never dereffed when HAS_D2T=False

    gen_logits_2d = gen_logits.contiguous().view(GK, vocab)
    _ddtree_topk_phase1_kernel[(GK, n_chunks)](
        gen_logits_2d,
        partial_max,
        partial_sumexp,
        partial_top,
        vocab=vocab,
        n_chunks=n_chunks,
        V=V,
        BLOCK_VOCAB=BLOCK_VOCAB,
        num_warps=4,
        num_stages=1,
    )

    BLOCK_CHUNKS = max(16, triton.next_power_of_2(n_chunks))
    BLOCK_MERGE = max(16, triton.next_power_of_2(n_chunks * V))

    _ddtree_topk_phase2_kernel[(GK,)](
        partial_max,
        partial_sumexp,
        partial_top,
        d2t_arg,
        topk_log_vals.view(GK, V),
        topk_ids.view(GK, V),
        n_chunks=n_chunks,
        V=V,
        BLOCK_CHUNKS=BLOCK_CHUNKS,
        BLOCK_MERGE=BLOCK_MERGE,
        HAS_D2T=has_d2t,
        num_warps=1,
        num_stages=1,
    )
    return topk_log_vals, topk_ids


def _ddtree_pack_retrieve(
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    slot_ids: torch.Tensor,
    count: int,
    out: torch.Tensor,
    stacked_src: torch.Tensor | None = None,
):
    """Pack ``retrieve_*[slot_ids[:count]]`` into a contiguous ``[count, n_dt, 3]``
    int32 tile for the verify op.

    When ``stacked_src`` is provided (the ``[3, S, n_dt]`` view aliasing the
    three retrieve fields), do one ``index_select(dim=1, ids)`` plus one
    permuted copy into ``out`` (2 dispatches). Otherwise fall back to 3
    separate fancy gathers.
    """
    if count == 0:
        return out[:0]
    ids = slot_ids[:count]
    if stacked_src is not None:
        gathered = stacked_src.index_select(1, ids)  # [3, count, n_dt]
        out[:count].copy_(gathered.permute(1, 2, 0))
    else:
        out[:count, :, 0] = retrieve_index.index_select(0, ids)
        out[:count, :, 1] = retrieve_next_token.index_select(0, ids)
        out[:count, :, 2] = retrieve_next_sibling.index_select(0, ids)
    return out[:count]


def _ddtree_slot_scatter(
    slot_storage, tree_dict, slot_ids: torch.Tensor, num_gens: int, dummy_slot_id: int
):
    """Scatter per-request DDTree build outputs into ``slot_storage[slot_ids]``.

    Both the slot storage and the build-side finalize buffers stack the 4
    same-shape ``[*, n_dt]`` int32 fields (positions, retrieve_index,
    retrieve_next_token, retrieve_next_sibling) onto a leading axis, so one
    ``index_copy_(dim=1, ...)`` scatters all four at once. Total: 1 stacked
    scatter + 1 packed_mask scatter + 1 ``index_fill_`` + 1 dummy-slot reset
    (the dummy reset keeps ``has_tree[dummy_slot]`` False for CUDA-graph
    dummy iters).
    """
    if num_gens == 0:
        return
    ids = slot_ids[:num_gens]
    slot_storage.packed_mask.index_copy_(0, ids, tree_dict["packed_mask"])
    slot_storage._slot_stack.index_copy_(1, ids, tree_dict["_stack"])
    slot_storage.has_tree.index_fill_(0, ids, True)
    slot_storage.has_tree.narrow(0, dummy_slot_id, 1).fill_(False)


def _ddtree_build_candidates(
    target_predict: torch.Tensor,
    candidates: torch.Tensor,
    tree_valid: torch.Tensor,
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    has_tree: torch.Tensor,
    slot_ids: torch.Tensor,
    num_contexts: int,
    num_gens: int,
    N: int,
):
    """Stage the verify-time inputs (target_predict, candidates, tree_valid).

    4 dispatches with no temp allocs:
      * ``copy_`` absorbs the int64 → int32 cast on the source slice.
      * ``candidates[:, 0]`` is filled from the same int64 slice via ``copy_``
        rather than reading back ``tp_view`` (saves a redundant int32 write
        path; PyTorch fuses the gather into the destination dtype).
    """
    if num_gens == 0:
        return
    G = num_gens
    tp_view = target_predict[:G]
    # Match target_tokens[(num_contexts + g) * N + n]. In practice
    # num_contexts == 0 during the gen path.
    start = num_contexts * N
    src_2d = target_tokens[start : start + G * N].view(G, N)
    tp_view.copy_(src_2d)  # int64 -> int32, no temp tensor
    candidates[:G, 0].copy_(src_2d[:, 0])  # int64 -> int32, no temp tensor
    candidates[:G, 1:N].copy_(draft_tokens[:G])
    torch.index_select(has_tree, 0, slot_ids[:G], out=tree_valid[:G])


def _ddtree_pre_init(
    accepted_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    tree_accepted_indices: torch.Tensor,
    batch_size: int,
    max_path_len: int,
    max_draft_len: int,
):
    """Pre-init the 3 verify-output buffers for ``_sample_and_accept_ddtree``
    on the leading ``batch_size`` rows. Slices match column masking even when
    callers pass wider scratch buffers.
    """
    if batch_size == 0:
        return
    accepted_tokens[:batch_size, :max_path_len].zero_()
    num_accepted[:batch_size].fill_(1)
    tree_accepted_indices[:batch_size, :max_draft_len].fill_(-1)


def _ddtree_post_scatter(
    accepted_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    tree_accepted_indices: torch.Tensor,
    tree_accept_path: torch.Tensor,
    accept_token: torch.Tensor,
    accept_token_num: torch.Tensor,
    accept_index: torch.Tensor,
    num_contexts: int,
    num_gens: int,
    max_path_len: int,
    max_draft_len: int,
):
    """Scatter verify outputs into the per-batch destination buffers.

    4 dispatches with no temp allocs:
      * ``copy_`` instead of ``.to(int64)`` — writes through the destination's
        int64 view and skips a temp allocation.
      * ``add_(1)`` / ``sub_(1)`` are fused with the destination write by first
        copying the source slice and then adjusting in-place.
    """
    if num_gens == 0:
        return
    P = max_path_len
    K = max_draft_len
    end = num_contexts + num_gens
    accepted_tokens[num_contexts:end].copy_(accept_token[:num_gens])
    # num_accepted slice = accept_token_num[:G] + 1 — copy then in-place add.
    num_accepted[num_contexts:end].copy_(accept_token_num[:num_gens]).add_(1)
    # tree_accepted_indices slice = accept_index[:G, 1:K+1] - 1 — copy then sub_.
    tree_accepted_indices[num_contexts:end, :K].copy_(accept_index[:num_gens, 1 : K + 1]).sub_(1)
    # int64 dest absorbs the int32 cast via copy_ (no temp tensor).
    tree_accept_path[num_contexts:end, :P].copy_(accept_index[:num_gens, :P])


@triton.jit
def _ddtree_build_kernel(
    topk_log_vals_ptr,  # [G, K, V] float32
    topk_ids_ptr,  # [G, K, V] int64
    draft_tokens_ptr,  # [G, B] int32
    retrieve_index_ptr,  # [G, n_dt] int32
    retrieve_next_token_ptr,  # [G, n_dt] int32
    retrieve_next_sibling_ptr,  # [G, n_dt] int32
    positions_ptr,  # [G, n_dt] int32
    packed_mask_ptr,  # [G, n_dt, n_words] int32
    B: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    n_words: tl.constexpr,
    BLOCK_CAP: tl.constexpr,  # next_pow2(1 + 2*B)
    BLOCK_DT: tl.constexpr,  # next_pow2(n_dt)
    BLOCK_W: tl.constexpr,  # next_pow2(n_words)
):
    """Single-launch DDTree build: heap expansion + finalize fused into one CTA per request.

    The original pipeline ran two Triton kernels back-to-back:

      1) heap_kernel  grid (G,)      ->  writes [G, n_dt] int64 parent/token/depth scratch
      2) finalize     grid (G, n_dt) ->  reads scratch, emits first-child / next-sibling /
                                         positions / packed_mask / draft_tokens

    With G typically 1-8 (BS<=batch), the launches sit at <0.1% wave occupancy and
    most of the cost is host dispatch + the int64 round-trip. Folding both into a
    single CTA keeps parent/token/depth row in registers ([BLOCK_DT] register tiles)
    and runs the finalize work as 2D-broadcast register reductions. Net effect: 1
    launch, 0 intermediate global stores, all heap+finalize work parallelized
    over BLOCK_DT lanes within the same warp.
    """
    g = tl.program_id(0)

    NEG_INF = -1.0e30
    n_dt = B + 1
    cap_arange = tl.arange(0, BLOCK_CAP)
    dt_arange = tl.arange(0, BLOCK_DT)
    dt_mask = dt_arange < n_dt

    # Strides (row-major contiguous):
    #   topk_log_vals[g, k, v] -> ptr + g*(K*V) + k*V + v  (float32)
    #   topk_ids     [g, k, v] -> ptr + g*(K*V) + k*V + v  (int64)
    base_kv = g * K * V

    # ---- Heap frontier registers (slot 0 = root child; rest = -inf) ----
    root_log = tl.load(topk_log_vals_ptr + base_kv + 0).to(tl.float32)
    root_id = tl.load(topk_ids_ptr + base_kv + 0)
    f_score = tl.where(cap_arange == 0, root_log, tl.full((BLOCK_CAP,), NEG_INF, tl.float32))
    f_depth = tl.where(
        cap_arange == 0, tl.full((BLOCK_CAP,), 1, tl.int32), tl.zeros((BLOCK_CAP,), tl.int32)
    )
    f_rank = tl.zeros((BLOCK_CAP,), tl.int32)
    f_parent = tl.zeros((BLOCK_CAP,), tl.int32)
    f_token = tl.where(cap_arange == 0, root_id, tl.zeros((BLOCK_CAP,), root_id.dtype))

    # ---- Output rows in registers (replaces the old global parent/token/depth scratch) ----
    # Root: parent=-1, token=0, depth=0. Invalid slots also default to parent=-1.
    parent_row = tl.full((BLOCK_DT,), -1, tl.int32)
    token_row = tl.zeros((BLOCK_DT,), root_id.dtype)
    depth_row = tl.zeros((BLOCK_DT,), tl.int32)

    for t in tl.static_range(B):
        node_idx = t + 1

        # Argmax over the frontier; tl.argmax breaks ties by lowest index, which
        # matches torch.argmax's semantics on the same data.
        idx = tl.argmax(f_score, axis=0)
        sel = cap_arange == idx
        pscore = tl.sum(tl.where(sel, f_score, 0.0), axis=0)
        pdepth = tl.sum(tl.where(sel, f_depth, 0), axis=0)
        prank = tl.sum(tl.where(sel, f_rank, 0), axis=0)
        pparent = tl.sum(tl.where(sel, f_parent, 0), axis=0)
        ptoken = tl.sum(tl.where(sel, f_token, tl.zeros_like(f_token)), axis=0)

        pvalid = pscore > NEG_INF
        out_parent_val = tl.where(pvalid, pparent.to(tl.int32), tl.full((), -1, tl.int32))
        out_token_val = tl.where(pvalid, ptoken, tl.zeros_like(ptoken))
        out_depth_val = tl.where(pvalid, pdepth.to(tl.int32), tl.zeros((), tl.int32))

        # Update parent/token/depth row at slot node_idx (in registers).
        slot_sel = dt_arange == node_idx
        parent_row = tl.where(slot_sel, out_parent_val, parent_row)
        token_row = tl.where(slot_sel, out_token_val, token_row)
        depth_row = tl.where(slot_sel, out_depth_val, depth_row)

        # Mark popped slot as -inf so it can never be re-selected.
        f_score = tl.where(sel, tl.full((BLOCK_CAP,), NEG_INF, tl.float32), f_score)

        # ---- sibling expansion ----
        pos_last = tl.maximum(pdepth - 1, 0)
        sib_rank = prank + 1
        sib_valid = pvalid & (sib_rank < V)
        rank_c = tl.minimum(prank, V - 1)
        sib_rank_c = tl.minimum(sib_rank, V - 1)
        sib_off_old = base_kv + pos_last * V + rank_c
        sib_off_new = base_kv + pos_last * V + sib_rank_c
        old_val = tl.load(topk_log_vals_ptr + sib_off_old).to(tl.float32)
        new_val = tl.load(topk_log_vals_ptr + sib_off_new).to(tl.float32)
        sib_score = pscore - old_val + new_val
        sib_token = tl.load(topk_ids_ptr + sib_off_new)

        s_sib = 1 + 2 * t
        sib_sel = cap_arange == s_sib
        f_score = tl.where(
            sib_sel,
            tl.where(sib_valid, sib_score, tl.full((BLOCK_CAP,), NEG_INF, tl.float32)),
            f_score,
        )
        f_depth = tl.where(sib_sel, pdepth.to(f_depth.dtype), f_depth)
        f_rank = tl.where(sib_sel, sib_rank_c.to(f_rank.dtype), f_rank)
        f_parent = tl.where(sib_sel, pparent.to(f_parent.dtype), f_parent)
        f_token = tl.where(sib_sel, sib_token.to(f_token.dtype), f_token)

        # ---- first-child expansion ----
        child_valid = pvalid & (pdepth < K)
        pos_child = tl.minimum(pdepth, K - 1)
        child_off = base_kv + pos_child * V + 0
        child_val = tl.load(topk_log_vals_ptr + child_off).to(tl.float32)
        child_score = pscore + child_val
        child_token = tl.load(topk_ids_ptr + child_off)

        s_child = 2 + 2 * t
        child_sel = cap_arange == s_child
        f_score = tl.where(
            child_sel,
            tl.where(child_valid, child_score, tl.full((BLOCK_CAP,), NEG_INF, tl.float32)),
            f_score,
        )
        f_depth = tl.where(
            child_sel, tl.full((BLOCK_CAP,), 1, f_depth.dtype) + pdepth.to(f_depth.dtype), f_depth
        )
        f_rank = tl.where(child_sel, tl.zeros((BLOCK_CAP,), f_rank.dtype), f_rank)
        f_parent = tl.where(child_sel, tl.full((BLOCK_CAP,), node_idx, f_parent.dtype), f_parent)
        f_token = tl.where(child_sel, child_token.to(f_token.dtype), f_token)

    # =========================================================================
    # Finalize phase: derive first-child / next-sibling / packed_mask from
    # parent_row (register tile). All work is parallel over BLOCK_DT lanes.
    # =========================================================================

    # valid_row[i]: true iff slot i holds a real tree node (i==0 or parent>=0).
    # parent_row is -1-initialized for slots >= n_dt and only slots 1..n_dt-1
    # are written by the heap loop, so valid_row already implies dt_mask.
    valid_row = (dt_arange == 0) | (parent_row >= 0)

    out_off = g * n_dt + dt_arange

    # positions = depth_row, retrieve_index = identity. draft_tokens[g, i-1] = token_row[i].
    tl.store(positions_ptr + out_off, depth_row, mask=dt_mask)
    tl.store(retrieve_index_ptr + out_off, dt_arange.to(tl.int32), mask=dt_mask)
    tl.store(
        draft_tokens_ptr + g * (n_dt - 1) + (dt_arange - 1),
        token_row.to(tl.int32),
        mask=dt_mask & (dt_arange >= 1),
    )

    # next-sibling[i] = smallest j > i with parent_row[j] == parent_row[i] (and >= 0).
    # first-child[i]  = smallest j with parent_row[j] == i.
    # 2D-broadcast register reductions over BLOCK_DT * BLOCK_DT (1024 i32 at typical sizes).
    BIG = n_dt
    j_grid = dt_arange[None, :]  # [1, BLOCK_DT]
    i_grid = dt_arange[:, None]  # [BLOCK_DT, 1]
    pj = parent_row[None, :]  # [1, BLOCK_DT] — parent of slot j
    pi = parent_row[:, None]  # [BLOCK_DT, 1] — parent of slot i

    valid_j_grid = ((j_grid == 0) | (pj >= 0)) & (j_grid < n_dt)

    sib_cand = valid_j_grid & (pj == pi) & (pi >= 0) & (j_grid > i_grid)
    sib_min = tl.min(tl.where(sib_cand, j_grid, tl.full(sib_cand.shape, BIG, tl.int32)), axis=1)
    sib_out = tl.where(valid_row & (sib_min < BIG), sib_min, tl.full((BLOCK_DT,), -1, tl.int32))
    tl.store(retrieve_next_sibling_ptr + out_off, sib_out, mask=dt_mask)

    fc_cand = valid_j_grid & (pj == i_grid)
    fc_min = tl.min(tl.where(fc_cand, j_grid, tl.full(fc_cand.shape, BIG, tl.int32)), axis=1)
    fc_out = tl.where(valid_row & (fc_min < BIG), fc_min, tl.full((BLOCK_DT,), -1, tl.int32))
    tl.store(retrieve_next_token_ptr + out_off, fc_out, mask=dt_mask)

    # ---- bit-packed ancestor mask ----
    # mask[i, w]: bit (c%32) of word (c//32) is set iff i attends to col c
    # (where c is i itself or any ancestor of i along the parent chain).
    w_arange = tl.arange(0, BLOCK_W)
    w_mask = w_arange < n_words

    # Self-bit per row: bit (i%32) of word (i//32) -- only for valid rows.
    self_word_2d = i_grid // 32  # [BLOCK_DT, 1]
    self_bit_2d = i_grid % 32  # [BLOCK_DT, 1]
    self_w_2d = w_arange[None, :]  # [1, BLOCK_W]
    self_set = (self_w_2d == self_word_2d).to(tl.int32) * (1 << self_bit_2d)
    valid_i_2d = valid_row[:, None]  # [BLOCK_DT, 1]
    mask_words = tl.where(valid_i_2d, self_set, tl.zeros((BLOCK_DT, BLOCK_W), tl.int32))

    # Walk parent chain at most K steps. cur[i] = current ancestor of row i.
    cur = dt_arange.to(tl.int32)  # [BLOCK_DT]
    for _ in tl.static_range(K):
        # Gather parent_row[cur[i]] in registers via where-broadcast:
        # cur_eq[i, j] = (j == cur[i]); for each i exactly one j matches (or none if cur[i] >= n_dt).
        cur_eq = j_grid == cur[:, None]
        cur_parent = tl.sum(tl.where(cur_eq, pj, tl.zeros((BLOCK_DT, BLOCK_DT), tl.int32)), axis=1)
        has_parent = cur_parent >= 0
        safe_parent = tl.maximum(cur_parent, 0)
        p_word = (safe_parent // 32)[:, None]  # [BLOCK_DT, 1]
        p_bit = (safe_parent % 32)[:, None]  # [BLOCK_DT, 1]
        bit_set = (w_arange[None, :] == p_word).to(tl.int32) * (1 << p_bit)
        bit_set = tl.where(has_parent[:, None], bit_set, tl.zeros((BLOCK_DT, BLOCK_W), tl.int32))
        mask_words = mask_words | bit_set
        cur = tl.where(has_parent, safe_parent, cur)

    pm_off = g * n_dt * n_words + i_grid * n_words + w_arange[None, :]
    pm_store_mask = dt_mask[:, None] & w_mask[None, :]
    tl.store(packed_mask_ptr + pm_off, mask_words, mask=pm_store_mask)


def _ddtree_build_triton(
    topk_log_vals: torch.Tensor,
    topk_ids: torch.Tensor,
    B: int,
    K: int,
    V: int,
    buffers: dict | None = None,
):
    """Single-launch heap-expand + finalize. Replaces the previous two-kernel
    pipeline (`_ddtree_heap_expand` + `_ddtree_finalize_triton`).

    Returns ``(draft_tokens, retrieve_index, retrieve_next_token,
    retrieve_next_sibling, positions, packed_mask)`` — all int32, on the same
    device as the inputs. If ``buffers`` is provided it must contain the same
    keys (each sliced to leading G rows); using persistent buffers saves 6
    ``torch.empty`` dispatches per step and keeps storage stable across
    iterations for CUDA-graph capture.
    """
    G = topk_log_vals.shape[0]
    n_dt = B + 1
    n_words = (n_dt + 31) // 32
    device = topk_log_vals.device

    if buffers is None:
        draft_tokens = torch.empty((G, B), dtype=torch.int32, device=device)
        retrieve_index = torch.empty((G, n_dt), dtype=torch.int32, device=device)
        retrieve_next_token = torch.empty((G, n_dt), dtype=torch.int32, device=device)
        retrieve_next_sibling = torch.empty((G, n_dt), dtype=torch.int32, device=device)
        positions = torch.empty((G, n_dt), dtype=torch.int32, device=device)
        packed_mask = torch.empty((G, n_dt, n_words), dtype=torch.int32, device=device)
    else:
        draft_tokens = buffers["draft_tokens"]
        retrieve_index = buffers["retrieve_index"]
        retrieve_next_token = buffers["retrieve_next_token"]
        retrieve_next_sibling = buffers["retrieve_next_sibling"]
        positions = buffers["positions"]
        packed_mask = buffers["packed_mask"]

    cap = 1 + 2 * B
    BLOCK_CAP = max(16, triton.next_power_of_2(cap))
    BLOCK_DT = max(16, triton.next_power_of_2(n_dt))
    BLOCK_W = max(1, triton.next_power_of_2(n_words))

    log_vals_f32 = topk_log_vals.to(torch.float32, copy=False).contiguous()
    ids_i64 = topk_ids.to(torch.int64, copy=False).contiguous()

    _ddtree_build_kernel[(G,)](
        log_vals_f32,
        ids_i64,
        draft_tokens,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        positions,
        packed_mask,
        B=B,
        K=K,
        V=V,
        n_words=n_words,
        BLOCK_CAP=BLOCK_CAP,
        BLOCK_DT=BLOCK_DT,
        BLOCK_W=BLOCK_W,
        num_warps=1,
        num_stages=1,
    )
    return (
        draft_tokens,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        positions,
        packed_mask,
    )


@torch.no_grad()
@nvtx_range("ddtree_build_ddtree")
def build_ddtree(
    topk_log_vals: torch.Tensor,
    topk_ids: torch.Tensor,
    node_budget: int,
    max_draft_len: int,
    vocab_fanout: int,
    finalize_buffers: dict | None = None,
) -> Dict[str, torch.Tensor]:
    """Build a DDTree draft tree from per-position top-k log marginals.

    Best-first expansion over rank-tuples: the score of a prefix is the sum of the
    per-position log-probabilities of its chosen ranks. We pop the top ``B`` prefixes;
    each pop pushes (a) the next sibling (increment the last rank) and (b) the first
    child (append rank 0 at the next position). The popped prefixes are prefix-closed,
    so they form a valid tree rooted at the bonus token (node index 0).

    All shapes are static and there is no ``.item()`` / host sync, so the whole
    routine is CUDA-graph safe given fixed ``B``, ``max_draft_len`` and ``vocab_fanout``.

    Args:
        topk_log_vals: ``[G, K, V]`` descending log-probabilities of the top-``V``
            tokens at each of the ``K`` draft positions, per generation request.
        topk_ids: ``[G, K, V]`` token ids matching ``topk_log_vals`` (already mapped
            through ``d2t`` if the draft model uses a reduced vocab).
        node_budget: ``B`` — number of tree nodes excluding the root.
        max_draft_len: ``K`` — number of draft positions (max tree depth).
        vocab_fanout: ``V`` — per-position branching cap (== ``topk_*`` last dim).

    Returns:
        dict with (all on the same device as inputs):
            ``draft_tokens``        ``[G, B]`` int32 — token id per node 1..B.
            ``retrieve_index``      ``[G, B+1]`` int32 — identity (root at 0).
            ``retrieve_next_token`` ``[G, B+1]`` int32 — first-child pointer (-1=none).
            ``retrieve_next_sibling````[G, B+1]`` int32 — next-sibling pointer (-1=none).
            ``positions``           ``[G, B+1]`` int32 — node depth (root=0).
            ``packed_mask``         ``[G, B+1, ceil((B+1)/32)]`` int32 — bit-packed
                                    ancestor mask (bit ``c%32`` of word ``c//32``
                                    set iff row attends to col ``c``). Layout
                                    matches ``SpecTreeManager.spec_dec_packed_mask``.
    """
    K = max_draft_len
    V = vocab_fanout
    B = node_budget

    # Single-launch fused build: heap expansion + finalize fused into one Triton
    # kernel with grid (G,). Each persistent CTA per request keeps the
    # parent/token/depth row in registers (BLOCK_DT lanes) and runs the
    # finalize work as 2D-broadcast register reductions. Replaces the previous
    # two-kernel pipeline that wrote/read [G, n_dt] int64 scratch tensors and
    # paid one extra launch boundary (visible at BS=1 where each launch is
    # under-occupied).
    with nvtx_range("ddtree_build"):
        (
            draft_tokens,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            positions,
            packed_mask,
        ) = _ddtree_build_triton(topk_log_vals, topk_ids, B, K, V, buffers=finalize_buffers)

    return {
        "draft_tokens": draft_tokens,
        "retrieve_index": retrieve_index,
        "retrieve_next_token": retrieve_next_token,
        "retrieve_next_sibling": retrieve_next_sibling,
        "positions": positions,
        "packed_mask": packed_mask,
    }
