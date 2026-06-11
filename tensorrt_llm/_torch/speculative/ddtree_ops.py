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

This module implements DDTree Algorithm 1 (greedy / temperature 0). The B-step
heap expansion is a hand-written Triton kernel (one persistent CTA per request,
frontier in register-tile of size ``next_pow2(1+2B)``); the downstream tree
finalize (first-child/next-sibling pointers + bit-packed ancestor mask) stays
in torch under ``torch.compile`` since it's already a handful of shape-static
broadcasts. All outputs are statically shaped for fixed ``(B, K, V)`` so the
whole routine captures cleanly into an outer CUDA graph.

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
def _ddtree_topk_kernel(
    logits_ptr,  # [G, K, vocab] (float32 / float16 / bfloat16)
    d2t_ptr,  # [vocab] int64 (or null if not used)
    topk_log_vals_ptr,  # [G, K, V] float32 -- log-prob
    topk_ids_ptr,  # [G, K, V] int64 -- token ids (post-d2t if HAS_D2T)
    vocab: tl.int32,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,  # ddtree_max_topk; small (<=16 in practice)
    BLOCK_VOCAB: tl.constexpr,
    BLOCK_TOP: tl.constexpr,  # next_pow2(V) -- output staging tile
    HAS_D2T: tl.constexpr,
):
    """One CTA per (g, k) row. Single streaming pass over the vocab in chunks
    of ``BLOCK_VOCAB``: online log-softmax (Welford-style max + sum_exp
    rescale) plus a running top-V tracked as ``V`` sequential mask-max
    reductions per chunk.

    Top-V packing layout (signed int64, high → low):
      bit 63    : XOR'd so signed-int order matches descending logit order.
      bits 62..32: float-as-monotonic-uint32 of the logit.
      bits 31..0 : ``~vocab_id`` so smaller id wins ties under descending sort.

    Per chunk: V mask-max reductions extract the chunk's top-V into the upper
    half of a 2V register tile; the lower half holds the running top-V.
    A ``tl.topk`` over 2V (cheap: bitonic sort over a tiny tile) keeps the
    overall top-V. This avoids the bitonic-sort-over-BLOCK_VOCAB cost of the
    original two-pass kernel.
    """
    g = tl.program_id(0)
    k = tl.program_id(1)

    NEG_INF_F32 = float("-inf")
    NEG_PACK = -0x8000000000000000  # int64 most-negative
    base = g * K * vocab + k * vocab
    v_arange = tl.arange(0, BLOCK_VOCAB)
    top_arange = tl.arange(0, BLOCK_TOP)
    is_run_slot = top_arange < V

    # Running state: log-softmax (running max + scaled sum_exp) and top-V
    # packed-int64 register tile. Lanes [0..V) hold descending top-V;
    # lanes [V..BLOCK_TOP) are NEG_PACK pad.
    running_max = tl.full((), NEG_INF_F32, tl.float32)
    sum_exp = tl.zeros((), tl.float32)
    running_top = tl.full((BLOCK_TOP,), NEG_PACK, tl.int64)

    n_chunks = (vocab + BLOCK_VOCAB - 1) // BLOCK_VOCAB
    for c in range(0, n_chunks):
        v_off = c * BLOCK_VOCAB + v_arange
        v_mask = v_off < vocab
        x = tl.load(logits_ptr + base + v_off, mask=v_mask, other=NEG_INF_F32).to(tl.float32)

        # ---- Online log-softmax recurrence ----
        chunk_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, chunk_max)
        rescale = tl.exp(running_max - new_max)
        e = tl.exp(x - new_max)
        e = tl.where(v_mask, e, 0.0)
        sum_exp = sum_exp * rescale + tl.sum(e, axis=0)
        running_max = new_max

        # ---- Pack (logit, ~vocab_id) -> signed int64; mask oob to NEG_PACK ----
        bits = x.to(tl.uint32, bitcast=True)
        sign = (bits >> 31) & 1
        ord_bits = tl.where(sign == 0, bits ^ 0x80000000, ~bits)
        tie_low = (~v_off.to(tl.uint32)).to(tl.uint64)
        packed_u = (ord_bits.to(tl.uint64) << 32) | tie_low
        packed_signed = (packed_u ^ 0x8000000000000000).to(tl.int64, bitcast=True)
        packed_signed = tl.where(v_mask, packed_signed, NEG_PACK)

        # ---- Extract chunk's top-V via V mask-max reductions, into [V..2V) ----
        chunk_top = tl.full((BLOCK_TOP,), NEG_PACK, tl.int64)
        ps = packed_signed
        for v in tl.static_range(V):
            cmax = tl.max(ps, axis=0)
            chunk_top = tl.where(top_arange == (V + v), cmax, chunk_top)
            ps = tl.where(ps == cmax, NEG_PACK, ps)
        # Merge running_top (lanes 0..V) ∪ chunk_top (lanes V..2V) into a
        # single 2V tile and bitonic-topk back to V.
        merged = tl.where(is_run_slot, running_top, chunk_top)
        new_top_v = tl.topk(merged, V, dim=0)
        # Scatter the V-element result back into the first V lanes.
        running_top = tl.full((BLOCK_TOP,), NEG_PACK, tl.int64)
        for i in tl.static_range(V):
            ri = tl.sum(tl.where(tl.arange(0, V) == i, new_top_v, 0).to(tl.int64), axis=0)
            running_top = tl.where(top_arange == i, ri, running_top)

    log_norm = running_max + tl.log(sum_exp)

    # ---- Unpack running_top (first V slots) to (log_val, id) ----
    top_u = running_top.to(tl.uint64, bitcast=True) ^ 0x8000000000000000
    top_ord = (top_u >> 32).to(tl.uint32)
    top_tie = (top_u & 0xFFFFFFFF).to(tl.uint32)
    top_vid = (~top_tie).to(tl.int32)
    sm = (top_ord >> 31) & 1
    rec_bits = tl.where(sm == 1, top_ord ^ 0x80000000, ~top_ord)
    rec_logit = rec_bits.to(tl.float32, bitcast=True)
    log_prob = rec_logit - log_norm

    out_id_i64 = top_vid.to(tl.int64)
    valid_slot = top_arange < V
    if HAS_D2T:
        safe_vid = tl.where(valid_slot, top_vid, 0)
        d2t_offset = tl.load(d2t_ptr + safe_vid, mask=valid_slot, other=0).to(tl.int64)
        out_id_i64 = out_id_i64 + d2t_offset

    out_base = g * K * V + k * V
    out_off = out_base + top_arange
    tl.store(topk_log_vals_ptr + out_off, log_prob, mask=valid_slot)
    tl.store(topk_ids_ptr + out_off, out_id_i64, mask=valid_slot)


def _ddtree_topk(gen_logits: torch.Tensor, d2t: torch.Tensor | None, V: int):
    """Fused log-softmax + top-V over [G, K, vocab] with optional d2t mapping.

    Replaces ``F.log_softmax(gen_logits.float(), dim=-1) + torch.topk + d2t``
    (3-4 dispatches + a full ``[G, K, vocab]`` fp32 materialization) with a
    single Triton launch. Grid ``(G, K)``; each CTA streams the vocab once.
    """
    G, K, vocab = gen_logits.shape
    device = gen_logits.device
    topk_log_vals = torch.empty((G, K, V), dtype=torch.float32, device=device)
    topk_ids = torch.empty((G, K, V), dtype=torch.int64, device=device)

    # Tuned on B300 for vocab≈152k V≤8: BLOCK_VOCAB=32k + 16 warps fits the
    # whole large-vocab workload in a couple of chunks per CTA and saturates
    # the SM with low CTA count (G*K=16). For tiny vocab (e.g. unit tests at
    # vocab<=4096) clamp BLOCK_VOCAB so the loop has at least one iteration
    # without crossing power-of-2.
    BLOCK_VOCAB = min(32768, max(1024, triton.next_power_of_2(vocab)))
    # 2V tile: lanes 0..V hold running top-V, V..2V hold chunk top-V.
    BLOCK_TOP = max(16, triton.next_power_of_2(2 * V))
    has_d2t = d2t is not None
    d2t_arg = d2t if has_d2t else gen_logits  # dummy ptr; never dereffed when HAS_D2T=False

    _ddtree_topk_kernel[(G, K)](
        gen_logits,
        d2t_arg,
        topk_log_vals,
        topk_ids,
        vocab=vocab,
        G=G,
        K=K,
        V=V,
        BLOCK_VOCAB=BLOCK_VOCAB,
        BLOCK_TOP=BLOCK_TOP,
        HAS_D2T=has_d2t,
        num_warps=16,
        num_stages=1,
    )
    return topk_log_vals, topk_ids


@triton.jit
def _ddtree_pack_retrieve_kernel(
    out_ptr,  # [count, n_dt, 3] int32
    retrieve_index_ptr,  # [S, n_dt] int32
    retrieve_next_token_ptr,  # [S, n_dt] int32
    retrieve_next_sibling_ptr,  # [S, n_dt] int32
    slot_ids_ptr,  # [count] int64
    n_dt: tl.constexpr,
    BLOCK_DT: tl.constexpr,
):
    """One CTA per (gen, sub-tile). Replaces 3 fancy-indexed slice copies.

    Layout of ``out``: out[g, i, 0|1|2] = retrieve_*[slot_ids[g], i].
    """
    g = tl.program_id(0)
    slot = tl.load(slot_ids_ptr + g)  # int64
    dt_off = tl.arange(0, BLOCK_DT)
    dt_mask = dt_off < n_dt

    src_off = slot * n_dt + dt_off
    ri = tl.load(retrieve_index_ptr + src_off, mask=dt_mask, other=0)
    rnt = tl.load(retrieve_next_token_ptr + src_off, mask=dt_mask, other=-1)
    rns = tl.load(retrieve_next_sibling_ptr + src_off, mask=dt_mask, other=-1)

    base = g * n_dt * 3 + dt_off * 3
    tl.store(out_ptr + base + 0, ri, mask=dt_mask)
    tl.store(out_ptr + base + 1, rnt, mask=dt_mask)
    tl.store(out_ptr + base + 2, rns, mask=dt_mask)


def _ddtree_pack_retrieve(
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    slot_ids: torch.Tensor,
    count: int,
    out: torch.Tensor,
):
    """One Triton launch fuses 3 fancy-indexed gather+copy + makes the result
    contiguous for the verify op. ``out`` must be ``[>=count, n_dt, 3]`` int32.
    """
    if count == 0:
        return out[:0]
    n_dt = retrieve_index.shape[1]
    BLOCK_DT = max(16, triton.next_power_of_2(n_dt))
    _ddtree_pack_retrieve_kernel[(count,)](
        out,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        slot_ids,
        n_dt=n_dt,
        BLOCK_DT=BLOCK_DT,
        num_warps=1,
    )
    return out[:count]


@triton.jit
def _ddtree_slot_scatter_kernel(
    # Slot storage (destination):
    ss_packed_mask_ptr,  # [S, n_dt, n_words] int32
    ss_pos_offsets_ptr,  # [S, n_dt] int32
    ss_retrieve_index_ptr,  # [S, n_dt] int32
    ss_retrieve_next_token_ptr,  # [S, n_dt] int32
    ss_retrieve_next_sibling_ptr,  # [S, n_dt] int32
    ss_has_tree_ptr,  # [S] bool (uint8)
    # build_ddtree outputs (source):
    src_packed_mask_ptr,  # [G, n_dt, n_words] int32
    src_pos_offsets_ptr,  # [G, n_dt] int32
    src_retrieve_index_ptr,  # [G, n_dt] int32
    src_retrieve_next_token_ptr,  # [G, n_dt] int32
    src_retrieve_next_sibling_ptr,  # [G, n_dt] int32
    # Slot ids (source row -> dst row mapping):
    slot_ids_ptr,  # [G] int64
    n_dt: tl.constexpr,
    n_words: tl.constexpr,
    BLOCK_DT: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Single launch fuses 5 ``index_copy_`` + 1 ``index_fill_`` (= 6 kernels)
    in :py:meth:`SpecTreeManager.scatter_to_slot_storage` plus the 5 work-buf
    ``.copy_()`` it previously stage into. Grid: (G,). Each CTA scatters one
    request's tree row into ``slot_storage[slot_id]``.
    """
    g = tl.program_id(0)
    slot = tl.load(slot_ids_ptr + g)  # int64

    dt_off = tl.arange(0, BLOCK_DT)
    dt_mask = dt_off < n_dt

    # ----- 1D-rank rows (4 of them: pos_offsets / retrieve_index / next_token / next_sibling)
    src_row = g * n_dt + dt_off
    dst_row = slot * n_dt + dt_off

    po = tl.load(src_pos_offsets_ptr + src_row, mask=dt_mask, other=0)
    ri = tl.load(src_retrieve_index_ptr + src_row, mask=dt_mask, other=0)
    rnt = tl.load(src_retrieve_next_token_ptr + src_row, mask=dt_mask, other=-1)
    rns = tl.load(src_retrieve_next_sibling_ptr + src_row, mask=dt_mask, other=-1)
    tl.store(ss_pos_offsets_ptr + dst_row, po, mask=dt_mask)
    tl.store(ss_retrieve_index_ptr + dst_row, ri, mask=dt_mask)
    tl.store(ss_retrieve_next_token_ptr + dst_row, rnt, mask=dt_mask)
    tl.store(ss_retrieve_next_sibling_ptr + dst_row, rns, mask=dt_mask)

    # ----- 2D rows for packed_mask: [n_dt, n_words]
    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < n_words
    mask_idx_src = (g * n_dt + dt_off[:, None]) * n_words + w_off[None, :]
    mask_idx_dst = (slot * n_dt + dt_off[:, None]) * n_words + w_off[None, :]
    pm_load_mask = dt_mask[:, None] & w_mask[None, :]
    pm = tl.load(src_packed_mask_ptr + mask_idx_src, mask=pm_load_mask, other=0)
    tl.store(ss_packed_mask_ptr + mask_idx_dst, pm, mask=pm_load_mask)

    # ----- has_tree[slot] = True (1 byte each)
    tl.store(ss_has_tree_ptr + slot, tl.full((), 1, tl.int8))


def _ddtree_slot_scatter(
    slot_storage, tree_dict, slot_ids: torch.Tensor, num_gens: int, dummy_slot_id: int
):
    """Fuse the 11-dispatch ``ddtree_scatter_slot_storage`` chain into one
    Triton launch:

      * 5 work-buf ``.copy_()`` (eliminated — never written; nothing reads them
        on the DDTree path)
      * 5 ``index_copy_`` into ``slot_storage.{packed_mask, position_offsets,
        retrieve_index, retrieve_next_token, retrieve_next_sibling}``
      * 1 ``index_fill_`` for ``has_tree[ids] = True``

    The dummy-slot reset (``has_tree[dummy_slot] = False``) stays as a small
    scalar write since it doesn't depend on per-request data.
    """
    if num_gens == 0:
        return
    n_dt = tree_dict["packed_mask"].shape[1]
    n_words = tree_dict["packed_mask"].shape[2]
    BLOCK_DT = max(16, triton.next_power_of_2(n_dt))
    BLOCK_W = max(8, triton.next_power_of_2(n_words))
    _ddtree_slot_scatter_kernel[(num_gens,)](
        slot_storage.packed_mask,
        slot_storage.position_offsets,
        slot_storage.retrieve_index,
        slot_storage.retrieve_next_token,
        slot_storage.retrieve_next_sibling,
        slot_storage.has_tree,
        tree_dict["packed_mask"],
        tree_dict["positions"],
        tree_dict["retrieve_index"],
        tree_dict["retrieve_next_token"],
        tree_dict["retrieve_next_sibling"],
        slot_ids,
        n_dt=n_dt,
        n_words=n_words,
        BLOCK_DT=BLOCK_DT,
        BLOCK_W=BLOCK_W,
        num_warps=2,
    )
    # Keep dummy slot invalid for CUDA-graph dummy iters.
    slot_storage.has_tree.narrow(0, dummy_slot_id, 1).fill_(False)


@triton.jit
def _ddtree_build_candidates_kernel(
    target_predict_ptr,  # [G, N] int32 — out
    candidates_ptr,  # [G, N] int32 — out
    tree_valid_ptr,  # [G] bool/int8 — out
    target_tokens_ptr,  # [num_flat] int64 — in (already argmaxed)
    draft_tokens_ptr,  # [G, B] int32 — in (B = N - 1)
    has_tree_ptr,  # [S] int8 — in
    slot_ids_ptr,  # [G] int64 — in (gen_slot_ids = all_ids_buf[ctx:ctx+G])
    num_contexts: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fuse 4 small ops:
    1. ``target_predict[:G] = target_tokens[ctx:ctx+G*N].view(G,N).int()``
    2. ``candidates[:G, 0]   = target_predict[:G, 0]``
    3. ``candidates[:G, 1:]  = draft_tokens[:G]``
    4. ``tree_valid[:G]      = has_tree[slot_ids[:G]]``
    """
    g_off = tl.arange(0, BLOCK_G)
    g_mask = g_off < G
    n_off = tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    # 1) target_predict[g, n] = (int32) target_tokens[num_contexts*N + g*N + n]
    src_idx = (num_contexts + g_off[:, None]) * N + n_off[None, :]
    src_mask = g_mask[:, None] & n_mask[None, :]
    tp = tl.load(target_tokens_ptr + src_idx, mask=src_mask, other=0).to(tl.int32)

    dst_idx = g_off[:, None] * N + n_off[None, :]
    tl.store(target_predict_ptr + dst_idx, tp, mask=src_mask)

    # 2 + 3) candidates[g, 0] = tp[g, 0];  candidates[g, 1:] = draft_tokens[g, n-1]
    # Build candidate tile: col 0 from tp[:, 0]; col n>=1 from draft_tokens[:, n-1].
    is_col0 = n_off == 0
    # Broadcast tp's column 0 across the row.
    tp_col0 = tl.where(n_off[None, :] == 0, tp, 0)
    # Sum across n axis to get [G] vector then broadcast back so col 0 holds it.
    tp_first = tl.sum(tp_col0, axis=1, keep_dims=True)  # [G, 1]
    # Load draft_tokens[g, n-1] for n>=1; safe index when n==0.
    dn = tl.maximum(n_off - 1, 0)
    dt_idx = g_off[:, None] * (N - 1) + dn[None, :]
    dt_load_mask = g_mask[:, None] & (n_off[None, :] >= 1) & ((n_off[None, :] - 1) < (N - 1))
    dt = tl.load(draft_tokens_ptr + dt_idx, mask=dt_load_mask, other=0)

    cand = tl.where(is_col0[None, :], tp_first, dt)
    tl.store(candidates_ptr + dst_idx, cand, mask=src_mask)

    # 4) tree_valid[g] = has_tree[slot_ids[g]]
    slot = tl.load(slot_ids_ptr + g_off, mask=g_mask, other=0)
    tv = tl.load(has_tree_ptr + slot, mask=g_mask, other=0)
    tl.store(tree_valid_ptr + g_off, tv, mask=g_mask)


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
    """One Triton launch fuses the pre-verify staging chain."""
    if num_gens == 0:
        return
    BLOCK_G = max(8, triton.next_power_of_2(num_gens))
    BLOCK_N = max(16, triton.next_power_of_2(N))
    _ddtree_build_candidates_kernel[(1,)](
        target_predict,
        candidates,
        tree_valid,
        target_tokens,
        draft_tokens,
        has_tree,
        slot_ids,
        num_contexts=num_contexts,
        G=num_gens,
        N=N,
        BLOCK_G=BLOCK_G,
        BLOCK_N=BLOCK_N,
        num_warps=2,
    )


@triton.jit
def _ddtree_pre_init_kernel(
    accepted_tokens_ptr,  # [B, P] int32
    num_accepted_ptr,  # [B] int32
    tree_accepted_indices_ptr,  # [B, K] int32
    batch_size: tl.constexpr,
    P: tl.constexpr,  # max_path_len = K+1
    K: tl.constexpr,  # max_draft_len
    BLOCK_B: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single launch replaces 3 separate ``zero_/fill_`` kernels.
    Sized for small batches (BLOCK_B >= batch_size); one CTA covers all rows.
    """
    b_off = tl.arange(0, BLOCK_B)
    b_mask = b_off < batch_size

    # accepted_tokens[:B, :P] = 0
    p_off = tl.arange(0, BLOCK_P)
    p_mask = p_off < P
    at_idx = b_off[:, None] * P + p_off[None, :]
    at_mask = b_mask[:, None] & p_mask[None, :]
    tl.store(accepted_tokens_ptr + at_idx, tl.zeros((BLOCK_B, BLOCK_P), tl.int32), mask=at_mask)

    # num_accepted[:B] = 1
    tl.store(num_accepted_ptr + b_off, tl.full((BLOCK_B,), 1, tl.int32), mask=b_mask)

    # tree_accepted_indices[:B, :K] = -1
    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < K
    tk_idx = b_off[:, None] * K + k_off[None, :]
    tk_mask = b_mask[:, None] & k_mask[None, :]
    tl.store(
        tree_accepted_indices_ptr + tk_idx, tl.full((BLOCK_B, BLOCK_K), -1, tl.int32), mask=tk_mask
    )


def _ddtree_pre_init(
    accepted_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    tree_accepted_indices: torch.Tensor,
    batch_size: int,
    max_path_len: int,
    max_draft_len: int,
):
    """Fused 3-buffer pre-init for ``_sample_and_accept_ddtree``.

    Replaces ``accepted_tokens[:B].zero_(); num_accepted[:B].fill_(1);
    tree_accepted_indices[:B].fill_(-1)`` (3 dispatches → 1).
    Caller supplies the underlying full-rank tensors; in-place writes are
    done on the leading B rows.
    """
    if batch_size == 0:
        return
    BLOCK_B = max(8, triton.next_power_of_2(batch_size))
    BLOCK_P = max(16, triton.next_power_of_2(max_path_len))
    BLOCK_K = max(16, triton.next_power_of_2(max_draft_len))
    _ddtree_pre_init_kernel[(1,)](
        accepted_tokens,
        num_accepted,
        tree_accepted_indices,
        batch_size=batch_size,
        P=max_path_len,
        K=max_draft_len,
        BLOCK_B=BLOCK_B,
        BLOCK_P=BLOCK_P,
        BLOCK_K=BLOCK_K,
        num_warps=2,
    )


@triton.jit
def _ddtree_post_scatter_kernel(
    accepted_tokens_ptr,  # [B, P] int32
    num_accepted_ptr,  # [B] int32
    tree_accepted_indices_ptr,  # [B, K] int32
    tree_accept_path_ptr,  # [B, P] int64
    accept_token_ptr,  # [num_gens, P] int32
    accept_token_num_ptr,  # [num_gens] int32
    accept_index_ptr,  # [num_gens, P] int32
    num_contexts: tl.constexpr,
    num_gens: tl.constexpr,
    P: tl.constexpr,  # max_path_len
    K: tl.constexpr,  # max_draft_len
    BLOCK_G: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Fuse the post-verify scatter chain (4 small kernels → 1)."""
    g_off = tl.arange(0, BLOCK_G)
    g_mask = g_off < num_gens

    p_off = tl.arange(0, BLOCK_P)
    p_mask = p_off < P
    k_mask = p_off < K

    src_idx = g_off[:, None] * P + p_off[None, :]
    src_mask = g_mask[:, None] & p_mask[None, :]

    # Load accept_token / accept_index into a [G, P] tile.
    acc_tok = tl.load(accept_token_ptr + src_idx, mask=src_mask, other=0)
    acc_idx = tl.load(accept_index_ptr + src_idx, mask=src_mask, other=0)
    acc_num = tl.load(accept_token_num_ptr + g_off, mask=g_mask, other=0)  # int32

    dst_b = num_contexts + g_off
    dst_idx_p = dst_b[:, None] * P + p_off[None, :]
    dst_idx_k = dst_b[:, None] * K + p_off[None, :]

    # accepted_tokens[ctx:ctx+G] = acc_tok
    tl.store(accepted_tokens_ptr + dst_idx_p, acc_tok, mask=g_mask[:, None] & p_mask[None, :])
    # num_accepted[ctx:ctx+G] = acc_num + 1
    tl.store(num_accepted_ptr + dst_b, acc_num + 1, mask=g_mask)
    # tree_accepted_indices[ctx:ctx+G, :K] = acc_idx[:, 1:K+1] - 1
    # Load shifted column and subtract 1.
    src_shift_idx = g_off[:, None] * P + (p_off[None, :] + 1)
    src_shift_mask = g_mask[:, None] & ((p_off[None, :] + 1) < P) & k_mask[None, :]
    acc_idx_shift = tl.load(
        accept_index_ptr + src_shift_idx, mask=src_shift_mask, other=1
    )  # other=1 -> stored as 0 after -1
    tl.store(
        tree_accepted_indices_ptr + dst_idx_k,
        acc_idx_shift - 1,
        mask=g_mask[:, None] & k_mask[None, :],
    )
    # tree_accept_path[ctx:ctx+G, :P] = acc_idx (cast to int64)
    tl.store(
        tree_accept_path_ptr + dst_idx_p,
        acc_idx.to(tl.int64),
        mask=g_mask[:, None] & p_mask[None, :],
    )


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
    """Fused post-verify scatter (4 kernels → 1)."""
    if num_gens == 0:
        return
    BLOCK_G = max(8, triton.next_power_of_2(num_gens))
    BLOCK_P = max(16, triton.next_power_of_2(max_path_len))
    _ddtree_post_scatter_kernel[(1,)](
        accepted_tokens,
        num_accepted,
        tree_accepted_indices,
        tree_accept_path,
        accept_token,
        accept_token_num,
        accept_index,
        num_contexts=num_contexts,
        num_gens=num_gens,
        P=max_path_len,
        K=max_draft_len,
        BLOCK_G=BLOCK_G,
        BLOCK_P=BLOCK_P,
        num_warps=2,
    )


@triton.jit
def _ddtree_heap_kernel(
    topk_log_vals_ptr,  # [G, K, V] float32
    topk_ids_ptr,  # [G, K, V] int64
    out_parent_ptr,  # [G, n_dt] int64
    out_token_ptr,  # [G, n_dt] int64
    out_depth_ptr,  # [G, n_dt] int64
    G: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_CAP: tl.constexpr,  # next_pow2(1 + 2*B)
):
    """One persistent CTA per request runs the entire B-step heap expansion.

    Frontier (score/depth/rank/parent/token) is held as ``[BLOCK_CAP]`` register
    tiles inside the program. Each iteration:
      1) ``argmax`` over the frontier → pop best prefix
      2) write ``out_parent / out_token / out_depth[node_idx]``
      3) overwrite popped slot with -inf, push next-sibling at ``1+2t`` and
         first-child at ``2+2t``
    No global atomics, no inter-CTA syncs; one launch replaces the B serial
    Inductor-generated kernels the original torch path produced.
    """
    g = tl.program_id(0)

    NEG_INF = -1.0e30
    n_dt = B + 1
    cap_arange = tl.arange(0, BLOCK_CAP)

    # Strides (row-major contiguous):
    #   topk_log_vals[g, k, v] -> ptr + g*(K*V) + k*V + v  (float32)
    #   topk_ids     [g, k, v] -> ptr + g*(K*V) + k*V + v  (int64)
    #   out_*[g, i]            -> ptr + g*n_dt + i
    base_kv = g * K * V

    # Frontier registers; slot 0 seeded with the (depth=1, rank=0) root child.
    # Slots 1..BLOCK_CAP-1 are NEG_INF, which already covers the inactive tail
    # past cap = 1 + 2*B (those slots are never selected by argmax).
    root_log = tl.load(topk_log_vals_ptr + base_kv + 0).to(tl.float32)
    root_id = tl.load(topk_ids_ptr + base_kv + 0)
    f_score = tl.where(cap_arange == 0, root_log, tl.full((BLOCK_CAP,), NEG_INF, tl.float32))
    f_depth = tl.where(
        cap_arange == 0, tl.full((BLOCK_CAP,), 1, tl.int32), tl.zeros((BLOCK_CAP,), tl.int32)
    )
    f_rank = tl.zeros((BLOCK_CAP,), tl.int32)
    f_parent = tl.zeros((BLOCK_CAP,), tl.int32)
    f_token = tl.where(cap_arange == 0, root_id, tl.zeros((BLOCK_CAP,), root_id.dtype))

    out_base = g * n_dt
    # Root: parent=-1, token=0, depth=0.
    tl.store(out_parent_ptr + out_base + 0, tl.full((), -1, tl.int64))
    tl.store(out_token_ptr + out_base + 0, tl.zeros((), tl.int64))
    tl.store(out_depth_ptr + out_base + 0, tl.zeros((), tl.int64))

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
        out_parent_val = tl.where(pvalid, pparent.to(tl.int64), tl.full((), -1, tl.int64))
        out_token_val = tl.where(pvalid, ptoken.to(tl.int64), tl.zeros((), tl.int64))
        out_depth_val = tl.where(pvalid, pdepth.to(tl.int64), tl.zeros((), tl.int64))
        tl.store(out_parent_ptr + out_base + node_idx, out_parent_val)
        tl.store(out_token_ptr + out_base + node_idx, out_token_val)
        tl.store(out_depth_ptr + out_base + node_idx, out_depth_val)

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


def _ddtree_heap_expand(
    topk_log_vals: torch.Tensor,
    topk_ids: torch.Tensor,
    B: int,
    K: int,
    V: int,
    out_parent: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    out_depth: torch.Tensor | None = None,
):
    """Run the heap-expansion Triton kernel; return (out_parent, out_token, out_depth).

    The three outputs may be passed in pre-allocated (sliced from worker
    persistent buffers) so the caller can avoid 3 dispatches per step.
    """
    G = topk_log_vals.shape[0]
    n_dt = B + 1
    device = topk_log_vals.device

    if out_parent is None:
        out_parent = torch.empty((G, n_dt), dtype=torch.int64, device=device)
    if out_token is None:
        out_token = torch.empty((G, n_dt), dtype=torch.int64, device=device)
    if out_depth is None:
        out_depth = torch.empty((G, n_dt), dtype=torch.int64, device=device)

    cap = 1 + 2 * B
    BLOCK_CAP = max(16, triton.next_power_of_2(cap))

    log_vals_f32 = topk_log_vals.to(torch.float32, copy=False).contiguous()
    ids_i64 = topk_ids.to(torch.int64, copy=False).contiguous()

    _ddtree_heap_kernel[(G,)](
        log_vals_f32,
        ids_i64,
        out_parent,
        out_token,
        out_depth,
        G=G,
        B=B,
        K=K,
        V=V,
        BLOCK_CAP=BLOCK_CAP,
        num_warps=1,
        num_stages=1,
    )
    return out_parent, out_token, out_depth


@triton.jit
def _ddtree_finalize_kernel(
    out_parent_ptr,  # [G, n_dt] int64
    out_token_ptr,  # [G, n_dt] int64
    out_depth_ptr,  # [G, n_dt] int64
    draft_tokens_ptr,  # [G, B] int32
    retrieve_index_ptr,  # [G, n_dt] int32
    retrieve_next_token_ptr,  # [G, n_dt] int32
    retrieve_next_sibling_ptr,  # [G, n_dt] int32
    positions_ptr,  # [G, n_dt] int32
    packed_mask_ptr,  # [G, n_dt, n_words] int32
    n_dt: tl.constexpr,
    n_words: tl.constexpr,
    K: tl.constexpr,
    BLOCK_DT: tl.constexpr,  # next_pow2(n_dt)
    BLOCK_W: tl.constexpr,  # next_pow2(n_words)
):
    """Per-row finalize: one block per (request g, node row i).

    Computes first-child / next-sibling pointers (min-j search over a
    register-tile of the parent row) and bit-packs the ancestor mask via a
    K-step parent-chain walk. Replaces the torch.compile path (~50 fused
    Inductor kernels per call) with a single launch.
    """
    g = tl.program_id(0)
    i = tl.program_id(1)

    parent_row_off = g * n_dt
    j = tl.arange(0, BLOCK_DT)
    j_in_range = j < n_dt

    # Load this request's parent row once into a register tile.
    parent_row = tl.load(out_parent_ptr + parent_row_off + j, mask=j_in_range, other=-1).to(
        tl.int32
    )
    valid_j = j_in_range & ((j == 0) | (parent_row >= 0))

    pi = tl.load(out_parent_ptr + parent_row_off + i).to(tl.int32)
    valid_i = (i == 0) | (pi >= 0)

    BIG = n_dt  # sentinel beyond any valid index

    # next-sibling: smallest j > i with parent_row[j] == pi (and pi >= 0).
    sib_cand = valid_j & (parent_row == pi) & (pi >= 0) & (j > i)
    sib_min = tl.min(tl.where(sib_cand, j, BIG), axis=0)
    sib_out = tl.where(valid_i & (sib_min < BIG), sib_min, -1).to(tl.int32)

    # first-child: smallest j with parent_row[j] == i.
    fc_cand = valid_j & (parent_row == i)
    fc_min = tl.min(tl.where(fc_cand, j, BIG), axis=0)
    fc_out = tl.where(valid_i & (fc_min < BIG), fc_min, -1).to(tl.int32)

    out_off = g * n_dt + i
    tl.store(retrieve_next_sibling_ptr + out_off, sib_out)
    tl.store(retrieve_next_token_ptr + out_off, fc_out)
    tl.store(retrieve_index_ptr + out_off, i)

    depth_i = tl.load(out_depth_ptr + out_off).to(tl.int32)
    tl.store(positions_ptr + out_off, depth_i)

    # draft_tokens[g, i-1] = out_token[g, i] for i >= 1.
    if i >= 1:
        tok = tl.load(out_token_ptr + out_off).to(tl.int32)
        tl.store(draft_tokens_ptr + g * (n_dt - 1) + (i - 1), tok)

    # ---- bit-packed ancestor mask ----
    w = tl.arange(0, BLOCK_W)
    w_in_range = w < n_words
    mask_words = tl.zeros((BLOCK_W,), tl.int32)

    # Self-bit: row i attends col i (only if valid).
    if valid_i:
        self_word = i // 32
        self_bit = i % 32
        mask_words = tl.where(w == self_word, (1 << self_bit).to(tl.int32), mask_words)

    # Walk parent chain at most K steps (max depth from any node to root <= K).
    cur = i
    for _ in tl.static_range(K):
        cur_parent = tl.load(out_parent_ptr + parent_row_off + cur).to(tl.int32)
        has_parent = cur_parent >= 0
        safe_parent = tl.maximum(cur_parent, 0)
        p_word = safe_parent // 32
        p_bit = safe_parent % 32
        bit_vec = tl.where(w == p_word, (1 << p_bit).to(tl.int32), tl.zeros((BLOCK_W,), tl.int32))
        if has_parent:
            mask_words = mask_words | bit_vec
            cur = safe_parent

    pm_off = g * n_dt * n_words + i * n_words + w
    tl.store(packed_mask_ptr + pm_off, mask_words, mask=w_in_range)


def _ddtree_finalize_triton(
    out_parent: torch.Tensor,
    out_token: torch.Tensor,
    out_depth: torch.Tensor,
    B: int,
    K: int,
    buffers: dict | None = None,
):
    """Run the single-launch Triton finalize kernel.

    Returns ``(draft_tokens, retrieve_index, retrieve_next_token,
    retrieve_next_sibling, positions, packed_mask)`` — all int32, on the same
    device as ``out_parent``.

    If ``buffers`` is provided it must contain the same keys as the return
    tuple, each sliced to the leading G/G*n_dt rows; using persistent buffers
    saves 6 ``torch.empty`` dispatches per step.
    """
    G = out_parent.shape[0]
    n_dt = B + 1
    n_words = (n_dt + 31) // 32
    device = out_parent.device

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

    BLOCK_DT = max(16, triton.next_power_of_2(n_dt))
    BLOCK_W = max(1, triton.next_power_of_2(n_words))

    _ddtree_finalize_kernel[(G, n_dt)](
        out_parent,
        out_token,
        out_depth,
        draft_tokens,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        positions,
        packed_mask,
        n_dt=n_dt,
        n_words=n_words,
        K=K,
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
    heap_buffers: dict | None = None,
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

    # 1) Heap expansion: a single Triton launch (one persistent CTA per request)
    #    runs the full B-step pop/push loop in registers. Replaces the 30..254
    #    serialized Inductor-generated kernels the torch.compile path emitted.
    with nvtx_range("ddtree_heap_expand"):
        if heap_buffers is None:
            hp = ht = hd = None
        else:
            hp = heap_buffers["out_parent"]
            ht = heap_buffers["out_token"]
            hd = heap_buffers["out_depth"]
        out_parent, out_token, out_depth = _ddtree_heap_expand(
            topk_log_vals, topk_ids, B, K, V, out_parent=hp, out_token=ht, out_depth=hd
        )

    # 2) Finalize: parent pointers -> first-child / next-sibling / packed mask.
    #    Single Triton launch with grid (G, n_dt); each block produces one
    #    output row (sibling/first-child via min-j over the parent row, plus a
    #    K-step parent-chain walk for the bit-packed ancestor mask). Replaces
    #    the previous torch.compile path (~50 fused Inductor kernels per call,
    #    each paying full python/CUDA dispatch overhead at TP=8).
    with nvtx_range("ddtree_finalize"):
        (
            draft_tokens,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            positions,
            packed_mask,
        ) = _ddtree_finalize_triton(
            out_parent, out_token, out_depth, B, K, buffers=finalize_buffers
        )

    return {
        "draft_tokens": draft_tokens,
        "retrieve_index": retrieve_index,
        "retrieve_next_token": retrieve_next_token,
        "retrieve_next_sibling": retrieve_next_sibling,
        "positions": positions,
        "packed_mask": packed_mask,
    }
