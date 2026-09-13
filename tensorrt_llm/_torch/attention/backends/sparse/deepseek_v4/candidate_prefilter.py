# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's two-level candidate prefilter, as pure tensor ops.

V4.1 runs indexer selection in two levels. The layer named by
``candidate_source_layer_id`` (20 in the release) reduces its own indexer logits
to per-``candidate_block_size`` blocks, keeps the best ``candidate_topk_blocks``
of them, and publishes that block set. Every *later* index source
(24/28/32/36 in the release) restricts its own logits to the published blocks
before taking its own top-``index_topk``. Level one therefore decides what
level two is even allowed to see.

Producer and consumers necessarily share a compressed-position axis: the
reference applies the published set with a ``masked_fill`` against a tensor of
the consumer's own shape, which only type-checks when the widths agree. In the
release they do -- ``compress_ratios`` is 1 for every layer from 20 up, so
layer 20 and layers 24/28/32/36 all count raw positions. This module inherits
that assumption and ``select_candidate_blocks`` asserts nothing about it,
because the caller is the only place that knows both ratios.

Two representation choices differ deliberately from the reference:

* **The published carrier is block ids, not a dense mask.** The reference keeps
  a bool tensor shaped like the logits, which at 128 K and ratio 1 is
  ~17 GB for one layer. Block ids cost ``candidate_topk_blocks * 4`` = 8 KiB per
  query row, i.e. 64 MiB for an 8192-token prefill chunk, and the cost does not
  grow with context length. Chunked prefill is therefore a precondition for the
  long-context path, not an optimisation.
* **The dense mask is still materialised, but only per logits tile.** A tile is
  already bounded by the indexer's own MQA-logits element budget, so a bool
  mask of the same shape is strictly smaller than the logits it masks. What was
  unaffordable was one mask for the whole prefill, not one mask per tile.

Blocks are anchored at each row's ``row_start`` rather than at column 0. In
prefill the logits' column 0 is the *chunk's* first compressed KV row, which for
the second and later requests of a multi-request chunk is not that request's
position 0 and is not a multiple of ``block_size``. Anchoring at ``row_start``
reproduces the reference's per-request block grid; anchoring at column 0 would
silently shift every block boundary for those requests.

Reference-behaviour details that are load-bearing and were measured rather than
assumed (see ``select_candidate_blocks``):

* the block holding the newest reachable position is pinned and always kept;
* blocks whose score is ``-inf`` (nothing reachable in them) are dropped rather
  than padded in, so fewer than ``topk_blocks`` ids can come back;
* the kept set is a *superset* of the reachable set -- blocks are kept whole, so
  a partially filled newest block drags its unreachable positions along. The
  overshoot never passes that block's own boundary, and downstream code must
  keep clamping selected positions against the sequence length regardless.
"""

import dataclasses
from typing import Optional, Tuple

import torch

__all__ = [
    "CandidatePublication",
    "apply_candidate_blocks",
    "candidate_prefilter_is_inert",
    "candidate_prefilter_is_on",
    "inert_up_to_positions",
    "select_candidate_blocks",
]


@dataclasses.dataclass
class CandidatePublication:
    """What a candidate source publishes for one forward pass.

    One record rather than parallel per-layer dicts, because the three fields are
    only meaningful together: ``rows`` is the guard that ``blocks`` was actually
    written where a consumer is about to read (an unwritten row is all -1, i.e.
    keep nothing, i.e. a silently arbitrary top-k), and ``compress_ratio`` is what
    lets a consumer refuse block ids from a different column space. Three separate
    dicts could go out of step with each other under a partially-run producer;
    one record cannot.
    """

    #: ``[num_tokens, topk_blocks]`` int32 block ids, indexed by *global query
    #: row* so a consumer with different tiling still slices the rows it scores.
    blocks: torch.Tensor
    #: The producer's compress ratio. Block ids are row-local to the producer's
    #: ``cu_seqlen_ks``, so consuming across ratios is not meaningful.
    compress_ratio: int
    #: How many rows of ``blocks`` the producer has written so far this pass.
    rows: int = 0


def candidate_prefilter_is_on(
    source_layer_id: Optional[int],
    topk_blocks: int,
    block_size: int,
) -> bool:
    """Whether the three configured values switch both levels on.

    Hoisted into one function because two places have to answer it and must not
    disagree: :class:`Indexer` decides per layer whether it publishes or consumes
    candidate blocks, and ``DeepseekV41ForCausalLM`` decides whether the
    "prefilter is missing, refuse a long ceiling" guard still applies. If those
    two ever diverged, the guard would either refuse a run the runtime handles
    correctly or wave through one it does not.

    A missing ``source_layer_id`` is the off switch — V4, and any V4.1 config
    that omits it — and a non-positive block count or block size is treated the
    same way, matching :func:`inert_up_to_positions`, which returns 0 there.
    """
    if source_layer_id is None:
        return False
    return int(topk_blocks) > 0 and int(block_size) > 0


def inert_up_to_positions(topk_blocks: int, block_size: int, compress_ratio: int = 1) -> int:
    """Longest sequence for which level one provably keeps everything.

    Level one is a no-op exactly while a row has no more blocks than it is
    allowed to keep, i.e. ``ceil(compressed_len / block_size) <= topk_blocks``,
    which is ``compressed_len <= topk_blocks * block_size``. The candidate
    source counts *compressed* positions, so a pooled source (ratio 2) stays
    inert to twice the raw length. The release's source is layer 20 at ratio 1,
    where the two coincide at 2048 * 8 = 16384.
    """
    if topk_blocks <= 0 or block_size <= 0:
        return 0
    return topk_blocks * block_size * max(1, int(compress_ratio))


def candidate_prefilter_is_inert(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> bool:
    """Whether level one would keep every reachable position for these rows.

    Cheap enough to be worth checking per call: it collapses to one device-side
    max and one comparison, and when it is true both levels can be skipped with
    a *provable* (not approximate) bit-identical result. The bound is on the
    block count, so it holds row by row -- a batch mixing a short row with a
    long one is not inert.

    Synchronises, because the answer decides host-side control flow. Callers on
    a CUDA-graph-captured path must not use it; use the configured ceiling from
    :func:`inert_up_to_positions` instead, which is a static property of the
    run.
    """
    if topk_blocks <= 0 or block_size <= 0:
        return True
    lens = (row_ends - row_starts).clamp_(min=0)
    if lens.numel() == 0:
        return True
    max_blocks = (int(lens.max().item()) + block_size - 1) // block_size
    return max_blocks <= topk_blocks


def _block_index(
    width: int,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-element block id, with unreachable elements sent to a sink bucket.

    Returns ``(block_id, lens)`` where ``block_id`` is ``[num_rows, width]``
    int64 with values in ``[0, num_blocks]`` -- ``num_blocks`` being a sink that
    collects everything outside ``[row_start, row_end)`` so that scatter and
    gather can stay unmasked.
    """
    device = row_starts.device
    cols = torch.arange(width, device=device, dtype=torch.int64)
    starts = row_starts.to(torch.int64).unsqueeze(1)
    ends = row_ends.to(torch.int64).unsqueeze(1)
    rel = cols.unsqueeze(0) - starts
    reachable = (rel >= 0) & (cols.unsqueeze(0) < ends)
    block_id = rel.div(block_size, rounding_mode="floor")
    # Scalar ``other``: a ``full_like`` here would allocate and fill a second
    # int64 ``[num_rows, width]`` tile purely to carry one constant.
    block_id = torch.where(reachable, block_id, num_blocks)
    lens = (ends - starts).squeeze(1).clamp_(min=0)
    return block_id, lens


def select_candidate_blocks(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one: reduce ``logits`` to the block ids this row may select from.

    Args:
        logits: ``[num_rows, width]`` indexer logits. Positions outside
            ``[row_start, row_end)`` are ignored whatever they contain, so the
            caller does not have to pre-mask -- but note the *consumer* still
            must clamp against the sequence length, because the returned set is
            a superset of the reachable set (see the module docstring).
        row_starts: ``[num_rows]`` first reachable column, i.e. the indexer's
            ``cu_seqlen_ks``. Also the block grid's anchor.
        row_ends: ``[num_rows]`` one past the last reachable column, i.e.
            ``cu_seqlen_ke``.
        topk_blocks: ``candidate_topk_blocks``.
        block_size: ``candidate_block_size``.

    Returns:
        ``[num_rows, min(topk_blocks, num_blocks)]`` int32 block ids, ascending
        within a row, ``-1``-padded. Ids are *row-local*: block ``b`` of row
        ``r`` spans columns ``row_starts[r] + b * block_size`` up to
        ``+ block_size``.
    """
    if logits.dim() != 2:
        raise ValueError(f"expected [num_rows, width] logits, got shape {tuple(logits.shape)}")
    num_rows, width = logits.shape
    if row_starts.shape[0] != num_rows or row_ends.shape[0] != num_rows:
        raise ValueError(
            "row_starts/row_ends must have one entry per logits row: "
            f"got {row_starts.shape[0]}/{row_ends.shape[0]} for {num_rows} rows"
        )
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError(
            f"topk_blocks and block_size must be positive, got {topk_blocks}/{block_size}"
        )

    # A row's blocks are anchored at its own start, so the widest possible row
    # spans the whole tile: `width` columns is the bound regardless of where the
    # individual rows begin.
    num_blocks = (width + block_size - 1) // block_size
    block_id, lens = _block_index(width, row_starts, row_ends, block_size, num_blocks)

    # `-inf` rather than `finfo.min` for the *internal* block score, so that
    # "nothing reachable in this block" is distinguishable from a real logit no
    # matter how negative that logit is. (The mask value written back into the
    # logits in `apply_candidate_blocks` is `finfo.min` instead -- see there.)
    neg_inf = float("-inf")
    # `include_self=True` on a -inf-filled buffer means an untouched block keeps
    # -inf, which is exactly the "nothing reachable here" marker the drop below
    # relies on. `amax` (not `mean`) matches the reference: a block is worth its
    # best position.
    scores = logits.new_full((num_rows, num_blocks + 1), neg_inf)
    scores.scatter_reduce_(1, block_id, logits, reduce="amax", include_self=True)
    scores = scores[:, :num_blocks]

    # Pin the block holding the newest reachable position. Not an optimisation:
    # that block is partially filled, so on a decode step it can lose to an
    # older full block and the row would drop its own most recent tokens.
    last_block = (lens - 1).div(block_size, rounding_mode="floor")
    pin = torch.zeros_like(scores, dtype=torch.bool)
    has_any = lens > 0
    pin.scatter_(1, last_block.clamp_(min=0).unsqueeze(1), has_any.unsqueeze(1))
    # Scalar overloads throughout: the ``full_like`` forms allocated and filled a
    # whole extra ``[num_rows, num_blocks + 1]`` tile per call to carry a constant.
    scores = torch.where(pin, float("inf"), scores)

    keep = min(topk_blocks, num_blocks)
    top = scores.topk(keep, dim=-1)
    # A row with fewer reachable blocks than `keep` gets -inf picks back; they
    # are dropped rather than kept, otherwise a short row would select garbage
    # columns that happen to sit in an unreachable block.
    dropped = ~(top.values > neg_inf)
    # Ascending ids make the level-two gather sequential and make the result
    # independent of top-k tie-breaking order, which is not specified. Dropped
    # picks are parked on `num_blocks` so they sort to the end, then rewritten
    # to -1: sorting -1 directly would put the padding at the *front*.
    ids = torch.where(dropped, num_blocks, top.indices)
    ids, _ = ids.sort(dim=-1)
    ids = torch.where(ids >= num_blocks, -1, ids)
    return ids.to(torch.int32)


def apply_candidate_blocks(
    logits: torch.Tensor,
    block_ids: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    block_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Level two: send every logit outside the published blocks to ``-inf``.

    Equivalent to the reference's ``index_score.masked_fill(~candidates, -inf)``
    and, like it, applied *before* the consumer's own top-k. Positions already
    outside ``[row_start, row_end)`` are masked too, which the indexer's top-k
    would have excluded anyway via its own row bounds -- doing it here as well
    costs nothing and keeps this function's contract independent of the top-k's.

    ``block_ids`` may be the slice of a wider published tensor, as long as its
    rows line up with ``logits``'s and it was produced with the same
    ``row_starts`` (guaranteed when producer and consumer share a compress
    ratio, since ``cu_seqlen_ks`` is then the same tensor).
    """
    if block_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            "block_ids must have one row per logits row: "
            f"got {block_ids.shape[0]} for {logits.shape[0]} rows"
        )
    num_rows, width = logits.shape
    num_blocks = (width + block_size - 1) // block_size
    block_id, _ = _block_index(width, row_starts, row_ends, block_size, num_blocks)

    kept = torch.zeros((num_rows, num_blocks + 1), dtype=torch.bool, device=logits.device)
    # -1 padding lands in the sink column, which is then cleared, so padding can
    # never keep a block.
    ids = block_ids.to(torch.int64)
    ids = torch.where(ids >= 0, ids, num_blocks)
    kept.scatter_(1, ids, True)
    kept[:, num_blocks] = False

    # Negated in place into the drop mask `masked_fill` actually wants: the gather
    # result is private to this call, and `~mask` would allocate a second
    # full-size bool tile over the whole logits tile on every consumer layer.
    drop = kept.gather(1, block_id).logical_not_()
    # `finfo.min`, not `-inf`, for what goes back into the logits: the downstream
    # top-k only ever compares these values, so the two are equivalent for
    # selection, and `finfo.min` cannot turn into a NaN if any later kernel adds
    # to or scales a masked entry.
    neg_inf = torch.finfo(logits.dtype).min
    if out is None:
        return logits.masked_fill(drop, neg_inf)
    out.copy_(logits)
    return out.masked_fill_(drop, neg_inf)
