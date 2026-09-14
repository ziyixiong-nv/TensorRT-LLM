# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoder-half bounded replay for DeepSeek-V4.1 (tech report §3.2.2).

V4.1 splits its 40 layers at ``L/2``: layers 0..19 are the *encoder* half, whose
job is to produce the compressed latent that layer 20's shared ``W_KV`` turns
into the one global KV cache all 20 *decoder* layers read. The decoder layers own
no compressor -- ``kv_source_layer_ids`` has exactly one entry >= 20 -- so the
only cache they write is their own layer-wise sliding window.

That is what makes the halving legal. Once the encoder half has run over the full
prompt and layer 20 has published the complete global KV, the decoder half only
has to produce *correct logits for the last token*, and every quantity it needs
for that is either (a) in the complete global cache or (b) inside a window of
``n_win`` positions. So the decoder half can be replayed over just the last
``n_win`` tokens instead of all ``N``, turning roughly half the model's prefill
into a fixed cost.

**The approximation, stated plainly.** Layer 21's sliding-window KV for the
replayed tokens is *exact*: its input is layer 20's output, and layer 20 ran over
the whole prompt. Layer 21's *outputs* for the earliest replayed tokens are not,
because their windows were truncated at the replay floor -- so layer 22's window
KV is approximate, and the error compounds with depth. The last replayed token,
the one whose logits are actually consumed, keeps a full ``n_win``-wide window at
every layer, which is the property the whole scheme rests on. The report is
explicit that the replayed window KV is therefore "used only for decoding, not
for prefix caching"; :func:`plan_decoder_replay` refuses to run when the cached
window KV it would poison could be read back.

**What this module does and does not touch.** It is pure metadata bookkeeping
around one extra ``prepare()``: no kernel, no C++, and no scheduler change. The
truncated window itself is one line in the index builder (see
``DeepseekV4TrtllmAttentionMetadata.swa_window_floored_to_pass``), because SWA
positions are materialized as explicit per-token index lists rather than derived
from a cached-token count.

The pass boundary has exactly one non-obvious rule, and it follows from *whose*
row space a cross-layer handoff lives in:

* **KV-row handoffs survive it.** ``v41_index_keys`` is indexed by compressed-KV
  entry, and the global cache is complete after the encoder half, so the decoder
  layers want it whole and unsliced. Reinstating it is not optional even though
  the replayed context scoring reads keys from the cache rather than from the
  handoff: every non-owner index source asserts its source published in *this*
  forward (``indexer.py::_run_serial_indexer_prepare``), and the generation rows
  in a mixed batch do consume the activation.
* **Query-row handoffs are re-sliced.** ``v41_candidate_blocks`` and
  ``v41_topk_indices`` are both indexed by global query row, so both are gathered
  down to the replayed rows. Dropping the top-k instead would be wrong twice
  over: the 12 decoder layers that own no indexer cannot recompute one -- they
  assert their source published in *this* forward
  (``module.py``, ``v41_topk_indices.get(source)``) -- and there would be nothing
  to recompute *from* either, since a gathered row is already exactly the
  selection that row's token got over a global cache the replay does not change.
  The four decoder layers that do own an indexer overwrite their own entry with a
  128-query selection, which their consumers then read; both kinds of consumer
  therefore see a 128-row carrier.

Getting that backwards is silent: a query-row carrier read at encoder offsets
returns rows belonging to other tokens, which is a plausible-looking top-k over
the wrong positions rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from .candidate_prefilter import CandidatePublication
    from .metadata import DeepseekV4TrtllmAttentionMetadata

__all__ = [
    "DecoderReplayPlan",
    "enter_decoder_replay",
    "exit_decoder_replay",
    "gather_replayed_rows",
    "plan_decoder_replay",
    "scatter_replayed_rows",
]


@dataclass
class DecoderReplayPlan:
    """One forward's replay bookkeeping: what to run, and how to undo it.

    One record rather than loose locals because the "undo" half is not derivable
    from the metadata once :func:`enter_decoder_replay` has overwritten it -- the
    original ``seq_lens`` is the only place the encoder-pass row count survives,
    and ``rows`` is the only map back to it.
    """

    #: Global query rows, in the *encoder* pass's token space, that the decoder
    #: half replays. Packed per request in request order, so it doubles as the
    #: gather index for every per-token activation crossing the boundary.
    rows: torch.Tensor
    #: Per-request query count for the replay pass (``min(seq_len, window)`` for a
    #: context request, unchanged for a generation request).
    replay_seq_lens: torch.Tensor
    #: Per-request cached-token count for the replay pass, raised by exactly the
    #: number of query rows the pass drops.
    replay_num_cached: List[int]
    #: ``seq_lens`` as the encoder pass saw it. Restoring this restores
    #: ``num_tokens``, ``num_ctx_tokens`` and ``seq_lens_cuda`` with it, because
    #: they are all derived through the ``seq_lens`` setter.
    saved_seq_lens: torch.Tensor
    saved_num_cached: List[int]
    #: Row count of the encoder pass, i.e. the length every per-token activation
    #: crossing the boundary has on the way in and must have again on the way out.
    num_encoder_tokens: int

    @property
    def num_replay_tokens(self) -> int:
        return int(self.rows.shape[0])


def plan_decoder_replay(
    metadata: "DeepseekV4TrtllmAttentionMetadata",
    window: int,
) -> Optional[DecoderReplayPlan]:
    """Decide whether this batch can be replayed, and lay out the row map.

    Returns None -- run the ordinary single-pass forward -- whenever replay would
    not be both safe and worthwhile:

    * **No context requests.** A decode batch already runs one query row per
      sequence; there is nothing to skip, and this is also the CUDA-graph-captured
      shape, which must not grow a second ``prepare()``.
    * **Nothing long enough.** With every context at or under ``window`` the row
      map is the identity and the second pass is pure overhead.
    * **Context-with-cached-KV is not enabled.** The replay pass presents context
      requests whose KV starts below their first query row, and
      ``enable_context_mla_with_cached_kv`` is the flag that makes that shape
      legal -- twice over. The MLA context path asserts it off before taking its
      no-cached-KV branch (``trtllm.py:827``), and the indexer only routes ctx
      scoring through the cache gather when it is on: with it off and one chunk
      group, ``dsa/indexer.py`` falls back to a single-pass branch that slices
      ``k_fp8[:num_ctx_kv_tokens]`` off the *activation* handoff, and this pass's
      ``num_ctx_kv_tokens`` counts only its own ``window`` new rows -- so it would
      silently score against the first 128 keys of the prefix instead of all of
      them. On the chunked path each query instead gets
      ``cu_seqlen_ks = 0 .. (num_cached + i)//ratio``, the whole prefix, which is
      what replay needs.
    * **Window KV is reusable across requests.** §3.2.2's replayed window KV is
      approximate and must not be read back by a later request, so a cache
      configured to reuse it rules replay out. This is the one refusal that is
      about correctness rather than benefit.
    * **Attention DP.** Each DP rank batches independently, so the replay row
      count is a per-rank quantity, but MoE sizes its all-to-all from
      ``all_rank_num_tokens``, which was all-gathered *before* the forward at
      encoder shape. Re-gathering it mid-forward would put a collective behind a
      per-rank predicate, and a rank that declined to replay would not reach it.
      Without DP ``all_rank_num_tokens`` is None and every rank sees the same
      batch, so the decision is rank-invariant by construction.
    """
    if window <= 0 or metadata.num_contexts == 0:
        return None
    if not metadata.enable_context_mla_with_cached_kv:
        return None
    if metadata.mapping is not None and metadata.mapping.enable_attention_dp:
        return None
    if _swa_is_reused(metadata):
        return None

    seq_lens = metadata.seq_lens
    num_requests = metadata.num_seqs
    num_contexts = metadata.num_contexts
    host_seq_lens = seq_lens[:num_requests].tolist()
    num_cached = list(metadata.kv_cache_params.num_cached_tokens_per_seq[:num_requests])

    # A generation request is left alone: its one (or 1 + draft) query row is
    # already inside any window, so `min` would be the identity anyway, and
    # touching its cached count would desynchronize it from the KV manager.
    replay_lens = [
        min(length, window) if req < num_contexts else length
        for req, length in enumerate(host_seq_lens)
    ]
    if replay_lens == host_seq_lens:
        return None

    rows: List[torch.Tensor] = []
    start = 0
    for length, kept in zip(host_seq_lens, replay_lens):
        rows.append(torch.arange(start + length - kept, start + length, dtype=torch.int64))
        start += length

    replay_num_cached = [
        cached + (length - kept)
        for cached, length, kept in zip(num_cached, host_seq_lens, replay_lens)
    ]

    return DecoderReplayPlan(
        rows=torch.cat(rows).to(device=metadata.seq_lens_cuda.device, non_blocking=True),
        replay_seq_lens=torch.tensor(replay_lens, dtype=seq_lens.dtype),
        replay_num_cached=replay_num_cached,
        saved_seq_lens=seq_lens,
        saved_num_cached=num_cached,
        num_encoder_tokens=metadata.num_tokens,
    )
    # No cross-layer handoff is captured here on purpose. Planning happens before
    # the first layer runs, so at this point every one of those dicts is empty --
    # the publications the decoder half consumes are written *by* the encoder half.
    # enter_decoder_replay() reads them at the boundary instead, which is the only
    # moment they exist and the only moment they are about to be cleared.


def _swa_is_reused(metadata: "DeepseekV4TrtllmAttentionMetadata") -> bool:
    """Whether a later request could read this pass's sliding-window KV.

    Read off the manager rather than the config so a manager that disabled the
    feature for its own reasons (a draft cache does) is believed over the config
    that asked for it. An unrecognized manager answers True: refusing to replay
    costs prefill time, while replaying into a reused window cache corrupts
    another request's prefix.
    """
    manager = metadata.kv_cache_manager
    reuse = getattr(manager, "enable_swa_scratch_reuse", None)
    if reuse is None:
        return True
    return bool(reuse)


def enter_decoder_replay(
    metadata: "DeepseekV4TrtllmAttentionMetadata",
    plan: DecoderReplayPlan,
) -> None:
    """Re-point ``metadata`` at the replayed rows, then rebuild it.

    Shrinking ``seq_lens`` while raising ``num_cached_tokens_per_seq`` by the same
    amount is the shape a chunked prefill's final chunk already has, which is why
    ``prepare()`` needs no new branch to serve it. What is *not* already
    supported is that the window cache holds nothing below the new cached
    boundary, and :attr:`swa_window_floored_to_pass` is what tells the index
    builder so.

    The cross-layer handoffs are read here rather than carried on the plan
    because *here* is the only place they exist. Planning runs before layer 0, so
    a snapshot taken then is empty; the encoder half writes these dicts as it
    goes, and ``prepare()`` below is what clears them. Capturing between those two
    events is not a detail of this function, it is the whole reason it is a
    function -- a snapshot on either side of the boundary silently hands the
    decoder half nothing, and the first indexed decoder layer then asserts that
    its source never published.
    """
    saved_index_keys = dict(metadata.v41_index_keys)
    saved_candidate_blocks = dict(metadata.v41_candidate_blocks)
    saved_topk_indices = dict(metadata.v41_topk_indices)

    metadata.kv_cache_params.num_cached_tokens_per_seq[: len(plan.replay_num_cached)] = (
        plan.replay_num_cached
    )
    metadata.swa_window_floored_to_pass = True
    # Assigning through the property recomputes num_tokens / num_ctx_tokens and
    # refreshes seq_lens_cuda; prepare() then reads the new shape everywhere.
    metadata.seq_lens = plan.replay_seq_lens
    metadata.prepare()

    # prepare() cleared the cross-layer handoffs. Reinstate the ones the decoder
    # half still needs, in the row space it will read them in -- see the module
    # docstring for why these two are treated differently.
    #
    # The fourth dict prepare() clears, `v41_candidate_rows_consumed`, is
    # deliberately *not* reinstated: it is a per-consumer tally that
    # `_check_candidate_rows` compares against its producer's row count
    # (`dsa/indexer.py:1730`), and after the boundary a decoder consumer masks
    # `window` rows against a publication `_gather_publication` has just retyped to
    # `window`. Carrying the encoder pass's tally forward would make that equality
    # compare a replay-height consumption against an encoder-height record and fail
    # on a correct forward. The check runs per layer inside that layer's own
    # indexer, so the encoder consumers have already been checked by this point.
    metadata.v41_index_keys.update(saved_index_keys)
    for layer_idx, published in saved_candidate_blocks.items():
        metadata.v41_candidate_blocks[layer_idx] = _gather_publication(published, plan.rows)
    for layer_idx, topk in saved_topk_indices.items():
        metadata.v41_topk_indices[layer_idx] = topk.index_select(0, plan.rows)


def exit_decoder_replay(
    metadata: "DeepseekV4TrtllmAttentionMetadata",
    plan: DecoderReplayPlan,
) -> None:
    """Restore the encoder pass's shape.

    Only the *inputs* are restored -- ``seq_lens`` (and with it ``num_tokens``,
    ``num_ctx_tokens`` and ``seq_lens_cuda``), the cached-token counts, and the
    floor switch. The device index buffers are deliberately left holding replay
    values: every forward rebuilds them from these inputs in ``prepare()``, and
    copying them back would cost a second rebuild to no observable end.
    """
    metadata.kv_cache_params.num_cached_tokens_per_seq[: len(plan.saved_num_cached)] = (
        plan.saved_num_cached
    )
    metadata.swa_window_floored_to_pass = False
    metadata.seq_lens = plan.saved_seq_lens


def gather_replayed_rows(
    tensor: Optional[torch.Tensor],
    plan: DecoderReplayPlan,
) -> Optional[torch.Tensor]:
    """Slice a per-token tensor down to the replayed rows.

    The model body carries two layouts and must not have to tell them apart at
    each call site: activations are row-major ``[N, ...]``, while position ids
    arrive as ``[N]`` or ``[1, N]`` -- per-token along the *last* axis. The axis
    is identified by which one is ``N``, and a tensor that is ``N`` on neither is
    an error rather than something to pass through, because passing an
    encoder-length tensor into the replay pass is the silent-corruption case this
    whole module is built to avoid.
    """
    if tensor is None:
        return None
    num_tokens = plan.num_encoder_tokens
    rows = plan.rows
    if tensor.shape[0] == num_tokens:
        return tensor.index_select(0, rows)
    if tensor.shape[-1] == num_tokens:
        return tensor.index_select(tensor.dim() - 1, rows)
    raise ValueError(
        f"tensor of shape {tuple(tensor.shape)} is not per-token for a pass of "
        f"{num_tokens} tokens, so the decoder replay boundary cannot re-index it."
    )


def scatter_replayed_rows(
    replayed: torch.Tensor,
    plan: DecoderReplayPlan,
) -> torch.Tensor:
    """Widen the replay pass's output back to the encoder pass's row count.

    Everything downstream of the model body indexes hidden states with
    encoder-space rows -- ``_get_last_token_states`` takes ``cumsum(seq_lens) -
    1``, and the padded-token slice takes ``[:num_tokens]``. Rather than teach
    those the row map, the replayed rows are scattered back into a full-height
    buffer. The rows the decoder half did not run are zero, which is correct only
    because replay is refused when any consumer needs them (see
    ``DeepseekV41Model._plan_bounded_replay``); zeros rather than uninitialized
    memory so that a future consumer that does read them is wrong reproducibly
    instead of wrong intermittently.
    """
    full = replayed.new_zeros((plan.num_encoder_tokens,) + tuple(replayed.shape[1:]))
    full.index_copy_(0, plan.rows, replayed)
    return full


def _gather_publication(
    published: "CandidatePublication",
    rows: torch.Tensor,
) -> "CandidatePublication":
    """Re-index a query-row candidate carrier onto the replayed rows.

    ``rows`` is set to the gathered height rather than carried over from the
    producer: ``_check_candidate_rows`` asserts that every row a consumer masked
    is a row the producer wrote, and after the boundary the consumers mask
    exactly this many.
    """
    from .candidate_prefilter import CandidatePublication

    return CandidatePublication(
        blocks=published.blocks.index_select(0, rows),
        compress_ratio=published.compress_ratio,
        rows=int(rows.shape[0]),
    )
