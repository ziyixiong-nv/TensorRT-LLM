# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential tests for DeepSeek-V4.1's two-level candidate prefilter.

`_reference_candidate_mask` below is a transcription of the reference
implementation (`inference/model.py::select_candidate_blocks` in the
DeepSeek-V4.1-Flash release). It is transcribed rather than imported because
importing the reference pulls in tilelang and therefore a GPU and a pinned
`tvm_ffi`, neither of which a unit test may require. The transcription itself is
checked against the reference's own bytes by
`DS/workspace/deepseek-v41-bringup/gates/verify_candidate_blocks.py`, which
extracts the function by text and runs the same characterisation on CPU; that
gate is what makes trusting this copy legitimate.

Everything here runs on CPU with tensors of a few hundred elements, so it is
cheap enough to sit in the default unit-test lane.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.candidate_prefilter import (
    CandidatePublication,
    apply_candidate_blocks,
    candidate_prefilter_is_inert,
    candidate_prefilter_is_on,
    inert_up_to_positions,
    select_candidate_blocks,
)
from tensorrt_llm._torch.configs.deepseek_v41 import build_layer_descriptors
from tensorrt_llm._torch.models.modeling_deepseekv41 import (
    _assert_candidate_prefilter_inert,
    _candidate_prefilter_is_wired,
    _candidate_source_ratio,
)
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

from ..dsa.test_dsa_indexer import create_indexer

pytestmark = pytest.mark.cpu_only


def _reference_candidate_mask(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """The reference algorithm, transcribed. Returns a dense bool mask."""
    width = logits.size(-1)
    pad = -width % block_size
    padded = F.pad(logits, (0, pad), value=float("-inf"))
    scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1).div(block_size, rounding_mode="floor")
    block_ar = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_ar.unsqueeze(0) == last.unsqueeze(1), float("inf"))
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _mask_from_block_ids(
    block_ids: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    width: int,
    block_size: int,
) -> torch.Tensor:
    """Turn the block-id carrier back into a dense mask, via level two."""
    logits = torch.zeros((block_ids.shape[0], width))
    masked = apply_candidate_blocks(logits, block_ids, row_starts, row_ends, block_size)
    return masked == 0.0


def _premask_unreachable(logits: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """The caller's own precondition: unreachable positions arrive at -inf."""
    ar = torch.arange(logits.size(-1), device=logits.device)
    return logits.masked_fill(ar.unsqueeze(0) >= lens.unsqueeze(1), float("-inf"))


def _reference_effective_mask(
    logits: torch.Tensor,
    lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """What the reference's consumer actually sees, which is what must match.

    The reference's raw mask is *not* reachable-tight -- blocks are kept whole,
    so a partially filled newest block carries its unreachable positions along.
    Those positions are nonetheless dead on arrival, because the reference
    applies the mask to an `index_score` whose unreachable entries are already
    `-inf`. `apply_candidate_blocks` folds that second condition in directly, so
    the comparable quantity is the raw mask AND reachability. Comparing against
    the raw mask instead would report a difference that no consumer can observe.
    """
    raw = _reference_candidate_mask(logits, lens, topk_blocks, block_size)
    ar = torch.arange(logits.size(-1), device=logits.device)
    return raw & (ar.unsqueeze(0) < lens.unsqueeze(1))


# Widths that are and are not multiples of block_size, and lengths that land
# mid-block, on a boundary, and at both extremes of the row.
@pytest.mark.parametrize("width,block_size", [(32, 4), (30, 4), (33, 8), (64, 8), (17, 3)])
@pytest.mark.parametrize("topk_blocks", [1, 2, 3, 5, 99])
def test_matches_reference_dense_mask(width, block_size, topk_blocks):
    """Level one + level two reproduce the reference's dense mask exactly."""
    torch.manual_seed(0x4109 ^ width ^ (block_size << 8) ^ (topk_blocks << 16))
    num_blocks = (width + block_size - 1) // block_size
    lens = torch.tensor(
        [1, block_size, block_size + 1, width // 2, width - 1, width][: max(2, num_blocks)],
        dtype=torch.int32,
    ).clamp_(min=1, max=width)
    logits = _premask_unreachable(torch.randn(lens.numel(), width), lens)
    starts = torch.zeros_like(lens)

    want = _reference_effective_mask(logits, lens, topk_blocks, block_size)
    ids = select_candidate_blocks(logits, starts, lens, topk_blocks, block_size)
    got = _mask_from_block_ids(ids, starts, lens, width, block_size)
    assert torch.equal(got, want), (
        f"mask mismatch at width={width} block_size={block_size} topk_blocks={topk_blocks}\n"
        f"lens={lens.tolist()}\ngot  ={got.int().tolist()}\nwant ={want.int().tolist()}"
    )


def test_newest_block_is_pinned_despite_worst_score():
    """The partially filled newest block must survive an otherwise losing score.

    Without the pin a decode step can drop its own most recent tokens, which is
    the one failure mode that would not show up as an obvious accuracy cliff.
    """
    block_size, num_blocks = 4, 8
    width = block_size * num_blocks
    logits = torch.zeros(1, width)
    logits[0, :28] = 10.0  # blocks 0..6
    logits[0, 28:30] = -10.0  # block 7's two reachable positions
    lens = torch.tensor([30], dtype=torch.int32)
    logits = _premask_unreachable(logits, lens)

    ids = select_candidate_blocks(logits, torch.zeros_like(lens), lens, 3, block_size)
    kept = sorted(i for i in ids[0].tolist() if i >= 0)
    assert 7 in kept, f"newest block dropped: kept {kept}"
    assert len(kept) == 3, f"pin inflated the kept count to {len(kept)}: {kept}"


def test_unreachable_blocks_are_dropped_not_padded():
    """Asking for more blocks than exist yields -1 padding, not junk ids."""
    block_size, width = 4, 32
    lens = torch.tensor([10], dtype=torch.int32)  # blocks 0,1,2 (2 partial)
    logits = _premask_unreachable(torch.randn(1, width), lens)

    ids = select_candidate_blocks(logits, torch.zeros_like(lens), lens, 6, block_size)
    kept = [i for i in ids[0].tolist() if i >= 0]
    assert sorted(kept) == [0, 1, 2], f"expected blocks [0,1,2], got {ids[0].tolist()}"
    # -1 padding sorts to the end, so the carrier can be read as a prefix.
    assert ids[0].tolist()[: len(kept)] == sorted(kept), (
        f"padding is not trailing: {ids[0].tolist()}"
    )


def test_kept_set_is_a_superset_not_reachable_tight():
    """Blocks are kept whole, so the newest one drags unreachable positions in.

    This is the property that was *falsified* when the reference was measured
    (the design doc originally claimed the mask was reachable-tight). It is only
    safe because reachability is enforced twice elsewhere; an implementation
    that trusted the mask and dropped the downstream `idxs < compress_lens`
    clamp would select phantom positions past end-of-sequence. Pinning it here
    so that "optimisation" fails loudly.
    """
    block_size, width = 4, 32
    lens = torch.tensor([10], dtype=torch.int32)
    logits = _premask_unreachable(torch.randn(1, width), lens)
    starts = torch.zeros_like(lens)

    ids = select_candidate_blocks(logits, starts, lens, 6, block_size)
    mask = _mask_from_block_ids(ids, starts, lens, width, block_size)
    # `apply_candidate_blocks` also masks by row bounds, so read the block set
    # directly to see the overshoot the *mask itself* would allow.
    assert 2 in ids[0].tolist(), "the partial newest block should be kept"
    block_end = ((int(lens[0]) - 1) // block_size + 1) * block_size
    assert block_end == 12
    # Everything past the newest block's own boundary is excluded.
    assert not bool(mask[0, block_end:].any()), f"overshoot past block boundary: {mask[0].tolist()}"


def test_blocks_are_anchored_at_row_start_not_column_zero():
    """A row whose reachable span does not start at 0 keeps its own block grid.

    In a multi-request prefill chunk the logits' column 0 is the *chunk's* first
    compressed KV row, so the second request's positions begin at a
    non-multiple-of-block_size offset. Anchoring at column 0 would shift every
    boundary for that request; this test is the guard against that regression.
    """
    block_size = 4
    offset = 6  # deliberately not a multiple of block_size
    inner_width, lens_val = 20, 13
    torch.manual_seed(0x41A0)

    inner = _premask_unreachable(
        torch.randn(1, inner_width), torch.tensor([lens_val], dtype=torch.int32)
    )
    want = _reference_effective_mask(
        inner, torch.tensor([lens_val], dtype=torch.int32), 2, block_size
    )

    shifted = torch.full((1, offset + inner_width), float("-inf"))
    shifted[0, offset:] = inner[0]
    starts = torch.tensor([offset], dtype=torch.int32)
    ends = torch.tensor([offset + lens_val], dtype=torch.int32)
    ids = select_candidate_blocks(shifted, starts, ends, 2, block_size)
    got = _mask_from_block_ids(ids, starts, ends, shifted.size(-1), block_size)

    assert torch.equal(got[0, offset : offset + inner_width], want[0]), (
        "shifted row does not match the zero-anchored reference: "
        f"got {got[0, offset : offset + inner_width].int().tolist()} want {want[0].int().tolist()}"
    )
    assert not bool(got[0, :offset].any()), "columns before row_start must not be kept"


def test_rows_are_independent():
    """One row's block grid must not leak into another's."""
    block_size, width = 4, 32
    starts = torch.tensor([0, 5, 13], dtype=torch.int32)
    ends = torch.tensor([9, 25, 32], dtype=torch.int32)
    torch.manual_seed(0x41B0)
    logits = torch.randn(3, width)

    ids = select_candidate_blocks(logits, starts, ends, 2, block_size)
    for r in range(3):
        one = select_candidate_blocks(
            logits[r : r + 1], starts[r : r + 1], ends[r : r + 1], 2, block_size
        )
        assert torch.equal(ids[r : r + 1], one), f"row {r} depends on its neighbours"


def test_inert_regime_keeps_every_reachable_position():
    """`num_blocks <= topk_blocks` must be a provable no-op, not nearly one.

    This is what lets the runtime skip both levels below the configured ceiling
    and still claim bit-identical output.
    """
    block_size, width = 4, 32
    num_blocks = width // block_size
    for lens_val in (1, 7, 16, 29, 32):
        lens = torch.tensor([lens_val], dtype=torch.int32)
        starts = torch.zeros_like(lens)
        logits = _premask_unreachable(torch.randn(1, width), lens)
        assert candidate_prefilter_is_inert(starts, lens, num_blocks, block_size)
        ids = select_candidate_blocks(logits, starts, lens, num_blocks, block_size)
        mask = _mask_from_block_ids(ids, starts, lens, width, block_size)
        assert bool(mask[0, :lens_val].all()), (
            f"inert regime dropped a reachable position at lens={lens_val}: {mask[0].tolist()}"
        )


def test_is_inert_is_per_row_not_per_batch():
    """A batch mixing a short row with a long one is not inert."""
    block_size, topk_blocks = 4, 2  # inert only up to 8 positions
    starts = torch.tensor([0, 0], dtype=torch.int32)
    assert candidate_prefilter_is_inert(
        starts, torch.tensor([5, 8], dtype=torch.int32), topk_blocks, block_size
    )
    assert not candidate_prefilter_is_inert(
        starts, torch.tensor([5, 9], dtype=torch.int32), topk_blocks, block_size
    )


def test_empty_row_selects_nothing():
    """A row with no reachable positions must publish only padding.

    Reachable in practice: a pooled (ratio 2) layer whose context rounded down
    to zero compressed rows.
    """
    block_size, width = 4, 16
    starts = torch.tensor([3], dtype=torch.int32)
    ends = torch.tensor([3], dtype=torch.int32)
    ids = select_candidate_blocks(torch.randn(1, width), starts, ends, 2, block_size)
    assert ids[0].tolist() == [-1, -1], f"empty row published {ids[0].tolist()}"
    mask = _mask_from_block_ids(ids, starts, ends, width, block_size)
    assert not bool(mask.any())


def test_level_two_masks_with_finite_sentinel():
    """The value written back must be finite, so no later kernel can make a NaN."""
    block_size, width = 4, 16
    starts = torch.tensor([0], dtype=torch.int32)
    ends = torch.tensor([width], dtype=torch.int32)
    logits = torch.randn(1, width)
    ids = select_candidate_blocks(logits, starts, ends, 1, block_size)
    out = apply_candidate_blocks(logits, ids, starts, ends, block_size)
    assert torch.isfinite(out).all(), f"non-finite mask value in {out.tolist()}"
    assert (out == torch.finfo(logits.dtype).min).any(), "nothing was masked"


def test_level_two_out_parameter_matches_functional_form():
    block_size, width = 4, 16
    starts = torch.tensor([0], dtype=torch.int32)
    ends = torch.tensor([13], dtype=torch.int32)
    logits = torch.randn(1, width)
    ids = select_candidate_blocks(logits, starts, ends, 2, block_size)
    out = torch.empty_like(logits)
    assert torch.equal(apply_candidate_blocks(logits, ids, starts, ends, block_size, out=out), out)
    assert torch.equal(out, apply_candidate_blocks(logits, ids, starts, ends, block_size))


def test_release_shape_inert_boundary_is_exact():
    """2048 x 8 is inert at exactly 16384 positions and not one block further.

    The refusal in `_assert_candidate_prefilter_inert` is stated as an exact
    bound rather than a conservative one, so the boundary is worth pinning: the
    release's candidate source is layer 20 at compress ratio 1, where compressed
    and raw positions coincide.
    """
    assert inert_up_to_positions(2048, 8) == 16384
    # A pooled source would stay inert to twice the raw length.
    assert inert_up_to_positions(2048, 8, compress_ratio=2) == 32768

    starts = torch.zeros(1, dtype=torch.int32)
    assert candidate_prefilter_is_inert(starts, torch.tensor([16384], dtype=torch.int32), 2048, 8)
    assert not candidate_prefilter_is_inert(
        starts, torch.tensor([16385], dtype=torch.int32), 2048, 8
    )


def test_release_shape_tensor_ops_are_the_exact_identity():
    """At 2048 x 8 the two levels are bit-identical, not merely mask-equivalent.

    ``test_release_shape_inert_boundary_is_exact`` pins the *arithmetic* of the
    bound and ``test_inert_regime_keeps_every_reachable_position`` pins the mask
    at a toy 32-column shape; neither runs the ops at the width the release
    actually configures. This does, because ``_assert_candidate_prefilter_inert``
    states its ceiling as an exact promise about output equality, and 2048 x 8 is
    where an off-by-one in ``num_blocks`` or in ``min(topk_blocks, num_blocks)``
    would first bite -- at 32 columns those two expressions are too small to
    disagree in an interesting way.

    Exactness is claimed only for rows reachable across the whole tile. A ragged
    row's unreachable tail arrives at ``-inf`` from the indexer and leaves at
    ``finfo.min``: a different bit pattern for the same meaning, which the
    downstream top-k excludes by its own row bounds either way. See
    :func:`test_release_shape_identity_holds_on_a_ragged_batch` for that half.
    """
    block_size, topk_blocks = CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS
    width = inert_up_to_positions(topk_blocks, block_size)
    assert width == 16384, "the release shape moved; the constants above are stale"

    starts = torch.zeros(2, dtype=torch.int32)
    ends = torch.full((2,), width, dtype=torch.int32)
    torch.manual_seed(0x4101)
    logits = torch.randn(2, width)

    ids = select_candidate_blocks(logits, starts, ends, topk_blocks, block_size)
    # Every block kept, so the carrier is the identity permutation with no
    # padding -- a stronger statement than "the mask is all True", because it
    # also rules out a duplicate id standing in for a dropped one.
    assert torch.equal(ids, torch.arange(topk_blocks, dtype=torch.int32).expand(2, -1)), (
        "a fully reachable inert row did not publish every block exactly once"
    )

    out = apply_candidate_blocks(logits, ids, starts, ends, block_size)
    changed = int((out != logits).sum())
    assert changed == 0, f"{changed} of {logits.numel()} logits changed inside the inert bound"


def test_release_shape_identity_holds_on_a_ragged_batch():
    """Below the bound, every *reachable* logit survives whatever the row lengths.

    The batch mixes a row shorter than one block, one that ends mid-block, and
    one that fills the tile, because level one's block grid is anchored per row:
    a bug that shifted the grid would leave short rows intact and only corrupt
    the long one, or vice versa.
    """
    block_size, topk_blocks = CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS
    width = inert_up_to_positions(topk_blocks, block_size)
    lens = torch.tensor([1, 7, 8193, width], dtype=torch.int32)
    starts = torch.zeros_like(lens)
    torch.manual_seed(0x4102)
    logits = _premask_unreachable(torch.randn(lens.numel(), width), lens)

    ids = select_candidate_blocks(logits, starts, lens, topk_blocks, block_size)
    out = apply_candidate_blocks(logits, ids, starts, lens, block_size)
    neg_inf = torch.finfo(logits.dtype).min
    for r, n in enumerate(lens.tolist()):
        assert torch.equal(out[r, :n], logits[r, :n]), (
            f"row {r} (len {n}) changed inside its reachable span"
        )
        assert bool((out[r, n:] == neg_inf).all()), f"row {r} left an unreachable column unmasked"


def test_one_position_past_the_bound_drops_exactly_one_block():
    """The bound must be tight from above too, or the refusal is over-strict.

    One column past ``inert_up_to_positions`` there are 2049 blocks and room for
    2048, so exactly one is cut. Pinning *which* one -- via a monotone ramp that
    makes block 0 the unique worst scorer -- turns "the identity broke" into
    "the identity broke where the algorithm says it should", which is what
    distinguishes a tight bound from an accidental one.
    """
    block_size, topk_blocks = CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS
    width = inert_up_to_positions(topk_blocks, block_size) + 1
    starts = torch.zeros(1, dtype=torch.int32)
    ends = torch.full((1,), width, dtype=torch.int32)
    # Block b scores 8b + 7, so block 0 loses; the newest block is pinned to
    # +inf and survives despite being one column wide.
    logits = torch.arange(width, dtype=torch.float32).unsqueeze(0)

    ids = select_candidate_blocks(logits, starts, ends, topk_blocks, block_size)
    assert ids[0].tolist() == list(range(1, topk_blocks + 1)), (
        "the block that was cut is not the worst-scoring one"
    )

    out = apply_candidate_blocks(logits, ids, starts, ends, block_size)
    neg_inf = torch.finfo(logits.dtype).min
    assert bool((out[0, :block_size] == neg_inf).all()), "block 0 survived the cut"
    assert torch.equal(out[0, block_size:], logits[0, block_size:]), "a kept block was masked"


def test_level_two_aliases_its_input_the_way_production_calls_it():
    """``out=logits`` is the production call, so the self-copy must be harmless.

    ``Indexer._apply_candidate_blocks`` passes the live logits as both source and
    destination, which makes ``out.copy_(logits)`` a self-copy.
    ``test_level_two_out_parameter_matches_functional_form`` uses a separate
    destination and therefore cannot see an aliasing bug at all. The shape here
    masks five of eight blocks, so a read-after-write fault would show as a wrong
    value rather than as no difference.
    """
    block_size, topk_blocks, width = 8, 3, 64
    starts = torch.zeros(2, dtype=torch.int32)
    lens = torch.tensor([width, 41], dtype=torch.int32)
    torch.manual_seed(0x4103)
    logits = _premask_unreachable(torch.randn(2, width), lens)

    ids = select_candidate_blocks(logits, starts, lens, topk_blocks, block_size)
    want = apply_candidate_blocks(logits, ids, starts, lens, block_size)
    assert bool((want == torch.finfo(want.dtype).min).any()), (
        "nothing was masked, so this shape does not exercise aliasing"
    )

    live = logits.clone()
    got = apply_candidate_blocks(live, ids, starts, lens, block_size, out=live)
    assert got.data_ptr() == live.data_ptr(), "production relies on this being in place"
    assert torch.equal(got, want)


def test_bf16_logits_round_trip():
    """The indexer's logits are not always fp32."""
    block_size, width = 8, 64
    starts = torch.zeros(1, dtype=torch.int32)
    ends = torch.tensor([50], dtype=torch.int32)
    logits = torch.randn(1, width, dtype=torch.bfloat16)
    ids = select_candidate_blocks(logits, starts, ends, 3, block_size)
    kept = [i for i in ids[0].tolist() if i >= 0]
    assert len(kept) == 3 and 6 in kept  # block 6 holds position 49
    out = apply_candidate_blocks(logits, ids, starts, ends, block_size)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="one entry per logits row"):
        select_candidate_blocks(
            torch.randn(2, 16),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            2,
            4,
        )
    with pytest.raises(ValueError, match=r"\[num_rows, width\]"):
        select_candidate_blocks(
            torch.randn(2, 3, 16),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            2,
            4,
        )
    with pytest.raises(ValueError, match="must be positive"):
        select_candidate_blocks(
            torch.randn(1, 16),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            0,
            4,
        )
    with pytest.raises(ValueError, match="one row per logits row"):
        apply_candidate_blocks(
            torch.randn(2, 16),
            torch.zeros(1, 2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            4,
        )


# ---------------------------------------------------------------------------
# Wiring tests: the primitives above are correct in isolation; these pin that
# the runtime actually calls them, for the right layers, with matching column
# spaces, and that every row a consumer masks was really published.
#
# The layer bands below are transcribed from the release checkpoint's
# `text_config` rather than read from it, because unit tests run in CI without
# model weights. That the transcription still matches the live checkpoint is
# checked on real bytes by `DS/workspace/deepseek-v41-bringup/gates/
# candidate_roles.py` check A. So this file pins the *logic* and that gate pins
# the *configuration*; neither is asked to do the other's job.
# ---------------------------------------------------------------------------

NUM_HIDDEN_LAYERS = 40
NUM_NEXTN_PREDICT_LAYERS = 3
# layers 0-1 -> 0 (SWA only), 2-19 -> 2 (pooled), 20-39 -> 1 (unpooled), MTP -> 0.
RELEASE_COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0] * NUM_NEXTN_PREDICT_LAYERS
KV_SOURCE_LAYER_IDS = [2, 8, 14, 20]
INDEX_SOURCE_LAYER_IDS = [2, 8, 14, 20, 24, 28, 32, 36]
ENGRAM_LAYER_IDS = [1, 14]
CANDIDATE_SOURCE_LAYER_ID = 20
CANDIDATE_TOPK_BLOCKS = 2048
CANDIDATE_BLOCK_SIZE = 8
ROPE_THETA = 10000.0
COMPRESS_ROPE_THETA = 160000.0
WINDOW_SIZE = 128


def _release_descriptors(candidate_source=CANDIDATE_SOURCE_LAYER_ID):
    return build_layer_descriptors(
        compress_ratios=list(RELEASE_COMPRESS_RATIOS),
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        rope_theta=ROPE_THETA,
        compress_rope_theta=COMPRESS_ROPE_THETA,
        window_size=WINDOW_SIZE,
        kv_source_layer_ids=list(KV_SOURCE_LAYER_IDS),
        index_source_layer_ids=list(INDEX_SOURCE_LAYER_IDS),
        engram_layer_ids=list(ENGRAM_LAYER_IDS),
        candidate_source_layer_id=candidate_source,
    )


def _release_sparse_config():
    return DeepSeekV4SparseAttentionConfig(
        variant="v41",
        compress_ratios=list(RELEASE_COMPRESS_RATIOS),
        kv_source_layer_ids=list(KV_SOURCE_LAYER_IDS),
        index_source_layer_ids=list(INDEX_SOURCE_LAYER_IDS),
        window_size=WINDOW_SIZE,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
    )


def _pretrained(
    candidate_source=CANDIDATE_SOURCE_LAYER_ID,
    topk_blocks=CANDIDATE_TOPK_BLOCKS,
    block_size=CANDIDATE_BLOCK_SIZE,
    compress_ratios=None,
):
    """Stand-in for ``DeepseekV41Config``.

    A namespace suffices because ``to_sparse_params`` reads these with
    ``getattr`` and the real composite config's ``__getattr__`` forwards any
    public name to ``text_config`` -- so the attribute is reachable at the top
    level on the real object too, and the test is not relying on a shape the
    production path does not have.
    """
    return SimpleNamespace(
        candidate_source_layer_id=candidate_source,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=block_size,
        compress_ratios=(
            list(RELEASE_COMPRESS_RATIOS) if compress_ratios is None else list(compress_ratios)
        ),
    )


def _indexer(layer_idx, pretrained=None, compress_ratio=None):
    """Build an Indexer the way ``deepseek_v4/backend.py:174-185`` does.

    The ratio defaults to the layer's own, as production's does
    (``backend.py:110``), rather than to the base class's ``= 1``.
    """
    if compress_ratio is None:
        compress_ratio = RELEASE_COMPRESS_RATIOS[layer_idx]
    return create_indexer(
        _release_sparse_config(),
        layer_idx=layer_idx,
        pretrained_config=_pretrained() if pretrained is None else pretrained,
        compress_ratio=compress_ratio,
    )


class TestIndexerCandidateRoles:
    """The Indexer derives its prefilter role instead of being handed it."""

    def test_indexer_candidate_roles_match_descriptors(self):
        """The two derivations agree on every layer that builds an Indexer.

        The Indexer drops the descriptor's ``is_index_source`` term from
        ``consumes_candidates`` on the grounds that only index sources construct
        an ``Indexer`` at all -- so the equivalence is asserted exactly on those
        layers, and the layers that build no Indexer are checked separately
        below.
        """
        descriptors = {d.layer_idx: d for d in _release_descriptors()}
        checked = 0
        for layer_idx in INDEX_SOURCE_LAYER_IDS:
            indexer = _indexer(layer_idx)
            desc = descriptors[layer_idx]
            assert indexer.is_candidate_source == desc.is_candidate_source, (
                f"layer {layer_idx}: indexer says is_candidate_source="
                f"{indexer.is_candidate_source}, descriptor says {desc.is_candidate_source}"
            )
            assert indexer.consumes_candidates == desc.consumes_candidates, (
                f"layer {layer_idx}: indexer says consumes_candidates="
                f"{indexer.consumes_candidates}, descriptor says {desc.consumes_candidates}"
            )
            checked += 1
        assert checked == len(INDEX_SOURCE_LAYER_IDS)

    def test_exactly_one_source_and_the_expected_consumers(self):
        """Pins the roles to values, not just to agreement.

        Without this, ``test_..._match_descriptors`` would pass if *both*
        derivations were False everywhere -- which is precisely what a wrong
        ``candidate_source_layer_id`` would produce. Two derivations that agree
        on nothing are not a check.
        """
        sources, consumers = [], []
        for layer_idx in INDEX_SOURCE_LAYER_IDS:
            indexer = _indexer(layer_idx)
            if indexer.is_candidate_source:
                sources.append(layer_idx)
            if indexer.consumes_candidates:
                consumers.append(layer_idx)
        assert sources == [CANDIDATE_SOURCE_LAYER_ID]
        assert consumers == [24, 28, 32, 36]
        # A layer is never both: level one's ids are row-local to the producer's
        # own `cu_seqlen_ks`, so a self-consuming layer would mask its scores
        # with a set derived from the very scores being masked.
        assert not set(sources) & set(consumers)

    def test_consumers_share_the_source_compress_ratio(self):
        """The carrier's block ids are row-local, so column spaces must match.

        This holds in the release because the ratio-2 index sources (2, 8, 14)
        all *precede* layer 20 and are therefore not consumers -- but it is a
        property of the checkpoint, not of the code, so it is worth failing
        loudly on a future one.
        """
        descriptors = {d.layer_idx: d for d in _release_descriptors()}
        source_ratio = descriptors[CANDIDATE_SOURCE_LAYER_ID].compress_ratio
        for layer_idx in INDEX_SOURCE_LAYER_IDS:
            if _indexer(layer_idx).consumes_candidates:
                assert descriptors[layer_idx].compress_ratio == source_ratio, (
                    f"layer {layer_idx} consumes layer {CANDIDATE_SOURCE_LAYER_ID}'s "
                    f"candidate blocks at ratio {descriptors[layer_idx].compress_ratio}, "
                    f"but they were published in ratio-{source_ratio} column space"
                )

    def test_non_index_source_layers_never_consume(self):
        """The half of the simplification the Indexer cannot check for itself."""
        index_sources = set(INDEX_SOURCE_LAYER_IDS)
        for desc in _release_descriptors():
            if desc.layer_idx not in index_sources:
                assert not desc.consumes_candidates, (
                    f"layer {desc.layer_idx} builds no Indexer but its descriptor asks it "
                    "to mask with candidate blocks; the dropped is_index_source term would "
                    "then be unsound"
                )
                assert not desc.is_candidate_source

    @pytest.mark.parametrize(
        "kwargs, why",
        [
            (dict(candidate_source=None), "no source layer configured"),
            (dict(topk_blocks=0), "topk_blocks 0"),
            (dict(block_size=0), "block_size 0"),
        ],
    )
    def test_roles_are_off_when_the_config_is_partial(self, kwargs, why):
        """A partial config disables the feature rather than half-enabling it.

        A source with ``topk_blocks=0`` would publish a zero-width carrier and
        every consumer would mask everything away -- all logits at
        ``finfo.min``, an arbitrary top-k, no error.
        ``inert_up_to_positions`` already guards ``<= 0`` by returning 0; this
        pins that the *role* derivation refuses the same inputs rather than
        relying on that.
        """
        pretrained = _pretrained(**kwargs)
        for layer_idx in (CANDIDATE_SOURCE_LAYER_ID, 24):
            indexer = _indexer(layer_idx, pretrained=pretrained)
            assert not indexer.is_candidate_source, why
            assert not indexer.consumes_candidates, why

    def test_shared_predicate_is_the_one_the_indexer_uses(self):
        """`candidate_prefilter_is_on` exists so two callers cannot disagree.

        The model's length refusal and the Indexer's role derivation ask the same
        question of the same three values. Pinning that the Indexer's answer is
        literally this function's is what makes the model's use of it evidence
        about the Indexer.
        """
        for layer_idx in INDEX_SOURCE_LAYER_IDS:
            indexer = _indexer(layer_idx)
            assert indexer._candidate_prefilter_enabled == candidate_prefilter_is_on(
                indexer.candidate_source_layer_id,
                indexer.candidate_topk_blocks,
                indexer.candidate_block_size,
            )


def _publication(*, rows: int, compress_ratio: int = 1) -> CandidatePublication:
    """A publication carrying only the row counter the guard under test reads.

    The blocks themselves are never dereferenced by ``_check_candidate_rows`` --
    it compares counters -- so an empty carrier keeps these cases about the
    invariant rather than about the mask.
    """
    return CandidatePublication(
        blocks=torch.empty((0, 0), dtype=torch.int32),
        compress_ratio=compress_ratio,
        rows=rows,
    )


class TestCandidateCarrierGuards:
    """The two ways a correct primitive can still be wired wrongly."""

    def _metadata_stub(self):
        """Just the publication carrier and the consumer-side counter."""
        return SimpleNamespace(
            num_tokens=8,
            v41_candidate_blocks={},
            v41_candidate_rows_consumed={},
        )

    @pytest.mark.parametrize("layer_idx, expected_ratio", [(2, 2), (14, 2), (20, 1), (24, 1)])
    def test_candidate_ratio_is_the_layers_own(self, layer_idx, expected_ratio):
        """The recorded ratio is the *layer's*, not a constant.

        Without this the mismatch test below is decorative: the base ``Indexer``
        defaults ``compress_ratio`` to 1 and every consumer in the release is
        ratio 1, so a helper that never forwarded a ratio would still produce a
        passing mismatch test. This pins the value, on the two ratio-2 layers
        where a constant 1 would show.
        """
        indexer = _indexer(layer_idx)
        assert indexer._candidate_ratio == expected_ratio
        assert indexer.compress_ratio == expected_ratio

    def test_ratio_mismatch_is_rejected(self):
        """A ratio mismatch cannot happen in this release -- hence the test.

        An assertion whose condition is unreachable in every configuration
        anyone runs is indistinguishable from an assertion that is wrong, and
        the cost of finding out on a future checkpoint is a silent accuracy loss
        rather than a crash.
        """
        consumer = _indexer(24)
        assert consumer._candidate_ratio == 1, (
            "the mismatch below is only a mismatch if the consumer really holds ratio 1; "
            "see test_candidate_ratio_is_the_layers_own"
        )
        metadata = self._metadata_stub()
        source = CANDIDATE_SOURCE_LAYER_ID
        metadata.v41_candidate_blocks[source] = CandidatePublication(
            blocks=torch.full((8, 4), -1, dtype=torch.int32),
            compress_ratio=2,  # producer pooled; consumer is ratio 1
            rows=8,
        )

        logits = torch.randn(8, 32)
        row_starts = torch.zeros(8, dtype=torch.int32)
        row_ends = torch.full((8,), 32, dtype=torch.int32)
        with pytest.raises(AssertionError, match="ratio"):
            consumer._apply_candidate_blocks(metadata, logits, row_starts, row_ends, 0)

    def test_apply_rejects_an_unpublished_source(self):
        """Consuming before the producer ran is a wiring bug, not a fallback.

        The carrier is a dict, so a missing entry is the natural shape of "the
        producer never ran" -- and `-1` means "keep nothing", so silently
        skipping the mask is not a safe default either way.
        """
        consumer = _indexer(24)
        metadata = self._metadata_stub()
        logits = torch.randn(8, 32)
        row_starts = torch.zeros(8, dtype=torch.int32)
        row_ends = torch.full((8,), 32, dtype=torch.int32)
        with pytest.raises(AssertionError, match="candidate blocks"):
            consumer._apply_candidate_blocks(metadata, logits, row_starts, row_ends, 0)

    def test_guard_fires_when_a_consumer_outruns_the_producer(self):
        """Drives the counter invariant directly, without a q-split forward.

        The guard exists because under q-split each rank publishes only its own
        query slice, and an unwritten row is all ``-1``, i.e. keep nothing, i.e.
        an arbitrary top-k with no error. It is the invariant, not any one
        scenario, that the guard claims to protect.
        """
        consumer = _indexer(24)
        metadata = self._metadata_stub()
        source = CANDIDATE_SOURCE_LAYER_ID
        metadata.v41_candidate_blocks[source] = _publication(rows=8)
        metadata.v41_candidate_rows_consumed[24] = 5  # three rows never published
        with pytest.raises(AssertionError, match="published"):
            consumer._check_candidate_rows(metadata)

    def test_guard_is_not_vacuous(self):
        """The counters must actually be maintained, or the guard passes on all.

        An equality assertion between two integers that are always zero is not a
        check. So a matched, *nonzero* pair must pass and an all-zero pair must
        not be mistaken for a matched one -- the latter means the producer never
        ran, which is impossible for a wired consumer.
        """
        consumer = _indexer(24)
        source = CANDIDATE_SOURCE_LAYER_ID

        matched = self._metadata_stub()
        matched.v41_candidate_blocks[source] = _publication(rows=8)
        matched.v41_candidate_rows_consumed[24] = 8
        consumer._check_candidate_rows(matched)  # must not raise

        never_ran = self._metadata_stub()  # both absent -> both default 0
        with pytest.raises(AssertionError):
            consumer._check_candidate_rows(never_ran)

    def test_guard_is_silent_for_a_non_consumer(self):
        """The source layer publishes and consumes nothing, so it must not check."""
        producer = _indexer(CANDIDATE_SOURCE_LAYER_ID)
        assert producer.is_candidate_source and not producer.consumes_candidates
        producer._check_candidate_rows(self._metadata_stub())  # must not raise

    def test_publish_then_apply_round_trips(self):
        """The carrier hop, not the algorithm.

        The primitives are already proven against the reference's own bytes. What
        is unproven is the hop -- lazy allocation to ``[num_tokens, keep]``,
        left-alignment of a narrower ``select_candidate_blocks`` output, ``-1``
        re-padding of the tail, and the global row offset.
        """
        producer = _indexer(CANDIDATE_SOURCE_LAYER_ID)
        consumer = _indexer(24)
        metadata = self._metadata_stub()

        rows, width = 4, 32
        logits = torch.randn(rows, width)
        row_starts = torch.zeros(rows, dtype=torch.int32)
        row_ends = torch.full((rows,), width, dtype=torch.int32)

        producer._publish_candidate_blocks(metadata, logits, row_starts, row_ends, 0)
        publication = metadata.v41_candidate_blocks[CANDIDATE_SOURCE_LAYER_ID]
        published = publication.blocks
        assert published.shape[0] == metadata.num_tokens, (
            "the carrier is allocated over num_tokens, not over this call's row count -- "
            "decode rows live at [num_ctx_tokens, num_tokens)"
        )
        assert published.dtype == torch.int32
        assert publication.rows == rows
        assert publication.compress_ratio == 1
        # Rows this call did not write stay -1, which is what the row guard
        # exists to notice.
        assert bool((published[rows:] == -1).all())

        masked = consumer._apply_candidate_blocks(metadata, logits.clone(), row_starts, row_ends, 0)
        assert masked.shape == logits.shape
        assert metadata.v41_candidate_rows_consumed[24] == rows
        # At 2048x8 the release is inert, and this stub is 4 blocks of 8 over
        # width 32 with topk_blocks=2048, so every block is kept and the mask is
        # the identity. That is the round trip succeeding, not the mask doing
        # nothing interesting: a carrier that failed to round-trip would show up
        # as `finfo.min` rows, not as equality.
        assert torch.equal(masked, logits)

    def test_publish_writes_rows_at_the_global_offset(self):
        """A decode publish lands at `token_offset`, not at row 0.

        Prefill tiles and decode rows share one carrier indexed by global token,
        so an offset dropped anywhere in the chain would overwrite the context
        rows with generation rows -- self-consistent, and wrong.
        """
        producer = _indexer(CANDIDATE_SOURCE_LAYER_ID)
        metadata = self._metadata_stub()

        rows, width, offset = 3, 32, 5
        logits = torch.randn(rows, width)
        row_starts = torch.zeros(rows, dtype=torch.int32)
        row_ends = torch.full((rows,), width, dtype=torch.int32)
        producer._publish_candidate_blocks(metadata, logits, row_starts, row_ends, offset)

        publication = metadata.v41_candidate_blocks[CANDIDATE_SOURCE_LAYER_ID]
        published = publication.blocks
        assert bool((published[:offset] == -1).all()), "rows before the offset were written"
        assert bool((published[offset : offset + rows] >= 0).any()), "the offset rows are empty"
        assert publication.rows == rows


class _UnwiredSparseConfig:
    """A sparse config that drops the three candidate values at lowering.

    Models the one disagreement that matters: a checkpoint carrying the fields
    against a ``to_sparse_params`` that does not forward them. The Indexer reads
    the lowered params, so in this state it runs neither level -- and the length
    refusal must therefore still apply.
    """

    def to_sparse_params(self, **kwargs):
        return SimpleNamespace()


def _model_config(sparse_config, pretrained, max_seq_len):
    return SimpleNamespace(
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained,
        max_seq_len=max_seq_len,
    )


class TestCandidatePrefilterLengthRefusal:
    """The refusal is now conditional on the wiring, and its bound is compressed."""

    def test_wired_config_is_not_refused_at_any_length(self):
        """The point of the whole patch: a wired build has no length ceiling."""
        config = _model_config(_release_sparse_config(), _pretrained(), 1 << 20)
        assert _candidate_prefilter_is_wired(config)
        _assert_candidate_prefilter_inert(config)  # must not raise
        _assert_candidate_prefilter_inert(
            _model_config(_release_sparse_config(), _pretrained(), None)
        )

    def test_wiredness_is_read_off_the_lowered_params(self):
        """Not off the checkpoint config, which is the wrong question.

        ``_UnwiredSparseConfig``'s pretrained config carries all three values, so
        an implementation that asked the checkpoint would call this wired and
        drop the refusal while the runtime silently skipped level one.
        """
        config = _model_config(_UnwiredSparseConfig(), _pretrained(), 1 << 20)
        assert not _candidate_prefilter_is_wired(config)
        with pytest.raises(ValueError, match="not wired up"):
            _assert_candidate_prefilter_inert(config)

    def test_unwired_boundary_is_exact_at_the_release_shape(self):
        """2048 x 8 at a ratio-1 source: inert to 16384 and not one token more."""
        unwired = _UnwiredSparseConfig()
        _assert_candidate_prefilter_inert(_model_config(unwired, _pretrained(), 16384))
        with pytest.raises(ValueError, match="max_seq_len=16384"):
            _assert_candidate_prefilter_inert(_model_config(unwired, _pretrained(), 16385))

    def test_unwired_bound_scales_with_the_source_compress_ratio(self):
        """The bound counts compressed positions, so a pooled source doubles it.

        The bare ``blocks * block_size`` product would refuse a ratio-2 source at
        half the length it is actually exact to. Nothing in the release exercises
        this -- layer 20 is in the ratio-1 band -- which is why it is pinned here
        rather than left to the first checkpoint that moves the source.
        """
        unwired = _UnwiredSparseConfig()
        pooled = _pretrained(compress_ratios=[2] * NUM_HIDDEN_LAYERS)
        assert _candidate_source_ratio(pooled, CANDIDATE_SOURCE_LAYER_ID) == 2
        _assert_candidate_prefilter_inert(_model_config(unwired, pooled, 32768))
        with pytest.raises(ValueError, match="max_seq_len=32768"):
            _assert_candidate_prefilter_inert(_model_config(unwired, pooled, 32769))

    def test_message_still_quotes_the_configured_shape(self):
        """The bound moved to compressed positions; the *reason* is still blocks x size.

        Two different numbers now appear in the message when the source is
        pooled, and both are load-bearing: the ceiling tells the operator what to
        set, the product tells them where it came from.
        """
        unwired = _UnwiredSparseConfig()
        pooled = _pretrained(compress_ratios=[2] * NUM_HIDDEN_LAYERS)
        with pytest.raises(ValueError) as excinfo:
            _assert_candidate_prefilter_inert(_model_config(unwired, pooled, 1 << 20))
        message = str(excinfo.value)
        assert "candidate_topk_blocks=2048 x candidate_block_size=8" in message
        assert "max_seq_len=32768" in message

    @pytest.mark.parametrize(
        "sparse_config, pretrained, why",
        [
            (None, _pretrained(), "no sparse attention configured"),
            (_UnwiredSparseConfig(), SimpleNamespace(), "V4: no candidate fields at all"),
            (
                _UnwiredSparseConfig(),
                _pretrained(candidate_source=None),
                "V4.1 config that omits the source layer",
            ),
            (_UnwiredSparseConfig(), _pretrained(topk_blocks=0), "topk_blocks 0"),
            (_UnwiredSparseConfig(), _pretrained(block_size=0), "block_size 0"),
        ],
    )
    def test_no_refusal_when_there_is_no_prefilter_to_miss(self, sparse_config, pretrained, why):
        """Nothing to skip means nothing to be inexact about, at any length."""
        _assert_candidate_prefilter_inert(_model_config(sparse_config, pretrained, 1 << 20))

    @pytest.mark.parametrize(
        "ratios, expected",
        [
            (None, 1),
            ([], 1),
            ([0] * NUM_HIDDEN_LAYERS, 1),  # the SWA-only sentinel floors to 1
            ([1] * NUM_HIDDEN_LAYERS, 1),
            ([2] * NUM_HIDDEN_LAYERS, 2),
            (list(RELEASE_COMPRESS_RATIOS), 1),  # layer 20 is in the unpooled band
        ],
    )
    def test_candidate_source_ratio(self, ratios, expected):
        text = SimpleNamespace(compress_ratios=ratios)
        assert _candidate_source_ratio(text, CANDIDATE_SOURCE_LAYER_ID) == expected

    def test_candidate_source_ratio_out_of_range_falls_back(self):
        """A source index past the ratio list is a malformed config, not a crash.

        ``build_layer_descriptors`` already refuses that config; this function is
        only reached on the way to a refusal, so it must not be the thing that
        raises first and hide the better message.
        """
        text = SimpleNamespace(compress_ratios=[2, 2, 2])
        assert _candidate_source_ratio(text, 99) == 1
