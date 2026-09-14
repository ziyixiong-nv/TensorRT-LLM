# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 decoder-half bounded replay (tech report §3.2.2).

Three things are checked here, and all three are checkable without a GPU or a
checkpoint:

1. **The truncated window.** ``_prepare_deepseek_v4_indices_compiled`` is a pure
   function of ``token_positions`` and the new ``swa_floor``, so the exact index
   lists and the exact valid-slot counts can be asserted rather than sampled. The
   properties that matter are that the *last* replayed token keeps a full
   ``window_size``-wide window -- the one the scheme's correctness rests on -- and
   that ``swa_floor=None`` reproduces today's output bit-for-bit, so an ordinary
   forward cannot regress.
2. **The row bookkeeping.** ``plan_decoder_replay`` and the gather/scatter pair
   are metadata arithmetic; the interesting cases are the refusals (each of which
   is a named hazard, not a tuning choice) and the mixed context+generation batch,
   where the generation rows must pass through untouched.
3. **The pass boundary.** ``enter_decoder_replay`` has to capture the encoder
   half's cross-layer handoffs in the one window where they exist -- after the
   encoder layers published them and before the replay ``prepare()`` clears them.
   Getting that window wrong crashes the first indexed decoder layer, and it is
   invisible to a test that only inspects the plan, so the handoffs are published
   *between* the planning call and the boundary call here, exactly as a real
   forward does.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.decoder_replay import (
    DecoderReplayPlan,
    enter_decoder_replay,
    exit_decoder_replay,
    gather_replayed_rows,
    plan_decoder_replay,
    scatter_replayed_rows,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)

WINDOW = 128


# ---------------------------------------------------------------------------
# 1. The truncated sliding window
# ---------------------------------------------------------------------------


def _build_indices(token_positions, swa_floor, ratio_specs):
    """Run the index builder in isolation and hand back the two buffers it fills."""
    num_tokens = token_positions.shape[0]
    swa_buf = torch.zeros(num_tokens, WINDOW, dtype=torch.int32)
    compressed_buf = torch.zeros(num_tokens, 1024, dtype=torch.int32)
    topk_lens = {ratio: torch.zeros(num_tokens, dtype=torch.int32) for ratio, _, _ in ratio_specs}
    DeepseekV4TrtllmAttentionMetadata._prepare_deepseek_v4_indices_compiled(
        token_positions,
        WINDOW,
        512,
        num_tokens,
        512,
        swa_buf,
        compressed_buf,
        topk_lens,
        ratio_specs,
        swa_floor,
    )
    return swa_buf, topk_lens


def _replay_positions(prompt_len: int, window: int = WINDOW):
    """``(token_positions, swa_floor)`` for one prompt replayed over its last window."""
    base = prompt_len - window
    positions = base + torch.arange(window, dtype=torch.int32)
    return positions, torch.full((window,), base, dtype=torch.int32)


def test_no_floor_is_bit_identical_to_today():
    """``swa_floor=None`` must not change a single index of an ordinary forward.

    The floor is threaded through the same traced graph every forward uses, so
    this is the regression that matters most: the default path has to be untouched.
    """
    positions = torch.arange(300, dtype=torch.int32)
    specs = ((1, 1, 0), (2, 2, 1))
    swa_ref, lens_ref = _build_indices(positions, None, specs)
    swa_zero, lens_zero = _build_indices(positions, torch.zeros(300, dtype=torch.int32), specs)
    # A zero floor is the identity, which is the statement that the maximum and
    # the subtraction are both no-ops at floor 0 -- i.e. that None and 0 agree.
    torch.testing.assert_close(swa_ref, swa_zero, rtol=0, atol=0)
    for ratio in lens_ref:
        torch.testing.assert_close(lens_ref[ratio], lens_zero[ratio], rtol=0, atol=0)


def test_last_replayed_token_keeps_a_full_window():
    """The token whose logits are consumed must see an exact, unapproximated window.

    This is the load-bearing property of §3.2.2. If it fails the scheme is not an
    approximation at the margins, it is wrong at the one row that is read.
    """
    positions, floor = _replay_positions(prompt_len=4096)
    swa, lens = _build_indices(positions, floor, ((1, 1, 0),))
    last = swa[-1]
    assert (last >= 0).all(), "the last replayed token has masked slots in its window"
    expected = torch.arange(4096 - WINDOW, 4096, dtype=torch.int32)
    torch.testing.assert_close(last, expected, rtol=0, atol=0)
    assert int(lens[1][-1]) == WINDOW


def test_first_replayed_token_sees_only_itself():
    """The earliest replayed row is maximally truncated: one valid slot, its own.

    Its window would have reached ``window - 1`` positions below the replay floor,
    and this pass wrote none of them. Reading them is what the floor prevents.
    """
    positions, floor = _replay_positions(prompt_len=4096)
    swa, lens = _build_indices(positions, floor, ((1, 1, 0),))
    first = swa[0]
    assert int(first[0]) == 4096 - WINDOW
    assert (first[1:] == -1).all(), "slots below the replay floor were not masked"
    assert int(lens[1][0]) == 1


def test_swa_only_valid_count_is_the_offset_plus_one():
    """An SWA-only layer's reported length is the *actual* valid count, so it must shrink.

    The padded kinds always spend a full ``window_size`` of slots and rely on the
    ``-1``s to mask; kind 0 does not, so a floored window that kept the unfloored
    count would have the kernel read the padding.
    """
    positions, floor = _replay_positions(prompt_len=1024)
    _, lens = _build_indices(positions, floor, ((1, 1, 0),))
    torch.testing.assert_close(
        lens[1], torch.arange(1, WINDOW + 1, dtype=torch.int32), rtol=0, atol=0
    )


def test_global_lengths_ignore_the_floor():
    """Only the window is truncated; the global cache is complete and stays so.

    The whole point of replaying the decoder half rather than recomputing it is
    that layer 20 already published the entire global KV. A floor applied to the
    indexed kinds would throw that away.
    """
    positions, floor = _replay_positions(prompt_len=1024)
    _, floored = _build_indices(positions, floor, ((2, 2, 1),))
    _, unfloored = _build_indices(positions, None, ((2, 2, 1),))
    torch.testing.assert_close(floored[2], unfloored[2], rtol=0, atol=0)


def test_floor_never_widens_a_short_window():
    """A prompt at or under the window is unaffected by a floor of zero cached tokens.

    This is the boundary the planner refuses on benefit grounds; checked here so
    that the *builder* is also safe if it ever sees one.
    """
    positions = torch.arange(64, dtype=torch.int32)
    floor = torch.zeros(64, dtype=torch.int32)
    swa_ref, _ = _build_indices(positions, None, ((1, 1, 0),))
    swa_floored, _ = _build_indices(positions, floor, ((1, 1, 0),))
    torch.testing.assert_close(swa_ref, swa_floored, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. The row bookkeeping
# ---------------------------------------------------------------------------


class _FakeManager:
    def __init__(self, enable_swa_scratch_reuse: bool):
        self.enable_swa_scratch_reuse = enable_swa_scratch_reuse


class _FakeMapping:
    def __init__(self, enable_attention_dp: bool):
        self.enable_attention_dp = enable_attention_dp


class _FakeKvParams:
    def __init__(self, num_cached):
        self.num_cached_tokens_per_seq = list(num_cached)


class _FakeMetadata:
    """The eight attributes ``plan_decoder_replay`` reads, and nothing else.

    A real ``DeepseekV4TrtllmAttentionMetadata`` needs a KV cache manager, pools
    and a device; the planner is pure host arithmetic over these fields, so
    standing them up here keeps this a CPU test and keeps the refusals -- which
    are the part worth pinning -- independent of cache-manager plumbing.

    ``prepare()`` is modelled too, because the boundary's correctness depends on
    what it clears; see the method.
    """

    def __init__(
        self,
        seq_lens,
        num_contexts,
        num_cached=None,
        *,
        cached_kv=True,
        attention_dp=False,
        swa_reuse=False,
    ):
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self.seq_lens_cuda = self.seq_lens
        self.num_contexts = num_contexts
        self.num_seqs = len(seq_lens)
        self.num_tokens = int(sum(seq_lens))
        self.enable_context_mla_with_cached_kv = cached_kv
        self.mapping = _FakeMapping(attention_dp)
        self.kv_cache_manager = _FakeManager(swa_reuse)
        self.kv_cache_params = _FakeKvParams(num_cached or [0] * len(seq_lens))
        self.v41_index_keys = {}
        self.v41_candidate_blocks = {}
        self.v41_topk_indices = {}
        self.swa_window_floored_to_pass = False
        self.prepare_calls = 0

    def prepare(self):
        """Only the part of ``prepare()`` the boundary has to survive.

        The real one rebuilds every device index buffer; what matters to
        ``enter_decoder_replay`` is the tail of it (``metadata.py:753-755``), which
        drops all three cross-layer handoffs. Modelling just that keeps the
        boundary testable on CPU and keeps the test honest about *why* the
        reinstatement exists.
        """
        self.prepare_calls += 1
        self.v41_index_keys.clear()
        self.v41_topk_indices.clear()
        self.v41_candidate_blocks.clear()


def test_plan_selects_the_window_suffix_of_each_context():
    plan = plan_decoder_replay(_FakeMetadata([300, 500], num_contexts=2), WINDOW)
    assert plan is not None
    assert plan.num_replay_tokens == 2 * WINDOW
    assert plan.num_encoder_tokens == 800
    torch.testing.assert_close(
        plan.rows[:WINDOW], torch.arange(300 - WINDOW, 300, dtype=torch.int64), rtol=0, atol=0
    )
    torch.testing.assert_close(
        plan.rows[WINDOW:], torch.arange(800 - WINDOW, 800, dtype=torch.int64), rtol=0, atol=0
    )
    # The cached count rises by exactly the rows dropped, which is what makes the
    # replay pass the shape a chunked prefill's final chunk already has.
    assert plan.replay_num_cached == [300 - WINDOW, 500 - WINDOW]


def test_generation_rows_pass_through_untouched():
    """A mixed batch replays its contexts and leaves its decodes exactly alone.

    A decode row is already inside any window, so ``min`` would be the identity --
    but its cached count must not move, or it desynchronizes from the KV manager.
    """
    metadata = _FakeMetadata([400, 1, 1], num_contexts=1, num_cached=[0, 900, 950])
    plan = plan_decoder_replay(metadata, WINDOW)
    assert plan is not None
    assert plan.num_replay_tokens == WINDOW + 2
    assert plan.replay_num_cached == [400 - WINDOW, 900, 950]
    torch.testing.assert_close(
        plan.rows[-2:], torch.tensor([400, 401], dtype=torch.int64), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "kwargs,why",
    [
        (dict(cached_kv=False), "the indexer would score the wrong 128 prefix keys"),
        (dict(attention_dp=True), "MoE all-to-all was sized before the forward"),
        (dict(swa_reuse=True), "a later request could read the approximate window KV"),
    ],
)
def test_named_hazards_refuse(kwargs, why):
    assert plan_decoder_replay(_FakeMetadata([600], 1, **kwargs), WINDOW) is None, why


def test_unrecognized_manager_refuses():
    """Absence of the reuse flag is read as reuse, because that way round is safe.

    Refusing costs prefill time; replaying into a reused window cache corrupts
    another request's prefix.
    """
    metadata = _FakeMetadata([600], 1)
    del metadata.kv_cache_manager.enable_swa_scratch_reuse
    assert plan_decoder_replay(metadata, WINDOW) is None


@pytest.mark.parametrize(
    "seq_lens,num_contexts",
    [
        ([1, 1, 1], 0),  # pure decode: also the CUDA-graph shape
        ([WINDOW, 64], 2),  # nothing longer than the window
    ],
)
def test_no_benefit_refuses(seq_lens, num_contexts):
    assert plan_decoder_replay(_FakeMetadata(seq_lens, num_contexts), WINDOW) is None


def test_zero_window_refuses():
    assert plan_decoder_replay(_FakeMetadata([600], 1), 0) is None


# ---------------------------------------------------------------------------
# 3. The pass boundary
# ---------------------------------------------------------------------------


def test_boundary_captures_handoffs_published_after_planning():
    """The handoffs the decoder half reads are written *after* the plan is built.

    This is the ordering the whole boundary turns on. ``plan_decoder_replay`` runs
    before layer 0, so every handoff dict is empty then; the encoder half fills
    them as it goes, and the replay ``prepare()`` clears them. Only the window
    between those two events holds anything, so a capture taken on either side of
    it hands the decoder half nothing -- and the first decoder layer that reuses an
    encoder layer's top-k then asserts its source never published, which is a hard
    crash on a real model.

    The failure is invisible to any test that only inspects the plan, because an
    empty dict crossing the boundary intact looks exactly like a correct no-op.
    """
    metadata = _FakeMetadata([400], num_contexts=1)
    plan = plan_decoder_replay(metadata, WINDOW)
    assert plan is not None

    # ... the encoder half runs here. Layer 20 publishes a per-query-row top-k and
    # a per-KV-row key handoff, both at full encoder height.
    topk = torch.arange(400 * 3, dtype=torch.int32).reshape(400, 3)
    metadata.v41_topk_indices[20] = topk
    metadata.v41_index_keys[20] = ("postprocessed-keys", None)

    enter_decoder_replay(metadata, plan)

    assert metadata.prepare_calls == 1, "the replay pass must re-prepare the metadata"
    assert metadata.swa_window_floored_to_pass is True
    assert 20 in metadata.v41_topk_indices, (
        "layer 20's top-k did not survive the boundary, so every decoder layer that "
        "reuses it will assert that its source never published in this forward"
    )
    # Gathered, not carried: the carrier is indexed by global query row, so the
    # decoder half must read the replayed rows' selections at replay offsets.
    torch.testing.assert_close(metadata.v41_topk_indices[20], topk[400 - WINDOW :], rtol=0, atol=0)
    # The KV-row handoff crosses whole -- it is indexed by compressed-KV entry and
    # the global cache the decoder half reads is complete after the encoder half.
    assert metadata.v41_index_keys[20] == ("postprocessed-keys", None)


def test_boundary_regathers_the_candidate_carrier_and_its_row_count():
    """A candidate publication is re-sliced *and* told its new height.

    ``_check_candidate_rows`` asserts every row a consumer masks is a row the
    producer wrote; after the boundary the consumers mask ``window`` rows, so a
    carrier that kept the producer's count would trip that assert rather than
    return wrong data.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.candidate_prefilter import (
        CandidatePublication,
    )

    metadata = _FakeMetadata([400], num_contexts=1)
    plan = plan_decoder_replay(metadata, WINDOW)
    assert plan is not None
    blocks = torch.arange(400 * 2, dtype=torch.int32).reshape(400, 2)
    metadata.v41_candidate_blocks[20] = CandidatePublication(
        blocks=blocks, compress_ratio=2, rows=400
    )

    enter_decoder_replay(metadata, plan)

    got = metadata.v41_candidate_blocks[20]
    assert got.rows == WINDOW
    assert got.compress_ratio == 2
    torch.testing.assert_close(got.blocks, blocks[400 - WINDOW :], rtol=0, atol=0)


def test_exit_restores_the_encoder_shape_and_clears_the_floor():
    """The next forward must not inherit this one's replay shape or its floor."""
    metadata = _FakeMetadata([400], num_contexts=1)
    plan = plan_decoder_replay(metadata, WINDOW)
    assert plan is not None
    enter_decoder_replay(metadata, plan)
    exit_decoder_replay(metadata, plan)
    assert metadata.swa_window_floored_to_pass is False
    assert metadata.kv_cache_params.num_cached_tokens_per_seq == [0]
    torch.testing.assert_close(metadata.seq_lens, plan.saved_seq_lens, rtol=0, atol=0)


def _plan_for(rows, num_encoder_tokens):
    return DecoderReplayPlan(
        rows=torch.tensor(rows, dtype=torch.int64),
        replay_seq_lens=torch.tensor([len(rows)], dtype=torch.int32),
        replay_num_cached=[num_encoder_tokens - len(rows)],
        saved_seq_lens=torch.tensor([num_encoder_tokens], dtype=torch.int32),
        saved_num_cached=[0],
        num_encoder_tokens=num_encoder_tokens,
    )


def test_gather_picks_the_row_axis_for_activations():
    plan = _plan_for([2, 3], num_encoder_tokens=4)
    residual = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
    torch.testing.assert_close(gather_replayed_rows(residual, plan), residual[2:4])


def test_gather_picks_the_last_axis_for_position_ids():
    """``position_ids`` arrives as ``[1, N]``, so the per-token axis is the last one."""
    plan = _plan_for([2, 3], num_encoder_tokens=4)
    position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
    torch.testing.assert_close(
        gather_replayed_rows(position_ids, plan), torch.tensor([[2, 3]], dtype=torch.int32)
    )


def test_gather_refuses_a_tensor_that_is_not_per_token():
    """Silently passing an encoder-length tensor through is the corruption case."""
    plan = _plan_for([2, 3], num_encoder_tokens=4)
    with pytest.raises(ValueError, match="not per-token"):
        gather_replayed_rows(torch.zeros(7, 5), plan)


def test_gather_of_none_is_none():
    assert gather_replayed_rows(None, _plan_for([0], 4)) is None


def test_scatter_restores_height_and_zeroes_the_rest():
    """Rows the decoder half never ran come back zero, not uninitialized.

    Replay is refused whenever a consumer needs them, so zero is unobserved -- but
    a future consumer that does read them should be wrong reproducibly.
    """
    plan = _plan_for([2, 3], num_encoder_tokens=4)
    replayed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    full = scatter_replayed_rows(replayed, plan)
    assert full.shape == (4, 2)
    torch.testing.assert_close(full[2:], replayed)
    assert not full[:2].any()


def test_gather_then_scatter_is_the_identity_on_replayed_rows():
    plan = _plan_for([1, 3], num_encoder_tokens=4)
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    round_tripped = scatter_replayed_rows(gather_replayed_rows(hidden, plan), plan)
    torch.testing.assert_close(round_tripped[plan.rows], hidden[plan.rows])
