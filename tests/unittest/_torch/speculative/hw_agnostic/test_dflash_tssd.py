# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DFlash + Target-Side SSD (T-SSD).

These tests exercise the math and state-machine pieces of
``tensorrt_llm/_torch/speculative/dflash_tssd.py`` without requiring real
model weights or a multi-GPU setup. They run on a single GPU.

For end-to-end perf validation see Phase 2.6 (Task #21) — needs 8x B200
for gpt-oss-120b TP=8.
"""

import os
import sys
import unittest

import torch

# Allow `from tensorrt_llm._torch.speculative.dflash_tssd import ...` to
# resolve from the repo even when not pip-installed.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")),
)

from tensorrt_llm._torch.speculative.dflash_tssd import (  # noqa: E402
    build_tssd_mask,
    build_tssd_packed_mask,
    geometric_fanout,
    sample_candidates,
    tssd_should_enable,
)

# ---------------------------------------------------------------------------
# 1. geometric_fanout
# ---------------------------------------------------------------------------


class TestGeometricFanout(unittest.TestCase):
    def test_K4_F8_a78(self):
        F = geometric_fanout(K=4, B_budget=8, a_p=0.78, r=0.5)
        self.assertEqual(len(F), 5)
        self.assertEqual(min(F), 1)
        self.assertEqual(sum(F), 8)  # exact budget (post drift correction)

    def test_K4_F16_a78(self):
        F = geometric_fanout(K=4, B_budget=16, a_p=0.78, r=0.5)
        self.assertEqual(len(F), 5)
        self.assertEqual(sum(F), 16)
        # Last slot (full-accept) must dominate at high a_p; check ratio.
        self.assertGreater(F[-1], F[0])

    def test_high_acceptance_concentrates_in_last_slot(self):
        """Per Saguaro Theorem 12, high a_p drives mass into the
        full-accept bonus slot (k = K)."""
        F_low = geometric_fanout(K=4, B_budget=16, a_p=0.5, r=0.5)
        F_high = geometric_fanout(K=4, B_budget=16, a_p=0.95, r=0.5)
        ratio_low = F_low[-1] / max(1, F_low[0])
        ratio_high = F_high[-1] / max(1, F_high[0])
        self.assertGreater(ratio_high, ratio_low)

    def test_tight_budget_falls_back_to_one_per_slot(self):
        # Tight budget < K+1: assign 1 to first B groups, 0 to rest.
        # Sum is exactly B_budget (matches grown shape contracts).
        F = geometric_fanout(K=4, B_budget=3, a_p=0.78, r=0.5)
        self.assertEqual(F, [1, 1, 1, 0, 0])
        self.assertEqual(sum(F), 3)


# ---------------------------------------------------------------------------
# 2. build_tssd_mask  (matches §4.2 of the design doc)
# ---------------------------------------------------------------------------


class TestBuildTSSDMask(unittest.TestCase):
    def test_shape(self):
        K, F, prefix = 4, [2, 2, 2, 1, 1], 100
        m = build_tssd_mask(K=K, F=F, prefix_len=prefix)
        Q = K + sum(F)
        kv_len = prefix + Q
        self.assertEqual(m.shape, (Q, kv_len))
        self.assertEqual(m.dtype, torch.bool)

    def test_prefix_visibility(self):
        """Every query attends to every prefix token."""
        K, F, prefix = 3, [1, 1, 1, 1], 50
        m = build_tssd_mask(K, F, prefix)
        self.assertTrue(m[:, :prefix].all())

    def test_main_verify_diagonal(self):
        """Main verify row i sees d_1..d_i and self (positions
        prefix..prefix+i inclusive)."""
        K, F, prefix = 4, [1, 1, 1, 1, 1], 10
        m = build_tssd_mask(K, F, prefix)
        for i in range(K):
            # Should see prefix..prefix+i.
            self.assertTrue(m[i, prefix : prefix + i + 1].all())
            # Should NOT see prefix+i+1 .. prefix+K-1.
            if i + 1 < K:
                self.assertFalse(m[i, prefix + i + 1 : prefix + K].any())

    def test_candidate_self_attention(self):
        """Each candidate attends to its own diagonal slot only (plus
        d_1..d_k for group k)."""
        K, F, prefix = 2, [1, 2, 3], 10
        m = build_tssd_mask(K, F, prefix)
        a = 0
        for k in range(K + 1):
            for _ in range(F[k]):
                row = K + a
                own_kv = prefix + K + a
                # Own slot is set.
                self.assertTrue(bool(m[row, own_kv]))
                # No other candidate slot is set.
                cand_slots = list(range(prefix + K, prefix + K + sum(F)))
                cand_slots.remove(own_kv)
                for s in cand_slots:
                    self.assertFalse(
                        bool(m[row, s]),
                        f"candidate row {row} should not see slot {s}",
                    )
                # Visible main verify positions = prefix..prefix+k-1.
                for j in range(K):
                    expected = j < k
                    self.assertEqual(
                        bool(m[row, prefix + j]),
                        expected,
                        f"row {row} (group k={k}) attending to d_{j + 1} expected={expected}",
                    )
                a += 1


# ---------------------------------------------------------------------------
# 2b. build_tssd_packed_mask (TRTLLM bit-packed format)
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBuildTSSDPackedMask(unittest.TestCase):
    def test_shape_and_dtype(self):
        K, F, B = 4, [1, 1, 1, 1, 1], 3
        m = build_tssd_packed_mask(K, F, B)
        n = K + 1 + sum(F)  # 4+1+5 = 10
        num_blocks = (n + 31) // 32  # = 1
        self.assertEqual(m.shape, (B, n, num_blocks))
        self.assertEqual(m.dtype, torch.int32)
        # All requests have identical pattern.
        self.assertTrue(torch.equal(m[0], m[1]))
        self.assertTrue(torch.equal(m[0], m[2]))

    def test_bit_pattern_matches_design(self):
        K, F, B = 2, [1, 1, 1], 1
        m = build_tssd_packed_mask(K, F, B)
        # n = 2 + 1 + 3 = 6 positions: [bonus, d_1, d_2, c_{0,0}, c_{1,0}, c_{2,0}]
        # Decoded mask should be:
        # q=0 bonus:      bits[0]=1                              → 0b000001 = 1
        # q=1 d_1:        bits[0,1]=1                            → 0b000011 = 3
        # q=2 d_2:        bits[0,1,2]=1                          → 0b000111 = 7
        # q=3 c_{0,0}: bonus+self                                → bits[0,3] = 0b001001 = 9
        # q=4 c_{1,0}: bonus+d_1+self                            → bits[0,1,4] = 0b010011 = 19
        # q=5 c_{2,0}: bonus+d_1+d_2+self                        → bits[0,1,2,5] = 0b100111 = 39
        expected = [1, 3, 7, 9, 19, 39]
        for q, exp in enumerate(expected):
            got = int(m[0, q, 0].item())
            self.assertEqual(
                got,
                exp,
                f"q={q}: bit pattern {bin(got)} != expected {bin(exp)} "
                f"({['bonus', 'd_1', 'd_2', 'c_00', 'c_10', 'c_20'][q]})",
            )

    def test_no_cross_candidate_attention(self):
        """Candidate rows must not have bits set for other candidates."""
        K, F, B = 4, [2, 2, 2, 1, 1], 1
        m = build_tssd_packed_mask(K, F, B)
        n = K + 1 + sum(F)
        # Each candidate row q in [K+1..n) should have:
        #  - some prefix bits (bonus + main verify subset)
        #  - exactly one diagonal candidate bit (its own slot)
        #  - zero other candidate bits
        for q in range(K + 1, n):
            # Reassemble per-bit row from packed int32s.
            num_blocks = m.shape[2]
            row_bits = []
            for blk in range(num_blocks):
                val = int(m[0, q, blk].item())
                for j in range(32):
                    col = blk * 32 + j
                    if col < n:
                        row_bits.append(bool(val & (1 << j)))

            # Count candidate-region bits (cols in [K+1..n)) that are set.
            cand_bits = sum(row_bits[K + 1 : n])
            self.assertEqual(
                cand_bits,
                1,
                f"row q={q}: expected exactly one candidate bit (own slot), "
                f"got {cand_bits}: {row_bits}",
            )
            # That one bit must be at column q (own diagonal).
            self.assertTrue(row_bits[q], f"row q={q}: own diagonal bit not set")


# ---------------------------------------------------------------------------
# 3. tssd_should_enable  (matches PHASE0_RESULTS.md viable envelope)
# ---------------------------------------------------------------------------


class TestTSSDGate(unittest.TestCase):
    def test_phase0_viable_configs_pass(self):
        # Configs measured viable in Phase 0 (B=1, prefix=1024, F=8/16):
        self.assertTrue(tssd_should_enable(B=1, prefix_len=1024, K=4, F_total=8))
        self.assertTrue(tssd_should_enable(B=1, prefix_len=1024, K=4, F_total=16))
        # B=1, prefix=4096, F=8: marginal pass.
        self.assertTrue(tssd_should_enable(B=1, prefix_len=4096, K=4, F_total=8))
        # B=2, prefix=1024, F=8: pass.
        self.assertTrue(tssd_should_enable(B=2, prefix_len=1024, K=4, F_total=8))

    def test_phase0_failed_configs_fail(self):
        # B>=4: hard rule.
        self.assertFalse(tssd_should_enable(B=4, prefix_len=1024, K=4, F_total=8))
        self.assertFalse(tssd_should_enable(B=8, prefix_len=512, K=4, F_total=8))
        # F_total > 16: out of budget.
        self.assertFalse(tssd_should_enable(B=1, prefix_len=1024, K=4, F_total=32))
        # Long prefix at B=2: gate trips.
        self.assertFalse(tssd_should_enable(B=2, prefix_len=16384, K=4, F_total=16))

    def test_zero_budget(self):
        self.assertFalse(tssd_should_enable(B=1, prefix_len=1024, K=4, F_total=0))


# ---------------------------------------------------------------------------
# 4. sample_candidates
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for sample_candidates")
class TestSampleCandidates(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B = 2
        self.K = 4
        self.V = 100
        self.device = "cuda"

    def test_topk_shape_and_padding(self):
        F = [1, 2, 2, 1, 2]
        target = torch.randn(self.B, self.K + 1, self.V, device=self.device)
        draft = torch.randn(self.B, self.K, self.V, device=self.device)

        cand, log_p = sample_candidates(draft, target, F, mode="topk")

        self.assertEqual(cand.shape, (self.B, self.K + 1, max(F)))
        self.assertEqual(log_p.shape, (self.B, self.K + 1, max(F)))
        # Slots beyond F[k] are padded with -1 / -inf.
        for k in range(self.K + 1):
            for j in range(F[k], max(F)):
                self.assertTrue((cand[:, k, j] == -1).all())
                self.assertTrue(torch.isinf(log_p[:, k, j]).all())

    def test_target_mode_picks_topk_target(self):
        F = [3, 3, 3, 3, 3]
        target = torch.randn(self.B, self.K + 1, self.V, device=self.device)
        cand, _ = sample_candidates(None, target, F, mode="target")

        # Verify the picked tokens match argmax / top-3 of target.
        for b in range(self.B):
            for k in range(self.K + 1):
                expected = torch.topk(target[b, k], k=3).indices.sort().values
                got = cand[b, k, :3].sort().values
                self.assertTrue(torch.equal(expected, got))

    def test_topk_residual_drops_low_residual_tokens(self):
        """When draft prob == target prob at every token, residual is 0;
        topk on log(0) produces arbitrary picks. Check that when one
        token has high target+low draft (high residual), it's selected."""
        target = torch.zeros(1, self.K + 1, self.V, device=self.device)
        draft = torch.zeros(1, self.K, self.V, device=self.device)
        # Token id 7: target high, draft low → high residual.
        target[0, 0, 7] = 10.0
        draft[0, 0, 7] = -10.0

        F = [1, 1, 1, 1, 1]
        cand, _ = sample_candidates(draft, target, F, mode="topk")
        self.assertEqual(int(cand[0, 0, 0].item()), 7)


# ---------------------------------------------------------------------------
# 5. DFlashTSSDWorker.try_commit_candidate
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTryCommit(unittest.TestCase):
    def setUp(self):
        from tensorrt_llm._torch.speculative.dflash_tssd import DFlashTSSDWorker
        from tensorrt_llm.mapping import Mapping

        # Stub spec_config (avoid pulling Pydantic at import time).
        class StubCfg:
            max_draft_len = 4
            tssd_enabled = True
            tssd_F_total = 8
            tssd_a_p = 0.78
            tssd_max_batch = 2

        self.worker = DFlashTSSDWorker(
            spec_config=StubCfg(),
            mapping=Mapping(world_size=1, rank=0, gpus_per_node=1),
        )

    def test_hit_path_returns_correct_index(self):
        # B=1 request, K=4. accepted_tokens[:, 0..K] all accepted (k* = K).
        # Candidate row at k=K contains x*.
        B, K = 1, 4
        max_F_k = 4
        accepted = torch.zeros(B, K + 1, dtype=torch.long, device="cuda")
        accepted[0, K] = 42  # x* is 42 at k_star = K
        num_accepted = torch.tensor([K + 1], dtype=torch.long, device="cuda")

        cand = torch.full((B, K + 1, max_F_k), -1, dtype=torch.long, device="cuda")
        # Place x* in slot j=2 of group k=K.
        cand[0, K, 2] = 42

        hit, idx = self.worker.try_commit_candidate(num_accepted, accepted, cand)
        self.assertTrue(bool(hit[0]))
        self.assertEqual(int(idx[0].item()), 2)

    def test_miss_path_returns_minus_one(self):
        B, K = 1, 4
        max_F_k = 4
        accepted = torch.zeros(B, K + 1, dtype=torch.long, device="cuda")
        accepted[0, K] = 42
        num_accepted = torch.tensor([K + 1], dtype=torch.long, device="cuda")

        cand = torch.full((B, K + 1, max_F_k), -1, dtype=torch.long, device="cuda")
        # x* (=42) not in any candidate.
        cand[0, K, :] = torch.tensor([1, 2, 3, 4], device="cuda")

        hit, idx = self.worker.try_commit_candidate(num_accepted, accepted, cand)
        self.assertFalse(bool(hit[0]))
        self.assertEqual(int(idx[0].item()), -1)

    def test_partial_accept_uses_correct_k_star(self):
        """When only the bonus token (position 0) is accepted, k_star = 0
        and x* is read from accepted[:, 0]."""
        B, K = 1, 4
        max_F_k = 4
        accepted = torch.zeros(B, K + 1, dtype=torch.long, device="cuda")
        accepted[0, 0] = 99  # x* at k_star=0
        accepted[0, 1:] = 0  # rejected positions
        num_accepted = torch.tensor([1], dtype=torch.long, device="cuda")

        cand = torch.full((B, K + 1, max_F_k), -1, dtype=torch.long, device="cuda")
        cand[0, 0, 1] = 99

        hit, idx = self.worker.try_commit_candidate(num_accepted, accepted, cand)
        self.assertTrue(bool(hit[0]))
        self.assertEqual(int(idx[0].item()), 1)

    def test_mixed_batch(self):
        """Some requests hit, others miss."""
        B, K = 3, 4
        max_F_k = 2
        accepted = torch.zeros(B, K + 1, dtype=torch.long, device="cuda")
        accepted[0, K] = 10  # request 0: x* = 10 at k* = K
        accepted[1, K] = 20  # request 1: x* = 20
        accepted[2, K] = 30  # request 2: x* = 30
        num_accepted = torch.tensor([K + 1, K + 1, K + 1], dtype=torch.long, device="cuda")

        cand = torch.full((B, K + 1, max_F_k), -1, dtype=torch.long, device="cuda")
        cand[0, K, 0] = 10  # hit
        cand[1, K, :] = torch.tensor([99, 100], device="cuda")  # miss
        cand[2, K, 1] = 30  # hit

        hit, idx = self.worker.try_commit_candidate(num_accepted, accepted, cand)
        self.assertEqual(hit.tolist(), [True, False, True])
        self.assertEqual(idx.tolist(), [0, -1, 1])


# ---------------------------------------------------------------------------
# 6. DFlashTSSDWorker.commit_candidate_to_buffers
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCommitToBuffers(unittest.TestCase):
    def setUp(self):
        from tensorrt_llm._torch.speculative.dflash_tssd import DFlashTSSDWorker, TSSDState
        from tensorrt_llm.mapping import Mapping

        class StubCfg:
            max_draft_len = 4
            tssd_enabled = True
            tssd_F_total = 8
            tssd_a_p = 0.78
            tssd_max_batch = 2

        self.worker = DFlashTSSDWorker(
            spec_config=StubCfg(),
            mapping=Mapping(world_size=1, rank=0, gpus_per_node=1),
        )
        # Manually wire scratch state without going through real model.
        K, F_total = 4, 8
        self.worker.tssd_state = TSSDState(
            K=K,
            F_total=F_total,
            F=[2, 2, 2, 1, 1],
            max_batch=2,
            fan_out=torch.tensor([2, 2, 2, 1, 1], dtype=torch.int32, device="cuda"),
            candidate_tokens=torch.full((2, K + 1, 2), -1, dtype=torch.long, device="cuda"),
            cand_target_hidden=torch.zeros(
                (1, 2, F_total, 16), dtype=torch.bfloat16, device="cuda"
            ),
            cand_target_k=torch.zeros((4, 2, F_total, 2, 8), dtype=torch.bfloat16, device="cuda"),
            cand_target_v=torch.zeros((4, 2, F_total, 2, 8), dtype=torch.bfloat16, device="cuda"),
            cand_dflash_k=torch.zeros((4, 2, F_total, 2, 8), dtype=torch.bfloat16, device="cuda"),
            cand_dflash_v=torch.zeros((4, 2, F_total, 2, 8), dtype=torch.bfloat16, device="cuda"),
        )
        # Stand-in for DFlashWorker._ctx_k_buf / _ctx_v_buf.
        self.worker._ctx_k_buf = torch.zeros((2, 4, 32, 2, 8), dtype=torch.bfloat16, device="cuda")
        self.worker._ctx_v_buf = torch.zeros_like(self.worker._ctx_k_buf)
        self.worker._ctx_len = torch.zeros(2, dtype=torch.long, device="cuda")

    def test_commit_writes_to_correct_position(self):
        st = self.worker.tssd_state
        # Mark candidate (slot=0, F-index=3) with a recognizable value.
        st.cand_dflash_k[:, 0, 3, :, :] = 1.5
        st.cand_dflash_v[:, 0, 3, :, :] = -2.5

        # Single hit: B=1, slot=0, k* = 2, prefix = 5 → write at pos 7.
        hit_flag = torch.tensor([True], device="cuda")
        cand_idx = torch.tensor([3], dtype=torch.long, device="cuda")
        num_accepted = torch.tensor([3], dtype=torch.long, device="cuda")  # k_star = 2
        slots = torch.tensor([0], dtype=torch.long, device="cuda")
        prefix = torch.tensor([5], dtype=torch.long, device="cuda")

        self.worker.commit_candidate_to_buffers(hit_flag, cand_idx, num_accepted, slots, prefix)

        # Position 7 = prefix(5) + k*(2) on slot 0.
        self.assertTrue(
            torch.allclose(
                self.worker._ctx_k_buf[0, :, 7, :, :].float(),
                torch.full_like(self.worker._ctx_k_buf[0, :, 7, :, :], 1.5).float(),
            )
        )
        self.assertTrue(
            torch.allclose(
                self.worker._ctx_v_buf[0, :, 7, :, :].float(),
                torch.full_like(self.worker._ctx_v_buf[0, :, 7, :, :], -2.5).float(),
            )
        )
        # Other positions unchanged (== 0).
        self.assertTrue(
            torch.allclose(
                self.worker._ctx_k_buf[0, :, 0:7, :, :],
                torch.zeros_like(self.worker._ctx_k_buf[0, :, 0:7, :, :]),
            )
        )
        # Stats updated.
        self.assertEqual(self.worker._tssd_hits, 1)
        self.assertEqual(self.worker._tssd_misses, 0)

    def test_miss_does_nothing(self):
        hit_flag = torch.tensor([False], device="cuda")
        cand_idx = torch.tensor([-1], dtype=torch.long, device="cuda")
        num_accepted = torch.tensor([3], dtype=torch.long, device="cuda")
        slots = torch.tensor([0], dtype=torch.long, device="cuda")
        prefix = torch.tensor([5], dtype=torch.long, device="cuda")

        before = self.worker._ctx_k_buf.clone()
        self.worker.commit_candidate_to_buffers(hit_flag, cand_idx, num_accepted, slots, prefix)
        self.assertTrue(torch.equal(self.worker._ctx_k_buf, before))
        self.assertEqual(self.worker._tssd_hits, 0)


# ---------------------------------------------------------------------------
# 7. Worker fallback when disabled
# ---------------------------------------------------------------------------


class TestFallbackBehavior(unittest.TestCase):
    def test_disabled_worker_constructs_with_zero_F(self):
        from tensorrt_llm._torch.speculative.dflash_tssd import DFlashTSSDWorker
        from tensorrt_llm.mapping import Mapping

        class StubCfg:
            max_draft_len = 4
            tssd_enabled = False
            tssd_F_total = 8
            tssd_a_p = 0.78
            tssd_max_batch = 2

        w = DFlashTSSDWorker(
            spec_config=StubCfg(),
            mapping=Mapping(world_size=1, rank=0, gpus_per_node=1),
        )
        self.assertFalse(w.tssd_enabled)
        self.assertEqual(w.F, [0, 0, 0, 0, 0])
        self.assertFalse(w.gate(batch_size=1, prefix_len=1024))

    def test_enabled_worker_gates_correctly(self):
        from tensorrt_llm._torch.speculative.dflash_tssd import DFlashTSSDWorker
        from tensorrt_llm.mapping import Mapping

        class StubCfg:
            max_draft_len = 4
            tssd_enabled = True
            tssd_F_total = 8
            tssd_a_p = 0.78
            tssd_max_batch = 2

        w = DFlashTSSDWorker(
            spec_config=StubCfg(),
            mapping=Mapping(world_size=1, rank=0, gpus_per_node=1),
        )
        self.assertTrue(w.gate(batch_size=1, prefix_len=1024))
        self.assertFalse(w.gate(batch_size=4, prefix_len=1024))
        self.assertFalse(w.gate(batch_size=2, prefix_len=16384))


if __name__ == "__main__":
    unittest.main(verbosity=2)
