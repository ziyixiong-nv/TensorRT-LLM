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
"""The two Engram table cuts, reassembled by their real collectives on real ranks.

``tests/unittest/_torch/modules/test_engram.py`` covers the sharding arithmetic by
building every rank's table in one process and concatenating or summing the pieces
in Python. That validates the index math and nothing else: the collective that
actually reassembles them at serving time is stubbed out by the very thing being
tested. This suite is the other half -- the same comparison, but with each rank in
its own process on its own GPU and the pieces combined by the collective
``DeepseekV41Engram._combine_embeddings`` would issue.

What only shows up here:

* **All-gather on the last dimension.** The head cut relies on ``allgather(...,
  dim=-1)`` landing each rank's head columns in rank order. For ``dim != 0`` that
  is not one NCCL call but a reshape-gather-chunk-cat (see ``_allgather``), and
  whether it round-trips a head-major ``[T, H * D]`` layout is a property of that
  implementation, not of the arithmetic above it.
* **The choice between the two collectives.** ``num_heads % tp_size`` decides the
  cut, and getting it wrong is not a crash -- an all-gather where a sum was needed
  produces a wrongly *shaped* tensor, but a sum where a concatenation was needed
  produces a right-shaped tensor full of overlapping garbage.
* **DeepSeek-V4.1's real head count against real world sizes.** 24 heads divides 4
  and 8 but not 16, so the shipped model takes different paths on different nodes.

Both cuts are checked against the same reference -- the unsharded table, built
redundantly on every rank -- and bit-exactly, since neither cut changes any
arithmetic. The sum all-reduce is exact too: every element has exactly one nonzero
addend, the rest having been masked to zero.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams, AllReduceStrategy, allgather
from tensorrt_llm._torch.modules.engram import ShardedFp8MultiHeadEmbedding
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# MPIPoolExecutor leaks a worker thread on first use; keep CI green.
pytestmark = pytest.mark.threadleak(enabled=False)

D = 128
BLOCK_SIZE = 32
NUM_TOKENS = 29

# DeepSeek-V4.1's own head count is 24 (`(engram_max_ngram_size - 1) *
# engram_n_heads`), which divides every world size reachable on one node. 18 divides
# none of them, and stands in for the case the shipped model actually hits at TP=16 --
# a head count that does not divide the ranks, so the row cut has to take over.
HEAD_COUNTS = (24, 18)


def _buckets(num_heads: int) -> list:
    """Deliberately unequal bucket sizes, so no two heads span the same rows.

    Equal buckets would let a shard whose row range is off by a whole bucket still
    line up, which is exactly the mistake the head cut can make.
    """
    return [16 * (1 + (7 * j) % 11) for j in range(num_heads)]


def _checkpoint(num_heads: int, seed: int = 11) -> dict:
    """The full unsharded table, regenerated identically on every rank.

    Built from a seed rather than broadcast because it has to be byte-identical
    across processes for the comparison to mean anything, and a CPU generator with
    a fixed seed gives that for free -- whereas shipping it through cloudpickle to
    every worker would put an 8-way copy of it in the MPI messages.

    e4m3 codes 0x7F/0xFF (NaN) are excluded and the e8m0 exponents kept near 127,
    so a mismatch cannot hide behind a NaN or a saturated product.
    """
    gen = torch.Generator().manual_seed(seed)
    rows = sum(_buckets(num_heads))
    magnitude = torch.randint(0, 127, (rows, D), dtype=torch.uint8, generator=gen)
    sign = torch.randint(0, 2, (rows, D), dtype=torch.uint8, generator=gen) << 7
    scale = torch.randint(120, 135, (rows, D // BLOCK_SIZE), dtype=torch.uint8, generator=gen)
    return {
        "weight": (magnitude | sign).view(torch.float8_e4m3fn),
        "scale": scale.view(torch.float8_e8m0fnu),
    }


def _indices(num_heads: int, seed: int = 5) -> torch.Tensor:
    """Bucket-local indices, as ``NgramHashMapping`` produces them.

    Head ``j`` indexes into ``[0, buckets[j])`` because the hash is taken mod that
    head's own bucket size. The head cut drops the shard mask on the strength of
    that invariant, so a test that violated it would be testing a case the model
    cannot produce.
    """
    gen = torch.Generator().manual_seed(seed)
    cols = [
        torch.randint(0, n, (NUM_TOKENS, 1), dtype=torch.long, generator=gen)
        for n in _buckets(num_heads)
    ]
    return torch.cat(cols, dim=1).cuda()


def _table(num_heads: int, tp_size: int, tp_rank: int) -> ShardedFp8MultiHeadEmbedding:
    with torch.device("cuda"):
        table = ShardedFp8MultiHeadEmbedding(
            list_of_N=_buckets(num_heads),
            D=D,
            block_size=BLOCK_SIZE,
            dtype=torch.bfloat16,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
    table.load_weights(_checkpoint(num_heads))
    return table


def run_single_rank(tensor_parallel_size, single_rank_forward_func, *args):
    """Wrapper used by MPIPoolExecutor; matches test_allgather.py."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(tensor_parallel_size, rank, *args)
    except Exception:
        traceback.print_exc()
        raise
    return True


@torch.inference_mode()
def run_engram_shard_roundtrip(tp_size: int, tp_rank: int, num_heads: int):
    """Shard, look up, combine -- and land back on the unsharded answer."""
    mapping = Mapping(world_size=tp_size, rank=tp_rank, tp_size=tp_size)
    table = _table(num_heads, tp_size, tp_rank)
    indices = _indices(num_heads)

    expect_head_cut = num_heads % tp_size == 0
    assert table.shard_heads == expect_head_cut, (
        f"{num_heads} heads over {tp_size} ranks should have taken the "
        f"{'head' if expect_head_cut else 'row'} cut"
    )

    # Exactly what `DeepseekV41Engram.precompute` does: flatten head-major, then
    # combine according to the cut.
    local = table(indices).flatten(start_dim=-2)
    if table.shard_heads:
        assert local.shape == (NUM_TOKENS, num_heads // tp_size * D), (
            f"a head-sharded rank must contribute only its own columns, got {tuple(local.shape)}"
        )
        combined = allgather(local, mapping, dim=-1)
    else:
        assert local.shape == (NUM_TOKENS, num_heads * D), (
            f"a row-sharded rank must contribute full width, got {tuple(local.shape)}"
        )
        # NCCL rather than AUTO: tactic selection is implicated in an unresolved
        # correctness bug on this model, and a nondeterministic strategy choice here
        # would make a failure unattributable.
        combined = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)(
            local, all_reduce_params=AllReduceParams(enable_allreduce=True)
        )

    reference = _table(num_heads, tp_size=1, tp_rank=0)(indices).flatten(start_dim=-2)
    # Guards the comparison itself: two all-zero tensors would agree bit-exactly and
    # say nothing about the wiring.
    assert reference.count_nonzero() > 0, "degenerate reference table"
    torch.testing.assert_close(combined, reference, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("num_heads", HEAD_COUNTS, ids=lambda n: f"heads:{n}")
@pytest.mark.parametrize("world_size", [4, 8], ids=lambda w: f"world:{w}")
def test_engram_shard_roundtrip(world_size, num_heads):
    """Either cut, combined by its own collective, equals the unsharded table.

    The grid crosses both cuts with both world sizes: 24 heads divides 4 and 8 and so
    takes the head cut on each, 18 divides neither and so takes the row cut on each.
    """
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.cuda.device_count()}")

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_engram_shard_roundtrip, num_heads)] * world_size),
        )
        for r in results:
            assert r is True
