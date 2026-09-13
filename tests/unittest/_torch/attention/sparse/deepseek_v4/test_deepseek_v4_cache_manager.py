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

import inspect
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DEEPSEEK_V4_VARIANT,
    DEEPSEEK_V41_VARIANT,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
)
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.pyexecutor._util import CacheCost
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _effective_partial_reuse,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._utils import binding_to_torch_dtype
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.logger import logger as trtllm_logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BatchDesc, KVCacheDesc, PageIndexMode
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

_RequestCache = Dict[
    Tuple[int, DeepseekV4AttentionType],  # (layer index, attention type)
    Tuple[torch.Tensor, torch.Tensor | None],  # (values tensor, scales tensor)
]


@pytest.mark.parametrize(("avg_seq_len", "expected"), [(None, 1024), (256, 256)])
def test_typical_seq_len_preserves_deepseek_v4_fallback(
    avg_seq_len: int | None, expected: int
) -> None:
    manager = object.__new__(DeepseekV4CacheManager)
    manager.max_seq_len = 1024

    assert manager._get_typical_seq_len(KvCacheConfig(avg_seq_len=avg_seq_len)) == expected


def _resolve_capturing_warnings(
    monkeypatch, requested: bool, supported: bool, manager_name: str
) -> Tuple[bool, List[str]]:
    """Run the partial-reuse resolver with TensorRT-LLM's logger captured.

    Not ``caplog``: ``tensorrt_llm.logger.logger`` is TensorRT-LLM's own object, not a
    stdlib ``logging.Logger``, so it never propagates to the root handler pytest
    installs and ``caplog.text`` stays empty no matter what was logged. An assertion
    written against it would be unfalsifiable in one direction and unsatisfiable in the
    other. Patching the method the resolver actually calls keeps both honest.

    Patched on the logger object rather than on the importing module, because
    ``kv_cache_manager_v2`` binds it with ``from tensorrt_llm.logger import logger`` --
    both names are the same singleton, and setting the attribute on the object holds
    whichever name the code under test happens to reach it by. Note that
    ``import tensorrt_llm.logger`` does not give you the module: the package re-exports
    the instance under that name, shadowing its own submodule.
    """
    messages: List[str] = []
    monkeypatch.setattr(trtllm_logger, "warning", messages.append)
    return _effective_partial_reuse(requested, supported, manager_name), messages


class TestPartialReuseOptOut:
    """Partial-block reuse must stay off for this manager.

    Why these are worth their upkeep: the defect they guard was found empirically and
    is *silent*. One fixed prompt, greedy decoding, one process per arm --
    ``enable_block_reuse=False``, ``enable_block_reuse=True``, and a cold default
    process all emitted byte-identical ids, while the second ``generate()`` call in a
    reuse-on process diverged at decode step 2 and produced a different, still fluent,
    still on-topic paragraph. Clearing only ``enable_partial_reuse`` restored exact
    agreement on an otherwise identical second call, so it is the whole factor and not
    merely correlated with one: being the second call is not sufficient, which is what
    rules out request-to-request state leakage as well as full-block reuse.

    The mechanism is structural rather than numerical. The reuse radix tree matches at
    one global token granularity, but for the COMPRESS / COMPRESSOR / INDEXER roles
    the block size is *not* ``tokens_per_block`` (see ``compressed_block_sizes``): a
    row covers N tokens, or is pooled over a window of them. A prefix that stops
    mid-block therefore hands the new sequence rows covering tokens that prefix does
    not own. They are well-formed, so nothing raises and no assertion fires -- which is
    exactly why the opt-out needs a test rather than a comment.

    These tests pin the plumbing only: the flag, the resolution rule, and the hand-off
    to the backend config. They cannot show the model emitting the right tokens, and
    they did not catch that the first version of the fix assigned the attribute after
    ``_build_base_config`` had already read it. The construction tests elsewhere in
    this file did.
    """

    def test_deepseek_v4_manager_opts_out(self) -> None:
        assert DeepseekV4CacheManager._supports_partial_reuse is False

    def test_the_base_manager_still_supports_it(self) -> None:
        # The opt-out has to be a narrowing of a supported default. If someone
        # "simplifies" by flipping the base to False, partial reuse silently dies for
        # every model in the repo and this file's opt-out looks redundant.
        assert KVCacheManagerV2._supports_partial_reuse is True

    @pytest.mark.parametrize(
        ("requested", "supported", "expected"),
        [
            (True, True, True),
            (True, False, False),  # the only case that overrides the caller
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_resolution_truth_table(self, requested: bool, supported: bool, expected: bool) -> None:
        assert _effective_partial_reuse(requested, supported, "FakeManager") is expected

    def test_forcing_it_off_warns_and_names_the_manager(self, monkeypatch) -> None:
        # Silent would be the wrong failure mode in the other direction: a caller who
        # asked for partial reuse and did not get it would have no way to explain
        # their hit rate. The manager name has to be in the message because the whole
        # point is that only *some* managers opt out.
        result, messages = _resolve_capturing_warnings(
            monkeypatch, True, False, "DeepseekV4CacheManager"
        )

        assert result is False
        assert len(messages) == 1
        assert "DeepseekV4CacheManager" in messages[0]
        assert "enable_partial_reuse=False" in messages[0]

    @pytest.mark.parametrize(("requested", "supported"), [(True, True), (False, False)])
    def test_honouring_the_request_stays_quiet(
        self, monkeypatch, requested: bool, supported: bool
    ) -> None:
        # Both quiet cases, not just one: warning when the caller asked for partial
        # reuse and got it would be noise on every supported manager, and warning when
        # they never asked for it would be noise on every unsupported one.
        result, messages = _resolve_capturing_warnings(
            monkeypatch, requested, supported, "SomeOtherManager"
        )

        assert result is requested
        assert messages == []

    def test_the_backend_config_reads_the_effective_value(self) -> None:
        """``_build_base_config`` must not re-read the raw ``kv_cache_config``.

        This is the regression that would silently undo the whole fix: the flag and
        the warning would still be there, the unit tests above would still pass, and
        the backend would still be handed ``True``. A source check rather than a
        behavioural one because ``_build_base_config`` needs a fully constructed
        manager, and standing one up would test the stub instead of the code. The
        sibling ``test_indexer_hadamard_coupling.py`` pins a hand-off the same way and
        for the same reason.
        """
        source = inspect.getsource(KVCacheManagerV2._build_base_config)
        assert "enable_partial_reuse=self.enable_partial_reuse" in source
        assert "enable_partial_reuse=kv_cache_config.enable_partial_reuse" not in source


def test_cache_size_estimation_uses_model_attention_layer_count():
    class FakeModelConfig:
        sparse_attention_config = SimpleNamespace(
            index_head_dim=128,
            compress_ratios=[1, 4, 1, 128],
            indexer_k_dtype="fp8",
            window_size=128,
        )
        pretrained_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        quant_config = None

        def get_num_attention_layers(self) -> int:
            return len(self.sparse_attention_config.compress_ratios)

    size_per_token = DeepseekV4CacheManager.get_cache_size_per_token(
        FakeModelConfig(),
        Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        tokens_per_block=128,
        is_disagg=True,
    )

    cost = CacheCost.from_raw(size_per_token)
    assert cost.slope > 0
    assert cost.intercept == 0


def test_quota_from_max_tokens_models_context_swa_scratch():
    manager = object.__new__(DeepseekV4CacheManager)
    manager.pp_layers = [0, 1]
    manager._compress_ratios = [4, 4]
    manager.dtype = DataType.BF16
    manager.head_dim = 512 + 64
    manager.index_head_dim = 128
    manager._indexer_k_dtype = "fp8"
    manager.use_fp8_ds_mla = False
    manager._swa_window_size = 128
    manager._max_draft_len = 0
    manager._max_num_tokens = 1024
    manager.tokens_per_block = 128
    manager.max_batch_size = 2

    manager.enable_swa_scratch_reuse = True
    small_quota = manager._get_quota_from_max_tokens(1)
    large_quota = manager._get_quota_from_max_tokens(4096)
    scratch_delta = large_quota - small_quota

    assert small_quota < large_quota
    assert small_quota > manager._get_extra_quota_padding()
    assert manager._get_max_tokens_from_quota(small_quota) == 1
    assert manager._get_max_tokens_from_quota(large_quota) == 4096

    manager.enable_swa_scratch_reuse = False
    no_scratch_small_quota = manager._get_quota_from_max_tokens(1)
    no_scratch_large_quota = manager._get_quota_from_max_tokens(4096)
    no_scratch_delta = no_scratch_large_quota - no_scratch_small_quota

    assert no_scratch_small_quota < no_scratch_large_quota
    assert no_scratch_delta > scratch_delta
    assert manager._get_max_tokens_from_quota(no_scratch_large_quota) == 4096


def test_needed_resource_uses_context_swa_scratch_slope():
    manager = object.__new__(DeepseekV4CacheManager)
    manager.pp_layers = [0, 1]
    manager._compress_ratios = [4, 4]
    manager.dtype = DataType.BF16
    manager.head_dim = 512 + 64
    manager.index_head_dim = 128
    manager._indexer_k_dtype = "fp8"
    manager.use_fp8_ds_mla = False
    manager._swa_window_size = 128
    manager.tokens_per_block = 128
    manager.num_extra_kv_tokens = 0
    manager.enable_swa_scratch_reuse = True

    context_request = SimpleNamespace(
        is_context_init_state=True,
        is_generation_in_progress_state=False,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.CONTEXT_INIT,
        prompt_len=10,
        max_new_tokens=100,
    )
    longer_context_request = SimpleNamespace(
        is_context_init_state=True,
        is_generation_in_progress_state=False,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.CONTEXT_INIT,
        prompt_len=11,
        max_new_tokens=100,
    )
    generation_request = SimpleNamespace(
        is_context_init_state=False,
        is_generation_in_progress_state=True,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        prompt_len=10,
        max_new_tokens=20,
    )
    longer_generation_request = SimpleNamespace(
        is_context_init_state=False,
        is_generation_in_progress_state=True,
        is_generation_to_complete_state=False,
        is_disagg_generation_init_state=False,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        prompt_len=10,
        max_new_tokens=21,
    )

    non_sliding_attn_size_per_token = manager.get_cache_bytes_per_token()
    context_bytes = manager.get_needed_resource_to_completion(context_request)
    longer_context_bytes = manager.get_needed_resource_to_completion(longer_context_request)
    generation_bytes = manager.get_needed_resource_to_completion(generation_request)
    longer_generation_bytes = manager.get_needed_resource_to_completion(longer_generation_request)

    assert non_sliding_attn_size_per_token > 0
    assert context_bytes > context_request.prompt_len * non_sliding_attn_size_per_token
    assert longer_context_bytes - context_bytes > non_sliding_attn_size_per_token
    assert longer_generation_bytes - generation_bytes == non_sliding_attn_size_per_token


def _view_fp8_as_uint8(buffer: torch.Tensor) -> torch.Tensor:
    """View an FP8 buffer as uint8. Non-FP8 buffers are returned as-is."""
    if buffer.dtype == torch.float8_e4m3fn:
        return buffer.view(torch.uint8)
    return buffer


@pytest.fixture(params=[False, True], ids=["scratch_reuse_disabled", "scratch_reuse_enabled"])
def scratch_reuse_enabled(request) -> bool:
    return request.param


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
class TestDeepseekV4CacheManager:
    # deepseek_v4 specific param
    head_dim = 512
    index_head_dim = 128
    window_size = 128
    vocab_size = 129280
    sparse_layer_ratio = 4
    overlap_compress_layer_ratio = 4

    # indexer quantization config
    indexer_dtype = DataType.FP8
    indexer_scale_dtype = DataType.FLOAT

    # cache manager specific param
    tokens_per_block = 128

    @staticmethod
    def _attention_op_block_offsets_shape(
        cache_manager: DeepseekV4CacheManager, num_seqs: int
    ) -> tuple[int, int, int, int]:
        return (
            cache_manager.num_attention_op_pools,
            num_seqs,
            2,
            cache_manager.max_blocks_per_seq,
        )

    @staticmethod
    def _sliding_block_tables_shape(
        cache_manager: DeepseekV4CacheManager, num_seqs: int
    ) -> tuple[int, int, int, int]:
        return (
            cache_manager.num_local_layers,
            len(DEEPSEEK_V4_SLIDING_ATTENTION),
            num_seqs,
            cache_manager.max_blocks_per_seq,
        )

    def _is_compress_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses compression based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses compression (ratio > 1)
        """
        return compress_ratio > 1

    def _is_sparse_layer(self, compress_ratio: int) -> bool:
        """Check if a layer uses sparse attention based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses sparse attention (ratio == 4)
        """
        return compress_ratio == self.sparse_layer_ratio

    def _is_overlap_compressor(self, compress_ratio: int) -> bool:
        """Check if a layer uses overlap compressor based on its compress ratio.

        Args:
            compress_ratio: The compression ratio for the layer

        Returns:
            True if the layer uses overlap compressor (ratio == 4)
        """
        return compress_ratio == self.overlap_compress_layer_ratio

    def _get_window_size(self, compress_ratio: int, attn_type: DeepseekV4AttentionType) -> int:
        """Get the window size for a layer based on its compress ratio and attention type.

        Args:
            compress_ratio: The compression ratio for the layer
            attn_type: The attention type

        Returns:
            The window size for the layer
        """
        state_factor = 2 if self._is_overlap_compressor(compress_ratio) else 1
        if attn_type == DeepseekV4AttentionType.SWA:
            return self.window_size
        elif attn_type in [
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            return state_factor * compress_ratio
        elif attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            return None

    def _create_deepseek_v4_cache_manager(
        self,
        tokens_per_block: int,
        max_batch_size: int,
        max_seq_len: int,
        compress_ratios: List[int],
        dtype: DataType,
        compressor_dtype: DataType,
        max_input_len: Optional[int] = None,
        is_draft: bool = False,
        tp_size: int = 1,
        enable_attention_dp: bool = False,
        spec_config: object | None = None,
        indexer_k_dtype: str | None = None,
        enable_swa_scratch_reuse: bool = True,
        variant: str = DEEPSEEK_V4_VARIANT,
        kv_source_layer_ids: Optional[List[int]] = None,
    ) -> Tuple[DeepseekV4CacheManager, DeepSeekV4SparseAttentionConfig]:
        """Helper to create a DeepseekV4CacheManager for testing.

        ``variant`` selects which generation's ``compress_ratios`` encoding the
        list is written in. It is passed through rather than inferred because the
        same integer means different things in each: V4's ``1`` is an SWA-only
        layer while V4.1's ``1`` is a genuine unpooled long-range layer.
        """

        # Create sparse attention config
        config_kwargs = {}
        if indexer_k_dtype is not None:
            config_kwargs["indexer_k_dtype"] = indexer_k_dtype
        if kv_source_layer_ids is not None:
            config_kwargs["kv_source_layer_ids"] = kv_source_layer_ids
        sparse_attn_config = DeepSeekV4SparseAttentionConfig(
            index_head_dim=self.index_head_dim,
            window_size=self.window_size,
            variant=variant,
            compress_ratios=compress_ratios,
            **config_kwargs,
        )

        # Create KV cache config
        if max_input_len is None:
            max_input_len = max_seq_len
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=max_seq_len * max_batch_size,
            event_buffer_max_size=0,
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
        )

        # Create mapping (single GPU, no parallelism)
        mapping = Mapping(
            world_size=tp_size,
            rank=0,
            tp_size=tp_size,
            pp_size=1,
            enable_attention_dp=enable_attention_dp,
        )

        # Create cache manager
        cache_manager = DeepseekV4CacheManager(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheTypeCpp.SELFKONLY,
            num_layers=len(compress_ratios),
            num_kv_heads=1,
            head_dim=self.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            mapping=mapping,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
            vocab_size=self.vocab_size,
            max_num_tokens=max_batch_size * (max_input_len + 1),
            sparse_attn_config=sparse_attn_config,
            is_draft=is_draft,
            spec_config=spec_config,
        )

        return cache_manager, sparse_attn_config

    # ------------------------------------------------------------------
    # DeepSeek-V4.1 compress-ratio encoding
    # ------------------------------------------------------------------
    # V4.1's `compress_ratios` are true pooling factors from {0, 1, 2}, not the
    # role sentinels {0, 4, 128} that V4 uses. The collision is on `1`: V4 only
    # ever produces it by normalizing a 0 (i.e. an SWA-only layer), while a V4.1
    # `1` is a genuine long-range layer whose compressor is the identity. These
    # tests pin the resulting role map so a future refactor cannot quietly
    # collapse the two encodings back together.

    # 6 layers: SWA-only, pooled, pooled, unpooled, unpooled, SWA-only. Shaped
    # like the real 40-layer checkpoint (leading 0s, a run of 2s, a run of 1s)
    # but small enough to allocate on one GPU.
    V41_RATIOS = [0, 2, 2, 1, 1, 0]

    def _layer_roles(
        self, cache_manager: DeepseekV4CacheManager, layer: int
    ) -> set[DeepseekV4AttentionType]:
        return {
            attn_type
            for (layer_idx, attn_type) in cache_manager._layer_attn_to_layer_id
            if layer_idx == layer
        }

    def test_v41_ratios_assign_roles_by_pooling_factor(self) -> None:
        """Each V4.1 ratio maps to exactly the buffers its reference module needs."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=2,
            max_seq_len=1024,
            compress_ratios=self.V41_RATIOS,
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            variant=DEEPSEEK_V41_VARIANT,
        )
        try:
            for layer, ratio in enumerate(self.V41_RATIOS):
                roles = self._layer_roles(cache_manager, layer)
                expected = {DeepseekV4AttentionType.SWA}
                if ratio > 0:
                    # Every long-range layer is indexed (model.py:775-778), so
                    # COMPRESS never appears without INDEXER_COMPRESS.
                    expected |= {
                        DeepseekV4AttentionType.COMPRESS,
                        DeepseekV4AttentionType.INDEXER_COMPRESS,
                    }
                if ratio > 1:
                    # Only a genuinely pooling compressor carries a cross-step
                    # accumulator; ratio 1 is `norm(wkv(x))` (model.py:461).
                    expected |= {
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    }
                assert roles == expected, f"layer {layer} (ratio {ratio}) got {roles}"

            # V4.1's Indexer owns no compressor of its own (model.py:496-525), so
            # neither indexer-compressor role may be allocated anywhere.
            all_roles = {attn_type for _, attn_type in cache_manager._layer_attn_to_layer_id}
            assert DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV not in all_roles
            assert DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE not in all_roles

            # Roles added together share one manager LayerId, which is what lets
            # the COMPRESS/INDEXER_COMPRESS pair share a page table.
            for layer, ratio in enumerate(self.V41_RATIOS):
                if ratio == 0:
                    continue
                ids = cache_manager._layer_attn_to_layer_id
                assert (
                    ids[layer, DeepseekV4AttentionType.COMPRESS]
                    == ids[layer, DeepseekV4AttentionType.INDEXER_COMPRESS]
                )
                if ratio > 1:
                    assert (
                        ids[layer, DeepseekV4AttentionType.COMPRESSOR_KV]
                        == ids[layer, DeepseekV4AttentionType.COMPRESSOR_SCORE]
                    )
        finally:
            cache_manager.shutdown()

    def test_v41_dependent_layers_alias_their_kv_source(self) -> None:
        """A shared compressed cache is allocated once, on the source layer.

        V4.1 has 4 KV sources for 38 long-range layers, so allocating per layer
        would reserve ~9.5x the compressed footprint to hold duplicates of the
        same bytes. The dependents resolve to the source's ``LayerId`` instead,
        which is what makes ``get_buffers(dependent, COMPRESS)`` return the
        source's pages with no copy and no special case at the call sites.
        """
        # Sources at 1 and 3: layer 2 depends on 1 (ratio 2), layer 4 on 3
        # (ratio 1). Both groups are ratio-uniform, as they are in the real
        # config -- aliasing across a ratio boundary is asserted against.
        ratios = self.V41_RATIOS
        kv_sources = [1, 3]
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=2,
            max_seq_len=1024,
            compress_ratios=ratios,
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            variant=DEEPSEEK_V41_VARIANT,
            kv_source_layer_ids=kv_sources,
        )
        compressed_roles = (
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        )
        try:
            ids = cache_manager._layer_attn_to_layer_id
            for dependent, source in ((2, 1), (4, 3)):
                for role in compressed_roles:
                    assert ids[dependent, role] == ids[source, role], (
                        f"layer {dependent} must share layer {source}'s {role.name}"
                    )
            # The two sources must not collapse into each other: aliasing has to
            # be "dependent -> its own source", not "everything -> one pool".
            assert (
                ids[1, DeepseekV4AttentionType.COMPRESS] != ids[3, DeepseekV4AttentionType.COMPRESS]
            )

            # A dependent runs no compressor, so it carries no pooling
            # accumulator -- not even an aliased one. Layer 2 is ratio 2, which
            # *would* have COMPRESSOR_* if it owned its cache (asserted by
            # test_v41_ratios_assign_roles_by_pooling_factor), so this
            # distinguishes ownership from the ratio.
            assert self._layer_roles(cache_manager, 2) == {
                DeepseekV4AttentionType.SWA,
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            }
            assert DeepseekV4AttentionType.COMPRESSOR_KV in self._layer_roles(cache_manager, 1)

            aliased_ids = {
                role: {ids[layer, role] for layer, r in enumerate(ratios) if r > 0}
                for role in compressed_roles
            }
        finally:
            cache_manager.shutdown()

        # The saving is the point, so measure it rather than inferring it. Note
        # that `get_buffers` cannot serve as the witness: it returns the whole
        # *pool's* base view, so two layers in one pool report the same
        # `data_ptr()` whether or not they alias. The count of distinct LayerIds
        # is what tracks how many `AttentionLayerConfig`s were reserved.
        unaliased, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=2,
            max_seq_len=1024,
            compress_ratios=ratios,
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            variant=DEEPSEEK_V41_VARIANT,
            # No sources declared: `owns_compressed_kv` degrades to
            # `is_compress_layer`, i.e. the pre-sharing behaviour.
        )
        try:
            unaliased_ids = {
                role: {
                    unaliased._layer_attn_to_layer_id[layer, role]
                    for layer, r in enumerate(ratios)
                    if r > 0
                }
                for role in compressed_roles
            }
        finally:
            unaliased.shutdown()

        for role in compressed_roles:
            assert len(unaliased_ids[role]) == 4, (
                f"expected all four long-range layers to reserve their own {role.name}"
            )
            assert len(aliased_ids[role]) == len(kv_sources), (
                f"{role.name} must be reserved once per KV source, not once per layer; "
                f"got {len(aliased_ids[role])} for sources {kv_sources}"
            )

    def test_v41_aliasing_rejects_a_mixed_ratio_sharing_group(self) -> None:
        """A sharing group must be ratio-uniform, or the pooled positions differ.

        Compressed row *j* means token *j x ratio*, so a ratio-1 layer reading a
        ratio-2 layer's cache would silently attend to the wrong positions. The
        real config never does this (every group is uniform), which is exactly
        why a regression here would be invisible without a check.
        """
        with pytest.raises(AssertionError, match="ratio-uniform"):
            cache_manager, _ = self._create_deepseek_v4_cache_manager(
                tokens_per_block=self.tokens_per_block,
                max_batch_size=2,
                max_seq_len=1024,
                # Layer 3 (ratio 1) would alias onto source layer 1 (ratio 2).
                compress_ratios=self.V41_RATIOS,
                dtype=DataType.BF16,
                compressor_dtype=DataType.FLOAT,
                variant=DEEPSEEK_V41_VARIANT,
                kv_source_layer_ids=[1],
            )
            cache_manager.shutdown()

    def test_v41_block_sizes_and_pools_follow_the_pooling_factor(self) -> None:
        """Pooling divides the compressed block size; ratio 0/1 do not shrink it."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=2,
            max_seq_len=1024,
            compress_ratios=self.V41_RATIOS,
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            variant=DEEPSEEK_V41_VARIANT,
        )
        try:
            # A ratio-0 layer's entry is never read (it owns no compressed
            # buffers); it must still be a safe value rather than a div-by-zero.
            assert cache_manager.compressed_block_sizes == [
                self.tokens_per_block,
                self.tokens_per_block // 2,
                self.tokens_per_block // 2,
                self.tokens_per_block,
                self.tokens_per_block,
                self.tokens_per_block,
            ]

            # One COMPRESS pool per distinct ratio that has a compressed branch,
            # keyed by ratio rather than by the V4 {4, 128} literals. Both are
            # indexed, so both also get an INDEXER_COMPRESS pool.
            assert sorted(cache_manager.compress_pool_ptrs) == [1, 2]
            assert sorted(cache_manager._compress_pool_meta) == [1, 2]
            assert sorted(cache_manager._indexer_compress_pool_meta) == [1, 2]

            # With more than one indexer pool the manager must refuse to guess.
            # Silently returning the ratio-1 table for a ratio-2 layer would
            # scatter indexer keys into the wrong pages and read as an accuracy
            # bug rather than a wiring bug.
            with pytest.raises(RuntimeError, match="compress_ratio is required"):
                cache_manager.copy_batch_indexer_compress_block_tables(
                    host_block_table=torch.zeros(
                        2, cache_manager.max_blocks_per_seq, dtype=torch.int32
                    ),
                    request_ids=[],
                    beam_width=1,
                    num_contexts=0,
                    num_seqs=0,
                )
        finally:
            cache_manager.shutdown()

    @pytest.mark.parametrize(
        ("variant", "expect_long_range"),
        [(DEEPSEEK_V4_VARIANT, False), (DEEPSEEK_V41_VARIANT, True)],
        ids=["v4_sees_swa_only", "v41_sees_long_range"],
    )
    def test_ratio_one_means_different_things_per_variant(
        self, variant: str, expect_long_range: bool
    ) -> None:
        """The irreducible collision: ``1`` is SWA-only in V4, long-range in V4.1.

        ``[0, 1, 1, 0]`` is a legal ratio list under both encodings, which makes
        it the sharpest available control. Under V4, ``normalize_compress_ratios``
        rewrites the ``0``\\ s to ``1`` and ``is_compress_layer`` is ``ratio > 1``,
        so *no* layer gets a compressed cache. Under V4.1 the two ``1``\\ s are
        genuine unpooled long-range layers and must get one. A single int cannot
        encode both role and pooling factor, so if this test ever passes for both
        variants with the same expectation, the variant is being ignored.
        """
        ratios = [0, 1, 1, 0]
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=2,
            max_seq_len=1024,
            compress_ratios=ratios,
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            variant=variant,
        )
        try:
            long_range = {
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            }
            for layer, ratio in enumerate(ratios):
                roles = self._layer_roles(cache_manager, layer)
                if expect_long_range and ratio == 1:
                    assert roles == {DeepseekV4AttentionType.SWA} | long_range
                else:
                    assert roles == {DeepseekV4AttentionType.SWA}
            # Unpooled compressors keep no cross-step accumulator either way.
            all_roles = {attn_type for _, attn_type in cache_manager._layer_attn_to_layer_id}
            assert DeepseekV4AttentionType.COMPRESSOR_KV not in all_roles
            assert DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV not in all_roles
            # A single unambiguous indexer pool: V4.1 may then be asked for the
            # compatibility table without naming a ratio.
            assert sorted(cache_manager._indexer_compress_pool_meta) == (
                [1] if expect_long_range else []
            )
        finally:
            cache_manager.shutdown()

    def _create_request(self, request_id: int, prompt_len: int) -> LlmRequest:
        """Helper to create a test LlmRequest.

        Args:
            request_id: Unique request identifier
            prompt_len: Prompt length (number of tokens)

        Returns:
            LlmRequest instance
        """
        input_tokens = list(range(prompt_len))
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=1024,
            input_tokens=input_tokens,
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )

        return request

    def _rand_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if dtype in (torch.uint8, torch.float8_e4m3fn):
            # Use uint8 for both uint8 and FP8 (same 1-byte layout)
            return torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
        else:
            return torch.randn(shape, dtype=dtype, device=device) * 1000.0

    def _create_random_cache(
        self,
        seq_len: int,
        head_dim: int,
        sparse_attn_config: DeepSeekV4SparseAttentionConfig,
        dtype: torch.dtype,
        compressor_dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> _RequestCache:
        """Helper to create random cache values for all layers and attention types.

        Args:
            seq_len: Sequence length
            head_dim: Head dimension for regular attention
            sparse_attn_config: Sparse attention configuration

        Returns:
            Dictionary mapping (layer_idx, attn_type) to (values, scales) tuples.
            scales is None for non-quantized attention types.
        """
        device = device or torch.device("cuda")
        cache: _RequestCache = {}

        for layer, ratio in enumerate(sparse_attn_config.compress_ratios):
            is_overlap = self._is_overlap_compressor(ratio)

            cache[layer, DeepseekV4AttentionType.SWA] = (
                self._rand_tensor((seq_len, head_dim), dtype, device),
                None,
            )

            if self._is_compress_layer(ratio):
                compressor_dim = 2 * head_dim if is_overlap else head_dim
                cache[layer, DeepseekV4AttentionType.COMPRESS] = (
                    self._rand_tensor((seq_len // ratio, head_dim), dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.COMPRESSOR_KV] = (
                    self._rand_tensor((seq_len, compressor_dim), compressor_dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.COMPRESSOR_SCORE] = (
                    self._rand_tensor((seq_len, compressor_dim), compressor_dtype, device),
                    None,
                )

            if self._is_sparse_layer(ratio):
                # Indexer KV cache stores raw quantized bytes plus raw scale bytes.
                indexer_dim = sparse_attn_config.index_head_dim
                indexer_num_tokens = seq_len // ratio
                indexer_k_dtype = sparse_attn_config.indexer_k_dtype
                value_dim, scale_dim, scale_dtype = self._indexer_cache_layout(indexer_k_dtype)
                indexer_values = self._rand_tensor(
                    (indexer_num_tokens, value_dim), torch.uint8, device
                )
                indexer_scales = self._rand_tensor(
                    (indexer_num_tokens, scale_dim), scale_dtype, device
                )
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESS] = (
                    indexer_values,
                    indexer_scales,
                )

                indexer_compressor_dim = 2 * indexer_dim if is_overlap else indexer_dim
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV] = (
                    self._rand_tensor((seq_len, indexer_compressor_dim), compressor_dtype, device),
                    None,
                )
                cache[layer, DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE] = (
                    self._rand_tensor((seq_len, indexer_compressor_dim), compressor_dtype, device),
                    None,
                )

        return cache

    def _indexer_cache_layout(self, indexer_k_dtype: str) -> Tuple[int, int, torch.dtype]:
        if indexer_k_dtype == "fp8":
            return self.index_head_dim, self.index_head_dim // 128, torch.float32
        if indexer_k_dtype == "fp4":
            return self.index_head_dim // 2, self.index_head_dim // 32, torch.uint8
        raise ValueError(f"Unsupported indexer_k_dtype {indexer_k_dtype!r}")

    def _split_blockwise_buffer(
        self,
        buffer: torch.Tensor,
        indexer_k_dtype: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split an indexer K cache buffer into value and scale buffers.

        Args:
            buffer: The raw indexer K cache buffer.

        Returns:
            Tuple of (value_buffer, scale_buffer)
        """
        num_blocks, tokens_per_block, bytes_per_token = buffer.shape
        bytes_per_block = bytes_per_token * tokens_per_block
        value_dim, scale_dim, scale_dtype = self._indexer_cache_layout(indexer_k_dtype)
        scale_bytes = scale_dim * scale_dtype.itemsize

        # Get value buffer
        value_shape = (num_blocks, tokens_per_block, value_dim)
        value_stride = (bytes_per_block, value_dim, 1)
        value_buffer = buffer.as_strided(value_shape, value_stride, 0).view(torch.uint8)

        # Get scale buffer
        scale_shape = (num_blocks, tokens_per_block, scale_bytes)
        scale_stride = (bytes_per_block, scale_bytes, 1)
        scale_offset = value_dim * tokens_per_block
        scale_buffer = buffer.as_strided(scale_shape, scale_stride, scale_offset).view(scale_dtype)

        return value_buffer, scale_buffer

    def _prefill_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        values: torch.Tensor,
    ) -> None:
        """Write context values to a paged cache buffer.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            values: Values to write (shape: [seq_len, dim_per_token])
        """
        assert buffer.size(2) == values.size(1), f"{buffer.size(2)=} != {values.size(1)=}"
        tokens_per_block = buffer.size(1)
        seq_len, dim_per_token = values.shape

        num_blocks = (seq_len + tokens_per_block - 1) // tokens_per_block
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[:num_blocks]), (
            f"{block_indices[:num_blocks]=} contains BAD_PAGE_INDEX"
        )

        if seq_len % tokens_per_block != 0:
            # pad the values to the nearest multiple of tokens_per_block
            pad_len = tokens_per_block - (seq_len % tokens_per_block)
            values = torch.cat(
                [
                    values,
                    self._rand_tensor((pad_len, dim_per_token), values.dtype, values.device),
                ],
                dim=0,
            )

        values_blocks = values.reshape(num_blocks, tokens_per_block, dim_per_token)
        buffer[block_indices[:num_blocks]] = values_blocks

    def _decode_write_paged_cache(
        self,
        buffer: torch.Tensor,
        block_indices: List[int],
        token_idx: int,
        value: torch.Tensor,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            buffer: The cache buffer to write to (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block indices to write to
            token_idx: Index of the new token to write
            value: Value to write (shape: [dim_per_token])
        """
        assert buffer.size(2) == value.size(0), f"{buffer.size(2)=} != {value.size(0)=}"
        num_blocks, tokens_per_block, _ = buffer.shape

        block_idx = token_idx // tokens_per_block
        block_offset = token_idx % tokens_per_block
        assert block_idx < num_blocks, f"{block_idx=} >= {num_blocks=}"
        assert block_indices[block_idx] != BAD_PAGE_INDEX, (
            f"{block_indices[block_idx]=} == BAD_PAGE_INDEX"
        )

        buffer[block_indices[block_idx], block_offset] = value

    def _read_paged_cache(
        self, buffer: torch.Tensor, block_indices: List[int], seq_len: int, window_size: int | None
    ) -> torch.Tensor:
        """Read values from a paged cache buffer.

        Args:
            buffer: The cache buffer to read from (shape: [num_blocks, tokens_per_block, dim_per_token])
            block_indices: List of block/page indices to read from
            seq_len: Sequence length
            window_size: sliding window size to read from the cache

        Returns:
            Tensor containing the read values (shape: [seq_len, dim_per_token] or [window_size, dim_per_token]
            if window_size is given and seq_len > window_size)
        """
        _, tokens_per_block, dim_per_token = buffer.shape

        # check if all blocks within the window are valid
        end_block_idx = (seq_len + tokens_per_block - 1) // tokens_per_block
        if window_size is not None:
            start_block_idx = (seq_len - window_size + tokens_per_block - 1) // tokens_per_block
        else:
            start_block_idx = 0
        assert all(idx != BAD_PAGE_INDEX for idx in block_indices[start_block_idx:end_block_idx]), (
            f"{block_indices[start_block_idx:end_block_idx]=} contains BAD_PAGE_INDEX"
        )

        # read values from the cache
        values = buffer[block_indices].reshape(-1, dim_per_token)[:seq_len]
        if window_size is not None and seq_len > window_size:
            values = values[-window_size:]
        return values

    def _get_page_indices(
        self,
        req: LlmRequest,
        cache_manager: DeepseekV4CacheManager,
        layer_idx: int,
        attn_type: DeepseekV4AttentionType,
        num_contexts: int = 1,
    ) -> torch.Tensor:
        if attn_type == DeepseekV4AttentionType.COMPRESS:
            compress_block_tables = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            cache_manager.copy_batch_compress_block_tables(
                compress_block_tables,
                [req.py_request_id],
                compress_ratio=cache_manager._compress_ratios[layer_idx],
                beam_width=1,
                num_contexts=num_contexts,
                num_seqs=1,
            )
            return compress_block_tables[0]

        if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            host_block_table = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            cache_manager.copy_batch_indexer_compress_block_tables(
                host_block_table,
                [req.py_request_id],
                beam_width=1,
                num_contexts=num_contexts,
                num_seqs=1,
            )
            return host_block_table[0]

        sliding_block_tables = torch.empty(
            self._sliding_block_tables_shape(cache_manager, 1),
            dtype=torch.int32,
            device="cuda",
        )
        with torch.cuda.stream(cache_manager._stream):
            cache_manager.compute_sliding_block_tables(
                [req.py_request_id],
                num_contexts=num_contexts,
            )
            cache_manager.copy_batch_sliding_block_tables(
                sliding_block_tables,
                [req.py_request_id],
                num_contexts=num_contexts,
                num_seqs=1,
            )
        cache_manager._stream.synchronize()
        return sliding_block_tables.cpu()[
            cache_manager.layer_offsets[layer_idx],
            attn_type.value,
            0,
        ]

    def _prepare_mixed_copy_batch(
        self,
        cache_manager: DeepseekV4CacheManager,
        prompt_len: int,
    ) -> tuple[list[LlmRequest], int]:
        requests = [self._create_request(request_id, prompt_len) for request_id in range(3)]
        for req in requests:
            assert cache_manager.prepare_context(req)
            assert cache_manager.resize_context(req, req.context_chunk_size)

        gen_req = requests[-1]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests_last_chunk = [gen_req]
        gen_req.context_current_position = prompt_len
        gen_req.add_new_token(prompt_len, 0)
        cache_manager.update_context_resources(scheduled_batch)
        cache_manager.update_resources(scheduled_batch)
        assert cache_manager.try_allocate_generation(gen_req)
        return requests, len(requests) - 1

    def _reference_copy_batch_page_indices(
        self,
        cache_manager: DeepseekV4CacheManager,
        request_ids: list[int],
        num_contexts: int,
        layer_idx: int,
        attn_type: DeepseekV4AttentionType,
        page_index_mode: PageIndexMode,
    ) -> torch.Tensor:
        layer_id = cache_manager._layer_attn_to_layer_id[layer_idx, attn_type]
        pool_id = cache_manager.layer_to_pool_mapping_dict[layer_id]
        converter = cache_manager.impl.get_page_index_converter(layer_id, attn_type.role)
        copy_idx = cache_manager.index_mapper.get_copy_index(request_ids, num_contexts, 1)

        expected = torch.full(
            (len(request_ids), cache_manager.max_blocks_per_seq),
            BAD_PAGE_INDEX,
            dtype=torch.int32,
            device="cpu",
        )
        for row, request_id in enumerate(request_ids):
            base_indices = cache_manager.host_kv_cache_block_offsets[
                pool_id,
                int(copy_idx[row]),
                0,
            ].tolist()
            scratch = None
            if row < num_contexts and page_index_mode == PageIndexMode.PER_LAYER:
                scratch = cache_manager.kv_cache_map[request_id].get_scratch_desc(pool_id)
            converted = converter(base_indices, page_index_mode, scratch)
            expected[row, : len(converted)] = torch.tensor(converted, dtype=torch.int32)
        return expected

    def _write_request_prefill(
        self,
        req: LlmRequest,
        prompt_len: int,
        cache_manager: DeepseekV4CacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Write cache values for a request to the cache manager.

        Args:
            req: The request to write cache for
            prompt_len: Prompt length
            cache_manager: The cache manager instance
            cache_values: Request's cache to write
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), (values, scales) in cache_values.items():
            page_indices = self._get_page_indices(
                req,
                cache_manager,
                layer_idx,
                attn_type,
            )

            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                seq_len = prompt_len // compress_ratios[layer_idx]
            else:
                seq_len = prompt_len

            buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer_idx, attn_type))
            if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                values_buffer, scales_buffer = self._split_blockwise_buffer(
                    buffer, cache_manager._indexer_k_dtype
                )
                self._prefill_write_paged_cache(
                    buffer=values_buffer,
                    block_indices=page_indices,
                    values=values[:seq_len],
                )
                self._prefill_write_paged_cache(
                    buffer=scales_buffer,
                    block_indices=page_indices,
                    values=scales[:seq_len],
                )
            else:
                self._prefill_write_paged_cache(
                    buffer=buffer,
                    block_indices=page_indices,
                    values=values[:seq_len],
                )

    def _write_request_decode(
        self,
        req: LlmRequest,
        token_idx: int,
        cache_manager: DeepseekV4CacheManager,
        cache_values: _RequestCache,
    ) -> None:
        """Simulate the decode phrase. Write one new token to the cache.

        Args:
            req: The request to write cache for
            token_idx: Index of the new token to write
            cache_manager: The cache manager instance
            cache_values: Request's cache to write
        """
        compress_ratios = cache_manager._compress_ratios
        for (layer_idx, attn_type), (values, scales) in cache_values.items():
            block_indices = self._get_page_indices(
                req,
                cache_manager,
                layer_idx,
                attn_type,
            )

            compressed_token_idx = token_idx
            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                if (token_idx + 1) % compress_ratios[layer_idx] != 0:
                    # skip if current token will not trigger compression
                    continue
                compressed_token_idx = token_idx // compress_ratios[layer_idx]

            buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer_idx, attn_type))
            if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                values_buffer, scales_buffer = self._split_blockwise_buffer(
                    buffer, cache_manager._indexer_k_dtype
                )
                self._decode_write_paged_cache(
                    buffer=values_buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=values[compressed_token_idx],
                )
                self._decode_write_paged_cache(
                    buffer=scales_buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=scales[compressed_token_idx],
                )
            else:
                self._decode_write_paged_cache(
                    buffer=buffer,
                    block_indices=block_indices,
                    token_idx=compressed_token_idx,
                    value=values[compressed_token_idx],
                )

    def _read_request(
        self,
        req: LlmRequest,
        seq_len: int,
        cache_manager: DeepseekV4CacheManager,
        compress_ratios: List[int],
    ) -> _RequestCache:
        """Read cache values for a request from the cache manager.

        Args:
            req: The request to read cache for
            seq_len: Sequence length
            cache_manager: The cache manager instance
            compress_ratios: Compression ratios for each layer

        Returns:
            Request's cache
        """
        cache_values: _RequestCache = {}
        for layer, ratio in enumerate(compress_ratios):
            attn_types = [DeepseekV4AttentionType.SWA]
            if self._is_compress_layer(ratio):
                attn_types.extend(
                    [
                        DeepseekV4AttentionType.COMPRESS,
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    ]
                )
            if self._is_sparse_layer(ratio):
                attn_types.extend(
                    [
                        DeepseekV4AttentionType.INDEXER_COMPRESS,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
                    ]
                )

            # read cache values for each attention type
            for attn_type in attn_types:
                page_indices = self._get_page_indices(
                    req,
                    cache_manager,
                    layer,
                    attn_type,
                )
                if attn_type in [
                    DeepseekV4AttentionType.COMPRESS,
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                ]:
                    attn_len = seq_len // ratio
                else:
                    attn_len = seq_len
                window_size = self._get_window_size(ratio, attn_type)

                buffer = _view_fp8_as_uint8(cache_manager.get_buffers(layer, attn_type))
                if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                    values_buffer, scales_buffer = self._split_blockwise_buffer(
                        buffer, cache_manager._indexer_k_dtype
                    )
                    values = self._read_paged_cache(
                        buffer=values_buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                    scales = self._read_paged_cache(
                        buffer=scales_buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                else:
                    values = self._read_paged_cache(
                        buffer=buffer,
                        block_indices=page_indices,
                        seq_len=attn_len,
                        window_size=window_size,
                    )
                    scales = None

                cache_values[layer, attn_type] = (values, scales)

        return cache_values

    def _assert_cache_equal(
        self, seq_len: int, compress_ratios: List[int], expect: _RequestCache, actual: _RequestCache
    ) -> None:
        """Assert that two cache dictionaries contain equal values.

        Args:
            seq_len: Sequence length
            compress_ratios: Compression ratios for each layer
            expected: Expected cache values
            actual: Actual cache values read from cache manager
        """
        # Check that keys match
        assert set(expect.keys()) == set(actual.keys()), (
            f"Cache keys don't match. Expected: {set(expect.keys())}, Actual: {set(actual.keys())}"
        )

        # Check each tensor value
        for layer_idx, attn_type in expect.keys():
            if attn_type in [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]:
                attn_len = seq_len // compress_ratios[layer_idx]
            else:
                attn_len = seq_len

            expect_values, expect_scales = expect[layer_idx, attn_type]
            actual_values, actual_scales = actual[layer_idx, attn_type]

            # Slice to attention length
            expect_values = expect_values[:attn_len]
            if expect_scales is not None:
                expect_scales = expect_scales[:attn_len]

            # Apply window size if applicable
            window_size = self._get_window_size(compress_ratios[layer_idx], attn_type)
            if window_size is not None:
                expect_values = expect_values[-window_size:]
                if expect_scales is not None:
                    expect_scales = expect_scales[-window_size:]

            # Assert values match
            torch.testing.assert_close(
                actual_values,
                expect_values,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch for layer {layer_idx}, attention type {attn_type.name} (values)",
            )

            # Assert scales match (both should be None or both should be tensors)
            if expect_scales is None:
                assert actual_scales is None, (
                    f"Expected no scales for layer {layer_idx}, attention type {attn_type.name}, "
                    f"but got scales with shape {actual_scales.shape}"
                )
            else:
                assert actual_scales is not None, (
                    f"Expected scales for layer {layer_idx}, attention type {attn_type.name}, "
                    f"but got None"
                )
                torch.testing.assert_close(
                    actual_scales,
                    expect_scales,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Mismatch for layer {layer_idx}, attention type {attn_type.name} (scales)",
                )

    def test_max_num_tokens_is_used_by_base_config(self):
        max_batch_size = 2
        max_seq_len = 1024
        max_input_len = 127
        max_num_tokens = max_batch_size * (max_input_len + 1)
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_input_len=max_input_len,
            compress_ratios=[1, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )

        assert cache_manager.kv_cache_manager_py_config.typical_step == BatchDesc(
            [
                KVCacheDesc(capacity=max_num_tokens, history_length=0),
                KVCacheDesc(capacity=max_seq_len, history_length=max_seq_len - 1),
            ]
        )

    def test_indexer_cache_layout_default(self):
        """DeepSeek-V4 defaults to FP4 indexer K cache on Blackwell+."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )
        try:
            buffer = cache_manager.get_buffers(0, DeepseekV4AttentionType.INDEXER_COMPRESS)
            assert buffer.dtype == torch.uint8
            assert buffer.shape[-1] == 64 + 4
            assert cache_manager.quant_block_size == 32
        finally:
            cache_manager.shutdown()

    def test_indexer_cache_layout_fp8(self):
        """The legacy FP8 blockwise indexer K cache remains available."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            indexer_k_dtype="fp8",
        )
        try:
            buffer = cache_manager.get_buffers(0, DeepseekV4AttentionType.INDEXER_COMPRESS)
            assert buffer.dtype == torch.float8_e4m3fn
            assert buffer.shape[-1] == 128 + 4
            assert cache_manager.quant_block_size == 128
        finally:
            cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    @pytest.mark.parametrize("prompt_lens", [[512, 128, 160], [1024, 2048, 4096]])
    @pytest.mark.parametrize("num_generation_steps", [2, 100])
    def test_write_read_cache(
        self,
        compress_ratios: List[int],
        prompt_lens: List[int],
        num_generation_steps: int,
        dtype: DataType,
        compressor_dtype: DataType,
        scratch_reuse_enabled: bool,
    ):
        max_batch_size = len(prompt_lens)
        max_seq_len = max(prompt_lens) + num_generation_steps + 1
        max_input_len = max(prompt_lens)
        # Create cache manager and sparse attention config
        cache_manager, sparse_attn_config = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
            max_input_len=max_input_len,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        # Create requests and their cache values
        requests = list[LlmRequest]()
        try:
            cache_values = dict[int, _RequestCache]()
            for req_id, prompt_len in enumerate(prompt_lens):
                req = self._create_request(req_id, prompt_len)
                requests.append(req)

                # Generate random cache values for this request
                cache_values[req_id] = self._create_random_cache(
                    seq_len=prompt_len + num_generation_steps + 1,
                    head_dim=self.head_dim,
                    sparse_attn_config=sparse_attn_config,
                    dtype=binding_to_torch_dtype(dtype),
                    compressor_dtype=binding_to_torch_dtype(compressor_dtype),
                )

            # Simulate the prefill phrase
            scheduled_batch = ScheduledRequests()
            scheduled_batch.context_requests_last_chunk = requests
            for req in requests:
                cache_manager.prepare_context(req)
                cache_manager.resize_context(req, req.context_chunk_size)

            # Write context to cache
            for req in requests:
                self._write_request_prefill(
                    req=req,
                    prompt_len=prompt_lens[req.py_request_id],
                    cache_manager=cache_manager,
                    cache_values=cache_values[req.py_request_id],
                )

            # Update requests state and call update_resources
            for req in requests:
                req.context_current_position = prompt_lens[req.py_request_id]
                req.add_new_token(prompt_lens[req.py_request_id], 0)
            cache_manager.update_context_resources(scheduled_batch)
            cache_manager.update_resources(scheduled_batch)

            # Read context from cache and verify
            for req in requests:
                actual_cache_values = self._read_request(
                    req=req,
                    seq_len=prompt_lens[req.py_request_id],
                    cache_manager=cache_manager,
                    compress_ratios=compress_ratios,
                )
                self._assert_cache_equal(
                    seq_len=prompt_lens[req.py_request_id],
                    compress_ratios=compress_ratios,
                    expect=cache_values[req.py_request_id],
                    actual=actual_cache_values,
                )

            # Simulate the decode phrase
            for i in range(num_generation_steps):
                seq_lens = [prompt_len + i + 1 for prompt_len in prompt_lens]
                scheduled_batch = ScheduledRequests()
                scheduled_batch.generation_requests = requests
                for req in requests:
                    cache_manager.try_allocate_generation(req)

                # Write new token to cache
                for req in requests:
                    self._write_request_decode(
                        req=req,
                        token_idx=seq_lens[req.py_request_id] - 1,
                        cache_manager=cache_manager,
                        cache_values=cache_values[req.py_request_id],
                    )

                # Read context from cache and verify
                for req in requests:
                    actual_cache_values = self._read_request(
                        req=req,
                        seq_len=seq_lens[req.py_request_id],
                        cache_manager=cache_manager,
                        compress_ratios=compress_ratios,
                    )
                    self._assert_cache_equal(
                        seq_len=seq_lens[req.py_request_id],
                        compress_ratios=compress_ratios,
                        expect=cache_values[req.py_request_id],
                        actual=actual_cache_values,
                    )

                for req in requests:
                    req.add_new_token(seq_lens[req.py_request_id], 0)
                cache_manager.update_resources(scheduled_batch)
        finally:
            try:
                for req in requests:
                    cache_manager.free_resources(req)
            finally:
                cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    def test_pool_mapping_tensors(
        self, compress_ratios: List[int], dtype: DataType, compressor_dtype: DataType
    ):
        # Create cache manager and sparse attention config
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=4,
            max_seq_len=1024,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        try:
            expected_mapping = torch.stack(
                [
                    torch.arange(cache_manager.num_local_layers, dtype=torch.int32),
                    torch.zeros(cache_manager.num_local_layers, dtype=torch.int32),
                ],
                dim=1,
            )
            expected_pointers = torch.tensor(
                [
                    [
                        cache_manager.impl.get_mem_pool_base_address(
                            cache_manager._layer_attn_to_layer_id[
                                pp_layer, DeepseekV4AttentionType.SWA
                            ],
                            DeepseekV4AttentionType.SWA.role,
                            PageIndexMode.PER_LAYER,
                        ),
                        0,
                    ]
                    for pp_layer in cache_manager.pp_layers
                ],
                dtype=torch.int64,
            )

            assert cache_manager.kv_cache_pool_mapping.shape == expected_mapping.shape
            assert cache_manager.kv_cache_pool_mapping.dtype == expected_mapping.dtype
            assert torch.equal(cache_manager.kv_cache_pool_mapping, expected_mapping)
            assert cache_manager.kv_cache_pool_pointers.shape == expected_pointers.shape
            assert cache_manager.kv_cache_pool_pointers.dtype == expected_pointers.dtype
            assert torch.equal(cache_manager.kv_cache_pool_pointers, expected_pointers)
        finally:
            cache_manager.shutdown()

    def test_disagg_page_table_accepts_deepseek_v4_roles(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[1, 4, 128],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )

        try:
            page_table = build_page_table_from_manager(cache_manager)
            saw_pool_view = False
            for layer_group in page_table.layer_groups:
                for pool_view in layer_group.pool_views:
                    saw_pool_view = True
                    # All DSv4 pools publish per-layer buffer_entries (INDEXED);
                    # the FLAT layout is reserved for the DSA (DeepSeek v3.2)
                    # indexer K cache pool.
                    assert pool_view.mapper_kind == MapperKind.INDEXED
                    # pool_role carries the manager-native role-name strings,
                    # which are all DSv4 attention-type names like
                    # "deepseek_v4_swa", "deepseek_v4_compress", etc.
                    assert pool_view.pool_role
                    for role_name in pool_view.pool_role:
                        assert role_name.startswith("deepseek_v4_"), role_name
            assert saw_pool_view
        finally:
            cache_manager.shutdown()

    def test_disagg_pool_mapping_prefers_exact_pool_view_layer_set(self):
        ctx_cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[1, 4, 128],
            dtype=DataType.FP8,
            compressor_dtype=DataType.FLOAT,
            tp_size=2,
        )
        gen_cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=512,
            compress_ratios=[1, 4, 128],
            dtype=DataType.FP8,
            compressor_dtype=DataType.FLOAT,
            tp_size=2,
            enable_attention_dp=True,
        )

        try:
            ctx_rank_info = RankInfo.from_kv_cache_manager("ctx", ctx_cache_manager, 0)
            gen_rank_info = RankInfo.from_kv_cache_manager("gen", gen_cache_manager, 0)
            registrar = PeerRegistrar(gen_rank_info, KVRegionExtractorV1(gen_cache_manager))
            registrar.register("ctx", 0, ctx_rank_info)

            pool_mapping = registrar.get_pool_mapping(ctx_rank_info)
            assert pool_mapping
            for self_pool_key, peer_pool_key in pool_mapping.items():
                registrar.get_kv_map(ctx_rank_info, self_pool_key, peer_pool_key)
        finally:
            ctx_cache_manager.shutdown()
            gen_cache_manager.shutdown()

    def test_dsv4_block_tables(self, scratch_reuse_enabled: bool):
        prompt_len = self.tokens_per_block * 2
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=prompt_len,
            compress_ratios=[4, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        req = self._create_request(0, prompt_len)
        allocated = False
        try:
            assert cache_manager.prepare_context(req)
            assert cache_manager.resize_context(req, req.context_chunk_size)
            allocated = True

            attention_types = {
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
                DeepseekV4AttentionType.COMPRESSOR_KV,
            }
            sliding_block_tables_cuda = torch.empty(
                self._sliding_block_tables_shape(cache_manager, 1),
                dtype=torch.int32,
                device="cuda",
            )
            compress_block_table = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            host_indexer_compress_block_table = torch.empty(
                1,
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )
            with torch.cuda.stream(cache_manager._stream):
                cache_manager.compute_sliding_block_tables(
                    [req.py_request_id],
                    num_contexts=1,
                )
                cache_manager.copy_batch_sliding_block_tables(
                    sliding_block_tables_cuda,
                    [req.py_request_id],
                    num_contexts=1,
                    num_seqs=1,
                )
            cache_manager.copy_batch_compress_block_tables(
                compress_block_table,
                [req.py_request_id],
                compress_ratio=4,
                beam_width=1,
                num_contexts=1,
                num_seqs=1,
            )
            cache_manager.copy_batch_indexer_compress_block_tables(
                host_indexer_compress_block_table,
                [req.py_request_id],
                beam_width=1,
                num_contexts=1,
                num_seqs=1,
            )
            cache_manager._stream.synchronize()
            sliding_block_tables = sliding_block_tables_cuda.cpu()
            for attn_type in attention_types:
                layer0_buffer = _view_fp8_as_uint8(cache_manager.get_buffers(0, attn_type))
                layer1_buffer = _view_fp8_as_uint8(cache_manager.get_buffers(1, attn_type))
                attn_len = prompt_len
                if attn_type in [
                    DeepseekV4AttentionType.COMPRESS,
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                ]:
                    attn_len //= 4
                num_blocks = (attn_len + layer0_buffer.shape[1] - 1) // layer0_buffer.shape[1]
                if attn_type == DeepseekV4AttentionType.COMPRESS:
                    layer0_offsets = compress_block_table[0, :num_blocks].tolist()
                    layer1_offsets = compress_block_table[0, :num_blocks].tolist()
                elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
                    layer0_offsets = host_indexer_compress_block_table[0, :num_blocks].tolist()
                    layer1_offsets = host_indexer_compress_block_table[0, :num_blocks].tolist()
                else:
                    layer0_offsets = sliding_block_tables[
                        cache_manager.layer_offsets[0],
                        attn_type.value,
                        0,
                        :num_blocks,
                    ].tolist()
                    layer1_offsets = sliding_block_tables[
                        cache_manager.layer_offsets[1],
                        attn_type.value,
                        0,
                        :num_blocks,
                    ].tolist()
                assert all(offset != BAD_PAGE_INDEX for offset in layer0_offsets)
                assert all(offset != BAD_PAGE_INDEX for offset in layer1_offsets)

                layer0_values = torch.full(
                    (num_blocks, layer0_buffer.shape[1], layer0_buffer.shape[-1]),
                    1,
                    dtype=layer0_buffer.dtype,
                    device=layer0_buffer.device,
                )
                layer1_values = torch.full_like(layer0_values, 2)

                layer0_buffer[layer0_offsets] = layer0_values
                layer1_buffer[layer1_offsets] = layer1_values

                torch.testing.assert_close(layer0_buffer[layer0_offsets], layer0_values)
                torch.testing.assert_close(layer1_buffer[layer1_offsets], layer1_values)
        finally:
            if allocated:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_block_offsets_matches_python_converter(self, scratch_reuse_enabled: bool):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]
            actual = torch.empty(
                self._attention_op_block_offsets_shape(cache_manager, len(request_ids)),
                dtype=torch.int32,
                device="cuda",
            )
            sliding_block_tables = torch.empty(
                self._sliding_block_tables_shape(cache_manager, len(request_ids)),
                dtype=torch.int32,
                device="cuda",
            )

            with torch.cuda.stream(cache_manager._stream):
                cache_manager.compute_sliding_block_tables(
                    request_ids,
                    num_contexts=num_contexts,
                )
                cache_manager.copy_batch_block_offsets(
                    actual,
                    request_ids,
                    beam_width=1,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )
                cache_manager.copy_batch_sliding_block_tables(
                    sliding_block_tables,
                    request_ids,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )

            expected = torch.full(
                (
                    cache_manager.num_attention_op_pools,
                    len(request_ids),
                    cache_manager.max_blocks_per_seq,
                ),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                device="cpu",
            )
            for local_layer_idx, pp_layer in enumerate(cache_manager.pp_layers):
                ref = self._reference_copy_batch_page_indices(
                    cache_manager,
                    request_ids,
                    num_contexts,
                    pp_layer,
                    DeepseekV4AttentionType.SWA,
                    PageIndexMode.PER_LAYER,
                )
                expected[local_layer_idx, : len(request_ids)] = ref

            # DSV4 AttentionOp only consumes the key table.
            cache_manager._stream.synchronize()
            actual_cpu = actual.cpu()
            sliding_block_tables_cpu = sliding_block_tables.cpu()
            torch.testing.assert_close(
                actual_cpu[:, : len(request_ids), 0],
                expected[:, : len(request_ids)],
            )
            torch.testing.assert_close(
                actual_cpu[:, : len(request_ids), 0],
                sliding_block_tables_cpu[
                    :,
                    DeepseekV4AttentionType.SWA.value,
                    : len(request_ids),
                ],
            )
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_sliding_block_tables_matches_python_converter(
        self, scratch_reuse_enabled: bool
    ):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]
            actual = torch.empty(
                self._sliding_block_tables_shape(cache_manager, len(request_ids)),
                dtype=torch.int32,
                device="cuda",
            )

            with torch.cuda.stream(cache_manager._stream):
                cache_manager.compute_sliding_block_tables(
                    request_ids,
                    num_contexts=num_contexts,
                )
                cache_manager.copy_batch_sliding_block_tables(
                    actual,
                    request_ids,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )

            expected = torch.full(
                self._sliding_block_tables_shape(cache_manager, len(request_ids)),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                device="cpu",
            )
            for pp_layer in cache_manager.pp_layers:
                compress_ratio = cache_manager._compress_ratios[pp_layer]
                local_layer_idx = cache_manager.layer_offsets[pp_layer]
                for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
                    if not compress_ratio_has_attention(compress_ratio, attn_type):
                        continue
                    expected[local_layer_idx, attn_type.value, : len(request_ids)] = (
                        self._reference_copy_batch_page_indices(
                            cache_manager,
                            request_ids,
                            num_contexts,
                            pp_layer,
                            attn_type,
                            PageIndexMode.PER_LAYER,
                        )
                    )

            cache_manager._stream.synchronize()
            torch.testing.assert_close(
                actual.cpu()[:, :, : len(request_ids)],
                expected[:, :, : len(request_ids)],
            )
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_compress_block_tables_matches_python_converter(
        self, scratch_reuse_enabled: bool
    ):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]

            for compress_ratio in (4, 128):
                actual = torch.empty(
                    len(request_ids),
                    cache_manager.max_blocks_per_seq,
                    dtype=torch.int32,
                    device="cpu",
                )

                cache_manager.copy_batch_compress_block_tables(
                    actual,
                    request_ids,
                    compress_ratio=compress_ratio,
                    beam_width=1,
                    num_contexts=num_contexts,
                    num_seqs=len(request_ids),
                )

                pp_layer = next(
                    layer
                    for layer in cache_manager.pp_layers
                    if cache_manager._compress_ratios[layer] == compress_ratio
                )
                expected = self._reference_copy_batch_page_indices(
                    cache_manager,
                    request_ids,
                    num_contexts,
                    pp_layer,
                    DeepseekV4AttentionType.COMPRESS,
                    PageIndexMode.SHARED,
                )
                torch.testing.assert_close(actual, expected)
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_copy_batch_indexer_compress_block_tables_matches_python_converter(
        self, scratch_reuse_enabled: bool
    ):
        prompt_len = self.tokens_per_block * 2 + 1
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=3,
            max_seq_len=1024,
            compress_ratios=[1, 4, 128, 4],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=scratch_reuse_enabled,
        )
        assert cache_manager.enable_swa_scratch_reuse == scratch_reuse_enabled

        requests = []
        try:
            requests, num_contexts = self._prepare_mixed_copy_batch(cache_manager, prompt_len)
            request_ids = [req.py_request_id for req in requests]
            actual = torch.empty(
                len(request_ids),
                cache_manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cpu",
            )

            cache_manager.copy_batch_indexer_compress_block_tables(
                actual,
                request_ids,
                beam_width=1,
                num_contexts=num_contexts,
                num_seqs=len(request_ids),
            )

            pp_layer = next(
                layer
                for layer in cache_manager.pp_layers
                if compress_ratio_has_attention(
                    cache_manager._compress_ratios[layer],
                    DeepseekV4AttentionType.INDEXER_COMPRESS,
                )
            )
            expected = self._reference_copy_batch_page_indices(
                cache_manager,
                request_ids,
                num_contexts,
                pp_layer,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
                PageIndexMode.SHARED,
            )
            torch.testing.assert_close(actual, expected)
            assert num_contexts == 2
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_swa_scratch_reuse_enabled_by_default_for_main_manager(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
        )

        try:
            assert cache_manager.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse is not None
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse.max_rewind_len == 0
            assert cache_manager.num_attention_op_pools == cache_manager.num_local_layers
        finally:
            cache_manager.shutdown()

    def test_swa_scratch_reuse_disabled_by_config_for_main_manager(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=False,
        )

        try:
            assert not cache_manager.enable_swa_scratch_reuse
            assert not cache_manager.kv_cache_manager_py_config.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse is None
            assert cache_manager.num_attention_op_pools == cache_manager.num_local_layers
        finally:
            cache_manager.shutdown()

    def test_swa_scratch_reuse_uses_extra_kv_tokens_for_rewind(self):
        spec_config = SimpleNamespace(
            max_draft_len=7,
            max_total_draft_tokens=7,
            spec_dec_mode=SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL,
        )
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            spec_config=spec_config,
            enable_swa_scratch_reuse=True,
        )

        try:
            scratch_reuse = cache_manager.kv_cache_manager_py_config.swa_scratch_reuse
            assert scratch_reuse is not None
            assert scratch_reuse.max_rewind_len == spec_config.max_draft_len - 1
        finally:
            cache_manager.shutdown()

    def test_draft_cache_manager_disables_swa_scratch_reuse(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            is_draft=True,
            enable_swa_scratch_reuse=True,
        )

        try:
            assert not cache_manager.enable_swa_scratch_reuse
            assert not cache_manager.kv_cache_manager_py_config.enable_swa_scratch_reuse
            assert cache_manager.kv_cache_manager_py_config.swa_scratch_reuse is None
            assert cache_manager.num_attention_op_pools == cache_manager.num_local_layers
        finally:
            cache_manager.shutdown()

    def test_context_request_enable_scratch_reuse_until_generation(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )

        req = self._create_request(request_id=0, prompt_len=self.tokens_per_block + 1)
        allocated = False
        try:
            assert cache_manager.prepare_context(req)
            assert cache_manager.resize_context(req, req.context_chunk_size)
            allocated = True

            kv_cache = cache_manager.kv_cache_map[req.py_request_id]
            assert kv_cache.enable_swa_scratch_reuse

            scheduled_batch = ScheduledRequests()
            scheduled_batch.context_requests_last_chunk = [req]
            req.context_current_position = req.prompt_len
            req.add_new_token(req.prompt_len, 0)
            cache_manager.update_context_resources(scheduled_batch)
            assert not kv_cache.enable_swa_scratch_reuse

            assert cache_manager.try_allocate_generation(req)
            assert not kv_cache.enable_swa_scratch_reuse
        finally:
            if allocated:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_disagg_generation_init_disables_swa_scratch_reuse(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )

        req = self._create_request(request_id=0, prompt_len=self.tokens_per_block + 1)
        req.state = LlmRequestState.DISAGG_GENERATION_INIT
        allocated = False
        try:
            assert cache_manager.prepare_disagg_gen_init(req)
            allocated = True

            kv_cache = cache_manager.kv_cache_map[req.py_request_id]
            assert not kv_cache.enable_swa_scratch_reuse
            assert kv_cache.history_length == req.prompt_len
        finally:
            if allocated:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    def test_disagg_generation_init_rejects_context_apis(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=1,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )
        req = self._create_request(request_id=0, prompt_len=self.tokens_per_block + 1)
        req.state = LlmRequestState.DISAGG_GENERATION_INIT
        try:
            with pytest.raises(AssertionError, match="prepare_disagg_gen_init"):
                cache_manager.prepare_context(req)
            with pytest.raises(AssertionError, match="prepare_disagg_gen_init"):
                cache_manager.resize_context(req, req.context_remaining_length)
        finally:
            cache_manager.shutdown()

    def test_dummy_generation_requests_with_swa_scratch_reuse(self):
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=2,
            max_seq_len=1024,
            compress_ratios=[1],
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            enable_swa_scratch_reuse=True,
        )

        requests = []
        token_nums = [1, self.tokens_per_block * 4 + 1]
        try:
            requests = cache_manager.add_dummy_requests(
                request_ids=[0, 1],
                token_nums=token_nums,
                is_gen=True,
            )
            assert requests is not None
            assert len(requests) == len(token_nums)

            short_kv_cache = cache_manager.kv_cache_map[requests[0].py_request_id]
            long_kv_cache = cache_manager.kv_cache_map[requests[1].py_request_id]
            assert not short_kv_cache.enable_swa_scratch_reuse
            assert not long_kv_cache.enable_swa_scratch_reuse
            assert short_kv_cache.history_length == 0
            assert short_kv_cache.capacity == token_nums[0] + 1
            assert long_kv_cache.history_length == token_nums[1] - 1
            assert long_kv_cache.capacity == token_nums[1] + 1
        finally:
            for req in requests:
                cache_manager.free_resources(req)
            cache_manager.shutdown()

    @pytest.mark.parametrize("compress_ratios", [[1, 4, 128]])
    @pytest.mark.parametrize(
        "dtype,compressor_dtype", [(DataType.BF16, DataType.FLOAT), (DataType.FP8, DataType.FLOAT)]
    )
    @pytest.mark.parametrize("invalid", [False, True])
    @pytest.mark.parametrize("fill_with_zero", [False, True])
    def test_check_invalid_values_in_kv_cache(
        self,
        compress_ratios: List[int],
        dtype: DataType,
        compressor_dtype: DataType,
        invalid: bool,
        fill_with_zero: bool,
    ):
        """Test invalid value detection and optional zero-fill behavior in KV cache."""
        cache_manager, _ = self._create_deepseek_v4_cache_manager(
            tokens_per_block=self.tokens_per_block,
            max_batch_size=4,
            max_seq_len=1024,
            compress_ratios=compress_ratios,
            dtype=dtype,
            compressor_dtype=compressor_dtype,
        )

        needs_invalid_cleanup = False
        try:
            # Fresh cache (zero-initialized) should have no invalid values
            result = cache_manager.check_invalid_values_in_kv_cache()
            assert not result, "Fresh cache should have no invalid values"

            if invalid:
                # Inject invalid into a float buffer so NaN/Inf checks are supported.
                layer_idx = next(i for i, ratio in enumerate(compress_ratios) if ratio > 1)
                buffer = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESSOR_KV)
                buffer[0, 0, 0] = torch.nan
                needs_invalid_cleanup = True

            result = cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=fill_with_zero)
            if invalid and fill_with_zero:
                needs_invalid_cleanup = False
            assert result == invalid, (
                f"Expected invalid={invalid} from check_invalid_values_in_kv_cache, got {result}"
            )

            # Verify whether invalid values remain after the check.
            post_check = cache_manager.check_invalid_values_in_kv_cache()
            expected_post_check = invalid and not fill_with_zero
            assert post_check == expected_post_check, (
                f"Expected post-check invalid={expected_post_check}, got {post_check}"
            )

            if expected_post_check:
                # Cleanup for shutdown path when zero-fill wasn't requested above.
                cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
                needs_invalid_cleanup = False
        finally:
            try:
                if needs_invalid_cleanup:
                    cache_manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
            finally:
                cache_manager.shutdown()


if __name__ == "__main__":
    tester = TestDeepseekV4CacheManager()
    print("=== FP8, prompt_lens=[1024, 2048, 4096], steps=100 ===")
    tester.test_write_read_cache([1, 4, 128], [1024, 2048, 4096], 100, DataType.FP8, DataType.FLOAT)
    print("Test passed")
