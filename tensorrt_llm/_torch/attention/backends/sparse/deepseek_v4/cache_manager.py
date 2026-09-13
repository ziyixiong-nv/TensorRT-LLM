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

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.pyexecutor import llm_request
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import GPU_LEVEL, KVCacheManagerV2
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    get_sm_version,
    nvtx_range_debug,
    prefer_pinned,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import ModelConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    LayerId,
    PageIndexMode,
    ScratchDesc,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

from .compressor import KVCacheDtype
from .params import (
    DEEPSEEK_V4_NON_SLIDING_ATTENTION,
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DEEPSEEK_V4_SPARSE_RATIO,
    DEEPSEEK_V4_VARIANT,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
    has_compressor_state,
    is_compress_layer,
    is_overlap_compressor,
    is_sparse_layer,
    is_v41,
    owns_compressed_kv,
    pool_factor,
    source_layer_for,
)


def get_attn_dim(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> int:
    state_factor = 2 if is_overlap_compressor(compress_ratio, variant) else 1
    if attn_type == DeepseekV4AttentionType.SWA:
        return head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESS:
        return head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_KV:
        return state_factor * head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return state_factor * head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return index_head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV:
        return state_factor * index_head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return state_factor * index_head_dim
    raise ValueError(f"Unsupported DeepSeek-V4 attention type: {attn_type}")


def get_token_bytes(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    has_fp8_kv_cache: bool,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> int:
    if not compress_ratio_has_attention(compress_ratio, attn_type, variant):
        raise ValueError(
            f"Layer with compress ratio {compress_ratio} does not have attention type {attn_type}"
        )

    if use_fp8_ds_mla and attn_type in (
        DeepseekV4AttentionType.SWA,
        DeepseekV4AttentionType.COMPRESS,
    ):
        from . import footer_scale_kv

        if head_dim != footer_scale_kv.DIM_NOPE + footer_scale_kv.DIM_ROPE:
            raise ValueError(
                f"footer-scale KV layout requires head_dim "
                f"{footer_scale_kv.DIM_NOPE + footer_scale_kv.DIM_ROPE}, got {head_dim}"
            )
        return footer_scale_kv.TOKEN_BYTES

    attn_dim = get_attn_dim(head_dim, index_head_dim, compress_ratio, attn_type, variant)

    dtype_bytes = 1 if has_fp8_kv_cache else 2
    # (indexer) compressor kv and score always use float32
    if attn_type in [
        DeepseekV4AttentionType.COMPRESSOR_KV,
        DeepseekV4AttentionType.COMPRESSOR_SCORE,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
    ]:
        dtype_bytes = 4  # (indexer) compressor kv and score use float32
    # Indexer cache always packs data + per-block scales into one row.  Only
    # the two indexer presets ("fp8" blockwise / "fp4" mxfp4) are valid
    # here — bf16 / fp8_pertensor are reserved for the main-attention
    # compressor.
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        if indexer_k_dtype == "fp8":
            return attn_dim + index_head_dim // 128 * 4
        if indexer_k_dtype == "fp4":
            return index_head_dim // 2 + index_head_dim // 32
        raise ValueError(
            f"Unsupported indexer_k_dtype {indexer_k_dtype!r}; expected 'fp8' or 'fp4'."
        )

    return attn_dim * dtype_bytes


def _estimate_non_sliding_attn_size_per_token(
    head_dim: int,
    index_head_dim: int,
    compress_ratios: List[int],
    has_fp8_kv_cache,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> int:
    total_bytes = 0
    for compress_ratio in compress_ratios:
        for attn_type in DEEPSEEK_V4_NON_SLIDING_ATTENTION:
            if compress_ratio_has_attention(compress_ratio, attn_type, variant):
                total_bytes += _get_attn_bytes_per_token(
                    head_dim,
                    index_head_dim,
                    compress_ratio,
                    attn_type,
                    has_fp8_kv_cache,
                    indexer_k_dtype=indexer_k_dtype,
                    use_fp8_ds_mla=use_fp8_ds_mla,
                    variant=variant,
                )
    return total_bytes


def _estimate_swa_cache_size(
    head_dim: int,
    index_head_dim: int,
    compress_ratios: List[int],
    has_fp8_kv_cache,
    tokens_per_block: int,
    swa_window_size: int | None,
    *,
    context: bool,
    scratch: bool,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> Tuple[int, int]:
    tokens_per_block = int(tokens_per_block)
    size_per_token = 0
    size_per_request = 0
    scratch_keys = set()
    for compress_ratio in compress_ratios:
        for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
            if not compress_ratio_has_attention(compress_ratio, attn_type, variant):
                continue
            if attn_type == DeepseekV4AttentionType.SWA:
                if swa_window_size is None:
                    continue
                window_size = swa_window_size
            else:
                state_factor = 2 if is_overlap_compressor(compress_ratio, variant) else 1
                window_size = state_factor * compress_ratio
            if window_size <= 0:
                continue
            window_tokens = (
                (int(window_size) + tokens_per_block - 1) // tokens_per_block
            ) * tokens_per_block
            token_bytes = _get_attn_bytes_per_token(
                head_dim,
                index_head_dim,
                compress_ratio,
                attn_type,
                has_fp8_kv_cache,
                indexer_k_dtype=indexer_k_dtype,
                use_fp8_ds_mla=use_fp8_ds_mla,
                variant=variant,
            )
            if not context:
                size_per_request += window_tokens * token_bytes
            elif not scratch:
                size_per_token += token_bytes
            else:
                scratch_key = (attn_type, compress_ratio)
                if scratch_key in scratch_keys:
                    size_per_request += window_tokens * token_bytes
                else:
                    scratch_keys.add(scratch_key)
                    size_per_token += token_bytes
    return size_per_token, size_per_request


def _get_attn_bytes_per_token(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    has_fp8_kv_cache: bool,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> int:
    token_bytes = get_token_bytes(
        head_dim,
        index_head_dim,
        compress_ratio,
        attn_type,
        has_fp8_kv_cache,
        indexer_k_dtype=indexer_k_dtype,
        use_fp8_ds_mla=use_fp8_ds_mla,
        variant=variant,
    )
    if attn_type in [DeepseekV4AttentionType.COMPRESS, DeepseekV4AttentionType.INDEXER_COMPRESS]:
        token_bytes //= pool_factor(compress_ratio)
    return token_bytes


def _get_index_mode(attn_type: DeepseekV4AttentionType) -> PageIndexMode:
    if attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
        return PageIndexMode.PER_LAYER
    else:
        return PageIndexMode.SHARED


class DeepseekV4CacheManager(KVCacheManagerV2):
    _supports_reuse_match_backoff = False

    # Partial-block reuse is unsound for this manager, and it was measured to be, not
    # reasoned to be. One fixed prompt, greedy, four arms one process each: block
    # reuse off, block reuse on, and a cold default process all emit byte-identical
    # ids, while the 2nd call in a reuse-on process diverges at decode step 2 and
    # produces a different -- still fluent, still on-topic -- paragraph. Turning only
    # `enable_partial_reuse` off restores exact agreement, so it is the whole factor;
    # neither general block reuse nor request-to-request state leakage survives that
    # control. (`_supports_reuse_match_backoff` above is unrelated: it governs the
    # draft/spec-decode backoff window, and was checked and ruled out first.)
    #
    # The structural reason is visible in `compressed_block_sizes` a few lines below:
    # for the COMPRESS / COMPRESSOR / INDEXER roles the block size is *not*
    # `tokens_per_block`, while `AttentionLayerConfig` carries no per-layer token
    # granularity for the reuse radix tree to match against. So the tree matches at one
    # global granularity over pools that are not addressed at it, and a prefix stopping
    # mid-block hands the new sequence rows covering tokens it does not own. The rows
    # are well-formed, so nothing raises.
    #
    # How far the measurement pins that down: a second arm whose warm-up generated one
    # token instead of 64 -- so the same 18-token prompt but a much shorter populated
    # region -- diverges to the *same* ids as the 64-token warm-up. The corruption is
    # therefore a function of the matched prompt prefix alone, not of how much the
    # earlier request went on to write past it. That is consistent with the above but
    # narrower than it: which specific rows are wrong is not established here, only
    # that partial matching over these pools produces them. Do not read the paragraph
    # above as a verified row-level account.
    #
    # Full-block reuse stays enabled: a whole-block prefix lands on a row boundary
    # for every ratio in `compressed_block_sizes`, which is what makes it sound.
    _supports_partial_reuse = False

    # Which generation's `compress_ratios` encoding this manager was built for.
    # A class attribute rather than an instance-only one so the sizing helpers stay
    # reachable on the bare `object.__new__` instances the sizing tests construct,
    # and so V4 remains the default for any caller that predates the distinction.
    _variant: str = DEEPSEEK_V4_VARIANT

    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_pointers contains one virtual attention-op pool per local
    # SWA layer, shape: [num_local_layers, 2]. The second column is always 0.
    kv_cache_pool_pointers: torch.Tensor
    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_mapping contains pool id and layer offset for each layer's swa attention,
    # shape: [num_local_layers, 2]
    kv_cache_pool_mapping: torch.Tensor
    # The block size of the (indexer) compressed cache.
    # For other attention types, block size is tokens_per_block.
    compressed_block_sizes: List[int]

    def _get_typical_seq_len(self, kv_cache_config: KvCacheConfig) -> int:
        """Retain DeepSeek-V4's max-length pool-sizing model by default."""
        return (
            kv_cache_config.avg_seq_len
            if kv_cache_config.avg_seq_len is not None
            else self.max_seq_len
        )

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: int = 1,
        max_batch_size: int,
        max_beam_width: int = 1,
        tokens_per_block: int,
        max_seq_len: int,
        vocab_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.BF16,
        compressor_dtype: DataType = DataType.FLOAT,
        sparse_attn_config: Optional[DeepSeekV4SparseAttentionConfig] = None,
        max_input_len: Optional[int] = None,
        max_num_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        if sparse_attn_config is None:
            sparse_attn_config = kwargs.pop("sparse_attention_config", None)
        if sparse_attn_config is None and kwargs.get("model_config") is not None:
            sparse_attn_config = kwargs["model_config"].sparse_attention_config
        if sparse_attn_config is None:
            raise ValueError(
                "sparse_attn_config or sparse_attention_config is required "
                "for DeepseekV4CacheManager"
            )

        # DeepSeek-V4 specific attributes initialization
        assert kv_cache_type == CacheTypeCpp.SELFKONLY, "DeepSeek-V4 only supports SELFKONLY"
        assert num_kv_heads == 1, "DeepSeek-V4 only supports num_kv_heads == 1"
        assert len(sparse_attn_config.compress_ratios) >= num_layers, (
            "The length of compress ratios must be >= the number of layers"
        )
        assert dtype in [DataType.BF16, DataType.FP8], (
            f"Unsupported dtype: {dtype}, only support BF16 and FP8"
        )
        assert compressor_dtype == DataType.FLOAT, (
            f"Unsupported compressor dtype: {compressor_dtype}, only support FP32/TF32"
        )

        assert tokens_per_block in [128, 256], (
            f"DeepseekV4CacheManager requires tokens_per_block in [128, 256], got {tokens_per_block}. "
            f"Set kv_cache_config.tokens_per_block to 128 or 256."
        )

        self.use_fp8_ds_mla = kv_cache_config.dtype == "fp8_ds_mla"
        sm_version = get_sm_version()
        if sm_version == 90 and not self.use_fp8_ds_mla:
            raise ValueError(
                "DeepSeek-V4 on Hopper requires kv_cache_config.dtype='fp8_ds_mla'; "
                f"got {kv_cache_config.dtype!r}."
            )
        if self.use_fp8_ds_mla and sm_version != 90 and tokens_per_block != 256:
            raise ValueError(
                "DeepSeek-V4 fp8_ds_mla KV cache requires tokens_per_block=256 "
                f"on SM{sm_version}, got {tokens_per_block}."
            )

        self.index_head_dim = sparse_attn_config.index_head_dim
        # Which generation's `compress_ratios` encoding this config uses. See
        # `params.py` — the same integer means different things in V4 and V4.1,
        # so every ratio predicate needs it.
        self._variant = getattr(sparse_attn_config, "variant", DEEPSEEK_V4_VARIANT)
        # Which layers actually write a compressed cache. Absent (V4) means every
        # long-range layer does, so `owns_compressed_kv` degrades to
        # `is_compress_layer` and nothing here changes shape.
        self._kv_source_layer_ids = getattr(sparse_attn_config, "kv_source_layer_ids", None)
        self._compress_ratios = sparse_attn_config.compress_ratios
        # When MTP is enabled, enlarge the sliding window sizes by
        # max_draft_len so that rewinding rejected draft tokens can still
        # reach the KV entries that would otherwise have been evicted by the
        # sliding-window policy.
        spec_config = kwargs.get("spec_config", None)
        self._max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self._swa_window_size = sparse_attn_config.window_size
        self._compressor_dtype = compressor_dtype
        # If MTP is enabled, append compress ratios for MTP virtual layers.
        # MTP adds (max_draft_len - 1) extra layers that mirror the last real
        # layer's attention pattern.  Only NEW entries are appended; existing
        # per-layer ratios are never modified, so a real layer with ratio==1
        # stays SWA-only.
        if self._max_draft_len > 0:
            self._compress_ratios = self._compress_ratios + [self._compress_ratios[-1]] * (
                self._max_draft_len - 1
            )
        self.compressed_block_sizes = [
            tokens_per_block // pool_factor(ratio) for ratio in self._compress_ratios
        ]

        self._init_indexer_dtype(sparse_attn_config)

        self._max_input_len = max_input_len
        self._max_num_tokens = max_num_tokens

        # General initialization
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            mapping=mapping,
            dtype=dtype,
            max_num_tokens=max_num_tokens,
            **kwargs,
        )
        self.is_vswa = True  # DeepSeek-V4 must has VSWA

        # DeepSeek-V4 expects cache of all layers with the same attention type and compress ratio
        # to be in the same pool and have the same scale.
        self._assert_layer_pool_scale()

        # For DeepSeek-V4 Attention, the base pointer for SWA pool
        # Use first PP layer instead of hardcoded 0 for pipeline parallelism.
        first_pp_layer = self.pp_layers[0]
        self.swa_pool_ptr = self.impl.get_mem_pool_base_address(
            self._layer_attn_to_layer_id[first_pp_layer, DeepseekV4AttentionType.SWA],
            DeepseekV4AttentionType.SWA.role,
            PageIndexMode.PER_LAYER,
        )

        # One COMPRESS pool per distinct compress ratio that actually has a
        # compressed branch. All layers sharing a ratio share a pool (asserted by
        # `_assert_layer_pool_scale`), so the first PP-local layer with each ratio
        # is representative. Driven off `compress_ratio_has_attention` rather than
        # a hard-coded {4, 128}: V4.1's ratios are {0, 1, 2}, and its ratio-1
        # layers own a full-length compressed cache that the V4 encoding has no
        # equivalent for.
        self.compress_pool_ptrs = {}
        pp_compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        for compress_ratio in sorted(set(pp_compress_ratios)):
            if not compress_ratio_has_attention(
                compress_ratio, DeepseekV4AttentionType.COMPRESS, self._variant
            ):
                continue
            first_layer = self.pp_layers[pp_compress_ratios.index(compress_ratio)]
            self.compress_pool_ptrs[compress_ratio] = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[first_layer, DeepseekV4AttentionType.COMPRESS],
                DeepseekV4AttentionType.COMPRESS.role,
                PageIndexMode.SHARED,
            )

    def _format_kv_cache_pool_lifecycle_entry(self, layer_id: LayerId, role: DataRole) -> str:
        layer_semantics = self._manager_layer_id_to_layer_attn.get((layer_id, role))
        if layer_semantics is None:
            return super()._format_kv_cache_pool_lifecycle_entry(layer_id, role)

        model_layer_idx, attn_type = layer_semantics
        return (
            f"deepseek_role={attn_type.name}, "
            f"compress_ratio={self._compress_ratios[model_layer_idx]}, "
            f"{super()._format_kv_cache_pool_lifecycle_entry(layer_id, role)}"
        )

    def get_buffers(self, layer_idx: int, attn_type: DeepseekV4AttentionType) -> torch.Tensor:
        """
        Get the buffers for a specific layer and attention type.

        Args:
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The buffer tensor (shape: [num_blocks, tokens_per_block, attn_dim])
            For blockwise FP8 layers, shape is [num_blocks, tokens_per_block, attn_dim + scale_size]
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        data_role = attn_type.role
        page_index_mode = _get_index_mode(attn_type)
        addr = self.impl.get_mem_pool_base_address(layer_id, data_role, page_index_mode)

        block_size = self.tokens_per_block
        if attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            block_size = self.compressed_block_sizes[layer_idx]

        attn_dim = get_attn_dim(
            self.head_dim,
            self.index_head_dim,
            self._compress_ratios[layer_idx],
            attn_type,
            self._variant,
        )
        footer_scale = self.use_fp8_ds_mla and attn_type in (
            DeepseekV4AttentionType.SWA,
            DeepseekV4AttentionType.COMPRESS,
        )
        if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            # Indexer always pack data + per-block scales into the same row.
            dim_per_token = self._indexer_data_size + self._indexer_scale_size
        elif footer_scale:
            # Footer-scale pages are byte-packed rather than token-major.
            from . import footer_scale_kv

            dim_per_token = footer_scale_kv.TOKEN_BYTES
        else:
            dim_per_token = attn_dim

        page_index_upper_bound = self.impl.get_page_index_upper_bound(layer_id, data_role)
        if page_index_mode == PageIndexMode.PER_LAYER:
            converter = self.impl.get_page_index_converter(layer_id, data_role)
            if converter.layer_offset is not None:
                page_index_upper_bound += converter.layer_offset * converter.expansion

        shape = (page_index_upper_bound, block_size, dim_per_token)

        dtype = self.dtype
        # (indexer) compressor kv and score use compressor_dtype
        if attn_type in [
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            dtype = self._compressor_dtype
        elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            dtype = self._indexer_dtype
        elif footer_scale:
            dtype = DataType.UINT8

        return convert_to_torch_tensor(TensorWrapper(addr, dtype, shape))

    def _get_window_size(
        self, compress_ratio: int, attn_type: DeepseekV4AttentionType
    ) -> int | None:
        if attn_type == DeepseekV4AttentionType.SWA:
            base_window_size = self._swa_window_size
        elif attn_type in (
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ):
            state_factor = 2 if is_overlap_compressor(compress_ratio, self._variant) else 1
            base_window_size = state_factor * compress_ratio
        else:
            return None
        return base_window_size + self._max_draft_len

    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        # Tensors for compatibility with AttentionOp, only contains swa attention.
        # SWA uses per-layer page indices, so each SWA layer has a virtual
        # attention-op pool.
        # shape: [num_local_layers, 2]
        self.num_attention_op_pools = self.num_local_layers
        self.kv_cache_pool_pointers = torch.tensor(
            [
                [
                    self.impl.get_mem_pool_base_address(
                        self._layer_attn_to_layer_id[pp_layer, DeepseekV4AttentionType.SWA],
                        DeepseekV4AttentionType.SWA.role,
                        PageIndexMode.PER_LAYER,
                    ),
                    0,
                ]
                for pp_layer in self.pp_layers
            ],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # shape: [num_local_layers, 2]
        self.kv_cache_pool_mapping = torch.tensor(
            [[local_layer_idx, 0] for local_layer_idx in range(self.num_local_layers)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.host_kv_cache_block_offsets = torch.empty(
            self.num_pools,
            index_mapper_capacity * self.max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        staging_capacity = self.max_batch_size * self.max_beam_width
        self._host_compress_block_tables_staging = {
            compress_ratio: torch.full(
                (staging_capacity, self.max_blocks_per_seq),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            for compress_ratio in set(self._compress_ratios)
            if compress_ratio_has_attention(
                compress_ratio, DeepseekV4AttentionType.COMPRESS, self._variant
            )
        }

        # layer offsets per layer and attn, shape [num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)].
        self._layer_offsets = torch.full(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        # Pool ids per layer and sliding attention type, shape [num_local_layers, num_sliding_attention_types].
        self._layer_attn_pool_ids = torch.full(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        # Scales per layer and sliding attention type, shape [num_local_layers, num_sliding_attention_types].
        self._layer_attn_scales = torch.ones(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            dtype=torch.int32,
            device="cpu",
        )
        # Scratch pages per block per layer and sliding attention type, shape
        # [num_local_layers, num_sliding_attention_types].
        self._scratch_pages = torch.zeros(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            dtype=torch.int32,
            device="cpu",
        )
        # (pool_id, scale) of the shared COMPRESS / INDEXER_COMPRESS pool, keyed by
        # compress ratio. V4 fills ratio 4 (both roles) and ratio 128 (COMPRESS
        # only, since ratio-128 layers are not indexed); V4.1 fills ratios 1 and 2
        # with both. All layers sharing a ratio share a pool, so the last write per
        # ratio is the same value as the first — `_assert_layer_pool_scale` is what
        # guarantees that.
        self._compress_pool_meta: Dict[int, Tuple[int, int]] = {}
        self._indexer_compress_pool_meta: Dict[int, Tuple[int, int]] = {}

        for layer_idx in self.pp_layers:
            compress_ratio = self._compress_ratios[layer_idx]
            if compress_ratio_has_attention(
                compress_ratio, DeepseekV4AttentionType.COMPRESS, self._variant
            ):
                compress_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.COMPRESS
                ]
                compress_converter = self.impl.get_page_index_converter(
                    compress_layer_id, DeepseekV4AttentionType.COMPRESS.role
                )
                self._compress_pool_meta[compress_ratio] = (
                    self.layer_to_pool_mapping_dict[compress_layer_id],
                    int(compress_converter.scale),
                )
            if compress_ratio_has_attention(
                compress_ratio, DeepseekV4AttentionType.INDEXER_COMPRESS, self._variant
            ):
                indexer_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS
                ]
                indexer_converter = self.impl.get_page_index_converter(
                    indexer_layer_id, DeepseekV4AttentionType.INDEXER_COMPRESS.role
                )
                self._indexer_compress_pool_meta[compress_ratio] = (
                    self.layer_to_pool_mapping_dict[indexer_layer_id],
                    int(indexer_converter.scale),
                )

            local_layer_idx = self.layer_offsets[layer_idx]
            for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
                # Presence in the map is the ground truth for "this layer has this
                # role", and strictly narrower than the ratio predicate: under
                # V4.1's cross-layer sharing a dependent long-range layer has the
                # ratio of a compressor but runs none, so it has no COMPRESSOR_*
                # buffers to describe. Asking the map instead of re-deriving from
                # the ratio keeps this loop correct for free as ownership evolves.
                layer_id = self._layer_attn_to_layer_id.get((layer_idx, attn_type))
                if layer_id is None:
                    continue
                pool_id = self.layer_to_pool_mapping_dict[layer_id]
                converter = self.impl.get_page_index_converter(layer_id, attn_type.role)
                self._layer_attn_pool_ids[local_layer_idx, attn_type.value] = pool_id
                self._layer_attn_scales[local_layer_idx, attn_type.value] = converter.scale
                self._layer_offsets[local_layer_idx, attn_type.value] = converter.layer_offset
                self._scratch_pages[local_layer_idx, attn_type.value] = (
                    converter.scratch_pages_per_block
                )

        device = torch.device("cuda", torch.cuda.current_device())
        self._device_kv_cache_block_offsets_input = torch.empty_like(
            self.host_kv_cache_block_offsets,
            device=device,
        )
        self._precomputed_sliding_block_tables = torch.empty(
            (
                self.num_local_layers,
                len(DEEPSEEK_V4_SLIDING_ATTENTION),
                self.host_kv_cache_block_offsets.size(1),
                self.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._device_copy_idx_staging = torch.zeros(
            self.host_kv_cache_block_offsets.size(1),
            dtype=torch.int32,
            device=device,
        )
        self._device_num_contexts = torch.empty((), dtype=torch.int32, device=device)
        self._device_layer_offsets = self._layer_offsets.to(device=device)
        self._device_layer_attn_pool_ids = self._layer_attn_pool_ids.to(
            device=device,
            dtype=torch.long,
        )
        self._device_layer_attn_scales = self._layer_attn_scales.to(device=device)
        self._device_scratch_pages = self._scratch_pages.to(device=device)
        self._device_valid_sliding_pool = self._device_layer_attn_pool_ids >= 0
        self._device_block_positions = torch.arange(
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )

        if self.enable_swa_scratch_reuse:
            valid_scales = self._layer_attn_scales[self._layer_attn_pool_ids >= 0]
            min_scale = int(valid_scales.min().item()) if valid_scales.numel() > 0 else 1
            max_scratch_pages = int(self._scratch_pages.max().item())
            self._max_scratch_slots = max(
                1,
                (self.max_blocks_per_seq * max_scratch_pages + min_scale - 1) // min_scale,
            )
            scratch_slots_shape = (
                self.num_pools,
                staging_capacity,
                self._max_scratch_slots,
            )
            self._host_scratch_begs_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._host_scratch_ends_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._host_scratch_slots_staging = torch.empty(
                scratch_slots_shape,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._device_scratch_begs_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                device=device,
            )
            self._device_scratch_ends_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                device=device,
            )
            self._device_scratch_slots_staging = torch.empty(
                scratch_slots_shape,
                dtype=torch.int32,
                device=device,
            )

    @property
    def blocks_in_primary_pool(self) -> int:
        first_pp_layer = self.pp_layers[0]
        swa_layer_id = self._layer_attn_to_layer_id[first_pp_layer, DeepseekV4AttentionType.SWA]
        return self.impl.get_page_index_upper_bound(swa_layer_id, DeepseekV4AttentionType.SWA.role)

    def get_num_free_blocks(self) -> int:
        # This method reports primary-pool capacity while the manager is empty.
        # DSV4 does not allocate the generic Role.KEY buffer, so use SWA's
        # model-specific DataRole for warmup capacity estimation.
        assert len(self.kv_cache_map) == 0, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
        max_num_pages = max(
            self.impl.get_page_index_upper_bound(
                self._layer_attn_to_layer_id[layer_idx, DeepseekV4AttentionType.SWA],
                DeepseekV4AttentionType.SWA.role,
            )
            for layer_idx in self.pp_layers
        )
        return max_num_pages

    def get_cache_indices(
        self,
        request_id: int,
        layer_idx: int,
        attn_type: DeepseekV4AttentionType,
    ) -> List[int]:
        """
        Get the cache block indices for a batch of requests at a specific layer and attention type.

        Args:
            request_id: The request id
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The cache block indices, shape (max_blocks_per_seq,)
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        data_role = attn_type.role
        pool_id = self.layer_to_pool_mapping_dict[layer_id]
        kv_cache = self.kv_cache_map[request_id]
        base_indices = kv_cache.get_base_page_indices(pool_id).tolist()
        converter = self.impl.get_page_index_converter(layer_id, data_role)
        page_index_mode = _get_index_mode(attn_type)
        return converter(
            base_indices,
            page_index_mode,
            kv_cache.get_scratch_desc(pool_id),
        )

    def _get_extra_quota_padding(self) -> int:
        """Ensure each attention type has minimal space when max_tokens is small."""
        return len(DeepseekV4AttentionType) * (2 << 20)

    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        has_fp8_kv_cache = self.dtype == DataType.FP8
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        (
            context_swa_size_per_token,
            _,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        max_context_tokens = (
            self._max_num_tokens if self._max_num_tokens is not None else max_tokens
        )
        context_tokens = min(max_tokens, max_context_tokens)
        generation_tokens = max_tokens - context_tokens
        generation_quota = (
            max_tokens * non_sliding_attn_size_per_token
            + generation_tokens * generation_swa_size_per_token
            + self.max_batch_size * generation_swa_size_per_request
        )
        context_extra_quota = context_tokens * context_swa_size_per_token
        padding = self._get_extra_quota_padding()
        return int(generation_quota + context_extra_quota + padding)

    def _get_max_tokens_from_quota(self, quota: int) -> float:
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        has_fp8_kv_cache = self.dtype == DataType.FP8
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        context_swa_size_per_token, _ = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        padding = self._get_extra_quota_padding()
        size_per_batch = self.max_batch_size * generation_swa_size_per_request + padding
        if quota < size_per_batch:
            return 0
        context_size_per_token = non_sliding_attn_size_per_token + context_swa_size_per_token
        if self._max_num_tokens is None:
            return (quota - size_per_batch) / context_size_per_token

        context_limit_quota = self._max_num_tokens * context_size_per_token + size_per_batch
        if quota <= context_limit_quota:
            return (quota - size_per_batch) / context_size_per_token

        generation_size_per_token = non_sliding_attn_size_per_token + generation_swa_size_per_token
        if generation_size_per_token <= 0:
            return float("inf")
        return self._max_num_tokens + (quota - context_limit_quota) / generation_size_per_token

    def _build_cache_config(self, config: KVCacheManagerConfigPy) -> KVCacheManagerConfigPy:
        """
        Add DeepSeek-V4 layers to the cache config.
        """
        layers: List[AttentionLayerConfig] = []
        layer_attn_to_layer_id: Dict[Tuple[int, DeepseekV4AttentionType], LayerId] = {}
        manager_layer_id_to_layer_attn: Dict[
            Tuple[LayerId, DataRole], Tuple[int, DeepseekV4AttentionType]
        ] = {}

        def _add_layer(
            layer_idx: int,
            attention_types: List[DeepseekV4AttentionType],
            sliding_window_size: int | None,
        ) -> None:
            layer_id = LayerId(len(layers))
            for attn_type in attention_types:
                layer_attn_to_layer_id[layer_idx, attn_type] = layer_id
                manager_layer_id_to_layer_attn[layer_id, attn_type.role] = (
                    layer_idx,
                    attn_type,
                )
            layer_config = AttentionLayerConfig(
                layer_id=layer_id,
                buffers=[
                    BufferConfig(
                        role=attn_type.role,
                        size=self._get_attn_bytes_per_block(attn_type, layer_idx),
                    )
                    for attn_type in attention_types
                ],
                sliding_window_size=sliding_window_size,
                num_sink_tokens=None,
            )
            layers.append(layer_config)

        def _alias_layer(
            layer_idx: int,
            attention_types: List[DeepseekV4AttentionType],
            source_layer_idx: int,
        ) -> None:
            """Point ``layer_idx`` at the buffers ``source_layer_idx`` allocated.

            No ``AttentionLayerConfig`` is appended, so nothing extra is reserved:
            the dependent layer resolves to the source's ``LayerId`` and therefore
            to the source's pages. Everything downstream goes through
            ``_layer_attn_to_layer_id`` -- ``get_buffers``, the pool-meta lookup,
            the block tables -- so a dependent needs no special case anywhere else.

            The reverse map is deliberately *not* extended: it names which layer a
            pool belongs to for the lifecycle log, and the answer is the owner.
            """
            for attn_type in attention_types:
                layer_attn_to_layer_id[layer_idx, attn_type] = layer_attn_to_layer_id[
                    source_layer_idx, attn_type
                ]

        if is_v41(self._variant):
            self._add_v41_cache_layers(_add_layer, _alias_layer)
        else:
            self._add_v4_cache_layers(_add_layer)

        # the mapping from layer index and attention type to layer id
        self._layer_attn_to_layer_id = layer_attn_to_layer_id
        self._manager_layer_id_to_layer_attn = manager_layer_id_to_layer_attn
        # number of layers in the KVCacheManagerPy
        self._num_manager_layers = len(layers)

        return replace(
            config,
            layers=layers,
        )

    def _add_v4_cache_layers(self, add_layer) -> None:
        """Assign cache roles for the V4 ``{1, 4, 128}`` compress-ratio encoding."""
        for layer in self.pp_layers:
            compress_ratio = self._compress_ratios[layer]
            if compress_ratio == 1:
                add_layer(
                    layer,
                    [DeepseekV4AttentionType.SWA],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
            elif compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
                add_layer(
                    layer,
                    [DeepseekV4AttentionType.SWA],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
                add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.COMPRESS,
                        DeepseekV4AttentionType.INDEXER_COMPRESS,
                    ],
                    None,
                )
                add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
                    ],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.COMPRESSOR_KV),
                )
            elif compress_ratio == 128:
                add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.SWA,
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    ],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
                add_layer(layer, [DeepseekV4AttentionType.COMPRESS], None)
            else:
                raise ValueError(f"Unsupported DeepSeek-V4 compress ratio {compress_ratio}.")

    def _add_v41_cache_layers(self, add_layer, alias_layer) -> None:
        """Assign cache roles for the V4.1 ``{0, 1, 2}`` compress-ratio encoding.

        Three shapes, driven by the reference ``Attention``/``Compressor``
        (``<checkpoint>/inference/model.py``):

        * ``0`` — no compressed branch (``if self.compress_ratio:`` at model.py:775
          is false), so SWA only. This is the role V4 spells ``1``.
        * ``1`` — a long-range branch whose "pooling" is the identity: the
          compressor is just ``self.norm(self.wkv(x))`` (model.py:461) over a
          full-length compressed cache. Indexed, but with no cross-step pooling
          accumulator, hence no ``COMPRESSOR_*`` buffers.
        * ``2`` — the same plus genuine 2-token pooling with a learned fp32 gate,
          which does need the accumulator.

        Every long-range layer is indexed, so ``INDEXER_COMPRESS`` always
        accompanies ``COMPRESS``. Neither ``INDEXER_COMPRESSOR_*`` role appears:
        V4.1's ``Indexer`` owns no compressor of its own (model.py:496-525), it
        derives its keys from the main compressor's latent.

        Cross-layer sharing is what makes the compressed side affordable. Only the
        four ``kv_source_layer_ids`` run a ``Compressor`` and write a compressed
        cache (``model.py:657``); the 34 long-range layers after them read the
        nearest preceding source's, which the reference expresses by having a
        source overwrite the ``shared_attn`` globals its dependents then read
        (``model.py:749``). So a dependent allocates nothing and is *aliased* onto
        its source's ``LayerId``. Allocating per layer instead and broadcasting the
        latent into 34 copies would cost the full compressed footprint 9.5x over
        plus a copy per layer per step, to store 34 duplicates of the same bytes.

        The index-key cache follows the *same* four layers, not the eight index
        sources: ``owns_k`` is ``layer_id in kv_source_layers``
        (``model.py:499``), because index keys are a reprojection of the
        compressor's latent and only a layer that produced a latent can project
        one. The eight ``index_source_layer_ids`` own the top-k *computation*
        (queries and ``weights_proj``), which is a module, not a cache -- see
        ``owns_index_topk`` versus ``owns_compressed_kv`` in ``params.py``. Hence
        ``INDEXER_COMPRESS`` is allocated with ``COMPRESS``, on the KV sources.

        Aliasing is safe here only because every sharing group is
        ratio-uniform -- kv source 2 owns layers 2-7 and 8 owns 9-13 at ratio 2,
        20 owns 21-39 at ratio 1 -- so a dependent never resolves into a pool
        built for a different pooling factor. ``_assert_layer_pool_scale`` is what
        keeps that from silently regressing.
        """
        for layer in self.pp_layers:
            compress_ratio = self._compress_ratios[layer]
            if compress_ratio < 0 or compress_ratio > 2:
                raise ValueError(
                    f"Unsupported DeepSeek-V4.1 compress ratio {compress_ratio} on layer "
                    f"{layer}; expected 0 (SWA only), 1 (unpooled long-range) or 2 (pooled)."
                )
            add_layer(
                layer,
                [DeepseekV4AttentionType.SWA],
                self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
            )
            if not is_compress_layer(compress_ratio, self._variant):
                continue
            assert is_sparse_layer(compress_ratio, self._variant), (
                "every V4.1 long-range layer must be indexed"
            )
            compressed_roles = [
                DeepseekV4AttentionType.COMPRESS,
                DeepseekV4AttentionType.INDEXER_COMPRESS,
            ]
            if owns_compressed_kv(layer, compress_ratio, self._kv_source_layer_ids, self._variant):
                add_layer(layer, compressed_roles, None)
            else:
                source = source_layer_for(layer, self._kv_source_layer_ids)
                assert source is not None, (
                    f"layer {layer} has a compressed branch but no KV source at or "
                    f"before it in {list(self._kv_source_layer_ids or [])}, so nothing "
                    "would ever write the cache it reads."
                )
                assert self._compress_ratios[source] == compress_ratio, (
                    f"layer {layer} (ratio {compress_ratio}) would alias onto layer "
                    f"{source} (ratio {self._compress_ratios[source]}); a sharing "
                    "group must be ratio-uniform or the pooled positions do not line up."
                )
                # Aliasing is a pointer into a pool this rank allocated, so it
                # cannot cross a pipeline stage. V4.1's groups are wide (kv source
                # 20 owns layers 21-39), so a PP split can land inside one; that
                # needs the source's compressed cache sent across the stage
                # boundary, which is a separate feature. Fail here rather than
                # deeper, where the symptom would be a bare KeyError.
                assert source in self.pp_layers, (
                    f"layer {layer} shares layer {source}'s compressed KV cache, but "
                    f"{source} is on another pipeline stage. DeepSeek-V4.1 cross-layer "
                    "sharing does not span PP stages: choose a PP split that keeps each "
                    f"kv_source_layer_ids group {list(self._kv_source_layer_ids or [])} "
                    "within one stage."
                )
                alias_layer(layer, compressed_roles, source)
                # A dependent runs no compressor, so it needs no pooling
                # accumulator either -- skip the COMPRESSOR_* block below.
                continue
            if has_compressor_state(compress_ratio, self._variant):
                add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    ],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.COMPRESSOR_KV),
                )

    def _init_indexer_dtype(self, sparse_attn_config: DeepSeekV4SparseAttentionConfig) -> None:
        # Indexer compressor cache layout. Two modes are supported:
        #   - "fp8" (FP8 blockwise): 1 byte per value + 1 fp32 scale per 128
        #     values.
        #   - "fp4" (MXFP4 blockwise): ½ byte per value (two FP4 codes packed
        #     per byte) + 1 ue8m0 byte per 32 values. At index_head_dim=128
        #     this halves the per-token indexer-K footprint vs FP8.
        self._indexer_k_dtype = sparse_attn_config.indexer_k_dtype
        if self._indexer_k_dtype == "fp8":
            self._indexer_cache_dtype = KVCacheDtype.FP8_BLOCKWISE
            self._indexer_dtype = DataType.FP8
            self.quant_block_size = 128
            self._indexer_data_size = self.index_head_dim
            self._indexer_scale_size = get_size_in_bytes(
                self.index_head_dim // self.quant_block_size, DataType.FLOAT
            )
        elif self._indexer_k_dtype == "fp4":
            assert self.index_head_dim == 128, (
                f"FP4 indexer K cache requires index_head_dim=128, got {self.index_head_dim}."
            )
            self._indexer_cache_dtype = KVCacheDtype.MXFP4_BLOCKWISE
            # Pool dtype is uint8 because PyTorch can't allocate float4
            # backing storage; downstream consumers reinterpret these
            # raw bytes as packed E2M1 + UE8M0 exponents.
            self._indexer_dtype = DataType.UINT8
            self.quant_block_size = 32
            # Two E2M1 codes pack into one byte → half the data footprint.
            self._indexer_data_size = self.index_head_dim // 2
            # 1 UE8M0 byte per 32-element block.
            self._indexer_scale_size = self.index_head_dim // self.quant_block_size
        else:
            raise ValueError(
                f"Unsupported indexer_k_dtype "
                f"{sparse_attn_config.indexer_k_dtype!r}; expected "
                "'fp8' or 'fp4'."
            )
        # FP4 indexer flag mirrors `DSACacheManager.use_fp4` so the shared
        # base Indexer can branch without knowing the V4-specific enum.
        self.use_fp4 = self._indexer_k_dtype == "fp4"
        assert self.index_head_dim % self.quant_block_size == 0, (
            f"indexer_head_dim {self.index_head_dim} must be divisible by {self.quant_block_size}"
        )

    def _assert_layer_pool_scale(self) -> None:
        attn_ratio_to_pool_id = defaultdict[DeepseekV4AttentionType, dict[int, int]](lambda: {})
        attn_ratio_to_scale = defaultdict[DeepseekV4AttentionType, dict[int, int]](lambda: {})

        # Presence in the map, not the ratio predicate, decides which (layer, role)
        # pairs exist. The two agree under V4, but under V4.1's cross-layer sharing
        # a dependent long-range layer carries a compressor's ratio while running no
        # compressor, so the predicate would claim COMPRESSOR_* roles it never
        # allocated. Asking the map is also strictly the right question here: this
        # check is about the pools that were actually built.
        comb = [
            (attn_type, layer_idx)
            for attn_type in DeepseekV4AttentionType
            for layer_idx in self.pp_layers
            if (layer_idx, attn_type) in self._layer_attn_to_layer_id
        ]
        for attn_type, layer_idx in comb:
            compress_ratio = self._compress_ratios[layer_idx]
            layer_id = self._layer_attn_to_layer_id[layer_idx, attn_type]
            pool_id = self.layer_to_pool_mapping_dict[layer_id]
            converter = self.impl.get_page_index_converter(layer_id, attn_type.role)
            assert converter.expansion == 1, "DeepSeek-V4 page index expansion must be 1"
            scale = converter.scale

            # check if the pool id is consistent
            if compress_ratio in attn_ratio_to_pool_id[attn_type]:
                other_pool_id = attn_ratio_to_pool_id[attn_type][compress_ratio]
                assert other_pool_id == pool_id, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has pool id {pool_id}, "
                    f"but another layer with the same compress ratio and attention type has pool id {other_pool_id}."
                    "DeepSeek-V4 expects they share the same pool."
                )
            else:
                attn_ratio_to_pool_id[attn_type][compress_ratio] = pool_id

            # check if the scale is consistent
            if compress_ratio in attn_ratio_to_scale[attn_type]:
                other_scale = attn_ratio_to_scale[attn_type][compress_ratio]
                assert other_scale == scale, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has scale {scale}, "
                    f"but another layer with the same compress ratio and attention type has scale {other_scale}."
                    "DeepSeek-V4 expects they share the same scale."
                )
            else:
                attn_ratio_to_scale[attn_type][compress_ratio] = scale

        # check if all swa attentions are in the same pool and have the same scale
        swa_pool_ids = set(attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA].values())
        swa_scales = set(attn_ratio_to_scale[DeepseekV4AttentionType.SWA].values())
        assert len(swa_pool_ids) == 1, "All swa attentions must be in the same pool"
        assert len(swa_scales) == 1, "All swa attentions must have the same scale"

        # Ensure all compress ratios have SWA entries, not just PP-local ones.
        # The attention metadata uses compress_ratio=1 as a hardcoded SWA key,
        # but with pipeline parallelism a PP stage may not have any layers with
        # ratio 1. Since all SWA layers share the same pool, we populate entries
        # for every compress ratio in the model.
        swa_pool_id = next(iter(swa_pool_ids))
        swa_scale = next(iter(swa_scales))
        for ratio in set(self._compress_ratios):
            if ratio not in attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA]:
                attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA][ratio] = swa_pool_id
                attn_ratio_to_scale[DeepseekV4AttentionType.SWA][ratio] = swa_scale

    def _get_attn_bytes_per_block(
        self,
        attn_type: DeepseekV4AttentionType,
        layer_idx: int,
    ) -> int:
        """
        Get the cache bytes per block for a specific attention type and layer.
        """
        has_fp8_kv_cache = self.dtype == DataType.FP8
        token_bytes = get_token_bytes(
            self.head_dim,
            self.index_head_dim,
            self._compress_ratios[layer_idx],
            attn_type,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )

        block_size = self.tokens_per_block
        if attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            block_size = self.compressed_block_sizes[layer_idx]

        return token_bytes * block_size

    def get_cache_bytes_per_token(self) -> int:
        """Get the average cache bytes per token for DeepSeek-V4."""
        has_fp8_kv_cache = self.dtype == DataType.FP8
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        return _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )

    def get_max_resource_count(self) -> int:
        # Keep scheduler capacity tied to physical GPU KV quota in bytes.
        return int(self.impl.get_quota(GPU_LEVEL))

    def _is_context_request(self, request: llm_request.LlmRequest) -> bool:
        if request.is_context_init_state:
            return True
        return request.state == llm_request.LlmRequestState.CONTEXT_INIT

    def _is_generation_request(self, request: llm_request.LlmRequest) -> bool:
        if (
            request.is_generation_in_progress_state
            or request.is_generation_to_complete_state
            or request.is_disagg_generation_init_state
        ):
            return True
        return request.state in (
            llm_request.LlmRequestState.GENERATION_IN_PROGRESS,
            llm_request.LlmRequestState.GENERATION_TO_COMPLETE,
        )

    def _get_context_bytes(self, request: llm_request.LlmRequest) -> int:
        prompt_len = max(0, request.prompt_len)
        total_tokens = prompt_len + self.num_extra_kv_tokens
        return self._get_cache_bytes_for_tokens(total_tokens, context=True)

    def _get_generation_bytes(self, request: llm_request.LlmRequest) -> int:
        prompt_len = max(0, request.prompt_len)
        max_new_tokens = max(0, request.max_new_tokens)
        total_tokens = prompt_len + max_new_tokens + self.num_extra_kv_tokens
        return self._get_cache_bytes_for_tokens(total_tokens, context=False)

    def _get_cache_bytes_for_tokens(self, total_tokens: int, *, context: bool) -> int:
        has_fp8_kv_cache = self.dtype == DataType.FP8
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        swa_size_per_token, swa_size_per_request = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=context,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self._variant,
        )
        return int(
            total_tokens * (non_sliding_attn_size_per_token + swa_size_per_token)
            + swa_size_per_request
        )

    def get_needed_resource_to_completion(self, request: llm_request.LlmRequest) -> int:
        if self._is_generation_request(request):
            return self._get_generation_bytes(request)
        if self._is_context_request(request):
            return self._get_context_bytes(request)
        raise ValueError(f"Unsupported request state: {request.state}")

    def get_layer_bytes_per_token(
        self,
        local_layer_idx: int,
        data_role: DataRole,
    ) -> int:
        # The generic layers in the base config are replaced by
        # _build_cache_config, so their buffer sizes are only placeholders.
        return 1

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """
        Get the buffers for the indexer k cache for a specific layer.
        """
        buffer = self.get_buffers(layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS).unsqueeze(2)
        return buffer.view(torch.uint8)

    def _compute_shared_block_table(
        self, pool_id: int, scale: int, copy_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the shared offset for one pool and copy index.
        Return shape: [num_seqs, max_blocks_per_seq]
        """
        base = self.host_kv_cache_block_offsets[pool_id, copy_idx, 0, :]
        return torch.where(base == BAD_PAGE_INDEX, BAD_PAGE_INDEX, base * scale)

    def _copy_idx_to_device(self, copy_idx: torch.Tensor) -> torch.Tensor:
        num_tables = copy_idx.size(0)
        device_copy_idx = self._device_copy_idx_staging[:num_tables]
        device_copy_idx.copy_(copy_idx, non_blocking=True)
        # Keep the compiled graph independent of the active table count.
        return self._device_copy_idx_staging

    def _copy_scratch_metadata_to_device(
        self,
        scratch_descs_by_pool: list[list[ScratchDesc | None]],
        num_contexts: int,
        host_begs_staging: torch.Tensor,
        host_ends_staging: torch.Tensor,
        host_slots_staging: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [num_pools, num_contexts]
        host_begs = host_begs_staging[:, :num_contexts]
        # shape: [num_pools, num_contexts]
        host_ends = host_ends_staging[:, :num_contexts]
        # shape: [num_pools, num_contexts, num_slots]
        host_slots = host_slots_staging[:, :num_contexts, :]
        host_begs.zero_()
        host_ends.zero_()
        host_slots.zero_()

        for pool_idx, scratch_descs in enumerate(scratch_descs_by_pool):
            for context_idx, desc in enumerate(scratch_descs):
                if desc is None:
                    continue
                slot_ids = desc.slot_ids
                if len(slot_ids) > self._max_scratch_slots:
                    raise RuntimeError(
                        f"Scratch slot count {len(slot_ids)} exceeds staging capacity "
                        f"{self._max_scratch_slots}"
                    )
                host_begs[pool_idx, context_idx] = int(desc.range.beg)
                host_ends[pool_idx, context_idx] = int(desc.range.end)
                for slot_idx, slot_id in enumerate(slot_ids):
                    host_slots[pool_idx, context_idx, slot_idx] = int(slot_id)

        self._device_scratch_begs_staging.copy_(host_begs_staging, non_blocking=True)
        self._device_scratch_ends_staging.copy_(host_ends_staging, non_blocking=True)
        self._device_scratch_slots_staging.copy_(host_slots_staging, non_blocking=True)
        # Keep scratch tensor shapes fixed; the device-side mask gates num_contexts.
        return (
            self._device_scratch_begs_staging,
            self._device_scratch_ends_staging,
            self._device_scratch_slots_staging,
        )

    @nvtx_range_debug("dsv4_compute_sliding_block_tables")
    def compute_sliding_block_tables(
        self,
        request_ids: List[int],
        num_contexts: int,
    ) -> None:
        """Compute all per-layer sliding-window block tables for this batch."""
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, 1)
        num_tables = copy_idx.size(0)
        self._num_tables = num_tables

        scratch_descs_by_pool = None
        if self.enable_swa_scratch_reuse and num_contexts > 0:
            scratch_descs_by_pool = [
                [
                    self.kv_cache_map[req].get_scratch_desc(pool_id)
                    for req in request_ids[:num_contexts]
                ]
                for pool_id in range(self.num_pools)
            ]

        device_copy_idx = self._copy_idx_to_device(copy_idx)
        self._device_kv_cache_block_offsets_input.copy_(
            self.host_kv_cache_block_offsets,
            non_blocking=True,
        )

        if scratch_descs_by_pool is not None:
            scratch_begs, scratch_ends, scratch_slots = self._copy_scratch_metadata_to_device(
                scratch_descs_by_pool,
                num_contexts,
                self._host_scratch_begs_staging,
                self._host_scratch_ends_staging,
                self._host_scratch_slots_staging,
            )
            self._device_num_contexts.fill_(num_contexts)
            torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables_with_scratch(
                self._device_kv_cache_block_offsets_input,
                device_copy_idx,
                self._device_layer_attn_pool_ids,
                self._device_valid_sliding_pool,
                self._device_layer_attn_scales,
                self._device_layer_offsets,
                self._device_scratch_pages,
                scratch_begs,
                scratch_ends,
                scratch_slots,
                self._device_num_contexts,
                self._precomputed_sliding_block_tables,
            )
        else:
            torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables(
                self._device_kv_cache_block_offsets_input,
                device_copy_idx,
                self._device_layer_attn_pool_ids,
                self._device_valid_sliding_pool,
                self._device_layer_attn_scales,
                self._device_layer_offsets,
                self._precomputed_sliding_block_tables,
            )

    @nvtx_range_debug("dsv4_copy_batch_block_offsets")
    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        """For compatibility with AttentionOp, copy only the SWA block offsets.

        max_blocks is accepted for signature parity with KVCacheManager; the
        copy below is already bounded by the precomputed SWA table width.
        """
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        assert dst_tensor.is_cuda, "copy_batch_block_offsets expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        dst_tensor[:, :num_seqs, 0, :].copy_(
            self._precomputed_sliding_block_tables[
                :, DeepseekV4AttentionType.SWA.value, :num_seqs, :
            ],
            non_blocking=True,
        )

    @nvtx_range_debug("dsv4_copy_batch_sliding_block_tables")
    def copy_batch_sliding_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """
        Copy the per-layer block tables for attentions managed in sliding-window mode to the GPU tensor.
        """
        assert dst_tensor.is_cuda, "copy_batch_sliding_block_tables expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        dst_tensor[:, :, :num_seqs, :].copy_(
            self._precomputed_sliding_block_tables[:, :, :num_seqs, :],
            non_blocking=True,
        )

    @nvtx_range_debug("dsv4_copy_batch_compress_block_tables")
    def copy_batch_compress_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        compress_ratio: int,
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """Build the COMPRESS block table for one compression ratio and copy it to the destination."""
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        staging = self._host_compress_block_tables_staging[compress_ratio]
        meta = self._compress_pool_meta.get(compress_ratio)
        if meta is None:
            raise ValueError(
                f"Unsupported compress ratio {compress_ratio} for "
                f"copy_batch_compress_block_tables; this manager has COMPRESS pools for "
                f"{sorted(self._compress_pool_meta)}"
            )
        pool_id, scale = meta
        staging[:num_seqs] = self._compute_shared_block_table(pool_id, scale, copy_idx)
        dst_tensor[:num_seqs].copy_(staging[:num_seqs], non_blocking=True)

    @property
    def indexer_compress_ratios(self) -> List[int]:
        """Compress ratios that have an INDEXER_COMPRESS pool on this rank, ascending.

        One entry for V4 (only its ratio-4 layers are indexed) and two for V4.1
        (every long-range layer is indexed, at ratio 1 or 2). The metadata needs
        this to size one page table per pool: the pools have different page
        scales, so a single table cannot address both.
        """
        return sorted(self._indexer_compress_pool_meta)

    @nvtx_range_debug("dsv4_copy_batch_indexer_compress_block_tables")
    def copy_batch_indexer_compress_block_tables(
        self,
        host_block_table: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        compress_ratio: Optional[int] = None,
    ) -> None:
        """Build the shared INDEXER_COMPRESS compatibility block table.

        ``compress_ratio=None`` means "the one indexer pool this model has", which
        is always the case for V4: only its ratio-4 layers are indexed. V4.1 indexes
        every long-range layer, so it has one indexer pool per distinct ratio and
        callers must say which. Refusing to guess is deliberate — silently returning
        the ratio-1 table for a ratio-2 layer would scatter indexer keys into the
        wrong pages and read as an accuracy bug rather than a wiring bug.
        """
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        if compress_ratio is None:
            if len(self._indexer_compress_pool_meta) != 1:
                raise RuntimeError(
                    "compress_ratio is required when the model has "
                    f"{len(self._indexer_compress_pool_meta)} INDEXER_COMPRESS pools "
                    f"(ratios {sorted(self._indexer_compress_pool_meta)})"
                )
            (compress_ratio,) = self._indexer_compress_pool_meta
        meta = self._indexer_compress_pool_meta.get(compress_ratio)
        if meta is None:
            raise RuntimeError(
                f"Missing INDEXER_COMPRESS pool metadata for compress ratio "
                f"{compress_ratio}; this manager has {sorted(self._indexer_compress_pool_meta)}"
            )
        pool_id, scale = meta
        host_block_table[:num_seqs] = self._compute_shared_block_table(pool_id, scale, copy_idx)

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping, **kwargs):
        config = model_config.pretrained_config
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim
        index_head_dim = model_config.sparse_attention_config.index_head_dim
        pp_layers = mapping.pp_layers(model_config.get_num_attention_layers())
        compress_ratios = [
            model_config.sparse_attention_config.compress_ratios[layer] for layer in pp_layers
        ]
        quant_config = model_config.quant_config
        if quant_config is not None:
            has_fp8_kv_cache = quant_config.quant_mode.has_fp8_kv_cache()
        else:
            has_fp8_kv_cache = False
        indexer_k_dtype = model_config.sparse_attention_config.indexer_k_dtype
        variant = getattr(model_config.sparse_attention_config, "variant", DEEPSEEK_V4_VARIANT)
        kv_cache_config = kwargs.get("kv_cache_config")
        use_fp8_ds_mla = (
            kv_cache_config is not None
            and getattr(kv_cache_config, "dtype", "auto") == "fp8_ds_mla"
        )
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=indexer_k_dtype,
            use_fp8_ds_mla=use_fp8_ds_mla,
            variant=variant,
        )
        swa_size_per_token, swa_size_per_request = _estimate_swa_cache_size(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            kwargs["tokens_per_block"],
            model_config.sparse_attention_config.window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=indexer_k_dtype,
            use_fp8_ds_mla=use_fp8_ds_mla,
            variant=variant,
        )
        max_batch_size = int(kwargs.get("max_batch_size") or 0)
        return (
            non_sliding_attn_size_per_token + swa_size_per_token,
            swa_size_per_request * max_batch_size,
        )

    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor(
            [False], dtype=torch.bool, device=torch.cuda.current_device()
        )
        buffers_handled = set()

        # Handle each attention buffer from start to end to traverse the whole
        # KV cache. Multiple attention buffers can now share one cache layer.
        for (layer, attn), layer_id in self._layer_attn_to_layer_id.items():
            data_role = attn.role
            buffer_key = (layer_id, data_role)
            if buffer_key in buffers_handled:
                continue
            buffer = self.get_buffers(layer, attn)
            # process in chunks of 256 pages to avoid OoM
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i : i + 256]
                try:
                    has_invalid_values.logical_or_(torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
            buffers_handled.add(buffer_key)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, "
                "related checks are skipped"
            )
        return bool(has_invalid_values)
