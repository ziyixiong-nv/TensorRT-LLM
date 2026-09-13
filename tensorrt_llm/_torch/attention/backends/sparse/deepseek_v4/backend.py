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

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm.models.modeling_utils import QuantConfig

from .cache_manager import get_token_bytes
from .compressor import Compressor, DeepseekV41Compressor
from .indexer import DeepseekV4Indexer, DeepseekV41Indexer
from .kernels import deepseek_v4_local_to_global_indices
from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import (
    DEEPSEEK_V4_VARIANT,
    DeepseekV4AttentionType,
    DeepSeekV4Params,
    is_compress_layer,
    is_sparse_layer,
    is_v41,
    owns_compressed_kv,
    owns_index_topk,
    source_layer_for,
)

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class DeepseekV4TrtllmAttention(TrtllmAttention):
    Metadata = DeepseekV4TrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        sparse_params: Optional[DeepSeekV4Params] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        if sparse_attention_config is None:
            sparse_attention_config = sparse_params
        assert sparse_attention_config is not None, (
            "sparse_attention_config is required for DeepseekV4TrtllmAttention and cannot be None"
        )
        if sparse_params is None:
            sparse_params = sparse_attention_config.to_sparse_params()
        assert sparse_params is not None, (
            "sparse_params is required for DeepseekV4TrtllmAttention and cannot be None"
        )
        kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
        self.use_fp8_ds_mla = kv_cache_dtype == "fp8_ds_mla"
        assert mla_params is not None, "DeepSeek-V4 attention requires MLA parameters"
        mla_params = replace(
            mla_params,
            v_head_dim=head_dim,
            rope_append=False,
        )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_params=sparse_params,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs,
        )

        self.sparse_attention_config = sparse_attention_config
        self.compress_ratio = sparse_attention_config.compress_ratios[layer_idx]
        # `compress_ratio` is both a pooling factor and a role sentinel, and the
        # two generations encode the roles differently (see `params.py`). Resolve
        # the roles once here so the rest of the backend never re-derives them
        # from the raw integer: V4.1's ratio 1 is a long-range layer where V4's
        # ratio 1 is SWA-only, so `ratio > 1` is not a portable test.
        self.variant = getattr(sparse_attention_config, "variant", DEEPSEEK_V4_VARIANT)
        self.has_compressed_kv = is_compress_layer(self.compress_ratio, self.variant)
        self.is_indexed = is_sparse_layer(self.compress_ratio, self.variant)
        # `has_compressed_kv` / `is_indexed` say what this layer *reads*; these say
        # what it *produces*. V4 always produces what it reads, so both collapse to
        # the pair above and the V4 path is unchanged. V4.1 splits them: 4 of its 38
        # long-range layers own a Compressor and 8 own an Indexer, and the rest read
        # the nearest preceding source's output. A non-owner must not build the
        # module (it has no weights for it) and must not allocate the cache (nothing
        # would write it); it is pointed at its source's buffer instead, see
        # `DeepseekV4TrtllmAttentionMetadata._build_cache_buffer_data_pointers`.
        self.kv_source_layer_ids = sparse_params.kv_source_layer_ids
        self.index_source_layer_ids = sparse_params.index_source_layer_ids
        self.owns_compressed_kv = owns_compressed_kv(
            layer_idx, self.compress_ratio, self.kv_source_layer_ids, self.variant
        )
        self.owns_indexer = owns_index_topk(
            layer_idx, self.compress_ratio, self.index_source_layer_ids, self.variant
        )
        # Whose compressed KV / top-k this layer consumes. Identity for an owner.
        self.kv_source_layer_idx = (
            source_layer_for(layer_idx, self.kv_source_layer_ids)
            if self.has_compressed_kv
            else None
        )
        self.index_source_layer_idx = (
            source_layer_for(layer_idx, self.index_source_layer_ids) if self.is_indexed else None
        )
        if self.has_compressed_kv and self.kv_source_layer_ids:
            assert self.kv_source_layer_idx is not None, (
                f"layer {layer_idx} reads a compressed-KV cache but no layer in "
                f"kv_source_layer_ids={list(self.kv_source_layer_ids)} runs at or "
                "before it, so nothing would ever write that cache."
            )
        if self.is_indexed and self.index_source_layer_ids:
            assert self.index_source_layer_idx is not None, (
                f"layer {layer_idx} reads an index top-k but no layer in "
                f"index_source_layer_ids={list(self.index_source_layer_ids)} runs "
                "at or before it, so no selection would ever be published."
            )

        if self.owns_indexer:
            indexer_kwargs = {}
            if is_v41(self.variant):
                indexer_cls = DeepseekV41Indexer
                # V4.1's index keys are a reprojection of the compressed latent, so
                # only a layer that compresses its own KV can build them
                # (``owns_k`` at model.py:499). The four index sources that are not
                # KV sources read keys from their kv source's cache and carry no
                # `wk` in the checkpoint.
                indexer_kwargs["owns_index_keys"] = self.owns_compressed_kv
                # Which cache a non-owner scores against. Also the key into
                # `metadata.v41_index_keys` on the single-pass prefill path,
                # where the keys are handed over as activations rather than read
                # back out of the cache.
                indexer_kwargs["kv_source_layer_idx"] = self.kv_source_layer_idx
            else:
                indexer_cls = DeepseekV4Indexer
            self.indexer = indexer_cls(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_params,
                dtype,
                self.compress_ratio,
                layer_idx,
                aux_stream,
                **indexer_kwargs,
            )

        if self.owns_compressed_kv:
            # The reference passes `args.norm_eps` at every compressor call site.
            # V4 ships 1e-6 and V4.1 ships 1e-20, so take it from the resolved
            # sparse params rather than hardcoding either generation's value.
            rms_norm_eps = sparse_params.rms_norm_eps
            has_fp8_kv_cache = False
            if quant_config is not None:
                has_fp8_kv_cache = quant_config.layer_quant_mode.has_fp8_kv_cache()
            kv_cache_dtype = "fp8_pertensor" if has_fp8_kv_cache else "default"
            compressor_cls = DeepseekV41Compressor if is_v41(self.variant) else Compressor
            self.compressor = compressor_cls(
                mla_params,
                layer_idx,
                self.compress_ratio,
                rms_norm_eps,
                skip_create_weights_in_init,
                pos_embd_params,
                kv_cache_dtype=kv_cache_dtype,
                dtype=dtype,
                rotate_activation=False,
            )
            if self.use_fp8_ds_mla:
                self.compressor.enable_footer_scale_cache()

    def _prepare_sparse_forward_args(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> None:
        attention_input_type = forward_args.attention_input_type
        if attention_input_type == AttentionInputType.context_only:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
        else:
            start_idx = 0
            end_idx = metadata.num_tokens

        sparse_args = forward_args.sparse_runtime_params
        sparse_args.sparse_attn_kv_lens = metadata.sparse_mla_topk_lens[self.compress_ratio][
            start_idx:end_idx
        ]
        if self.has_compressed_kv:
            sparse_args.aux_kv_cache_pool_ptr = metadata.sparse_mla_base_ptrs[self.compress_ratio]
        else:
            sparse_args.aux_kv_cache_pool_ptr = None

        metadata.num_sparse_topk = (
            self.sparse_attention_config.window_size
            + metadata.max_compressed_indices[self.compress_ratio]
        )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ):
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        attn_sink = getattr(self, "attn_sink", None)
        if attn_sink is not None:
            if forward_args.attention_sinks is None:
                forward_args = replace(forward_args, attention_sinks=attn_sink.data)

        self._prepare_sparse_forward_args(metadata, forward_args)
        return super().forward(q, k, v, metadata, forward_args=forward_args)

    def _unit_scale(self, like: torch.Tensor) -> torch.Tensor:
        """Cached [1.0], for scale pointers the Triton kernel cannot take as null."""
        cached = getattr(self, "_unit_scale_tensor", None)
        if cached is None or cached.device != like.device:
            cached = torch.ones(1, dtype=torch.float32, device=like.device)
            self._unit_scale_tensor = cached
        return cached

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert local indices (SWA + compressed) to global pool indices."""
        layer_idx = self.layer_idx
        kv_cache_manager = metadata.kv_cache_manager
        attention_input_type = forward_args.attention_input_type

        swa_pool_base_ptr = metadata.swa_pool_base_ptr

        # Get cached buffer pointers
        swa_buffer_ptr = metadata.swa_buffer_ptrs[layer_idx]

        # Token stride
        index_head_dim = self.sparse_attention_config.index_head_dim
        has_fp8_kv_cache = False
        if self.quant_config is not None:
            has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache()
        token_stride = get_token_bytes(
            self.head_dim,
            index_head_dim,
            self.compress_ratio,
            DeepseekV4AttentionType.SWA,
            has_fp8_kv_cache,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            variant=self.variant,
        )

        # Select token range based on phase
        if attention_input_type == AttentionInputType.context_only:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
        else:
            start_idx = 0
            end_idx = metadata.num_tokens

        # Use global req_id directly
        req_id = metadata.req_idx_per_token[start_idx:end_idx]
        swa_local_indices = metadata.swa_local_indices_cuda[start_idx:end_idx]
        local_layer_idx = kv_cache_manager.layer_offsets[layer_idx]
        block_table_swa = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]

        if self.has_compressed_kv:
            compressed_buffer_ptr = metadata.compressed_buffer_ptrs[layer_idx]
            compress_pool_base_ptr = metadata.sparse_mla_base_ptrs[self.compress_ratio]
            block_table_compressed = metadata.compress_block_tables[self.compress_ratio]
            if self.is_indexed:
                sparse_backend_args = forward_args.sparse_backend_args
                assert sparse_backend_args is not None, (
                    "sparse_backend_args is required for an indexed layer"
                )
                topk_indices = sparse_backend_args.topk_indices
                assert topk_indices is not None, "topk_indices is required for an indexed layer"
                compressed_local_indices = topk_indices
            else:
                compressed_local_indices = metadata.compressed_local_indices_cuda[start_idx:end_idx]
        else:
            compressed_buffer_ptr = 0
            compress_pool_base_ptr = 0
            block_table_compressed = None
            compressed_local_indices = None

        # FMHA scheduler prologue: this kernel is the last one before FMHA, so it owns
        # the tile-counter reset and bmm scale derivation the MLA RoPE kernels used to
        # do two launches earlier. Generation only -- context uses the attention
        # workspace.
        sched_kwargs = {}
        if (
            attention_input_type == AttentionInputType.generation_only
            and has_fp8_kv_cache
            and forward_args.fmha_scheduler_counter is not None
            and forward_args.mla_bmm1_scale is not None
            and forward_args.mla_bmm2_scale is not None
        ):
            sched_kwargs = dict(
                fmha_tile_counter=forward_args.fmha_scheduler_counter,
                bmm1_scale=forward_args.mla_bmm1_scale,
                bmm2_scale=forward_args.mla_bmm2_scale,
                # Mirrors attentionOp.cpp: quant_scale_o is the attention-output
                # quant scale, and both dequant scales are the KV cache scale.
                # The Triton kernel always dereferences quant_scale_o, so an absent
                # out_scale needs an explicit 1.0 -- the value mlaKernels.cu:490
                # substitutes for a null pointer. Falling back to the KV scale here
                # would square it into bmm2.
                quant_scale_o=(
                    forward_args.out_scale
                    if forward_args.out_scale is not None
                    else self._unit_scale(self.kv_scale_quant_orig)
                ),
                dequant_scale_q=self.kv_scale_quant_orig,
                dequant_scale_kv=self.kv_scale_quant_orig,
                host_bmm1_scale=1.0
                / (
                    self.q_scaling * math.sqrt(float(self.qk_nope_head_dim + self.qk_rope_head_dim))
                ),
            )

        result = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table_swa,
            swa_local_indices=swa_local_indices,
            swa_pool_base_ptr=swa_pool_base_ptr,
            swa_buffer_ptr=swa_buffer_ptr,
            tokens_per_block=kv_cache_manager.tokens_per_block,
            token_stride=token_stride,
            block_table_compressed=block_table_compressed,
            compressed_local_indices=compressed_local_indices,
            compress_pool_base_ptr=compress_pool_base_ptr,
            compressed_buffer_ptr=compressed_buffer_ptr,
            compress_ratio=self.compress_ratio,
            has_compressed=self.has_compressed_kv,
            num_compressed_indices=metadata.max_compressed_indices[self.compress_ratio],
            **sched_kwargs,
            split_extra=self.use_fp8_ds_mla,
        )

        if self.use_fp8_ds_mla:
            return result
        return result, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None
