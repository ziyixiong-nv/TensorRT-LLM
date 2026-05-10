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

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from ..pyexecutor.resource_manager import BaseResourceManager
from .interface import SpecMetadata, SpecWorkerBase


@triton.jit
def _build_dflash_query_inputs_kernel(
    # Inputs
    accepted_tokens_ptr,  # [num_gens, accepted_stride] int32
    accepted_tokens_stride,
    num_accepted_ptr,  # [num_gens] int32
    ctx_len_ptr,  # [max_batch] int64
    slots_ptr,  # [num_gens] int64 (already sliced from _batch_to_slot)
    # Outputs
    input_ids_out_ptr,  # [num_gens, block_size] int64
    query_positions_out_ptr,  # [num_gens, block_size] int64
    ctx_positions_out_ptr,  # [num_gens, K_plus_1] int64
    # Scalars
    mask_token_id,
    block_size,
    K_plus_1,
    BLOCK: tl.constexpr,
):
    """Build DFlash per-gen-request query inputs in one kernel:

      input_ids[i, 0]        = bonus (last accepted token)
      input_ids[i, 1..block]  = mask_token_id
      query_positions[i, j]   = ctx_len[slot(i)] + num_accepted(i) + j
      ctx_positions[i, j]     = ctx_len[slot(i)] + j   (j < K+1)

    Replaces ~10 small PyTorch ops (expand + clone + gather + scatter + arange
    broadcasts) that ran sequentially on the host per draft step.
    """
    req_idx = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    query_mask = j < block_size
    ctx_mask = j < K_plus_1

    slot = tl.load(slots_ptr + req_idx)
    ctx_len = tl.load(ctx_len_ptr + slot)
    num_acc = tl.load(num_accepted_ptr + req_idx)
    bonus_idx = tl.maximum(num_acc - 1, 0)
    bonus_token = tl.load(accepted_tokens_ptr + req_idx * accepted_tokens_stride + bonus_idx)

    # input_ids
    input_id = tl.where(
        j == 0, bonus_token.to(tl.int64), tl.full((BLOCK,), mask_token_id, dtype=tl.int64)
    )
    tl.store(input_ids_out_ptr + req_idx * block_size + j, input_id, mask=query_mask)

    # query positions: ctx_len + num_accepted + j
    query_pos = ctx_len + num_acc.to(tl.int64) + j.to(tl.int64)
    tl.store(query_positions_out_ptr + req_idx * block_size + j, query_pos, mask=query_mask)

    # ctx positions: ctx_len + j for j < K+1
    ctx_pos = ctx_len + j.to(tl.int64)
    tl.store(ctx_positions_out_ptr + req_idx * K_plus_1 + j, ctx_pos, mask=ctx_mask)


if TYPE_CHECKING:
    from ...llmapi.llm_args import DFlashDecodingConfig


@dataclass
class DFlashSpecMetadata(SpecMetadata):
    """Metadata for DFlash speculative decoding.

    Captures hidden states from specific target model layers during the target
    forward pass, which are then projected through fc + hidden_norm and fed to
    the DFlash draft model as cross-attention context.
    """

    batch_indices_cuda: Optional[torch.Tensor] = None
    spec_resource_manager: Optional[BaseResourceManager] = None

    # Hidden state capture fields
    layers_to_capture: Optional[List[int]] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    captured_hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

        # Set up hidden state capture buffer
        if self.layers_to_capture is not None and len(self.layers_to_capture) > 0:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            self.num_capture_layers = len(self.layers_to_capture)
            # O(1) lookups for is_layer_capture() and maybe_capture_hidden_states()
            self._capture_layer_set = frozenset(self.layers_to_capture)
            self._layer_to_idx = {lid: i for i, lid in enumerate(self.layers_to_capture)}
            self.captured_hidden_states = torch.empty(
                (self.max_num_tokens, self.hidden_size * self.num_capture_layers),
                dtype=self.dtype,
                device="cuda",
            )
            logger.info(
                f"DFlash: capturing hidden states from layers {self.layers_to_capture}, "
                f"buffer shape {self.captured_hidden_states.shape}"
            )
        else:
            self.num_capture_layers = 0
            self._capture_layer_set = frozenset()
            self._layer_to_idx = {}

    def prepare(self):
        assert self.request_ids is not None

        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        # Update slot mapping for DFlash context buffers
        worker = getattr(self, "_dflash_worker", None)
        if worker is not None and worker._ctx_buf_inited:
            current = set(self.request_ids)
            for rid in list(worker._req_to_slot.keys()):
                if rid not in current:
                    slot = worker._req_to_slot.pop(rid)
                    worker._release_slot(slot)

            # Default to slot 0 for unknown request IDs (e.g. during warmup
            # where synthetic requests may not have assigned slots).
            mapping = torch.tensor(
                [worker._req_to_slot.get(rid, 0) for rid in self.request_ids],
                dtype=torch.long,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            worker._batch_to_slot[:num_seqs].copy_(mapping, non_blocking=True)

    def is_layer_capture(self, layer_id: int) -> bool:
        return layer_id in self._capture_layer_set

    def maybe_capture_hidden_states(
        self, layer_id: int, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> None:
        """Capture hidden states from a target model layer into the buffer."""
        if self.captured_hidden_states is None:
            return
        i = self._layer_to_idx.get(layer_id)
        if i is not None:
            num_tokens = hidden_states.shape[0]
            to_save = hidden_states + residual if residual is not None else hidden_states
            self.captured_hidden_states[
                :num_tokens, i * self.hidden_size : (i + 1) * self.hidden_size
            ].copy_(to_save, non_blocking=True)

    def get_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Get captured hidden states (all layers concatenated)."""
        if self.captured_hidden_states is None:
            return None
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


class DFlashWorker(SpecWorkerBase):
    """
    Worker for DFlash speculative decoding.

    DFlash uses the draft model with mask tokens to predict multiple
    draft tokens in parallel. The DFlash draft model uses cross-attention
    to target model hidden states captured from specific layers.

    The target features are projected through fc + hidden_norm and fed
    to the draft model as K/V context in cross-attention. The context
    accumulates across steps - each step adds newly accepted tokens'
    projected hidden states to a per-request buffer, giving the draft
    model the full history of target features.

    Reference: https://arxiv.org/pdf/2602.06036
    """

    def __init__(
        self,
        spec_config: "DFlashDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        self._resolved_mask_token_id = None
        self._resolved_block_size = None

        # Pre-allocated context buffers (lazy-init on first forward).
        # Replaces per-request Python dicts with fixed-size CUDA buffers
        # indexed by slot, enabling CUDA graph compatibility.
        self._ctx_buf_inited = False
        self._ctx_buf = None  # [max_batch, max_ctx, proj_dim] (fallback path)
        self._ctx_pos_buf = None
        self._ctx_len = None
        self._batch_to_slot = None
        self._max_ctx = 0
        self._proj_dim = 0

        # Paged per-layer K/V pool (fast path).
        # Layout [L, num_blocks, page_block_size, nkv, hd]: per-layer view is
        # contiguous so flash_attn_with_kvcache's block_table mode can read
        # each layer directly without a gather. num_blocks defaults to
        # max_batch * max_blocks_per_req (same total capacity as the old
        # dense [max_batch, L, max_ctx+block, nkv, hd] layout); users can
        # tune spec_config.ctx_kv_cache_num_blocks lower to trade capacity
        # for memory.
        self._ctx_k_pool = None
        self._ctx_v_pool = None
        self._slot_block_table = None  # [max_batch, max_blocks_per_req] int32
        self._page_block_size = 0
        self._max_blocks_per_req = 0

        # Slot management (Python, updated in prepare() and eager mode)
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._free_block_ids = deque()  # available block ids (paged pool)
        self._slot_blocks = {}  # slot_id -> list[int] of blocks owned

        logger.info(
            f"DFlashWorker initialized with use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    @property
    def _draft_tokens_per_req(self) -> int:
        """Total tokens per gen request in the draft forward.

        Uses 2K to fit all accepted tokens (up to K+1) plus K-1 mask tokens,
        ensuring K unique predictions regardless of how many tokens were accepted.
        """
        return 2 * self.max_draft_len

    def _lazy_init_ctx_buffers(self, draft_model, spec_metadata):
        if self._ctx_buf_inited:
            return

        max_batch = spec_metadata.max_num_requests

        if hasattr(draft_model, "fc"):
            self._proj_dim = draft_model.fc.weight.shape[0]
        else:
            self._proj_dim = spec_metadata.hidden_size

        config_max_ctx = getattr(self.spec_config, "max_ctx_len", None)
        if config_max_ctx is not None:
            self._max_ctx = config_max_ctx
        else:
            config = getattr(draft_model, "config", None)
            max_pos = getattr(config, "max_position_embeddings", 8192) if config else 8192
            self._max_ctx = min(max_pos, 8192)

        dtype = draft_model.fc.weight.dtype if hasattr(draft_model, "fc") else torch.bfloat16

        self._ctx_buf = torch.zeros(
            (max_batch, self._max_ctx, self._proj_dim), dtype=dtype, device="cuda"
        )
        self._ctx_pos_buf = torch.zeros((max_batch, self._max_ctx), dtype=torch.long, device="cuda")
        self._ctx_len = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        self._batch_to_slot = torch.zeros(max_batch, dtype=torch.long, device="cuda")

        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}

        if hasattr(draft_model, "_build_fused_kv_buffers"):
            draft_model._build_fused_kv_buffers()
            L = draft_model._num_attn_layers
            nkv = draft_model._num_kv_heads
            hd = draft_model._head_dim
            model_block_size = getattr(draft_model, "block_size", None) or (self.max_draft_len + 1)

            # Paged pool geometry. flash_attn_with_kvcache requires the page
            # block size to be a multiple of 256.
            page_block = getattr(self.spec_config, "ctx_kv_cache_page_block_size", 256)
            assert page_block % 256 == 0, (
                "ctx_kv_cache_page_block_size must be a multiple of 256 "
                "(flash_attn_with_kvcache requirement)."
            )
            per_req_capacity = self._max_ctx + model_block_size
            max_blocks_per_req = (per_req_capacity + page_block - 1) // page_block
            default_num_blocks = max_batch * max_blocks_per_req
            num_blocks = getattr(self.spec_config, "ctx_kv_cache_num_blocks", default_num_blocks)
            assert num_blocks >= max_blocks_per_req, (
                f"ctx_kv_cache_num_blocks={num_blocks} is too small to hold one "
                f"request's max context (needs {max_blocks_per_req} blocks)."
            )

            self._page_block_size = page_block
            self._max_blocks_per_req = max_blocks_per_req
            pool_shape = (L, num_blocks, page_block, nkv, hd)
            self._ctx_k_pool = torch.zeros(pool_shape, dtype=dtype, device="cuda")
            self._ctx_v_pool = torch.zeros(pool_shape, dtype=dtype, device="cuda")
            self._slot_block_table = torch.zeros(
                (max_batch, max_blocks_per_req), dtype=torch.int32, device="cuda"
            )
            self._free_block_ids = deque(range(num_blocks))
            self._slot_blocks = {}
            logger.info(
                f"DFlash: paged ctx pool: L={L}, num_blocks={num_blocks}, "
                f"page_block={page_block}, nkv={nkv}, hd={hd}, dtype={dtype}, "
                f"max_blocks_per_req={max_blocks_per_req}"
            )
        self._ctx_buf_inited = True

        logger.info(
            f"DFlash: allocated ctx buffers: max_batch={max_batch}, "
            f"max_ctx={self._max_ctx}, proj_dim={self._proj_dim}, dtype={dtype}, "
            f"ctx_kv_cache_enabled={self._ctx_k_pool is not None}"
        )

    def _assign_slot(self, req_id: int) -> int:
        """Pop a free slot + reserve its page blocks. Python-only, called on
        prefill only (rare), never inside the hot draft forward."""
        slot = self._free_slots.popleft()
        self._req_to_slot[req_id] = slot
        if self._ctx_k_pool is not None:
            if len(self._free_block_ids) < self._max_blocks_per_req:
                raise RuntimeError(
                    f"DFlash paged ctx pool exhausted: need "
                    f"{self._max_blocks_per_req} blocks for a new slot but "
                    f"only {len(self._free_block_ids)} free. Raise "
                    f"spec_config.ctx_kv_cache_num_blocks."
                )
            blocks = [self._free_block_ids.popleft() for _ in range(self._max_blocks_per_req)]
            self._slot_blocks[slot] = blocks
            self._slot_block_table[slot].copy_(
                torch.tensor(blocks, dtype=torch.int32), non_blocking=True
            )
        return slot

    def _release_slot(self, slot: int) -> None:
        """Return a slot's blocks to the free pool."""
        self._ctx_len[slot] = 0
        self._free_slots.append(slot)
        if self._ctx_k_pool is not None:
            for bid in self._slot_blocks.pop(slot, ()):
                self._free_block_ids.append(bid)

    def _scatter_ctx_kv_to_pool(
        self, slot: int, start_col: int, chunk_k: torch.Tensor, chunk_v: torch.Tensor
    ) -> None:
        """Scatter prefill K/V into this slot's pre-allocated pool blocks.

        chunk_k / chunk_v are [N, L, nkv, hd]. Positions land at
        [start_col, start_col + N) within this request's logical space, which
        maps into blocks in self._slot_blocks[slot] at offsets computed mod
        page_block_size.
        """
        N = chunk_k.shape[0]
        if N == 0:
            return
        page = self._page_block_size
        blocks = self._slot_blocks[slot]
        # chunk_k: [N, L, nkv, hd] -> [L, N, nkv, hd] so a layer axis indexes
        # cleanly when writing to pool[:, block, off].
        k_ln = chunk_k.permute(1, 0, 2, 3).contiguous()
        v_ln = chunk_v.permute(1, 0, 2, 3).contiguous()
        col = start_col
        src_off = 0
        while src_off < N:
            block_idx = col // page
            off = col % page
            bid = blocks[block_idx]
            take = min(page - off, N - src_off)
            self._ctx_k_pool[:, bid, off : off + take] = k_ln[:, src_off : src_off + take]
            self._ctx_v_pool[:, bid, off : off + take] = v_ln[:, src_off : src_off + take]
            col += take
            src_off += take

    def _prepare_attn_metadata_for_dflash(self, attn_metadata, spec_metadata):
        """Save attn_metadata fields that DFlash modifies during forward."""
        is_capturing = torch.cuda.is_current_stream_capturing()

        if spec_metadata.is_cuda_graph and not is_capturing:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """Adjust kv_lens_cuda so the draft model sees correct RoPE positions."""
        if hasattr(attn_metadata, "kv_lens_cuda"):
            self._kv_rewind_amount = 1 - num_accepted_tokens[num_contexts:batch_size]
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size

            if batch_size > num_contexts:
                attn_metadata.kv_lens_cuda[num_contexts:batch_size] += 1

            attn_metadata.update_for_spec_dec()

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        """Apply the deferred kv_lens rewind after the draft forward."""
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if is_warmup:
            return

        if hasattr(self, "_kv_rewind_amount") and hasattr(attn_metadata, "kv_lens_cuda"):
            nc = self._kv_rewind_nc
            bs = self._kv_rewind_bs
            attn_metadata.kv_lens_cuda[nc:bs] -= self._kv_rewind_amount
            attn_metadata.kv_lens_cuda[nc:bs].clamp_(min=0)

    def _store_prefill_context(
        self,
        draft_model,
        spec_metadata: "DFlashSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ):
        """Capture prefill hidden states and store as initial accumulated context.

        During prefill (context requests), the target model processes all prompt
        tokens.  We project their captured hidden states through fc + hidden_norm
        and store them per-request so the draft model can use the full prompt
        as cross-attention context on subsequent gen steps.
        """
        if not hasattr(draft_model, "fc") or not hasattr(draft_model, "hidden_norm"):
            return

        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return

        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            return

        # Project context tokens through fc + hidden_norm
        ctx_hs = captured_hs[:num_ctx_tokens]
        ctx_proj = draft_model.fc(ctx_hs.to(draft_model.fc.weight.dtype))
        ctx_proj = draft_model.hidden_norm(ctx_proj)

        # Split by request and store/append accumulated context.
        # Context requests may arrive in chunks (chunked prefill), so we
        # must APPEND successive chunks for the same request rather than
        # overwriting.  If a previously-finished request id is reused for a
        # brand-new request, the new chunk's first position will be 0, which
        # signals a fresh start → replace instead of append.
        offset = 0
        num_contexts = attn_metadata.num_contexts
        for i in range(num_contexts):
            req_id = spec_metadata.request_ids[i]
            slen = int(attn_metadata._seq_lens[i])
            chunk_proj = ctx_proj[offset : offset + slen].detach()
            chunk_pos = position_ids[offset : offset + slen].long().detach()

            first_pos = chunk_pos[0].item() if slen > 0 else 0

            # Assign slot for new requests or reset for reused IDs
            if req_id not in self._req_to_slot or first_pos == 0:
                if req_id in self._req_to_slot:
                    old_slot = self._req_to_slot.pop(req_id)
                    self._release_slot(old_slot)
                if not self._free_slots:
                    logger.warning("DFlash: no free slots, skipping context store")
                    offset += slen
                    continue
                self._assign_slot(req_id)

            slot = self._req_to_slot[req_id]
            cur = int(self._ctx_len[slot].item())
            end = min(cur + slen, self._max_ctx)
            actual = end - cur
            if actual > 0:
                chunk_proj_cast = chunk_proj[:actual].to(self._ctx_buf.dtype)
                self._ctx_buf[slot, cur:end] = chunk_proj_cast
                self._ctx_pos_buf[slot, cur:end] = chunk_pos[:actual]
                self._ctx_len[slot] = end
                if self._ctx_k_pool is not None:
                    # Precompute post-norm/post-RoPE K,V for this prefill
                    # chunk so decode iters can read without re-projecting,
                    # then scatter into the paged pool at this slot's blocks.
                    chunk_k, chunk_v = draft_model.precompute_context_kv(
                        chunk_proj_cast, chunk_pos[:actual]
                    )  # [actual, L, nkv, hd]
                    self._scatter_ctx_kv_to_pool(slot, cur, chunk_k, chunk_v)
            offset += slen

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits
        K = self.max_draft_len

        # Lazy init buffers and attach worker reference for prepare()
        self._lazy_init_ctx_buffers(draft_model, spec_metadata)
        spec_metadata._dflash_worker = self

        # Save context lengths before warmup to prevent accumulation
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if is_warmup:
            saved_ctx_len = self._ctx_len.clone()

        self._execute_guided_decoder_if_present(logits)

        # draft_tokens buffer has (2K-1) entries per gen request; extract the K real drafts
        if num_gens > 0:
            draft_tokens_raw = spec_metadata.draft_tokens
            draft_tokens = draft_tokens_raw.reshape(num_gens, 2 * K - 1)[:, :K]
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(0, K)

        # logits have 2K entries per gen request; extract K+1 for acceptance
        if num_gens > 0:
            ctx_logits = logits[:num_contexts]
            vocab_size = logits.shape[-1]
            gen_logits_2k = logits[num_contexts:].reshape(num_gens, 2 * K, vocab_size)
            gen_logits_kp1 = gen_logits_2k[:, : K + 1, :].reshape(-1, vocab_size)
            logits_for_accept = torch.cat([ctx_logits, gen_logits_kp1], dim=0)
        else:
            logits_for_accept = logits

        accepted_tokens, num_accepted_tokens = self._sample_and_accept_draft_tokens_base(
            logits_for_accept, draft_tokens, num_contexts, batch_size, spec_metadata
        )

        # Update GDN/Mamba recurrent states to the accepted token's state.
        if num_gens > 0 and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager):
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        # Pad accepted_tokens from (batch, K+1) to (batch, 2K) to match sampler buffer
        if K > 1:
            acc_padding = torch.zeros(
                (batch_size, K - 1), dtype=accepted_tokens.dtype, device=accepted_tokens.device
            )
            accepted_tokens = torch.cat([accepted_tokens, acc_padding], dim=1)

        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        self._prepare_kv_for_draft_forward(
            attn_metadata, num_accepted_tokens, num_contexts, batch_size
        )

        # Collapse mrope [3, 1, N] to 1D by taking the first (temporal) dimension.
        # The draft model uses standard 1D RoPE, so only scalar positions are needed.
        if position_ids.ndim == 3:
            position_ids = position_ids[0, 0]
        else:
            position_ids = position_ids.squeeze(0)

        # Get total tokens processed by target model (for hidden state extraction)
        total_target_tokens = input_ids.shape[0]

        # Capture prefill (context) hidden states for future gen steps.
        # This gives the draft model the full prompt context, not just gen tokens.
        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            # Rebuild batch_to_slot after prefill assigns new slots
            if self._ctx_buf_inited and spec_metadata.request_ids:
                num_seqs = len(spec_metadata.request_ids)
                mapping = [self._req_to_slot.get(rid, 0) for rid in spec_metadata.request_ids]
                self._batch_to_slot[:num_seqs].copy_(
                    torch.tensor(mapping, dtype=torch.long, device="cuda")
                )

        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
            total_target_tokens=total_target_tokens,
        )

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        if num_gens > 0:
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                # Use custom dflash_forward with cross-attention.
                # ctx_k_cache / ctx_v_cache (when populated) let the drafter
                # skip per-layer context-KV projection/norm/rope entirely
                # and just read from the persistent per-layer buffers.
                hidden_states_out = draft_model.dflash_forward(
                    noise_embedding=inputs["noise_embedding"],
                    target_hidden=inputs["target_hidden"],
                    query_positions=inputs["query_positions"],
                    context_positions=inputs["context_positions"],
                    num_ctx_per_req=inputs["num_ctx_per_req"],
                    ctx_k_cache=inputs.get("ctx_k_cache"),
                    ctx_v_cache=inputs.get("ctx_v_cache"),
                    ctx_block_table=inputs.get("ctx_block_table"),
                )

                # Gather K logits per gen request from mask positions (1..K).
                # hidden_states_out is flat: [num_gens * block_size, hidden_dim]
                block_size = self._resolved_block_size
                request_bases = torch.arange(num_gens, dtype=torch.long, device="cuda") * block_size
                offsets = torch.arange(K, dtype=torch.long, device="cuda")
                # Masks are at positions 1..K in each request's block_size output
                gen_gather_ids = (request_bases.unsqueeze(1) + 1 + offsets.unsqueeze(0)).flatten()
                gen_gather_ids = gen_gather_ids.clamp(max=hidden_states_out.shape[0] - 1)

                gen_logits = draft_model.logits_processor(
                    hidden_states_out[gen_gather_ids], draft_model.lm_head, attn_metadata, True
                )

                vocab_size = gen_logits.shape[-1]
                gen_logits = gen_logits.reshape(num_gens, self.max_draft_len, vocab_size)

                d2t = getattr(draft_model.model, "d2t", None)
                gen_draft_tokens = torch.argmax(gen_logits, dim=-1, keepdim=False).long()

                if d2t is not None:
                    gen_draft_tokens = d2t[gen_draft_tokens] + gen_draft_tokens

                gen_draft_tokens = gen_draft_tokens.type(torch.int32)

                # Pad from (num_gens, K) to (num_gens, 2K-1).
                if K > 1:
                    pad = torch.zeros((num_gens, K - 1), dtype=torch.int32, device="cuda")
                    gen_draft_tokens = torch.cat([gen_draft_tokens, pad], dim=1)

        else:
            gen_draft_tokens = torch.empty((0, 2 * K - 1), dtype=torch.int32, device="cuda")

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros(
                (num_contexts, 2 * K - 1), dtype=torch.int32, device="cuda"
            )
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros(
                (num_contexts, 2 * K - 1), dtype=torch.int32, device="cuda"
            )
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        # Restore context lengths after warmup
        if is_warmup:
            self._ctx_len.copy_(saved_ctx_len)

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int = 0,
    ):
        """
        Prepare inputs for DFlash draft model with proper cross-attention.

        For gen requests, builds:
        - noise_embedding: token embeddings for [accepted_tokens + mask_tokens] (2K per req)
        - target_hidden: projected target features at accepted positions (context for cross-attn)
        - query_positions: position IDs for all 2K query tokens
        - context_positions: position IDs for the accepted (context) tokens
        - num_ctx_per_req: per-request context token counts

        The target_hidden stays CONSTANT across layers and provides K/V context.
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        # Resolve mask_token_id and block_size once, cache for subsequent calls
        if self._resolved_mask_token_id is None:
            if (
                hasattr(self.spec_config, "mask_token_id")
                and self.spec_config.mask_token_id is not None
            ):
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            elif hasattr(draft_model.model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.model.mask_token_id
            else:
                raise ValueError(
                    "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                    "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
                )
        mask_token_id = self._resolved_mask_token_id

        if self._resolved_block_size is None:
            # Default to the draft model's trained block_size, but cap at
            # K+1 (minimum needed for K draft predictions + 1 bonus) when
            # the user opts in. The trained block_size may be larger than
            # K+1 — e.g. Qwen3-8B-DFlash-b16 uses 16 but we only read back
            # K predictions, so the extra mask tokens are wasted compute.
            trained_block_size = getattr(draft_model, "block_size", None) or (
                self.max_draft_len + 1
            )
            override = getattr(self.spec_config, "block_size", None)
            if override is not None:
                self._resolved_block_size = int(override)
            else:
                self._resolved_block_size = trained_block_size
            logger.info(
                f"DFlash: resolved block_size={self._resolved_block_size} "
                f"(trained={trained_block_size}, K={self.max_draft_len})"
            )

        # Get the embed_tokens layer from the draft model
        embed_tokens = draft_model.draft_model_full.model.embed_tokens
        hidden_dim = (
            spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
        )

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # 2K
            K = self.max_draft_len

            # Get captured multi-layer hidden states from spec_metadata
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            has_target_features = (
                captured_hs is not None
                and hasattr(draft_model, "fc")
                and hasattr(draft_model, "hidden_norm")
            )

            # Use cached block_size (resolved once on first call)
            block_size = self._resolved_block_size
            query_tokens_per_req = block_size

            # Get slots for gen requests from pre-computed mapping
            slots = self._batch_to_slot[num_contexts : num_contexts + num_gens]

            # Fused build of input_ids + query_positions + ctx_positions in
            # one Triton kernel. Replaces the expand+clone+gather+scatter
            # pattern and the two arange-broadcast-add chains with a single
            # per-gen-request launch.
            K_plus_1 = K + 1
            input_ids_2d = torch.empty(
                (num_gens, query_tokens_per_req), dtype=torch.long, device="cuda"
            )
            query_position_ids = torch.empty(
                (num_gens, query_tokens_per_req), dtype=torch.long, device="cuda"
            )
            ctx_position_ids = torch.empty((num_gens, K_plus_1), dtype=torch.long, device="cuda")
            _build_dflash_query_inputs_kernel[(num_gens,)](
                gen_accepted_tokens,
                gen_accepted_tokens.stride(0),
                gen_num_accepted,
                self._ctx_len,
                slots,
                input_ids_2d,
                query_position_ids,
                ctx_position_ids,
                mask_token_id=int(mask_token_id),
                block_size=query_tokens_per_req,
                K_plus_1=K_plus_1,
                BLOCK=triton.next_power_of_2(max(query_tokens_per_req, K_plus_1)),
            )
            # One embed_tokens call replaces the expand+clone+per-position
            # overwrite; also avoids caching a mask embedding tensor.
            noise_embed_2d = embed_tokens(input_ids_2d.view(-1)).view(
                num_gens, query_tokens_per_req, hidden_dim
            )

            offsets_kp1 = torch.arange(K + 1, dtype=torch.long, device="cuda")

            # Accumulate new accepted features into context buffers
            if has_target_features:
                gen_start = attn_metadata.num_ctx_tokens
                gen_hs = captured_hs[gen_start : gen_start + num_gens * total_tokens_per_req]
                gen_hs = gen_hs.reshape(num_gens, total_tokens_per_req, -1)

                # Only project K+1 tokens (the accepted ones), not all 2K
                gen_hs_to_project = gen_hs[:, : K + 1, :].reshape(-1, gen_hs.shape[-1])
                projected_to_store = draft_model.fc(
                    gen_hs_to_project.to(draft_model.fc.weight.dtype)
                )
                projected_to_store = draft_model.hidden_norm(projected_to_store)
                gen_num_accepted_long = gen_num_accepted.long()
                col_idx = self._ctx_len[slots].unsqueeze(1) + offsets_kp1.unsqueeze(0)
                write_mask = offsets_kp1.unsqueeze(0) < gen_num_accepted_long.unsqueeze(1)
                col_idx = col_idx.clamp(max=self._max_ctx - 1)

                # Fixed-size writes for CUDA graph compatibility:
                # Write ALL entries but zero out invalid ones. Invalid
                # writes land at clamped column indices beyond valid
                # _ctx_len range, so they're harmless.
                slot_flat = slots.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                col_flat = col_idx.reshape(-1)
                proj_flat = projected_to_store  # already [num_gens*(K+1), proj_dim]
                pos_flat = ctx_position_ids.long().reshape(-1)
                mask_1d = write_mask.reshape(-1)

                if self._ctx_k_pool is not None:
                    # Fast path: store only the pre-projected/pre-RoPE'd K/V
                    # into the paged pool. dflash_forward reads the pool in
                    # place via block_table; no gather, no per-request
                    # hidden-state scatter.
                    k_new, v_new = draft_model.precompute_context_kv(
                        proj_flat.to(self._ctx_k_pool.dtype), pos_flat
                    )  # [num_gens*(K+1), L, nkv, hd]
                    mask_bc = mask_1d.view(-1, 1, 1, 1).to(k_new.dtype)
                    k_new = k_new * mask_bc
                    v_new = v_new * mask_bc
                    # Map each (slot, col) to (block_id, offset) on the
                    # paged pool. col_flat is already clamped to _max_ctx-1
                    # which is inside the last pre-allocated block for every
                    # slot, so no OOB risk on masked/invalid entries.
                    page = self._page_block_size
                    block_idx_in_req = (col_flat // page).clamp(max=self._max_blocks_per_req - 1)
                    offsets = col_flat % page
                    block_ids = self._slot_block_table[
                        slot_flat.long(), block_idx_in_req.long()
                    ].long()
                    # pool: [L, num_blocks, page, nkv, hd]; write layer-major.
                    k_ln = k_new.permute(1, 0, 2, 3).contiguous()
                    v_ln = v_new.permute(1, 0, 2, 3).contiguous()
                    self._ctx_k_pool[:, block_ids, offsets] = k_ln
                    self._ctx_v_pool[:, block_ids, offsets] = v_ln
                else:
                    # Fallback path: dflash_forward will recompute K/V from
                    # target_hidden each layer, so we must persist the
                    # projected hidden states and positions.
                    proj_masked = proj_flat * mask_1d.unsqueeze(1).to(proj_flat.dtype)
                    pos_masked = pos_flat * mask_1d.long()
                    self._ctx_buf[slot_flat, col_flat] = proj_masked.to(self._ctx_buf.dtype)
                    self._ctx_pos_buf[slot_flat, col_flat] = pos_masked

                self._ctx_len[slots] += gen_num_accepted_long
                self._ctx_len.clamp_(max=self._max_ctx)

            # Read padded context from buffers. Fast path uses the paged
            # pool + per-slot block_table; flash_attn reads each layer in
            # place. Fallback path still gathers the projected hidden
            # states + positions for per-layer recomputation.
            has_ctx_kv_cache = self._ctx_k_pool is not None
            num_ctx_per_req_t = self._ctx_len[slots]
            if has_ctx_kv_cache:
                target_hidden = self._ctx_buf.new_empty(num_gens, 0, self._proj_dim)
                context_positions = self._ctx_pos_buf.new_empty(num_gens, 0)
                ctx_k_cache = self._ctx_k_pool
                ctx_v_cache = self._ctx_v_pool
                # Per-forward block_table gather: [B, max_blocks_per_req]
                # int32. Tiny (a few KB even at bs=64) compared to the
                # GB-scale gather the old cache_batch_idx mode replaced.
                ctx_block_table = self._slot_block_table[slots.long()]
            else:
                target_hidden = self._ctx_buf[slots]
                context_positions = self._ctx_pos_buf[slots].long()
                ctx_k_cache = None
                ctx_v_cache = None
                ctx_block_table = None

            # 3D padded tensors for CUDA graph compatible dflash_forward
            noise_embedding = noise_embed_2d
            query_positions = query_position_ids.long()

            # Update seq_lens for gen requests to 2K
            attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = (
                total_tokens_per_req
            )
            attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        else:
            noise_embedding = hidden_states.new_empty(0, 0, hidden_dim)
            target_hidden = hidden_states.new_empty(0, 0, hidden_dim)
            query_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
            context_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
            num_ctx_per_req_t = torch.empty(0, dtype=torch.long, device="cuda")
            ctx_k_cache = None
            ctx_v_cache = None
            ctx_block_table = None

        return {
            "noise_embedding": noise_embedding,
            "target_hidden": target_hidden,
            "query_positions": query_positions,
            "context_positions": context_positions,
            "num_ctx_per_req": num_ctx_per_req_t,
            "ctx_k_cache": ctx_k_cache,
            "ctx_v_cache": ctx_v_cache,
            "ctx_block_table": ctx_block_table,
        }
