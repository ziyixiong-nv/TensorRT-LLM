from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch
from torch import nn

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import DFlashDecodingConfig

_DEFAULT_MAX_CTX_LEN = 2048

# Bridges ctx-only and gen-only metadata instances under CUDA graphs:
# ctx stores prefill projections, gen consumes them during prepare().
_shared_prefill_proj_store: Dict[int, torch.Tensor] = {}


@dataclass
class DFlashSpecMetadata(SpecMetadata):
    """Metadata for DFlash speculative decoding.

    Captures hidden states from specific target model layers and manages
    per-request accumulated projected target hidden states across
    generation steps.
    """

    hidden_states: Optional[torch.Tensor] = None
    layers_to_capture: Optional[Set[int]] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    batch_indices_cuda: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.layers_to_capture is None:
            self.layers_to_capture = ()
        else:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
        self.num_capture_layers = len(self.layers_to_capture)

        logger.debug(
            f"DFlashSpecMetadata: layers_to_capture={self.layers_to_capture}, "
            f"num_capture_layers={self.num_capture_layers}, hidden_size={self.hidden_size}, "
            f"max_num_tokens={self.max_num_tokens}"
        )

        if self.num_capture_layers > 0:
            self.hidden_states = torch.zeros(
                (self.max_num_tokens, self.hidden_size * self.num_capture_layers),
                dtype=self.dtype,
                device="cuda",
            )

        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

        K = self.max_draft_len
        max_acc_cols = 2 * K
        self.max_ctx_len = _DEFAULT_MAX_CTX_LEN
        self._max_acc_cols = max_acc_cols
        total_target_cols = self.max_ctx_len + max_acc_cols

        self.proj_cache = torch.zeros(
            (self.max_num_requests, self.max_ctx_len, self.hidden_size),
            dtype=self.dtype,
            device="cuda",
        )
        self.proj_cache_lens = torch.zeros(
            self.max_num_requests,
            dtype=torch.int32,
            device="cuda",
        )

        self.full_target_proj = torch.zeros(
            (self.max_num_requests, total_target_cols, self.hidden_size),
            dtype=self.dtype,
            device="cuda",
        )
        self.full_target_pos_ids = torch.zeros(
            (self.max_num_requests, total_target_cols),
            dtype=torch.int32,
            device="cuda",
        )

        self.historical_lens = torch.zeros(
            self.max_num_requests,
            dtype=torch.int32,
            device="cuda",
        )

        self._current_proj_buffer = torch.zeros(
            (self.max_num_requests, max_acc_cols, self.hidden_size),
            dtype=self.dtype,
            device="cuda",
        )
        self._current_num_acc = torch.zeros(
            self.max_num_requests,
            dtype=torch.int32,
            device="cuda",
        )

        self._rid_to_slot: Dict[int, int] = {}
        self._free_slots: List[int] = list(range(self.max_num_requests - 1, -1, -1))
        self._prev_gen_rids: List[int] = []

        self._ar_total_accepted = 0
        self._ar_total_steps = 0
        self._ar_log_interval = 20
        self._ar_warmup_done = False

    def is_layer_capture(self, layer_id: int):
        return layer_id in self.layers_to_capture

    def maybe_capture_hidden_states(
        self, layer_id: int, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> None:
        for i, captured_layer_id in enumerate(self.layers_to_capture):
            if captured_layer_id == layer_id:
                num_tokens = hidden_states.shape[0]
                to_save = hidden_states + residual if residual is not None else hidden_states
                self.hidden_states[
                    :num_tokens, i * self.hidden_size : (i + 1) * self.hidden_size
                ].copy_(to_save, non_blocking=True)
                break

    def prepare(self):
        assert self.request_ids is not None

        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        num_contexts = num_seqs - self.num_generations
        current_gen_rids = self.request_ids[num_contexts:num_seqs]
        current_gen_rid_set = set(current_gen_rids)

        for rid in list(self._rid_to_slot.keys()):
            if rid not in current_gen_rid_set:
                slot = self._rid_to_slot.pop(rid)
                self._free_slots.append(slot)
                self.proj_cache_lens[slot] = 0

        for rid in current_gen_rids:
            if rid not in self._rid_to_slot:
                slot = self._free_slots.pop()
                self._rid_to_slot[rid] = slot
                if rid in _shared_prefill_proj_store:
                    prefill = _shared_prefill_proj_store.pop(rid)
                    length = min(prefill.shape[0], self.max_ctx_len)
                    self.proj_cache[slot, :length].copy_(prefill[:length], non_blocking=True)
                    self.proj_cache_lens[slot] = length
                else:
                    self.proj_cache_lens[slot] = 0

        # Clean stale prefill projections (only from gen-capable instances).
        if _shared_prefill_proj_store and self.num_generations > 0:
            active_rids = set(self.request_ids)
            stale = [r for r in _shared_prefill_proj_store if r not in active_rids]
            for r in stale:
                del _shared_prefill_proj_store[r]

        if not self._ar_warmup_done and self._prev_gen_rids:
            if set(self._prev_gen_rids) & current_gen_rid_set:
                self._ar_warmup_done = True
                self._ar_total_accepted = 0
                self._ar_total_steps = 0

        for prev_idx, prev_rid in enumerate(self._prev_gen_rids):
            if prev_rid in self._rid_to_slot:
                slot = self._rid_to_slot[prev_rid]
                curr_len = self.proj_cache_lens[slot].item()
                num_acc = self._current_num_acc[prev_idx].item()
                new_len = min(curr_len + num_acc, self.max_ctx_len)
                copy_len = new_len - curr_len
                if copy_len > 0:
                    self.proj_cache[slot, curr_len:new_len].copy_(
                        self._current_proj_buffer[prev_idx, :copy_len],
                        non_blocking=True,
                    )
                    self.proj_cache_lens[slot] = new_len

                if self._ar_warmup_done:
                    self._ar_total_accepted += num_acc
                    self._ar_total_steps += 1

        num_gens = self.num_generations
        for i, rid in enumerate(current_gen_rids):
            slot = self._rid_to_slot[rid]
            length = self.proj_cache_lens[slot].item()

            if length > 0:
                self.full_target_proj[i, :length].copy_(
                    self.proj_cache[slot, :length], non_blocking=True
                )
            if length < self.max_ctx_len:
                self.full_target_proj[i, length : self.max_ctx_len].zero_()

            self.historical_lens[i] = length

        if self._ar_total_steps > 0 and self._ar_total_steps % self._ar_log_interval == 0:
            avg_al = self._ar_total_accepted / self._ar_total_steps
            sample_hl = self.historical_lens[0].item() if num_gens > 0 else -1
            logger.info(
                f"DFlash AR: avg_acceptance_length={avg_al:.2f} "
                f"over {self._ar_total_steps} steps "
                f"(sample hist_lens={sample_hl})"
            )

        self._prev_gen_rids = list(current_gen_rids)


class DFlashWorker(SpecWorkerBase):
    """Worker for DFlash (Block Diffusion Flash) speculative decoding.

    Reference: https://arxiv.org/abs/2602.06036
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
        logger.info(
            f"DFlashWorker initialized with max_draft_len={spec_config.max_draft_len}, "
            f"use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    @property
    def _draft_tokens_per_req(self) -> int:
        return 2 * self.max_draft_len

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

        self._execute_guided_decoder_if_present(logits)

        if num_gens > 0:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, 2 * K - 1)[:, :K]
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(0, K)

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

        if K > 1:
            acc_padding = torch.zeros(
                (batch_size, K - 1), dtype=accepted_tokens.dtype, device=accepted_tokens.device
            )
            accepted_tokens = torch.cat([accepted_tokens, acc_padding], dim=1)

        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        self._prepare_kv_for_draft_forward(
            attn_metadata, num_accepted_tokens, num_contexts, batch_size
        )

        position_ids = position_ids.squeeze(0)
        inputs = self.prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
        )

        if num_gens > 0:
            noise_embedding_gen = inputs["noise_embedding_gen"]
            noise_pos_gen = inputs["noise_position_ids_gen"]
            target_proj_gen = inputs["target_hidden_proj_gen"]
            target_pos_gen = inputs["target_position_ids_gen"]
            gen_num_accepted_for_ctx = inputs["gen_num_accepted"]
            noise_block_size = noise_embedding_gen.shape[1]

            hidden_states_out = draft_model.forward_dflash(
                noise_embedding=noise_embedding_gen,
                target_hidden_proj=target_proj_gen,
                noise_position_ids=noise_pos_gen,
                target_position_ids=target_pos_gen,
                num_target_tokens=gen_num_accepted_for_ctx,
            )

            # Gather K logits per gen request from mask positions 1..K
            offsets = torch.arange(num_gens, device="cuda", dtype=torch.long) * noise_block_size
            base_positions = offsets + 1
            gather_deltas = torch.arange(K, device="cuda", dtype=torch.long)
            gen_gather_ids = (base_positions.unsqueeze(1) + gather_deltas.unsqueeze(0)).reshape(-1)
            max_idx = num_gens * noise_block_size - 1
            gen_gather_ids = gen_gather_ids.clamp(min=0, max=max_idx)

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

            if K > 1:
                pad = torch.zeros((num_gens, K - 1), dtype=torch.int32, device="cuda")
                gen_draft_tokens = torch.cat([gen_draft_tokens, pad], dim=1)

        else:
            gen_draft_tokens = torch.zeros((0, 2 * K - 1), dtype=torch.int32, device="cuda")

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

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def draft_decoder(
        self,
        logits: torch.Tensor,
        draft_model: nn.Module,
    ):
        d2t = getattr(draft_model.model, "d2t", None)
        return self._draft_sampler_greedy(logits, d2t)

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
    ):
        """Prepare inputs for DFlash draft model with accumulated target context."""
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        if (
            hasattr(self.spec_config, "mask_token_id")
            and self.spec_config.mask_token_id is not None
        ):
            mask_token_id = self.spec_config.mask_token_id
        elif hasattr(draft_model, "mask_token_id"):
            mask_token_id = draft_model.mask_token_id
        elif hasattr(draft_model.model, "mask_token_id"):
            mask_token_id = draft_model.model.mask_token_id
        else:
            raise ValueError(
                "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                "or ensure the draft model config has 'mask_token_id'."
            )

        embed_tokens = draft_model.model.embed_tokens

        if num_contexts > 0:
            input_ids_ctx = input_ids[: attn_metadata.num_ctx_tokens].to(torch.int32).clone()
            inputs_embeds_ctx = embed_tokens(input_ids_ctx.long())
            position_ids_ctx = position_ids[: attn_metadata.num_ctx_tokens]

            # Project prefill hidden states and store for gen phase.
            num_ctx_tokens_val = attn_metadata.num_ctx_tokens
            _num_cap = spec_metadata.num_capture_layers
            _h = spec_metadata.hidden_size
            _hup = _num_cap * _h
            if num_ctx_tokens_val > 0 and _hup > 0:
                ctx_captured = spec_metadata.hidden_states[:num_ctx_tokens_val, :_hup]
                ctx_proj_all = draft_model.hidden_norm(draft_model.fc(ctx_captured))
                offset = 0
                for ci in range(num_contexts):
                    rid = spec_metadata.request_ids[ci]
                    seq_len = int(attn_metadata._seq_lens[ci])
                    _shared_prefill_proj_store[rid] = (
                        ctx_proj_all[offset : offset + seq_len].detach().clone()
                    )
                    offset += seq_len
        else:
            inputs_embeds_ctx = torch.empty(
                0, draft_model.config.hidden_size, dtype=hidden_states.dtype, device="cuda"
            )
            position_ids_ctx = torch.empty(0, dtype=torch.int32, device="cuda")

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # 2K
            num_capture_layers = spec_metadata.num_capture_layers
            h_size = spec_metadata.hidden_size
            hidden_size_up = num_capture_layers * h_size
            hidden_size = h_size

            # Cap noise block at model's training block_size.
            if not hasattr(self, "_noise_block_size"):
                model_block_size = getattr(draft_model.config, "block_size", None)
                if model_block_size is not None and model_block_size < total_tokens_per_req:
                    self._noise_block_size = model_block_size
                else:
                    self._noise_block_size = total_tokens_per_req
                logger.info(
                    f"DFlash: noise_block_size={self._noise_block_size} "
                    f"(model_block_size={model_block_size}, 2K={total_tokens_per_req})"
                )
            noise_block_size = self._noise_block_size

            num_ctx_tokens = attn_metadata.num_ctx_tokens
            target_block_size = total_tokens_per_req
            gen_hidden_all = spec_metadata.hidden_states[
                num_ctx_tokens : num_ctx_tokens + num_gens * target_block_size, :hidden_size_up
            ].reshape(num_gens, target_block_size, hidden_size_up)

            max_acc_cols = gen_accepted_tokens.shape[1]
            mask_token_ids = torch.full(
                (num_gens, noise_block_size), mask_token_id, dtype=torch.long, device="cuda"
            )
            bonus_indices = (gen_num_accepted - 1).clamp(min=0).long()
            bonus_tokens = gen_accepted_tokens[
                torch.arange(num_gens, device="cuda"), bonus_indices
            ].long()
            mask_token_ids[:, 0] = bonus_tokens

            noise_embedding_gen = embed_tokens(mask_token_ids)

            gen_hidden_accepted = gen_hidden_all[:, :max_acc_cols, :]
            gen_hidden_accepted_flat = gen_hidden_accepted.reshape(-1, hidden_size_up)
            current_proj_flat = draft_model.hidden_norm(draft_model.fc(gen_hidden_accepted_flat))
            current_proj = current_proj_flat.reshape(num_gens, max_acc_cols, -1)

            spec_metadata._current_proj_buffer[:num_gens].copy_(current_proj, non_blocking=True)
            spec_metadata._current_num_acc[:num_gens].copy_(
                gen_num_accepted.int(), non_blocking=True
            )

            max_ctx_len = spec_metadata.max_ctx_len
            total_target_cols = max_ctx_len + max_acc_cols

            hist_lens = spec_metadata.historical_lens[:num_gens].long()
            gen_range = torch.arange(num_gens, device="cuda", dtype=torch.long)
            col_range = torch.arange(max_acc_cols, device="cuda", dtype=torch.long)

            write_positions = hist_lens.unsqueeze(1) + col_range.unsqueeze(0)
            valid_mask = col_range.unsqueeze(0) < gen_num_accepted.unsqueeze(1)

            dummy_col = total_target_cols - 1
            write_positions = write_positions.clamp(max=dummy_col)
            write_positions = torch.where(valid_mask, write_positions, dummy_col)

            flat_idx = gen_range.unsqueeze(1) * total_target_cols + write_positions
            flat_idx_expanded = flat_idx.reshape(-1, 1).expand(-1, hidden_size)
            current_proj_flat_for_scatter = current_proj.reshape(-1, hidden_size)

            full_tp_2d = spec_metadata.full_target_proj[:num_gens].reshape(-1, hidden_size)
            full_tp_2d.scatter_(0, flat_idx_expanded, current_proj_flat_for_scatter)

            gen_num_accepted_total = hist_lens + gen_num_accepted.long()
            target_proj = spec_metadata.full_target_proj[:num_gens]

            attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = (
                total_tokens_per_req
            )
            attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req

            target_offsets = torch.arange(total_target_cols, dtype=torch.int32, device="cuda")
            target_position_ids_gen = target_offsets.unsqueeze(0).expand(num_gens, -1)

            noise_offsets = torch.arange(noise_block_size, dtype=torch.int32, device="cuda")
            noise_position_ids_gen = gen_num_accepted_total.unsqueeze(
                1
            ).int() + noise_offsets.unsqueeze(0)
        else:
            max_acc_cols = spec_metadata._max_acc_cols
            total_target_cols = spec_metadata.max_ctx_len + max_acc_cols
            noise_embedding_gen = torch.empty(
                0, 0, draft_model.config.hidden_size, dtype=hidden_states.dtype, device="cuda"
            )
            noise_position_ids_gen = torch.empty(0, 0, dtype=torch.int32, device="cuda")
            target_proj = torch.empty(
                0,
                total_target_cols,
                draft_model.config.hidden_size,
                dtype=hidden_states.dtype,
                device="cuda",
            )
            target_position_ids_gen = torch.empty(
                0, total_target_cols, dtype=torch.int32, device="cuda"
            )
            gen_num_accepted_total = torch.empty(0, dtype=torch.long, device="cuda")
            position_ids_ctx = (
                position_ids[: attn_metadata.num_ctx_tokens]
                if num_contexts > 0
                else torch.empty(0, dtype=torch.int32, device="cuda")
            )

        return {
            "inputs_embeds_ctx": inputs_embeds_ctx,
            "position_ids_ctx": position_ids_ctx,
            "noise_embedding_gen": noise_embedding_gen,
            "noise_position_ids_gen": noise_position_ids_gen,
            "target_hidden_proj_gen": target_proj,
            "target_position_ids_gen": target_position_ids_gen,
            "gen_num_accepted": gen_num_accepted_total,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
