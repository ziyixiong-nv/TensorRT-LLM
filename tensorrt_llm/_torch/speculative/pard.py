from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import PARDDecodingConfig


@dataclass
class PARDSpecMetadata(SpecMetadata):
    """Metadata for PARD speculative decoding."""

    batch_indices_cuda: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

    def prepare(self):
        assert self.request_ids is not None

        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(num_seqs, dtype=torch.int, device="cpu", pin_memory=True)
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)


class PARDWorker(SpecWorkerBase):
    """
    Worker for PARD (PARallel Draft) speculative decoding.

    PARD is a target-independent speculative decoding method that uses
    mask tokens to predict multiple draft tokens in parallel within a
    single forward pass.

    Key differences from EAGLE3:
    1. Target-Independence: PARD doesn't use target model hidden states.
       The draft model relies only on its own embeddings and mask tokens.

    2. Parallel Prediction: All K draft tokens are predicted in ONE forward
       pass using mask tokens, rather than autoregressive generation.
       Input: [last_accepted_token, mask_0, mask_1, ..., mask_(K-1)]
       Output: K draft tokens predicted from K positions in parallel.

    3. Shared Mask Token: Uses the same mask_token_id across all positions
       to enable better generalization and extrapolation capability.

    Reference: https://arxiv.org/pdf/2504.18583
    """

    def __init__(
        self,
        spec_config: "PARDDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        logger.info(
            f"PARDWorker initialized with use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _prepare_attn_metadata_for_pard(self, attn_metadata, spec_metadata):
        """
        Save attn_metadata fields that PARD modifies during forward.

        CUDA graph handling: During warmup, save and restore kv_lens_cuda to
        prevent accumulation. During capture, don't save kv_lens_cuda so the
        modification is captured but restoration is not — ensuring the
        modification persists correctly during replay.
        """
        is_capturing = torch.cuda.is_current_stream_capturing()

        if spec_metadata.is_cuda_graph and not is_capturing:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _adjust_kv_cache_for_accepted_tokens(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """
        Rewind KV cache by K to remove draft token entries after target verification.

        The accepted tokens will be re-fed in prepare_1st_drafter_inputs.
        After target forward: kv_lens = old_kv_lens + (K+1)
        After adjustment:     kv_lens = old_kv_lens + 1
        """
        if hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= self.max_draft_len
            attn_metadata.kv_lens_cuda[num_contexts:batch_size].clamp_(min=0)
            if num_contexts > 0:
                attn_metadata.kv_lens_cuda[:num_contexts] += 1
            attn_metadata.update_for_spec_dec()

    # Skip torch.compile for now since current Torch is not compatible with Triton 3.4
    # @torch.compile(options={"max-autotune": True})
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

        self._execute_guided_decoder_if_present(logits)

        draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, self.max_draft_len)
        accepted_tokens, num_accepted_tokens = self._sample_and_accept_draft_tokens_base(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )

        self._prepare_attn_metadata_for_pard(attn_metadata, spec_metadata)
        self._adjust_kv_cache_for_accepted_tokens(
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

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        if num_gens > 0:
            logger.debug(
                f"PARD: draft_kv_cache_manager is {'None' if draft_kv_cache_manager is None else 'set'}"
            )
            logger.debug(f"PARD: use_separate_draft_kv_cache={self.use_separate_draft_kv_cache}")
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                hidden_states_out = draft_model.model(**inputs)

                # Gather logits from K positions per generation request.
                # With N accepted tokens in the previous iteration, we gather from
                # positions [N-1, N, ..., N+K-2] (capped at K) to get K draft predictions.
                gen_start_idx = attn_metadata.num_ctx_tokens

                request_bases = (
                    torch.arange(num_gens, dtype=torch.long, device="cuda")
                    * (self.max_draft_len + 1)
                    + gen_start_idx
                )

                gen_num_accepted = num_accepted_tokens[num_contexts:batch_size].long()
                base_offsets = (gen_num_accepted - 1).clamp(min=0)
                offsets = torch.arange(self.max_draft_len, dtype=torch.long, device="cuda")

                gen_gather_offsets = (base_offsets.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
                    max=self.max_draft_len
                )

                gen_gather_ids = (request_bases.unsqueeze(1) + gen_gather_offsets).flatten()
                gen_gather_ids = gen_gather_ids.clamp(max=hidden_states_out.shape[0] - 1)

                gen_logits = draft_model.logits_processor(
                    hidden_states_out[gen_gather_ids], draft_model.lm_head, attn_metadata, True
                )

                vocab_size = gen_logits.shape[-1]
                gen_logits = gen_logits.reshape(num_gens, self.max_draft_len, vocab_size)

                # Use torch.argmax directly to avoid cute_argmax stride issues
                d2t = getattr(draft_model.model, "d2t", None)
                gen_draft_tokens = torch.argmax(gen_logits, dim=-1, keepdim=False).long()

                if d2t is not None:
                    gen_draft_tokens = d2t[gen_draft_tokens] + gen_draft_tokens

                gen_draft_tokens = gen_draft_tokens.type(torch.int32)

        elif num_contexts > 0 and self.use_separate_draft_kv_cache:
            # Pure context batch: populate the draft KV cache so it's
            # ready when generation starts.
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                draft_model.model(**inputs)
            gen_draft_tokens = torch.empty(
                (0, self.max_draft_len), dtype=torch.int32, device="cuda"
            )

        else:
            gen_draft_tokens = torch.empty(
                (0, self.max_draft_len), dtype=torch.int32, device="cuda"
            )

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros(
                (num_contexts, self.max_draft_len), dtype=torch.int32, device="cuda"
            )
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros(
                (num_contexts, self.max_draft_len), dtype=torch.int32, device="cuda"
            )
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)

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
        """
        Sampling draft tokens with support for non-greedy sampling.

        Args:
            logits: torch.Tensor
                [num_tokens, vocab_size]
                Logits produced by the draft model.
            draft_model: nn.Module
                The draft model.

        Returns:
            draft_tokens: torch.Tensor
                [batch_size * max_draft_len]
                Draft token ids. Flattened.
        """

        # Note: using greedy for draft tokens is a bit easier to implement and
        # faster. It doesn't affect the final output and seems to have a negligible
        # impact on AR.
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
        spec_metadata: PARDSpecMetadata,
        draft_model: nn.Module,
    ):
        """
        Prepare inputs for PARD draft model.

        For generation requests, constructs input_ids as:
        [accepted_tokens, mask_tokens] to fill K+1 slots per request.
        """
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
                "PARD requires mask_token_id to be set. Please set it in PARDDecodingConfig "
                "or ensure the draft model config has 'pard_token' or 'mask_token_id'."
            )

        if num_contexts > 0:
            input_ids_ctx = self._prepare_context_input_ids(
                input_ids,
                attn_metadata.num_ctx_tokens,
                spec_metadata.gather_ids,
                accepted_tokens,
                num_contexts,
            )
            position_ids_ctx = position_ids[: attn_metadata.num_ctx_tokens]
        else:
            input_ids_ctx = torch.empty(0, dtype=torch.int32, device="cuda")
            position_ids_ctx = torch.empty(0, dtype=torch.int32, device="cuda")

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self.max_draft_len + 1

            # Position j uses accepted token if j < num_accepted[i], else mask token
            col_indices = torch.arange(total_tokens_per_req, dtype=torch.int32, device="cuda")
            use_accepted_mask = col_indices.unsqueeze(0) < gen_num_accepted.unsqueeze(1)

            gen_accepted_int32 = gen_accepted_tokens[:, :total_tokens_per_req].to(dtype=torch.int32)
            request_ids_2d = torch.where(
                use_accepted_mask,
                gen_accepted_int32,
                torch.full_like(gen_accepted_int32, mask_token_id),
            )

            input_ids_gen = request_ids_2d.flatten()

            # Position IDs start from kv_lens_cuda (post-rewind) for each request
            if hasattr(attn_metadata, "kv_lens_cuda"):
                gen_pos_starts = attn_metadata.kv_lens_cuda[
                    num_contexts : num_contexts + num_gens
                ].int()
            else:
                gen_pos_starts = position_ids[
                    attn_metadata.num_ctx_tokens :: self.max_draft_len + 1
                ][:num_gens]

            offsets = torch.arange(self.max_draft_len + 1, dtype=torch.int32, device="cuda")
            position_ids_gen = (gen_pos_starts.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
        else:
            input_ids_gen = torch.empty(0, dtype=torch.int32, device="cuda")
            position_ids_gen = torch.empty(0, dtype=torch.int32, device="cuda")

        input_ids_final = torch.cat([input_ids_ctx, input_ids_gen], dim=0)
        position_ids_final = torch.cat([position_ids_ctx, position_ids_gen], dim=0).int()

        return {
            "input_ids": input_ids_final,
            "position_ids": position_ids_final,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
