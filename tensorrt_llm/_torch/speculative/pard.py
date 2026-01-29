"""
PARD (Parallel Draft) speculative decoding implementation.

PARD is a parallel draft model adaptation method that generates multiple
draft tokens in parallel using special PARD tokens, enabling efficient
one-model speculative decoding.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tensorrt_llm.logger import logger

from ..pyexecutor.sampler import TorchSampler
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import PARDDecodingConfig


@dataclass
class PARDSpecMetadata(SpecMetadata):
    """
    Metadata for PARD speculative decoding.

    PARD generates draft tokens in parallel by using special unused tokens
    as placeholders, allowing the draft model to predict multiple tokens
    in a single forward pass.
    """

    def __post_init__(self):
        """Initialize buffers for CUDA graph compatibility."""
        # Draft tokens: [max_num_requests, max_draft_len]
        self.draft_tokens = torch.empty(
            (self.max_num_requests, self.max_draft_len), dtype=torch.int32, device="cuda"
        )

        # Draft token lengths: [max_num_requests]
        self.draft_lens = torch.empty(self.max_num_requests, dtype=torch.int32, device="cuda")

        # Number of accepted draft tokens: [max_num_requests]
        self.num_accepted_draft_tokens = torch.empty(
            self.max_num_requests, dtype=torch.int32, device="cuda"
        )

        # Batch indices for gather operations
        self.batch_indices_cuda = torch.arange(
            self.max_num_requests, dtype=torch.int64, device="cuda"
        )

        # Gather ids for extracting specific tokens from sequences
        self.gather_ids = torch.empty(self.max_num_requests, dtype=torch.int64, device="cuda")

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """
        pass


class PARDWorker(SpecWorkerBase):
    """
    Worker for PARD one-model speculative decoding.

    PARD generates draft tokens in parallel by injecting special unused tokens
    as placeholders into the draft model input. The draft model then predicts
    all draft tokens in a single forward pass.

    Key differences from MTP/EAGLE:
    - Generates all draft tokens in parallel (not sequentially)
    - Uses special PARD tokens as placeholders
    - Requires draft model trained with PARD training strategy
    """

    def __init__(
        self,
        spec_config: "PARDDecodingConfig",
        model_config=None,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.model_config = model_config

        # Get PARD token IDs from config
        if spec_config.pard_token_ids is not None:
            self.pard_token_ids = torch.tensor(
                spec_config.pard_token_ids[: self.max_draft_len], dtype=torch.int32, device="cuda"
            )
        else:
            # Default: use large token IDs that are unlikely to appear in normal text
            # These should be configured to match unused tokens in the vocabulary
            self.pard_token_ids = torch.arange(
                128000, 128000 + self.max_draft_len, dtype=torch.int32, device="cuda"
            )

        logger.info(f"PARD initialized with {self.max_draft_len} draft tokens")
        logger.info(f"PARD token IDs: {self.pard_token_ids.tolist()}")

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

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
        """
        PARD forward pass for one-model speculative decoding.

        Algorithm:
        1. Sample and verify draft tokens from previous iteration
        2. Generate new draft tokens in parallel using PARD tokens
        3. Return accepted tokens and new draft tokens for next iteration

        Example (max_draft_len=3):

        Context phase:
        - Target model: Input: "ABCD" -> Output logits for: E
        - Draft model: Input: "E[P1][P2][P3]" -> Output logits for: FGH (parallel)
        - Next iteration input: "E" + draft tokens "FGH"

        Generation phase:
        - Target model: Input: "E+FGH" -> Output logits for: FGXY
        - Accept: "FGX" (3 tokens accepted, H rejected)
        - Draft model: Input: "X[P1][P2][P3]" -> Output logits for: NOP (parallel)
        - Next iteration input: "X" + draft tokens "NOP"

        Args:
            input_ids: Input token IDs [num_tokens]
            position_ids: Position IDs [1, num_tokens]
            hidden_states: Hidden states from target model
            logits: Logits from target model [num_tokens, vocab_size]
            attn_metadata: Attention metadata
            spec_metadata: PARD speculative decoding metadata
            draft_model: Draft model for generating draft tokens
            resource_manager: Optional resource manager

        Returns:
            Dictionary with:
                - logits: Target model logits
                - new_tokens: Accepted tokens [batch_size, max_draft_len + 1]
                - new_tokens_lens: Number of accepted tokens per request [batch_size]
                - next_draft_tokens: Draft tokens for next iteration [batch_size, max_draft_len]
                - next_new_tokens: All tokens for next iteration [batch_size, max_draft_len + 1]
        """
        batch_size = attn_metadata.num_seqs
        # Execute guided decoder if present
        self._execute_guided_decoder_if_present(logits)

        # Sample and verify draft tokens from previous iteration
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        # Prepare attention metadata for draft model forward
        self._prepare_attn_metadata_for_spec_dec(attn_metadata)

        # Prepare inputs for draft model
        position_ids = position_ids.squeeze(0)
        draft_inputs = self.prepare_draft_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
        )

        # Generate draft tokens in parallel
        next_draft_tokens, draft_logits_list = self.generate_parallel_draft_tokens(
            draft_inputs=draft_inputs,
            draft_model=draft_model,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            batch_size=batch_size,
            resource_manager=resource_manager,
        )

        # Compute and store draft probs for rejection sampling if enabled
        if spec_metadata.use_rejection_sampling and draft_logits_list:
            self._compute_and_store_draft_probs(draft_logits_list, spec_metadata, batch_size)

        # Restore attention metadata
        self._restore_attn_metadata_from_spec_dec(attn_metadata)

        # Prepare next_new_tokens for overlap scheduler
        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens=accepted_tokens,
            next_draft_tokens=next_draft_tokens,
            batch_indices_cuda=spec_metadata.batch_indices_cuda,
            batch_size=batch_size,
            num_accepted_tokens=num_accepted_tokens,
        )

        return {
            "logits": logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata,
        spec_metadata: PARDSpecMetadata,
    ):
        """
        Sample tokens from target model logits and verify draft tokens.

        Uses either rejection sampling (if enabled) or greedy acceptance.

        Args:
            logits: Target model logits [num_tokens, vocab_size]
            attn_metadata: Attention metadata
            spec_metadata: PARD metadata containing draft tokens

        Returns:
            accepted_tokens: [batch_size, max_draft_len + 1]
            num_accepted_tokens: [batch_size]
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        # Get draft tokens from previous iteration
        draft_tokens = spec_metadata.draft_tokens[:num_gens]

        # Check if we can use rejection sampling
        if self._can_use_rejection_sampling(spec_metadata, num_contexts):
            # Reshape draft_probs from flat buffer
            vocab_size = spec_metadata.draft_probs_vocab_size
            draft_probs_flat = spec_metadata.draft_probs[
                : num_gens * self.max_draft_len * vocab_size
            ]
            draft_probs = draft_probs_flat.reshape(num_gens, self.max_draft_len, vocab_size)

            # Use rejection sampling
            return self._sample_and_accept_draft_tokens_rejection(
                logits=logits,
                draft_tokens=draft_tokens,
                draft_probs=draft_probs,
                batch_size=batch_size,
                spec_metadata=spec_metadata,
            )
        else:
            # Use greedy acceptance (base implementation)
            return self._sample_and_accept_draft_tokens_base(
                logits=logits,
                draft_tokens=draft_tokens,
                num_contexts=num_contexts,
                batch_size=batch_size,
                spec_metadata=spec_metadata,
            )

    def prepare_draft_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
    ):
        """
        Prepare inputs for parallel draft token generation.

        For PARD, we create a parallel input by concatenating:
        - The last accepted token
        - PARD placeholder tokens

        This allows the draft model to predict all draft tokens in parallel.

        Args:
            input_ids: Original input IDs
            position_ids: Original position IDs
            hidden_states: Hidden states from target model
            accepted_tokens: Accepted tokens from verification
            num_accepted_tokens: Number of accepted tokens per request
            attn_metadata: Attention metadata
            spec_metadata: PARD metadata

        Returns:
            Dictionary with draft model inputs
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        # Get the last accepted token for each request
        batch_indices = spec_metadata.batch_indices_cuda[:batch_size]
        last_accepted_idx = num_accepted_tokens - 1
        last_accepted_tokens = accepted_tokens[batch_indices, last_accepted_idx]

        # For context requests: use context input directly
        if num_contexts > 0:
            num_ctx_tokens = attn_metadata.num_ctx_tokens
            gather_ids = spec_metadata.gather_ids[:num_contexts]

            # Prepare context input: shift and append accepted token
            ctx_input_ids = self._prepare_context_input_ids(
                input_ids=input_ids,
                num_ctx_tokens=num_ctx_tokens,
                gather_ids=gather_ids,
                accepted_tokens=accepted_tokens,
                num_contexts=num_contexts,
            )
        else:
            ctx_input_ids = torch.empty(0, dtype=torch.int32, device="cuda")

        # For generation requests: create parallel input with PARD tokens
        if num_gens > 0:
            # Create parallel input: [last_token, PARD_token_1, ..., PARD_token_K]
            gen_last_tokens = last_accepted_tokens[num_contexts:].unsqueeze(1)
            pard_tokens = self.pard_token_ids.unsqueeze(0).expand(num_gens, -1)
            gen_input_ids = torch.cat([gen_last_tokens, pard_tokens], dim=1).flatten()
        else:
            gen_input_ids = torch.empty(0, dtype=torch.int32, device="cuda")

        # Concatenate context and generation inputs
        draft_input_ids = torch.cat([ctx_input_ids, gen_input_ids], dim=0)

        # Prepare position IDs
        # Context: continue from existing positions
        # Generation: each request has positions [pos, pos+1, ..., pos+K]
        if num_contexts > 0:
            ctx_position_ids = position_ids[:num_ctx_tokens]
        else:
            ctx_position_ids = torch.empty(0, dtype=torch.int64, device="cuda")

        if num_gens > 0:
            # Get position of last accepted token for each generation request
            gen_last_positions = position_ids[attn_metadata.num_ctx_tokens :][
                torch.cumsum(attn_metadata.seq_lens_cuda[num_contexts:], dim=0) - 1
            ]
            # Create parallel positions: [pos, pos+1, ..., pos+K]
            gen_position_offsets = torch.arange(
                self.max_draft_len + 1, dtype=torch.int64, device="cuda"
            )
            gen_position_ids = gen_last_positions.unsqueeze(1) + gen_position_offsets
            gen_position_ids = gen_position_ids.flatten()
        else:
            gen_position_ids = torch.empty(0, dtype=torch.int64, device="cuda")

        draft_position_ids = torch.cat([ctx_position_ids, gen_position_ids], dim=0)

        # Update sequence lengths in attention metadata
        # Context: keep original lengths
        # Generation: set to (max_draft_len + 1) for parallel input
        if num_gens > 0:
            attn_metadata._seq_lens[num_contexts:batch_size].fill_(self.max_draft_len + 1)
            attn_metadata._seq_lens_cuda[num_contexts:batch_size].fill_(self.max_draft_len + 1)
            attn_metadata.on_update()

        return {
            "input_ids": draft_input_ids,
            "position_ids": draft_position_ids,
            "hidden_states": None,  # PARD doesn't use hidden states from target
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def generate_parallel_draft_tokens(
        self,
        draft_inputs,
        draft_model,
        attn_metadata,
        spec_metadata,
        batch_size,
        resource_manager=None,
    ):
        """
        Generate draft tokens in parallel using PARD draft model.

        The draft model takes parallel input with PARD tokens and predicts
        all draft tokens in a single forward pass.

        Args:
            draft_inputs: Prepared draft model inputs
            draft_model: PARD draft model
            attn_metadata: Attention metadata
            spec_metadata: PARD metadata
            batch_size: Number of requests
            resource_manager: Optional resource manager

        Returns:
            next_draft_tokens: [batch_size, max_draft_len]
            draft_logits_list: List of draft logits for rejection sampling
        """
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        # Get draft KV cache manager if using separate layouts
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        # Forward through draft model
        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            # Run draft model forward
            hidden_states = draft_model.model(
                input_ids=draft_inputs["input_ids"],
                position_ids=draft_inputs["position_ids"],
                attn_metadata=draft_inputs["attn_metadata"],
                spec_metadata=draft_inputs["spec_metadata"],
            )

            # Get logits from draft model
            logits = draft_model.logits_processor(
                hidden_states,
                draft_model.lm_head,
                attn_metadata,
                True,  # return_logits
            ).float()

        # Extract draft token logits
        # For context: extract last token logits (no parallel generation yet)
        # For generation: extract all K draft token logits (positions 1 to K)
        next_draft_tokens = torch.empty(
            (batch_size, self.max_draft_len), dtype=torch.int32, device="cuda"
        )

        draft_logits_list = []

        if num_contexts > 0:
            # Context: only predict first draft token (no parallel yet)
            ctx_gather_ids = spec_metadata.gather_ids[:num_contexts]
            ctx_logits = logits[ctx_gather_ids]
            ctx_draft_token = self._draft_sampler_greedy(ctx_logits)

            # Fill all draft positions with same token for context (will be updated next iteration)
            next_draft_tokens[:num_contexts, :] = ctx_draft_token.unsqueeze(1).expand(
                -1, self.max_draft_len
            )

            if spec_metadata.use_rejection_sampling:
                draft_logits_list.append(ctx_logits.clone())

        if num_gens > 0:
            # Generation: extract parallel draft tokens (positions 1 to K)
            # The logits are arranged as: [req0_pos1, ..., req0_posK, req1_pos1, ...]
            gen_start_idx = attn_metadata.num_ctx_tokens if num_contexts > 0 else 0
            gen_logits = logits[gen_start_idx:].reshape(num_gens, self.max_draft_len + 1, -1)

            # Extract draft token logits (positions 1 to K, skip position 0)
            gen_draft_logits = gen_logits[:, 1:, :]

            # Sample draft tokens
            if spec_metadata.use_rejection_sampling:
                # Store logits for rejection sampling
                for k in range(self.max_draft_len):
                    draft_logits_list.append(gen_draft_logits[:, k, :].clone())

            # Greedy sampling for draft tokens
            gen_draft_tokens = self._draft_sampler_greedy(
                gen_draft_logits.reshape(-1, gen_draft_logits.shape[-1])
            ).reshape(num_gens, self.max_draft_len)

            next_draft_tokens[num_contexts:] = gen_draft_tokens

        # Store draft tokens in metadata for next iteration
        spec_metadata.draft_tokens[:batch_size] = next_draft_tokens

        return next_draft_tokens, draft_logits_list


class PARDSampler(TorchSampler):
    """
    Sampler for PARD speculative decoding.

    Similar to MTPSampler but adapted for PARD's parallel drafting approach.
    Handles token updates and KV cache management for PARD workers.
    """

    def __init__(self, args: "TorchSampler.Args", nextn: int = 1):
        super().__init__(args)
        self.nextn = nextn

    def sample(self, outputs, requests, logits_post_processor_names):
        """
        Sample tokens and update requests for PARD speculative decoding.

        Args:
            outputs: Model outputs containing accepted tokens
            requests: List of requests to update
            logits_post_processor_names: Names of logits processors

        Returns:
            Updated requests
        """
        if "new_tokens" not in outputs:
            # Fallback to regular sampling if no spec-dec outputs
            return super().sample(outputs, requests, logits_post_processor_names)

        new_tokens = outputs["new_tokens"]
        new_tokens_lens = outputs["new_tokens_lens"]

        # Update each request with accepted tokens
        for i, request in enumerate(requests):
            num_accepted = new_tokens_lens[i].item()
            accepted = new_tokens[i, :num_accepted].tolist()

            # Update request tokens
            request.seq.extend(accepted)

            # Update KV cache length
            request.kv_cache_len += num_accepted

        return requests
