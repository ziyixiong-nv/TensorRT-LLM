from typing import Optional, Tuple

import flashinfer.sampling
import torch


def forward_native(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    PyTorch-native implementation of top-k and top-p sampling.

    The logits tensor may be updated in-place.
    """
    logits = apply_top_k_top_p(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs)


def random_sample(
    probs: torch.Tensor,
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs).exponential_()
    return probs.div_(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        top_k_mask = top_k_mask.clamp(min=0)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def apply_temperature(
    logits: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    return logits.div_(temp.unsqueeze(dim=1))


@torch.compile(options={"max-autotune": True})
def sampling_batch_spec_dec_one_model(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-graph compatible sampling. Supports mixed sampling params.

    We can't do dynamic kernel selection inside graphs, so this might
    be slower than a torch.argmax for greedy requests. This is why advanced
    sampling is opt-in for now.
    """
    logits = apply_temperature(logits, temperatures)
    random_sampled = forward_native(logits, top_k, top_p)
    return random_sampled


def rejection_sample_and_accept_draft_tokens(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform rejection sampling for speculative decoding using FlashInfer's
    chain_speculative_sampling kernel.

    This implements the rejection sampling algorithm from the paper
    "Accelerating Large Language Model Decoding with Speculative Sampling"
    (https://arxiv.org/pdf/2302.01318).

    Args:
        draft_probs: Probabilities from the draft model.
            Shape: (batch_size, num_speculate_tokens, vocab_size)
        draft_token_ids: Token IDs sampled by the draft model.
            Shape: (batch_size, num_speculate_tokens)
        target_probs: Probabilities from the target model. Has one more position
            than draft_probs because the target model generates a bonus token.
            Shape: (batch_size, num_speculate_tokens + 1, vocab_size)
        deterministic: Whether to use deterministic kernel implementation.
            Default is True.
        generator: Optional random number generator for reproducibility.

    Returns:
        output_token_ids: Accepted token IDs, with rejected positions padded with -1.
            Shape: (batch_size, num_speculate_tokens + 1)
        output_accepted_token_num: Number of tokens that could be accepted if each
            token were considered independently (measures draft/target alignment).
            Shape: (batch_size,)
        output_emitted_draft_token_num: Number of draft tokens actually emitted
            (not including the bonus token).
            Shape: (batch_size,)
    """
    return flashinfer.sampling.chain_speculative_sampling(
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids,
        target_probs=target_probs,
        deterministic=deterministic,
        generator=generator,
    )


def rejection_sample_from_logits(
    draft_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform rejection sampling for speculative decoding from logits.

    This is a convenience wrapper that converts logits to probabilities
    before calling rejection_sample_and_accept_draft_tokens.

    Args:
        draft_logits: Logits from the draft model.
            Shape: (batch_size, num_speculate_tokens, vocab_size)
        draft_token_ids: Token IDs sampled by the draft model.
            Shape: (batch_size, num_speculate_tokens)
        target_logits: Logits from the target model. Has one more position
            than draft_logits because the target model generates a bonus token.
            Shape: (batch_size, num_speculate_tokens + 1, vocab_size)
        temperatures: Optional temperature values for scaling logits before softmax.
            If provided, shape should be (batch_size,) or (batch_size, 1).
            If None, no temperature scaling is applied (equivalent to temperature=1.0).
        deterministic: Whether to use deterministic kernel implementation.
            Default is True.
        generator: Optional random number generator for reproducibility.

    Returns:
        output_token_ids: Accepted token IDs, with rejected positions padded with -1.
            Shape: (batch_size, num_speculate_tokens + 1)
        output_accepted_token_num: Number of tokens that could be accepted if each
            token were considered independently (measures draft/target alignment).
            Shape: (batch_size,)
        output_emitted_draft_token_num: Number of draft tokens actually emitted
            (not including the bonus token).
            Shape: (batch_size,)
    """
    # Convert logits to probabilities
    if temperatures is not None:
        # Expand temperatures for broadcasting: (batch_size,) -> (batch_size, 1, 1)
        if temperatures.dim() == 1:
            temperatures = temperatures.unsqueeze(-1).unsqueeze(-1)
        elif temperatures.dim() == 2:
            temperatures = temperatures.unsqueeze(-1)
        draft_logits = draft_logits / temperatures
        target_logits = target_logits / temperatures

    draft_probs = torch.softmax(draft_logits, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)

    return rejection_sample_and_accept_draft_tokens(
        draft_probs=draft_probs,
        draft_token_ids=draft_token_ids,
        target_probs=target_probs,
        deterministic=deterministic,
        generator=generator,
    )
