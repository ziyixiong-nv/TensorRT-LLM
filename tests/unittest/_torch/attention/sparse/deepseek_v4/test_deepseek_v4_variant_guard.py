# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An unrecognized ``variant`` must fail loudly, not fall back to V4.

Every variant-dependent predicate in ``deepseek_v4.params`` is shaped "V4.1 does
X, otherwise do what V4 did". Before :func:`is_v41` existed, each one compared
the string itself, so a typo did not raise -- it reinterpreted a V4.1 model under
V4 semantics, where the two disagree on what a compress ratio *means*. The
damage is silent and model-wide: V4.1's ratio-2 layers are pooling factors, but
V4's ``is_sparse_layer`` tests ``== 4``, so it goes false everywhere and the
indexer is disabled on every layer while the model still produces fluent text.

These tests pin the guard, and pin that both *real* variants still answer exactly
what they answered before it was introduced -- the guard is a new error path, not
a change of semantics.
"""

import pytest

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
    DEEPSEEK_V4_VARIANT,
    DEEPSEEK_V4_VARIANTS,
    DEEPSEEK_V41_VARIANT,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
    has_compressor_state,
    is_compress_layer,
    is_overlap_compressor,
    is_sparse_layer,
    is_v41,
    swa_only_ratio,
)

# Near-misses a caller actually produces: the HF model_type, the config class
# name's suffix, a dotted version, and an unset field read as an empty string.
BAD_VARIANTS = ("deepseek_v41", "deepseek_v4", "V41", "v4.1", "v41 ", "", "4.1", "v5")


def test_is_v41_accepts_exactly_the_two_known_variants():
    assert is_v41(DEEPSEEK_V41_VARIANT) is True
    assert is_v41(DEEPSEEK_V4_VARIANT) is False
    assert DEEPSEEK_V4_VARIANTS == (DEEPSEEK_V4_VARIANT, DEEPSEEK_V41_VARIANT)


@pytest.mark.parametrize("variant", BAD_VARIANTS)
def test_every_variant_predicate_rejects_an_unknown_string(variant):
    """The guard has to cover all of them; one unguarded predicate is enough.

    A single predicate still comparing the raw string would silently answer the
    V4 way while its neighbours raised, which is harder to diagnose than either
    consistent behaviour.
    """
    for call in (
        lambda: is_v41(variant),
        lambda: swa_only_ratio(variant),
        lambda: is_overlap_compressor(2, variant),
        lambda: is_sparse_layer(2, variant),
        lambda: is_compress_layer(2, variant),
        lambda: has_compressor_state(2, variant),
        lambda: compress_ratio_has_attention(2, DeepseekV4AttentionType.COMPRESS, variant),
    ):
        with pytest.raises(ValueError, match="Unknown DeepSeek-V4 variant"):
            call()


def test_known_variants_keep_their_pre_guard_answers():
    """Regression net: the guard must not have moved any real answer.

    Hand-written rather than derived from the predicates, so a change in either
    variant's semantics shows up here as a diff instead of agreeing with itself.
    """
    # V4: ratio 4 is the sparse/overlap sentinel, 128 is dense-compressed.
    assert swa_only_ratio(DEEPSEEK_V4_VARIANT) == 1
    assert is_sparse_layer(4, DEEPSEEK_V4_VARIANT) is True
    assert is_sparse_layer(128, DEEPSEEK_V4_VARIANT) is False
    assert is_overlap_compressor(4, DEEPSEEK_V4_VARIANT) is True
    assert is_compress_layer(1, DEEPSEEK_V4_VARIANT) is False

    # V4.1: ratios are pooling factors, so every non-zero ratio is indexed and
    # nothing is an overlap compressor.
    assert swa_only_ratio(DEEPSEEK_V41_VARIANT) == 0
    assert is_sparse_layer(1, DEEPSEEK_V41_VARIANT) is True
    assert is_sparse_layer(2, DEEPSEEK_V41_VARIANT) is True
    assert is_sparse_layer(0, DEEPSEEK_V41_VARIANT) is False
    assert is_overlap_compressor(2, DEEPSEEK_V41_VARIANT) is False
    assert is_compress_layer(1, DEEPSEEK_V41_VARIANT) is True
    # Only the pooling ratio keeps cross-step accumulators.
    assert has_compressor_state(2, DEEPSEEK_V41_VARIANT) is True
    assert has_compressor_state(1, DEEPSEEK_V41_VARIANT) is False


@pytest.mark.parametrize(
    "variant, ratio, expected",
    [
        (DEEPSEEK_V4_VARIANT, 1, {"SWA"}),
        (
            DEEPSEEK_V4_VARIANT,
            4,
            {
                "SWA",
                "COMPRESSOR_KV",
                "COMPRESSOR_SCORE",
                "INDEXER_COMPRESSOR_KV",
                "INDEXER_COMPRESSOR_SCORE",
                "COMPRESS",
                "INDEXER_COMPRESS",
            },
        ),
        # Ratio 128 still pools, so it keeps the compressor accumulators; what it
        # drops is the indexer, because it attends over its whole cache densely.
        (
            DEEPSEEK_V4_VARIANT,
            128,
            {"SWA", "COMPRESSOR_KV", "COMPRESSOR_SCORE", "COMPRESS"},
        ),
        (DEEPSEEK_V41_VARIANT, 0, {"SWA"}),
        # V4.1's indexer has no compressor of its own -- its keys come from the
        # main latent -- so the two INDEXER_COMPRESSOR_* roles never appear.
        (DEEPSEEK_V41_VARIANT, 1, {"SWA", "COMPRESS", "INDEXER_COMPRESS"}),
        (
            DEEPSEEK_V41_VARIANT,
            2,
            {"SWA", "COMPRESSOR_KV", "COMPRESSOR_SCORE", "COMPRESS", "INDEXER_COMPRESS"},
        ),
    ],
)
def test_cache_roles_per_variant_and_ratio(variant, ratio, expected):
    """The role set is what sizes the pools, so pin it per (variant, ratio)."""
    roles = {
        attn_type.name
        for attn_type in DeepseekV4AttentionType
        if compress_ratio_has_attention(ratio, attn_type, variant)
    }
    assert roles == expected
