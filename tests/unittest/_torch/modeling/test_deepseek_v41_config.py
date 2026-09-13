# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-mapping tests for DeepSeek-V4.1.

Every silent failure mode in the V4.1 bring-up that these tests guard is a
*classification* error rather than an arithmetic one, so the assertions here are
deliberately exhaustive per layer rather than spot checks. In particular
:func:`test_all_43_ratio_entries_classified` pins the classification of all 43
``compress_ratios`` entries, and :func:`test_composed_v4_bugs_are_not_reproduced`
is the negative control: it reconstructs what the two pre-existing V4 code paths
would produce and asserts our descriptor differs on exactly the twenty layers
they get wrong.

These run on CPU in well under a second and require no checkpoint.
"""

import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from transformers import AutoConfig

from tensorrt_llm._torch.configs.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41QuantLayout,
    DeepseekV41QuantRole,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
    assert_weight_layout,
    build_layer_descriptors,
    derive_block,
    layout_for_role,
    parse_quantization_layout,
    quant_role_for_weight_key,
)

# The released DeepSeek-V4.1-Flash topology: 40 decoder layers + 3 DSpark stages.
# [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
RELEASE_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
RELEASE_KV_SOURCES = [2, 8, 14, 20]
RELEASE_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]
RELEASE_ENGRAM_LAYERS = [1, 14]


def release_text_config(**overrides) -> DeepseekV41TextConfig:
    kwargs = dict(
        compress_ratios=RELEASE_RATIOS,
        kv_source_layer_ids=RELEASE_KV_SOURCES,
        index_source_layer_ids=RELEASE_INDEX_SOURCES,
        engram_layer_ids=RELEASE_ENGRAM_LAYERS,
        engram_num_embeddings=[384006168, 384016682],
    )
    kwargs.update(overrides)
    return DeepseekV41TextConfig(**kwargs)


# ---------------------------------------------------------------------------
# Per-layer descriptor
# ---------------------------------------------------------------------------


def test_release_ratio_list_shape():
    """Guard the fixture itself, so a typo cannot weaken every other test."""
    assert len(RELEASE_RATIOS) == 43
    assert set(RELEASE_RATIOS) == {0, 1, 2}
    assert RELEASE_RATIOS.count(0) == 5
    assert RELEASE_RATIOS.count(2) == 18
    assert RELEASE_RATIOS.count(1) == 20


def test_all_43_ratio_entries_classified():
    """Pin the classification of every ratio entry.

    This is the test the bring-up brief calls for explicitly: assert the per-layer
    classification for all 43 entries. The four expected bands are layers 0-1 pure
    SWA, 2-19 indexed pooled 2:1, 20-39 indexed unpooled, 40-42 pure SWA.
    """
    descriptors = release_text_config().layer_descriptors
    assert len(descriptors) == 43

    for d in descriptors:
        if d.layer_idx in (0, 1) or d.layer_idx >= 40:
            # Pure sliding window: no long-range path, no YaRN, base theta.
            assert d.compress_ratio == 0, d
            assert d.has_long_range is False, d
            assert d.pools_kv is False, d
            assert d.pool_factor == 1, d
            assert d.rope_theta == 10000, d
            assert d.yarn_enabled is False, d
        elif 2 <= d.layer_idx <= 19:
            # Indexed and pooled 2:1.
            assert d.compress_ratio == 2, d
            assert d.has_long_range is True, d
            assert d.pools_kv is True, d
            assert d.pool_factor == 2, d
            assert d.rope_theta == 160000, d
            assert d.yarn_enabled is True, d
        else:
            # 20-39: indexed but UNPOOLED. Ratio 1 is not "no compression" --
            # compress_len is the full sequence, so these layers retrieve over
            # all history and take YaRN with the compressed theta.
            assert 20 <= d.layer_idx <= 39, d
            assert d.compress_ratio == 1, d
            assert d.has_long_range is True, d
            assert d.pools_kv is False, d
            assert d.pool_factor == 1, d
            assert d.rope_theta == 160000, d
            assert d.yarn_enabled is True, d

        # Every layer keeps its sliding window; the indexed path is additive.
        assert d.window_size == 128, d
        assert d.kind == ("mtp" if d.layer_idx >= 40 else "decoder"), d


def test_ratio_one_layers_are_long_range_not_uncompressed():
    """The single highest-risk misclassification in the task, isolated.

    V4's ``is_compress_layer(r) = r > 1`` would mark all twenty ratio-1 layers as
    non-long-range, stripping their retrieval and flipping them to base theta.
    """
    descriptors = release_text_config().layer_descriptors
    ratio_one = [d for d in descriptors if d.compress_ratio == 1]
    assert len(ratio_one) == 20
    assert all(d.has_long_range for d in ratio_one)
    assert all(d.yarn_enabled for d in ratio_one)
    assert all(d.rope_theta == 160000 for d in ratio_one)
    # ... but they do NOT pool, which is the narrower `> 1` condition.
    assert not any(d.pools_kv for d in ratio_one)


def test_bands_collapse_to_exactly_four():
    """The descriptor must collapse to the four bands the reference gate prints."""
    descriptors = release_text_config().layer_descriptors
    bands = []
    for d in descriptors:
        key = (d.has_long_range, d.pools_kv, d.pool_factor, d.rope_theta, d.yarn_enabled)
        if not bands or bands[-1][0] != key:
            bands.append((key, [d.layer_idx]))
        else:
            bands[-1][1].append(d.layer_idx)

    assert len(bands) == 4
    spans = [(b[1][0], b[1][-1]) for b in bands]
    assert spans == [(0, 1), (2, 19), (20, 39), (40, 42)]


def test_kv_and_index_source_asymmetry():
    """Four KV sources but eight index sources; only KV sources own a compressor."""
    descriptors = release_text_config().layer_descriptors

    kv_sources = [d.layer_idx for d in descriptors if d.is_kv_source]
    index_sources = [d.layer_idx for d in descriptors if d.is_index_source]
    assert kv_sources == RELEASE_KV_SOURCES
    assert index_sources == RELEASE_INDEX_SOURCES

    # Only the four KV sources own a compressor / indexer key projection --
    # allocating either per compressing layer would over-allocate ~10x.
    assert [d.layer_idx for d in descriptors if d.owns_compressor] == RELEASE_KV_SOURCES
    assert [d.layer_idx for d in descriptors if d.owns_indexer_wk] == RELEASE_KV_SOURCES

    # The extra four index sources contribute queries but own no key projection.
    extra = set(index_sources) - set(kv_sources)
    assert extra == {24, 28, 32, 36}
    for d in descriptors:
        if d.layer_idx in extra:
            assert d.is_index_source and not d.owns_indexer_wk


def test_compressor_dtype_follows_pooling_not_membership():
    """Pooling compressors run fp32; the ratio-1 compressor runs bf16 with no gate."""
    by_idx = {d.layer_idx: d for d in release_text_config().layer_descriptors}
    # Layers 2, 8, 14 are ratio 2 -> pooled -> fp32.
    for i in (2, 8, 14):
        assert by_idx[i].pools_kv is True
        assert by_idx[i].compressor_wkv_dtype == "fp32"
    # Layer 20 is ratio 1 -> unpooled -> bf16 (and it has no wgate in the ckpt).
    assert by_idx[20].pools_kv is False
    assert by_idx[20].compressor_wkv_dtype == "bf16"
    # Non-source layers own no compressor at all.
    assert by_idx[3].compressor_wkv_dtype is None


def test_source_layer_routing_is_most_recent_upstream():
    """A consumer layer reads the most recent source's published cache."""
    by_idx = {d.layer_idx: d for d in release_text_config().layer_descriptors}
    # Consumers between sources read the preceding source.
    assert by_idx[3].kv_source_layer_idx == 2
    assert by_idx[9].kv_source_layer_idx == 8
    assert by_idx[21].kv_source_layer_idx == 20
    assert by_idx[39].kv_source_layer_idx == 20
    # A source layer points at itself.
    assert by_idx[14].kv_source_layer_idx == 14
    # Pure-SWA layers have no long-range path, so no source.
    assert by_idx[0].kv_source_layer_idx is None
    assert by_idx[42].kv_source_layer_idx is None
    # Index sources advance independently of KV sources.
    assert by_idx[30].index_source_layer_idx == 28
    assert by_idx[39].index_source_layer_idx == 36


def test_two_level_candidate_topk_wiring():
    """Layer 20 publishes candidates; only *later* index sources consume them."""
    descriptors = release_text_config().layer_descriptors
    producers = [d.layer_idx for d in descriptors if d.is_candidate_source]
    consumers = [d.layer_idx for d in descriptors if d.consumes_candidates]
    assert producers == [20]
    # 24/28/32/36 mask their own scores with ~candidates before their own top-k.
    # The earlier index sources (2, 8, 14) and layer 20 itself do not.
    assert consumers == [24, 28, 32, 36]


def test_descriptor_predicts_the_checkpoints_own_tensor_counts():
    """The strongest cheap check available: predict the checkpoint's inventory.

    These counts were read off all 48 safetensors headers of
    DeepSeek-V4.1-Flash. They are an *independent* witness to the descriptor's
    classification, because the checkpoint only ships a tensor where the reference
    model actually built a submodule:

    * ``attn.compressor.wkv``       n=4  -- one per KV source
    * ``attn.compressor.wgate``     n=3  -- only the *pooling* sources have a gate
    * ``attn.indexer.wk``           n=4  -- key projection, KV sources only
    * ``attn.indexer.wq_b``         n=8  -- query projection, all index sources
    * ``attn.indexer.weights_proj`` n=8  -- ditto
    * ``engram.q_weight``           n=2

    The 4-vs-8 split is why ``owns_indexer_wk`` is a separate field from
    ``is_index_source``, and the 3-vs-4 split is why ``pools_kv`` is separate from
    ``is_kv_source``. A descriptor that collapsed either pair would predict the
    wrong inventory here.
    """
    descriptors = release_text_config().layer_descriptors

    assert sum(d.owns_compressor for d in descriptors) == 4
    assert sum(d.is_kv_source and d.pools_kv for d in descriptors) == 3
    assert sum(d.owns_indexer_wk for d in descriptors) == 4
    assert sum(d.is_index_source for d in descriptors) == 8
    assert sum(d.has_engram for d in descriptors) == 2

    # And the compressor dtype split lines up with the wgate split exactly.
    assert sum(d.compressor_wkv_dtype == "fp32" for d in descriptors) == 3
    assert sum(d.compressor_wkv_dtype == "bf16" for d in descriptors) == 1
    assert sum(d.compressor_wkv_dtype is None for d in descriptors) == 39


def test_engram_layers_and_table_sizes():
    cfg = release_text_config()
    engram = [d.layer_idx for d in cfg.layer_descriptors if d.has_engram]
    assert engram == [1, 14]
    # The two tables have DIFFERENT sizes, so they are not one shared table.
    assert cfg.engram_num_embeddings_for_layer(1) == 384006168
    assert cfg.engram_num_embeddings_for_layer(14) == 384016682
    assert cfg.engram_num_embeddings_for_layer(1) != cfg.engram_num_embeddings_for_layer(14)
    with pytest.raises(ValueError, match="does not host an Engram table"):
        cfg.engram_num_embeddings_for_layer(2)


def test_engram_pad_comes_from_config_not_tokenizer():
    """The tokenizer reports pad 1; Engram must hash the config's pad 2."""
    assert release_text_config().engram_pad_token_id == 2


def test_absent_v4_keys_are_not_defaulted():
    """Absent keys are instructions: V4 defaults here would build a wrong net."""
    cfg = release_text_config()
    # No dense FFN prefix and no dense intermediate size -- all layers are MoE.
    assert not hasattr(cfg, "first_k_dense_replace")
    assert not hasattr(cfg, "intermediate_size")
    # V4's spelling of the hash-routing count stays absent; ours is pinned off
    # instead of absent, for the reason in the next test.
    assert not hasattr(cfg, "num_hash_layers")


def test_v4_router_knobs_are_pinned_to_their_identity_setting():
    """Three router knobs cannot merely be absent -- V4's gate reads them.

    ``n_hash_layers`` / ``n_group`` / ``topk_group`` are absent from V4.1's
    ``config.json``, but the shared DeepSeek-V4 router reads all three
    unconditionally, so leaving them off raises instead of building the
    reference behaviour, and letting V4's own defaults apply builds a
    *different* net (three hash-routed layers; eight-group node-limited
    routing). This class therefore pins them to the settings that reproduce
    V4.1's plain global top-k gate, and this test is what stops that pinning
    from drifting into V4's values.
    """
    cfg = release_text_config()
    # Hash routing off: the checkpoint has no `tid2eid` table and every
    # `ffn.gate` ships a real `bias`, so all 40 layers route by top-k.
    assert cfg.n_hash_layers == 0
    # One group holding every expert, and that one group selected == global
    # top-k, which is what the reference gate does.
    assert cfg.n_group == 1
    assert cfg.topk_group == 1
    # Pinned by this class, not accepted from a caller: a checkpoint that really
    # does declare group routing must not be able to land on the degenerate
    # setting by omission.
    params = inspect.signature(DeepseekV41TextConfig.__init__).parameters
    assert "n_hash_layers" not in params
    assert "n_group" not in params
    assert "topk_group" not in params


def test_release_dimensions_and_eps():
    cfg = release_text_config()
    assert cfg.hidden_size == 5120
    assert cfg.q_lora_rank == 1280
    assert cfg.n_routed_experts == 384
    assert cfg.moe_intermediate_size == 2304
    assert cfg.index_n_heads == 32
    assert cfg.num_hidden_layers == 40
    assert cfg.head_dim == 512
    assert cfg.qk_rope_head_dim == 64
    assert cfg.qk_nope_head_dim == 448  # derived: 512 - 64
    assert cfg.num_key_value_heads == 1  # all 64 heads share one 512-dim latent
    # TRT-LLM's MLA splits that one 512-wide latent into `kv_lora_rank`
    # un-rotated lanes + `qk_rope_head_dim` rotated ones, so `kv_lora_rank` is
    # 448 and NOT the latent width. Same pair of numbers V4 pins literally.
    assert cfg.kv_lora_rank == 448
    assert cfg.kv_lora_rank + cfg.qk_rope_head_dim == cfg.head_dim
    # ... while the value MLA reads out of that latent is the whole thing.
    assert cfg.v_head_dim == 512
    assert cfg.o_groups == 8
    assert cfg.o_lora_rank == 1024
    assert cfg.vocab_size == 129280
    assert cfg.max_position_embeddings == 1048576
    # 1e-20, down from V4's 1e-6, and load-bearing.
    assert cfg.rms_norm_eps == 1e-20
    # The mHC epsilon is separate and stays at 1e-6.
    assert cfg.hc_eps == 1e-6
    assert cfg.hc_mult == 4
    assert cfg.hc_sinkhorn_iters == 20


def test_mhc_derived_shapes_match_checkpoint():
    """``mix_hc = (2 + hc_mult) * hc_mult = 24`` and ``hc_dim = hc_mult * dim``."""
    cfg = release_text_config()
    assert (2 + cfg.hc_mult) * cfg.hc_mult == 24
    assert cfg.hc_mult * cfg.hidden_size == 20480


def test_dspark_config():
    cfg = release_text_config()
    assert cfg.num_nextn_predict_layers == 3
    assert cfg.dspark_target_layer_ids == []  # not set by the fixture
    cfg = release_text_config(dspark_target_layer_ids=[37, 38, 39])
    assert cfg.dspark_target_layer_ids == [37, 38, 39]
    # Each DSpark stage runs a SMALLER 128-expert top-3 MoE.
    assert cfg.dspark_n_routed_experts == 128
    assert cfg.dspark_num_experts_per_tok == 3
    assert cfg.dspark_markov_rank == 256
    assert cfg.dspark_noise_token_id == 128799


def test_kv_bytes_per_token_matches_hand_derivation():
    """890 B/token: 3 ratio-2 sources at 178 B + 1 ratio-1 source at 356 B."""
    cfg = release_text_config()
    assert cfg.bytes_per_compressed_entry() == pytest.approx(356.0)
    assert cfg.kv_bytes_per_token() == pytest.approx(890.0)
    # Sanity: a V4-style 128:1 pooling assumption would under-provision by ~64x
    # on the pooled band, which is exactly the capacity trap to avoid.
    assert cfg.kv_bytes_per_token() > 10 * (4 * 356.0 / 128)


def test_descriptor_rejects_inconsistent_source_lists():
    """A source layer with ratio 0 is a config contradiction, not a warning."""
    with pytest.raises(ValueError, match="listed as a kv/index source"):
        build_layer_descriptors(
            compress_ratios=[0, 0, 2, 2],
            num_hidden_layers=4,
            rope_theta=10000,
            compress_rope_theta=160000,
            window_size=128,
            kv_source_layer_ids=[0],  # ratio 0 -> cannot own a compressor
            index_source_layer_ids=[2],
        )
    with pytest.raises(ValueError, match="outside the"):
        build_layer_descriptors(
            compress_ratios=[0, 2],
            num_hidden_layers=2,
            rope_theta=10000,
            compress_rope_theta=160000,
            window_size=128,
            kv_source_layer_ids=[7],
            index_source_layer_ids=[],
        )


def test_descriptor_refuses_empty_ratio_list():
    """Never synthesize a default ratio list -- it changes attention semantics."""
    with pytest.raises(ValueError, match="requires the per-layer"):
        build_layer_descriptors(
            compress_ratios=[],
            num_hidden_layers=0,
            rope_theta=10000,
            compress_rope_theta=160000,
            window_size=128,
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
        )


# ---------------------------------------------------------------------------
# Negative control: the two pre-existing V4 bugs
# ---------------------------------------------------------------------------


def _simulate_composed_v4_bugs():
    """What the two pre-existing V4 code paths produce for V4.1's ratios.

    Bug A (``_torch/model_config.py``): rewrite ratio 0 -> 1 so "cache allocation
    math works". Bug B (``sparse/deepseek_v4/params.py``): classify long-range
    with ``ratio > 1``.

    The two **cancel** on ratio-0 layers -- 0 becomes 1, then ``1 > 1`` is False,
    landing them back on base theta with YaRN off, which is accidentally correct.
    So only the twenty ratio-1 layers come out wrong, there is no short-context
    symptom, and fixing either bug alone breaks the cancellation.
    """
    normalized = [r if r > 0 else 1 for r in RELEASE_RATIOS]  # bug A
    return [
        {
            "layer_idx": i,
            "has_long_range": r > 1,  # bug B
            "rope_theta": 160000 if r > 1 else 10000,
            "yarn_enabled": r > 1,
        }
        for i, r in enumerate(normalized)
    ]


def test_composed_v4_bugs_are_not_reproduced():
    """Our descriptor must differ from the buggy path on exactly 20 layers x 3 fields."""
    buggy = {row["layer_idx"]: row for row in _simulate_composed_v4_bugs()}
    ours = {d.layer_idx: d for d in release_text_config().layer_descriptors}

    disagreeing = set()
    field_mismatches = 0
    for i, d in ours.items():
        for field in ("has_long_range", "rope_theta", "yarn_enabled"):
            if getattr(d, field) != buggy[i][field]:
                disagreeing.add(i)
                field_mismatches += 1

    # Exactly the twenty ratio-1 layers, three fields each.
    assert disagreeing == set(range(20, 40)), sorted(disagreeing)
    assert field_mismatches == 60

    # And the cancellation really does leave the pure-SWA layers agreeing, which
    # is why a partial fix is worse than none.
    for i in (0, 1, 40, 41, 42):
        assert ours[i].has_long_range == buggy[i]["has_long_range"]
        assert ours[i].rope_theta == buggy[i]["rope_theta"]
        assert ours[i].yarn_enabled == buggy[i]["yarn_enabled"]


def test_v4_and_v41_ratio_encodings_are_disjoint():
    """The value sets share only 0, so no injective remap onto V4 exists."""
    v4_ratios = {0, 4, 128}
    v41_ratios = set(RELEASE_RATIOS)
    assert v4_ratios & v41_ratios == {0}
    # In particular no V4.1 layer has V4's sparse sentinel 4, which is why every
    # `ratio == 4` test declares the whole model non-sparse.
    assert 4 not in v41_ratios


def test_v4_ratio_classifiers_must_not_be_reused_for_v41():
    """Negative control on the module boundary, measured rather than asserted.

    `.../attention/backends/sparse/deepseek_v4/params.py` classifies layers by
    comparing the ratio to V4's sentinels. Those helpers are *correct for V4* and
    must stay untouched, but they are catastrophically wrong for V4.1, so V4.1 has
    to read `DeepseekV41LayerDescriptor` instead (plan section 4.2). This test
    measures how wrong, so that a future refactor which quietly routes V4.1 through
    the V4 helpers fails here instead of at a GSM8K run.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DEEPSEEK_V4_SPARSE_RATIO,
        is_compress_layer,
        is_sparse_layer,
    )

    descriptors = release_text_config().layer_descriptors

    # V4's sparse sentinel appears nowhere in V4.1, so `is_sparse_layer` declares
    # *every* layer non-sparse -- the whole indexed long-range path disappears with
    # no error anywhere.
    assert DEEPSEEK_V4_SPARSE_RATIO == 4
    assert sum(is_sparse_layer(d.compress_ratio) for d in descriptors) == 0
    assert sum(d.has_long_range for d in descriptors) == 38

    # `is_compress_layer` is `ratio > 1`, so it finds the 18 pooled layers and
    # misses all 20 ratio-1 layers that retrieve over full history unpooled.
    v4_compress = {d.layer_idx for d in descriptors if is_compress_layer(d.compress_ratio)}
    ours_long_range = {d.layer_idx for d in descriptors if d.has_long_range}
    assert v4_compress == set(range(2, 20))
    assert ours_long_range - v4_compress == set(range(20, 40))
    # It does agree with our narrower `pools_kv`, which is what it actually means.
    assert v4_compress == {d.layer_idx for d in descriptors if d.pools_kv}


# ---------------------------------------------------------------------------
# Quantization layout
# ---------------------------------------------------------------------------


RELEASE_QUANT_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def test_three_distinct_quant_layouts():
    """``quantization_config`` advertises one block; the checkpoint uses three."""
    layout = parse_quantization_layout(RELEASE_QUANT_CONFIG)

    expert = layout[DeepseekV41QuantRole.EXPERT]
    engram = layout[DeepseekV41QuantRole.ENGRAM_EMBED]
    dense = layout[DeepseekV41QuantRole.DENSE]

    # Routed experts: OCP MXFP4 -- 32-element blocks along K only, e8m0 scales,
    # two elements per int8 container.
    assert (expert.weight_dtype, expert.scale_dtype) == ("mxfp4_e2m1", "float8_e8m0")
    assert expert.element_block == (1, 32)
    assert (expert.storage_dtype, expert.pack_factor, expert.packed_dim) == ("int8", 2, 1)
    # Engram tables: fp8 e4m3 with one e8m0 scale per 32 columns, unpacked.
    assert (engram.weight_dtype, engram.element_block, engram.pack_factor) == (
        "float8_e4m3fn",
        (1, 32),
        1,
    )
    # Dense: fp8 e4m3 blockwise 32x32 -- the only role the config's block fits.
    assert (dense.weight_dtype, dense.element_block, dense.pack_factor) == (
        "float8_e4m3fn",
        (32, 32),
        1,
    )

    # The advertised block is right for exactly one of the three roles.
    advertised = tuple(RELEASE_QUANT_CONFIG["weight_block_size"])
    assert sum(1 for e in layout.values() if e.element_block == advertised) == 1


def test_expert_stored_block_is_16_and_that_is_the_trap():
    """The on-disk block of an MXFP4 expert is 16, which is NVFP4's block size.

    So the check cannot simply trust ``weight.shape // scale.shape``: it has to
    know the pack factor. Getting this wrong reads MXFP4 as NVFP4, and because
    both pack two nibbles per byte no shape mismatch ever surfaces.
    """
    expert = layout_for_role(DeepseekV41QuantRole.EXPERT)
    assert expert.element_block == (1, 32)  # semantics: 32 elements per scale
    assert expert.stored_block == (1, 16)  # on disk: 16 int8 containers per scale
    # ... and the real header shapes derive the stored block, not the element one.
    assert derive_block((2304, 2560), (2304, 160)) == expert.stored_block
    assert expert.logical_shape((2304, 2560)) == (2304, 5120)
    # NVFP4 would also have block 16 on disk but e4m3 scales and a global scale.
    assert expert.scale_dtype == "float8_e8m0"


def test_unpacked_roles_have_no_stored_vs_element_gap():
    for role in (DeepseekV41QuantRole.DENSE, DeepseekV41QuantRole.ENGRAM_EMBED):
        entry = layout_for_role(role)
        assert entry.stored_block == entry.element_block
        assert entry.logical_shape((7, 11)) == (7, 11)


def test_expert_dtype_fp8_escape_hatch():
    """``--expert-dtype fp8`` widens experts losslessly to fp8 e4m3 32x32."""
    layout = parse_quantization_layout(dict(RELEASE_QUANT_CONFIG, expert_dtype="fp8"))
    expert = layout[DeepseekV41QuantRole.EXPERT]
    assert (expert.weight_dtype, expert.storage_dtype) == ("float8_e4m3fn",) * 2
    assert expert.element_block == (32, 32)
    assert expert.pack_factor == 1  # no longer nibble-packed
    assert expert.stored_block == (32, 32)


def test_quant_config_rejects_unknown_formats():
    for bad in (
        {"quant_method": "awq"},
        {"scale_fmt": "e4m3"},
        {"expert_dtype": "int4"},
        {"weight_block_size": [32, 32, 32]},
    ):
        with pytest.raises(ValueError):
            parse_quantization_layout(dict(RELEASE_QUANT_CONFIG, **bad))


def test_dense_block_override_never_leaks_to_the_other_roles():
    layout = parse_quantization_layout(dict(RELEASE_QUANT_CONFIG, weight_block_size=[128, 128]))
    assert layout[DeepseekV41QuantRole.DENSE].element_block == (128, 128)
    # The two 1x32 roles are not derived from weight_block_size and must not move.
    assert layout[DeepseekV41QuantRole.EXPERT].element_block == (1, 32)
    assert layout[DeepseekV41QuantRole.ENGRAM_EMBED].element_block == (1, 32)


def test_quant_layout_rejects_incoherent_packing():
    with pytest.raises(ValueError, match="requires packed_dim"):
        DeepseekV41QuantLayout("a", "b", "c", (1, 32), pack_factor=2)
    with pytest.raises(ValueError, match="not divisible by"):
        DeepseekV41QuantLayout("a", "b", "c", (1, 32), pack_factor=2, packed_dim=0)
    with pytest.raises(ValueError, match="pack_factor must be"):
        DeepseekV41QuantLayout("a", "b", "c", (1, 32), pack_factor=0)


def test_quant_role_classification():
    """Routed experts are MXFP4; shared experts next door are not."""
    assert (
        quant_role_for_weight_key("layers.5.ffn.experts.17.w1.weight")
        == DeepseekV41QuantRole.EXPERT
    )
    # The DSpark stages' experts are routed experts too.
    assert quant_role_for_weight_key("mtp.0.ffn.experts.3.w2.weight") == DeepseekV41QuantRole.EXPERT
    # `shared_experts` is plain fp8 32x32 -- it must NOT be read as MXFP4.
    assert (
        quant_role_for_weight_key("layers.5.ffn.shared_experts.w1.weight")
        == DeepseekV41QuantRole.DENSE
    )
    assert (
        quant_role_for_weight_key("layers.1.engram.embed.weight")
        == DeepseekV41QuantRole.ENGRAM_EMBED
    )
    assert quant_role_for_weight_key("layers.5.attn.wq_b.weight") == DeepseekV41QuantRole.DENSE


def test_quant_role_classification_spans_both_naming_conventions():
    """The classifier runs on both sides of the load, so it must accept both names.

    The checkpoint spells the block `ffn`; TensorRT-LLM's own modules spell it `mlp`.
    A classifier keyed on `.ffn.experts.` alone would silently call every TRT-LLM-side
    routed expert DENSE and dequantize MXFP4 as blocked fp8 -- the same class of
    silent-wrongness the stored-block check exists to catch, arriving through the
    naming door instead. `$W/gates/quant_layout.py` accepts the same alternation,
    and this test is what keeps the two from drifting apart.
    """
    for key in (
        "layers.5.mlp.experts.17.w1.weight",
        "model.layers.5.mlp.experts.0.w3.weight",
        "ffn.experts.2.w2.weight",  # leading-anchored, no dotted prefix
    ):
        assert quant_role_for_weight_key(key) == DeepseekV41QuantRole.EXPERT, key

    # `experts.` must be followed by an index: the shared-expert block and a bare
    # `experts` attribute are both DENSE, under either spelling.
    for key in (
        "layers.5.mlp.shared_experts.w1.weight",
        "layers.5.ffn.experts.gate.weight",
        "layers.5.mlp.experts_bias",
    ):
        assert quant_role_for_weight_key(key) == DeepseekV41QuantRole.DENSE, key

    # The Engram role covers the scale as well as the weight -- callers classify a
    # stem, and both members of the pair have to land on the same layout.
    for key in (
        "layers.1.engram.embed.weight",
        "layers.14.engram.embed.scale",
        "layers.1.engram.embed",
    ):
        assert quant_role_for_weight_key(key) == DeepseekV41QuantRole.ENGRAM_EMBED, key
    # ... but a neighbouring Engram tensor is not the embedding table.
    assert (
        quant_role_for_weight_key("layers.1.engram.embed_norm.weight") == DeepseekV41QuantRole.DENSE
    )


def test_load_assert_accepts_every_dtype_spelling_a_loader_can_hand_it():
    """safetensors says `I8`, torch says `int8`, `str(t.dtype)` says `torch.int8`.

    The caller is a loader walking safetensors headers, so if the check only accepted
    torch names every call site would need its own mapping table -- and a wrong table
    would make the dtype half of the check compare a name it never matched, i.e. pass
    silently. All three spellings must land on the same verdict.
    """
    expert = ("layers.0.ffn.experts.0.w1.weight", (2304, 2560), (2304, 160))
    dense = ("layers.0.attn.wkv.weight", (512, 5120), (16, 160))
    for dtypes, (key, wsh, ssh) in (
        (("I8", "int8", "torch.int8"), expert),
        (("F8_E4M3", "float8_e4m3fn", "torch.float8_e4m3fn"), dense),
    ):
        for dtype in dtypes:
            assert_weight_layout(key, wsh, ssh, storage_dtype=dtype)

    # Normalization must not turn into "accept anything": a genuinely wrong dtype
    # still has to raise, under every spelling.
    for dtype in ("BF16", "bfloat16", "torch.bfloat16"):
        with pytest.raises(ValueError, match="expects on-disk dtype"):
            assert_weight_layout(*expert, storage_dtype=dtype)


def test_layout_for_role_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unknown DeepSeek-V4.1 quant role"):
        layout_for_role("nope")


def test_derive_block_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="different ranks"):
        derive_block((32, 32), (32,))
    with pytest.raises(ValueError, match="not an integer multiple"):
        derive_block((32, 33), (1, 32))


# A *sample* of the (key, weight shape, scale shape, on-disk dtype) quantized
# layouts in DeepSeek-V4.1-Flash, read straight off the 48 safetensors headers:
# one or two per distinct shape class, not one per tensor. The checkpoint holds
# 96085 tensors and this list holds 16, so it proves the assert accepts every
# *shape class* on disk -- it does not prove the assert accepts every tensor.
# `gates/quant_layout.py --assert-load-layout` is the exhaustive walk; run that
# for the whole-checkpoint claim.
RELEASE_QUANTIZED_TENSORS = [
    ("layers.6.attn.wq_a.weight", (1280, 5120), (40, 160), "float8_e4m3fn"),
    ("layers.6.attn.wq_b.weight", (32768, 1280), (1024, 40), "float8_e4m3fn"),
    ("layers.6.attn.wkv.weight", (512, 5120), (16, 160), "float8_e4m3fn"),
    ("layers.6.attn.wo_a.weight", (8192, 4096), (256, 128), "float8_e4m3fn"),
    ("layers.6.attn.wo_b.weight", (5120, 8192), (160, 256), "float8_e4m3fn"),
    ("layers.2.attn.indexer.wq_b.weight", (4096, 1280), (128, 40), "float8_e4m3fn"),
    ("layers.6.ffn.shared_experts.w1.weight", (2304, 5120), (72, 160), "float8_e4m3fn"),
    ("layers.6.ffn.shared_experts.w2.weight", (5120, 2304), (160, 72), "float8_e4m3fn"),
    ("layers.6.ffn.experts.0.w1.weight", (2304, 2560), (2304, 160), "int8"),
    ("layers.6.ffn.experts.0.w2.weight", (5120, 1152), (5120, 72), "int8"),
    ("layers.6.ffn.experts.0.w3.weight", (2304, 2560), (2304, 160), "int8"),
    ("layers.1.engram.embed.weight", (384006168, 256), (384006168, 8), "float8_e4m3fn"),
    ("layers.14.engram.embed.weight", (384016682, 256), (384016682, 8), "float8_e4m3fn"),
    ("mtp.0.main_proj.weight", (5120, 15360), (160, 480), "float8_e4m3fn"),
    ("mtp.1.ffn.experts.7.w1.weight", (2304, 2560), (2304, 160), "int8"),
    ("mtp.2.ffn.shared_experts.w3.weight", (2304, 5120), (72, 160), "float8_e4m3fn"),
]


def test_release_shape_sample_passes_the_load_assert():
    """Every shape class on disk must validate -- no false positives.

    Sixteen sampled tensors, not the checkpoint's 96085. The exhaustive walk is
    `python3 gates/quant_layout.py --assert-load-layout`, which feeds every
    header through this same `assert_weight_layout`.
    """
    for key, wshape, sshape, dtype in RELEASE_QUANTIZED_TENSORS:
        layout = assert_weight_layout(
            key, wshape, sshape, dtype, quantization_config=RELEASE_QUANT_CONFIG
        )
        assert layout.stored_block == derive_block(wshape, sshape), key


def test_load_assert_raises_on_a_corrupted_expectation():
    """Demonstrate the assert is load-bearing, not decorative."""
    # An expert tensor whose scale implies block 32 on disk -- i.e. someone wrote
    # it as if MXFP4 were unpacked. Must raise, and must say so in MXFP4 terms.
    with pytest.raises(ValueError, match=r"expects on-disk block \(1, 16\)"):
        assert_weight_layout("layers.6.ffn.experts.0.w1.weight", (2304, 2560), (2304, 80), "int8")
    # A dense tensor read with the experts' 1x32 block.
    with pytest.raises(ValueError, match=r"expects on-disk block \(32, 32\)"):
        assert_weight_layout(
            "layers.6.attn.wq_b.weight", (32768, 1280), (32768, 40), "float8_e4m3fn"
        )
    # Right block, wrong container dtype: an expert stored unpacked as fp8 while
    # quantization_config still says fp4.
    with pytest.raises(ValueError, match="expects on-disk dtype 'int8'"):
        assert_weight_layout(
            "layers.6.ffn.experts.0.w1.weight",
            (2304, 2560),
            (2304, 160),
            "float8_e4m3fn",
        )


def test_modeling_pack_factor_is_what_rules_nvfp4_out():
    """Why the pack factor has to be modeled rather than derived.

    The naive derivation ``2560 // 160`` yields 16, and 16 is *also* NVFP4's
    element block. A loader that maps a derived block straight onto a format reads
    the experts as NVFP4 on that numeric coincidence alone.

    Once packing is modeled the coincidence disappears: NVFP4's 16-element block
    at two elements per byte would put 8 containers under each scale, so the
    checkpoint's 16 positively excludes NVFP4 rather than suggesting it.
    """
    naive = derive_block((2304, 2560), (2304, 160))
    nvfp4_like = DeepseekV41QuantLayout(
        weight_dtype="nvfp4_e2m1",
        storage_dtype="int8",
        scale_dtype="float8_e4m3fn",  # NVFP4 uses e4m3 scales, not e8m0
        element_block=(1, 16),
        pack_factor=2,
        packed_dim=1,
    )
    expert = layout_for_role(DeepseekV41QuantRole.EXPERT)

    # The coincidence: the naive derivation equals NVFP4's element block.
    assert naive == (1, 16) == nvfp4_like.element_block
    # The discrimination: NVFP4 would store 8 containers per scale, MXFP4 16.
    assert nvfp4_like.stored_block == (1, 8)
    assert expert.stored_block == (1, 16) == naive
    assert naive != nvfp4_like.stored_block
    # And the scale dtype is a second, independent discriminator.
    assert expert.scale_dtype == "float8_e8m0" != nvfp4_like.scale_dtype


# ---------------------------------------------------------------------------
# Composite config
# ---------------------------------------------------------------------------


def test_composite_config_rebuilds_nested_dicts():
    """Sub-configs arrive as dicts from AutoConfig and must become real classes."""
    cfg = DeepseekV41Config(
        text_config={
            "compress_ratios": RELEASE_RATIOS,
            "kv_source_layer_ids": RELEASE_KV_SOURCES,
            "index_source_layer_ids": RELEASE_INDEX_SOURCES,
            "engram_layer_ids": RELEASE_ENGRAM_LAYERS,
        },
        vision_config={"num_hidden_layers": 32, "hidden_size": 1024},
        quantization_config=RELEASE_QUANT_CONFIG,
    )
    assert isinstance(cfg.text_config, DeepseekV41TextConfig)
    assert isinstance(cfg.vision_config, DeepseekV41VisionConfig)
    assert cfg.model_type == "deepseek_v41"
    # Language-model fields are forwarded so flat-config call sites keep working.
    assert cfg.hidden_size == 5120
    assert cfg.n_routed_experts == 384
    assert len(cfg.layer_descriptors) == 43
    # The top-level quantization_config drives the per-role layout.
    assert cfg.quantization_layout[DeepseekV41QuantRole.EXPERT].element_block == (1, 32)
    # ... and must reach the text tower, which is where the weight loader reads it.
    # V4.1 publishes it at the top level only, so a sub-config built from
    # `text_config` verbatim has no quantization policy at all and every weight
    # then loads as unquantized bf16 with nothing raising.
    assert cfg.text_config.quantization_config == RELEASE_QUANT_CONFIG
    assert cfg.text_config.quantization_layout == cfg.quantization_layout
    # Unknown attributes still raise rather than silently returning None.
    with pytest.raises(AttributeError):
        _ = cfg.definitely_not_a_field


def test_composite_config_refuses_to_drop_the_quantization_policy(monkeypatch):
    """The propagation is asserted at construction, not hoped for.

    A text tower that arrives already carrying a policy keeps its own; one that
    arrives bare inherits the top-level dict; and a composite that somehow ends up
    with a quantized top level and a bare text tower must raise rather than load.
    """
    explicit = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [32, 32]}
    cfg = DeepseekV41Config(
        text_config={"quantization_config": explicit},
        quantization_config=RELEASE_QUANT_CONFIG,
    )
    assert cfg.text_config.quantization_config == explicit

    # An unquantized checkpoint stays unquantized on both levels.
    bare = DeepseekV41Config()
    assert getattr(bare, "quantization_config", None) is None
    assert getattr(bare.text_config, "quantization_config", None) is None

    class _Dropper(DeepseekV41TextConfig):
        """A text config that silently discards the key, as a refactor might."""

        def __init__(self, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(**kwargs)

    # Swap the class through `sub_configs` rather than by subclassing the composite:
    # transformers 5.x replaces `__init__` on any config subclass that does not
    # define its own, so a subclass here would never run the code under test.
    monkeypatch.setitem(DeepseekV41Config.sub_configs, "text_config", _Dropper)
    with pytest.raises(RuntimeError, match="did not survive into text_config"):
        DeepseekV41Config(text_config={}, quantization_config=RELEASE_QUANT_CONFIG)


def test_vision_config_derived_geometry():
    """Each number here is confirmed by a checkpoint tensor shape."""
    v = DeepseekV41VisionConfig()
    assert v.head_dim == 64  # wqkv (3072, 1024) / 16 heads
    assert v.rope_dim == 32  # 2D RoPE splits head_dim in half
    # patch_embed.proj.weight is (1024, 588): a Linear, not a Conv2d.
    assert v.patch_input_dim == 588  # 3 * 14**2
    assert v.downsample_ratio == 3
    # aligner.w1.weight is (5120, 9216) = hidden * downsample_ratio**2.
    assert v.hidden_size * v.downsample_ratio**2 == 9216
    # mlp.w1 is (5632, 1024) = 2 * intermediate: gate and up are fused in w1,
    # and mlp.w2 is (1024, 2816), so the tower is gated with only two tensors.
    assert 2 * v.intermediate_size == 5632
    # The vision tower keeps eps 1e-6; only the text stack moved to 1e-20.
    assert v.rms_norm_eps == 1e-6


def test_layer_plan_dump_is_gate_shaped():
    """The dump must carry every field the layer_plan gate compares."""
    dump = release_text_config().layer_plan_dump()
    assert dump["num_ratio_entries"] == 43
    assert len(dump["layers"]) == 43
    required = {
        "has_long_range",
        "pools_kv",
        "pool_factor",
        "rope_theta",
        "yarn_enabled",
        "window_size",
        "is_kv_source",
        "is_index_source",
        "owns_compressor",
        "owns_indexer_wk",
        "has_engram",
        "compressor_wkv_dtype",
    }
    for row in dump["layers"]:
        assert required <= set(row), required - set(row)
    # Round-trips through JSON, since that is how the gate consumes it.
    assert json.loads(json.dumps(dump))["layers"][20]["rope_theta"] == 160000


def test_descriptors_are_cached_and_immutable():
    cfg = release_text_config()
    assert cfg.layer_descriptors is cfg.layer_descriptors
    with pytest.raises(Exception):
        cfg.layer_descriptors[0].has_long_range = True


# ---------------------------------------------------------------------------
# The released checkpoint, through the real loader
#
# Everything above builds its configs from hand-written fixtures, which is why an
# earlier revision of this file passed 40 tests while `ModelConfig.from_pretrained`
# on the actual checkpoint raised `ValueError: ... model type 'deepseek_v41' ...`
# and while the composite exposed only 30 of the text tower's 59 fields. A fixture
# cannot notice either. These four tests load the release config.
# ---------------------------------------------------------------------------

RELEASE_CHECKPOINT = Path("/code/llm-models/DeepSeek-V4.1-Flash")

requires_release_checkpoint = pytest.mark.skipif(
    not (RELEASE_CHECKPOINT / "config.json").is_file(),
    reason=f"release checkpoint not present at {RELEASE_CHECKPOINT}",
)


@requires_release_checkpoint
def test_release_config_json_is_the_shape_the_loader_assumes():
    """Pin the three structural facts the loader branch depends on."""
    raw = json.loads((RELEASE_CHECKPOINT / "config.json").read_text())
    assert raw["model_type"] == "deepseek_v41"
    assert raw["architectures"] == ["DeepseekV41ForCausalLM"]
    # Nested towers, and `quantization_config` at the top level *only*.
    assert isinstance(raw["text_config"], dict) and raw["text_config"]
    assert isinstance(raw["vision_config"], dict) and raw["vision_config"]
    assert "quantization_config" not in raw["text_config"]
    assert raw["quantization_config"]["quant_method"] == "fp8"
    assert raw["quantization_config"]["scale_fmt"] == "ue8m0"
    # `dtype` is top-level only, so the text tower needs it resolved explicitly.
    assert "dtype" not in raw["text_config"] and "torch_dtype" not in raw["text_config"]
    # The inline dict is the only quantization policy here. `model_config.py` keeps V4.1
    # out of the `hf_quant_config.json` (modelopt-format) gate on that basis; if a future
    # release ships one, this fails and that decision gets revisited rather than silently
    # taking a branch that was never exercised.
    assert not (RELEASE_CHECKPOINT / "hf_quant_config.json").exists()


@requires_release_checkpoint
def test_release_config_exposes_every_text_field_off_the_composite():
    """All 59 `text_config` keys must be readable on a **bare** composite.

    The allowlist this replaced covered 30 of them, and the failure mode is
    silent: `modeling_deepseekv4.py` reads `getattr(config, "swiglu_limit", None)`,
    so a forgotten name disables the asymmetric SwiGLU clamp with nothing raising.
    The five names asserted individually below are the read sites that a partial
    allowlist actually broke.

    Deliberately `DeepseekV41Config.from_pretrained`, not `ModelConfig.from_pretrained`:
    the latter runs `_mirror_text_subconfig_attrs` (`model_config.py:1643`), which copies
    every public text field onto the parent. Reachability measured *after* the mirror is
    0-unreachable for any implementation, including one whose own forwarding covers 30 of
    59 keys, so it cannot distinguish this build from the one it replaced.
    """
    raw = json.loads((RELEASE_CHECKPOINT / "config.json").read_text())
    config = DeepseekV41Config.from_pretrained(RELEASE_CHECKPOINT)
    # `AutoConfig` must land on the same class -- that is the path production takes.
    assert type(AutoConfig.from_pretrained(RELEASE_CHECKPOINT)) is type(config)
    assert isinstance(config, DeepseekV41Config)

    unreachable = [key for key in raw["text_config"] if not hasattr(config, key)]
    assert not unreachable, f"{len(unreachable)} text_config field(s) unreachable: {unreachable}"

    # `model_type` is the one field both levels legitimately own and disagree on:
    # the composite is `deepseek_v41` (what CONFIG_MAPPING keys off) and the text
    # tower is `deepseek_v41_text`. Forwarding must not shadow the composite's own,
    # so its value is checked separately rather than against the sub-config.
    assert config.model_type == raw["model_type"] == "deepseek_v41"
    assert config.text_config.model_type == raw["text_config"]["model_type"]

    for key, value in raw["text_config"].items():
        if key == "model_type":
            continue
        expected = getattr(config.text_config, key)
        if isinstance(expected, (list, tuple)):
            assert list(getattr(config, key)) == list(expected), key
        else:
            assert getattr(config, key) == expected, key

    assert config.swiglu_limit == raw["text_config"]["swiglu_limit"] is not None
    assert config.routed_scaling_factor == raw["text_config"]["routed_scaling_factor"]
    assert config.hc_sinkhorn_iters == raw["text_config"]["hc_sinkhorn_iters"]
    # `engram_vocab_size` is the one forwarded field whose *type* the config
    # deliberately changes, so it is checked against both meanings rather than
    # against the raw scalar. config.json publishes the scalar per-bucket target
    # the prime search starts from; `EngramConfig.engram_vocab_size` is indexed
    # per n-gram order (`vocab_size_per_ngram[ngram - 2]`, engram.py:262). The
    # scalar survives under `engram_bucket_vocab_size` and the forwarded field is
    # its broadcast over the `engram_max_ngram_size - 1` orders. Asserting the raw
    # scalar here would pin the wrong one of the two.
    assert config.engram_bucket_vocab_size == raw["text_config"]["engram_vocab_size"]
    assert config.engram_vocab_size == [raw["text_config"]["engram_vocab_size"]] * (
        raw["text_config"]["engram_max_ngram_size"] - 1
    )
    assert config.engram_max_ngram_size == raw["text_config"]["engram_max_ngram_size"]

    # Delegation, not duplication: the composite must not have copied the text
    # fields into its own __dict__, or the two levels can diverge under mutation
    # and `to_dict()` serializes each field twice.
    copied = sorted(set(raw["text_config"]) & set(vars(config)) - {"model_type"})
    assert not copied, f"text fields copied onto the composite: {copied}"

    # And a name that is in neither config still raises rather than reading None.
    with pytest.raises(AttributeError):
        _ = config.definitely_not_a_field


@requires_release_checkpoint
def test_release_config_layer_plan_matches_the_hand_written_fixture():
    """Close the loop between the fixtures above and the real checkpoint.

    Every classification test in this file drives `release_text_config()`. This is
    what makes those tests statements about DeepSeek-V4.1-Flash rather than about a
    fixture: the checkpoint's own config must produce a byte-identical layer plan.
    """
    from tensorrt_llm._torch.model_config import ModelConfig

    pretrained_config = ModelConfig.from_pretrained(RELEASE_CHECKPOINT).pretrained_config
    assert pretrained_config.layer_plan_dump() == release_text_config().layer_plan_dump()


@requires_release_checkpoint
def test_release_checkpoint_loads_in_a_fresh_interpreter():
    """`ModelConfig.from_pretrained` works from a cold interpreter, no `trust_remote_code`.

    A subprocess rather than an in-process call because this file's own imports pull in
    `tensorrt_llm._torch.configs`, so in-process the model type is already registered and
    the assertion would be about test-module import order rather than about the loader.
    The child starts with no `tensorrt_llm` in `sys.modules`.

    What this does *not* claim: that needing that import is a V4.1-specific defect.
    `deepseek_v4` and `deepseek_v32` behave identically, and every production entry point
    imports `tensorrt_llm._torch.models`, which registers all of them
    (`_torch/models/__init__.py:18-22`). `transformers_alone` is recorded for the record,
    not asserted to be a failure -- the load-bearing assertions below are about the class,
    the architectures, the propagated quantization policy, and the resolved dtype.
    """
    child = textwrap.dedent(
        f"""
        import json, sys
        assert "tensorrt_llm" not in sys.modules

        import transformers
        try:
            transformers.AutoConfig.from_pretrained({str(RELEASE_CHECKPOINT)!r})
            transformers_alone = "loaded"
        except Exception as exc:
            transformers_alone = type(exc).__name__

        from tensorrt_llm._torch.model_config import ModelConfig
        pc = ModelConfig.from_pretrained({str(RELEASE_CHECKPOINT)!r}).pretrained_config
        print("@@" + json.dumps({{
            "transformers_alone": transformers_alone,
            "cls": type(pc).__name__,
            "architectures": pc.architectures,
            "text_architectures": pc.text_config.architectures,
            "text_quantization_config": pc.text_config.quantization_config,
            "text_dtype": str(pc.text_config.torch_dtype),
            "dtype": str(pc.torch_dtype),
            "num_descriptors": len(pc.layer_descriptors),
            "hidden_size": pc.hidden_size,
        }}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=900
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    payload = json.loads(
        next(line for line in completed.stdout.splitlines() if line.startswith("@@"))[2:]
    )

    assert payload["cls"] == "DeepseekV41Config"
    assert payload["architectures"] == ["DeepseekV41ForCausalLM"]
    assert payload["text_architectures"] == ["DeepseekV41ForCausalLM"]
    # The gate: V4.1's quantization_config is top level only, so a text tower with
    # an empty one means 475 GiB would load as unquantized bf16 with nothing raising.
    assert payload["text_quantization_config"]["quant_method"] == "fp8"
    assert payload["text_quantization_config"]["scale_fmt"] == "ue8m0"
    assert payload["text_quantization_config"]["expert_dtype"] == "fp4"
    # `dtype` is top level only; an unresolved text dtype is a None.itemsize crash
    # in the KV-cache byte sizing rather than a wrong answer.
    assert payload["dtype"] == payload["text_dtype"] == "torch.bfloat16"
    assert payload["num_descriptors"] == 43
    assert payload["hidden_size"] == 5120
    # If transformers ever learns `deepseek_v41` this stops being a discriminator;
    # say so out loud rather than letting the test quietly weaken.
    assert payload["transformers_alone"] in ("ValueError", "loaded")


@requires_release_checkpoint
@pytest.mark.parametrize(
    "name",
    [
        "layers.0.attn.wkv.weight",  # attention projection
        "layers.0.ffn.shared_experts.w1.weight",  # shared-expert FFN
        "layers.1.engram.wkv.weight",  # Engram projection
    ],
)
def test_dense_fp8_widens_to_bf16_without_loss(name):
    """One tensor per dense flavour: bf16 dequant must be bit-exact, not merely close.

    `model_config.py::_build_deepseek_v41_quant_config` loads V4.1's dense weights as
    bf16 rather than keeping them FP8, because their 32x32 block scales cannot use the
    `FP8_BLOCK_SCALES` path (which assumes 128x128). That is only acceptable if the
    widening is lossless, and it is: `float8_e4m3fn` has 3 mantissa bits against bf16's
    7, and a `float8_e8m0fnu` scale is a pure power of two, so scaling shifts the
    exponent and leaves the mantissa untouched. A nonzero delta here means the wrong
    block extent or the wrong role's layout, and would make the memory trade a silent
    accuracy trade.

    `gates/dense_dequant.py --all-patterns` runs this over all 19 dense patterns
    (565,182,464 elements); this keeps one tensor per flavour in the repo suite.
    """
    import torch
    from safetensors import safe_open

    weight_map = json.loads((RELEASE_CHECKPOINT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    scale_name = name.removesuffix(".weight") + ".scale"

    tensors = {}
    for key in (name, scale_name):
        with safe_open(RELEASE_CHECKPOINT / weight_map[key], framework="pt") as handle:
            tensors[key] = handle.get_tensor(key)
    weight, scale = tensors[name], tensors[scale_name]

    assert weight.dtype is torch.float8_e4m3fn
    assert scale.dtype is torch.float8_e8m0fnu

    layout = parse_quantization_layout(
        json.loads((RELEASE_CHECKPOINT / "config.json").read_text())["quantization_config"]
    )[DeepseekV41QuantRole.DENSE]
    block = tuple(layout.element_block)
    assert block == (32, 32)
    assert tuple(a // b for a, b in zip(weight.shape, scale.shape)) == block
    assert tuple(a % b for a, b in zip(weight.shape, scale.shape)) == (0, 0)

    scale_f32 = scale.to(torch.float32)
    # Exponent-only means every scale is exactly 2**k; anything else and the exactness
    # argument below does not hold.
    log2 = torch.log2(scale_f32[scale_f32 > 0])
    assert torch.equal(log2, log2.round())

    expanded = scale_f32.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    reference = weight.to(torch.float32) * expanded
    via_bf16 = weight.to(torch.bfloat16) * expanded.to(torch.bfloat16)

    assert torch.isfinite(reference).all()
    assert (reference - via_bf16.to(torch.float32)).abs().max().item() == 0.0
    assert (reference - reference.to(torch.bfloat16).to(torch.float32)).abs().max().item() == 0.0

    # Control: the zero above is a measurement, not an identity. Nudge the scale off a
    # power of two -- what an fp32-scale checkpoint looks like -- and bf16 must lose bits.
    nudged = expanded * 1.0000001
    lossy_ref = weight.to(torch.float32) * nudged
    lossy_bf16 = weight.to(torch.bfloat16) * nudged.to(torch.bfloat16)
    assert (lossy_ref - lossy_bf16.to(torch.float32)).abs().max().item() > 0.0
