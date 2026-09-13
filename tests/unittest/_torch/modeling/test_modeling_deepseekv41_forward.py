# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward tests for DeepSeek-V4.1.

``test_deepseek_v41_config.py`` covers the config-to-descriptor mapping on CPU and
``test_modeling_deepseekv41.py`` covers weight loading against a meta-tensor
checkpoint. This file covers the part that only a real forward pass can: V4.1's
*cross-layer* sparse-attention plumbing, which has no analogue in V4 and therefore
no existing coverage. It is kept separate from the loader suite because it is the
only V4.1 modeling test that needs a GPU.

The tiny topology below is chosen so that every V4.1 role appears at least once
while every sharing group stays ratio-uniform (an invariant the cache manager
asserts, because a group's members alias one ratio-keyed pool):

===== ===== ========== ============ =========================================
layer ratio kv source  index source role
===== ===== ========== ============ =========================================
0     0     --         --           sliding window only
1     0     --         --           sliding window only
2     2     2 (own)    2 (own)      pooled owner: compressor + indexer
3     2     2          2            reads layer 2's KV *and* its top-k
4     2     2          2            same
5     1     5 (own)    5 (own)      unpooled owner: compressor + indexer
6     1     5          5            reads layer 5's KV and top-k
7     1     5          7 (own)      index source that is *not* a kv source
===== ===== ========== ============ =========================================

Layer 7 is the interesting one and the reason this test exists: it owns an
``Indexer`` but no ``Compressor``, so its index keys come from layer 5 -- out of
the cache on the decode path (the manager aliases its ``LayerId``) and out of
``metadata.v41_index_keys`` on the single-pass prefill path. Layers 3, 4 and 6
are the other new shape: indexed layers that run no ``Indexer`` at all and
consume ``metadata.v41_topk_indices``.

Requires one GPU and no checkpoint; weights are random.
"""

import unittest.mock
import weakref
from copy import deepcopy

import pytest
import torch
from utils.util import skip_blackwell_geforce

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
    _is_fused_prologue_active,
    _is_fused_q_fp8_quant_enabled,
)
from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41Config
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Attention
from tensorrt_llm._torch.models.modeling_deepseekv41 import (
    DeepseekV41Attention,
    DeepseekV41ForCausalLM,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

# See the module docstring for what each entry buys. `num_hidden_layers` equals
# `len(compress_ratios)` because MTP is not constructed for V4.1 yet, so an MTP
# tail here would only add unverified descriptor rows.
V41_TINY_RATIOS = [0, 0, 2, 2, 2, 1, 1, 1]
V41_TINY_KV_SOURCES = [2, 5]
V41_TINY_INDEX_SOURCES = [2, 5, 7]

# Attention geometry is the released one -- 512-wide latent head, 64 of it rope --
# because the compressor and indexer kernels are specialized on those widths.
# Everything that only costs memory (vocab, expert width, hidden size) is shrunk.
# `n_routed_experts` stays at the released 384: the routing kernel is specialized
# on the expert count, so shrinking it would test a topology V4.1 never runs.
V41_TINY_TEXT_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 2048,
    "moe_intermediate_size": 128,
    "num_hidden_layers": len(V41_TINY_RATIOS),
    "num_attention_heads": 16,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 512,
    "o_lora_rank": 512,
    "o_groups": 8,
    "max_position_embeddings": 65536,
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "compress_ratios": V41_TINY_RATIOS,
    "kv_source_layer_ids": V41_TINY_KV_SOURCES,
    "index_source_layer_ids": V41_TINY_INDEX_SOURCES,
    "index_n_heads": 16,
    "index_head_dim": 128,
    "index_topk": 512,
    "sliding_window": 128,
    # The two-level candidate prefilter is a long-context-only path (it is a
    # mathematical no-op until a layer has more compressed entries than
    # `index_topk`), and layer 20 does not exist here. `None` turns it off
    # explicitly rather than leaving an out-of-range layer id in the descriptors.
    "candidate_source_layer_id": None,
    # Engram is off: its tables are sized by prime search over a 16M vocab and
    # nothing in the sparse-attention path depends on them.
    "engram_layer_ids": [],
    "engram_num_embeddings": [],
    # V4.1's three MTP layers are heterogeneous and not constructed yet;
    # `DeepseekV41ForCausalLM` refuses a `spec_config` outright.
    "num_nextn_predict_layers": 0,
    "dspark_target_layer_ids": [],
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 4.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    },
}


def _tiny_v41_model_config(*, tokens_per_block: int):
    """A V4.1 model config plus the sparse config the backend reads.

    Both are returned because the sparse config has to be handed to the cache
    manager as well, and it must be the *same object*: the manager sizes its
    ratio-keyed pools from `compress_ratios` and its aliases from the two source
    lists, so a second copy that had been through
    `normalize_compress_ratios` under the wrong variant would allocate a
    different set of pools than the layers read.
    """
    config = DeepseekV41Config(text_config=deepcopy(V41_TINY_TEXT_CONFIG))
    config.dtype = torch.bfloat16
    config.tie_word_embeddings = False
    config.mapping = Mapping(world_size=1, tp_size=1, rank=0)

    sparse_attn_config = DeepSeekV4SparseAttentionConfig(
        variant="v41",
        index_n_heads=V41_TINY_TEXT_CONFIG["index_n_heads"],
        index_head_dim=V41_TINY_TEXT_CONFIG["index_head_dim"],
        window_size=V41_TINY_TEXT_CONFIG["sliding_window"],
        compress_ratios=V41_TINY_RATIOS,
        kv_source_layer_ids=V41_TINY_KV_SOURCES,
        index_source_layer_ids=V41_TINY_INDEX_SOURCES,
        index_topk=V41_TINY_TEXT_CONFIG["index_topk"],
    )
    config.sparse_attention_config = sparse_attn_config

    model_config = ModelConfig(
        pretrained_config=config,
        sparse_attention_config=sparse_attn_config,
        attn_backend="TRTLLM",
        quant_config=QuantConfig(),
    )
    model_config.extra_attrs["kv_cache_dtype"] = "auto"
    return model_config, sparse_attn_config


def _init_random_weights(model, seed: int = 0) -> None:
    """Give every parameter a finite value.

    TRT-LLM's ``Linear`` allocates with ``torch.empty``, so a model built from a
    config and never loaded holds uninitialized memory -- which is why the V4
    sanity test can only check shapes. Filling the parameters here is what makes
    the finiteness assertions below meaningful: a NaN then means the *wiring*
    produced one (an all-masked softmax row, a division by an empty sum), not
    that the weights were garbage to begin with.

    Norms are set to 1 rather than sampled: a norm weight near zero would scale a
    whole residual stream to ~0 and turn a real NaN downstream into a silent
    zero.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for name, param in model.named_parameters():
        if not param.dtype.is_floating_point:
            continue
        if "norm" in name or name.endswith("_weight"):
            param.data.fill_(1.0)
        else:
            param.data.normal_(0.0, 0.02, generator=generator)


def _assert_v41_module_topology(model) -> None:
    """The module tree must match the two source lists, not merely be buildable.

    This is the assertion that catches the whole class of V4.1 bring-up bug where
    a layer builds a module it has no weights for: an index source that is not a
    kv source must have an ``Indexer`` and no ``Compressor``, and a
    non-source long-range layer must have neither.
    """
    for layer_idx, ratio in enumerate(V41_TINY_RATIOS):
        attn = model.model.layers[layer_idx].self_attn
        expect_compressor = layer_idx in V41_TINY_KV_SOURCES
        expect_indexer = layer_idx in V41_TINY_INDEX_SOURCES
        assert (attn.compressor is not None) == expect_compressor, (
            f"layer {layer_idx} (ratio {ratio}) compressor presence is wrong"
        )
        assert (attn.indexer is not None) == expect_indexer, (
            f"layer {layer_idx} (ratio {ratio}) indexer presence is wrong"
        )
        if expect_indexer:
            # `owns_index_keys` is what decides whether the indexer projects its
            # own keys from a latent or reads a published set. Layer 7 is the
            # only one here where it differs from `is_index_source`.
            assert attn.indexer.owns_index_keys == expect_compressor
            expected_source = max(i for i in V41_TINY_KV_SOURCES if i <= layer_idx)
            assert attn.indexer.kv_source_layer_idx == expected_source, (
                f"layer {layer_idx} scores against the wrong index-key cache"
            )


@pytest.mark.skip_less_device_memory(40000)
@skip_blackwell_geforce
def test_deepseek_v41_sanity() -> None:
    tokens_per_block = 128
    model_config, sparse_attn_config = _tiny_v41_model_config(tokens_per_block=tokens_per_block)
    config = model_config.pretrained_config
    vocab_size = config.vocab_size
    num_layers = config.num_hidden_layers

    device = torch.device("cuda")
    model = DeepseekV41ForCausalLM(model_config).to(device)
    _init_random_weights(model)
    _assert_v41_module_topology(model)
    # Cheap and total: proves every constructed layer agrees with its
    # config-derived descriptor, so a topology regression cannot hide behind a
    # forward pass that merely does not crash.
    model.model.describe_layers()

    # Two context requests and two generation requests, so a single forward pass
    # exercises the prefill and decode branches of the cross-layer handoff at
    # once. The generation requests start past `window_size` (128) so their
    # long-range branch has compressed entries to select from at all.
    context_sequence_length = [3, 5]
    num_contexts = len(context_sequence_length)
    sequence_length = context_sequence_length + [1, 1]
    past_seen_tokens = [0, 0, 200, 137]

    input_ids = torch.randint(
        0, vocab_size, (sum(sequence_length),), dtype=torch.int32, device=device
    )
    request_ids = list(range(len(sequence_length)))
    token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_length)).tolist()
    prompt_lens = token_nums[:num_contexts] + past_seen_tokens[num_contexts:]
    max_new_tokens = 256
    required_blocks = sum(
        (token_num + max_new_tokens + tokens_per_block - 1) // tokens_per_block
        for token_num in token_nums
    )
    num_blocks = max(16, required_blocks)
    max_seq_len = num_blocks * tokens_per_block
    batch_size = len(sequence_length)

    mapping = config.mapping
    kv_cache_config = KvCacheConfig(
        dtype="auto",
        enable_block_reuse=False,
        max_tokens=num_blocks * tokens_per_block,
        event_buffer_max_size=0,
    )
    kv_cache_manager = DeepseekV4CacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.BF16,
        compressor_dtype=tensorrt_llm.bindings.DataType.FLOAT,
        vocab_size=vocab_size,
        max_num_tokens=max_seq_len * batch_size,
        sparse_attn_config=sparse_attn_config,
        model_config=model_config,
    )

    reqs = []
    for i, req_id in enumerate(request_ids):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=max_new_tokens,
            input_tokens=list(range(token_nums[i])),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert kv_cache_manager.prepare_context(req), f"prepare_context failed for {req_id}"
        if i < num_contexts:
            assert kv_cache_manager.resize_context(req, req.context_chunk_size)
        else:
            # Warm-cache setup for a generation request: give it
            # `past_seen_tokens[i]` of history without running forward for it.
            kv_cache = kv_cache_manager.kv_cache_map[req.py_request_id]
            kv_cache.enable_swa_scratch_reuse = False
            target = (
                req.context_current_position + token_nums[i] + kv_cache_manager.num_extra_kv_tokens
            )
            assert kv_cache.resize(max(kv_cache.capacity, target), past_seen_tokens[i])
        reqs.append(req)

    attn_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(sequence_length, dtype=torch.int32),
        num_contexts=num_contexts,
        max_num_requests=len(sequence_length),
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=past_seen_tokens,
        ),
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        max_num_tokens=8192,
        mapping=mapping,
        sparse_attention_config=sparse_attn_config,
    )

    position_ids = []
    seq_lens = []
    for i, tokens in enumerate(past_seen_tokens):
        seq_len = context_sequence_length[i] if i < num_contexts else 1
        position_ids.append(torch.arange(tokens, tokens + seq_len, device=device))
        seq_lens.append(seq_len)
    position_ids = torch.cat(position_ids).unsqueeze(0).to(torch.int32)

    extra_attrs = model_config.extra_attrs
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        scheduled_batch = ScheduledRequests()
        scheduled_batch.context_requests_last_chunk = reqs[:num_contexts]
        scheduled_batch.generation_requests = reqs[num_contexts:]
        kv_cache_manager.prepare_resources(scheduled_batch)
        attn_metadata.prepare()
        # Both cross-layer channels are per-pass scratch and must start empty, or
        # a consumer could be satisfied by a producer that did not run this step.
        assert not attn_metadata.v41_index_keys
        assert not attn_metadata.v41_topk_indices

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
        )

        # Exactly the index sources publish a top-k, and exactly the kv sources
        # publish index keys. Checked after the forward rather than inside it so
        # this stays a statement about the plumbing, not about call order.
        assert sorted(attn_metadata.v41_topk_indices) == V41_TINY_INDEX_SOURCES
        assert sorted(attn_metadata.v41_index_keys) == V41_TINY_KV_SOURCES

        for req in reqs[:num_contexts]:
            req.context_current_position = seq_lens[req.py_request_id]
        for req in reqs:
            req.add_new_token(seq_lens[req.py_request_id], 0)
        kv_cache_manager.update_context_resources(scheduled_batch)
        kv_cache_manager.update_resources(scheduled_batch)
    assert logits.shape[0] == len(past_seen_tokens)
    assert torch.isfinite(logits).all(), "prefill produced non-finite logits"

    # Second pass: every request is now a generation request, so this is the
    # decode-only shape, where a dependent layer reads its source's index keys
    # out of the aliased cache instead of off `v41_index_keys`.
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with torch.inference_mode(), model_extra_attrs(extra_attrs):
        seq_lens = [seq_len + 1 for seq_len in seq_lens]
        scheduled_batch = ScheduledRequests()
        scheduled_batch.generation_requests = reqs
        for req in reqs:
            assert kv_cache_manager.try_allocate_generation(req)
        kv_cache_manager.prepare_resources(scheduled_batch)
        attn_metadata.prepare()
        assert not attn_metadata.v41_topk_indices, "prepare() must drop last step's handoffs"

        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            return_context_logits=True,
        )
        for req in reqs:
            req.add_new_token(seq_lens[req.py_request_id], 0)
        kv_cache_manager.update_resources(scheduled_batch)
    assert input_ids.shape == logits.shape[:-1]
    assert torch.isfinite(logits).all(), "decode produced non-finite logits"

    for req in reqs:
        kv_cache_manager.free_resources(req)
    kv_cache_manager.shutdown()


# ---------------------------------------------------------------------------
# The per-head query norm V4.1 removed
# ---------------------------------------------------------------------------
#
# V4 rescales each head of the query by its own RMS after ``q_b_proj``, with no
# learned gain (``DeepSeek-V4-Flash/inference/model.py:498``). V4.1 deleted that
# line (``DeepSeek-V4.1-Flash/inference/model.py:770-772``). The first V4.1 port
# inherited it, and the cost was not obvious: a uniform ~0.52x on q rescales every
# logit, which changes the softmax *temperature* rather than the output magnitude.
# The result stays a convex combination of the same value vectors, so its norm
# barely moves while its direction shifts -- measured against a captured reference
# forward, layer 0's attention output sat at rel 0.58 / cos 0.82 while every operand
# feeding the kernel was exact to 0.002, and the model emitted fluent but wrong
# text. GSM8K read 19/20 after removing the norm.
#
# So these tests are deliberately about the *switch*, not about numerics: the
# numerics are covered by the reference-parity work, and what regresses in code is
# someone re-deriving ``q_b_layernorm`` unconditionally during a refactor. A test
# that only asserted the class attribute would restate the source; each of these
# instead exercises the place the flag is consumed.


@pytest.mark.skip_less_device_memory(40000)
@skip_blackwell_geforce
def test_v41_builds_no_per_head_query_norm_and_the_flag_is_what_decides() -> None:
    """Every V4.1 attention layer must have no ``q_b_layernorm``, because of the flag.

    Building twice is the point. The first build pins V4.1's behaviour; the second
    flips only ``q_b_norm_enabled`` and must produce the module again. If a
    refactor hardcodes either answer, exactly one of the two assertions fails, and
    the failure names which direction was hardcoded.
    """
    model_config, _ = _tiny_v41_model_config(tokens_per_block=128)
    model = DeepseekV41ForCausalLM(model_config).to(torch.device("cuda"))
    offenders = [
        idx
        for idx, layer in enumerate(model.model.layers)
        if getattr(layer.self_attn, "q_b_layernorm", None) is not None
    ]
    assert not offenders, (
        f"layers {offenders} built a per-head query norm; V4.1 removed it, and an "
        f"inherited one costs ~0.52x on every query"
    )
    del model
    torch.cuda.empty_cache()

    assert DeepseekV41Attention.q_b_norm_enabled is False
    # V4 keeps it: the flag defaults to the base class's behaviour so that a
    # subclass which forgets to state it stays V4-correct.
    assert DeepseekV4Attention.q_b_norm_enabled is True

    with unittest.mock.patch.object(DeepseekV41Attention, "q_b_norm_enabled", True):
        patched_config, _ = _tiny_v41_model_config(tokens_per_block=128)
        patched = DeepseekV41ForCausalLM(patched_config).to(torch.device("cuda"))
        norms = [getattr(layer.self_attn, "q_b_layernorm", None) for layer in patched.model.layers]
        assert all(norm is not None for norm in norms), (
            "flipping q_b_norm_enabled did not restore the norm, so "
            "initialize_sparse_attn no longer reads the flag"
        )
        norm = norms[0]
        # Shape and gainlessness are the parts that made the bug expensive: a
        # per-*head* norm over qk_head_dim, with no learned weight to absorb it.
        attn = patched.model.layers[0].self_attn
        # ``has_weights=False`` does not give ``weight is None``: RMSNorm registers a
        # non-persistent all-ones *buffer* under that name. Asserting ``is None`` here
        # pinned a spelling the implementation never had, so it failed for a reason
        # unrelated to the property it meant to protect. Gainlessness is the property,
        # and these are the three ways it is actually observable.
        assert not isinstance(norm.weight, torch.nn.Parameter), (
            "V4's query norm has no learned gain; a Parameter here means one was added"
        )
        assert bool(torch.all(norm.weight == 1)), "the gain must be identity"
        assert "weight" not in norm.state_dict(), (
            "the gain must stay out of the state dict, or a checkpoint could load one"
        )
        # Likewise not `norm.hidden_size`: RMSNorm does not keep the constructor arg.
        # The normalized width is observable as the gain's own width.
        assert norm.weight.shape[0] == attn.qk_head_dim, (
            f"the query norm normalizes over {norm.weight.shape[0]}, not the per-head "
            f"{attn.qk_head_dim}; a norm over the full projection is a different op"
        )
        del patched
        torch.cuda.empty_cache()


@pytest.mark.skip_less_device_memory(40000)
@skip_blackwell_geforce
def test_the_fused_fp8_query_prologue_refuses_to_run_without_the_norm() -> None:
    """``deepseek_v4_q_norm_fused_fp8`` always normalizes, so V4.1 must not reach it.

    This is the coupling that is easy to lose. The fused kernel folds norm + RoPE +
    FP8 quant into one launch and has no "skip the norm" flag, so a model without
    the per-head query norm cannot use it -- and because
    ``_is_fused_prologue_active`` requires the Q and KV folds together, the gate has
    to be checked on the Q side or the KV fold silently leaves a raw latent behind.
    Restoring the fused path for V4.1 is a C++ change; until then this must hold.
    """
    model_config, _ = _tiny_v41_model_config(tokens_per_block=128)
    model = DeepseekV41ForCausalLM(model_config).to(torch.device("cuda"))
    attn = model.model.layers[0].self_attn
    assert attn.q_b_layernorm is None
    assert _is_fused_q_fp8_quant_enabled(attn, num_generations=0, num_contexts=1) is False
    assert _is_fused_q_fp8_quant_enabled(attn, num_generations=1, num_contexts=0) is False
    assert (
        _is_fused_prologue_active(attn, rope_specs=[object()], num_generations=1, num_contexts=0)
        is False
    )
    del model
    torch.cuda.empty_cache()
