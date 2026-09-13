# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DeepseekV41WeightLoader``'s three load-time audits, on the real checkpoint.

V4.1's checkpoint is 96085 tensors / 475 GiB across 48 shards, and V4's loader
walks it without ever asking whether it consumed all of it. Three things can go
wrong silently:

* a tensor nobody loads (a renamed module, a variant nobody wired) is dropped,
* a parameter nobody fills keeps its uninitialized value,
* a quantized weight is read with the wrong block extent -- the expensive one,
  because MXFP4's ``1 x 32`` element block packs two e2m1 per byte and therefore
  *stores* ``1 x 16``, which is exactly NVFP4's element block. Reading the stored
  extent as if it were the element extent is a silent 2x scale-stride error.

So ``DeepseekV41WeightLoader`` adds a layout check, a census, and a coverage
audit, and all three raise. This file drives them from the released checkpoint.

The audits are decided entirely by tensor *names, shapes and dtypes*, and all
three of those live in the safetensors headers -- so the fixtures read the 48
headers and materialize each tensor as a zero-cost ``meta`` tensor of its true
shape and dtype. That is the same signal a 475 GiB load would produce, minus
numerics: ``_dequantize_block32`` is a Triton kernel and cannot run on meta
storage, so it is stubbed. Dequantization exactness is covered separately by
``test_deepseek_v41_config.py::test_dense_fp8_widens_to_bf16_without_loss``.
"""

import dataclasses
import inspect
import json
import struct
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.configs import deepseek_v41 as v41_config
from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41QuantRole
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_deepseekv41 as v41
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate
from tensorrt_llm.mapping import Mapping

RELEASE_CHECKPOINT = Path("/code/llm-models/DeepSeek-V4.1-Flash")
RELEASE_TENSOR_COUNT = 96085
RELEASE_SHARDS = 48

requires_release_checkpoint = pytest.mark.skipif(
    not (RELEASE_CHECKPOINT / "config.json").is_file(),
    reason=f"release checkpoint not present at {RELEASE_CHECKPOINT}",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the attention modules are built on the GPU"
)

# safetensors dtype tag -> torch dtype name. e2m1 has no torch dtype, which is
# why the checkpoint stores packed experts as ``I8``.
_SAFETENSORS_DTYPES = {
    "F8_E4M3": "float8_e4m3fn",
    "F8_E8M0": "float8_e8m0fnu",
    "I8": "int8",
    "U8": "uint8",
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}

# Only size-bearing dims, and only enough to construct 40 real layers on one GPU.
# No per-layer role is touched, so layer i keeps its real compress ratio, its real
# pooling flags and its real Engram membership -- which is what decides how many
# tensors each layer claims.
_REDUCTIONS = {
    "vocab_size": 1024,
    "n_routed_experts": 8,
    "moe_intermediate_size": 256,
    "engram_vocab_size": 4096,
}


def _reduce(before, value):
    if isinstance(before, (list, tuple)):
        return type(before)(value for _ in before)
    return value


@pytest.fixture(scope="module")
def release_meta_weights():
    """Every released tensor as a ``meta`` tensor of its true shape and dtype."""
    weights = {}
    shards = sorted(RELEASE_CHECKPOINT.glob("model-*-of-*.safetensors"))
    assert len(shards) == RELEASE_SHARDS, f"expected {RELEASE_SHARDS} shards, saw {len(shards)}"
    for shard in shards:
        with open(shard, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            dtype = getattr(torch, _SAFETENSORS_DTYPES[entry["dtype"]])
            weights[key] = torch.empty(tuple(entry["shape"]), dtype=dtype, device="meta")
    assert len(weights) == RELEASE_TENSOR_COUNT
    return weights


def _build_release_model():
    """A really-constructed 40-layer ``DeepseekV41ForCausalLM`` at real geometry.

    A function and not only a fixture because the dense-fp8 switch is read *inside*
    ``ModelConfig.from_pretrained``: a test that needs the fp8 arm cannot reuse the
    module-scoped model, which was frozen on the bf16 path when it was first built.
    """
    # ModelConfig freezes itself at the end of from_pretrained, so both the mapping
    # and the sequence ceiling have to be handed in rather than assigned afterwards.
    # The model refuses to construct without a ceiling it can serve exactly, because
    # the two-level candidate prefilter is not implemented; the ceiling is
    # ``candidate_topk_blocks * candidate_block_size``, read off the checkpoint so
    # this test keeps agreeing with whatever the release publishes.
    raw = json.loads((RELEASE_CHECKPOINT / "config.json").read_text())
    raw_text = raw.get("text_config", raw)
    inert_up_to = int(raw_text.get("candidate_topk_blocks", 0) or 0) * int(
        raw_text.get("candidate_block_size", 0) or 0
    )
    assert inert_up_to > 0, "the release publishes a two-level candidate prefilter"
    model_config = ModelConfig.from_pretrained(
        str(RELEASE_CHECKPOINT),
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1, rank=0),
        max_seq_len=inert_up_to,
    )
    text = getattr(model_config.pretrained_config, "text_config", model_config.pretrained_config)
    for name, value in _REDUCTIONS.items():
        setattr(text, name, _reduce(getattr(text, name), value))
    text.__dict__.pop("_layer_descriptors", None)

    # Construct on the host: a ``torch.device("cuda")`` default-device context
    # breaks YaRN RoPE table creation, which calls ``.numpy()``.
    torch.cuda.set_device(0)
    return v41.DeepseekV41ForCausalLM(model_config)


@pytest.fixture(scope="module")
def release_model():
    """``_build_release_model()``, paid once for the whole file."""
    return _build_release_model()


def _audited_loader(model, weights, monkeypatch):
    """Run the loader's audits over ``weights``, returning the loader.

    Numerics are stubbed (see the module docstring) and the ``attn_sink``
    parameters V4's walk *assigns* rather than declares are stood up, because the
    coverage audit runs after the walk in a real load but before it here.

    Where they are stood up matters, and getting it wrong is why a real 8-way load
    failed the coverage audit while this file was green: the checkpoint keys them
    on the attention module (``self_attn.attn_sink``) but both of the walk's
    branches register them on the inner implementation, so the parameter is
    ``self_attn.mqa.attn_sink``. Creating them under the checkpoint's own name
    instead makes the audit's ``_V41_RELOCATED_PARAMS`` rewrite look unnecessary
    and hides its absence. So this mirrors the walk rather than the key.
    """
    monkeypatch.setattr(
        v41,
        "_dequantize_block32",
        lambda weight, scale: torch.empty(tuple(weight.shape), dtype=torch.bfloat16, device="meta"),
    )

    loader = v41.DeepseekV41WeightLoader(model)
    forwarded = loader.remap_checkpoint_keys(
        dict(weights),
        num_hidden_layers=model.config.num_hidden_layers,
        kv_lora_rank=model.config.kv_lora_rank,
    )
    for name, module in model.named_modules():
        key = f"{name}.attn_sink"
        if key not in forwarded:
            continue
        # The walk relocates the sinks onto `mqa`; refuse to invent a home for
        # them rather than quietly falling back to the parent and re-hiding the
        # mismatch this helper exists to expose.
        target = getattr(module, "mqa", None)
        assert target is not None, f"{name} holds an attn_sink key but has no mqa child"
        # `attn_sink` is *preset to None* by the attention module's __init__ so that
        # forward can skip the sinks when the checkpoint has none -- so `hasattr` is
        # already true here and testing it would silently create nothing.
        if getattr(target, "attn_sink", None) is None:
            target.attn_sink = torch.nn.Parameter(
                torch.empty(tuple(forwarded[key].shape), dtype=torch.float32, device="meta"),
                requires_grad=False,
            )
    loader.assert_load_complete()
    return loader


# --------------------------------------------------------------------------- #
# Name rules. These are pure, so they are also the cheapest place to pin down
# *why* a key is claimed -- the release fixtures below only prove that it is.
# --------------------------------------------------------------------------- #


def test_structural_key_rewrites_are_exactly_the_walk_s_two_renames():
    """Flat mHC names become structured; a fused sub-module takes the fused name."""
    assert v41._v41_structural_key("model.layers.3.hc_attn_fn") == "model.layers.3.hc_attn.fn"
    assert v41._v41_structural_key("model.layers.3.hc_ffn_base") == "model.layers.3.hc_ffn.base"
    assert v41._v41_structural_key("model.layers.3.hc_head_scale") == "model.layers.3.hc_head.scale"

    # ``_fn`` only splits off a real mHC stem, never an arbitrary module.
    assert v41._v41_structural_key("model.layers.3.mlp.act_fn") == "model.layers.3.mlp.act_fn"

    for source in ("gate_proj", "up_proj"):
        assert (
            v41._v41_structural_key(f"model.layers.0.mlp.shared_experts.{source}.weight")
            == "model.layers.0.mlp.shared_experts.gate_up_proj.weight"
        )
    assert (
        v41._v41_structural_key("model.layers.0.self_attn.q_a_proj.weight")
        == "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    )

    # `attn_sink` is *not* rewritten even though the walk relocates it: `mqa` is a
    # plain attribute rather than a sub-module, so no name reaches the destination
    # and `_v41_load_coverage` resolves it as an attribute instead. Rewriting here
    # would produce a name that can never match and re-hide the mismatch.
    assert (
        v41._v41_structural_key("model.layers.0.self_attn.attn_sink")
        == "model.layers.0.self_attn.attn_sink"
    )

    # A name that is already the model's is left alone.
    assert (
        v41._v41_structural_key("model.layers.0.self_attn.kv_a_proj_with_mqa.weight")
        == "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    )


@pytest.mark.parametrize(
    "key,pattern",
    [
        ("vision.blocks.0.attn.qkv.weight", "vision.* | aligner.*"),
        ("aligner.proj.weight", "vision.* | aligner.*"),
        ("image_newline", "image_start | image_end | image_newline"),
        ("mtp.0.embed_tokens.weight", "mtp.*"),
        ("mtp.2.shared_head.norm.weight", "mtp.*"),
    ],
)
def test_every_deliberate_drop_reports_a_pattern_and_a_reason(key, pattern):
    reason = v41._v41_ignore_reason(key)
    assert reason is not None, f"{key} would be forwarded, not ignored"
    assert reason[0] == pattern
    assert reason[1].strip(), "an ignored tensor must carry a reason, not just a pattern"


def test_a_loaded_tensor_is_never_reported_as_ignored():
    for key in (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.bias",
        "model.layers.1.engram.multi_head_embedding.embed_2.weight",
        "lm_head.weight",
    ):
        assert v41._v41_ignore_reason(key) is None, key


def test_the_per_token_gate_bias_is_dropped_and_the_drop_stays_lossless():
    """``ffn.gate.bias_vl`` is a *deliberate* drop, and the drop is provably inert.

    The checkpoint ships one ``bias_vl`` per MoE layer -- 40 text layers plus the 3
    MTP layers, 43 in all -- because ``inference/model.py:809`` allocates it under
    ``args.vision_enabled``, which the released config sets. The routing bias is
    *replaced* per token rather than added to:
    ``bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)``
    (``inference/model.py:818``). So on a text-only path, where ``image_mask`` is
    never set, the resolved bias is bitwise ``bias`` and ``bias_vl`` cannot affect a
    single routing decision.

    That is why the loader drops it with a reason instead of forwarding it: with
    ``has_vl_bias=False`` there is no parameter to forward *to*, and a forwarded key
    with no destination is how a silent unfilled parameter gets created.

    The drop is only sound while the parameter is genuinely absent, so this test
    pins both halves. If someone enables the vision bias without removing the drop
    rule, the second assertion fails and names the coupling -- otherwise the gate
    would allocate ``e_score_correction_bias_vl``, the loader would still ignore its
    checkpoint tensor, and the parameter would route off uninitialized memory the
    moment image tokens appeared.
    """
    reason = v41._v41_ignore_reason("layers.7.ffn.gate.bias_vl")
    assert reason is not None, "bias_vl must be reported as a deliberate drop, not dropped silently"
    pattern, why = reason
    assert pattern == "layers.<i>.ffn.gate.bias_vl"
    # The reason has to state *why* it is safe, not merely that it is dropped.
    assert "bitwise unchanged" in why

    # The bias routing actually reads is never ignored.
    assert v41._v41_ignore_reason("layers.7.ffn.gate.bias") is None

    # The coupling: dropping the tensor is only lossless while no parameter wants it.
    # V4.1 inherits the decoder layer's gate construction from V4, so that is where
    # the switch lives.
    from tensorrt_llm._torch.models import modeling_deepseekv4 as v4

    src = inspect.getsource(v4)
    assert "has_vl_bias=False" in src, (
        "the gate now builds e_score_correction_bias_vl, but _v41_ignore_reason still "
        "drops layers.<i>.ffn.gate.bias_vl -- the parameter would stay unfilled. Enable "
        "the vision bias and its checkpoint key together, or neither."
    )

    # An MTP layer's copy is doubly unclaimed -- inert vision bias *and* unbuilt
    # layer -- and the ``bias_vl`` rule is tested first, so that is the reason it
    # gets attributed to. Either attribution is a deliberate drop and the census
    # counts it exactly once, so the ordering is harmless; it is asserted so that a
    # future reorder is a visible decision rather than a silent one.
    mtp = v41._v41_ignore_reason("mtp.1.ffn.gate.bias_vl")
    assert mtp is not None and mtp[0] == "layers.<i>.ffn.gate.bias_vl"
    # A non-bias_vl MTP tensor still lands on the unbuilt-MTP rule.
    other = v41._v41_ignore_reason("mtp.1.ffn.gate.weight")
    assert other is not None and other[0] == "mtp.*"

    # The sibling that *is* forwarded still lands on the parameter the gate builds.
    forwarded = v41._remap_deepseek_v41_checkpoint_keys(
        {"layers.7.ffn.gate.bias": torch.zeros(4)},
        num_hidden_layers=40,
    )
    assert set(forwarded) == {"model.layers.7.mlp.gate.e_score_correction_bias"}
    assert forwarded.census.ignored_count == 0


@pytest.mark.parametrize("has_vl_bias", [False, True])
def test_the_gate_builds_the_vision_bias_only_when_asked(has_vl_bias):
    """And loading it leaves the static bias -- the one routing reads -- untouched.

    ``bias_vl`` *replaces* ``bias`` for image tokens rather than adding to it, so
    a text-only batch must route off bit-identical ``e_score_correction_bias``
    values no matter whether the second vector was loaded.
    """
    gate = DeepseekV4Gate(
        hidden_size=8,
        num_experts=4,
        top_k=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.5,
        is_hashed=False,
        dtype=torch.bfloat16,
        has_vl_bias=has_vl_bias,
    )
    assert (gate.e_score_correction_bias_vl is not None) is has_vl_bias

    bias = torch.tensor([0.25, -0.5, 1.0, -2.0])
    weights = {"weight": torch.zeros(4, 8, dtype=torch.bfloat16), "e_score_correction_bias": bias}
    if has_vl_bias:
        weights["e_score_correction_bias_vl"] = torch.tensor([-9.0, -9.0, -9.0, -9.0])

    gate.load_weights([weights])
    assert torch.equal(gate.e_score_correction_bias, bias.to(torch.float32))
    assert torch.equal(gate.routing_method.e_score_correction_bias, bias.to(torch.float32))
    if has_vl_bias:
        assert torch.equal(gate.e_score_correction_bias_vl, weights["e_score_correction_bias_vl"])


def test_keeping_one_mtp_layer_ignores_only_the_rest():
    assert v41._v41_ignore_reason("mtp.0.eh_proj.weight", keep_mtp_layers=1) is None
    kept = v41._v41_ignore_reason("mtp.1.eh_proj.weight", keep_mtp_layers=1)
    assert kept is not None and kept[0] == "mtp.<i>.* for i >= 1"


def test_census_verify_rejects_a_count_that_does_not_close():
    census = v41.DeepseekV41LoadCensus(total=10, folded=2, forwarded=3)
    census.ignore("a", ("pattern", "because"))
    with pytest.raises(ValueError, match="does not close"):
        census.verify()

    census.forwarded = 7
    census.verify()
    assert census.consumed == 9 and census.ignored_count == 1
    rendered = census.render()
    assert "consumed + ignored == 10" in rendered and "because" in rendered


def test_forwarded_weights_separates_forwarded_from_synthesized_keys():
    forwarded = v41._V41ForwardedWeights({"a": 1}, census=v41.DeepseekV41LoadCensus(total=1))
    forwarded["b"] = 2
    forwarded.update({"c": 3, "a": 4})
    assert forwarded.forwarded_keys == frozenset({"a"})
    assert forwarded.synthesized_keys == {"b", "c"}
    assert forwarded.all_keys == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# The release checkpoint.
# --------------------------------------------------------------------------- #


@requires_cuda
@requires_release_checkpoint
def test_release_census_closes_and_coverage_is_empty(
    release_model, release_meta_weights, monkeypatch
):
    """The audit accepts the released checkpoint, and accounts for all of it."""
    loader = _audited_loader(release_model, release_meta_weights, monkeypatch)
    census = loader.census

    assert census.total == RELEASE_TENSOR_COUNT
    assert census.consumed + census.ignored_count == RELEASE_TENSOR_COUNT
    assert census.consumed > 0 and census.ignored_count > 0

    # Every ignored group carries both a pattern and a reason, and nothing is
    # ignored by accident: the multimodal tower, its three separator embeddings, the
    # per-token vision router bias, and the unbuilt MTP layers. Why the vision bias
    # is a *separate* group rather than folded into the tower, and why dropping it
    # is provably lossless, is
    # `test_the_per_token_gate_bias_is_dropped_and_the_drop_stays_lossless`.
    assert {pattern for pattern, _ in census.ignored} == {
        "vision.* | aligner.*",
        "image_start | image_end | image_newline",
        "layers.<i>.ffn.gate.bias_vl",
        "mtp.*",
    }
    for pattern, why in census.ignored:
        assert why.strip(), pattern

    # The vision-bias group must contain exactly the checkpoint's `bias_vl` tensors
    # and nothing else. Asserting the membership and not just the pattern is what
    # makes this catch the failure mode that matters: a too-greedy match that
    # swallowed `ffn.gate.bias` as well would leave the pattern set identical and
    # close the census just as neatly, while silently stripping the bias that
    # routing actually reads. The expected set is derived from the checkpoint rather
    # than hardcoded, so it keeps agreeing with whatever the release ships.
    vl_group = next(
        names
        for (pattern, _), names in census.ignored.items()
        if pattern == "layers.<i>.ffn.gate.bias_vl"
    )
    expected_vl = {k for k in release_meta_weights if k.endswith("ffn.gate.bias_vl")}
    assert set(vl_group) == expected_vl, (
        f"the vision-bias drop claimed {len(vl_group)} tensors, the checkpoint ships "
        f"{len(expected_vl)}; symmetric difference "
        f"{set(vl_group) ^ expected_vl}"
    )
    assert expected_vl, "the release is expected to ship a per-token vision router bias"
    # The static bias is never swept up with it.
    all_ignored = {name for names in census.ignored.values() for name in names}
    swept = {k for k in all_ignored if k.endswith("ffn.gate.bias") and not k.startswith("mtp.")}
    assert not swept, f"the bias routing reads was ignored for {sorted(swept)[:5]}"

    # The layout check is the expensive audit; prove it actually ran, on more
    # pairs than the dense tail alone.
    assert census.checked_pairs > 40000, census.checked_pairs
    assert census.folded > 0 and census.forwarded > census.folded

    unexpected, unfed = v41._v41_load_coverage(release_model, loader._forwarded.all_keys)
    assert unexpected == [] and unfed == []


@requires_cuda
@requires_release_checkpoint
def test_a_sink_the_walk_failed_to_deliver_is_not_silently_accepted(
    release_model, release_meta_weights, monkeypatch
):
    """The one attribute-resolved claim in the audit is a real check, not a waiver.

    ``attn_sink`` is the only checkpoint tensor with no reachable model name -- the
    walk assigns it onto ``self_attn.mqa``, which is not an ``nn.Module``. The audit
    therefore resolves it as an attribute *and* requires it to be populated. If that
    requirement were only "``mqa`` exists", a walk that dropped every sink would load
    clean and the kernel would run without sinks: wrong logits, no error. So clear
    them and prove the audit notices.
    """
    loader = _audited_loader(release_model, release_meta_weights, monkeypatch)
    sinks = [
        getattr(module, "mqa")
        for name, module in release_model.named_modules()
        if f"{name}.attn_sink" in loader._forwarded
    ]
    assert sinks, "no attention module claims an attn_sink key"
    assert all(getattr(inner, "attn_sink", None) is not None for inner in sinks)

    # Undone by the next `_audited_loader` call, which repopulates any sink it finds
    # cleared -- but restore anyway so an assertion failure here cannot leak state
    # into the rest of the module-scoped fixture's lifetime.
    saved = [inner.attn_sink for inner in sinks]
    for inner in sinks:
        inner.attn_sink = None
    try:
        with pytest.raises(ValueError, match="no module would load"):
            loader.assert_load_complete()
        unexpected, _ = v41._v41_load_coverage(release_model, loader._forwarded.all_keys)
        assert len(unexpected) == len(sinks)
        assert all(key.endswith(".attn_sink") for key in unexpected), unexpected[:4]
    finally:
        for inner, sink in zip(sinks, saved):
            inner.attn_sink = sink


def _quant_config(model):
    """The ``quantization_config`` dict the loader will actually consult."""
    pretrained = model.model_config.pretrained_config
    text = getattr(pretrained, "text_config", pretrained)
    return text.quantization_config


@requires_cuda
@requires_release_checkpoint
@pytest.mark.parametrize(
    "role,block,stored",
    [
        # 16 is what the experts *store* (two e2m1 per byte); reading it as the
        # element block is the silent 2x scale-stride bug this check exists for.
        (DeepseekV41QuantRole.EXPERT, (1, 16), (1, 8)),
        (DeepseekV41QuantRole.EXPERT, (1, 64), (1, 32)),
        (DeepseekV41QuantRole.ENGRAM_EMBED, (1, 16), (1, 16)),
    ],
)
def test_loader_refuses_a_wrong_built_in_block_extent(
    release_model, release_meta_weights, monkeypatch, role, block, stored
):
    """Corrupt one role's built-in expectation; the load must fail loudly, not warn."""
    expected = v41_config._EXPECTED_LAYOUT[role]
    assert expected.element_block != block
    corrupted = dataclasses.replace(expected, element_block=block)
    # ``stored_block`` is derived, so corrupting the element extent moves the
    # on-disk expectation with it -- which is the whole point: the two are only
    # equal for the unpacked roles.
    assert corrupted.stored_block == stored
    monkeypatch.setitem(v41_config._EXPECTED_LAYOUT, role, corrupted)

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, release_meta_weights, monkeypatch)

    message = str(excinfo.value)
    assert f"role {role!r}" in message
    assert f"expects on-disk block {stored}" in message
    assert "derives" in message
    if expected.pack_factor > 1:
        # Only the packed role can hide a stored/element gap, so only it has to
        # spell both extents out in the failure.
        assert f"Elements per scale are {block}" in message


@requires_cuda
@requires_release_checkpoint
@pytest.mark.parametrize("block", [(128, 128), (1, 32), (64, 64)])
def test_loader_refuses_a_wrong_published_dense_block(
    release_model, release_meta_weights, monkeypatch, block
):
    """Dense is the one role the checkpoint itself gets to specify.

    ``quantization_config.weight_block_size`` overrides the built-in dense entry,
    so corrupting the built-in cannot move the dense check while the checkpoint
    publishes a block -- this is the vector that can, and 128x128 is specifically
    V4's block, i.e. the value a V4-shaped assumption would supply.
    """
    quant = _quant_config(release_model)
    assert tuple(quant["weight_block_size"]) == (32, 32)
    monkeypatch.setitem(quant, "weight_block_size", list(block))

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, release_meta_weights, monkeypatch)
    message = str(excinfo.value)
    assert f"role {DeepseekV41QuantRole.DENSE!r}" in message
    assert f"expects on-disk block {block}" in message


@requires_cuda
@requires_release_checkpoint
def test_the_built_in_dense_block_is_the_fallback_and_is_itself_checked(
    release_model, release_meta_weights, monkeypatch
):
    """Neither dense expectation source is unchecked.

    Drop the published block and the built-in takes over -- and it agrees with the
    checkpoint, so the load still passes. Corrupt the built-in *then*, and the load
    fails. So a checkpoint that publishes nothing is still audited.
    """
    quant = _quant_config(release_model)
    monkeypatch.delitem(quant, "weight_block_size")
    loader = _audited_loader(release_model, release_meta_weights, monkeypatch)
    assert loader.census.checked_pairs > 40000

    role = DeepseekV41QuantRole.DENSE
    monkeypatch.setitem(
        v41_config._EXPECTED_LAYOUT,
        role,
        dataclasses.replace(v41_config._EXPECTED_LAYOUT[role], element_block=(128, 128)),
    )
    with pytest.raises(ValueError, match=r"expects on-disk block \(128, 128\)"):
        _audited_loader(release_model, release_meta_weights, monkeypatch)


@requires_cuda
@requires_release_checkpoint
# Raw checkpoint names, not model names: the checkpoint ships DeepSeek's own
# ``layers.N.attn.w*`` spelling and the remap renames it.
@pytest.mark.parametrize("dropped", ["attn.wq_b", "attn.wo_b", "hc_attn_fn"])
def test_loader_refuses_to_leave_a_parameter_unfilled(
    release_model, release_meta_weights, monkeypatch, dropped
):
    """Hide a checkpoint tensor; the parameter it fed must be reported by name."""
    weights = {k: v for k, v in release_meta_weights.items() if dropped not in k}
    assert len(weights) < len(release_meta_weights)

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, weights, monkeypatch)
    assert "no checkpoint tensor would fill" in str(excinfo.value)


@requires_cuda
@requires_release_checkpoint
def test_loader_refuses_a_tensor_no_module_would_load(
    release_model, release_meta_weights, monkeypatch
):
    """A key that is neither ignorable nor addressable must be named, not dropped."""
    weights = dict(release_meta_weights)
    weights["layers.0.attn.wq_c.weight"] = torch.empty(8, dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError) as excinfo:
        _audited_loader(release_model, weights, monkeypatch)
    message = str(excinfo.value)
    assert "no module would load" in message
    assert "wq_c" in message or "q_c_proj" in message
    # The census rides along with the failure, so a rejection is diagnosable.
    assert "raw checkpoint tensors" in message


@requires_cuda
@requires_release_checkpoint
def test_v41_every_fp8_consumer_receives_its_scale(release_meta_weights, monkeypatch):
    """Under ``TRTLLM_V41_DENSE_FP8=1``, no fp8 module is left without its scale.

    This is the audit the coverage check structurally cannot perform, and it exists
    because a real bug got through: ``layers.<i>.attn.indexer.wq_b`` was missing from
    ``_V41_REQUANT_STEM_TAILS``, so the remap dequantized it to bf16 and folded its
    scale away while the module -- a ``Linear`` handed the ``quant_config`` *object*,
    so unreachable by any ``exclude_modules`` name -- allocated fp8 and a ``(32, 10)``
    ``weight_scale``. The weight was then cast bf16 -> fp8 unscaled, clipping every
    value above 448, against a scale tensor that was still ``torch.empty``.

    Three things independently declined to report it, which is why this is a test and
    not a comment:

    * ``_v41_load_coverage`` trusts a consumer that received *any* key to fill *all*
      its parameters. That prefix rule is required -- a ``Linear`` fuses a TP shard,
      ``ConfigurableMoE`` stacks 384 experts into ``w3_w1_weight`` -- and it is exactly
      wrong for a module fed one half of a pair.
    * ``Linear``'s block-scale load is ``if scale_name in weights[0]``
      (``modules/linear.py:1276``), so a missing ``weight_scale_inv`` is a skipped
      branch rather than a ``KeyError``.
    * a short-prompt end-to-end smoke test passed on this configuration and produced
      fluent, physically correct text in both languages asked. At ~146 total sequence
      length the indexer's top-k budget covers the whole sequence, so its ranking is a
      no-op and its scores never reach the output.

    **The invariant is pairing, not naming.** The obvious form -- "for module ``M``
    with an fp8 ``weight`` and a ``weight_scale``, some forwarded key is named
    ``M.weight_scale_inv``" -- false-positives on every fused consumer: 40 layers of
    ``mlp.shared_experts.gate_up_proj`` are fed ``shared_experts.w1``/``w3``, whose
    scales arrive under the *source* names. What fusion cannot hide is the count. A
    consumer that receives N fp8 weights must receive N scales; anything less means
    the difference is being multiplied by uninitialized memory.

    Run in the switch direction on purpose. The default bf16 path satisfies this
    trivially (it declares no dense block-scaled modules at all), so a test that only
    ran there would pass while asserting nothing.
    """
    monkeypatch.setenv("TRTLLM_V41_DENSE_FP8", "1")
    assert v41_config.dense_fp8_requant_enabled()
    # A second 40-layer build rather than the shared fixture: the switch is read
    # inside `ModelConfig.from_pretrained`, so the module-scoped `release_model` was
    # already frozen on the bf16 path before this test ran.
    model = _build_release_model()
    assert model.model_config.quant_config.quant_algo is not None, (
        "the switch must reach the quant config, or this test proves nothing"
    )

    monkeypatch.setattr(
        v41,
        "_dequantize_block32",
        lambda weight, scale: torch.empty(tuple(weight.shape), dtype=torch.bfloat16, device="meta"),
    )
    loader = v41.DeepseekV41WeightLoader(model)
    forwarded = loader.remap_checkpoint_keys(
        dict(release_meta_weights),
        num_hidden_layers=model.config.num_hidden_layers,
        kv_lora_rank=model.config.kv_lora_rank,
    )
    keys = set(forwarded.all_keys)

    # Who consumes a key is decided the same way the coverage audit decides it, so
    # this test and that one cannot disagree about module ownership.
    modules = dict(model.named_modules())
    consumers = {name for name, mod in modules.items() if name and hasattr(mod, "load_weights")}

    def consumer_of(key):
        parts = key.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in consumers:
                return prefix
        return None

    weights_per, scales_per = {}, {}
    for key in keys:
        owner = consumer_of(key)
        if owner is None:
            continue
        if key.endswith(".weight"):
            weights_per[owner] = weights_per.get(owner, 0) + 1
        elif key.endswith(".weight_scale_inv") or key.endswith(".weight_scale"):
            scales_per[owner] = scales_per.get(owner, 0) + 1

    needs_scale = [
        name
        for name, mod in modules.items()
        if getattr(getattr(mod, "weight", None), "dtype", None) is torch.float8_e4m3fn
        and getattr(mod, "weight_scale", None) is not None
    ]
    assert needs_scale, "the switch built no fp8 block-scaled module; the arm is inert"

    starved = []
    for name in needs_scale:
        owner = name if name in consumers else (consumer_of(f"{name}.weight") or name)
        n_w, n_s = weights_per.get(owner, 0), scales_per.get(owner, 0)
        if n_s < n_w:
            starved.append(f"{name} (consumer {owner}: {n_w} weight(s), {n_s} scale(s))")

    assert not starved, (
        f"{len(starved)} of {len(needs_scale)} fp8 module(s) allocate a weight_scale that "
        f"no forwarded key fills. Each will cast its bf16 weight into fp8 unscaled and "
        f"multiply by uninitialized memory, and neither the load census nor `Linear` will "
        f"say so. Add the missing stem tail to `_V41_REQUANT_STEM_TAILS`.\n  "
        + "\n  ".join(sorted(starved)[:12])
    )


def test_no_v41_attention_subclass_reacquires_the_per_head_query_norm():
    """Every V4.1 attention class must declare ``q_b_norm_enabled = False``.

    The flag defaults to ``True`` on the base class so that a V4 subclass which
    says nothing stays V4-correct. That default is exactly the trap for V4.1: the
    three MTP layers are not constructed yet (``DeepseekV41ForCausalLM`` refuses
    ``model_nextn != 0``), and whoever implements them will write a DSpark
    cross-attention class. If it derives from ``DeepseekV4Attention`` rather than
    from ``DeepseekV41Attention`` it inherits the default, re-acquires the norm,
    and costs ~0.52x on every draft query -- producing fluent, plausible, wrong
    drafts rather than a crash.

    The reference is unambiguous that MTP has no such norm: ``DSparkAttention``
    (``model.py:1032``) overrides ``forward`` and its query path is
    ``q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(...)`` with no per-head
    renormalization, the same as V4.1's main attention and unlike V4's.

    This runs on CPU and constructs nothing: it walks the classes the V4.1 module
    actually defines, so it starts covering the MTP class the moment one exists.
    """
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Attention

    offenders = []
    for name in dir(v41):
        obj = getattr(v41, name)
        if not isinstance(obj, type) or not issubclass(obj, DeepseekV4Attention):
            continue
        if obj is DeepseekV4Attention:
            continue  # V4 itself keeps the norm and is not defined here
        if obj.__module__ != v41.__name__:
            continue  # only classes this module owns
        if getattr(obj, "q_b_norm_enabled", True) is not False:
            offenders.append(name)

    assert not offenders, (
        f"V4.1 attention classes {offenders} do not set q_b_norm_enabled = False, so "
        f"they inherit V4's per-head query norm. V4.1 removed it (model.py:770-772 "
        f"vs V4's model.py:498) and inheriting it costs ~0.52x on every query."
    )
    # The walk has to actually find something, or a rename turns this into a
    # test that passes by looking at nothing.
    assert v41.DeepseekV41Attention.q_b_norm_enabled is False
    assert DeepseekV4Attention.q_b_norm_enabled is True


class _StubEngramConfig:
    """A config whose every attribute access raises.

    The attention-DP guard is supposed to fire before it reads anything. Making that
    structural rather than incidental: if a future edit moves the raise below the
    first config read, this stops being a passing test.
    """

    def __getattr__(self, name):
        raise AssertionError(
            f"the attention-DP guard read config.{name}; it is supposed to raise "
            f"before touching the config at all"
        )


def test_engram_refuses_attention_dp():
    """enable_attention_dp=True at tp_size>1 must be refused, not silently sharded."""
    mapping = Mapping(world_size=8, tp_size=8, rank=0, enable_attention_dp=True)
    with pytest.raises(NotImplementedError, match="attention data parallelism"):
        v41.DeepseekV41Engram(layer_id=0, config=_StubEngramConfig(), mapping=mapping)


def test_engram_guard_is_scoped_to_attention_dp():
    """Without attention DP the guard must not fire.

    Asserted as "raises something else, but not our message" rather than "constructs
    successfully", because a real construction wants a real config and allocates the
    table -- which is what this CPU test is avoiding. What matters is that the refusal
    is scoped to the configuration it names.
    """
    mapping = Mapping(world_size=8, tp_size=8, rank=0, enable_attention_dp=False)
    with pytest.raises(Exception) as excinfo:
        v41.DeepseekV41Engram(layer_id=0, config=_StubEngramConfig(), mapping=mapping)
    assert "attention data parallelism" not in str(excinfo.value)


def test_engram_allows_attention_dp_at_tp1():
    """tp_size=1 shards nothing, so there is no collective to malform."""
    mapping = Mapping(world_size=1, tp_size=1, rank=0, enable_attention_dp=True)
    with pytest.raises(Exception) as excinfo:
        v41.DeepseekV41Engram(layer_id=0, config=_StubEngramConfig(), mapping=mapping)
    assert "attention data parallelism" not in str(excinfo.value)
