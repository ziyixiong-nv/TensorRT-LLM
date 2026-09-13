# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 parameter and cache-role types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional

from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

from ..dsa.params import DSAMetadataParams, DSAParams

DEEPSEEK_V4_SPARSE_RATIO = 4
DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO = 4

# Which generation's `compress_ratios` encoding a config uses. See
# `DeepSeekV4SparseAttentionConfig.variant` in ``llmapi/llm_args.py`` for the
# user-facing description; the short version is that V4's ratios are role
# sentinels from {0, 4, 128} while V4.1's are true pooling factors from
# {0, 1, 2}, so the same integer means different things in each.
DEEPSEEK_V4_VARIANT = "v4"
DEEPSEEK_V41_VARIANT = "v41"
DEEPSEEK_V4_VARIANTS = (DEEPSEEK_V4_VARIANT, DEEPSEEK_V41_VARIANT)


def is_v41(variant: str) -> bool:
    """Whether ``variant`` selects V4.1 — raising on anything unrecognized.

    Every variant-dependent predicate in this module reads as "V4.1 does X,
    otherwise do what V4 did", so an unrecognized string does not fail: it
    silently reinterprets a V4.1 model under V4 semantics. That is the worst
    available outcome, because the two disagree on what a compress ratio *means*
    (see the note above ``DEEPSEEK_V4_VARIANT``): a V4.1 ratio of 2 is a pooling
    factor, but ``is_sparse_layer`` under V4 tests ``== 4``, so it goes false on
    every layer and disables the indexer model-wide with no error anywhere.
    Funnelling the comparison through here makes that a startup exception
    instead.
    """
    if variant == DEEPSEEK_V41_VARIANT:
        return True
    if variant == DEEPSEEK_V4_VARIANT:
        return False
    raise ValueError(
        f"Unknown DeepSeek-V4 variant {variant!r}; expected one of {DEEPSEEK_V4_VARIANTS}."
    )


class DeepseekV4AttentionType(Enum):
    """DeepSeek-V4 cache roles."""

    # Sliding-window roles must remain contiguous.
    SWA = 0
    COMPRESSOR_KV = 1
    COMPRESSOR_SCORE = 2
    INDEXER_COMPRESSOR_KV = 3
    INDEXER_COMPRESSOR_SCORE = 4

    # Non-sliding cache roles.
    COMPRESS = 5
    INDEXER_COMPRESS = 6

    @property
    def role(self) -> DataRole:
        return DataRole(f"deepseek_v4_{self.name.lower()}")


DEEPSEEK_V4_SLIDING_ATTENTION = (
    DeepseekV4AttentionType.SWA,
    DeepseekV4AttentionType.COMPRESSOR_KV,
    DeepseekV4AttentionType.COMPRESSOR_SCORE,
    DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
    DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
)
assert tuple(attn_type.value for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION) == tuple(
    range(len(DEEPSEEK_V4_SLIDING_ATTENTION))
)

DEEPSEEK_V4_NON_SLIDING_ATTENTION = (
    DeepseekV4AttentionType.COMPRESS,
    DeepseekV4AttentionType.INDEXER_COMPRESS,
)


def pool_factor(compress_ratio: int) -> int:
    """The number of source tokens per compressed entry, floored at 1.

    ``compress_ratio`` doubles as a role sentinel, and the V4.1 encoding spends
    ``0`` on "no compressed branch at all". Dividing by the raw ratio is only
    meaningful for layers that *do* have a compressed branch, but the per-ratio
    buffers and block-size tables are built for every ratio in the model up
    front, so the floor keeps a V4.1 SWA-only layer from raising
    ``ZeroDivisionError`` before its unused entry is ignored. Identity on every
    V4 ratio (1, 4, 128), so the V4 path is unchanged.
    """
    return max(int(compress_ratio), 1)


def swa_only_ratio(variant: str = DEEPSEEK_V4_VARIANT) -> int:
    """The ratio value that means "this layer has no compressed branch".

    V4 spells it ``1`` (the LLM API normalizes a checkpoint ``0`` up to ``1``);
    V4.1 keeps the raw ``0``, because it needs ``1`` for a genuine unpooled
    long-range layer. Several tables are keyed by ratio and need an entry for
    the SWA-only layers, so the sentinel has to be nameable.
    """
    return 0 if is_v41(variant) else 1


def is_dense_compress_layer(compress_ratio: int, variant: str = DEEPSEEK_V4_VARIANT) -> bool:
    """Whether the layer attends over its *whole* compressed cache, unindexed.

    True only for V4's ratio-128 layers, which pool so aggressively that a top-k
    would be pointless. V4.1 indexes every long-range layer (model.py:775-778),
    so it has no dense-compressed layers at all — which is why its
    ``compressed_local_indices`` buffer is empty.
    """
    return is_compress_layer(compress_ratio, variant) and not is_sparse_layer(
        compress_ratio, variant
    )


def is_overlap_compressor(compress_ratio: int, variant: str = DEEPSEEK_V4_VARIANT) -> bool:
    """Whether the compressor keeps two pooling groups in flight at once.

    V4's ratio-4 compressor does (hence a ``2 * ratio`` state window and a
    doubled compressor cache dim). V4.1's reference ``Compressor`` keeps exactly
    one ``[max_batch, ratio, head_dim]`` group (model.py:452), so no V4.1 layer
    is an overlap compressor.
    """
    if is_v41(variant):
        return False
    return compress_ratio == DEEPSEEK_V4_OVERLAP_COMPRESSOR_RATIO


def is_sparse_layer(compress_ratio: int, variant: str = DEEPSEEK_V4_VARIANT) -> bool:
    """Whether the layer participates in indexer-based top-k selection.

    In V4 only the ratio-4 layers do: ratio-128 layers compress so aggressively
    that they attend over the whole compressed cache densely. In V4.1 *every*
    long-range layer is indexed — the reference gates the compressed branch on
    ``if self.compress_ratio:`` (model.py:775) and unconditionally concatenates
    ``compress_idxs`` into ``topk_idxs`` (model.py:778). Note this is about the
    *cache*, not about which layer runs the indexer: index-source layers own the
    top-k computation, but the keys they score live in the cache owned by the
    nearest preceding KV source, and every long-range layer reads a selection.
    """
    if is_v41(variant):
        return compress_ratio > 0
    return compress_ratio == DEEPSEEK_V4_SPARSE_RATIO


def source_layer_for(layer_idx: int, source_layer_ids: Optional[List[int]]) -> Optional[int]:
    """The nearest source at or before ``layer_idx``, or None when there is none.

    V4.1 publishes compressed KV and index top-k from a handful of source
    layers; every later layer reads whatever the most recent source left behind
    (the reference does this implicitly, by having a source overwrite the
    ``shared_attn`` globals its dependents then read -- ``model.py:749`` for the
    compressed cache, ``model.py:731`` for the top-k). "Most recent" in
    execution order is "largest id not greater than mine", which is what this
    returns.

    A ``None`` result means no source has run yet, which for a layer that needs
    one is a malformed config rather than a state to handle -- callers assert.
    """
    if not source_layer_ids:
        return None
    candidates = [int(src) for src in source_layer_ids if int(src) <= layer_idx]
    return max(candidates) if candidates else None


def owns_compressed_kv(
    layer_idx: int,
    compress_ratio: int,
    kv_source_layer_ids: Optional[List[int]] = None,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> bool:
    """Whether this layer *writes* the compressed-KV cache it attends over.

    Distinct from :func:`is_compress_layer`, which asks whether the layer *reads*
    one. They coincide in V4 -- every long-range layer runs its own
    ``Compressor`` -- but V4.1 splits them: only ``kv_source_layer_ids`` own a
    ``Compressor`` (``model.py:657``), and the layers after each source read that
    same cache. Owning is also what decides who allocates the buffer, so a
    dependent layer must answer False here or it would reserve a cache nothing
    ever writes.

    ``kv_source_layer_ids`` unset means "no cross-layer sharing", i.e. the V4
    behaviour, so this reduces to :func:`is_compress_layer`.
    """
    if not is_compress_layer(compress_ratio, variant):
        return False
    if not kv_source_layer_ids:
        return True
    return layer_idx in {int(src) for src in kv_source_layer_ids}


def owns_index_topk(
    layer_idx: int,
    compress_ratio: int,
    index_source_layer_ids: Optional[List[int]] = None,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> bool:
    """Whether this layer runs an ``Indexer`` to compute the top-k itself.

    The counterpart of :func:`owns_compressed_kv` for the selection rather than
    the cache, and deliberately a *different* layer set: V4.1 has eight
    ``index_source_layer_ids`` against four ``kv_source_layer_ids``
    (``model.py:660``). The four extra index sources contribute their own queries
    and ``weights_proj`` against keys they did not produce, which is why
    ownership of the top-k and ownership of the cache have to be asked
    separately.

    ``index_source_layer_ids`` unset reduces to :func:`is_sparse_layer`, the V4
    behaviour.
    """
    if not is_sparse_layer(compress_ratio, variant):
        return False
    if not index_source_layer_ids:
        return True
    return layer_idx in {int(src) for src in index_source_layer_ids}


def is_compress_layer(compress_ratio: int, variant: str = DEEPSEEK_V4_VARIANT) -> bool:
    """Whether the layer has a compressed-KV branch at all.

    ``> 1`` is correct for V4, where ratio 1 only ever arises from the LLM API
    normalizing a 0, i.e. from an SWA-only layer. It is wrong for V4.1, whose
    ratio-1 layers are genuine long-range layers with an unpooled compressed
    cache (``compress_len = (start_pos + seqlen) // 1``); there the test is
    truthiness, matching model.py:775.
    """
    if is_v41(variant):
        return compress_ratio > 0
    return compress_ratio > 1


def has_compressor_state(compress_ratio: int, variant: str = DEEPSEEK_V4_VARIANT) -> bool:
    """Whether the compressor needs a cross-step pooling accumulator.

    Only a *pooling* compressor does. V4.1's ratio-1 compressor is
    ``self.norm(self.wkv(x))`` — one token per group, so no gate, no fp32, and no
    ``kv_state``/``score_state`` buffers (model.py:461, and the ``if
    compress_ratio > 1`` guard around their registration at model.py:447-456).
    A ratio-1 layer that allocated compressor-state cache would reserve memory
    the kernels never touch.
    """
    if is_v41(variant):
        return compress_ratio > 1
    return is_compress_layer(compress_ratio, variant)


def compress_ratio_has_attention(
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    variant: str = DEEPSEEK_V4_VARIANT,
) -> bool:
    is_sparse = is_sparse_layer(compress_ratio, variant)
    is_compress = is_compress_layer(compress_ratio, variant)
    has_state = has_compressor_state(compress_ratio, variant)
    # V4's indexer owns a compressor of its own, so it needs its own pooling
    # state. V4.1's `Indexer` has no compressor at all (model.py:496-525: just
    # `wq_b`, `weights_proj`, and — on a KV source — `wk`/`k_norm`/`k_cache`),
    # because its keys are derived from the main compressor's latent. So the two
    # INDEXER_COMPRESSOR_* roles are V4-only.
    has_indexer_state = is_sparse and not is_v41(variant)

    if attn_type == DeepseekV4AttentionType.SWA:
        return True
    if attn_type == DeepseekV4AttentionType.COMPRESS:
        return is_compress
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_KV:
        return has_state
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return has_state
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return is_sparse
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV:
        return has_indexer_state
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return has_indexer_state
    raise ValueError(f"Unsupported DeepSeek-V4 attention type: {attn_type}")


@dataclass(frozen=True)
class DeepSeekV4Params(DSAParams):
    """DeepSeek-V4 backend parameters."""

    algorithm: Literal["deepseek_v4"] = field(init=False, default="deepseek_v4")
    compress_ratios: List[int] = field(default_factory=list)
    window_size: int = 128
    variant: str = DEEPSEEK_V4_VARIANT
    kv_source_layer_ids: Optional[List[int]] = None
    index_source_layer_ids: Optional[List[int]] = None
    # The compressor's and indexer's RMSNorms use the model's own eps. The
    # default is V4's value; V4.1 ships 1e-20, so this is read off the
    # checkpoint config rather than hardcoded at the consumers.
    rms_norm_eps: float = 1e-6
    # V4.1's two-level candidate prefilter. Read off the checkpoint config, not
    # exposed as a user knob, for the same reason as `rms_norm_eps` above: these
    # are architecture facts, and a value that disagrees with the checkpoint does
    # not tune anything, it selects positions the model was never trained to
    # attend over. `candidate_source_layer_id=None` (V4, and any V4.1 config that
    # omits it) leaves both levels off.
    candidate_source_layer_id: Optional[int] = None
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0


@dataclass(frozen=True)
class DeepSeekV4MetadataParams(DSAMetadataParams):
    """DeepSeek-V4 metadata parameters."""

    compress_ratios: List[int] = field(default_factory=list)
    window_size: int = 128
    variant: str = DEEPSEEK_V4_VARIANT
    kv_source_layer_ids: Optional[List[int]] = None
    index_source_layer_ids: Optional[List[int]] = None
