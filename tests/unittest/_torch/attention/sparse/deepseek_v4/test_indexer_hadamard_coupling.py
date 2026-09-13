# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The indexer's Hadamard rotation must be applied to Q and K together or not at all.

The rotation exists to condition the indexer cache's FP8/MXFP4 quantization, and it
is safe *only* because a Hadamard is orthogonal: ``(Hq) . (Hk) == q . k``, so it
cancels in the index logits when both sides carry it. Apply it to one side and every
indexer logit becomes a projection onto a rotated basis -- the top-k still returns
``k`` plausible-looking token indices, nothing asserts, and the model degrades into
attending to the wrong tokens. There is no numerical tripwire for that, which is why
it is pinned structurally here.

Two facts make this worth a test rather than a comment:

* The **decision** is shared but the **implementations are not**. Q is rotated in
  Python by ``fast-hadamard-transform`` via
  ``dsa.indexer.rotate_activation`` (scale ``head_dim ** -0.5``); K is rotated inside
  the CUDA op ``compressor_postprocess_scatter``, which carries its own Hadamard and
  does not consult the Python package at all. Only ``DeepseekV41Indexer`` passing the
  same ``HAS_FAST_HADAMARD`` to both keeps them in step.
* ``rotate_activation`` **silently no-ops** when the package is missing (it logs a
  warning and returns its input). So Q's rotation is decided by an import, while K's
  is decided by a boolean argument. Those two can disagree without anything failing.

The reference applies no Hadamard anywhere -- there is no Hadamard in
``DeepSeek-V4.1-Flash/inference/`` -- so *off on both sides* is the reference-faithful
setting, and it is the setting under which this bring-up's parity and GSM8K numbers
were measured. That also means the rotation cannot be justified as reference fidelity;
it is purely a quantization-conditioning choice, and enabling it puts two
never-cross-validated Hadamard implementations on opposite sides of a dot product.

These tests are pure CPU: they inspect the wiring, and construct nothing.
"""

import inspect

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import indexer as v4_indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as dsa_indexer


def test_rotate_activation_is_a_no_op_without_the_package():
    """Q's rotation is decided by an import, and its absence must be inert, not partial."""
    source = inspect.getsource(dsa_indexer.rotate_activation)
    assert "if not HAS_FAST_HADAMARD" in source
    # The fallback has to return the input unchanged. A fallback that raised would be
    # safe too, but a fallback that rotated by some other means would not.
    fallback = source.split("if not HAS_FAST_HADAMARD")[1].split("hidden_size")[0]
    assert "return x" in fallback, (
        "rotate_activation must return its input unrotated when the package is "
        "missing; anything else desynchronizes Q from K"
    )


def test_q_and_k_rotation_are_driven_by_the_same_decision():
    """Both sides read ``HAS_FAST_HADAMARD``, so neither can rotate alone.

    Q reaches the rotation through ``rotate_activation(...)``, which gates on the
    flag internally. K reaches it through the ``rotate_activation=`` argument of
    ``postprocess_scatter_compressed``, which must therefore be handed exactly that
    flag -- not ``True``, not a config field, not a separate import check.
    """
    source = inspect.getsource(v4_indexer)

    # Every K-side hand-off must pass the flag itself.
    kwargs = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("rotate_activation=")
    ]
    assert kwargs, "no rotate_activation= hand-off found; did the indexer stop wiring K?"
    for kwarg in kwargs:
        assert kwarg == "rotate_activation=HAS_FAST_HADAMARD,", (
            f"K-side rotation is driven by {kwarg!r} instead of HAS_FAST_HADAMARD. Q's "
            f"rotation gates on that flag inside rotate_activation(), so any other "
            f"value lets one side of the q.k dot product be rotated while the other "
            f"is not -- which silently selects the wrong tokens."
        )

    # And the Q side must actually go through the gating helper.
    assert "q = rotate_activation(q)" in source, (
        "Q no longer goes through rotate_activation(), so it no longer shares K's on/off decision"
    )


def test_the_footer_scale_cache_refuses_to_combine_with_the_rotation():
    """The footer-scale postprocess has no Hadamard, so it must reject a rotated Q.

    This is the one place where the two could legitimately drift -- a different
    postprocess path for K -- and it is guarded by an assert rather than by a
    silently different result.
    """
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.compressor import Compressor

    source = inspect.getsource(Compressor.enable_footer_scale_cache)
    assert "assert not self.rotate_activation" in source
