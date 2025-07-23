import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..attention_backend.interface import AttentionMetadata
from ..speculative.interface import SpecMetadata
from ..utils import make_weak_ref, set_piecewise_cuda_graph_flag


class graph_capturing_local(threading.local):

    def __init__(self):
        self.is_graph_capturing = False


_local = graph_capturing_local()


def set_graph_capturing(enable: bool):
    _local.is_graph_capturing = enable


def is_graph_capturing() -> bool:
    return _local.is_graph_capturing


class DecodingCUDAGraphRunner:

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        use_mrope: bool = False,
    ) -> None:
        """
        Stores a CUDA graph and its associated input buffers.

        Each CUDA graph runner is associated with an AttentionMetadata object
        if flashinfer is being used. Make sure to call attn_metadata.prepare()
        before run()!

        Note that torch.compile w/ mode reduce-overhead supports CUDA graphs
        with memory pool sharing. However, we have our own manager here because,
        at the time of writing this, torch.compile takes way too long to warmup
        graphs compared to doing it manually (not to mention, custom ops from
        e.g. FlashInfer cause graph breaks).
        """
        self.batch_size = batch_size

        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        token_per_request = spec_metadata.max_draft_len + 1 if spec_metadata is not None else 1

        # Using ones instead of zeros prevents NaNs in e.g. Deepseek
        self.input_ids = torch.ones((batch_size * token_per_request, ),
                                    device=device,
                                    dtype=torch.int32)
        self.position_ids = torch.zeros((1, batch_size * token_per_request),
                                        device=device,
                                        dtype=torch.int32)
        self.mrope_position_deltas = torch.zeros(
            (batch_size,
             1), device=device, dtype=torch.int32) if use_mrope else None

        self.attn_metadata = attn_metadata
        self.spec_metadata = spec_metadata
        self._output = None
        self._graph = None
        self.optional_extra_model_inputs = ["mrope_position_deltas"]

    def __del__(self):
        if self._graph is not None:
            self._graph.reset()

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        self._graph = torch.cuda.CUDAGraph()
        inputs = {
            "attn_metadata": self.attn_metadata,
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "inputs_embeds": None,
            "spec_metadata": self.spec_metadata,
            "mrope_position_deltas": self.mrope_position_deltas,
        }

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        set_graph_capturing(True)
        set_piecewise_cuda_graph_flag(False)
        for _ in range(2):
            forward_fn(inputs)
        with torch.cuda.graph(self._graph, pool=pool):
            output = forward_fn(inputs)
        set_graph_capturing(False)
        set_piecewise_cuda_graph_flag(True)
        # Mark weak ref here. The output tensor should be freed properly.
        self._output = make_weak_ref(output)
        return self._graph.pool()

    def needs_capture(self, use_spec: bool = False) -> bool:
        return self._output is None

    def run(self,
            inputs: Dict[str, Any],
            use_spec: bool = False) -> torch.Tensor:
        assert "input_ids" in inputs
        assert "position_ids" in inputs
        assert "attn_metadata" in inputs

        attn_metadata = inputs["attn_metadata"]
        assert attn_metadata is self.attn_metadata, (
            "attn_metadata does not match the attn_metadata instance that was used to "
            "capture this graph.")

        if "spec_metadata" in inputs:
            spec_metadata = inputs["spec_metadata"]
            assert spec_metadata is self.spec_metadata, (
                "spec_metadata does not match the spec_metadata instance that was used to "
                "capture this graph.")

        input_ids = inputs["input_ids"]
        position_ids = inputs["position_ids"]
        seqlen = input_ids.shape[0]
        self.input_ids[:seqlen].copy_(input_ids)
        self.position_ids[:, :seqlen].copy_(position_ids)
        if "mrope_position_deltas" in inputs:
            self.mrope_position_deltas[:self.batch_size].copy_(
                inputs["mrope_position_deltas"])

        assert self._output is not None and self._graph is not None
        self._graph.replay()
        return self._output


class DynamicDecodingCUDAGraphRunner:
    """
    A CUDA graph runner that supports both speculative and non-speculative modes.
    This allows dynamic toggling of speculation without performance penalties.
    Can also function as a single-mode runner when only one mode is needed.
    """

    def __init__(
        self,
        batch_size: int,
        device: str,
        max_draft_len: int,
        attn_metadata_spec: Optional[AttentionMetadata] = None,
        attn_metadata_non_spec: Optional[AttentionMetadata] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        use_mrope: bool = False,
    ) -> None:
        """
        Initialize multi-mode CUDA graph runner.

        Args:
            batch_size: Number of requests in the batch
            device: Device to run on
            use_spec: Whether this runner supports speculative decoding
            attn_metadata_spec: Attention metadata for speculative mode (required if SPECULATIVE mode supported)
            attn_metadata_non_spec: Attention metadata for non-speculative mode (required if NON_SPECULATIVE mode supported)
            spec_metadata: Speculative decoding metadata
            use_mrope: Whether to use MRoPE
        """
        self.batch_size = batch_size
        self.device = device
        self.use_mrope = use_mrope
        self.max_draft_len = max_draft_len

        # Create graph runners based on supported modes
        self.spec_runner = None
        self.non_spec_runner = None

        if max_draft_len > 0:
            if attn_metadata_spec is None:
                raise ValueError(
                    "attn_metadata_spec is required when SPECULATIVE mode is supported"
                )
            self.spec_runner = DecodingCUDAGraphRunner(batch_size, device,
                                                       attn_metadata_spec,
                                                       spec_metadata, use_mrope)
        # CUDA graph runner for non-speculative mode is always created, so that we can dynamically turn on/off speculation.
        if attn_metadata_non_spec is None:
            raise ValueError(
                "attn_metadata_non_spec is required when NON_SPECULATIVE mode is supported"
            )
        self.non_spec_runner = DecodingCUDAGraphRunner(batch_size, device,
                                                       attn_metadata_non_spec,
                                                       None, use_mrope)

        self._captured_spec = False
        self._captured_non_spec = False

    def __del__(self):
        # Cleanup is handled by individual runners
        pass

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Capture graphs for the configured mode.
        During initial capture, only capture the mode that matches the current configuration.
        """

        # During initial capture (e.g., warmup), only capture the speculative mode
        # if it's supported, since the forward_fn is typically designed for that mode.
        # The non-speculative mode can be captured later on-demand when actually needed.
        if self.max_draft_len > 0:
            assert self.spec_runner is not None
            spec_pool = self.spec_runner.capture(forward_fn, pool)
            return spec_pool

        assert self.non_spec_runner is not None
        non_spec_pool = self.non_spec_runner.capture(forward_fn, pool)
        return non_spec_pool

    def needs_capture(self, max_draft_len: int = 0) -> bool:
        """Returns True if the mode needs capture."""
        if max_draft_len > 0:
            return self.spec_runner is not None and self.spec_runner.needs_capture(
            )

        return self.non_spec_runner.needs_capture()

    def run(self,
            inputs: Dict[str, Any],
            max_draft_len: int = 0) -> torch.Tensor:
        """
        Run the appropriate graph based on the inputs and supported modes.

        The mode is automatically determined by examining the spec_metadata in inputs
        and the max_draft_len parameter. If spec_metadata is present and max_draft_len > 0,
        use speculative mode. Otherwise, use non-speculative mode.
        """
        # Automatically determine which mode to use based on inputs
        use_spec = (max_draft_len > 0 and "spec_metadata" in inputs
                    and inputs["spec_metadata"] is not None)

        if use_spec:
            assert self.spec_runner is not None
            return self.spec_runner.run(inputs)

        assert self.non_spec_runner is not None
        return self.non_spec_runner.run(inputs)

    @property
    def attn_metadata(self):
        """
        Get attention metadata. Returns the first available metadata.
        For compatibility with existing code that expects this property.
        """
        if self.spec_runner is not None:
            return self.spec_runner.attn_metadata
        elif self.non_spec_runner is not None:
            return self.non_spec_runner.attn_metadata
        else:
            return None

    @property
    def spec_metadata(self):
        """
        Get spec metadata. Returns metadata from speculative runner if available.
        For compatibility with existing code that expects this property.
        """
        if self.spec_runner is not None:
            return self.spec_runner.spec_metadata
        else:
            return None
