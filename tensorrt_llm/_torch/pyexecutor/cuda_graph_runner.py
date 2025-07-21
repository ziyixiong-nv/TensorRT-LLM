import threading
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..attention_backend.interface import AttentionMetadata
from ..speculative.interface import SpecMetadata, SpeculationMode
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

    def needs_capture(self) -> bool:
        return self._output is None

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
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


class MultiModeDecodingCUDAGraphRunner:
    """
    A CUDA graph runner that supports both speculative and non-speculative modes.
    This allows dynamic toggling of speculation without performance penalties.
    Can also function as a single-mode runner when only one mode is needed.
    """

    def __init__(
        self,
        batch_size: int,
        device: str,
        attn_metadata_spec: Optional[AttentionMetadata] = None,
        attn_metadata_non_spec: Optional[AttentionMetadata] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        use_mrope: bool = False,
        supported_modes: SpeculationMode = SpeculationMode.NONE,
    ) -> None:
        """
        Initialize multi-mode CUDA graph runner.

        Args:
            batch_size: Number of requests in the batch
            device: Device to run on
            attn_metadata_spec: Attention metadata for speculative mode (required if SPECULATIVE mode supported)
            attn_metadata_non_spec: Attention metadata for non-speculative mode (required if NON_SPECULATIVE mode supported)
            spec_metadata: Speculative decoding metadata
            use_mrope: Whether to use MRoPE
            supported_modes: Which speculation modes this runner should support
        """
        self.batch_size = batch_size
        self.device = device
        self.use_mrope = use_mrope
        self.supported_modes = supported_modes

        # Create graph runners based on supported modes
        self.spec_runner = None
        self.non_spec_runner = None

        if supported_modes.has_speculative:
            if attn_metadata_spec is None:
                raise ValueError(
                    "attn_metadata_spec is required when SPECULATIVE mode is supported"
                )
            self.spec_runner = DecodingCUDAGraphRunner(batch_size, device,
                                                       attn_metadata_spec,
                                                       spec_metadata, use_mrope)

        if supported_modes.has_non_speculative:
            if attn_metadata_non_spec is None:
                raise ValueError(
                    "attn_metadata_non_spec is required when NON_SPECULATIVE mode is supported"
                )
            self.non_spec_runner = DecodingCUDAGraphRunner(
                batch_size, device, attn_metadata_non_spec, None, use_mrope)

        self._captured_spec = False
        self._captured_non_spec = False
        self._forward_fn = None

    def __del__(self):
        # Cleanup is handled by individual runners
        pass

    def capture(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Capture graphs for supported modes.
        During initial capture, only capture the mode that matches the current configuration.
        Store the forward function for potential fallback.
        """
        # Store the forward function for potential fallback
        self._forward_fn = forward_fn

        # During initial capture (e.g., warmup), only capture the speculative mode
        # if it's supported, since the forward_fn is typically designed for that mode.
        # The non-speculative mode can be captured later on-demand when actually needed.
        if (self.supported_modes.has_speculative
                and self.spec_runner is not None and not self._captured_spec):
            spec_pool = self.spec_runner.capture(forward_fn, pool)
            self._captured_spec = True
            return spec_pool

        # If speculative mode is not supported, capture non-speculative
        if (self.supported_modes.has_non_speculative
                and self.non_spec_runner is not None
                and not self._captured_non_spec):
            non_spec_pool = self.non_spec_runner.capture(forward_fn, pool)
            self._captured_non_spec = True
            return non_spec_pool

        # If we get here, both modes are already captured
        return pool

    def needs_capture(self) -> bool:
        """Returns True if the primary mode needs capture."""
        # During initial setup, we only care about capturing the primary mode
        # (speculative mode if supported, otherwise non-speculative)
        if self.supported_modes.has_speculative and self.spec_runner is not None:
            return not self._captured_spec
        if self.supported_modes.has_non_speculative and self.non_spec_runner is not None:
            return not self._captured_non_spec
        return False

    def capture_non_speculative_on_demand(
        self,
        forward_fn: Callable[[Dict[str, Any]], torch.Tensor],
        pool: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Capture the non-speculative mode on-demand when it's actually needed.
        This is called when we need to run in non-speculative mode but it hasn't been captured yet.
        """
        if (self.supported_modes.has_non_speculative
                and self.non_spec_runner is not None
                and not self._captured_non_spec):
            # For non-speculative capture, we need a forward function that doesn't use spec metadata
            def non_spec_forward_fn(inputs: Dict[str, Any]):
                # Create inputs without spec_metadata for non-speculative mode
                non_spec_inputs = inputs.copy()
                non_spec_inputs["spec_metadata"] = None
                return forward_fn(non_spec_inputs)

            non_spec_pool = self.non_spec_runner.capture(
                non_spec_forward_fn, pool)
            self._captured_non_spec = True
            return non_spec_pool
        return None

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Run the appropriate graph based on the inputs and supported modes.

        The mode is determined by examining the spec_metadata in inputs.
        If spec_metadata is present and contains draft tokens, use speculative mode.
        Otherwise, use non-speculative mode if available.
        Falls back to eager execution if graphs are not captured.
        """
        # Determine mode based on spec_metadata
        use_spec_mode = self._should_use_speculative_mode(inputs)

        if use_spec_mode and self.supported_modes.has_speculative:
            if self.spec_runner is None:
                raise RuntimeError("Speculative mode not supported")
            if not self._captured_spec:
                # Fall back to _forward_fn
                return self._forward_fn(inputs)
            return self.spec_runner.run(inputs)

        if not use_spec_mode and self.supported_modes.has_non_speculative:
            if self.non_spec_runner is None:
                # Fall back to _forward_fn
                return self._forward_fn(inputs)
            if not self._captured_non_spec:
                # Fall back to _forward_fn instead of throwing error
                return self._forward_fn(inputs)
            return self.non_spec_runner.run(inputs)

        # If no suitable mode is available, fall back to _forward_fn
        return self._forward_fn(inputs)

    def _should_use_speculative_mode(self, inputs: Dict[str, Any]) -> bool:
        """
        Determine whether to use speculative mode based on the inputs.

        Args:
            inputs: Input dictionary containing model inputs

        Returns:
            True if speculative mode should be used, False otherwise
        """
        # Check if spec_metadata is present and has draft tokens
        if "spec_metadata" not in inputs or inputs["spec_metadata"] is None:
            return False

        spec_metadata = inputs["spec_metadata"]

        # Check if any draft tokens are present and meaningful
        if hasattr(spec_metadata,
                   'draft_tokens') and spec_metadata.draft_tokens is not None:
            # If all draft tokens are zero/dummy tokens, treat as non-speculative
            if hasattr(spec_metadata.draft_tokens, 'sum'):
                return spec_metadata.draft_tokens.sum() > 0
            elif hasattr(spec_metadata.draft_tokens, '__len__'):
                # Check if it's a non-empty tensor/list
                return len(spec_metadata.draft_tokens) > 0
            return True

        # Additional check: if max_draft_len is 0, use non-speculative mode
        if hasattr(spec_metadata,
                   'max_draft_len') and spec_metadata.max_draft_len == 0:
            return False

        return False

    def get_current_mode(self, inputs: Dict[str, Any]) -> SpeculationMode:
        """
        Get the current mode being used for debugging/logging.

        Returns:
            SpeculationMode.SPECULATIVE or SpeculationMode.NON_SPECULATIVE
        """
        return SpeculationMode.SPECULATIVE if self._should_use_speculative_mode(
            inputs) else SpeculationMode.NON_SPECULATIVE

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
