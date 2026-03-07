# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulated quantization utilities for mixed K/V precision KV cache.

Applies round-trip (quantize then dequantize) to simulate precision loss
without modifying the actual storage format. This allows quality validation
of mixed K/V precision configurations in the full inference pipeline.
"""

import torch

_INT_BITS = {"int2": 2, "int3": 3, "int4": 4, "int8": 8}


def simulated_quantize(x: torch.Tensor, dtype_str: str, group_size: int = 128) -> torch.Tensor:
    """Round-trip quantization: quantize then dequantize to simulate precision loss.

    Args:
        x: Input tensor of any shape, with quantization applied along the last dim.
        dtype_str: Target simulated dtype. One of 'int2','int3','int4','int8','fp8'.
        group_size: Group size for per-group INT symmetric quantization.

    Returns:
        Tensor with same shape and dtype as input, with simulated quantization noise.
    """
    if dtype_str is None:
        return x

    if dtype_str == "fp8":
        return x.to(torch.float8_e4m3fn).to(x.dtype)

    n_bits = _INT_BITS.get(dtype_str)
    if n_bits is None:
        raise ValueError(
            f"Unsupported simulated dtype: '{dtype_str}'. "
            f"Supported: {sorted(list(_INT_BITS.keys()) + ['fp8'])}"
        )

    # Symmetric per-group quantization
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])

    if group_size > 0 and x_flat.shape[-1] % group_size == 0:
        x_grouped = x_flat.reshape(-1, group_size)
    else:
        x_grouped = x_flat

    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    scale = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / qmax
    x_q = (x_grouped / scale).round().clamp(qmin, qmax)
    x_deq = x_q * scale

    return x_deq.reshape(orig_shape)
