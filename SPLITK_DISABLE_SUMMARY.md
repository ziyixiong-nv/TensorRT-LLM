# Split-K Disabled for TRTLLM-gen MoE Backend

## Summary
Split-K parallelization has been disabled for TRTLLM-gen MoE kernels by filtering out kernel configurations with `mNumSlicesForSplitK > 1`.

## Changes Made

### 1. C++ Kernel Runner Filter
**File**: `cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp`

Added filter in the `TrtllmGenBatchedGemmRunner` constructor initialization:

```cpp
// Filter out split-K configs (mNumSlicesForSplitK > 1)
// Split-K uses multiple slices along the K dimension which can impact performance
// For MoE workloads, we typically want single-slice (mNumSlicesForSplitK == 1) configs
if (options.mNumSlicesForSplitK > 1)
{
    continue;
}
```

**Location**: Line ~165-171, during `mPassingConfigIndices` population

**Impact**: 
- Configs with `mNumSlicesForSplitK = 2` are excluded from valid kernel configs
- Only configs with `mNumSlicesForSplitK = 1` (no split-K) will be available for tuning
- Affects all TRTLLM-gen MoE backends:
  - FP4 Block Scale MoE
  - FP8 Block Scale MoE  
  - MxE4m3/MxE2m1 Block Scale MoE
  - FP8/FP4 Block Scale MoE

### 2. Python Documentation Update
**File**: `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py`

Added comment in `FP4BlockScaleMoERunner.get_valid_tactics()`:

```python
# Note: Split-K filtering (mNumSlicesForSplitK > 1) is now handled in C++
# See TrtllmGenBatchedGemmRunner initialization in KernelRunner.cpp
```

### 3. Documentation Updates
**File**: `TACTIC_ANALYSIS_GUIDE.md`

Updated to reflect split-K filtering and explain the parameter.

## Split-K Background

### TRTLLM-gen Split-K Details

- **Parameter**: `mNumSlicesForSplitK`
  - Controls number of slices along K dimension
  - Value 1 = no split-K (default)
  - Value 2+ = split-K enabled

- **Implementation Modes**:
  1. **`SplitK::None`**: No split-K parallelization
  2. **`SplitK::Gmem`**: Partial results exchanged via global memory
  3. **`SplitK::Dsmem`**: Partial results exchanged via distributed shared memory (CGA)

- **Usage Before Filter**:
  - 350 total TRTLLM-gen configs in kernel metadata
  - ~9 configs had `mNumSlicesForSplitK = 2`
  - Most configs already used `mNumSlicesForSplitK = 1`

### Comparison with CUTLASS

- **CUTLASS**: Split-K indicated by `cluster_shape_enum` (e.g., `2001001` means 2 SMs in Z dimension)
- **TRTLLM-gen**: Explicit `mNumSlicesForSplitK` parameter with separate exchange mechanisms

## Next Steps

### 1. Recompile the C++ Code

The changes require recompiling TensorRT-LLM:

```bash
cd /home/scratch.fxiong_gpu/git_repo/TensorRT-LLM
# Follow your standard build process, e.g.:
python3 scripts/build_wheel.py --clean
```

### 2. Verify the Filter Works

After recompiling, run your test:

```bash
pytest tests/unittest/_torch/speculative/test_eagle3.py::test_gpt_oss_eagle3
```

Check the output for:
- No split-K tactics (mNumSlicesForSplitK=2) in the available configs
- Debug prints show only single-slice configs

### 3. Performance Testing

Compare performance before/after:
- Latency measurements
- Throughput benchmarks
- Memory usage

## Files Modified

1. `cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp`
2. `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py`
3. `TACTIC_ANALYSIS_GUIDE.md`
4. `SPLITK_DISABLE_SUMMARY.md` (this file)

## Rollback Instructions

If you need to re-enable split-K configs, simply comment out or remove the filter:

```cpp
// In KernelRunner.cpp, line ~165-171:
// if (options.mNumSlicesForSplitK > 1)
// {
//     continue;
// }
```

Then recompile.
