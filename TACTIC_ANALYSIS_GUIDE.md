# MoE Tactic Analysis Guide

## Overview

This guide explains the tactic encoding differences between CUTLASS and TRTLLM-gen MoE kernels.

## Backend Selection

In your test, the backend is selected via:
```python
moe_config=MoeConfig(backend="CUTLASS")  # or "TRTLLM"
```

## Tactic Encoding Systems

### CUTLASS Tactics (Standard MoE)

**Format**: Single integer (e.g., `37`, `66`)

**Structure for SM100**:
- GEMM_1: 42 tactics total (2 groups × 21 configs)
- GEMM_2: 84 tactics total (4 groups × 21 configs)

**Configuration Parameters** (Tactic ID maps to):
- `tile_config_sm90_enum`: CTA tile dimensions (128×64×128, 128×128×128, 128×256×128)
- `cluster_shape_enum`: 1001001 (1×1×1 SM) or 2001001 (2 SMs)  
- `dynamic_cluster_shape_enum`: Dynamic cluster shape
- `epilogue_fusion_type`: 0=NONE, 1=FINALIZE
- `swap_ab`: false/true (transpose GEMM operands)
- `mainloop_schedule_enum`: Mainloop scheduling (0=AUTO)
- `epilogue_schedule_enum`: Epilogue scheduling (2=TMA)

**Example Output**:
```
[CUTLASS MoE] Selected tactics - GEMM1: 37, GEMM2: 66
```

**Decoding Tactic 37** (from GEMM_1):
- Group: swap_ab=true (tactics 21-41)
- Within group offset: 37-21 = 16
- Configuration: tile=128×64×128, cluster=2×1×1, dynamic_cluster=0, epilogue=NONE, swap_ab=true

### TRTLLM-gen Tactics (FP4/FP8 Block Scale MoE)

**Format**: List of two integers `[tileN, configIndex]`

**Structure**:
- `tileN`: Tile dimension in token (N) direction
  - Common values: 8, 16, 32, 64, 128
- `configIndex`: Index into precompiled kernel config array

**Configuration Parameters** (configIndex maps to):
- `mTileM`, `mTileN`, `mTileK`: CTA tile dimensions
- `mNumStages`: Pipeline depth (typically 3-5)
- `mNumStagesMma`: MMA pipeline stages
- `mNumSlicesForSplitK`: Split-K factor
  - 1 = no split-K (default)
  - 2+ = split-K enabled (filtered out in KernelRunner initialization)
- `mSplitK`: Split-K location (None, GMEM, DSMEM)
- `mClusterDimX`, `mClusterDimY`, `mClusterDimZ`: Cluster dimensions
- Activation type, bias handling, TMA optimizations, etc.

**Note**: Split-K configs (`mNumSlicesForSplitK > 1`) are automatically filtered out during kernel runner initialization to optimize MoE performance.

**Example Output**:
```
[TRTLLM-gen FP4 MoE] Selected tactic: [64, 12] (format: [tileN, configIndex])
```

**Decoding Tactic [64, 12]**:
- `tileN=64`: Uses 64 tokens per CTA tile
- `configIndex=12`: Uses precompiled kernel config #12
  - To see config details, check `cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/KernelMetaInfo.h`

## Key Differences

| Feature | CUTLASS | TRTLLM-gen |
|---------|---------|------------|
| **Tactic Format** | Single integer | `[tileN, configIndex]` |
| **Enumeration** | Dynamic at runtime | Static precompiled list |
| **Split-K Control** | Via `cluster_shape` | Via `mNumSlicesForSplitK` |
| **Kernel Source** | CUTLASS templates | TRTLLM-gen DSL + cubins |
| **Typical Count** | 42-84 per GEMM | 10-50 per tileN |
| **Data Types** | All standard types | FP4, FP8 block scale |

## Filtering Tactics

### CUTLASS (in torch_custom_ops.py)

Current implementation filters out 2-SM tactics:
```python
single_sm_offsets = [0, 1, 2, 3, 5, 6, 9, 11, 12, 15, 17, 18]
```

To modify, edit `get_valid_tactics()` in `MoERunner` class.

### TRTLLM-gen (in trtllm_gen_custom_ops.py)

Tactics are filtered via `get_valid_configs()` in the C++ runner.

To disable split-K, filter configs where `mNumSlicesForSplitK > 1`.

## Debugging Your Test

When running `test_gpt_oss_eagle3`, you'll see output like:

**CUTLASS backend**:
```
[CUTLASS MoE] Selected tactics - GEMM1: 15, GEMM2: 38
```

**TRTLLM backend** (requires FP4/FP8 quantized model):
```
[TRTLLM-gen FP4 MoE] Selected tactic: [32, 7] (format: [tileN, configIndex])
```

## Running the Test

```bash
# CUTLASS backend
pytest tests/unittest/_torch/speculative/test_eagle3.py::test_gpt_oss_eagle3 -k "CUTLASS"

# TRTLLM-gen backend  
pytest tests/unittest/_torch/speculative/test_eagle3.py::test_gpt_oss_eagle3 -k "TRTLLM"
```

## Next Steps

1. Run the test and collect tactic outputs
2. Compare performance between CUTLASS and TRTLLM-gen tactics
3. Analyze which tactics are fastest for your workload
4. Optionally hardcode best tactics to skip autotuning

## Reference Files

- CUTLASS tactics: `cpp/tensorrt_llm/thop/moeOp.cpp`
- CUTLASS config generation: `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp`
- TRTLLM-gen runner: `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu`
- TRTLLM-gen configs: `cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/KernelMetaInfo.h`
