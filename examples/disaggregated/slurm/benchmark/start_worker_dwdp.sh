#! /bin/bash
set -u
set -e
set -x

config_file=${1}
numa_bind=${2}
log_dir=${3}
enable_nsys=${4}
ctx_profile_range=${5}
gen_profile_range=${6}
num_ctx_gpus=${7}
ctx_worker_env_var=${8}
gen_worker_env_var=${9}

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname)"

# NOTE: do NOT export CUDA_VISIBLE_DEVICES from this script.
#
# Restricting CUDA visibility to a single GPU (via CUDA driver isolation)
# breaks DWDP's intra-node peer GPU discovery:
#   - VA composite cuMemMap imports of peer GPUs' MNNVL fabric handles
#   - UCX cuda_ipc / cuda_copy intra-node transports for CTX->GEN KV
#   - PyTorch torch.cuda.device_count() peer enumeration
# All of these need the process to *see* peer GPUs on the same node,
# even if it only computes on one of them.
#
# Empirically (R1/T2/T3/T4 vs T5 on dwdp3 dg=4): exporting
# ``CUDA_VISIBLE_DEVICES=<single_gpu>`` blows TTFT std up 3x and drops
# per-CTX-GPU throughput by 15%, with TPOT unchanged.  Letting
# trtllm-serve auto-pick the device from SLURM_LOCALID restores Phase
# D's full perf.
#
# For audit, log which GPU SLURM would have given this rank.  With our
# compact-packing allocate_gpus, the natural mapping is:
#   gpu_id = (gpu_map_mpi_worker.txt[rank][2])   if gpu_map exists
#         == SLURM_LOCALID                       (always, by construction)
# so we log it but don't export.
gpu_map_file="${log_dir}/gpu_map_mpi_worker.txt"
if [ -f "${gpu_map_file}" ]; then
    expected_gpu=$(awk -v p="${SLURM_PROCID}" '$1==p {print $3; exit}' "${gpu_map_file}")
    echo "rank-to-gpu (arbitrary-dist path): SLURM_PROCID=${SLURM_PROCID} LOCALID=${SLURM_LOCALID} expected_gpu=${expected_gpu}"
else
    echo "rank-to-gpu (block-dist path): SLURM_PROCID=${SLURM_PROCID} LOCALID=${SLURM_LOCALID}"
fi

if [ "${SLURM_PROCID}" -lt "${num_ctx_gpus}" ]; then
    worker_role="CTX"
    worker_env_var=${ctx_worker_env_var}
    profile_range=${ctx_profile_range}
else
    worker_role="GEN"
    worker_env_var=${gen_worker_env_var}
    profile_range=${gen_profile_range}
fi

echo "worker_role: ${worker_role}, profile_range: ${profile_range}"

for env_var in ${worker_env_var}; do
    export "${env_var}"
    echo "Exported: ${env_var}"
done

# Container runtimes (pyxis/enroot) reset image-defined variables like PATH and
# PYTHONPATH at container start, so a value the launcher config wants for the
# worker cannot simply be exported into the container -- it has to be prepended
# from inside it.  Same contract as start_worker.sh.
#
# This MUST run after the worker_env_var loop above.  On the DWDP path the
# request does NOT arrive via srun --export (submit_dwdp.py's worker srun has
# none); it arrives inside the ctx_worker_env_var / gen_worker_env_var positional
# argument, so a block placed before the loop would read an unset variable and
# drop the request without a word.  Same ordering trap as the UCX block below.
#
# The load-bearing use is CuTe DSL.  Some container images ship the public PyPI
# nvidia_cutlass_dsl frontend, whose .pth puts its self-contained
# python_packages/ tree on sys.path and thereby SHADOWS the internal nightly
# frontend installed alongside it under dsl_packages/.  When the public tree is
# older than the target architecture, every CuTe DSL kernel then dies at compile
# time with "invalid chip string".  Prepending
#   /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/dsl_packages
# restores the nightly frontend, because PYTHONPATH entries precede .pth-added
# dirs in sys.path.
#
# That same wheel also drops a nvidia_cutlass_dsl/lib/libcute_dsl_runtime.so,
# and the frontend's runtime discovery tries "lib" BEFORE "cu<major>/lib" while
# walking its ancestors -- so restoring the frontend alone makes it load the
# public runtime, which lacks CuteDSLRT_TVMFFISetRaisedCudaError and fails every
# --enable-tvm-ffi compile.  Pin the matching runtime alongside the prepend with
#   CUTE_DSL_LIBS=<...>/nvidia_cutlass_dsl/cu<CTK major>/lib/libcute_dsl_runtime.so
# (an ordinary variable -- the export loop above handles it; the value is the
# full path to the .so, not a directory).  A wrong CTK major compiles and then
# dies at launch with cudaErrorInvalidValue.
# srun --export keeps any quotes in the exported values literal; strip them.
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND:-}"
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND#\'}"; TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND%\'}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND:-}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND#\'}"; TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND%\'}"
if [ -n "${TRTLLM_PATH_PREPEND:-}" ]; then
    export PATH="${TRTLLM_PATH_PREPEND}:${PATH}"
    echo "PATH prepended from TRTLLM_PATH_PREPEND: ${TRTLLM_PATH_PREPEND}"
fi
if [ -n "${TRTLLM_PYTHONPATH_PREPEND:-}" ]; then
    export PYTHONPATH="${TRTLLM_PYTHONPATH_PREPEND}${PYTHONPATH:+:${PYTHONPATH}}"
    echo "PYTHONPATH prepended from TRTLLM_PYTHONPATH_PREPEND: ${TRTLLM_PYTHONPATH_PREPEND}"
fi

# Clear the container-provided UCX_TLS / UCX_NET_DEVICES, then re-pin them if
# the launcher config asked for an explicit transport list.  Same contract as
# start_worker.sh, but this MUST run after the worker_env_var loop above: the
# pin arrives as TRTLLM_WORKER_UCX_TLS inside ctx_worker_env_var /
# gen_worker_env_var, so clearing before the loop exports the request and then
# never acts on it, silently dropping the pin.
#
# This is a CONNECTIVITY knob, not a throughput knob.  Some clusters cannot
# establish the CTX->GEN KV connection at all unless both UCX_TLS and
# UCX_NET_DEVICES are pinned (observed on GB300 nodes that need
# UCX_NET_DEVICES=eth0); on those, auto-selection picks an interface with no
# route to the peer and the transfer never connects.  Do NOT set it expecting a
# speedup: pinning UCX_TLS was measured on a prefill-heavy NVL72-class run and
# left throughput unchanged.
if [ -n "${TRTLLM_WORKER_UCX_TLS:-}" ]; then
    export UCX_TLS="${TRTLLM_WORKER_UCX_TLS}"
    echo "UCX_TLS pinned from TRTLLM_WORKER_UCX_TLS: ${UCX_TLS}"
else
    unset UCX_TLS
    echo "UCX_TLS cleared (no TRTLLM_WORKER_UCX_TLS in worker_env_var)"
fi
if [ -n "${TRTLLM_WORKER_UCX_NET_DEVICES:-}" ]; then
    export UCX_NET_DEVICES="${TRTLLM_WORKER_UCX_NET_DEVICES}"
    echo "UCX_NET_DEVICES pinned from TRTLLM_WORKER_UCX_NET_DEVICES: ${UCX_NET_DEVICES}"
else
    unset UCX_NET_DEVICES
    echo "UCX_NET_DEVICES cleared (no TRTLLM_WORKER_UCX_NET_DEVICES in worker_env_var)"
fi

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
    echo "Not binding memory. If on GB200/GB300 NVL72, use \"numactl -m 0,1\" to only allocate memory from nodes."
fi

echo "config_file: ${config_file}"

nsys_prefix=""
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_file=${log_dir}/nsys_worker_proc_${worker_role}_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    export NSYS_MPI_STORE_TEAMS_PER_RANK=1
    export TLLM_PROFILE_START_STOP=${profile_range}
    echo "nsys is enabled on ${worker_role} ranks, TLLM_PROFILE_START_STOP=${profile_range}"
    nsys_prefix="nsys profile -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

${nsys_prefix} ${numa_bind_cmd} trtllm-serve disaggregated_mpi_worker -c ${config_file}
