"""
Phase 0 microbenchmark for the DFlash + Target-Side SSD design.

Goal: measure the wall-clock overhead of running target's verify attention
with the §4.2 custom mask (main verify K queries + ΣF_k candidate queries)
relative to a dense baseline (just the K main verify queries).

This benchmark requires only ONE GPU and does NOT need full target model
weights. We measure the attention layer in isolation, with shapes matching
gpt-oss-120b:

    num_qo_heads = 64
    num_kv_heads = 8 (GQA)
    head_dim     = 64
    hidden_size  = 2880

Run:
    python tensorrt_llm/_torch/speculative/bench_dflash_tssd_mask_overhead.py

Or with a single config:
    python ... --batch 1 --K 4 --F_total 16 --prefix_len 4096

The output is a table of wall-clock microseconds and overhead percentages.

Gate condition (see DFLASH_TARGET_SIDE_SSD_DESIGN.md §11 Phase 0):
    proceed only if attention overhead at B=1, ΣF_k=16 is < 25% of dense
    verify-attention time.
"""

import argparse

# Suppress harmless urllib3 warning that flashinfer triggers in this env.
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import torch

warnings.filterwarnings("ignore", category=Warning)

import flashinfer  # noqa: E402

# -----------------------------------------------------------------------------
# Attention shape: gpt-oss-120b (extracted from /code/llm-models/gpt_oss/gpt-oss-120b/config.json)
# -----------------------------------------------------------------------------
NUM_QO_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
HIDDEN = (
    NUM_QO_HEADS * HEAD_DIM
)  # 4096; note hidden_size in config is 2880 but per-head is 64*64=4096
PAGE_SIZE = 16
DTYPE = torch.bfloat16  # bf16 query/output; we use bf16 KV for simplicity


# -----------------------------------------------------------------------------
# Geometric fan-out solver (Saguaro Theorem 12)
# -----------------------------------------------------------------------------
def geometric_fanout(K: int, B_budget: int, a_p: float = 0.78, r: float = 0.5) -> List[int]:
    """
    Compute geometric fan-out F_k for k in [0, K] given total budget B.

    Per Saguaro Theorem 12:
        F_k     = F_0 * a_p^(k/(1+r))           for k < K
        F_K     = F_0 * a_p^(K/(1+r)) * (1-a_p)^(-1/(1+r))
        sum_k F_k = B_budget   --> solve F_0

    Returns integer F_k list (rounded), with min 1 per slot, summing
    approximately to B_budget.
    """
    rho = 1.0 / (1.0 + r)
    coeffs = [a_p ** (k * rho) for k in range(K)]
    coeffs.append(a_p ** (K * rho) * (1.0 - a_p) ** (-rho))
    F0 = B_budget / sum(coeffs)
    F = [max(1, int(round(F0 * c))) for c in coeffs]
    return F


# -----------------------------------------------------------------------------
# Mask construction (per §4.2 of the design doc)
# -----------------------------------------------------------------------------
def build_tssd_mask_per_request(
    K: int,
    F: List[int],  # length K+1
    prefix_len: int,
) -> torch.Tensor:
    """
    Build the §4.2 mask for ONE request (corrected: candidates have own
    K/V slots and self-attention).

    Q = K + sum(F)
    KV-length = prefix_len + K + sum(F) = prefix_len + Q
        [0, prefix_len)              prefix (everyone attends)
        [prefix_len, prefix_len+K)   d_1..d_K (main verify K/V)
        [prefix_len+K, prefix_len+Q) candidate K/V (diagonal)

    Each query attends to its own K/V (self) plus the visible context.
    """
    Q = K + sum(F)
    kv_len = prefix_len + Q
    mask = torch.zeros(Q, kv_len, dtype=torch.bool)

    # Prefix: every query attends to all prefix positions
    mask[:, :prefix_len] = True

    # Main verify rows (q = 0..K-1): row i attends to d_1..d_i AND self.
    for i in range(K):
        mask[i, prefix_len : prefix_len + i + 1] = True

    # Candidate rows: row q (candidate index a = q-K) in group k attends to
    # d_1..d_k AND its own K/V slot at prefix_len + K + a.
    a = 0
    for k in range(K + 1):
        for _ in range(F[k]):
            row = K + a
            own_kv = prefix_len + K + a
            if k > 0:
                mask[row, prefix_len : prefix_len + k] = True
            mask[row, own_kv] = True
            a += 1

    return mask


def build_dense_mask_per_request(K: int, prefix_len: int) -> torch.Tensor:
    """
    Dense baseline: K main verify queries only, standard spec-dec mask.
    Q = K, KV-length = prefix_len + K (each query attends to self).
    """
    Q = K
    kv_len = prefix_len + K
    mask = torch.zeros(Q, kv_len, dtype=torch.bool)
    mask[:, :prefix_len] = True
    for i in range(K):
        mask[i, prefix_len : prefix_len + i + 1] = True  # +1 for self
    return mask


# -----------------------------------------------------------------------------
# FlashInfer plan + run wrapper
# -----------------------------------------------------------------------------
@dataclass
class BenchConfig:
    batch_size: int
    K: int
    F: List[int]  # length K+1; for dense baseline F=[0]*(K+1)
    prefix_len: int
    label: str = ""

    @property
    def Q_per_req(self) -> int:
        return self.K + sum(self.F)

    @property
    def kv_per_req(self) -> int:
        # Corrected: KV includes prefix + K main + sum(F) candidate K/V slots.
        return self.prefix_len + self.Q_per_req


def run_attention_once(
    wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    q: torch.Tensor,
    paged_kv_cache: torch.Tensor,
) -> torch.Tensor:
    """One attention call. Assumes wrapper already planned."""
    return wrapper.run(q, paged_kv_cache)


def setup_wrapper(
    cfg: BenchConfig,
    use_custom_mask: bool,
    workspace: torch.Tensor,
) -> Tuple[
    flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Plan a FlashInfer wrapper for the given config.

    Returns (wrapper, q_buf, paged_kv_cache).
    """
    B = cfg.batch_size
    Q_per_req = cfg.Q_per_req
    kv_per_req = cfg.kv_per_req

    # qo_indptr: cumulative Q offsets, shape [B+1]
    qo_indptr = torch.tensor(
        [i * Q_per_req for i in range(B + 1)], dtype=torch.int32, device="cuda"
    )

    # Paged KV layout: each request gets ceil(kv_per_req / PAGE_SIZE) pages.
    pages_per_req = (kv_per_req + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = B * pages_per_req
    paged_kv_indptr = torch.tensor(
        [i * pages_per_req for i in range(B + 1)], dtype=torch.int32, device="cuda"
    )
    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last_page_len = ((kv_per_req - 1) % PAGE_SIZE) + 1
    paged_kv_last_page_len = torch.full((B,), last_page_len, dtype=torch.int32, device="cuda")

    # Allocate the paged KV cache tensor.
    # Layout: [num_pages, 2, page_size, num_kv_heads, head_dim]
    # (the "2" is K vs V; this is the standard FlashInfer layout)
    paged_kv_cache = torch.randn(
        (total_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM),
        dtype=DTYPE,
        device="cuda",
    )

    # Build per-request masks and concatenate.
    if use_custom_mask:
        masks = []
        for _ in range(B):
            masks.append(build_tssd_mask_per_request(cfg.K, cfg.F, cfg.prefix_len))
        # FlashInfer expects flat: sum_b q_len[b] * k_len[b] booleans.
        flat_mask = torch.cat([m.flatten() for m in masks]).to(device="cuda")
    else:
        flat_mask = None

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE,
        custom_mask=flat_mask,
        causal=False,  # ignored when custom_mask is provided
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )

    # Allocate query tensor: [total_Q, num_qo_heads, head_dim]
    q_buf = torch.randn((B * Q_per_req, NUM_QO_HEADS, HEAD_DIM), dtype=DTYPE, device="cuda")

    return wrapper, q_buf, paged_kv_cache


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------
def bench(
    cfg: BenchConfig,
    use_custom_mask: bool,
    workspace: torch.Tensor,
    n_warmup: int = 20,
    n_iters: int = 200,
) -> float:
    """
    Returns mean wall-clock microseconds per attention call.
    """
    wrapper, q, kv = setup_wrapper(cfg, use_custom_mask, workspace)

    # Warmup
    for _ in range(n_warmup):
        _ = run_attention_once(wrapper, q, kv)
    torch.cuda.synchronize()

    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        _ = run_attention_once(wrapper, q, kv)
    end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    return ms_total * 1000.0 / n_iters  # microseconds per call


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------
def make_dense_cfg(B: int, K: int, prefix_len: int) -> BenchConfig:
    return BenchConfig(
        batch_size=B,
        K=K,
        F=[0] * (K + 1),  # no candidates
        prefix_len=prefix_len,
        label="dense",
    )


def make_tssd_cfg(B: int, K: int, F_total: int, prefix_len: int, a_p: float) -> BenchConfig:
    F = geometric_fanout(K=K, B_budget=F_total, a_p=a_p)
    return BenchConfig(
        batch_size=B,
        K=K,
        F=F,
        prefix_len=prefix_len,
        label=f"tssd_F={F_total}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Run a single config (debug)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--F_total", type=int, default=16)
    parser.add_argument("--prefix_len", type=int, default=4096)
    parser.add_argument(
        "--a_p",
        type=float,
        default=0.78,
        help="Per-token acceptance rate for fanout calc (gpt-oss=0.78, kimi=0.89)",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)

    # FlashInfer workspace (256 MiB is plenty)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"FlashInfer version: {flashinfer.__version__}")
    print(
        f"Attention shape: Q_heads={NUM_QO_HEADS}, KV_heads={NUM_KV_HEADS}, "
        f"head_dim={HEAD_DIM}, page_size={PAGE_SIZE}, dtype={DTYPE}"
    )
    print()

    if args.single:
        # Single config: prints both dense and TSSD timings + overhead
        cfg_dense = make_dense_cfg(args.batch, args.K, args.prefix_len)
        cfg_tssd = make_tssd_cfg(args.batch, args.K, args.F_total, args.prefix_len, args.a_p)

        t_dense = bench(cfg_dense, use_custom_mask=False, workspace=workspace)
        t_tssd = bench(cfg_tssd, use_custom_mask=True, workspace=workspace)
        ovh = (t_tssd / t_dense - 1.0) * 100.0

        print(
            f"Config: B={args.batch} K={args.K} prefix_len={args.prefix_len} "
            f"F_total={args.F_total} F={cfg_tssd.F}"
        )
        print(
            f"  dense       (Q={cfg_dense.Q_per_req:3d}, kv={cfg_dense.kv_per_req:5d}): {t_dense:8.1f} us"
        )
        print(
            f"  tssd        (Q={cfg_tssd.Q_per_req:3d}, kv={cfg_tssd.kv_per_req:5d}): {t_tssd:8.1f} us"
        )
        print(f"  overhead vs dense: {ovh:+.1f}%")
        return

    # Full sweep: prints a table.
    K = args.K
    a_p = args.a_p
    BATCH_SIZES = [1, 2, 4, 8, 16]
    F_TOTALS = [0, 8, 16, 32]
    PREFIX_LENS = [1024, 4096, 16384]

    # Header
    print(
        f"{'prefix':>7s}  {'B':>3s}  "
        + "  ".join(f"F={f:<3d}" + " " * 5 for f in F_TOTALS)
        + "  | overhead vs F=0"
    )
    print(f"{'-' * 7}  {'-' * 3}  " + "  ".join("-" * 9 for _ in F_TOTALS) + "  | " + "-" * 25)

    for prefix_len in PREFIX_LENS:
        for B in BATCH_SIZES:
            timings = {}
            for F_total in F_TOTALS:
                if F_total == 0:
                    cfg = make_dense_cfg(B, K, prefix_len)
                    t = bench(cfg, use_custom_mask=False, workspace=workspace)
                else:
                    cfg = make_tssd_cfg(B, K, F_total, prefix_len, a_p)
                    t = bench(cfg, use_custom_mask=True, workspace=workspace)
                timings[F_total] = t
            row = f"{prefix_len:>7d}  {B:>3d}  " + "  ".join(
                f"{timings[f]:>7.1f}us" for f in F_TOTALS
            )
            base = timings[0]
            ovhs = [(timings[f] / base - 1.0) * 100.0 for f in F_TOTALS]
            ovh_str = "  | " + "  ".join(f"{ovh:+5.1f}%" for ovh in ovhs[1:])
            print(row + ovh_str)
        print()  # blank line between prefix_len groups

    print()
    print("Gate (DFLASH_TARGET_SIDE_SSD_DESIGN.md §11 Phase 0):")
    print("  Proceed if overhead at B=1, F_total=16 is < 25%.")


if __name__ == "__main__":
    main()
