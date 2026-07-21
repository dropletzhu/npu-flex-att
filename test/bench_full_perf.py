"""Comprehensive performance benchmark for NPU Flex Attention.

Tests forward/backward across sequence lengths, GQA configs, and soft-cap,
comparing with SDPA (npu_fusion_attention) baseline.
"""
import torch
import torch_npu  # noqa: F401
import triton
import time
import sys
import os
import math

os.environ["ASCEND_RT_LOG_LEVEL"] = "ERROR"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from npu_flash_attention import (
    npu_flash_attention_forward,
    npu_flash_attention_backward,
    npu_flex_attention,
)


def bench_fn(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.time()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.time() - start) / rep * 1000  # ms


def sdpa_forward(q, k, v, causal=False, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=scale
    )


def sdpa_forward_lse(q, k, v, causal=False, scale=None):
    B, H, S, D = q.shape
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=scale
    )
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    return out, lse


def print_table_header():
    print(f"  {'Config':35s} {'Fwd(ms)':>8s} {'Bwd(ms)':>8s} {'Total':>8s} {'vs SDPA':>8s} {'Fwd T':>7s} {'Bwd T':>7s}")
    print("  " + "-" * 95)


def print_row(label, fwd_ms, bwd_ms, sdpa_ms, flops_fwd, flops_bwd):
    total = (fwd_ms or 0) + (bwd_ms or 0)
    vs_sdpa = f"{sdpa_ms / fwd_ms:.2f}x" if (sdpa_ms and fwd_ms) else "N/A"
    tf_fwd = f"{flops_fwd / fwd_ms / 1e9:.1f}" if fwd_ms else "FAIL"
    tf_bwd = f"{flops_bwd / bwd_ms / 1e9:.1f}" if bwd_ms else "FAIL"
    fwd_s = f"{fwd_ms:.2f}" if fwd_ms else "FAIL"
    bwd_s = f"{bwd_ms:.2f}" if bwd_ms else "FAIL"
    total_s = f"{total:.2f}" if fwd_ms and bwd_ms else "N/A"
    print(f"  {label:35s} {fwd_s:>8s} {bwd_s:>8s} {total_s:>8s} {vs_sdpa:>8s} {tf_fwd:>7s} {tf_bwd:>7s}")


def bench_suite():
    print("=" * 100)
    print("NPU Flex Attention — Comprehensive Performance Benchmark (Ascend 910B3)")
    print("=" * 100)

    # ── Section 1: Core sweep (B=1, H=2, D=64, fp16, causal) ──
    print("\n--- 1. Core Sequence Length Sweep (B=1, H=2, D=64, fp16, causal) ---")
    print_table_header()

    for S in [128, 256, 512, 1024, 2048]:
        B, H, D = 1, 2, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        do = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

        sdpa_ms = bench_fn(lambda: sdpa_forward(q, k, v, causal=True))
        fwd_ms = bench_fn(lambda: npu_flash_attention_forward(q, k, v, causal=True))

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        bwd_ms = bench_fn(lambda: npu_flash_attention_backward(q, k, v, out, lse, do, causal=True))

        flops_fwd = 2 * B * H * S * S * D * 2
        flops_bwd = 2 * B * H * S * S * D * 4
        print_row(f"S={S}", fwd_ms, bwd_ms, sdpa_ms, flops_fwd, flops_bwd)

    # ── Section 2: H=4 sweep ──
    print("\n--- 2. Head Count Sweep (B=1, H=4, D=64, fp16, causal) ---")
    print_table_header()

    for S in [128, 256, 512, 1024, 2048]:
        B, H, D = 1, 4, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        do = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

        sdpa_ms = bench_fn(lambda: sdpa_forward(q, k, v, causal=True))
        fwd_ms = bench_fn(lambda: npu_flash_attention_forward(q, k, v, causal=True))

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        bwd_ms = bench_fn(lambda: npu_flash_attention_backward(q, k, v, out, lse, do, causal=True))

        flops_fwd = 2 * B * H * S * S * D * 2
        flops_bwd = 2 * B * H * S * S * D * 4
        print_row(f"S={S}", fwd_ms, bwd_ms, sdpa_ms, flops_fwd, flops_bwd)

    # ── Section 3: GQA sweep ──
    print("\n--- 3. GQA Ratio Sweep (B=1, D=64, fp16, causal, S=512) ---")
    print_table_header()

    S = 512
    for Hq, Hkv in [(4, 2), (8, 2), (16, 4), (32, 8)]:
        B, D = 1, 64
        q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
        do = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)

        # SDPA needs GQA expansion for comparison
        k_exp = k.repeat_interleave(Hq // Hkv, dim=1)
        v_exp = v.repeat_interleave(Hq // Hkv, dim=1)
        sdpa_ms = bench_fn(lambda: sdpa_forward(q, k_exp, v_exp, causal=True))
        fwd_ms = bench_fn(lambda: npu_flash_attention_forward(q, k, v, causal=True))

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        bwd_ms = bench_fn(lambda: npu_flash_attention_backward(q, k, v, out, lse, do, causal=True))

        flops_fwd = 2 * B * Hq * S * S * D * 2
        flops_bwd = 2 * B * Hq * S * S * D * 4
        print_row(f"Hq={Hq} Hkv={Hkv} (ratio={Hq // Hkv})", fwd_ms, bwd_ms, sdpa_ms, flops_fwd, flops_bwd)

    # ── Section 4: Soft-cap performance ──
    print("\n--- 4. Soft-Cap Performance (B=1, H=2, D=64, fp16, causal) ---")
    print_table_header()

    for S in [128, 512, 1024, 2048]:
        B, H, D = 1, 2, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        do = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

        fwd_ms = bench_fn(lambda: npu_flash_attention_forward(q, k, v, causal=True, soft_cap=50.0))

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True, soft_cap=50.0)
        bwd_ms = bench_fn(lambda: npu_flash_attention_backward(q, k, v, out, lse, do, causal=True, soft_cap=50.0))

        flops_fwd = 2 * B * H * S * S * D * 2
        flops_bwd = 2 * B * H * S * S * D * 4
        print_row(f"S={S} soft_cap=50", fwd_ms, bwd_ms, None, flops_fwd, flops_bwd)

    # ── Section 5: Batch scaling ──
    print("\n--- 5. Batch Scaling (H=4, S=512, D=64, fp16, causal) ---")
    print_table_header()

    H, S, D = 4, 512, 64
    for B in [1, 2, 4, 8]:
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        do = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

        sdpa_ms = bench_fn(lambda: sdpa_forward(q, k, v, causal=True))
        fwd_ms = bench_fn(lambda: npu_flash_attention_forward(q, k, v, causal=True))

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        bwd_ms = bench_fn(lambda: npu_flash_attention_backward(q, k, v, out, lse, do, causal=True))

        flops_fwd = 2 * B * H * S * S * D * 2
        flops_bwd = 2 * B * H * S * S * D * 4
        print_row(f"B={B}", fwd_ms, bwd_ms, sdpa_ms, flops_fwd, flops_bwd)

    # ── Section 6: D=128 ──
    print("\n--- 6. D=128 Sweep (B=1, H=2, fp16, causal) ---")
    print_table_header()

    H, D = 2, 128
    for S in [128, 256, 512, 1024, 2048]:
        B = 1
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        do = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

        sdpa_ms = bench_fn(lambda: sdpa_forward(q, k, v, causal=True))
        fwd_ms = bench_fn(lambda: npu_flash_attention_forward(q, k, v, causal=True))

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        bwd_ms = bench_fn(lambda: npu_flash_attention_backward(q, k, v, out, lse, do, causal=True))

        flops_fwd = 2 * B * H * S * S * D * 2
        flops_bwd = 2 * B * H * S * S * D * 4
        print_row(f"S={S} D=128", fwd_ms, bwd_ms, sdpa_ms, flops_fwd, flops_bwd)

    # ── Section 7: Non-causal ──
    print("\n--- 7. Non-Causal (B=1, H=2, D=64, fp16) ---")
    print_table_header()

    H, D = 2, 64
    for S in [128, 256, 512, 1024]:
        B = 1
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        do = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

        sdpa_ms = bench_fn(lambda: sdpa_forward(q, k, v, causal=False))
        fwd_ms = bench_fn(lambda: npu_flash_attention_forward(q, k, v, causal=False))

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=False, return_lse=True)
        bwd_ms = bench_fn(lambda: npu_flash_attention_backward(q, k, v, out, lse, do, causal=False))

        flops_fwd = 2 * B * H * S * S * D * 2
        flops_bwd = 2 * B * H * S * S * D * 4
        print_row(f"S={S} non-causal", fwd_ms, bwd_ms, sdpa_ms, flops_fwd, flops_bwd)

    print("\n" + "=" * 100)
    print("Benchmark complete.")


if __name__ == "__main__":
    bench_suite()
