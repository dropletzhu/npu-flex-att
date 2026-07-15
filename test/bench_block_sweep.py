"""Focused benchmark: optimal block size + num_warps tuning."""
import torch
import torch_npu  # noqa: F401
import triton
import time
from npu_flash_attention import npu_flash_attention_forward


def bench_fn(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.time()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.time() - start) / rep * 1000


def bench_config(B, H, S, D, dtype, causal, bm, bn, nw=1, ns=1):
    q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    v = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    try:
        ms = bench_fn(lambda: npu_flash_attention_forward(
            q, k, v, causal=causal, block_m=bm, block_n=bn))
        flops = 2 * B * H * S * S * D * 2 * (0.5 if causal else 1.0)
        tflops = flops / ms / 1e9
        tag = f"bm={bm} bn={bn}"
        if nw != 1 or ns != 1:
            tag += f" nw={nw} ns={ns}"
        print(f"    {tag:25s} {ms:7.2f} ms  {tflops:5.1f} TFLOPS")
        return ms
    except Exception as e:
        tag = f"bm={bm} bn={bn}"
        if nw != 1 or ns != 1:
            tag += f" nw={nw} ns={ns}"
        print(f"    {tag:25s} FAIL")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("Block Size Sweep — Ascend 910B3")
    print("=" * 70)

    cases = [
        (1, 1, 128, 64, torch.float16, "S=128 D=64 fp16"),
        (1, 1, 256, 64, torch.float16, "S=256 D=64 fp16"),
        (1, 1, 512, 64, torch.float16, "S=512 D=64 fp16"),
        (1, 1, 1024, 64, torch.float16, "S=1024 D=64 fp16"),
        (1, 1, 2048, 64, torch.float16, "S=2048 D=64 fp16"),
        (1, 1, 128, 64, torch.float32, "S=128 D=64 fp32"),
        (1, 1, 512, 64, torch.float32, "S=512 D=64 fp32"),
        (1, 1, 1024, 64, torch.float32, "S=1024 D=64 fp32"),
        (1, 1, 512, 128, torch.float16, "S=512 D=128 fp16"),
        (1, 1, 1024, 128, torch.float16, "S=1024 D=128 fp16"),
        (2, 4, 512, 64, torch.float16, "B=2 H=4 S=512 D=64 fp16"),
        (1, 8, 1024, 64, torch.float16, "B=1 H=8 S=1024 D=64 fp16"),
    ]

    block_configs = [
        (16, 16), (16, 32), (16, 64),
        (32, 16), (32, 32), (32, 64),
        (64, 16), (64, 32), (64, 64),
    ]

    for B, H, S, D, dtype, label in cases:
        dtype_name = "fp16" if dtype == torch.float16 else "fp32"
        print(f"\n--- {label} ---")
        grid_blocks = triton.cdiv(S, 32) * B * H
        print(f"    grid blocks (bm=32): {grid_blocks} / 48 cores = {min(grid_blocks,48)/48*100:.0f}% utilization")
        for bm, bn in block_configs:
            bench_config(B, H, S, D, dtype, True, bm, bn)
