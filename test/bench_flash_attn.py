"""Benchmark NPU Flash Attention performance."""
import torch
import torch_npu  # noqa: F401
import triton
import time
from npu_flash_attention import npu_flash_attention_forward, npu_flash_attention_backward


def bench_fn(fn, warmup=3, rep=10, *args, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    start = time.time()
    for _ in range(rep):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    elapsed = (time.time() - start) / rep
    return elapsed * 1000  # ms


def bench_fwd(B, H, S, D, dtype, causal, block_m, block_n, label=""):
    q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    v = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    try:
        ms = bench_fn(
            lambda: npu_flash_attention_forward(q, k, v, causal=causal, block_m=block_m, block_n=block_n)
        )
        # Compute FLOPs: 2 * B * H * S * S * D * 2 (fwd: QK^T + PV)
        flops = 2 * B * H * S * S * D * 2
        tflops = flops / ms / 1e9
        print(f"  {label:30s} {ms:8.2f} ms  {tflops:6.1f} TFLOPS")
        return ms
    except Exception as e:
        print(f"  {label:30s} FAIL: {type(e).__name__}")
        return None


def bench_bwd(B, H, S, D, dtype, causal, block_m, block_n, label=""):
    q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    v = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=causal, block_m=block_m, block_n=block_n, return_lse=True)
    grad_out = torch.randn_like(out)
    try:
        ms = bench_fn(
            lambda: npu_flash_attention_backward(q, k, v, out, lse, grad_out, causal=causal, block_m=block_m, block_n=block_n)
        )
        # Backward is ~2x forward FLOPs
        flops = 2 * B * H * S * S * D * 4
        tflops = flops / ms / 1e9
        print(f"  {label:30s} {ms:8.2f} ms  {tflops:6.1f} TFLOPS")
        return ms
    except Exception as e:
        print(f"  {label:30s} FAIL: {type(e).__name__}")
        return None


def bench_sdpa_ref(B, H, S, D, dtype, causal, label=""):
    q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    v = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    try:
        ms = bench_fn(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        )
        flops = 2 * B * H * S * S * D * 2
        tflops = flops / ms / 1e9
        print(f"  {label:30s} {ms:8.2f} ms  {tflops:6.1f} TFLOPS")
        return ms
    except Exception as e:
        print(f"  {label:30s} FAIL: {type(e).__name__}")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("NPU Flash Attention Benchmark (Ascend 910B3)")
    print("=" * 70)
    print(f"NPU count: {torch.npu.device_count()}")
    print()

    configs = [
        (1, 1, 128, 64, torch.float16, True),
        (1, 1, 256, 64, torch.float16, True),
        (1, 1, 512, 64, torch.float16, True),
        (1, 1, 1024, 64, torch.float16, True),
        (1, 4, 512, 64, torch.float16, True),
        (1, 1, 512, 128, torch.float16, True),
        (1, 1, 1024, 128, torch.float16, True),
        (1, 1, 128, 64, torch.float32, True),
        (1, 1, 512, 64, torch.float32, True),
    ]

    for B, H, S, D, dtype, causal in configs:
        dtype_name = "fp16" if dtype == torch.float16 else "fp32"
        print(f"\n--- B={B} H={H} S={S} D={D} {dtype_name} causal={causal} ---")

        # Reference SDPA
        bench_sdpa_ref(B, H, S, D, dtype, causal, f"SDPA ref")

        # Forward: sweep block sizes
        for bm, bn in [(32, 32), (32, 64), (64, 32), (64, 64)]:
            bench_fwd(B, H, S, D, dtype, causal, bm, bn, f"fwd bm={bm} bn={bn}")

        # Backward: only 32x32 works
        bench_bwd(B, H, S, D, dtype, causal, 32, 32, f"bwd bm=32 bn=32")
