"""Test backward with smaller block sizes to diagnose AICore exception."""
import math
import torch
import torch_npu  # noqa: F401
from npu_flash_attention import npu_flash_attention_forward, npu_flash_attention_backward


def test_bwd_with_block_size(block_m, block_n, dtype=torch.float32):
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
    v = torch.randn(B, H, S, D, device="npu", dtype=dtype)

    with torch.no_grad():
        out, lse = npu_flash_attention_forward(
            q, k, v, causal=True, block_m=block_m, block_n=block_n, return_lse=True
        )

    grad_out = torch.randn_like(out)
    try:
        DQ, DK, DV = npu_flash_attention_backward(
            q, k, v, out, lse, grad_out,
            causal=True, block_m=block_m, block_n=block_n,
        )
        print(f"[block_m={block_m}, block_n={block_n}] DQ max: {DQ.abs().max().item():.4e}")
        print(f"[block_m={block_m}, block_n={block_n}] DK max: {DK.abs().max().item():.4e}")
        print(f"[block_m={block_m}, block_n={block_n}] DV max: {DV.abs().max().item():.4e}")
        return True
    except Exception as e:
        err = str(e)[:200]
        print(f"[block_m={block_m}, block_n={block_n}] FAIL: {type(e).__name__}: {err}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Backward kernel: block size sweep")
    print("=" * 60)

    configs = [
        (32, 32),
        (32, 64),
        (64, 32),
        (16, 64),
        (64, 16),
        (16, 32),
        (32, 16),
        (16, 16),
    ]

    for bm, bn in configs:
        result = test_bwd_with_block_size(bm, bn)
        if result:
            print(f"  => First working config: block_m={bm}, block_n={bn}")
            break
        print()
