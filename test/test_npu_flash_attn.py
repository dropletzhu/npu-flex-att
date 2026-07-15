"""Test NPU Flash Attention forward + backward."""
import math
import torch
import torch_npu  # noqa: F401
from npu_flash_attention import npu_flex_attention, npu_flash_attention_forward, npu_flash_attention_backward

def test_fwd_causal_fp32():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    max_diff = (out - ref).abs().max().item()
    print(f"[fwd causal fp32] max_diff={max_diff:.6e} PASS={torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return max_diff


def test_fwd_causal_fp16():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    max_diff = (out - ref).abs().max().item()
    print(f"[fwd causal fp16] max_diff={max_diff:.6e} PASS={torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return max_diff


def test_fwd_gqa_fp16():
    B, Hq, Hkv, S, D = 1, 4, 1, 128, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)

    out = npu_flash_attention_forward(q, k, v, causal=True)
    k_exp = k.expand(B, Hq, S, D)
    v_exp = v.expand(B, Hq, S, D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
    max_diff = (out - ref).abs().max().item()
    print(f"[fwd GQA fp16]   max_diff={max_diff:.6e} PASS={torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return max_diff


def test_fwd_sliding_window():
    B, H, S, D = 1, 1, 256, 64
    W = 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    out = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W)
    ref = torch.zeros_like(out)
    for m in range(S):
        start = max(0, m - W)
        attn = q[0, 0, m] @ k[0, 0, start:m+1].T * (1.0 / math.sqrt(D))
        attn = torch.softmax(attn, dim=-1)
        ref[0, 0, m] = attn @ v[0, 0, start:m+1]
    max_diff = (out - ref).abs().max().item()
    print(f"[fwd sliding fp32] max_diff={max_diff:.6e} PASS={torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return max_diff


def test_fwd_full_attn():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    out = npu_flash_attention_forward(q, k, v, causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    max_diff = (out - ref).abs().max().item()
    print(f"[fwd full fp32]  max_diff={max_diff:.6e} PASS={torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return max_diff


def test_bwd_causal_fp32():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, block_m=32, block_n=32, return_lse=True)

    grad_out = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, grad_out, causal=True, block_m=32, block_n=32)

    # Reference
    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    ref.backward(grad_out)

    print(f"[bwd causal fp32] dQ max_diff={(DQ - q_ref.grad).abs().max().item():.4e}")
    print(f"[bwd causal fp32] dK max_diff={(DK - k_ref.grad).abs().max().item():.4e}")
    print(f"[bwd causal fp32] dV max_diff={(DV - v_ref.grad).abs().max().item():.4e}")
    print(f"[bwd causal fp32] dQ PASS={torch.allclose(DQ, q_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"[bwd causal fp32] dK PASS={torch.allclose(DK, k_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"[bwd causal fp32] dV PASS={torch.allclose(DV, v_ref.grad, atol=0.2, rtol=0.2)}")


if __name__ == "__main__":
    print("=" * 60)
    print("NPU Flash Attention Tests")
    print("=" * 60)

    tests = [
        ("Full Attention fp32", lambda: test_fwd_full_attn()),
        ("Causal fp32", lambda: test_fwd_causal_fp32()),
        ("Causal fp16", lambda: test_fwd_causal_fp16()),
        ("GQA fp16", lambda: test_fwd_gqa_fp16()),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"[{name}] FAIL: {type(e).__name__}: {str(e)[:200]}")
        print()

    try:
        test_fwd_sliding_window()
    except Exception as e:
        print(f"[Sliding Window] FAIL: {type(e).__name__}: {str(e)[:200]}")
    print()

    try:
        test_bwd_causal_fp32()
    except Exception as e:
        print(f"[Backward] FAIL: {type(e).__name__}: {str(e)[:300]}")
