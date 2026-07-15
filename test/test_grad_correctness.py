"""Verify backward gradient correctness against PyTorch SDPA reference."""
import math
import torch
import torch_npu  # noqa: F401
from npu_flash_attention import npu_flash_attention_forward, npu_flash_attention_backward


def grad_check_causal_fp32():
    B, H, S, D = 1, 1, 128, 64
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    with torch.no_grad():
        out, lse = npu_flash_attention_forward(
            q, k, v, causal=True, block_m=32, block_n=32, return_lse=True
        )

    grad_out = torch.randn_like(out)

    DQ, DK, DV = npu_flash_attention_backward(
        q, k, v, out, lse, grad_out,
        causal=True, block_m=32, block_n=32,
    )

    # Reference using PyTorch SDPA
    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=True)
    ref.backward(grad_out)

    dq_diff = (DQ - q_ref.grad).abs().max().item()
    dk_diff = (DK - k_ref.grad).abs().max().item()
    dv_diff = (DV - v_ref.grad).abs().max().item()

    q_norm = q_ref.grad.abs().max().item()
    k_norm = k_ref.grad.abs().max().item()
    v_norm = v_ref.grad.abs().max().item()

    print(f"[grad check causal fp32]")
    print(f"  dQ: max_diff={dq_diff:.4e}, ref_max={q_norm:.4e}, rel_err={dq_diff/max(q_norm,1e-6):.4e}")
    print(f"  dK: max_diff={dk_diff:.4e}, ref_max={k_norm:.4e}, rel_err={dk_diff/max(k_norm,1e-6):.4e}")
    print(f"  dV: max_diff={dv_diff:.4e}, ref_max={v_norm:.4e}, rel_err={dv_diff/max(v_norm,1e-6):.4e}")
    print(f"  dQ PASS: {torch.allclose(DQ, q_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"  dK PASS: {torch.allclose(DK, k_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"  dV PASS: {torch.allclose(DV, v_ref.grad, atol=0.2, rtol=0.2)}")


def grad_check_full_fp32():
    B, H, S, D = 1, 1, 128, 64
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    with torch.no_grad():
        out, lse = npu_flash_attention_forward(
            q, k, v, causal=False, block_m=32, block_n=32, return_lse=True
        )

    grad_out = torch.randn_like(out)

    DQ, DK, DV = npu_flash_attention_backward(
        q, k, v, out, lse, grad_out,
        causal=False, block_m=32, block_n=32,
    )

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref)
    ref.backward(grad_out)

    dq_diff = (DQ - q_ref.grad).abs().max().item()
    dk_diff = (DK - k_ref.grad).abs().max().item()
    dv_diff = (DV - v_ref.grad).abs().max().item()

    print(f"[grad check full attn fp32]")
    print(f"  dQ: max_diff={dq_diff:.4e}")
    print(f"  dK: max_diff={dk_diff:.4e}")
    print(f"  dV: max_diff={dv_diff:.4e}")
    print(f"  dQ PASS: {torch.allclose(DQ, q_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"  dK PASS: {torch.allclose(DK, k_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"  dV PASS: {torch.allclose(DV, v_ref.grad, atol=0.2, rtol=0.2)}")


def grad_check_gqa_fp32():
    B, Hq, Hkv, S, D = 1, 4, 1, 128, 64
    torch.manual_seed(42)
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float32)

    with torch.no_grad():
        out, lse = npu_flash_attention_forward(
            q, k, v, causal=True, block_m=32, block_n=32, return_lse=True
        )

    grad_out = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(
        q, k, v, out, lse, grad_out,
        causal=True, block_m=32, block_n=32,
    )

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    k_exp = k_ref.expand(B, Hq, S, D)
    v_exp = v_ref.expand(B, Hq, S, D)
    ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_exp, v_exp, is_causal=True)
    ref.backward(grad_out)

    dq_diff = (DQ - q_ref.grad).abs().max().item()
    dk_diff = (DK - k_ref.grad).abs().max().item()
    dv_diff = (DV - v_ref.grad).abs().max().item()

    print(f"[grad check GQA fp32]")
    print(f"  dQ: max_diff={dq_diff:.4e}")
    print(f"  dK: max_diff={dk_diff:.4e}")
    print(f"  dV: max_diff={dv_diff:.4e}")
    print(f"  dQ PASS: {torch.allclose(DQ, q_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"  dK PASS: {torch.allclose(DK, k_ref.grad, atol=0.2, rtol=0.2)}")
    print(f"  dV PASS: {torch.allclose(DV, v_ref.grad, atol=0.2, rtol=0.2)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Backward Gradient Correctness Verification")
    print("=" * 60)
    print()
    try:
        grad_check_causal_fp32()
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")
    print()
    try:
        grad_check_full_fp32()
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")
    print()
    try:
        grad_check_gqa_fp32()
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")
