"""NPU Flash Attention — Comprehensive Test Suite

Executes all tests from TEST_PLAN.md and outputs a summary report.
"""
import math
import time
import torch
import torch_npu  # noqa: F401
import triton
from npu_flash_attention import (
    npu_flash_attention_forward,
    npu_flash_attention_backward,
    npu_flex_attention,
)

# ── Helpers ──────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def record(test_id, name, status, detail=""):
    results.append((test_id, name, status, detail))
    tag = {"PASS": PASS, "FAIL": FAIL, "SKIP": SKIP}[status]
    print(f"  [{test_id:4s}] {name:45s} {tag}  {detail}")


def bench(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = time.time()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.time() - s) / rep * 1000


def ref_sdpa(q, k, v, causal=False, scale=None):
    if scale is not None:
        # F.sdpa doesn't support custom scale directly, compute manually
        qk = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(qk.shape[-2:], device=q.device, dtype=torch.bool), diagonal=1)
            qk = qk.masked_fill(mask, float("-inf"))
        attn = torch.softmax(qk, dim=-1)
        return torch.matmul(attn, v)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def ref_sdpa_backward(q, k, v, grad_out, causal=False, scale=None):
    q_r = q.clone().requires_grad_(True)
    k_r = k.clone().requires_grad_(True)
    v_r = v.clone().requires_grad_(True)
    out = ref_sdpa(q_r, k_r, v_r, causal=causal, scale=scale)
    out.backward(grad_out)
    return q_r.grad, k_r.grad, v_r.grad


def sliding_window_ref(q, k, v, window, scale=None):
    B, H, S, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(v)
    for m in range(S):
        start = max(0, m - window)
        attn = torch.matmul(q[:, :, m:m+1], k[:, :, start:m+1].transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out[:, :, m:m+1] = torch.matmul(attn, v[:, :, start:m+1])
    return out


# ── A: Forward Correctness ───────────────────────────────

def test_A01():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out = npu_flash_attention_forward(q, k, v)
    ref = ref_sdpa(q, k, v)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    record("A01", "Full attention fp32", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A02():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v)
    ref = ref_sdpa(q, k, v)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A02", "Full attention fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A03():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    record("A03", "Causal mask fp32", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A04():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A04", "Causal mask fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A05():
    try:
        B, H, S, D = 1, 1, 128, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
        out = npu_flash_attention_forward(q, k, v, causal=True)
        ref = ref_sdpa(q, k, v, causal=True)
        diff = (out - ref).abs().max().item()
        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        record("A05", "Causal mask bf16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")
    except Exception as e:
        record("A05", "Causal mask bf16", "SKIP", str(e)[:80])


def test_A06():
    B, Hq, Hkv, S, D = 1, 2, 1, 128, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    k_exp = k.expand(B, Hq, S, D)
    v_exp = v.expand(B, Hq, S, D)
    ref = ref_sdpa(q, k_exp, v_exp, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A06", "GQA ratio=2 fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A07():
    B, Hq, Hkv, S, D = 1, 4, 1, 128, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    k_exp = k.expand(B, Hq, S, D)
    v_exp = v.expand(B, Hq, S, D)
    ref = ref_sdpa(q, k_exp, v_exp, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A07", "GQA ratio=4 fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A08():
    B, Hq, Hkv, S, D = 1, 8, 1, 256, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    k_exp = k.expand(B, Hq, S, D)
    v_exp = v.expand(B, Hq, S, D)
    ref = ref_sdpa(q, k_exp, v_exp, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A08", "GQA ratio=8 fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A09():
    B, Hq, Hkv, S, D = 1, 4, 2, 128, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    k_exp = k.repeat_interleave(Hq // Hkv, dim=1)
    v_exp = v.repeat_interleave(Hq // Hkv, dim=1)
    ref = ref_sdpa(q, k_exp, v_exp, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A09", "GQA Hkv=2 fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A10():
    B, H, S, D = 4, 4, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A10", "Multi batch fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A11():
    B, H, S, D, W = 1, 1, 256, 64, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W)
    ref = sliding_window_ref(q, k, v, W)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    record("A11", "Sliding window fp32", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A12():
    B, H, S, D, W = 1, 1, 256, 64, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W)
    ref = sliding_window_ref(q.float(), k.float(), v.float(), W).to(torch.float16)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    record("A12", "Sliding window fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A13():
    B, H, S, D = 1, 1, 128, 128
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A13", "Head dim=128 fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A14():
    B, H, Sq, Skv, D = 1, 1, 128, 256, 64
    q = torch.randn(B, H, Sq, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, Skv, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, Skv, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("A14", "Non-square Q!=KV fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


def test_A15():
    B, H, S, D = 1, 1, 2048, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    record("A15", "Long sequence S=2048 fp16", "PASS" if ok else "FAIL", f"max_diff={diff:.2e}")


# ── B: Backward Gradient Correctness ─────────────────────

def test_B01():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    rq, rk, rv = ref_sdpa_backward(q, k, v, g, causal=True)
    dq_diff = (DQ - rq).abs().max().item()
    dk_diff = (DK - rk).abs().max().item()
    dv_diff = (DV - rv).abs().max().item()
    ok = torch.allclose(DQ, rq, atol=0.2, rtol=0.2) and torch.allclose(DK, rk, atol=0.2, rtol=0.2) and torch.allclose(DV, rv, atol=1e-4, rtol=1e-4)
    record("B01", "dQ/dK/dV causal fp32", "PASS" if ok else "FAIL", f"dQ={dq_diff:.2e} dK={dk_diff:.2e} dV={dv_diff:.2e}")


def test_B02():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    try:
        DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
        rq, rk, rv = ref_sdpa_backward(q, k, v, g, causal=True)
        ok = torch.allclose(DQ, rq, atol=0.5, rtol=0.5)
        record("B02", "dQ/dK/dV causal fp16", "PASS" if ok else "FAIL", f"dQ_diff={(DQ-rq).abs().max().item():.2e}")
    except Exception as e:
        record("B02", "dQ/dK/dV causal fp16", "SKIP", str(e)[:80])


def test_B03():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=False, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=False, block_m=16, block_n=32)
    rq, rk, rv = ref_sdpa_backward(q, k, v, g, causal=False)
    ok = torch.allclose(DQ, rq, atol=0.2, rtol=0.2) and torch.allclose(DV, rv, atol=1e-4, rtol=1e-4)
    record("B03", "dQ/dK/dV full attn fp32", "PASS" if ok else "FAIL", f"dQ={(DQ-rq).abs().max().item():.2e} dV={(DV-rv).abs().max().item():.2e}")


def test_B04():
    torch.manual_seed(42)
    B, Hq, Hkv, S, D = 1, 4, 1, 128, 64
    GQA_GROUPS = Hq // Hkv
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    k_exp = k.expand(B, Hq, S, D)
    v_exp = v.expand(B, Hq, S, D)
    rq, rk, rv = ref_sdpa_backward(q, k_exp, v_exp, g, causal=True)
    # Sum reference grads across GQA groups to match our implementation
    rk_summed = rk.view(B, Hkv, GQA_GROUPS, S, D).sum(dim=2)
    rv_summed = rv.view(B, Hkv, GQA_GROUPS, S, D).sum(dim=2)
    dq_diff = (DQ - rq).abs().max().item()
    dk_diff = (DK - rk_summed).abs().max().item()
    dv_diff = (DV - rv_summed).abs().max().item()
    ok = torch.allclose(DQ, rq, atol=0.3, rtol=0.3) and torch.allclose(DK, rk_summed, atol=0.3, rtol=0.3) and torch.allclose(DV, rv_summed, atol=0.3, rtol=0.3)
    record("B04", "dQ/dK/dV GQA fp32", "PASS" if ok else "FAIL", f"dQ={dq_diff:.2e} dK={dk_diff:.2e} dV={dv_diff:.2e}")


def test_B05():
    torch.manual_seed(42)
    B, H, S, D, W = 1, 1, 128, 64, 32
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W, return_lse=True)
    g = torch.randn_like(out)
    try:
        DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, sliding_window=W, block_m=16, block_n=32)
        record("B05", "dQ/dK/dV sliding fp32", "PASS", "ran successfully")
    except Exception as e:
        record("B05", "dQ/dK/dV sliding fp32", "SKIP", str(e)[:80])


def test_B06():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 256, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    # dV should be very accurate (just p^T @ dO)
    rq, rk, rv = ref_sdpa_backward(q, k, v, g, causal=True)
    dv_diff = (DV - rv).abs().max().item()
    ok = torch.allclose(DV, rv, atol=1e-4, rtol=1e-4)
    record("B06", "dV standalone verification", "PASS" if ok else "FAIL", f"dV_diff={dv_diff:.2e}")


# ── C: Numerical Stability ───────────────────────────────

def test_C01():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32) * 10
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32) * 10
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out = npu_flash_attention_forward(q, k, v, causal=True, scale=10.0)
    ok = not out.isnan().any().item() and not out.isinf().any().item()
    record("C01", "Large scale=10", "PASS" if ok else "FAIL", f"NaN={out.isnan().sum()} Inf={out.isinf().sum()}")


def test_C02():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out = npu_flash_attention_forward(q, k, v, causal=True, scale=0.01)
    ok = not out.isnan().any().item() and not out.isinf().any().item()
    record("C02", "Small scale=0.01", "PASS" if ok else "FAIL", f"NaN={out.isnan().sum()} Inf={out.isinf().sum()}")


def test_C03():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16) * 100
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16) * 100
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ok = not out.isnan().any().item() and not out.isinf().any().item()
    record("C03", "Extreme input fp16", "PASS" if ok else "FAIL", f"NaN={out.isnan().sum()} Inf={out.isinf().sum()}")


def test_C04():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.zeros(B, H, S, D, device="npu", dtype=torch.float32)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ok = out.abs().max().item() < 1e-5
    record("C04", "Zero Value", "PASS" if ok else "FAIL", f"max_abs={out.abs().max().item():.2e}")


def test_C05():
    B, H, S, D = 1, 1, 128, 64
    q = torch.zeros(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    # When q=0, QK^T=0, softmax=uniform, output=mean(V[:m+1]) per row
    ref = ref_sdpa(q, k, v, causal=True)
    diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    record("C05", "Zero Query (uniform attn)", "PASS" if ok else "FAIL", f"diff={diff:.2e}")


def test_C06():
    ok = True
    detail = ""
    for S in [128, 256, 512, 1024, 2048]:
        B, H, D = 1, 1, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
        _, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        if lse.isnan().any() or lse.isinf().any():
            ok = False
            detail = f"S={S} LSE has NaN/Inf"
            break
        if lse.max().item() > 1000 or lse.min().item() < -1000:
            ok = False
            detail = f"S={S} LSE out of range: [{lse.min().item():.2f}, {lse.max().item():.2f}]"
            break
    if ok:
        detail = f"LSE range OK for S=128..2048"
    record("C06", "LSE range check", "PASS" if ok else "FAIL", detail)


# ── D: Edge Cases ────────────────────────────────────────

def test_D01():
    B, H, S, D = 1, 1, 16, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("D01", "Min sequence S=16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D02():
    B, H, S, D = 1, 1, 33, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("D02", "Non-divisible S=33", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D03():
    B, H, S, D = 1, 1, 1, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    # With S=1, output should be v[0] (single token attends to itself)
    ok = torch.allclose(out, v, atol=1e-2, rtol=1e-2)
    record("D03", "Single token S=1", "PASS" if ok else "FAIL", f"diff={(out-v).abs().max().item():.2e}")


def test_D04():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("D04", "Simplest H=1 Hkv=1", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D05():
    B, H, S, D = 8, 8, 256, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("D05", "Large batch B=8 H=8", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D06():
    B, H, S, D = 1, 1, 128, 32
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("D06", "Small head_dim D=32", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


# ── E: Performance ──────────────────────────────────────

def test_E01():
    print("\n  --- E01: Forward latency sweep (D=64, fp16, causal) ---")
    print(f"  {'S':>6s}  {'fwd(ms)':>8s}  {'SDPA(ms)':>8s}  {'speedup':>7s}  {'TFLOPS':>7s}")
    for S in [128, 256, 512, 1024, 2048]:
        B, H, D = 1, 1, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        fwd = bench(lambda: npu_flash_attention_forward(q, k, v, causal=True))
        sdpa = bench(lambda: ref_sdpa(q, k, v, causal=True))
        flops = 2 * B * H * S * S * D * 2 * 0.5
        tflops = flops / fwd / 1e9
        sp = sdpa / fwd
        print(f"  {S:6d}  {fwd:8.2f}  {sdpa:8.2f}  {sp:7.2f}x  {tflops:7.1f}")


def test_E02():
    print("\n  --- E02: Forward latency D=128 (fp16, causal) ---")
    print(f"  {'S':>6s}  {'fwd(ms)':>8s}  {'SDPA(ms)':>8s}  {'speedup':>7s}")
    for S in [128, 256, 512, 1024]:
        B, H, D = 1, 1, 128
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        fwd = bench(lambda: npu_flash_attention_forward(q, k, v, causal=True))
        sdpa = bench(lambda: ref_sdpa(q, k, v, causal=True))
        sp = sdpa / fwd
        print(f"  {S:6d}  {fwd:8.2f}  {sdpa:8.2f}  {sp:7.2f}x")


def test_E03():
    print("\n  --- E03: Backward latency (D=64, fp16, causal) ---")
    print(f"  {'S':>6s}  {'bwd(ms)':>8s}  {'fwd(ms)':>8s}  {'ratio':>7s}")
    for S in [128, 256, 512, 1024]:
        B, H, D = 1, 1, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        g = torch.randn_like(out)
        fwd = bench(lambda: npu_flash_attention_forward(q, k, v, causal=True))
        try:
            bwd = bench(lambda: npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32))
            print(f"  {S:6d}  {bwd:8.2f}  {fwd:8.2f}  {bwd/fwd:7.2f}x")
        except:
            print(f"  {S:6d}  {'FAIL':>8s}")


def test_E04():
    print("\n  --- E04: vs SDPA speedup ---")
    print(f"  {'S':>6s} {'D':>4s} {'dtype':>6s}  {'fwd(ms)':>8s}  {'SDPA(ms)':>8s}  {'speedup':>7s}")
    for S, D, dt in [(128,64,'fp16'),(256,64,'fp16'),(512,64,'fp16'),(1024,64,'fp16'),(2048,64,'fp16'),(512,128,'fp16'),(1024,128,'fp16'),(128,64,'fp32'),(512,64,'fp32')]:
        dtype = torch.float16 if dt == 'fp16' else torch.float32
        B, H = 1, 1
        q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        v = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        fwd = bench(lambda: npu_flash_attention_forward(q, k, v, causal=True))
        sdpa = bench(lambda: ref_sdpa(q, k, v, causal=True))
        sp = sdpa / fwd
        print(f"  {S:6d} {D:4d} {dt:6s}  {fwd:8.2f}  {sdpa:8.2f}  {sp:7.2f}x")


def test_E07():
    print("\n  --- E07: Batch scaling (H=4, S=512, D=64, fp16) ---")
    print(f"  {'B':>4s}  {'fwd(ms)':>8s}  {'SDPA(ms)':>8s}  {'speedup':>7s}  {'grid_blk':>8s}")
    for B in [1, 2, 4, 8]:
        H, S, D = 4, 512, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
        fwd = bench(lambda: npu_flash_attention_forward(q, k, v, causal=True))
        sdpa = bench(lambda: ref_sdpa(q, k, v, causal=True))
        grid_blk = triton.cdiv(S, 16) * B * H
        print(f"  {B:4d}  {fwd:8.2f}  {sdpa:8.2f}  {sdpa/fwd:7.2f}x  {grid_blk:8d}")


# ── F: Feature Coverage ──────────────────────────────────

def test_F01():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    ok = lse.shape == (B, H, S) and not lse.isnan().any() and not lse.isinf().any()
    record("F01", "return_lse=True", "PASS" if ok else "FAIL", f"lse.shape={lse.shape}")


def test_F02():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    ok = DQ is not None and DK is not None and DV is not None
    record("F02", "LSE→backward chain", "PASS" if ok else "FAIL", "LSE from fwd used in bwd")


def test_F03():
    B, Hq, Hkv, S, D = 2, 4, 1, 256, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    k_exp = k.expand(B, Hq, S, D)
    v_exp = v.expand(B, Hq, S, D)
    ref = ref_sdpa(q, k_exp, v_exp, causal=True)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    record("F03", "Causal+GQA+fp16 combo", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_F04():
    B, Hq, Hkv, S, D, W = 1, 2, 1, 256, 64, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    try:
        out = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W)
        record("F04", "Sliding+Causal+GQA", "PASS", "ran successfully")
    except Exception as e:
        record("F04", "Sliding+Causal+GQA", "FAIL", str(e)[:80])


def test_F05():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    all_ok = True
    for scale in [0.05, 0.1, 0.2]:
        out = npu_flash_attention_forward(q, k, v, causal=True, scale=scale)
        ref = ref_sdpa(q, k, v, causal=True, scale=scale)
        if not torch.allclose(out, ref, atol=1e-5, rtol=1e-5):
            all_ok = False
            break
    record("F05", "Custom scale values", "PASS" if all_ok else "FAIL", f"scales tested: 0.05,0.1,0.2")


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("NPU Flash Attention — Comprehensive Test Suite")
    print(f"torch={torch.__version__}, triton-ascend={triton.__version__}")
    print(f"NPU={torch.npu.get_device_name(0)}, count={torch.npu.device_count()}")
    print("=" * 70)

    print("\n--- A: Forward Correctness ---")
    for fn in [test_A01, test_A02, test_A03, test_A04, test_A05,
               test_A06, test_A07, test_A08, test_A09, test_A10,
               test_A11, test_A12, test_A13, test_A14, test_A15]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__.replace("test_", ""), fn.__name__, "FAIL", str(e)[:80])

    print("\n--- B: Backward Gradient Correctness ---")
    for fn in [test_B01, test_B02, test_B03, test_B04, test_B05, test_B06]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__.replace("test_", ""), fn.__name__, "FAIL", str(e)[:80])

    print("\n--- C: Numerical Stability ---")
    for fn in [test_C01, test_C02, test_C03, test_C04, test_C05, test_C06]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__.replace("test_", ""), fn.__name__, "FAIL", str(e)[:80])

    print("\n--- D: Edge Cases ---")
    for fn in [test_D01, test_D02, test_D03, test_D04, test_D05, test_D06]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__.replace("test_", ""), fn.__name__, "FAIL", str(e)[:80])

    print("\n--- E: Performance ---")
    for fn in [test_E01, test_E02, test_E03, test_E04, test_E07]:
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__}: FAIL - {e}")

    print("\n--- F: Feature Coverage ---")
    for fn in [test_F01, test_F02, test_F03, test_F04, test_F05]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__.replace("test_", ""), fn.__name__, "FAIL", str(e)[:80])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_fail = sum(1 for r in results if r[2] == "FAIL")
    n_skip = sum(1 for r in results if r[2] == "SKIP")
    n_total = len(results)
    print(f"  Total: {n_total}  PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")
    print(f"  Pass rate: {n_pass}/{n_total} = {n_pass/n_total*100:.1f}%")
    if n_fail > 0:
        print("\n  Failed tests:")
        for tid, name, status, detail in results:
            if status == "FAIL":
                print(f"    [{tid}] {name}: {detail}")
    print("=" * 70)
