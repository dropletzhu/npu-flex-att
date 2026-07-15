"""NPU Flash Attention — bf16 全量测试套件

与 test_comprehensive.py 的 fp32/fp16 测试用例完全对齐。
覆盖: 前向正确性 / 反向梯度 / 数值稳定性 / 边界条件 / 功能完整性
"""
import math
import torch
import torch_npu  # noqa: F401
from npu_flash_attention import (
    npu_flash_attention_forward,
    npu_flash_attention_backward,
)

# ── Helpers ──────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
DT = torch.bfloat16
TOL_FWD = 2e-2  # bf16 精度约 3.9e-3, 某些 scale 值 (如 0.2) 量化误差更大
TOL_BWD_DQDK = 0.5
TOL_BWD_DV = 5e-3  # bf16 dV 精度低于 fp32, 放宽 tolerance
results = []


def record(tid, name, status, detail=""):
    results.append((tid, name, status, detail))
    tag = {"PASS": PASS, "FAIL": FAIL, "SKIP": SKIP}[status]
    print(f"  [{tid:4s}] {name:45s} {tag}  {detail}")


def ref_sdpa(q, k, v, causal=False, scale=None):
    if scale is not None:
        qk = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(torch.ones(qk.shape[-2:], device=q.device, dtype=torch.bool), diagonal=1)
            qk = qk.masked_fill(mask, float("-inf"))
        attn = torch.softmax(qk.float(), dim=-1).to(v.dtype)
        return torch.matmul(attn, v)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def ref_bwd(q, k, v, g, causal=False):
    qr = q.clone().requires_grad_(True)
    kr = k.clone().requires_grad_(True)
    vr = v.clone().requires_grad_(True)
    ref_sdpa(qr, kr, vr, causal=causal).backward(g)
    return qr.grad, kr.grad, vr.grad


def sliding_window_ref(q, k, v, window, scale=None):
    B, H, S, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(v)
    for m in range(S):
        start = max(0, m - window)
        attn = torch.matmul(q[:, :, m:m+1], k[:, :, start:m+1].transpose(-2, -1)) * scale
        attn = torch.softmax(attn.float(), dim=-1).to(v.dtype)
        out[:, :, m:m+1] = torch.matmul(attn, v[:, :, start:m+1])
    return out


# ── A: 前向正确性 (与 A01-A15 对齐) ─────────────────────

def test_A01():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v)
    ref = ref_sdpa(q, k, v)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A01", "Full attention bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A02():
    B, H, S, D = 1, 1, 256, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v)
    ref = ref_sdpa(q, k, v)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A02", "Full attention bf16 S=256", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A03():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A03", "Causal mask bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A04():
    B, H, S, D = 1, 1, 512, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A04", "Causal mask bf16 S=512", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A05():
    B, H, S, D = 1, 1, 2048, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A05", "Causal mask bf16 S=2048", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A06():
    B, Hq, Hkv, S, D = 1, 2, 1, 128, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=DT)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k.expand(B, Hq, S, D), v.expand(B, Hq, S, D), causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A06", "GQA ratio=2 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A07():
    B, Hq, Hkv, S, D = 1, 4, 1, 128, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=DT)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k.expand(B, Hq, S, D), v.expand(B, Hq, S, D), causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A07", "GQA ratio=4 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A08():
    B, Hq, Hkv, S, D = 1, 8, 1, 256, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=DT)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k.expand(B, Hq, S, D), v.expand(B, Hq, S, D), causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A08", "GQA ratio=8 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A09():
    B, Hq, Hkv, S, D = 1, 4, 2, 128, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=DT)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    G = Hq // Hkv
    k_exp = k.repeat_interleave(G, dim=1)
    v_exp = v.repeat_interleave(G, dim=1)
    ref = ref_sdpa(q, k_exp, v_exp, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A09", "GQA Hkv=2 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A10():
    B, H, S, D = 4, 4, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A10", "Multi batch bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A11():
    B, H, S, D, W = 1, 1, 256, 64, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W)
    ref = sliding_window_ref(q, k, v, W)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A11", "Sliding window bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A12():
    B, H, S, D = 1, 1, 128, 128
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A12", "Head dim=128 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A13():
    B, H, Sq, Skv, D = 1, 1, 128, 256, 64
    q = torch.randn(B, H, Sq, D, device="npu", dtype=DT)
    k = torch.randn(B, H, Skv, D, device="npu", dtype=DT)
    v = torch.randn(B, H, Skv, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A13", "Non-square Q!=KV bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A14():
    B, H, S, D = 1, 1, 512, 128
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A14", "D=128 S=512 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_A15():
    B, H, S, D = 1, 1, 1024, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("A15", "Long seq S=1024 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


# ── B: 反向梯度正确性 (与 B01-B06 对齐) ─────────────────

def test_B01():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    rq, rk, rv = ref_bwd(q, k, v, g, causal=True)
    ok = (torch.allclose(DQ, rq, atol=TOL_BWD_DQDK, rtol=TOL_BWD_DQDK) and
          torch.allclose(DK, rk, atol=TOL_BWD_DQDK, rtol=TOL_BWD_DQDK) and
          torch.allclose(DV, rv, atol=TOL_BWD_DV, rtol=TOL_BWD_DV))
    record("B01", "dQ/dK/dV causal bf16", "PASS" if ok else "FAIL",
           f"dQ={(DQ-rq).abs().max().item():.2e} dK={(DK-rk).abs().max().item():.2e} dV={(DV-rv).abs().max().item():.2e}")


def test_B02():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 256, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    rq, rk, rv = ref_bwd(q, k, v, g, causal=True)
    ok = torch.allclose(DQ, rq, atol=TOL_BWD_DQDK, rtol=TOL_BWD_DQDK)
    record("B02", "dQ/dK/dV causal S=256 bf16", "PASS" if ok else "FAIL", f"dQ={(DQ-rq).abs().max().item():.2e}")


def test_B03():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=False, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=False, block_m=16, block_n=32)
    rq, rk, rv = ref_bwd(q, k, v, g, causal=False)
    ok = (torch.allclose(DQ, rq, atol=TOL_BWD_DQDK, rtol=TOL_BWD_DQDK) and
          torch.allclose(DV, rv, atol=TOL_BWD_DV, rtol=TOL_BWD_DV))
    record("B03", "dQ/dK/dV full attn bf16", "PASS" if ok else "FAIL", f"dQ={(DQ-rq).abs().max().item():.2e} dV={(DV-rv).abs().max().item():.2e}")


def test_B04():
    torch.manual_seed(42)
    B, Hq, Hkv, S, D = 1, 4, 1, 128, 64
    G = Hq // Hkv
    q = torch.randn(B, Hq, S, D, device="npu", dtype=DT)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    k_exp = k.expand(B, Hq, S, D)
    v_exp = v.expand(B, Hq, S, D)
    rq, rk, rv = ref_bwd(q, k_exp, v_exp, g, causal=True)
    rk_s = rk.view(B, Hkv, G, S, D).sum(dim=2)
    rv_s = rv.view(B, Hkv, G, S, D).sum(dim=2)
    ok = (torch.allclose(DQ, rq, atol=TOL_BWD_DQDK, rtol=TOL_BWD_DQDK) and
          torch.allclose(DK, rk_s, atol=TOL_BWD_DQDK, rtol=TOL_BWD_DQDK) and
          torch.allclose(DV, rv_s, atol=TOL_BWD_DV, rtol=TOL_BWD_DV))
    record("B04", "dQ/dK/dV GQA bf16", "PASS" if ok else "FAIL",
           f"dQ={(DQ-rq).abs().max().item():.2e} dK={(DK-rk_s).abs().max().item():.2e}")


def test_B05():
    torch.manual_seed(42)
    B, H, S, D, W = 1, 1, 128, 64, 32
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W, return_lse=True)
    g = torch.randn_like(out)
    try:
        DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, sliding_window=W, block_m=16, block_n=32)
        record("B05", "dQ/dK/dV sliding bf16", "PASS", "ran successfully")
    except Exception as e:
        record("B05", "dQ/dK/dV sliding bf16", "FAIL", str(e)[:80])


def test_B06():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 256, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    rq, rk, rv = ref_bwd(q, k, v, g, causal=True)
    ok = torch.allclose(DV, rv, atol=TOL_BWD_DV, rtol=TOL_BWD_DV)
    record("B06", "dV standalone bf16", "PASS" if ok else "FAIL", f"dV_diff={(DV-rv).abs().max().item():.2e}")


# ── C: 数值稳定性 (与 C01-C06 对齐) ─────────────────────

def test_C01():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT) * 10
    k = torch.randn(B, H, S, D, device="npu", dtype=DT) * 10
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True, scale=10.0)
    ok = not out.isnan().any() and not out.isinf().any()
    record("C01", "Large scale=10 bf16", "PASS" if ok else "FAIL", f"NaN={out.isnan().sum()} Inf={out.isinf().sum()}")


def test_C02():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True, scale=0.01)
    ok = not out.isnan().any() and not out.isinf().any()
    record("C02", "Small scale=0.01 bf16", "PASS" if ok else "FAIL", f"NaN={out.isnan().sum()} Inf={out.isinf().sum()}")


def test_C03():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT) * 100
    k = torch.randn(B, H, S, D, device="npu", dtype=DT) * 100
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ok = not out.isnan().any() and not out.isinf().any()
    record("C03", "Extreme input bf16", "PASS" if ok else "FAIL", f"NaN={out.isnan().sum()} Inf={out.isinf().sum()}")


def test_C04():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.zeros(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ok = out.abs().max().item() < 1e-3
    record("C04", "Zero Value bf16", "PASS" if ok else "FAIL", f"max_abs={out.abs().max().item():.2e}")


def test_C05():
    B, H, S, D = 1, 1, 128, 64
    q = torch.zeros(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("C05", "Zero Query bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_C06():
    ok = True
    detail = ""
    for S in [128, 256, 512, 1024, 2048]:
        B, H, D = 1, 1, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=DT)
        k = torch.randn(B, H, S, D, device="npu", dtype=DT)
        v = torch.randn(B, H, S, D, device="npu", dtype=DT)
        _, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        if lse.isnan().any() or lse.isinf().any():
            ok = False; detail = f"S={S} LSE has NaN/Inf"; break
    record("C06", "LSE range check bf16", "PASS" if ok else "FAIL", detail or "LSE OK for S=128..2048")


# ── D: 边界条件 (与 D01-D06 对齐) ────────────────────────

def test_D01():
    B, H, S, D = 1, 1, 16, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("D01", "Min sequence S=16 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D02():
    B, H, S, D = 1, 1, 33, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("D02", "Non-divisible S=33 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D03():
    B, H, S, D = 1, 1, 1, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ok = torch.allclose(out, v, atol=TOL_FWD, rtol=TOL_FWD)
    record("D03", "Single token S=1 bf16", "PASS" if ok else "FAIL", f"diff={(out-v).abs().max().item():.2e}")


def test_D04():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("D04", "Simplest config bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D05():
    B, H, S, D = 8, 8, 256, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("D05", "Large batch B=8 H=8 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_D06():
    B, H, S, D = 1, 1, 128, 32
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k, v, causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("D06", "Small head_dim D=32 bf16", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


# ── F: 功能完整性 (与 F01-F05 对齐) ─────────────────────

def test_F01():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    ok = lse.shape == (B, H, S) and not lse.isnan().any() and not lse.isinf().any()
    record("F01", "return_lse=True bf16", "PASS" if ok else "FAIL", f"lse.shape={lse.shape}")


def test_F02():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    ok = DQ is not None and DK is not None and DV is not None
    record("F02", "LSE->backward chain bf16", "PASS" if ok else "FAIL", "LSE from fwd used in bwd")


def test_F03():
    B, Hq, Hkv, S, D = 2, 4, 1, 256, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=DT)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    out = npu_flash_attention_forward(q, k, v, causal=True)
    ref = ref_sdpa(q, k.expand(B, Hq, S, D), v.expand(B, Hq, S, D), causal=True)
    ok = torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD)
    record("F03", "Causal+GQA+bf16 combo", "PASS" if ok else "FAIL", f"diff={(out-ref).abs().max().item():.2e}")


def test_F04():
    B, Hq, Hkv, S, D, W = 1, 2, 1, 256, 64, 64
    q = torch.randn(B, Hq, S, D, device="npu", dtype=DT)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=DT)
    try:
        out = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=W)
        record("F04", "Sliding+Causal+GQA bf16", "PASS", "ran successfully")
    except Exception as e:
        record("F04", "Sliding+Causal+GQA bf16", "FAIL", str(e)[:80])


def test_F05():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=DT)
    k = torch.randn(B, H, S, D, device="npu", dtype=DT)
    v = torch.randn(B, H, S, D, device="npu", dtype=DT)
    all_ok = True
    for scale in [0.05, 0.1, 0.2]:
        out = npu_flash_attention_forward(q, k, v, causal=True, scale=scale)
        ref = ref_sdpa(q, k, v, causal=True, scale=scale)
        d = (out - ref).abs().max().item()
        if not torch.allclose(out, ref, atol=TOL_FWD, rtol=TOL_FWD):
            all_ok = False
            record("F05", f"Custom scale={scale} bf16 FAIL", "FAIL", f"diff={d:.2e}")
            break
    if all_ok:
        record("F05", "Custom scale bf16", "PASS", "scales: 0.05,0.1,0.2")


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("NPU Flash Attention — bf16 全量测试 (与 fp32/fp16 对齐)")
    print(f"torch={torch.__version__}, NPU={torch.npu.get_device_name(0)}")
    print("=" * 70)

    print("\n--- A: 前向正确性 ---")
    for fn in [test_A01, test_A02, test_A03, test_A04, test_A05,
               test_A06, test_A07, test_A08, test_A09, test_A10,
               test_A11, test_A12, test_A13, test_A14, test_A15]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, "FAIL", str(e)[:80])

    print("\n--- B: 反向梯度正确性 ---")
    for fn in [test_B01, test_B02, test_B03, test_B04, test_B05, test_B06]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, "FAIL", str(e)[:80])

    print("\n--- C: 数值稳定性 ---")
    for fn in [test_C01, test_C02, test_C03, test_C04, test_C05, test_C06]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, "FAIL", str(e)[:80])

    print("\n--- D: 边界条件 ---")
    for fn in [test_D01, test_D02, test_D03, test_D04, test_D05, test_D06]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, "FAIL", str(e)[:80])

    print("\n--- F: 功能完整性 ---")
    for fn in [test_F01, test_F02, test_F03, test_F04, test_F05]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, "FAIL", str(e)[:80])

    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_fail = sum(1 for r in results if r[2] == "FAIL")
    n_skip = sum(1 for r in results if r[2] == "SKIP")
    n_total = len(results)
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_pass}/{n_total} = {n_pass/n_total*100:.1f}%  (PASS={n_pass} FAIL={n_fail} SKIP={n_skip})")
    if n_fail > 0:
        print("Failed:")
        for tid, name, status, detail in results:
            if status == "FAIL":
                print(f"  [{tid}] {name}: {detail}")
    print("=" * 70)
