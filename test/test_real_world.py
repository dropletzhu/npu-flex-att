"""真实场景补充测试: 非连续张量, 输入验证, 确定性, 大规模训练场景."""
import math
import torch
import torch_npu  # noqa: F401
from npu_flash_attention import npu_flash_attention_forward, npu_flash_attention_backward

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def record(tid, name, ok, detail=""):
    results.append((tid, name, ok, detail))
    print(f"  [{tid:4s}] {name:50s} {PASS if ok else FAIL}  {detail}")

def ref_sdpa(q, k, v, causal=False):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

# ── R1: 非连续张量测试 ──────────────────────────────────

def test_R01():
    """非连续 Q (transpose 后)"""
    B, H, S, D = 1, 2, 128, 64
    q = torch.randn(B, S, H, D, device="npu", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    try:
        out = npu_flash_attention_forward(q, k, v, causal=True)
        ref = ref_sdpa(q, k, v, causal=True)
        ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        record("R01", "非连续 Q (transpose)", ok, f"diff={(out-ref).abs().max().item():.2e} contig={q.is_contiguous()}")
    except Exception as e:
        record("R01", "非连续 Q (transpose)", False, str(e)[:80])

def test_R02():
    """非连续 K (slice 后)"""
    B, H, S, D = 1, 1, 256, 64
    k_full = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = k_full[:, :, :128, :]  # 非连续 slice
    q = torch.randn(B, H, 128, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, 128, D, device="npu", dtype=torch.float16)
    try:
        out = npu_flash_attention_forward(q, k, v, causal=True)
        ref = ref_sdpa(q, k, v, causal=True)
        ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        record("R02", "非连续 K (slice)", ok, f"diff={(out-ref).abs().max().item():.2e} contig={k.is_contiguous()}")
    except Exception as e:
        record("R02", "非连续 K (slice)", False, str(e)[:80])

def test_R03():
    """stride=0 (broadcast K/V)"""
    B, Hkv, S, D = 1, 1, 128, 64
    q = torch.randn(B, 4, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16).expand(B, 4, S, D)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16).expand(B, 4, S, D)
    try:
        out = npu_flash_attention_forward(q, k, v, causal=True)
        ref = ref_sdpa(q, k, v, causal=True)
        ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        record("R03", "stride=0 broadcast K/V", ok, f"diff={(out-ref).abs().max().item():.2e}")
    except Exception as e:
        record("R03", "stride=0 broadcast K/V", False, str(e)[:80])

# ── R2: 输入验证测试 ────────────────────────────────────

def test_R04():
    """Q 和 K head_dim 不匹配"""
    q = torch.randn(1, 1, 128, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 1, 128, 128, device="npu", dtype=torch.float16)
    v = torch.randn(1, 1, 128, 64, device="npu", dtype=torch.float16)
    try:
        npu_flash_attention_forward(q, k, v, causal=True)
        record("R04", "Q/K head_dim 不匹配", False, "应抛出异常但未抛出")
    except (AssertionError, RuntimeError) as e:
        record("R04", "Q/K head_dim 不匹配", True, f"正确拒绝: {type(e).__name__}")

def test_R05():
    """Hq 不能被 Hkv 整除"""
    q = torch.randn(1, 3, 128, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 2, 128, 64, device="npu", dtype=torch.float16)
    v = torch.randn(1, 2, 128, 64, device="npu", dtype=torch.float16)
    try:
        npu_flash_attention_forward(q, k, v, causal=True)
        record("R05", "Hq%Hkv!=0", False, "应抛出异常但未抛出")
    except (AssertionError, RuntimeError) as e:
        record("R05", "Hq%Hkv!=0", True, f"正确拒绝: {type(e).__name__}")

# ── R3: 确定性测试 ───────────────────────────────────────

def test_R06():
    """相同输入两次运行结果一致"""
    torch.manual_seed(123)
    q = torch.randn(1, 2, 256, 64, device="npu", dtype=torch.float32)
    k = torch.randn(1, 2, 256, 64, device="npu", dtype=torch.float32)
    v = torch.randn(1, 2, 256, 64, device="npu", dtype=torch.float32)
    out1 = npu_flash_attention_forward(q, k, v, causal=True)
    out2 = npu_flash_attention_forward(q, k, v, causal=True)
    ok = torch.equal(out1, out2)
    record("R06", "确定性 (两次结果一致)", ok, f"max_diff={(out1-out2).abs().max().item():.2e}")

# ── R4: 大规模训练场景 ──────────────────────────────────

def test_R07():
    """典型 LLM 训练: B=2 H=16 S=512 D=128 fp16"""
    B, H, S, D = 2, 16, 512, 128
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    try:
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        ref = ref_sdpa(q, k, v, causal=True)
        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        record("R07", "LLM 训练 B=2 H=16 S=512 D=128", ok, f"diff={(out-ref).abs().max().item():.2e}")
    except Exception as e:
        record("R07", "LLM 训练 B=2 H=16 S=512 D=128", False, str(e)[:80])

def test_R08():
    """GQA LLM: B=1 Hq=32 Hkv=8 S=1024 D=128 fp16"""
    B, Hq, Hkv, S, D = 1, 32, 8, 1024, 128
    q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
    try:
        out = npu_flash_attention_forward(q, k, v, causal=True)
        k_exp = k.repeat_interleave(Hq // Hkv, dim=1)
        v_exp = v.repeat_interleave(Hq // Hkv, dim=1)
        ref = ref_sdpa(q, k_exp, v_exp, causal=True)
        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        record("R08", "GQA LLM Hq=32 Hkv=8 S=1024 D=128", ok, f"diff={(out-ref).abs().max().item():.2e}")
    except Exception as e:
        record("R08", "GQA LLM Hq=32 Hkv=8 S=1024 D=128", False, str(e)[:80])

def test_R09():
    """长序列 S=4096"""
    B, H, S, D = 1, 2, 4096, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    try:
        out = npu_flash_attention_forward(q, k, v, causal=True)
        ref = ref_sdpa(q, k, v, causal=True)
        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        record("R09", "长序列 S=4096", ok, f"diff={(out-ref).abs().max().item():.2e}")
    except Exception as e:
        record("R09", "长序列 S=4096", False, str(e)[:80])

# ── R5: 内存布局测试 ────────────────────────────────────

def test_R10():
    """不同 stride 模式 (非标准 stride)"""
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16).contiguous()
    # 交换最后两个维度的 stride 但保持数据
    q_custom = q.as_strided(q.shape, (S*D, D, D, 1))  # 非标准 stride
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    try:
        out = npu_flash_attention_forward(q_custom, k, v, causal=True)
        ref = ref_sdpa(q_custom, k, v, causal=True)
        ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        record("R10", "非标准 stride", ok, f"diff={(out-ref).abs().max().item():.2e}")
    except Exception as e:
        record("R10", "非标准 stride", False, str(e)[:80])

# ── R6: 反向确定性 ──────────────────────────────────────

def test_R11():
    """反向两次运行结果一致"""
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ1, DK1, DV1 = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    DQ2, DK2, DV2 = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    ok = torch.equal(DQ1, DQ2) and torch.equal(DK1, DK2) and torch.equal(DV1, DV2)
    record("R11", "反向确定性", ok, f"dQ_diff={(DQ1-DQ2).abs().max().item():.2e}")

# ── R7: LSE 正确性验证 ──────────────────────────────────

def test_R12():
    """LSE = logsumexp 验证 (与 PyTorch 对比)"""
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    # 参考 LSE: log(sum(exp(qk*scale, dim=-1)))
    scale = 1.0 / math.sqrt(D)
    qk = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(S, S, device="npu", dtype=torch.bool), diagonal=1)
    qk = qk.masked_fill(mask, float("-inf"))
    ref_lse = torch.logsumexp(qk, dim=-1)  # natural log
    # 我们的 LSE 也是 natural log (已转换)
    diff = (lse - ref_lse).abs().max().item()
    ok = torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)
    record("R12", "LSE = logsumexp 验证", ok, f"diff={diff:.2e}")

if __name__ == "__main__":
    print("=" * 70)
    print("真实场景补充测试")
    print("=" * 70)

    print("\n--- R1: 非连续张量 ---")
    for fn in [test_R01, test_R02, test_R03]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, False, str(e)[:80])

    print("\n--- R2: 输入验证 ---")
    for fn in [test_R04, test_R05]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, False, str(e)[:80])

    print("\n--- R3: 确定性 ---")
    test_R06()

    print("\n--- R4: 大规模训练场景 ---")
    for fn in [test_R07, test_R08, test_R09]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, False, str(e)[:80])

    print("\n--- R5: 内存布局 ---")
    try: test_R10()
    except Exception as e: record("R10", "非标准 stride", False, str(e)[:80])

    print("\n--- R6: 反向确定性 ---")
    try: test_R11()
    except Exception as e: record("R11", "反向确定性", False, str(e)[:80])

    print("\n--- R7: LSE 正确性 ---")
    try: test_R12()
    except Exception as e: record("R12", "LSE 验证", False, str(e)[:80])

    print("\n" + "=" * 70)
    n_pass = sum(1 for r in results if r[2])
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} = {n_pass/n_total*100:.1f}%")
    if n_pass < n_total:
        print("Failed:")
        for tid, name, ok, detail in results:
            if not ok:
                print(f"  [{tid}] {name}: {detail}")
    print("=" * 70)
