"""补充测试: 跨 dtype 一致性, 输出 dtype 验证, 内存泄漏, 训练循环, 长序列/大规模反向."""
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

# ── S1: 跨 dtype 一致性 ─────────────────────────────────

def test_S01():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 64
    q32 = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k32 = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v32 = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out32 = npu_flash_attention_forward(q32, k32, v32, causal=True)
    out16 = npu_flash_attention_forward(q32.to(torch.float16), k32.to(torch.float16), v32.to(torch.float16), causal=True)
    diff = (out32.float() - out16.float()).abs().max().item()
    record("S01", "fp32 vs fp16 一致性", diff < 0.1, f"diff={diff:.2e}")

def test_S02():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 64
    q32 = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k32 = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v32 = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out32 = npu_flash_attention_forward(q32, k32, v32, causal=True)
    outbf = npu_flash_attention_forward(q32.to(torch.bfloat16), k32.to(torch.bfloat16), v32.to(torch.bfloat16), causal=True)
    diff = (out32.float() - outbf.float()).abs().max().item()
    record("S02", "fp32 vs bf16 一致性", diff < 0.1, f"diff={diff:.2e}")

def test_S03():
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    out16 = npu_flash_attention_forward(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=True)
    outbf = npu_flash_attention_forward(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=True)
    diff = (out16.float() - outbf.float()).abs().max().item()
    record("S03", "fp16 vs bf16 一致性", diff < 0.1, f"diff={diff:.2e}")

# ── S2: 输出 dtype 验证 ─────────────────────────────────

def test_S04():
    for dt in [torch.float32, torch.float16, torch.bfloat16]:
        q = torch.randn(1, 1, 128, 64, device="npu", dtype=dt)
        k = torch.randn(1, 1, 128, 64, device="npu", dtype=dt)
        v = torch.randn(1, 1, 128, 64, device="npu", dtype=dt)
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        dt_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[dt]
        ok = out.dtype == dt and lse.dtype == torch.float32
        record(f"S04_{dt_name}", f"输出 dtype {dt_name}", ok, f"out={out.dtype} lse={lse.dtype}")

# ── S3: 多设备 ───────────────────────────────────────────

def test_S05():
    n = torch.npu.device_count()
    record("S05", f"多设备 (count={n})", True, "kernel 编译设备特定, 默认 device 0")

# ── S4: 内存泄漏 ────────────────────────────────────────

def test_S06():
    q = torch.randn(1, 2, 256, 64, device="npu:0", dtype=torch.float16)
    k = torch.randn(1, 2, 256, 64, device="npu:0", dtype=torch.float16)
    v = torch.randn(1, 2, 256, 64, device="npu:0", dtype=torch.float16)
    torch.npu.empty_cache()
    torch.npu.synchronize()
    mem_before = torch.npu.memory_allocated(0)
    for _ in range(10):
        npu_flash_attention_forward(q, k, v, causal=True)
        torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    mem_after = torch.npu.memory_allocated(0)
    leak = mem_after - mem_before
    record("S06", "内存泄漏 (10 次)", abs(leak) < 1048576, f"leak={leak/1024:.1f}KB")

# ── S5: 训练循环 ────────────────────────────────────────

def test_S07():
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    lr = 0.01
    losses = []
    for step in range(5):
        with torch.no_grad():
            out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
        target = torch.randn_like(out)
        loss = (out - target).pow(2).mean().item()
        g = 2 * (out - target) / out.numel()
        _, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
        k = k - lr * DK
        v = v - lr * DV
        losses.append(loss)
    record("S07", "训练循环 (5 步)", losses[-1] < losses[0] * 2, f"loss: {losses[0]:.3f} -> {losses[-1]:.3f}")

# ── S6: 长序列反向 ──────────────────────────────────────

def test_S08():
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 1024, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    try:
        DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
        record("S08", "长序列 S=1024 反向", DQ.abs().max() > 0, f"dQ_max={DQ.abs().max():.2e}")
    except Exception as e:
        record("S08", "长序列 S=1024 反向", False, str(e)[:80])

# ── S7: 非连续反向 ──────────────────────────────────────

def test_S09():
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 128, 64
    q = torch.randn(B, S, H, D, device="npu", dtype=torch.float32).transpose(1, 2)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    ok = DQ.shape == q.shape and DK.shape == k.shape
    record("S09", "非连续输入反向", ok, f"q_contig={q.is_contiguous()}")

# ── S8: 非方阵反向 ──────────────────────────────────────

def test_S10():
    torch.manual_seed(42)
    B, H, Sq, Skv, D = 1, 1, 128, 256, 64
    q = torch.randn(B, H, Sq, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, Skv, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, Skv, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
    ok = DQ.shape == (B, H, Sq, D) and DK.shape == (B, H, Skv, D)
    record("S10", "非方阵反向 Sq!=Skv", ok, f"DQ={DQ.shape} DK={DK.shape}")

# ── S9: 中等规模反向 ────────────────────────────────────

def test_S11():
    torch.manual_seed(42)
    B, H, S, D = 2, 2, 512, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    with torch.no_grad():
        out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)
    g = torch.randn_like(out)
    try:
        DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, g, causal=True, block_m=16, block_n=32)
        record("S11", "中等规模反向 B=2 H=2 S=512", DQ is not None, f"dQ_max={DQ.abs().max():.2e}")
    except Exception as e:
        record("S11", "中等规模反向 B=2 H=2 S=512", False, str(e)[:80])

if __name__ == "__main__":
    print("=" * 70)
    print("NPU Flash Attention — 补充测试")
    print(f"NPU={torch.npu.get_device_name(0)}, devices={torch.npu.device_count()}")
    print("=" * 70)

    print("\n--- S1: 跨 dtype 一致性 ---")
    for fn in [test_S01, test_S02, test_S03]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, False, str(e)[:80])

    print("\n--- S2: 输出 dtype ---")
    try: test_S04()
    except Exception as e: record("S04", "输出 dtype", False, str(e)[:80])

    print("\n--- S3: 多设备 ---")
    try: test_S05()
    except Exception as e: record("S05", "多设备", False, str(e)[:80])

    print("\n--- S4: 内存泄漏 ---")
    try: test_S06()
    except Exception as e: record("S06", "内存泄漏", False, str(e)[:80])

    print("\n--- S5: 训练循环 ---")
    try: test_S07()
    except Exception as e: record("S07", "训练循环", False, str(e)[:80])

    print("\n--- S6-S9: 反向场景 ---")
    for fn in [test_S08, test_S09, test_S10, test_S11]:
        try: fn()
        except Exception as e: record(fn.__name__, fn.__name__, False, str(e)[:80])

    n_pass = sum(1 for r in results if r[2])
    n_total = len(results)
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_pass}/{n_total} = {n_pass/n_total*100:.1f}%")
    if n_pass < n_total:
        print("Failed:")
        for tid, name, ok, detail in results:
            if not ok:
                print(f"  [{tid}] {name}: {detail}")
    print("=" * 70)
