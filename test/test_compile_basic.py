"""Quick test: verify torch.compile + triton works on NPU."""
import torch
import torch_npu  # noqa: F401


def fn(x):
    return x * 2 + 1


x = torch.randn(128, 128, device="npu", dtype=torch.float32)
compiled = torch.compile(fn, backend="inductor")
out = compiled(x)
ref = fn(x)
print(f"Basic compile test: max_diff={ (out - ref).abs().max().item()}")
print(f"PASS: {torch.allclose(out, ref)}")
