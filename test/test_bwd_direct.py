"""Direct backward test (bypassing autograd) to isolate AICore issue."""
import math
import torch
import torch_npu  # noqa: F401
from npu_flash_attention import npu_flash_attention_forward, npu_flash_attention_backward

B, H, S, D = 1, 1, 128, 64
q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

with torch.no_grad():
    out, lse = npu_flash_attention_forward(q, k, v, causal=True, block_m=32, block_n=32, return_lse=True)

grad_out = torch.randn_like(out)
print("Forward done. Testing backward...")
DQ, DK, DV = npu_flash_attention_backward(q, k, v, out, lse, grad_out, causal=True, block_m=32, block_n=32)
print(f"DQ max: {DQ.abs().max().item()}")
print(f"DK max: {DK.abs().max().item()}")
print(f"DV max: {DV.abs().max().item()}")
print("Direct backward PASS")
