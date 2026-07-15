"""Phase 4: Forward Propagation Verification (Basic)

Tests flex_attention without score_mod or mask_mod.
Uses torch.compile(backend="inductor") to trigger Triton kernel generation.
"""
import torch
import torch_npu  # noqa: F401
from torch.nn.attention.flex_attention import flex_attention

B, H, S, D = 1, 1, 128, 64
q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

compiled_flex = torch.compile(flex_attention, backend="inductor")

with torch.no_grad():
    out = compiled_flex(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f"Output shape: {out.shape}")
    print(f"Max diff: {(out - ref).abs().max().item()}")
    print(f"Mean diff: {(out - ref).abs().mean().item()}")
    print(f"Pass: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
