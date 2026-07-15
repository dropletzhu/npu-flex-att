"""Phase 4: Forward Propagation Verification (Basic) - fp32 + smaller blocks

Tests flex_attention without score_mod or mask_mod.
"""
import torch
import torch_npu  # noqa: F401
from torch.nn.attention.flex_attention import flex_attention, FlexKernelOptions

B, H, S, D = 1, 1, 64, 64
q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

# Try with smaller block sizes
kernel_options = FlexKernelOptions(
    fwd_block_m=64,
    fwd_block_n=64,
)
compiled_flex = torch.compile(flex_attention, backend="inductor")

with torch.no_grad():
    out = compiled_flex(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f"Output shape: {out.shape}")
    print(f"Max diff: {(out - ref).abs().max().item()}")
    print(f"Mean diff: {(out - ref).abs().mean().item()}")
    print(f"Pass: {torch.allclose(out, ref, atol=1e-3, rtol=1e-3)}")
