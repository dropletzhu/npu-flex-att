import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

size = 1024
x = torch.randn(size, device='npu', dtype=torch.float32)
y = torch.randn(size, device='npu', dtype=torch.float32)
out = torch.empty(size, device='npu', dtype=torch.float32)

grid = ((size + 255) // 256,)
add_kernel[grid](x, y, out, size, BLOCK_SIZE=256)

ref = x + y
print('Triton kernel on NPU: max diff =', (out - ref).abs().max().item())
print('Test passed:', torch.allclose(out, ref))
