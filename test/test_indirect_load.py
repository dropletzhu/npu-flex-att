"""Test: indirect load (load index from memory, use as offset) on NPU."""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def test_indirect_load_kernel(data_ptr, indices_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.load(indices_ptr + pid)
    offs = idx * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    val = tl.load(data_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), val,
             mask=tl.arange(0, BLOCK) < BLOCK)


if __name__ == "__main__":
    x = torch.randn(1024, device="npu", dtype=torch.float32)
    indices = torch.tensor([0, 1, 2, 3], device="npu", dtype=torch.int32)
    out = torch.empty(1024, device="npu", dtype=torch.float32)
    try:
        test_indirect_load_kernel[(4,)](x, indices, out, 1024, BLOCK=256)
        print("Indirect load test: PASS")
        print("Max diff:", (out[:1024] - x[:1024]).abs().max().item())
    except Exception as e:
        print(f"Indirect load test: FAIL - {type(e).__name__}: {e}")
