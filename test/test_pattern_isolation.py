"""Isolate the bishengir-compile failure: test combinations of patterns."""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def test_runtime_loop_bound(n_blocks, out_ptr, BLOCK: tl.constexpr):
    """Test 1: runtime loop bound (no indirect load, no dot)"""
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n_blocks):
        acc += 1.0
    tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)


@triton.jit
def test_runtime_loop_with_indirect_load(
    indices_ptr, n_blocks, out_ptr, BLOCK: tl.constexpr
):
    """Test 2: runtime loop bound + indirect load (load index in loop)"""
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n_blocks):
        idx = tl.load(indices_ptr + i)
        acc += idx.to(tl.float32)
    tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)


@triton.jit
def test_runtime_loop_with_dot(
    a_ptr, n_blocks, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
):
    """Test 3: runtime loop bound + tl.dot (no indirect load)"""
    pid = tl.program_id(0)
    acc = tl.zeros([M, N], dtype=tl.float32)
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    for i in range(0, n_blocks):
        offs_k = i * K + tl.arange(0, K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(a_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + pid * M * N + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def test_runtime_loop_indirect_dot(
    a_ptr, indices_ptr, n_blocks, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
):
    """Test 4: runtime loop bound + indirect load + tl.dot (full pattern)"""
    pid = tl.program_id(0)
    acc = tl.zeros([M, N], dtype=tl.float32)
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    for i in range(0, n_blocks):
        blk_idx = tl.load(indices_ptr + i)
        offs_k = blk_idx * K + tl.arange(0, K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(a_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(out_ptr + pid * M * N + offs_m[:, None] * N + offs_n[None, :], acc)


if __name__ == "__main__":
    print("=" * 60)
    print("Pattern Isolation Tests on NPU")
    print("=" * 60)

    # Test 1: runtime loop bound only
    try:
        out = torch.empty(64, device="npu", dtype=torch.float32)
        test_runtime_loop_bound[(1,)](5, out, BLOCK=64)
        print("[1] Runtime loop bound: PASS")
    except Exception as e:
        print(f"[1] Runtime loop bound: FAIL - {type(e).__name__}")

    # Test 2: runtime loop + indirect load
    try:
        indices = torch.tensor([0, 1, 2, 3, 4], device="npu", dtype=torch.int32)
        out = torch.empty(64, device="npu", dtype=torch.float32)
        test_runtime_loop_with_indirect_load[(1,)](indices, 5, out, BLOCK=64)
        print("[2] Runtime loop + indirect load: PASS")
    except Exception as e:
        print(f"[2] Runtime loop + indirect load: FAIL - {type(e).__name__}")

    # Test 3: runtime loop + dot (no indirect)
    try:
        a = torch.randn(64, 64, device="npu", dtype=torch.float32)
        out = torch.empty(64 * 64, device="npu", dtype=torch.float32)
        test_runtime_loop_with_dot[(1,)](a, 2, out, M=64, N=64, K=32)
        print("[3] Runtime loop + dot (no indirect): PASS")
    except Exception as e:
        print(f"[3] Runtime loop + dot (no indirect): FAIL - {type(e).__name__}")

    # Test 4: runtime loop + indirect load + dot (full pattern)
    try:
        a = torch.randn(64, 64, device="npu", dtype=torch.float32)
        indices = torch.tensor([0, 1], device="npu", dtype=torch.int32)
        out = torch.empty(64 * 64, device="npu", dtype=torch.float32)
        test_runtime_loop_indirect_dot[(1,)](a, indices, 2, out, M=64, N=64, K=32)
        print("[4] Runtime loop + indirect + dot: PASS")
    except Exception as e:
        print(f"[4] Runtime loop + indirect + dot: FAIL - {type(e).__name__}")
