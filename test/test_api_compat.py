"""Phase 0: API Compatibility Verification for Triton on NPU.

Tests these Triton APIs on NPU:
1. tl.dot - Matrix multiplication (2D tiles)
2. tl.math.exp2 - Base-2 exponential
3. tl.math.log2 - Base-2 logarithm
4. tl.trans - Matrix transpose
5. tl.dot with input_precision parameter
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def dot_test_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * M + tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)  # (M, N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def exp2_test_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < n)
    result = tl.math.exp2(x)
    tl.store(out_ptr + offs, result, mask=offs < n)


@triton.jit
def log2_test_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < n)
    x = tl.maximum(x, 1e-10)  # avoid log(0)
    result = tl.math.log2(x)
    tl.store(out_ptr + offs, result, mask=offs < n)


@triton.jit
def trans_test_kernel(a_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    a = tl.load(a_ptr + offs_m[:, None] * N + offs_n[None, :])
    a_t = tl.trans(a)  # (N, M)
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :], a_t)


@triton.jit
def dot_precision_test_kernel(
    a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b, input_precision="ieee")
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)


def test_dot(dtype_name, dtype):
    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device="npu", dtype=dtype)
    b = torch.randn(K, N, device="npu", dtype=dtype)
    c = torch.empty(M, N, device="npu", dtype=dtype)
    grid = (triton.cdiv(M, 64),)
    dot_test_kernel[grid](a, b, c, 64, 64, 64)
    ref = a.float() @ b.float()
    max_diff = (c.float() - ref).abs().max().item()
    print(f"  [tl.dot {dtype_name}] max_diff={max_diff:.6e}")
    return True


def test_exp2(dtype_name, dtype):
    n = 128
    x = torch.randn(n, device="npu", dtype=dtype)
    out = torch.empty(n, device="npu", dtype=dtype)
    exp2_test_kernel[(1,)](x, out, n, BLOCK_SIZE=128)
    ref = torch.pow(2.0, x.float())
    max_diff = (out.float() - ref).abs().max().item()
    print(f"  [tl.math.exp2 {dtype_name}] max_diff={max_diff:.6e}")
    return True


def test_log2(dtype_name, dtype):
    n = 128
    x = torch.rand(n, device="npu", dtype=dtype) + 1.0
    out = torch.empty(n, device="npu", dtype=dtype)
    log2_test_kernel[(1,)](x, out, n, BLOCK_SIZE=128)
    ref = torch.log2(x.float())
    max_diff = (out.float() - ref).abs().max().item()
    print(f"  [tl.math.log2 {dtype_name}] max_diff={max_diff:.6e}")
    return True


def test_trans(dtype_name, dtype):
    M, N = 64, 64
    a = torch.randn(M, N, device="npu", dtype=dtype)
    out = torch.empty(N, M, device="npu", dtype=dtype)
    trans_test_kernel[(1,)](a, out, 64, 64)
    ref = a.float().t()
    max_diff = (out.float() - ref).abs().max().item()
    print(f"  [tl.trans {dtype_name}] max_diff={max_diff:.6e}")
    return True


def test_dot_precision(dtype_name, dtype):
    M, N, K = 64, 64, 64
    a = torch.randn(M, K, device="npu", dtype=dtype)
    b = torch.randn(K, N, device="npu", dtype=dtype)
    c = torch.empty(M, N, device="npu", dtype=dtype)
    grid = (triton.cdiv(M, 64),)
    dot_precision_test_kernel[grid](a, b, c, 64, 64, 64)
    ref = a.float() @ b.float()
    max_diff = (c.float() - ref).abs().max().item()
    print(f"  [tl.dot input_precision=ieee {dtype_name}] max_diff={max_diff:.6e}")
    return True


def run_test(name, fn, dtype_name, dtype):
    try:
        fn(dtype_name, dtype)
        print(f"  => PASS")
        return True
    except Exception as e:
        print(f"  => FAIL: {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("Phase 0: API Compatibility Verification on NPU")
    print("=" * 60)
    print(f"torch version: {torch.__version__}")
    print(f"triton-ascend version: {triton.__version__}")
    print(f"NPU available: {torch.npu.is_available()}")
    print(f"NPU count: {torch.npu.device_count()}")
    print()

    results = {}
    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        print(f"--- Testing with {dtype_name} ---")
        results[f"dot_{dtype_name}"] = run_test("dot", test_dot, dtype_name, dtype)
        results[f"exp2_{dtype_name}"] = run_test("exp2", test_exp2, dtype_name, dtype)
        results[f"log2_{dtype_name}"] = run_test("log2", test_log2, dtype_name, dtype)
        results[f"trans_{dtype_name}"] = run_test("trans", test_trans, dtype_name, dtype)
        results[f"dot_precision_{dtype_name}"] = run_test(
            "dot_precision", test_dot_precision, dtype_name, dtype
        )
        print()

    print("=" * 60)
    print("Summary:")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
