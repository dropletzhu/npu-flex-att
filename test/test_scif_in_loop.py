"""Test: does scf.if inside a for loop cause bishengir-compile failure?"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def flash_attn_with_if_kernel(
    Q, K, V, O,
    N_BLOCKS,
    stride_qm, stride_qk, stride_kn, stride_kk,
    stride_vn, stride_vk, stride_om, stride_ok,
    Q_LEN, KV_LEN,
    SM_SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    RCP_LN2: tl.constexpr = 1.44269504

    n_blocks = tl.load(N_BLOCKS)
    # Use KV_LEN (kernel arg) as loop bound, NOT tl.load value
    max_blocks = tl.cdiv(KV_LEN, BLOCK_N)

    for start_n_blk in range(0, max_blocks):
        start_n = start_n_blk * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # scf.if inside loop - check if we should process this block
        if start_n_blk < n_blocks:
            k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                         mask=offs_n[:, None] < KV_LEN, other=0.0)
            k = tl.trans(k)
            qk = tl.dot(q, k) * SM_SCALE
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
            p = tl.math.exp2((qk - m_ij[:, None]) * RCP_LN2)
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                         mask=offs_n[:, None] < KV_LEN, other=0.0)
            acc = tl.dot(p.to(v.dtype), v, acc)
            m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)


if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention with scf.if inside loop on NPU")
    print("=" * 60)

    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        B, H, S, D = 1, 1, 128, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        v = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        o = torch.empty_like(q)
        n_blocks = torch.tensor([S // 64], device="npu", dtype=torch.int32)

        try:
            grid = (triton.cdiv(S, 64),)
            flash_attn_with_if_kernel[grid](
                q, k, v, o, n_blocks,
                q.stride(2), q.stride(3), k.stride(2), k.stride(3),
                v.stride(2), v.stride(3), o.stride(2), o.stride(3),
                S, S, 1.0 / (D ** 0.5),
                BLOCK_M=64, BLOCK_N=64, HEAD_DIM=D,
            )
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            max_diff = (o - ref).abs().max().item()
            print(f"[{dtype_name}] Max diff: {max_diff:.6e}")
            print(f"[{dtype_name}] Pass: {torch.allclose(o, ref, atol=1e-2, rtol=1e-2)}")
        except Exception as e:
            err_str = str(e)[:300]
            print(f"[{dtype_name}] FAIL: {type(e).__name__}: {err_str}")
