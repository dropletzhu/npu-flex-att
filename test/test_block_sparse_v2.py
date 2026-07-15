"""Block-sparse flash attention with FIXED pointer arithmetic.
The previous version had a bug: used runtime kv_num_blocks as stride.
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def flash_attn_block_sparse_v2_kernel(
    Q, K, V, O,
    KV_NUM_BLKS, KV_IDX,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, Q_LEN, KV_LEN,
    SM_SCALE,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    MAX_KV_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=offs_m[:, None] < Q_LEN, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    RCP_LN2: tl.constexpr = 1.44269504

    # Use compile-time constant stride for kv_indices offset
    # KV_IDX shape: [Z, H, Q_BLOCKS, MAX_KV_BLOCKS]
    # stride for (z, h) = Q_BLOCKS * MAX_KV_BLOCKS (compile-time known)
    q_blocks = tl.cdiv(Q_LEN, SPARSE_KV_BLOCK_SIZE)
    bh_offset = (off_z * H + off_h) * q_blocks * MAX_KV_BLOCKS

    kv_num_blocks = tl.load(KV_NUM_BLKS + off_z * H * q_blocks + off_h * q_blocks + pid_m)
    kv_indices_base = KV_IDX + bh_offset + pid_m * MAX_KV_BLOCKS

    for blk_idx in range(0, kv_num_blocks):
        kv_blk_idx = tl.load(kv_indices_base + blk_idx)
        start_n = kv_blk_idx * SPARSE_KV_BLOCK_SIZE
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
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

        v = tl.load(V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                     mask=offs_n[:, None] < KV_LEN, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
             acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < Q_LEN)


if __name__ == "__main__":
    print("=" * 60)
    print("Block-Sparse Flash Attention v2 on NPU")
    print("=" * 60)

    BLOCK = 64
    B, H, S, D = 1, 1, 128, 64
    n_blocks = (S + BLOCK - 1) // BLOCK  # 2

    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        v = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        o = torch.empty_like(q)

        # Build causal block mask
        kv_num_blocks = torch.zeros(B, H, n_blocks, dtype=torch.int32, device="npu")
        kv_indices = torch.zeros(B, H, n_blocks, n_blocks, dtype=torch.int32, device="npu")
        for m_idx in range(n_blocks):
            kv_num_blocks[0, 0, m_idx] = m_idx + 1
            for j in range(m_idx + 1):
                kv_indices[0, 0, m_idx, j] = j

        try:
            grid = (triton.cdiv(S, 64), B, H)
            flash_attn_block_sparse_v2_kernel[grid](
                q, k, v, o,
                kv_num_blocks, kv_indices,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                B, H, S, S,
                1.0 / (D ** 0.5),
                SPARSE_KV_BLOCK_SIZE=BLOCK,
                MAX_KV_BLOCKS=n_blocks,
                BLOCK_M=64, BLOCK_N=64, HEAD_DIM=D,
            )
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            max_diff = (o - ref).abs().max().item()
            print(f"[{dtype_name}] Max diff: {max_diff:.6e}")
            print(f"[{dtype_name}] Pass: {torch.allclose(o, ref, atol=1e-2, rtol=1e-2)}")
        except Exception as e:
            err_str = str(e)[:200]
            print(f"[{dtype_name}] FAIL: {type(e).__name__}: {err_str}")
