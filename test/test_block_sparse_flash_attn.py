"""Test flash attention with block-sparse indexing (like flex_attention uses).
This isolates whether the indirect kv_indices loading causes bishengir-compile issues.
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def flash_attn_block_sparse_kernel(
    Q, K, V, O,
    KV_NUM_BLKS, KV_IDX,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, Q_LEN, KV_LEN,
    SM_SCALE,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
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

    # Block-sparse: load num_blocks and indices
    kv_num_blocks_ptr = KV_NUM_BLKS + off_z * H + off_h + pid_m
    kv_num_blocks = tl.load(kv_num_blocks_ptr)
    kv_indices_ptr = KV_IDX + (off_z * H + off_h) * kv_num_blocks

    for blk_idx in range(0, kv_num_blocks):
        # Indirect: load the KV block index
        kv_blk_idx = tl.load(kv_indices_ptr + blk_idx)
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


def flash_attn_block_sparse(q, k, v, block_size=128, scale=None):
    B, H, S, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)

    # Build simple causal block mask
    n_blocks = (S + block_size - 1) // block_size
    kv_num_blocks = torch.zeros(B, H, n_blocks, dtype=torch.int32, device="npu")
    kv_indices = torch.zeros(B, H, n_blocks, n_blocks, dtype=torch.int32, device="npu")

    for b in range(B):
        for h in range(H):
            for m in range(n_blocks):
                kv_num_blocks[b, h, m] = m + 1
                for j in range(m + 1):
                    kv_indices[b, h, m, j] = j

    grid = (triton.cdiv(S, 64), B, H)
    flash_attn_block_sparse_kernel[grid](
        q, k, v, o,
        kv_num_blocks, kv_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, S, S,
        scale,
        SPARSE_KV_BLOCK_SIZE=block_size,
        BLOCK_M=64, BLOCK_N=64, HEAD_DIM=D,
    )
    return o


if __name__ == "__main__":
    print("=" * 60)
    print("Block-Sparse Flash Attention on NPU")
    print("=" * 60)

    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        B, H, S, D = 1, 1, 128, 64
        q = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        k = torch.randn(B, H, S, D, device="npu", dtype=dtype)
        v = torch.randn(B, H, S, D, device="npu", dtype=dtype)

        try:
            out = flash_attn_block_sparse(q, k, v, block_size=64)
            ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            max_diff = (out - ref).abs().max().item()
            print(f"[{dtype_name}] Max diff: {max_diff:.6e}")
            print(f"[{dtype_name}] Pass: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
        except Exception as e:
            print(f"[{dtype_name}] FAIL: {type(e).__name__}: {e}")
