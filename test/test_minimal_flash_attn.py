"""Minimal Flash Attention kernel to test bishengir-compile compatibility.
This bypasses Inductor entirely - just raw Triton on NPU.
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, Q_LEN, KV_LEN,
    SM_SCALE,
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

    for start_n in range(0, KV_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                     mask=offs_n[:, None] < KV_LEN, other=0.0)
        k = tl.trans(k)
        qk = tl.dot(q, k) * SM_SCALE

        # Causal mask
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


def flash_attention(q, k, v, scale=None):
    B, H, S, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    o = torch.empty_like(q)

    grid = (triton.cdiv(S, 64), B, H)
    flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, S, S,
        scale,
        BLOCK_M=64, BLOCK_N=64, HEAD_DIM=D,
    )
    return o


def test_fp32():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float32)

    out = flash_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    max_diff = (out - ref).abs().max().item()
    print(f"[fp32] Max diff: {max_diff:.6e}")
    print(f"[fp32] Pass: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return max_diff


def test_fp16():
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

    out = flash_attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    max_diff = (out - ref).abs().max().item()
    print(f"[fp16] Max diff: {max_diff:.6e}")
    print(f"[fp16] Pass: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return max_diff


if __name__ == "__main__":
    print("=" * 60)
    print("Minimal Flash Attention on NPU")
    print("=" * 60)
    try:
        test_fp32()
    except Exception as e:
        print(f"[fp32] FAIL: {type(e).__name__}: {e}")
    try:
        test_fp16()
    except Exception as e:
        print(f"[fp16] FAIL: {type(e).__name__}: {e}")
