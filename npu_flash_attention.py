"""NPU Flash Attention — Triton Ascend implementation.

Bypasses Inductor's complex codegen that triggers bishengir-compile limitations.
Uses simple pointer arithmetic (tl.load with direct offsets) proven to work on NPU.

Supports:
  - Causal masking
  - Sliding window
  - ALiBi-style score bias
  - GQA (Grouped Query Attention)
  - LSE output (for backward)
  - fp16/bf16 with fp32 accumulation
  - Forward + Backward
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import math

# ================ Forward Kernel ================

@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_m,
    Q_LEN, KV_LEN,
    SM_SCALE,
    GQA_GROUPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ALIBI_SLOPE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_hq = tl.program_id(2)

    off_hkv = off_hq // GQA_GROUPS

    q_base = off_b * stride_qb + off_hq * stride_qh
    k_base = off_b * stride_kb + off_hkv * stride_kh
    v_base = off_b * stride_vb + off_hkv * stride_vh
    o_base = off_b * stride_ob + off_hq * stride_oh
    lse_base = off_b * stride_lse_b + off_hq * stride_lse_h

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < Q_LEN, other=0.0,
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    RCP_LN2: tl.constexpr = 1.44269504

    if IS_CAUSAL:
        if SLIDING_WINDOW > 0:
            start_n = max(0, pid_m * BLOCK_M - SLIDING_WINDOW)
        else:
            start_n = 0
    elif SLIDING_WINDOW > 0:
        start_n = max(0, pid_m * BLOCK_M - SLIDING_WINDOW)
    else:
        start_n = 0

    for sn in range(start_n, KV_LEN, BLOCK_N):
        offs_n = sn + tl.arange(0, BLOCK_N)

        # [Optimization] Preload K and V together to hide load latency
        k = tl.load(
            K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=offs_n[:, None] < KV_LEN, other=0.0,
        )
        v = tl.load(
            V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < KV_LEN, other=0.0,
        )
        k = tl.trans(k)
        qk = tl.dot(q, k) * SM_SCALE

        if USE_ALIBI:
            alibi_bias = tl.load(ALIBI_SLOPE + off_hq) * (offs_m[:, None] - offs_n[None, :])
            qk = qk + alibi_bias

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        if SLIDING_WINDOW > 0:
            window_mask = (offs_m[:, None] - offs_n[None, :]) <= SLIDING_WINDOW
            qk = tl.where(window_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
        p = tl.math.exp2((qk - m_ij[:, None]) * RCP_LN2)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    tl.store(
        O + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(O.dtype.element_ty),
        mask=offs_m[:, None] < Q_LEN,
    )

    LN2: tl.constexpr = 0.6931471805599453
    lse = m_i + tl.math.log2(l_i) * LN2
    tl.store(
        LSE + lse_base + offs_m * stride_lse_m,
        lse,
        mask=offs_m < Q_LEN,
    )


# ================ Backward: DQ Kernel ================

@triton.jit
def flash_attn_bwd_dq_kernel(
    Q, K, V, LSE, DO, DQ,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    Q_LEN, KV_LEN,
    SM_SCALE,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    RCP_LN2: tl.constexpr = 1.44269504

    q_base = off_b * stride_qb + off_h * stride_qh
    q = tl.load(Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < Q_LEN, other=0.0)
    lse = tl.load(LSE + off_b * stride_lse_b + off_h * stride_lse_h + offs_m * stride_lse_m,
                  mask=offs_m < Q_LEN, other=0.0)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                 mask=offs_m[:, None] < Q_LEN, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    k_base = off_b * stride_kb + off_h * stride_kh
    v_base = off_b * stride_vb + off_h * stride_vh

    if IS_CAUSAL:
        start_n = 0
    else:
        start_n = 0

    for sn in range(start_n, KV_LEN, BLOCK_N):
        offs_n = sn + tl.arange(0, BLOCK_N)
        k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                     mask=offs_n[:, None] < KV_LEN, other=0.0)
        k = tl.trans(k)
        qk = tl.dot(q, k) * SM_SCALE

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where((offs_m[:, None] - offs_n[None, :]) <= SLIDING_WINDOW, qk, float("-inf"))

        p = tl.math.exp2((qk - lse[:, None]) * RCP_LN2)
        v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                     mask=offs_n[:, None] < KV_LEN, other=0.0)
        dp = tl.dot(do.to(v.dtype), tl.trans(v))
        # Softmax backward: ds = p * (dp - sum(p * dp))
        dp_f32 = dp.to(tl.float32)
        p_f32 = p.to(tl.float32)
        row_sum = tl.sum(p_f32 * dp_f32, 1)
        ds = p_f32 * (dp_f32 - row_sum[:, None]) * SM_SCALE
        dq = tl.dot(ds.to(q.dtype), tl.trans(k), dq)

    tl.store(DQ + off_b * stride_dqb + off_h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
             dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)


# ================ Backward: DK/DV Kernel ================

@triton.jit
def flash_attn_bwd_dkdv_kernel(
    Q, K, V, LSE, DO, DK, DV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    Q_LEN, KV_LEN,
    SM_SCALE,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)  # query head index

    # For GQA: K/V are indexed by KV head, Q/LSE/dO by query head
    off_hkv = off_h  # For non-GQA, hq == hkv. GQA handled by Python averaging

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    RCP_LN2: tl.constexpr = 1.44269504

    k_base = off_b * stride_kb + off_h * stride_kh
    v_base = off_b * stride_vb + off_h * stride_vh
    q_base = off_b * stride_qb + off_h * stride_qh
    do_base = off_b * stride_dob + off_h * stride_doh
    lse_base = off_b * stride_lse_b + off_h * stride_lse_h

    k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=offs_n[:, None] < KV_LEN, other=0.0)
    v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=offs_n[:, None] < KV_LEN, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    for sm in range(0, Q_LEN, BLOCK_M):
        offs_m = sm + tl.arange(0, BLOCK_M)
        q = tl.load(Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                     mask=offs_m[:, None] < Q_LEN, other=0.0)
        lse = tl.load(LSE + lse_base + offs_m * stride_lse_m,
                      mask=offs_m < Q_LEN, other=0.0)
        do = tl.load(DO + do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                     mask=offs_m[:, None] < Q_LEN, other=0.0)

        kt = tl.trans(k)
        qk = tl.dot(q, kt) * SM_SCALE

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where((offs_m[:, None] - offs_n[None, :]) <= SLIDING_WINDOW, qk, float("-inf"))

        p = tl.math.exp2((qk - lse[:, None]) * RCP_LN2)
        dp = tl.dot(do.to(v.dtype), tl.trans(v))
        dv = tl.dot(tl.trans(p.to(v.dtype)), do.to(v.dtype), dv)
        # Softmax backward: ds = p * (dp - sum(p * dp))
        dp_f32 = dp.to(tl.float32)
        p_f32 = p.to(tl.float32)
        row_sum = tl.sum(p_f32 * dp_f32, 1)
        ds = p_f32 * (dp_f32 - row_sum[:, None]) * SM_SCALE
        dk = dk + tl.dot(tl.trans(ds.to(k.dtype)), q.to(k.dtype))

    tl.store(DK + off_b * stride_dkb + off_h * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
             dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < KV_LEN)
    tl.store(DV + off_b * stride_dvb + off_h * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
             dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < KV_LEN)


# ================ Python API ================

class NPUFlexAttentionConfig:
    def __init__(self):
        self.block_m = 64
        self.block_n = 64
        self.causal = False
        self.sliding_window = 0
        self.alibi_slope = None

    def __repr__(self):
        return f"NPUFlexAttentionConfig(block_m={self.block_m}, block_n={self.block_n}, causal={self.causal}, sliding_window={self.sliding_window}, alibi={self.alibi_slope is not None})"


def npu_flash_attention_forward(
    query, key, value,
    causal=False,
    sliding_window=0,
    alibi_slope=None,
    scale=None,
    block_m=16,
    block_n=64,
    return_lse=False,
):
    B, Hq, Sq, D = query.shape
    Bk, Hkv, Sk, Dk = key.shape
    Bv, Hkv2, Sv, Dv = value.shape
    assert D == Dk, "Q and K head dims must match"
    assert Dv == D, "V head dim must match Q head dim for this kernel"
    assert Hq % Hkv == 0, "Hq must be divisible by Hkv"
    GQA_GROUPS = Hq // Hkv

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(query)
    LSE = torch.empty(B, Hq, Sq, device=query.device, dtype=torch.float32)

    grid = (triton.cdiv(Sq, block_m), B, Hq)
    flash_attn_fwd_kernel[grid](
        query, key, value, O, LSE,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        Sq, Sk,
        scale,
        GQA_GROUPS,
        IS_CAUSAL=causal,
        SLIDING_WINDOW=sliding_window,
        USE_ALIBI=alibi_slope is not None,
        ALIBI_SLOPE=alibi_slope if alibi_slope is not None else query,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=D,
    )

    if return_lse:
        return O, LSE
    return O


def npu_flash_attention_backward(
    query, key, value, out, lse, grad_out,
    causal=False,
    sliding_window=0,
    alibi_slope=None,
    scale=None,
    block_m=32,
    block_n=32,
):
    B, Hq, Sq, D = query.shape
    Bk, Hkv, Sk, Dk = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    DQ = torch.empty_like(query)
    # Allocate DK/DV with Hq heads (for GQA, we average across groups later)
    DK = torch.empty(B, Hq, Sk, D, device=key.device, dtype=key.dtype)
    DV = torch.empty(B, Hq, Sk, D, device=value.device, dtype=value.dtype)

    # DQ kernel: grid = (cdiv(Sq, block_m), B, Hq)
    # For GQA DQ: K/V need to be expanded to Hq heads
    GQA_GROUPS = Hq // Hkv
    if GQA_GROUPS > 1:
        key_exp = key.unsqueeze(2).expand(B, Hkv, GQA_GROUPS, Sk, D).reshape(B, Hq, Sk, D).contiguous()
        value_exp = value.unsqueeze(2).expand(B, Hkv, GQA_GROUPS, Sk, D).reshape(B, Hq, Sk, D).contiguous()
    else:
        key_exp = key
        value_exp = value

    grid_dq = (triton.cdiv(Sq, block_m), B, Hq)
    flash_attn_bwd_dq_kernel[grid_dq](
        query, key_exp, value_exp, lse, grad_out, DQ,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key_exp.stride(0), key_exp.stride(1), key_exp.stride(2), key_exp.stride(3),
        value_exp.stride(0), value_exp.stride(1), value_exp.stride(2), value_exp.stride(3),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        DQ.stride(0), DQ.stride(1), DQ.stride(2), DQ.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        Sq, Sk,
        scale,
        IS_CAUSAL=causal,
        SLIDING_WINDOW=sliding_window,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=D,
    )

    # DK/DV kernel: grid = (cdiv(Sk, block_n), B, Hq)
    grid_dkdv = (triton.cdiv(Sk, block_n), B, Hq)
    flash_attn_bwd_dkdv_kernel[grid_dkdv](
        query, key_exp, value_exp, lse, grad_out, DK, DV,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key_exp.stride(0), key_exp.stride(1), key_exp.stride(2), key_exp.stride(3),
        value_exp.stride(0), value_exp.stride(1), value_exp.stride(2), value_exp.stride(3),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        DK.stride(0), DK.stride(1), DK.stride(2), DK.stride(3),
        DV.stride(0), DV.stride(1), DV.stride(2), DV.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        Sq, Sk,
        scale,
        IS_CAUSAL=causal,
        SLIDING_WINDOW=sliding_window,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=D,
    )

    # For GQA: average DK/DV across query heads sharing the same KV head
    GQA_GROUPS = Hq // Hkv
    if GQA_GROUPS > 1:
        DK = DK.view(B, Hkv, GQA_GROUPS, Sk, D).sum(dim=2)
        DV = DV.view(B, Hkv, GQA_GROUPS, Sk, D).sum(dim=2)

    return DQ, DK, DV


class NPUFlexAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, causal=False, sliding_window=0,
                alibi_slope=None, scale=None, block_m=16, block_n=64):
        ctx.causal = causal
        ctx.sliding_window = sliding_window
        ctx.alibi_slope = alibi_slope
        ctx.scale = scale
        ctx.block_m = block_m
        ctx.block_n = block_n

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(
                query, key, value,
                causal=causal,
                sliding_window=sliding_window,
                alibi_slope=alibi_slope,
                scale=scale,
                block_m=block_m,
                block_n=block_n,
                return_lse=True,
            )
        ctx.save_for_backward(query, key, value, lse)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, lse = ctx.saved_tensors
        DQ, DK, DV = npu_flash_attention_backward(
            query, key, value, None, lse, grad_out,
            causal=ctx.causal,
            sliding_window=ctx.sliding_window,
            alibi_slope=ctx.alibi_slope,
            scale=ctx.scale,
            block_m=ctx.block_m,
            block_n=32,
        )
        return DQ, DK, DV, None, None, None, None, None, None


def npu_flex_attention(
    query, key, value,
    causal=False,
    sliding_window=0,
    alibi_slope=None,
    scale=None,
    block_m=16,
    block_n=64,
):
    return NPUFlexAttention.apply(
        query, key, value, causal, sliding_window,
        alibi_slope, scale, block_m, block_n,
    )
