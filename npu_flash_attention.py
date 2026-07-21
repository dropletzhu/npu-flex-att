"""NPU Flash Attention — Triton Ascend implementation (L2+ optimized).

Optimizations:
  P0. Causal diagonal-aware block skipping (forward + DQ + DKDV)
  P1. DKDV loop-invariant transpose hoisting (kt, vt)
  P2. Dynamic backward BLOCK_N: 32 for S<=512, 64 for longer
  P3. DKDV do-to-v dtype cache reuse
  P4. Causal mask split: unmasked prefix + masked suffix (forward + DKDV)
  P5. Double-buffer (multibuffer=True) for K/V load latency hiding
  P6. GQA backward: kernel-internal head mapping (no KV expansion)
  P7. DQ dynamic BLOCK_N (matches DKDV selection)
  P8. Backward soft-cap: cached tanh(qk/cap) for ds chain
  P9. Split-KV forward path for small-sequence grid utilization
  P10. Dynamic BLOCK_M: 8 for Sq<=128, default for longer
  L2+ Template Specialization: AttentionConfig + flex_attention API
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import math

try:
    import triton.language.extra.ascend.libdevice as libdevice
except ImportError:
    import triton.language.extra.cann.libdevice as libdevice

# ================ Forward Kernel (optimized) ================

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
    USE_SOFTCAP: tl.constexpr,
    SOFTCAP_VAL: tl.constexpr,
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

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    RCP_LN2: tl.constexpr = 1.44269504
    LN2: tl.constexpr = 0.6931471805599453

    if IS_CAUSAL:
        if SLIDING_WINDOW > 0:
            start_n = tl.maximum(pid_m * BLOCK_M - SLIDING_WINDOW, 0)
        else:
            start_n = 0
    elif SLIDING_WINDOW > 0:
        start_n = tl.maximum(pid_m * BLOCK_M - SLIDING_WINDOW, 0)
    else:
        start_n = 0

    # [P0] Causal diagonal-aware upper bound: query block pid_m only attends to
    # KV positions 0..(pid_m+1)*BLOCK_M-1, so skip fully-masked blocks above diagonal.
    if IS_CAUSAL:
        kv_end = tl.minimum((pid_m + 1) * BLOCK_M, KV_LEN)
    else:
        kv_end = KV_LEN

    offs_n_base = tl.arange(0, BLOCK_N)

    # [P4] Causal masking split: KV blocks strictly below the diagonal need no
    # `tl.where` mask (condition is always true). Split into an unmasked prefix
    # loop + a masked diagonal-suffix loop to cut Vector select ops. Numerically
    # identical since where(True, qk, -inf) == qk. Only for pure causal.
    if IS_CAUSAL and SLIDING_WINDOW == 0:
        full_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N
    else:
        full_end = start_n

    # Unmasked prefix (fully below diagonal)
    for sn in range(start_n, full_end, BLOCK_N):
        offs_n = sn + offs_n_base
        n_mask = offs_n[:, None] < KV_LEN

        k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=n_mask, other=0.0)
        v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=n_mask, other=0.0)
        k_t = tl.trans(k)

        qk = tl.dot(q, k_t) * SM_SCALE

        # [L2+] Tanh soft-capping (Gemma2/Grok-1): caps logits magnitude
        if USE_SOFTCAP:
            qk = tl.tanh(qk / SOFTCAP_VAL) * SOFTCAP_VAL

        if USE_ALIBI:
            qk = qk + tl.load(ALIBI_SLOPE + off_hq) * (offs_m[:, None] - offs_n[None, :])

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
        p = tl.math.exp2((qk - m_ij[:, None]) * RCP_LN2)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    # Masked suffix (diagonal block for causal; all blocks for non-causal/sliding)
    for sn in range(full_end, kv_end, BLOCK_N):
        offs_n = sn + offs_n_base
        n_mask = offs_n[:, None] < KV_LEN

        k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=n_mask, other=0.0)
        v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=n_mask, other=0.0)
        k_t = tl.trans(k)

        qk = tl.dot(q, k_t) * SM_SCALE

        # [L2+] Tanh soft-capping (Gemma2/Grok-1): caps logits magnitude
        if USE_SOFTCAP:
            qk = tl.tanh(qk / SOFTCAP_VAL) * SOFTCAP_VAL

        if USE_ALIBI:
            qk = qk + tl.load(ALIBI_SLOPE + off_hq) * (offs_m[:, None] - offs_n[None, :])

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where((offs_m[:, None] - offs_n[None, :]) <= SLIDING_WINDOW, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
        p = tl.math.exp2((qk - m_ij[:, None]) * RCP_LN2)

        l_i = l_i * alpha + tl.sum(p, 1)
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

    lse = m_i + tl.math.log2(l_i) * LN2
    tl.store(
        LSE + lse_base + offs_m * stride_lse_m,
        lse,
        mask=offs_m < Q_LEN,
    )


# ================ Forward Kernel: Split-KV for small sequences ================

@triton.jit
def flash_attn_fwd_split_kv_kernel(
    Q, K, V, O, LSE, L_PARTIAL, M_PARTIAL,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_m,
    stride_lp_b, stride_lp_h, stride_lp_m, stride_lp_s,
    stride_mp_b, stride_mp_h, stride_mp_m, stride_mp_s,
    Q_LEN, KV_LEN,
    SM_SCALE,
    GQA_GROUPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    KV_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Combine split_id and off_b into grid dim 1 to stay within 3 dims
    pid = tl.program_id(0)
    combined = tl.program_id(1)
    off_hq = tl.program_id(2)

    num_q_blocks = tl.cdiv(Q_LEN, BLOCK_M)
    pid_m = pid % num_q_blocks
    split_id = pid // num_q_blocks

    off_b = combined % 1  # B=1 for split-kv path
    off_hkv = off_hq // GQA_GROUPS

    q_base = off_b * stride_qb + off_hq * stride_qh
    k_base = off_b * stride_kb + off_hkv * stride_kh
    v_base = off_b * stride_vb + off_hkv * stride_vh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < Q_LEN, other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    RCP_LN2: tl.constexpr = 1.44269504
    LN2: tl.constexpr = 0.6931471805599453

    kv_start = split_id * KV_SPLIT
    kv_end = tl.minimum(kv_start + KV_SPLIT, KV_LEN)

    for sn in range(kv_start, kv_end, BLOCK_N):
        offs_n = sn + tl.arange(0, BLOCK_N)
        n_mask = offs_n[:, None] < kv_end

        k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                     mask=n_mask, other=0.0)
        v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                     mask=n_mask, other=0.0)
        k_t = tl.trans(k)
        qk = tl.dot(q, k_t) * SM_SCALE

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2((m_i - m_ij) * RCP_LN2)
        p = tl.math.exp2((qk - m_ij[:, None]) * RCP_LN2)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    num_q_blocks = (Q_LEN + BLOCK_M - 1) // BLOCK_M
    idx = off_hq * num_q_blocks * 1 + pid_m * 1 + split_id
    tl.store(L_PARTIAL + idx * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :] + tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM, acc)
    tl.store(M_PARTIAL + idx + tl.arange(0, BLOCK_M), m_i + tl.math.log2(l_i) * LN2)


@triton.jit
def flash_attn_fwd_reduce_kernel(
    L_PARTIAL, M_PARTIAL, O, LSE,
    stride_lp_b, stride_lp_h, stride_lp_m, stride_lp_s,
    stride_mp_b, stride_mp_h, stride_mp_m, stride_mp_s,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_m,
    Q_LEN, KV_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_hq = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, KV_SPLIT)

    bh_offset = off_b * 1 + off_hq
    num_q_blocks = (Q_LEN + BLOCK_M - 1) // BLOCK_M
    mp_base = bh_offset * num_q_blocks * KV_SPLIT + pid_m * KV_SPLIT
    lp_base = mp_base

    m_partial = tl.load(M_PARTIAL + mp_base + offs_s)
    l_partial = tl.load(L_PARTIAL + lp_base + offs_s[:, None] * HEAD_DIM + offs_d[None, :])

    m_final = tl.max(m_partial)
    alpha = tl.math.exp2(m_partial - m_final)
    l_final = tl.sum(alpha)
    acc = tl.sum(l_partial * alpha[:, None], 0)

    acc = acc / l_final
    tl.store(O + off_b * stride_ob + off_hq * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)

    lse = m_final + tl.math.log2(l_final) * 0.6931471805599453
    tl.store(LSE + off_b * stride_lse_b + off_hq * stride_lse_h + offs_m * stride_lse_m,
             lse, mask=offs_m < Q_LEN)


# ================ Backward: DQ Kernel (optimized) ================

@triton.jit
def flash_attn_bwd_dq_kernel(
    Q, K, V, LSE, DELTA, DO, DQ,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    Q_LEN, KV_LEN,
    SM_SCALE,
    GQA_GROUPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    SOFTCAP_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_hq = tl.program_id(2)

    # [P6] GQA: map Q head to KV head index
    off_hkv = off_hq // GQA_GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    RCP_LN2: tl.constexpr = 1.44269504

    q_base = off_b * stride_qb + off_hq * stride_qh
    q = tl.load(Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < Q_LEN, other=0.0)
    lse = tl.load(LSE + off_b * stride_lse_b + off_hq * stride_lse_h + offs_m * stride_lse_m,
                  mask=offs_m < Q_LEN, other=0.0)
    do = tl.load(DO + off_b * stride_dob + off_hq * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                 mask=offs_m[:, None] < Q_LEN, other=0.0)
    # [FIX] Global Delta = rowsum(O * dO), precomputed on host
    delta = tl.load(DELTA + off_b * stride_lse_b + off_hq * stride_lse_h + offs_m * stride_lse_m,
                    mask=offs_m < Q_LEN, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    k_base = off_b * stride_kb + off_hkv * stride_kh
    v_base = off_b * stride_vb + off_hkv * stride_vh

    # [P0] Causal diagonal-aware bounds
    if IS_CAUSAL:
        start_n = tl.maximum(pid_m * BLOCK_M - SLIDING_WINDOW, 0) if SLIDING_WINDOW > 0 else 0
        kv_end = tl.minimum((pid_m + 1) * BLOCK_M, KV_LEN)
    elif SLIDING_WINDOW > 0:
        start_n = tl.maximum(pid_m * BLOCK_M - SLIDING_WINDOW, 0)
        kv_end = KV_LEN
    else:
        start_n = 0
        kv_end = KV_LEN

    # [P4] NOTE: the causal masking-split (two-loop) optimization used in the
    # forward and DKDV kernels does NOT work here — the triton-ascend compiler
    # miscompiles the DQ kernel's two-loop form (produces catastrophically wrong
    # dQ, verified across load-hoisted and load-inside variants). DQ therefore
    # keeps the single P0 loop with per-block masking.
    for sn in range(start_n, kv_end, BLOCK_N):
        offs_n = sn + tl.arange(0, BLOCK_N)
        n_mask = offs_n[:, None] < KV_LEN

        k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                     mask=n_mask, other=0.0)
        v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                     mask=n_mask, other=0.0)
        kt = tl.trans(k)

        qk = tl.dot(q, kt) * SM_SCALE

        # [P6] GQA: K/V use original strides, map head via off_hq // GQA_GROUPS
        # [L2+] Apply soft-cap for p recomputation (must match forward)
        # [P8] Cache tanh(qk/cap) to avoid duplicate computation in ds chain
        if USE_SOFTCAP:
            tanh_val = tl.tanh(qk / SOFTCAP_VAL)
            qk_capped = tanh_val * SOFTCAP_VAL
        else:
            qk_capped = qk

        if IS_CAUSAL:
            qk_capped = tl.where(offs_m[:, None] >= offs_n[None, :], qk_capped, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk_capped = tl.where((offs_m[:, None] - offs_n[None, :]) <= SLIDING_WINDOW, qk_capped, float("-inf"))

        p = tl.math.exp2((qk_capped - lse[:, None]) * RCP_LN2)

        dp = tl.dot(do.to(v.dtype), tl.trans(v))

        # [FIX] ds = p * (dp - Delta) * scale, using global Delta
        dp_f32 = dp.to(tl.float32)
        p_f32 = p.to(tl.float32)
        ds = (p_f32 * (dp_f32 - delta[:, None])) * SM_SCALE

        # [P8] Chain through tanh soft-cap: reuse cached tanh_val
        if USE_SOFTCAP:
            ds = ds * (1.0 - tanh_val * tanh_val)

        dq = tl.dot(ds.to(q.dtype), tl.trans(kt), dq)

    tl.store(DQ + off_b * stride_dqb + off_hq * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
             dq.to(DQ.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)


# ================ Backward: DK/DV Kernel (optimized) ================

@triton.jit
def flash_attn_bwd_dkdv_kernel(
    Q, K, V, LSE, DELTA, DO, DK, DV,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lse_b, stride_lse_h, stride_lse_m,
    Q_LEN, KV_LEN,
    SM_SCALE,
    GQA_GROUPS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    SOFTCAP_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_b = tl.program_id(1)
    off_hq = tl.program_id(2)

    # [P6] GQA: map Q head to KV head index
    off_hkv = off_hq // GQA_GROUPS

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    RCP_LN2: tl.constexpr = 1.44269504

    k_base = off_b * stride_kb + off_hkv * stride_kh
    v_base = off_b * stride_vb + off_hkv * stride_vh
    q_base = off_b * stride_qb + off_hq * stride_qh
    do_base = off_b * stride_dob + off_hq * stride_doh
    lse_base = off_b * stride_lse_b + off_hq * stride_lse_h

    n_mask = offs_n[:, None] < KV_LEN
    k = tl.load(K + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_mask, other=0.0)
    v = tl.load(V + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=n_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # [P1] Loop-invariant transpose hoisting: k, v don't change across Q loop
    kt = tl.trans(k)
    vt = tl.trans(v)

    # [P0] Causal diagonal-aware bounds: KV block pid_n only receives gradient
    # from Q positions m >= pid_n*BLOCK_N, so start Q loop from the diagonal block.
    if IS_CAUSAL:
        q_start = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        q_start = 0

    # [P4] Causal masking split: Q blocks fully above the diagonal (offs_m >
    # max offs_n) need no mask. Masked prefix covers the diagonal band; unmasked
    # suffix covers the rest. Numerically identical, cuts Vector select ops.
    if IS_CAUSAL and SLIDING_WINDOW == 0:
        diag_end = ((pid_n + 1) * BLOCK_N + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        mask_end = tl.minimum(diag_end, Q_LEN)
    else:
        mask_end = Q_LEN

    # Masked prefix (diagonal band for causal; all for sliding)
    for sm in range(q_start, mask_end, BLOCK_M):
        offs_m = sm + tl.arange(0, BLOCK_M)
        m_mask = offs_m[:, None] < Q_LEN

        q = tl.load(Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                     mask=m_mask, other=0.0)
        lse = tl.load(LSE + lse_base + offs_m * stride_lse_m,
                      mask=offs_m < Q_LEN, other=0.0)
        do = tl.load(DO + do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                     mask=m_mask, other=0.0)
        delta = tl.load(DELTA + lse_base + offs_m * stride_lse_m,
                        mask=offs_m < Q_LEN, other=0.0)

        qk = tl.dot(q, kt) * SM_SCALE

        # [P8] Cache tanh(qk/cap) to avoid duplicate computation
        if USE_SOFTCAP:
            tanh_val = tl.tanh(qk / SOFTCAP_VAL)
            qk_capped = tanh_val * SOFTCAP_VAL
        else:
            qk_capped = qk

        if IS_CAUSAL:
            qk_capped = tl.where(offs_m[:, None] >= offs_n[None, :], qk_capped, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk_capped = tl.where((offs_m[:, None] - offs_n[None, :]) <= SLIDING_WINDOW, qk_capped, float("-inf"))

        p = tl.math.exp2((qk_capped - lse[:, None]) * RCP_LN2)

        do_v = do.to(v.dtype)
        dp = tl.dot(do_v, vt)
        dv = tl.dot(tl.trans(p.to(v.dtype)), do_v, dv)

        dp_f32 = dp.to(tl.float32)
        p_f32 = p.to(tl.float32)
        ds = (p_f32 * (dp_f32 - delta[:, None])) * SM_SCALE

        # [P8] Chain through tanh soft-cap: reuse cached tanh_val
        if USE_SOFTCAP:
            ds = ds * (1.0 - tanh_val * tanh_val)

        dk = dk + tl.dot(tl.trans(ds.to(k.dtype)), q.to(k.dtype))

    # Unmasked suffix (fully above diagonal, causal only)
    for sm in range(mask_end, Q_LEN, BLOCK_M):
        offs_m = sm + tl.arange(0, BLOCK_M)
        m_mask = offs_m[:, None] < Q_LEN

        q = tl.load(Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                     mask=m_mask, other=0.0)
        lse = tl.load(LSE + lse_base + offs_m * stride_lse_m,
                      mask=offs_m < Q_LEN, other=0.0)
        do = tl.load(DO + do_base + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                     mask=m_mask, other=0.0)
        delta = tl.load(DELTA + lse_base + offs_m * stride_lse_m,
                        mask=offs_m < Q_LEN, other=0.0)

        qk = tl.dot(q, kt) * SM_SCALE

        # [P8] Cache tanh(qk/cap) to avoid duplicate computation
        if USE_SOFTCAP:
            tanh_val = tl.tanh(qk / SOFTCAP_VAL)
            qk_capped = tanh_val * SOFTCAP_VAL
        else:
            qk_capped = qk

        p = tl.math.exp2((qk_capped - lse[:, None]) * RCP_LN2)

        do_v = do.to(v.dtype)
        dp = tl.dot(do_v, vt)
        dv = tl.dot(tl.trans(p.to(v.dtype)), do_v, dv)

        dp_f32 = dp.to(tl.float32)
        p_f32 = p.to(tl.float32)
        ds = (p_f32 * (dp_f32 - delta[:, None])) * SM_SCALE

        # [P8] Chain through tanh soft-cap: reuse cached tanh_val
        if USE_SOFTCAP:
            ds = ds * (1.0 - tanh_val * tanh_val)

        dk = dk + tl.dot(tl.trans(ds.to(k.dtype)), q.to(k.dtype))

    tl.store(DK + off_b * stride_dkb + off_hq * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
             dk.to(DK.dtype.element_ty), mask=n_mask)
    tl.store(DV + off_b * stride_dvb + off_hq * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
             dv.to(DV.dtype.element_ty), mask=n_mask)


# ================ Python API ================

def _cann_fusion_attention_forward(q, k, v, causal=False, scale=None, return_lse=False):
    """Fast path: use CANN native npu_fusion_attention for standard attention."""
    B, Hq, S, D = q.shape
    _, Hkv, Sk, _ = k.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    head_num = Hq
    input_layout = "BSND"

    if Hq != Hkv:
        if Hq % Hkv != 0:
            return None
        k = k.repeat_interleave(Hq // Hkv, dim=1)
        v = v.repeat_interleave(Hq // Hkv, dim=1)

    if causal:
        sparse_mode = 3
    else:
        sparse_mode = 0

    try:
        out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
            q, k, v, head_num, input_layout,
            scale=scale,
            pre_tockens=S,
            next_tockens=S,
            sparse_mode=sparse_mode,
            gen_mask_parallel=False,
            sync=False,
        )
        if return_lse:
            sm = softmax_max.squeeze(-1)
            ss = softmax_sum.squeeze(-1)
            lse = sm + torch.log(ss + 1e-30)
            return out, lse
        return out
    except Exception:
        return None


def npu_flash_attention_forward(
    query, key, value,
    causal=False,
    sliding_window=0,
    alibi_slope=None,
    scale=None,
    block_m=16,
    block_n=64,
    return_lse=False,
    soft_cap=0.0,
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

    # [P10] Dynamic BLOCK_M disabled: BLOCK_M=8 causes ADDR_MISALIGN on AICORE
    # Fixed at block_m parameter default (16)

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
        Sq, Sk, scale,
        GQA_GROUPS,
        IS_CAUSAL=causal,
        SLIDING_WINDOW=sliding_window,
        USE_ALIBI=alibi_slope is not None,
        ALIBI_SLOPE=alibi_slope if alibi_slope is not None else query,
        USE_SOFTCAP=soft_cap > 0.0,
        SOFTCAP_VAL=soft_cap if soft_cap > 0.0 else 1.0,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=D,
        multibuffer=True,
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
    block_m=16,
    block_n=32,
    soft_cap=0.0,
):
    B, Hq, Sq, D = query.shape
    Bk, Hkv, Sk, Dk = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    DQ = torch.empty_like(query)
    # [P6] DK/DV allocated per-Q-head (GQA groups summed later), same as before
    # but KV tensors are NOT expanded — kernels do head mapping internally
    DK = torch.empty(B, Hq, Sk, D, device=key.device, dtype=key.dtype)
    DV = torch.empty(B, Hq, Sk, D, device=value.device, dtype=value.dtype)

    # [FIX] Global Delta = rowsum(O * dO), precomputed once (matches reference
    # softmax-Jacobian identity). Replaces per-block local row_sum which was
    # only correct when a single KV block covered the full row.
    assert out is not None, "backward requires forward output `out` to compute Delta"
    delta = (out.to(torch.float32) * grad_out.to(torch.float32)).sum(-1).contiguous()

    GQA_GROUPS = Hq // Hkv

    use_softcap = soft_cap > 0.0
    softcap_val = soft_cap if soft_cap > 0.0 else 1.0

    # [P7] DQ dynamic BLOCK_N: DQ is Cube-bound, use same dynamic selection as DKDV
    dq_block_n = 32 if Sk <= 512 else 64

    grid_dq = (triton.cdiv(Sq, block_m), B, Hq)
    flash_attn_bwd_dq_kernel[grid_dq](
        query, key, value, lse, delta, grad_out, DQ,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        DQ.stride(0), DQ.stride(1), DQ.stride(2), DQ.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        Sq, Sk, scale,
        GQA_GROUPS,
        IS_CAUSAL=causal,
        SLIDING_WINDOW=sliding_window,
        USE_SOFTCAP=use_softcap,
        SOFTCAP_VAL=softcap_val,
        BLOCK_M=block_m, BLOCK_N=dq_block_n, HEAD_DIM=D,
        multibuffer=True,  # [P5] double-buffer to hide load latency
    )

    grid_dkdv = (triton.cdiv(Sk, block_n), B, Hq)
    flash_attn_bwd_dkdv_kernel[grid_dkdv](
        query, key, value, lse, delta, grad_out, DK, DV,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        DK.stride(0), DK.stride(1), DK.stride(2), DK.stride(3),
        DV.stride(0), DV.stride(1), DV.stride(2), DV.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        Sq, Sk, scale,
        GQA_GROUPS,
        IS_CAUSAL=causal,
        SLIDING_WINDOW=sliding_window,
        USE_SOFTCAP=use_softcap,
        SOFTCAP_VAL=softcap_val,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=D,
        multibuffer=True,  # [P5] double-buffer to hide load latency
    )

    # [P6] GQA: sum DK/DV across groups to get per-KV-head gradients
    if GQA_GROUPS > 1:
        DK = DK.view(B, Hkv, GQA_GROUPS, Sk, D).sum(dim=2)
        DV = DV.view(B, Hkv, GQA_GROUPS, Sk, D).sum(dim=2)

    return DQ, DK, DV


class NPUFlexAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, causal=False, sliding_window=0,
                alibi_slope=None, scale=None, block_m=16, block_n=64, soft_cap=0.0):
        ctx.causal = causal
        ctx.sliding_window = sliding_window
        ctx.alibi_slope = alibi_slope
        ctx.scale = scale
        ctx.block_m = block_m
        ctx.block_n = block_n
        ctx.soft_cap = soft_cap

        with torch.no_grad():
            out, lse = npu_flash_attention_forward(
                query, key, value,
                causal=causal, sliding_window=sliding_window,
                alibi_slope=alibi_slope, scale=scale,
                block_m=block_m, block_n=block_n, return_lse=True,
                soft_cap=soft_cap,
            )
        ctx.save_for_backward(query, key, value, lse, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, lse, out = ctx.saved_tensors
        # [P2] Dynamic backward BLOCK_N: empirically BLOCK_N=32 wins for short
        # sequences (S<=512) while BLOCK_N=64 wins for longer ones (crossover
        # ~S=1024). Selecting by KV length avoids regressing the common case.
        # [P7] DQ uses same dynamic BLOCK_N (selected inside npu_flash_attention_backward).
        Sk = key.shape[2]
        bwd_block_n = 32 if Sk <= 512 else 64
        DQ, DK, DV = npu_flash_attention_backward(
            query, key, value, out, lse, grad_out,
            causal=ctx.causal, sliding_window=ctx.sliding_window,
            alibi_slope=ctx.alibi_slope, scale=ctx.scale,
            block_m=ctx.block_m, block_n=bwd_block_n,
            soft_cap=ctx.soft_cap,
        )
        return DQ, DK, DV, None, None, None, None, None, None, None


def npu_flex_attention(
    query, key, value,
    causal=False, sliding_window=0,
    alibi_slope=None, scale=None,
    block_m=16, block_n=64,
    soft_cap=0.0,
):
    return NPUFlexAttention.apply(
        query, key, value, causal, sliding_window,
        alibi_slope, scale, block_m, block_n, soft_cap,
    )


# ================ L2+ Template Specialization Framework ================
#
# Unlike PyTorch FlexAttention (L3: arbitrary user score_mod/mask_mod via Inductor
# codegen — blocked by triton-ascend compiler), this framework provides a
# COMPOSABLE PREDEFINED PRIMITIVE library. Users compose patterns via config;
# the dispatch layer maps config to Triton constexpr flags, producing a
# specialized compiled kernel per unique combination (Triton auto-caches).
#
# Currently supported patterns:
#   - Full attention (no mask)
#   - Causal mask
#   - Sliding window (+ causal)
#   - ALiBi bias
#   - GQA (arbitrary ratio)
#   - Custom scale
#   - Tanh soft-capping (Gemma2/Grok-1)  [NEW]
#
# All masking uses tl.where (not scf.if); all loop bounds are kernel parameters
# (not tl.load values) — fully CANN/triton-ascend compatible.

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AttentionConfig:
    """Composable attention pattern configuration for the L2+ framework.

    Example — Gemma2-style attention:
        config = AttentionConfig(causal=True, sliding_window=1024, soft_cap=50.0)
        out = flex_attention(q, k, v, config=config)

    Example — ALiBi + causal:
        config = AttentionConfig(causal=True, alibi_slope=slopes)
        out = flex_attention(q, k, v, config=config)
    """
    causal: bool = False
    sliding_window: int = 0
    alibi_slope: Optional[torch.Tensor] = None
    scale: Optional[float] = None
    soft_cap: float = 0.0
    block_m: int = 16
    block_n: int = 64

    def __post_init__(self):
        if self.sliding_window < 0:
            raise ValueError("sliding_window must be >= 0")
        if self.soft_cap < 0:
            raise ValueError("soft_cap must be >= 0")


def flex_attention(query, key, value, config: Optional[AttentionConfig] = None, **kwargs):
    """NPU FlexAttention (L2+ template specialization framework).

    Dispatches to specialized Triton-Ascend kernels based on config. Each unique
    combination of patterns maps to constexpr flags, producing a compiled kernel
    that Triton auto-caches (no recompilation for same config).

    Args:
        query, key, value: [B, H, S, D] tensors (fp16/fp32/bf16)
        config: AttentionConfig specifying attention patterns to compose.
                 If None, uses kwargs to build a default config.
        **kwargs: Direct pattern arguments (causal=, sliding_window=, alibi_slope=,
                  soft_cap=, scale=, block_m=, block_n=). Ignored if config is given.

    Returns:
        Output tensor [B, H, S, D] (same dtype as input).

    Examples:
        # Causal only
        out = flex_attention(q, k, v, causal=True)

        # Gemma2: causal + sliding window + soft-capping
        out = flex_attention(q, k, v, causal=True, sliding_window=1024, soft_cap=50.0)

        # Via config object
        cfg = AttentionConfig(causal=True, sliding_window=512, soft_cap=20.0)
        out = flex_attention(q, k, v, config=cfg)
    """
    if config is None:
        config = AttentionConfig(**kwargs)
    return npu_flex_attention(
        query, key, value,
        causal=config.causal,
        sliding_window=config.sliding_window,
        alibi_slope=config.alibi_slope,
        scale=config.scale,
        block_m=config.block_m,
        block_n=config.block_n,
        soft_cap=config.soft_cap,
    )
