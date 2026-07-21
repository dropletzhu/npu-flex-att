# NPU Flash Attention 设计文档

## 1. 背景与动机

### 1.1 FlexAttention 架构

PyTorch FlexAttention 是一种编译器驱动的注意力实现，通过 Jinja 模板 + Triton JIT 编译生成高效内核。其架构为：

```
Python API → HOP (设备无关) → Inductor Lowering (设备分派) → Triton Jinja 模板 → Triton JIT → 设备二进制
```

在 CUDA 上，Inductor 生成 Triton 内核并通过 NVIDIA Triton 编译为 GPU 二进制。该实现支持任意 `score_mod`（注意力分数修改函数）和 `mask_mod`（掩码函数），通过 `make_fx` 追踪用户函数并内联到 Triton 模板中。

### 1.2 NPU 上的挑战

将 FlexAttention 移植到 Ascend NPU 面临以下关键挑战：

1. **bishengir-compile 限制**：triton-ascend 3.2.0rc4 的 bishengir 编译器无法处理 Inductor 生成的复杂内存访问模式（`memref.reinterpret_cast` + `memref.copy` + `arith.truncf` 组合在循环内），表现为 `vcast`/`vexp` "Unsupported op for finding the root alloc" 错误。

2. **运行时加载循环上界**：FlexAttention 使用 `tl.load(kv_num_blocks)` 获取块稀疏索引的数量作为循环上界，bishengir-compiler 无法追踪此模式的内存分配链。

3. **scf.if 在 scf.for 内**：循环内的条件分支（如 `if start_n < block_n_end`）会导致 bishengir-compile 失败。

4. **torch_npu Inductor 兼容性**：torch_npu 的 Inductor 后端会强制 fallback 未在白名单中的 HOP，覆盖 FlexAttention 的专用 Triton 模板 lowering。

### 1.3 解决方案

采用**独立 Triton kernel** 方案：绕过 Inductor 的复杂代码生成，直接编写使用简单指针运算的 Triton kernel。该方案已被验证在 NPU 上可行——最小 Flash Attention kernel 使用 `tl.load(ptr + offsets)` 直接指针运算，在 bishengir-compile 上完全通过。

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户 API                              │
│  npu_flex_attention(q, k, v, causal=, sliding_window=) │
│  npu_flash_attention_forward(q, k, v, ...) → (O, LSE)  │
│  npu_flash_attention_backward(q, k, v, ...) → (DQ,DK,DV)│
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Autograd Function 层                        │
│  NPUFlexAttention (torch.autograd.Function)             │
│  forward: 调用 forward kernel, 保存 q/k/v/lse           │
│  backward: 调用 DQ kernel + DK/DV kernel                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Triton Kernel 层                            │
│                                                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ flash_attn_fwd  │  │ bwd_dq_kernel│  │bwd_dkdv_kernel│
│  │  _kernel        │  │              │  │              │ │
│  │                 │  │ 输入: Q,K,V, │  │ 输入: Q,K,V, │ │
│  │ 输入: Q,K,V     │  │ LSE,dO       │  │ LSE,dO       │ │
│  │ 输出: O, LSE    │  │ 输出: dQ      │  │ 输出: dK, dV │ │
│  └────────┬────────┘  └──────┬───────┘  └──────┬───────┘ │
│           │                  │                  │         │
│           ▼                  ▼                  ▼         │
│  ┌──────────────────────────────────────────────────────┐│
│  │          bishengir-compile (NPU 编译器)               ││
│  │   Triton IR → Linalg IR → HIVM IR → npubin 二进制     ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Ascend 910B3 NPU 硬件                       │
│  48 AI Cores | 192KB UB/core | Cube+Vector units         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 前向 Kernel 设计

#### 算法：在线 Softmax Flash Attention

```
输入: Q [B, H, S, D], K [B, Hkv, S, D], V [B, Hkv, S, D]
输出: O [B, H, S, D], LSE [B, H, S]

Grid: (cdiv(S, BLOCK_M), B, H)
每个 program 处理 BLOCK_M 个 query token

初始化:
  m_i = -inf  (运行最大值)
  l_i = 0     (运行 exp 和)
  acc = 0     (输出累加器)

[P0] causal 上界: kv_end = min((pid_m+1)*BLOCK_M, KV_LEN)  (跳过对角线以上全 masked 块, 省 44-48% 迭代)
[P4] mask 分裂: full_end = (pid_m*BLOCK_M // BLOCK_N)*BLOCK_N
  ├─ 无 mask 前缀 (对角线以下, 免 tl.where):  for n in range(start_n, full_end, BLOCK_N)
  └─ 带 mask 后缀 (对角线块):                  for n in range(full_end, kv_end, BLOCK_N)
每段循环体:
    1. 加载 K[n:n+BLOCK_N, :] 和 V[n:n+BLOCK_N, :]  (multibuffer 双缓冲隐藏延迟)
    2. qk = Q @ K^T * scale                          (tl.dot)
    2b. [L2+] if USE_SOFTCAP: qk = tanh(qk/cap) * cap (Gemma2/Grok-1 软截断)
    2c. [if USE_ALIBI] qk += alibi_slope[h] * (q_idx - kv_idx)
    3. [仅后缀] 应用 mask (causal / sliding window)   (tl.where)
    4. m_ij = max(m_i, max(qk))                      (在线 max)
    5. alpha = exp(m_i - m_ij)                       (缩放因子)
    6. p = exp(qk - m_ij)                            (注意力概率)
    7. l_i = l_i * alpha + sum(p)                    (更新归一化因子)
    8. acc = acc * alpha + p @ V                     (更新累加器, tl.dot)
    9. m_i = m_ij                                    (更新最大值)

输出:
  acc /= l_i                                        (归一化)
  LSE = m_i + ln(l_i)                               (logsumexp)
```

> **P4 mask 分裂**在数学上完全等价 (`where(True, x, -inf) == x`, 实测输出 bit 一致)，
> 但对角线以下的块免除了 `tl.where` Vector 选择，前向大序列因此额外获得 ~2x 加速。

#### 关键设计决策

| 决策 | 原因 | 替代方案 |
|------|------|----------|
| 直接指针运算 `tl.load(ptr + offs)` | bishengir-compile 不支持 `memref.reinterpret_cast` | Inductor 的 `load_checked_2d` |
| KV_LEN 作为循环上界 | bishengir-compile 不支持 `tl.load()` 值作为循环上界 | `tl.load(kv_num_blocks)` |
| `tl.where` 替代 `scf.if` | bishengir-compile 不支持循环内 `scf.if` | `if` 条件分支 |
| `tl.math.exp2` + `RCP_LN2` | triton-ascend libdevice 支持 exp2 | `tl.exp` (也可用) |
| BLOCK_M=16, BLOCK_N=64 | 最优 block size (全面扫描确认) | 32x32, 64x64 等 |
| 分离 DQ 和 DK/DV kernel | 避免复杂条件分支 | 单 kernel (会触发 bishengir 错误) |

### 2.3 反向 Kernel 设计

#### 全局 Delta (softmax 雅可比) — 关键正确性设计

softmax 反向的雅可比项需要**整行**的 `rowsum(p ⊙ dp)`。由于 flash attention 分块遍历 KV，
早期实现用**按块局部** `tl.sum(p*dp, 1)` 近似，仅当单个 KV 块覆盖整行时才正确 → dQ/dK 误差 0.15-0.30。

**正确方案**：利用恒等式 `rowsum(p ⊙ dp) = rowsum(O ⊙ dO) ≡ Delta`（O 为前向输出）。
在 Host 侧一次性预计算 `Delta = (out.float() * grad_out.float()).sum(-1)`，传入两个反向 kernel，
用 `ds = p * (dp - Delta[:, None]) * scale`。此举同时保证正确性（全局精确）并省去每块的 `tl.sum`。

#### DQ Kernel

```
Grid: (cdiv(S, BLOCK_M), B, H)
每个 program 计算一个 query 块的 dQ

[P0] causal 上界: kv_end = min((pid_m+1)*BLOCK_M, KV_LEN)  (跳过对角线以上全 masked 块)
[P7] 动态 BLOCK_N: block_n = 32 if Sk≤512 else 64  (DQ/DKDV 均使用)
循环 (遍历 KV 块 0..kv_end):
  for n in range(start_n, kv_end, BLOCK_N):
    1. 加载 K, V
    2. qk = Q @ K^T * scale
    3. 应用 mask (causal / sliding, tl.where)
    4. p = exp2((qk - LSE) * RCP_LN2)     (从存储的 LSE 恢复概率)
    5. dp = dO @ V^T                       (注意力概率的梯度)
    6. ds = p * (dp - Delta) * scale       (全局 Delta, 见上)
    7. dQ += ds @ K                        (Q 的梯度, tl.dot 累加)

注: DQ kernel 无法应用 P4 mask 分裂 —— triton-ascend 编译器对其双循环形式误编译, 保持单循环。
```

#### DK/DV Kernel

```
Grid: (cdiv(S, BLOCK_N), B, H)
每个 program 计算一个 KV 块的 dK 和 dV

[P0] causal 下界: q_start = (pid_n*BLOCK_N // BLOCK_M) * BLOCK_M  (从对角线块开始)
[P1] kt/vt = trans(k)/trans(v) 外提到循环前 (循环不变量)
[P3] do_v = dO.to(v.dtype) 缓存复用
[P6] GQA head 映射: off_hkv = off_hq // GQA_GROUPS (kernel 内部, 无需 expand KV)
[P7] 动态 BLOCK_N: block_n = 32 if Sk≤512 else 64
[P4] mask 分裂: 对角线带 (masked) + 对角线上方 (unmasked) 两段循环
循环 (遍历 Q 块 q_start..Q_LEN):
  for m in range(q_start, Q_LEN, BLOCK_M):
    1. 加载 Q, dO, LSE, Delta
    2. qk = Q @ kt * scale
    3. 应用 mask (仅对角带; 上方段免 tl.where)
    4. p = exp2((qk - LSE) * RCP_LN2)
    5. do_v = dO.to(v.dtype)              [P3] 缓存, dp 与 dv 复用
    6. dp = do_v @ vt
    7. dV += p^T @ do_v                   (V 的梯度)
    8. ds = p * (dp - Delta) * scale      (全局 Delta)
    9. dK += ds^T @ Q                     (K 的梯度)
```

#### GQA 处理

GQA (Grouped Query Attention) 在反向传播中的处理：

```
前向: K/V 有 Hkv 头, Q 有 Hq 头 (Hq = GQA_GROUPS * Hkv)
反向: 
  1. Host 侧: 不再 expand KV tensor (节省 4x 内存, Hq=32 Hkv=8)
  2. DQ kernel: 对每个 query 头独立计算 dQ, 使用 off_hq (无需 head 映射)
  3. DKDV kernel: 使用 off_hkv = off_hq // GQA_GROUPS 做 head 映射
  4. 将 dK/dV 沿 GQA_GROUPS 维度求和 (sum, 非 mean)
     因为 K/V 被多个 query head 共享, 梯度应累加
  5. 输出 dK/dV shape 为 [B, Hkv, S, D] (非 [B, Hq, S, D])
```

### 2.4 LSE (LogSumExp) 处理

前向存储的 LSE 使用 `exp2`/`log2` 优化：

```python
# 前向: 使用 exp2 代替 exp (性能优化)
RCP_LN2 = 1.44269504  # 1/ln(2) = log2(e)
LN2 = 0.6931471805599453  # ln(2)

p = tl.math.exp2((qk - m_ij) * RCP_LN2)  # = exp(qk - m_ij)
lse = m_i + tl.math.log2(l_i) * LN2       # = m_i + ln(l_i) (正确转换)

# 反向: 从 LSE 恢复概率
p = tl.math.exp2((qk - lse) * RCP_LN2)   # = exp(qk - lse) = exp(qk - m_i) / l_i
```

### 2.5 Block Size 选择

通过全面扫描 block size 组合确定最优配置（下表为**优化前基线**的前向延迟，用于**相对**比较选型；
优化后绝对延迟已快约 2x，但相对排序不变）：

| BLOCK_M | BLOCK_N | S=128 | S=512 | S=1024 | S=2048 |
|---------|---------|-------|-------|--------|--------|
| 16 | 16 | 0.72ms | 2.33ms | 4.07ms | 7.87ms |
| 16 | 32 | 0.48ms | 2.24ms | 3.25ms | 6.56ms |
| **16** | **64** | **0.48ms** | **2.22ms** | **3.38ms** | **6.28ms** |
| 32 | 32 | 0.73ms | 2.21ms | 3.62ms | 7.31ms |
| 32 | 64 | 0.73ms | 2.22ms | 3.59ms | 7.20ms |
| 64 | 64 | 0.94ms | 3.04ms | 3.51ms | 7.02ms |

最优: **BLOCK_M=16, BLOCK_N=64**
- BLOCK_M=16: 增加 grid 块数, 提升小序列核心利用率
- BLOCK_N=64: 减少循环次数, 更好的开销摊销

> 复测确认 (含 multibuffer): 16/64 在各序列长度均近最优 (7% 以内)。BLOCK_M=32 会让主力低 batch
> 配置退化最多 24% —— 印证前向为 **Vector-bound 而非 Cube-bound** (增大 BLOCK_M 主帮 Cube)。

**反向**使用 BLOCK_M=16, **BLOCK_N 动态** (`32 if Sk≤512 else 64`, autograd 路径)：
实测短序列 BLOCK_N=32 快 20%、长序列 BLOCK_N=64 快 9%，交叉点约 S=1024。

### 2.5b 编译选项: multibuffer

三个 kernel 的 launch 均传入 `multibuffer=True` 启用**双缓冲**，隐藏 K/V 加载的 MTE2 访存延迟
(前向为访存/Vector 密集型, MTE2-Cube 不重叠)。小序列 (S≤256) latency-bound，收益显著 (~1.45x)，
使 S=128 前向超过 SDPA；大序列已计算饱和，收益中性。

### 2.6 NPU 硬件约束

| 约束 | 值 | 影响 |
|------|------|------|
| AI Core 数量 | 48 | Grid 应接近 48 的倍数 |
| UB (Unified Buffer) | 192KB/core | BLOCK_M×BLOCK_N×dtype 不能超过 UB |
| Cube 单元 | 支持 tl.dot | fp16/fp32 GEMM |
| Vector 单元 | 支持 exp2/load/store | 数学函数和访存 |
| bishengir-compile | 3.2.0rc4 | 不支持 scf.if/memref.reinterpret_cast 在循环内 |

### 2.7 优化演进总览

在基础 Triton kernel (v2) 之上，按 profiling 结论 (Cube ~6%、Vector 80-90%、因果浪费 44-48% 循环)
和 triton-latency-optimizer skill 系统化实施了以下优化：

| 优化 | 作用对象 | 核心思想 | 实测收益 (vs v2) |
|------|---------|---------|-----------------|
| P0 因果块跳过 | 前向+DQ+DKDV | 对角线感知循环上/下界，跳过全 masked 块 | 大 S ~1.5-2x |
| 🔴 全局 Delta 修复 | DQ+DKDV | `Delta=rowsum(O⊙dO)` 替代按块局部 row_sum | **精度 ~10^5x** + 省 Vector |
| P1 转置外提 | DKDV | `kt/vt` 循环不变量外提 | 反向 ~1.1x |
| P3 缓存 do_v | DKDV | 复用 `do.to(v.dtype)` | 反向微增 |
| P4 因果 mask 分裂 | 前向+DKDV | 对角线以下块免 `tl.where` | 前向大 S 额外 ~2x |
| multibuffer | 三 kernel | 双缓冲隐藏 MTE2 访存延迟 | 前向小 S ~1.45x |
| 动态反向 BLOCK_N | DQ+DKDV | `32 if Sk≤512 else 64` | 大 S 反向 ~9-16% |
| P6 GQA KV 消除 | DQ+DKDV | kernel 内部 `off_hkv = off_hq // GQA_GROUPS` head 映射 | dK/dV 内存 4x 节省 |
| P7 DQ 动态 BLOCK_N | DQ | DQ 也用动态 block size | DQ 大序列 ~16% |
| P8 soft-cap tanh 缓存 | DQ+DKDV | 反向 tanh(qk/cap) 只算一次，ds 链复用 | soft-cap 模式 ~10% Vector ops |
| L2+ 模板特化框架 | 全部 | `AttentionConfig` + constexpr dispatch | 新增 soft-cap (Gemma2),可组合原语库 |

**综合结果**: 前向 1.44x-4.37x、反向 1.23x-2.48x (vs 优化前 v2)，前向 vs SDPA 最高 **1.81x (S=128)**，
反向 D=128 最高 **1.18x (S=512)**。

**尝试后放弃的优化** (遵循实测驱动原则):
- Split-KV forward (P9)：死代码修复成本高，`triton.cdiv` JIT 内不可用 + L_PARTIAL/M_PARTIAL 索引错误
- BLOCK_M 动态化 (P10)：`BLOCK_M=8` 触发 AICORE `ADDR_MISALIGN` 硬件错误
- Forward 循环合并 (P11)：合并会消除 P4 mask 分裂的 Vector 节省，反向 S=1024 回退 19%
- 前向 BLOCK_M=32：数据推翻理论，主力配置退化最多 24% → 保持 16/64
- i32→float 比较 (避免标量降级)：`.to(float32)` 转换开销 > 收益，大序列回退 36-50% → 弃用
- DQ kernel P4 分裂：编译器误编译 → 保持单循环
- CANN 原生快速路径：会绕过 Triton 前向实现，改变项目本质 → 决定保持纯 Triton

## 3. 支持的特性

| 特性 | 前向 | 反向 | 备注 |
|------|------|------|------|
| Full Attention (无 mask) | ✅ | ✅ | |
| Causal Mask | ✅ | ✅ | 下三角因果掩码 |
| Sliding Window | ✅ | ✅ | 与 causal 组合使用 |
| GQA (任意 ratio) | ✅ | ✅ | kernel 内部 head 映射,无需 Python 层 expand |
| ALiBi Bias | ✅ | ✅ | 注意力分数线性偏置,per-head slope |
| Tanh Soft-Capping | ✅ | ✅ | Gemma2/Grok-1 风格 logits 软截断 |
| 自定义 Scale | ✅ | ✅ | |
| LSE 输出 | ✅ | — | 用于反向传播 |
| fp32 / fp16 / bf16 | ✅ | ✅ | |
| 任意 head_dim | ✅ | ✅ | D=32/64/128 已验证 |
| **模式任意组合** | ✅ | ✅ | L2+ 框架: 多模式可同时启用 |

## 3.5 L2+ 模板特化框架

### 动机

PyTorch FlexAttention (L3) 通过 Inductor 追踪**任意**用户 `score_mod`/`mask_mod` 函数生成 Triton kernel。这在 Ascend NPU 上被 bishengir-compile 阻断 (§1.2)。

L2+ 框架采用**折中方案**:提供**可组合的预定义原语库**,用户通过 `AttentionConfig` 组合模式,框架映射到 Triton `tl.constexpr` flags,编译期特化生成专用 kernel。每个唯一 flag 组合 = 1 个特化编译 kernel (Triton 自动缓存,相同 config 不重编译)。

### API

```python
from npu_flash_attention import flex_attention, AttentionConfig

cfg = AttentionConfig(causal=True, sliding_window=1024, soft_cap=50.0, alibi_slope=slopes)
out = flex_attention(q, k, v, config=cfg)
# 或: flex_attention(q, k, v, causal=True, sliding_window=1024, soft_cap=50.0)
```

### Dispatch 机制

```
AttentionConfig → host 预处理(alibi_bias/delta) → Triton kernel[constexpr flags]
                                                      ↓
                                        编译期特化(每唯一组合 1 个 kernel)
```

### 原语库

| 模式 | Kernel constexpr flag | 实现机制 | CANN 兼容 |
|------|----------------------|---------|-----------|
| Full attention | (无 flag) | 无掩码 | ✅ |
| Causal | `IS_CAUSAL` | `tl.where` 掩码 | ✅ |
| Sliding window | `SLIDING_WINDOW` | `tl.where` 掩码 | ✅ |
| ALiBi | `USE_ALIBI` | 分数偏置 `score += slope*(q-k)` | ✅ |
| GQA | `GQA_GROUPS` | head 映射 `off_hkv = off_hq // groups` | ✅ |
| Tanh soft-cap | `USE_SOFTCAP` + `SOFTCAP_VAL` | `qk = tanh(qk/cap)*cap` | ✅ |
| Custom scale | `SM_SCALE` | 标量乘法 | ✅ |

### CANN 兼容约束

- 掩码用 `tl.where`(非 `scf.if`) — 规避循环内条件分支限制
- 循环上界用 kernel 参数(非 `tl.load` 值) — 规避块稀疏限制
- 简单指针运算 `tl.load(ptr + offsets)` — 规避 `memref.reinterpret_cast` 限制

### Tanh Soft-Capping 实现 (Gemma2/Grok-1)

**前向**: 在 `qk = Q@K^T * scale` 之后、掩码之前施加:
```
qk_capped = tanh(qk / SOFTCAP_VAL) * SOFTCAP_VAL
```

**反向**: 重计算 soft-cap 用于 `p` 恢复,并在 `ds` 乘 tanh 导数:
```
# 重计算 p (匹配前向)
qk_capped = tanh(qk / cap) * cap
p = exp2((qk_capped - lse) * RCP_LN2)

# softmax 反向 + tanh 链式法则
ds = p * (dp - Delta) * SM_SCALE
ds *= (1 - tanh^2(qk / cap))  # tanh 导数
```

### 可扩展性

添加新模式遵循统一模式:
1. kernel 签名加 `tl.constexpr` flag
2. kernel 体加条件逻辑 (`if FLAG: ...`)
3. host 函数加参数
4. `AttentionConfig` 加字段

**未来候选模式**: Relative Position Encoding、PrefixLM、Document Masking。

## 4. 已知限制

1. **不支持任意 score_mod/mask_mod**：仅支持预定义原语 (causal、sliding window、ALiBi、GQA、tanh soft-capping 等) 的可组合特化，不支持 PyTorch FlexAttention 的任意用户函数追踪
2. **大序列性能为 SDPA 的 0.56-0.70x**：原生 SDPA 使用 CANN 硬件级优化内核 (软件流水、TMA 等)，本实现受 triton-ascend 编译器限制无法企及
3. **GQA 场景 Flex 较慢**：SDPA 有原生 GQA 支持，Flex 在 Python 层 expand KV tensor (Hq=32 Hkv=8 时 4x 复制)
4. **DQ 反向 kernel 无法应用 P4 mask 分裂**：triton-ascend 编译器对 DQ 的双循环形式误编译 (产生错误 dQ)，前向/DKDV 同结构正常；DQ 保持单循环
5. **不支持 block-sparse BlockMask**：不支持 PyTorch FlexAttention 的块稀疏索引格式
6. **KV 长度需为 BLOCK_N 整数倍**：非整除时 padding 位屏蔽为已知边界待完善项 (当前测试序列均满足)

> **已修复的历史问题**:
> - dQ/dK ~10-15% 误差 → 全局 Delta 修复后达 fp32 ~1e-7 / fp16 ~2e-4 (§2.3)
> - autograd.Function 路径此前 backward 未传 `out` (梯度不完整) → 已修正为保存并传递 `out`
> - 反向 block size 此前硬编码 BLOCK_N=32 → 改为按 Sk 动态 (§2.5)

## 5. 适用场景

- 需要自定义掩码组合 (causal + sliding window + ALiBi + soft-cap) 的注意力计算
- 需要 LSE 输出用于自定义反向传播或 KV cache 场景
- 需要 bf16/fp16 混合精度训练的前向 + 反向
- GQA (任意 ratio) 注意力计算，包括 Hq=32 Hkv=8 等大规模配置
- 研究 Triton-Ascend 在 NPU 上的 kernel 开发
- 原生 SDPA 不支持的灵活掩码组合场景
- 小序列前向低延迟场景 (S≤128, Flex 比 SDPA 快 1.81x)

## 6. 不适用场景

- 需要任意 score_mod (如 ALiBi、相对位置编码等动态修改) — 需等待 triton-ascend 修复编译器限制
- 需要极致性能的大规模推理 — 建议使用 `npu_fusion_attention` 原生算子
- 需要 block-sparse 掩码 (如文档级掩码、自定义稀疏模式) — 当前不支持 BlockMask
- 需要与 `torch.compile` 无缝集成 — 直接调用方式不支持 autograd tracing
- GQA 大比例场景 (Hq=32 Hkv=8) — SDPA 有原生 GQA 支持，Flex 需 Python 层 expand 较慢

## 7. triton-ascend 版本兼容性

| triton-ascend | CANN | FlexAttention Inductor 路径 | 独立 kernel 方案 (本项目) |
|---------------|------|------------------------------|-------------------------|
| 3.2.0rc4 | 8.3.RC1 | ❌ vcast/vexp 失败 | ✅ 可用 |
| 3.2.0 | 8.5.0 | ❌ 同上 | ✅ 可用 |
| 3.2.1 | 9.0.0 | ❌ vcast/vexp 仍未修复 ([Issue #448](https://github.com/triton-lang/triton-ascend/issues/448)) | ✅ 可用 |
| 未来 (Triton 3.5) | TBD | ⚠️ 取决于 BiShengIR 动态循环支持 | ✅ 可用 |

> **结论**：截至 2026 年 7 月，升级 triton-ascend 版本不能解决 FlexAttention 的核心限制。独立 Triton kernel 方案在可预见的未来仍是唯一可行路径。
