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
  1. 在 Python 层将 K/V expand 到 Hq 头 (每个 KV 头复制 GQA_GROUPS 次)
  2. 对每个 query 头独立计算 dK/dV
  3. 将 dK/dV 沿 GQA_GROUPS 维度求和 (sum, 非 mean)
     因为 K/V 被多个 query head 共享, 梯度应累加
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
| 动态反向 BLOCK_N | 反向 | `32 if Sk≤512 else 64` | 大 S 反向 ~9% |

**综合结果**: 前向 1.47x-4.34x、反向 1.20x-2.52x (vs 优化前 v2)，前向 vs SDPA 由 0.12-0.31x 提升至
0.54-0.66x，S=128 前向超过 SDPA。

**尝试后放弃的优化** (遵循实测驱动原则):
- 前向 BLOCK_M=32：数据推翻理论，主力配置退化最多 24% → 保持 16/64
- i32→float 比较 (避免标量降级)：`.to(float32)` 转换开销 > 收益，大序列回退 36-50% → 弃用
- DQ kernel P4 分裂：编译器误编译 → 保持单循环
- CANN 原生快速路径：会绕过 Triton 前向实现，改变项目本质 → 决定保持纯 Triton

详细优化历程、实测数据与决策依据见 [OPTIMIZATION.md](OPTIMIZATION.md)。

## 3. 支持的特性

| 特性 | 前向 | 反向 | 备注 |
|------|------|------|------|
| Full Attention (无 mask) | ✅ | ✅ | |
| Causal Mask | ✅ | ✅ | |
| Sliding Window | ✅ | ✅ | 与 causal 组合使用 |
| GQA (任意 ratio) | ✅ | ✅ | 通过 Python 层 expand + sum |
| 自定义 Scale | ✅ | ✅ | |
| LSE 输出 | ✅ | — | 用于反向传播 |
| fp32 | ✅ | ✅ | |
| fp16 | ✅ | ✅ | |
| bf16 | ✅ | ✅ | |
| 任意 head_dim | ✅ | ✅ | D=32/64/128 已验证 |

## 4. 已知限制

1. **不支持任意 score_mod/mask_mod**：仅支持预定义的 causal、sliding window、alibi 模式，不支持 PyTorch FlexAttention 的任意用户函数追踪
2. **大序列性能为 SDPA 的 0.54-0.66x**：原生 SDPA 使用 CANN 硬件级优化内核 (软件流水、TMA 等)，本实现受 triton-ascend 编译器限制无法企及
3. **DQ 反向 kernel 无法应用 P4 mask 分裂**：triton-ascend 编译器对 DQ 的双循环形式误编译 (产生错误 dQ)，前向/DKDV 同结构正常；DQ 保持单循环
4. **不支持 block-sparse BlockMask**：不支持 PyTorch FlexAttention 的块稀疏索引格式
5. **KV 长度需为 BLOCK_N 整数倍**：非整除时 padding 位屏蔽为已知边界待完善项 (当前测试序列均满足)

> **已修复的历史问题**:
> - dQ/dK ~10-15% 误差 → 全局 Delta 修复后达 fp32 ~1e-7 / fp16 ~2e-4 (§2.3)
> - autograd.Function 路径此前 backward 未传 `out` (梯度不完整) → 已修正为保存并传递 `out`
> - 反向 block size 此前硬编码 BLOCK_N=32 → 改为按 Sk 动态 (§2.5)

## 5. 适用场景

- 需要自定义掩码组合 (causal + sliding window) 的注意力计算
- 需要 LSE 输出用于自定义反向传播或 KV cache 场景
- 需要 bf16/fp16 混合精度训练的前向 + 反向
- GQA (任意 ratio) 注意力计算，包括 Hq=32 Hkv=8 等大规模配置
- 研究 Triton-Ascend 在 NPU 上的 kernel 开发
- 原生 SDPA 不支持的灵活掩码组合场景

## 6. 不适用场景

- 需要任意 score_mod (如 ALiBi、相对位置编码等动态修改) — 需等待 triton-ascend 修复编译器限制
- 需要极致性能的大规模推理 — 建议使用 `npu_fusion_attention` 原生算子
- 需要 block-sparse 掩码 (如文档级掩码、自定义稀疏模式) — 当前不支持 BlockMask
- 需要与 `torch.compile` 无缝集成 — 直接调用方式不支持 autograd tracing

## 7. triton-ascend 版本兼容性

| triton-ascend | CANN | FlexAttention Inductor 路径 | 独立 kernel 方案 (本项目) |
|---------------|------|------------------------------|-------------------------|
| 3.2.0rc4 | 8.3.RC1 | ❌ vcast/vexp 失败 | ✅ 可用 |
| 3.2.0 | 8.5.0 | ❌ 同上 | ✅ 可用 |
| 3.2.1 | 9.0.0 | ❌ vcast/vexp 仍未修复 ([Issue #448](https://github.com/triton-lang/triton-ascend/issues/448)) | ✅ 可用 |
| 未来 (Triton 3.5) | TBD | ⚠️ 取决于 BiShengIR 动态循环支持 | ✅ 可用 |

> **结论**：截至 2026 年 7 月，升级 triton-ascend 版本不能解决 FlexAttention 的核心限制。独立 Triton kernel 方案在可预见的未来仍是唯一可行路径。
