# NPU Flex Attention

在 Ascend NPU 上基于 Triton-Ascend 实现的 Flash Attention 内核与 **L2+ 模板特化框架**，支持前向传播与反向传播。通过可组合的预定义原语库（causal、sliding window、ALiBi、GQA、tanh soft-capping 等）覆盖主流注意力模式，适用于 LLM 训练与推理。

> **与 PyTorch FlexAttention 的关系**: FlexAttention (L3) 依赖 Inductor 追踪任意用户函数,被 triton-ascend 编译器阻断。本项目采用 **L2+ 模板特化框架**——预定义可组合原语 + constexpr 特化 dispatch,绕过编译器限制,覆盖 90%+ 实际注意力模式需求。详见 [DESIGN.md](DESIGN.md) §2.7。

## 快速开始

### 环境要求

| 组件 | 版本 |
|------|------|
| NPU | Ascend 910B3 (或其他 Ascend A2 系列) |
| CANN | 8.3.RC1+ |
| PyTorch | 2.9.1 |
| torch_npu | 2.9.1 |
| triton-ascend | 3.2.0rc4+ |

### 安装

```bash
# 1. 安装 triton-ascend (兼容 CANN 8.3.RC1)
pip install triton-ascend==3.2.0rc4

# 2. 配置 bishengir-compile 路径
source setup_env.sh

# 3. 验证环境
python test/test_triton_ascend.py
```

### 基本用法

```python
import torch
import torch_npu
from npu_flash_attention import npu_flash_attention_forward, npu_flash_attention_backward

# 前向传播
B, H, S, D = 1, 1, 128, 64
q = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
k = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)
v = torch.randn(B, H, S, D, device="npu", dtype=torch.float16)

out, lse = npu_flash_attention_forward(q, k, v, causal=True, return_lse=True)

# 反向传播
grad_out = torch.randn_like(out)
dQ, dK, dV = npu_flash_attention_backward(q, k, v, out, lse, grad_out, causal=True)
```

### GQA (Grouped Query Attention)

```python
Hq, Hkv = 4, 1  # GQA ratio = 4
q = torch.randn(B, Hq, S, D, device="npu", dtype=torch.float16)
k = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)
v = torch.randn(B, Hkv, S, D, device="npu", dtype=torch.float16)

out = npu_flash_attention_forward(q, k, v, causal=True)
```

### Sliding Window

```python
out = npu_flash_attention_forward(q, k, v, causal=True, sliding_window=64)
```

### 自定义 Scale

```python
out = npu_flash_attention_forward(q, k, v, scale=0.05)
```

### L2+ 框架 API (推荐)

使用 `flex_attention` + `AttentionConfig` 组合多种注意力模式:

```python
from npu_flash_attention import flex_attention, AttentionConfig

# Gemma2 风格: causal + sliding window + tanh soft-capping
cfg = AttentionConfig(causal=True, sliding_window=1024, soft_cap=50.0)
out = flex_attention(q, k, v, config=cfg)

# 或直接传 kwargs
out = flex_attention(q, k, v, causal=True, sliding_window=1024, soft_cap=50.0)

# autograd 端到端 (前向+反向自动支持)
q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
out = flex_attention(q, k, v, causal=True, soft_cap=50.0)
out.backward(grad_out)
```

### Tanh Soft-Capping (Gemma2 / Grok-1)

```python
# 对 attention logits 施加 tanh 软截断,防止 logits 过大
out = npu_flash_attention_forward(q, k, v, causal=True, soft_cap=50.0)
```

## 项目结构

```
flex-attn/
├── npu_flash_attention.py   # 核心实现: 前向+反向 Triton kernel + L2+ 框架 API
├── setup_env.sh             # bishengir-compile PATH 配置
├── opencode.json             # opencode skills 配置
├── DESIGN.md                 # 设计文档
├── Testcase.md               # 测试用例文档
├── RESULT.md                 # 测试结果文档
├── README.md                 # 本文件
└── test/                     # 测试脚本目录
    ├── test_comprehensive.py # 全面测试套件 (38 项)
    ├── test_bf16_full.py     # bf16 全量测试 (38 项)
    ├── test_real_world.py    # 真实场景测试 (12 项)
    ├── test_supplementary.py # 补充测试 (13 项)
    ├── bench_full_perf.py    # 全面性能测试 (7 组场景)
    ├── bench_vs_sdpa.py      # Flex vs SDPA 前向+反向对比 (6 组场景)
    └── ...
```

## 支持的特性

| 特性 | 前向 | 反向 | 备注 |
|------|------|------|------|
| Full Attention | ✅ | ✅ | 无掩码的全注意力 |
| Causal Mask | ✅ | ✅ | 因果掩码 (下三角) |
| Sliding Window | ✅ | ✅ | 滑动窗口掩码 |
| GQA (任意 ratio) | ✅ | ✅ | Grouped Query Attention |
| ALiBi Bias | ✅ | ✅ | 注意力分数线性偏置 |
| Tanh Soft-Capping | ✅ | ✅ | Gemma2/Grok-1 风格 logits 软截断 |
| 自定义 Scale | ✅ | ✅ | |
| LSE 输出 | ✅ | — | 用于反向传播 |
| fp32 / fp16 / bf16 | ✅ | ✅ | |
| **模式任意组合** | ✅ | ✅ | L2+ 框架: causal+sliding+alibi+softcap 等可同时启用 |

## 设计要点

本实现绕过了 PyTorch Inductor 的代码生成路径，直接编写 Triton kernel。原因：

1. **bishengir-compile 限制**：triton-ascend 3.2.0rc4 的编译器无法处理 Inductor 生成的 `memref.reinterpret_cast` + `memref.copy` 模式在循环内与 `arith.truncf` 组合的情况
2. **简单指针运算**：使用 `tl.load(ptr + offsets)` 直接指针运算已被验证在 NPU 上完全可用
3. **不用 scf.if**：所有条件分支用 `tl.where` 实现
4. **编译时常量循环上界**：使用 kernel 参数 (非 `tl.load` 值) 作为循环上界

详细设计见 [DESIGN.md](DESIGN.md)。

## 性能概览

优化后 (P0 因果跳过 + P4 mask 分裂 + P6 GQA KV 消除 + P7 DQ动态BLOCK_N + multibuffer 等) 实测 (fp16 causal, B=1 H=2 D=64):

| 序列长度 | 前向延迟 | 反向延迟 | SDPA 延迟 | 前向/SDPA | 前向 vs 优化前 |
|----------|----------|----------|-----------|-----------|---------------|
| S=128 | **0.33ms** | 1.37ms | 0.59ms | **1.81x (Flex 快)** | 2.13x |
| S=512 | 0.98ms | 3.27ms | 0.92ms | 0.94x (接近) | 2.13x |
| S=1024 | 1.55ms | 4.24ms | 1.08ms | 0.70x | 2.42x |
| S=2048 | 2.26ms | 9.27ms | 1.26ms | 0.56x | **4.10x** |

> **相比优化前基线**: 前向 1.44x-4.37x、反向 1.23x-2.48x 加速。
> **S=128 前向比 SDPA 快 81%**，S=512 接近持平；大序列 SDPA CANN 硬件级优化占主导。
> Flex 反向在 D=128 S=512 场景**快于 SDPA 18%**（无 autograd 图开销）。
> Flex 独特价值：支持 causal+sliding+ALiBi+soft-cap 组合、非标准 GQA 比例、自定义 LSE 输出。

详细测试结果与优化历程见 [RESULT.md](RESULT.md)。

## 运行测试

```bash
# 环境配置
source setup_env.sh

# 全面测试 (38 项)
python test/test_comprehensive.py

# bf16 全量测试 (38 项)
python test/test_bf16_full.py

# 真实场景测试 (12 项)
python test/test_real_world.py

# 补充测试 (13 项)
python test/test_supplementary.py

# 全面性能 benchmark (7 组场景)
python test/bench_full_perf.py

# Flex vs SDPA 前向+反向对比 (6 组场景)
python test/bench_vs_sdpa.py
```

## 已知限制

1. 不支持 PyTorch FlexAttention 的任意 `score_mod` / `mask_mod` 函数追踪 — 仅支持预定义原语 (causal、sliding window、ALiBi、GQA、tanh soft-capping 等) 的可组合特化
2. 大序列性能为 SDPA 的 0.56-0.70x — 原生 SDPA 使用 CANN `npu_fusion_attention` 硬件级内核 (软件流水、TMA 等)
3. GQA 场景 Flex 较慢 — SDPA 有原生 GQA 支持，Flex 在 Python 层 expand KV (Hq=32 Hkv=8 时 4x 复制)
4. 前向 DQ 反向 kernel 无法应用 P4 mask 分裂 — triton-ascend 编译器对 DQ 双循环形式误编译 (前向/DKDV 同结构正常)，DQ 保持单循环
5. 不支持 block-sparse BlockMask — 不支持 PyTorch FlexAttention 的块稀疏索引格式
6. KV 长度需为 BLOCK_N 的整数倍才能保证正确 (当前测试序列均满足)；非整除 KV 长度的 padding 位屏蔽为已知边界待完善项

> **已修复**: 早期版本 dQ/dK 有 ~10-15% 误差 (按块局部 row_sum 近似 softmax 雅可比)，已通过**全局 Delta 修复**
> 消除，现 fp32 达 ~1e-7、fp16 达 ~2e-4 精度。autograd Function 路径也已修正 (此前 backward 未传 out)。

## 适用场景

- Gemma2 / Grok-1 等使用 tanh soft-capping 的模型注意力
- Mistral 等使用 sliding window + causal 的模型
- 需要自定义掩码组合 (causal + sliding window + ALiBi + soft-cap) 的注意力计算
- 需要 LSE 输出用于自定义反向传播或 KV cache 场景
- 需要 bf16/fp16 混合精度训练的前向 + 反向
- 研究 Triton-Ascend 在 NPU 上的 kernel 开发
- 非标准 GQA ratio (如 32:8) 场景，SDPA 不直接支持
- 小序列前向低延迟场景 (S≤128, Flex 比 SDPA 快 81%)

## 不适用场景

- 需要**任意** `score_mod`/`mask_mod` (超出预定义原语库的自定义函数) — 需等待 triton-ascend 修复编译器限制后通过 Inductor 路径实现
- 需要极致性能的大规模推理 — 建议使用 `npu_fusion_attention` 原生算子
- 需要 block-sparse 掩码 (如文档级掩码、自定义稀疏模式) — 当前不支持 BlockMask
- 需要与 `torch.compile` 无缝集成的场景 — 直接调用方式不支持 autograd tracing
- GQA 大比例场景 (Hq=32 Hkv=8) — SDPA 有原生 GQA 支持，Flex 需 Python 层 expand 较慢

## triton-ascend 版本现状与展望

### 当前版本 (3.2.0rc4 + CANN 8.3.RC1)

本项目使用的 triton-ascend 3.2.0rc4 存在以下编译器限制，是采用独立 kernel 方案的根本原因：

| 限制 | 影响 | 状态 |
|------|------|------|
| `vcast`/`vexp` "root alloc" 错误 | 循环内类型转换无法编译 | ❌ 未修复 |
| `scf.if` 在 `scf.for` 内 | 循环内条件分支失败 | ⚠️ 部分改善 |
| `memref.reinterpret_cast` + `memref.copy` 在循环内 | Inductor 生成的内存模式失败 | ⚠️ 部分修复 |
| 运行时加载循环上界 (`tl.load` 值作为循环上界) | 块稀疏索引无法使用 | ❌ 不支持 |

### 最新版本 (3.2.1 + CANN 9.0.0)

截至 2026 年 7 月，triton-ascend 最新稳定版本为 **3.2.1** (2026-04-30 发布)，需要 **CANN 9.0.0**。相比 3.2.0rc4 的改进：

- ✅ 修复 fp8/bf16/half 在 dot/cast/pow 中的类型处理
- ✅ 改进 `scf.if` lowering (简单场景可用)
- ✅ 修复动态 memref offset 不匹配 (PR #965)
- ✅ 新增 Ascend 950 硬件支持
- ✅ 新增 AutoTune 自动优化 (试验性)
- ❌ **`vcast`/`vexp` root alloc 错误仍存在** ([Issue #448](https://github.com/triton-lang/triton-ascend/issues/448))
- ❌ **运行时加载循环上界仍不支持** ([Issue #888](https://github.com/triton-lang/triton-ascend/issues/888))
- ❌ **FlexAttention 无原生支持** (未在路线图中)

### 结论

升级到 triton-ascend 3.2.1 + CANN 9.0.0 **不能** 解决 FlexAttention 的核心限制。独立 Triton kernel 方案（本项目方案）在可预见的未来仍是唯一可行路径。triton-ascend 的路线图计划升级到 **Triton 3.5**，但即使升级后，BiShengIR 编译器对动态循环上界和复杂内存模式的限制仍需解决。

详细版本兼容性：

| triton-ascend | CANN | FlexAttention 可行性 |
|---------------|------|---------------------|
| 3.2.0rc4 | 8.3.RC1 | ❌ Inductor 路径; ✅ 独立 kernel (本项目) |
| 3.2.0 | 8.5.0 | ❌ 同上 |
| 3.2.1 | 9.0.0 | ❌ 同上 (vcast/vexp 未修复) |
| 未来 (Triton 3.5) | TBD | ⚠️ 取决于 BiShengIR 动态循环支持 |

## 技术背景

本项目源于对 PyTorch FlexAttention 在 Ascend NPU 上的可行性分析。FlexAttention 是编译器驱动的代码生成式注意力实现，其 Inductor 生成的 Triton 内核在 bishengir-compile 上失败。通过系统化的模式隔离测试定位了根因（循环内 `memref.reinterpret_cast` + `vcast`/`vexp` 组合），最终采用独立 Triton kernel 方案绕过 Inductor。

详见 [DESIGN.md](DESIGN.md) 中的背景与动机章节。
# npu-flex-att
