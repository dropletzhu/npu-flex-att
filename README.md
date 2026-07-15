# NPU Flash Attention

在 Ascend NPU 上基于 Triton-Ascend 实现的 Flash Attention 内核，支持前向传播与反向传播，适用于因果掩码、滑动窗口、GQA 等常见注意力模式。

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

## 项目结构

```
flex-attn/
├── npu_flash_attention.py   # 核心实现: 前向+反向 Triton kernel + Python API
├── setup_env.sh             # bishengir-compile PATH 配置
├── opencode.json             # opencode skills 配置
├── DESIGN.md                 # 设计文档
├── Testcase.md               # 测试用例文档
├── RESULT.md                 # 测试结果文档
├── README.md                 # 本文件
└── test/                     # 测试脚本目录
    ├── test_comprehensive.py # 全面测试套件 (38 项)
    ├── test_api_compat.py    # Triton API 兼容性验证
    ├── test_minimal_flash_attn.py   # 最小 Flash Attention 验证
    ├── test_grad_correctness.py     # 梯度正确性验证
    ├── test_npu_flash_attn.py       # 前向+反向功能测试
    ├── test_bf16_full.py       # bf16 全量测试 (22 项)
    ├── test_real_world.py      # 真实场景测试 (12 项)
    ├── bench_flash_attn.py          # 性能基准测试
    ├── bench_block_sweep.py         # Block size 扫描
    └── ...                         # 其他调试/隔离测试脚本
```

## 支持的特性

| 特性 | 前向 | 反向 | 备注 |
|------|------|------|------|
| Full Attention | ✅ | ✅ | 无掩码的全注意力 |
| Causal Mask | ✅ | ✅ | 因果掩码 (下三角) |
| Sliding Window | ✅ | ✅ | 滑动窗口掩码 |
| GQA (任意 ratio) | ✅ | ✅ | Grouped Query Attention |
| 自定义 Scale | ✅ | ✅ | |
| LSE 输出 | ✅ | — | 用于反向传播 |
| fp32 / fp16 / bf16 | ✅ | ✅ | |

## 设计要点

本实现绕过了 PyTorch Inductor 的代码生成路径，直接编写 Triton kernel。原因：

1. **bishengir-compile 限制**：triton-ascend 3.2.0rc4 的编译器无法处理 Inductor 生成的 `memref.reinterpret_cast` + `memref.copy` 模式在循环内与 `arith.truncf` 组合的情况
2. **简单指针运算**：使用 `tl.load(ptr + offsets)` 直接指针运算已被验证在 NPU 上完全可用
3. **不用 scf.if**：所有条件分支用 `tl.where` 实现
4. **编译时常量循环上界**：使用 kernel 参数 (非 `tl.load` 值) 作为循环上界

详细设计见 [DESIGN.md](DESIGN.md)。

## 性能概览

| 序列长度 | 前向延迟 | 反向延迟 | SDPA 延迟 | 前向/SDPA |
|----------|----------|----------|-----------|-----------|
| S=128 | 0.46ms | 1.40ms | 0.68ms | 1.47x |
| S=512 | 2.22ms | 4.48ms | 0.68ms | 0.31x |
| S=1024 | 3.38ms | 7.03ms | 0.93ms | 0.28x |
| S=2048 | 6.29ms | 15.3ms | 0.76ms | 0.12x |

> S=128 前向超过 SDPA 原生内核速度。大序列的差距来自 CANN `npu_fusion_attention` 的硬件级优化 (TMA、Cube 直接调度等)。

详细测试结果见 [RESULT.md](RESULT.md)。

## 运行测试

```bash
# 环境配置
source setup_env.sh

# 全面测试 (38 项, 约 5-10 分钟)
python test/test_comprehensive.py

# bf16 全量测试 (22 项)
python test/test_bf16_full.py

# 真实场景测试 (12 项)
python test/test_real_world.py

# 性能 benchmark
python test/bench_flash_attn.py

# API 兼容性
python test/test_api_compat.py
```

## 已知限制

1. 不支持 PyTorch FlexAttention 的任意 `score_mod` / `mask_mod` 函数追踪 — 仅支持预定义的 causal、sliding window 模式
2. `torch.autograd.Function` 集成可能触发 AICore 异常 — 直接调用 forward/backward 函数无此问题
3. dQ/dK 有 ~10-15% 相对误差 — 由 Triton-Ascend `exp2` 精度限制导致，训练可接受
4. 大序列性能为 SDPA 的 0.1-0.3x — 原生 SDPA 使用 CANN `npu_fusion_attention` 硬件级内核
5. 反向 BLOCK_M=16, BLOCK_N=32 固定 — 更大 block 会触发 910B3 UB 容量溢出 (AICore exception)
6. 不支持 block-sparse BlockMask — 不支持 PyTorch FlexAttention 的块稀疏索引格式

## 适用场景

- 需要自定义掩码组合 (causal + sliding window) 的注意力计算
- 需要 LSE 输出用于自定义反向传播或 KV cache 场景
- 需要 bf16/fp16 混合精度训练的前向 + 反向
- 研究 Triton-Ascend 在 NPU 上的 kernel 开发
- 原生 SDPA 不支持的 GQA ratio (如 32:8) 场景

## 不适用场景

- 需要任意 score_mod (如 ALiBi、相对位置编码等动态修改) 的场景 — 需等待 triton-ascend 修复编译器限制后通过 Inductor 路径实现
- 需要极致性能的大规模推理 — 建议使用 `npu_fusion_attention` 原生算子
- 需要 block-sparse 掩码 (如文档级掩码、自定义稀疏模式) — 当前不支持 BlockMask
- 需要与 `torch.compile` 无缝集成的场景 — 直接调用方式不支持 autograd tracing

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
