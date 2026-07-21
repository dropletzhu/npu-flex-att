# NPU Flash Attention 测试用例文档

## 测试环境

| 项目 | 配置 |
|------|------|
| NPU 硬件 | Ascend 910B3 × 8 (64GB HBM) |
| CANN | 8.3.RC1 (aarch64) |
| PyTorch | 2.9.1+cpu |
| torch_npu | 2.9.1 |
| triton-ascend | 3.2.0rc4 |
| Block Size | 前向 BLOCK_M=16, BLOCK_N=64; 反向 BLOCK_M=16, BLOCK_N=32/64 (按 Sk 动态, DQ/DKDV 均使用) |

## 测试用例

### A. 前向正确性 (15 项)

| ID | 测试名称 | 输入配置 | 参考实现 | 验证标准 |
|----|----------|----------|----------|----------|
| A01 | 全注意力 fp32 | B=1 H=1 S=128 D=64, fp32, 无 mask | F.scaled_dot_product_attention | atol=1e-5 |
| A02 | 全注意力 fp16 | B=1 H=1 S=128 D=64, fp16, 无 mask | F.scaled_dot_product_attention | atol=1e-3 |
| A03 | Causal mask fp32 | B=1 H=1 S=128 D=64, causal, fp32 | F.sdpa(is_causal=True) | atol=1e-5 |
| A04 | Causal mask fp16 | B=1 H=1 S=128 D=64, causal, fp16 | F.sdpa(is_causal=True) | atol=1e-3 |
| A05 | Causal mask bf16 | B=1 H=1 S=128 D=64, causal, bf16 | F.sdpa(is_causal=True) | atol=1e-2 |
| A06 | GQA ratio=2 fp16 | B=1 Hq=2 Hkv=1 S=128 D=64, causal, fp16 | F.sdpa(expand K/V) | atol=1e-3 |
| A07 | GQA ratio=4 fp16 | B=1 Hq=4 Hkv=1 S=128 D=64, causal, fp16 | F.sdpa(expand K/V) | atol=1e-3 |
| A08 | GQA ratio=8 fp16 | B=1 Hq=8 Hkv=1 S=256 D=64, causal, fp16 | F.sdpa(expand K/V) | atol=1e-3 |
| A09 | GQA Hkv=2 fp16 | B=1 Hq=4 Hkv=2 S=128 D=64, causal, fp16 | F.sdpa(expand K/V) | atol=1e-3 |
| A10 | 多 batch fp16 | B=4 H=4 S=128 D=64, causal, fp16 | F.sdpa(is_causal=True) | atol=1e-3 |
| A11 | Sliding window fp32 | B=1 H=1 S=256 D=64, causal, window=64, fp32 | 逐行 softmax 参考 | atol=1e-4 |
| A12 | Sliding window fp16 | B=1 H=1 S=256 D=64, causal, window=64, fp16 | 逐行 softmax 参考 | atol=1e-2 |
| A13 | 大 head_dim=128 fp16 | B=1 H=1 S=128 D=128, causal, fp16 | F.sdpa(is_causal=True) | atol=1e-3 |
| A14 | 非方阵 Q≠KV fp16 | B=1 H=1 Sq=128 Skv=256 D=64, causal, fp16 | F.sdpa(is_causal=True) | atol=1e-3 |
| A15 | 长序列 S=2048 fp16 | B=1 H=1 S=2048 D=64, causal, fp16 | F.sdpa(is_causal=True) | atol=1e-2 |

### B. 反向梯度正确性 (6 项)

| ID | 测试名称 | 输入配置 | 参考实现 | 验证标准 |
|----|----------|----------|----------|----------|
| B01 | dQ/dK/dV causal fp32 | B=1 H=1 S=128 D=64, fp32, causal | torch.autograd + F.sdpa | dQ/dK: atol=0.2, dV: atol=1e-4 |
| B02 | dQ/dK/dV causal fp16 | B=1 H=1 S=128 D=64, fp16, causal | torch.autograd + F.sdpa | dQ: atol=0.5 |
| B03 | dQ/dK/dV 全注意力 fp32 | B=1 H=1 S=128 D=64, fp32, 无 causal | torch.autograd + F.sdpa | dQ/dK: atol=0.2, dV: atol=1e-4 |
| B04 | dQ/dK/dV GQA fp32 | B=1 Hq=4 Hkv=1 S=128 D=64, fp32, causal | torch.autograd + F.sdpa(expand) | dQ/dK: atol=0.3, dV: atol=0.3 |
| B05 | dQ/dK/dV sliding fp32 | B=1 H=1 S=128 D=64, fp32, causal, window=32 | 逐行参考 | 运行无异常 |
| B06 | dV 独立验证 | B=1 H=1 S=256 D=64, fp32, causal | dV = P^T @ dO 数值验证 | atol=1e-4 |

### C. 数值稳定性 (6 项)

| ID | 测试名称 | 输入配置 | 验证标准 |
|----|----------|----------|----------|
| C01 | 大 scale 值 | scale=10.0, S=128 D=64 fp32 | 输出无 NaN/Inf |
| C02 | 小 scale 值 | scale=0.01, S=128 D=64 fp32 | 输出无 NaN/Inf |
| C03 | 极端输入 fp16 | q/k = randn * 100, S=128 D=64 fp16 | 输出无 NaN/Inf |
| C04 | 全零 Value | v = zeros, S=128 D=64 fp32 | 输出全零 |
| C05 | 全零 Query | q = zeros, S=128 D=64 fp32 | 输出 = 均匀分布注意力 (mean of V) |
| C06 | LSE 范围检查 | S=128~2048, 检查 LSE 无 Inf | LSE ∈ [-1000, 1000] |

### D. 边界条件 (6 项)

| ID | 测试名称 | 输入配置 | 验证标准 |
|----|----------|----------|----------|
| D01 | 最小序列 S=16 | B=1 H=1 S=16 D=64, causal, fp16 | atol=1e-3 |
| D02 | 非整除序列 | B=1 H=1 S=33 D=64, causal, fp16 | atol=1e-3 |
| D03 | 单 token S=1 | B=1 H=1 S=1 D=64, causal, fp16 | 输出 = V[0] |
| D04 | 最简配置 | B=1 H=1 S=128 D=64, fp16 | atol=1e-3 |
| D05 | 大 batch | B=8 H=8 S=256 D=64, fp16 | atol=1e-3 |
| D06 | 小 head_dim D=32 | B=1 H=1 S=128 D=32, fp16 | atol=1e-3 |

### E. 性能测试 (7 组)

| ID | 测试名称 | 扫描维度 | 指标 |
|----|----------|----------|------|
| E00 | 优化前后对比 | S ∈ {128,256,512,1024,2048}, D=64, fp16, causal | ms + vs v2 加速比 + vs SDPA |
| E01 | 前向延迟扫描 D=64 | S ∈ {128,256,512,1024,2048}, D=64, fp16, causal | ms + vs SDPA |
| E01b | 前向延迟扫描 D=128 | S ∈ {128,256,512,1024,2048}, D=128, fp16, causal | ms + vs SDPA |
| E03 | 反向延迟扫描 D=64 | S ∈ {128,256,512,1024,2048}, D=64, fp16, causal | ms + vs SDPA + vs v2 |
| E03b | 反向延迟 D=128 | S ∈ {128,256,512,1024,2048}, D=128, fp16, causal+non-causal | ms + vs SDPA |
| E07 | Batch scaling | B ∈ {1,2,4,8}, H ∈ {4,16}, S=512, D=64, fp16 | ms + vs SDPA |
| E08 | D=128 多场景 | B×H×S 组合, D=128, fp16, causal | ms + vs SDPA |

### F. 功能完整性 (5 项)

| ID | 测试名称 | 验证内容 |
|----|----------|----------|
| F01 | return_lse=True | LSE shape=(B,H,S), 无 NaN |
| F02 | LSE→backward 链路 | 前向 LSE 用于反向, 梯度正确 |
| F03 | Causal+GQA+fp16 组合 | 多特性组合正确性 |
| F04 | Sliding+Causal+GQA 组合 | 多特性组合运行无异常 |
| F05 | 自定义 scale | scale=0.05/0.1/0.2 与参考一致 |

> **注意**: tanh soft-capping 和 ALiBi 功能测试见 `test_supplementary.py` 中的 L2+ 框架 API 验证和 soft-cap 精度测试。

## 执行方式

```bash
# 环境配置
source setup_env.sh

# 全面测试套件 (38 项, 约 5-10 分钟)
python test/test_comprehensive.py

# bf16 全量测试 (38 项, 与 fp32/fp16 对齐)
python test/test_bf16_full.py

# 真实场景测试 (12 项)
python test/test_real_world.py

# 补充测试 (13 项: 跨dtype一致性/输出dtype/内存/训练循环/L2+框架/soft-cap)
python test/test_supplementary.py

# 全面性能 benchmark (7 组场景)
python test/bench_full_perf.py

# Flex vs SDPA 前向+反向对比 (6 组场景)
python test/bench_vs_sdpa.py
```

## 测试统计

| 类别 | 测试数 | 通过 | 通过率 |
|------|--------|------|--------|
| A. 前向正确性 (fp32/fp16) | 15 | 15 | 100% |
| B. 反向梯度 (fp32/fp16) | 6 | 6 | 100% |
| C. 数值稳定性 (fp32/fp16) | 6 | 6 | 100% |
| D. 边界条件 (fp32/fp16) | 6 | 6 | 100% |
| F. 功能完整性 (fp32/fp16) | 5 | 5 | 100% |
| bf16 全量测试 (前向+反向+稳定性+边界+功能) | 38 | 38 | 100% |
| 真实场景测试 (非连续/验证/确定性/大规模) | 12 | 12 | 100% |
| 补充测试 (跨dtype一致性/输出dtype/内存/训练循环) | 13 | 13 | 100% |
| **合计** | **101** | **101** | **100%** |
