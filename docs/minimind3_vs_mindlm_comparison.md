# MiniMind3 vs MindLM-1B 模型结构对比分析

> 基于源码 `minimind/model/model_minimind.py` 与 `modeling_mindlm.py` 的逐行对比

---

## 1. 总览

| 特性 | MiniMind3 | MindLM-1B |
|------|-----------|-----------|
| 参数量 | ~64M (Dense) | ~95.3M (Dense) |
| 隐藏维度 (dim) | 768 | 768 |
| Transformer 层数 | 8 | 16 |
| Q Heads | 8 | 12 |
| KV Heads | 4 | 3 |
| GQA 压缩比 | 2:1 | 4:1 |
| FFN 中间维度 | 2048 | 2048 |
| 词表大小 | 6,400 | 6,400 |
| 最大上下文长度 | 32,768 | 1,024 |
| 权重共享 | embed == lm_head | embed == output |

---

## 2. 注意力机制 — 最核心差异

### 2.1 MiniMind3：纯标准 GQA + QK Norm

所有 8 层均使用标准 Multi-Head Attention，实现位于 `model_minimind.py:90-132`。

```python
class Attention(nn.Module):
    self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
    self.q_norm = RMSNorm(head_dim)   # QK Norm — Qwen3 标志性特征
    self.k_norm = RMSNorm(head_dim)
```

前向流程：
1. Q/K/V 线性投影
2. **QK RMSNorm 归一化**（稳定训练，防止注意力分数爆炸）
3. RoPE 旋转位置编码
4. KV Repeat（4 heads → 8 heads）
5. Flash Attention 或手动 Softmax

### 2.2 MindLM-1B：混合注意力架构

16 层分为两类，由 `layer_types` 配置控制：

```
层  0-2:  GatedDeltaNet (线性注意力)
层  3:    标准 GQA Attention
层  4-6:  GatedDeltaNet
层  7:    标准 GQA Attention
层  8-10: GatedDeltaNet
层 11:    标准 GQA Attention
层 12-14: GatedDeltaNet
层 15:    标准 GQA Attention
→ 75% 线性注意力 + 25% 标准注意力
```

#### 标准注意力层（`modeling_mindlm.py:74-133`）

```python
class Attention(nn.Module):
    self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
    self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
    # 注意：无 QK Norm
```

与 MiniMind3 的标准注意力差异：

| | MiniMind3 | MindLM |
|---|---|---|
| QK Norm | 有（RMSNorm） | 无 |
| RoPE theta | 1,000,000 | 10,000 |
| GQA 比例 | 2:1 (8:4) | 4:1 (12:3) |
| RoPE 实现 | cos/sin 分离存储 | 复数极坐标 `torch.polar` |

#### GatedDeltaNet 线性注意力层（`modeling_mindlm.py:136-305`）

这是 MindLM 的核心创新，实现固定状态矩阵的循环注意力：

```python
class GatedDeltaNet(nn.Module):
    # 输入投影
    self.in_proj_qkv = nn.Linear(hidden_size, key_dim*2 + value_dim, bias=False)
    self.in_proj_z   = nn.Linear(hidden_size, value_dim, bias=False)  # 门控信号
    self.in_proj_b   = nn.Linear(hidden_size, num_v_heads, bias=False)  # 写入门
    self.in_proj_a   = nn.Linear(hidden_size, num_v_heads, bias=False)  # 衰减参数

    # 因果卷积（替代位置编码）
    self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=4, groups=conv_dim)

    # 可学习衰减
    self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
    self.A_log   = nn.Parameter(torch.log(torch.arange(1, num_v_heads+1).float()))
```

核心递推公式：

```
S_t = exp(g_t) × S_{t-1} + k_t × (v_t × β_t)
out_t = q_t × S_t

其中：
  S_t  — 固定大小 (head_dim, value_dim) 的循环状态矩阵
  g_t  — 衰减门（控制遗忘）
  β_t  — 输入门（控制写入强度）
```

分块并行计算（`chunk_size=64`）：
- 序列切分为 64 token 的块
- 块内：矩阵乘法并行计算衰减加权注意力
- 块间：传递 recurrent state
- Python 循环从 T 次降到 T/64 次

数值稳定性：全程 log 空间计算，`clamp(max=0)` 防止 exp 溢出。

---

## 3. 位置编码

### MiniMind3：RoPE + YaRN 扩展

```python
# precompute_freqs_cis — model_minimind.py:61
freqs = 1.0 / (rope_base ** (arange(0, dim, 2) / dim))
# rope_theta = 1,000,000（远大于常见的 10,000）
# 支持 YaRN 长度扩展：beta_fast=32, beta_slow=1, factor=16
```

- 超大 theta (1M) 使 RoPE 在更长序列上仍保持分辨能力
- YaRN（Yet another RoPE extensioN method）支持上下文从 2048 扩展到 32768
- cos/sin 分离存储，`rotate_half` 实现旋转

### MindLM-1B：分层策略

| 层类型 | 位置编码方式 |
|--------|------------|
| 标准注意力 | RoPE (theta=10,000，复数极坐标实现) |
| 线性注意力 | **无位置编码** — 因果 Conv1d (kernel=4) 提供局部位置感知 |

```python
# precompute_pos_cis — modeling_mindlm.py:30
freqs = 1.0 / (theta ** (arange(0, dim, 2) / dim))
pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数极坐标
```

线性注意力层不需要位置编码的原因：
- Conv1d 的因果卷积天然编码了局部 token 顺序
- 循环状态的衰减机制隐式编码了时序距离信息

---

## 4. FFN / MoE

两者均使用 **SwiGLU** 激活函数：`down_proj(SiLU(gate_proj(x)) * up_proj(x))`

### MiniMind3 MoE（`model_minimind.py:146-174`）

```python
class MOEFeedForward:
    num_experts = 4
    num_experts_per_tok = 1          # Top-1 路由
    norm_topk_prob = True
    router_aux_loss_coef = 5e-4
    # 无共享专家
```

路由策略：Softmax 门控 → Top-K 选择 → 归一化权重 → 加权聚合。训练时通过 aux_loss 负载均衡。

### MindLM MoE（`modeling_mindlm.py:325-444`）

```python
class MOEFeedForward:
    n_routed_experts = 8             # 8 个路由专家
    num_experts_per_tok = 2          # Top-2 路由
    n_shared_experts = 1             # 1 个共享专家（始终激活）
    scoring_func = 'softmax'
    aux_loss_alpha = 0.01
    seq_aux = True                   # 序列级辅助损失
    norm_topk_prob = True
```

与 MiniMind3 MoE 的关键区别：
- **共享专家机制**：1 个专家始终激活，提供稳定的基础能力
- **Top-2 路由**：每个 token 同时激活 2 个专家（而非 1 个）
- **推理优化**：`moe_infer` 方法按专家排序批量处理，减少计算浪费
- 更复杂的负载均衡损失（序列级别）

> 注：当前 MindLM-1B 配置 `use_moe=false`，MoE 功能已实现但未启用。

---

## 5. 归一化

两者均使用 **RMSNorm** + **Pre-Norm** 架构，在每个 Transformer Block 中：

```python
# MiniMind3
self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)

# MindLM — 完全一致
self.attention_norm = RMSNorm(dim, eps=1e-6)
self.ffn_norm = RMSNorm(dim, eps=1e-6)
```

MindLM 线性注意力层额外使用 **GateNorm**：

```python
def gated_norm(x, gate):
    x = x * rsqrt(mean(x^2) + eps)   # RMSNorm
    x = x * SiLU(gate)                # 门控
    return x
```

以及 Q/K 上的 **L2 归一化**（仅线性注意力层）：

```python
def l2norm(x, dim=-1, eps=1e-6):
    return x * rsqrt(sum(x^2, dim) + eps)
```

---

## 6. 参数量估算

### MiniMind3 (Dense)

```
Embedding (tied):         6,400 × 768   =   4.9M
Per Layer (×8):
  Attention:    768² + 768×384×2 + 768² + 2×768  =   1.77M
  FFN:          768×2048×3                        =   4.72M
  LayerNorm:    768 × 2                           ≈   0.001M
  小计:                                            ≈   6.49M
8 层合计:                                          ≈  51.9M
Final Norm + Head:                                 ≈   4.9M
─────────────────────────────────────────────────────────
总计:                                               ≈ ~64M
```

### MindLM-1B (Dense)

```
Embedding (tied):         6,400 × 768   =   4.9M
标准注意力层 (×4):
  Attention:    768² + 768×192×2 + 768²  =   1.47M
  FFN:          768×2048×3                =   4.72M
  RMSNorm×2:    768 × 2                   ≈   0.002M
  小计:                                    ≈   6.19M
  4 层合计:                                ≈  24.8M

线性注意力层 (×12):
  GatedDeltaNet:  n_kv_heads=3, 所以 K/V 维度远小于 n_heads=12
    in_proj_qkv:  768 × (192×2 + 192) = 768 × 576 = 0.44M
    in_proj_z:    768 × 192                      = 0.15M
    in_proj_b/a:  768 × 3 × 2                    ≈ 0.005M
    conv1d:       576 × 4 (groups=576)            ≈ 0.002M
    dt_bias+A_log:3 × 2                           ≈ 0M
    out_proj:     192 × 768                       = 0.15M
    RMSNorm×2:    768 × 2                         ≈ 0.002M
    小计:                                          ≈ 0.75M
  FFN:           768×2048×3                       = 4.72M
  每层小计:                                      ≈   5.47M
  12 层合计:                                      ≈  65.6M

Final Norm + Head:          ≈   4.9M
─────────────────────────────────────────────────────────
总计:                        ≈ ~95.3M  (实际训练输出: 95.267M)
```

> **注意**：GatedDeltaNet 使用 `n_kv_heads=3`（而非 `n_heads=12`），
> 因此 `key_dim = 64×3 = 192`，`conv_dim = 192×2+192 = 576`，
> 参数量远小于假设全部 12 个头时的估算。

---

## 7. 计算复杂度与推理效率

| 指标 | MiniMind3 | MindLM-1B |
|------|-----------|-----------|
| 注意力计算复杂度 | O(n² × d) 全层 | 混合：75% 层 O(n × d²)，25% 层 O(n² × d) |
| KV Cache | 全部层，随序列长度线性增长 | 仅 4 层标准注意力有 KV Cache |
| 推理内存 | 较高 | 更低（75% 层无 KV Cache 膨胀） |
| 长序列扩展性 | 受限于 O(n²) 和 KV Cache | 线性注意力层天然支持任意长度 |

---

## 8. 架构溯源

### MiniMind3 → Qwen3

MiniMind3 的架构与 Qwen3 几乎完全对应，核心特征逐一吻合：

| 组件 | MiniMind3 | Qwen3 (`modeling_qwen3.py`) | 一致性 |
|------|-----------|------------------------------|----|
| 注意力 | GQA | GQA | 完全一致 |
| QK Norm | `RMSNorm(head_dim)` | `Qwen3RMSNorm(head_dim)` | 完全一致 |
| FFN | SwiGLU (gate/up/down) | SwiGLU (gate/up/down) | 完全一致 |
| 位置编码 | RoPE (theta=1e6) | RoPE | 结构一致，theta 不同 |
| 归一化 | RMSNorm Pre-Norm | RMSNorm Pre-Norm | 完全一致 |
| MoE | Top-1，4 专家 | Qwen3MoE 变体 | 结构相似 |

**标志性特征 — QK Norm** 是 Qwen3 区别于其他模型的关键设计，MiniMind3 完整保留了这一特征。MiniMind3 本质上是一个**小规模的 Qwen3 架构复现**。

### MindLM → Qwen3.5

MindLM 的混合注意力架构与 Qwen3.5 高度对应：

| 组件 | MindLM | Qwen3.5 (`modeling_qwen3_5.py`) | 一致性 |
|------|--------|----------------------------------|----|
| 混合注意力策略 | 75% 线性 + 25% 标准 | 同样的混合策略 | 完全一致 |
| layer_types 配置 | `["linear_attention", ..., "attention"]` | 同样的配置机制 | 完全一致 |
| GatedDeltaNet | in_proj_qkv/z/b/a + Conv1d + chunk | 同样结构 | 完全一致 |
| 衰减参数 | A_log + dt_bias | A_log + dt_bias | 完全一致 |
| chunk_size | 64 | 64 | 完全一致 |
| L2 归一化 | l2norm(Q), l2norm(K) | 同样 | 完全一致 |
| 门控归一化 | RMSNorm + SiLU(gate) | RMSNormGated | 结构一致 |

GatedDeltaNet 输入投影的逐行对应：

```python
# Qwen3.5
self.in_proj_qkv = nn.Linear(hidden_size, key_dim*2 + value_dim, bias=False)
self.in_proj_z   = nn.Linear(hidden_size, value_dim, bias=False)
self.in_proj_b   = nn.Linear(hidden_size, num_v_heads, bias=False)
self.in_proj_a   = nn.Linear(hidden_size, num_v_heads, bias=False)
self.dt_bias     = nn.Parameter(torch.ones(num_v_heads))
self.A_log       = nn.Parameter(torch.log(A))

# MindLM — 完全相同
self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
self.in_proj_z   = nn.Linear(hidden_size, value_dim, bias=False)
self.in_proj_b   = nn.Linear(hidden_size, num_v_heads, bias=False)
self.in_proj_a   = nn.Linear(hidden_size, num_v_heads, bias=False)
self.dt_bias     = nn.Parameter(torch.ones(num_v_heads))
self.A_log       = nn.Parameter(torch.log(torch.arange(1, num_v_heads+1).float()))
```

**差异主要在工程层面：**

| 方面 | MindLM | Qwen3.5 |
|------|--------|---------|
| 实现语言 | 纯 PyTorch | FLA/causal-conv1d 加速内核 + PyTorch 回退 |
| 推理模式 | 统一 chunk 算法 | 训练用 chunk，推理用 fused recurrent |
| 状态缓存 | 未实现 | Qwen3_5DynamicCache（conv_states + recurrent_states） |
| 多模态 | 无 | MRoPE 3D 位置编码 + 视觉编码器 |
| A 参数初始化 | `arange(1, num_v_heads+1)` | `Uniform(0, 16)` |

### 与其他模型的距离

#### GLM4/GLM5

| 差异点 | MiniMind/MindLM | GLM4 |
|--------|-----------------|------|
| RoPE 方式 | 标准旋转 | 部分旋转 + 交错 (interleave) |
| QK Norm | 有 | 可选，默认关闭 |
| FFN 投影 | gate/up/down 三分离 | gate_up 融合投影 |
| Block 内 Norm 数 | 2 个 | 4 个（extra post_self_attn + post_mlp） |

差异较大，不属于同一架构族。

#### Kimi K2.5

Kimi K2.5 在 HuggingFace transformers 库中尚无实现（截至当前版本），其公开架构为 MoE 密集路由，与 MindLM 的 Dense/GatedDeltaNet 混合架构路线不同。

---

## 9. 优劣总结

### MiniMind3

**优势：**
- 结构简洁，纯标准 Transformer，实现和调试门槛低
- QK Norm 稳定训练，防止注意力分数溢出
- 超大 RoPE theta + YaRN，明确的 32K 长上下文支持
- 与 Qwen3 生态完全兼容，可直接对接 HuggingFace
- 参数量小（64M），训练和推理资源需求低

**劣势：**
- O(n²) 复杂度，长序列计算和内存开销大
- 仅 8 层，模型深度和容量有限
- GQA 压缩比低（2:1），KV Cache 节省较少
- 无线性注意力，无法享受 O(n) 的效率优势

### MindLM-1B

**优势：**
- GatedDeltaNet 线性注意力带来 O(n) 复杂度，75% 的层无 KV Cache 膨胀
- 16 层深度，模型表达力更强
- GQA 4:1 压缩比，标准注意力层 KV Cache 更小
- 混合策略兼顾效率与精度（保留 25% 标准注意力）
- MoE 实现更完善（共享专家 + Top-2 + 推理优化）

**劣势：**
- GatedDeltaNet 实现复杂度高（chunked 并行、log 空间数值稳定性）
- 纯 PyTorch 实现缺少 FLA 等加速内核，训练速度不及优化版本
- 推理时缺少 fused recurrent 模式，无法 O(1) 逐 token 生成
- 尚未做 HuggingFace transformers 库适配（但架构本身不存在障碍）
- 默认上下文仅 1024，虽然线性注意力天然支持更长序列，但标准层仍是瓶颈

---

## 10. 演进关系

```
Qwen3 (标准 GQA + QK Norm + SwiGLU)
  │
  ├─→ MiniMind3：小规模复现，保留全部核心特征
  │
  └─→ Qwen3.5 (混合 GatedDeltaNet + 标准 Attention)
        │
        └─→ MindLM：简化实现，聚焦语言建模，去除多模态和加速内核
```

MindLM 相对 MiniMind3 的核心进步是引入了 Qwen3.5 的线性注意力混合策略，这是当前大模型架构从纯标准注意力向**线性注意力混合架构**演进的重要方向。两者都是对前沿架构的教学级复现，尚未引入原创性结构创新。
