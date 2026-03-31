# MindLM-1B 架构设计文档

> 混合标准注意力 + GatedDeltaNet 线性注意力的小型语言模型

---

## 1. 设计概览

### 1.1 设计目标

MindLM-0.1B 的核心目标是：**在 ~425M 参数量级，构建一个推理高效、支持长上下文的语言模型。**

关键设计决策：
- 混合注意力架构（75% 线性 + 25% 标准），兼顾效率与精度
- GQA (Grouped Query Attention) 降低 KV Cache 开销
- 标准 SwiGLU FFN（不用 MoE，保持简洁）
- 自研 6400 词表 BPE 分词器，适配中英文

### 1.2 最终配置

```python
MindLM_1B = {
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "n_kv_heads": 3,
    "vocab_size": 6400,
    "max_seq_len": 8192,
    "use_moe": False,
    "layer_types": [
        "linear_attention",  # 0
        "linear_attention",  # 1
        "linear_attention",  # 2
        "attention",         # 3  ← 标准
        "linear_attention",  # 4
        "linear_attention",  # 5
        "linear_attention",  # 6
        "attention",         # 7  ← 标准
        "linear_attention",  # 8
        "linear_attention",  # 9
        "linear_attention",  # 10
        "attention",         # 11 ← 标准
        "linear_attention",  # 12
        "linear_attention",  # 13
        "linear_attention",  # 14
        "attention",         # 15 ← 标准
    ],
}
```

### 1.3 架构全图

```
Input IDs (vocab: 6400)
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│  Token Embedding (6400 → 768)                                  │
│  weight tied with lm_head                                      │
└──────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────────┐
  │  ×16 TransformerBlock (混合架构)                             │
  │                                                              │
  │  ┌── Layer 0,1,2,4,5,6,8,9,10,12,13,14: ──────────────┐   │
  │  │  GatedDeltaNet Linear Attention (75% 层)             │   │
  │  │                                                       │   │
  │  │  RMSNorm                                              │   │
  │  │    │                                                  │   │
  │  │    ▼                                                  │   │
  │  │  in_proj_qkv → Conv1d(因果) → SiLU                   │   │
  │  │    │                                                  │   │
  │  │    ├── Query (12 heads × 64 dim)  → l2norm           │   │
  │  │    ├── Key   (12 heads × 64 dim)  → l2norm           │   │
  │  │    └── Value (12 heads × 64 dim)                      │   │
  │  │                                                       │   │
  │  │  β = sigmoid(in_proj_b)    ← 写入门控                 │   │
  │  │  g = -exp(A_log) × softplus(a + dt_bias)  ← 衰减     │   │
  │  │                                                       │   │
  │  │  Chunked Gated Delta Attention:                       │   │
  │  │    S_t = exp(g_t) × S_{t-1} + k_t × (v_t × β_t)    │   │
  │  │    out_t = q_t × S_t                                  │   │
  │  │    → O(T × d²) 复杂度, 状态矩阵固定 O(d²)            │   │
  │  │                                                       │   │
  │  │  GateNorm(out, z) → SiLU gating                       │   │
  │  │  out_proj                                              │   │
  │  │  ✗ 不需要 RoPE (卷积感知局部位置)                      │   │
  │  └───────────────────────────────────────────────────────┘   │
  │                                                              │
  │  ┌── Layer 3,7,11,15: ─────────────────────────────────┐   │
  │  │  Standard Attention (25% 层, GQA 4:1)               │   │
  │  │                                                       │   │
  │  │  RMSNorm                                              │   │
  │  │    │                                                  │   │
  │  │    ├── wq: 768 → 768  (12 Q heads × 64 dim)         │   │
  │  │    ├── wk: 768 → 192  ( 3 KV heads × 64 dim)        │   │
  │  │    ├── wv: 768 → 192  ( 3 KV heads × 64 dim)        │   │
  │  │    └── wo: 768 → 768                                  │   │
  │  │                                                       │   │
  │  │  Q, K → RoPE (theta=10000)                           │   │
  │  │  Flash Attention / SDPA (causal)                     │   │
  │  │  → O(T²) 复杂度, KV Cache 随 T 线性增长              │   │
  │  └───────────────────────────────────────────────────────┘   │
  │                                                              │
  │  ┌── 所有层共享: ──────────────────────────────────────┐   │
  │  │  SwiGLU FFN                                          │   │
  │  │    RMSNorm                                            │   │
  │  │    w1: 768 → 2048 (gate)                              │   │
  │  │    w3: 768 → 2048 (up)                                │   │
  │  │    SiLU(gate) × up → w2: 2048 → 768                  │   │
  │  └───────────────────────────────────────────────────────┘   │
  └──────────────────────────┬────────────────────────────────────┘
                           │
              ┌────────────▼───────────┐
              │  Final RMSNorm          │
              │  lm_head (768 → 6400)   │
              │  tied with embedding    │
              └────────────────────────┘
```

### 1.4 参数量明细

| 组件 | 计算 | 参数量 |
|------|------|--------|
| Embedding (tied) | 6400 × 768 | **4.9M** |
| 每层 Standard Attn (GQA) | wq:768² + wk:768×192 + wv:768×192 + wo:768² | **1.47M** |
| 每层 GatedDeltaNet | qkv_proj + z_proj + a/b_proj + conv1d + out_proj | **2.97M** |
| 每层 SwiGLU FFN | 3 × 768 × 2048 | **4.72M** |
| 标准 Attn 层 × 4 | 4 × (1.47M + 4.72M) | **24.76M** |
| GatedDeltaNet 层 × 12 | 12 × (2.97M + 4.72M) | **92.28M** |
| Final RMSNorm | 768 | ~0 |
| **总计** | | **~122M** |

> 注：实际参数可能因 FFN hidden_dim 对齐到 multiple_of=256 而略有差异，最终约 120-130M。

---

## 2. 核心组件原理与设计理由

### 2.1 混合注意力架构

#### 原理

MindLM 在 16 层中混合使用两种注意力机制：

| 类型 | 层数 | 占比 | 复杂度 | KV Cache | 位置编码 |
|------|------|------|--------|----------|---------|
| GatedDeltaNet | 12 | 75% | O(T × d²) | O(d²) 固定 | 不需要 |
| Standard Attention | 4 | 25% | O(T² × d) | O(T × d) 增长 | RoPE |

```
序列中信息流动:

  Token 1  2  3  4  5  6  7  8 ... T
           │                    │
  ┌────────▼────────────────────▼──── GatedDeltaNet ──┐
  │  维护固定大小递归状态 S ∈ (d, d)                    │
  │  每步: S = g×S_old + k×(v×β),  out = q×S          │
  │  新信息写入，旧信息按衰减因子 g 遗忘                 │
  │  → 天然支持任意长度序列                              │
  └───────────────────────────────────────────────────┘

  ┌────────▼────────────────────▼──── Standard Attn ──┐
  │  全局 softmax 注意力                                │
  │  每个token 直接看到序列中所有其他token               │
  │  → 精确的 token-to-token 关系                       │
  └───────────────────────────────────────────────────┘

  交替使用: Linear层"压缩传递" → Standard层"精确校准"
  避免纯 Linear Attention 的精度损失累积
```

#### 设计理由

**为什么是 75/25 而不是 50/50？**

Qwen3.5 的验证：Qwen3.5 全系列（0.8B 到 27B）都使用 75/25 配比（每 4 层一个标准 Attention），且支持 256K 上下文。这证明了：

1. **75% 线性层足以保证长序列效率** — 只有 25% 的层受 O(T²) 约束
2. **25% 标准层足以保证精度** — 定期精确校准，防止信息在递归传递中累积偏差
3. **比 50/50 更激进但已被工业验证** — Qwen 团队的工程经验表明这个比例是最优权衡

**为什么不是 100% 线性？**

纯线性注意力在实践中精度不如混合架构。标准 Attention 层提供：
- 精确的 softmax 归一化（而非线性近似的无界求和）
- 精确的 token-to-token 对应关系
- 对关键位置（如首 token、最新 token）的准确关注

---

### 2.2 GatedDeltaNet 线性注意力

#### 原理

GatedDeltaNet 是一种**门控线性注意力**机制，核心思想是将标准 Attention 的 softmax(QK^T)V 替换为**带衰减的递归状态更新**：

```
标准 Attention:                    GatedDeltaNet:

scores = Q × Kᵀ                   # 递归状态: S ∈ (d_k, d_v) 固定大小
   → (T, T) O(T²) 显存
                                    for t = 1, 2, ..., T:
attn = softmax(scores)                # 1. 衰减旧记忆
   → 精确归一化                        S_t = exp(g_t) × S_{t-1}

output = attn × V                     # 2. 写入新信息 (β 门控写入强度)
   → 加权求和                          S_t = S_t + k_t × (v_t × β_t)

                                    # 3. 查询记忆
                                      o_t = q_t × S_t
```

关键参数：
- **β (beta)**: 写入门控，控制新信息写入递归状态的强度。`β = sigmoid(b_proj(x))`
- **g (衰减因子)**: 控制旧信息的遗忘速度。`g = -exp(A_log) × softplus(a + dt_bias)`，值域 (0, 1)
- **A_log**: 可学习的衰减参数，初始化为 `log(1..n_heads)`，不同头学习不同的衰减速率
- **dt_bias**: 时间步偏置，类似 S4/Mamba 的离散化参数

#### 分块并行计算

纯逐时间步的 Python 循环太慢。MindLM 采用 **Chunked Gated Delta Attention**：

```
将序列分成 chunk_size=64 的块:

┌──── Chunk 0 ────┐  ┌──── Chunk 1 ────┐  ┌──── Chunk 2 ────┐
│ 块内: 矩阵乘法   │  │ 块内: 矩阵乘法   │  │ 块内: 矩阵乘法   │
│ 并行计算 C×C    │  │ 并行计算 C×C    │  │ 并行计算 C×C    │
│                  │  │                  │  │                  │
│ 块间: 状态传递   │──│ 块间: 状态传递   │──│ 块间: 状态传递   │
│ S_0 → S_1       │  │ S_1 → S_2       │  │ S_2 → S_3       │
└──────────────────┘  └──────────────────┘  └──────────────────┘

Python 循环: T 次 → T/chunk_size 次 (如 8192 → 128 次)
块内: 全部用矩阵乘法，GPU 并行
```

块内计算使用 **log 空间累积和** 保证数值稳定：

```python
# g 已经是 log 空间（负值）
log_cg = torch.cumsum(log_g, dim=-1)       # log 空间累积衰减

# 块内注意力: log_ratio = log_cg[i] - log_cg[j] → g[j+1]*...*g[i]
log_ratio = log_cg.unsqueeze(-1) - log_cg.unsqueeze(-2)
decay_attn = torch.exp(log_ratio.clamp(max=0)) * causal_mask * (Q @ K^T)
```

#### 因果卷积

GatedDeltaNet 在 QKV 投影后加了一层**因果 1D 卷积**：

```
in_proj_qkv → Conv1d(kernel=4, groups=dim) → SiLU

作用:
  1. 让每个 token 能"看到"前 3 个 token 的局部上下文
  2. 替代位置编码 — 卷积天然感知局部相对位置
  3. 类似 Mamba 的 causal conv1d 设计
```

#### 门控 RMSNorm (GateNorm)

输出经过门控归一化：

```python
def gated_norm(x, gate):
    x = x / sqrt(mean(x²) + eps)    # RMSNorm
    x = x * SiLU(gate)              # 门控
    return x
```

门控信号 `z = in_proj_z(x)` 提供额外的选择性：哪些信息通道应该被激活。

#### 设计理由

**为什么选 GatedDeltaNet 而不是 Mamba/RetNet/RWKV？**

| 方案 | 优点 | 缺点 | 被主流采用 |
|------|------|------|-----------|
| **GatedDeltaNet** | 线性复杂度、支持 chunk 并行、门控灵活 | 实现复杂 | **Qwen3.5** |
| Mamba (S6) | 选择性状态空间、高效 | 确定性衰减，无门控 | 未被主流LLM采用 |
| RetNet |Retention机制简单 | 表达力不如门控 | 未被主流LLM采用 |
| RWKV | 线性RNN、推理快 | 纯RNN，训练并行难 | 未被主流LLM采用 |

**Qwen3.5 直接验证了 GatedDeltaNet 的选择** — 阿里云从 0.8B 到 27B 全系列采用，并支持 256K 上下文。这是目前唯一被主流大厂采用的线性注意力方案。

---

### 2.3 标准 Attention + GQA

#### 原理

标准 Multi-Head Attention 的 GQA (Grouped Query Attention) 变体：

```
MHA (MindLM 旧版):              GQA (MindLM-1B):

12 Q heads, 12 KV heads         12 Q heads, 3 KV heads
每 Q 独享 1 组 KV                每 4 个 Q 共享 1 组 KV

Q: [h0 h1 h2 h3 h4 h5          Q: [h0 h1 h2 h3 h4 h5
    h6 h7 h8 h9 h10 h11]            h6 h7 h8 h9 h10 h11]
K: [k0 k1 k2 k3 k4 k5          K: [k0 k0 k0 k0 k1 k1
    k6 k7 k8 k9 k10 k11]            k1 k1 k2 k2 k2 k2]
                                 ↑  4个Q共享1个K/V

KV Cache: 12 × 2 × d × T       KV Cache: 3 × 2 × d × T
                                → 减少 75% 的 KV Cache
```

#### 设计理由

**为什么用 GQA 4:1？**

| 参考 | 模型 | Q heads | KV heads | GQA比例 |
|------|------|---------|----------|---------|
| SmolLM2-135M | 135M | 9 | 3 | 3:1 |
| SmolLM3-3B | 3B | 16 | 4 | **4:1** |
| Qwen2.5-1.5B | 1.5B | 12 | 2 | 6:1 |
| Qwen3-2B | 2B | 16 | 4 | **4:1** |
| **MindLM-1B** | **~125M** | **12** | **3** | **4:1** |

4:1 是当前 1-3B 级别模型的主流选择（SmolLM3、Qwen3 都用这个比例）。
对于 MindLM-1B 的 8K 上下文：

```
KV Cache 大小 (fp16, T=8192):
  MHA: 4层 × 2(K+V) × 12头 × 64dim × 8192 × 2字节 = 96MB
  GQA: 4层 × 2(K+V) ×  3头 × 64dim × 8192 × 2字节 = 24MB  ← 75% 减少
```

**为什么不用更激进的 MQA (1 KV head)？**

MQA (所有 Q 共享 1 个 KV head) 过于极端，会降低模型表达力。GQA 4:1 在效率和精度间取得平衡，是当前业界共识。

---

### 2.4 SwiGLU FFN

#### 原理

SwiGLU (SwiSH-Gated Linear Unit) 是当前 LLM 的标准 FFN 选择：

```
传统 FFN:               SwiGLU FFN:

y = W2 × ReLU(W1 × x)  y = W2 × (SiLU(W1 × x) ⊙ W3 × x)
                             ↑ gate          ↑ up
参数: W1(d→4d) + W2(4d→d)  参数: W1(d→4d) + W2(4d→d) + W3(d→4d)
= 8d² 参数                 = 12d² 参数 (多 50%, 但效果更好)
```

MindLM 的 FFN hidden_dim 计算：
```python
hidden_dim = int(2 × 4 × dim / 3)  # dim=768 → 2048
# 然后对齐到 multiple_of=256
```

#### 设计理由

SwiGLU 已被所有主流模型采用（LLaMA、Qwen、Mistral、SmolLM），无需解释选择理由。

**为什么不用 MoE？**

| 对比 | 纯 FFN | MoE (4+1专家) |
|------|--------|---------------|
| 参数量/层 | ~4.72M | ~23.6M |
| 总参数 (16层) | ~125M | ~425M |
| 实现复杂度 | 简单 | 需要负载均衡、aux loss、dispatch逻辑 |
| 训练稳定性 | 稳定 | 需要调 expert 数量、top-k、aux_loss_alpha |
| 数据需求 | 适中 | 需要更多数据填满专家 |
| 推理效率 | 恒定 | 取决于 expert 路由 |

对于 ~125M 参数的小模型：
- MoE 会使参数膨胀到 ~425M，但数据量可能不足以训练充分
- 纯 FFN 更简单、更稳定、更容易调试
- SmolLM2 全系列也没有用 MoE，证明了小模型纯 FFN 的可行性

---

### 2.5 RoPE 旋转位置编码

#### 原理

RoPE (Rotary Position Embedding) 通过旋转向量来编码位置信息：

```python
# 预计算旋转角度
freqs = 1.0 / (theta ^ (2i/d))   # d 是 head_dim
pos_cis = exp(i × pos × freqs)    # 复数形式

# 应用到 Q 和 K
q_out = q ⊗ pos_cis               # 复数乘法 = 旋转
k_out = k ⊗ pos_cis
```

核心性质：**两个位置的内积只依赖相对位置差** → 天然的相对位置编码。

MindLM 使用预计算方式：在模型初始化时一次性计算好所有位置的 cos/sin 值，前向传播时直接查表。

#### 设计理由

**为什么标准层用 RoPE、线性层不用？**

- **标准层**: 需要精确的位置信息来区分不同位置 token 的注意力权重
- **线性层**: GatedDeltaNet 有因果卷积，天然感知局部相对位置；递归状态的衰减机制本身编码了时间距离。加 RoPE 反而可能干扰递归状态的学习

Qwen3.5 也采用了类似策略：线性层不使用 RoPE，且标准层只对 head_dim 的 25% 应用 RoPE（partial rotary）。

---

### 2.6 RMSNorm

#### 原理

RMSNorm 是 LayerNorm 的简化版，移除均值中心化，只做缩放：

```
LayerNorm:                    RMSNorm:

x' = (x - μ) / √(σ² + ε)    x' = x / √(mean(x²) + ε)
x' = x' × γ + β              x' = x' × γ
                              (无 bias, 无均值中心化)
```

#### 设计理由

所有主流模型（LLaMA、Qwen、Mistral、SmolLM）都使用 RMSNorm：
- 计算更快（省去均值计算和减法）
- 效果与 LayerNorm 相当
- 与 pre-norm（在每个子层前归一化）配合使用，训练更稳定

---

### 2.7 权重绑定 (Weight Tying)

#### 原理

```python
self.tok_embeddings.weight = self.output.weight
# Embedding 矩阵和 lm_head 矩阵共享同一份参数
```

#### 设计理由

| 模型 | 权重绑定 | 说明 |
|------|---------|------|
| SmolLM 全系列 | **tied** | 小模型标配，节省参数 |
| Qwen2.5 | **tied** | 1.5B 也用 |
| Qwen3 | **tied** | 2B 也用 |
| Qwen3.5 | **不 tied** | 0.8B 起就不 tied，可能是为了多模态 |
| MindLM-1B | **tied** | 6400 词表，Embedding 只有 4.9M |

小词表 (6400) 的 Embedding 只占 4.9M，tied 可以节省 4.9M 参数（~4%），且不影响性能。

---

### 2.8 BPE 分词器

#### 设计

| 配置 | 值 | 设计理由 |
|------|------|---------|
| 算法 | ByteLevel BPE | 天然支持多语言，无需未知字符处理 |
| 词表大小 | 6400 | 平衡覆盖率和模型效率 |
| 特殊 token | `<unk>` `<s>` `</s>` `<pad>` | 最小必要集合 |
| 训练数据 | 60万条中文对话 | 中英文混合 |

#### 设计理由

**为什么词表只有 6400？**

| 模型 | 词表大小 | 参数占比 |
|------|---------|---------|
| MindLM-1B | 6,400 | 3.9% (4.9M/125M) |
| SmolLM2 | 49,152 | ~20-30% |
| SmolLM3 | 128,256 | ~15% |
| Qwen2.5 | 151,936 | ~15% |
| Qwen3.5 | 248,320 | ~20% |

小词表的优势：
- Embedding 参数少（4.9M vs Qwen 的 150M+），参数更多花在模型深度上
- 每个 token 的训练数据更密集，学得更好
- 适合小模型实验和教学

劣势：
- 编码效率低（中文平均 5-8 个 token/字 vs Qwen 的 1-2 个）
- 限制了模型的语义理解粒度

**未来方向**：如果数据量充足，可以考虑升级到 32K 词表（如 LLaMA-3 的 32K），但需要更多训练数据。

---

## 3. 上下文长度分析

### 3.1 各层的上下文影响

```
T=8192 时:

标准 Attention 层 (4层, GQA 4:1):
  注意力矩阵: 4 × 8192² = 2.68 亿 → Flash Attention 处理，无压力
  KV Cache: 4 × 2 × 3 × 64 × 8192 × 2字节 = 24MB → 很小

GatedDeltaNet 层 (12层):
  计算量: 12 × O(T × d²) = 12 × 8192 × 64² ≈ 线性增长
  状态矩阵: 12 × O(d²) = 12 × 64² = 固定，与 T 无关
  → 天然支持任意长度
```

### 3.2 不同上下文的可行性

| 上下文 | 标准层 (4层) | 线性层 (12层) | 整体 | 备注 |
|--------|-------------|--------------|------|------|
| 512 | 极轻松 | 极轻松 | 完全可行 | 当前 |
| 2048 | 轻松 | 轻松 | 完全可行 | MindLM-0.5B 目标 |
| 4096 | 轻松 | 轻松 | 完全可行 | |
| **8192** | **可行** | **轻松** | **推荐上限** | **MindLM-1B 目标** |
| 16384 | 需滑动窗口 | 轻松 | 可行 | 标准层加 SWA |
| 32768 | 需滑动窗口 | 轻松 | 有挑战 | 参考 Qwen2.5 |
| 262144 | 不行 | 轻松 | 需要 Qwen3.5 级别优化 | |

### 3.3 扩展上下文的路径

```
Stage 1: max_seq_len=512,  ~5亿 tokens   → 打基础
Stage 2: max_seq_len=2048, ~3亿 tokens   → 4x 扩展
Stage 3: max_seq_len=8192, ~1.5亿 tokens → 4x 扩展到目标
```

如果要到 16K+：
1. 给标准层加 sliding window (如 window=4096)
2. 提高 GatedDeltaNet 层比例到 87.5% (14/16)
3. 参考 Qwen3.5 的 partial RoPE (只对 25% head_dim 应用)

---

## 4. 与其他模型的对比

### 4.1 架构对比

| 特性 | MindLM-1B | SmolLM2-135M | Qwen2.5-1.5B | Qwen3-2B | **Qwen3.5-2B** | Kimi K2.5 |
|------|-----------|-------------|--------------|---------|---------------|-----------|
| 参数量 | ~125M | 135M | 1.54B | ~2B | ~2B | 1T (32B激活) |
| 层数 | 16 | 30 | 28 | 36 | 32 | 61 |
| 隐藏维度 | 768 | 576 | 1536 | 1536 | 4096 | ~4096+ |
| Q heads | 12 | 9 | 12 | 16 | 16 | 64 |
| KV heads | 3 (GQA 4:1) | 3 (GQA 3:1) | 2 (GQA 6:1) | 4 (GQA 4:1) | 4 (GQA 4:1) | MLA |
| head_dim | 64 | 64 | 128 | 128 | 256 | ~128 |
| **线性注意力** | **75% GatedDeltaNet** | **无** | **无** | **无** | **75% GatedDeltaNet** | **无** |
| FFN | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| MoE | 无 | 无 | 无 | 无 | 无 | 384专家 |
| QK Norm | 无 | 无 | 无 | 有 | 有 | 有 |
| 词表 | 6,400 | 49,152 | 151,936 | 151,936 | 248,320 | 160,000 |
| 上下文 | 8,192 | 2,048 | 32,768 | 32,768 | **262,144** | 131,072 |
| 权重绑定 | tied | tied | tied | tied | **不 tied** | tied |

### 4.2 从各模型学到的设计经验

#### SmolLM2/3 — 小模型的工程实践

| 经验 | 详情 | MindLM 应用 |
|------|------|------------|
| GQA 是标配 | 135M 就用 GQA 3:1 | 采用 GQA 4:1 |
| 权重绑定 | 全系列 tied | 采用 tied |
| head_dim=64 | 135M 到 1.7B 都用 64 | 采用 head_dim=64 |
| no_rope_layer | 前4层不用 RoPE | 线性层不用 RoPE（更激进） |
| rope_theta=50000 | 配合长上下文 | 可在扩展上下文时参考 |
| 分阶段训练 | 短→长渐进扩展 | 采用同样策略 |

#### Qwen3 — QK Norm 的引入

| 经验 | 详情 | MindLM 应用 |
|------|------|------------|
| QK Norm | Q/K 后加 RMSNorm 稳定训练 | 可选：未来训练不稳定时加入 |
| 去 Attention bias | QK Norm 替代了 bias | 当前无 bias，方向一致 |
| GQA 4:1 | 2B 级别用 4:1 | 直接采用 |

#### Qwen3.5 — 混合注意力的工业验证

| 经验 | 详情 | MindLM 应用 |
|------|------|------------|
| **75/25 混合** | 每4层一个标准Attention | **直接采用相同配比** |
| **GatedDeltaNet** | 从 FLA 库引入 | **核心架构组件** |
| 因果卷积 | conv1d(kernel=4) | 采用 |
| GateNorm | 门控 RMSNorm | 采用 |
| Attention Gate | 标准层输出加 sigmoid gate | 可选改进方向 |
| Partial RoPE | 只对 25% head_dim 应用 | 可选改进方向 |
| RMSNorm(1+w) | 权重初始化为 0 | 可选改进方向 |

#### Kimi K2.5 — 超大规模 MoE 的经验

| 经验 | 详情 | MindLM 应用 |
|------|------|------------|
| MLA (KV压缩) | KV 压缩到低维 latent | GatedDeltaNet 的递归状态起到类似作用 |
| MoE 极端稀疏 | 384选8 (2.1%激活) | 小模型不适合，保持纯 FFN |
| 大词表 | 160K | 不适合 125M 模型 |

---

## 5. 可考虑的改进方向

### 5.1 短期（可直接加入）

#### QK Norm

```python
# 在标准 Attention 层的 Q/K 投影后加 RMSNorm
self.q_norm = RMSNorm(head_dim)
self.k_norm = RMSNorm(head_dim)

query = self.q_norm(self.wq(x))  # 新增
key = self.k_norm(self.wk(x))    # 新增
```

- **收益**: 稳定训练，防止注意力分数爆炸
- **代价**: 几乎无额外参数 (2 × head_dim)
- **参考**: Qwen3、Qwen3.5、Kimi K2.5 都使用

#### Attention Output Gate

```python
# Qwen3.5 的做法: q_proj 输出 2x 维度，拆出 gate
query, gate = self.wq(x).chunk(2, dim=-1)
# ... attention 计算 ...
output = output * torch.sigmoid(gate)  # 门控
```

- **收益**: 对 Attention 输出做选择性过滤
- **代价**: wq 参数翻倍
- **参考**: Qwen3.5 独有设计

#### Z-loss

```python
# SmolLM3 的做法: 对 logits 做正则化
z_loss = 1e-5 * log(sum(exp(logits))²)
total_loss = ce_loss + z_loss
```

- **收益**: 防止 logits 数值爆炸
- **代价**: 几乎无
- **参考**: SmolLM3

### 5.2 中期（需要数据支持）

#### 升级词表

从 6400 → 32000（LLaMA-3 风格）或 49152（SmolLM 风格）：
- 提升中文编码效率
- 需要重新训练 tokenizer 和模型
- 需要更多训练数据

#### Partial RoPE

```python
# 只对 head_dim 的一部分应用 RoPE
partial_rotary_factor = 0.5  # 50% of head_dim
```

- 参考 Qwen3.5 的 25% 设计
- 减少位置编码对注意力的干扰
- 配合线性注意力层使用效果更好

#### 滑动窗口注意力

给标准层加 sliding window：
```python
# 只看最近 W 个 token
window_size = 4096
```

- 使 16K-32K 上下文成为可能
- 参考 Qwen2.5/3 的设计

### 5.3 长期（架构级变化）

#### 提高 GatedDeltaNet 层比例

```
当前 75% (12/16):  可行上下文 8K
未来 87.5% (14/16): 可行上下文 16K-32K
未来 93.75% (15/16): 可行上下文 32K-64K
```

#### 使用 FLA 库的 CUDA kernel

Qwen3.5 使用了 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 库的优化 kernel：
- `fused_recurrent_gated_delta_rule`: 推理时的 fused kernel
- `chunk_gated_delta_rule`: 训练时的分块并行 kernel

这些 kernel 比纯 PyTorch 实现快 3-5 倍。

#### 引入多模态

参考 Qwen3.5 的 MoonViT 视觉编码器，将 MindLM 扩展为多模态模型。

---

## 6. 总结

MindLM-1B 的架构设计可以用三个关键词概括：

1. **混合** — 75% GatedDeltaNet + 25% 标准 Attention，兼顾效率与精度
2. **紧凑** — GQA 4:1、权重绑定、小词表，每一分参数都花在刀刃上
3. **可扩展** — 线性注意力天然支持长序列，架构为未来扩展留好空间

这个设计不是凭空想象，而是站在 Qwen3.5 的肩膀上——Qwen3.5 全系列（0.8B 到 27B）验证了 75/25 混合 GatedDeltaNet 架构的可行性，包括 256K 上下文。MindLM-1B 将同样的架构思想应用到 125M 参数级别，并保持实现简洁（无 MoE、无 QK Norm），适合研究和教学。
