# LLM 模型架构分析对比

> MindLM vs Qwen2.5-1.5B vs Qwen3-2B vs SmolLM3-3B vs Kimi K2.5

---

## 1. Qwen2.5-1.5B

### 基本信息

| 参数 | 值 |
|------|------|
| 开发方 | 阿里云 |
| 参数量 | 1.54B (非嵌入 1.31B) |
| 层数 | 28 |
| 隐藏维度 | 1536 |
| 上下文长度 | 32,768 |
| 词表大小 | 151,936 |

### 架构图

```
Input IDs (vocab: 151936)
  │
  ▼
┌─────────────────────────────────┐
│  Embedding (151936 → 1536)      │
│  tie_word_embeddings=True       │
└─────────────┬───────────────────┘
              │
     ┌────────▼────────┐
     │  ×28 DecoderLayer│
     │                  │
     │  RMSNorm ──────► Attention (GQA)
     │    12 Q heads, 2 KV heads (6:1)
     │    QKV bias=True (Qwen独有)
     │    RoPE (theta=10000)
     │    │
     │    ├── q_proj (1536→1536, bias=True)
     │    ├── k_proj (1536→256,  bias=True)
     │    ├── v_proj (1536→256,  bias=True)
     │    └── o_proj (1536→1536, bias=False)
     │                  │
     │  RMSNorm ──────► MLP (SwiGLU)
     │    gate_proj (1536→8960, bias=False)
     │    up_proj   (1536→8960, bias=False)
     │    down_proj (8960→1536, bias=False)
     │    act: SiLU(gate) * up
     └────────┬─────────┘
              │
     ┌────────▼─────────┐
     │  Final RMSNorm    │
     │  lm_head (tied)   │
     └──────────────────┘
```

### 核心特征

- **GQA (Grouped Query Attention)**: 12个Q头共享2个KV头，比例为6:1，大幅减少KV cache显存占用
- **QKV bias**: Qwen2.5 在 Q/K/V 投影上都加了 bias，这是区别于 LLaMA 系列的标志性设计
- **权重绑定**: lm_head.weight 与 embed_tokens.weight 共享
- **SwiGLU FFN**: 使用门控线性单元，中间维度 8960（约 5.83× hidden_size）
- **RMSNorm pre-norm**: 每个 sublayer 前加 RMSNorm（eps=1e-6）
- **滑动窗口注意力**: 支持 sliding_window（默认关闭），可配置部分层使用

---

## 2. Qwen3-2B

### 基本信息

| 参数 | 值 |
|------|------|
| 开发方 | 阿里云 |
| 参数量 | ~2B |
| 层数 | 36 |
| 隐藏维度 | 1536 |
| head_dim | 128 |
| 上下文长度 | 32,768 |
| 词表大小 | 151,936 |

### 架构图

```
Input IDs (vocab: 151936)
  │
  ▼
┌─────────────────────────────────┐
│  Embedding (151936 → 1536)      │
│  tie_word_embeddings=True       │
└─────────────┬───────────────────┘
              │
     ┌────────▼────────┐
     │  ×36 DecoderLayer│
     │                  │
     │  RMSNorm ──────► Attention (GQA)
     │    16 Q heads, 4 KV heads (4:1)
     │    attention_bias=False  ← 去掉了 bias
     │    ┌──────────────────────┐
     │    │  q_proj → q_norm    │  ← 新增 QK Norm
     │    │  k_proj → k_norm    │  ← 新增 QK Norm
     │    │  v_proj              │
     │    │  RoPE → attn → o_proj│
     │    └──────────────────────┘
     │                  │
     │  RMSNorm ──────► MLP (SwiGLU)
     └────────┬─────────┘
              │
     ┌────────▼─────────┐
     │  Final RMSNorm    │
     │  lm_head (tied)   │
     └──────────────────┘
```

### 相比 Qwen2.5 的核心变化

| 改动 | Qwen2.5 | Qwen3 |
|------|---------|-------|
| **QK Norm** | 无 | **新增 RMSNorm** (q_norm, k_norm) |
| **Attention bias** | True | **False** |
| **Q头数** | 12 | 16 |
| **KV头数** | 2 | 4 |
| **GQA比例** | 6:1 | 4:1 |
| **层数** | 28 | 36 |

### 核心特征

- **QK Norm**: 在 Q/K 投影后各加一个 RMSNorm（作用于 head_dim），稳定大模型训练时的注意力分数，防止 attention logits 爆炸
- **去除 Attention bias**: 因为 QK Norm 已提供足够稳定性，不再需要 QKV bias
- **head_dim 显式配置**: 配置中有明确的 `head_dim: 128`，不再通过 hidden_size/heads 推导

---

## 3. SmolLM 系列

### 3.1 SmolLM2 架构（全部基于 LLaMA）

| 配置 | 135M | 360M | 1.7B |
|------|------|------|------|
| hidden_size | 576 | 960 | 2048 |
| layers | 30 | 32 | 24 |
| Q heads | 9 | 15 | 32 |
| KV heads | 3 (GQA 3:1) | 5 (GQA 3:1) | 32 (MHA) |
| FFN (intermediate) | 1,536 | 2,560 | 8,192 |
| 词表 | 49,152 | 49,152 | 49,152 |
| 上下文 | 2,048 | 2,048 | 2,048 |
| rope_theta | 10,000 | 10,000 | 10,000 |
| 基座 | LLaMA | LLaMA | LLaMA |
| tie_word_embeddings | True | True | True |
| rope_interleaved | False | False | **True** |

### 3.2 SmolLM3-3B（升级到 Qwen2 基座）

| 参数 | 值 |
|------|------|
| hidden_size | 2,048 |
| layers | 36 |
| Q heads | 16 |
| KV heads | 4 (GQA 4:1) |
| FFN (intermediate) | 11,008 |
| 词表 | 128,256 (Llama-3.2 tokenizer) |
| 上下文 | 4,096 → 64K (分阶段扩展) |
| rope_theta | **50,000** |
| 基座 | **Qwen2** |
| tie_word_embeddings | True |
| attention_bias | **False** |
| **no_rope_layer** | **4** |

### SmolLM3 关键设计

- **no_rope_layer=4**: 前4层不使用 RoPE 位置编码，让底层专注于局部模式学习
- **切换到 Qwen2 基座**: 从 LLaMA 切换到 Qwen2 架构（is_qwen2_config: true）
- **升级词表**: 从 49K 升到 128K（Llama-3.2 tokenizer），提升多语言和代码能力
- **rope_theta=50000**: 更大的 theta 值，配合长上下文扩展
- **分阶段训练**: Stage1(8T tokens) → Stage2(8-9T) → Stage3(9-11T)，然后 4K→32K→64K 长上下文扩展
- **Z-loss**: 可选的训练稳定技术（z_loss_coefficient: 1e-5）

### SmolLM 系列设计哲学

SmolLM 系列的核心是**小模型 + 大数据**的理念：
- 135M/360M 用 GQA（3:1 比例），1.7B 用 MHA，3B 回归 GQA（4:1）
- 权重始终 tied（embedding = lm_head），减少小模型参数浪费
- 训练数据量远超同规模模型（百万级 steps）

---

## 4. Kimi K2.5（~1T 参数 MoE）

### 基本信息

| 参数 | 值 |
|------|------|
| 开发方 | Moonshot AI (月之暗面) |
| 总参数量 | ~1.04T |
| 激活参数量 | ~32B (每 token) |
| 层数 | 61 |
| 词表大小 | 160,000 |
| 上下文长度 | 128,000 |
| 激活率 | ~3% (32B/1T) |

### 架构图

```
Input IDs (vocab: 160K)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  Embedding (160K → hidden)                           │
└──────────────┬───────────────────────────────────────┘
               │
      ┌────────▼──────────────────────────┐
      │  ×61 Layers                        │
      │                                    │
      │  RMSNorm                           │
      │  ──────► MLA                       │
      │           (Multi-head Latent       │
      │            Attention)              │
      │           64 attention heads       │
      │           KV 压缩到低维 latent     │
      │           无显式 KV cache          │
      │                                    │
      │  RMSNorm                           │
      │  ──────► MoE FFN                   │
      │           384 路由专家              │
      │           + 1 共享专家              │
      │           每次激活 8/384 = 2.1%    │
      │           Expert hidden: 2048      │
      │                                    │
      └────────┬──────────────────────────┘
               │
      ┌────────▼──────────┐
      │  Final RMSNorm     │
      │  lm_head (tied)    │
      └───────────────────┘

附加: MoonViT (400M 视觉编码器) ← K2.5 新增多模态能力
```

### 核心技术

#### MLA (Multi-head Latent Attention)

MLA 源自 DeepSeek 架构，核心思想是将 KV 压缩到低维 latent space：

```
标准 Attention:                    MLA:
K = k_proj(x)                     c_kv = kv_proj(x)    ← 压缩到 latent
V = v_proj(x)                     K = k_up(c_kv)       ← 从 latent 恢复
                                   V = v_up(c_kv)       ← 从 latent 恢复
KV Cache: 存储完整 K, V            Cache: 只存 c_kv (低维)
```

- 推理时只需缓存低维 latent 向量，大幅减少 KV cache 显存
- 64个注意力头，支持128K长上下文
- 本质目标与 MindLM 的 GatedDeltaNet 一致：减少推理显存占用

#### MoE (Mixture of Experts)

| 配置 | 值 |
|------|------|
| 总专家数 | 384 |
| 共享专家 | 1 |
| 每token激活 | 8 |
| Expert hidden | 2048 |
| 激活率 | 2.1% (8/384) |
| 动态路由 | 有 |

对比 Kimi K2 (256专家) → K2.5 (384专家)，专家数增加50%

#### Agent Swarm

K2.5 独有的 Agent Swarm 技术：
- 多个 agent 实例协作处理大规模任务
- 适用于复杂编程、推理场景
- 原生多模态（通过 MoonViT 视觉编码器）

---

## 5. MindLM

### 基本信息

| 参数 | 值 |
|------|------|
| 参数量 | ~30M |
| 层数 | 8 |
| 隐藏维度 | 512 |
| Q/KV heads | 8/8 (MHA) |
| 词表大小 | 6,400 |
| 上下文长度 | 512 |
| MoE专家 | 4路由 + 1共享, 每 token 激活 2 |
| 聚焦方向 | 高效混合注意力小模型 |

### 架构图

```
Input IDs (vocab: 6400)
  │
  ▼
┌─────────────────────────────────────────────────┐
│  Embedding (6400 → 512)                         │
│  weight tied with lm_head                       │
└──────────────┬──────────────────────────────────┘
               │
  ┌────────────▼────────────────────────────┐
  │  ×8 TransformerBlock (混合架构)          │
  │                                          │
  │  偶数层 (0,2,4,6):                       │
  │    RMSNorm → Standard Attention          │
  │              8 heads MHA, RoPE           │
  │    RMSNorm → MoE FFN (SwiGLU)           │
  │              4专家选2 + 1共享             │
  │                                          │
  │  奇数层 (1,3,5,7):                       │
  │    RMSNorm → GatedDeltaNet               │
  │              Linear Attention, 无RoPE     │
  │              因果卷积 + 递归状态           │
  │              l2norm(Q), l2norm(K)         │
  │              门控 RMSNorm                 │
  │    RMSNorm → MoE FFN (SwiGLU)           │
  │              4专家选2 + 1共享             │
  └────────────┬────────────────────────────┘
               │
  ┌────────────▼──────────┐
  │  Final RMSNorm         │
  │  lm_head (tied)        │
  └───────────────────────┘
```

### 核心设计

#### 混合注意力

| 特性 | Standard Attention | GatedDeltaNet (Linear) |
|------|--------------------|-----------------------|
| 复杂度 | O(n²) | O(n) (分块并行) |
| KV Cache | 需要缓存所有 K/V | 只需递归状态矩阵 (D×V) |
| 长序列推理 | 显存随长度线性增长 | 显存恒定 |
| 位置编码 | 需要 RoPE | 不需要（卷积感知局部位置） |
| 门控机制 | 无 | 有（gated RMSNorm + beta gate） |
| 衰减机制 | softmax 自然归一化 | 指数衰减门控 (exp(A_log)) |

#### MoE (Mixture of Experts)

- 4 个路由专家 + 1 个共享专家
- 每 token 激活 2 个路由专家（50% 激活率）
- 辅助损失 (aux_loss) 保证负载均衡
- 支持 softmax 评分 + top-k 选择

---

## 6. 五模型横向对比

### 基础参数

| 特性 | MindLM | SmolLM3-3B | Qwen2.5-1.5B | Qwen3-2B | Kimi K2.5 |
|------|--------|-----------|--------------|---------|-----------|
| 参数量 | ~30M | 3B | 1.54B | ~2B | 1T (32B激活) |
| 层数 | 8 | 36 | 28 | 36 | 61 |
| 隐藏维度 | 512 | 2048 | 1536 | 1536 | ~4096+ |
| 词表 | 6,400 | 128,256 | 151,936 | 151,936 | 160,000 |
| 上下文 | 512 | 64K | 32K | 32K | 128K |

### 注意力机制

| 特性 | MindLM | SmolLM3-3B | Qwen2.5-1.5B | Qwen3-2B | Kimi K2.5 |
|------|--------|-----------|--------------|---------|-----------|
| 注意力类型 | **混合** (MHA+Linear) | GQA (4:1) | GQA (6:1) | GQA (4:1) | **MLA** |
| Q头数 | 8 | 16 | 12 | 16 | 64 |
| KV头数 | 8 | 4 | 2 | 4 | latent压缩 |
| QKV bias | 无 | 无 | **有** | 无 | 无 |
| QK Norm | 无* | 无 | 无 | **有** | 有 |
| 位置编码 | RoPE (部分层无) | RoPE (前4层无) | RoPE | RoPE | RoPE |

> *MindLM 的 GatedDeltaNet 层用 l2norm 做 Q/K 归一化，功能类似 QK Norm

### FFN 与 MoE

| 特性 | MindLM | SmolLM3-3B | Qwen2.5-1.5B | Qwen3-2B | Kimi K2.5 |
|------|--------|-----------|--------------|---------|-----------|
| FFN类型 | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| MoE | **有** (4+1) | 无 | 无 | 无 | **有** (384+1) |
| 激活专家 | 2/4 (50%) | — | — | — | 8/384 (2.1%) |

### 其他

| 特性 | MindLM | SmolLM3-3B | Qwen2.5-1.5B | Qwen3-2B | Kimi K2.5 |
|------|--------|-----------|--------------|---------|-----------|
| Norm | RMSNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| 权重绑定 | tied | tied | tied | tied | tied |
| 多模态 | 无 | 无 | 无 | 无 | **MoonViT** |
| 滑动窗口 | 无 | 无 | 支持 | 支持 | — |

---

## 7. 对 MindLM 的启示

### 已有的设计验证

MindLM 的多项设计在主流大模型中得到验证：

1. **部分层不用位置编码** — SmolLM3 的 `no_rope_layer=4` 和 MindLM 的 Linear Attention 层不用 RoPE 是同一思路：底层更关注局部模式，不需要全局位置信息

2. **MoE 稀疏激活** — Kimi K2.5 用 384 专家 MoE，MindLM 用 4 专家 MoE，架构方向一致。小模型用高激活率（50%）合理，因为参数本来就不够

3. **KV 压缩** — Kimi K2.5 的 MLA 将 KV 压缩到 latent space，MindLM 的 GatedDeltaNet 用递归状态替代 KV cache，目标一致：减少推理显存

### 可借鉴的改进方向

1. **QK Norm** — Qwen3 和 Kimi K2.5 都在 Q/K 后加了 RMSNorm。MindLM 的标准 Attention 层可以考虑加上，提升训练稳定性

2. **GQA** — SmolLM3/Qwen3 都用 4:1 GQA。MindLM 当前是纯 MHA，引入 GQA 可以减少 KV cache，对推理更友好

3. **更大的 rope_theta** — SmolLM3 用 50000（vs 默认 10000），配合长上下文扩展。如果 MindLM 未来扩展上下文，可以参考

4. **Z-loss** — SmolLM3 可选的训练稳定技术，对 logits 做正则化防止数值爆炸，小模型训练也可以受益

5. **分阶段训练** — SmolLM3 的多阶段训练策略（短→长上下文）值得参考，先用 512 训练基础能力，再扩展到更长序列
