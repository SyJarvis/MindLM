# 大模型训练显存与参数量计算指南

> 以 MindLM-1B (dim=768, n_layers=16) 为例，详解显存占用计算方法

---

## 1. 核心概念：显存占用构成

训练时显存 = **模型状态** + **激活值** + **其他开销**

```
┌─────────────────────────────────────────────────────────┐
│                    总显存占用 (~71GB)                    │
├─────────────────────────────────────────────────────────┤
│ 模型状态 (~1GB)                                          │
│   ├── 模型参数: 95M × 2B = 0.2GB                        │
│   ├── 梯度: 95M × 2B = 0.2GB                            │
│   └── 优化器状态(AdamW): 95M × 8B = 0.8GB               │
├─────────────────────────────────────────────────────────┤
│ 激活值 (~35GB) ← 最大头                                  │
│   ├── FFN中间值: ~16GB                                  │
│   ├── Attention缓存: ~8GB                               │
│   ├── GatedDeltaNet缓存: ~15GB                          │
│   └── 各层输入: ~2GB                                    │
├─────────────────────────────────────────────────────────┤
│ 其他开销 (~35GB)                                         │
│   ├── DDP同步缓存: ~6GB                                 │
│   ├── DataLoader(pin_memory): ~5GB                      │
│   └── PyTorch显存池: ~24GB                              │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Attention 显存计算

### 2.1 标准 Attention（O(T²) 复杂度）

**计算步骤显存**:
```
Q = X @ Wq          → (batch, n_heads, T, d_head)
K = X @ Wk          → (batch, n_kv_heads, T, d_head)
V = X @ Wv          → (batch, n_kv_heads, T, d_head)

Scores = Q @ K.T    → (batch, n_heads, T, T)  ← 显存杀手
Attention = softmax(Scores)
Output = Attention @ V  → (batch, n_heads, T, d_head)
```

**显存公式**:
```python
# Attention Scores 矩阵 (峰值显存)
attn_vram = batch × n_heads × T × T × 4 bytes  # fp32 softmax
         ≈ batch × T² × dim × 4 bytes          # 简化 (n_heads × d_head = dim)

# 示例: batch=80, T=1024, dim=768
= 80 × 1024 × 1024 × 768 × 4 / 1e9
= 256 GB? 不对，n_heads=12, d_head=64

# 修正: 分头计算
= 80 × 12 × 1024 × 1024 × 4 / 1e9
= 4.2 GB (单次)

# 反向传播需要保存: Q, K, V, Scores
= 4 × 4.2 GB ≈ 16 GB (4层标准Attention)
```

**关键规律**:
```
seq_len 翻倍 → Attention显存 ×4
batch 翻倍   → Attention显存 ×2
```

### 2.2 Linear Attention（O(T×d) 复杂度）

**GatedDeltaNet 原理**:
```
不需要存 T×T 矩阵!
只需维护固定大小的状态矩阵: S ∈ (d_k, d_v)

每步更新: S_t = g_t × S_{t-1} + k_t ⊗ (v_t × β_t)
输出: o_t = q_t × S_t
```

**显存公式**:
```python
# 状态矩阵 (固定大小，与T无关)
state_vram = n_heads × d_head × d_head × 4 bytes
           = 12 × 64 × 64 × 4 / 1e9
           = 0.2 MB (可忽略)

# 但训练时需要存每个token的q,k,v,g,beta
# 以及chunk并行时的中间结果
linear_vram ≈ batch × T × dim × n_layers × 8 bytes
            ≈ 80 × 1024 × 768 × 12 × 8 / 1e9
            ≈ 6 GB (12层Linear Attention)
```

**对比**:

| 上下文 | 标准Attention (4层) | LinearAttention (12层) | 总节省 |
|--------|---------------------|------------------------|--------|
| 1K     | ~8 GB              | ~6 GB                  | -2 GB  |
| 2K     | ~32 GB             | ~12 GB                 | -20 GB |
| 8K     | ~512 GB            | ~48 GB                 | -464 GB|

**结论**: Linear Attention 在长序列时优势巨大，但短序列时优势不明显。

---

## 3. FFN (Feed-Forward Network) 显存计算

### 3.1 SwiGLU 结构

```python
# SwiGLU FFN
gate = x @ W1    # (batch, T, hidden_dim)
up   = x @ W3    # (batch, T, hidden_dim)
hidden = SiLU(gate) * up  # (batch, T, hidden_dim)
out = hidden @ W2  # (batch, T, dim)
```

**参数量**:
```
W1: dim → hidden_dim
W3: dim → hidden_dim
W2: hidden_dim → dim

总参数量 = 3 × dim × hidden_dim
        = 3 × 768 × 2048
        = 4.7M / 层
```

**激活显存** (反向传播需要保存):
```python
# 需要保存: gate, up, hidden, 以及输入x
ffn_vram ≈ batch × T × hidden_dim × 3 × 2 bytes  # fp16
         = 80 × 1024 × 2048 × 3 × 2 / 1e9
         = 1 GB / 层

16层总计 = 16 GB
```

**关键规律**:
```
seq_len 翻倍 → FFN显存 ×2
hidden_dim 翻倍 → FFN显存 ×2
```

---

## 4. 模型参数量计算

### 4.1 完整公式

```python
def count_params(vocab_size, dim, n_layers, n_heads, n_kv_heads,
                 hidden_dim, use_moe=False):

    # 1. Embedding
    embed = vocab_size × dim

    # 2. 每层参数
    # 2.1 Standard Attention (GQA)
    if layer == "standard":
        # Wq: dim → n_heads × d_head
        # Wk: dim → n_kv_heads × d_head
        # Wv: dim → n_kv_heads × d_head
        # Wo: n_heads × d_head → dim
        attn = dim × n_heads × d_head + \
               dim × n_kv_heads × d_head × 2 + \
               n_heads × d_head × dim

    # 2.2 Linear Attention (GatedDeltaNet)
    else:
        # qkv_proj + z_gate + a/b_decay + out_proj + conv1d
        attn ≈ dim × (3 + 1 + 2) × d_head × n_heads + dim × dim

    # 2.3 FFN (SwiGLU)
    ffn = 3 × dim × hidden_dim

    # 单层总计
    per_layer = attn + ffn

    # 3. 总参数量
    total = embed + n_layers × per_layer + dim  # +dim是final norm

    return total
```

### 4.2 MindLM-1B 实例

| 组件 | 计算 | 参数量 |
|------|------|--------|
| Embedding | 6400 × 768 | 4.9M |
| 每层Standard(4层) | 4 × (2.4M + 4.7M) | 28.4M |
| 每层Linear(12层) | 12 × (1.8M + 4.7M) | 78M |
| Final Norm | 768 | ~0 |
| **总计** | | **~111M** |

> 实际显示 95M，差异来自权重绑定(Embedding=LM head)和计算细节。

---

## 5. 训练显存完整计算

### 5.1 理论公式

```python
def estimate_training_vram(batch, seq, dim, n_layers, n_heads, n_kv_heads,
                           vocab_size, n_standard_layers=4, dtype_bytes=2):
    """
    估算训练显存 (GB)
    """
    hidden_dim = int(2 * 4 * dim / 3)  # SwiGLU计算
    n_linear_layers = n_layers - n_standard_layers

    # 1. 模型状态 (参数 + 梯度 + 优化器)
    params = vocab_size * dim + n_layers * (3 * dim * hidden_dim + 4 * dim * dim)
    model_state = params * (dtype_bytes + dtype_bytes + 8) / 1e9  # +8是AdamW状态

    # 2. 激活值
    # 2.1 FFN激活
    ffn_act = batch * seq * hidden_dim * n_layers * 3 * dtype_bytes / 1e9

    # 2.2 Standard Attention激活
    d_head = dim // n_heads
    std_attn_act = n_standard_layers * batch * seq * seq * n_heads * 4 / 1e9

    # 2.3 Linear Attention激活
    linear_act = n_linear_layers * batch * seq * dim * 8 * dtype_bytes / 1e9

    # 2.4 其他激活(输入、norm等)
    other_act = batch * seq * dim * n_layers * 2 * dtype_bytes / 1e9

    activation = ffn_act + std_attn_act + linear_act + other_act

    # 3. DDP开销 (~20%)
    ddp_overhead = (model_state + activation) * 0.2

    # 4. PyTorch缓存 (~30%碎片)
    pytorch_cache = (model_state + activation + ddp_overhead) * 0.3

    total = model_state + activation + ddp_overhead + pytorch_cache

    return {
        'model_state': model_state,
        'activation': activation,
        'ddp_overhead': ddp_overhead,
        'pytorch_cache': pytorch_cache,
        'total': total
    }
```

### 5.2 实际案例对比

**配置**: batch=80, dim=768, n_layers=16, n_standard=4

| 上下文 | 激活值 | Attention | 其他 | 总计(理论) | 实际占用 | 误差 |
|--------|--------|-----------|------|-----------|----------|------|
| 512    | 8GB    | 2GB       | 5GB  | ~25GB     | ~45GB    | 缓存 |
| 1024   | 16GB   | 8GB       | 5GB  | ~40GB     | **~71GB**| 大 |
| 2048   | 32GB   | 32GB      | 5GB  | ~90GB     | **爆**   | - |

> 理论与实际差距主要来自 PyTorch 显存池策略和实现细节。

---

## 6. 参数配置影响关系

### 6.1 各参数对显存的影响系数

```
显存 ∝ batch_size¹ × seq_len^(1-2) × dim² × n_layers¹

具体:
- batch_size: 线性增长 (×2)
- seq_len:    线性~平方 (×2~×4)
- dim:        平方增长 (×4, 因为 hidden_dim 也翻倍)
- n_layers:   线性增长 (×2)
```

### 6.2 调参决策树

```
显存不够 (OOM)?
│
├─ 首先降低 batch_size (影响最小，速度换显存)
│   └─ 从 80 → 40 → 20
│
├─ 其次降低 seq_len (如果任务允许)
│   └─ 从 2048 → 1024 → 512
│
├─ 然后开启梯度检查点 (activation checkpointing)
│   └─ 激活值减半，速度-30%
│
├─ 给标准Attention层加滑动窗口
│   └─ window_size=4096，支持32K但总显存按4K算
│
└─ 最后考虑模型并行或流水线并行
    └─ 多卡分层，但通信开销大
```

### 6.3 典型配置速查表

**MindLM-1B (dim=768, n_layers=16)** on A800 80GB:

| 目标上下文 | batch_size | accumulation_steps | 有效batch | 预计显存 | 可行? |
|-----------|-----------|-------------------|----------|---------|------|
| 1K        | 80        | 2                 | 160      | ~70GB   | ✅   |
| 2K        | 40        | 4                 | 160      | ~75GB   | ✅   |
| 4K        | 20        | 8                 | 160      | ~78GB   | ⚠️   |
| 8K        | 10        | 16                | 160      | ~80GB   | ❌   |
| 8K+滑动窗  | 40        | 4                 | 160      | ~70GB   | ✅   |
| 32K+滑动窗 | 16        | 10                | 160      | ~75GB   | ✅   |

---

## 7. 实际优化技巧

### 7.1 立即降低显存的方法

```python
# 1. 梯度检查点 (换速度)
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self._forward_impl, x)

# 节省: ~40% 激活值显存
# 代价: 前向传播算两次，速度-20%~30%
```

```python
# 2. 混合精度训练 (已经默认)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss = model(input)

# fp16: 显存减半，但可能有精度问题
# bf16: 显存减半，数值稳定性更好
```

```python
# 3. 梯度累积 (保持有效batch)
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 显存占用 = 单步batch，效果 = 大batch
```

### 7.2 滑动窗口实现

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, n_heads, window_size=4096):
        self.window_size = window_size

    def forward(self, x):
        seq_len = x.shape[1]

        # 只计算局部窗口内的attention
        for i in range(seq_len):
            window_start = max(0, i - self.window_size)
            local_k = k[:, window_start:i+1, :]
            local_v = v[:, window_start:i+1, :]

            # attention 计算范围从 T×T 降为 T×W
```

---

## 8. 总结

### 关键公式速记

| 计算项 | 公式 | 复杂度 |
|--------|------|--------|
| **标准Attention** | batch × T² × dim × 4B | O(T²) |
| **LinearAttention** | batch × T × dim × n_layers × 8B | O(T) |
| **FFN激活** | batch × T × hidden_dim × 3 × 2B | O(T) |
| **模型参数** | vocab×dim + n_layers×(3×dim×hidden + 4×dim²) | O(n_layers × dim²) |

### 黄金法则

1. **seq_len 是最大杀手**: 翻倍 → Attention显存×4
2. **Linear Attention 在长序列救命**: 12层Linear + 4层Standard是8K+的最佳配比
3. **batch_size 是线性调节器**: 显存不够先砍batch，对效果影响最小
4. **PyTorch显存池**: 实际占用比理论多30-50%，预留余量

### MindLM-1B 配置建议

- **1K上下文**: batch=80, 无滑动窗, 正常训练
- **2K上下文**: batch=40, accumulation=4, 保持有效batch
- **8K上下文**: batch=20, 加滑动窗(window=4K), 4层标准Attention受限
- **32K上下文**: batch=8, 必须滑动窗, 考虑减少标准层到2层

---

> 文档版本: v1.0
> 适用模型: MindLM-1B (dim=768, n_layers=16)
> 硬件参考: NVIDIA A800 80GB × 2 (DDP)
