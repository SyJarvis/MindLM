# MindLM v2 改进方案

## 概述

基于 MindLM v1（95.3M 参数）的训练经验，规划下一版本的架构改进。所有改进项在重新训练时一次性加入，不中途修改已训练模型。

---

## 改进清单

### 0. 标准注意力与线性注意力参数分离（最关键）

**问题**：v1 中两种注意力共享了 `n_kv_heads` 参数，导致 GatedDeltaNet 复用了标准 Attention 的 `n_kv_heads=3`，线性注意力层只工作在 192 维（3×64）而非完整的 768 维（12×64）。12 层线性注意力的表达力被严重压缩。

**设计原则**：两种注意力共享所有能共享的参数，只分离必须不同的参数。

**参数分离对照表**：

| 参数 | 共享/分离 | 标准注意力 | 线性注意力 | 说明 |
|------|----------|-----------|-----------|------|
| `dim` | **共享** | 768 | 768 | 隐藏维度，必须一致 |
| `n_heads` | **共享** | 12 | 12 | Q 头数，决定 head_dim=64 |
| `n_kv_heads` | **分离** | 3 (GQA) | — | 标准注意力独有，用于 GQA 压缩 |
| `linear_attn_heads` | **分离** | — | 12 | 线性注意力独有，默认等于 n_heads（不压缩） |
| `vocab_size` | **共享** | 6400 | 6400 | 词表 |
| `max_seq_len` | **共享** | 2048 | 2048 | 最大序列长度 |
| `norm_eps` | **共享** | 1e-6 | 1e-6 | RMSNorm epsilon |
| `dropout` | **共享** | 0.0 | 0.0 | Dropout 比率 |
| `hidden_dim` / `multiple_of` | **共享** | 自动/256 | 自动/256 | FFN 维度（两种注意力共享同一 FFN） |
| MoE 参数 | **共享** | — | — | 专家配置与注意力类型无关 |
| 位置编码 | **分离** | RoPE | 因果 Conv1d | 标准注意力用 RoPE，线性注意力用 conv_kernel_size |
| `rope_theta` | **分离** | 100000.0 | — | 仅标准注意力使用 |
| `rope_scaling_factor` | **分离** | 1.0 | — | 仅标准注意力使用 |
| `sliding_window` | **分离** | 0/4096 | — | 仅标准注意力使用（长上下文时） |
| `use_qk_norm` | **分离** | true | — | 仅标准注意力使用（RoPE 前归一化） |
| `conv_kernel_size` | **分离** | — | 4 | 仅线性注意力使用 |

**改法**：

```python
# MindLMConfig 新增参数（仅线性注意力专属）
linear_attn_heads: int = None  # None 时使用 n_heads（不压缩）

# GatedDeltaNet.__init__ 中 —— 使用独立参数
self.num_k_heads = getattr(args, 'linear_attn_heads', None) or args.n_heads  # 默认12
self.num_v_heads = getattr(args, 'linear_attn_heads', None) or args.n_heads  # 默认12

# Attention.__init__ 中 —— 继续使用 n_kv_heads（GQA）
self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
```

**效果**：

| 指标 | v1 (3 heads) | v2 (12 heads) |
|------|-------------|--------------|
| 每层 GatedDeltaNet 参数 | ~0.75M | ~2.97M |
| 12 层合计 | ~65.6M | ~92.3M |
| 总参数量 | ~95.3M | ~122M |
| 线性注意力表达力 | 192 维瓶颈 | 768 维完整 |

**额外参数**：+27M，但全部用于提升线性注意力层质量，值得。

---

### 1. 标准注意力层加 QK Norm

**问题**：v1 的 4 层标准 Attention 没有 QK 归一化，训练大规模或长序列时可能出现注意力分数爆炸。

**改法**（`modeling_mindlm.py` `Attention` 类）：

```python
# __init__ 中新增
self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)

# forward 中，RoPE 之前加
xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
xq = self.q_norm(xq)   # ← 新增
xk = self.k_norm(xk)   # ← 新增
if pos_cis is not None:
    xq, xk = apply_rotary_emb(xq, xk, pos_cis)
```

**额外参数**：4 层 × 2 个 RMSNorm × 64 维 = 512 个参数（可忽略）。

**参考**：Qwen3 标准做法。

---

### 2. RoPE 基础频率调大

**问题**：v1 使用 theta=10000，上下文扩展能力有限，需要 NTK-aware scaling 补偿。

**改法**：将默认 theta 从 10000 调到 100000 或 1000000。

```python
# MindLMConfig 默认值
rope_theta: float = 100000.0   # 原来 10000.0

# 或更激进（Qwen3 风格）
rope_theta: float = 1000000.0
```

**效果**：更大的 theta 使 RoPE 在更长序列上仍保持分辨能力，减少上下文扩展时的位置编码失真。

**配合**：仍然保留 `rope_scaling_factor` 参数，扩展时按需调整。

**参考**：Qwen3 使用 theta=1000000。

---

### 3. 预训练直接用 2K 上下文

**问题**：v1 默认 max_seq_len=1024，后续扩展需要额外阶段。

**改法**：预训练直接用 2048 上下文。

```json
// config/mindlm_1b.json
{
    "max_seq_len": 2048,
    "rope_theta": 100000.0
}
```

**显存影响**：1024→2048 序列翻倍，但 Linear Attention 层天然 O(T)，只有 4 层标准 Attention 是 O(T²)。加梯度检查点后 2×A800 完全够用。

---

### 4. 上下文扩展配置参数化

**问题**：v1 没有位置编码扩展参数，扩展时需要硬编码。

**改法**：在 `MindLMConfig` 中加入 RoPE 扩展参数。

```python
class MindLMConfig(PretrainedConfig):
    def __init__(
        self,
        # ... 原有参数 ...
        rope_theta: float = 100000.0,           # RoPE 基础频率
        rope_scaling_factor: float = 1.0,        # NTK-aware 缩放因子
        original_max_seq_len: int = None,        # 原始预训练序列长度
        # ...
    ):
```

并在 `precompute_pos_cis` 中支持 NTK-aware scaling（详见 `docs/rope_extension_design.md`）。

---

### 5. 滑动窗口 Attention

**问题**：扩展到 32K 上下文时，4 层标准 Attention 的 O(T²) 显存爆炸。

**改法**：在 `Attention` 类中加滑动窗口支持（详见 `docs/sliding_window_design.md`）。

```python
# MindLMConfig 新增
sliding_window: int = 0    # 0 = 不启用
```

32K 阶段设为 `sliding_window=4096`，只有标准 Attention 层受影响，12 层 Linear Attention 仍保持全局视野。

---

### 6. 完整 Checkpoint 保存

**问题**：v1 只保存 `model.state_dict()`，无法恢复训练状态（优化器、epoch、学习率位置）。

**改法**：保存完整训练状态 + JSON 元数据（已在 `sft.py` 中实现，需要回移到 `pretrain.py`）。

```python
checkpoint = {
    'model': state_dict,
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'epoch': epoch,
    'step': epoch * iter_per_epoch + step,
}
```

---

### 7. 训练脚本统一化

**问题**：`pretrain.py` 和 `sft.py` 大量代码重复。

**改法**：抽取公共训练逻辑为 `trainer.py`，两个脚本只负责参数解析和数据集选择。

```
MindLM/
├── trainer.py          # 公共训练逻辑（训练循环、保存、恢复、DDP）
├── pretrain.py         # 只负责：预训练参数 + PretrainDataset
├── sft.py              # 只负责：SFT 参数 + SFTDataset
└── train_vl.py         # 只负责：VL 参数 + VLDataset
```

---

## 改动优先级

| 优先级 | 改进项 | 改动量 | 影响范围 |
|--------|--------|--------|---------|
| **P0** | 标准注意力与线性注意力参数分离 | 小 | `modeling_mindlm.py` + config |
| P0 | 完整 Checkpoint 保存 | 小 | `pretrain.py` |
| P0 | 上下文扩展参数化 | 中 | `modeling_mindlm.py` + config |
| P1 | QK Norm | 小 | `modeling_mindlm.py` Attention |
| P1 | RoPE theta 调大 | 极小 | config JSON |
| P1 | 预训练 2K 上下文 | 极小 | config JSON |
| P2 | 滑动窗口 | 中 | `modeling_mindlm.py` Attention |
| P2 | 训练脚本统一化 | 大 | 重构 |

---

## v2 配置文件参考

### config/mindlm_1b_v2.json

```json
{
    "model_type": "mindlm",
    // ========== 共享参数（标准注意力 + 线性注意力共用） ==========
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "vocab_size": 6400,
    "max_seq_len": 2048,
    "dropout": 0.0,
    "norm_eps": 1e-6,
    "hidden_dim": null,
    "multiple_of": 256,
    "use_moe": false,
    "use_linear_attn": true,
    // ========== 标准注意力专属参数 ==========
    "n_kv_heads": 3,
    "rope_theta": 100000.0,
    "rope_scaling_factor": 1.0,
    "sliding_window": 0,
    "use_qk_norm": true,
    // ========== 线性注意力专属参数 ==========
    "linear_attn_heads": null,
    "conv_kernel_size": 4,
    // ========== 层类型配置 ==========
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention"
    ]
}
```

### 8K 扩展配置 config/mindlm_1b_v2_8k.json

```json
{
    "model_type": "mindlm",
    // ========== 共享参数 ==========
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "vocab_size": 6400,
    "max_seq_len": 8192,
    "dropout": 0.0,
    "norm_eps": 1e-6,
    "hidden_dim": null,
    "multiple_of": 256,
    "use_moe": false,
    "use_linear_attn": true,
    // ========== 标准注意力专属参数 ==========
    "n_kv_heads": 3,
    "rope_theta": 100000.0,
    "rope_scaling_factor": 4.0,
    "sliding_window": 0,
    "use_qk_norm": true,
    // ========== 线性注意力专属参数 ==========
    "linear_attn_heads": null,
    "conv_kernel_size": 4,
    // ========== 层类型配置 ==========
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention"
    ]
}
```

### 32K 扩展配置 config/mindlm_1b_v2_32k.json

```json
{
    "model_type": "mindlm",
    // ========== 共享参数 ==========
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "vocab_size": 6400,
    "max_seq_len": 32768,
    "dropout": 0.0,
    "norm_eps": 1e-6,
    "hidden_dim": null,
    "multiple_of": 256,
    "use_moe": false,
    "use_linear_attn": true,
    // ========== 标准注意力专属参数 ==========
    "n_kv_heads": 3,
    "rope_theta": 100000.0,
    "rope_scaling_factor": 16.0,
    "sliding_window": 4096,
    "use_qk_norm": true,
    // ========== 线性注意力专属参数 ==========
    "linear_attn_heads": null,
    "conv_kernel_size": 4,
    // ========== 层类型配置 ==========
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention"
    ]
}
```

---

## 预计参数量变化

| 组件 | v1 | v2 | 变化 |
|------|----|----|------|
| 线性注意力头数 3→12（参数分离） | 65.6M (12层) | 92.3M (12层) | **+26.7M** |
| QK Norm (4层) | 0 | +512 | 可忽略 |
| 其余不变 | 95.3M | ~122M | **+27M** |

v2 约 122M 参数，比 v1 多 27M，全部用于提升线性注意力层质量。
