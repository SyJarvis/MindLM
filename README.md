# MindLM

<div align="center">

<h2>MindLM: 混合架构语言模型</h2>
<p><b>Linear Attention + MoE 混合架构语言模型</b></p>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📋 项目简介

**MindLM** 是一个采用 **Linear Attention** 与 **混合架构** 设计的语言模型，旨在探索更高效的注意力机制与稀疏专家模型的结合。

### 核心特性

- ✅ **混合架构**: 标准 Attention (O(n²)) + Linear Attention (O(n)) 自由组合
- ✅ **稀疏专家 (MoE)**: Top-K 路由 + 共享专家机制
- ✅ **线性复杂度**: Gated DeltaNet 实现线性注意力，大幅降低长序列计算开销
- ✅ **灵活配置**: JSON 配置文件驱动，支持层级别自定义，按需组合不同注意力类型

---

## 🆚 MindLM 架构特点

| 特性 | **MindLM** |
|------|-----------|
| **注意力类型** | 混合：标准 Attention + Linear Attention |
| **计算复杂度** | O(n²) / O(n) 可选 |
| **长序列支持** | 优秀（Linear Attention） |
| **MoE支持** | Top-K 路由 + 共享专家 |
| **架构灵活性** | 层级别自定义 |
| **GQA** | Grouped Query Attention，降低 KV-Cache 开销 |
| **门控机制** | Gated DeltaNet |

### 架构示意

```
MindLM Layer (混合):
  Input → RMSNorm → Attention/LinearAttn → + → RMSNorm → MoE/FFN → + → Output
                  (根据配置自动选择)
```

---

## 🏗️ 模型架构

### 1. 核心组件

#### A. 混合注意力层

| 层类型 | 复杂度 | 适用场景 | 实现 |
|--------|--------|---------|------|
| `attention` | O(n²) | 短序列、高精度需求 | 标准 RoPE Attention |
| `linear_attention` | O(n) | 长序列、效率优先 | Gated DeltaNet |

**Gated DeltaNet 核心机制：**
```python
# 线性注意力通过门控机制更新状态
h_t = g_t ⊙ h_{t-1} + β_t ⊙ (k_t^T ⊗ v_t)  # 隐藏状态更新
o_t = q_t^T h_t                              # 输出

# 其中 g_t 为 decay gate，β_t 为 input weight
```

#### B. MoE (混合专家)

```
Top-K Router → Expert Selection → Parallel Computation → Weighted Sum
                    ↓
            ┌───────┼───────┐
           Exp0    Exp1    ExpN (路由专家)
            └───────┬───────┘
             Shared Expert (全局)
```

### 2. 架构配置

项目通过 JSON 配置文件定义模型架构，位于 `config/` 目录：

| 配置文件 | 参数量 | 隐藏维度 | 层数 | Linear Attn | 标准Attn | 序列长度 |
|---------|--------|---------|------|------------|---------|---------|
| `mindlm_0.5b.json` | 43.6M | 576 | 12 | 8 层 | 4 层 | 512 |
| `mindlm_1b.json` | 95.3M | 768 | 16 | 12 层 | 4 层 | 1024 |

以 `mindlm_1b.json` 为例：

```json
{
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "n_kv_heads": 3,
    "use_moe": false,
    "use_linear_attn": true,
    "max_seq_len": 1024,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention",
        "linear_attention", "linear_attention", "linear_attention", "attention"
    ]
}
```

---

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone https://github.com/SyJarvis/MindLM.git
cd MindLM

# 安装依赖
pip install -r requirements.txt

# 验证PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 基础训练

#### 1. MindLM-0.5B 训练（单卡 RTX 4090 即可）

```bash
python pretrain.py \
    --model_config mindlm_0.5b \
    --batch_size 64 \
    --epochs 5 \
    --accumulation_steps 8
```

- 参数量：43.6M（隐藏维度 576，12 层，9 头）
- 层类型：8 层 Linear Attention + 4 层标准 Attention
- 序列长度：512
- 训练数据：536 万条样本，每轮 83827 次迭代
- 硬件需求：单张 RTX 4090，batch_size=64

#### 2. MindLM-1B 训练

```bash
python pretrain.py \
    --model_config mindlm_1b \
    --batch_size 64 \
    --epochs 3 \
    --accumulation_steps 8
```

- 参数量：95.3M（隐藏维度 768，16 层，12 头）
- 层类型：12 层 Linear Attention + 4 层标准 Attention
- 序列长度：1024
- 训练数据：536 万条样本，每轮 33531 次迭代

#### 3. 分布式训练

```bash
# 单机多卡
torchrun --nproc_per_node=2 pretrain.py \
    --model_config mindlm_1b \
    --batch_size 80 \
    --accumulation_steps 2 \
    --epochs 3 \
    --ddp \
    --num_workers 8
```

### 模型推理

```bash
# 快速评估预训练模型
cd examples
python eval_pretrain.py
```

```python
# 自定义推理（精简版）
from modeling_mindlm import MindLM, MindLMConfig
from config import load_config
import torch
from transformers import AutoTokenizer

config = MindLMConfig(**load_config("mindlm_1b"))
model = MindLM(config)
model.load_state_dict(torch.load("out/mindlm_pretrain_768_linear_epoch0.pth", map_location="cpu", weights_only=True))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("mindlm_tokenizer", trust_remote_code=True)
input_ids = tokenizer.encode("人工智能的发展", return_tensors="pt")
with torch.no_grad():
    for out in model.generate(input_ids, eos=0, max_new_tokens=100):
        print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## ⚙️ 详细配置说明

### MindLMConfig 参数

```python
from modeling_mindlm import MindLMConfig

config = MindLMConfig(
    # ===== 基础架构 =====
    dim=768,                    # 隐藏层维度
    n_layers=16,                # 层数
    n_heads=12,                 # 注意力头数
    n_kv_heads=3,               # KV头数（GQA，降低KV-Cache开销）
    vocab_size=6400,            # 词表大小（会被tokenizer覆盖）
    max_seq_len=1024,           # 最大序列长度
    dropout=0.0,                # Dropout率
    norm_eps=1e-6,              # RMSNorm epsilon

    # ===== MoE配置 =====
    use_moe=False,              # 启用MoE（默认关闭）
    n_routed_experts=4,         # 路由专家数量
    num_experts_per_tok=2,      # 每token选择专家数
    n_shared_experts=1,         # 共享专家数
    scoring_func='softmax',     # 路由评分函数
    aux_loss_alpha=0.01,        # 辅助损失系数
    seq_aux=True,               # 序列级辅助损失
    norm_topk_prob=True,        # 归一化TopK概率

    # ===== Linear Attention配置 =====
    use_linear_attn=True,       # 启用Linear Attention
    layer_types=None,           # 层类型列表（None=自动交替）
    conv_kernel_size=4,         # 因果卷积核大小
)
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_config` | str | "mindlm_0.5b" | 模型配置名，从 config/ 目录加载 JSON |
| `--epochs` | int | 5 | 训练轮数 |
| `--batch_size` | int | 64 | 批次大小 |
| `--learning_rate` | float | 2e-4 | 学习率 |
| `--accumulation_steps` | int | 8 | 梯度累积步数 |
| `--grad_clip` | float | 1.0 | 梯度裁剪阈值 |
| `--warmup_iters` | int | 100 | Warmup步数 |
| `--data_path` | str | "data/pretrain_data.csv" | 训练数据路径 |
| `--tokenizer_path` | str | None | Tokenizer路径 |
| `--dtype` | str | "bfloat16" | 数据类型 |
| `--ddp` | bool | False | 启用分布式训练 |
| `--compile` | bool | False | 启用torch.compile优化 |
| `--use_wandb` | bool | False | 启用wandb记录 |

---

## 📁 项目结构

```
.
├── README.md                 # 本文件
├── config.py                 # 配置加载工具
├── config/                   # 模型配置文件
│   ├── mindlm_0.5b.json      # MindLM-0.5B 配置
│   └── mindlm_1b.json        # MindLM-1B 配置
├── modeling_mindlm.py        # MindLM模型实现
│   ├── MindLMConfig          # 配置类
│   ├── Attention             # 标准Attention
│   ├── GatedDeltaNet         # Linear Attention
│   ├── MOEFeedForward        # MoE层
│   ├── TransformerBlock      # Transformer块
│   └── MindLM                # 主模型
│
├── pretrain.py               # 预训练脚本
├── dataset.py                # 数据集
├── train_tokenizer.py        # Tokenizer训练脚本
│
├── mindlm_tokenizer/         # 自定义Tokenizer
├── data/                     # 训练数据目录
├── eval/                     # 模型推理
│   └── eval_pretrain.py      # 预训练模型评估/推理
├── examples/                 # 示例脚本
│   └── load_tokenizer.py      # 使用分词器
├── docs/                     # 技术文档
│
└── out/                      # 输出目录
    └── mindlm_pretrain_*.pth # 预训练权重
```

---

## 🔬 技术细节

### Gated DeltaNet 原理

**Delta Rule (增量规则)** 是线性注意力的核心，通过维护一个隐藏状态矩阵避免重复计算：

```
标准Attention:  O(n²·d)  时间和空间复杂度
Linear Attention: O(n·d²) 对于长序列更优
```

**关键组件：**
1. **Causal Conv1D**: 局部依赖建模
2. **Decay Gate**: 控制历史信息衰减
3. **Beta Gate**: 控制新信息输入
4. **L2 Norm**: Query/Key归一化

### MoE 负载均衡

```python
# 辅助损失 = α · Σ(fi · Pi)
# fi: 专家i的实际负载比例
# Pi: 专家i的路由概率比例
# α: 辅助损失系数 (默认0.01)

aux_loss = config.aux_loss_alpha * torch.sum(tokens_per_expert * router_prob_per_expert)
```

---

## 📈 训练建议

### 1. 超参数调优

| 场景 | 推荐配置 |
|------|---------|
| **快速验证** | `mindlm_0.5b`（43.6M，训练快） |
| **标准训练** | `mindlm_1b`（95.3M，效果好） |
| **长文本** | 修改配置中 `max_seq_len`，增加 Linear Attention 层比例 |

### 2. 学习率调度

```python
# Warmup + Cosine Decay
warmup_iters = 100
lr_decay_iters = total_steps
min_lr = init_lr / 10

# 学习率曲线:
# 0 -> warmup: 线性增加
# warmup -> end: Cosine衰减到min_lr
```

---

## 🧪 实验对比

### 序列长度扩展性

| 序列长度 | 标准 Attention (O(n²)) | Linear Attention (O(n)) |
|---------|----------------------|------------------------|
| 512 | 1x | ~0.9x |
| 2048 | 1x | ~1.2x |
| 8192 | 1x | ~2.5x |
| 32768 | OOM | 可处理 |

### 下游任务性能（待补充）

| 任务 | MindLM-混合 | MindLM-全Linear |
|------|------------|----------------|
| 文本续写 | - | - |
| 问答任务 | - | - |
| 长文本理解 | ✅ | ✅✅ |

---

## ⚠️ 注意事项

1. **预训练 vs 对话**: 预训练模型用于文本续写，做对话需要先进行 SFT 微调

---

## 🆚 与 MiniMind 架构对比

| 特性 | **MiniMind** | **MindLM** |
|------|-------------|-----------|
| **注意力机制** | 标准 Multi-Head Attention | 混合：标准 Attention + Gated DeltaNet Linear Attention |
| **计算复杂度** | O(n²) | O(n²) / O(n) 混合，长序列更高效 |
| **KV-Cache** | MHA，KV头数=Query头数 | GQA，KV头数=3，显存占用更低 |
| **长序列支持** | 有限（O(n²) 内存瓶颈） | 优秀（Linear Attention 层 O(n) 复杂度） |
| **架构灵活性** | 固定全 Attention | 层级别自定义，支持任意组合 |
| **位置编码** | RoPE | RoPE（标准层）+ Causal Conv1D（线性层） |
| **门控机制** | 无 | Gated DeltaNet（decay gate + beta gate） |
| **配置方式** | Python 代码 | JSON 配置文件，便于管理和复现 |

### 架构对比示意

```
MiniMind Layer:
  Input → RMSNorm → Attention (O(n²)) → + → RMSNorm → FFN → + → Output

MindLM Layer:
  Input → RMSNorm → Attention/LinearAttn → + → RMSNorm → FFN → + → Output
                  (根据 layer_types 配置自动选择)

  标准 Attention 层: RoPE + GQA → O(n²)
  Linear Attention 层: Causal Conv1D + Gated DeltaNet → O(n)
```

### 典型配置对比

```
MiniMind (8层):
  [Attention, Attention, Attention, Attention,
   Attention, Attention, Attention, Attention]

MindLM-0.5B (12层):
  [Linear, Linear, Attention, Linear, Linear, Attention,
   Linear, Linear, Attention, Linear, Linear, Attention]

MindLM-1B (16层):
  [Linear, Linear, Linear, Attention, Linear, Linear, Linear, Attention,
   Linear, Linear, Linear, Attention, Linear, Linear, Linear, Attention]
```

---

## 📝 引用

如果本项目对你有帮助，欢迎引用：

```bibtex
@software{mindlm2024,
  title = {MindLM: Hybrid Architecture Language Model},
  author = {Runke Zhong},
  year = {2026},
  url = {https://github.com/SyJarvis/MindLM}
}
```

---

## 🙏 致谢

- 训练数据集和 Tokenizer 来自 [MiniMind](https://github.com/jingyaogong/minimind)，感谢其开源贡献
- Linear Attention 参考 [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- Gated DeltaNet 参考 [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)

---

## 📄 License

MIT License

---

<div align="center">

**[⬆ 回到顶部](#mindlm)**

</div>
