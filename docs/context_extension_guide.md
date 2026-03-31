# MindLM 分阶段预训练与上下文扩展指南

## 概述

分阶段提升上下文窗口是训练长上下文语言模型的标准做法。**所有阶段都属于预训练**，SFT（有监督微调）是最后一步。

```
Stage 1 (预训练): 2K 上下文 → 基础语言能力
    ↓
Stage 2 (预训练): 8K 上下文 → 扩展长文本理解
    ↓
Stage 3 (预训练): 32K 上下文 → 长文档/书籍理解（可选）
    ↓
SFT (有监督微调): 对话数据 → 学会对话格式
    ↓
DPO/RLHF: 人类偏好对齐（可选）
```

---

## 为什么分阶段预训练？

### 错误做法 ❌
只在 SFT 阶段扩展上下文：
- 预训练权重只"见过" 2K 长度
- 突然给 32K 的对话，模型无法处理长距离信息
- 效果差：遗忘、幻觉、注意力崩溃

### 正确做法 ✅
预训练阶段逐步扩展：
- 模型先学会"短程依赖"（2K）
- 再适应"中程依赖"（8K）
- 最后掌握"长程依赖"（32K+）
- SFT 时上下文长度与预训练最终阶段保持一致

### 参考实践

| 模型 | 预训练扩展路径 | SFT 上下文 |
|------|--------------|-----------|
| Qwen2.5 | 4K → 32K（渐进） | 32K |
| LLaMA-3 | 8K → 128K（预训练后段） | 128K |
| MiniMind | 512 → 实际数据长度 | 同预训练 |
| **MindLM-1B** | **2K → 8K → (可选 32K)** | **8K 或 32K** |

---

## 分阶段配置详解

### Stage 1: 基础预训练（2K 上下文）

```bash
python pretrain.py \
    --model_config mindlm_1b \
    --epochs 5 \
    --learning_rate 2e-4 \
    --batch_size 64 \
    --accumulation_steps 8
```

| 参数 | 值 | 说明 |
|------|-----|------|
| max_seq_len | 2048 | JSON 配置文件中设置 |
| 训练轮数 | 3-5 轮 | 从随机初始化学基础 |
| 学习率 | 2e-4 | 较高，快速收敛 |
| 数据 | pretrain_data.csv | 完整数据集 |
| 样本处理 | text[:2048] | 截断到 2K |

**输出**: `mindlm_pretrain_768_epoch4.pth`

---

### Stage 2: 上下文扩展（8K）

```bash
python pretrain.py \
    --resume_from out/mindlm_pretrain_768_epoch4.pth \
    --model_config mindlm_1b \
    --max_seq_len 8192 \
    --epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --accumulation_steps 4
```

| 参数 | 值 | 说明 |
|------|-----|------|
| max_seq_len | 8192 | 覆盖 JSON 配置 |
| 训练轮数 | 1-2 轮 | 权重已有基础，只需适应 |
| 学习率 | 1e-4 | 减半，微调模式 |
| 数据 | pretrain_data.csv | **同一数据集** |
| 样本处理 | text[:8192] | 更长截断 |
| 续训 | --resume_from | 加载 Stage 1 权重 |

**关键**: 同一数据，但每条样本提供更多 token（8K vs 2K）

---

### Stage 3: 长上下文（32K，可选）

```bash
python pretrain.py \
    --resume_from out/mindlm_pretrain_768_stage2.pth \
    --model_config mindlm_1b \
    --max_seq_len 32768 \
    --sliding_window 4096 \
    --epochs 1 \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --accumulation_steps 2
```

| 参数 | 值 | 说明 |
|------|-----|------|
| max_seq_len | 32768 | 长文档级别 |
| sliding_window | 4096 | **滑动窗口注意力** |
| 训练轮数 | 0.5-1 轮 | 避免过拟合 |
| 学习率 | 5e-5 | 更低，精细调整 |
| 数据 | 长文档/书籍 | 可能需额外收集 |
| 样本处理 | text[:32768] | 完整长文档 |

**滑动窗口**: 标准 Attention 层只看最近 4096 token，降低显存压力

---

## 数据集处理策略

### 各阶段数据对比

| 阶段 | 平均序列长度 | 单条样本 token | 有效样本数 | 总 token 数 |
|------|-------------|---------------|-----------|------------|
| Stage 1 (2K) | 1K | 2K | ~500万 | ~50亿 |
| Stage 2 (8K) | 4K | 8K | ~125万 | ~50亿 |
| Stage 3 (32K) | 16K | 32K | ~30万 | ~50亿 |

**注意**: 虽然"样本数"变少了，但总 token 量大致相同。

### 数据准备代码示例

```python
# 同一数据源，不同截断策略
def prepare_data(text, stage="stage1"):
    if stage == "stage1":
        return text[:2048]
    elif stage == "stage2":
        return text[:8192]
    elif stage == "stage3":
        # 优先选择长文档
        if len(text) < 8192:
            return None  # 跳过太短文本
        return text[:32768]
```

### Stage 3 长文档来源

- **书籍**: Gutenberg, 中文网络小说
- **论文**: arXiv, 学术论文
- **代码**: GitHub 完整项目文件
- **百科**: Wikipedia 长词条

---

## 续训关键参数

### 必须设置

```python
# 1. 加载已有权重
--resume_from out/mindlm_pretrain_768_epoch4.pth

# 2. 更大的上下文（覆盖 JSON 配置）
--max_seq_len 8192  # 或 32768

# 3. 更低的学习率
--learning_rate 1e-4  # Stage 2
--learning_rate 5e-5  # Stage 3

# 4. 减少训练轮数
--epochs 2  # Stage 2
--epochs 1  # Stage 3
```

### 显存调整

```python
# Stage 2 (8K): batch_size 减半
--batch_size 32  # 原来是 64
--accumulation_steps 4  # 保持有效 batch

# Stage 3 (32K): batch_size 再减半
--batch_size 16
--accumulation_steps 2
```

---

## 滑动窗口注意力实现

### 原理

标准 Attention 计算复杂度 **O(T²)**，32K 时内存爆炸。

滑动窗口限制：只看最近的 `W` 个 token（如 4K），复杂度降为 **O(T×W)**。

```
正常 Attention:     滑动窗口 Attention:
Q × K^T (32K×32K)   Q × K^T (32K×4K)
   ↓                      ↓
 显存爆炸              可控
```

### 代码改动（modeling_mindlm.py）

```python
class Attention(nn.Module):
    def __init__(self, args: MindLMConfig):
        super().__init__()
        ...
        self.sliding_window = args.sliding_window  # 新增参数

    def forward(self, x):
        if self.sliding_window > 0 and seq_len > self.sliding_window:
            # 创建局部 attention mask
            mask = self.create_sliding_window_mask(seq_len, self.sliding_window)
            # 只计算窗口内的 attention
```

### MindLM-1B 的特殊性

你的模型只有 **4 层标准 Attention**，其余 12 层是 Linear Attention。

```
总层数: 16
标准 Attention (需滑动窗口): 4 层
Linear Attention (天然长上下文): 12 层
```

**建议**:
- 8K 阶段：**不需要**滑动窗口（4 层 8K 显存吃得住）
- 32K 阶段：给那 4 层标准 Attention 加滑动窗口

---

## 实践建议

### 推荐路径

1. **现在**: 跑完 Stage 1 (2K, 5轮)
2. **第二阶段**: 扩展到 8K (2轮，学习率 1e-4)
3. **评估**: 测试 8K 长文本理解能力
4. **决定**: 是否需要 Stage 3 (32K)

### 检查点管理

```
out/
├── mindlm_pretrain_768.pth              # Stage 1 快照
├── mindlm_pretrain_768_epoch4.pth       # Stage 1 最终
├── mindlm_pretrain_768_stage2.pth       # Stage 2 最终
└── mindlm_pretrain_768_stage3.pth       # Stage 3 最终 (可选)
```

### 关键指标监控

| 阶段 | 关注指标 | 正常范围 |
|------|---------|---------|
| Stage 1 | loss 下降 | 9 → 3 |
| Stage 2 | 长程依赖 | PPL 不爆炸 |
| Stage 3 | 上下文利用 | 首尾信息不遗忘 |

---

## 常见问题

### Q: 为什么 Stage 2 轮数更少？

**A**: 权重已有基础，只需"适应"更长上下文，类似微调。多轮容易过拟合。

### Q: 可以用不同数据吗？

**A**: 可以，但建议保持领域一致。Stage 3 必须补充长文档。

### Q: 显存不够怎么办？

**A**:
- 减小 batch_size，增大 accumulation_steps
- 开滑动窗口（sliding_window=4096）
- 使用梯度检查点（gradient checkpointing）

### Q: 如何验证长上下文效果？

**A**:
- "大海捞针"测试：长文本中藏关键信息，看能否召回
- 长文档摘要：输入 8K/32K 文档，检查输出准确性
- 代码补全：跨文件依赖理解

---

## 参考资源

- **Qwen2.5 技术报告**: 长上下文扩展策略
- **LLaMA-3 论文**: 预训练后段扩展上下文
- **MiniMind 源码**: 分阶段训练实现
- **MindLM 设计文档**: `docs/mindlm_1b_architecture_design.md`
