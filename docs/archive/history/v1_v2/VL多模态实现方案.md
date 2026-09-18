# MindLM-VL 多模态实现方案

## 概述

给 MindLM 加上视觉理解能力，采用 VL（Vision-Language）路线。用预训练好的 CLIP-ViT-Base-Patch32 作为视觉编码器，通过投影层连接到 MindLM。

---

## 架构设计

```
输入图像 (224×224)
    ↓
CLIP-ViT-Base-Patch32 (冻结)
    ↓
50 个视觉 token (每个 768 维)     ← hidden_size=768
    ↓
投影层 (MLP, 768→768)             ← 维度完全匹配！
    ↓
[视觉token] + [文本token] 拼接
    ↓
MindLM Transformer (16层)
    ↓
文本输出（描述图像内容、回答视觉问题）
```

### 关键参数匹配

| 组件 | 维度 | 说明 |
|------|------|------|
| CLIP-ViT-Base-Patch32 hidden_size | **768** | 视觉编码器输出维度 |
| MindLM-1B dim | **768** | 语言模型隐藏维度 |
| 视觉 token 数量 | **50** | 49个patch + 1个[CLS]，224/32=7, 7×7=49 |

**维度完全匹配！** 投影层可以非常简单（甚至可以是恒等映射）。

---

## 整体训练流程

```
阶段 1: 视觉-语言对齐（只训投影层）
  - 冻结 ViT + 冻结 MindLM
  - 只训练投影层 MLP
  - 数据：图文对（图片 + 描述）
  - 目标：让投影层学会把视觉特征"翻译"成语言模型的输入
  - 轮数：1-2 轮
  - 学习率：1e-3（只训小模块，可以高一些）

阶段 2: 端到端微调（解冻 MindLM）
  - 冻结 ViT（不动）
  - 解冻 MindLM + 投影层
  - 数据：图文对话（VQA 数据）
  - 目标：让 MindLM 学会理解视觉信息并回答问题
  - 轮数：1-2 轮
  - 学习率：5e-5（跟 SFT 一样）
```

---

## 需要新建/修改的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `modeling_mindlm_vl.py` | 新建 | VL 模型类，封装 ViT + 投影层 + MindLM |
| `dataset_vl.py` | 新建 | 图文数据集处理 |
| `train_vl.py` | 新建 | VL 两阶段训练脚本 |
| `inference_vl.py` | 新建 | VL 推理脚本（输入图片+问题，输出回答） |
| `modeling_mindlm.py` | 不改 | MindLM 原始模型不动 |
| `config/mindlm_1b_vl.json` | 新建 | VL 配置文件 |

---

## 模型代码设计

### MindVLM 类

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from modeling_mindlm import MindLM, MindLMConfig


class VisionProjection(nn.Module):
    """视觉特征投影层：把 ViT 输出映射到语言模型空间"""

    def __init__(self, vision_hidden_size=768, lm_hidden_size=768):
        super().__init__()
        # 因为维度匹配(768==768)，用两层 MLP 足够
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, lm_hidden_size),
            nn.GELU(),
            nn.Linear(lm_hidden_size, lm_hidden_size),
        )

    def forward(self, vision_features):
        # vision_features: (batch, num_patches, vision_hidden_size)
        return self.proj(vision_features)


class MindVLM(nn.Module):
    """MindLM Vision-Language 模型"""

    def __init__(self, lm_config: MindLMConfig, vision_model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        # 1. 视觉编码器（CLIP ViT，预训练好）
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)

        # 2. 视觉投影层
        self.vision_proj = VisionProjection(
            vision_hidden_size=self.vision_model.config.hidden_size,  # 768
            lm_hidden_size=lm_config.dim,                             # 768
        )

        # 3. 语言模型（MindLM）
        self.lm = MindLM(lm_config)

        # 4. 特殊 token 的 embedding
        # 添加一个 <image> 占位符 token
        self.image_token_id = None  # 初始化时设置

    def encode_image(self, pixel_values):
        """用 ViT 编码图像，返回视觉 token"""
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # 取最后一层所有 patch token（不含 [CLS]）
        # shape: (batch, 49, 768)  或 (batch, 50, 768)
        image_features = vision_outputs.last_hidden_state
        # 投影到语言模型空间
        vision_tokens = self.vision_proj(image_features)
        return vision_tokens  # (batch, num_patches, lm_dim)

    def forward(self, input_ids, labels=None, pixel_values=None, image_mask=None):
        """
        前向传播

        Args:
            input_ids: 文本 token ids，包含 <image> 占位符
            labels: 目标 token ids
            pixel_values: 预处理后的图像张量 (batch, 3, 224, 224)
            image_mask: 标记哪些位置是 <image> 占位符
        """
        if pixel_values is not None:
            # 编码图像
            vision_tokens = self.encode_image(pixel_values)  # (B, 50, 768)

            # 获取文本 embedding
            text_embeddings = self.lm.tok_embeddings(input_ids)  # (B, T, 768)

            # 把 <image> 占位符位置替换为视觉 token
            # image_mask: (B, T), 1 表示该位置是 <image> 占位符
            batch_size, seq_len, dim = text_embeddings.shape
            num_vision_tokens = vision_tokens.shape[1]

            # 方案：把所有 <image> 占位符合并为视觉 token
            # 构建 new_embeddings: 用视觉 token 替换 <image> 位置
            new_embeddings = self._merge_vision_text(
                text_embeddings, vision_tokens, image_mask
            )

            # 直接调用 MindLM 的 transformer 层
            # 跳过 tok_embeddings（已经手动做了）
            h = new_embeddings
            h = self.lm.dropout(h)

            pos_cis = None
            if self.lm.pos_cis is not None:
                pos_cis = self.lm.pos_cis[:h.shape[1]]

            total_aux_loss = 0.0
            for layer in self.lm.layers:
                h, aux_loss = layer(h, pos_cis, False)
                if aux_loss is not None:
                    total_aux_loss += aux_loss

            h = self.lm.norm(h)
            logits = self.lm.output(h)

            loss = None
            if labels is not None:
                pad_id = getattr(self.lm.config, 'pad_token_id', None)
                if pad_id is None:
                    pad_id = self.lm.config.vocab_size - 1
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=pad_id
                )
                if total_aux_loss > 0:
                    loss = loss + total_aux_loss

            return CausalLMOutputWithPast(
                loss=loss, logits=logits,
                past_key_values=None, hidden_states=None, attentions=None
            )
        else:
            # 纯文本模式，走原始 MindLM
            return self.lm(input_ids, targets=labels)

    def _merge_vision_text(self, text_embeddings, vision_tokens, image_mask):
        """将视觉 token 融合到文本 embedding 中

        把 <image> 占位符替换为实际的视觉 token。

        例如：
        文本:  "请描述 <image><image>...<image> 这张图片"
        替换后: "请描述 [vis1][vis2]...[vis50] 这张图片"
        """
        batch_size = text_embeddings.shape[0]
        results = []

        for b in range(batch_size):
            mask = image_mask[b]  # (T,)
            text_emb = text_embeddings[b]  # (T, 768)
            vis_tok = vision_tokens[b]     # (50, 768)

            # 找到 <image> 占位符的起始位置
            image_positions = mask.nonzero(as_tuple=True)[0]

            if len(image_positions) == 0:
                # 没有 <image>，直接返回文本
                results.append(text_emb)
                continue

            # 第一个 <image> 的位置
            image_start = image_positions[0].item()
            # <image> 占位符的数量（应该等于 num_vision_tokens）
            num_placeholders = len(image_positions)

            # 构建新的 embedding 序列
            # [image之前的文本] + [视觉token] + [image之后的文本]
            before = text_emb[:image_start]                          # <image> 之前
            visual = vis_tok[:num_placeholders]                      # 视觉 token
            after = text_emb[image_start + num_placeholders:]        # <image> 之后

            merged = torch.cat([before, visual, after], dim=0)
            results.append(merged)

        # padding 到同一长度
        max_len = max(r.shape[0] for r in results)
        padded = []
        for r in results:
            if r.shape[0] < max_len:
                pad = torch.zeros(max_len - r.shape[0], r.shape[1], device=r.device)
                padded.append(torch.cat([r, pad], dim=0))
            else:
                padded.append(r)

        return torch.stack(padded)  # (B, new_T, 768)
```

### 模型初始化和冻结

```python
def create_vl_model(lm_checkpoint_path, device="cuda"):
    """创建 VL 模型"""
    # 加载 MindLM 配置和权重
    config_dict = load_config("mindlm_1b")
    config = MindLMConfig(**config_dict)

    model = MindLMVL(config)

    # 加载预训练的 MindLM 权重
    lm_state_dict = torch.load(lm_checkpoint_path, map_location=device)
    model.lm.load_state_dict(lm_state_dict, strict=False)

    # 添加 <image> token 到 tokenizer
    model.lm.config.vocab_size += 1  # 6400 → 6401
    # 扩展 embedding 层
    old_emb = model.lm.tok_embeddings
    new_emb = nn.Embedding(model.lm.config.vocab_size, config.dim)
    new_emb.weight[:config.vocab_size - 1] = old_emb.weight
    model.lm.tok_embeddings = new_emb
    # 同样扩展 output 层
    old_output = model.lm.output
    new_output = nn.Linear(config.dim, model.lm.config.vocab_size, bias=False)
    new_output.weight[:config.vocab_size - 1] = old_output.weight
    model.lm.output = new_output
    # tied weights
    model.lm.tok_embeddings.weight = model.lm.output.weight
    # <image> token id
    model.image_token_id = config.vocab_size - 1  # 6400

    return model.to(device)


def freeze_for_alignment(model):
    """阶段1：只训练投影层，冻结其余"""
    # 冻结视觉编码器
    for param in model.vision_model.parameters():
        param.requires_grad = False
    # 冻结语言模型
    for param in model.lm.parameters():
        param.requires_grad = False
    # 只保留投影层可训练
    for param in model.vision_proj.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable/1e6:.2f}M / 总参数: {total/1e6:.2f}M")


def freeze_for_finetune(model):
    """阶段2：解冻 MindLM，保持 ViT 冻结"""
    # ViT 仍然冻结
    for param in model.vision_model.parameters():
        param.requires_grad = False
    # 解冻 MindLM
    for param in model.lm.parameters():
        param.requires_grad = True
    # 投影层也继续训练
    for param in model.vision_proj.parameters():
        param.requires_grad = True
```

---

## 图文数据集处理

### 数据格式

CSV 文件，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `image_path` | str | 图片文件路径 |
| `conversations` | str | 对话内容 JSON |

示例：

```csv
image_path,conversations
data/images/001.jpg,"[{""role"":""user"",""content"":""<image>\n描述这张图片""},{""role"":""assistant"",""content"":""一只橘色的猫趴在窗台上晒太阳""}]"
data/images/002.jpg,"[{""role"":""user"",""content"":""<image>\n图片中有多少人？""},{""role"":""assistant"",""content"":""图片中有3个人""}]"
, "[{""role"":""user"",""content"":""你好""},{""role"":""assistant"",""content"":""你好！有什么可以帮助你的吗？""}]"
```

注意：`image_path` 为空时是纯文本对话，不影响训练。

### VLDataset 类

```python
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPImageProcessor


class VLDataset(Dataset):
    """图文多模态数据集"""

    def __init__(self, df, tokenizer, image_processor, max_length=1024,
                 image_token_id=6400, num_image_tokens=50):
        self.df = df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens
        self.padding = tokenizer.pad_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row.get('image_path', None)
        conversations = json.loads(row['conversations'])

        # 构建 prompt
        messages = []
        for conv in conversations:
            content = conv['content']
            # 替换 <image> 为多个 image token 占位符
            if '<image>' in content:
                content = content.replace(
                    '<image>',
                    ' '.join([f'<image_token_{i}>' for i in range(self.num_image_tokens)])
                )
            messages.append({"role": conv['role'], "content": content})

        # 用 chat template 格式化
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 将 <image_token_X> 占位符替换为实际的 image_token_id
        # 先正常 tokenize，然后把占位符位置替换
        input_ids = self.tokenizer(prompt).data['input_ids']

        # 找到 image token 占位符的位置并替换
        # ... (具体实现需要处理占位符 token)

        # 加载图像
        pixel_values = None
        image_mask = None
        if image_path and pd.notna(image_path):
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values']
            pixel_values = pixel_values.squeeze(0)  # (3, 224, 224)

        # 截断 + padding
        text_len = len(input_ids)
        padding_len = self.max_length - text_len
        input_ids = input_ids[:self.max_length] + [self.padding] * max(0, padding_len)

        # 构建 labels 和 loss_mask
        # ... 与 SFTDataset 类似，只对 assistant 回答部分计算 loss

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'loss_mask': torch.tensor(loss_mask),
            'pixel_values': pixel_values,
            'image_mask': torch.tensor(image_mask),
        }
```

---

## 训练脚本设计

### 阶段 1：对齐训练

```bash
torchrun --nproc_per_node=2 train_vl.py \
    --stage align \
    --lm_checkpoint out/mindlm_sft_768_linear_epoch2.pth \
    --model_config mindlm_1b \
    --data_path data/vl_align_data.csv \
    --epochs 2 \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --accumulation_steps 4 \
    --ddp \
    --num_workers 4
```

关键点：
- `--stage align`：只训练投影层
- `learning_rate=1e-3`：投影层从头训，可以用高学习率
- `batch_size=16`：有图像，显存占用更大

### 阶段 2：端到端微调

```bash
torchrun --nproc_per_node=2 train_vl.py \
    --stage finetune \
    --vl_checkpoint out/mindlm_vl_align_epoch1.pth \
    --model_config mindlm_1b \
    --data_path data/vl_sft_data.csv \
    --epochs 2 \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --accumulation_steps 8 \
    --ddp \
    --num_workers 4
```

关键点：
- `--stage finetune`：解冻 MindLM，ViT 仍冻结
- `learning_rate=5e-5`：跟 SFT 一样
- `batch_size=8`：视觉 token 占序列空间，batch 更小

---

## 推理脚本设计

```python
from PIL import Image
from transformers import CLIPImageProcessor

def chat_vl(model, tokenizer, image_processor, question, image_path=None):
    """多模态对话"""
    messages = [{"role": "user", "content": ""}]

    if image_path:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        pixel_values = image_processor(images=image, return_tensors="pt")['pixel_values']
        pixel_values = pixel_values.to(model.lm.tok_embeddings.weight.device)

        # 构建带 <image> 的 prompt
        content = f"<image>\n{question}"
    else:
        pixel_values = None
        content = question

    # 用 chat template 格式化
    prompt = f"<s>user\n{content}</s>\n<s>assistant\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 构建 image_mask
    image_mask = torch.zeros_like(input_ids)
    if pixel_values is not None:
        # 找到 <image> token 位置
        image_positions = (input_ids == model.image_token_id).nonzero(as_tuple=True)[1]
        image_mask[:, image_positions] = 1

    # 生成
    with torch.no_grad():
        # 先获取视觉 embedding
        if pixel_values is not None:
            vision_tokens = model.encode_image(pixel_values)
            text_embeddings = model.lm.tok_embeddings(input_ids)
            merged = model._merge_vision_text(text_embeddings, vision_tokens, image_mask)
            # 自回归生成
            # ... (需要实现 generate 方法支持视觉 token)
        else:
            output = model.lm.generate(input_ids, eos=tokenizer.eos_token_id, max_new_tokens=256)

    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response
```

---

## 数据准备指南

### 阶段 1 数据：图文对（图片描述）

| 数据集 | 规模 | 说明 | 来源 |
|--------|------|------|------|
| LAION-400M 子集 | 10-50万对 | 图文对，训练对齐 | HuggingFace |
| COCO Captions | 12万对 | 图片+描述 | HuggingFace |
| 自建中文图文对 | 5-10万对 | 中文场景 | 自己爬取/API生成 |

格式：图片 + 一句描述，不需要对话。

### 阶段 2 数据：视觉问答（VQA）

| 数据集 | 规模 | 说明 | 来源 |
|--------|------|------|------|
| LLaVA-Instruct-150K | 15万条 | 图文对话 | HuggingFace |
| ShareGPT4V | 10万条 | 图文对话 | HuggingFace |
| COCO-VQA | 20万条 | 视觉问答 | HuggingFace |
| 自建中文 VQA | 5万条 | 中文视觉问答 | API生成 |

### 数据量建议

| 阶段 | 数据量 | 效果预期 |
|------|--------|---------|
| 对齐（阶段1） | 5万对 | 基本对齐 |
| 对齐（阶段1） | 20万对 | 良好对齐 |
| VQA 微调（阶段2） | 5万条 | 能回答简单视觉问题 |
| VQA 微调（阶段2） | 20万条 | 较好的视觉理解 |

---

## 显存估算

| 组件 | 参数量 | 显存 (bf16) |
|------|--------|------------|
| CLIP-ViT-Base | 88M | ~0.18GB（冻结，不需要优化器） |
| 投影层 MLP | 1.2M | ~0.01GB |
| MindLM-1B | 95M | ~0.2GB + ~0.8GB（优化器） |
| **总计** | ~184M | ~1.2GB（模型部分） |

加上激活值（含 50 个视觉 token + 图像处理）：

| 阶段 | batch_size | seq_len | 显存/卡 | 2×A800 可行 |
|------|-----------|---------|---------|:-----------:|
| 对齐 | 16 | 1024 | ~15GB | 轻松 |
| 微调 | 8 | 1024 | ~25GB | 可行 |
| 微调 | 4 | 2048 | ~20GB | 可行 |

---

## 依赖安装

```bash
pip install transformers pillow
# CLIP 会随 transformers 自动下载
```

---

## 实施顺序

```
1. 新建 modeling_mindlm_vl.py（VisionProjection + MindLMVL 类）
2. 新建 dataset_vl.py（VLDataset 类）
3. 准备对齐数据（COCO Captions 或 LAION 子集）
4. 阶段1：对齐训练（只训投影层）
5. 验证对齐效果（图像描述是否合理）
6. 准备 VQA 数据（LLaVA-Instruct 或自建）
7. 阶段2：端到端微调
8. 新建 inference_vl.py（推理脚本）
9. 测试：输入图片+问题，验证回答质量
```

---

## 注意事项

1. **ViT 始终冻结**：CLIP 的视觉编码器已经训练得很好，不需要再训
2. **<image> token**：需要在 tokenizer 中新增一个特殊 token（vocab 6400→6401）
3. **序列长度**：50 个视觉 token 会占掉一部分上下文窗口，实际可用文本长度减少
4. **中文数据**：CLIP 对中文图片的理解能力弱于英文，建议用中英混合数据
5. **图像预处理**：CLIP 要求 224×224 输入，非标准尺寸图片会自动缩放裁剪

---

## 参考资源

- **LLaVA 论文**: Visual Instruction Tuning，VL 路线的经典实现
- **MiniGPT-4**: 类似的 VL 架构，代码简洁可参考
- **CLIP 论文**: Contrastive Language-Image Pre-training
- **ShareGPT4V**: 开源图文对话数据集
- **LLaVA-Instruct-150K**: 开源 VQA 训练数据
