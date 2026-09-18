# MindLM Pretrain Data Format v1

> Status: Draft
> Target: MindLM 0.7B / Qwen3 tokenizer
> Focus: General knowledge pretraining, code/FIM, API/structured data, tool-calling and agent continued pretraining

---

## 1. 目标

本文定义 MindLM 的预训练数据格式与数据流水线。

设计目标：

1. 兼容当前 Qwen3 tokenizer
2. Base Pretrain 与 Chat/SFT 解耦
3. 兼容 Hugging Face Causal LM 训练习惯
4. 支持普通文本、代码/FIM、API/JSON/Schema
5. 为后续 Tool Calling / Agent 训练保留统一接口
6. Raw Dataset 与 tokenizer-specific 序列化解耦
7. 支持离线 tokenize、固定长度 packing 和高吞吐训练

整体分三层：

```text
Raw Dataset
    │
    ▼
Renderer / Tokenizer
    │
    ▼
Packed Token Dataset
```

Raw Dataset 保存尽可能原始、结构化的数据；Renderer 决定如何序列化；Packed Token Dataset 是训练器最终读取的形式。

---

## 2. Tokenizer 约束

MindLM 0.7B 当前使用从 Qwen3 复制的 tokenizer。

| 项目 | 值 |
|---|---:|
| Tokenizer | Qwen3 / Qwen2Tokenizer |
| 基础 BPE 词表 | 151,643 |
| 完整词表 | 151,669 |
| `model_max_length` | 131072 |
| BOS | 无 |
| Chat EOS | `<|im_end|>` |
| Padding | `<|endoftext|>` |

训练边界使用聊天 EOS；padding 仍然是独立的 tokenizer 配置项，不能混用：

```text
<|endoftext|> = 151643
<|im_end|>    = 151645
```

### 2.1 Base Pretrain 文档结束符

MindLM 当前以 Qwen3 聊天 EOS 作为统一 document boundary：

```text
<|im_end|>
```

例如：

```text
Document A<|im_end|>Document B<|im_end|>Document C<|im_end|>
```

预训练代码必须按 token 字符串显式解析，不能依赖不同 tokenizer 配置下含义可能变化的
`tokenizer.eos_token_id`：

```python
CHAT_EOS_TOKEN = "<|im_end|>"
CHAT_EOS_ID = tokenizer.convert_tokens_to_ids(CHAT_EOS_TOKEN)
```

`<|endoftext|>` 仍可作为 padding token，但不是当前预训练文档边界。

---

## 3. Base Pretrain 不使用 ChatML

这里使用 `<|im_end|>` 仅表示文档边界，并不意味着把普通文本包装成对话消息。

普通文本预训练不应该转成：

```text
<|im_start|>system
...
<|im_start|>user
...
<|im_start|>assistant
...
```

Base Pretrain 主要学习自然语言、知识、代码和结构化文本分布，而不是对话协议。

普通文档应保持：

```text
太阳是太阳系中心的恒星……<|im_end|>
```

ChatML 主要用于：

- instruction tuning
- tool calling
- agent trajectory
- conversation continued pretraining
- SFT / RL

---

## 4. Raw Dataset 类型

MindLM v1 建议只定义三类 Raw Record：

```text
MindLM Raw Data
│
├── document
├── code
└── tool_trajectory
```

这样足以覆盖通用预训练、代码/FIM 和 Agent/Tool 数据。

---

## 5. Document Record

网页、百科、书籍、论文、教材、API 文档、JSON、YAML、XML、Markdown 等统一使用 `document` schema。

推荐：

```json
{
  "id": "fineweb:CC-MAIN-2025-13:xxxx",
  "text": "太阳是太阳系中心的恒星……",
  "source": "fineweb",
  "language": "zh",
  "domain": "web",
  "quality_score": 0.91,
  "url": "https://example.com/...",
  "license": null
}
```

最小必需字段：

```json
{
  "id": "doc_xxx",
  "text": "...",
  "source": "dataset_name",
  "language": "zh"
}
```

推荐字段：

| 字段 | 用途 |
|---|---|
| `id` | 全局唯一标识 |
| `text` | 原始文本 |
| `source` | 数据来源 |
| `language` | 语言 |
| `domain` | web / wiki / book / api / stem / etc |
| `quality_score` | 数据采样与过滤 |
| `url` | 溯源 |
| `license` | 许可证 |
| `timestamp` | 时间信息 |

---

## 6. Document Renderer

Raw：

```json
{
  "text": "水在标准大气压下的沸点约为100摄氏度。"
}
```

Renderer：

```text
水在标准大气压下的沸点约为100摄氏度。<|im_end|>
```

多个 document 连续拼接：

```text
Doc A<|im_end|>Doc B<|im_end|>Doc C<|im_end|>
```

---

## 7. Packing

不要采用：

```text
one document = one training sample
```

否则短文档会造成大量 padding。

推荐把文档构造成连续 token stream：

```text
A<EOD>B<EOD>C<EOD>D<EOD>...
```

然后按照固定长度切块：

```text
tokens[0:4096]
tokens[4096:8192]
tokens[8192:12288]
...
```

因此一个 training sample 可以跨越多个 document：

```text
[Doc A tail]
<|im_end|>
[Doc B]
<|im_end|>
[Doc C head]
```

Base Pretrain v1 不要求在 EOD 位置 reset Delta state，先采用标准 causal pretraining 的连续流做法。

---

## 8. Packed Dataset 格式

Base Pretrain 最终推荐保存：

```python
{
    "input_ids": np.ndarray(shape=(seq_len,), dtype=np.uint32),
    "loss_mask": np.ndarray(shape=(seq_len,), dtype=np.uint8)
}
```

例如：

```text
seq_len = 4096
```

则：

```python
input_ids.shape == (4096,)
loss_mask.shape == (4096,)
```

### 8.1 为什么用 uint32

当前词表为 151,669，`uint16` 最大只能表示 65,535，因此不足。

推荐：

```text
input_ids → uint32
loss_mask → uint8
```

进入 PyTorch 时再转换：

```python
input_ids = torch.tensor(input_ids, dtype=torch.long)
```

---

## 9. Labels 规范

MindLM 应遵循 Hugging Face CausalLM 标准：

```python
labels = input_ids.clone()
```

即：

```text
input_ids = [A, B, C, D]
labels    = [A, B, C, D]
```

模型内部做 causal shift：

```text
A → B
B → C
C → D
```

推荐 loss：

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

---

## 10. 当前 MindLM 需要修正的 loss

当前实现直接：

```python
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1),
    ignore_index=-100,
)
```

没有 next-token shift。

如果数据采用标准 `labels = input_ids`，会变成同位置预测，而不是 next-token prediction。

因此建议优先修改 MindLM `forward()`，不要长期让数据格式迁就这个实现。

---

## 11. Loss Mask

统一训练接口：

```python
{
    "input_ids": ...,
    "loss_mask": ...
}
```

Collator：

```python
labels = input_ids.clone()
labels[loss_mask == 0] = -100
```

Base Pretrain 通常：

```text
loss_mask = [1, 1, 1, ..., 1]
```

只有真实 padding 的位置设为 0。

---

## 12. API / JSON / Structured Data

面向 Agent 的 Base 模型应该直接看到大量：

- JSON
- JSON Schema
- YAML
- XML
- TOML
- OpenAPI
- API documentation
- CLI documentation
- SDK documentation
- Markdown
- tables

这些仍然作为普通 `document` 做 causal pretraining。

例如：

```text
{
  "name": "get_weather",
  "description": "Returns weather information",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string"
      }
    },
    "required": ["city"]
  }
}
<|im_end|>
```

不要因为最终目标是 Agent，就把所有 API 数据都包装成 ChatML。

---

## 13. Code Record

代码使用独立 schema：

```json
{
  "id": "github:owner/repo:src/main.py",
  "repo_name": "owner/repo",
  "path": "src/main.py",
  "language": "Python",
  "content": "def main():\n    pass\n",
  "license": "Apache-2.0"
}
```

推荐字段：

| 字段 | 用途 |
|---|---|
| `id` | 唯一标识 |
| `repo_name` | repository |
| `path` | 文件路径 |
| `language` | 编程语言 |
| `content` | 文件内容 |
| `license` | 许可证 |

---

## 14. 普通代码预训练

普通代码可直接：

```text
def add(a, b):
    return a + b
<|im_end|>
```

按普通 causal pretraining 处理。

---

## 15. Repository-Level Code

当前 tokenizer 已存在：

```text
<|repo_name|>
<|file_sep|>
```

MindLM 推荐 renderer：

```text
<|repo_name|>owner/repo
<|file_sep|>src/main.py
def main():
    ...

<|file_sep|>src/utils.py
def helper():
    ...

<|file_sep|>README.md
# Project
...
<|im_end|>
```

该格式是 MindLM 推荐设计，并不宣称与 Qwen 私有 pretraining serialization 完全一致。

---

## 16. FIM

Tokenizer 已存在：

```text
<|fim_prefix|>
<|fim_middle|>
<|fim_suffix|>
<|fim_pad|>
```

Raw Dataset 中不要提前保存 FIM 字符串，只保存完整代码。

训练阶段随机采样：

```text
prefix
middle
suffix
```

Renderer：

```text
<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|im_end|>
```

推荐在线生成 FIM，而不是把相同代码提前复制为 causal/FIM 两份。

例如可实验：

```text
50% causal
50% FIM
```

具体比例通过训练实验决定。

---

## 17. Tool Trajectory Record

真正的 Tool Calling / Agent 数据不要提前保存成 ChatML 字符串。

Raw Dataset 推荐使用标准化的：

```text
messages + tools
```

示例：

```json
{
  "id": "agent_000001",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询指定城市天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": ["city"]
        }
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "深圳今天会下雨吗？"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "city": "深圳"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "{\"weather\":\"rain\",\"temperature\":28}"
    },
    {
      "role": "assistant",
      "content": "深圳今天有雨，气温约 28℃。"
    }
  ]
}
```

---

## 18. Tool Trajectory Renderer

Tool trajectory 不重新定义协议。

统一使用当前 tokenizer 自带：

```python
tokenizer.apply_chat_template(...)
```

例如：

```python
text = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=False,
)
```

当前 tokenizer 已经定义：

```text
<tool_call>
</tool_call>
<tool_response>
</tool_response>
```

以及 ChatML：

```text
<|im_start|>{role}
...
<|im_end|>
```

因此 train 和 inference 必须使用同一套 template。

不要额外创建另一套不属于当前 tokenizer 的 tool protocol。

---

## 19. Tool 数据存储格式

普通 document/code 推荐：

```text
Parquet
```

Tool trajectory 中存在任意嵌套 JSON Schema：

```text
tools[].function.parameters.properties
```

不同样本的 schema 可能不同，因此推荐：

```text
tool_trajectory → JSONL
```

或者使用 Parquet，但把复杂 JSON 对象保存成 JSON string。

推荐目录：

```text
data/
├── document/
│   ├── train-00000.parquet
│   └── ...
├── code/
│   ├── train-00000.parquet
│   └── ...
└── tool_trajectory/
    ├── train-00000.jsonl
    └── ...
```

---

## 20. 训练阶段

### Stage 1 — General Pretrain

目标：

```text
language
general knowledge
world knowledge
reasoning substrate
code
structured data literacy
```

主要数据：

```text
web
books
wiki
STEM
code
JSON/YAML/XML
API documentation
SDK documentation
CLI documentation
```

主要序列形式：

```text
text<|im_end|>
```

### Stage 2 — Knowledge / Agent Continued Pretrain

提高：

```text
high-quality knowledge
STEM
code
API docs
OpenAPI
JSON Schema
structured reasoning data
少量 tool trajectory
```

让模型逐步熟悉 schema、function signature、arguments 和 structured output。

### Stage 3 — Agent SFT

重点训练：

```text
when to call
which tool to call
arguments
when not to call
clarification
multi-turn tool use
tool result grounding
error handling
```

主要数据形式：

```text
messages + tools
```

---

## 21. SFT Loss Mask

SFT 继续复用：

```python
{
    "input_ids": ...,
    "loss_mask": ...
}
```

推荐：

```text
system            0
tools             0
user              0
assistant          1
tool_call          1
tool_response      0
assistant final    1
```

即：

| 内容 | Loss |
|---|---:|
| System | × |
| Tool definitions | × |
| User | × |
| Assistant tool call | ✓ |
| Tool response | × |
| Assistant final answer | ✓ |

这样 Base Pretrain 和 SFT 可以共享同一套训练接口。

---

## 22. 训练张量接口

建议统一为：

```python
class MindLMSample:
    input_ids: Tensor[seq_len]
    loss_mask: Tensor[seq_len]
```

Collator：

```python
input_ids = ...
labels = input_ids.clone()
labels[loss_mask == 0] = -100

batch = {
    "input_ids": input_ids,
    "labels": labels,
}
```

如果以后需要对 packed SFT/Agent sample 做 sequence isolation，再扩展：

```python
segment_ids
```

但 Base Pretrain v1 不要求必须保存该字段。

---

## 23. 离线 Tokenize 流程

大规模预训练不要在 trainer 中反复运行 BPE。

推荐：

```text
Raw Parquet / JSONL
        │
        ▼
 Dataset Cleaning
        │
        ▼
 Dedup / Filtering
        │
        ▼
 Renderer
        │
        ▼
 Qwen3 Tokenizer
        │
        ▼
 append <|im_end|>
        │
        ▼
 token stream
        │
        ▼
 packing
        │
        ▼
 binary/token shards
        │
        ▼
 MindLM Trainer
```

---

## 24. Token Shard

可以使用：

```text
.npy
memmap
Arrow
WebDataset
自定义 binary + index
```

核心目标是：

```text
sequential read
low CPU overhead
no runtime BPE
high GPU utilization
```

当前 MindLM 实现使用一个轻量的 binary + manifest 变体：

```text
<prefix>.bin   # little-endian uint32，连续保存 sequence_length + 1 token 的 record
<prefix>.json  # format、sequence_length、tokenizer_vocab_size、boundary_token_id 等元数据
```

训练器把每个 record 的前 `sequence_length` 个 token 作为 `input_ids`，后移一位
作为 next-token `targets`；packed Base Pretrain 没有 padding，因此 `loss_mask`
在内存中是全 1 的 `uint8` 张量。数据准备器会显式写入 `boundary_token_id`，并使用
Qwen3 聊天 EOS `<|im_end|>` 作为文档边界。

逻辑字段仍保持：

```text
input_ids : uint32
loss_mask : uint8
```

---

## 25. Base Pretrain 示例

Raw：

```json
{"id":"wiki_001","text":"太阳是太阳系的中心恒星。","source":"wiki","language":"zh"}
```

```json
{"id":"wiki_002","text":"地球是太阳系第三颗行星。","source":"wiki","language":"zh"}
```

Render：

```text
太阳是太阳系的中心恒星。<|im_end|>地球是太阳系第三颗行星。<|im_end|>
```

Tokenize：

```text
[...., 151643, ...., 151643]
```

Packing：

```text
sample 0 = tokens[0:4096]
sample 1 = tokens[4096:8192]
...
```

---

## 26. API Documentation 示例

Raw：

```json
{
  "id": "api_get_weather",
  "text": "# get_weather\n\nReturns weather information...\n\nParameters:\n- city: string\n...",
  "source": "api_docs",
  "language": "en",
  "domain": "api"
}
```

Render：

```text
# get_weather

Returns weather information...

Parameters:
- city: string
...
<|im_end|>
```

仍然属于普通 causal pretraining。

---

## 27. 不推荐的格式

### 27.1 不推荐所有数据 ChatML 化

不要：

```text
<|im_start|>user
百科文章
<|im_end|>
```

Base 文本应保持自然形式。

### 27.2 不推荐 Raw Dataset 保存 token

Raw Dataset 应保留文本；token shard 是下游训练产物。

### 27.3 不推荐 Raw Tool Dataset 只保存 ChatML string

应该保留：

```text
messages
tools
```

否则不便于以后修改 chat template、tool protocol、loss mask 或 thinking mode。

### 27.4 显式使用 Qwen3 聊天 EOS 作为边界

当前 MindLM 预训练以 Qwen3 聊天 EOS `<|im_end|>` 为文档边界。代码应按 token
字符串显式解析，避免不同 tokenizer 配置下 `tokenizer.eos_token_id` 的语义漂移：

```python
CHAT_EOS_TOKEN = "<|im_end|>"
CHAT_EOS_ID = tokenizer.convert_tokens_to_ids(CHAT_EOS_TOKEN)
```

`<|endoftext|>` 仅作为 padding token 保留，不参与文档拼接。

---

## 28. 推荐目录结构

```text
mindlm-data/
├── raw/
│   ├── documents/
│   ├── code/
│   └── tool_trajectory/
├── cleaned/
│   ├── documents/
│   ├── code/
│   └── tool_trajectory/
├── tokenized/
│   ├── stage1/
│   ├── stage2/
│   └── sft/
└── manifests/
    ├── stage1.json
    ├── stage2.json
    └── sft.json
```

Manifest 示例：

```json
{
  "name": "mindlm-stage1-v1",
  "tokenizer": "qwen3_tokenizer",
  "vocab_size": 151669,
  "sequence_length": 4096,
  "boundary_token": "<|im_end|>",
  "boundary_token_id": 151645,
  "datasets": [
    {"name": "fineweb_zh", "weight": 0.35},
    {"name": "fineweb_en", "weight": 0.25},
    {"name": "code", "weight": 0.20},
    {"name": "stem", "weight": 0.10},
    {"name": "api_structured", "weight": 0.10}
  ]
}
```

这些权重只是 manifest 表达示例，实际比例需要根据数据统计和训练实验确定。

---

## 29. MindLM Data v1 总体结构

```text
                         MindLM Data v1
                                │
             ┌──────────────────┼─────────────────┐
             │                  │                 │
             ▼                  ▼                 ▼
         document              code        tool_trajectory
          Parquet             Parquet          JSONL
             │                  │                 │
             │           causal / FIM             │
             │          repository packing        │
             │                  │          apply_chat_template
             └──────────────────┼─────────────────┘
                                ▼
                         Qwen3 Tokenizer
                         vocab = 151669
                                │
                                ▼
                   Base document boundary:
                  <|im_end|> = 151645
                                │
                                ▼
                         Token Stream
                                │
                   concatenate + fixed packing
                                │
                                ▼
                          seq_len = N
                                │
                                ▼
                  ┌────────────────────────┐
                  │ input_ids : uint32     │
                  │ loss_mask : uint8      │
                  └────────────────────────┘
                                │
                                ▼
                     labels = input_ids
                     masked → -100
                                │
                                ▼
                  Causal shift inside MindLM
```

---

## 30. Implementation Checklist

在下一次正式预训练前建议确认：

- [ ] Base Pretrain 使用 Qwen3 `<|im_end|>` 作为文档边界
- [ ] 通过 token 字符串显式解析 `<|im_end|>` 的 id
- [ ] 普通文档不经过 ChatML
- [ ] Raw 数据保留 text / messages / tools
- [ ] Base 数据执行 concatenate + fixed-length packing
- [ ] `input_ids` 磁盘使用 uint32
- [ ] `loss_mask` 使用 uint8
- [ ] MindLM `forward()` 内实现 causal label shift
- [ ] API docs / JSON / Schema 作为普通 document 参与 pretrain
- [ ] 代码保留 repo/path metadata
- [ ] FIM 在 renderer/tokenization 阶段生成
- [ ] Tool trajectory 统一使用 `messages + tools`
- [ ] Tool renderer 使用 tokenizer 自带 `apply_chat_template`
- [ ] SFT 通过 `loss_mask` 仅训练目标 assistant token
- [ ] train / inference 使用完全相同的 tool protocol

---

## 31. 参考资料

本规范基于以下资料和当前 MindLM 工程约束整理：

1. MindLM `tokenizer.md`
   - Qwen3 tokenizer
   - `<|endoftext|>` / `<|im_end|>`
   - ChatML
   - tool call / tool response tokens
   - FIM / repository tokens

2. Hugging Face Transformers — Causal Language Modeling
   - concatenate documents
   - fixed block packing
   - `labels = input_ids`
   - causal shift semantics

3. Hugging Face Qwen3 Base
   - Base tokenizer / EOS 约定

4. Hugging Face FineWeb
   - text-first raw dataset
   - metadata-preserving document schema

5. BigCode / StarCoder / The Stack
   - code metadata
   - repository information
   - Fill-In-the-Middle

6. Hugging Face tool-calling datasets
   - `messages + tools`
   - OpenAI-style function/tool schema
   - structured trajectory storage

---

## 32. Summary

MindLM Pretrain Data Format v1 的核心原则：

> **Base Pretrain 使用自然文本，不使用 ChatML。**

> **普通 document 使用 Qwen3 `<|im_end|>` 结束，然后 concatenate + fixed-length packing。**

> **代码保留原始 repository/file 结构，FIM 在 renderer/tokenization 阶段动态构造。**

> **Tool/Agent 数据保存为 `messages + tools`，只有训练时才通过当前 Qwen3 tokenizer 的 chat template 渲染。**
