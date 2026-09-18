# Qwen3 Tokenizer 标记说明

`qwen3_tokenizer/` 是从 Qwen3 复制的 BPE tokenizer，MindLM 的
`mindlm_0.7b`、`mindlm_0.2b_gdn` 配置与 SFT 流程使用该 tokenizer
（见 `pretrain.py` 默认值与 `docs/sft.md`）。

## 文件组成

| 文件 | 说明 |
| --- | --- |
| `tokenizer_config.json` | 配置：添加标记、特殊 token、chat template、长度上限等 |
| `tokenizer.json` | HF fast tokenizer 完整定义（词表 + BPE merge 规则 + 预分词规则） |
| `vocab.json` | 基础 BPE 词表，token → ID，共 151,643 项（ID 0–151642） |
| `merges.txt` | BPE merge 规则，与 `vocab.json` 对应 |

## 关键配置

| 配置 | 值 |
| --- | --- |
| `tokenizer_class` | `Qwen2Tokenizer` |
| 词表总大小 | 151,669 = 151,643 基础 BPE + 26 添加标记（ID 151643–151668） |
| `eos_token` | `<|im_end|>` |
| `pad_token` | `<|endoftext|>` |
| `bos_token` / `unk_token` | `null`（不添加 BOS，无 UNK；未知字符按 `errors: replace` 处理） |
| `model_max_length` | 131072 |
| `add_bos_token` / `add_prefix_space` | `false` |

## 添加标记总表（ID 151643–151668）

### 基础控制

| ID | 标记 | `special` | 用途 |
| --- | --- | --- | --- |
| 151643 | `<|endoftext|>` | true | `pad_token`；预训练文档结束 |
| 151644 | `<|im_start|>` | true | ChatML 对话轮开始 |
| 151645 | `<|im_end|>` | true | `eos_token`；对话轮结束 |

### 视觉 / 多模态（均属于 `additional_special_tokens`）

| ID | 标记 | `special` |
| --- | --- | --- |
| 151646 | `<|object_ref_start|>` | true |
| 151647 | `<|object_ref_end|>` | true |
| 151648 | `<|box_start|>` | true |
| 151649 | `<|box_end|>` | true |
| 151650 | `<|quad_start|>` | true |
| 151651 | `<|quad_end|>` | true |
| 151652 | `<|vision_start|>` | true |
| 151653 | `<|vision_end|>` | true |
| 151654 | `<|vision_pad|>` | true |
| 151655 | `<|image_pad|>` | true |
| 151656 | `<|video_pad|>` | true |

### 工具调用

| ID | 标记 | `special` | 用途 |
| --- | --- | --- | --- |
| 151657 | `<tool_call>` | false | 工具调用 JSON 开始 |
| 151658 | `</tool_call>` | false | 工具调用 JSON 结束 |
| 151665 | `<tool_response>` | false | 工具返回结果开始（嵌在 `user` 轮内） |
| 151666 | `</tool_response>` | false | 工具返回结果结束 |

### 代码补全（FIM）与仓库级预训练

| ID | 标记 | `special` | 用途 |
| --- | --- | --- | --- |
| 151659 | `<|fim_prefix|>` | false | FIM 前缀模式 |
| 151660 | `<|fim_middle|>` | false | FIM 中间模式 |
| 151661 | `<|fim_suffix|>` | false | FIM 后缀模式 |
| 151662 | `<|fim_pad|>` | false | FIM 填充 |
| 151663 | `<|repo_name|>` | false | 仓库名标记 |
| 151664 | `<|file_sep|>` | false | 仓库内文件分隔 |

### 思考（Qwen3 推理模式）

| ID | 标记 | `special` | 用途 |
| --- | --- | --- | --- |
| 151667 | `<think>` | false | 思考内容开始 |
| 151668 | `</think>` | false | 思考内容结束 |

`special` 为 `true` 的标记不会被分词器切分，也不会被 BPE 拆开；
`special` 为 `false` 的标记（`<tool_call>` 起）在词表中占独立 ID，但允许按
普通文本参与切分。

## Chat Template

模板采用 ChatML 格式：

```
<|im_start|>{role}\n{content}<|im_end|>\n
```

要点：

- **system**：`<|im_start|>system\n...<|im_end|>\n`；传入 `tools` 时，system
  轮内嵌入 `<tools></tools>`（函数签名 JSON 列表）与
  `<tool_call></tool_call>` 调用格式说明。
- **assistant**：默认输出
  `<|im_start|>assistant\n<think>\n{推理}\n</think>\n\n{回答}<|im_end|>\n`。
  推理内容优先取 `message.reasoning_content`，否则从 `<think>...</think>`
  中解析；最后一条用户提问之前的轮次不带 `<think>` 包裹。
- **tool**：结果合并进 `user` 轮，形如
  `<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n`，
  连续多个 tool 结果只包一对 `im_start/im_end`。
- **生成提示**：`add_generation_prompt=True` 时追加
  `<|im_start|>assistant\n`；同时若 `enable_thinking=False`，会预填空的
  `<think>\n\n</think>\n\n` 强制关闭思考模式。

## 与 MindLM 相关的注意点

- 该 tokenizer 为因果 LM tokenizer，没有 BERT 风格的 `<mask>` token；
  预训练依赖数据集的二元 `loss_mask` 屏蔽 padding 位置（见 `README.md`）。
- `mindlm_tokenizer/` 是项目自训的小词表 tokenizer（6,400 词表，
  `PreTrainedTokenizerFast`）。`pretrain.py` 默认按模型配置选择：
  `mindlm_0.7b` 和 `mindlm_0.2b_gdn` 用 `qwen3_tokenizer`，其余用
  `mindlm_tokenizer`。
- 训练/推理代码若引用 token ID，基础词表上界为 151642，完整上界为
  151668；embedding 尺寸按 151,669 配置。
