# MindLM Dense 0.8B 方案（Qwen3 tokenizer）

## 1. 目标

Dense 0.8B 主要学习稳定的语言建模、可验证推理、任务规划和工具调用。知识不追求覆盖所有百科内容，外部检索、计算器、代码执行器和业务 API 负责提供事实和确定性计算。

```text
用户请求 -> 判断 -> 直接回答或生成工具调用 -> 执行器运行工具 -> 读取结果并总结
```

## 2. Tokenizer 决策

本项目只复用本机目录：

```text
/data/runke/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B
```

已复制到仓库的 `qwen3_tokenizer/`，包含 `tokenizer.json`、`tokenizer_config.json`、`vocab.json` 和 `merges.txt`。复用的是 tokenizer 文件，不复用 Qwen3-0.6B 的模型权重；Qwen3 权重的层数、注意力结构和参数形状与 MindLM 不兼容。

Qwen3 tokenizer 的实际词表大小是 **151,936**。关键 token 为：

| 用途 | token | id |
| --- | --- | ---: |
| padding | `<|endoftext|>` | 151643 |
| 对话起始 | `<|im_start|>` | 151644 |
| 对话结束 / EOS | `<|im_end|>` | 151645 |
| 工具调用 | `<tool_call>` / `</tool_call>` | 151657 / 151658 |
| 工具结果 | `<tool_response>` / `</tool_response>` | 151665 / 151666 |
| 思考段 | `<think>` / `</think>` | 151667 / 151668 |

Qwen3 是 causal LM，**没有 BERT 式 `<mask>` token**。预训练不向文本插入 `<mask>`；数据集只生成独立的二值 `loss_mask`，padding 位置为 0，真实 token 位置为 1。训练 loss 由 `masked_language_model_loss` 按 token mask 计算。

## 3. Dense 0.8B 配置

| 参数 | 值 |
| --- | ---: |
| `dim` | 1152 |
| `n_layers` | 40 |
| `n_heads` / `n_kv_heads` | 16 / 4 |
| `linear_attn_heads` | 16 |
| `head_dim` | 72 |
| `hidden_dim` | 2816 |
| `max_seq_len` | 4096 |
| 标准 Attention | 10 层 |
| Gated DeltaNet | 30 层 |
| `use_moe` | `false` |
| embedding | tied |

层模式为 `[L, L, L, A] x 10`。按当前实现和 151,936 词表估算约 **798.2M 参数**，最终以实际实例化后的 `sum(p.numel())` 为准。

配置文件是 `config/mindlm_0.8b.json`。`build_model_config` 会再次从 tokenizer 读取 `len(tokenizer)` 及 pad/bos/eos id，避免配置与 tokenizer 漂移。

## 4. 数据和 mask 约定

预训练 CSV 只需要 `text` 列。`PretrainDataset` 截断到 `max_seq_len`，右侧用 Qwen3 的 pad id 补齐，并返回：

```text
input_ids = tokens[:-1]
targets   = tokens[1:]
loss_mask = [1 for real token] + [0 for padding]
```

SFT 使用 Qwen3 自带 `apply_chat_template`。答案边界优先查找 `<|im_start|>assistant\n`，旧 MindLM tokenizer 才回退到 `<s>assistant\n`。带最终答案的完整模板不追加 generation prompt，避免产生第二个 assistant marker；loss 只计算最终 assistant 内容。

## 5. 工具调用协议

优先使用 Qwen3 tokenizer 已有的 token，不新增 special token：

```text
<tool_call>
{"name":"calculator","arguments":{"expression":"(17*23)+4"}}
</tool_call>
<tool_response>
{"name":"calculator","result":"395"}
</tool_response>
```

工具执行器负责校验工具名、arguments schema 和执行结果。第一版每轮最多 3 次调用；解析失败或工具失败都返回结构化错误，不伪造成功结果。

## 6. 训练阶段

1. **基础预训练**：中文/英文文本 50%，代码和技术文档 20%，数学逻辑 15%，高质量知识问答 10%，JSON/表格 5%。先以 2048 长度验证收敛。
2. **长上下文继续训练**：扩展到 4096，加入多轮对话、长文档、工具 schema 和工具结果。
3. **推理 SFT**：使用短、可验证的计划、分解、计算、代码验证和错误修正轨迹。
4. **工具调用 SFT**：普通回答 50%，单步调用 25%，调用后总结 15%，多步调用和错误修正 10%。
5. **可验证优化**：先按 JSON 合法、schema 通过、执行成功、答案正确过滤轨迹，再尝试小规模 DPO/GRPO。

## 7. 运行时和验收

正式 Agent benchmark 前必须实现标准 Attention KV cache、DeltaNet convolution/recurrent state cache、工具轮数限制和逐样本 EOS 处理。评估除 perplexity 外，还要记录 JSON 合法率、工具选择和 schema 通过率、执行成功率、最终答案准确率、不必要调用率及失败恢复率。

实施顺序：固定 `qwen3_tokenizer/` 版本 -> 运行 2048 smoke train -> 完成两类 cache -> 加入工具执行器测试 -> 4096 继续训练 -> 工具调用 SFT。
