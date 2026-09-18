# MindLM Pretrain

本文描述当前正式预训练方案与实现。

## 1. 数据处理

入口脚本：`prepare_data.py --type pretrain`。

处理流程：

1. 读取包含 `text` 和 `source` 列的 CSV。
2. 对文本做 NFC 规范化、换行统一和首尾去空格。
3. 按规范化全文 SHA256 精确去重，保留首次出现。
4. 按规范化文本折叠空白后的前 150 个字符分组，保证同组文档进入同一 split。
5. 使用固定 `seed=1337`，按组抽取约 0.5% heldout。
6. 按确定性哈希混排后 tokenize，保留正文内部空白和代码缩进。
7. train 与 heldout 独立追加 Qwen3 聊天 EOS `<|im_end|>`，并分别 packing 成 4097-token records，对应 4096 个输入/监督 token。
8. 输出 `.bin`、`.json`、`documents.jsonl` 和 `manifest.json`，并审计 token 范围、SHA256 和 train/heldout 交集。

文档边界固定为 Qwen3 聊天 EOS `<|im_end|>`（151645），由 tokenizer 按 token 字符串显式解析。

数据资产目录由数据准备命令的 `--output-dir` 指定；train 与 heldout manifest 会记录各自的边界 token、词表和 SHA256。

## 2. 模型配置

配置文件：[`config/mindlm_0.2b_gdn.json`](../config/mindlm_0.2b_gdn.json)。

- 约 206.8M 参数，hidden size 768，16 层
- 12 层线性注意力、4 层标准注意力
- 上下文长度 4096，词表 151,669
- 线性注意力：`gated_delta_rule`，CUDA 后端为 FLA
- 标准注意力后端：FlashAttention 4
- `gdn_v3` 初始化，dropout=0
- BF16 autocast、FP32 参数和 optimizer
- 全量 gradient checkpointing

模型使用随机初始化。训练入口会检查模型数学规则、初始化方案和 CUDA backend，不满足要求时直接报错。

## 3. 训练实现

入口：[`pretrain.py`](../pretrain.py)。使用 `--model_config mindlm_0.2b_gdn`，训练和 held-out 数据分别通过 `--train_data_prefix` 与 `--validation_data_prefix` 提供。

当前训练计划：

| 参数 | 值 |
|---|---:|
| epochs | 3 |
| batch size | 64 |
| gradient accumulation steps | 8 |
| 每次 update 监督 token | 由序列长度、batch 和 accumulation 共同决定 |
| 总 updates | 由 train manifest 大小和 epochs 共同决定 |
| learning rate | 2e-4 |
| warmup | 总 updates 的 2% |
| cosine 最低学习率 | 2e-5 |
| optimizer | AdamW |
| matrix weight decay | 0.1 |
| gradient clip | 1.0 |
| loss chunk | 256 tokens |
| workers | 4 |

每个累积窗口按实际监督 token 数归一化 loss 和梯度，再执行 clip、学习率更新和 optimizer step；epoch 尾部不足完整窗口时采用相同口径。

## 4. 评估、保存与恢复

- 每 100 个 update 记录训练指标
- 每 1000 个 update 保存 latest checkpoint
- 每 1000 个 update 及每个 epoch 结束执行完整 heldout 评估
- checkpoint 保存模型、optimizer、scaler、训练游标、随机数状态、W&B ID 和数据/tokenizer/backend contract
- 恢复时严格校验配置、数据、tokenizer、运行时和训练参数；contract 不一致则拒绝恢复

正式实现已完成 CPU/GPU 数值验证、200 update smoke 和 200→201 恢复验证。
