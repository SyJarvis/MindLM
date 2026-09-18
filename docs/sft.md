# MindLM SFT

本文描述当前正式 SFT 设计与实现要求。

## 1. 基线与训练配置

SFT 从完成预训练的 `mindlm_0.2b_gdn` checkpoint 初始化，使用 `qwen3_tokenizer`。训练采用 BF16、全量 gradient checkpointing、AdamW、learning rate `5e-5`、warmup `916`、gradient clip `1.0`、5 epochs，以及当前 grouped/packed 训练入口。

在重建数据 bin 之前必须完成 baseline 归档。bin 的行序或长度变化后，旧 checkpoint 的 `next_micro` 不再对应新数据，不能继续恢复。

## 2. 数据与监督实现

训练数据必须保留完整消息角色和 `tools` 字段，并通过与线上评测一致的 chat template 渲染。

监督 mask 使用 role-aware 区间：每个区间覆盖 assistant 正文及其结尾 `<|im_end|>`，不覆盖 user、tool response 或其他上下文。数据元信息使用 `sup` 区间列表，训练 loader 根据区间生成 mask。

数据处理要求：

1. 在渲染后进行长度检查，避免 tools 块引入额外 token 后发生错配。
2. 先过滤，再按子集、domain 和函数族分层随机采样。
3. 去除重复消息，保留长样本覆盖。
4. 注入“有工具清单但不应调用工具”的负样本，默认比例 10%。

## 3. 代码改动边界

- `prepare_sft_qwen3.py` 保留并输出 `tools` 字段。
- `prepare_data.py --type sft` 生成 `sup` 区间并在渲染后过滤长度。
- `full_sft.py` 按 `sup` 区间计算 loss，不再使用纯位置式 `mask[-ans:]`。
- 评测使用 `eval/run_sft_eval.sh` 和 `eval/eval_sft_tool.py`，分别覆盖生成质量和工具调用行为。

## 4. 实验顺序与验收

实验按单变量顺序执行：

1. E1：只修 role-aware 监督 mask。
2. E2：E1 加入 tools 渲染贯通。
3. E3：E2 加入 10% 负样本。

验收要求：监督区间内不得出现 `<tool_response>` 或 `<|im_start|>`；每个监督段必须以 `<|im_end|>` 收尾；正例工具调用、负例拒答和普通生成评测均需通过；200-update smoke 的 loss、梯度、显存和吞吐正常。

每个实验使用独立 run 目录和 checkpoint，不能混用不同数据 bin 的恢复状态。
