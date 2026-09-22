# MindLM SFT

本文描述当前正式 SFT 设计与实现要求。

## 1. 基线与训练配置

SFT 从完成预训练的 `mindlm_0.2b_gdn` checkpoint 初始化，使用 `qwen3_tokenizer`。训练采用 BF16、全量 gradient checkpointing、AdamW、learning rate `5e-5`、warmup `916`、gradient clip `1.0`、5 epochs，以及当前 grouped/packed 训练入口。模型上下文 4096（`config/mindlm_0.2b_gdn.json` 的 `max_seq_len`）。

在重建数据 bin 之前必须完成 baseline 归档。bin 的行序或长度变化后，旧 checkpoint 的 `next_micro` 不再对应新数据，不能继续恢复。

## 2. 数据与监督实现

训练数据必须保留完整消息角色和 `tools` 字段，并通过与线上评测一致的 chat template 渲染。

监督 mask 使用 role-aware 区间：每个区间覆盖 assistant 正文及其结尾 `<|im_end|>`，不覆盖 user、tool response 或其他上下文。数据元信息使用 `sup` 区间列表，训练 loader 根据区间生成 mask。

数据处理要求：

1. 在渲染后进行长度检查，避免 tools 块引入额外 token 后发生错配。长度过滤阈值与模型上下文一致：`prepare_sft_qwen3.py --max_tokens 4096`（渲染含 tools 块后计量）。
2. 先过滤，再按子集、domain 和函数族分层随机采样。
3. 去除重复消息，保留长样本覆盖。
4. 注入“有工具清单但不应调用工具”的负样本，默认比例 10%。

### 2.1 每行工具数上限（已定：8）

`prepare_sft_qwen3.py --max_tools 8`（默认；0 = 关闭）。每行渲染进 `<tools>` 块的 schema 数量按 **called-first** 规则裁剪：

1. 本行消息中实际被调用的函数必须全部保留——监督区间里的 `<tool_call>` 不允许引用未定义工具；
2. 剩余名额从本行自己的 schema 列表按原顺序填充（干扰项与被调函数同域，迫使模型读 schema 选函数而非靠域偏好猜）；
3. 被调用函数数本身超过上限的行原样通过，不做截断。

依据（2026-09 实测，Tool_Use 全量 82,760 行 + 3,600 行渲染抽样）：

- 原始数据每行挂 p50=15 / p90=35 / max=55 个 schema，全集 14,026 个不同函数名；
- 单条 schema 渲染后约 150–170 token。cap=8 时 `<tools>` 块约 1,400 token（占 4096 的 ~34%），cap=10 约 1,666（41%），cap=20 达 63%、cap=30 达 76%，后两者会显著挤占对话与监督目标的空间并使可用行数下降；
- 参照 MiniMind：其 tool call 样本每条实际只挂 2–3 个工具（工具池 ~10 个），在 768 上下文即完成训练；
- called-first 裁剪经全量验证：0 行丢失被调用函数。

## 3. 代码改动边界

- `prepare_sft_qwen3.py` 保留并输出 `tools` 字段，写入前完成 called-first 裁剪（`--max_tools`）。
- `prepare_data.py --type sft` 生成 `sup` 区间并在渲染后过滤长度（`--max_tokens`，默认 4096）。
- `full_sft.py` 按 `sup` 区间计算 loss，不再使用纯位置式 `mask[-ans:]`。
- 评测使用 `eval/run_sft_eval.sh`、`eval/eval_sft_tool.py`（单轮探针）与 `eval/eval_sft_tool_loop.py`（闭环：调用 → mock 执行 → 回填 → 终答必须使用工具返回数据）。

## 4. 实验顺序与验收

实验按单变量顺序执行：

1. E1：只修 role-aware 监督 mask。
2. E2：E1 加入 tools 渲染贯通。
3. E3：E2 加入 10% 负样本。

验收要求：监督区间内不得出现 `<tool_response>` 或 `<|im_start|>`；每个监督段必须以 `<|im_end|>` 收尾；正例工具调用、负例拒答和普通生成评测均需通过；200-update smoke 的 loss、梯度、显存和吞吐正常。

每个实验使用独立 run 目录和 checkpoint，不能混用不同数据 bin 的恢复状态。

当前状态：E1 已落地（`sup` 区间 + grouped trainer）；E2 代码已完成（tools 列贯通渲染、审计指标 `rows_with_tools` / `tools_block_rendered` / `tools_parse_errors`、闭环评测脚本），待服务器重建 CSV/bin 后启动训练；E3 未实施。
