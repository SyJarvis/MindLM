# Pretraining Asset Inventory

本文记录本地正式预训练主线，以及 SSH alias `60` 上截至 2026-10-23 已核实的权重、数据与运行记录。远端状态为盘点时可见状态；路径、文件存在性和日志可见范围不代表数据质量、许可状态或 checkpoint 生成流程已独立复现。

## 本地主线与边界

- 正式支持配置只有 `mindlm_0.2b_gdn`：`config.py` 的 `SUPPORTED_CONFIGS` 与 `pretrain.py` 的 `--model_config` choices 均只列出该配置。
- 正式配置 `config/mindlm_0.2b_gdn.json` 为 16 层、dim 768、4096 context 的混合 GDN/标准注意力模型，约 206.8M 唯一参数。训练和数据合同说明见 [`pretrain.md`](pretrain.md)。
- 本地 `config/mindlm.json` 是未跟踪的 dense 变体，不受正式 loader/入口支持；本地 `model/` 是未跟踪的第二套代码，检索未发现 import 引用。检查到的空文件 SHA256 为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。这些漂移项均保留，未修改。

## 60 上的 MindLM 权重与数据

以下路径均位于 `60` alias 对应的 `deeplearning-v2-mindlm-cdi` 容器挂载中。文件状态仅代表本次盘点时可见内容，不与 H100 主机混合。

| 分类 | 路径 / 数量 | 已核实事实 | 解释边界 |
|---|---|---|---|
| 预训练 checkpoint | `/mnt/data2/reexen_datasets/llm_training/MindLM/runs/sft_v3_e2_20260922/baseline/mindlm_pretrain_mindlm_0.2b_gdn_epoch2.pt` | 2,482,262,379 bytes；SHA256 `0e5727e093b0970e1ef3a60645c8f7b28d8e214d9da8014d4e091861d409e30e`；记录 step 4020、epoch 2 complete | 参数和 tensor 数值审计通过；生成输出呈重复退化，不能据此宣称生成质量合格 |
| SFT checkpoint | `/mnt/data2/reexen_datasets/llm_training/MindLM/out` 与 `/mnt/data2/reexen_datasets/llm_training/MindLM/runs` | 本次 `find` 统计 42 个 MindLM `*.pt` 文件，其中绝大多数为 SFT；常见 grouped SFT 文件大小 2,482,249,195 bytes。`MindLM/out` 约 9.3G，`MindLM/runs` 约 148G | 不应把 SFT 权重计为预训练权重；其中还有 MiniCPM5 的 safetensors checkpoint，需按 run 名区分 |

同目录 `baseline/SHA256SUMS` 记录 tokenizer `qwen3_tokenizer/tokenizer.json` 的 SHA256 为 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，正式配置哈希为 `9041a1f468ebd662cd75e511f2a152d937bea20de8689f91caad5d739d37f390`。

## 训练合同、运行记录与复核边界

该 checkpoint 的保存合同/manifest 记录：预计总计 4020 updates；训练集 685,756 sequences、heldout 3,254 sequences；sequence length 4096。manifest 数据前缀为 `/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v5_packed/{train,heldout}`，对应 train bin SHA256 前缀 `b441bc0f...`、heldout bin SHA256 前缀 `d477d0bf...`。盘点时这两个 packed 文件不存在/未找到，因此不能独立复核 checkpoint 所用的确切训练和 heldout 数据。

可见 W&B summary 中途 heldout 记录如下；update 2656 是该 summary 对应的 training update：

| Update | Heldout loss |
|---:|---:|
| 1340 | 3.4825 |
| 2000 | 3.3096 |
| 2656 | 3.3096 |
| 2680 | 3.2176 |
| 3000 | 3.1906 |

另一个恢复 run 的可见日志到约 update 3346；日志末尾被截断，未见 update 4020 或最终 heldout 结果。另一个恢复 run 的可见输出出现过 `flash_attn.cute` 缺失的后端启动错误。现有可见日志不足以据此断言某次 run 最终因该错误失败，也不足以证明完整 4020-update 训练与最终 heldout 评估过程。checkpoint 的保存合同/记录与可独立复核的训练证据是两回事。

## 数据资产分类

| 资产 | 路径 / 元数据 | 与 checkpoint 的关系 |
|---|---|---|
| 原训练 packed train/heldout | 预期前缀 `/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v5_packed/{train,heldout}`；manifest 哈希前缀分别 `b441bc0f...`、`d477d0bf...` | 该路径及 MindLM/data 下同名候选均不存在/未找到；确切训练/验证数据无法复核 |
| 当前可找到的 V6 合并语料 | `/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl`；31,331,876,573 bytes；meta 记录 7,217,908 rows | merge report 标为 2026-09-29 构建，晚于该 checkpoint（2026-09-22）；不能称为 checkpoint 实际训练数据或验证集 |
| V6 来源行数 | `zh_t2t` 2,840,723；`zh_cosmopedia` 2,579,892；`fineweb_edu` 902,566；`open-perfectblend` 793,204；`xlam_fc` 58,980；`tbench_toolcall` 41,803；`toolace` 740 | 记录来源和元数据，不据此推断质量或许可 |
| V6 merge report | policy 为 v5 merged，排除 minimind 并包含 full zh_cosmopedia；无 filtering/dedup；15.434B chars | 只描述该合并产物，不等同训练合同 |
| V5 merge 产物/报告 | `/mnt/data2/reexen_datasets/llm_training/pretrain/v5_merge_report.json` 记录其预期输出 `pretrain_v5_merged.csv` 为 13,095,642,691 bytes、9,801,851 rows、8.606B chars；CSV 当前不在 `pretrain/` 目录 | 报告能说明历史构建信息，不表示 CSV 当前可用；不等于 packed 训练文件 |
| 可找到的其他语料原料 | `/mnt/data2/reexen_datasets/llm_training/minimind_orig/pretrain_t2t.jsonl`（8,275,074,893 bytes）；`/mnt/data2/reexen_datasets/llm_training/chinese_cosmopedia/data/` 下 10 个 parquet（约 11G）；`/mnt/data2/reexen_datasets/llm_training/pretrain/ToolACE/data.json`（37,154,735 bytes） | 是可定位原料，不可替代缺失的原 packed train/heldout 文件 |
| SFT/MiniCPM5 数据 | `/mnt/data2/reexen_datasets/llm_training/MindLM_full_data_v2`（约 15GB）；另有 `/mnt/data2/reexen_datasets/llm_training/MindLM/runs/` 下各 SFT run 的 `full_data`、`data` 与缓存 | 属于 SFT/MiniCPM5 或 SFT tokenized 数据，不是原 pretrain packed 数据 |

以上数据路径和元数据仅按盘点时可见材料记录；不对数据质量、来源许可或训练适用性作推断。

## 当前最高优先级：训练时流式 JSONL

基于 V6 原始 JSONL，当前 P0 是实现并验证**训练时流式读取 JSONL、边 tokenize/pack 边计算**。不要求离线生成完整 packed 文件；当前入口尚不支持该路径，见 [`training-roadmap.md`](project/2026-10-23/training-roadmap.md) 的验收条件。先完成只读数据验收、确定性 train/held-out 切分与防泄漏规则、在线 4096-token 样本构造和小规模可复现验证；此优先级声明本身不是启动正式训练的授权。

旧 checkpoint 与新 V6 流式训练必须作为不同数据合同记录。正式训练前固定 tokenizer/config/data 指纹、边界 token、拆分种子，并保存完整训练日志、held-out 评估结果和终态 checkpoint/hash；预训练与 SFT 权重分开归档。
