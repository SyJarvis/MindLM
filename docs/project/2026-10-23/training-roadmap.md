# MindLM 训练方向梳理与清理计划

日期：2026-10-23

## 目的与边界

当前项目同时包含自研 MindLM 预训练/SFT、MiniCPM5 工具调用 SFT、数据准备、评测和多个历史实验。本文先统一这些工作的分类、证据标准和先后顺序；它不是启动训练的授权，也不改变已有 checkpoint、数据、报告或两端工作树。

“清理”在本计划中指整理索引、状态和复现信息。历史源码、失败实验、旧评测、模型与数据均先保留；删除、覆盖、合并双端差异或清理存储必须单独审查和确认。

## 训练方向分轨

### 当前最高优先级（P0）：MindLM 预训练流式数据通路

基于 `60` 上的 `/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl`，优先实现并验证**训练时流式读取 JSONL、边 tokenize/pack 边计算**。本阶段不把离线生成完整 packed `.bin` 作为前置，也不启动正式训练；先完成数据验收、流式数据通路与小规模可复现验证。

验收要求：

1. 只读验收 V6 JSONL 的行数、schema、来源、异常行/空文本/长度分布和重复情况；与 sidecar metadata、merge report 对账。
2. 明确训练/held-out 的确定性切分与防泄漏规则，保留来源和 tokenizer/config/数据指纹；流式消费不能使验证集信息泄漏到训练。
3. 流式 loader 按文档边界追加 `<|im_end|>`，增量 tokenize 并形成 4096-token 训练样本；不要求先落盘完整 tokenized corpus。
4. 先用可复现的小数据/短运行验证顺序、恢复/随机性、loss mask、held-out 评估和内存上界，再决定是否进行正式训练。

当前实现差距：现有 `pretrain.py` 只消费已有 packed `.bin`/`.json`，`PackedPretrainDataset` 使用 memory map；`data_process/pretrain.py` 只接受 CSV 并离线输出 packed 数据。它们不是目标中的在线 JSONL 流式路径，不能直接视作已实现。

| 轨道 | 目标 | 当前判断 | 近期处理 |
|---|---|---|---|
| 自研 MindLM 预训练 | 验证模型架构、数据和语言建模能力 | 当前最高优先级为面向 V6 的训练时 JSONL 流式 tokenize/pack/计算；旧入口仍要求离线 packed 数据 | 先完成上方 P0 的数据验收、流式路径与小规模验证；通过后另行确认正式训练计划 |
| 自研 MindLM 工具调用 SFT | 学会调用决策、工具选择、参数、闭环与拒绝 | 已有 E2、repair 和 pilot 对照没有形成可靠的通用工具调用候选；小规模实验不能证明唯一根因或容量上限 | 先恢复单轮/闭环评测和 CPU 测试；再按明确失败类型设计单变量实验 |
| MiniCPM5 工具调用 SFT | 在固定任务集上验证训练是否带来泛化改善 | 已有候选 dev 结果存在分项回退，尚无候选独立 test 结论；不能把格式专项通过等同于工具能力通过 | 先冻结候选、评测版本与选择规则；按 dev 选点后只对冻结候选做独立 test |

以上是项目状态摘要，不表示自研模型与 MiniCPM5 的分数可直接横向比较。它们的模型、模板、评测集和训练流程不同。

## 资料与实验分类规则

### 当前基线

- **工程基线**：Git 共同 HEAD `160be1f` 及明确列出的两端未提交差异；本地和容器状态分开记录，不互相覆盖。
- **自研训练对照**：E2 终点及 repair/pilot 的相关起点，用于后续同设置比较；已有结果主要是问题诊断，不等于工具能力已达标。
- **MiniCPM5 评测基线**：每个评测版本内的官方/起始模型、模板、题目、运行时、指标和 manifest。不同 suite 之间不拼接分数。
- 基线必须有可定位的 checkpoint 或明确说明只有报告/元数据；不得只凭 run 名称认定模型身份。

### 待验证候选

- 当前记录中的 MiniCPM5 `token100`、`format200` 保留为候选，不预先宣布胜出。
- dev 选择规则、case/suite 哈希、运行时、生成参数和对照模型必须先冻结；选择后保存 selection 文件，再跑独立 test。
- test 未完成前不得依据 test 结果继续挑 checkpoint、改提示或改评分规则。若发现评测缺陷，应标注该轮无效并另建版本。

### 历史与诊断资料

- 失败、回退、作废评测和负面实验继续保存，并保留原始结果、命令、manifest 与作废原因。
- `docs/archive/history/` 保存历史设计；现行结论放在当前项目台账或本路线图中，以链接说明修正，不静默改写旧实验事实。
- 旧文档中的假设若与后续证据冲突，标为“历史假设/待核实”，不作为当前事实引用。
- `runs/`、数据和 checkpoint 是实验产物，不与源码清理或文档整理混做。

分类索引：[`2026-10-16/README.md`](../2026-10-16/README.md)、[`experiments.csv`](../2026-10-16/experiments.csv)、[`2026-10-16/backlog.md`](../2026-10-16/backlog.md)。

## 分阶段计划与验收

### 阶段 0：保全双端现状

1. 本地与 60/容器分别保存 Git 状态、差异、未跟踪清单和 HEAD；不得用一端覆盖另一端。
2. 每一项待纳管代码按功能审查，区分源码、文档、生成数据、模型和日志。

**验收**：双端差异清单完整；关键文件有来源与哈希；未执行清理或覆盖。

### 阶段 1：恢复评测可信度

1. 修复自研工具调用评测入口及缺失依赖；固定单轮和闭环 case/runtime。
2. 全量 CPU 测试通过；实际运行小样本单轮和闭环流程，并保存命令、代码版本、退出码与日志。
3. 明确格式合法、工具名、参数值/grounding、调用/不调用、多步状态与最终回答等分项；不以单一总分掩盖失败。

**验收**：可从新目录按记录命令复跑；无效 case 和历史评测版本有清晰隔离；基础设施失败不会计为模型答错。

### 阶段 2：固定模型与数据身份

1. 为每个可比较候选登记 checkpoint 路径及可用时的内容哈希、训练起点、训练状态、数据和 tokenizer 指纹、配置、模板与评测版本。
2. 分开标记训练最终点、dev 选中点、验证 loss 点和诊断快照。
3. 查清关键外部数据/模型依赖的实际所有者、路径与可用性；不能精确再生的标为不可精确再生，不用近似数据替代原件。

**验收**：任何报告分数都能对应到唯一模型、数据、模板和 evaluator；恢复检查不修改保留产物。

### 阶段 3：执行有边界的候选验证

1. 先对 MiniCPM5 候选按冻结 dev 协议做选择；若没有候选整体胜过起点，保留起点为选择结果。
2. 只对冻结选择做一次独立 test；按预先约定的分项结果报告改善和回退。
3. 自研线只有在评测稳定后，才根据主要错误类型另立单变量实验；一次只改变可解释的一个因素，保留同题/同模板基线与通用能力回归检查。

**验收**：实验计划先于训练冻结；有明确成功、回退和停止标准；未达标就记录负面结论并停止该设置，不以降低标准补救。

## 外部参考资料及适用范围

- [BFCL：Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)，及 [ICML 2025 论文](https://proceedings.mlr.press/v267/patil25a.html)：参考单轮、并行/多步、工具相关性与状态式 agent 评测，以及分项报告。它是评测设计参考，不是 MindLM 的目标分数承诺。
- [Granite Function Calling（EMNLP Industry 2024）](https://aclanthology.org/2024.emnlp-industry.85/)：参考把函数选择、参数抽取、调用顺序、并行/链式调用和最终响应拆成不同训练/分析任务。其规模、底座和数据条件与本项目不同，不能直接照搬配比或效果结论。
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 与 [数据格式说明](https://huggingface.co/docs/trl/dataset_formats)：参考 conversational/prompt-completion 格式、chat template 和 assistant/completion-only loss 的边界。用于审计本项目数据与 loss mask，不要求迁移训练框架。
- 项目内部的 [`MiniCPM5 独立评测合同`](../../../../MindLM-MiniCPM5/docs/minicpm5_eval.md)、[`MiniCPM5 SFT 说明`](../../../../MindLM-MiniCPM5/docs/minicpm5_sft.md)、[`实验台账`](../2026-10-16/README.md) 是本项目执行依据；外部材料不覆盖项目自身测得的负面结果。

## 明确不做

- 本计划不启动新训练，不更改架构、损失、数据配比或 tokenizer。
- 不删除历史 run、checkpoint、报告、缓存或数据，不清理 `.git`、外部 symlink 或备份目录。
- 不将本地和容器工作树强行同步；先审差异，再逐项决定归属和同步方向。
- 不将不同 suite、不同模板或不同评分规则的分数合并排名。

## 完成判断

这轮梳理完成的标准是：训练方向分轨清楚；基线、候选、历史分类有可追溯链接；双端未提交差异受保护；评测恢复、模型身份和候选验证有可执行验收条件。完成计划文档不代表上述工程任务或新训练已经完成。

## 执行进展（增量记录）

- **2026-11-20 阶段 1 完成**：`runs/eval_repair_20261016T030501Z/` 的 10-16 断网中断项全部收尾——strict 历史评分比较 0 未解释差异、candidate 与生产树全量 pytest 89 passed + 440 subtests、CLI ×2 与真实 tokenizer 300-case 回放 300/300 通过，带 base SHA 守卫原子同步 7 文件到生产树。HANDOFF 状态 complete_20261120。
- **2026-11-20 阶段 2 完成（登记部分）**：`runs/stage2_identity_20261120/` 建成模型身份登记册 model-identity-register.json（30 条目，runs/ 全部 MindLM .pt 与 MiniCPM5 权重含 SHA256、角色、谱系；pretrain_epoch2 哈希与 2026-09-22 训练合同完全一致）；tokenizer/config/模板指纹固定；V6 语料在盘完好、v5 packed 仍缺失、repair v2 生成器确认不可精确再生。受控恢复演练未做，列为独立后续任务。
