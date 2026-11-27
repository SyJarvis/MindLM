# MindLM 项目与实验收敛台账

截至 2026-10-16，MindLM 的训练、数据处理和断点恢复已有实现，模型工具调用效果仍需收敛。当前将自研 repair 系列归入负面结果与诊断基线，将 MiniCPM5 token100 和 format200 归入待验证候选。现有独立评测没有选出优于官方 Base 或起点 SFT 的训练版本；保留检查点不代表确认其为最佳模型。

实验明细见 [实验台账](experiments.csv)，后续工作见 [待办与验收标准](backlog.md)，资料位置、校验与恢复见 [资料保全](preservation.md)。本轮新增台账与资料快照，现有源码、历史报告、模型和数据路径保持原状。

## 开发状态

| 工作线 | 已有实现或产物 | 当前管理状态 |
|---|---|---|
| 自研 MindLM | 约 206.8M 参数的 16 层 GDN 混合模型，预训练、SFT、数据审计、断点恢复；attention KV 与 GDN 状态缓存均已实现，生成默认使用缓存 | 工程基线保留；工具调用实验作为负面结果与诊断对照 |
| MiniCPM5 | HF 模型训练、按需分词缓存、完整恢复、独立 dev 选择与 test 评测、格式专项 | 源码与评测套件已移至相邻的 `MindLM-MiniCPM5/` 快照；token100 和 format200 保留为候选 |
| 自研工具评测 | 单轮、闭环历史报告及 300 题修正评测资料 | 当前入口和测试接口断裂，先恢复验收基线 |

当前 CPU 全量测试在收集 `tests/test_tool_loop.py` 时因缺少 `eval.eval_sft_tool.build_summary` 失败。排除该文件后为 **53 passed、27 subtests passed、16 warnings**；这不等于全套通过。闭环入口还引用缺失的 `tool_case_loader.py` 与 `tool_runtime.py`。日志见[资料保全中的验证记录](preservation.md#验证记录)，本轮未重新验证 GPU 数值或吞吐。

两端 HEAD 均为 `160be1f9f304ed09a24c93fbb2a6270eef3ff0d9`，`main` 比各自记录的 `origin/main` 超前 1 个提交；本轮未 fetch。60 上有 4 个已跟踪文件修改，本地有 3 个，未跟踪成果和资料也不相同，因此分别保全两端工作树。

## 资料分类

| 类别 | 范围与保存方式 |
|---|---|
| 正式源码基线 | 模型、训练、数据、评测、测试与依赖文件；保全工作树和 Git 差异，后续按功能审查 |
| MiniCPM5 快照 | `../MindLM-MiniCPM5/`；共享基础设施与历史生成产物仍留在主项目 |
| 实验研究证据 | 60 上 13 个顶层 run 及其日志、命令、manifest、原始评测、负面报告和无效评测说明；按源端分别归档 |
| 模型保留产物 | 60 上 `runs/` 约 148G；现有最终点、候选点和诊断快照继续原位保存，清单不构成模型备份 |
| 数据资产 | 数据格式、生成脚本、审计与哈希；60 上 `full_data_v2` 指向外部 data2 数据目录，大数据与二进制只做清单 |
| 历史与待核实文档 | `docs/archive/history/` 和训练记录保留原文；过时内容与假设在当前台账中明确标识 |
| 备份与临时探针 | `full_data_v2.migration-backup/`、实验内脚本及旧快照单独标识；`.git` 大对象来源待只读审计 |

## 实验收敛结论

- **自研线**：E2、repair v1、v2、from_e2 和 v3 pilot 保留为对照链。不同数据和起点未在这些设置下解决调用与拒绝的取舍；小规模探针不足以确认架构容量上限或唯一退化原因。
- **MiniCPM5 线**：v1 选择官方 Base，960 条 pilot 选择起点 `start`。pilot 的 step400 是验证 token NLL 候选，step500 是该次训练候选中的任务分数最高点，均低于起点的 dev 宏平均。
- **待验证候选**：token100 的 v2 dev 宏平均为 27%；format200 为 33.875%，格式专项 seen 64/64、holdout 96/96 通过，但 v2 不调用仍为 0/50，多步工具从 30/40 降至 19/40；两者都没有独立 test 结论。
- **无效与历史口径**：旧 `formal/` 因 35 道不可解地址题及 50 道缺内容 oracle 的负例排除模型选择，使用修正的 `formal_v2/`。8、28、150、200、300 等不同探针分母分别记录，不拼接排名。

本轮按上述状态收拢资料，优先恢复评测、纳管源码和补齐复现信息；算法修改与新训练列为后续独立任务。

## 源码与证据入口

以下相对链接指向两端共有的项目文件；CSV 中 `local:` 表示本地资料，`ssh60:` 表示服务器资料，远端路径不表示本地存在。

- [当前模型配置](../../../config/mindlm_0.2b_gdn.json)、[缓存实现](../../../modeling_mindlm.py)、[闭环测试](../../../tests/test_tool_loop.py)、[闭环入口](../../../eval/eval_sft_tool_loop.py)。
- [MiniCPM 训练说明](../../../../MindLM-MiniCPM5/docs/minicpm5_sft.md)、[独立评测合同](../../../../MindLM-MiniCPM5/docs/minicpm5_eval.md)、[v1 结果](../../../runs/minicpm5_eval_v1_20260929T095427Z/report.md)、[token100 对照](../../../runs/minicpm5_token100_control_20260930T091217Z/report.md)、[format200 结果](../../../runs/minicpm5_format_pilot_20261002T155024Z/report.md)。
- [repair v2 历史训练记录](../../training/sft_tool_repair_training_record.md)、[10 月 3 日改进计划与后续实验](../../training/tool_call_improvement_plan.md)。其中“未实现 cache”已经过时，“起点错了”与后半段 from_e2 结果矛盾，容量与退化机制判断仍属假设；当前配置文件数量也与旧 README 枚举不同。
