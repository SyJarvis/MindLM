# MindLM 新主线：重新预训练 + SFT（收敛版计划）

日期：2026-11-20。本文取代之前的发散工作分轨（预训练流式 P0 / 工具调用修复线 / MiniCPM5 候选线并行推进的状态），收敛为单一主线：**用固定结构重新预训练 MindLM，然后做 SFT**。旧主线产物不迁移、不比较，只归档。

## 主线决定（用户 2026-11-20 指令）

1. **两套 tokenizer 并列保留**：`tokenizers/mindlm_tokenizer`（6,400 词表，自研小词表，新预训练用）+ `tokenizers/minicpm5_tokenizer`（130,072 词表 + chat_template，SFT/对话线保留）。
2. **模型结构保留**：`mindlm_0.2b_gdn`（GDN×12 + Attn×4，16 层，dim 768，context 4096，约 206.8M 参数 151K 词表口径）。`modeling_mindlm.py` 与 `config/mindlm_0.2b_gdn.json` 不改。
3. **重新预训练 + SFT，之前的状态全都不要**：旧 checkpoint、旧 run、旧候选不作为新主线的起点、对照或证据。保留在原位仅作归档。

## 新预训练合同（待启动前冻结指纹）

| 项 | 决定 | 状态 |
|---|---|---|
| 模型 | mindlm_0.2b_gdn 原结构（GDN×12+Attn×4, dim768, ctx4096） | 固定 |
| Tokenizer | `tokenizers/mindlm_tokenizer`（6,400 BPE；embedding ~4.9M vs 151K 词表 116.8M） | 固定，指纹见 tokenizers/README.md |
| 预训练语料 | 60:`/mnt/data2/reexen_datasets/llm_training/pretrain/pretrain_v6_merged.jsonl`（31.3GB、7,217,908 行，在盘完好） | 固定 |
| 数据通路 | 训练时流式 JSONL tokenize/pack（10-23 roadmap P0 的通路，**尚未实现**） | 待实现 |
| 切分/防泄漏 | 确定性 train/held-out 切分、追加 `<|endoftext|>` 边界（按 mindlm_tokenizer 的边界 token 重新核验）、保留来源指纹 | 待设计验收 |
| SFT 数据 | 新预训练完成后另立合同；chat 模板线用 minicpm5_tokenizer 的模板作参考 | 待预训练完成后设计 |

## 实施阶段

### 阶段 A：结构收敛（本次完成）

- `tokenizers/` 双 tokenizer 入库（本地+60，SHA 双端核验一致）
- 本文档固定三条主线决定
- 旧主线文档归档见 §旧主线处理

### 阶段 B：流式数据通路（下一步，唯一开发项）

在 60 容器（data2 有 11T 余量）实现并验收：
1. 只读验收 V6 JSONL（行数/来源/异常行/长度分布/重复）——meta 与 merge report 对账
2. 确定性 train/held-out 切分规则 + 防泄漏（按行号/哈希切分，种子固定）
3. 流式 loader：增量 tokenize + 4096-token pack，追加边界 token
4. 小规模可复现验证：顺序、恢复/随机性、loss mask、held-out 评估、内存上界
5. **词表有效性预检**：用 mindlm_tokenizer 对 V6 抽样 tokenize，实测 coverage/fertility（是否存在大量 UNK、中英 fertility 是否合理）——6,400 词表对 V6 的适配性是本合同的前置事实，不通过则回到词表决定重新评审

### 阶段 C：启动预训练（需用户确认后执行）

合同指纹冻结（tokenizer/config/数据切分种子/入口参数）→ 小规模验证 run → 正式 run。此阶段才会产生新的 checkpoint，全部写入新身份登记册（延续 stage2 格式，run 目录带独立 manifest）。

### 阶段 D：SFT

预训练 endpoint 确定后：SFT 数据合同、模板选择（minicpm5 模板线 vs mindlm 自有模板）、训练与评测。评测入口用已修复的 `eval/`（阶段 1 成果）。

## 旧主线处理

- **归档不删除**：60 `runs/`（148G）、身份登记册 `runs/stage2_identity_20261120/`（30 条目）、实验台账 `docs/project/2026-10-16/experiments.csv`、路线图与 HANDOFF 全部原地保留
- **不迁移不比较**：新主线不引用旧 checkpoint 作为起点；旧探针分数不作为新基线
- **评测资产复用**：`eval/eval_sft_tool*.py` + `tool_cases_v1.jsonl`（300 case，已修复并验证）是跨主线可复用的基础设施，不在归档范围
- **qwen3_tokenizer/ 保留原位**：属旧主线历史资产（151K 词表），新主线不使用；是否移入 archive 待后续清理决策
- 旧主线"不可精确再生"清单（v5 packed 缺失、repair v2 生成器丢失、外部 symlink 依赖）继续有效，见于 stage2 报告

## 启动边界

- 本计划是收敛声明，不是训练启动授权；阶段 C 启动前需单独确认
- 阶段 B 的开发与验收（CPU/小样本）不需要训练授权，可先行
- 新主线一切决定以本文件为准；与旧文档冲突处以本文为准
