# 阶段D SFT 数据合同（草案 v0.1）

> 状态：草案，GPU 空闲前利用等待期完成的前期调研成果。预训练（阶段C）完成、冻结 checkpoint 后再最终定稿并执行。
> 日期：2026-11-20
> 关联：`docs/project/2026-11-20/refocus-plan.md`（主线决定：重预训练+SFT，tokenizer 双线保留但主线为 mindlm_tokenizer 6.4K）

## 1. 目标

从冻结的 stageC 预训练 checkpoint 出发，训练 mindlm_0.2b_gdn 的第一个正式 SFT 模型，使 6.4K 词表线具备基础对话能力。验收沿用已修复的 `eval/` 300-case 评测。

## 2. 候选数据盘点（60 服务器实测 2026-11-20）

### 2.1 主数据候选：`sft_e2_combined_v5.csv`（旧 E2 SFT 主数据，复用）

- 路径：`/mnt/data2/reexen_datasets/llm_training/MindLM_full_data_v2/sft_e2_combined_v5.csv`
- 规模：15.7GB，**8,606,853 行**，schema `messages,tools,source,domain`（messages 为 JSON 数组，与 tokenizer 内置 chat_template 直接兼容）
- **全量 source 分布（实测，`runs/stageC_pretrain_20261120/sft_v5_source_stats.json`）**：

| source | 行数 | 占比 |
|---|---|---|
| t2t_qa（中文 QA） | 3,988,894 | 46.3% |
| legacy（中文 QA） | 3,600,720 | 41.8% |
| perfectblend_en（英文） | 797,285 | 9.3% |
| ultra（英文多轮） | 134,661 | 1.6% |
| xlam_fc（工具调用） | 49,749 | 0.58% |
| multilingual_zh | 24,979 | 0.29% |
| pb_zh | 9,829 | 0.11% |
| toolace（工具调用） | 736 | 0.009% |

  中文合计 ≈ 761.4万行（88.5%），英文 ≈ 93.2万（10.8%），工具调用 5.05万（0.59%）
- **mindlm_tokenizer 适配性已实测验证**（1/700 抽样，12000 行，chat 模板渲染 + `add_special_tokens=False`）：
  - **UNK = 0（0/5,237,611 token）** —— 与 V6 预训练语料结论一致：6.4K 词表 + byte-level BPE 无 UNK 没有残留
  - 长度分布：mean 436 / p50 284 / p90 902 / p99 2313 / max 5413；**>4096 仅 0.08%**
  - 抽样 12000 行 JSON 解析零失败
- 注意：该数据是旧 E2 SFT（qwen3 模板线）的主力数据。旧线数据用作新线 SFT 训练数据不存在污染问题；但评测 300-case 若与其中样本重叠，评测满分不能证有效性——**定稿前需检查 300-case 与该数据的重叠**（hash 对比）。

### 2.2 原始件：open-perfectblend parquet（1.4G，142 万条，8 源）

- 路径：`/mnt/data2/reexen_datasets/llm_training/sft/open-perfectblend/data/train-*.parquet`
- schema：`conversations`（from: human/gpt, value）+ `source`
- 源构成：MetaMathQA 38.6万 / UltraInteract 27.2万 / ultrachat_200k 20.8万 / orca-math 20万 / ultrafeedback 12.5万 / evol-codealpaca 11.1万 / AutoIF 6.1万 / lmsys-arena 5.7万
- **英文为主**。旧管线（`convert_perfectblend_to_csv.py`、`convert_perfectblend_to_sft.py`）将其转成 ChatML 文本与 `history,q,a` schema。旧 MindLM/data/ 下 3 个转换产物 symlink 已断（MindLM_bask 目录已不存在）。

### 2.3 备选：UltraData-SFT-Agent-2609（51G，SWE-agent 长轨迹）

- 路径：`/mnt/data2/reexen_datasets/llm_training/sft/UltraData-SFT-Agent-2609/data/{Code,General,Search_Agent,Tool_Use}`
- 内容：SWE-bench 风格 agent 轨迹（system 含工具 XML 说明、uploaded_files、PR description 等长上下文）
- 判断：**对 0.2B 模型第一轮 SFT 过重**（长 system + 多轮工具轨迹，token 效率低且与 6.4K 词表场景不匹配），记录备用，第一轮不做主力。

### 2.4 断链的旧产物

`MindLM/data/{perfectblend_sft_zh.jsonl, sft_data_multi.csv, sft_data_single.csv}` 三个 symlink 指向 `~/workspace/MindLM_bask/data/`（已不存在），实际为空。不可用。

## 3. 数据处理合同要点（定稿前需确认）

1. **渲染管线（已实现）**：`data_process/sft.py --template_style mindlm` 走 mindlm_tokenizer 内置模板渲染 + `sup` 区间预计算（实测：300 行真实数据全保留，sup 精确覆盖 assistant 体 + `</s>`，<s>user 段不计 loss）；训练侧 `trainers/sft_grouped.py` 直接消费 bin+rows.jsonl，无需改动
2. **长度过滤**：> 4096 token 的样本（0.08%）丢弃（SFT 不做跨样本拼接，保持样本边界完整）
3. **数据规模（已实测 pack 完成）**：正式采样 830,955 行（seed 20261121，zh 7% / en 10% / 其余全留）→ tokenize 后 **kept 721,407 行 / 340,122,872 token（其中 answer 250,701,795）**，产物 `runs/stageD_sft_20261121/pack/train.{bin,rows.jsonl,stats.json}`（2026-11-21 实测）。长度 p50 294 / p90 1053 / p99 2563 / max 4096。
   - **丢弃明细（实测按源）**：over_long 1,140（0.14%）；errors 108,408 几乎全部来自 ultra 的 107,311 行（79.9% NOSUP，工具 8.4 万行按模板降级为 tools_unrendered 计数、不丢弃）
   - **ultra NOSUP 根因（实测诊断）**：ultra 是 search-agent 轨迹，~80% 行为 `system,tool,assistant,...` 无 user 轮；mindlm 模板只渲染 user→assistant 转换（tool/system 轮跳过），这些行渲染为破碎文本且无 `<s>assistant` 头。**丢弃是正确防护**（防破碎文本进训练）；有 user 轮的 27,093 行正常训练。若未来需要 agent 轨迹数据，须改模板加 tool/system role 支持——但那会改变与预训练 V6 渲染的一致性，属后续独立决策。
4. **Eval 防泄漏（已实测通过）**：minicpm5_eval_v2（dev 200 + test 200）与 sft_e2_combined_v5 全量 8,606,853 行做了消息级 sha256 对比——**对话级重叠 0、user 轮重叠 0**（`runs/stageC_pretrain_20261120/sft_v5_overlap_audit.log`）。评测设计与数据实测双确认无泄漏。
5. **冻结超参（已随 launch_formal.sh 落地）**：lr 1e-4、2 epoch、token_budget 16384 / max_batch_count 32 / accumulation 8（等效全局 batch ≈ 8×32 中位长度序列）、warmup 100、cosine、bfloat16、单 H200。参照旧 E2 SFT（lr 5e-5 → 1e-5 区间）与新预训练起点（lr 6e-4 收敛后），1e-4 为首轮 SFT 的中间起点，训练中观察 loss 前 500 update 走势可回滚重跑（checkpoint 合同支持）。

## 3.5 每源处理规则（草案，已按全量统计更新）

| 来源 | 全量行数 | 处理 |
|---|---|---|
| t2t_qa（中文 QA） | 3,988,894 | 主体，直接用 |
| legacy（中文 QA） | 3,600,720 | 主体，直接用 |
| perfectblend_en | 797,285 | ~10% 配额采样，保持英文能力 |
| ultra（英文多轮） | 134,661 | 直接用 |
| xlam_fc / toolace（工具调用） | 50,485 | 全量保留，格式参考 minicpm5 模板的工具 XML 段 |
| multilingual_zh / pb_zh | 34,808 | 全量保留 |

## 4. 决定点（等预训练完成后与用户确认）

1. SFT 起点确认：stageC 冻结 checkpoint（唯一入口，不再有其他起点）
2. 数据 scope：全量 860万 vs 子集 30-60万（影响训练时长数倍）
3. 工具调用能力是否纳入第一轮 SFT（xlam_fc+toolace 抽样仅 72/12000 ≈ 0.6%，全量占比待统计）
4. 是否需要 chat_template 增强版（加入 system 角色支持；现模板仅在首个 message 为 system 时渲染其 content，无标记包裹）
   - 现模板：`system_content<s>user\n...` —— 无包裹直接前置
   - 可选强化：显式 system 段（`<s>system\n...`）或延续 MiniCPM5 做法；**Draft 建议第一轮保持现模板不动**——tokenizer 是预训练使用的同款，SFT 阶段不改模板，保持与预训练渲染一致
   - 可选强化方案记录在 §6，第一轮不用
5. minicpm5 线（130K 线）的 SFT：本轮不做，待 6.4K 线验证后另行立项

## 5. 模板渲染样例（实测）

mindlm_tokenizer 内置 chat_template 渲染多轮对话（jinja2 sandbox 本地实测）：

```
You are a helpful assistant.<s>user\n你好\n...
→ 'You are a helpful assistant.<s>user
你好</s>
<s>assistant
你好！有什么可以帮你？</s>
<s>user
1+1=?</s>
<s>assistant
2</s>
```

- 无 `<|im_start|>`/`<|im_end|>` token（6.4K 词表无此 token，ABSENT 已实测）——SFT 渲染必须走 tokenizer 内置模板，不能用 ChatML 模板
- 工具调用渲染参考 `chat_templates/minicpm5_no_think.jinja`（60 上，XML 工具段 + `<think>\n\n</think>` 前缀设计），第一轮 SFT 不启用 think 块
- UNK=0 已实测（§2.1），6.4K 词表对该数据无覆盖缺口

## 6. 备用：system 段强化方案（第一轮不启用）

若后续需要显式 system 段（例如让模型区分系统指令与用户输入），两个选项：

- 选项A：改 tokenizer_config.json chat_template，加 `<s>system\n{...}</s>\n` 段——但这会改变预训练/SFT 渲染一致性，需重训或接受首 token softly mismatch
- 选项建议：**不做**。第一轮 SFT 与预训练保持同一模板（same tokenizer, same template, same special token 语义），SFT 后评测再定

## 7. 下一步（阶段C 完成后）

1. 全量 source 分布统计 ✅（`sft_v5_source_stats.json`）
2. 300-case 重叠检测 ✅（2026-11-20 实测 0 重叠，见 §3 条目4 与 `sft_v5_overlap_audit.log`）
3. SFT 渲染管线 ✅（`data_process/sft.py --template_style mindlm`：commit dc54096）
4. SFT 采样管线 ✅（`data_process/stageD_sample_mix.py` + 4 测试：commit 5137762）
5. 训练侧数据合同冒烟 ✅（2026-11-21 实测：GroupedSFTDataset 载入 721,407 记录、32,419 micro-batches/epoch、padding 5.7-8.2%）
6. 启动：`runs/stageD_sft_20261121/launch_formal.sh`（GPU<1GB + 阶段C train.exit=0 + pack 完整性三重门禁；lr 1e-4、2 epoch 冻结）

## 8. 验收边界（2026-11-21 实测）

mindlm_tokenizer 的 chat_template 忽略 `tools` 参数（渲染不含工具定义段，`add_generation_prompt` 正常）。这与训练数据的 tools-free 降级（§3.5）一致：模型从未在训练中见过"读 schema → 发 XML"，因此第一轮 SFT 后跑 minicpm5_eval_v2 的工具类 case（single_tool/stateful_tool，200 题依赖工具定义注入）会系统性低分——**属预期边界而非回归**。第一轮验收以 logic + no_call（200 题）与格式/指令遵循为主。工具能力路线（模板加 tools 段或 renderer 注入）留待第二轮；改 render_prompt 属于改评测协议，须冻结存档后明示，不得静默调整。
