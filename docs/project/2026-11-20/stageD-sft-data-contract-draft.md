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

1. **渲染管线**：messages → `tokenizer.apply_chat_template`（6.4K 线内置 Llama 式模板，已验证可渲染，见 §5 样例）→ loss masking 只对 assistant 回复段计算（需 SFT 训练脚本支持 `assistant_masks` 或等价机制）
2. **长度过滤**：> 4096 token 的样本（0.08%）丢弃（SFT 不做跨样本拼接，保持样本边界完整）
3. **数据规模控制**：860万行全量 ≈ 3.7B token，对 0.2B 模型 SFT 过大。第一轮建议 scope 到 30-60 万样本（1-3 亿 token），来源配比优先中文 QA
4. **Eval 防泄漏（已实测通过）**：minicpm5_eval_v2（dev 200 + test 200）与 sft_e2_combined_v5 全量 8,606,853 行做了消息级 sha256 对比——**对话级重叠 0、user 轮重叠 0**（`runs/stageC_pretrain_20261120/sft_v5_overlap_audit.log`）。评测设计与数据实测双确认无泄漏。
5. **冻结超参**（初稿，待 stageC 预训练完成后校准）：lr 1e-5 ~ 2e-5、1-2 epoch、bs 与 ga 组合与预训练对齐（bs8×ga8 等价全局 batch）、cosine、bfloat16、单 H200

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

1. 全量 source 分布统计（60 宿主 python3，不占 GPU）
2. 300-case 重叠检测 ✅ 已完成（2026-11-20 实测 0 重叠，见 §3 条目4 与 `sft_v5_overlap_audit.log`）
3. SFT 读取管线实现（参照 `data_process/v6_stream_loader.py` 的流式与合同纪律，复用 tokenizer 模板渲染）
4. SFT 训练脚本检查（SFTDataset 兼容 messages schema / assistant masks）
5. CPU 微型冒烟 → 全量回归 → pack → 启动（放行检查同 stageC：GPU<1GB + 磁盘 + 不撞车）
