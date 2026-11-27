# 自研 MindLM 工具调用训练改进计划

> **日期**: 2026-10-03
> **模型**: MindLM 0.2B GDN (206.8M, 12×linear_attention + 4×attention)
> **背景**: 对照 Needle（Cactus Compute 端侧工具调用模型）的公开代码与二进制引擎分析结论，审查自研线的工具调用训练路径
> **参照分析**: `~/research/docs/needle-模型格式与训练目标分析.md`

---

## 1. 现状基线（全部有实测/代码依据）

| 实验 | 起点 | 数据 | 20+8 探针结果 |
|---|---|---|---|
| E2 全量 SFT (step15220 探针) | pretrain epoch2 | E2 v1 4.05M 条 | retail 6 用例探针 call 6/6, name 4/6, overall 5/8 |
| E2 终点 (step15678，2026-10-03 补测) | — | — | **call 20/20, name 5/20, overall 7/28, neg 2/8**；retail overall 2/8, neg 0/2 |
| tool repair full (179 upd) | smoke200_base | repair v2 45.8K | call 2/20, overall 9/28 |
| tool repair redo (1793 upd) | smoke200_base | repair v2 45.8K | call 3/20, overall 10/28 |

闭环多轮探针（调用→mock 执行→回填→终答，2026-10-03 补测）：

| 模型 | 首调 name | 终答产出 | loop_stuck | 闭环 overall |
|---|---|---|---|---|
| E2 终点 (15678) | 2/6 | 0/6 | 7/8 | **0/8** |
| repair redo e19 | 0/6 | 6/6（但不调工具直接幻觉作答，相关仅 1/6） | 0 | 1/8 |

两份模型以互补的方式全挂：E2 终点乱调+卡死循环，redo 不调+直接编答案。单轮探针会做 ≠ 闭环能跑通；后续验收以闭环为主。

三个已确认的错误模式：**过度调用**（E2 终点正例 20/20 全调、负例 6/8 误调）、**函数名坍缩**（见瓶颈 3a）、**参数残缺**（args 6/20）。负例边界在 repair 训练后是 100%（redo neg 8/8）。

## 2. 诊断

### 瓶颈 1：词表吞掉一半参数（结构性上限）

`vocab 151,669 × dim 768 ≈ 116.5M` 参数在 embedding（tied head），总参 206.8M 中占 56%，主干仅 ~90M 做计算。Qwen3 词表为大模型设计，对 0.2B 是反模式；Needle 用 8192 词表把参数几乎全部留给主干。当前模型的"有效智力"只有 90M 主干，这决定了学习"读 schema → 决策调用"这类复杂行为的容量上限。

### 瓶颈 2：repair 数据的措辞分布决定行为，起点不是主因（2026-10-03 实验修正）

原假设：repair 从 `smoke200_base.pt` 起点训练丢了 E2 的调用能力。**已证伪**：从 E2 终点续训 repair v2 后，模型在 1 个 epoch 内就收敛到与 redo 相同的行为（调用率 2–5/20、闭环 0/6、直接幻觉作答），E2 的调用倾向被完全覆盖。真正机制：**模型的调用决策由查询措辞驱动，而非 schema 理解**——repair 数据的 no-call 域（`synthetic_permanent_honest` 1,500 + `high_quality_negative` 4,184）教会了模板化拒绝话术（"我无法实时查询汇率，建议访问 IMF 官网"），这些模板按查询类型触发，把自然措辞的正例也拒掉了（甚至 kg→磅 的题拒成了"无法获取天气"——拒绝模板串台）。

### 瓶颈 3：函数名坍缩（2026-10-03 实测确认）

E2 终点在 20+8 探针上 13+/20 个非天气正例全部调 `get_current_weather`，把任务内容塞进 `city` 参数（`text_len`→`city="Hello World"`、时间查询→`city="Khan Academy"`）；retail 域上坍缩默认换成了 `find_user_id_by_email`/`calculate`——**坍缩默认函数是域相对的，模型学的是"要调用=调最常见函数"而非"读 schema 选函数"**。还出现重复 JSON key（`to_currency` 两次），属解码约束可物理消灭的错误。

成因：E2 v1 的 4.05M 条中高质量工具数据仅 ~83K（约 2%），且函数分布头部集中；无足够的同域干扰项反例。repair v2 数据实测 3,015 个被监督函数、头部均衡（random_number 3445 / weather 3390 / translate 3383 / text_length 3380），探针目标函数全部有 700–3,400 条监督样本——**数据本身是对的，起点错了**。另：E2 训练末期自身也在退化（step15220 retail 5/8 → step15678 retail 2/8，neg 1/2→0/2）。

### 瓶颈 4：监督 token 配比失衡

repair 数据 31.7% 监督 token 中，7.5K 条 legacy 纯对话贡献大份额；no-call 短答案几乎无梯度。MiniCPM5 pilot 的实测（逻辑 0.64% / no-call 1.27% / 单调用 49.2% / 多调用 48.9%）证明了记录数配比 ≠ 梯度配比。决策类（调/不调）信号被稀释，与"该调不调"直接相关。

### 瓶颈 5：推理无 cache，闭环迭代被卡死

模型未实现 KV cache（4 层 attention）与 GDN state cache（12 层线性注意力），生成逐 token 重算全文。多轮闭环评测（`eval_sft_tool_loop.py`）无法快速迭代，拖慢一切实验。GDN 天然适合递推 state cache（FLA 有 fused_recurrent kernel）。

### 瓶颈 6：闭环评测代码在 HEAD 处于断裂状态

清理提交（160be1f）重写了 `eval/eval_sft_tool_loop.py` 并引用不存在的 `tool_case_loader.py`/`tool_runtime.py`，同时删除了 `eval_sft_tool.py` 的 `GENERIC_TOOLS`/`RETAIL_TOOLS` 导出——闭环评测在当前 commit 无法运行。本次基线用父提交（160be1f^）的自洽版本在容器 `/tmp/legacy_eval` 恢复运行。**建议修复仓库后再迭代闭环**（修复属仓库变更，另行提交）。

## 3. 改进方案（按优先级）

### P0：数据与流程（不动模型）

1. ~~修正训练链路~~（已实验证伪：起点不是主因，见 5.3 节）。改为：**正例措辞多样化**——同一意图的自然语言变体 × 工具类型，覆盖探针/真实查询分布；拒绝话术同样要多样化，避免模板化触发与串台。
2. **同措辞空间的 call/refuse 对比对**：同一句式下有/无对应工具、够/不够参数两种样本成对出现，迫使决策依赖 schema 而非句式（repair 的 no-call 三分法方向对，但措辞必须与正例同一分布）。
3. **E2 v2（40GB, 5.82M 条）训前配比重整**：legacy 占 68%，直接全训必重演"通用好工具差"。分层采样将工具轨迹提到 20–30% 样本份额，并按监督 token 份额监控（复用 pilot 审计方法）。
4. **域外场景合成扩充**：repair 探针的 20 个通用场景就是分布缺口。复用 MiniCPM5 线生成题族方案，但**措辞覆盖自然查询变体**（这是本次实验的直接教训）。

### P1：训练目标（Needle 式，改训练代码不动架构）

4. **解码约束（单项最大杠杆）**：评测侧 schema validator 已完备，升级为生成时 logits mask（xgrammar/outlines 或手写 JSON 子集语法）。收益：schema 违反长尾归零 + 0.2B 容量全部让给"何时调/调哪个/参数语义"。约束作用在 lm_head，GDN 混合架构不受影响。
5. **每轮强制首 token + 一行溯源**：渲染 assistant 前缀固定（如空 `<think>` 或直接 `<tool_call>`），可选一行参数溯源训练。Needle 实证 reasoning 是监督目标的一部分。[需消融]
6. **决策类监督加权**：用 `--loss_normalization example` 或类加权，保证决策类监督 token ≥ 15–20%。
7. **干扰项对比对**：语义近似工具同场（`find_user_id_by_email` vs `get_user_details`），配合已有 called-first + max_tools=8，逼模型读 schema 而非域偏好。
8. **grounding 闭环**：数据侧 `strict_grounding` 已实现（与 Needle 引擎检查同构）。补：评测侧"参数值出处"维度 + ungrounded 负例（编造参数 → 拒绝/修正）。

### P2：结构改造（需重训预训练）

9. **词表瘦身 151K → 32–50K**：基于 E2 v2 语料频率重训 BPE（保留中文高频 + `<tool_call>` 等结构 token 为特殊 token）或按频率截断。embedding 116M → ~38–58M，主干 +60–80% 或总参降至 ~150M。
10. **推理 cache**：GDN 递推 state cache + attention KV cache（此项决定迭代速度，优先级实际等同 P0）。
11. **校准置信度头**：Needle 的 ConfidenceHead 是探针池化小头（几十 K 参数）。repair 的 no-call 三分法（上下文足够/缺参/无能力）天然是校准训练集，产出 act/confirm/refuse 三态路由。

### P3：远期可选

- 弹性深度（ladder 式训练 + 端点保持切片导出，混合架构需保持层型平衡）。
- QAT 量化对齐（若部署 int4/int8，训练末期 fake-quant）。
- 统一两线评测口径：MiniCPM5 线确定性闭环（固定 oracle + 离线回放 + family 持出）迁移到自研线格式。

## 4. 预期路径

```
第 1 步（已完成 2026-10-03）：从 E2 续训 repair → 假设证伪，定位措辞依赖问题
第 2 步（下周）        ：重生成 repair 数据（正例措辞多样化 + 同措辞 call/refuse 对比对）
                        → 2–4 epochs 续训 → 三套探针验收
第 3 步（1–2 周）      ：E2 v2 配比重整 + 场景族合成 + 监督加权 → 域外探针
第 4 步（2–4 周）      ：解码约束上线 + 干扰项对比对 + grounding 评测维度
第 5 步（一个月+）     ：小词表重训预训练 + cache + 置信度头
```

第 2 步之前不动架构——当前所有症状（过度调用/拒绝模板串台/闭环全挂）都是数据分布问题，不是 0.2B 容量问题。解码约束（第 4 步）可提前到第 2 步并行开发，因为它不依赖训练结果。

## 5. 第 1 步执行记录（2026-10-03，60 服务器 H200）

### 5.1 基线补测（训练前）

```bash
# E2 终点从未在 20+8 探针上测过；retail 基线用归档用例重放（retail_probe.py）
python eval/eval_sft_tool.py \
  --checkpoint runs/sft_v3_e2_20260922/eval/final_dialogue/mindlm_sft_final_update15678.pt \
  --device cuda:0 --json_out runs/sft_t2t_tool_repair_v2_from_e2/eval/e2_final_probe.json
python runs/sft_t2t_tool_repair_v2_from_e2/retail_probe.py \
  --checkpoint runs/sft_v3_e2_20260922/eval/final_dialogue/mindlm_sft_final_update15678.pt \
  --json_out runs/sft_t2t_tool_repair_v2_from_e2/eval/e2_final_retail.json
```

结果（见上表）：E2 终点 call 20/20 但 name 5/20、负例 2/8；retail overall 2/8、负例 0/2。

### 5.2 单变量对照训练（唯一差异：resume_from = E2 终点）

```bash
python3 full_sft.py \
  --bin runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled_tokens_4096.bin \
  --meta runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled_tokens_4096.rows.jsonl \
  --model_config mindlm_0.2b_gdn \
  --out_dir runs/sft_t2t_tool_repair_v2_from_e2/out \
  --resume_from runs/sft_v3_e2_20260922/eval/final_dialogue/mindlm_sft_final_update15678.pt \
  --resume_weights_only \
  --token_budget 524288 --accumulation_steps 2 \
  --learning_rate 1e-6 --warmup_iters 50 --grad_clip 1.0 \
  --epochs 20 --num_workers 8 --max_batch_count 2048 \
  --gradient_checkpointing all --dtype bfloat16 \
  --save_interval 200 --log_interval 10 --seed 1337
```

训练后自动跑 epoch1/4/9/19 × 双探针曲线 + 闭环曲线 + 对话 sanity。

### 5.3 实测结果（2026-10-03）

**20+8 探针曲线**（温度 0.1 / seed 1337，与历史记录可比）：

| 模型 | call | json | name | args | overall | 负例不调 |
|---|---|---|---|---|---|---|
| E2 终点（基线） | 20/20 | 20/20 | 5/20 | 6/20 | 7/28 | 2/8 |
| from_e2 epoch1 | 2/20 | 2/20 | 0/20 | 0/20 | 7/28 | 7/8 |
| from_e2 epoch4 | 4/20 | 4/20 | 1/20 | 1/20 | 9/28 | 8/8 |
| from_e2 epoch9 | 5/20 | 5/20 | 1/20 | 1/20 | 9/28 | 8/8 |
| from_e2 epoch19 | 2/20 | 2/20 | 1/20 | 1/20 | 9/28 | 8/8 |
| （对照）redo e19 | 3/20 | 3/20 | 2/20 | 2/20 | 10/28 | 8/8 |

**闭环多轮**（legacy 套件，8 用例）：

| 模型 | 首调 name | 工具执行 | 终答产出 | 终答相关 | 闭环过 | stuck | 总过 |
|---|---|---|---|---|---|---|---|
| E2 终点 | 2/6 | 1/6 | 0/6 | 1/6 | 0/6 | 7 | 0/8 |
| from_e2 epoch1/4/9/19 | 0–1/6 | 0–1/6 | 6/6 | 0–1/6 | 0/6 | 0 | 1/8 |
| （对照）redo e19 | 0/6 | 0/6 | 6/6 | 1/6 | 0/6 | 0 | 1/8 |

**retail 回归**：from_e2 各 epoch 均 2/8–3/8（E2 终点 2/8，step15220 为 5/8），未恢复。**对话 sanity**：epoch19 与 E2 终点同水平（Q1/Q2 正常、Q3 重复退化），无额外损伤。

### 5.4 结论

1. **原假设证伪**：起点不是主因。从 E2 续训 1 个 epoch 内，repair 数据就把 E2 的 100% 调用倾向完全覆盖，收敛到与 redo（smoke200 起点）相同的行为——调用率 2–5/20、闭环 0/6、直接幻觉作答。**repair v2 数据本身压制调用，与起点无关**。
2. **负例边界可学且不依赖起点**：单轮 8/8、闭环负例 1/2，epoch1 即生效。
3. **机制定性**：调用决策由查询措辞驱动。repair 的 honest-refusal 域教出了模板化拒绝（kg→磅的题拒成"无法获取天气"，模板串台），自然措辞的正例被拒。模型没有学会"读 schema 判断能力"，学的是措辞→模式的映射。
4. **20 epochs 过多**：epoch1 后各项指标基本不动，后续实验 2–4 epochs 足够。
5. 验收标准 1/2/3 均未达成，但实验定位了真正的杠杆：**数据措辞多样性 + 同措辞 call/refuse 对比对**（见 P0 修订）。所有评测产物在 `runs/sft_t2t_tool_repair_v2_from_e2/eval/`。

## 6. 第 2 步执行记录（2026-10-03 晚，repair v3 pilot 自造数据）

### 6.1 repair v2 数据措辞审计（动机证据）

v2 preparer 已从仓库丢失（从未提交，pyc 常量反汇编确认其结构），直接审计产物 CSV（`runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled.csv`）：

| 域 | 正例问法 | 拒绝话术 | 问题 |
|---|---|---|---|
| call_required_grounded (29,378) | 英文 API 腔（"Fetch the last events for team ID 465171"） | — | 与自然问法零重叠 |
| high_quality_negative (4,184) | 中文自然长句（多任务串联） | **"目前我无法直接搜索/获取实时XX内容…"模板族** | 与探针问法同语域，正好教出模板拒绝 |
| 同上部分样本 | 公里→英里（无换算工具时） | **用公式直接算**（"英里=公里×0.621371"） | 教会 convert 题直接作答 |
| synthetic_permanent_honest (1,500) | — | 固定句 "The tool failed, so the task is not complete. Please try again later." | 极端模板化 |

### 6.2 v3 pilot 设计（`runs/sft_t2t_tool_repair_v3_pilot_20261003/gen_data.py`）

- 10 个探针函数族（weather/exchange/time/news/search/translate/textlen/convert/math/random），各族配同域干扰工具
- 样本类型：P 闭环正例 2,400（调用→工具结果→接地终答）｜N1 无工具孪生 1,800（**同一句 query 原样复用**，事实族→多样化拒绝，确定族→直接计算）｜N2 缺参数追问 800｜N3 闲聊带工具 200｜N4 明确禁用工具 120
- 参数值全部"表面接地"（抄问句原文，0.2B 无需学 韩元→KRW 映射）；zh:en ≈ 59:41；终答模板每族 3–4 变体；拒绝 5+ 变体（含说明可用能力）
- 防 probe/retail/loop 泄漏：34 条评测 query 归一化精确匹配拒绝（命中 0）
- 结构审计（audit.py）：P 参数接地、终答含 payload 值、N1 无目标工具泄漏、N2/N4 形状——5,320 行 0 错误；convert 真值 100%

### 6.3 两轮训练（同一起点 E2 终点、lr 1e-6、warmup 20、12 epochs、36 updates、4.1 min）

**第 1 轮（v1，作废）**：P 样本目标函数恒排第一位 → 模型学到"调清单第一个工具"的位置捷径，epoch1 探针 20 例全调 `get_current_weather`（探针清单第一位恰是它）。教训：**工具顺序必须随机化**（已修，P 目标首位占比 41%）。

**第 2 轮（v2，正式）** 20+8 探针曲线（每 epoch 仅 3 updates）：

| epoch | call | name | args | 总过 | neg |
|---|---|---|---|---|---|
| 1 | 18/20 | 4/20 | 5/20 | 5/28 | 1/8 |
| **2** | **11/20** | 2/20 | 3/20 | 6/28 | **4/8** |
| 3 | 5/20 | 1/20 | 2/20 | 6/28 | 5/8 |
| 7 | 1/20 | 0/20 | 0/20 | 7/28 | 7/8 |
| 11 | 1/20 | 0/20 | 0/20 | 8/28 | 8/8 |

retail：e1 call 6/6 → e3 5/6（pass 4/8，历史最佳区间）→ e7 起 0/6。闭环：e3 loop_pass 1/6（罕见非零）。epoch3 生成文本出现语域损伤（word salad：echo 问句、混语种胡话）——pilot 无 legacy replay 锚，纯合成模板 36 个大步就把 0.2B 拽离通用语域。

### 6.4 工具块形状消融（mini_probe.py：full10/shuffled/small3）

epoch3：call 5/20 → 乱序 9/20 → 3 工具 6/20（位置偏置约 2×，次要）；epoch11：三种全灭。**工具块规模/顺序不是主因**。small3 中一旦决定调用，name 多半正确（math/convert/exchange ✓）——名字能力未死，调用决策在临界点摇摆。

### 6.5 结论（假设二证伪 + 结构性定性）

1. **"措辞配对"假设证伪**：正负例同措辞、同语域、同语言、位置随机化、参数接地、多样化拒绝——全部到位后，call/neg 仍沿同一根轴单调互换（18/1 → 11/4 → 5/5 → 1/8），**不存在双赢 checkpoint**。三套数据（repair v2 redo / from_e2 / v3 pilot）同构收敛。
2. **函数名 ≤5/20 是全程地板**：E2 终点、repair v2 全系、v3 两轮，name_accuracy 从未超过 5/20。函数名选择能力从未建立（E2 历史的 weather 吸引子 + GDN 线性注意力对 800+ token 前工具块的衰减嫌疑）。
3. **定性**：0.2B GDN 在自由解码下，call/refuse 是一个全局开关而非逐 query 判别；数据怎么配只能滑动它，不能劈开它。继续在数据侧投入边际收益已尽。
4. **P0 改道**：解码约束（schema→logits mask，xgrammar/outlines 路线）升为最高优先——它机械地保证 name∈工具清单 + JSON 合法，把"选名字"从模型能力里拿走，训练只需学"调不调 + 填参数"。Needle 三道闸门的同一哲学。
5. 数据侧保留两条廉价经验：工具顺序随机化（防位置捷径）；纯合成数据必须混 replay 锚（防语域损伤）。
6. 产物：`runs/sft_t2t_tool_repair_v3_pilot_20261003/`（gen_data.py、audit.py、mini_probe.py、两轮 out/eval、manifest）。
