# 阶段C→D 全链收尾报告：预训练、SFT 与 400-case 评测（2026-11-22/23）

> 状态：正式执行完成。本报告覆盖阶段C 正式预训练、阶段D 正式 SFT、400-case 受控评测与两个执行层缺陷的定位与修复。
> 日期：2026-11-22 ~ 2026-11-23
> 关联：`docs/project/2026-11-20/stageD-sft-data-contract-draft.md`（数据合同与验收边界）、`docs/project/2026-11-20/refocus-plan.md`

## 1. 执行总览

| 阶段 | 启动 | 完成 | 结果 |
|---|---|---|---|
| C 预训练 | 11-21 14:59（门禁自动） | 11-22 18:33 | exit=0，update 34,887/34,887，最终 val_loss 1.2949（全程单调降 8.76→1.295） |
| C→D 衔接 | — | 11-22 20:29 | watcher（宿主部署）检测共卡租户退卡，三重门禁通过后自动启动 D |
| D SFT | 11-22 20:29（watcher 自动） | 11-22 23:20 | exit=0，update 8,106/8,106（2 epoch），epoch0/epoch1/latest 三 checkpoint 落盘 |
| 400-case 评测 | 11-23 01:08（修复版） | 11-23 01:52 | dev 0/200、test 0/200；失败形态量化见 §4 |

- 训练全程无人值守：C 的 exit=0 → watcher 等退卡 → D 自动起跑 → 评测手动起跑（模板修复后）。
- 共卡纪律：C/D 全程与 zeyasong（70-96GB 波动）、liqiang（4.9GB）共存，峰峰值显存 4.6GiB（C）/3.1GiB（D），无 OOM、未动任何他人任务。

## 2. 阶段C 结果

- 数据：`runs/stageC_pretrain_20261120/full_data/`（2,232,761 train 行 / 10,852 heldout / 9.15B token，字节级 EXACT）
- 超参：lr 6e-4 cosine、warmup 2%、batch 8×8 accum、`--boundary_token </s>`，mindlm_tokenizer 6.4K
- 产物：`mindlm_pretrain_mindlm_0.2b_gdn_epoch0.pt` + `_latest.pt`（各 1.14GB）
- wandb offline：`offline-run-20261121_070420-3cfyx2ov`（云端 sync 待定）

## 3. 阶段D 结果

- 数据：`runs/stageD_sft_20261121/pack/`（721,407 行 / 3.40亿 token，answer 2.51亿；工具行 8.4万 tools-free 降级；ultra 无 user 轮 10.7万行丢弃）
- 超参：lr 1e-4、2 epoch、token_budget 16384、resume_weights_only 自 C latest
- 训练健康：SFT loss 1.16-1.56 区间（自 pretrain 1.29 基础上贴合 SFT 数据），grad_norm 0.40-0.54
- wandb offline：`offline-run-20261122_122923-qhdxj4bi`

## 4. 400-case 受控评测（关键结论）

**协议**：冻结评测器 `eval/eval_minicpm5_suite_v2.py`（判定/manifest/summary 原样），模型入口经 `eval/run_v2_suite_mindlm.py`（.pt 直读，见 §5），CPU float32（共卡租户占满 GPU 算力），dev+test 各 200。

**结果：dev 0/200、test 0/200。**

失败形态（dev 200 例人工分类）：
| 形态 | 数量 | 说明 |
|---|---|---|
| 截断（512 token 无 EOS） | 131 | 生成不终止，多为循环复述 |
| 空响应 | 33 | 首次生成即 EOS 或空 |
| 有内容但偏题 | 43 | 复述题面候选词/答非所问，无一命中公开规格格式 |

- 各分类（logic 60 / no_call 50 / single_tool 50 / stateful_tool 40）全部 0 分，logic+no_call 也不例外。
- stop_correct 48%（dev）/56%（test）——仅在"是否该停下"维度接近随机。
- **对照证据（重要）**：模型基础对话能力正常——`eval_sft.py` 内置 3 问探针（你好/AI 介绍/如何学编程）回答连贯、正常 EOS 终止，"我是MOSS，一个智能助手"式自我介绍完整。**受控规格答题失败是 0.2B/6.4K 词表模型的真实能力边界，不是训练或管线故障。**

### 4.1 v1 错配对照（为什么先得 0 分、修复后仍 0 分但含义不同）

- v1 评测 wrapper 误用 `minicpm5_io.configure_tokenizer`（把 tokenizer 换成 MiniCPM5 ChatML 模板，130K 线产物）；训练 pack 实际用 mindlm_tokenizer **内置模板**（`<s>user\n…</s>\n<s>assistant\n`）。
- 训练渲染与评测渲染的 byte-exact 一致性已验证（重渲染 pack 首行逐 id 比对 prefix match=True）。v1 结果归档 `runs/stageD_sft_20261121/eval_400_v1_chatml_mismatch/`。
- 修复后 0 分与错配 0 分的失败形态不同（乱码→截断/偏题），排除管线变量后结论归于模型能力。

## 5. 执行层缺陷定位与修复（本轮新增知识）

1. **HF 导出 tie 断裂**：`export_model.py` 在容器 transformers 4.57.6 下，`save_pretrained(safe_serialization=True)` 剔除 tied `output.weight`，重载不恢复 tie → 重载模型 lm_head 随机（实测 GEN_MATCH=False、输出乱码）。`.pt` 直读路径正常（tie 保持）。**评测与推理一律走 .pt 直读**；HF 导出线路需先修 modeling 的 `get_output_embeddings`/`tie_weights` 兼容。
2. **评测模板两套线**：`minicpm5_io` 是 MiniCPM5（130K）模板线；stageD（6.4K 主线）用内置模板。评测 mindlm 主线模型必须用内置模板渲染（`apply_chat_template` 直调），不能经 configure_tokenizer 换模板。
3. **容器/宿主双 venv**：`/home/.../venvs/mindlm-kernels-20260911` 宿主侧是 python3.12（缺 referencing 等包），容器侧 python3.10 齐备。作用评测/训练必须 `docker exec` 进容器执行。
4. **宿主 nohup 权限边界**：容器内进程（root）写宿主侧属主目录会 Permission denied——评测驱动改为容器内写 exit（或评测产物目录预先 chmod）。

## 6. 技术产物索引

- 评测 wrapper：`eval/run_v2_suite_mindlm.py`（冻结协议 + .pt 直读 + 内置模板渲染）
- C→D watcher：`runs/stageD_sft_20261121/watcher.sh`（60 宿主，等 C exit=0 且 GPU<1GB，自动跑 launch_formal.sh；已退出）
- 评测产物：`runs/stageD_sft_20261121/eval_400/{dev,test}.jsonl*`（含 manifest/summary）；对照 `eval_400_v1_chatml_mismatch/`
- 模型 checkpoint：`runs/stageD_sft_20261121/formal_sft/mindlm_sft_grouped_{epoch0,epoch1,latest}.pt`

## 7. 遗留决策（待用户）

1. push 授权：本地 main 领先 origin 17+ 提交（GitHub `SyJarvis/MindLM`）。
2. develop 分支 `e4ef885`（runs 删除记录）是否 cherry-pick 回 main。
3. `config/mindlm.json` + `model/`（mindlm.py 空文件）历史漂移项清理。
4. wandb offline 两轮数据是否事后 sync。
5. 评测协议明示冻结：本轮 wrapper 已按 §8 要求记录模板/ckpt/tokenizer 哈希于 manifest；后续改协议须先冻结存档。

## 8. 下一步建议（不自动执行）

- 受控答题能力提升路线：a) 扩训更多受控格式样本（当前答案规格教学样本占比极小）；b) 第二轮 SFT 加入模板 tools 段渲染（工具 case 破局）；c) 提升模型规模或词表覆盖的取舍讨论。
- HF 导出修复（`get_output_embeddings` 实现）后再跑一次导出→重载一致性测试，供未来部署链路使用。
