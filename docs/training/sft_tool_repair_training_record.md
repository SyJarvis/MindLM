# SFT Tool Repair V2 — 训练实验记录

> **日期**: 2026-09-28
> **模型**: MindLM 0.2B GDN (206.8M params)
> **目标**: 提升模型在 SFT 阶段的工具调用（tool call）能力
> **数据集**: `sft_t2t_repair_v2_controlled`（45,830 条）

---

## 1. 背景

在完成 E2 全量 SFT 训练（`sft_v3_e2`，4M 条数据）后，模型在通用对话上表现良好，但在工具调用场景下存在以下问题：

- 模型倾向于**直接回答**而非调用工具
- 即使调用了工具，函数名和参数也经常出错
- 对"有工具可用但不应调用"的负样本边界模糊

为此，我们构建了专门的 **tool repair 数据集**，针对工具调用场景进行修复训练。

---

## 2. 数据集

### 2.1 数据来源

| Source | 行数 | 占比 | 说明 |
|--------|------|------|------|
| `sft_t2t_repair_v2` | 38,330 | 83.6% | 工具修复核心数据（合成） |
| `legacy` | 7,431 | 16.2% | 历史通用对话 |
| `multilingual_zh` | 49 | 0.1% | 中文多轮对话 |
| `pb_zh` | 20 | <0.1% | PerfectBlend 中文 |
| **Total** | **45,830** | 100% | |

### 2.2 Domain 分布

| Domain | 行数 | 说明 |
|--------|------|------|
| `call_required_grounded` | 29,378 | 必须调用工具的 grounded 样本 |
| *(空)* | 7,500 | 无工具定义（纯对话） |
| `high_quality_negative` | 4,184 | 高质量负样本（有工具但不调用） |
| `synthetic_transient_retry` | 2,500 | 合成临时性错误重试 |
| `synthetic_permanent_honest` | 1,500 | 合成永久性错误诚实回答 |
| `error_honest_final` | 693 | 错误后诚实最终回答 |
| `error_retry` | 75 | 错误重试 |

### 2.3 Token 统计

| 指标 | 值 |
|------|------|
| 总 Token 数 | 23,461,800 |
| Supervised Token 数 | 7,432,459 |
| Supervised 占比 | 31.7% |
| 平均序列长度 | 512 tokens |
| 最长序列 | 2,334 tokens |
| 最短序列 | 23 tokens |
| 含 tools 定义的行 | 38,330 (83.6%) |
| 含 tool_call 消息的行 | 34,146 (74.5%) |
| 含中文对话 | 21,100 (46.0%) |

### 2.4 数据格式

每条记录为 CSV 格式，包含：
- `messages`: JSON 数组，多轮对话（user / assistant / tool 角色）
- `tools`: JSON 数组，OpenAI 格式的函数定义
- `source`: 数据来源标签
- `domain`: 领域标签

Token 化后生成 `.bin` + `.rows.jsonl`，其中 `rows.jsonl` 的 `sup` 字段记录了需要计算 loss 的 assistant 回复区间（role-aware mask）。

---

## 3. 训练配置

### 3.1 实验对比

| 配置项 | 首次训练 (full) | 重训 (redo) | 变化 |
|--------|:---:|:---:|:---:|
| **Epochs** | 2 | **20** | 10x ↑ |
| **学习率** | 5e-7 | **1e-6** | 2x ↑ |
| **Warmup iters** | 20 | **50** | 2.5x ↑ |
| **总更新数** | 179 | **1,793** | 10x ↑ |
| **恢复起点** | smoke200_base.pt | smoke200_base.pt | 相同 |
| **Token budget** | 524,288 | 524,288 | 相同 |
| **Accumulation steps** | 2 | 2 | 相同 |
| **Max batch count** | 2,048 | 2,048 | 相同 |
| **Mega batch** | 512 | 512 | 相同 |
| **Gradient checkpointing** | all | all | 相同 |
| **Dtype** | bfloat16 | bfloat16 | 相同 |
| **Grad clip** | 1.0 | 1.0 | 相同 |
| **Weight decay** | 0.1 | 0.1 | 相同 |
| **Save interval** | 200 | 200 | 相同 |
| **Log interval** | 10 | 10 | 相同 |

### 3.2 训练命令

```bash
# 首次训练 (full)
python3 full_sft.py \
  --bin runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled_tokens_4096.bin \
  --meta runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled_tokens_4096.rows.jsonl \
  --model_config mindlm_0.2b_gdn \
  --out_dir runs/sft_t2t_tool_repair_v2_20260924/full/out \
  --resume_from runs/sft_t2t_tool_repair_v2_20260924/train100/input/mindlm_sft_smoke200_base.pt \
  --resume_weights_only \
  --token_budget 524288 --accumulation_steps 2 \
  --learning_rate 5e-7 --warmup_iters 20 --grad_clip 1.0 \
  --epochs 2 --num_workers 8 --max_batch_count 2048 \
  --gradient_checkpointing all --dtype bfloat16 \
  --save_interval 200 --log_interval 10 --seed 1337

# 重训 (redo)
python3 full_sft.py \
  --bin runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled_tokens_4096.bin \
  --meta runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled_tokens_4096.rows.jsonl \
  --model_config mindlm_0.2b_gdn \
  --out_dir runs/sft_t2t_tool_repair_v2_20260924/redo/out \
  --resume_from runs/sft_t2t_tool_repair_v2_20260924/train100/input/mindlm_sft_smoke200_base.pt \
  --resume_weights_only \
  --token_budget 524288 --accumulation_steps 2 \
  --learning_rate 1e-6 --warmup_iters 50 --grad_clip 1.0 \
  --epochs 20 --num_workers 8 --max_batch_count 2048 \
  --gradient_checkpointing all --dtype bfloat16 \
  --save_interval 200 --log_interval 10 --seed 1337
```

---

## 4. 训练过程

### 4.1 硬件环境

| 硬件 | 规格 |
|------|------|
| GPU | 1× NVIDIA H200 NVL |
| 显存 | 143,771 MiB |
| Python | mindlm-kernels-20260911 venv (Python 3.10) |
| 容器 | deeplearning-v2-mindlm-cdi |

### 4.2 吞吐量优化

在首次训练前进行了 9 组配置对比实验，最优配置：

| 参数 | 最优值 | 吞吐量 |
|------|:---:|:---:|
| Token budget | 524,288 | ~14,800 tok/s |
| Accumulation steps | 2 | |
| Max batch count | 2,048 | |
| Mega batch | 512 | |

> 详细实验记录见 `docs/training/sft_throughput_optimization.md`

### 4.3 Loss 曲线

```
epoch=1/20  loss=1.93  →  epoch=5/20  loss≈1.15  →  epoch=10/20  loss≈1.05
→  epoch=15/20  loss≈1.00  →  epoch=20/20  loss≈1.03
```

- Loss 从 1.93 快速下降，前 5 个 epoch 降幅最大
- 约 epoch 10 后进入平台期（loss ~1.0）
- 最终 loss ~1.03，仍未完全收敛

### 4.4 训练耗时

| 训练 | 总更新数 | 耗时 | 平均每更新 |
|------|:---:|:---:|:---:|
| 首次 (full) | 179 | 15.7 min | 5.3s |
| 重训 (redo) | 1,793 | 155.2 min (~2.6h) | 5.2s |

---

## 5. Eval 结果

### 5.1 评测方法

使用 `eval/eval_sft_tool.py` 进行单轮探针测试：
- **20 个 positive case**：需要调用工具的场景（天气查询、数学计算、单位换算、汇率、翻译、新闻、时间、随机数等）
- **8 个 negative case**：有工具清单但不应调用（问候、自我介绍、哲学问题、常识问答等）
- **温度**: 0.1, **Top-K**: 8, **Max new tokens**: 256

### 5.2 结果对比

| 指标 | 首次训练 (update=179) | 重训 (update=1793) | 变化 |
|------|:---:|:---:|:---:|
| **call_emitted_rate** | **2/20 (10%)** | **3/20 (15%)** | ↑ +5% |
| **json_valid_rate** | 2/20 (10%) | 3/20 (15%) | ↑ +5% |
| **name_accuracy** | 1/20 (5%) | 2/20 (10%) | ↑ +5% |
| **name_in_provided_tools** | 2/20 (10%) | 3/20 (15%) | ↑ +5% |
| **args_complete_rate** | 1/20 (5%) | 2/20 (10%) | ↑ +5% |
| **overall_pass_rate** | **9/28 (32.1%)** | **10/28 (35.7%)** | ↑ +3.6% |
| **negative_no_call_rate** | **8/8 (100%)** | **8/8 (100%)** | — |

### 5.3 通过的 Positive Case 明细

| Case | 首次训练 | 重训 |
|------|:---:|:---:|
| `weather_beijing` | ❌ | ✅ `get_current_weather` |
| `weather_then_convert` | ❌ | ✅ `get_current_weather` |
| `math_complex` | ✅ `calculate_math` | ❌ （退步） |
| `text_len` | ❌ | ✅ 但函数名错误（调了 weather） |

### 5.4 关键发现

1. **训练不足是主要瓶颈**：10x epoch 后调用率从 10% 提升到 15%，说明更多训练确实有效，但提升幅度有限
2. **负样本边界已学到位**：negative no_call 从首次就达到 100%，20 epochs 后仍然保持
3. **模型倾向于"直接回答"**：即使在有工具的场景下，模型更依赖自身的知识而非调用工具
4. **函数名混淆**：即使调用了工具，也经常调用错误的函数（如需要 `text_length` 却调了 `get_current_weather`）
5. **Loss 仍在 ~1.0**：可能还未完全收敛，更多训练可能继续提升

---

## 6. 完整 SFT 数据全景

### 6.1 所有 SFT 数据集

| 数据集 | 文件 | 大小 | 行数 | 状态 |
|--------|------|:---:|:---:|:---:|
| E2 (v1) | `runs/sft_v3_e2/.../sft_e2_combined.csv` | 2.3 GB | 4,053,775 | ✅ 已训练 |
| E2 (v2) | `full_data_v2/sft_e2_combined_v2.csv` | **40 GB** | **5,823,341** | ❌ 未训练 |
| Tool Repair V1 | `runs/sft_t2t_tool_repair/.../sft_t2t_tool_repair.csv` | 616 MB | 287,422 | ✅ 已训练 |
| Tool Repair V2 | `runs/.../sft_t2t_repair_v2_controlled.csv` | 92 MB | 45,830 | ✅ **当前** |

### 6.2 E2 v2 数据集详情

`full_data_v2/sft_e2_combined_v2.csv`（40 GB, 5,823,341 行）相比 v1 新增了 **1,420,909 条 `perfectblend_en`** 英文数据，其余来源基本一致：

| Source | v1 | v2 |
|--------|:---:|:---:|
| legacy | 3,935,005 | 3,935,005 |
| perfectblend_en | — | **1,420,909** |
| xlam_fc | 58,607 | 53,267 |
| ultra | 24,310 | 24,297 |
| multilingual_zh | 25,129 | 25,129 |
| pb_zh | 9,988 | 9,988 |
| toolace | 736 | 736 |

---

## 7. 结论与下一步

### 7.1 结论

1. **增加 10x epoch 训练带来了有限的提升**（调用率 10% → 15%），说明单纯的重复训练不足以让模型学会工具调用
2. **模型已掌握 tool_call 格式和负样本边界**，但"主动调用工具"的决策能力仍然不足
3. **训练数据本身的质量和多样性**可能是更大的瓶颈

### 7.2 下一步建议

| 方向 | 优先级 | 说明 |
|------|:---:|------|
| **使用 E2 v2 全量数据（40GB）训练** | 🔴 高 | 增加 142 万条英文数据，提升模型对 tool use 的理解 |
| **增加正样本比例** | 🟡 中 | 当前正样本中模型不调用工具的比例可能过高 |
| **调整训练策略** | 🟡 中 | 尝试更高的学习率、更长的 warmup |
| **改进 eval 套件** | 🟢 低 | 增加闭环 eval（调用 → mock 执行 → 回填 → 终答） |
| **尝试从 E2 checkpoint 继续训练** | 🟢 低 | 而非从 smoke200 base 开始 |

---

## 8. 附录

### 8.1 文件路径

| 文件 | 路径 |
|------|------|
| 首次训练日志 | `runs/sft_t2t_tool_repair_v2_20260924/full/train.log` |
| 首次训练 checkpoint | `runs/sft_t2t_tool_repair_v2_20260924/full/out/` |
| 首次训练 eval 结果 | `runs/sft_t2t_tool_repair_v2_20260924/full/eval/` |
| 重训日志 | `train_sft_tool_repair_v2_redo.log` |
| 重训 checkpoint | `runs/sft_t2t_tool_repair_v2_20260924/redo/out/` |
| 训练数据 CSV | `runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled.csv` |
| 训练数据 bin | `runs/sft_t2t_tool_repair_v2_20260924/full_v2/data/sft_t2t_repair_v2_controlled_tokens_4096.bin` |
| E2 v2 数据（未训练） | `full_data_v2/sft_e2_combined_v2.csv` |
| Eval 脚本 | `eval/eval_sft_tool.py` |
| SFT 设计文档 | `docs/sft.md` |
| 吞吐量优化文档 | `docs/training/sft_throughput_optimization.md` |

### 8.2 Checkpoint 列表（重训）

| Epoch | Update | 文件 |
|:---:|:---:|------|
| 0 | 90 | `mindlm_sft_grouped_epoch0.pt` |
| 1 | 179 | `mindlm_sft_grouped_epoch1.pt` |
| 2 | 269 | `mindlm_sft_grouped_epoch2.pt` |
| 3 | 359 | `mindlm_sft_grouped_epoch3.pt` |
| 4 | 448 | `mindlm_sft_grouped_epoch4.pt` |
| 5 | 538 | `mindlm_sft_grouped_epoch5.pt` |
| 6 | 628 | `mindlm_sft_grouped_epoch6.pt` |
| 7 | 718 | `mindlm_sft_grouped_epoch7.pt` |
| 8 | 808 | `mindlm_sft_grouped_epoch8.pt` |
| 9 | 897 | `mindlm_sft_grouped_epoch9.pt` |
| 10 | 987 | `mindlm_sft_grouped_epoch10.pt` |
| 11 | 1077 | `mindlm_sft_grouped_epoch11.pt` |
| 12 | 1166 | `mindlm_sft_grouped_epoch12.pt` |
| 13 | 1256 | `mindlm_sft_grouped_epoch13.pt` |
| 14 | 1346 | `mindlm_sft_grouped_epoch14.pt` |
| 15 | 1435 | `mindlm_sft_grouped_epoch15.pt` |
| 16 | 1525 | `mindlm_sft_grouped_epoch16.pt` |
| 17 | 1615 | `mindlm_sft_grouped_epoch17.pt` |
| 18 | 1704 | `mindlm_sft_grouped_epoch18.pt` |
| 19 | 1793 | `mindlm_sft_grouped_epoch19.pt` |
| latest | 1793 | `mindlm_sft_grouped_latest.pt` |

### 8.3 Loss 曲线（完整）

```
epoch  1: 1.93 → 1.30 → 1.34 → 1.58 → 1.34 → 1.92 → 1.27 → 1.21 → 1.25
epoch  2: 1.25 → 1.21 → 1.40 → 1.25 → 1.28 → 1.20 → 1.17 → 1.16
epoch  3: 1.16 → 1.20 → 1.15 → 1.49 → 1.19 → 1.24 → 1.19 → 1.19 → 2.21
epoch  4: 1.17 → 1.12 → 1.12 → 1.19 → 1.15 → 1.14 → 1.15 → 1.12 → 1.12
epoch  5: 1.18 → 1.15 → 1.11 → 1.09 → 1.09 → 1.10 → 1.11 → 1.10 → 1.10
epoch  6: 1.14 → 1.09 → 1.08 → 1.09 → 1.08 → 1.09 → 1.08 → 1.08 → 1.07
epoch  7: 1.07 → 1.08 → 1.07 → 1.08 → 1.07 → 1.07 → 1.06 → 1.07 → 1.05
epoch  8: 1.10 → 1.39 → 0.98 → 1.09 → 1.13 → 1.03 → 1.05 → 1.05 → 1.05
epoch  9: 1.01 → 1.06 → 1.22 → 1.05 → 1.09 → 0.94 → 1.19 → 1.57 → 1.60
epoch 10: 1.02 → 1.05 → 1.02 → 1.08 → 1.10 → 1.05 → 1.12 → 1.05 → 1.12
epoch 11: 1.45 → 1.06 → 1.05 → 1.06 → 1.02 → 1.02 → 1.05 → 1.02 → 1.00
epoch 12: 1.03 → 1.05 → 1.04 → 1.05 → 1.02 → 1.04 → 1.03 → 1.02 → 1.01
epoch 13: 1.00 → 1.00 → 1.03 → 1.02 → 1.00 → 1.02 → 1.02 → 1.00 → 1.00
epoch 14: 0.97 → 1.00 → 1.06 → 1.06 → 1.00 → 1.00 → 1.00 → 1.07 → 1.08
epoch 15: 1.02 → 1.02 → 1.02 → 1.04 → 1.01 → 1.01 → 1.00 → 0.99 → 1.00
epoch 16: 1.00 → 1.02 → 1.00 → 1.00 → 1.00 → 0.98 → 1.00 → 0.99 → 0.99
epoch 17: 0.99 → 1.00 → 0.98 → 0.99 → 0.98 → 0.99 → 0.98 → 0.98 → 0.97
epoch 18: 0.97 → 0.98 → 0.97 → 0.97 → 0.97 → 0.97 → 0.97 → 0.97 → 0.96
epoch 19: 0.96 → 0.97 → 0.95 → 0.96 → 0.95 → 0.95 → 0.94 → 0.94 → 0.92
epoch 20: 1.07 → 1.00 → 0.98 → 1.04 → 1.40 → 0.99 → 1.03 → 1.04 → 1.03
```
