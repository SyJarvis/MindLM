# MindLM 代码改动梳理

> 基于 `d24d73b`（HEAD）与工作区未提交变更
> 日期：2026-09-28

---

## 一、已提交的 Git 历史（从近到远）

```
d24d73b feat(sft): finalize E2 data config - max_tools=8, max_tokens=4096
7400363 feat(data): cap per-row tool schemas with called-first priority
5146a60 feat(data): add tool schema audit script
11a4767 feat(sft): complete E2 tools rendering + closed-loop tool eval
395b146 feat(train): upgrade pretrain checkpoint to v2 with elastic resume
7d56a62 chore(data): remove obsolete prepare_pretrain_data.py
e690625 Merge pull request #3
5fbb45f chore(repo): remove obsolete training assets
97f36eb docs: reorganize project documentation
3a98b17 refactor(eval): update evaluation and export tooling
cbafc59 refactor(train): consolidate MindLM model and trainers
0aca2ee refactor(data): add deterministic packed data preparation
7bda3a1 fix(model): make GDN V3 initialization explicit and idempotent
4f852cc refactor(eval): isolate tool call report generation
8f8b428 refactor(train): separate pretrain and sft trainer boundaries
48acb89 refactor(train): share cosine learning rate schedule
00df669 refactor(train): remove duplicate checkpoint loader
...
```

---

## 二、未提交的已跟踪文件变更（6 个文件，+1420 / -257）

### 1. `prepare_sft_qwen3.py`（+785 / -???，核心改动）

**目的**：将 E2 实验的数据流水线升级为支持 V2 数据集（含 tool repair）

**主要改动**：
- 重构了 tools 列的渲染逻辑，支持 called-first 裁剪（`--max_tools`）
- 新增 `--max_tools 8` 参数，控制每行渲染进 `<tools>` 块的 schema 数量
- 新增 `--max_tokens 4096` 参数，渲染后过滤超长序列
- 新增 `domain` 字段的传递和审计
- 新增负样本注入逻辑（`high_quality_negative`）
- 修复了 whitespace 处理（assistant 回复前的空格问题）

### 2. `data_process/sft.py`（+39）

**目的**：配合 `prepare_sft_qwen3.py` 的数据处理变更

**主要改动**：
- 更新了 SFT 数据处理的 pipeline
- 支持新的 CSV 格式（含 `domain` 字段）

### 3. `eval/eval_sft_tool.py`（+363 / 重写）

**目的**：从旧的 retail agent eval 重写为使用训练数据中真实函数的 eval

**主要改动**：
- 删除了旧的 `tool_cases_v1.jsonl` 依赖（8 个 retail agent 函数）
- 新增 20 个 positive case + 8 个 negative case
- 使用训练数据中的真实函数（`get_current_weather`、`calculate_math`、`unit_converter` 等）
- 新增详细的指标统计（`call_emitted_rate`、`json_valid_rate`、`name_accuracy` 等）
- 新增 JSON 格式的结果输出

### 4. `eval/eval_sft_tool_loop.py`（+242）

**目的**：闭环 eval（调用 → mock 执行 → 回填 → 终答）

**主要改动**：
- 新增完整的 tool call loop 测试
- 支持 mock 函数执行
- 验证终答是否使用了工具返回数据

### 5. `tests/test_tool_loop.py`（+246）

**目的**：新增闭环 eval 的单元测试

### 6. `docs/sft.md`（+2）

**目的**：更新 SFT 设计文档，反映最新实现

---

## 三、未跟踪的新增文件

### 3.1 核心数据准备脚本

| 文件 | 用途 |
|------|------|
| `prepare_sft_t2t_tool.py` | Tool repair V1 数据准备 |
| `prepare_sft_t2t_repair_v2.py` | Tool repair V2 数据准备（含负样本控制） |
| `prepare_sft_v2.py` | E2 V2 数据准备（含 perfectblend_en） |
| `data_process/build_v4.py` | V4 数据构建 |
| `data_process/clean_sft_v2.py` | SFT V2 数据清洗 |
| `append_data.py` | 数据追加工具 |
| `patch_prepare.py` | 数据修补工具 |
| `fix_append.py` | 修复 append 逻辑 |

### 3.2 审计与分析脚本

| 文件 | 用途 |
|------|------|
| `scripts/audit_sft_t2t_repair_v2_candidates.py` | 审计 V2 候选数据 |
| `scripts/audit_tool_cases_leakage.py` | 审计 eval 数据泄露 |
| `scripts/gen_tool_cases_v1.py` | 生成 V1 eval case |
| `check_stats.py` / `check_stats_v2.py` | 数据统计检查 |
| `quick_stats.py` / `quick_stats_tail.py` | 快速统计 |
| `source_dist.py` / `stats_source.py` | 来源分布统计 |

### 3.3 测试文件

| 文件 | 用途 |
|------|------|
| `tests/test_prepare_sft_data.py` | SFT 数据准备测试 |
| `tests/test_prepare_sft_t2t_repair_v2.py` | V2 repair 数据准备测试 |
| `tests/test_prepare_sft_t2t_tool.py` | V1 tool 数据准备测试 |
| `tests/test_repair_v2_candidates.py` | V2 候选数据测试 |
| `tests/test_tool_case_leakage.py` | eval 泄露测试 |
| `tests/test_tool_cases_v1.py` | V1 eval case 测试 |

### 3.4 文档

| 文件 | 用途 |
|------|------|
| `docs/training/sft_t2t_tool_repair_v2_training_record.md` | **本次训练实验记录（新写）** |
| `docs/training/sft_throughput_optimization.md` | 吞吐量优化实验记录（新写） |

### 3.5 数据/配置

| 文件 | 用途 |
|------|------|
| `full_data_v2/prepare_stats_v2.json` | E2 V2 数据准备统计 |
| `full_data_v2/prepare_stats_v3.json` | E2 V3 数据准备统计 |
| `eval/tool_cases_v1.leakage.json` | Eval 泄露分析结果 |
| `train_sft_tool_repair_v2_redo.log` | **重训日志（1,793 updates）** |

---

## 四、代码架构总览

```
MindLM/
├── full_sft.py                    # SFT 入口（委托给 trainers/sft_grouped.py）
├── pretrain.py                    # 预训练入口
├── prepare_data.py                # 数据 tokenize 入口
├── prepare_sft_qwen3.py           # ★ SFT CSV 构建（核心改动）
│
├── trainers/
│   ├── sft_grouped.py             # ★ Grouped SFT 训练器（含 LengthGroupedBatchSampler）
│   └── __init__.py
│
├── eval/
│   ├── eval_sft_tool.py           # ★ 单轮探针 eval（重写）
│   └── eval_sft_tool_loop.py      # ★ 闭环 eval（新增）
│
├── data_process/
│   ├── sft.py                     # ★ SFT 数据处理（更新）
│   ├── build_v4.py                # 新增：V4 构建
│   └── clean_sft_v2.py            # 新增：V2 清洗
│
├── prepare_sft_t2t_tool.py        # 新增：V1 tool repair
├── prepare_sft_t2t_repair_v2.py   # 新增：V2 tool repair
├── prepare_sft_v2.py              # 新增：E2 V2
│
├── scripts/
│   ├── audit_sft_t2t_repair_v2_candidates.py  # 新增
│   ├── audit_tool_cases_leakage.py             # 新增
│   └── gen_tool_cases_v1.py                    # 新增
│
├── docs/
│   ├── sft.md                     # SFT 设计文档（更新）
│   └── training/
│       ├── sft_t2t_tool_repair_v2_training_record.md  # ★ 新增
│       └── sft_throughput_optimization.md              # ★ 新增
│
├── tests/
│   ├── test_tool_loop.py          # ★ 闭环 eval 测试（更新）
│   ├── test_prepare_sft_data.py   # 新增
│   ├── test_prepare_sft_t2t_repair_v2.py  # 新增
│   └── ...
│
└── full_data_v2/                  # 新增：E2 V2 数据目录
    ├── sft_e2_combined_v2.csv     # 40GB，5.8M 行（未训练）
    └── prepare_stats_v2.json
```

---

## 五、实验流水线总结

```
                    ┌──────────────────────────────┐
                    │    原始数据源                  │
                    │  (legacy, ultra, xlam_fc,     │
                    │   perfectblend, toolace, ...)  │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  prepare_sft_qwen3.py         │
                    │  prepare_sft_t2t_repair_v2.py │  ← 新增
                    │  prepare_sft_v2.py            │  ← 新增
                    └──────────────┬───────────────┘
                                   ▼  CSV (messages + tools)
                    ┌──────────────────────────────┐
                    │  prepare_data.py --type sft   │
                    └──────────────┬───────────────┘
                                   ▼  bin + rows.jsonl
                    ┌──────────────────────────────┐
                    │  full_sft.py                  │  ← grouped trainer
                    │  (trainers/sft_grouped.py)    │
                    └──────────────┬───────────────┘
                                   ▼  checkpoint .pt
                    ┌──────────────────────────────┐
                    │  eval/eval_sft_tool.py        │  ← 重写
                    │  eval/eval_sft_tool_loop.py   │  ← 新增
                    └──────────────┬───────────────┘
                                   ▼  eval results
```

---

## 六、已运行的实验

| 实验 | 数据集 | 行数 | 状态 | Checkpoint 路径 |
|------|--------|:---:|:---:|:---:|
| E2 (v1) | `sft_e2_combined.csv` | 4.05M | ✅ 完成 | `out/mindlm_sft_grouped_epoch1.pt` |
| Tool Repair V1 | `sft_t2t_tool_repair.csv` | 287K | ✅ 完成 | `runs/.../train_full_epoch/out/` |
| Tool Repair V2 (full) | `sft_t2t_repair_v2_controlled.csv` | 45.8K | ✅ 完成 | `runs/.../full/out/` |
| Tool Repair V2 (redo) | `sft_t2t_repair_v2_controlled.csv` | 45.8K | ✅ **完成** | `runs/.../redo/out/` |
| E2 (v2) | `sft_e2_combined_v2.csv` | 5.82M | ❌ 未训练 | — |
