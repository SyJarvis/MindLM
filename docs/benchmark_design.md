# MindLM Benchmark 评测方案

> 参照 MiniMind 的 lm-evaluation-harness 方案，针对 MindLM 混合注意力架构的特点重新设计，使其更科学、标准、可复现。

---

## 1. MiniMind 现有方案分析

### 1.1 MiniMind 做了什么

| 环节 | MiniMind 做法 |
|------|--------------|
| 测评框架 | EleutherAI lm-evaluation-harness |
| 测评集 | C-Eval, CMMLU, ARC-Easy, PIQA, OpenBookQA, HellaSwag, Social-IQa（共 7 个） |
| 模型转换 | `convert_torch2transformers()` 将权重映射为 Qwen3 结构，再用 `lm_eval --model hf` 加载 |
| 定性评测 | `eval_llm.py`（8 条中文 prompt 文本生成）、`eval_toolcall.py`（8 条工具调用） |
| Few-shot 设置 | 未明确记录 |
| 复现性 | 未固定随机种子，未锁定框架版本 |

### 1.2 MiniMind 方案可以直接复用的部分

- lm-evaluation-harness 框架选型 — 业界标准，没有更好的替代
- 7 个基准数据集 — 覆盖中英文、知识推理、常识判断，选择合理
- 定性评测思路（文本生成 + 工具调用）— 作为补充有参考价值

### 1.3 MiniMind 方案的不足（MindLM 需要改进）

| 问题 | 说明 |
|------|------|
| **权重转换不可行** | MiniMind 能转 Qwen3 是因为架构一致；MindLM 有 GatedDeltaNet 层，无法映射到任何现有 HF 模型 |
| **评测集偏少** | 7 个数据集覆盖面有限，缺少数学推理、幻觉检测、长文本等维度 |
| **Few-shot 未标准化** | 未记录每个任务使用的 few-shot 数量，不同 run 可能结果不一致 |
| **无复现性保障** | 未固定 seed、未锁定 lm-eval-harness 版本 |
| **无基线对比** | 只报告绝对分数，缺少同参数量级模型的横向对比 |
| **无 per-category 分析** | 只报告总分数，无学科/类别维度的细分 |
| **定性评测不标准** | 8 条 prompt 是主观选择，无标准化评分机制 |

---

## 2. MindLM Benchmark 方案设计

### 2.1 前置条件

MindLM 需要以 HuggingFace 格式加载才能被 lm-evaluation-harness 调用。有两种路径：

| 路径 | 方式 | 适用时机 |
|------|------|---------|
| **A. trust_remote_code** | 上传到 Hub，`lm_eval --model hf --model_args ...,trust_remote_code=True` | 路线 1 完成前 |
| **B. 注册模型类型** | 在本地 fork 中注册 MindLM，`lm_eval --model hf --model_args pretrained=mindlm-1b` | 路线 1 完成后 |

**建议先用路径 A 快速跑通，路线 1 完成后切换到路径 B。**

### 2.2 评测集选择

分为**核心评测集**（必须）和**扩展评测集**（可选），按能力维度组织：

#### 核心评测集（7 个，与 MiniMind 对齐，确保可比性）

| 维度 | 评测集 | 语言 | Few-shot | 说明 |
|------|--------|------|----------|------|
| 中文知识 | C-Eval | 中文 | 5-shot | 52 个学科的多选题，中文 LLM 标准基准 |
| 中文知识 | CMMLU | 中文 | 5-shot | 67 个学科，与 C-Eval 互补 |
| 英文推理 | ARC-Challenge | 英文 | 25-shot | 科学推理，Challenge 子集（难度更高） |
| 英文推理 | HellaSwag | 英文 | 10-shot | 句子补全，常识推理 |
| 常识判断 | PIQA | 英文 | 10-shot | 物理交互常识 |
| 常识判断 | OpenBookQA | 英文 | 10-shot | 开卷科学问答 |
| 社会常识 | Social-IQa | 英文 | 10-shot | 社会情境理解 |

> **注意**：MiniMind 用的是 ARC-Easy，我们改用 ARC-Challenge，因为 Easy 太简单，区分度低。如果需要与 MiniMind 直接对比，可同时报告两个分数。

#### 扩展评测集（按需选用）

| 维度 | 评测集 | Few-shot | 说明 |
|------|--------|----------|------|
| 数学推理 | GSM8K | 8-shot | 小学数学应用题，推理能力核心指标 |
| 知识广度 | MMLU | 5-shot | 57 个学科，英文 LLM 最广泛使用的基准 |
| 幻觉检测 | TruthfulQA | 0-shot | 检测模型是否产生事实性错误 |
| 常识推理 | WinoGrande | 10-shot | 共指消解，Winograd Schema 升级版 |
| 综合推理 | MUSR | 0-shot | 多步推理，难度较高 |
| 长文本 | LongBench | 0-shot | 长上下文理解（验证线性注意力的长文本优势） |

### 2.3 评测协议标准化

以下参数必须固定并在结果中明确报告：

```yaml
# eval_config.yaml — MindLM 评测标准配置
framework: lm-evaluation-harness==0.4.8   # 锁定版本
model_dtype: bfloat16                       # 统一精度
batch_size: 16                              # 根据显存调整，需记录
num_fewshot: 见上表                         # 每个任务固定
seed: 42                                    # 固定随机种子
device: cuda                                # 评测设备
trust_remote_code: true                     # 路径 A 需要
```

**结果报告必须包含：**

- lm-evaluation-harness 版本号
- 每个 task 的 few-shot 数量
- 模型精度（fp16 / bf16 / fp32）
- 评测设备（GPU 型号）
- acc（准确率）和 acc_norm（归一化准确率，如适用）
- 每个 task 的 per-category 细分（C-Eval/CMMLU 报告各学科分数）

### 2.4 与 MiniMind 的公平对比

为确保对比的科学性：

| 对比项 | 要求 |
|--------|------|
| 参数量级 | 报告总参数和可训练参数，对比同参数量级模型 |
| 训练数据 | 记录训练 token 数、数据来源 |
| Few-shot 数量 | 严格一致 |
| 评测框架版本 | 严格一致 |
| 模型精度 | 严格一致 |
| 评测设备 | 记录但不要求一致 |

### 2.5 基线对比模型

选择与 MindLM 参数量相近的模型作为基线：

| 模型 | 参数量 | 架构 | 对比意义 |
|------|--------|------|---------|
| MiniMind3 | ~64M | 标准 GQA | 同系基座对比 |
| MiniMind3-MoE | ~198M | MoE | MoE 对比 |
| Qwen3-0.6B | ~600M | GQA + QK Norm | 同架构大一号 |
| Pythia-160M | ~160M | 标准 Transformer | 经典基线 |
| SmolLM-135M | ~135M | LLaMA 架构 | 同规模主流模型 |

---

## 3. 实现步骤

### 3.1 阶段一：快速跑通（1-2 天）

目标：用最少的改动跑通 lm-evaluation-harness。

**Step 1：准备 HF 格式模型**

MindLM 目前无法像 MiniMind 那样映射到 Qwen3 结构。最简方案是直接用 `save_pretrained()`：

```python
# scripts/convert_mindlm_to_hf.py
from modeling_mindlm import MindLM, MindLMConfig
from transformers import AutoTokenizer
import torch

def convert(torch_path, output_path, config_name="mindlm_1b"):
    # 加载配置
    config = load_config(config_name)
    # 创建模型
    model = MindLM(MindLMConfig(**config))
    # 加载权重
    state_dict = torch.load(torch_path, map_location="cpu")
    model.load_state_dict(state_dict)
    # 保存为 HF 格式（含 modeling_mindlm.py 代码）
    model.save_pretrained(output_path)
    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_path)
```

> `save_pretrained()` 会自动将 `modeling_mindlm.py` 复制到输出目录，使 `trust_remote_code=True` 可以加载。

**Step 2：运行核心评测**

```bash
# 安装 lm-evaluation-harness
pip install lm-eval==0.4.8

# 单任务测试
lm_eval --model hf \
    --model_args pretrained=out/mindlm-1b-hf,dtype=bfloat16,trust_remote_code=True \
    --tasks ceval-valid \
    --num_fewshot 5 \
    --batch_size 16 \
    --seed 42 \
    --output_path results/mindlm-1b/

# 全部核心评测
lm_eval --model hf \
    --model_args pretrained=out/mindlm-1b-hf,dtype=bfloat16,trust_remote_code=True \
    --tasks ceval-valid,cmmlu,arc_challenge,hellaswag,piqa,openbookqa,siqa \
    --batch_size 16 \
    --seed 42 \
    --output_path results/mindlm-1b/
```

**Step 3：收集并整理结果**

```bash
# 结果已在 output_path 目录下
ls results/mindlm-1b/
```

### 3.2 阶段二：标准化评测脚本（2-3 天）

目标：编写一键运行的评测脚本，自动生成标准化报告。

**文件结构：**

```
MindLM/
├── benchmark/
│   ├── eval_config.yaml          # 评测配置（任务、few-shot、seed 等）
│   ├── run_eval.py               # 一键评测脚本
│   ├── report_results.py         # 结果整理 + 格式化输出
│   └── results/                  # 结果存放目录
│       └── mindlm-1b/
│           ├── ceval.json
│           ├── cmmlu.json
│           └── ...
```

**`run_eval.py` 要点：**

```python
"""
MindLM 标准化评测脚本
用法：python benchmark/run_eval.py --model_path out/mindlm-1b-hf --config benchmark/eval_config.yaml
"""
import subprocess, yaml, json, os

def run_eval(model_path, config_path):
    config = yaml.safe_load(open(config_path))
    tasks = ",".join(config["tasks"])
    results = {}
    for task_config in config["tasks_detail"]:
        task_name = task_config["name"]
        cmd = [
            "lm_eval", "--model", "hf",
            "--model_args", f"pretrained={model_path},dtype={config['dtype']},trust_remote_code=True",
            "--tasks", task_name,
            "--num_fewshot", str(task_config["num_fewshot"]),
            "--batch_size", str(config["batch_size"]),
            "--seed", str(config["seed"]),
            "--output_path", f"{config['output_dir']}/{task_name}",
            "--log_samples",  # 保存每个样本的预测，用于后续分析
        ]
        subprocess.run(cmd, check=True)
    # 整理结果
    generate_report(config["output_dir"], config)
```

**`report_results.py` 要点：**

- 读取所有 task 结果 JSON
- 生成标准化 Markdown 表格
- C-Eval/CMMLU 按 52/67 个学科分类报告 per-category 分数
- 与基线模型对比（读取基线结果文件）
- 输出格式：

```markdown
## MindLM-1B Benchmark Results

| Task | Few-shot | MindLM-1B (122M) | MiniMind3 (64M) | SmolLM (135M) |
|------|----------|-------------------|-----------------|---------------|
| C-Eval | 5 | xx.xx | 24.89 | — |
| CMMLU | 5 | xx.xx | 25.38 | — |
| ARC-C | 25 | xx.xx | — | — |
| ... | ... | ... | ... | ... |
| **Average** | — | **xx.xx** | **xx.xx** | **xx.xx** |

### C-Eval 分学科详情
| 学科 | 分数 |
|------|------|
| ... | ... |
```

### 3.3 阶段三：扩展评测（按需）

**Step 1：添加扩展评测集**

```bash
# GSM8K 数学推理
lm_eval --model hf \
    --model_args pretrained=out/mindlm-1b-hf,dtype=bfloat16,trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 8 \
    --batch_size 16 --seed 42

# MMLU 知识广度
lm_eval --model hf \
    --model_args pretrained=out/mindlm-1b-hf,dtype=bfloat16,trust_remote_code=True \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 16 --seed 42
```

**Step 2：长文本专项评测**（验证线性注意力优势）

```bash
# LongBench — 长上下文理解
lm_eval --model hf \
    --model_args pretrained=out/mindlm-1b-hf,dtype=bfloat16,trust_remote_code=True \
    --tasks longbench \
    --batch_size 4 --seed 42
```

对比实验：固定参数量，MindLM vs MiniMind3 在不同序列长度下的表现。

**Step 3：推理效率评测**

```python
# benchmark/bench_inference.py
"""推理速度和内存基准测试"""
import torch, time

def bench_inference(model, tokenizer, seq_lengths=[128, 256, 512, 1024]):
    for seq_len in seq_lengths:
        # 测量：首 token 延迟、吞吐量、峰值显存
        # 对比：MindLM vs MiniMind3 在不同长度下的效率差异
        pass
```

---

## 4. 评测结果报告模板

每次正式评测完成后，按以下模板记录：

```markdown
# MindLM Benchmark Report

## 实验信息
- **模型**: MindLM-1B (122M params)
- **权重**: [commit hash 或路径]
- **训练数据**: [token 数, 来源]
- **评测日期**: YYYY-MM-DD
- **框架**: lm-evaluation-harness==0.4.8
- **精度**: bfloat16
- **设备**: [GPU 型号]
- **Seed**: 42

## 核心评测结果

| Task | Few-shot | Metric | Score |
|------|----------|--------|-------|
| C-Eval | 5 | acc | xx.xx |
| C-Eval | 5 | acc_norm | xx.xx |
| CMMLU | 5 | acc | xx.xx |
| ARC-Challenge | 25 | acc | xx.xx |
| ARC-Challenge | 25 | acc_norm | xx.xx |
| HellaSwag | 10 | acc | xx.xx |
| HellaSwag | 10 | acc_norm | xx.xx |
| PIQA | 10 | acc | xx.xx |
| PIQA | 10 | acc_norm | xx.xx |
| OpenBookQA | 10 | acc | xx.xx |
| Social-IQa | 10 | acc | xx.xx |
| **Average** | — | — | **xx.xx** |

## 与基线对比
[表格]

## C-Eval 分学科详情
[表格]

## 分析
[对结果的简要分析]
```

---

## 5. 与 MiniMind 方案的对比总结

| 方面 | MiniMind | MindLM（本方案） |
|------|----------|-----------------|
| 模型加载方式 | 映射到 Qwen3 结构 | save_pretrained + trust_remote_code |
| 评测集数量 | 7 个 | 核心 7 个 + 可选扩展 6 个 |
| ARC 版本 | Easy | Challenge（区分度更高） |
| Few-shot 记录 | 未记录 | 每个任务固定并在报告中标注 |
| 版本锁定 | 无 | lm-eval==0.4.8 |
| 随机种子 | 未固定 | seed=42 |
| Per-category 分析 | 无 | C-Eval/CMMLU 按学科细分 |
| 基线对比 | 无绝对对比 | 横向对比同参数量级模型 |
| 报告模板 | 无 | 标准化模板，含完整实验信息 |
| 推理效率评测 | 仅 tokens/s | 首 token 延迟 + 吞吐 + 峰值显存 |
| 长文本评测 | 无 | LongBench 专项（验证线性注意力优势） |
| 一键运行 | 手动命令 | run_eval.py + eval_config.yaml |

---

## 6. 执行优先级

```
阶段 1（快速验证）       阶段 2（标准化）          阶段 3（深度分析）
─────────────────      ──────────────────      ──────────────────
convert_mindlm_to_hf   run_eval.py              GSM8K / MMLU
单任务跑通 lm_eval      report_results.py        LongBench
7 个核心评测集          eval_config.yaml          推理效率 benchmark
收集原始分数            标准化报告模板             与基线全面对比
                        基线对比表格
```
