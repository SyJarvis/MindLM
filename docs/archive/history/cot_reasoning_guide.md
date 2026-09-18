# MindLM COT (Chain of Thought) 推理能力实现指南

## 概述

让 MindLM 学会"先思考再回答"，提升数学、逻辑推理等任务的表现。核心思路：**用大模型生成带思考过程的数据，蒸馏到小模型上**。

---

## 什么是 COT

普通模型直接给答案：

```
用户: 农场有鸡和兔共35只，脚共94只，各多少？
助手: 鸡23只，兔12只。
```

COT 模型先展示推理过程：

```
用户: 农场有鸡和兔共35只，脚共94只，各多少？
助手: <think step>
设鸡x只，兔y只
x + y = 35
2x + 4y = 94
从第一个方程: x = 35 - y
代入: 2(35-y) + 4y = 94 -> 2y = 24 -> y = 12
所以 x = 23
验证: 23*2 + 12*4 = 94
</think step>
鸡23只，兔12只。
```

---

## 三种实现路径

| 路径 | 难度 | 效果 | 适合 MindLM |
|------|------|------|------------|
| SFT + CoT 数据蒸馏 | 低 | 中 | **推荐** |
| SFT + 自生成数据迭代 | 中 | 中高 | 可选 |
| 强化学习 (GRPO) | 高 | 高 | 不推荐（模型太小） |

---

## 路径 1：蒸馏（推荐）

### 整体流程

```
1. 收集问题（数学、逻辑、常识推理）
2. 用 DeepSeek-R1 / QwQ API 生成带 <think step> 的回答
3. 过滤低质量数据
4. 与普通 SFT 数据混合训练
```

### 第一步：收集问题

数据来源：

| 类型 | 来源 | 示例 |
|------|------|------|
| 数学推理 | GSM8K、MATH 数据集 | 应用题、计算题 |
| 逻辑推理 | BigBench、LogiQA | 逻辑判断、三段论 |
| 常识推理 | CommonsenseQA | 生活常识判断 |
| 中文推理 | 自建/爬取 | 中文应用题 |

### 第二步：用大模型生成 CoT 数据

```python
import pandas as pd
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="your-key"
)

questions = [
    "小明有15个苹果，给了小红3个，又给了小刚5个，还剩多少？",
    "一个长方形的长是宽的2倍，周长是36cm，求面积。",
    "如果所有猫都怕水，Tom是一只猫，Tom怕水吗？",
]

system_prompt = """你是一个推理助手。回答问题时，先用 <think step> 和 </think step> 标签展示推理过程，然后给出最终答案。"""

samples = []
for q in questions:
    resp = client.chat.completions.create(
        model="deepseek-reasoner",  # DeepSeek-R1
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
    )
    reasoning = resp.choices[0].message.reasoning_content
    answer = resp.choices[0].message.content
    full_answer = f"<think step>\n{reasoning}\n</think step>\n{answer}"
    samples.append({"history": "[]", "q": q, "a": full_answer})

df = pd.DataFrame(samples)
df.to_csv("data/sft_cot_data.csv", index=False)
print(f"生成 {len(df)} 条 CoT 训练数据")
```

### 第三步：数据质量过滤

```python
def filter_cot_data(df):
    """过滤低质量 CoT 数据"""
    good = []
    for _, row in df.iterrows():
        a = row["a"]
        # 必须包含完整的 think step 标签
        if "<think step>" not in a or "</think step>" not in a:
            continue
        # 思考过程不能太短（至少30字）
        think_part = a.split("<think step>")[1].split("</think step>")[0]
        if len(think_part) < 30:
            continue
        # 思考过程不能太长（小模型学不动）
        if len(think_part) > 500:
            continue
        good.append(row)
    return pd.DataFrame(good)
```

### 第四步：混合训练

```bash
# 将 CoT 数据与普通 SFT 数据混合
torchrun --nproc_per_node=2 sft.py \
    --resume_from out/mindlm_sft_768_linear_epoch2.pth \
    --model_config mindlm_1b \
    --epochs 2 \
    --learning_rate 3e-5 \
    --batch_size 80 \
    --accumulation_steps 2 \
    --data_path data/sft_cot_mixed.csv \
    --ddp \
    --num_workers 8
```

混合比例：**70% 普通对话 + 30% CoT 数据**。

---

## 路径 2：自生成迭代（STaR 方法）

如果不想花钱调 API，可以让模型自己生成 CoT 数据，然后筛选正确的继续训练：

```python
def star_iteration(model, tokenizer, questions_with_answers, iteration=0):
    """STaR: Self-Taught Reasoner 迭代"""
    new_data = []

    for item in questions_with_answers:
        q = item["question"]
        expected = item["answer"]

        # 让模型自己尝试推理
        prompt = f"<s>user\n{q}\n请一步步思考。</s>\n<s>assistant\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_new_tokens=512)
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=False)

        # 只保留答案正确的样本
        if str(expected) in response:
            new_data.append({"history": "[]", "q": q, "a": response})

    print(f"第 {iteration} 轮: {len(new_data)}/{len(questions_with_answers)} 答对")
    return new_data
```

流程：生成 -> 筛选正确 -> 训练 -> 再生成 -> 筛选 -> 再训练，迭代 3-5 轮。

---

## 推理脚本中的 CoT 支持

```python
def chat_with_cot(model, tokenizer, question, show_thinking=True):
    """带 CoT 的推理"""
    prompt = f"<s>user\n{question}</s>\n<s>assistant\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output = model.generate(
        input_ids,
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        max_new_tokens=512,
        temperature=0.3,  # CoT 推理用低温，更稳定
    )

    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=False)
    response = response.replace("</s>", "").strip()

    # 分离思考过程和最终答案
    if "<think step>" in response and "</think step>" in response:
        think = response.split("<think step>")[1].split("</think step>")[0].strip()
        answer = response.split("</think step>")[-1].strip()
        if show_thinking:
            print(f"思考过程:\n{think}\n")
        return answer
    return response
```

---

## 效果评估

```python
def evaluate_cot(model, tokenizer, test_data):
    """评估 CoT 准确率"""
    correct = 0
    total = len(test_data)

    for item in test_data:
        q = item["question"]
        expected = str(item["answer"])
        answer = chat_with_cot(model, tokenizer, q, show_thinking=False)
        if expected in answer:
            correct += 1

    accuracy = correct / total
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    return accuracy
```

### 推荐测试集

| 测试集 | 类型 | 难度 | 来源 |
|--------|------|------|------|
| GSM8K 前100题 | 数学 | 小学 | HuggingFace |
| BigBench Logic | 逻辑 | 中等 | HuggingFace |
| 自建中文推理50题 | 综合 | 自定义 | 自己写 |

---

## 95M 模型的 CoT 天花板

| 任务 | 能否学会 | 说明 |
|------|---------|------|
| 两步以内算术 | 能 | 15+23=? |
| 简单应用题 | 能 | GSM8K 简单题 |
| 三段论推理 | 能 | A->B, B->C 所以 A->C |
| 多步数学证明 | 勉强 | 3步以上容易出错 |
| 复杂逻辑链 | 难 | 超过模型容量 |
| 抽象数学 | 不能 | 需要更大参数量 |

**预期**：95M 模型做 CoT 后，GSM8K 准确率可从约 5% 提升到 20-30%。

---

## 数据质量检查清单

- [ ] 所有样本都有完整的 `<think step>...</think step>` 标签
- [ ] 思考过程中的计算无错误
- [ ] 思考过程长度在 50-500 字之间
- [ ] 训练数据混合比例约 70% 普通对话 + 30% CoT
- [ ] 覆盖从简单到困难的梯度
- [ ] API 生成时用 temperature=0.3

---

## 参考资源

- **DeepSeek-R1 论文**: 纯 RL 涌现 CoT 的开创性工作
- **STaR (Self-Taught Reasoner)**: 自生成迭代方法
- **Distilling Step-by-Step**: 蒸馏 CoT 到小模型的方法论
- **OpenMathInstruct-2**: 开源数学推理训练数据
- **GSM8K**: 小学数学推理基准测试集
