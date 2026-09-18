# MindLM Function Call 实现指南

## 概述

让 MindLM 学会根据用户指令调用外部工具（查天气、计算器、搜索等）。不需要改模型架构，只需要构造专门的 SFT 数据再训一轮。

---

## 整体流程

```
当前 SFT 模型（已有对话能力）
    ↓
构造 Function Call 训练数据
    ↓
混合普通 SFT 数据 + FC 数据训练
    ↓
推理时解析模型输出，执行函数，返回结果
```

---

## 第一步：定义 Function Call 格式

### 工具定义格式

在 system prompt 中描述可用工具，使用简单文本格式（小模型友好）：

```
你可以使用以下工具：
1. get_weather(city: str) - 查询指定城市的天气
2. calculator(expression: str) - 计算数学表达式
3. search(query: str) - 搜索互联网信息

需要调用工具时，严格使用以下格式：
<tool_call name="函数名">参数名=参数值</tool_call >

不需要工具时，直接回答即可。
```

### 三种对话场景

**场景 A：直接回答（不需要工具）**

```
<s>user
你好，你是谁？</s >
<s>assistant
你好！我是 MindLM，一个 AI 助手，有什么可以帮助你的吗？</s >
```

**场景 B：需要调用工具**

```
<s>user
北京今天天气怎么样？</s >
<s>assistant
<tool_call name="get_weather">city=北京</tool_call ></s >
```

**场景 C：拿到工具结果后回复**

```
<s>user
北京今天天气怎么样？</s >
<s>assistant
<tool_call name="get_weather">city=北京</tool_call ></s >
<s>tool
北京：晴天，气温 25°C，湿度 40%，微风</s >
<s>assistant
北京今天的天气是晴天，气温 25°C，湿度 40%，有微风，体感舒适。适合出门活动！</s >
```

---

## 第二步：构造训练数据

### 数据格式

CSV 文件，列为 `history`, `q`, `a`，与现有 SFTDataset 兼容。

### 数据比例建议

| 数据类型 | 占比 | 数量（参考） |
|---------|------|------------|
| 普通对话（不用工具） | 60% | 3 万条 |
| 工具调用（场景 B） | 15% | 7500 条 |
| 工具调用 + 回复（场景 C） | 25% | 1.25 万条 |

### 数据生成方法

**方法 1：用大模型 API 批量生成（推荐）**

```python
import json
from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com", api_key="your-key")

# 定义工具列表
tools = [
    {"name": "get_weather", "desc": "查询天气", "params": "city: str"},
    {"name": "calculator", "desc": "计算数学表达式", "params": "expression: str"},
    {"name": "search", "desc": "搜索信息", "params": "query: str"},
    {"name": "get_time", "desc": "获取当前时间", "params": "timezone: str"},
    {"name": "translate", "desc": "翻译文本", "params": "text: str, target_lang: str"},
]

tool_desc = "\n".join([f'{i+1}. {t["name"]}({t["params"]}) - {t["desc"]}' for i, t in enumerate(tools)])

system_prompt = f"""你是一个 AI 助手。你可以使用以下工具：
{tool_desc}

需要调用工具时，严格使用以下格式：
<tool_call name="函数名">参数名=参数值</tool_call >
不需要工具时，直接回答。"""

# 生成训练样本
prompts = [
    "北京今天天气怎么样？",       # → get_weather
    "123 * 456 等于多少？",       # → calculator
    "苹果公司的创始人是谁？",     # → search
    "你好，帮我写一首诗",         # → 直接回答
    "现在东京几点了？",           # → get_time
    "把'我爱学习'翻译成英文",     # → translate
]

samples = []
for q in prompts:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
        temperature=0.7,
    )
    a = resp.choices[0].message.content
    samples.append({"history": "[]", "q": q, "a": a})

# 保存为 CSV
import pandas as pd
df = pd.DataFrame(samples)
df.to_csv("data/sft_fc_data.csv", index=False)
```

**方法 2：模板生成（免费，快速）**

```python
import pandas as pd
import random

# 城市、表达式等模板
cities = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "南京"]
templates = []

# 查天气模板
weather_questions = [
    "{city}今天天气怎么样？",
    "{city}明天会下雨吗？",
    "帮我查一下{city}的天气",
]
weather_answers = [
    '<tool_call name="get_weather">city={city}</tool_call >',
]

for _ in range(5000):
    city = random.choice(cities)
    q = random.choice(weather_questions).format(city=city)
    a = random.choice(weather_answers).format(city=city)
    templates.append({"history": "[]", "q": q, "a": a})

# 计算器模板
calc_questions = [
    "{a} + {b} 等于多少？",
    "{a} 乘以 {b} 是多少？",
    "计算 {a} / {b}",
]
for _ in range(3000):
    a_num = random.randint(10, 999)
    b_num = random.randint(1, 100)
    ops = {"+": f"{a_num}+{b_num}", "乘以": f"{a_num}*{b_num}", "/": f"{a_num}/{b_num}"}
    q = random.choice(calc_questions).format(a=a_num, b=b_num)
    expr = ops[random.choice(list(ops.keys()))]
    templates.append({"history": "[]", "q": q, "a": f'<tool_call name="calculator">expression={expr}</tool_call >'})

# 直接回答模板（不需要工具）
direct_qa = [
    ("你好", "你好！有什么可以帮助你的吗？"),
    ("你是谁", "我是 MindLM，一个 AI 助手。"),
    ("讲个笑话", "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25。"),
]
for q, a in direct_qa:
    for _ in range(1000):
        templates.append({"history": "[]", "q": q, "a": a})

df = pd.DataFrame(templates)
df.to_csv("data/sft_fc_data.csv", index=False)
```

---

## 第三步：训练

### 修改 SFTDataset 的 chat template

当前 tokenizer 的 chat template 不支持 `tool` 角色。需要在 `tokenizer_config.json` 的 `chat_template` 中加入对 tool 角色的处理：

```
{% if message['role'] == 'tool' %}{{ '<s>tool\n' + content + '</s>\n' }}{% endif %}
```

完整 chat_template：

```
{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ '<s>system\n' + system_message + '</s>\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\n' + content + '</s>\n<s>assistant\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>\n' }}{% elif message['role'] == 'tool' %}{{ '<s>tool\n' + content + '</s>\n<s>assistant\n' }}{% endif %}{% endfor %}
```

### 训练命令

```bash
torchrun --nproc_per_node=2 sft.py \
    --resume_from out/mindlm_sft_768_linear_epoch2.pth \
    --model_config mindlm_1b \
    --epochs 2 \
    --learning_rate 3e-5 \
    --batch_size 80 \
    --accumulation_steps 2 \
    --data_path data/sft_fc_data.csv \
    --ddp \
    --num_workers 8
```

注意：
- `resume_from` 从普通 SFT 的 checkpoint 开始
- `learning_rate` 比普通 SFT 更低（3e-5 vs 5e-5），防止遗忘
- `epochs` 只需要 1-2 轮

---

## 第四步：推理时解析

推理脚本中需要解析模型的 `<tool_call >` 输出：

```python
import re

def parse_tool_call(text):
    """解析模型输出中的工具调用"""
    pattern = r'<tool_call name="(\w+)">(.*?)</tool_call >'
    match = re.search(pattern, text)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        # 解析 key=value 参数
        args = {}
        for pair in args_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                args[k.strip()] = v.strip()
        return func_name, args
    return None, None

def execute_tool(func_name, args):
    """执行工具调用"""
    if func_name == "get_weather":
        return get_weather(args["city"])
    elif func_name == "calculator":
        return str(eval(args["expression"]))  # 注意：生产环境需沙箱化
    elif func_name == "search":
        return search(args["query"])
    return None

# 推理循环
def chat_with_tools(model, tokenizer, user_input, max_rounds=3):
    messages = [{"role": "user", "content": user_input}]

    for _ in range(max_rounds):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        output = model.generate(input_ids, eos=tokenizer.eos_token_id, max_new_tokens=256)
        response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=False)
        response = response.replace("</s>", "").strip()

        # 检查是否调用了工具
        func_name, args = parse_tool_call(response)
        if func_name:
            print(f"[调用工具] {func_name}({args})")
            tool_result = execute_tool(func_name, args)
            # 把工具结果加入对话，继续生成
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "content": str(tool_result)})
        else:
            return response

    return response
```

---

## 建议的工具列表（适合小模型）

小模型参数少，工具定义要简单，参数要少：

| 工具 | 参数 | 难度 |
|------|------|------|
| `get_weather` | city | 简单 |
| `calculator` | expression | 简单 |
| `search` | query | 简单 |
| `get_time` | timezone | 简单 |
| `translate` | text, target_lang | 中等 |
| `send_email` | to, subject, body | 中等 |
| `query_database` | sql | 困难（不推荐小模型） |

**原则**：每个工具参数不超过 3 个，参数名语义清晰。

---

## 训练数据质量检查

训练前务必检查以下几点：

1. **格式一致性**：所有 `<tool_call >` 的格式完全一致，不能有变体
2. **参数正确性**：参数名和工具定义匹配，不能出现幻觉参数
3. **混合比例**：60% 普通对话 + 40% 工具调用，防止模型"过度调用"
4. **覆盖度**：每个工具至少 2000 条训练样本
5. **负样本**：包含"不需要工具"的问题，防止模型什么都调工具

---

## 参考资源

- **Toolformer**: 教模型使用工具的经典论文
- **Gorilla**: 基于 LLaMA 的 API 调用模型
- **Qwen2.5 Function Call**: 实用的 function call 格式设计
- **Hermes Function Calling**: 开源的 function call 训练数据集
