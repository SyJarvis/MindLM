"""Quick mid-training multi-turn conversation test on the latest SFT checkpoint."""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling_mindlm import MindLM
from training_utils import build_model_config, load_model_checkpoint


def chat(model, tokenizer, messages, device, max_new_tokens=200, temperature=0.1, top_k=8):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = model.generate(
        input_ids=input_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0, input_ids.size(1):], skip_special_tokens=True).strip()


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "out/mindlm_sft_mindlm_0.1b_latest.pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / "qwen3_tokenizer", trust_remote_code=True)
    model = MindLM(build_model_config("mindlm_0.1b", tokenizer)).to(device)
    load_model_checkpoint(model, checkpoint, device)
    model.eval()
    print(f"checkpoint: {checkpoint}\n")

    conversations = [
        # 测试 1: 上下文记忆（名字）
        [
            {"role": "user", "content": "你好，我叫小明，我喜欢打篮球。"},
            {"role": "assistant", "content": "你好小明！打篮球是很棒的运动。请问有什么可以帮你的吗？"},
            {"role": "user", "content": "我叫什么名字？我喜欢什么运动？"},
        ],
        # 测试 2: 指代消解（“它”指向）
        [
            {"role": "user", "content": "我想养一只猫，但是担心照顾不好。"},
            {"role": "assistant", "content": "养猫其实不难，每天喂食、清理猫砂即可。猫比较独立，适合忙碌的人。"},
            {"role": "user", "content": "那它一个月大概要花多少钱？"},
        ],
        # 测试 3: 多轮递进（话题延续）
        [
            {"role": "user", "content": "推荐几个适合初学者的编程语言。"},
            {"role": "assistant", "content": "初学者可以从Python入手，语法简单；如果对网页感兴趣可以学JavaScript；想深入底层可以选C语言。"},
            {"role": "user", "content": "第一个语言学大概要多久？之后能做什么项目？"},
        ],
    ]

    for i, msgs in enumerate(conversations, 1):
        print(f"===== 多轮测试 {i} =====")
        for m in msgs:
            print(f"{'用户' if m['role']=='user' else '助手'}: {m['content']}")
        reply = chat(model, tokenizer, msgs, device)
        print(f"→ 模型回答: {reply}\n")


if __name__ == "__main__":
    main()
