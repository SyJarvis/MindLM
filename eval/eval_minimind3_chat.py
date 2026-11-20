"""Chat (dialogue) evaluation for minimind-3 (HuggingFace AutoModel).

Same questions and multi-turn conversations as eval_sft_batch.py and
eval_sft_multi.py for direct comparison with MindLM.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BATCH_QUESTIONS = ("你好，你是谁？", "请介绍一下人工智能。", "如何学习编程？")

MULTI_CONVERSATIONS = [
    # 测试 1: 上下文记忆（名字）
    [
        {"role": "user", "content": "你好，我叫小明，我喜欢打篮球。"},
        {"role": "assistant", "content": "你好小明！打篮球是很棒的运动。请问有什么可以帮你的吗？"},
        {"role": "user", "content": "我叫什么名字？我喜欢什么运动？"},
    ],
    # 测试 2: 指代消解（"它"指向）
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


def chat(model, tokenizer, messages, device, max_new_tokens=256, temperature=0.7, top_k=8):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prompt, str):
        parts = []
        for block in prompt:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        prompt = "".join(parts)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
        )
    response_ids = generated[0, input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return {"text": text, "new_tokens": int(response_ids.numel()),
            "hit_eos": int(response_ids.numel()) < max_new_tokens}


def main():
    parser = argparse.ArgumentParser(description="minimind-3 chat evaluation")
    parser.add_argument("--checkpoint",
                        default="/root/.cache/modelscope/models/gongjy--minimind-3/snapshots/master")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json_out", default=None)
    parser.add_argument("--mode", choices=["batch", "multi", "all"], default="all",
                        help="batch: single-turn 3 questions; multi: multi-turn 3 conversations; all: both")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Loading model from {args.checkpoint} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True)
    model.eval()
    print(f"Model loaded. device={args.device} temperature={args.temperature} seed={args.seed}\n")

    output = {"model": "minimind-3", "checkpoint": args.checkpoint, "device": args.device,
              "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed}

    if args.mode in ("batch", "all"):
        print("===== BATCH (single-turn) =====")
        batch_results = []
        for question in BATCH_QUESTIONS:
            reply = chat(model, tokenizer, [{"role": "user", "content": question}],
                         args.device, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
            print(f"用户: {question}\n助手: {reply['text']}\n")
            batch_results.append({"question": question, **reply})
        output["batch"] = {
            "questions": len(batch_results),
            "new_tokens_total": sum(item["new_tokens"] for item in batch_results),
            "eos_hits": f"{sum(item['hit_eos'] for item in batch_results)}/{len(batch_results)}",
            "results": batch_results,
        }

    if args.mode in ("multi", "all"):
        print("===== MULTI-TURN =====")
        multi_results = []
        for index, messages in enumerate(MULTI_CONVERSATIONS, 1):
            print(f"===== 多轮测试 {index} =====")
            for msg in messages:
                print(f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}")
            reply = chat(model, tokenizer, messages, args.device,
                         max_new_tokens=min(args.max_new_tokens, 200),
                         temperature=args.temperature, top_k=args.top_k)
            print(f"→ 模型回答: {reply['text']}\n")
            multi_results.append({"conversation": index, "messages": messages, **reply})
        output["multi"] = {
            "conversations": len(multi_results),
            "new_tokens_total": sum(item["new_tokens"] for item in multi_results),
            "eos_hits": f"{sum(item['hit_eos'] for item in multi_results)}/{len(multi_results)}",
            "results": multi_results,
        }

    print(json.dumps({k: v for k, v in output.items() if k != "results"}, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
