"""Quick mid-training multi-turn conversation test on the latest SFT checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import default_tokenizer_path, inference_context, load_checkpoint_with_tokenizer_check, supported_configs
from modeling_mindlm import MindLM
from training_utils import build_model_config

CONVERSATIONS = [
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


def chat(model, tokenizer, messages, device, max_new_tokens=200, temperature=0.1, top_k=8):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with inference_context(device):
        generated = model.generate(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    response_ids = generated[0, input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return {"text": text, "new_tokens": int(response_ids.numel()),
            "hit_eos": int(response_ids.numel()) < max_new_tokens}


def main():
    parser = argparse.ArgumentParser(description="MindLM SFT multi-turn eval")
    parser.add_argument("checkpoint", help="SFT checkpoint to evaluate")
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.2b_gdn")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed; makes temperature-based decoding reproducible")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # sampling is stochastic: pin it so a quality record can be replayed
    tokenizer = AutoTokenizer.from_pretrained(
        default_tokenizer_path(args.tokenizer_path), trust_remote_code=True
    )
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    metadata = load_checkpoint_with_tokenizer_check(model, args.checkpoint, args.device, tokenizer)
    model.eval()
    print(f"config={args.config} checkpoint={args.checkpoint} device={args.device} "
          f"step={metadata.get('step') if isinstance(metadata, dict) else None}\n")

    results = []
    for index, messages in enumerate(CONVERSATIONS, 1):
        print(f"===== 多轮测试 {index} =====")
        for message in messages:
            print(f"{'用户' if message['role'] == 'user' else '助手'}: {message['content']}")
        reply = chat(model, tokenizer, messages, args.device,
                     max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(f"→ 模型回答: {reply['text']}\n")
        results.append({"conversation": index, "messages": messages, **reply})

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "step": metadata.get("step") if isinstance(metadata, dict) else None,
        "training_stage": metadata.get("training_stage") if isinstance(metadata, dict) else None,
        "device": args.device,
        "temperature": args.temperature,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "conversations": len(results),
        "new_tokens_total": sum(item["new_tokens"] for item in results),
        "eos_hits": f"{sum(item['hit_eos'] for item in results)}/{len(results)}",
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
