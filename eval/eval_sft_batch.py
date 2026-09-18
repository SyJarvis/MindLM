"""Batch three-question eval for a MindLM SFT checkpoint (CPU/GPU selectable).

Derived from eval/eval_sft.py with explicit sampling controls: --temperature
and --top_k are exposed so quality checks are reproducible (low temperature
reduces sampling variance).

Model config and tokenizer handling come from ``eval_common`` so every current
config (including ``mindlm_0.2b_gdn``) is accepted instead of a hardcoded list.
"""

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

QUESTIONS = ("你好，你是谁？", "请介绍一下人工智能。", "如何学习编程？")


def chat(model, tokenizer, messages, device, max_new_tokens=256, temperature=0.7, top_k=8, enable_thinking=False):
    # The Qwen3 template prepends an empty <think></think> block when thinking
    # is disabled, matching how SFT samples were rendered during training.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(prompt, str):
        # Qwen3 template may return a list of content blocks when history
        # contains non-string/structured content; flatten to a string.
        parts = []
        for block in prompt:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        prompt = "".join(parts)
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
    return {"text": text, "new_tokens": int(response_ids.numel()), "hit_eos": int(response_ids.numel()) < max_new_tokens}


def main():
    parser = argparse.ArgumentParser(description="MindLM SFT batch eval with sampling controls")
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.2b_gdn")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default 0.1 for stable eval)")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--greedy", action="store_true", help="decode greedily instead of sampling")
    parser.add_argument("--enable_thinking", action="store_true", help="Let the model generate a <think> block (off by default)")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed; makes temperature-based decoding reproducible")
    parser.add_argument("--json_out", default=None, help="write the questions and outputs as JSON to this path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)  # sampling is stochastic: pin it so a quality record can be replayed
    tokenizer = AutoTokenizer.from_pretrained(
        default_tokenizer_path(args.tokenizer_path), trust_remote_code=True
    )
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    metadata = load_checkpoint_with_tokenizer_check(model, args.checkpoint, args.device, tokenizer)
    model.eval()

    print(f"config={args.config} checkpoint={args.checkpoint} device={args.device} "
          f"do_sample={not args.greedy} temperature={args.temperature} top_k={args.top_k}\n")

    results = []
    for question in QUESTIONS:
        reply = chat(
            model, tokenizer, [{"role": "user", "content": question}], args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            enable_thinking=args.enable_thinking,
        )
        print(f"用户: {question}\n助手: {reply['text']}\n")
        results.append({"question": question, **reply})

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "step": metadata.get("step") if isinstance(metadata, dict) else None,
        "training_stage": metadata.get("training_stage") if isinstance(metadata, dict) else None,
        "device": args.device,
        "do_sample": not args.greedy,
        "temperature": args.temperature,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "questions": len(results),
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
