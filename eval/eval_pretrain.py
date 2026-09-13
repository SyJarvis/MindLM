"""Generate text from a MindLM pretraining checkpoint."""

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


PROMPTS = ["人工智能", "中国的首都是", "机器学习是"]


def main():
    parser = argparse.ArgumentParser(description="MindLM pretraining evaluation")
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.1b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--greedy", action="store_true", help="decode greedily instead of sampling")
    parser.add_argument("--json_out", default=None, help="write the prompts and outputs as JSON to this path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        default_tokenizer_path(args.tokenizer_path), trust_remote_code=True
    )
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    metadata = load_checkpoint_with_tokenizer_check(model, args.checkpoint, args.device, tokenizer)
    model.eval()

    print(f"config={args.config} checkpoint={args.checkpoint} device={args.device} "
          f"do_sample={not args.greedy}\n")

    results = []
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        with inference_context(args.device):
            generated = model.generate(
                input_ids=input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                do_sample=not args.greedy,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "text": text,
            "new_tokens": int(generated.shape[1] - input_ids.shape[1]),
        })
        print(f"输入: {prompt}\n输出: {text}\n")

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "step": metadata.get("step") if isinstance(metadata, dict) else None,
        "do_sample": not args.greedy,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "prompts": len(results),
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
