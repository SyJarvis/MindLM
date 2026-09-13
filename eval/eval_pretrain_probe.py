"""Generation probe for MindLM pretraining checkpoints.

Runs a fixed prompt set through a checkpoint and reports per-prompt statistics
(length, distinct-token ratio, EOS behaviour) plus an aggregate summary. Meant to
be run at regular update milestones: a plain loss number cannot show whether a
checkpoint still produces coherent text, loops, or stops at all.

Sampling is seeded per prompt, so repeated runs on the same checkpoint are
directly comparable.
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


DEFAULT_PROMPTS = [
    "人工智能",
    "中国的首都是",
    "机器学习是",
    "今天天气",
    "请介绍一下北京",
    "Python是一种",
    "水在标准大气压下的沸点是",
    "深度学习",
]


def parse_prompts(spec):
    """Accept a comma-separated list, a newline-delimited file, or None."""
    if not spec:
        return list(DEFAULT_PROMPTS)
    candidate = Path(spec)
    if candidate.is_file():
        return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item.strip() for item in spec.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="MindLM pretraining generation probe")
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.1b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompts", default=None, help="comma-separated prompts, or a file with one prompt per line")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--greedy", action="store_true", help="decode greedily instead of sampling")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--show_ids", action="store_true", help="print generated token ids")
    parser.add_argument("--json_out", default=None, help="write results as JSON to this path")
    args = parser.parse_args()

    prompts = parse_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        default_tokenizer_path(args.tokenizer_path), trust_remote_code=True
    )
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    metadata = load_checkpoint_with_tokenizer_check(model, args.checkpoint, args.device, tokenizer)
    model.eval()
    step = metadata.get("step") if isinstance(metadata, dict) else None
    print(f"config={args.config}\ncheckpoint={args.checkpoint}\nstep={step}\n"
          f"do_sample={not args.greedy} temperature={args.temperature} top_k={args.top_k} "
          f"max_new_tokens={args.max_new_tokens}\neos_id={tokenizer.eos_token_id}\n")

    results = []
    total_new = total_distinct = eos_hits = 0
    for index, prompt in enumerate(prompts):
        torch.manual_seed(args.seed + index)
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
        new_ids = generated[0][input_ids.shape[1]:].tolist()
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        hit_eos = tokenizer.eos_token_id in new_ids
        distinct = len(set(new_ids))
        ratio = distinct / len(new_ids) if new_ids else 0.0

        total_new += len(new_ids)
        total_distinct += distinct
        eos_hits += int(hit_eos)
        results.append({
            "prompt": prompt,
            "new_tokens": len(new_ids),
            "distinct_tokens": distinct,
            "distinct_ratio": round(ratio, 3),
            "hit_eos": hit_eos,
            "text": text,
            "ids": new_ids if args.show_ids else None,
        })
        print(f"[{prompt}] new_tokens={len(new_ids)} distinct={distinct} "
              f"ratio={ratio:.2f} hit_eos={hit_eos}")
        if args.show_ids:
            print(f"  raw_ids={new_ids}")
        print(f"  text={text!r}\n")

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "step": step,
        "prompts": len(prompts),
        "total_new_tokens": total_new,
        "distinct_token_ratio": round(total_distinct / total_new, 3) if total_new else 0.0,
        "avg_new_tokens": round(total_new / len(prompts), 1) if prompts else 0.0,
        "eos_hits": f"{eos_hits}/{len(prompts)}",
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.json_out:
        payload = {"summary": summary, "results": results}
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
