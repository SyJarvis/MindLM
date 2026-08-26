"""Generate text from a MindLM pretraining checkpoint."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling_mindlm import MindLM
from training_utils import build_model_config, load_model_checkpoint


PROMPTS = ["人工智能", "中国的首都是", "机器学习是"]


def main():
    parser = argparse.ArgumentParser(description="MindLM pretraining evaluation")
    parser.add_argument("--config", choices=("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.7b"), default="mindlm_0.1b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = str(
            PROJECT_ROOT / ("qwen3_tokenizer" if args.config == "mindlm_0.7b" else "mindlm_tokenizer")
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    load_model_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        generated = model.generate(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"输入: {prompt}\n输出: {tokenizer.decode(generated[0], skip_special_tokens=True)}\n")


if __name__ == "__main__":
    main()
