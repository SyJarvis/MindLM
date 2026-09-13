"""Batch three-question eval for a MindLM SFT checkpoint (CPU/GPU selectable).

Derived from eval/eval_sft.py with explicit sampling controls: --temperature
and --top_k are exposed so quality checks are reproducible (low temperature
reduces sampling variance).
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling_mindlm import MindLM
from training_utils import build_model_config, load_model_checkpoint


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
    generated = model.generate(
        input_ids=input_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    response_ids = generated[0, input_ids.size(1):]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="MindLM SFT batch eval with sampling controls")
    parser.add_argument("--config", choices=("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.7b"), default="mindlm_0.1b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default 0.1 for stable eval)")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--enable_thinking", action="store_true", help="Let the model generate a <think> block (off by default)")
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = str(PROJECT_ROOT / "qwen3_tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    load_model_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    for question in ("你好，你是谁？", "请介绍一下人工智能。", "如何学习编程？"):
        answer = chat(
            model, tokenizer, [{"role": "user", "content": question}], args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            enable_thinking=args.enable_thinking,
        )
        print(f"用户: {question}\n助手: {answer}\n")


if __name__ == "__main__":
    main()
