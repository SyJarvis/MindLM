"""Interactive or batch chat evaluation for a MindLM SFT checkpoint."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modeling_mindlm import MindLM
from training_utils import build_model_config, load_model_checkpoint


def chat(model, tokenizer, messages, device, max_new_tokens=256, temperature=0.7, top_k=8):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    parser = argparse.ArgumentParser(description="MindLM SFT evaluation")
    parser.add_argument("--config", choices=("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.8b"), default="mindlm_0.1b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = str(
            PROJECT_ROOT / ("qwen3_tokenizer" if args.config == "mindlm_0.8b" else "mindlm_tokenizer")
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    load_model_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    if not args.interactive:
        for question in ("你好，你是谁？", "请介绍一下人工智能。", "如何学习编程？"):
            answer = chat(model, tokenizer, [{"role": "user", "content": question}], args.device)
            print(f"用户: {question}\n助手: {answer}\n")
        return

    history = []
    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        messages = history[-6:] + [{"role": "user", "content": user_input}]
        answer = chat(model, tokenizer, messages, args.device)
        print(f"助手: {answer}\n")
        history.extend(({"role": "user", "content": user_input}, {"role": "assistant", "content": answer}))


if __name__ == "__main__":
    main()
