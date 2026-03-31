"""
MindLM 推理示例
使用 transformers 库加载 mindlm_1b_sft 模型进行对话推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./mindlm_1b_sft"


def load_model(model_path=MODEL_PATH, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def chat(model, tokenizer, messages, max_new_tokens=256, temperature=0.7, top_k=8):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    eos_token_id = tokenizer.convert_tokens_to_ids("</s>")

    with torch.inference_mode():
        generated = inputs["input_ids"]
        for _ in range(max_new_tokens):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                logits = logits.nan_to_num(posinf=0.0, neginf=0.0)
                probs = torch.softmax(logits, dim=-1)
                probs = torch.clamp(probs, min=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                next_token = torch.multinomial(probs, num_samples=1)
                print
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            if next_token.item() == eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=1)

    response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def interactive_chat(model, tokenizer):
    print("MindLM 对话 (输入 'quit' 退出)\n")
    messages = []

    while True:
        user_input = input("用户: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})
        reply = chat(model, tokenizer, messages)
        print(f"助手: {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"设备: {args.device}\n")

    # 单轮示例
    messages = [{"role": "user", "content": "你好，请介绍一下你自己"}]
    reply = chat(model, tokenizer, messages)
    print(f"用户: {messages[0]['content']}")
    print(f"助手: {reply}\n")

    # 进入交互模式
    interactive_chat(model, tokenizer)
