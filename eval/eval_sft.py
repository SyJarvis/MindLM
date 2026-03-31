"""
MindLM SFT 推理测试脚本
加载 SFT 权重，进行对话生成测试
"""

import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from modeling_mindlm import MindLM, MindLMConfig
from config import load_config

# ==================== 配置 ====================
CONFIG_NAME = "mindlm_0.1b"
CHECKPOINT_PATH = "out/mindlm_sft_768_linear_epoch0.pth"
TOKENIZER_PATH = "mindlm_tokenizer"

# 生成参数
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_K = 8

# ==================== 测试问题 ====================
TEST_PROMPTS = [
    "你好，你是谁？",
    "请介绍一下人工智能。",
    "什么是深度学习？",
    "请给我讲一个故事。",
    "中国的首都是哪里？",
    "请解释一下机器学习的概念。",
    "如何学习编程？",
    "请用一句话描述春天。",
    "1+1等于多少？",
]


def load_model(config_name, checkpoint_path, tokenizer_path, device):
    """加载模型和tokenizer"""
    # 加载配置
    model_config_dict = load_config(config_name)
    print(f"模型配置: {config_name} (dim={model_config_dict['dim']}, layers={model_config_dict['n_layers']})")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer 词表大小: {len(tokenizer)}")
    model_config_dict['vocab_size'] = len(tokenizer)

    # 创建模型
    config = MindLMConfig(
        dim=model_config_dict['dim'],
        n_layers=model_config_dict['n_layers'],
        n_heads=model_config_dict['n_heads'],
        n_kv_heads=model_config_dict['n_kv_heads'],
        linear_attn_heads=model_config_dict.get('linear_attn_heads'),
        vocab_size=model_config_dict['vocab_size'],
        max_seq_len=model_config_dict['max_seq_len'],
        dropout=0.0,
        norm_eps=model_config_dict.get('norm_eps', 1e-6),
        hidden_dim=model_config_dict.get('hidden_dim'),
        multiple_of=model_config_dict.get('multiple_of', 256),
        use_moe=model_config_dict.get('use_moe', False),
        n_routed_experts=model_config_dict.get('n_routed_experts', 4),
        num_experts_per_tok=model_config_dict.get('num_experts_per_tok', 2),
        n_shared_experts=model_config_dict.get('n_shared_experts', 1),
        scoring_func='softmax',
        aux_loss_alpha=model_config_dict.get('aux_loss_alpha', 0.01),
        seq_aux=True,
        norm_topk_prob=True,
        use_linear_attn=model_config_dict.get('use_linear_attn', True),
        layer_types=model_config_dict.get('layer_types'),
        conv_kernel_size=model_config_dict.get('conv_kernel_size', 4),
    )

    model = MindLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 加载权重
    print(f"加载权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 兼容完整 checkpoint 和纯 state_dict
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("模型加载成功！\n")

    return model, tokenizer, config


def chat(model, tokenizer, question, max_new_tokens=256, temperature=0.7, top_k=8, device="cpu"):
    """单轮对话"""
    # 构建 prompt
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_text = prompt
    with torch.no_grad():
        for generated in model.generate(
            input_ids,
            eos=tokenizer.convert_tokens_to_ids("</s>"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)

    # 提取 assistant 回复
    if "<s>assistant\n" in generated_text:
        answer = generated_text.split("<s>assistant\n")[-1]
        answer = answer.replace("</s>", "").strip()
    else:
        answer = generated_text.strip()

    return answer


def interactive_chat(model, tokenizer, device="cpu"):
    """交互式对话"""
    print("\n" + "=" * 60)
    print("MindLM SFT 对话模式（输入 'quit' 退出）")
    print("=" * 60 + "\n")

    history = []
    while True:
        try:
            user_input = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话。")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("退出对话。")
            break

        # 构建带历史的 prompt
        messages = []
        for h in history[-6:]:  # 保留最近3轮对话
            messages.append(h)
        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        answer = ""
        with torch.no_grad():
            for generated in model.generate(
                input_ids,
                eos=tokenizer.convert_tokens_to_ids("</s>"),
                max_new_tokens=256,
                temperature=0.7,
                top_k=8,
            ):
                full_text = tokenizer.decode(generated[0], skip_special_tokens=False)

        # 提取最后一段 assistant 回复
        if "<s>assistant\n" in full_text:
            answer = full_text.split("<s>assistant\n")[-1]
            answer = answer.replace("</s>", "").strip()
        else:
            answer = full_text.strip()

        print(f"助手: {answer}\n")

        # 更新历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MindLM SFT 推理测试")
    parser.add_argument("--config", type=str, default=CONFIG_NAME, help="模型配置名")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="SFT 权重路径")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--interactive", action="store_true", help="交互式对话模式")
    args = parser.parse_args()

    # 加载模型
    model, tokenizer, config = load_model(args.config, args.checkpoint, TOKENIZER_PATH, args.device)

    if args.interactive:
        # 交互模式
        interactive_chat(model, tokenizer, args.device)
    else:
        # 批量测试
        print("=" * 60)
        print("SFT 模型批量测试")
        print("=" * 60)

        for question in TEST_PROMPTS:
            answer = chat(model, tokenizer, question, device=args.device)
            print(f"用户: {question}")
            print(f"助手: {answer}")
            print("-" * 40)

        print("\n测试完成！使用 --interactive 进入交互对话模式：")
        print(f"  python eval/eval_sft.py --checkpoint {args.checkpoint} --interactive")
