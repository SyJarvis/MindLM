"""
MindLM Tokenizer Training Script
基于 BPE 算法训练词表大小为 6400 的分词器
"""

import json
import os
import random

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import AutoTokenizer

random.seed(42)

# ==================== 配置 ====================
VOCAB_SIZE = 6400
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tokenizer_train.jsonl")
TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "mindlm_tokenizer")

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]


def read_texts_from_jsonl(file_path):
    """从 JSONL 文件逐行读取文本，节省内存"""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data["text"]


def train():
    # 初始化 BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # ByteLevel 预分词器：先将文本转为 UTF-8 字节再分词，天然支持多语言
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 训练
    texts = read_texts_from_jsonl(DATA_PATH)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 验证特殊 token 索引
    for i, tok in enumerate(SPECIAL_TOKENS):
        assert tokenizer.token_to_id(tok) == i, f"{tok} index mismatch"

    # 保存 tokenizer.json
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    tokenizer.model.save(TOKENIZER_DIR)

    # 生成 tokenizer_config.json（兼容 transformers PreTrainedTokenizerFast）
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(i): {
                "content": tok,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for i, tok in enumerate(SPECIAL_TOKENS)
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "clean_up_tokenization_spaces": False,
        "legacy": True,
        "model_max_length": 32768,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "spaces_between_special_tokens": False,
        "use_default_system_prompt": False,
        "chat_template": (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% endif %}"
            "{% if system_message is defined %}"
            "{{ system_message }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ content + '</s>' + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        ),
    }

    with open(os.path.join(TOKENIZER_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 打印统计信息
    vocab = tokenizer.get_vocab()
    print(f"Tokenizer 训练完成")
    print(f"  词表大小: {len(vocab)}")
    print(f"  保存路径: {TOKENIZER_DIR}")
    print(f"  特殊 token: {SPECIAL_TOKENS}")


def eval():
    """加载并验证训练好的 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

    print(f"词表大小: {len(tokenizer)}")

    # 基本编解码测试
    text = "你好世界，这是一个测试。"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"原文:   {text}")
    print(f"编码:   {ids}")
    print(f"解码:   {decoded}")
    print(f"编解码一致: {decoded == text}")

    # 聊天模板测试
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人。"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\n聊天模板:\n{prompt}")

    # 编码效率测试
    sample_texts = [
        "今天天气真好！",
        "The quick brown fox jumps over the lazy dog.",
        "Python是一种流行的编程语言。",
        "1+1=2",
    ]
    print("\n编码效率:")
    for t in sample_texts:
        token_ids = tokenizer.encode(t)
        print(f"  '{t}' -> {len(token_ids)} tokens: {token_ids}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MindLM Tokenizer")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        eval()


if __name__ == "__main__":
    main()
