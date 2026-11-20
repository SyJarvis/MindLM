"""Analyze MindLM tokenizer vocab distribution."""
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("qwen3_tokenizer", trust_remote_code=True)
print("Type:", type(t).__name__)
print("Vocab size:", len(t))

all_tokens = list(t.vocab.keys())
print(f"Total vocab keys: {len(all_tokens)}")

# Check Chinese chars
chinese = [k for k in all_tokens if any("\u4e00" <= c <= "\u9fff" for c in k)]
print(f"Chinese-containing tokens: {len(chinese)}")

# English alpha tokens
english = [k for k in all_tokens if k and all(c.isascii() and c.isalpha() for c in k)]
print(f"Pure alpha tokens: {len(english)}")

# Byte-level tokens (single bytes)
byte_tokens = [k for k in all_tokens if k.startswith("<0x") or k.startswith("0x")]
print(f"Byte-level tokens: {len(byte_tokens)}")

# Special tokens
special = [k for k in all_tokens if k.startswith("<") and k.endswith(">")]
print(f"Special tokens: {len(special)}")

print("\nSample tokens:", all_tokens[:30])
print("\nSample Chinese:", chinese[:20])
print("\nSample English:", english[:20])

# Suggested reduced vocab
print("\n--- Suggestion ---")
print(f"Current vocab: {len(t)}")
print(f"Suggested vocab for 0.2B model: 8,000 - 16,000")
print(f"Embedding savings: {(len(t) - 8000) * 768 / 1e6:.1f}M to {(len(t) - 16000) * 768 / 1e6:.1f}M params freed up for Transformer layers")
