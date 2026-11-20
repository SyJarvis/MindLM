"""Compare tokenizer vocab sizes between MindLM and MiniMind-3."""
from transformers import AutoTokenizer

t1 = AutoTokenizer.from_pretrained("qwen3_tokenizer", trust_remote_code=True)
t2 = AutoTokenizer.from_pretrained(
    "/root/.cache/modelscope/models/gongjy--minimind-3/snapshots/master",
    trust_remote_code=True,
)

print(f"MindLM vocab:    {len(t1):>8,}")
print(f"MiniMind-3 vocab: {len(t2):>8,}")
print()

texts = [
    "你好，你是谁？",
    "请介绍一下人工智能。",
    "如何学习编程？",
    "What is the weather like in Beijing today?",
]
for text in texts:
    ids1 = t1.encode(text)
    ids2 = t2.encode(text)
    print(f"Text: {text}")
    print(f"  MindLM:     {len(ids1):>3} tokens")
    print(f"  MiniMind-3: {len(ids2):>3} tokens")
    print()

hidden = 768
vocab1 = len(t1)
vocab2 = len(t2)
print(f"--- Parameter impact (hidden={hidden}, tied embeddings) ---")
print(f"MindLM embedding:     {vocab1} x {hidden} = {vocab1*hidden/1e6:.1f}M")
print(f"MiniMind-3 embedding: {vocab2} x {hidden} = {vocab2*hidden/1e6:.1f}M")
print(f"Ratio: {vocab1/vocab2:.1f}x")
print(f"Embedding params alone: MindLM {vocab1*hidden/1e6:.1f}M vs MiniMind {vocab2*hidden/1e6:.1f}M")
