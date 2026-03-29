"""
MindLM 推理测试脚本
加载预训练权重，进行文本生成测试
"""

import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent
mindlm_root = project_root / "MindLM"
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from modeling_mindlm import MindLM, MindLMConfig
from config import load_config

# ==================== 配置 ====================
# 使用你训练时的配置（对应 mindlm_1b.json）
CONFIG_NAME = "mindlm_1b"
CHECKPOINT_PATH = str(Path(__file__).parent.parent / "out" / "mindlm_pretrain_768_linear_epoch0.pth")
TOKENIZER_PATH = str(Path(__file__).parent.parent / "mindlm_tokenizer")

# 生成参数
PROMPTS = [
    "人工智能",
    "中国的首都是",
    "世界上最高的山是",
    "机器学习是",
    "深度学习在自然语言处理中的应用包括",
]
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_K = 8
EOS_TOKEN_ID = 0  # EOS token id

# ==================== 加载模型 ====================
print("=" * 60)
print("MindLM 推理测试")
print("=" * 60)

# 加载配置
model_config_dict = load_config(CONFIG_NAME)
print(f"模型配置: {CONFIG_NAME} (dim={model_config_dict['dim']}, layers={model_config_dict['n_layers']})")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
print(f"Tokenizer 词表大小: {len(tokenizer)}")

# 更新 vocab_size
model_config_dict['vocab_size'] = len(tokenizer)

# 创建模型配置
config = MindLMConfig(
    dim=model_config_dict['dim'],
    n_layers=model_config_dict['n_layers'],
    n_heads=model_config_dict['n_heads'],
    n_kv_heads=model_config_dict['n_kv_heads'],
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

# 创建并加载模型
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

model = MindLM(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params / 1e6:.2f}M")

# 加载权重
print(f"加载权重: {CHECKPOINT_PATH}")
state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

# 检查是纯 state_dict 还是包含其他信息的 checkpoint
if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
    model.load_state_dict(state_dict['model_state_dict'])
else:
    model.load_state_dict(state_dict)

model.eval()
print("模型加载成功！")

# ==================== 生成测试 ====================
print("=" * 60)
print("开始生成测试")
print("=" * 60)

for prompt in PROMPTS:
    print(f"\n{'─' * 40}")
    print(f"输入: {prompt}")
    print(f"{'─' * 40}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"编码: {input_ids.shape}")

    generated_text = prompt
    with torch.no_grad():
        for generated in model.generate(
            input_ids,
            eos=EOS_TOKEN_ID,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        ):
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"输出: {generated_text}")

print(f"\n{'=' * 60}")
print("测试完成！")
