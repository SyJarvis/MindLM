"""
MindLM 模型导出脚本
将 .pth 权重导出为 HuggingFace transformers 格式 (model.safetensors)
"""

import os
import sys
import json
import torch
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from modeling_mindlm import MindLM, MindLMConfig
from config import load_config


def export_model(config_name, checkpoint_path, output_dir, dtype="bfloat16"):
    """
    导出模型为 HuggingFace transformers 格式

    Args:
        config_name: 模型配置名 (如 mindlm_0.1b)
        checkpoint_path: .pth 权重文件路径
        output_dir: 输出目录
        dtype: 权重精度 (bfloat16/float16/float32)
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype)

    # 1. 加载配置
    model_config_dict = load_config(config_name)
    print(f"模型配置: {config_name}")

    # 2. 加载 tokenizer
    tokenizer_path = str(project_root / "mindlm_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model_config_dict['vocab_size'] = len(tokenizer)
    print(f"Tokenizer 词表大小: {len(tokenizer)}")

    # 3. 创建模型
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

    # 注册 auto class，使 transformers 能自动识别
    MindLMConfig.register_for_auto_class()
    MindLM.register_for_auto_class("AutoModelForCausalLM")

    model = MindLM(config).to(device)

    # 4. 加载权重
    print(f"加载权重: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 兼容完整 checkpoint 和纯 state_dict
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    # 兼容 DDP 的 module. 前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    print("权重加载成功")

    # 5. 转换精度
    model = model.to(torch_dtype)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 6. 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"模型已保存: {output_dir}/")

    # 补写 auto_map 到 config.json（save_pretrained 不会自动加）
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["auto_map"] = {
        "AutoConfig": "modeling_mindlm.MindLMConfig",
        "AutoModelForCausalLM": "modeling_mindlm.MindLM",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 复制模型代码文件（transformers trust_remote_code 需要这些文件）
    import shutil
    src_dir = Path(__file__).parent
    for fname in ["modeling_mindlm.py", "config.py"]:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, os.path.join(output_dir, fname))
    Path(output_dir, "__init__.py").touch(exist_ok=True)

    # 7. 保存 tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer 已保存: {output_dir}/")

    # 8. 验证：重新加载测试
    print("\n验证导出结果...")
    from transformers import AutoModelForCausalLM
    test_model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)
    test_params = sum(p.numel() for p in test_model.parameters())
    print(f"验证通过！重新加载参数量: {test_params / 1e6:.2f}M")

    # 9. 打印文件列表
    print(f"\n导出文件列表:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        if size > 1024 * 1024:
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {f}: {size / 1024:.1f} KB")

    print(f"\n使用方式:")
    print(f'  from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}", trust_remote_code=True)')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{output_dir}", trust_remote_code=True)')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MindLM 模型导出")
    parser.add_argument("--config", type=str, default="mindlm_0.1b",
                        help="模型配置名")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="权重文件路径 (.pth)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="权重精度")
    args = parser.parse_args()

    export_model(args.config, args.checkpoint, args.output_dir, args.dtype)
# ```

# ## 用法

# ```bash
# # 导出预训练模型
# python export_model.py \
#     --checkpoint out/mindlm_pretrain_768_linear_epoch0.pth \
#     --output_dir mindlm-0.1b \
#     --dtype bfloat16

# # 导出 SFT 模型
# python export_model.py \
#     --checkpoint out/mindlm_sft_768_linear_epoch0.pth \
#     --output_dir mindlm-0.1b-sft \
#     --dtype bfloat16
# ```

# ## 导出后的目录结构

# ```
# mindlm-0.1b/
# ├── config.json              # 模型配置 (transformers 格式)
# ├── model.safetensors        # 模型权重 (safetensors 格式)
# ├── tokenizer.json           # tokenizer
# ├── tokenizer_config.json    # tokenizer 配置
# ├── vocab.json               # 词表
# ├── merges.txt               # BPE 合并规则
# └── special_tokens_map.json  # 特殊 token
# ```

# ## 使用方式

# ```python
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("mindlm-0.1b", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("mindlm-0.1b", trust_remote_code=True)
# ```

# 关键点：`trust_remote_code=True` 是必须的，因为 MindLM 的架构是自定义的，transformers 需要从仓库中加载 `modeling_mindlm.py`。所以导出目录里需要放一份 `modeling_mindlm.py` 和 `config.py`。脚本里需要我加上自动复制这两个文件的逻辑吗？