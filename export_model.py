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
from training_utils import build_model_config, load_model_checkpoint


def export_model(config_name, checkpoint_path, output_dir, dtype="bfloat16", tokenizer_path=None):
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
    print(f"模型配置: {config_name}")

    # 2. 加载与模型词表匹配的 tokenizer
    if tokenizer_path is None:
        tokenizer_path = str(
            project_root / ("qwen3_tokenizer" if config_name == "mindlm_0.8b" else "mindlm_tokenizer")
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer 词表大小: {len(tokenizer)}")

    # 3. 创建模型
    config = build_model_config(config_name, tokenizer)

    # 注册 auto class，使 transformers 能自动识别
    MindLMConfig.register_for_auto_class()
    MindLM.register_for_auto_class("AutoModelForCausalLM")

    model = MindLM(config).to(device)

    # 4. 加载权重
    print(f"加载权重: {checkpoint_path}")
    load_model_checkpoint(model, checkpoint_path, device)
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
    for fname in ["modeling_mindlm.py"]:
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
    parser.add_argument("--config", choices=("mindlm_0.1b", "mindlm_0.1b_moe", "mindlm_0.8b"), default="mindlm_0.1b",
                        help="模型配置名")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="权重文件路径 (.pth)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer 目录；默认按模型配置选择")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="权重精度")
    args = parser.parse_args()

    export_model(args.config, args.checkpoint, args.output_dir, args.dtype, args.tokenizer_path)
