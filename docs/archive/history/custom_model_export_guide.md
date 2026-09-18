# 自定义模型导出 HuggingFace Transformers 格式指南

> 基于将 MindLM (PyTorch .pth) 导出为 transformers 5.x 格式 (model.safetensors) 的实践经验总结。

## 目录

- [概述](#概述)
- [踩过的坑](#踩过的坑)
  - [坑1: `_tied_weights_keys` 格式不兼容](#坑1-_tied_weights_keys-格式不兼容)
  - [坑2: 缺少 `post_init()` 调用](#坑2-缺少-post_init-调用)
  - [坑3: Config 缺少 `tie_word_embeddings`](#坑3-config-缺少-tie_word_embeddings)
  - [坑4: 缺少 `GenerationMixin` 继承](#坑4-缺少-generationmixin-继承)
- [问题排查流程](#问题排查流程)
- [完整代码](#完整代码)
  - [modeling_mindlm.py (关键部分)](#modeling_mindlmpy-关键部分)
  - [export_model.py](#export_modelpy)

---

## 概述

将一个自定义 PyTorch 模型导出为 HuggingFace transformers 格式，需要满足以下条件：

1. 模型类继承 `PreTrainedModel`（和 `GenerationMixin`）
2. 配置类继承 `PretrainedConfig`
3. 正确声明权重绑定关系
4. 正确调用 `post_init()` 初始化流程
5. 使用 `save_pretrained()` 保存，而非手动写 safetensors

## 踩过的坑

### 坑1: `_tied_weights_keys` 格式不兼容

**现象：**
```
AttributeError: 'list' object has no attribute 'keys'
```

**原因：**

transformers 4.x 中 `_tied_weights_keys` 是 **list** 格式：
```python
_tied_weights_keys = ["lm_head.weight"]  # transformers 4.x
```

transformers 5.x 中 `_tied_weights_keys` 改为 **dict** 格式，表达 `{target: source}` 的映射关系：
```python
_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # transformers 5.x
```

transformers 5.x 内部 `_get_tied_weight_keys` 函数会对每个 `_tied_weights_keys` 调用 `.keys()` 方法：
```python
# transformers/modeling_utils.py
def _get_tied_weight_keys(module):
    tied_weight_keys = []
    for name, submodule in module.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", {}) or {}
        tied_weight_keys.extend([f"{name}.{k}" if name else k for k in tied.keys()])
        #                                                                      ^^^^^ 报错：list 没有 keys()
    return tied_weight_keys
```

**解决办法：**

使用 dict 格式，key 是被绑定的权重（output），value 是源权重（embedding）：
```python
class MindLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"output.weight": "tok_embeddings.weight"}
```

---

### 坑2: 缺少 `post_init()` 调用

**现象：**
```
AttributeError: 'MindLM' object has no attribute 'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?
```

**原因：**

`PreTrainedModel.__init__()` **不会**自动调用 `post_init()`。`post_init()` 负责设置关键实例属性：
- `all_tied_weights_keys`：从 `_tied_weights_keys` 展开得到完整的权重绑定映射
- 并行计划 (`_tp_plan`, `_ep_plan`, `_pp_plan`)
- 调用 `init_weights()` → `tie_weights()`

如果不在子类 `__init__` 末尾调用 `post_init()`，`all_tied_weights_keys` 就不会被设置，后续 `from_pretrained` 加载时会崩溃。

```python
# PreTrainedModel.__init__ 的简化流程
class PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ... 基本设置
        # 注意：这里没有调用 self.post_init()!
```

**解决办法：**

在模型类 `__init__` 的**最后一步**调用 `self.post_init()`：
```python
class MindLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        # ... 定义所有层
        self.tok_embeddings = nn.Embedding(...)
        self.output = nn.Linear(...)
        self.tok_embeddings.weight = self.output.weight
        # ... 权重初始化
        self.apply(self._init_weights)
        # 最后一步！
        self.post_init()
```

> 注意：`post_init()` 内部会调用 `init_weights()` → `tie_weights()`，但只会执行一次（有 `_is_hf_initialized` 标志位保护），不会重复初始化。

---

### 坑3: Config 缺少 `tie_word_embeddings`

**现象：**

导出成功，加载无报错，但 `output.weight` 被**随机初始化**而非绑定到 `tok_embeddings.weight`：
```
MindLM LOAD REPORT from: mindlm_1b
Key           | Status  |
--------------+---------+-
output.weight | MISSING |
```

验证发现权重完全不共享：
```python
model.tok_embeddings.weight.data_ptr() == model.output.weight.data_ptr()  # False!
torch.equal(model.tok_embeddings.weight.data, model.output.weight.data)   # False!
```

**原因：**

这是最隐蔽的坑。transformers 5.x 的 `get_expanded_tied_weights_keys` 会检查 `config.tie_word_embeddings`：

```python
# transformers/modeling_utils.py
def get_expanded_tied_weights_keys(self, all_submodels=False) -> dict:
    tied_mapping = self._tied_weights_keys
    # 关键检查！默认是 False
    tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
    if not tie_word_embeddings:
        return {}  # 直接返回空 dict，所有绑定关系被忽略！
```

`PretrainedConfig` 在 transformers 5.x 中**不再默认包含** `tie_word_embeddings` 属性。如果自定义 Config 没有设置这个属性，`getattr` 返回默认值 `False`，导致所有 `_tied_weights_keys` 的绑定映射被忽略。

这导致：
1. `save_pretrained` 保存了**两份**权重（tok_embeddings + output），而不是一份
2. `from_pretrained` 时 `output.weight` 被随机初始化，因为绑定关系被跳过
3. 模型虽然能加载，但 `output` 层是**随机权重**，推理结果完全错误

**解决办法：**

在 Config 的 `__init__` 中显式设置：
```python
class MindLMConfig(PretrainedConfig):
    def __init__(self, ..., **kwargs):
        super().__init__(**kwargs)
        # ... 其他参数
        self.tie_word_embeddings = True  # 必须！
```

---

### 坑4: 缺少 `GenerationMixin` 继承

**现象：**

模型可以 `from_pretrained` 加载，但没有标准的 `generate()` 接口，也不兼容 transformers 的 generation 配置系统。

**原因：**

标准 transformers 的 CausalLM 模型同时继承 `PreTrainedModel` 和 `GenerationMixin`：
```python
# transformers 标准写法
class LlamaForCausalLM(PreTrainedModel, GenerationMixin):
    ...
```

**解决办法：**

```python
from transformers import PreTrainedModel, GenerationMixin

class MindLM(PreTrainedModel, GenerationMixin):
    config_class = MindLMConfig
    ...
```

---

## 问题排查流程

当自定义模型导出 transformers 格式出现问题时，按以下顺序排查：

```
1. 导出时报错
   ├── AttributeError: 'list' has no attribute 'keys'
   │   └── _tied_weights_keys 格式错误 → 改为 dict 格式
   │
   └── 其他保存错误
       └── 检查是否用了 save_pretrained()

2. 加载时报错
   ├── AttributeError: no attribute 'all_tied_weights_keys'
   │   └── 缺少 post_init() 调用 → 在 __init__ 末尾加上
   │
   └── 其他加载错误
       └── 检查 config.json 中的 auto_map 和模型文件是否复制到输出目录

3. 加载成功但结果异常
   ├── output.weight | MISSING
   │   └── 检查 config.tie_word_embeddings 是否为 True
   │
   └── 参数量不一致
       └── 检查权重绑定是否正确生效（data_ptr 是否相同）
```

**验证权重绑定是否正确：**
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("your_model_dir", trust_remote_code=True)

# 1. 检查是否共享内存（真正绑定）
print(model.tok_embeddings.weight.data_ptr() == model.output.weight.data_ptr())  # 应为 True

# 2. 检查值是否相同
print(torch.equal(model.tok_embeddings.weight.data, model.output.weight.data))   # 应为 True

# 3. 检查参数量是否一致
save_params = sum(p.numel() for p in model.parameters())
load_params = sum(p.numel() for p in model.parameters())
print(save_params == load_params)  # 应为 True
```

---

## 完整代码

### modeling_mindlm.py (关键部分)

展示自定义模型接入 transformers 所需的关键代码结构。省略了注意力、FFN 等内部实现，只保留与 transformers 兼容性相关的部分。

```python
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class MindLMConfig(PretrainedConfig):
    """自定义配置类"""
    model_type = "mindlm"

    def __init__(
        self,
        dim: int = 768,
        n_layers: int = 8,
        n_heads: int = 8,
        vocab_size: int = 6400,
        # ... 其他模型参数
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        # ... 其他参数赋值

        # 关键！必须设置，否则权重绑定不生效
        self.tie_word_embeddings = True


class MindLM(PreTrainedModel, GenerationMixin):
    """自定义模型类"""
    config_class = MindLMConfig

    # 关键！transformers 5.x 要求 dict 格式 {"target": "source"}
    _tied_weights_keys = {"output.weight": "tok_embeddings.weight"}

    def __init__(self, config: MindLMConfig = None):
        if config is None:
            config = MindLMConfig()
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        # 定义模型层
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([...])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 权重绑定
        self.tok_embeddings.weight = self.output.weight

        # 自定义权重初始化
        self.apply(self._init_weights)

        # 关键！必须在 __init__ 最后调用
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens=None, targets=None, **kwargs):
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        # ... 前向传播逻辑
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
```

### export_model.py

```python
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

    # 验证权重绑定
    tied = test_model.tok_embeddings.weight.data_ptr() == test_model.output.weight.data_ptr()
    print(f"权重绑定验证: {'通过' if tied else '失败'}")

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
```

## 总结：接入 transformers 5.x 的 Checklist

| # | 要点 | 代码 |
|---|------|------|
| 1 | Config 设置 `model_type` | `model_type = "mindlm"` |
| 2 | Config 设置 `tie_word_embeddings = True` | `self.tie_word_embeddings = True` |
| 3 | Model 继承 `PreTrainedModel` + `GenerationMixin` | `class MindLM(PreTrainedModel, GenerationMixin)` |
| 4 | Model 声明 `config_class` | `config_class = MindLMConfig` |
| 5 | Model 声明 `_tied_weights_keys` (dict格式) | `_tied_weights_keys = {"output.weight": "tok_embeddings.weight"}` |
| 6 | Model 在 `__init__` 末尾调用 `self.post_init()` | `self.post_init()` |
| 7 | 使用 `model.save_pretrained()` 保存 | `model.save_pretrained(output_dir, safe_serialization=True)` |
| 8 | 导出目录包含模型代码文件 | 复制 `modeling_mindlm.py`、`config.py` |
| 9 | `config.json` 中写入 `auto_map` | 指向本地模型类 |
| 10 | 加载时使用 `trust_remote_code=True` | `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` |
