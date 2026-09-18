# MindLM 集成到 HuggingFace Transformers 指南

> 基于 transformers 官方 [CONTRIBUTING.md](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) 和 [Modular Transformers](https://huggingface.co/docs/transformers/modular_transformers) 文档整理

---

## 当前状态

- MindLM 模型代码：`MindLM/modeling_mindlm.py`（已继承 `PreTrainedModel`）
- MiniMind 模型代码：`MindLM/minimind/model/model_minimind.py`（已继承 `PreTrainedModel`）
- 本地 transformers fork：`/Users/whoami/research/minimind/transformers/`
- **两者均未集成进 transformers 库的目录结构**

---

## 路线 1：本地 Fork 集成（推荐先做）

在本地 transformers fork 中直接添加 MindLM，使其支持 `AutoModelForCausalLM.from_pretrained("mindlm-1b")` 加载。

### 1.1 需要创建的文件

```
src/transformers/models/mindlm/
├── __init__.py                    # 懒加载入口
├── configuration_mindlm.py        # MindLMConfig
└── modeling_mindlm.py             # 模型实现
```

### 1.2 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `src/transformers/models/__init__.py` | 添加 `from .mindlm import *` |
| `src/transformers/models/auto/configuration_auto.py` | 添加 `("mindlm", "MindLMConfig")` |
| `src/transformers/models/auto/modeling_auto.py` | 添加 `("mindlm", "MindLMForCausalLM")` |

### 1.3 各文件实现要点

#### `configuration_mindlm.py`

```python
from ...configuration_utils import PretrainedConfig

class MindLMConfig(PretrainedConfig):
    model_type = "mindlm"

    def __init__(
        self,
        dim=768,
        n_layers=16,
        n_heads=12,
        n_kv_heads=3,
        vocab_size=6400,
        max_seq_len=1024,
        dropout=0.0,
        norm_eps=1e-6,
        hidden_dim=None,
        multiple_of=256,
        use_moe=False,
        use_linear_attn=True,
        conv_kernel_size=4,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.n_layers = n_layers
        # ... 其余参数
```

#### `modeling_mindlm.py`

基于现有 `MindLM/modeling_mindlm.py` 改造：

- [ ] `MindLMConfig` → 从 `configuration_mindlm.py` 导入
- [ ] `MindLMPreTrainedModel(PreTrainedModel)` → 基类，含 `_init_weights`
- [ ] `MindLMModel(MindLMPreTrainedModel)` → 核心模型（无 LM Head）
- [ ] `MindLMForCausalLM(MindLMPreTrainedModel)` → 带 LM Head，支持 `GenerationMixin`
- [ ] 所有内部类名加 `MindLM` 前缀：`MindLMRMSNorm`、`MindLMAttention`、`MindLMGatedDeltaNet`、`MindLMFeedForward` 等
- [ ] `generate()` 方法改用 `GenerationMixin` 标准接口，而非自定义实现
- [ ] `forward()` 返回 `CausalLMOutputWithPast`

#### `__init__.py`

```python
from typing import TYPE_CHECKING
from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from .configuration_mindlm import *
    from .modeling_mindlm import *
else:
    import sys
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
```

### 1.4 AutoClass 注册

#### `src/transformers/models/auto/configuration_auto.py`

在 `CONFIG_MAPPING_NAMES` 中添加：

```python
("mindlm", "MindLMConfig"),
```

#### `src/transformers/models/auto/modeling_auto.py`

在 `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 中添加：

```python
("mindlm", "MindLMForCausalLM"),
```

在 `MODEL_MAPPING_NAMES` 中添加：

```python
("mindlm", "MindLMModel"),
```

#### `src/transformers/models/__init__.py`

在 import 列表中添加：

```python
from .mindlm import *
```

### 1.5 验证步骤

```bash
# 安装本地 transformers
cd /Users/whoami/research/minimind/transformers
pip install -e ".[quality]"

# 测试能否加载
python -c "
from transformers import AutoConfig, AutoModelForCausalLM
config = AutoConfig.for_model('mindlm')
print(config)
model = AutoModelForCausalLM.from_config(config)
print(f'参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
"

# 测试从预训练权重加载
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/mindlm-1b-weights')
print('加载成功')
"
```

### 1.6 上传到 HuggingFace Hub

路线 1 完成后，可将模型上传到 Hub，用户通过 `trust_remote_code=True` 加载：

```python
model = AutoModelForCausalLM.from_pretrained("your-name/MindLM-1B", trust_remote_code=True)
```

---

## 路线 2：正式贡献到 Transformers 库

将 MindLM 合并进 HuggingFace 官方 transformers 库，用户无需 `trust_remote_code`。

### 2.1 前置条件

- [ ] 路线 1 已完成且稳定运行
- [ ] 有明确的差异化价值（不能只是 Qwen3.5 的简单改版）
- [ ] 有预训练权重且效果可验证
- [ ] 熟悉每一行代码，能向 reviewer 解释所有改动

### 2.2 流程

#### Step 1：开 GitHub Issue

在 [huggingface/transformers](https://github.com/huggingface/transformers/issues) 新建 Issue，标签选 `New model`：

**Issue 内容模板：**

```markdown
## New Model: MindLM

### Description
MindLM 是一个轻量级混合注意力语言模型，采用 GatedDeltaNet 线性注意力 + 标准 GQA 注意力的混合架构。

### Architecture
- 75% GatedDeltaNet 线性注意力 + 25% 标准 GQA Attention
- SwiGLU FFN + 可选 MoE
- RoPE 位置编码（标准层）+ Conv1d（线性注意力层）

### Differences from existing models
- 与 Qwen3.5 的区别：[说明差异化]
- 更小的参数规模用于教学和实验

### Implementation
- 代码仓库：[链接]
- 预训练权重：[链接]

### Contribution
我计划自己贡献这个模型的实现。
```

#### Step 2：等维护团队回复

维护团队会评估：
- 是否接受（与现有模型是否过于重复）
- 技术建议（应该继承哪些现有模型）
- 分配 reviewer

**可能的风险**：MindLM 架构与 Qwen3.5 高度相似，可能被要求说明差异化价值。建议准备：
- 独立的训练结果/基准测试
- 与 Qwen3.5 在同等参数量下的对比实验
- 明确的目标场景（教学、轻量级部署等）

#### Step 3：实现 Modular 文件

当前 transformers **要求所有新模型使用 modular 方式**：

```bash
# 生成骨架（可选）
transformers add-new-model-like --model qwen3_5 --new_model_name mindlm
```

创建 `src/transformers/models/mindlm/modular_mindlm.py`：

```python
# 因为 MindLM 最接近 Qwen3（标准注意力）+ Qwen3Next（GatedDeltaNet）
# 可以继承复用，大幅减少代码量

from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm, Qwen3MLP
from ..qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

class MindLMConfig(PreTrainedConfig):
    model_type = "mindlm"
    # ...

class MindLMAttention(Qwen3Attention):
    # 复用 Qwen3 的 GQA + QK Norm
    # 覆盖不同的参数（GQA 比例等）
    pass

class MindLMGatedDeltaNet(Qwen3NextGatedDeltaNet):
    # 复用 Qwen3Next 的线性注意力
    # 覆盖简化实现
    pass

class MindLMDecoderLayer(nn.Module):
    # 根据 layer_types 分派
    pass

class MindLMForCausalLM(PreTrainedModel):
    # 主模型
    pass
```

生成独立文件：

```bash
python utils/modular_model_converter.py mindlm
```

#### Step 4：创建权重转换脚本

```
src/transformers/models/mindlm/convert_mindlm_to_hf.py
```

功能：
- 加载原始 checkpoint
- 映射权重名称到 HF 标准命名
- 验证形状匹配
- 保存为 `save_pretrained()` 格式

```python
def convert_checkpoint(original_path, output_path):
    original_state_dict = torch.load(original_path)
    new_state_dict = {}
    for key, value in original_state_dict.items():
        new_key = key.replace("tok_embeddings", "embed_tokens")
        new_key = new_key.replace("output", "lm_head")
        new_key = new_key.replace("feed_forward", "mlp")
        # ... 其余映射
        new_state_dict[new_key] = value
    model = MindLMForCausalLM(MindLMConfig())
    model.load_state_dict(new_state_dict)
    model.save_pretrained(output_path)
```

#### Step 5：编写测试

```
tests/models/mindlm/test_modeling_mindlm.py
```

```python
import unittest
from transformers import MindLMConfig, MindLMForCausalLM, MindLMModel

class MindLMModelTester:
    """小模型配置用于快速测试"""
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 16
        self.dim = 128
        self.n_layers = 2
        self.n_heads = 4
        self.n_kv_heads = 2
        self.vocab_size = 64
        self.use_moe = False
        self.use_linear_attn = True

class MindLMModelTest(unittest.TestCase):
    all_model_classes = (MindLMModel, MindLMForCausalLM)

    def test_config(self): ...
    def test_forward_pass(self): ...
    def test_generation(self): ...
    def test_linear_attention_layers(self): ...
    def test_standard_attention_layers(self): ...

class MindLMIntegrationTest(unittest.TestCase):
    """用真实权重验证输出一致性"""
    @slow
    def test_logits_match(self):
        # 原始模型输出 vs HF 模型输出，atol=1e-3
        ...
```

运行测试：

```bash
pytest tests/models/mindlm/test_modeling_mindlm.py
RUN_SLOW=1 pytest -sv tests/models/mindlm/test_modeling_mindlm.py
```

#### Step 6：编写文档

```
docs/source/en/model_doc/mindlm.md
```

内容包括：
- 模型描述（混合注意力架构）
- 架构图/参数表
- 使用示例代码
- API 参考（Config、Model、ForCausalLM）

在 `docs/source/en/_toctree.yml` 中注册。

#### Step 7：代码质量检查

```bash
make style          # 自动格式化 + lint
make check-repo     # 仓库完整性检查
```

#### Step 8：提交 PR

```bash
git checkout -b add-mindlm
git add src/transformers/models/mindlm/
git add tests/models/mindlm/
git add docs/source/en/model_doc/mindlm.md
# 以及所有修改的注册文件
git commit -m "Add MindLM model"
git push origin add-mindlm
```

PR 标题：`[WIP] Add MindLM`，描述中关联 Step 1 的 Issue。

PR Checklist：

- [ ] PR 标题概括了贡献内容
- [ ] 关联了 Issue 编号
- [ ] 现有测试通过
- [ ] 新功能有新测试
- [ ] 公共方法有 docstring
- [ ] `make style` 通过
- [ ] Modular 文件已验证

---

## 两条路线对比

| | 路线 1：本地 Fork | 路线 2：正式贡献 |
|---|---|---|
| **目标** | 自己的 transformers fork 里能用 | 合并进官方 transformers 库 |
| **是否需要 Issue** | 不需要 | 需要 |
| **是否需要 PR** | 不需要 | 需要 |
| **代码规范** | 自由 | 严格（modular、测试、文档） |
| **加载方式** | `trust_remote_code=True` 或本地安装 | 直接 `from_pretrained()` |
| **实现方式** | 直接放 `modeling_mindlm.py` | 必须用 modular 继承模式 |
| **工作量** | ~3-5 天 | ~2-4 周（含审核迭代） |
| **适用场景** | 研究、教学、内部使用 | 社区广泛使用 |

---

## 建议执行顺序

```
阶段 1（路线 1）                    阶段 2（可选，路线 2）
─────────────────                  ──────────────────────
创建目录结构                        开 GitHub Issue
编写 configuration_mindlm.py        等维护团队确认
改造 modeling_mindlm.py             重构为 modular 继承模式
注册 AutoClass                      编写权重转换脚本
本地测试验证                         补充完整测试套件
上传权重到 Hub                       编写文档
                                    make style + check-repo
                                    提交 PR
```
