# RoPE 上下文扩展策略设计

## 1. 背景：RoPE 在 MindLM 中的作用

MindLM-1B 采用混合注意力架构，16 层中只有 **4 层标准 Attention**（第 3,7,11,15 层）使用 RoPE。其余 12 层 Linear Attention（GatedDeltaNet）通过 conv1d 感知局部位置，**不依赖 RoPE**。

```
Layer 0:  linear_attention  ─── 无 RoPE（conv1d 局部位置）
Layer 1:  linear_attention  ─── 无 RoPE
Layer 2:  linear_attention  ─── 无 RoPE
Layer 3:  attention         ─── 使用 RoPE ← 扩展重点
Layer 4:  linear_attention  ─── 无 RoPE
...
Layer 15: attention         ─── 使用 RoPE ← 扩展重点
```

**关键结论**：RoPE 扩展只影响 4 层，对模型整体影响较小。

---

## 2. 问题：为什么直接扩大 max_seq_len 效果差？

当前 `precompute_pos_cis` 使用固定 `theta=10000.0` 生成位置编码：

```
freq_i = 1 / (10000^(2i/d))    d = head_dim = 64
```

| 频率分量 i | 波长（token数） | 作用 |
|-----------|---------------|------|
| 0 | ~6280 | 全局结构 |
| 16 | ~39 | 段落级关系 |
| 32 | ~1 | 相邻 token 关系 |

模型在 Stage 1（2K）训练时，只"见过"位置 0-2047 对应的旋转角度。直接扩展到 8K 时：
- 位置 2048-8191 对应的旋转角度是模型**从未见过的**
- 高频分量（局部结构）外推效果尚可
- **低频分量（全局结构）外推会导致注意力分布崩溃**

---

## 3. 扩展策略对比

| 策略 | 原理 | 优点 | 缺点 | 推荐场景 |
|------|------|------|------|---------|
| **NTK-aware Scaling** | 调整基础频率 theta | 简单、有效、改动最小 | 低频分辨率略降 | Stage 2 续训（推荐） |
| **Position Interpolation** | 线性插值位置索引 | 实现最简单 | 损失分辨率、需要更多训练数据 | 快速验证 |
| **YaRN** | 分频段处理 + 温度缩放 | 效果最好、外推能力强 | 实现较复杂 | Stage 3（32K） |

**推荐路线**：
- Stage 2（2K→8K）：NTK-aware Scaling
- Stage 3（8K→32K）：NTK-aware 或 YaRN

---

## 4. 原始代码

### 4.1 位置编码预计算（modeling_mindlm.py:30-36）

```python
def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算旋转位置编码"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis
```

### 4.2 模型初始化调用（modeling_mindlm.py:590-594）

```python
if has_normal:
    pos_cis = precompute_pos_cis(
        config.dim // config.n_heads,
        config.max_seq_len
    )
    self.register_buffer("pos_cis", pos_cis, persistent=False)
```

### 4.3 MindLMConfig 定义（modeling_mindlm.py:447-516）

当前配置中没有 RoPE 相关参数：

```python
class MindLMConfig(PretrainedConfig):
    model_type = "mindlm"
    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = None,
        vocab_size: int = 6400,
        max_seq_len: int = 32768,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        hidden_dim: int = None,
        multiple_of: int = 256,
        use_moe: bool = True,
        n_routed_experts: int = 8,
        num_experts_per_tok: int = 2,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        use_linear_attn: bool = True,
        layer_types: List[str] = None,
        conv_kernel_size: int = 4,
        **kwargs
    ):
        # ... 省略赋值 ...
        # 注意：没有 rope_theta 和 rope_scaling_factor 参数
```

---

## 5. 策略一：NTK-aware Scaling（推荐）

### 5.1 原理

NTK-aware 的核心思想：**不缩放位置，而是调整频率基数 theta**，让高频分量（局部关系）保持不变，低频分量（全局关系）被拉伸。

```
原始:   theta = 10000
缩放后: theta_new = theta × scale^(d/(d-2))

其中:
  scale = new_max_seq_len / original_max_seq_len
  d = head_dim = 64 (MindLM-1B: 768/12)
```

**数学推导**：

原始频率：`freq_i = 1 / (theta^(2i/d))`

缩放后频率：`freq_i_new = 1 / (theta_new^(2i/d))`

代入 theta_new：
```
freq_i_new = 1 / ((theta × s^(d/(d-2)))^(2i/d))
           = 1 / (theta^(2i/d) × s^(2i/(d-2)))
```

- 当 i 较小（高频分量）：`s^(2i/(d-2)) ≈ 1`，频率几乎不变 → **局部关系保持**
- 当 i 较大（低频分量）：`s^(2i/(d-2)) ≈ s`，频率被缩放 → **全局关系被拉伸到新范围**

### 5.2 MindLM-1B 的具体数值

| 阶段 | scale | theta_new | 说明 |
|------|-------|-----------|------|
| Stage 1 (2K) | 1 | 10,000 | 原始 |
| Stage 2 (8K) | 4 | ~40,900 | `10000 × 4^(64/62)` |
| Stage 3 (32K) | 16 | ~167,000 | `10000 × 16^(64/62)` |

### 5.3 实现代码

#### 5.3.1 修改后的 `precompute_pos_cis`

```python
def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0,
                       rope_scaling_factor: float = 1.0,
                       original_max_seq_len: int = None):
    """
    预计算旋转位置编码，支持 NTK-aware 缩放

    Args:
        dim: 每个注意力头的维度（head_dim = model_dim / n_heads）
        end: 最大序列长度
        theta: RoPE 基础频率，默认 10000.0
        rope_scaling_factor: NTK-aware 缩放因子
            - 1.0: 不缩放（原始行为）
            - 4.0: 适合 2K→8K 扩展
            - 16.0: 适合 2K→32K 扩展
        original_max_seq_len: 原始训练时的最大序列长度
            - 用于 YaRN 混合策略（可选）
    """
    if rope_scaling_factor != 1.0:
        # NTK-aware scaling: 调整基础频率
        # theta_new = theta × scale^(dim/(dim-2))
        theta = theta * (rope_scaling_factor ** (dim / (dim - 2)))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis
```

#### 5.3.2 MindLMConfig 新增参数

```python
class MindLMConfig(PretrainedConfig):
    model_type = "mindlm"
    def __init__(
        self,
        # ... 原有参数保持不变 ...
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = None,
        vocab_size: int = 6400,
        max_seq_len: int = 32768,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        hidden_dim: int = None,
        multiple_of: int = 256,
        use_moe: bool = True,
        n_routed_experts: int = 8,
        num_experts_per_tok: int = 2,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        use_linear_attn: bool = True,
        layer_types: List[str] = None,
        conv_kernel_size: int = 4,
        # ========== 新增 RoPE 扩展参数 ==========
        rope_theta: float = 10000.0,              # RoPE 基础频率
        rope_scaling_factor: float = 1.0,          # NTK-aware 缩放因子（1.0=不缩放）
        original_max_seq_len: int = None,          # 原始预训练的序列长度（用于 YaRN）
        # ========================================
        **kwargs
    ):
        super().__init__(**kwargs)
        # ... 原有赋值保持不变 ...
        self.dim = dim
        self.n_layers = n_layers
        # ... 省略中间赋值 ...
        self.conv_kernel_size = conv_kernel_size

        # 新增 RoPE 参数
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.original_max_seq_len = original_max_seq_len
```

#### 5.3.3 MindLM.__init__ 调用修改

```python
# ---- 原始代码 ----
if has_normal:
    pos_cis = precompute_pos_cis(
        config.dim // config.n_heads,
        config.max_seq_len
    )
    self.register_buffer("pos_cis", pos_cis, persistent=False)

# ---- 修改后 ----
if has_normal:
    pos_cis = precompute_pos_cis(
        config.dim // config.n_heads,
        config.max_seq_len,
        theta=config.rope_theta,
        rope_scaling_factor=config.rope_scaling_factor,
        original_max_seq_len=config.original_max_seq_len,
    )
    self.register_buffer("pos_cis", pos_cis, persistent=False)
```

### 5.4 JSON 配置文件示例

#### Stage 1（2K 预训练）

```json
{
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "n_kv_heads": 3,
    "vocab_size": 6400,
    "max_seq_len": 2048,
    "rope_theta": 10000.0,
    "rope_scaling_factor": 1.0
}
```

#### Stage 2（8K 扩展）

```json
{
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "n_kv_heads": 3,
    "vocab_size": 6400,
    "max_seq_len": 8192,
    "rope_theta": 10000.0,
    "rope_scaling_factor": 4.0,
    "original_max_seq_len": 2048
}
```

#### Stage 3（32K 扩展）

```json
{
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "n_kv_heads": 3,
    "vocab_size": 6400,
    "max_seq_len": 32768,
    "rope_theta": 10000.0,
    "rope_scaling_factor": 16.0,
    "original_max_seq_len": 2048
}
```

---

## 6. 策略二：YaRN（高级，适合 32K+）

### 6.1 原理

YaRN 将频率分量分为三个区域，分别处理：

```
频率分量 i 的波长: λ_i = 2π / freq_i

低频区 (λ > L):    位置插值，保持全局结构
中频区 (L > λ > l): NTK-aware 缩放，平滑过渡
高频区 (λ < l):    不变，保持局部精度

其中:
  L = 2π × original_max_seq_len  (低频阈值)
  l = 2π × scale_factor^(d/(d-2)) (高频阈值, 通常设为固定值如 2π×32)
```

此外，YaRN 还对注意力分数施加温度缩放：
```
attention_temperature = 0.1 × ln(scale) + 1
```

### 6.2 实现代码

```python
def precompute_pos_cis_yarn(dim: int, end: int, theta: float = 10000.0,
                            scale_factor: float = 1.0,
                            original_max_seq_len: int = 2048,
                            beta_fast: float = 32.0,
                            beta_slow: float = 1.0):
    """
    YaRN 风格的 RoPE 位置编码

    将频率分量分为三个区域：
    - 高频（波长 < 阈值）：保持原始频率
    - 低频（波长 > 阈值）：线性插值
    - 中频：平滑过渡

    Args:
        dim: head_dim
        end: 最大序列长度
        theta: 基础频率
        scale_factor: 缩放因子（new_len / original_len）
        original_max_seq_len: 原始训练序列长度
        beta_fast: 高频区域阈值系数
        beta_slow: 低频区域阈值系数
    """
    if scale_factor == 1.0:
        return precompute_pos_cis(dim, end, theta)

    # 计算基础频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 波长: λ_i = 2π / freq_i
    wavelengths = 2 * math.pi / freqs

    # 计算阈值
    # NTK-aware 缩放后的新基础频率
    base = theta * (scale_factor ** (dim / (dim - 2)))
    ntk_freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 高频阈值：波长 < 2π × beta_fast 的分量保持不变
    threshold_fast = 2 * math.pi * beta_fast
    # 低频阈值：波长 > 2π × beta_slow × original_len 的分量做插值
    threshold_slow = 2 * math.pi * beta_slow * original_max_seq_len

    # 混合策略：对每个频率分量选择处理方式
    new_freqs = torch.empty_like(freqs)
    for i in range(len(freqs)):
        wl = wavelengths[i].item()
        if wl < threshold_fast:
            # 高频区：保持原始频率（局部精度）
            new_freqs[i] = freqs[i]
        elif wl > threshold_slow:
            # 低频区：位置插值（全局结构）
            new_freqs[i] = freqs[i] / scale_factor
        else:
            # 中频区：平滑混合
            # 在原始频率和 NTK 缩放频率之间线性插值
            ratio = (wl - threshold_fast) / (threshold_slow - threshold_fast)
            ratio = max(0.0, min(1.0, ratio))
            # 从 NTK 频率（ratio=0）平滑过渡到插值频率（ratio=1）
            interpolated_freq = freqs[i] / scale_factor
            ntk_freq = ntk_freqs[i]
            new_freqs[i] = (1 - ratio) * ntk_freq + ratio * interpolated_freq

    t = torch.arange(end, device=new_freqs.device)
    freqs_matrix = torch.outer(t, new_freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs_matrix), freqs_matrix)
    return pos_cis
```

### 6.3 Attention 温度缩放

YaRN 还需要对注意力分数施加温度调节，修改 `Attention.forward`：

```python
# ---- 原始代码 ----
scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

# ---- YaRN 温度缩放 ----
# temperature = 0.1 × ln(scale_factor) + 1
yarn_temperature = 0.1 * math.log(self.rope_scaling_factor) + 1
scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
scores = scores / yarn_temperature
```

---

## 7. 策略三：Position Interpolation（简单备选）

### 7.1 原理

最简单的方法：将位置索引线性压缩到原始范围内。

```
原始:   位置 0, 1, 2, ..., 8191
插值后: 位置 0, 0.25, 0.5, ..., 2047.75  (映射到 0-2048 范围内)
```

所有位置都"见过"，但分辨率降低了 4 倍。

### 7.2 实现代码

```python
def precompute_pos_cis_pi(dim: int, end: int, theta: float = 10000.0,
                          scale_factor: float = 1.0):
    """
    Position Interpolation 风格的 RoPE

    将位置索引线性压缩：pos_new = pos / scale_factor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 插值：位置索引除以缩放因子
    t = torch.arange(end, device=freqs.device) / scale_factor
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis
```

---

## 8. 策略对比实验

### 预期效果

| 策略 | 2K→8K 续训效果 | 8K→32K 续训效果 | 实现难度 | 推荐度 |
|------|--------------|--------------|---------|-------|
| 直接外推 | 差 | 很差 | 无 | 不推荐 |
| PI（位置插值） | 中等 | 较差 | 极低 | 快速验证 |
| **NTK-aware** | **好** | **中等** | **低** | **Stage 2 首选** |
| YaRN | 好 | 好 | 中等 | Stage 3 考虑 |

### 建议方案

```
Stage 1 (2K): rope_scaling_factor = 1.0 （默认，不缩放）
Stage 2 (8K): rope_scaling_factor = 4.0 （NTK-aware）
Stage 3 (32K): rope_scaling_factor = 16.0 （NTK-aware 或 YaRN）
```

---

## 9. 与续训流程的集成

### 续训时的关键点

1. **pos_cis 是 non-persistent buffer**：不会保存在 `state_dict` 中
2. 续训加载权重时，模型会用新的 `max_seq_len` 和 `rope_scaling_factor` 重新计算 pos_cis
3. 这意味着**无需特殊处理权重加载**，只需更新配置即可

### pretrain.py 需要的改动

```python
# 1. 命令行参数：添加 --resume_from 和 --max_seq_len
parser.add_argument("--resume_from", type=str, default=None,
                    help="续训权重路径")
parser.add_argument("--max_seq_len", type=int, default=None,
                    help="覆盖配置文件中的 max_seq_len")

# 2. 加载权重
if args.resume_from:
    state_dict = torch.load(args.resume_from, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # strict=False 因为 pos_cis 不在 state_dict 中

# 3. 覆盖 max_seq_len
if args.max_seq_len:
    config.max_seq_len = args.max_seq_len
    # 自动重算 rope_scaling_factor
    if config.original_max_seq_len:
        config.rope_scaling_factor = args.max_seq_len / config.original_max_seq_len
```

---

## 10. 验证方法

扩展 RoPE 后，在正式续训前应先验证：

```python
# 对比原始位置 0-2047 的编码在新旧 theta 下是否一致
from modeling_mindlm import precompute_pos_cis

head_dim = 64  # 768 / 12

# 原始 2K 编码
pos_cis_original = precompute_pos_cis(head_dim, 2048, theta=10000.0)

# NTK-aware 8K 编码
pos_cis_ntk = precompute_pos_cis(head_dim, 8192, theta=10000.0, rope_scaling_factor=4.0)

# 检查位置 0-2047 的差异
diff = (pos_cis_original - pos_cis_ntk[:2048]).abs().mean()
print(f"位置 0-2047 平均差异: {diff:.6f}")  # 应该很小
```
