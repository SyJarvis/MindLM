# Attention 滑动窗口实现设计

## 1. 背景：为什么需要滑动窗口？

### 1.1 MindLM-1B 的 Attention 显存瓶颈

MindLM-1B 有 4 层标准 Attention（第 3,7,11,15 层），计算复杂度 O(T²)：

| 上下文长度 | 单层 Q×K 矩阵大小 | 4层合计 (bf16) | 是否可行 |
|-----------|------------------|--------------|---------|
| 2K | 2K×2K = 4M | ~16MB | 轻松 |
| 8K | 8K×8K = 64M | ~256MB | 可行（Flash Attention） |
| 32K | 32K×32K = 1B | ~4GB | **需要滑动窗口** |

Flash Attention 通过不物化完整注意力矩阵来节省显存，但在 32K 时即使 Flash Attention 的显存开销也很高。

### 1.2 滑动窗口原理

限制每个 token 只关注最近的 W 个 token，将复杂度从 O(T²) 降为 O(T×W)：

```
标准 Attention (T=8):           滑动窗口 (T=8, W=4):
  0 1 2 3 4 5 6 7                0 1 2 3 4 5 6 7
0 ✓                            ✓
1 ✓ ✓                          ✓ ✓
2 ✓ ✓ ✓                        ✓ ✓ ✓
3 ✓ ✓ ✓ ✓                      ✓ ✓ ✓ ✓
4 ✓ ✓ ✓ ✓ ✓                      ✓ ✓ ✓ ✓
5 ✓ ✓ ✓ ✓ ✓ ✓                      ✓ ✓ ✓ ✓
6 ✓ ✓ ✓ ✓ ✓ ✓ ✓                      ✓ ✓ ✓ ✓
7 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓                      ✓ ✓ ✓ ✓

每个 token 看到所有前序 token    每个 token 只看最近 4 个 token
显存: O(T²)                     显存: O(T×W)
```

### 1.3 MindLM 混合架构的优势

```
12层 Linear Attention: O(T) 天然长上下文，无需滑动窗口
 4层标准 Attention:    O(T²) ← 滑动窗口只作用于这 4 层

组合效果:
  - Linear Attention 层提供全局信息传递（recurrent state 跨整个序列）
  - 标准 Attention 层通过滑动窗口处理局部精细关系
  - 信息通过 12 层 Linear Attention 在整个序列中流动
```

这意味着即使标准 Attention 只看 4K 窗口，模型仍然能通过 Linear Attention 层获得全局上下文。

---

## 2. 原始代码

### 2.1 Attention 类（modeling_mindlm.py:74-133）

```python
class Attention(nn.Module):
    """标准多头注意力"""
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if pos_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash and seqlen != 1:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

### 2.2 原始代码的问题

1. **`__init__` 中预分配固定大小 mask**：`max_seq_len × max_seq_len` 矩阵。32K 时为 32K×32K = 1G 元素 = **4GB 显存**
2. **Flash Attention 路径**：`is_causal=True` 只支持完整因果 mask，无法限制窗口
3. **无可配置的窗口大小参数**

---

## 3. 实现方案

### 3.1 MindLMConfig 新增参数

```python
class MindLMConfig(PretrainedConfig):
    model_type = "mindlm"
    def __init__(
        self,
        # ... 原有参数保持不变 ...
        # ========== 新增滑动窗口参数 ==========
        sliding_window: int = 0,    # 滑动窗口大小，0 = 不启用
        # ======================================
        **kwargs
    ):
        super().__init__(**kwargs)
        # ... 原有赋值保持不变 ...
        self.sliding_window = sliding_window
```

### 3.2 修改后的 Attention 类

```python
class Attention(nn.Module):
    """标准多头注意力，支持滑动窗口"""
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # 滑动窗口大小（0 = 不启用，使用完整因果注意力）
        self.sliding_window = getattr(args, 'sliding_window', 0)

        if self.sliding_window > 0:
            # 滑动窗口模式：mask 在 forward 中动态创建
            # 不预分配大矩阵，节省显存
            self.register_buffer("mask", None, persistent=False)
        else:
            # 原始模式：预分配完整因果 mask
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果滑动窗口 mask

        位置 i 可以关注位置 j 当且仅当：
            j <= i  （因果性：只看过去）
            AND i - j < sliding_window  （窗口限制：只看最近的 W 个 token）

        Returns:
            mask: shape (1, 1, seq_len, seq_len)
                  0.0 = 允许关注, -inf = 阻止
        """
        # 行坐标 i，列坐标 j
        rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)

        # 因果 + 窗口条件
        causal = cols <= rows                                    # j <= i
        within_window = (rows - cols) < self.sliding_window      # i - j < W
        allowed = causal & within_window                         # 两个条件同时满足

        # 构建 float mask: 允许=0.0, 阻止=-inf
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask[allowed] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if pos_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)   # (B, H, S, D)
        xk = xk.transpose(1, 2)   # (B, H, S, D)
        xv = xv.transpose(1, 2)   # (B, H, S, D)

        # 判断是否需要滑动窗口
        use_sliding_window = (
            self.sliding_window > 0
            and seqlen > self.sliding_window
        )

        if self.flash and seqlen != 1:
            if use_sliding_window:
                # 滑动窗口 + Flash Attention：需要显式 mask
                sw_mask = self._create_sliding_window_mask(seqlen, xq.device)
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=sw_mask.expand(bsz * self.n_local_heads, -1, -1),
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,  # mask 已包含因果性
                )
            else:
                # 正常 Flash Attention：利用 is_causal 优化
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
        else:
            # 非 Flash Attention 回退路径
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if use_sliding_window:
                mask = self._create_sliding_window_mask(seqlen, xq.device)
            else:
                mask = self.mask[:, :, :seqlen, :seqlen]
            scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

---

## 4. 代码改动对比

### 4.1 `__init__` 改动

```diff
  class Attention(nn.Module):
      def __init__(self, args):
          super().__init__()
          # ... 原有参数不变 ...
+
+         # 滑动窗口大小（0 = 不启用）
+         self.sliding_window = getattr(args, 'sliding_window', 0)

-         mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
-         mask = torch.triu(mask, diagonal=1)
-         self.register_buffer("mask", mask, persistent=False)
+         if self.sliding_window > 0:
+             # 滑动窗口：动态创建 mask，不预分配
+             self.register_buffer("mask", None, persistent=False)
+         else:
+             # 原始：预分配完整因果 mask
+             mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
+             mask = torch.triu(mask, diagonal=1)
+             self.register_buffer("mask", mask, persistent=False)
```

### 4.2 `forward` 改动

```diff
  def forward(self, x, pos_cis, kv_cache=False):
      # ... Q/K/V 投影和 reshape 不变 ...

+     # 判断是否需要滑动窗口
+     use_sliding_window = (
+         self.sliding_window > 0
+         and seqlen > self.sliding_window
+     )

      if self.flash and seqlen != 1:
-         output = torch.nn.functional.scaled_dot_product_attention(
-             xq, xk, xv, attn_mask=None,
-             dropout_p=self.dropout if self.training else 0.0,
-             is_causal=True
-         )
+         if use_sliding_window:
+             sw_mask = self._create_sliding_window_mask(seqlen, xq.device)
+             output = torch.nn.functional.scaled_dot_product_attention(
+                 xq, xk, xv,
+                 attn_mask=sw_mask.expand(bsz * self.n_local_heads, -1, -1),
+                 dropout_p=self.dropout if self.training else 0.0,
+                 is_causal=False,
+             )
+         else:
+             output = torch.nn.functional.scaled_dot_product_attention(
+                 xq, xk, xv, attn_mask=None,
+                 dropout_p=self.dropout if self.training else 0.0,
+                 is_causal=True,
+             )
      else:
          scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
-         scores = scores + self.mask[:, :, :seqlen, :seqlen]
+         if use_sliding_window:
+             mask = self._create_sliding_window_mask(seqlen, xq.device)
+         else:
+             mask = self.mask[:, :, :seqlen, :seqlen]
+         scores = scores + mask
          scores = F.softmax(scores.float(), dim=-1).type_as(xq)
          scores = self.attn_dropout(scores)
          output = torch.matmul(scores, xv)
```

### 4.3 新增方法

```python
def _create_sliding_window_mask(self, seq_len, device):
    """创建因果滑动窗口 mask"""
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = cols <= rows
    within_window = (rows - cols) < self.sliding_window
    allowed = causal & within_window
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask[allowed] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)
```

---

## 5. 显存分析

### 5.1 各配置下的 mask 显存占用

| 配置 | seq_len | window | mask 大小 | 显存占用 |
|------|---------|--------|----------|---------|
| Stage 1 | 2048 | 不启用 | 2K×2K | 16MB（预分配） |
| Stage 2 | 8192 | 不启用 | 8K×8K | 256MB（预分配） |
| Stage 3 (无窗口) | 32768 | 不启用 | 32K×32K | **4GB（预分配，不可行）** |
| Stage 3 (窗口4K) | 32768 | 4096 | 32K×32K | 4GB（动态创建） |

**注意**：滑动窗口 mask 本身仍然是 32K×32K 矩阵，只是大部分是 -inf。要真正节省显存，需要更高级的方案。

### 5.2 实际显存瓶颈分析

在使用 Flash Attention 时，注意力矩阵 **不会** 被完整物化到显存中。Flash Attention 的显存占用是 O(T) 而非 O(T²)。

当传入 `attn_mask` 参数时，`scaled_dot_product_attention` 的行为：

| 模式 | attn_mask | is_causal | 内部实现 | 显存 |
|------|-----------|-----------|---------|------|
| 原始 Flash | None | True | Flash Attention | O(T) |
| 窗口 Flash | dense mask | False | **可能回退到 math** | O(T²) |

**关键问题**：传入显式 attn_mask 可能导致 SDPA 回退到 O(T²) 的数学实现。

### 5.3 方案选择建议

| 上下文 | 建议 | 原因 |
|--------|------|------|
| 8K | 不启用滑动窗口 | Flash Attention 处理 8K 没问题 |
| 32K（显存充足） | 启用窗口 + 块状处理 | 减少 attention 计算量 |
| 32K（显存紧张） | 启用窗口 + Block Attention | 完全避免 O(T²) |

---

## 6. Block Attention 方案（32K 显存优化版）

### 6.1 原理

将序列分成 `T/W` 个大小为 W 的块，每个块只与自身和前一个块计算 attention：

```
块状 Attention (T=8, W=4):
      B0   B1
B0  [✓✓✓✓ ✓✓✓✓]
B1  [      ✓✓✓✓ ✓✓✓✓]

每行实际计算的 attention: W×2W = O(W²)
总计: (T/W) × W × 2W = O(T×W)
显存: O(W²) per block
```

### 6.2 实现代码

```python
def _block_sliding_attention(self, xq, xk, xv, seq_len):
    """
    块状滑动窗口注意力

    将序列分成 block_size 大小的块，每个块只与自身和前一个块计算 attention。
    避免创建完整的 T×T 矩阵。

    Args:
        xq: (B, H, T, D)
        xk: (B, H, T, D)
        xv: (B, H, T, D)
        seq_len: T

    Returns:
        output: (B, H, T, D)
    """
    W = self.sliding_window
    num_blocks = (seq_len + W - 1) // W
    head_dim = xq.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    output = torch.zeros_like(xq)

    for b in range(num_blocks):
        # 当前块的范围
        q_start = b * W
        q_end = min(q_start + W, seq_len)
        q_len = q_end - q_start

        # KV 范围：当前块 + 前一个块（滑动窗口）
        kv_start = max(0, q_start - W)
        kv_end = q_end
        kv_len = kv_end - kv_start

        # 取出对应块
        q_block = xq[:, :, q_start:q_end]           # (B, H, q_len, D)
        k_block = xk[:, :, kv_start:kv_end]          # (B, H, kv_len, D)
        v_block = xv[:, :, kv_start:kv_end]          # (B, H, kv_len, D)

        # 块内 attention（小矩阵：W × 2W）
        scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

        # 创建块级因果 mask
        # q 的绝对位置: q_start .. q_end-1
        # kv 的绝对位置: kv_start .. kv_end-1
        q_pos = torch.arange(q_start, q_end, device=xq.device)
        k_pos = torch.arange(kv_start, kv_end, device=xq.device)
        # 因果 + 窗口
        causal = k_pos[None, :] <= q_pos[:, None]
        within_window = (q_pos[:, None] - k_pos[None, :]) < W
        allowed = causal & within_window
        block_mask = torch.zeros(q_len, kv_len, device=xq.device)
        block_mask[~allowed] = float('-inf')

        scores = scores + block_mask.unsqueeze(0).unsqueeze(0)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq_block)
        scores = self.attn_dropout(scores)

        block_output = torch.matmul(scores, v_block)  # (B, H, q_len, D)
        output[:, :, q_start:q_end] = block_output

    return output
```

### 6.3 集成到 Attention.forward

```python
def forward(self, x, pos_cis, kv_cache=False):
    # ... Q/K/V 投影不变 ...

    use_sliding_window = (
        self.sliding_window > 0
        and seqlen > self.sliding_window
    )

    if self.flash and seqlen != 1 and not use_sliding_window:
        # 原始 Flash Attention 路径（无滑动窗口）
        output = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
    elif use_sliding_window:
        # 滑动窗口：使用块状 attention 节省显存
        output = self._block_sliding_attention(xq, xk, xv, seqlen)
    else:
        # 非 Flash 回退路径
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.mask[:, :, :seqlen, :seqlen]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)

    # ... 输出投影不变 ...
```

### 6.4 Block Attention 显存分析

| 场景 | 矩阵大小 | 显存/每块 | 总 attention 显存 |
|------|---------|----------|-----------------|
| 32K, W=4K | 4K × 8K | ~256KB | ~2MB（8 块 × 256KB） |
| 32K, W=8K | 8K × 16K | ~1MB | ~4MB（4 块 × 1MB） |

相比完整 32K×32K（4GB），节省了 **1000 倍以上**。

---

## 7. 配置示例

### 7.1 JSON 配置文件

#### Stage 1/2：不启用滑动窗口

```json
{
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "n_kv_heads": 3,
    "vocab_size": 6400,
    "max_seq_len": 8192,
    "sliding_window": 0
}
```

#### Stage 3（32K）：启用滑动窗口

```json
{
    "dim": 768,
    "n_layers": 16,
    "n_heads": 12,
    "n_kv_heads": 3,
    "vocab_size": 6400,
    "max_seq_len": 32768,
    "sliding_window": 4096
}
```

### 7.2 MindLM-1B 各阶段推荐配置

| 阶段 | max_seq_len | sliding_window | 理由 |
|------|------------|----------------|------|
| Stage 1 | 2048 | 0 | 不需要 |
| Stage 2 | 8192 | 0 | Flash Attention 够用 |
| Stage 3 | 32768 | 4096 | 32K 显存瓶颈，4K 窗口足够 |

---

## 8. 完整改动清单

### 8.1 需要修改的文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `modeling_mindlm.py` | `MindLMConfig` | 添加 `sliding_window` 参数 |
| `modeling_mindlm.py` | `Attention.__init__` | 条件化 mask 预分配 |
| `modeling_mindlm.py` | `Attention.forward` | 滑动窗口分支逻辑 |
| `modeling_mindlm.py` | `Attention` 新增方法 | `_create_sliding_window_mask` |
| `modeling_mindlm.py` | `Attention` 新增方法 | `_block_sliding_attention`（可选） |
| `config/mindlm_1b_32k.json` | 新文件 | Stage 3 配置 |

### 8.2 不需要修改的部分

- `GatedDeltaNet`（Linear Attention）：天然 O(T)，不受影响
- `TransformerBlock`：透传参数，无需改动
- `MindLM.forward`：pos_cis 切片逻辑不变
- `pretrain.py`：训练循环不变

---

## 9. 注意事项

### 9.1 滑动窗口大小的选择

- **太小**（如 512）：标准 Attention 层只能看到极近的上下文，丢失局部精细关系
- **太大**（如 16K）：显存优化效果有限
- **推荐**：4096，覆盖约 1-2 页文本，对大多数任务足够

### 9.2 与 Linear Attention 的协同

滑动窗口限制标准 Attention 只看局部，但 Linear Attention 层不受限制。这意味着：

```
Token 1 ─[Linear Attn]→ Token 10000  ✓ 全局信息可以传递
Token 1 ─[Standard Attn]→ Token 10000  ✗ 窗口限制，无法直接看到
Token 9999 ─[Standard Attn]→ Token 10000  ✓ 在窗口内，可以看到
```

模型通过交替层实现"全局 + 局部"的信息融合。

### 9.3 训练 vs 推理

- **训练时**：full sequence 并行，滑动窗口通过 mask 实现
- **推理时**：如果实现 KV-cache，需要限制缓存的 KV 对数量为 window_size（当前 MindLM 未实现 KV-cache，暂不影响）
