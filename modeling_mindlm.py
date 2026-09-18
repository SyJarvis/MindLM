"""
MindLM
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    # Optional fused training kernel. When available, ``gated_delta_rule``
    # inference and training dispatch to this chunk kernel instead of the
    # per-token PyTorch reference. The reference path remains the fallback for
    # single-token recurrence and for environments without fla installed.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdr
except ImportError:  # pragma: no cover - fla is an optional acceleration dep
    _fla_chunk_gdr = None

try:
    from flash_attn.cute import flash_attn_func as _flash_attn_4
except ImportError:  # pragma: no cover - FA4 is an optional CUDA dependency
    _flash_attn_4 = None

@dataclass(frozen=True)
class AttentionCache:
    """Rotated keys and values in [batch, time, KV heads, head dim]."""

    key: torch.Tensor
    value: torch.Tensor

@dataclass(frozen=True)
class GatedDeltaNetCache:
    """Raw convolution history [B, C, K-1] and FP32 state [B, H, Dk, Dv]."""

    conv_state: torch.Tensor
    recurrent_state: torch.Tensor

@dataclass(frozen=True)
class MindLMCache:
    """Request-local inference cache; pass only new tokens when reusing it."""

    seq_length: int
    layers: Tuple[Union[AttentionCache, GatedDeltaNetCache], ...]

def _validate_cache_tensor(tensor, shape, device, name, dtype=None):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must be a tensor with shape {tuple(shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")

@dataclass
class MindLMCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal LM output extended with the pre-head hidden state."""

    last_hidden_state: Optional[torch.FloatTensor] = None

class RMSNorm(nn.Module):
    """RMSNorm实现"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算旋转位置编码"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis

def apply_rotary_emb(xq, xk, pos_cis):
    """应用旋转位置编码"""
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复KV头"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """L2归一化"""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

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
        self.attention_backend = args.attention_backend
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        mask = torch.ones((1, 1, args.max_seq_len, args.max_seq_len), dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def _flash_attention(self, query, key, value):
        if _flash_attn_4 is None:
            raise RuntimeError("attention_backend='flash_attn_4' requires flash_attn.cute on CUDA")
        if self.training and self.dropout != 0:
            raise ValueError("flash_attn_4 does not support nonzero attention dropout during training")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash_attn_4 requires float16 or bfloat16 inputs")
        output, _lse = _flash_attn_4(query, key, value, causal=True)
        return output

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, past_key_value=None, use_cache=False):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        if pos_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        past_length = 0
        if past_key_value is not None:
            past_length = past_key_value.key.size(1)
            _validate_cache_tensor(past_key_value.key,
                                   (bsz, past_length, self.n_local_kv_heads, self.head_dim),
                                   xk.device, "attention key cache", xk.dtype)
            _validate_cache_tensor(past_key_value.value, past_key_value.key.shape,
                                   xv.device, "attention value cache", xv.dtype)
            xk = torch.cat((past_key_value.key, xk), dim=1)
            xv = torch.cat((past_key_value.value, xv), dim=1)
        # Store native GQA heads, before repeating them for the SDPA fallback.
        present = AttentionCache(xk.detach(), xv.detach()) if use_cache else None

        # Keep FA4 for square prefill/training. Cached rectangular attention uses
        # an explicit offset mask, independent of backend causal alignment.
        if self.attention_backend == 'flash_attn_4' and xq.is_cuda and not past_length:
            output = self._flash_attention(xq, xk, xv)
            output = self.resid_dropout(self.wo(output.reshape(bsz, seqlen, -1)))
            return (output, present) if use_cache else output

        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        xq = xq.transpose(1, 2)
        total_length = past_length + seqlen
        blocked = self.mask[:, :, past_length:total_length, :total_length]
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=~blocked if past_length else None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=not bool(past_length),
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(blocked, float("-inf"))
            scores = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq))
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.resid_dropout(self.wo(output))
        return (output, present) if use_cache else output

class GatedDeltaNet(nn.Module):
    """Gated DeltaNet with the complete prediction-residual update rule."""
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.dim
        self.num_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.num_k_heads = getattr(args, 'linear_attn_heads', None) or args.n_heads
        self.num_v_heads = getattr(args, 'linear_attn_heads', None) or args.n_heads
        self.head_k_dim = self.head_dim
        self.head_v_dim = self.head_dim

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.chunk_size = getattr(args, 'linear_attn_chunk_size', 64)
        self.linear_attn_backend = args.linear_attn_backend
        if self.chunk_size < 1:
            raise ValueError("linear_attn_chunk_size must be positive")

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_v_heads + 1).float()))
        self.dt_bias._no_weight_decay = True
        self.A_log._no_weight_decay = True

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

    def _select_backend(self, device):
        if device.type != 'cuda':
            return 'reference'
        if _fla_chunk_gdr is None:
            raise RuntimeError("linear_attn_backend='fla' requires flash-linear-attention on CUDA")
        return 'fla'

    def forward(self, x: torch.Tensor, pos_cis=None, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        conv_state = None
        initial_state = None
        if past_key_value is not None:
            _validate_cache_tensor(past_key_value.conv_state,
                                   (batch_size, self.conv_dim, self.conv_kernel_size - 1),
                                   mixed_qkv.device, "GDN convolution cache", mixed_qkv.dtype)
            _validate_cache_tensor(past_key_value.recurrent_state,
                                   (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                                   mixed_qkv.device, "GDN recurrent cache", torch.float32)
            conv_input = torch.cat((past_key_value.conv_state, mixed_qkv), dim=-1)
            initial_state = past_key_value.recurrent_state
            mixed_qkv = F.conv1d(conv_input, self.conv1d.weight,
                                 bias=self.conv1d.bias, groups=self.conv_dim)
        else:
            conv_input = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0)) if use_cache else None
            mixed_qkv = self.conv1d(mixed_qkv)[:, :, :seq_len]
        if use_cache:
            # clone prevents the small history view retaining the full prefill.
            history_start = conv_input.size(-1) - (self.conv_kernel_size - 1)
            conv_state = conv_input[:, :, history_start:].clone().detach()
        mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            n_rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(n_rep, dim=2)
            key = key.repeat_interleave(n_rep, dim=2)

        present = None
        if use_cache:
            rule = self.gated_delta_rule_attention
            if seq_len > 1 and self._select_backend(query.device) == 'fla':
                rule = self.gated_delta_rule_fla
            core_attn_out, recurrent_state = rule(
                query, key, value, g, beta, initial_state=initial_state, return_state=True,
            )
            present = GatedDeltaNetCache(conv_state, recurrent_state.detach())
        elif self._select_backend(query.device) == 'fla':
            core_attn_out = self.gated_delta_rule_fla(query, key, value, g, beta)
        else:
            core_attn_out = self.gated_delta_rule_attention(query, key, value, g, beta)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.gated_norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        output = self.resid_dropout(output)
        return (output, present) if use_cache else output

    def gated_delta_rule_attention(
        self,
        query,
        key,
        value,
        g,
        beta,
        chunk_size=None,
        initial_state=None,
        return_state=False,
    ):
        """Reference implementation of the complete gated delta rule.

        The recurrent state is updated with the prediction residual rather than
        adding the value directly.  This is intentionally kept as a compact
        PyTorch reference for single-token decoding and environments without FLA.
        The loop is chunked to bound the temporary output lists.

        For each token, with ``S`` shaped ``[D_K, D_V]``:

        ``S <- exp(g) S``
        ``v_new <- beta * (v - k @ S)``
        ``S <- S + outer(k, v_new)``
        ``o <- q @ S``
        """
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        # Public callers use [B, T, H, D], while the recurrence is more natural
        # in [B, H, T, D].  Keep the accumulator in FP32 for BF16 stability.
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()

        output_dtype = query.dtype
        query = l2norm(query.float(), dim=-1).to(output_dtype).float()
        key = l2norm(key.float(), dim=-1).to(output_dtype).float()
        value = value.float()
        beta = beta.float()
        g = g.float()

        batch_size, num_heads, seq_len, head_dim = query.shape
        value_dim = value.shape[-1]
        query = query * (head_dim ** -0.5)
        if initial_state is None:
            state = torch.zeros(
                batch_size,
                num_heads,
                head_dim,
                value_dim,
                device=query.device,
                dtype=torch.float32,
            )
        else:
            expected_shape = (batch_size, num_heads, head_dim, value_dim)
            if tuple(initial_state.shape) != expected_shape:
                raise ValueError(
                    f"initial_state must have shape {expected_shape}, "
                    f"got {tuple(initial_state.shape)}"
                )
            state = initial_state.to(device=query.device, dtype=torch.float32)
        if seq_len == 0:
            output = torch.empty(
                batch_size,
                0,
                num_heads,
                value_dim,
                device=query.device,
                dtype=output_dtype,
            )
            return (output, state) if return_state else output
        output_chunks = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_outputs = []
            for t in range(start, end):
                # Avoid in-place updates: all intermediate states remain valid
                # for autograd when training the reference implementation.
                state = state * torch.exp(g[:, :, t]).unsqueeze(-1).unsqueeze(-1)
                predicted = torch.einsum('bhd,bhdv->bhv', key[:, :, t], state)
                residual = beta[:, :, t].unsqueeze(-1) * (value[:, :, t] - predicted)
                state = state + key[:, :, t].unsqueeze(-1) * residual.unsqueeze(-2)
                chunk_outputs.append(torch.einsum('bhd,bhdv->bhv', query[:, :, t], state))
            output_chunks.append(torch.stack(chunk_outputs, dim=2))

        output = torch.cat(output_chunks, dim=2)
        output = output.transpose(1, 2).contiguous().to(dtype=output_dtype)
        return (output, state) if return_state else output

    def gated_delta_rule_fla(self, query, key, value, g, beta, initial_state=None, return_state=False):
        """Fused FLA chunk kernel for the gated delta rule.

        Numerically equivalent to :meth:`gated_delta_rule_attention` (the
        per-token reference) but offloads the chunked recurrence to the FLA
        Triton kernel. Both l2-normalization of ``query``/``key`` and the
        ``1/sqrt(head_dim)`` scale are fused into the kernel, matching the
        reference. ``query``, ``key``, ``value`` are ``[B, T, H, D]``;
        ``g``/``beta`` are ``[B, T, H]`` — exactly the public calling layout
        used by :meth:`forward`, so no transposes are needed here.

        Cached prefill and multi-token continuations use FLA's initial/final
        state interface, with the default [B, H, Dk, Dv] state layout.
        """
        if _fla_chunk_gdr is None:
            raise RuntimeError(
                "flash-linear-attention is required for the FLA kernel path "
                "but is not installed; install it or use the reference path."
            )
        if self.chunk_size not in {16, 32, 64}:
            raise ValueError(
                "The FLA gated delta rule kernel requires "
                "linear_attn_chunk_size to be 16, 32, or 64"
            )
        output_dtype = query.dtype
        scale = query.shape[-1] ** -0.5
        out = _fla_chunk_gdr(
            query,
            key,
            value,
            g.float(),
            beta.float(),
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            chunk_size=self.chunk_size,
            initial_state=initial_state,
            output_final_state=return_state,
        )
        if return_state:
            if not isinstance(out, tuple) or len(out) != 2 or out[1] is None:
                raise RuntimeError("FLA kernel did not return the requested final recurrent state")
            output, state = out
            return output.to(dtype=output_dtype), state.float()
        out = out[0] if isinstance(out, tuple) else out
        return out.to(dtype=output_dtype)

    def gated_norm(self, x, gate):
        """门控RMSNorm"""
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        x = x * F.silu(gate.float())
        return x.to(input_dtype)

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MindLMConfig(PretrainedConfig):
    """MindLM配置"""
    model_type = "mindlm"
    def __init__(
        self,
        # ========== 核心维度参数 ==========
        dim: int = 512,                   # 模型隐藏维度（如 576=0.5B, 768=1B）
        n_layers: int = 8,                # Transformer 层数（如 12=0.5B, 16=1B）
        n_heads: int = 8,                 # 查询注意力头数（Q heads）
        n_kv_heads: int = None,           # KV头数，None表示等于n_heads；设为更小值启用GQA（如3=4:1比例）
        linear_attn_heads: int = None,     # 线性注意力头数，None表示等于n_heads；设为更小值启用线性注意力（如4=4:4比例）
        # ========== 词表与上下文 ==========
        vocab_size: int = 6400,           # 分词器词表大小
        max_seq_len: int = 512,         # 最大序列长度（0.5B=512, 1B=1024）
        # ========== 归一化与FFN参数 ==========
        dropout: float = 0.0,             # Dropout 比率，预训练通常为 0
        norm_eps: float = 1e-6,           # RMSNorm 的 epsilon，防止除零
        hidden_dim: int = None,           # FFN 隐藏维度，None 时自动计算为 int(2*4*dim/3)
        multiple_of: int = 256,           # FFN hidden_dim 对齐到此值的倍数，提高硬件利用率
        # ========== 混合注意力架构 ==========
        use_linear_attn: bool = True,     # 是否启用混合注意力（False 则全部用标准 Attention）
        layer_types: List[str] = None,    # 每层的注意力类型列表，如 ["linear_attention","attention",...]
                                          # None 时自动生成交替模式（use_linear_attn=True）
                                          # 推荐每4层一个标准attention（如1B: 12线性+4标准）
        # ========== GatedDeltaNet 特定参数 ==========
        conv_kernel_size: int = 4,        # 因果卷积核大小，提供局部位置感知，替代位置编码
        linear_attn_chunk_size: int = 64, # 线性注意力块大小；须在训练前固定
        linear_attn_backend: str = 'fla',
        attention_backend: str = 'flash_attn_4',
        # ========== 训练优化 ==========
        gradient_checkpointing: str = 'off',  # 梯度检查点策略：'off'=关闭，'linear_attn'=仅linear层，'all'=所有层
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim                    # 模型隐藏维度
        self.n_layers = n_layers          # Transformer 层数
        self.n_heads = n_heads            # Q 注意力头数
        self.n_kv_heads = n_kv_heads      # KV 注意力头数（GQA）
        self.linear_attn_heads = linear_attn_heads
        self.vocab_size = vocab_size      # 词表大小
        self.max_seq_len = max_seq_len    # 最大序列长度
        self.dropout = dropout            # Dropout
        self.norm_eps = norm_eps          # RMSNorm epsilon
        self.hidden_dim = hidden_dim      # FFN 隐藏维度
        self.multiple_of = multiple_of    # FFN 维度对齐基数

        # 混合注意力架构
        self.use_linear_attn = use_linear_attn      # 是否启用混合注意力
        if layer_types is None:
            if use_linear_attn:
                # 默认：每隔一层交替（可被 JSON 配置覆盖）
                self.layer_types = ["linear_attention" if i % 2 == 1 else "attention"
                                    for i in range(n_layers)]
            else:
                self.layer_types = ["attention"] * n_layers
        else:
            self.layer_types = layer_types          # 从 JSON 配置加载的层类型列表
        self.tie_word_embeddings = True         # tok_embeddings 和 output 共享权重
        self.conv_kernel_size = conv_kernel_size    # 因果卷积核大小
        self.linear_attn_chunk_size = linear_attn_chunk_size
        self.linear_attn_backend = linear_attn_backend
        self.attention_backend = attention_backend
        self.gradient_checkpointing = gradient_checkpointing  # 梯度检查点策略
        self.use_cache = False

        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if conv_kernel_size < 1:
            raise ValueError("conv_kernel_size must be positive")
        if linear_attn_chunk_size < 1:
            raise ValueError("linear_attn_chunk_size must be positive")
        if linear_attn_backend != 'fla':
            raise ValueError("linear_attn_backend='fla' is required")
        if attention_backend != 'flash_attn_4':
            raise ValueError("attention_backend='flash_attn_4' is required")
        if n_kv_heads is not None and n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if linear_attn_heads is not None and n_heads % linear_attn_heads != 0:
            raise ValueError("n_heads must be divisible by linear_attn_heads")
        if len(self.layer_types) != n_layers:
            raise ValueError("layer_types must contain exactly n_layers entries")
        invalid_layer_types = set(self.layer_types) - {"attention", "linear_attention"}
        if invalid_layer_types:
            raise ValueError(f"Unsupported layer types: {sorted(invalid_layer_types)}")

class TransformerBlock(nn.Module):
    """MindLM Transformer块"""
    def __init__(self, layer_id: int, args: MindLMConfig):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.layer_type = args.layer_types[layer_id]

        if self.layer_type == "linear_attention":
            self.attention = GatedDeltaNet(args)
            self.use_pos_cis = False
        else:
            self.attention = Attention(args)
            self.use_pos_cis = True

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

    def _block_forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        """整块前向（attention + FFN），用于 gradient checkpointing。"""
        attn_output = self.attention(self.attention_norm(x),
                                     pos_cis if self.use_pos_cis else None,
                                     past_key_value=past_key_value, use_cache=use_cache)
        if use_cache:
            attn_output, present = attn_output
        else:
            present = None
        h = x + attn_output
        ffn_input = self.ffn_norm(h)
        return h + self.feed_forward(ffn_input), present

    def forward(self, x, pos_cis=None, past_key_value=None, use_cache=False):
        gc = self.args.gradient_checkpointing
        if self.training and (gc == 'all' or (gc == 'linear_attn' and self.layer_type == "linear_attention")):
            return cp.checkpoint(self._block_forward, x, pos_cis, use_reentrant=False)
        return self._block_forward(x, pos_cis, past_key_value, use_cache)

class MindLM(PreTrainedModel, GenerationMixin):
    """MindLM主模型"""
    config_class = MindLMConfig
    _tied_weights_keys = {"output.weight": "tok_embeddings.weight"}

    def __init__(self, config: MindLMConfig = None):
        if config is None:
            config = MindLMConfig()
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        has_normal = "attention" in config.layer_types

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

        if has_normal:
            pos_cis = precompute_pos_cis(
                config.dim // config.n_heads,
                config.max_seq_len
            )
            self.register_buffer("pos_cis", pos_cis, persistent=False)
        else:
            self.pos_cis = None

        for layer in self.layers:
            projection = layer.attention.out_proj if layer.layer_type == 'linear_attention' else layer.attention.wo
            projection._mindlm_residual_projection = True
            for name, module in layer.feed_forward.named_modules():
                if name == 'w2' or name.endswith('.w2'):
                    module._mindlm_residual_projection = True
        # Apply the V3 scheme explicitly so every Linear/Embedding receives
        # the intended initialization exactly once.
        self.apply(self._init_weights)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # The embedding and output head share one parameter.
            if not getattr(module.weight, '_mindlm_v3_initialized', False):
                std = 0.02 / math.sqrt(2 * self.config.n_layers) if getattr(module, '_mindlm_residual_projection', False) else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                module.weight._mindlm_v3_initialized = True
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GatedDeltaNet):
            # ``post_init`` may be called again by callers after model
            # construction (for example when integrating with
            # Transformers utilities).  Keep the V3 initialization
            # idempotent just like the tied embedding/output path above;
            # re-sampling these recurrent parameters would silently alter
            # a freshly loaded model.
            if getattr(module, '_mindlm_v3_initialized', False):
                return
            with torch.no_grad():
                amplitude = torch.empty_like(module.A_log).uniform_(0, 16).clamp_min_(torch.finfo(module.A_log.dtype).tiny)
                module.A_log.copy_(amplitude.log())
                dt = torch.exp(torch.empty_like(module.dt_bias).uniform_(math.log(0.001), math.log(0.1)))
                module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            module._mindlm_v3_initialized = True
            return

    def mark_tied_weights_as_initialized(self, loading_info):
        """覆写父类方法：只标记已初始化，不从 missing_keys 中移除绑定权重。
        transformers 5.3.0 的实现会移除 remote code 模型的绑定权重，
        导致 tie_weights 误判两个权重都"存在"而拒绝绑定。"""
        for tied_param in self.all_tied_weights_keys.keys():
            param = self.get_parameter(tied_param)
            param._is_hf_initialized = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        tokens=None,
        targets=None,
        return_logits=True,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        """Run training or cached inference on new tokens.

        Pass the returned ``MindLMCache`` as ``past_key_values`` with only new
        tokens. Cache use requires eval mode; training defaults to no cache.
        Only unpadded batches are supported (``attention_mask`` must be all ones).
        Labels are already shifted by the dataset; ``-100`` positions are ignored.
        ``tokens`` and ``targets`` remain compatibility aliases.
        """
        if input_ids is None:
            input_ids = tokens
        if labels is None:
            labels = targets
        if input_ids is None:
            raise ValueError("input_ids is required")
        if labels is not None and not return_logits:
            raise ValueError("labels require return_logits=True")

        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError("Caching is only supported in eval mode; use model.eval()")
        if labels is not None and use_cache:
            raise ValueError("labels are not supported with use_cache=True")
        bsz, seqlen = input_ids.shape
        past_length = self._validate_cache(past_key_values, bsz, input_ids.device)
        total_length = past_length + seqlen
        if total_length > self.config.max_seq_len:
            raise ValueError(
                f"Input length including cache {total_length} exceeds max_seq_len={self.config.max_seq_len}."
            )
        self._validate_attention_mask(attention_mask, bsz, total_length)
        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)

        pos_cis = None
        if self.pos_cis is not None:
            pos_cis = self.pos_cis[past_length:total_length]

        presents = []
        for index, layer in enumerate(self.layers):
            past = past_key_values.layers[index] if past_key_values is not None else None
            h, present = layer(h, pos_cis, past_key_value=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        h = self.norm(h)

        logits = self.output(h) if return_logits else None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return MindLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=MindLMCache(total_length, tuple(presents)) if use_cache else None,
            hidden_states=None,
            attentions=None,
            last_hidden_state=h if not return_logits else None,
        )

    @staticmethod
    def _validate_attention_mask(attention_mask, batch_size, seq_length):
        if attention_mask is None:
            return
        if tuple(attention_mask.shape) != (batch_size, seq_length):
            raise ValueError("attention_mask must have shape [batch, past length + new length]")
        if not torch.all(attention_mask == 1):
            raise ValueError("Only all-ones attention_mask is supported; padded or masked batches are not supported")

    def _validate_cache(self, cache, batch_size, device):
        if cache is None:
            return 0
        if not isinstance(cache, MindLMCache):
            raise ValueError("past_key_values must be a MindLMCache returned by this model")
        if type(cache.seq_length) is not int or not 0 < cache.seq_length <= self.config.max_seq_len:
            raise ValueError("cache seq_length must be positive and at most max_seq_len")
        if not isinstance(cache.layers, tuple) or len(cache.layers) != self.n_layers:
            raise ValueError("cache layers must contain exactly one entry per model layer")
        for layer, entry in zip(self.layers, cache.layers):
            attn = layer.attention
            if isinstance(attn, Attention):
                if not isinstance(entry, AttentionCache):
                    raise ValueError("attention layer requires AttentionCache")
                shape = (batch_size, cache.seq_length, attn.n_local_kv_heads, attn.head_dim)
                _validate_cache_tensor(entry.key, shape, device, "attention key cache")
                _validate_cache_tensor(entry.value, shape, device, "attention value cache")
            else:
                if not isinstance(entry, GatedDeltaNetCache):
                    raise ValueError("linear_attention layer requires GatedDeltaNetCache")
                _validate_cache_tensor(entry.conv_state,
                                       (batch_size, attn.conv_dim, attn.conv_kernel_size - 1),
                                       device, "GDN convolution cache")
                _validate_cache_tensor(entry.recurrent_state,
                                       (batch_size, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim),
                                       device, "GDN recurrent cache", torch.float32)
        return cache.seq_length

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        max_new_tokens=20,
        eos_token_id=None,
        pad_token_id=None,
        do_sample=True,
        temperature=0.7,
        top_k=8,
        eos=None,
        use_cache=True,
        attention_mask=None,
        **kwargs,
    ):
        """Generate with one prefill followed by cached single-token steps.

        Set ``use_cache=False`` to recompute the full context at each step.
        Generation temporarily uses eval mode and restores the prior mode.
        """
        if input_ids is None:
            input_ids = kwargs.pop("idx", None)
        if input_ids is None:
            raise ValueError("input_ids is required")
        if eos_token_id is None:
            eos_token_id = eos if eos is not None else getattr(self.config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        if isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = set(eos_token_id)
        elif eos_token_id is None:
            eos_token_ids = set()
        else:
            eos_token_ids = {eos_token_id}

        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"Input length exceeds max_seq_len={self.config.max_seq_len}")
        self._validate_attention_mask(attention_mask, input_ids.size(0), input_ids.size(1))
        generated = input_ids
        past_key_values = None
        unfinished = torch.ones(generated.size(0), dtype=torch.bool, device=generated.device)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                if generated.size(1) >= self.config.max_seq_len:
                    break

                model_input = generated[:, -1:] if past_key_values is not None else generated
                result = self(model_input, past_key_values=past_key_values, use_cache=use_cache)
                past_key_values = result.past_key_values
                logits = result.logits[:, -1, :]
                if do_sample and temperature > 0:
                    logits = logits / temperature
                    if top_k is not None and top_k > 0:
                        threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                        logits = logits.masked_fill(logits < threshold, -float("inf"))
                    probabilities = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                next_token = torch.where(
                    unfinished.unsqueeze(-1),
                    next_token,
                    torch.full_like(next_token, pad_token_id),
                )
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_ids:
                    is_eos = torch.zeros_like(unfinished)
                    for token_id in eos_token_ids:
                        is_eos |= next_token.squeeze(-1).eq(token_id)
                    unfinished &= ~is_eos
                    if not unfinished.any():
                        break
        finally:
            self.train(was_training)

        return generated
