"""
MindLM
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import List, Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    # Optional fused training kernel. When available, ``gated_delta_rule``
    # inference and training dispatch to this chunk kernel instead of the
    # per-token PyTorch reference. The reference path remains the fallback for
    # stateful recurrence and for environments without fla installed.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gdr
except ImportError:  # pragma: no cover - fla is an optional acceleration dep
    _fla_chunk_gdr = None

try:
    from flash_attn.cute import flash_attn_func as _flash_attn_4
except ImportError:  # pragma: no cover - FA4 is an optional CUDA dependency
    _flash_attn_4 = None


@dataclass
class MindLMCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal LM output extended with the MoE load-balancing loss."""

    aux_loss: Optional[torch.FloatTensor] = None
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

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if pos_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # CPU evaluation uses SDPA; FA4 receives native BTHD GQA on CUDA.
        if self.attention_backend == 'flash_attn_4' and xq.is_cuda:
            output = self._flash_attention(xq, xk, xv)
            return self.resid_dropout(self.wo(output.reshape(bsz, seqlen, -1)))

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
            scores = scores.masked_fill(self.mask[:, :, :seqlen, :seqlen], float("-inf"))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention with selectable legacy and full rules.

    ``simple`` preserves the original MindLM recurrence for old checkpoints;
    ``gated_delta_rule`` applies the prediction-residual update used by FLA.
    """
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
        self.linear_attn_impl = getattr(args, 'linear_attn_impl', 'simple')
        self.linear_attn_backend = args.linear_attn_backend
        if self.linear_attn_impl not in {'simple', 'gated_delta_rule'}:
            raise ValueError(
                "linear_attn_impl must be 'simple' or 'gated_delta_rule'"
            )
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
        self.register_buffer(
            "chunk_causal_mask",
            torch.tril(torch.ones(self.chunk_size, self.chunk_size, dtype=torch.bool)),
            persistent=False,
        )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_v_heads + 1).float()))
        if args.initialization_scheme == 'gdn_v3':
            self.dt_bias._no_weight_decay = True
            self.A_log._no_weight_decay = True

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

    def _select_backend(self, device):
        if device.type != 'cuda' or self.linear_attn_backend == 'reference':
            return 'reference'
        if self.linear_attn_backend == 'fla':
            if _fla_chunk_gdr is None:
                raise RuntimeError("linear_attn_backend='fla' requires flash-linear-attention on CUDA")
            return 'fla'
        return 'fla' if _fla_chunk_gdr is not None and not getattr(self, 'use_reference_gdr', False) else 'reference'

    def forward(self, x: torch.Tensor, pos_cis=None, kv_cache=False):
        batch_size, seq_len, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)

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

        if self.linear_attn_impl == 'gated_delta_rule':
            if self._select_backend(query.device) == 'fla':
                core_attn_out = self.gated_delta_rule_fla(query, key, value, g, beta)
            else:
                core_attn_out = self.gated_delta_rule_attention(query, key, value, g, beta)
        else:
            core_attn_out = self.simple_gated_delta_attention(query, key, value, g, beta)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.gated_norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        output = self.resid_dropout(output)
        return output

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
        PyTorch reference until a fused FLA/TileLang training kernel is wired in.
        The loop is chunked so the dispatch boundary is explicit and can later be
        replaced by a chunk kernel without changing the model interface.

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

    def gated_delta_rule_fla(self, query, key, value, g, beta):
        """Fused FLA chunk kernel for the gated delta rule.

        Numerically equivalent to :meth:`gated_delta_rule_attention` (the
        per-token reference) but offloads the chunked recurrence to the FLA
        Triton kernel. Both l2-normalization of ``query``/``key`` and the
        ``1/sqrt(head_dim)`` scale are fused into the kernel, matching the
        reference. ``query``, ``key``, ``value`` are ``[B, T, H, D]``;
        ``g``/``beta`` are ``[B, T, H]`` — exactly the public calling layout
        used by :meth:`forward`, so no transposes are needed here.

        Only the plain forward recurrence is supported; stateful calls
        (``initial_state``/``return_state``) still go through the reference.
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
        )
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

    def simple_gated_delta_attention(self, query, key, value, g, beta, chunk_size=None):
        """Chunked Gated Delta Attention — 分块并行计算，替代逐时间步Python循环

        将序列分成 chunk_size 大小的块，块内用矩阵乘法并行计算，
        块间传递 recurrent state。将 Python 循环从 T 次降到 T/chunk_size 次。
        """
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        batch_size, num_heads, seq_len, head_dim = query.shape
        v_dim = value.shape[-1]

        # g 已经是 log-space（负值），exp(g) 给出 (0,1) 的衰减因子
        log_g = g  # (B, H, T)

        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        output_chunks = []
        recurrent_state = torch.zeros(batch_size, num_heads, head_dim, v_dim,
                                      device=query.device, dtype=query.dtype)

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, seq_len)
            C = end - start

            q_c = query[:, :, start:end]       # (B, H, C, D)
            k_c = key[:, :, start:end]         # (B, H, C, D)
            v_c = value[:, :, start:end]       # (B, H, C, V)
            beta_c = beta[:, :, start:end]     # (B, H, C)
            log_g_c = log_g[:, :, start:end]   # (B, H, C)

            # 块内 log 空间累积和：log_cg[i] = sum_{u=0}^{i} log(g[u])
            log_cg = torch.cumsum(log_g_c, dim=-1)  # (B, H, C)

            # === 块内：衰减加权线性注意力 ===
            # 衰减比 log_ratio[i,j] = log_cg[i] - log_cg[j]，对应 g[j+1]*...*g[i]
            log_ratio = log_cg.unsqueeze(-1) - log_cg.unsqueeze(-2)  # (B, H, C, C)
            if chunk_size == self.chunk_size:
                causal_mask = self.chunk_causal_mask[:C, :C]
            else:
                causal_mask = torch.tril(
                    torch.ones(C, C, device=query.device, dtype=torch.bool)
                )

            # 衰减加权注意力矩阵
            # clamp(max=0) 防止上三角 exp 溢出：exp(大正数) * mask(0) = inf*0 = NaN
            qk = torch.matmul(q_c, k_c.transpose(-2, -1))  # (B, H, C, C)
            decay_attn = torch.exp(log_ratio.clamp(max=0)) * causal_mask * qk  # (B, H, C, C)

            v_beta = v_c * beta_c.unsqueeze(-1)  # (B, H, C, V)
            intra_out = torch.matmul(decay_attn, v_beta)  # (B, H, C, V)

            # === 块间：recurrent state 贡献 ===
            inter_out = torch.exp(log_cg).unsqueeze(-1) * torch.matmul(q_c, recurrent_state)
            # (B, H, C, 1) * (B, H, C, V) = (B, H, C, V)

            output_chunks.append(intra_out + inter_out)

            # === 更新 recurrent state 传给下一个 chunk ===
            # 直接计算 exp(log_cg_last - log_cg) 而非 exp(-log_cg) * exp(log_cg_last)
            # 因为 log_cg_last - log_cg <= 0，exp 不会溢出
            log_decay_to_last = log_cg[:, :, -1:] - log_cg  # (B, H, C)，始终 <= 0
            decay_weight = torch.exp(log_decay_to_last)      # (B, H, C)，(0, 1]
            v_beta_weighted = v_beta * decay_weight.unsqueeze(-1)  # (B, H, C, V)
            kv_sum = torch.matmul(k_c.transpose(-2, -1), v_beta_weighted)  # (B, H, D, V)

            cg_last = torch.exp(log_cg[:, :, -1:])  # (B, H, 1)
            recurrent_state = cg_last.unsqueeze(-1) * recurrent_state + kv_sum

        output = torch.cat(output_chunks, dim=2)  # (B, H, T, V)
        output = output.transpose(1, 2).contiguous()  # (B, T, H, V)
        return output


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


class MoEGate(nn.Module):
    """MoE门控"""
    def __init__(self, args):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        self.scoring_func = args.scoring_func
        self.alpha = args.aux_loss_alpha
        self.seq_aux = args.seq_aux
        self.norm_topk_prob = args.norm_topk_prob
        self.gating_dim = args.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'Unsupported scoring function: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        aux_loss = None
        if self.training and self.alpha > 0.0:
            if self.seq_aux:
                scores_for_seq_aux = scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx.view(bsz, -1),
                                torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device)
                                ).div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """MoE前馈网络"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.experts = nn.ModuleList([
            FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )
            for _ in range(args.n_routed_experts)
        ])
        self.gate = MoEGate(args)
        if args.n_shared_experts:
            self.shared_experts = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            x = x.repeat_interleave(self.args.num_experts_per_tok, dim=0)
            y = torch.empty_like(x)
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    y[mask] = expert(x[mask]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.args.n_shared_experts:
            y = y + self.shared_experts(identity)

        return y, aux_loss

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.args.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


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
        # ========== MoE 参数 ==========
        use_moe: bool = True,             # 是否使用 MoE（混合专家）FFN，False 则用普通 SwiGLU FFN
        n_routed_experts: int = 8,        # 路由专家总数（use_moe=True 时生效）
        num_experts_per_tok: int = 2,     # 每个 token 激活的专家数（Top-K）
        n_shared_experts: int = 1,        # 共享专家数（始终激活，不参与路由）
        scoring_func: str = 'softmax',    # 路由评分函数，可选 'softmax' 或 'sigmoid'
        aux_loss_alpha: float = 0.01,     # 负载均衡辅助损失系数
        seq_aux: bool = True,             # 是否在序列级别计算辅助损失
        norm_topk_prob: bool = True,      # 是否归一化 Top-K 专家概率权重
        # ========== 混合注意力架构 ==========
        use_linear_attn: bool = True,     # 是否启用混合注意力（False 则全部用标准 Attention）
        layer_types: List[str] = None,    # 每层的注意力类型列表，如 ["linear_attention","attention",...]
                                          # None 时自动生成交替模式（use_linear_attn=True）
                                          # 推荐每4层一个标准attention（如1B: 12线性+4标准）
        # ========== GatedDeltaNet 特定参数 ==========
        conv_kernel_size: int = 4,        # 因果卷积核大小，提供局部位置感知，替代位置编码
        linear_attn_chunk_size: int = 64, # 线性注意力块大小；须在训练前固定
        linear_attn_impl: str = 'simple',  # 'simple' 兼容旧权重；'gated_delta_rule' 为完整规则
        linear_attn_backend: str = 'auto',  # CPU uses reference; CUDA can require FLA explicitly
        attention_backend: str = 'sdpa',
        initialization_scheme: str = 'legacy',
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

        # MoE 混合专家
        self.use_moe = use_moe                      # 是否启用 MoE
        self.n_routed_experts = n_routed_experts    # 路由专家数
        self.num_experts_per_tok = num_experts_per_tok  # 每 token 激活专家数
        self.n_shared_experts = n_shared_experts    # 共享专家数
        self.scoring_func = scoring_func            # 路由评分函数
        self.aux_loss_alpha = aux_loss_alpha        # 辅助损失系数
        self.seq_aux = seq_aux                      # 序列级辅助损失
        self.norm_topk_prob = norm_topk_prob        # 归一化专家概率

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
        self.linear_attn_impl = linear_attn_impl
        self.linear_attn_backend = linear_attn_backend
        self.attention_backend = attention_backend
        self.initialization_scheme = initialization_scheme
        self.gradient_checkpointing = gradient_checkpointing  # 梯度检查点策略
        self.use_cache = False

        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if linear_attn_chunk_size < 1:
            raise ValueError("linear_attn_chunk_size must be positive")
        if linear_attn_impl not in {'simple', 'gated_delta_rule'}:
            raise ValueError("linear_attn_impl must be 'simple' or 'gated_delta_rule'")
        if linear_attn_backend not in {'auto', 'reference', 'fla'}:
            raise ValueError("linear_attn_backend must be 'auto', 'reference', or 'fla'")
        if attention_backend not in {'sdpa', 'flash_attn_4'}:
            raise ValueError("attention_backend must be 'sdpa' or 'flash_attn_4'")
        if initialization_scheme not in {'legacy', 'gdn_v3'}:
            raise ValueError("initialization_scheme must be 'legacy' or 'gdn_v3'")
        if linear_attn_backend == 'fla' and linear_attn_impl != 'gated_delta_rule':
            raise ValueError("linear_attn_backend='fla' requires linear_attn_impl='gated_delta_rule'")
        if n_kv_heads is not None and n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if linear_attn_heads is not None and n_heads % linear_attn_heads != 0:
            raise ValueError("n_heads must be divisible by linear_attn_heads")
        if len(self.layer_types) != n_layers:
            raise ValueError("layer_types must contain exactly n_layers entries")
        invalid_layer_types = set(self.layer_types) - {"attention", "linear_attention"}
        if invalid_layer_types:
            raise ValueError(f"Unsupported layer types: {sorted(invalid_layer_types)}")
        if use_moe and not 1 <= num_experts_per_tok <= n_routed_experts:
            raise ValueError("num_experts_per_tok must be between 1 and n_routed_experts")


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

        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )

    def _block_forward(self, x, pos_cis, kv_cache):
        """整块前向（attention + FFN），用于 gradient checkpointing"""
        attn_input = self.attention_norm(x)
        if self.use_pos_cis:
            h = x + self.attention(attn_input, pos_cis, kv_cache)
        else:
            h = x + self.attention(attn_input, None, kv_cache)

        ffn_input = self.ffn_norm(h)
        if isinstance(self.feed_forward, MOEFeedForward):
            ffn_out, aux_loss = self.feed_forward(ffn_input)
            out = h + ffn_out
            return out, aux_loss
        else:
            out = h + self.feed_forward(ffn_input)
            return out, None

    def forward(self, x, pos_cis=None, kv_cache=False):
        gc = self.args.gradient_checkpointing
        if gc == 'all' or (gc == 'linear_attn' and self.layer_type == "linear_attention"):
            return cp.checkpoint(self._block_forward, x, pos_cis, kv_cache,
                                 use_reentrant=False)
        return self._block_forward(x, pos_cis, kv_cache)


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

        if config.initialization_scheme == 'legacy':
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
        else:
            for layer in self.layers:
                projection = layer.attention.out_proj if layer.layer_type == 'linear_attention' else layer.attention.wo
                projection._mindlm_residual_projection = True
                for name, module in layer.feed_forward.named_modules():
                    if name == 'w2' or name.endswith('.w2'):
                        module._mindlm_residual_projection = True

        self.aux_loss = 0.0
        self.post_init()

    def _init_weights(self, module):
        if self.config.initialization_scheme == 'gdn_v3':
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # The embedding and output head share one parameter.
                if not getattr(module.weight, '_mindlm_v3_initialized', False):
                    std = 0.02 / math.sqrt(2 * self.config.n_layers) if getattr(module, '_mindlm_residual_projection', False) else 0.02
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    module.weight._mindlm_v3_initialized = True
                if getattr(module, 'bias', None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, GatedDeltaNet):
                with torch.no_grad():
                    amplitude = torch.empty_like(module.A_log).uniform_(0, 16).clamp_min_(torch.finfo(module.A_log.dtype).tiny)
                    module.A_log.copy_(amplitude.log())
                    dt = torch.exp(torch.empty_like(module.dt_bias).uniform_(math.log(0.001), math.log(0.1)))
                    module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        **kwargs,
    ):
        """Run a full causal-LM forward pass without a KV cache.

        ``labels`` follows the Transformers convention: positions marked ``-100``
        are ignored. ``tokens`` and ``targets`` remain as compatibility aliases for
        the project's older scripts.
        """
        if input_ids is None:
            input_ids = tokens
        if labels is None:
            labels = targets
        if input_ids is None:
            raise ValueError("input_ids is required")
        if labels is not None and not return_logits:
            raise ValueError("labels require return_logits=True")

        _bsz, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:
            raise ValueError(
                f"Input length {seqlen} exceeds max_seq_len={self.config.max_seq_len}."
            )
        h = self.tok_embeddings(input_ids)
        h = self.dropout(h)

        pos_cis = None
        if self.pos_cis is not None:
            pos_cis = self.pos_cis[:seqlen]

        total_aux_loss = 0.0

        for layer in self.layers:
            h, aux_loss = layer(h, pos_cis, False)
            if aux_loss is not None:
                total_aux_loss += aux_loss

        h = self.norm(h)

        logits = self.output(h) if return_logits else None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            if isinstance(total_aux_loss, torch.Tensor):
                loss = loss + total_aux_loss

        return MindLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            aux_loss=total_aux_loss if isinstance(total_aux_loss, torch.Tensor) else None,
            last_hidden_state=h if not return_logits else None,
        )

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
        **kwargs,
    ):
        """Generate tokens with a compact Transformers-compatible interface.

        MindLM currently recomputes the full context at each step because neither
        its standard attention nor DeltaNet path exposes a KV/state cache.
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

        generated = input_ids
        unfinished = torch.ones(generated.size(0), dtype=torch.bool, device=generated.device)
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.config.max_seq_len:
                break

            logits = self(generated).logits[:, -1, :]
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

        return generated
