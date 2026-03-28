"""
MindLM: MiniMind with Linear Attention + MoE
基于MiniMind架构，集成Gated DeltaNet Linear Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


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


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet Linear Attention
    简化版实现，适配MiniMind架构
    """
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.dim
        self.num_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.num_k_heads = getattr(args, 'n_kv_heads', None) or args.n_heads
        self.num_v_heads = getattr(args, 'n_kv_heads', None) or args.n_heads
        self.head_k_dim = self.head_dim
        self.head_v_dim = self.head_dim

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)

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

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, pos_cis=None, kv_cache=False):
        batch_size, seq_len, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
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

        core_attn_out = self.simple_gated_delta_attention(query, key, value, g, beta)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.gated_norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        output = self.resid_dropout(output)
        return output

    def gated_norm(self, x, gate):
        """门控RMSNorm"""
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        x = x * F.silu(gate.float())
        return x.to(input_dtype)

    def simple_gated_delta_attention(self, query, key, value, g, beta):
        """简化的Gated Delta Attention"""
        batch_size, seq_len, num_heads, head_dim = query.shape
        v_dim = value.shape[-1]

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        beta = beta.transpose(1, 2).contiguous()
        g = g.transpose(1, 2).contiguous()

        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        output = []
        recurrent_state = torch.zeros(batch_size, num_heads, head_dim, v_dim,
                                     device=query.device, dtype=query.dtype)

        for t in range(seq_len):
            q_t = query[:, :, t:t+1, :]
            k_t = key[:, :, t:t+1, :]
            v_t = value[:, :, t:t+1, :]
            beta_t = beta[:, :, t:t+1]
            g_t = g[:, :, t:t+1].exp()

            recurrent_state = recurrent_state * g_t.unsqueeze(-1)
            kv = torch.matmul(k_t.transpose(-2, -1), v_t * beta_t.unsqueeze(-1))
            recurrent_state = recurrent_state + kv

            out_t = torch.matmul(q_t, recurrent_state)
            output.append(out_t)

        output = torch.cat(output, dim=2)
        output = output.transpose(1, 2).contiguous()
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
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    y[mask] = expert(x[mask])
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
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = None,
        vocab_size: int = 10000,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        hidden_dim: int = None,
        multiple_of: int = 256,
        # MoE参数
        use_moe: bool = True,
        n_routed_experts: int = 8,
        num_experts_per_tok: int = 2,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        # Linear Attention参数
        use_linear_attn: bool = True,
        layer_types: List[str] = None,
        conv_kernel_size: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of

        # MoE
        self.use_moe = use_moe
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        # Linear Attention
        self.use_linear_attn = use_linear_attn
        if layer_types is None:
            if use_linear_attn:
                self.layer_types = ["linear_attention" if i % 2 == 1 else "attention"
                                    for i in range(n_layers)]
            else:
                self.layer_types = ["attention"] * n_layers
        else:
            self.layer_types = layer_types
        self.conv_kernel_size = conv_kernel_size


class TransformerBlock(nn.Module):
    """MindLM Transformer块"""
    def __init__(self, layer_id: int, args: MindLMConfig):
        super().__init__()
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

    def forward(self, x, pos_cis=None, kv_cache=False):
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


class MindLM(PreTrainedModel):
    """MindLM主模型"""
    config_class = MindLMConfig

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

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        self.aux_loss = 0.0

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens=None, targets=None, **kwargs):
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
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

        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                  ignore_index=0)
            if total_aux_loss > 0:
                loss = loss + total_aux_loss
        else:
            logits = self.output(h[:, [-1], :])
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @torch.inference_mode()
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=8):
        for _ in range(max_new_tokens):
            inference_res = self(idx)
            logits = inference_res.logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == eos:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            yield idx


if __name__ == "__main__":
    config = MindLMConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=10000,
        use_moe=True,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        aux_loss_alpha=0.01,
        use_linear_attn=True,
    )
    config = MindLMConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=10000,
        use_linear_attn=True        
    )

    model = MindLM(config)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    output = model(tokens)
    print(f"输出logits形状: {output.logits.shape}")
    print("✅ 测试通过!")
