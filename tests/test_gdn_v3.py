"""Numerical and configuration contracts for the opt-in GDN V3 model."""

import copy
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import modeling_mindlm
from config import load_config
from modeling_mindlm import Attention, GatedDeltaNet, MindLM, MindLMConfig, l2norm
from training_utils import masked_lm_head_loss


def small_config(**overrides):
    values = dict(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, linear_attn_heads=2,
                  vocab_size=67, max_seq_len=16, hidden_dim=64, multiple_of=8,
                  use_moe=False, layer_types=["attention", "linear_attention"],
                  linear_attn_backend="fla", attention_backend="flash_attn_4")
    values.update(overrides)
    return MindLMConfig(**values)


class GDNV3ConfigAndInitializationTest(unittest.TestCase):
    def test_named_config_and_round_trip_preserve_explicit_v3_choices(self):
        config = MindLMConfig(**load_config("mindlm_0.2b_gdn"))
        self.assertEqual((config.dim, config.n_layers, config.n_heads, config.n_kv_heads), (768, 16, 12, 3))
        self.assertEqual((config.linear_attn_heads, config.dim // config.n_heads), (3, 64))
        self.assertEqual((config.vocab_size, config.max_seq_len, config.dropout), (151669, 4096, 0))
        self.assertEqual(config.layer_types.count("linear_attention"), 12)
        self.assertEqual(config.layer_types.count("attention"), 4)
        expected = dict(linear_attn_backend="fla",
                        attention_backend="flash_attn_4",
                        gradient_checkpointing="all", conv_kernel_size=4)
        restored = MindLMConfig.from_dict(config.to_dict())
        for name, value in expected.items():
            self.assertEqual(getattr(config, name), value)
            self.assertEqual(getattr(restored, name), value)
        baseline = MindLMConfig()
        self.assertEqual((baseline.linear_attn_backend, baseline.attention_backend),
                         ("fla", "flash_attn_4"))

    def test_invalid_backend_choices_are_rejected(self):
        for name in ("linear_attn_backend", "attention_backend"):
            with self.subTest(field=name), self.assertRaises(ValueError):
                small_config(**{name: "unknown"})

    def test_final_initialization_scales_residuals_once_and_exempts_gates_from_decay(self):
        torch.manual_seed(404)
        config = small_config(dim=128, n_heads=8, n_kv_heads=2, linear_attn_heads=8,
                              vocab_size=1024, hidden_dim=256, n_layers=4,
                              layer_types=["attention", "linear_attention"] * 2)
        calls = {}
        original_normal = torch.nn.init.normal_

        def record_normal(parameter, mean=0, std=1, **kwargs):
            if std != 1:
                calls.setdefault(id(parameter), []).append(std)
            return original_normal(parameter, mean=mean, std=std, **kwargs)

        with mock.patch.object(torch.nn.init, "normal_", side_effect=record_normal):
            model = MindLM(config)
        expected_residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        residual_ids = set()
        for layer in model.layers:
            projection = layer.attention.wo if layer.layer_type == "attention" else layer.attention.out_proj
            residual_ids.update((id(projection.weight), id(layer.feed_forward.w2.weight)))
            if isinstance(layer.attention, GatedDeltaNet):
                amplitude = layer.attention.A_log.exp()
                dt = F.softplus(layer.attention.dt_bias)
                self.assertTrue(torch.all((amplitude > 0) & (amplitude < 16)))
                self.assertTrue(torch.all((dt >= 0.001) & (dt <= 0.1)))
                self.assertTrue(layer.attention.A_log._no_weight_decay)
                self.assertTrue(layer.attention.dt_bias._no_weight_decay)
                self.assertEqual(layer.attention.conv1d.kernel_size, (4,))
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                std = expected_residual_std if id(module.weight) in residual_ids else 0.02
                self.assertEqual(calls[id(module.weight)], [std])
                sampling_tolerance = 5 * std / math.sqrt(2 * (module.weight.numel() - 1))
                self.assertAlmostEqual(float(module.weight.detach().std()), std, delta=sampling_tolerance)
        self.assertIs(model.output.weight, model.tok_embeddings.weight)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        model.post_init()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


class GDNV3NumericsTest(unittest.TestCase):
    def test_delta_correction_overwrites_repeated_association(self):
        module = GatedDeltaNet(small_config())
        q = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
        value = torch.ones(1, 2, 1, 1)
        gate = torch.zeros(1, 2, 1)
        beta = torch.ones_like(gate)
        actual = module.gated_delta_rule_attention(q, q, value, gate, beta)
        torch.testing.assert_close(actual.flatten(), torch.tensor([1, 1]) / math.sqrt(2), rtol=0, atol=2e-6)

    def test_full_gdr_output_state_and_all_input_gradients_match_affine_oracle(self):
        torch.manual_seed(405)
        module = GatedDeltaNet(small_config())
        shapes = [(1, 5, 2, 3), (1, 5, 2, 3), (1, 5, 2, 2), (1, 5, 2), (1, 5, 2), (1, 2, 3, 2)]
        source = [torch.randn(shape) for shape in shapes]
        source[3] = -source[3].abs() - 0.1
        source[4] = source[4].sigmoid()
        actual_inputs = [value.clone().requires_grad_() for value in source]
        references = [value.double().requires_grad_() for value in source]
        actual, actual_state = module.gated_delta_rule_attention(
            *actual_inputs[:5], chunk_size=2, initial_state=actual_inputs[5], return_state=True,
        )
        q, k, v, g, beta, state = references
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        expected = []
        for t in range(q.shape[1]):
            key = k[:, t].unsqueeze(-1)
            transition = g[:, t].exp()[..., None, None] * (
                torch.eye(q.shape[-1], dtype=torch.double) - beta[:, t, :, None, None] * (key @ key.transpose(-2, -1)))
            state = transition @ state + beta[:, t, :, None, None] * (key @ v[:, t].unsqueeze(-2))
            expected.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2) / math.sqrt(q.shape[-1]))
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_state.double(), state, rtol=1e-5, atol=1e-6)
        output_weights = torch.randn_like(actual)
        state_weights = torch.randn_like(actual_state)
        actual_grads = torch.autograd.grad((actual * output_weights).sum() + (actual_state * state_weights).sum(), actual_inputs)
        reference_grads = torch.autograd.grad((expected * output_weights).sum() + (state * state_weights).sum(), references)
        for name, actual_grad, reference in zip(("q", "k", "v", "g", "beta", "initial_state"), actual_grads, reference_grads):
            with self.subTest(gradient=name):
                torch.testing.assert_close(actual_grad.double(), reference, rtol=1e-4, atol=2e-6)

    def test_bfloat16_full_rule_normalizes_in_fp32_before_cast(self):
        torch.manual_seed(406)
        module = GatedDeltaNet(small_config())
        q = torch.randn(1, 1, 2, 64).bfloat16()
        k = torch.randn_like(q)
        value = torch.randn(1, 1, 2, 3).bfloat16()
        beta = torch.ones(1, 1, 2)
        actual = module.gated_delta_rule_attention(q, k, value, torch.zeros_like(beta), beta)
        q_norm = l2norm(q.float()).bfloat16().float()
        k_norm = l2norm(k.float()).bfloat16().float()
        expected = ((q_norm * k_norm).sum(-1, keepdim=True) * value.float() / 8).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hidden_only_matches_logits_and_gradients_without_executing_head(self):
        torch.manual_seed(407)
        model = MindLM(small_config(gradient_checkpointing="all")).train()
        input_ids = torch.randint(67, (2, 7))
        targets = torch.randint(67, (2, 7))
        mask = torch.ones_like(targets)
        mask[:, :3] = 0
        regular = model(input_ids=input_ids)
        expected = F.cross_entropy(regular.logits[mask.bool()], targets[mask.bool()])
        expected.backward()
        reference_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        with mock.patch.object(model.output, "forward", side_effect=AssertionError("head must not execute")):
            hidden_only = model(input_ids=input_ids, return_logits=False)
        self.assertIsNone(hidden_only.logits)
        torch.testing.assert_close(model.output(hidden_only.last_hidden_state), regular.logits)
        actual = masked_lm_head_loss(model.output, hidden_only.last_hidden_state, targets, mask, chunk_tokens=2)
        actual.backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=3e-5, atol=2e-6)
        with self.assertRaisesRegex(ValueError, "labels require"):
            model(input_ids=input_ids, labels=targets, return_logits=False)


class GDNV3BackendTest(unittest.TestCase):
    def test_cpu_uses_reference_and_sdpa_even_with_explicit_cuda_backends(self):
        model = MindLM(small_config(linear_attn_backend="fla", attention_backend="flash_attn_4")).eval()
        with mock.patch.object(modeling_mindlm, "_fla_chunk_gdr", side_effect=AssertionError("no FLA on CPU")), \
                mock.patch.object(modeling_mindlm, "_flash_attn_4", side_effect=AssertionError("no FA4 on CPU")):
            actual = model(input_ids=torch.randint(67, (2, 7))).logits
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(model.layers[1].attention._select_backend(torch.device("cpu")), "reference")

    def test_explicit_fla_requires_dependency_on_cuda(self):
        module = GatedDeltaNet(small_config(linear_attn_backend="fla"))
        with mock.patch.object(modeling_mindlm, "_fla_chunk_gdr", None):
            with self.assertRaisesRegex(RuntimeError, "requires flash-linear-attention"):
                module._select_backend(torch.device("cuda"))
        with mock.patch.object(modeling_mindlm, "_fla_chunk_gdr", mock.Mock()):
            self.assertEqual(module._select_backend(torch.device("cuda")), "fla")
        self.assertEqual(module._select_backend(torch.device("cpu")), "reference")

    def test_fa4_forward_passes_native_gqa_and_unpacks_tuple(self):
        config = small_config(dim=48, n_heads=12, n_kv_heads=3, attention_backend="flash_attn_4")
        module = Attention(config).bfloat16().eval()
        x = torch.randn(2, 7, 48).bfloat16()
        observed = {}

        def fake_fa4(q, k, v, **kwargs):
            observed.update(q=tuple(q.shape), k=tuple(k.shape), v=tuple(v.shape), **kwargs)
            out = F.scaled_dot_product_attention(q.transpose(1, 2),
                k.repeat_interleave(4, dim=2).transpose(1, 2),
                v.repeat_interleave(4, dim=2).transpose(1, 2), is_causal=True)
            return out.transpose(1, 2), torch.empty(0)

        with mock.patch.object(modeling_mindlm, "_flash_attn_4", side_effect=fake_fa4), \
                mock.patch.object(torch.Tensor, "is_cuda", new_callable=mock.PropertyMock, return_value=True):
            actual = module(x, None)
        self.assertEqual(observed, {"q": (2, 7, 12, 4), "k": (2, 7, 3, 4), "v": (2, 7, 3, 4), "causal": True})
        module.attention_backend = "sdpa"
        torch.testing.assert_close(actual, module(x, None))

    def test_fa4_rejects_missing_dependency_and_training_dropout(self):
        module = Attention(small_config(attention_backend="flash_attn_4", dropout=0.1)).train()
        q = torch.ones(1, 2, 4, 8, dtype=torch.bfloat16)
        with mock.patch.object(modeling_mindlm, "_flash_attn_4", None):
            with self.assertRaisesRegex(RuntimeError, "requires flash_attn.cute"):
                module._flash_attention(q, q, q)
        with mock.patch.object(modeling_mindlm, "_flash_attn_4", mock.Mock()):
            with self.assertRaisesRegex(ValueError, "nonzero attention dropout"):
                module._flash_attention(q, q, q)


if __name__ == "__main__":
    unittest.main()
