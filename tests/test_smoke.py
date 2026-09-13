"""Small CPU smoke tests for the supported MindLM training contracts."""

import json
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

try:
    import numpy as np
    import pandas as pd
    import torch

    import modeling_mindlm
    from dataset import PackedPretrainDataset, SFTDataset
    from modeling_mindlm import GatedDeltaNet, MindLM, MindLMConfig, l2norm
    from training_utils import (
        extract_model_state,
        masked_language_model_loss,
        save_training_checkpoint,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and project dependencies are required")
class MindLMSmokeTest(unittest.TestCase):
    def make_config(self, use_moe=False):
        return MindLMConfig(
            dim=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            linear_attn_heads=2,
            vocab_size=32,
            max_seq_len=12,
            hidden_dim=32,
            multiple_of=8,
            use_moe=use_moe,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=1,
            layer_types=["attention", "linear_attention"],
        )

    def test_dense_and_moe_forward(self):
        input_ids = torch.randint(0, 32, (2, 6))

        dense = MindLM(self.make_config()).eval()
        dense_output = dense(input_ids=input_ids)
        self.assertEqual(tuple(dense_output.logits.shape), (2, 6, 32))
        self.assertIsNone(dense_output.aux_loss)

        moe = MindLM(self.make_config(use_moe=True)).train()
        moe_output = moe(input_ids=input_ids)
        self.assertEqual(tuple(moe_output.logits.shape), (2, 6, 32))
        self.assertIsNotNone(moe_output.aux_loss)
        self.assertGreaterEqual(moe_output.aux_loss.item(), 0.0)

        full_config = self.make_config()
        full_config.linear_attn_impl = "gated_delta_rule"
        full = MindLM(full_config).eval()
        full_output = full(input_ids=input_ids)
        self.assertEqual(tuple(full_output.logits.shape), (2, 6, 32))

    def test_masked_loss_and_context_limited_generation(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        targets = torch.tensor([[0, 1]])
        mask = torch.tensor([[1, 0]])
        expected = torch.nn.functional.cross_entropy(logits[:, :1].reshape(-1, 2), targets[:, :1].reshape(-1))
        self.assertTrue(torch.allclose(masked_language_model_loss(logits, targets, mask), expected))

        model = MindLM(self.make_config()).eval()
        full_prompt = torch.randint(0, 32, (2, model.config.max_seq_len))
        self.assertTrue(torch.equal(model.generate(full_prompt, max_new_tokens=4), full_prompt))

    def test_checkpoint_round_trip(self):
        model = MindLM(self.make_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = f"{directory}/checkpoint.pt"
            save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                model.config,
                epoch=2,
                step=7,
                epoch_complete=False,
                training_stage="pretrain",
            )
            checkpoint = torch.load(checkpoint_path, weights_only=True)

        self.assertEqual(checkpoint["training_stage"], "pretrain")
        self.assertEqual(checkpoint["step"], 7)
        self.assertEqual(set(extract_model_state(checkpoint)), set(model.state_dict()))

    def test_sft_dataset_keeps_the_final_answer_when_context_is_long(self):
        class FakeTokenizer:
            pad_token_id = 0

            def __call__(self, text, add_special_tokens=True):
                if text == "<s>assistant\n":
                    token_ids = [1, 9]
                else:
                    token_ids = [3, 4, 5, 1, 9, 6, 7, 8, 2]
                return SimpleNamespace(data={"input_ids": token_ids})

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
                return "formatted"

        dataframe = pd.DataFrame([{"history": "[]", "q": "question", "a": "answer"}])
        dataset = SFTDataset(dataframe, FakeTokenizer(), max_length=6)
        _input_ids, _targets, loss_mask = dataset[0]
        self.assertEqual(tuple(loss_mask.shape), (5,))
        self.assertEqual(loss_mask.tolist(), [0, 1, 1, 1, 1])

    def test_packed_pretrain_dataset_reads_padding_free_token_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = f"{directory}/packed"
            records = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.uint32)
            records.tofile(f"{prefix}.bin")
            with open(f"{prefix}.json", "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "format": "mindlm_packed_pretrain_v1",
                        "dtype": "uint32",
                        "sequence_length": 4,
                        "tokens_per_record": 5,
                        "num_sequences": 2,
                        "tokenizer_vocab_size": 32,
                    },
                    file,
                )

            dataset = PackedPretrainDataset(prefix, max_length=4, tokenizer_vocab_size=32)
            input_ids, targets, loss_mask = dataset[1]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(input_ids.tolist(), [6, 7, 8, 9])
        self.assertEqual(targets.tolist(), [7, 8, 9, 10])
        self.assertEqual(loss_mask.tolist(), [1, 1, 1, 1])

    def test_complete_gated_delta_rule_matches_token_recurrence_and_backpropagates(self):
        config = self.make_config()
        config.linear_attn_impl = "gated_delta_rule"
        module = GatedDeltaNet(config).double()
        batch, length = 2, 5
        heads, dim = module.num_v_heads, module.head_v_dim
        query = torch.randn(batch, length, heads, dim, dtype=torch.float64, requires_grad=True)
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn(batch, length, heads, dim, dtype=torch.float64, requires_grad=True)
        g = -torch.rand(batch, length, heads, dtype=torch.float64)
        beta = torch.rand(batch, length, heads, dtype=torch.float64)

        actual = module.gated_delta_rule_attention(query, key, value, g, beta, chunk_size=2)

        q_ref = l2norm(query.transpose(1, 2), dim=-1) * (dim ** -0.5)
        k_ref = l2norm(key.transpose(1, 2), dim=-1)
        v_ref = value.transpose(1, 2)
        g_ref = g.transpose(1, 2)
        b_ref = beta.transpose(1, 2)
        state = torch.zeros(batch, heads, dim, dim, dtype=torch.float64)
        expected = []
        for t in range(length):
            state = state * torch.exp(g_ref[:, :, t]).unsqueeze(-1).unsqueeze(-1)
            predicted = torch.einsum("bhd,bhdv->bhv", k_ref[:, :, t], state)
            residual = b_ref[:, :, t].unsqueeze(-1) * (v_ref[:, :, t] - predicted)
            state = state + k_ref[:, :, t].unsqueeze(-1) * residual.unsqueeze(-2)
            expected.append(torch.einsum("bhd,bhdv->bhv", q_ref[:, :, t], state))
        expected = torch.stack(expected, dim=2).transpose(1, 2)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(key.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())

        first, state = module.gated_delta_rule_attention(
            query.detach()[:, :3], key.detach()[:, :3], value.detach()[:, :3],
            g[:, :3], beta[:, :3], chunk_size=2, return_state=True,
        )
        second = module.gated_delta_rule_attention(
            query.detach()[:, 3:], key.detach()[:, 3:], value.detach()[:, 3:],
            g[:, 3:], beta[:, 3:], initial_state=state, chunk_size=2,
        )
        torch.testing.assert_close(torch.cat((first, second), dim=1), actual.detach())

    def test_fla_adapter_receives_the_configured_chunk_size(self):
        config = self.make_config()
        config.linear_attn_chunk_size = 32
        module = GatedDeltaNet(config)
        shape = (1, 3, module.num_v_heads, module.head_v_dim)
        query = torch.randn(shape)
        key = torch.randn(shape)
        value = torch.randn(shape)
        g = -torch.rand(shape[:-1])
        beta = torch.rand(shape[:-1])
        received = {}

        def fake_fla(q, k, v, gate, beta_value, **kwargs):
            received.update(kwargs)
            return q, None

        with mock.patch.object(modeling_mindlm, "_fla_chunk_gdr", fake_fla):
            output = module.gated_delta_rule_fla(query, key, value, g, beta)

        self.assertEqual(tuple(output.shape), shape)
        self.assertEqual(received["chunk_size"], 32)
        self.assertTrue(received["use_qk_l2norm_in_kernel"])


if __name__ == "__main__":
    unittest.main()
