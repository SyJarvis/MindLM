"""Small CPU smoke tests for the supported MindLM training contracts."""

import tempfile
import unittest
from types import SimpleNamespace

try:
    import pandas as pd
    import torch

    from dataset import SFTDataset
    from modeling_mindlm import MindLM, MindLMConfig
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

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "formatted"

        dataframe = pd.DataFrame([{"history": "[]", "q": "question", "a": "answer"}])
        dataset = SFTDataset(dataframe, FakeTokenizer(), max_length=6)
        _input_ids, _targets, loss_mask = dataset[0]
        self.assertEqual(tuple(loss_mask.shape), (5,))
        self.assertEqual(loss_mask.tolist(), [0, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
