"""CPU regression tests for grouped SFT correctness and resumability."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import full_sft_packed as trainer
import training_utils
from modeling_mindlm import MindLM, MindLMConfig


class GroupedDataTest(unittest.TestCase):
    def make_dataset(self, directory, tokens, records):
        bin_path = Path(directory) / "tokens.bin"
        meta_path = Path(directory) / "rows.jsonl"
        np.array(tokens, dtype=np.uint32).tofile(bin_path)
        meta_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return trainer.GroupedSFTDataset(bin_path, meta_path)

    def test_answer_mask_includes_first_and_last_answer_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(
                directory,
                [10, 11, 12, 13, 20, 21, 22, 23, 24],
                [{"off": 0, "n": 4, "ans": 1}, {"off": 4, "n": 5, "ans": 3}],
            )
            x, y, mask = dataset[0]
            self.assertEqual(x.tolist(), [10, 11, 12])
            self.assertEqual(y[mask.bool()].tolist(), [13])
            self.assertEqual(dataset[1][1][dataset[1][2].bool()].tolist(), [22, 23, 24])
            batch_x, batch_y, batch_mask = trainer.collate_variable([dataset[0], dataset[1]])
            self.assertEqual(batch_x.shape, (2, 4))
            self.assertEqual(batch_mask.tolist(), [[0, 0, 1, 0], [0, 1, 1, 1]])
            self.assertEqual(batch_y[batch_mask.bool()].tolist(), [13, 22, 23, 24])

    def test_invalid_metadata_is_rejected_before_training(self):
        invalid_records = [
            {"off": -1, "n": 4, "ans": 1},
            {"off": 0.5, "n": 4, "ans": 1},
            {"off": 0, "n": 1, "ans": 1},
            {"off": 0, "n": 5, "ans": 1},
            {"off": 2, "n": 4, "ans": 1},
            {"off": 0, "n": 4, "ans": 0},
            {"off": 0, "n": 4, "ans": -3},
            {"off": 0, "n": 4, "ans": 4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for record in invalid_records:
                with self.subTest(record=record), self.assertRaises(ValueError):
                    self.make_dataset(directory, [1, 2, 3, 4], [record])

    def test_sampler_bounds_padded_inputs_and_preserves_every_record(self):
        lengths = [10, 10, 80, 2, 9, 50, 51, 3, 16, 21, 29]
        sampler = trainer.LengthGroupedBatchSampler(lengths, token_budget=100, mega_batch=512, max_count=4)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            self.assertEqual(sorted(index for batch in batches for index in batch), list(range(len(lengths))))
            for batch in batches:
                self.assertLessEqual(len(batch) * max(lengths[index] - 1 for index in batch), 100)
                self.assertLessEqual(len(batch), 4)
            sampler.set_epoch(epoch)
            self.assertEqual(list(sampler), batches)

    def test_single_record_at_budget_is_allowed_and_over_budget_is_rejected(self):
        self.assertEqual(list(trainer.LengthGroupedBatchSampler([101], 100, 512)), [[0]])
        with self.assertRaises(ValueError):
            trainer.LengthGroupedBatchSampler([102], 100, 512)


class MaskedLossTest(unittest.TestCase):
    def test_chunked_ce_matches_full_fp32_loss_and_gradients(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 5, 17, requires_grad=True)
        targets = torch.randint(17, (2, 5))
        mask = torch.tensor([[0, 1, 0.25, 2, 0], [1, 0, 1, 0.5, 1]])
        reference_logits = logits.detach().clone().requires_grad_()
        expected = (F.cross_entropy(reference_logits.flatten(0, 1), targets.flatten(), reduction="none")
                    * mask.flatten()).sum() / mask.sum()
        actual, count = training_utils.chunked_masked_ce_loss(logits, targets, mask, chunk_tokens=2)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(count.item(), mask.sum().item())
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, reference_logits.grad)

    def test_bfloat16_mask_count_is_not_rounded(self):
        logits = torch.zeros(1, 257, 3, dtype=torch.bfloat16, requires_grad=True)
        targets = torch.zeros(1, 257, dtype=torch.long)
        mask = torch.ones(1, 257, dtype=torch.bfloat16)
        actual, count = training_utils.chunked_masked_ce_loss(logits, targets, mask, chunk_tokens=31)
        self.assertEqual(count.item(), 257)
        torch.testing.assert_close(actual, torch.tensor(3.0).log())
        actual.backward()
        expected_logits = logits.detach().float().requires_grad_()
        F.cross_entropy(expected_logits.flatten(0, 1), targets.flatten()).backward()
        torch.testing.assert_close(logits.grad.float(), expected_logits.grad, rtol=0.008, atol=1e-5)

    def test_zero_mask_returns_differentiable_zero(self):
        logits = torch.randn(2, 3, 7, requires_grad=True)
        loss, count = training_utils.chunked_masked_ce_loss(
            logits, torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3), chunk_tokens=2,
        )
        self.assertEqual(count.item(), 0)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_ce_forward_does_not_retain_fp32_vocab_buffers(self):
        logits = torch.randn(2, 7, 31, dtype=torch.bfloat16, requires_grad=True)
        saved = []

        def pack(tensor):
            saved.append((tuple(tensor.shape), tensor.dtype))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss, _ = training_utils.chunked_masked_ce_loss(
                logits, torch.zeros(2, 7, dtype=torch.long), torch.ones(2, 7), chunk_tokens=3,
            )
        self.assertFalse([
            shape for shape, dtype in saved if len(shape) >= 2 and shape[-1] == 31 and dtype == torch.float32
        ], saved)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_lm_head_chunks_match_full_projection_loss_and_gradients(self):
        torch.manual_seed(8)
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        head = torch.nn.Linear(8, 17, bias=False)
        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_head = copy.deepcopy(head)
        targets = torch.randint(17, (2, 5))
        mask = torch.tensor([[0, 1, 0.25, 2, 0], [1, 0, 1, 0.5, 1]])
        expected = (F.cross_entropy(reference_head(reference_hidden).flatten(0, 1), targets.flatten(), reduction="none")
                    * mask.flatten()).sum() / mask.sum()
        actual = training_utils.masked_lm_head_loss(head, hidden, targets, mask, chunk_tokens=2)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        torch.testing.assert_close(head.weight.grad, reference_head.weight.grad)

    def test_lm_head_uses_exact_bfloat16_mask_count(self):
        hidden = torch.ones(1, 257, 2, requires_grad=True)
        head = torch.nn.Linear(2, 3, bias=False)
        torch.nn.init.zeros_(head.weight)
        loss = training_utils.masked_lm_head_loss(
            head, hidden, torch.zeros(1, 257, dtype=torch.long),
            torch.ones(1, 257, dtype=torch.bfloat16), chunk_tokens=31,
        )
        torch.testing.assert_close(loss, torch.tensor(3.0).log())
        loss.backward()
        torch.testing.assert_close(head.weight.grad, torch.tensor([[-2 / 3, -2 / 3], [1 / 3, 1 / 3], [1 / 3, 1 / 3]]))

    def test_zero_lm_head_mask_has_zero_hidden_and_weight_gradients(self):
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        head = torch.nn.Linear(8, 17, bias=False)
        loss = training_utils.masked_lm_head_loss(
            head, hidden, torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3), chunk_tokens=2,
        )
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(hidden.grad.abs().sum().item(), 0)
        self.assertTrue(head.weight.grad is None or head.weight.grad.abs().sum().item() == 0)

    def test_lm_head_forward_does_not_retain_vocab_sized_activations(self):
        hidden = torch.randn(2, 7, 8, requires_grad=True)
        head = torch.nn.Linear(8, 31, bias=False)
        saved = []

        def pack(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = training_utils.masked_lm_head_loss(
                head, hidden, torch.zeros(2, 7, dtype=torch.long), torch.ones(2, 7), chunk_tokens=3,
            )
        self.assertFalse([shape for shape in saved if len(shape) >= 2 and shape[-1] == 31], saved)
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(head.weight.grad).all())


class GroupedTrainingTest(unittest.TestCase):
    def make_config(self, dropout=0):
        return MindLMConfig(
            dim=16, n_layers=2, n_heads=4, n_kv_heads=2, linear_attn_heads=2,
            vocab_size=32, max_seq_len=16, hidden_dim=32, multiple_of=8,
            use_moe=False, layer_types=["attention", "linear_attention"], dropout=dropout,
        )

    def make_inputs(self, directory, config):
        rows = [(4, 1), (5, 3), (7, 2), (8, 4), (6, 1)]
        tokens, records = [], []
        for index, (length, answer) in enumerate(rows):
            records.append({"off": len(tokens), "n": length, "ans": answer})
            tokens.extend([index + 1] + [((index + position) % 20) + 6 for position in range(length - 1)])
        bin_path, meta_path = Path(directory) / "tokens.bin", Path(directory) / "rows.jsonl"
        np.array(tokens, dtype=np.uint32).tofile(bin_path)
        meta_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        torch.manual_seed(93)
        initial_path = Path(directory) / "initial.pt"
        torch.save({"model": MindLM(config).state_dict(), "config": config.to_dict(), "training_stage": "pretrain"}, initial_path)
        return bin_path, meta_path, initial_path

    def run_training(self, directory, config, inputs, resume=None, limit=0, epochs=2, accumulation=3,
                     weights_only=False, extra_args=()):
        bin_path, meta_path, initial_path = inputs
        output_dir = Path(directory)
        output_dir.mkdir()
        argv = [
            "full_sft_packed.py", "--bin", str(bin_path), "--meta", str(meta_path),
            "--resume_from", str(resume or initial_path), "--out_dir", str(output_dir),
            "--device", "cpu", "--num_workers", "0", "--epochs", str(epochs),
            "--max_batch_count", "1", "--token_budget", "100", "--mega_batch", "3",
            "--accumulation_steps", str(accumulation), "--warmup_iters", "0",
            "--learning_rate", "0.003", "--weight_decay", "0.1", "--grad_clip", "1000000",
            "--save_interval", "1", "--log_interval", "1", "--limit_steps", str(limit),
        ]
        if resume is None or weights_only:
            argv.append("--resume_weights_only")
        argv.extend(extra_args)
        seen, updates, models = [], [], []
        original_step = torch.optim.AdamW.step

        def make_model(model_config):
            model = MindLM(model_config)
            model.tok_embeddings.register_forward_pre_hook(
                lambda _model, args: seen.append(args[0].detach().clone()),
            )
            models.append(model)
            return model

        def step(optimizer, *args, **kwargs):
            model = models[-1]
            updates.append({
                "model": copy.deepcopy(model.state_dict()),
                "grads": {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                          if parameter.grad is not None},
                "micro_count": len(seen),
                "lr": optimizer.param_groups[0]["lr"],
            })
            return original_step(optimizer, *args, **kwargs)

        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=object()), \
                mock.patch.object(trainer, "build_model_config", side_effect=lambda *_: copy.deepcopy(config)), \
                mock.patch.object(trainer, "MindLM", side_effect=make_model), \
                mock.patch.object(torch.optim.AdamW, "step", new=step), \
                mock.patch.object(torch, "autocast", side_effect=lambda **_: contextlib.nullcontext()), \
                contextlib.redirect_stdout(output):
            trainer.main()
        checkpoint_path = output_dir / "mindlm_sft_grouped_latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return checkpoint_path, checkpoint, seen, updates, output.getvalue()

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_nested_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def test_full_and_tail_accumulations_match_all_supervised_tokens(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(directory, config)
            _, checkpoint, seen, updates, _ = self.run_training(
                Path(directory) / "train", config, inputs, epochs=1,
            )
            dataset = trainer.GroupedSFTDataset(*inputs[:2])
            self.assertEqual([update["micro_count"] for update in updates], [3, 5])
            first_micro = 0
            for update in updates:
                reference = MindLM(copy.deepcopy(config)).train()
                reference.load_state_dict(update["model"])
                supervised_logits, supervised_targets = [], []
                for input_ids in seen[first_micro:update["micro_count"]]:
                    index = int(input_ids[0, 0]) - 1
                    _, targets, mask = dataset[index]
                    logits = reference(input_ids=input_ids).logits[0]
                    supervised_logits.append(logits[mask.bool()])
                    supervised_targets.append(targets[mask.bool()])
                F.cross_entropy(torch.cat(supervised_logits), torch.cat(supervised_targets)).backward()
                for name, parameter in reference.named_parameters():
                    torch.testing.assert_close(update["grads"][name], parameter.grad, rtol=2e-5, atol=2e-6)
                first_micro = update["micro_count"]
            self.assertTrue(checkpoint["epoch_complete"])
            self.assertEqual(checkpoint["step"], 2)

    def test_resume_matches_uninterrupted_training_including_dropout_and_sampling(self):
        config = self.make_config(dropout=0.2)
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(directory, config)
            torch.manual_seed(712)
            _, complete, all_seen, all_updates, _ = self.run_training(Path(directory) / "complete", config, inputs)
            for stop, epoch_complete in [(1, False), (2, True)]:
                with self.subTest(stop=stop):
                    torch.manual_seed(712)
                    path, partial, first_seen, first_updates, _ = self.run_training(
                        Path(directory) / f"partial{stop}", config, inputs, limit=stop,
                    )
                    self.assertEqual(partial["epoch_complete"], epoch_complete)
                    self.assertEqual(partial["step"], stop)
                    torch.manual_seed(9999)
                    _, resumed, rest_seen, rest_updates, _ = self.run_training(
                        Path(directory) / f"resumed{stop}", config, inputs, resume=path,
                    )
                    self.assert_nested_equal(first_seen + rest_seen, all_seen)
                    self.assertEqual([u["lr"] for u in first_updates + rest_updates], [u["lr"] for u in all_updates])
                    for key in ("model", "optimizer", "scaler", "epoch", "step", "epoch_complete"):
                        self.assert_nested_equal(resumed[key], complete[key])

    def test_full_resume_rejects_changes_to_batching_seed_and_learning_rate(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(directory, config)
            path, _, _, _, _ = self.run_training(Path(directory) / "partial", config, inputs, limit=1)
            for name, value in [("token_budget", "101"), ("seed", "1338"), ("learning_rate", "0.004")]:
                with self.subTest(argument=name), self.assertRaisesRegex(ValueError, "configuration mismatch"):
                    self.run_training(Path(directory) / name, config, inputs, resume=path,
                                      extra_args=[f"--{name}", value])

    def test_weights_only_start_resets_optimizer_schedule_and_run_identity(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(directory, config)
            path, partial, _, _, _ = self.run_training(Path(directory) / "partial", config, inputs, limit=1)
            partial["wandb_run_id"] = "previous-run"
            torch.save(partial, path)
            _, restarted, _, updates, _ = self.run_training(
                Path(directory) / "restart", config, inputs, resume=path, weights_only=True,
                limit=1, epochs=1, extra_args=["--token_budget", "101"],
            )
            self.assertEqual(restarted["step"], 1)
            self.assertEqual(len(updates), 1)
            self.assertEqual(updates[0]["lr"], 0.003)
            self.assert_nested_equal(updates[0]["model"], partial["model"])
            self.assertTrue(all(state["step"].item() == 1 for state in restarted["optimizer"]["state"].values()))
            self.assertIsNone(restarted["wandb_run_id"])

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        config = self.make_config()
        model = MindLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"previous checkpoint")
            with mock.patch.object(torch, "save", side_effect=OSError("simulated write failure")):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    training_utils.save_training_checkpoint(
                        path, model, optimizer, scaler, config, 0, 1, False, training_stage="sft",
                    )
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_wandb_public_init_supports_new_runs_and_full_resume_without_util(self):
        config = self.make_config()
        run = SimpleNamespace(id="public-run-id", log=mock.Mock(), finish=mock.Mock())
        wandb_module = SimpleNamespace(init=mock.Mock(return_value=run))
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"wandb": wandb_module}):
            inputs = self.make_inputs(directory, config)
            path, partial, _, _, _ = self.run_training(
                Path(directory) / "partial", config, inputs, limit=1, extra_args=["--use_wandb"],
            )
            first_init = wandb_module.init.call_args.kwargs
            self.assertIsNone(first_init["id"])
            self.assertEqual(first_init["resume"], "never")
            self.assertEqual(partial["wandb_run_id"], run.id)
            _, resumed, _, _, _ = self.run_training(
                Path(directory) / "resume", config, inputs, resume=path, extra_args=["--use_wandb"],
            )
            second_init = wandb_module.init.call_args.kwargs
            self.assertEqual(second_init["id"], run.id)
            self.assertEqual(second_init["resume"], "must")
            self.assertEqual(resumed["wandb_run_id"], run.id)
            self.assertEqual(wandb_module.init.call_count, 2)


if __name__ == "__main__":
    unittest.main()
