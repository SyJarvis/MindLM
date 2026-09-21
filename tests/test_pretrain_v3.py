"""Independent CPU oracles for the real V3 trainer entry and resume contract."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pretrain
from dataset import PackedPretrainDataset
from modeling_mindlm import MindLM, MindLMConfig

torch.set_num_threads(1)


class Tokenizer:
    special_tokens_map = {"eos_token": "end"}
    def __len__(self):
        return 32
    def get_vocab(self):
        vocab = {str(i): i for i in range(32)}
        vocab["<|im_end|>"] = 1
        return vocab
    def convert_tokens_to_ids(self, token):
        return self.get_vocab().get(token, 0)
    all_special_tokens = []


class PretrainV3Test(unittest.TestCase):
    def config(self, dropout=0):
        return MindLMConfig(dim=16, n_layers=2, n_heads=4, n_kv_heads=2, linear_attn_heads=2,
            vocab_size=32, max_seq_len=6, hidden_dim=32, multiple_of=8,
            layer_types=["attention", "linear_attention"], dropout=dropout,
            linear_attn_backend="fla", attention_backend="flash_attn_4", gradient_checkpointing="off")

    def inputs(self, directory):
        for name, count, offset in [("train", 5, 0), ("val", 3, 10)]:
            prefix = Path(directory) / name
            np.array([[i + offset + 1] + [(i + j + offset) % 20 + 6 for j in range(6)] for i in range(count)],
                     dtype=np.uint32).tofile(prefix.with_suffix(".bin"))
            prefix.with_suffix(".json").write_text(json.dumps({"format": "mindlm_packed_pretrain_v1", "dtype": "uint32",
                "loss_mask_dtype": "uint8", "sequence_length": 6, "tokens_per_record": 7, "num_sequences": count,
                "tokenizer_vocab_size": 32, "boundary_token": "<|im_end|>", "boundary_token_id": 1}))
        return Path(directory) / "train", Path(directory) / "val"

    def run_main(self, directory, inputs, config, resume=None, limit=0, accumulation=2, epochs=2, extra=()):
        argv = ["pretrain.py", "--model_config", "mindlm_0.2b_gdn", "--train_data_prefix", str(inputs[0]),
            "--validation_data_prefix", str(inputs[1]), "--tokenizer_path", str(inputs[0].parent / "tokenizer"),
            "--output_dir", str(directory), "--batch_size", "2", "--gradient_accumulation_steps", str(accumulation),
            "--epochs", str(epochs), "--warmup_updates", "0", "--learning_rate", "0.003", "--grad_clip", "1000000",
            "--device", "cpu", "--dtype", "float32", "--num_workers", "0", "--loss_chunk_tokens", "3",
            "--save_interval", "1", "--eval_interval", "1", "--log_interval", "1", "--limit_updates", str(limit)]
        if resume:
            argv += ["--resume_from", str(resume)]
        argv += list(extra)
        seen, updates, models, projections = [], [], [], []
        original_step = torch.optim.AdamW.step
        def make_model(cfg):
            model = MindLM(cfg)
            model.tok_embeddings.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()) if model.training else None)
            model.output.register_forward_pre_hook(lambda _, args: projections.append(tuple(args[0].shape)))
            models.append(model)
            return model
        def step(optimizer, *args, **kwargs):
            model = models[-1]
            updates.append({"model": copy.deepcopy(model.state_dict()), "micro_count": len(seen),
                "grads": {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None},
                "lr": optimizer.param_groups[0]["lr"]})
            return original_step(optimizer, *args, **kwargs)
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(pretrain.AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
             mock.patch.object(pretrain, "build_model_config", side_effect=lambda *_: copy.deepcopy(config)), \
             mock.patch.object(pretrain, "MindLM", side_effect=make_model), mock.patch.object(torch.optim.AdamW, "step", new=step), \
             contextlib.redirect_stdout(output):
            pretrain.main()
        path = Path(directory) / "mindlm_pretrain_mindlm_0.2b_gdn_latest.pt"
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        self.assertTrue(projections)
        self.assertTrue(all(len(shape) == 2 and shape[0] <= 3 for shape in projections), projections)
        return path, checkpoint, seen, updates, output.getvalue()

    def equal(self, a, b):
        if isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        elif isinstance(b, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in b:
                self.equal(a[k], b[k])
        elif isinstance(b, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_main_token_weighting_for_full_tail_windows_and_small_last_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(directory)
            dataset = PackedPretrainDataset(inputs[0], 6, 32)
            for accumulation in (2, 3):
                _, checkpoint, seen, updates, output = self.run_main(Path(directory) / str(accumulation), inputs, self.config(), accumulation=accumulation, epochs=1)
                first = 0
                for update in updates:
                    reference = MindLM(self.config()).train()
                    reference.load_state_dict(update["model"])
                    logits, targets = [], []
                    for ids in seen[first:update["micro_count"]]:
                        logits.append(reference(input_ids=ids).logits.reshape(-1, 32))
                        targets.append(torch.cat([dataset[int(row[0]) - 1][1] for row in ids]))
                    F.cross_entropy(torch.cat(logits), torch.cat(targets)).backward()
                    for n, p in reference.named_parameters():
                        torch.testing.assert_close(update["grads"][n], p.grad, atol=2e-6, rtol=3e-5)
                    first = update["micro_count"]
                self.assertEqual(first, 3)
                self.assertTrue(checkpoint["epoch_complete"])
                self.assertIn('"val_supervised_tokens": 18', output)

    def test_main_stop_resume_preserves_parameters_optimizer_rng_order_and_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(directory)
            config = self.config(0.2)
            _, complete, seen, updates, _ = self.run_main(Path(directory) / "full", inputs, config)
            for stop, complete_epoch in [(1, False), (2, True)]:
                path, partial, seen1, updates1, _ = self.run_main(Path(directory) / f"stop{stop}", inputs, config, limit=stop)
                self.assertEqual(partial["epoch_complete"], complete_epoch)
                self.assertEqual(partial["extra_state"]["pretrain"]["contract"]["total_updates"], 4)
                torch.manual_seed(987)
                _, resumed, seen2, updates2, _ = self.run_main(Path(directory) / f"resume{stop}", inputs, config, resume=path)
                self.equal(seen, seen1 + seen2)
                self.assertEqual([x["lr"] for x in updates], [x["lr"] for x in updates1 + updates2])
                for key in ("model", "optimizer", "scaler", "epoch", "step", "epoch_complete", "extra_state"):
                    self.equal(resumed[key], complete[key])

    def test_resume_allows_runtime_tuning_and_restores_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(directory)
            config = self.config(0.2)
            config.gradient_checkpointing = "all"
            path, partial, _, _, _ = self.run_main(
                Path(directory) / "first", inputs, config, limit=1
            )
            saved = partial["extra_state"]["pretrain"]
            self.assertEqual(saved["version"], 2)
            self.assertEqual(saved["next_record"], 4)
            self.assertEqual(saved["last_metrics"]["update"], 1)
            self.assertAlmostEqual(
                saved["optimizer_step"]["learning_rate"],
                partial["optimizer"]["param_groups"][0]["lr"],
            )

            _, resumed, _, resumed_updates, _ = self.run_main(
                Path(directory) / "resumed",
                inputs,
                config,
                resume=path,
                extra=[
                    "--batch_size", "1",
                    "--gradient_accumulation_steps", "4",
                    "--loss_chunk_tokens", "2",
                    "--gradient_checkpointing", "off",
                ],
            )
            resumed_state = resumed["extra_state"]["pretrain"]
            self.assertEqual(resumed_state["contract"]["batch_size"], 1)
            self.assertEqual(resumed_state["contract"]["gradient_accumulation_steps"], 4)
            self.assertEqual(resumed_state["contract"]["loss_chunk_tokens"], 2)
            self.assertEqual(resumed_state["contract"]["config"]["gradient_checkpointing"], "off")
            self.assertEqual(resumed_state["schedule"], saved["schedule"])
            self.assertGreaterEqual(resumed_state["global_update"], saved["global_update"])
            expected_lr = pretrain.learning_rate_at(
                saved["global_update"],
                saved["schedule"]["total_updates"],
                saved["schedule"]["warmup_updates"],
                saved["schedule"]["base_learning_rate"],
            )
            self.assertAlmostEqual(resumed_updates[0]["lr"], expected_lr)

    def test_resume_rejects_math_data_tokenizer_runtime_changes_and_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(directory)
            _, original, _, _, _ = self.run_main(Path(directory) / "first", inputs, self.config(), limit=1)
            for key in ("config", "train", "val", "tokenizer", "runtime"):
                checkpoint = copy.deepcopy(original)
                contract = checkpoint["extra_state"]["pretrain"]["contract"]
                if key == "config":
                    contract[key]["linear_attn_impl"] = "simple"
                    checkpoint["config"]["linear_attn_impl"] = "simple"
                else:
                    contract[key]["changed"] = True
                path = Path(directory) / f"changed_{key}.pt"
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "resume contract mismatch"):
                    self.run_main(Path(directory) / key, inputs, self.config(), resume=path)
            path = Path(directory) / "legacy.pt"
            torch.save({"model": original["model"], "training_stage": "pretrain"}, path)
            with self.assertRaisesRegex(ValueError, "complete pretraining checkpoint"):
                self.run_main(Path(directory) / "legacy", inputs, self.config(), resume=path)

    def test_heldout_token_mean_preserves_rng_gradient_and_train_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(directory)
            loader = DataLoader(PackedPretrainDataset(inputs[1], 6, 32), batch_size=2)
            model = MindLM(self.config(0.2)).train()
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            before = pretrain.capture_rng_state(torch.device("cpu"))
            metrics = pretrain.evaluate(model, loader, torch.device("cpu"), "float32", 3)
            self.equal(pretrain.capture_rng_state(torch.device("cpu")), before)
            self.assertTrue(model.training)
            self.assertTrue(all(torch.count_nonzero(p.grad).item() == 0 for p in model.parameters()))
            model.eval()
            with torch.no_grad():
                logits, labels = [], []
                for ids, targets, _ in loader:
                    logits.append(model(input_ids=ids).logits.reshape(-1, 32)); labels.append(targets.reshape(-1))
                expected = F.cross_entropy(torch.cat(logits), torch.cat(labels)).item()
            self.assertAlmostEqual(metrics["val_loss"], expected, places=6)
            self.assertEqual(metrics["val_supervised_tokens"], 18)

    def test_optimizer_decay_groups_and_fp32_parameters(self):
        model = MindLM(self.config())
        marked = model.layers[0].attention.wq.weight
        marked._no_weight_decay = True
        optimizer = pretrain.build_optimizer(model, 0.003, 0.1, False)
        groups = {id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]}
        for name, p in model.named_parameters():
            exempt = p.ndim < 2 or name.endswith(("bias", "A_log", "dt_bias")) or p is marked
            self.assertEqual(groups[id(p)], 0 if exempt else 0.1, name)
            self.assertEqual(p.dtype, torch.float32)

    def test_invalid_arguments_existing_assets_and_conflicting_cursor_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.inputs(directory)
            path, checkpoint, _, _, _ = self.run_main(Path(directory) / "first", inputs, self.config(), limit=1)
            with self.assertRaisesRegex(ValueError, "without existing pretraining checkpoints"):
                self.run_main(path.parent, inputs, self.config())
            for name in ("learning_rate", "weight_decay", "grad_clip"):
                with self.assertRaisesRegex(ValueError, "invalid warmup"):
                    self.run_main(Path(directory) / name, inputs, self.config(), extra=[f"--{name}", "nan"])
            checkpoint["step"] += 1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "metadata and resume cursor disagree"):
                self.run_main(Path(directory) / "cursor", inputs, self.config(), resume=path)

    def test_public_wandb_id_roundtrip_without_private_helpers(self):
        calls = []
        handle = SimpleNamespace(id="v3-public", log=lambda *a, **k: None, finish=lambda: None)
        def init(**kwargs):
            calls.append(kwargs)
            return handle
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"wandb": SimpleNamespace(init=init)}):
            inputs = self.inputs(directory)
            path, partial, _, _, _ = self.run_main(Path(directory) / "first", inputs, self.config(), limit=1, extra=["--use_wandb"])
            _, resumed, _, _, _ = self.run_main(Path(directory) / "next", inputs, self.config(), resume=path, extra=["--use_wandb"])
        self.assertIsNone(calls[0]["id"])
        self.assertEqual(calls[0]["resume"], "never")
        self.assertEqual(calls[1]["id"], handle.id)
        self.assertEqual(calls[1]["resume"], "must")
        self.assertEqual(partial["wandb_run_id"], resumed["wandb_run_id"])

if __name__ == "__main__":
    unittest.main()
