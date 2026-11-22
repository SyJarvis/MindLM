"""Run the frozen minicpm5_suite_v2 evaluation with a bare MindLM .pt checkpoint.

The frozen evaluator (eval/eval_minicpm5_suite_v2.py) loads models via
AutoModelForCausalLM in HF-dir format. The HF export of MindLM is broken on
this container's transformers version (the tied output.weight is dropped from
safetensors by save_pretrained and never re-tied on load, so the reloaded
model has a random lm_head), while loading the .pt directly via
modeling_mindlm.MindLM + training_utils.load_model_checkpoint is verified
correct (tie intact, sane generations).

This wrapper keeps the frozen evaluation protocol (suite loading, prompt
rendering, multi-round loop, judging, manifest, summary, resume) exactly as
is and only swaps the model entry point:

    v2.HFGenerator  ->  MindLMPTGenerator  (identical call/generation contract,
                        .pt loading in exchange)

and records the model identity (checkpoint / config / tokenizer sha256) for
the manifest so the run stays fingerprint-traceable.

Usage (mirrors eval_minicpm5_suite_v2.py flags plus mindlm ones):

  python eval/run_v2_suite_mindlm.py \
      --suite eval/data/minicpm5_eval_v2/dev.jsonl --split dev \
      --mindlm_checkpoint <...>.pt --template_mode mindlm \
      --output_jsonl <out>.jsonl [--resume] [--device cuda:0]

Output artifacts are identical in shape to the frozen evaluator:
  <out>.jsonl, <out>.manifest.json, <out>.summary.json
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.eval_minicpm5_suite_v2 as v2

# Wrapper-level configuration: the frozen main() rebuilds its own Namespace
# from argv, so mindlm-specific fields live here instead of the Namespace.
MINDLM_SETTINGS = {"checkpoint": None, "tokenizer": None, "config": "mindlm_0.2b_gdn"}


class MindLMPTGenerator:
    """Same generation contract as v2.HFGenerator; loads a bare MindLM .pt."""

    def __init__(self, args):
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        self.args, self.torch = args, torch
        tokenizer_path = MINDLM_SETTINGS["tokenizer"]
        checkpoint = MINDLM_SETTINGS["checkpoint"]
        # NOTE: deliberately NOT calling minicpm5_io.configure_tokenizer here.
        # The stage-D training pack (data_process/sft.py --template_style mindlm)
        # rendered every row with the mindlm_tokenizer BUILT-IN template
        # ("<s>user\n...</s>\n<s>assistant\n"), verified byte-exact against the
        # pack bin. configure_tokenizer swaps in the MiniCPM5 ChatML template,
        # which the stage-D model has never seen (v1 eval scored 0/200 with it).
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if args.template_mode == "mindlm":
            if not isinstance(tokenizer.chat_template, str) or "<s>user" not in tokenizer.chat_template:
                raise ValueError("expected the mindlm built-in chat template")
        elif not isinstance(tokenizer.chat_template, str) or "enable_thinking" not in tokenizer.chat_template:
            raise ValueError("native mode requires the original template with enable_thinking support")
        self.tokenizer = tokenizer
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        from modeling_mindlm import MindLM
        from training_utils import build_model_config, load_model_checkpoint
        config = build_model_config(MINDLM_SETTINGS["config"], tokenizer)
        self.model = MindLM(config).to(dtype).to(args.device).eval()
        load_model_checkpoint(self.model, MINDLM_SETTINGS["checkpoint"], "cpu")
        self.model.eval()
        eos = tokenizer.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.context_limit = min(args.max_context_tokens, self.model.config.max_seq_len)
        assert self.context_limit >= 64
        self.model.generation_config = GenerationConfig(do_sample=False, num_beams=1,
            repetition_penalty=1.0, use_cache=True, eos_token_id=self.eos,
            bos_token_id=self.model.config.bos_token_id, pad_token_id=tokenizer.pad_token_id)

    def __call__(self, messages, tools, round_index=0):
        if self.args.template_mode == "mindlm":
            # Training-faithful rendering: the stage-D pack used the mindlm
            # built-in template via apply_chat_template (verified byte-exact vs
            # the pack bin). The built-in template renders user/assistant turns
            # only; tool schemas and tool-role messages are dropped (same
            # downgrade the training pack applied — see stageD contract §3.5/§8).
            history = []
            for message in messages:
                role = message["role"]
                if role in ("user", "assistant", "system"):
                    history.append({"role": role,
                                    "content": message.get("content", "") or ""})
            prompt = self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True,
                tools=None, enable_thinking=False)
        else:
            prompt = v2.render_prompt(self.tokenizer, messages, tools, self.args.template_mode)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False,
                                return_tensors="pt").to(self.args.device)
        length = inputs["input_ids"].shape[1]
        if length + self.args.max_new_tokens > self.context_limit:
            raise ValueError(f"prompt ({length}) + max_new_tokens exceeds context limit {self.context_limit}")
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.args.max_new_tokens,
                eos_token_id=self.eos, pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        generated = output[0, length:].tolist()
        terminated = bool(generated) and generated[-1] in self.eos
        return {"text": self.tokenizer.decode(generated[:-1] if terminated else generated, skip_special_tokens=False),
                "prompt_tokens": length, "generated_tokens": len(generated), "truncated": not terminated}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--mindlm_checkpoint", required=True)
    parser.add_argument("--mindlm_tokenizer", default=str(ROOT / "tokenizers" / "mindlm_tokenizer"))
    parser.add_argument("--mindlm_config", default="mindlm_0.2b_gdn")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit_cases", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--template_mode", choices=("native", "mindlm"), required=True)
    args = parser.parse_args()
    if args.template_mode != "mindlm":
        parser.error("this wrapper is for the mindlm template line; use the frozen evaluator for native models")
    if args.limit_cases < 0:
        parser.error("limit_cases cannot be negative")
    # The frozen main() requires exactly one of model/responses; we satisfy
    # model mode with our own loader, surface compatibility flags accordingly.
    args.model_name_or_path = None
    args.responses_file = None
    if not 1 <= args.max_context_tokens <= 8192 or not 1 <= args.max_new_tokens <= 512 or not 1 <= args.max_rounds <= 8:
        parser.error("limits must be positive and at most 8192 context, 512 generated tokens, 8 rounds")

    # ---- swaps (kept minimal and reversible) ----
    original_HFGenerator = v2.HFGenerator
    original_model_identity = v2.model_identity
    original_load_tokenizer = v2.load_tokenizer

    def mindlm_model_identity(path):
        ckpt = Path(MINDLM_SETTINGS["checkpoint"])
        tok = Path(MINDLM_SETTINGS["tokenizer"])
        tok_files = {name: sha256_file(tok / name) for name in
                     ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
                     if (tok / name).exists()}
        return {"kind": "mindlm_pt_direct", "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
                "config": MINDLM_SETTINGS["config"], "tokenizer_path": str(tok), "tokenizer_sha256": tok_files}

    def load_tokenizer_for_manifest(_path, mode):
        # The manifest must hash the BUILT-IN mindlm template (training-faithful),
        # not the MiniCPM5 ChatML swap that the frozen load_tokenizer(mindlm)
        # would apply via configure_tokenizer.
        if mode == "mindlm":
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(MINDLM_SETTINGS["tokenizer"], local_files_only=True)
        return original_load_tokenizer(MINDLM_SETTINGS["tokenizer"], mode)

    MINDLM_SETTINGS["checkpoint"] = args.mindlm_checkpoint
    MINDLM_SETTINGS["tokenizer"] = args.mindlm_tokenizer
    MINDLM_SETTINGS["config"] = args.mindlm_config
    v2.HFGenerator = MindLMPTGenerator
    v2.model_identity = mindlm_model_identity
    v2.load_tokenizer = load_tokenizer_for_manifest
    try:
        # Frozen main() parses its own argv: rewrite argv to match its parser.
        frozen_argv = [sys.argv[0],
            "--suite", args.suite,
            "--split", args.split,
            "--model_name_or_path", str(Path(MINDLM_SETTINGS["checkpoint"])),
            "--output_jsonl", args.output_jsonl]
        if args.resume:
            frozen_argv += ["--resume"]
        if args.limit_cases:
            frozen_argv += ["--limit_cases", str(args.limit_cases)]
        frozen_argv += ["--device", args.device, "--dtype", args.dtype,
                        "--max_context_tokens", str(args.max_context_tokens),
                        "--max_new_tokens", str(args.max_new_tokens),
                        "--max_rounds", str(args.max_rounds),
                        "--template_mode", args.template_mode]
        sys.argv = frozen_argv
        return v2.main()
    finally:
        v2.HFGenerator = original_HFGenerator
        v2.model_identity = original_model_identity
        v2.load_tokenizer = original_load_tokenizer


if __name__ == "__main__":
    sys.exit(main())
