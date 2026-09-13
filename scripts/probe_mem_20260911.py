"""GPU memory probe for MindLM SFT (20260911).

Answers, with measurements:
  1. Which CE implementation is the memory hog: view-chunk (Run 1) vs
     index-gather (current training_utils) on identical logits.
  2. How peak memory scales with sequence length on the real model
     (b1s4096 -> b8s4096) and whether 4096-long sequences are inherently
     problematic vs raw token count (compare b8s4096 = 32k tokens against
     the Run 1 baseline 32x2048 = 65k tokens).
  3. Static config facts that decide the fix: linear_attn_impl, FLA kernel
     availability, gradient checkpointing policy.

Each case is isolated: OOM in one case does not abort the probe.
No training, no data files, no side effects beyond GPU memory churn.
"""
import json
import time

import torch
import torch.nn.functional as F

REPO = "/home/runke.zhong.srv/workspace/MindLM"

RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}


def log(msg):
    print(msg, flush=True)


def gpu_snapshot(tag):
    return {
        "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "live_alloc_GiB": round(torch.cuda.memory_allocated() / 2**30, 2),
        "secs": None,
    }


def run_case(name, fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        extra = fn() or {}
        extra.update(gpu_snapshot(name))
        extra["secs"] = round(time.time() - t0, 1)
        extra["status"] = "ok"
    except torch.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        extra = {"status": "OOM", "secs": round(time.time() - t0, 1),
                 "note": str(e).splitlines()[0][:160]}
    except Exception as e:  # noqa: BLE001
        torch.cuda.empty_cache()
        extra = {"status": "error", "secs": round(time.time() - t0, 1),
                 "note": f"{type(e).__name__}: {str(e)[:160]}"}
    RESULTS["cases"][name] = extra
    log(f"[{name}] {extra}")
    return extra


# ---------------------------------------------------------------- CE variants

def ce_view_chunk(logits, targets, loss_mask, chunk_tokens=16384):
    """Run 1 implementation: contiguous slicing (views, no gather)."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(logits.dtype)
    total = flat_logits.size(0)
    pieces = []
    for start in range(0, total, chunk_tokens):
        stop = min(start + chunk_tokens, total)
        per = F.cross_entropy(flat_logits[start:stop].float(), flat_targets[start:stop],
                              reduction="none")
        pieces.append(per * flat_mask[start:stop])
    return torch.cat(pieces).sum() / flat_mask.sum()


def ce_index_gather(logits, targets, loss_mask, chunk_tokens=4096):
    """Current training_utils implementation: nonzero + fancy-index gather."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = loss_mask.reshape(-1).to(logits.dtype)
    idx = torch.nonzero(flat_mask > 0, as_tuple=False).flatten()
    loss_sum = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, idx.numel(), chunk_tokens):
        sel = idx[start:start + chunk_tokens]
        per = F.cross_entropy(flat_logits[sel].float(), flat_targets[sel], reduction="none")
        loss_sum = loss_sum + (per * flat_mask[sel].float()).sum()
    return loss_sum / flat_mask.sum()


def make_logits(bt, vocab, supervised_ratio=0.57, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(bt, vocab, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
    targets = torch.randint(0, vocab, (bt,), generator=g, device="cuda")
    mask = (torch.rand(bt, generator=g, device="cuda") < supervised_ratio).float()
    logits = logits.requires_grad_(True)
    return logits, targets, mask


# ---------------------------------------------------------------- model parts

_MODEL = {}


def get_model(device):
    if "model" in _MODEL:
        return _MODEL["model"]
    import sys
    sys.path.insert(0, REPO)
    from transformers import AutoTokenizer
    from modeling_mindlm import MindLM
    from training_utils import build_model_config

    tok = AutoTokenizer.from_pretrained(f"{REPO}/qwen3_tokenizer", trust_remote_code=True)
    config = build_model_config("mindlm_0.1b", tok)
    model = MindLM(config).to(device)
    model.train()
    _MODEL["model"] = model
    _MODEL["config"] = config
    return model


def model_fwd_bwd(batch, seqlen, device="cuda"):
    model = get_model(device)
    vocab = _MODEL["config"].vocab_size
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randint(0, vocab, (batch, seqlen), generator=g, device="cuda")
    mask = torch.zeros_like(x, dtype=torch.bfloat16)
    mask[:, -int(seqlen * 0.57):] = 1  # assistant tail supervised, as in real data
    from training_utils import masked_language_model_loss
    out = model(input_ids=x)
    loss = masked_language_model_loss(out.logits, x, mask, out.aux_loss)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return {"tokens_per_micro": batch * seqlen}


# ---------------------------------------------------------------- static case

def case_static(device="cuda"):
    model = get_model(device)
    cfg = _MODEL["config"]
    import sys
    sys.path.insert(0, REPO)
    import modeling_mindlm as mm
    return {
        "linear_attn_impl": getattr(cfg, "linear_attn_impl", None),
        "fla_kernel_available": getattr(mm, "_fla_chunk_gdr", None) is not None,
        "gradient_checkpointing": getattr(cfg, "gradient_checkpointing", None),
        "max_seq_len": cfg.max_seq_len,
        "vocab_size": cfg.vocab_size,
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "full_attn_layers": sum(1 for t in cfg.layer_types if t == "attention"),
        "linear_attn_layers": sum(1 for t in cfg.layer_types if t == "linear_attention"),
    }


def main():
    device = "cuda"
    log(f"probe start, device={torch.cuda.get_device_name(0)}")

    run_case("static_config", lambda: case_static(device))

    def ce_view():
        logits, targets, mask = make_logits(8 * 4096, 151669)
        loss = ce_view_chunk(logits, targets, mask)
        loss.backward()
        return {"loss": round(loss.item(), 4)}

    def ce_gather():
        logits, targets, mask = make_logits(8 * 4096, 151669)
        loss = ce_index_gather(logits, targets, mask)
        loss.backward()
        return {"loss": round(loss.item(), 4)}

    run_case("ce_view_chunk_8x4096", ce_view)
    run_case("ce_index_gather_8x4096", ce_gather)

    for batch in (1, 2, 4, 8):
        run_case(f"model_b{batch}_s4096", lambda b=batch: model_fwd_bwd(b, 4096, device))

    run_case("model_b32_s2048_run1_baseline", lambda: model_fwd_bwd(32, 2048, device))

    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{REPO}/out/probe_mem_20260911.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2, ensure_ascii=False)
    log("probe complete -> out/probe_mem_20260911.json")


if __name__ == "__main__":
    main()
