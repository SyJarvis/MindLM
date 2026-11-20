import torch, json, sys
files = sys.argv[1:] if len(sys.argv) > 1 else [
    "runs/sft_v3_e2_20260922/baseline/mindlm_pretrain_mindlm_0.2b_gdn_epoch2.pt",
    "runs/sft_v3_e2_20260922/eval/final_dialogue/mindlm_sft_final_update15678.pt",
    "runs/sft_v3_e2_20260922/eval/update14400/mindlm_sft_update14400.pt",
    "runs/sft_t2t_tool_repair_v2_20260924/full/out/mindlm_sft_grouped_latest.pt",
    "runs/sft_t2t_tool_repair_v2_20260924/redo/out/mindlm_sft_grouped_latest.pt",
    "runs/sft_t2t_tool_repair_v2_20260924/train100/out/mindlm_sft_grouped_latest.pt",
]
for f in files:
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
        meta = {k: d.get(k) for k in ["step","epoch","training_stage","loss","epoch_complete"] if k in d}
        if "extra_state" in d:
            meta["extra_state_keys"] = list(d["extra_state"].keys())
        print(f + ": " + json.dumps(meta))
    except Exception as e:
        print(f + ": ERROR " + str(e))
