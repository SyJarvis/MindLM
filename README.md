# MindLM

自研小型 LLM 项目。当前为收敛主线：**固定结构、重新预训练 + SFT**。主线决定与实施计划见 [`docs/project/2026-11-20/refocus-plan.md`](docs/project/2026-11-20/refocus-plan.md)。

## 当前结构（2026-11-20 起）

- **模型**：`modeling_mindlm.py` + `config/mindlm_0.2b_gdn.json`（GDN×12 + Attn×4，16 层，dim 768，context 4096），结构保留不动。
- **Tokenizer**：`tokenizers/` 双套并列保留——`mindlm_tokenizer`（6,400 词表，新预训练用）与 `minicpm5_tokenizer`（130,072 词表 + chat 模板，SFT 对话线）。指纹与说明见 [`tokenizers/README.md`](tokenizers/README.md)。
- **预训练语料**：60 服务器 `pretrain_v6_merged.jsonl`（31.3GB / 7,217,908 行，在盘完好）。数据通路采用训练时流式 tokenize/pack，**尚未实现**，见 refocus-plan 阶段 B。
- **训练入口**：`pretrain.py`（`--tokenizer_path` 指向 `tokenizers/mindlm_tokenizer`）、`full_sft.py`。当前 packed 数据入口仅支持旧格式，流式通路实现前不满足新主线启动条件。

## 旧主线状态

2026-11-20 之前的预训练/SFT/工具调用实验**全部归档、不迁移、不作为新主线对照**。产物在 60 `runs/`（148G）原地保留；模型身份登记册 `runs/stage2_identity_20261120/`；实验台账 `docs/project/2026-10-16/experiments.csv`；评测基础设施（`eval/eval_sft_tool*.py`、300-case 语料，2026-11-20 已修复验证）跨主线复用。

历史文档：`docs/pretrain.md`、`docs/tokenizer.md`、`docs/project/2026-10-23/training-roadmap.md` 等描述旧主线，与 refocus-plan 冲突处以 refocus-plan 为准。

## 设置与测试

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

完整验证环境在 60 容器（`deeplearning-v2-mindlm-cdi`），本地缺 torch/transformers 时部分测试仅做静态检查。
