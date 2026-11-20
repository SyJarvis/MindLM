# Tokenizers

本目录并列保留两套 tokenizer，服务重新预训练+SFT 的收敛主线。来源与指纹均已核验（2026-11-20）。

## mindlm_tokenizer/ —— 新预训练线（自研小词表）

- 来源：h100-04:/home/runke/MiniDLLM/mindlm_tokenizer（4 文件，SHA256 与源端逐项一致）
- 词表 6,400（BPE，自研裁剪，Qwen3 派生谱系；tokenizer_config 内 added_tokens 4 个、model_max_length 32768）
- 用途：mindlm_0.2b_gdn 重新预训练的编码器。embedding 参数从 Qwen3 词表的 116.8M 降到约 4.9M（dim 768），把参数留给主干。
- 指纹：merges 29b0d4f6...、tokenizer_config 96c32156...、tokenizer.json e78f46fc...、vocab 8a49c1ad...

## minicpm5_tokenizer/ —— SFT/对话线（MiniCPM5 官方词表+模板）

- 来源：60:/mnt/data2/reexen_datasets/llm_training/MindLM/minicpm5_tokenizer（4 文件；tokenizer.json 与 MiniCPM5-1B-SFT 官方模型逐位一致，SHA 3e065a55...）
- 词表 130,072（BPE + 510 added tokens），含 chat_template.jinja（7451a05c...）与 special_tokens_map（82d96d7a...）
- 用途：SFT 阶段对话渲染与模板固定的参考线；不替代上述预训练编码器。
- 与旧 qwen3_tokenizer 的关系：qwen3_tokenizer（151,669 词表，指纹 aeb13307...）仍留在仓库根目录并保持原位，属旧主线历史资产；新主线以本目录为准。

## 边界

- 两套 tokenizer 并存不表示两个模型：模型结构不变（mindlm_0.2b_gdn，GDN×12+Attn×4）。
- 新预训练的 tokenizer/config/数据指纹、边界 token 与切分种子在启动训练前统一冻结并写入训练合同；本目录文件一经冻结不再改动。
- 词表 6,400 对 4096-context 的预训练合同有效性（coverage/fertility 对 V6 语料）需在流式通路验收时用实测数据确认，此处不作承诺。
