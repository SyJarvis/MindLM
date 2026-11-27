# 阶段 B：流式数据通路 —— 完成报告（2026-11-20）

对应 refocus-plan 阶段 B。全部产物在 `runs/stageB_v6_20261120/`（60）+ 本地 `runs/stageB_v6_20261120/`。对语料只读，未启动任何训练。

## 验收结果总览

| 步骤 | 结论 | 证据 |
|---|---|---|
| V6 只读全量验收 | **PASS** | v6-acceptance-report.json：7,217,908 行、31,331,876,573 bytes 与 meta 逐项对账；7 来源行数分布全对；0 坏 JSON、0 未知来源、0 空文本；重复文本仅 422 行（xlam_fc 373 + tbench 26 + perfectblend 23 = 0.006%）；108 秒全量扫描 |
| tokenizer 适配性（mindlm_tokenizer 6,400 词表） | **PASS** | tokenizer-fitness-report.json：分源各采样 2,000 行（toolace 全量 740）共 19.7M token；**UNK 率 0.000000**（全源零未知 token）；fertility：英文 0.30–0.44 token/char、中文 0.75–0.78；词表利用率 42%–83%；全语料估算 **≈9.03B token** |
| 流式 loader（data_process/v6_stream_loader.py） | **实现+验证** | 确定性、split 零泄漏、换种子顺序改变全部通过；见下 |
| 单窗口=全局排序 合同对齐 | **EXACT** | 20,000 行切片：流式（window≥全量）与旧离线 packer 逐记录一致 3565/3565 |
| 内存上界 | 有界 | 流式全程峰值 RSS 168 MiB（含 tokenizer），窗口/token 缓冲受 window_rows/CHUNK 上界控制 |
| 吞吐 | 1,804 行/s 单线程 | 20,000 行 11.1s → 全量单 epoch ≈ 67 分钟（单线程）；多进程可再压缩 |

## 新合同要点（相对旧 v1 离线合同的差异）

1. **边界 token 来自 live tokenizer**：mindlm_tokenizer 的 EOS `</s>`（id=2），替代 v1 硬编码的 Qwen3 `<|im_end|>`。`PackedPretrainDataset`（dataset.py）与 `data_process/pretrain.py` 的硬编码检查在流式路径上一律不走——流式 loader 不产 `.bin`，训练时直接消费记录流。
2. **窗口化确定性混合**：替换离线的全局 `(order_hash, full_hash)` 排序为按窗口（默认 50,000 行）内排序，保持 SHA256(seed+'/mix/'+full_hash) 语义 + 字节级可复现；单窗口模式与全局排序逐字节等价（已验证 EXACT）。
3. **train/heldout 切分**：沿用 v1 规则（SHA256(seed+'/heldout/'+group_hash)，group=前 30 归一化字符，NFC 归一化），实测同种子零交集、跨种子敏感。
4. 与 `pretrain.py` 训练入口的对接入口在阶段 C 落地：`--tokenizer_path tokenizers/mindlm_tokenizer` + 流式 loader 适配层（把 `PackedPretrainDataset` 换成流式迭代器，保持 4097 宽记录、input=[ :-1]/target=[1:] 合同不变）。

## 训练量级事实（阶段 C 排期用）

- 语料总 token ≈ **9.03B**（按分源采样外推，中文两源占 ~7B）
- 4097-token 记录全量 ≈ **2.2M 条**；heldout_ppm=5000（0.5%）→ train ≈ 2.19M 条
- 单 epoch 单线程 tokenize+pact ≈ 67 分钟；H200 上 0.2B 模型的计算时间远小于此，tokenize 可与训练 overlap 或预 pack

## 数据决定（待阶段 C 冻结）

- heldout_ppm 建议 5,000（0.5%，约 45M token）——比旧管线（盘点推定数千 ppm）更充裕，src 分桶防泄漏已由 group-hash 规则保证
- seed 建议 20261120（现验证所用），prefix_chars=30 沿用

## 遗留与边界

- fertility 的跨分位效应（超长文档截断在 pack 边界自然丢弃尾 token，合同同 v1 discarded_tail）已在 packer 内实现，全量数字需正式 pack 时记录
- v6_acceptance_audit.py / v6_tokenizer_fitness.py / v6_stream_loader.py / v6_stream_validate.py 四个脚本已入库 data_process/，本地与 60 双端同步
- 本通路未做：多进程 encode 的 worker 池化（吞吐富余，阶段 C 按需）；与 `pretrain.py` 主循环的 wiring（阶段 C 第一项）
