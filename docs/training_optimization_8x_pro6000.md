# 8x RTX PRO 6000 训练优化方案

## 1. 已确认环境

本机有 8 张 RTX PRO 6000 Blackwell Server Edition，每张约 95 GiB 可用显存，compute capability 为 12.0，驱动为 595.58.03。硬件分成两个 NUMA 域：GPU 0-3 接近 NUMA 0，GPU 4-7 接近 NUMA 1；卡间没有 NVLink，但 P2P 读写均可用。

`vllm_deploy` 环境可用 PyTorch 2.11.0 + CUDA 13.0、NCCL 2.28.9、Triton 3.6.0、TileLang 0.1.9 和 FlashInfer。第三方 `flash-attn` 与 `flash-linear-attention` 包没有安装。Qwen3 tokenizer 在当前 `transformers` 中的 `len(tokenizer)` 是 151,669；原模型配置中的 151,936 是 embedding 预留尺寸。

当前 8 张卡被 VLLM worker 占满。开始训练或性能基准前必须先释放这些进程，不能与服务抢占同一组 GPU。

## 2. 结论

| 路径 | 决策 | 原因 |
| --- | --- | --- |
| 标准 Attention | 使用 PyTorch SDPA | `head_dim=64` 原生对齐 Flash SDPA 后端；不需要第三方 flash-attn。 |
| 线性 Attention reference | 完整 Gated Delta Rule，FP32 state | `config/mindlm_0.7b.json` 已启用；`simple` 仅用于旧 checkpoint 兼容。 |
| FLA Gated Delta Rule | CUDA 上检测到 FLA 时自动启用 | 未安装 FLA、CPU 或显式调试 reference 时回退到 PyTorch 实现。 |
| TileLang Gated Delta Rule | 作为独立候选训练 kernel | 本机已有完整前反向样例；必须与 PyTorch reference 做前向、反向和完整 step 对齐。 |
| 参数 BF16 | 暂不启用 | 当前 RoPE 缓存为 complex64；对整个模型 `to(bfloat16)` 会把复数缓存错误转换为实数，需先重构缓存 dtype。 |

标准 Attention 的 `head_dim=64` 原生对齐 Flash SDPA 后端，也兼容未来的 Flash Attention 3 优化。

## 3. 第一阶段：不改变模型定义

### 3.1 固定长度 packing

使用 `prepare_pretrain_data.py` 将 CSV 连续 token 化；文档之间插入 Qwen3 EOS，再打包成 `max_seq_len + 1` 的无 padding 块。

```bash
source /data/runke/miniconda/etc/profile.d/conda.sh
conda activate vllm_deploy

python prepare_pretrain_data.py \
  --input_csv data/pretrain_data.csv \
  --output_prefix data/packed_qwen3_4096 \
  --max_seq_len 4096
```

训练时传入 `--packed_data_prefix data/packed_qwen3_4096`。metadata 会校验序列长度和 tokenizer 词表，防止把 Qwen3 token 块喂给错误的 tokenizer。packing 的收益取决于原始长度分布；在短文本语料上，通常是首个、且最确定的吞吐提升来源。

### 3.2 DDP 和数据调度

使用 NCCL DDP，参数含义如下：

- `--batch_size`：每卡每个 micro-step 的样本数；
- global batch：`8 x batch_size x accumulation_steps`；
- `--ddp_bucket_cap_mb 100`：先用较大 bucket 减少通信发射次数；应通过 profile 在 50、100、200 间选择；
- `gradient_as_bucket_view=True`、`broadcast_buffers=False`：减少 DDP 额外内存和无用同步；
- `no_sync()`：已经在梯度累积的非更新步启用；
- Dense 模型可尝试 `--ddp_static_graph`，MoE 不使用该选项；
- `--num_workers 8 --prefetch_factor 4` 是起点。两个 NUMA 域应使用 `torchrun --numa-binding=node` 让 rank 和 DataLoader worker 跟随本地 CPU/内存。

建议启动命令：

```bash
source /data/runke/miniconda/etc/profile.d/conda.sh
conda activate vllm_deploy

TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
NCCL_DEBUG=WARN \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --numa-binding=node \
  pretrain.py \
  --ddp \
  --ddp_static_graph \
  --model_config mindlm_0.7b \
  --packed_data_prefix data/packed_qwen3_4096 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --ddp_bucket_cap_mb 100 \
  --dtype bfloat16
```

上述起点的 global batch 是 64 条 4096-token 序列。应根据目标 token batch、显存和 loss 曲线，而不是按 GPU 数线性放大学习率。

### 3.3 编译和 profile

先跑固定形状基准，再决定是否传 `--compile`：

```bash
CUDA_VISIBLE_DEVICES=0 python bench_train_step.py \
  --model_config mindlm_0.7b \
  --batch_size 1 \
  --seq_len 2048 \
  --warmup 5 \
  --steps 20 \
  --dtype bfloat16
```

第二次加 `--compile` 比较；仅在稳态 tokens/s 至少提升 10%、首轮编译时间可接受且 loss/grad 对齐时保留。使用 `--profile` 生成 Chrome trace，再检查：线性层 Python chunk、`aten::matmul` 小 kernel、卷积、SDPA、DataLoader 空洞及 NCCL all-reduce 的占比。

验收门槛：固定 seed、固定 batch/seq/dtype 下，profile 前后 loss 相同量级，梯度无 NaN；报告单卡 tokens/s、8 卡 global tokens/s、每卡峰值显存、MFU 以及 all-reduce 占比。吞吐不得只报样本/s。

## 4. 第二阶段：完整 Gated Delta Rule reference（已完成）

当前 `simple_gated_delta_attention` 更新为：

```text
S_t = exp(g_t) * S_(t-1) + K_t^T (beta_t * V_t)
O_t = Q_t * S_t
```

完整 Gated Delta Rule 还要计算状态对应的预测值，并以校正残差更新：

```text
V_hat_t = K_t * S_(t-1)
S_t = exp(g_t) * S_(t-1) + K_t^T (beta_t * (V_t - V_hat_t))
```

仓库现在提供 `linear_attn_impl: "gated_delta_rule"` 的纯 PyTorch 完整规则 reference，并在 `config/mindlm_0.7b.json` 中启用它。它按 token 做数学上明确的递推，state 用 FP32 累积，适合作为 FLA/TileLang kernel 的正确性基线。CUDA 环境检测到 FLA 时会自动切换到 chunk kernel；`"simple"` 仍保留给旧 checkpoint 兼容。两种实现的权重语义不同，已有使用 `simple` 的 checkpoint 不能中途切换。

推荐顺序：

1. 已完成纯 PyTorch reference，并做逐 token 前向及反向有限性测试。
2. 已接入 FLA chunk forward/backward adapter，保留 reference fallback，并传递配置的 chunk size。
3. 在本机 Blackwell 以 `H=16, K=V=64, T=2048/4096, BF16` 对比 reference 与 FLA 的前向、反向、峰值显存和 tokens/s。
4. FLA 无法覆盖、或 profile 显示 post-conv/split/L2Norm/gate 占比明显时，再使用 TileLang 的 GDN 样例写融合 kernel。
5. TileLang kernel 必须有 BF16 forward/backward 数值测试、完整训练 step 对齐测试、自动调参缓存和 CPU fallback；每个候选 kernel 要与 FLA baseline 比较。

本机已有可参考的 TileLang GDN 前反向样例：`/data/runke/workspace/tilelang/examples/gdn/`。另有 vLLM 中 FLA 派生实现可作接口和调度参考：`/data/runke/workspace/deepseekv4_deploy/vllm-jasl-production/vllm/model_executor/layers/fla/`。这些是参考代码，不应直接复制到训练仓库，因为 vLLM 路径主要面向推理。

## 5. 不建议现在做的事情

- 不安装或手写第三方 FlashAttention：当前 SDPA 已确认走 Flash backend。
- 不把未经 reference 对齐的 Gated Delta Rule kernel 直接用于训练。
- 不因为 custom kernel 过早引入 8 卡 FSDP/ZeRO：0.7B 全参模型在 95 GiB 卡上 DDP 足够，先消除计算和 padding 浪费。
- 不把整个模型直接转为 BF16 参数，直到复数 RoPE cache 设计被修复并完成收敛验证。
