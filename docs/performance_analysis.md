# MindLM 训练性能分析 Runbook

本文定义 MindLM 单卡训练的性能采集流程。目标是用可复现数据定位训练 step 的主要瓶颈，并为 batch size、kernel、编译和数据流水线优化提供依据。

## 1. 分析目标

需要回答以下问题：

1. 一个完整 optimizer update 的时间花在哪里：CPU、数据加载、H2D、CUDA kernel、同步还是 optimizer。
2. GPU 是否受计算吞吐、HBM 带宽、kernel launch、低 occupancy 或 CPU gap 限制。
3. FLA Gated Delta Rule、FlashAttention 4、GEMM、loss 和 optimizer 各占多少时间。
4. 当前 micro batch size 是否达到最佳 tokens/s，继续增大是否仍有收益。
5. 温度降频对吞吐的影响有多大。
6. 优化后是否在相同配置和硬件状态下产生稳定、可重复的收益。

最终结果必须报告 tokens/s、step time、峰值显存和关键 kernel 指标，不能只使用 `GPU-Util` 或功耗判断性能。

## 2. 当前基线

当前已确认的训练环境：

| 项目 | 当前值 |
| --- | --- |
| GPU | NVIDIA H200 NVL，compute capability 9.0 |
| PCIe | Gen 5 x16，当前和最大链路均为 32.0 GT/s x16 |
| 模型 | MindLM 0.2B GDN，206,833,224 参数 |
| 层结构 | 12 层 linear attention，4 层标准 attention |
| Linear attention | FLA `chunk_gated_delta_rule`，chunk size 64 |
| 标准 attention | FlashAttention 4 `flash_attn.cute.flash_attn_func` |
| 数值类型 | BF16 autocast，FP32 参数和 optimizer |
| 序列长度 | 4096 |
| Micro batch size | 64 |
| Gradient accumulation | 8 |
| 每 update token 数 | 2,097,152 |
| 峰值 allocated 显存 | 约 46.47 GiB |
| 峰值 reserved 显存 | 约 57.37 GiB |
| FLA | `flash-linear-attention==0.6.0` |
| FA4 | `flash-attn-4==4.0.0b31.dev3+g0f3fb00` |

已观察到两种明显不同的运行状态：

| 状态 | SM 频率 | 温度 | 功耗 | 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 热降频阶段 | 约 345-765 MHz | 86-87 C | 约 237-300 W | 约 29-31k tokens/s |
| 高频阶段 | 约 1740 MHz | 85-87 C | 约 387-437 W | 约 55-58k tokens/s |

驱动曾明确报告 `SW Thermal Slowdown: Active`。后续基准必须记录温度、SM 频率和 throttle reason；不能把不同热状态下的吞吐差异归因于代码改动。

## 3. 当前阻塞条件

### 3.1 不影响正在运行的长训

不要对当前长训直接执行 Nsight Compute。NCU 会暂停和 replay kernel，可能显著降低吞吐，并改变训练的时序与热状态。当前训练结束前只允许只读监控。

### 3.2 Nsight Systems 尚未安装

当前宿主机和容器中都没有 `nsys`。开始分析前，需要安装与当前驱动兼容的 Nsight Systems，并确认：

```bash
nsys --version
```

### 3.3 GPU performance counter 权限未开放

当前驱动参数为：

```text
RmProfilingAdminOnly: 1
```

NCU 探针返回：

```text
ERR_NVGPUCTRPERM
```

管理员需要选择以下一种方式：

- 将 `NVreg_RestrictProfilingToAdminUsers` 设置为 `0`，然后按机器运维要求重载驱动或重启；
- 使用具备 `CAP_SYS_ADMIN` 的专用 profiling 容器。

不要在训练运行期间重载 NVIDIA 驱动。

权限开放后先验证：

```bash
ncu --query-metrics-mode suffix \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed
```

命令不应再返回 `ERR_NVGPUCTRPERM`。

## 4. 实验控制

每轮对比必须固定以下变量：

- 同一台机器和同一张 GPU；
- 相同模型代码、配置、tokenizer 和依赖版本；
- 相同 batch size、sequence length、dtype 和 gradient checkpointing；
- 相同 warmup 和 measured step 数；
- GPU 上没有其他计算进程；
- GPU 温度和 SM 频率处于可比较区间；
- 固定随机种子；
- 每个候选配置至少重复 3 次，报告中位数和范围。

每轮开始前记录：

```bash
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi -q -d PERFORMANCE,TEMPERATURE,CLOCK,POWER
```

性能报告必须保存代码 commit、dirty diff 摘要、完整命令、环境变量和 Python package 版本。

## 5. 推荐采集顺序

### 5.1 前置任务：建立训练等价 benchmark

现有 `bench_train_step.py` 可用于小规模 kernel 探针，但目前不能作为正式长训的端到端基线：

- benchmark 调用 `model(..., return_logits=True)`，会物化完整词表 logits；
- 正式训练调用 `model(..., return_logits=False)`，再使用 `masked_lm_head_loss` 分块计算 LM head 和 CE；
- 正式训练的 `loss_chunk_tokens` 为 16384，并使用 8 次 gradient accumulation；
- benchmark 每个 micro-step 都执行 `zero_grad` 和 optimizer step。

因此，不要直接用现有脚本运行 `batch_size=64, seq_len=4096` 并把结果与长训 tokens/s 比较。profiling 前应为 benchmark 增加与 `pretrain.py::forward_loss`、gradient accumulation 和 optimizer update 等价的路径，并通过一次正式训练 update 的 loss、峰值显存和 kernel 构成验证代表性。

在该前置任务完成前，现有脚本只用于较小 batch 的工具链和 kernel 探针，例如：

```bash
cd /home/runke.zhong.srv/MindLM
source /home/runke.zhong.srv/workspace/venvs/mindlm-kernels-20260911/bin/activate

CUDA_VISIBLE_DEVICES=0 python bench_train_step.py \
  --model_config mindlm_0.2b_gdn \
  --batch_size 1 \
  --seq_len 4096 \
  --warmup 5 \
  --steps 5 \
  --dtype bfloat16
```

### 5.2 阶段 A：无 profiler 稳态基准

训练等价 benchmark 完成后，使用与正式训练相同的参数建立 profiler 之外的基线：

```bash
CUDA_VISIBLE_DEVICES=0 python <TRAIN_EQUIVALENT_BENCHMARK> \
  --model_config mindlm_0.2b_gdn \
  --batch_size 64 \
  --seq_len 4096 \
  --gradient_accumulation_steps 8 \
  --loss_chunk_tokens 16384 \
  --warmup_updates 2 \
  --updates 10 \
  --dtype bfloat16
```

记录：

- 每个 micro-step 和完整 optimizer update 的时间；
- `tokens_per_second`；
- `peak_memory_gib`；
- forward、backward、optimizer 各阶段时间；
- 测量期间的温度、功耗和 SM 频率范围。

Micro batch sweep 从 8、16、32、64 开始，再根据显存余量决定是否增加。比较 micro batch 时保持每个 optimizer update 的总 token 数一致，按需反向调整 gradient accumulation。不要把增大 accumulation 当作增大 micro batch，因为前者不会增加单个 kernel 的并行工作量。

### 5.3 阶段 B：PyTorch Profiler

训练等价 benchmark 应支持生成一个完整 optimizer update 的 Chrome trace：

```bash
CUDA_VISIBLE_DEVICES=0 python <TRAIN_EQUIVALENT_BENCHMARK> \
  --model_config mindlm_0.2b_gdn \
  --batch_size 64 \
  --seq_len 4096 \
  --gradient_accumulation_steps 8 \
  --loss_chunk_tokens 16384 \
  --warmup_updates 2 \
  --updates 1 \
  --dtype bfloat16 \
  --profile \
  --trace_dir out/profile
```

先用该 trace 确认 operator 级热点、CPU gap、同步和显存分配，再进入 Nsight。PyTorch Profiler 自身有开销，其吞吐不能作为正式基线。

### 5.4 阶段 C：Nsight Systems

Nsight Systems 用于定位完整 step 的时间线和最耗时 kernel，不用于获取单个 kernel 的深层硬件计数器。

```bash
mkdir -p out/nsys

nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --force-overwrite=true \
  --output=out/nsys/mindlm_b64_s4096 \
  python <TRAIN_EQUIVALENT_BENCHMARK> \
    --model_config mindlm_0.2b_gdn \
    --batch_size 64 \
    --seq_len 4096 \
    --gradient_accumulation_steps 8 \
    --loss_chunk_tokens 16384 \
    --warmup_updates 2 \
    --updates 2 \
    --dtype bfloat16
```

生成统计摘要：

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,os_runtime_sum \
  out/nsys/mindlm_b64_s4096.nsys-rep \
  > out/nsys/mindlm_b64_s4096_stats.txt
```

重点检查：

- GPU 时间线是否存在持续空洞；
- CUDA API 和同步是否阻塞 CPU；
- kernel launch 是否过碎；
- FLA、FA4、GEMM、loss 和 optimizer kernel 的 GPU 时间占比；
- forward、backward 和 optimizer 的时间比例；
- H2D 和数据准备是否与计算重叠；
- CUDA Graph 或 `torch.compile` 是否有明确收益空间。

如果 trace 太大，减少 measured steps，不要跳过 warmup。

### 5.5 阶段 D：Nsight Compute

先从 Nsight Systems 的 kernel summary 中取得准确的 kernel 名称。NCU 每次只分析一种关键 kernel，且限制采集次数。

第一轮使用 `basic` set：

```bash
mkdir -p out/ncu

ncu \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:<EXACT_KERNEL_PATTERN_FROM_NSYS>' \
  --launch-count 1 \
  --force-overwrite \
  --export out/ncu/<kernel>_basic \
  python <TRAIN_EQUIVALENT_BENCHMARK> \
    --model_config mindlm_0.2b_gdn \
    --batch_size 64 \
    --seq_len 4096 \
    --gradient_accumulation_steps 8 \
    --loss_chunk_tokens 16384 \
    --warmup_updates 2 \
    --updates 1 \
    --dtype bfloat16
```

只有 `basic` 数据显示该 kernel 确实是主要瓶颈时，再使用 `detailed` 或精确 section：

```bash
ncu \
  --set detailed \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:<EXACT_KERNEL_PATTERN_FROM_NSYS>' \
  --launch-count 1 \
  --force-overwrite \
  --export out/ncu/<kernel>_detailed \
  python <TRAIN_EQUIVALENT_BENCHMARK> \
    --model_config mindlm_0.2b_gdn \
    --batch_size 64 \
    --seq_len 4096 \
    --gradient_accumulation_steps 8 \
    --loss_chunk_tokens 16384 \
    --warmup_updates 2 \
    --updates 1 \
    --dtype bfloat16
```

不要从 `--set full` 开始。当前 NCU 版本估算 `full` 需要约 5907 个 metrics，replay 成本高，并且容易让采集后的热状态失去代表性。

优先分析的 kernel 类型：

1. FLA `chunk_gated_delta_rule` forward/backward；
2. FA4 SM90 forward/backward；
3. GPU 时间排名靠前的 GEMM；
4. loss、RMSNorm、RoPE、conv 和 optimizer 中占比明显的 kernel。

## 6. 指标解释

### 6.1 整体训练指标

- `tokens/s`：主吞吐指标；
- `step_ms`：固定 token 数下的延迟；
- peak allocated/reserved memory：决定 micro batch 上限；
- GPU active time：判断 CPU 或数据流水线是否制造空洞；
- 稳态 SM frequency、temperature、power：识别热降频和功耗限制。

### 6.2 Kernel 指标

至少检查：

- SM throughput；
- DRAM throughput；
- achieved occupancy；
- eligible/active warps；
- Tensor Core pipe utilization；
- warp stall reasons；
- memory sectors、cache hit rate 和访问合并情况；
- kernel duration、grid/block size、register 和 shared memory 使用量。

判断原则：

- 高 SM throughput、低 DRAM throughput：更可能是 compute-bound；
- 高 DRAM throughput、低 SM throughput：更可能是 memory-bound；
- 两者都低且 kernel 很短：优先检查 launch overhead、shape 和融合机会；
- occupancy 低：检查寄存器、shared memory、block 配置和工作规模；
- Tensor Core 利用率低：确认 GEMM shape、dtype、layout 和对齐；
- GPU 时间线有空洞：先解决 CPU、数据加载或同步，不要先改 CUDA kernel。

`GPU-Util=100%` 只说明采样窗口内持续有 kernel 运行，不能代替上述指标。

## 7. Micro Batch Sweep

在独占 GPU、温度稳定的条件下测试：

```text
batch_size = 8, 16, 32, 64，以及显存允许的下一档
```

每档记录：

| batch | step ms | tokens/s | peak GiB | avg power | avg SM MHz | max temp | throttle reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| | | | | | | | |

选择标准是 tokens/s 已进入平台期且留有合理显存余量，不是单纯把显存占满。若增大 batch 后吞吐提升不足 3%-5%，通常不值得继续增加由其带来的优化器延迟和 OOM 风险。

## 8. 优化实验顺序

只在 profile 证据支持时逐项实验，每次只改变一个变量：

1. 排除散热和频率不稳定；
2. micro batch sweep；
3. DataLoader、H2D 和同步问题；
4. `torch.compile` 或 CUDA Graph；
5. FLA/FA4 参数和版本；
6. GEMM shape、layout 和融合；
7. 自定义 Triton/TileLang/CUDA kernel。

自定义 kernel 必须先通过前向、反向和完整训练 step 数值对齐，再比较性能。不能以单 kernel benchmark 的提升直接推断端到端训练收益。

## 9. 验收标准

一次性能优化只有同时满足以下条件才可接受：

- 固定配置下至少重复 3 次；
- 稳态 tokens/s 中位数有明确提升，建议门槛至少 5%；
- 温度、SM 频率和 throttle reason 可比较；
- loss、梯度和参数更新与基线在既定容差内对齐；
- 无 NaN/Inf，短训稳定；
- 峰值显存可接受；
- 保存原始 `.nsys-rep`、`.ncu-rep`、统计文本和完整命令；
- 端到端训练提升与 profiler 中被优化部分的占比一致。

## 10. 结果记录模板

```text
日期：
机器/GPU：
代码 commit：
工作区状态：
Python/PyTorch/CUDA：
FLA/FA4 版本：
命令：
环境变量：

模型配置：
batch / seq / dtype：
warmup / measured steps：

基线 tokens/s：
候选 tokens/s：
提升百分比：
step ms：
峰值显存：
温度范围：
SM 频率范围：
功耗范围：
throttle reason：

Nsight Systems 主要热点：
Nsight Compute 关键指标：
数值验证结果：
结论：
后续动作：
```

## 11. 预期产物

每轮正式分析至少保留：

```text
out/perf/<date>-<experiment>/
  environment.txt
  command.txt
  benchmark.txt
  gpu_samples.csv
  nsys/<name>.nsys-rep
  nsys/<name>_stats.txt
  ncu/<kernel>_basic.ncu-rep
  ncu/<kernel>_detailed.ncu-rep
  summary.md
```

原始 profiler 报告通常较大，是否纳入 Git 由仓库策略决定；默认只提交 `summary.md` 和小型统计文本，大型二进制报告存放在实验产物目录或对象存储中。
