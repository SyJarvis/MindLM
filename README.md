# MindLM

MindLM is an experimental causal language model that combines standard RoPE
attention, Gated DeltaNet linear attention, and an optional sparse MoE feed-forward
layer. The runnable training, SFT, export, and inference scripts share one
configuration contract across the supported 0.1B and 0.7B variants.

## Supported Configurations

Only the following configurations are supported by the runnable scripts:

| Name | FFN | MoE | Context | Use |
| --- | --- | --- | --- | --- |
| `mindlm_0.1b` | SwiGLU | No | 1024 | Default Dense baseline |
| `mindlm_0.1b_moe` | 4 routed + 1 shared expert | Top-2 | 1024 | MoE experiment |
| `mindlm_0.7b` | SwiGLU | No | 4096 | Qwen3 tokenizer Dense model |

The 0.1B configurations use 16 layers with 12 query heads, 3 KV heads, and 12
linear-attention layers plus 4 standard attention layers. The 0.7B configuration
uses 40 layers, 16 query heads, 4 KV heads, 30 linear-attention layers, and 10
standard attention layers. Its copied Qwen3 tokenizer has 151,669 usable tokens.
Its linear-attention layers use the complete gated delta rule. CUDA runs switch
automatically to the fused FLA kernel when `flash-linear-attention` is installed;
otherwise they use the PyTorch reference. The legacy `simple` recurrence remains
available for old checkpoints.

Qwen3 is a causal LM tokenizer and has no BERT-style `<mask>` token. Pretraining
uses the dataset's binary `loss_mask` to exclude padding positions.

The model does not currently implement standard attention KV cache or DeltaNet state
cache. Generation recomputes its context on every token and stops once the configured
context length is reached.

## Setup

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The test suite is a CPU smoke suite covering Dense and MoE forwards, token loss
masking, SFT truncation, generation length limits, and checkpoint serialization.

## Data Formats

Pretraining input is a CSV with a `text` column:

```csv
text
人工智能正在改变软件开发。
```

SFT input is a CSV with `history`, `q`, and `a` columns. `history` is a Python-list
literal containing previous `[question, answer]` pairs. It is parsed with
`ast.literal_eval`, not `eval`.

```csv
history,q,a
"[]",你好,你好，我是 MindLM。
```

Long SFT samples retain the final assistant marker and answer. Loss is calculated
only on the final answer tokens.

## Training

Train the Dense baseline:

```bash
python pretrain.py \
  --model_config mindlm_0.1b \
  --data_path data/pretrain_data.csv \
  --batch_size 64 \
  --accumulation_steps 8
```

For the 0.7B design, use `--model_config mindlm_0.7b`; it defaults to the copied
`qwen3_tokenizer/` directory. Older 0.1B checkpoints continue to use
`mindlm_tokenizer/`. Pass `--tokenizer_path` to override either default.

For 0.7B pretraining, pack variable-length source text into fixed token blocks before
launching a long run. This removes short-sample padding from the training hot path:

```bash
python prepare_pretrain_data.py \
  --input_csv data/pretrain_data.csv \
  --output_prefix data/packed_qwen3_4096 \
  --max_seq_len 4096
```

Train the MoE variant:

```bash
python pretrain.py \
  --model_config mindlm_0.1b_moe \
  --data_path data/pretrain_data.csv
```

Fine-tune from a matching pretraining checkpoint:

```bash
python full_sft.py \
  --model_config mindlm_0.1b \
  --resume_from out/mindlm_pretrain_mindlm_0.1b_epoch4.pt \
  --data_path data/sft_data_single.csv
```

多卡训练使用 PyTorch DDP。`--batch_size` 是每张 GPU 的 batch size，实际 global
batch size 为 `GPU 数 x batch_size x accumulation_steps`。例如 4 张卡训练 0.7B：

```bash
torchrun --nproc_per_node=4 pretrain.py \
  --ddp \
  --model_config mindlm_0.7b \
  --packed_data_prefix data/packed_qwen3_4096 \
  --batch_size 2 \
  --accumulation_steps 8 \
  --dtype bfloat16
```

这里的 global batch size 是 `4 x 2 x 8 = 64`。SFT 使用相同方式：

```bash
torchrun --nproc_per_node=4 full_sft.py \
  --ddp \
  --model_config mindlm_0.7b \
  --resume_from out/mindlm_pretrain_mindlm_0.7b_epoch0.pt \
  --data_path data/sft_data_single.csv \
  --batch_size 2 \
  --accumulation_steps 8
```

每个 rank 只读取自己的 `DistributedSampler` 分片，rank 0 负责日志和 checkpoint。
梯度累积的非更新步不会执行梯度同步，以降低通信开销；MoE 配置会自动启用未使用
专家参数检测。需要使用 CUDA 多进程环境，GPU 数量应与 `--nproc_per_node` 一致。

Checkpoints use a single format containing model, optimizer, scaler, config, epoch,
and batch step. A `latest` checkpoint is written only after an optimizer update.
Pretraining weights can initialize SFT, but SFT only resumes optimizer and epoch state
from a checkpoint whose `training_stage` is `sft`.

## Evaluation And Export

```bash
python eval/eval_pretrain.py \
  --config mindlm_0.1b \
  --checkpoint out/mindlm_pretrain_mindlm_0.1b_epoch4.pt

python eval/eval_sft.py \
  --config mindlm_0.1b \
  --checkpoint out/mindlm_sft_mindlm_0.1b_epoch2.pt \
  --interactive

python export_model.py \
  --config mindlm_0.1b \
  --checkpoint out/mindlm_sft_mindlm_0.1b_epoch2.pt \
  --output_dir mindlm-0.1b-sft
```

The exported directory can be loaded with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mindlm-0.1b-sft", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("mindlm-0.1b-sft", trust_remote_code=True)
output = model.generate(
    tokenizer("你好", return_tensors="pt").input_ids,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=64,
)
```

## Repository Layout

```text
config/                 Supported Dense and MoE JSON configurations
qwen3_tokenizer/       Copied Qwen3-0.6B tokenizer files (no model weights)
modeling_mindlm.py      Model, hybrid attention, and generation implementation
dataset.py              Pretraining and answer-only SFT datasets
training_utils.py       Shared config, loss, sampler, and checkpoint helpers
prepare_pretrain_data.py  Offline EOS packing for padding-free pretraining
bench_train_step.py     Synthetic train-step benchmark and profiler entry point
pretrain.py             Pretraining entry point
full_sft.py             Full SFT entry point
eval/                   Checkpoint evaluation scripts
export_model.py         HuggingFace export script
tests/                  CPU smoke tests
```

The documents under `docs/` include exploratory designs for long context, multimodal,
and larger models. They are not part of the supported runnable configuration surface.
