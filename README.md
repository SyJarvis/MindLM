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

Train MindLM pretraining from packed train/heldout streams:

```bash
python pretrain.py \
  --model_config mindlm_0.2b_gdn \
  --train_data_prefix data/packed/train \
  --validation_data_prefix data/packed/heldout \
  --batch_size 16 \
  --gradient_accumulation_steps 8
```

The trainer uses the copied `qwen3_tokenizer/` by default. The packed manifest must
declare the Qwen3 chat EOS `<|im_end|>` as `boundary_token`.

Pack variable-length source text into fixed token blocks before launching a long run.
This removes short-sample padding from the training hot path:

```bash
python prepare_data.py --type pretrain \
  --input-csv data/pretrain_data.csv \
  --tokenizer-path qwen3_tokenizer \
  --output-dir data/packed \
  --max-seq-len 4096
```

Fine-tune from a matching pretraining checkpoint:

```bash
python full_sft.py \
  --model_config mindlm_0.1b \
  --resume_from out/mindlm_pretrain_mindlm_0.1b_epoch4.pt \
  --data_path data/sft_data_single.csv
```

`--batch_size` 是单进程读取的 batch size，实际 token 数还取决于
`--gradient_accumulation_steps`。

```bash
python pretrain.py \
  --model_config mindlm_0.2b_gdn \
  --train_data_prefix data/packed/train \
  --validation_data_prefix data/packed/heldout \
  --batch_size 16 \
  --gradient_accumulation_steps 8
```

SFT 使用各自的训练入口和参数。

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
prepare_data.py           Canonical train/heldout data preparation entry point
bench_train_step.py     Synthetic train-step benchmark and profiler entry point
pretrain.py             Pretraining entry point
full_sft.py             Full SFT entry point
eval/                   Checkpoint evaluation scripts
export_model.py         HuggingFace export script
tests/                  CPU smoke tests
```

The documents under `docs/` include exploratory designs for long context, multimodal,
and larger models. They are not part of the supported runnable configuration surface.
