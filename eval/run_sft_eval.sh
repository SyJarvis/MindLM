#!/bin/bash
# Run the SFT eval suite (3-question batch at two temperatures + 3 multi-turn probes)
# for one checkpoint and persist every result as JSON under <run_dir>/eval/.
#
# Usage: run_sft_eval.sh <run_dir> <checkpoint> <tag> [config] [device]
#   <run_dir>     run directory that receives eval/*.json
#   <checkpoint>  pretrain or SFT checkpoint (copied to /tmp so a concurrently
#                 running job cannot rewrite it mid-eval)
#   <tag>         filename suffix for the JSON outputs
#   [config]      model config name (default mindlm_0.2b_gdn)
#   [device]      default cpu: the training run owns the GPU, and the CPU kernel
#                 fallbacks (SDPA attention, reference GDR) make inference exact
#                 but must stay thread-pinned or they starve the trainer
set -uo pipefail

RUN_DIR=${1:?usage: run_sft_eval.sh <run_dir> <checkpoint> <tag> [config] [device]}
CKPT=${2:?missing checkpoint}
TAG=${3:?missing tag}
CONFIG=${4:-mindlm_0.2b_gdn}
DEVICE=${5:-cpu}

ROOT=/home/runke.zhong.srv/workspace/MindLM
cd "$ROOT"

PY=/home/runke.zhong.srv/workspace/venvs/mindlm-kernels-20260911/bin/python
# Without these the reference-kernel fallbacks spawn ~900 threads per process and
# three concurrent evals measurably slow the trainer (observed 18K -> 7.8K tok/s).
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

EVALDIR=$RUN_DIR/eval
SNAP=/tmp/sft_eval_ckpt.$$.pt
FILTER='UserWarning|cannot be converted|exe_args|registered at|previous kernel|new kernel|dispatch key|self.m.impl|^  operator|overridden|recommended to upgrade|Warning only once'

mkdir -p "$EVALDIR"
cp "$CKPT" "$SNAP"
trap 'rm -f "$SNAP"' EXIT

echo "run_dir=$RUN_DIR checkpoint=$CKPT tag=$TAG config=$CONFIG device=$DEVICE"
$PY -c "
import torch
ck = torch.load('$SNAP', map_location='cpu', mmap=True, weights_only=False)
print('checkpoint step=%s stage=%s epoch=%s' % (ck.get('step'), ck.get('training_stage'), ck.get('epoch')))
" 2>&1 | tail -1

echo "### 1/3 three-question batch, temperature 0.1 -> $EVALDIR/gen_sft_batch_t010_${TAG}.json"
$PY -u eval/eval_sft_batch.py --config "$CONFIG" --checkpoint "$SNAP" --device "$DEVICE" \
  --temperature 0.1 --top_k 8 --max_new_tokens 256 \
  --json_out "$EVALDIR/gen_sft_batch_t010_${TAG}.json" 2>&1 | grep -viE "$FILTER" | tail -8

echo "### 2/3 three-question batch, temperature 0.7 (observation) -> $EVALDIR/gen_sft_batch_t070_${TAG}.json"
$PY -u eval/eval_sft_batch.py --config "$CONFIG" --checkpoint "$SNAP" --device "$DEVICE" \
  --temperature 0.7 --top_k 8 --max_new_tokens 256 \
  --json_out "$EVALDIR/gen_sft_batch_t070_${TAG}.json" 2>&1 | grep -viE "$FILTER" | tail -8

echo "### 3/3 multi-turn (3 conversations) -> $EVALDIR/gen_sft_multi_t010_${TAG}.json"
$PY -u eval/eval_sft_multi.py "$SNAP" --config "$CONFIG" --device "$DEVICE" \
  --temperature 0.1 --top_k 8 --max_new_tokens 200 \
  --json_out "$EVALDIR/gen_sft_multi_t010_${TAG}.json" 2>&1 | grep -viE "$FILTER" | tail -12

echo "SFT_EVAL_DONE_FOR_${TAG}"
