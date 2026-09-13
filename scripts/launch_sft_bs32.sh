#!/bin/bash
# MindLM SFT bs32 定时启动脚本（2026-09-10 由训练暂停操作生成）
# 计划启动时间：2026-09-10 19:00 Asia/Shanghai（约 5.8 小时后）
# 到点后自动检查 GPU 空闲（<2GB 视为空闲，最多等 2 小时），空闲即启动 bs32 SFT。
# 训练参数与 docs/sft_training_plan.md 4.3 节方案 B 完全一致（含分块 CE 修复）。

LOG=/home/runke.zhong.srv/workspace/MindLM/out/sft_qwen3_combined_2048_bs32.log

cd /home/runke.zhong.srv/workspace/MindLM

# 等 GPU 空闲（最多 120 分钟，每 5 分钟查一次）
for i in $(seq 1 24); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$USED" -lt 2000 ]; then
        break
    fi
    echo "$(date '+%F %T') GPU busy (${USED} MiB), waiting..." >> /tmp/sft_autostart.log
    sleep 300
done

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "$(date '+%F %T') launching bs32 SFT, GPU used=${USED} MiB" >> /tmp/sft_autostart.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 -u full_sft.py \
    --data_path data/sft_qwen3_combined.csv \
    --model_config mindlm_0.1b \
    --resume_from out_pretrain_minimind_pb/mindlm_pretrain_mindlm_0.1b_epoch2.pt \
    --resume_weights_only \
    --max_seq_len 2048 --batch_size 32 --accumulation_steps 8 \
    --epochs 2 --learning_rate 5e-5 \
    --save_interval 2000 --log_interval 50 --num_workers 8 \
    --use_wandb --wandb_run_name mindlm_0.1b_sft_qwen3combined_2048_bs32 \
    >> "$LOG" 2>&1
