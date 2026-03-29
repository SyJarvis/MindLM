"""
MindLM Pretrain Script
使用混合架构(Attention + Linear Attention) + MoE进行预训练
"""

import os
import sys
import platform
import argparse
import time
import math
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from MindLM.modeling_mindlm import MindLM, MindLMConfig
from MindLM.dataset import PretrainDataset
from MindLM.config import load_config

warnings.filterwarnings('ignore')


def Logger(content):
    """日志输出，支持DDP"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(it, all_steps, args):
    """
    学习率调度：warmup + cosine decay
    """
    warmup_iters = args.warmup_iters
    lr_decay_iters = all_steps
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch, wandb):
    """训练一个epoch"""
    start_time = time.time()
    model.train()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 学习率调度
        global_step = epoch * iter_per_epoch + step
        lr = get_lr(global_step, args.epochs * iter_per_epoch, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        with ctx:
            outputs = model(X, targets=Y)

            # 基础语言模型损失
            loss = outputs.loss / args.accumulation_steps

            # 应用loss mask（只计算非padding部分的损失）
            if loss is not None:
                loss_mask_flat = loss_mask.view(-1)
                loss_flat = loss.view(-1)
                loss = torch.sum(loss_flat * loss_mask_flat) / loss_mask_flat.sum()

        # 反向传播
        if loss is not None:
            scaler.scale(loss).backward()

        # 梯度累积
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps if loss is not None else 0.0
            current_lr = optimizer.param_groups[-1]['lr']

            # 获取当前层类型分布
            layer_types_str = ", ".join([
                f"L{i}:{config.layer_types[i][:3]}"
                for i in range(min(4, config.n_layers))
            ]) + "..." if config.n_layers > 4 else ""

            Logger(
                f'Epoch:[{epoch}/{args.epochs}]({step}/{iter_per_epoch}) '
                f'loss:{current_loss:.3f} '
                f'lr:{current_lr:.7f} '
                f'layers:[{layer_types_str}] '
                f'Time:{spend_time / (step + 1) * (iter_per_epoch - step) / 60:.1f}min'
            )

            if wandb is not None and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": current_loss,
                    "lr": current_lr,
                    "epoch": epoch,
                    "step": global_step,
                })

        # 定期保存（覆盖写，只保留最新）
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            save_snapshot(epoch)


def save_snapshot(epoch):
    """定期保存模型快照（覆盖写，固定文件名）"""
    model.eval()
    moe_path = '_moe' if config.use_moe else ''
    linear_path = '_linear' if config.use_linear_attn else ''
    ckpt_path = os.path.join(args.save_dir, f'mindlm_pretrain_{config.dim}{moe_path}{linear_path}.pth')

    if isinstance(model, DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, ckpt_path)
    Logger(f'💾 快照已保存: {ckpt_path}')
    model.train()


def save_checkpoint(epoch):
    """每轮结束保存完整检查点（带 epoch 编号）"""
    model.eval()
    moe_path = '_moe' if config.use_moe else ''
    linear_path = '_linear' if config.use_linear_attn else ''
    ckpt_path = os.path.join(args.save_dir, f'mindlm_pretrain_{config.dim}{moe_path}{linear_path}_epoch{epoch}.pth')

    if isinstance(model, DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, ckpt_path)
    Logger(f'💾 模型已保存: {ckpt_path}')
    model.train()


def init_model():
    """初始化模型和tokenizer"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载MindLM分词器
    tokenizer_path = args.tokenizer_path or str(project_root / 'MindLM/mindlm_tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 确保vocab_size和pad_token_id匹配
    if config.vocab_size != len(tokenizer):
        Logger(f'⚠️ 警告: config.vocab_size({config.vocab_size}) != tokenizer.vocab_size({len(tokenizer)})')
        Logger(f'自动调整为tokenizer.vocab_size')
        config.vocab_size = len(tokenizer)
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id

    # 创建模型
    model = MindLM(config).to(args.device)

    # 打印模型信息
    total_params = count_parameters(model)
    Logger(f'🧠 MindLM模型配置:')
    Logger(f'   - 总层数: {config.n_layers}')
    Logger(f'   - 隐藏维度: {config.dim}')
    Logger(f'   - 注意力头数: {config.n_heads}')
    Logger(f'   - 层类型分布: {config.layer_types}')
    Logger(f'   - 使用MoE: {config.use_moe}')
    if config.use_moe:
        Logger(f'   - 路由专家数: {config.n_routed_experts}')
        Logger(f'   - 每token选择专家数: {config.num_experts_per_tok}')
    Logger(f'   - 使用Linear Attention: {config.use_linear_attn}')
    Logger(f'💪 总参数量: {total_params / 1e6:.3f} 百万')

    return model, tokenizer


def init_distributed_mode():
    """初始化分布式训练"""
    if not ddp:
        return

    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    Logger(f'🚀 DDP模式启动: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}')


def print_architecture_info():
    """打印架构信息"""
    Logger("\n" + "="*60)
    Logger("🎯 MindLM 架构配置")
    Logger("="*60)

    # 统计各层类型数量
    attn_count = config.layer_types.count("attention")
    linear_count = config.layer_types.count("linear_attention")

    Logger(f"总层数: {config.n_layers}")
    Logger(f"标准Attention层: {attn_count}")
    Logger(f"Linear Attention层: {linear_count}")
    Logger(f"层类型序列: {config.layer_types}")

    if config.use_moe:
        Logger(f"\nMoE配置:")
        Logger(f"  - 路由专家数: {config.n_routed_experts}")
        Logger(f"  - 每token选择专家数: {config.num_experts_per_tok}")
        Logger(f"  - 共享专家数: {config.n_shared_experts}")
        Logger(f"  - 辅助损失系数: {config.aux_loss_alpha}")

    Logger("="*60 + "\n")


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindLM Pretraining")

    # 基础配置
    parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="数据类型")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="续训权重路径")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="最大序列长度（默认576，适合0.5B模型；1B模型建议1024）")
    # 数据配置
    parser.add_argument("--data_path", type=str,
                        default="data/pretrain_data.csv",
                        help="训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Tokenizer路径（默认使用MindLM分词器）")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    
    # 模型架构配置
    parser.add_argument("--model_config", type=str, default="mindlm_0.5b",
                        help="模型配置名， 从config目录加载json")

    # 训练优化配置
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=100,
                        help="warmup步数")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="模型保存间隔")

    # 分布式训练
    parser.add_argument("--ddp", action="store_true",
                        help="启用DistributedDataParallel")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的local rank")

    # 编译优化
    parser.add_argument("--compile", action="store_true",
                        help="启用torch.compile优化（默认关闭，可能导致编译卡住）")

    # wandb配置
    parser.add_argument("--use_wandb", action="store_true",
                        help="使用Weights & Biases记录训练")
    parser.add_argument("--wandb_project", type=str, default="MindLM-Pretrain",
                        help="wandb项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb运行名称")

    args = parser.parse_args()

    # 创建输出目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(1337)

    # 设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 自动混合精度上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(
        dtype=getattr(torch, args.dtype)
    )

    # 检测DDP模式
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, args.device

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 创建MindLM配置
    if args.model_config:
        # 从JSON配置文件加载模型架构
        model_config_dict = load_config(args.model_config)
        Logger(f'加载模型配置: config/{args.model_config}.json')
        config = MindLMConfig(                                                                                                                    
            dim=model_config_dict['dim'],                                                                                                         
            n_layers=model_config_dict['n_layers'],                                                                                               
            n_heads=model_config_dict['n_heads'],                                                                                                 
            n_kv_heads=model_config_dict['n_kv_heads'],                                                                                           
            vocab_size=model_config_dict['vocab_size'],                                                                                           
            max_seq_len=model_config_dict['max_seq_len'],                                                                                         
            dropout=model_config_dict.get('dropout', 0.0),                                                                                        
            norm_eps=model_config_dict.get('norm_eps', 1e-6),                                                                                     
            hidden_dim=model_config_dict.get('hidden_dim'),                                                                                       
            multiple_of=model_config_dict.get('multiple_of', 256),                                                                                
            use_moe=model_config_dict.get('use_moe', False),                                                                                      
            n_routed_experts=model_config_dict.get('n_routed_experts', 4),                                                                        
            num_experts_per_tok=model_config_dict.get('num_experts_per_tok', 2),                                                                  
            n_shared_experts=model_config_dict.get('n_shared_experts', 1),                                                                        
            scoring_func='softmax',                                                                                                               
            aux_loss_alpha=model_config_dict.get('aux_loss_alpha', 0.01),                                                                         
            seq_aux=True,                                                                                                                         
            norm_topk_prob=True,                                                                                                                  
            use_linear_attn=model_config_dict.get('use_linear_attn', True),                                                                       
            layer_types=model_config_dict.get('layer_types'),                                                                                     
            conv_kernel_size=model_config_dict.get('conv_kernel_size', 4),                                                                        
        ) 
    else:
        Logger(f'加载默认模型配置: MindLMConfig')
        config = MindLMConfig()

    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len
        Logger(f'⚡ 使用命令行指定的最大序列长度: {config.max_seq_len}')
        
    # 打印架构信息
    print_architecture_info()

    # wandb初始化
    if args.use_wandb and (not ddp or dist.get_rank() == 0):
        import wandb
        wandb_run_name = args.wandb_run_name or (
            f"MindLM-D{config.dim}-L{config.n_layers}-H{config.n_heads}-"
            f"MoE{config.n_routed_experts if config.use_moe else 0}-"
            f"Linear{config.use_linear_attn}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
    else:
        wandb = None

    # 初始化模型和tokenizer
    model, tokenizer = init_model()

    # 加载续训权重
    if args.resume_from:
        Logger(f"加载续训权重: {args.resume_from}")
        state_dict = torch.load(args.resume_from, map_location=args.device)
        # 处理DDP的module，前缀
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        Logger("续训权重加载成功!")

    # 加载数据
    Logger(f'📂 加载训练数据: {args.data_path}')
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1.0)  # 打乱
    train_ds = PretrainDataset(df, tokenizer, max_length=config.max_seq_len)

    # 创建DataLoader
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    Logger(f'📊 数据集大小: {len(train_ds)}, 批次数量: {len(train_loader)}')

    # 优化器和梯度缩放器
    # bfloat16 不需要 GradScaler，只有 float16 需要
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # torch.compile加速（Linux + PyTorch 2.0+）
    if args.compile and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("⚡ 启用torch.compile优化...")
        model = torch.compile(model)

    # DDP包装
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 开始训练
    iter_per_epoch = len(train_loader)
    Logger(f'🚀 开始训练! 总轮数: {args.epochs}, 每轮迭代: {iter_per_epoch}')

    for epoch in range(args.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb)

        # 每轮结束后保存
        if not ddp or dist.get_rank() == 0:
            save_checkpoint(epoch)

    Logger('✅ 训练完成!')

    if wandb is not None:
        wandb.finish()
