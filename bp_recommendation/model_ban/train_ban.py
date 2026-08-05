"""
BPTacticalTransformer Ban 训练脚本 (极致加速版)
====================================
支持 Mac M1 (MPS) 与 云服务器 (CUDA) 的自适应硬件加速。
新增参数:
    --amp           开启自动混合精度 (极大加速)
    --compile       开启 torch.compile 计算图编译 (仅 CUDA 生效)
    --num_workers   Dataloader 工作进程数 (推荐设为 4)
"""

# ============================================================
# 最佳超参数 (Optuna TPE 搜索 2026-06-17, Ban@10=0.8298)
# 开发模式与生产模式共用，生产模式仅 best_epoch 从配置文件读取
# ============================================================
BEST_PARAMS_BAN = {
    "h_dim": 384, "n_layers": 6, "n_heads": 6,
    "query_dim": 256, "c_dim": 64,
    "dropout": 0.1004155499576746, "attention_dropout": 0.13362160892243613,
    "candidate_hidden": 512,
    "lr": 0.00029644199186398216, "weight_decay": 0.009017448968875661,
    "warmup_ratio": 0.19486781371292378, "grad_clip": 1.4148690600178437,
    "aux_loss_weight": 0.8,  # 压缩自 1.242，防止 aux 任务过拟合（val aux 在 ep11 触底后反弹至 0.21+）
    "batch_size": 32, "n_epochs": 50, "patience": 20,
}

import os
import sys
import json
import time
import logging
import argparse
import inspect
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bp_recommendation.model_ban.dataloader_ban import create_train_val_dataloaders
from bp_recommendation.model_ban.model_ban import BPTacticalTransformer
from bp_recommendation.feature_pipeline import load_champion_vocabulary, CANDIDATE_FEAT_MAP
from bp_recommendation.config import (
    is_production_mode,
    get_production_val_ratio,
    get_production_num_epochs,
    save_best_params,
    record_training_environment,
    record_feature_dimensions,
    record_production_params,
    print_config_summary,
    get_config,
)

# CS 特征索引: ally_synergy(15) ~ ally_counter(18)，用于 NoCS 模型零化上下文敏感特征
CS_FEATURE_INDICES = slice(CANDIDATE_FEAT_MAP["ally_synergy"], CANDIDATE_FEAT_MAP["ally_counter"] + 1)

from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())

# 共享数据异常检测工具
sys.path.insert(0, _PROJECT_ROOT)
from data_checks import check_array, check_labels, check_predictions
from logger_config import get_logger, setup_logging, log_context, timed

log = get_logger(__name__)
SHARED_FEATURES_DIR = os.path.join(_PROJECT_ROOT, "bp_recommendation", "features")
CLEANED_DIR = os.path.join(_PROJECT_ROOT, "cleaned_data")
BAN_DIR = os.path.dirname(os.path.abspath(__file__))
BAN_FEATURES_DIR = os.path.join(BAN_DIR, "features")
VOCAB_PATH = os.path.join(CLEANED_DIR, "champion_vocabulary.json")
POS_JSON = os.path.join(CLEANED_DIR, "champion_position_mapping.json")
CKPT_DIR = os.path.join(BAN_DIR, "checkpoints")
LOG_DIR = os.path.join(BAN_DIR, "logs")


def setup_logger(run_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{run_name}_{timestamp}.log")

    logger = get_logger("BPTrain_Ban")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)

    logger.addHandler(fh)

    return logger, log_path


def create_dataloaders(vocab_path=VOCAB_PATH, pos_json=POS_JSON, cleaned_dir=CLEANED_DIR,
                       features_dir=SHARED_FEATURES_DIR, val_ratio=0.15, batch_size=32,
                       logger=None, force_unroll_train=False, num_workers=0):
    context_parquet = os.path.join(features_dir, "ALL_context.parquet")
    meta_parquet = os.path.join(features_dir, "ALL_meta_store.parquet")
    player_parquet = os.path.join(features_dir, "ALL_player_store.parquet")

    # 【加速】传入 num_workers 开启异步数据加载
    train_loader, val_loader = create_train_val_dataloaders(
        context_parquet, meta_parquet, player_parquet,
        vocab_path, pos_json,
        batch_size=batch_size, num_workers=num_workers, val_ratio=val_ratio,
        force_unroll_train=force_unroll_train,
    )

    if logger:
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)} (Workers: {num_workers})")

    return train_loader, val_loader


def get_bare_model(model):
    """剥离 DDP 的 .module 和 torch.compile 的 ._orig_mod 包装"""
    bare_model = model
    if hasattr(bare_model, "_orig_mod"):
        bare_model = bare_model._orig_mod
    if hasattr(bare_model, "module"):
        bare_model = bare_model.module
    return bare_model


def recall_at_k(logits, labels, available_mask, k, is_pick=None):
    masked = logits.clone()
    masked[available_mask == 0] = -1e9

    _, topk_indices = masked.topk(k, dim=-1)
    hit = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()

    if is_pick is not None:
        valid_mask = is_pick.float()
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return 0.0
        # 只有在 mask 范围内的样本才参与算分
        return ((hit * valid_mask).sum() / valid_count).item()

    if hit.numel() == 0:
        return 0.0
    return hit.mean().item()


@torch.no_grad()
def evaluate(model, val_loader, device, mask_cs=False, use_amp=False):
    model.eval()
    
    # 【加速】获取总样本数以进行预分配连续内存
    n_samples = len(val_loader.dataset)
    vocab_size = getattr(model, 'vocab_size', 180)

    
    # 从第一个 batch 动态获取 cand_dim
    sample_batch = next(iter(val_loader))
    cand_dim = sample_batch["candidate_matrix"].shape[-1]

    all_logits = torch.empty((n_samples, vocab_size), dtype=torch.float32)
    all_labels = torch.empty(n_samples, dtype=torch.long)
    all_masks = torch.empty((n_samples, vocab_size), dtype=torch.float32)
    all_is_pick = torch.empty(n_samples, dtype=torch.float32)
    all_cands = torch.empty((n_samples, vocab_size, cand_dim), dtype=torch.float32)

    all_time_weights = torch.ones(n_samples, dtype=torch.float32)
    all_bp_steps = torch.full((n_samples,), -1, dtype=torch.long)

    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_val_loss = 0.0  # 累加 total_loss (main + aux)，与 train 口径一致
    n_batches = 0
    start_idx = 0
    
    device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

    for batch in val_loader:
        batch_size = batch["label"].size(0)
        end_idx = start_idx + batch_size

        bp_seq = batch["bp_sequence"].to(device)
        ctx = batch["global_context"].to(device)
        cand = batch["candidate_matrix"].to(device)
        mask = batch["available_mask"].to(device)
        labels = batch["label"].to(device)
        is_pick = batch["is_pick"].to(device)

        # 【新增】：捕获和保存 bp_step
        bp_step = batch.get("bp_step")
        if bp_step is not None:
            all_bp_steps[start_idx:end_idx] = bp_step.cpu()

        time_weight = batch.get("time_weight")
        if time_weight is not None:
            time_weight = time_weight.to(device)
            # 【新增】：保存 time_weight
            all_time_weights[start_idx:end_idx] = time_weight.cpu()

        cand_model = cand
        if mask_cs:
            cand_model = cand.clone()
            cand_model[:, :, CS_FEATURE_INDICES] = 0.0

        history_positions = batch.get("history_positions")
        if history_positions is not None:
            history_positions = history_positions.to(device)

        time_weight = batch.get("time_weight")
        if time_weight is not None:
            time_weight = time_weight.to(device)

        if device.type == 'mps':
            use_amp = False
            device_type = 'cpu' # autocast 引擎要求，但在 disabled 状态下不影响计算
        else:
            device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.autocast(device_type=device_type, enabled=use_amp):
            out = model(bp_seq, ctx, cand_model, mask, history_positions)
            main_loss, total_loss = model.compute_loss(out["logits"], labels, out["aux_loss"], is_pick, time_weight)

        # 写入预分配张量
        all_logits[start_idx:end_idx] = out["logits"].float().cpu()
        all_labels[start_idx:end_idx] = labels.cpu()
        all_masks[start_idx:end_idx] = mask.cpu()
        all_is_pick[start_idx:end_idx] = is_pick.cpu()
        all_cands[start_idx:end_idx] = cand.cpu()
        
        total_main_loss += main_loss.item()
        total_aux_loss += out["aux_loss"].item() if isinstance(out["aux_loss"], torch.Tensor) else out["aux_loss"]
        total_val_loss += total_loss.item()
        
        start_idx = end_idx
        n_batches += 1

    is_pick_flag = (all_is_pick > 0.5).float()
    is_ban_flag = 1.0 - is_pick_flag

    ban_r1 = recall_at_k(all_logits, all_labels, all_masks, 1, is_ban_flag)
    ban_r5 = recall_at_k(all_logits, all_labels, all_masks, 5, is_ban_flag)
    ban_r10 = recall_at_k(all_logits, all_labels, all_masks, 10, is_ban_flag)
    ban_r20 = recall_at_k(all_logits, all_labels, all_masks, 20, is_ban_flag)
    pick_r10 = recall_at_k(all_logits, all_labels, all_masks, 10, is_pick_flag)
    pick_r20 = recall_at_k(all_logits, all_labels, all_masks, 20, is_pick_flag)
    overall_r10 = recall_at_k(all_logits, all_labels, all_masks, 10)
    overall_r20 = recall_at_k(all_logits, all_labels, all_masks, 20)

    metrics = {
        "main_loss": total_main_loss / max(n_batches, 1),
        "aux_loss": total_aux_loss / max(n_batches, 1),
        "total_loss": total_val_loss / max(n_batches, 1),
        "Ban@1": ban_r1,
        "Ban@5": ban_r5,
        "Ban@10": ban_r10,
        "Ban@20": ban_r20,
        "Pick@10": pick_r10,
        "Pick@20": pick_r20,
        "Overall@10": overall_r10,
        "Overall@20": overall_r20,
    }
    model.train()
    raw_data = {
        "logits": all_logits.numpy(),
        "labels": all_labels.numpy(),
        "masks": all_masks.numpy(),
        "is_pick": all_is_pick.numpy(),
        "candidates": all_cands.numpy(),
        "time_weights": all_time_weights.numpy(),  
        "bp_steps": all_bp_steps.numpy(),          
    }
    return metrics, raw_data


def train_one_epoch(model, train_loader, optimizer, scheduler, device, grad_clip=1.0, 
                    mask_cs=False, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    n_batches = 0
    
    device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

    for batch in train_loader:
        # 【加速】异步传输数据
        bp_seq = batch["bp_sequence"].to(device)
        ctx = batch["global_context"].to(device)
        cand = batch["candidate_matrix"].to(device)
        mask = batch["available_mask"].to(device)
        labels = batch["label"].to(device)
        is_pick = batch["is_pick"].to(device)

        cand_model = cand
        if mask_cs:
            cand_model = cand.clone()
            cand_model[:, :, CS_FEATURE_INDICES] = 0.0

        history_positions = batch.get("history_positions")
        if history_positions is not None:
            history_positions = history_positions.to(device)

        time_weight = batch.get("time_weight")
        if time_weight is not None:
            time_weight = time_weight.to(device)

        # 【加速】极速梯度清零
        optimizer.zero_grad(set_to_none=True)

        # 【加速】混合精度前向传播
        if device.type == 'mps':
            use_amp = False
            device_type = 'cpu' # autocast 引擎要求，但在 disabled 状态下不影响计算
        else:
            device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.autocast(device_type=device_type, enabled=use_amp):
            out = model(bp_seq, ctx, cand_model, mask, history_positions)
            main_loss, loss = model.compute_loss(out["logits"], labels, out["aux_loss"], is_pick, time_weight)

        # 【加速】混合精度反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        scheduler.step()

        total_loss += loss.item()
        total_main += main_loss.item()
        total_aux += out["aux_loss"].item() if isinstance(out["aux_loss"], torch.Tensor) else out["aux_loss"]
        n_batches += 1

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "main_loss": total_main / max(n_batches, 1),
        "aux_loss": total_aux / max(n_batches, 1),
    }


def save_checkpoint(model, optimizer, epoch, metrics, path, candidate_dim=None, context_dim=None, vocab_size=None, role_token_start=None, champion_start_idx=None):
    bare_model = get_bare_model(model)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": bare_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if candidate_dim is not None:
        ckpt["candidate_dim"] = candidate_dim
    if context_dim is not None:
        ckpt["context_dim"] = context_dim
    if vocab_size is not None:
        ckpt["vocab_size"] = vocab_size
    if role_token_start is not None:
        ckpt["role_token_start"] = role_token_start
    if champion_start_idx is not None:
        ckpt["champion_start_idx"] = champion_start_idx
    torch.save(ckpt, path)

def main(override_args=None):
    parser = argparse.ArgumentParser(description="Train BPTacticalTransformer for BAN")
    if override_args is not None:
        args = parser.parse_args(override_args)
    else:
        try:
            args = parser.parse_args()
        except SystemExit:
            args, _ = parser.parse_known_args()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.96e-4)
    parser.add_argument("--weight_decay", type=float, default=0.00902)
    parser.add_argument("--warmup_ratio", type=float, default=0.195)
    parser.add_argument("--grad_clip", type=float, default=1.415)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mask_cs", action="store_true", default=False,
                        help="Train with CS features masked (NoCS model). Default: False (CS model)")
    parser.add_argument("--export_only", action="store_true", default=False,
                        help="Skip training and only export features (using force_unroll_train=True).")
    
    # 【新增】硬件加速相关参数
    parser.add_argument("--amp", action="store_true", default=False, help="开启自动混合精度加速 (AMP)")
    parser.add_argument("--compile", action="store_true", default=False, help="使用 torch.compile 编译计算图 (仅限 Linux/CUDA)")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader 数据读取进程数 (非 CUDA 推荐 0, CUDA 推荐 2~4)")
    parser.add_argument("--run_name", type=str, default=None, help="运行时名称，用于日志和输出文件命名")
    parser.add_argument("--production", action="store_true", default=None,
                        help="强制启用生产模式 (覆盖环境变量 BP_PRODUCTION_MODE)")

    # 架构超参数 (Optuna TPE 搜索 2026-06-17, Ban@10=0.8298)
    parser.add_argument("--h_dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--c_dim", type=int, default=64)
    parser.add_argument("--query_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.100)
    parser.add_argument("--attention_dropout", type=float, default=0.134)
    parser.add_argument("--aux_loss_weight", type=float, default=0.8)
    
    args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"))

    # === 生产模式 / 开发模式 ===
    is_production = args.production if args.production is not None else is_production_mode()
    # 同步环境变量，确保 config.py 中的 get_production_* 函数能正确识别生产模式
    # （get_production_num_epochs / get_production_val_ratio 等依赖环境变量 BP_PRODUCTION_MODE）
    if is_production:
        os.environ["BP_PRODUCTION_MODE"] = "true"
    model_type_label = "NoCS" if args.mask_cs else "CS"
    model_subtype = "NoCS" if args.mask_cs else "CS"

    # === 注入硬编码最佳超参数（搜索结果，开发模式和生产模式共用）===
    # 超参数为权威源，不依赖配置文件；生产模式仅 best_epoch 从配置文件读取
    best_params = BEST_PARAMS_BAN
    args.h_dim = best_params["h_dim"]
    args.c_dim = best_params["c_dim"]
    args.query_dim = best_params["query_dim"]
    args.n_layers = best_params["n_layers"]
    args.n_heads = best_params["n_heads"]
    args.dropout = best_params["dropout"]
    args.attention_dropout = best_params["attention_dropout"]
    args.lr = best_params["lr"]
    args.weight_decay = best_params["weight_decay"]
    args.warmup_ratio = best_params["warmup_ratio"]
    args.grad_clip = best_params["grad_clip"]
    args.aux_loss_weight = best_params["aux_loss_weight"]
    args.batch_size = best_params["batch_size"]
    args.epochs = best_params["n_epochs"]
    args.patience = best_params["patience"]

    # MPS 兼容：设置多进程 start_method='spawn' 避免死锁
    if device.type == "mps" and args.num_workers > 0:
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    run_name = args.run_name or f"train_ban_{model_type_label.lower()}"
    logger, log_path = setup_logger(run_name)

    from datetime import datetime as _dt
    training_date = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_production:
        config = get_config("ban", model_subtype)
        args.epochs = get_production_num_epochs(config)
        args.val_ratio = get_production_val_ratio()
        args.patience = args.epochs + 10

        logger.info("=" * 70)
        logger.info("=" * 70)
        logger.info("  BAN MODEL - PRODUCTION MODE (Blind Training)")
        logger.info("=" * 70)
        logger.info("  Training Date : %s", training_date)
        logger.info("  Model Type    : %s", model_type_label)
        logger.info("  val_ratio     : %s", args.val_ratio)
        logger.info("  epochs        : %s", args.epochs)
        logger.info("  Early Stopping: DISABLED")
        logger.info("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("=" * 70)
        logger.info("  BAN MODEL - DEVELOPMENT MODE (Search & Validate)")
        logger.info("=" * 70)
        logger.info("  Training Date : %s", training_date)
        logger.info("  Model Type    : %s", model_type_label)
        logger.info("  val_ratio     : %s", args.val_ratio)
        logger.info("  epochs        : %s", args.epochs)
        logger.info("  patience      : %s", args.patience)
        logger.info("  Early Stopping: ENABLED")
        logger.info("=" * 70)

    logger.info("=" * 70)
    logger.info("HYPERPARAMETERS:")
    logger.info("  Device        : %s", device)
    logger.info("  lr            : %.6f", args.lr)
    logger.info("  batch_size    : %s", args.batch_size)
    logger.info("  weight_decay  : %.6f", args.weight_decay)
    logger.info("  warmup_ratio  : %.4f", args.warmup_ratio)
    logger.info("  grad_clip     : %.4f", args.grad_clip)
    logger.info("  dropout       : %.4f", args.dropout)
    logger.info("  h_dim         : %s", args.h_dim)
    logger.info("  n_layers      : %s", args.n_layers)
    logger.info("  n_heads       : %s", args.n_heads)
    logger.info("  aux_loss_w    : %.4f", args.aux_loss_weight)
    logger.info("  CS Features   : %s", not args.mask_cs)
    logger.info("  NoCS Features : %s", args.mask_cs)
    logger.info("  [Acceleration] AMP: %s, Compile: %s, Workers: %s", args.amp, args.compile, args.num_workers)
    logger.info("=" * 70)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(BAN_FEATURES_DIR, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        val_ratio=args.val_ratio, batch_size=args.batch_size, logger=logger,
        force_unroll_train=args.export_only, num_workers=args.num_workers
    )

    _, _, vocab_size, _, _ = load_champion_vocabulary(VOCAB_PATH)

    # v3 方案: 从 vocab 文件读取 role_token_start 和 champion_start_idx
    with open(VOCAB_PATH, "r", encoding="utf-8") as _f:
        _vocab_meta = json.load(_f)
    role_token_start = _vocab_meta.get("role_token_start", 2)
    champion_start_idx = _vocab_meta.get("champion_start_idx", 7)
    n_positions_vocab = _vocab_meta.get("n_positions", 5)
    logger.info(f"  vocab_size={vocab_size}, role_token_start={role_token_start}, champion_start_idx={champion_start_idx}")

    sample_batch = next(iter(train_loader))
    context_dim_inferred = sample_batch["global_context"].shape[-1]
    candidate_dim_inferred = sample_batch["candidate_matrix"].shape[-1]

    # === 数据加载后异常检查 ===
    logger.info("=" * 70)
    logger.info("  数据加载异常检查")
    logger.info("=" * 70)
    check_array("train_global_context", sample_batch["global_context"].numpy(), logger, context="全局上下文特征")
    check_array("train_candidate_matrix", sample_batch["candidate_matrix"].numpy(), logger, context="候选特征矩阵")
    check_labels("train_label", sample_batch["label"].numpy(), logger, context="训练标签")
    if "is_pick" in sample_batch:
        check_labels("train_is_pick", sample_batch["is_pick"].numpy(), logger, context="pick/ban标识")
    if "available_mask" in sample_batch:
        mask_arr = sample_batch["available_mask"].numpy()
        logger.info(f"  [数据检查] available_mask: shape={mask_arr.shape}, "
                    f"可用率={mask_arr.mean():.4f}")
    logger.info(f"  训练集 batches: {len(train_loader)}, 验证集 batches: {len(val_loader)}")
    logger.info("=" * 70)

    # 维度守卫
    from bp_recommendation.model_ban.dataloader_ban import BAN_CONTEXT_DIM, EXTENDED_CANDIDATE_DIM
    if context_dim_inferred != BAN_CONTEXT_DIM:
        logger.error(f"Feature dimension mismatch! context_dim={context_dim_inferred}, expected={BAN_CONTEXT_DIM}.")
        return
    if candidate_dim_inferred != EXTENDED_CANDIDATE_DIM:
        logger.error(f"Feature dimension mismatch! candidate_dim={candidate_dim_inferred}, expected={EXTENDED_CANDIDATE_DIM}.")
        return

    model = BPTacticalTransformer(
        vocab_size=vocab_size,
        context_dim=context_dim_inferred,
        candidate_dim=candidate_dim_inferred,
        h_dim=args.h_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        c_dim=args.c_dim,
        query_dim=args.query_dim,
        aux_loss_weight=args.aux_loss_weight,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
    ).to(device)

    # =====================================================================
    # [ALIGN CHECK] Token/Embedding 维度对齐验证 (Ban 模型)
    # =====================================================================
    bare_model = get_bare_model(model)
    _emb_ban = bare_model.bert.embeddings.word_embeddings.weight
    logger.info("=" * 70)
    logger.info("  [ALIGN CHECK] Ban 模型 Token/Embedding 维度对齐验证")
    logger.info("=" * 70)
    logger.info(f"  vocab_size (模型)     = {bare_model.vocab_size}")
    logger.info(f"  word_embeddings shape = {tuple(_emb_ban.shape)}")
    logger.info(f"  champion_start_idx    = {champion_start_idx}")
    _assert_ok = True
    if bare_model.vocab_size != vocab_size:
        logger.error(f"  [MISMATCH] model.vocab_size ({bare_model.vocab_size}) != vocab_size from file ({vocab_size})")
        _assert_ok = False
    if _emb_ban.shape[0] != bare_model.vocab_size:
        logger.error(f"  [MISMATCH] embedding dim 0 ({_emb_ban.shape[0]}) != vocab_size ({bare_model.vocab_size})")
        _assert_ok = False
    if champion_start_idx != 7:
        logger.error(f"  [MISMATCH] champion_start_idx should be 7 in v3, got {champion_start_idx}")
        _assert_ok = False
    if vocab_size != 180:
        logger.warning(f"  [WARNING] vocab_size={vocab_size}, expected 180 for v3 scheme")
    logger.info(f"  PAD=0, UNK=1, TOP/JNG/MID/BOT/SUP=2-6, champions=7..{vocab_size-1} ({vocab_size-champion_start_idx} heroes)")
    if _assert_ok:
        logger.info("  [ALIGN CHECK] ✓ Ban 模型所有维度对齐验证通过!")
    else:
        logger.critical("  [ALIGN CHECK] ✗ Ban 模型维度对齐失败! 训练已终止。")
        raise RuntimeError("Ban model Token/Embedding alignment check FAILED.")
    logger.info("=" * 70)

    # 【加速】Torch Compile 计算图编译
    if args.compile and device.type == "cuda":
        logger.info("Compiling model with torch.compile() for ultimate speed...")
        model = torch.compile(model)
    elif args.compile:
        logger.info(f"Skipping torch.compile (Not fully supported/stable on {device.type}).")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} (mask_cs={args.mask_cs}, {model_type_label.upper()} model)")

    best_val_raw = None
    if args.export_only:
        ckpt_path = os.path.join(CKPT_DIR, f"best_model_{model_type_label.lower()}.pt")
        if not os.path.exists(ckpt_path):
            logger.error(f"Cannot find trained model at {ckpt_path}. Please train without --export_only first.")
            return

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_to_load = get_bare_model(model)
        model_to_load.load_state_dict(ckpt["model_state_dict"])

        logger.info("Evaluating Val Dataloader to extract logits...")
        val_metrics, best_val_raw = evaluate(model, val_loader, device, mask_cs=args.mask_cs, use_amp=args.amp)
        logger.info(f"Val Check: Ban@10={val_metrics['Ban@10']:.4f}")

    else:
        # 【加速】使用 Fused AdamW
        use_fused = (device.type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=use_fused)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # 【加速】初始化 GradScaler
        scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

        best_metric = 0.0
        patience_counter = 0
        history = []
        best_epoch_val = 0
        prev_val_metric = None

        for epoch in range(1, args.epochs + 1):
            with log_context(Epoch=epoch):
                t0 = time.time()
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, scheduler, device,
                    args.grad_clip, mask_cs=args.mask_cs, scaler=scaler, use_amp=args.amp
                )

                if np.isnan(train_metrics['total_loss']) or np.isinf(train_metrics['total_loss']):
                    logger.error(f"  Epoch {epoch}: NaN/Inf loss detected! Training may be diverging.")

                if is_production:
                    val_metrics = None
                    val_raw = None
                else:
                    val_metrics, val_raw = evaluate(model, val_loader, device, mask_cs=args.mask_cs, use_amp=args.amp)
                elapsed = time.time() - t0

                epoch_record = {
                    "epoch": epoch,
                    "elapsed_sec": round(elapsed, 1),
                    "train": {k: round(v, 6) for k, v in train_metrics.items()},
                    "val": {k: round(v, 6) for k, v in val_metrics.items()} if val_metrics else {},
                }
                history.append(epoch_record)

                if is_production:
                    logger.info(
                        f"Epoch {epoch:03d} | "
                        f"Train Total: {train_metrics['total_loss']:.4f} "
                        f"(Main: {train_metrics['main_loss']:.4f}, Aux: {train_metrics['aux_loss']:.4f}) | "
                        f"{elapsed:.1f}s"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch:03d} | "
                        f"Train Total: {train_metrics['total_loss']:.4f} "
                        f"(Main: {train_metrics['main_loss']:.4f}, Aux: {train_metrics['aux_loss']:.4f}) | "
                        f"Val Total: {val_metrics['total_loss']:.4f} "
                        f"(Main: {val_metrics['main_loss']:.4f}, Aux: {val_metrics['aux_loss']:.4f}) | "
                        f"Ban@1: {val_metrics['Ban@1']:.4f} | Ban@5: {val_metrics['Ban@5']:.4f} | Ban@10: {val_metrics['Ban@10']:.4f} | "
                        f"{elapsed:.1f}s"
                    )

                    current_metric = val_metrics["Ban@10"]
                    if prev_val_metric is not None and current_metric < prev_val_metric - 0.02:
                        logger.warning(f"  Epoch {epoch}: Significant metric drop detected: {prev_val_metric:.4f} -> {current_metric:.4f}")
                    prev_val_metric = current_metric

                    if not is_production:
                        if current_metric > best_metric:
                            best_metric = current_metric
                            patience_counter = 0
                            best_val_raw = val_raw
                            best_epoch_val = epoch
                            versioned_ckpt = os.path.join(CKPT_DIR, f"best_model_{model_type_label.lower()}_ep{epoch}.pt")
                            canonical_ckpt = os.path.join(CKPT_DIR, f"best_model_{model_type_label.lower()}.pt")
                            save_checkpoint(model, optimizer, epoch, val_metrics, versioned_ckpt,
                                            candidate_dim=candidate_dim_inferred, context_dim=context_dim_inferred,
                                            vocab_size=vocab_size, role_token_start=role_token_start,
                                            champion_start_idx=champion_start_idx)
                            import shutil
                            shutil.copy2(versioned_ckpt, canonical_ckpt)
                            src_size = os.path.getsize(versioned_ckpt)
                            dst_size = os.path.getsize(canonical_ckpt)
                            if dst_size != src_size:
                                logger.warning(f"  shutil.copy2 verification failed (size mismatch: {src_size} vs {dst_size}), retrying with os.replace...")
                                shutil.copy(versioned_ckpt, canonical_ckpt)
                                dst_size = os.path.getsize(canonical_ckpt)
                            if dst_size != src_size:
                                raise RuntimeError(
                                    f"Failed to update canonical checkpoint {canonical_ckpt} "
                                    f"(size: {dst_size}, expected: {src_size})"
                                )
                            logger.info(f"  -> New best: Ban@10={best_metric:.4f} @ epoch {epoch} | saved to {os.path.basename(versioned_ckpt)} ({dst_size/1024/1024:.2f} MB)")
                        else:
                            patience_counter += 1
                            if patience_counter >= args.patience:
                                logger.info(f"Early stopping triggered at epoch {epoch} (patience={args.patience})")
                                logger.info(f"Best Ban@10={best_metric:.4f} achieved at epoch {best_epoch_val}")
                                break

        # 【修复 1】：生产模式训练结束后强制保存最后一轮权重 last_model.pt
        total_train_samples = len(train_loader.dataset)
        total_val_samples = len(val_loader.dataset) if val_loader else 0
        training_duration = sum(r.get("elapsed_sec", 0) for r in history)

        if is_production:
            last_ckpt_path = os.path.join(CKPT_DIR, f"last_model_{model_type_label.lower()}.pt")
            save_checkpoint(model, optimizer, args.epochs, {}, last_ckpt_path,
                            candidate_dim=candidate_dim_inferred, context_dim=context_dim_inferred,
                            vocab_size=vocab_size, role_token_start=role_token_start,
                            champion_start_idx=champion_start_idx)
            canonical_ckpt = os.path.join(CKPT_DIR, f"best_model_{model_type_label.lower()}.pt")
            import shutil
            shutil.copy2(last_ckpt_path, canonical_ckpt)
            ckpt_size = os.path.getsize(canonical_ckpt) / (1024*1024)
            logger.info("=" * 70)
            logger.info("PRODUCTION BAN MODEL TRAINING COMPLETE")
            logger.info("=" * 70)
            logger.info("  Final model   : %s", canonical_ckpt)
            logger.info("  Model size    : %.2f MB", ckpt_size)
            logger.info("  Epochs trained: %s", args.epochs)
            logger.info("  Training samples: %s", total_train_samples)
            logger.info("=" * 70)
        else:
            history_path = os.path.join(LOG_DIR, f"{run_name}_history.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            canonical_ckpt = os.path.join(CKPT_DIR, f'best_model_{model_type_label.lower()}.pt')
            ckpt_size = os.path.getsize(canonical_ckpt) / (1024*1024) if os.path.exists(canonical_ckpt) else 0
            logger.info("=" * 70)
            logger.info("DEVELOPMENT BAN MODEL TRAINING COMPLETE")
            logger.info("=" * 70)
            logger.info("  Best Ban@10   : %.4f", best_metric)
            logger.info("  Best epoch    : %s", best_epoch_val)
            logger.info("  Best model    : %s", canonical_ckpt)
            logger.info("  Model size    : %.2f MB", ckpt_size)
            logger.info("  History saved : %s", history_path)
            logger.info("=" * 70)

        if not is_production:
            save_best_params(
                model_type="ban",
                best_epoch=best_epoch_val,
                best_metric=best_metric,
                best_metric_name="Ban@10",
                model_subtype=model_subtype,
                architecture={
                    "h_dim": args.h_dim, "n_layers": args.n_layers,
                    "n_heads": args.n_heads, "c_dim": args.c_dim,
                    "query_dim": args.query_dim, "dropout": args.dropout,
                    "attention_dropout": args.attention_dropout,
                },
                optimizer={
                    "learning_rate": args.lr, "weight_decay": args.weight_decay,
                    "warmup_ratio": args.warmup_ratio, "grad_clip": args.grad_clip,
                },
                loss={
                    "aux_loss_weight": args.aux_loss_weight,
                },
                training={
                    "batch_size": args.batch_size, "patience": args.patience,
                    "val_ratio": args.val_ratio, "seed": args.seed,
                    "use_amp": args.amp, "num_workers": args.num_workers,
                },
            )
            logger.info(f"Best params saved to training config for [ban/{model_subtype}]")

        record_feature_dimensions(
            "ban", context_dim_inferred, candidate_dim_inferred, model_subtype
        )
        record_training_environment(
            get_config("ban", model_subtype),
            "ban",
            train_samples=total_train_samples,
            val_samples=total_val_samples,
            training_duration_sec=training_duration,
        )

        # 生产模式：单独记录实际使用的参数，便于审计
        if is_production:
            record_production_params(
                "ban", model_subtype,
                best_epoch=args.epochs,  # 【修复 1】：生产模式使用最后一轮 epoch
                num_epochs=args.epochs,
                val_ratio=args.val_ratio,
                train_samples=total_train_samples,
                val_samples=total_val_samples,
                device=str(device),
            )
            print_config_summary(get_config("ban", model_subtype))

        ckpt = torch.load(os.path.join(CKPT_DIR, f"best_model_{model_type_label.lower()}.pt"), map_location=device, weights_only=False)
        model_to_load = get_bare_model(model)
        model_to_load.load_state_dict(ckpt["model_state_dict"])

        # ========== PyTorch 2.x 原生 Checkpoint 部署 ==========
        # 直接使用 .pt checkpoint + torch.compile (可选) 进行推理
        # 不再使用 TorchScript/jit.trace, 依赖 PyTorch 2.x 原生执行图
        logger.info("PyTorch 2.x native checkpoint serving (no TorchScript export).")
        try:
            bare_model = get_bare_model(model)
            bare_model.eval()

            # 前向传播 sanity check: 验证 checkpoint 可正确推理
            dummy_seq = torch.zeros((1, 20), dtype=torch.long, device=device)
            dummy_ctx = torch.zeros((1, context_dim_inferred), dtype=torch.float32, device=device)
            dummy_cand = torch.zeros((1, vocab_size, candidate_dim_inferred), dtype=torch.float32, device=device)
            dummy_mask = torch.ones((1, vocab_size), dtype=torch.float32, device=device)
            dummy_hist = torch.full((1, 20), -1, dtype=torch.long, device=device)

            with torch.no_grad():
                test_out = bare_model(
                    dummy_seq, dummy_ctx, dummy_cand, dummy_mask, dummy_hist
                )
                logits_shape = test_out["logits"].shape
                logger.info(f"Checkpoint serving check: logits={logits_shape} "
                            f"(epoch={ckpt.get('epoch', '?')})")
            logger.info(f"PyTorch checkpoint ready: {os.path.join(CKPT_DIR, f'best_model_{model_type_label.lower()}.pt')}")
            bare_model.train()
        except Exception as ckpt_err:
            import traceback
            logger.warning(f"Checkpoint serving check 失败（不影响训练流程）: {ckpt_err}")
            logger.warning(f"详细 traceback:\n{traceback.format_exc()}")

    # ========== 导出部分 ==========
    os.makedirs(BAN_FEATURES_DIR, exist_ok=True)

    val_save_path = os.path.join(BAN_FEATURES_DIR, f"ALL_val_logits_{model_type_label.lower()}.npz")
    if best_val_raw is not None:
        np.savez_compressed(val_save_path, **best_val_raw)

    model.eval()
    train_save_path = os.path.join(BAN_FEATURES_DIR, f"ALL_train_logits_{model_type_label.lower()}.npz")

    if not args.export_only:
        train_loader_export, _ = create_dataloaders(
            val_ratio=args.val_ratio, batch_size=args.batch_size, logger=logger,
            force_unroll_train=True, num_workers=args.num_workers
        )
    else:
        train_loader_export = train_loader

    logger.info("Extracting Train Dataloader logits using pre-allocated tensors...")
    
    n_train_samples = len(train_loader_export.dataset)
    vocab_size = getattr(model, 'vocab_size', 180)

    
    # 从第一个 batch 动态获取 cand_dim
    sample_export = next(iter(train_loader_export))
    cand_dim = sample_export["candidate_matrix"].shape[-1]
    
    train_logits_arr = np.empty((n_train_samples, vocab_size), dtype=np.float32)
    train_masks_arr = np.empty((n_train_samples, vocab_size), dtype=np.float32)
    train_cands_arr = np.empty((n_train_samples, vocab_size, cand_dim), dtype=np.float32)
    train_labels_arr = np.empty(n_train_samples, dtype=np.int64)
    train_is_pick_arr = np.empty(n_train_samples, dtype=np.float32)
    train_time_weights_arr = np.ones(n_train_samples, dtype=np.float32)
    train_bp_steps_arr = np.full(n_train_samples, -1, dtype=np.int64)

    start_idx = 0
    device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

    with torch.no_grad():
        for batch in train_loader_export:
            batch_size = batch["label"].size(0)
            end_idx = start_idx + batch_size
            
            bp_seq = batch["bp_sequence"].to(device)
            ctx = batch["global_context"].to(device)
            cand = batch["candidate_matrix"].to(device)
            mask = batch["available_mask"].to(device)
            labels = batch["label"]
            is_pick = batch["is_pick"]

            history_positions = batch.get("history_positions")
            if history_positions is not None:
                history_positions = history_positions.to(device)
                
            cand_model = cand
            if args.mask_cs:
                cand_model = cand.clone()
                cand_model[:, :, CS_FEATURE_INDICES] = 0.0

            if device.type == 'mps':
                use_amp = False
                device_type = 'cpu' # autocast 引擎要求，但在 disabled 状态下不影响计算
            else:
                device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
            with torch.autocast(device_type=device_type, enabled=args.amp):
                out = model(bp_seq, ctx, cand_model, mask, history_positions)
            
            # 【极速拷贝】直接覆写到连续内存
            train_logits_arr[start_idx:end_idx] = out["logits"].float().cpu().numpy()
            train_masks_arr[start_idx:end_idx] = mask.cpu().numpy()
            train_cands_arr[start_idx:end_idx] = cand.cpu().numpy() # 保证导出的是完整特征
            train_labels_arr[start_idx:end_idx] = labels.numpy()
            train_is_pick_arr[start_idx:end_idx] = is_pick.numpy()
            if "time_weight" in batch:
                train_time_weights_arr[start_idx:end_idx] = batch["time_weight"].numpy()
            if "bp_step" in batch:
                train_bp_steps_arr[start_idx:end_idx] = batch["bp_step"].numpy()
            
            start_idx = end_idx

    np.savez_compressed(
        train_save_path, 
        logits=train_logits_arr, masks=train_masks_arr, 
        candidates=train_cands_arr, labels=train_labels_arr, is_pick=train_is_pick_arr,
        time_weights=train_time_weights_arr, bp_steps=train_bp_steps_arr
    )
    
    # 【修复 1】：生产模式无验证集指标，使用 0.0 作为占位
    if is_production:
        final_metric = 0.0
    elif args.export_only:
        final_metric = val_metrics['Ban@10']
    else:
        final_metric = best_metric
    logger.info(f"Logits export complete: {train_save_path}")
    return final_metric

if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))
    main()