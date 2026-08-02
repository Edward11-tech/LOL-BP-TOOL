"""
BPTacticalTransformerPick 训练脚本 (极致加速版)
====================================
支持 Mac M1 (MPS) 与 云服务器 (CUDA) 的自适应硬件加速。
新增参数:
    --amp           开启自动混合精度 (极大加速)
    --compile       开启 torch.compile 计算图编译 (仅 CUDA 生效)
    --num_workers   Dataloader 工作进程数 (推荐设为 4)
"""

# ============================================================
# 最佳超参数 (Optuna TPE 搜索 2026-06-17)
# CS:  Pick@10=0.7665 | NoCS: Pick@10=0.7663
# 开发模式与生产模式共用，生产模式仅 best_epoch 从配置文件读取
# ============================================================
BEST_PARAMS_CS = {
    "h_dim": 384, "n_layers": 3, "n_heads": 16,
    "query_dim": 128, "c_dim": 128,
    "dropout": 0.18118910790805948, "attention_dropout": 0.13638900372842316,
    "candidate_hidden": 256, "tactical_hidden": 256,
    "lr": 0.00035384612595255167, "weight_decay": 0.009093929525644107,
    "warmup_ratio": 0.15284688768272234, "grad_clip": 1.3886218532930636,
    "aux_loss_weight": 0.7090268572399898, "ban_sample_weight": 0.04050837781329675,
    "step6_downweight": 0.2534717113185624, "batch_size": 32,
    "n_epochs": 80, "patience": 25,
}

BEST_PARAMS_NOCS = {
    "h_dim": 384, "n_layers": 3, "n_heads": 12,
    "query_dim": 128, "c_dim": 128,
    "dropout": 0.18821167603120068, "attention_dropout": 0.12311823689532547,
    "candidate_hidden": 256, "tactical_hidden": 256,
    "lr": 0.0004296709540923395, "weight_decay": 0.04650497670980488,
    "warmup_ratio": 0.22856776713038615, "grad_clip": 1.4484419615378876,
    "aux_loss_weight": 0.9, "ban_sample_weight": 0.08233083651413035,
    "step6_downweight": 0.18314479445333554, "batch_size": 32,
    "n_epochs": 50, "patience": 20,
}
CS_FEATURE_INDICES = None  # 延迟初始化，见文件末尾导入后

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

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(TEST_DIR)))
sys.path.insert(0, TEST_DIR)

from bp_recommendation.model_pick.dataloader_pick import create_train_val_dataloaders
from bp_recommendation.model_pick.model_pick import BPTacticalTransformerPick
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

from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())

# 共享数据异常检测工具
sys.path.insert(0, _PROJECT_ROOT)
from data_checks import check_array, check_labels, check_predictions
from logger_config import get_logger, setup_logging, log_context, timed

log = get_logger(__name__)
SHARED_FEATURES_DIR = os.path.join(os.path.dirname(TEST_DIR), "features")
CLEANED_DIR = os.path.join(_PROJECT_ROOT, "cleaned_data")
PICK_DIR = TEST_DIR
PICK_FEATURES_DIR = os.path.join(PICK_DIR, "features")
VOCAB_PATH = os.path.join(CLEANED_DIR, "champion_vocabulary.json")
POS_JSON = os.path.join(CLEANED_DIR, "champion_position_mapping.json")
CKPT_DIR = os.path.join(PICK_DIR, "checkpoints")
LOG_DIR = os.path.join(PICK_DIR, "logs")

# CS 特征索引: ally_synergy(15) ~ ally_counter(18)，用于 NoCS 模型零化上下文敏感特征
CS_FEATURE_INDICES = slice(CANDIDATE_FEAT_MAP["ally_synergy"], CANDIDATE_FEAT_MAP["ally_counter"] + 1)


def setup_logger(run_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{run_name}_{timestamp}.log")

    logger = get_logger("BPTrain_Pick")
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

    # 【加速】传入 num_workers 以开启异步数据加载
    train_loader, val_loader = create_train_val_dataloaders(
        context_parquet, meta_parquet, player_parquet,
        vocab_path, pos_json,
        batch_size=batch_size, num_workers=num_workers, val_ratio=val_ratio,
        force_unroll_train=force_unroll_train,
    )

    if logger:
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)} (Workers: {num_workers})")

    return train_loader, val_loader


def recall_at_k(logits, labels, available_mask, k, is_pick=None):
    masked = logits.clone()
    masked[available_mask == 0] = -1e9

    _, topk_indices = masked.topk(k, dim=-1)
    hit = (topk_indices == labels.unsqueeze(1)).any(dim=1)

    if is_pick is not None:
        hit = hit[is_pick.bool()] if is_pick.sum() > 0 else hit[:0]

    if hit.numel() == 0:
        return 0.0
    return hit.float().mean().item()


@torch.no_grad()
def evaluate(model, val_loader, device, mask_cs=False, use_amp=False):
    model.eval()
    
    n_samples = len(val_loader.dataset)
    vocab_size = getattr(model, 'vocab_size', 175) # 兼容编译后的模型
    
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

    # 提取正确的设备类型用于 autocast
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

        history_positions = batch.get("history_positions")
        if history_positions is not None:
            history_positions = history_positions.to(device)

        time_weight = batch.get("time_weight")
        if time_weight is not None:
            time_weight = time_weight.to(device)
            all_time_weights[start_idx:end_idx] = time_weight.cpu()

        bp_step = batch.get("bp_step")
        if bp_step is not None:
            all_bp_steps[start_idx:end_idx] = bp_step.cpu()

        last_ally_pos = batch.get("last_ally_pos")
        if last_ally_pos is not None:
            last_ally_pos = last_ally_pos.to(device)

        # 纯张量数学操作，torch.compile 对这种静态操作的优化极佳
        cand_model = cand
        if mask_cs:
            mask_tensor = torch.ones_like(cand)
            mask_tensor[:, :, CS_FEATURE_INDICES] = 0.0
            cand_model = cand * mask_tensor 

        # 【加速】推理时启用 AMP
        with torch.autocast(device_type=device_type, enabled=use_amp):
            out = model(bp_seq, ctx, cand_model, mask, history_positions, last_ally_pos=last_ally_pos)
            main_loss, total_loss = model.compute_loss(out["logits"], labels, out["aux_loss"], is_pick, time_weight)

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

    pick_r1 = recall_at_k(all_logits, all_labels, all_masks, 1, is_pick_flag)
    pick_r3 = recall_at_k(all_logits, all_labels, all_masks, 3, is_pick_flag)
    pick_r5 = recall_at_k(all_logits, all_labels, all_masks, 5, is_pick_flag)
    pick_r10 = recall_at_k(all_logits, all_labels, all_masks, 10, is_pick_flag)
    pick_r20 = recall_at_k(all_logits, all_labels, all_masks, 20, is_pick_flag)
    ban_r10 = recall_at_k(all_logits, all_labels, all_masks, 10, is_ban_flag)
    overall_r10 = recall_at_k(all_logits, all_labels, all_masks, 10)

    metrics = {
        "main_loss": total_main_loss / max(n_batches, 1),
        "aux_loss": total_aux_loss / max(n_batches, 1),
        "total_loss": total_val_loss / max(n_batches, 1),
        "Pick@1": pick_r1,
        "Pick@3": pick_r3,
        "Pick@5": pick_r5,
        "Pick@10": pick_r10,
        "Pick@20": pick_r20,
        "Ban@10": ban_r10,
        "Overall@10": overall_r10,
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
                    step6_downweight=1.0, mask_cs=False, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    n_batches = 0
    
    device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

    for batch in train_loader:
        # 内存拷贝
        bp_seq = batch["bp_sequence"].to(device)
        ctx = batch["global_context"].to(device)
        cand = batch["candidate_matrix"].to(device)
        mask = batch["available_mask"].to(device)
        labels = batch["label"].to(device)
        is_pick = batch["is_pick"].to(device)

        history_positions = batch.get("history_positions")
        if history_positions is not None:
            history_positions = history_positions.to(device)

        time_weight = batch.get("time_weight")
        if time_weight is not None:
            time_weight = time_weight.to(device)

        last_ally_pos = batch.get("last_ally_pos")
        if last_ally_pos is not None:
            last_ally_pos = last_ally_pos.to(device)

        # 纯张量数学操作，torch.compile 对这种静态操作的优化极佳
        cand_model = cand
        if mask_cs:
            mask_tensor = torch.ones_like(cand)
            mask_tensor[:, :, CS_FEATURE_INDICES] = 0.0
            cand_model = cand * mask_tensor 

        sample_weight = None
        if step6_downweight < 1.0:
            bp_step = batch.get("bp_step")
            if bp_step is not None:
                sample_weight = torch.ones(labels.shape[0], device=device)
                step6_mask = (bp_step == 6)
                sample_weight[step6_mask] = step6_downweight

        tuple_partners = batch.get("tuple_partner")
        if tuple_partners is not None:
            tuple_partners = tuple_partners.to(device)

        bp_step_for_loss = batch.get("bp_step")
        if bp_step_for_loss is not None:
            bp_step_for_loss = bp_step_for_loss.to(device)

        # 【加速】极速梯度清零
        optimizer.zero_grad(set_to_none=True)

        # 【加速】混合精度前向传播
        with torch.autocast(device_type=device_type, enabled=use_amp):
            out = model(bp_seq, ctx, cand_model, mask, history_positions, last_ally_pos=last_ally_pos)
            main_loss, loss = model.compute_loss(out["logits"], labels, out["aux_loss"], is_pick, time_weight, sample_weight,
                                                  tuple_partners=tuple_partners, bp_steps=bp_step_for_loss)

        # 【加速】反向传播 (带/不带 Scaler)
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


def get_bare_model(model):
    """剥离 DDP 的 .module 和 torch.compile 的 ._orig_mod 包装"""
    bare_model = model
    if hasattr(bare_model, "_orig_mod"):
        bare_model = bare_model._orig_mod
    if hasattr(bare_model, "module"):
        bare_model = bare_model.module
    return bare_model


def save_checkpoint(model, optimizer, epoch, metrics, path, context_dim=None, candidate_dim=None):
    bare_model = get_bare_model(model)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": bare_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if context_dim is not None:
        ckpt["context_dim"] = context_dim
    if candidate_dim is not None:
        ckpt["candidate_dim"] = candidate_dim
    torch.save(ckpt, path)


def main(override_args=None):
    parser = argparse.ArgumentParser(description="Train BPTacticalTransformerPick")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3.54e-4)
    parser.add_argument("--weight_decay", type=float, default=0.00909)
    parser.add_argument("--warmup_ratio", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.26)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mask_cs", action="store_true", default=False)
    parser.add_argument("--export_only", action="store_true", default=False)
    
    # 架构超参数
    parser.add_argument("--h_dim", type=int, default=384)
    parser.add_argument("--c_dim", type=int, default=128)
    parser.add_argument("--query_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.181)
    parser.add_argument("--attention_dropout", type=float, default=0.136)
    parser.add_argument("--candidate_hidden", type=int, default=256)
    parser.add_argument("--tactical_hidden", type=int, default=256)
    parser.add_argument("--aux_loss_weight", type=float, default=0.709)
    parser.add_argument("--ban_sample_weight", type=float, default=0.0405)
    parser.add_argument("--step6_downweight", type=float, default=0.253)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None, help="自定义运行名称，覆盖自动生成的 run_name")
    
    # 【新增】硬件加速相关参数
    parser.add_argument("--amp", action="store_true", default=False, help="开启自动混合精度加速 (AMP)")
    parser.add_argument("--compile", action="store_true", default=False, help="使用 torch.compile 编译计算图 (仅限 Linux/CUDA)")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader 数据读取进程数 (非 CUDA 推荐 0, CUDA 推荐 2~4)")
    parser.add_argument("--production", action="store_true", default=None,
                        help="强制启用生产模式 (覆盖环境变量 BP_PRODUCTION_MODE)")
    
    # 【修复】：支持外部 HPO 框架传入超参数
    if override_args is not None:
        args = parser.parse_args(override_args)
    else:
        # 如果是在 Jupyter 等特殊环境中直接运行，防止 sys.argv 报错
        try:
            args = parser.parse_args()
        except SystemExit:
            # Jupyter 环境下的 fallback
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
    best_params = BEST_PARAMS_NOCS if args.mask_cs else BEST_PARAMS_CS
    args.h_dim = best_params["h_dim"]
    args.c_dim = best_params["c_dim"]
    args.query_dim = best_params["query_dim"]
    args.n_layers = best_params["n_layers"]
    args.n_heads = best_params["n_heads"]
    args.dropout = best_params["dropout"]
    args.attention_dropout = best_params["attention_dropout"]
    args.candidate_hidden = best_params["candidate_hidden"]
    args.tactical_hidden = best_params["tactical_hidden"]
    args.lr = best_params["lr"]
    args.weight_decay = best_params["weight_decay"]
    args.warmup_ratio = best_params["warmup_ratio"]
    args.grad_clip = best_params["grad_clip"]
    args.aux_loss_weight = best_params["aux_loss_weight"]
    args.ban_sample_weight = best_params["ban_sample_weight"]
    args.step6_downweight = best_params["step6_downweight"]
    args.batch_size = best_params["batch_size"]
    args.epochs = best_params["n_epochs"]
    args.patience = best_params["patience"]

    # MPS 兼容：设置多进程 start_method='spawn' 避免死锁
    if device.type == "mps" and args.num_workers > 0:
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    run_name = f"train_pick_{model_type_label.lower()}"
    logger, log_path = setup_logger(run_name)

    from datetime import datetime as _dt
    training_date = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    if is_production:
        config = get_config("pick", model_subtype)
        args.epochs = get_production_num_epochs(config)
        args.val_ratio = get_production_val_ratio()
        args.patience = args.epochs + 10

        logger.info("=" * 70)
        logger.info("=" * 70)
        logger.info("  PICK MODEL - PRODUCTION MODE (Blind Training)")
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
        logger.info("  PICK MODEL - DEVELOPMENT MODE (Search & Validate)")
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
    os.makedirs(PICK_FEATURES_DIR, exist_ok=True)
    effective_ckpt_dir = args.ckpt_dir or CKPT_DIR
    os.makedirs(effective_ckpt_dir, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        features_dir=SHARED_FEATURES_DIR,
        val_ratio=args.val_ratio, batch_size=args.batch_size, logger=logger,
        force_unroll_train=args.export_only, num_workers=args.num_workers
    )

    _, _, vocab_size, _, _ = load_champion_vocabulary(VOCAB_PATH)

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

    model = BPTacticalTransformerPick(
        vocab_size=vocab_size,
        context_dim=context_dim_inferred,
        candidate_dim=candidate_dim_inferred,
        h_dim=args.h_dim,
        c_dim=args.c_dim,
        query_dim=args.query_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        candidate_hidden=args.candidate_hidden,
        tactical_hidden=args.tactical_hidden,
        aux_loss_weight=args.aux_loss_weight,
        ban_sample_weight=args.ban_sample_weight,
    ).to(device)

    # 【加速】Torch Compile 计算图编译
    if args.compile and device.type == "cuda":
        logger.info("Compiling model with torch.compile() for ultimate speed...")
        model = torch.compile(model)
    elif args.compile:
        logger.info(f"Skipping torch.compile (Not fully supported/stable on {device.type}).")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    best_val_raw = None
    if args.export_only:
        ckpt_path = os.path.join(effective_ckpt_dir, f"best_model_{model_type_label.lower()}.pt")
        if not os.path.exists(ckpt_path):
            logger.error(f"Cannot find trained model at {ckpt_path}. Please train without --export_only first.")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # 兼容性读取
        model_to_load = model.module if hasattr(model, 'module') else model
        bare_model = get_bare_model(model_to_load)
        bare_model.load_state_dict(ckpt["model_state_dict"])
        
        val_metrics, best_val_raw = evaluate(bare_model, val_loader, device, mask_cs=args.mask_cs, use_amp=args.amp)
        logger.info(f"Val Check: Pick@10={val_metrics['Pick@10']:.4f}")

    else:
        # 【加速】使用 Fused AdamW (仅 CUDA 且支持的 PyTorch 版本可用)
        use_fused = (device.type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        if use_fused:
            logger.info("Using Fused AdamW Optimizer.")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=use_fused)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # 【加速】初始化 GradScaler (仅用于 CUDA)
        scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None
        if args.amp:
            logger.info(f"AMP Enabled. Scaling backend: {'GradScaler' if scaler else 'MPS Autocast natively'}")

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
                    args.grad_clip, args.step6_downweight,
                    mask_cs=args.mask_cs, scaler=scaler, use_amp=args.amp
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
                        f"Train Loss: {train_metrics['total_loss']:.4f} "
                        f"(Main: {train_metrics['main_loss']:.4f}, Aux: {train_metrics['aux_loss']:.4f}) | "
                        f"{elapsed:.1f}s"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch:03d} | "
                        f"Train Loss: {train_metrics['total_loss']:.4f} "
                        f"(Main: {train_metrics['main_loss']:.4f}, Aux: {train_metrics['aux_loss']:.4f}) | "
                        f"Val Loss: {val_metrics['total_loss']:.4f} | "
                        f"Pick@1: {val_metrics['Pick@1']:.4f} | Pick@5: {val_metrics['Pick@5']:.4f} | Pick@10: {val_metrics['Pick@10']:.4f} | "
                        f"Ban@10: {val_metrics['Ban@10']:.4f} | "
                        f"{elapsed:.1f}s"
                    )

                    current_metric = val_metrics["Pick@10"]
                    if prev_val_metric is not None and current_metric < prev_val_metric - 0.02:
                        logger.warning(f"  Epoch {epoch}: Significant metric drop detected: {prev_val_metric:.4f} -> {current_metric:.4f}")
                    prev_val_metric = current_metric

                if not is_production:
                    current_metric = val_metrics["Pick@10"]
                    if current_metric > best_metric:
                        best_metric = current_metric
                        patience_counter = 0
                        best_val_raw = val_raw
                        best_epoch_val = epoch
                        versioned_ckpt = os.path.join(effective_ckpt_dir, f"best_model_{model_type_label.lower()}_ep{epoch}.pt")
                        canonical_ckpt = os.path.join(effective_ckpt_dir, f"best_model_{model_type_label.lower()}.pt")
                        save_checkpoint(model, optimizer, epoch, val_metrics, versioned_ckpt,
                                        context_dim=context_dim_inferred, candidate_dim=candidate_dim_inferred)
                        import shutil
                        shutil.copy2(versioned_ckpt, canonical_ckpt)
                        src_size = os.path.getsize(versioned_ckpt)
                        dst_size = os.path.getsize(canonical_ckpt)
                        if dst_size != src_size:
                            logger.warning(f"  shutil.copy2 verification failed (size mismatch: {src_size} vs {dst_size}), retrying...")
                            shutil.copy(versioned_ckpt, canonical_ckpt)
                            dst_size = os.path.getsize(canonical_ckpt)
                        if dst_size != src_size:
                            raise RuntimeError(
                                f"Failed to update canonical checkpoint {canonical_ckpt} "
                                f"(size: {dst_size}, expected: {src_size})"
                            )
                        logger.info(f"  -> New best: Pick@10={best_metric:.4f} @ epoch {epoch} | saved to {os.path.basename(versioned_ckpt)} ({dst_size/1024/1024:.2f} MB)")
                    else:
                        patience_counter += 1
                        if patience_counter >= args.patience:
                            logger.info(f"Early stopping triggered at epoch {epoch} (patience={args.patience})")
                            logger.info(f"Best Pick@10={best_metric:.4f} achieved at epoch {best_epoch_val}")
                            break

        # 【修复 1】：生产模式训练结束后强制保存最后一轮权重 last_model.pt
        total_train_samples = len(train_loader.dataset)
        if is_production:
            last_ckpt_path = os.path.join(effective_ckpt_dir, f"last_model_{model_type_label.lower()}.pt")
            save_checkpoint(model, optimizer, args.epochs, {}, last_ckpt_path,
                            context_dim=context_dim_inferred, candidate_dim=candidate_dim_inferred)
            canonical_ckpt = os.path.join(effective_ckpt_dir, f"best_model_{model_type_label.lower()}.pt")
            import shutil
            shutil.copy2(last_ckpt_path, canonical_ckpt)
            ckpt_size = os.path.getsize(canonical_ckpt) / (1024*1024)
            logger.info("=" * 70)
            logger.info("PRODUCTION PICK MODEL TRAINING COMPLETE")
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
            canonical_ckpt = os.path.join(effective_ckpt_dir, f'best_model_{model_type_label.lower()}.pt')
            ckpt_size = os.path.getsize(canonical_ckpt) / (1024*1024) if os.path.exists(canonical_ckpt) else 0
            logger.info("=" * 70)
            logger.info("DEVELOPMENT PICK MODEL TRAINING COMPLETE")
            logger.info("=" * 70)
            logger.info("  Best Pick@10  : %.4f", best_metric)
            logger.info("  Best epoch    : %s", best_epoch_val)
            logger.info("  Best model    : %s", canonical_ckpt)
            logger.info("  Model size    : %.2f MB", ckpt_size)
            logger.info("  History saved : %s", history_path)
            logger.info("=" * 70)

        # === 开发模式：保存最佳参数到配置文件 ===
        if not is_production:
            save_best_params(
                model_type="pick",
                best_epoch=best_epoch_val,
                best_metric=best_metric,
                best_metric_name="Pick@10",
                model_subtype=model_subtype,
                architecture={
                    "h_dim": args.h_dim, "c_dim": args.c_dim,
                    "query_dim": args.query_dim, "n_layers": args.n_layers,
                    "n_heads": args.n_heads, "dropout": args.dropout,
                    "attention_dropout": args.attention_dropout,
                    "candidate_hidden": args.candidate_hidden,
                    "tactical_hidden": args.tactical_hidden,
                },
                optimizer={
                    "learning_rate": args.lr, "weight_decay": args.weight_decay,
                    "warmup_ratio": args.warmup_ratio, "grad_clip": args.grad_clip,
                },
                loss={
                    "aux_loss_weight": args.aux_loss_weight,
                    "ban_sample_weight": args.ban_sample_weight,
                    "step6_downweight": args.step6_downweight,
                },
                training={
                    "batch_size": args.batch_size, "patience": args.patience,
                    "val_ratio": args.val_ratio, "seed": args.seed,
                    "use_amp": args.amp, "num_workers": args.num_workers,
                },
            )
            logger.info(f"Best params saved to training config for [pick/{model_subtype}]")

        # === 记录训练环境信息 ===
        total_train_samples = len(train_loader.dataset)
        total_val_samples = len(val_loader.dataset) if val_loader else 0
        training_duration = sum(r.get("elapsed_sec", 0) for r in history)
        record_feature_dimensions(
            "pick", context_dim_inferred, candidate_dim_inferred, model_subtype
        )
        record_training_environment(
            get_config("pick", model_subtype),
            "pick",
            train_samples=total_train_samples,
            val_samples=total_val_samples,
            training_duration_sec=training_duration,
        )

        # 生产模式：单独记录实际使用的参数，便于审计
        if is_production:
            record_production_params(
                "pick", model_subtype,
                best_epoch=args.epochs,  # 【修复 1】：生产模式使用最后一轮 epoch
                num_epochs=args.epochs,
                val_ratio=args.val_ratio,
                train_samples=total_train_samples,
                val_samples=total_val_samples,
                device=str(device),
            )
            print_config_summary(get_config("pick", model_subtype))

        # 训练结束后读取最佳模型准备导出
        ckpt = torch.load(os.path.join(effective_ckpt_dir, f"best_model_{model_type_label.lower()}.pt"), map_location=device, weights_only=False)
        model_to_load = model.module if hasattr(model, 'module') else model
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
            dummy_last_ally = torch.tensor([-1], dtype=torch.long, device=device)

            with torch.no_grad():
                test_out = bare_model(
                    dummy_seq, dummy_ctx, dummy_cand, dummy_mask,
                    dummy_hist, dummy_last_ally
                )
                logits_shape = test_out["logits"].shape
                logger.info(f"Checkpoint serving check: logits={logits_shape} "
                            f"(epoch={ckpt.get('epoch', '?')})")
            logger.info(f"PyTorch checkpoint ready: {os.path.join(effective_ckpt_dir, f'best_model_{model_type_label.lower()}.pt')}")
            bare_model.train()
        except Exception as ckpt_err:
            import traceback
            logger.warning(f"Checkpoint serving check 失败（不影响训练流程）: {ckpt_err}")
            logger.warning(f"详细 traceback:\n{traceback.format_exc()}")

    # ========== 导出部分 ==========
    os.makedirs(PICK_FEATURES_DIR, exist_ok=True)
    val_save_path = os.path.join(PICK_FEATURES_DIR, f"ALL_val_logits_{model_type_label.lower()}.npz")
    if best_val_raw is not None:
        np.savez_compressed(val_save_path, **best_val_raw)

    train_save_path = os.path.join(PICK_FEATURES_DIR, f"ALL_train_logits_{model_type_label.lower()}.npz")
    if not args.export_only:
        train_loader_export, _ = create_dataloaders(
            val_ratio=args.val_ratio, batch_size=args.batch_size, logger=logger,
            force_unroll_train=True, num_workers=args.num_workers
        )
    else:
        train_loader_export = train_loader

    logger.info("Extracting Train Dataloader logits...")
    model.eval()
    
    n_train_samples = len(train_loader_export.dataset)
    vocab_size = getattr(model, 'vocab_size', 175)
    
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

            last_ally_pos = batch.get("last_ally_pos")
            if last_ally_pos is not None:
                last_ally_pos = last_ally_pos.to(device)

            # 纯张量数学操作，torch.compile 对这种静态操作的优化极佳
            cand_model = cand
            if args.mask_cs:
                mask_tensor = torch.ones_like(cand)
                mask_tensor[:, :, CS_FEATURE_INDICES] = 0.0
                cand_model = cand * mask_tensor 

            with torch.autocast(device_type=device_type, enabled=args.amp):
                out = model(bp_seq, ctx, cand_model, mask, history_positions, last_ally_pos=last_ally_pos)
            
            train_logits_arr[start_idx:end_idx] = out["logits"].float().cpu().numpy()
            train_masks_arr[start_idx:end_idx] = mask.cpu().numpy()
            train_cands_arr[start_idx:end_idx] = cand.cpu().numpy()
            train_labels_arr[start_idx:end_idx] = labels.numpy()
            train_is_pick_arr[start_idx:end_idx] = is_pick.numpy()
            
            if "time_weight" in batch:
                train_time_weights_arr[start_idx:end_idx] = batch["time_weight"].numpy()
            if "bp_step" in batch:
                train_bp_steps_arr[start_idx:end_idx] = batch["bp_step"].numpy()
            
            start_idx = end_idx

    np.savez_compressed(
        train_save_path, logits=train_logits_arr, masks=train_masks_arr, 
        candidates=train_cands_arr, labels=train_labels_arr, is_pick=train_is_pick_arr, 
        time_weights=train_time_weights_arr, bp_steps=train_bp_steps_arr
    )

    # 【修复 1】：生产模式无验证集指标，使用 0.0 作为占位
    if is_production:
        final_metric = 0.0
    elif args.export_only:
        final_metric = val_metrics['Pick@10']
    else:
        final_metric = best_metric
    logger.info(f"Logits export complete: {train_save_path}")

    return final_metric

if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))
    main()