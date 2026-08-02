"""
Transformer 特征提取脚本 (PIT 隔离版)
======================================
每折独立训练 NoCS Transformer 快照, 实现严格的 Point-In-Time 数据隔离。

流程:
  1. 对每个 OOT Fold, 仅使用训练窗口内的 context 数据训练 NoCS Transformer
  2. 用训练好的快照对训练集+测试集提取 4 种深层特征
  3. 保存为 parquet 供 train_walk_forward.py 使用

用法:
  python extract_transformer_features.py [--window 12] [--tf_epochs 15]
"""

import os
import sys
import time
import json
import hashlib
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# =====================================================================
# 路径配置
# =====================================================================
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTION_DIR = str(Path(__file__).parent.parent.resolve())

FEATURES_DIR = os.path.join(PREDICTION_DIR, "features")
WIDE_FEATURES_PATH = os.path.join(FEATURES_DIR, "ALL_prediction_wide_features.parquet")

MODEL_PICK_DIR = os.path.join(PROJECT_ROOT, "bp_recommendation", "model_pick")
NOCS_CKPT_PATH = os.path.join(MODEL_PICK_DIR, "checkpoints", "best_model_nocs.pt")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "cleaned_data", "champion_vocabulary.json")
POS_JSON = os.path.join(PROJECT_ROOT, "cleaned_data", "champion_position_mapping.json")

CONTEXT_PARQUET = os.path.join(PROJECT_ROOT, "bp_recommendation", "features", "ALL_context.parquet")
META_PARQUET = os.path.join(PROJECT_ROOT, "bp_recommendation", "features", "ALL_meta_store.parquet")
PLAYER_PARQUET = os.path.join(PROJECT_ROOT, "bp_recommendation", "features", "ALL_player_store.parquet")

TF_FEATURES_DIR = os.path.join(PREDICTION_DIR, "tf_features")
TF_SNAPSHOTS_DIR = os.path.join(PREDICTION_DIR, "tf_snapshots")
LOGS_DIR = os.path.join(PREDICTION_DIR, "logs")

LEAGUES = ["LPL", "LCK", "LEC"]

for d in [TF_FEATURES_DIR, TF_SNAPSHOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# =====================================================================
# 日志配置
# =====================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
from logger_config import get_logger, setup_logging

log = get_logger(__name__)


def log_info(msg):
    log.info(msg)

# =====================================================================
# 导入模型和数据加载器
# =====================================================================
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODEL_PICK_DIR)

from bp_recommendation.model_pick.model_pick import BPTacticalTransformerPick
from bp_recommendation.model_pick.dataloader_pick import BPRecommendationDataset
from bp_recommendation.feature_pipeline import load_champion_vocabulary

# =====================================================================
# NoCS 最佳超参数 (Optuna TPE 搜索结果)
# =====================================================================
NOCS_BEST_PARAMS = {
    "h_dim": 384,
    "c_dim": 64,
    "query_dim": 256,
    "n_layers": 2,
    "n_heads": 4,
    "dropout": 0.155,
    "attention_dropout": 0.117,
    "candidate_hidden": 256,
    "tactical_hidden": 256,
    "aux_loss_weight": 3.01,
    "ban_sample_weight": 0.034,
    "n_positions": 5,
}

# =====================================================================
# PIT 隔离的 NoCS Transformer 训练器
# =====================================================================
class PITNoCSTrainer:
    """每折独立训练 NoCS Transformer, 实现严格的 PIT 数据隔离。

    核心思路:
      - 从全局 context parquet 中筛选训练窗口内的比赛
      - 用这些比赛构建 DataLoader, 训练 NoCS Transformer
      - 保存快照到 tf_snapshots/fold_{i}_nocs.pt
    """

    def __init__(self, device="cpu", tf_epochs=15, tf_lr=3e-4, tf_batch_size=32,
                 tf_patience=8, force_retrain=False):
        self.device = torch.device(device)
        self.tf_epochs = tf_epochs
        self.tf_lr = tf_lr
        self.tf_batch_size = tf_batch_size
        self.tf_patience = tf_patience
        # 【修复 2】：强制重训练标志，跳过快照缓存
        self.force_retrain = force_retrain

    def _compute_snapshot_fingerprint(self, train_start, train_end):
        """计算快照指纹，用于判断快照是否与当前数据/配置完全一致。

        指纹包含：
          1. 训练窗口 (train_start~train_end)
          2. NOCS_BEST_PARAMS 超参数
          3. 训练超参数 (tf_epochs, tf_lr, tf_batch_size, tf_patience)
          4. 数据文件 (context/meta/player parquet) 的 mtime + size
          5. VOCAB_PATH 和 POS_JSON 的 mtime + size

        Returns:
            str: 16 字符 hex 指纹
        """
        fingerprint_parts = []

        # 1. 训练窗口
        fingerprint_parts.append(f"window={train_start}~{train_end}")

        # 2. NoCS 超参数 (排序后序列化，确保 key 顺序不影响 hash)
        params_str = json.dumps(NOCS_BEST_PARAMS, sort_keys=True)
        fingerprint_parts.append(f"params={params_str}")

        # 3. 训练超参数
        fingerprint_parts.append(
            f"train_cfg=epochs{self.tf_epochs}_lr{self.tf_lr}_bs{self.tf_batch_size}_pat{self.tf_patience}"
        )

        # 4. 数据文件指纹 (mtime + size)
        data_files = [CONTEXT_PARQUET, META_PARQUET, PLAYER_PARQUET, VOCAB_PATH, POS_JSON]
        for fpath in data_files:
            if os.path.exists(fpath):
                st = os.stat(fpath)
                fingerprint_parts.append(f"{os.path.basename(fpath)}:mtime{int(st.st_mtime)},size{st.st_size}")
            else:
                fingerprint_parts.append(f"{os.path.basename(fpath)}:MISSING")

        # 计算 MD5 并取前 16 字符
        raw = "|".join(fingerprint_parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _filter_context_by_date(self, context_df, meta_df, start_date, end_date):
        """根据日期范围筛选 context 数据, 实现 PIT 隔离。"""
        # meta_df 包含 gameid -> date 映射
        meta = meta_df[["gameid", "date"]].copy()
        meta["date"] = pd.to_datetime(meta["date"])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # 筛选日期范围内的比赛
        valid_gameids = meta[
            (meta["date"] >= start_dt) & (meta["date"] <= end_dt)
        ]["gameid"].unique()

        filtered = context_df[context_df["gameid"].isin(valid_gameids)].copy()
        filtered = filtered.sort_values("match_seq_idx").reset_index(drop=True)
        return filtered

    def _build_dataloaders(self, context_df, player_df):
        """从筛选后的 context 构建 DataLoader。"""
        # 80/20 划分训练/验证
        n = len(context_df)
        n_val = max(int(n * 0.15), 1)
        n_train = n - n_val

        train_ctx = context_df.iloc[:n_train].reset_index(drop=True)
        val_ctx = context_df.iloc[n_train:].reset_index(drop=True)

        # 加载 CS lookup (NoCS 模式不需要, 但 Dataset 接口需要)
        features_dir = os.path.dirname(CONTEXT_PARQUET)
        league_prefix = os.path.basename(CONTEXT_PARQUET).split('_')[0]

        def load_json_safe(filename):
            path = os.path.join(features_dir, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        counter_dict = load_json_safe(f"{league_prefix}_counter_lookup.json")
        synergy_dict = load_json_safe(f"{league_prefix}_synergy_lookup.json")
        grudge_store = load_json_safe(f"{league_prefix}_grudge_store.json")
        respect_store = load_json_safe(f"{league_prefix}_respect_store.json")
        hot_streak_store = load_json_safe(f"{league_prefix}_hot_streak_store.json")

        train_dataset = BPRecommendationDataset(
            train_ctx, player_df, pd.read_parquet(META_PARQUET),
            counter_dict, synergy_dict, VOCAB_PATH, POS_JSON,
            grudge_store=grudge_store, respect_store=respect_store,
            hot_streak_store=hot_streak_store,
            is_train=True,
        )
        val_dataset = BPRecommendationDataset(
            val_ctx, player_df, pd.read_parquet(META_PARQUET),
            counter_dict, synergy_dict, VOCAB_PATH, POS_JSON,
            grudge_store=grudge_store, respect_store=respect_store,
            hot_streak_store=hot_streak_store,
            is_train=False, anchor_date=train_dataset.anchor_date,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.tf_batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.tf_batch_size,
                                shuffle=False, num_workers=0)
        return train_loader, val_loader

    def _create_model(self, context_dim, candidate_dim):
        """创建 NoCS Transformer 模型。"""
        _, _, vocab_size, _, _ = load_champion_vocabulary(VOCAB_PATH)
        model = BPTacticalTransformerPick(
            vocab_size=vocab_size,
            context_dim=context_dim,
            candidate_dim=candidate_dim,
            **NOCS_BEST_PARAMS,
        ).to(self.device)
        return model

    def train_fold(self, fold_idx, train_start, train_end):
        """为指定折训练 NoCS Transformer 快照。

        Parameters:
            fold_idx: 折索引 (0-based)
            train_start: 训练窗口起始日期 (str, YYYY-MM-DD)
            train_end: 训练窗口结束日期 (str, YYYY-MM-DD)

        Returns:
            model: 训练好的模型
            snapshot_path: 快照保存路径
        """
        snapshot_path = os.path.join(TF_SNAPSHOTS_DIR, f"fold_{fold_idx}_nocs.pt")

        # 【修复 2】：计算当前快照指纹，用于判断是否需要重新训练
        expected_fingerprint = self._compute_snapshot_fingerprint(train_start, train_end)

        # 如果快照已存在, 验证指纹是否完全一致
        if os.path.exists(snapshot_path) and not self.force_retrain:
            ckpt = torch.load(snapshot_path, map_location=self.device, weights_only=False)
            stored_fingerprint = ckpt.get("snapshot_fingerprint")

            if stored_fingerprint == expected_fingerprint:
                log_info(f"  [Fold {fold_idx}] 快照指纹一致 ({expected_fingerprint}), 直接加载")
                context_dim = ckpt.get("context_dim", 15)
                candidate_dim = ckpt.get("candidate_dim", 31)
                model = self._create_model(context_dim, candidate_dim)
                model.load_state_dict(ckpt["model_state_dict"])
                return model, snapshot_path
            else:
                log_info(f"  [Fold {fold_idx}] 快照指纹不匹配 (stored={stored_fingerprint}, expected={expected_fingerprint})")
                log_info(f"  [Fold {fold_idx}] 数据/配置已变更, 重新训练快照")
                # 备份旧快照
                backup_path = snapshot_path.replace(".pt", "_stale_backup.pt")
                try:
                    os.rename(snapshot_path, backup_path)
                    log_info(f"  [Fold {fold_idx}] 旧快照已备份: {backup_path}")
                except Exception:
                    pass  # 备份失败不阻塞
        elif self.force_retrain:
            log_info(f"  [Fold {fold_idx}] --force_retrain 已启用, 忽略已有快照, 重新训练")

        log_info(f"  [Fold {fold_idx}] 开始 PIT 训练: 窗口 [{train_start} ~ {train_end}]")

        # 加载并筛选数据
        context_df = pd.read_parquet(CONTEXT_PARQUET)
        meta_df = pd.read_parquet(META_PARQUET)
        player_df = pd.read_parquet(PLAYER_PARQUET)

        filtered_ctx = self._filter_context_by_date(
            context_df, meta_df, train_start, train_end
        )
        log_info(f"  [Fold {fold_idx}] PIT 训练样本: {len(filtered_ctx)} 条 context")

        if len(filtered_ctx) < 50:
            log_info(f"  [Fold {fold_idx}] 样本不足 50, 使用全局 checkpoint 作为回退")
            ckpt = torch.load(NOCS_CKPT_PATH, map_location=self.device, weights_only=False)
            context_dim = ckpt.get("context_dim", 15)
            candidate_dim = ckpt.get("candidate_dim", 31)
            model = self._create_model(context_dim, candidate_dim)
            model.load_state_dict(ckpt["model_state_dict"])
            # 保存为快照 (标记为回退)
            ckpt["pit_fallback"] = True
            ckpt["train_window"] = f"{train_start}~{train_end}"
            ckpt["snapshot_fingerprint"] = expected_fingerprint  # 【修复 2】
            torch.save(ckpt, snapshot_path)
            return model, snapshot_path

        # 构建 DataLoader
        train_loader, val_loader = self._build_dataloaders(filtered_ctx, player_df)

        # 推断输入维度
        sample_batch = next(iter(train_loader))
        context_dim = sample_batch["global_context"].shape[-1]
        candidate_dim = sample_batch["candidate_matrix"].shape[-1]

        # 创建模型 (从全局 checkpoint 初始化, 加速收敛)
        model = self._create_model(context_dim, candidate_dim)
        if os.path.exists(NOCS_CKPT_PATH):
            log_info(f"  [Fold {fold_idx}] 从全局 checkpoint 初始化权重 (迁移学习)")
            global_ckpt = torch.load(NOCS_CKPT_PATH, map_location=self.device, weights_only=False)
            # 部分加载: 跳过维度不匹配的层
            model_sd = model.state_dict()
            loaded_keys = []
            skipped_keys = []
            for k, v in global_ckpt["model_state_dict"].items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(k)
            model.load_state_dict(model_sd)
            log_info(f"  [Fold {fold_idx}] 迁移加载: {len(loaded_keys)} keys loaded, {len(skipped_keys)} skipped")

        # 训练
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.tf_lr, weight_decay=0.01
        )
        total_steps = len(train_loader) * self.tf_epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.tf_epochs):
            # ---- Train ----
            model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                bp_seq = batch["bp_sequence"].to(self.device)
                ctx = batch["global_context"].to(self.device)
                cand = batch["candidate_matrix"].to(self.device)
                mask = batch["available_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                is_pick = batch["is_pick"].to(self.device)

                history_positions = batch.get("history_positions")
                if history_positions is not None:
                    history_positions = history_positions.to(self.device)
                last_ally_pos = batch.get("last_ally_pos")
                if last_ally_pos is not None:
                    last_ally_pos = last_ally_pos.to(self.device)

                out = model(bp_seq, ctx, cand, mask, history_positions,
                            last_ally_pos=last_ally_pos)
                main_loss, loss = model.compute_loss(
                    out["logits"], labels, out["aux_loss"], is_pick
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)

            # ---- Validate ----
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    bp_seq = batch["bp_sequence"].to(self.device)
                    ctx = batch["global_context"].to(self.device)
                    cand = batch["candidate_matrix"].to(self.device)
                    mask = batch["available_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    is_pick = batch["is_pick"].to(self.device)

                    history_positions = batch.get("history_positions")
                    if history_positions is not None:
                        history_positions = history_positions.to(self.device)
                    last_ally_pos = batch.get("last_ally_pos")
                    if last_ally_pos is not None:
                        last_ally_pos = last_ally_pos.to(self.device)

                    out = model(bp_seq, ctx, cand, mask, history_positions,
                                last_ally_pos=last_ally_pos)
                    _, loss = model.compute_loss(
                        out["logits"], labels, out["aux_loss"], is_pick
                    )
                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                log_info(f"  [Fold {fold_idx}] Epoch {epoch+1}/{self.tf_epochs} "
                         f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                    "context_dim": context_dim,
                    "candidate_dim": candidate_dim,
                    "pit_fallback": False,
                    "train_window": f"{train_start}~{train_end}",
                    "nocs_best_params": NOCS_BEST_PARAMS,
                    "snapshot_fingerprint": expected_fingerprint,  # 【修复 2】
                }
                torch.save(ckpt, snapshot_path)
            else:
                patience_counter += 1
                if patience_counter >= self.tf_patience:
                    log_info(f"  [Fold {fold_idx}] Early stopping at epoch {epoch+1}")
                    break

        # 加载最佳模型
        ckpt = torch.load(snapshot_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log_info(f"  [Fold {fold_idx}] 训练完成, best_val_loss={best_val_loss:.4f}")
        log_info(f"  [Fold {fold_idx}] 快照保存: {snapshot_path}")

        return model, snapshot_path


# =====================================================================
# Transformer 特征提取器 (Forward Hook 版)
# =====================================================================
class TransformerFeatureExtractor:
    """使用 Forward Hook 从 NoCS Transformer 提取 4 种深层特征。

    特征列表:
      1. tf_win_logits   - 胜负预测 logits (从 BP 序列隐状态投影)
      2. tf_cosine_sim   - 蓝红双方隐向量余弦相似度
      3. tf_blue_l2norm  - 蓝方隐向量 L2 范数
      4. tf_red_l2norm   - 红方隐向量 L2 范数
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self._captured_hidden = None
        self._register_hook()

    def _register_hook(self):
        """注册 Forward Hook 捕获 BERT last_hidden_state。"""
        def hook_fn(module, input, output):
            # DistilBertModel 的输出是 BaseModelOutput, last_hidden_state 是第一个元素
            if hasattr(output, "last_hidden_state"):
                self._captured_hidden = output.last_hidden_state.detach()
            elif isinstance(output, tuple):
                self._captured_hidden = output[0].detach()
            else:
                self._captured_hidden = output.detach()

        # 注册到 DistilBertModel
        self.model.bert.register_forward_hook(hook_fn)

    def _build_inputs_from_context(self, context_df):
        """从 context parquet 构建模型输入张量。"""
        n = len(context_df)
        vocab_size = self.model.vocab_size
        context_dim = self.model.context_mlp[0].in_features  # 15

        # --- 1. BP 序列 (B, 20) ---
        bp_cols = [f"bp_step{i}_champion_id" for i in range(20)]
        if all(col in context_df.columns for col in bp_cols):
            bp_sequence = context_df[bp_cols].values.astype(np.int64)
            bp_sequence = np.clip(bp_sequence, 0, vocab_size - 1)
        else:
            bp_sequence = np.zeros((n, 20), dtype=np.int64)
        bp_sequence = torch.as_tensor(bp_sequence, dtype=torch.long)

        # --- 2. Global Context (B, 15) ---
        global_context = np.zeros((n, context_dim), dtype=np.float32)

        # 联赛 one-hot (列 0-2)
        league_map = {lg: idx for idx, lg in enumerate(LEAGUES)}
        if "league" in context_df.columns:
            league_indices = context_df["league"].map(league_map).fillna(0).astype(int).values
            global_context[np.arange(n), league_indices] = 1.0

        # 战队统计特征 (列 3-12)
        stat_cols = [
            "blue_team_avg_ckpm", "blue_team_avg_golddiffat15",
            "blue_team_avg_gamelength", "blue_team_firstdragon_rate",
            "blue_team_firsttower_rate", "red_team_avg_ckpm",
            "red_team_avg_golddiffat15", "red_team_avg_gamelength",
            "red_team_firstdragon_rate", "red_team_firsttower_rate",
        ]
        stat_defaults = [0.7, 0, 1954, 0.5, 0.5, 0.7, 0, 1954, 0.5, 0.5]
        for j, (col, default) in enumerate(zip(stat_cols, stat_defaults)):
            if col in context_df.columns:
                global_context[:, 3 + j] = context_df[col].fillna(default).values.astype(np.float32)
            else:
                global_context[:, 3 + j] = default

        # 季后赛 & 先选边 (列 13-14)
        if "playoffs" in context_df.columns:
            global_context[:, 13] = context_df["playoffs"].fillna(0).values.astype(np.float32)
        if "first_pick_map_side" in context_df.columns:
            global_context[:, 14] = context_df["first_pick_map_side"].fillna(1).values.astype(np.float32)

        global_context = torch.as_tensor(global_context, dtype=torch.float32)

        # --- 3. Candidate Matrix (B, V, 31) - Dummy ---
        candidate_dim = self.model.candidate_mlp[0].in_features  # 31
        candidate_matrix = torch.zeros(n, vocab_size, candidate_dim, dtype=torch.float32)

        # --- 4. Available Mask (B, V) - Dummy ---
        available_mask = torch.ones(n, vocab_size, dtype=torch.float32)
        available_mask[:, :5] = 0.0  # 特殊 token 不可选

        return {
            "bp_sequence": bp_sequence,
            "global_context": global_context,
            "candidate_matrix": candidate_matrix,
            "available_mask": available_mask,
        }

    def extract_features(self, gameids_fold, context_df, batch_size=64):
        """对指定 gameid 列表提取 Transformer 深层特征。

        Parameters:
            gameids_fold: list[str], 需要提取特征的 gameid 列表
            context_df: DataFrame, context parquet 数据
            batch_size: int, 推理批大小

        Returns:
            dict: {gameid: {tf_win_logits, tf_cosine_sim, tf_blue_l2norm, tf_red_l2norm}}
        """
        self.model.eval()
        gameids_list = list(gameids_fold)

        # 筛选有 context 数据的比赛
        ctx_gameids = [gid for gid in gameids_list if gid in context_df["gameid"].values]
        no_ctx_gameids = [gid for gid in gameids_list if gid not in context_df["gameid"].values]

        results = {}

        # 无 context 数据的比赛 -> 使用默认值
        if no_ctx_gameids:
            for gid in no_ctx_gameids:
                results[gid] = {
                    "tf_win_logits": 0.0,
                    "tf_cosine_sim": 0.5,
                    "tf_blue_l2norm": 10.0,
                    "tf_red_l2norm": 10.0,
                }

        if not ctx_gameids:
            return results

        # 有 context 数据的比赛 -> 真实推理
        ctx_subset = context_df[context_df["gameid"].isin(ctx_gameids)].copy()
        ctx_subset = ctx_subset.drop_duplicates(subset=["gameid"], keep="first")
        ctx_subset = ctx_subset.set_index("gameid").loc[ctx_gameids].reset_index()

        all_features = []
        all_gids = []

        with torch.no_grad():
            for start in range(0, len(ctx_subset), batch_size):
                end = min(start + batch_size, len(ctx_subset))
                batch_df = ctx_subset.iloc[start:end]

                inputs = self._build_inputs_from_context(batch_df)
                # 移到设备
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                # 前向传播 (Hook 自动捕获 hidden states)
                _ = self.model(
                    inputs["bp_sequence"],
                    inputs["global_context"],
                    inputs["candidate_matrix"],
                    inputs["available_mask"],
                )

                hidden = self._captured_hidden  # (B, seq_len+5, h_dim)
                seq_len = inputs["bp_sequence"].shape[1]  # 20

                # BP 步骤隐状态 (去掉 role tokens)
                bp_hidden = hidden[:, :seq_len, :]  # (B, 20, h_dim)

                # 蓝方步骤: 0,2,4,6,9,11,13,15,17,18 (10步)
                blue_steps = [0, 2, 4, 6, 9, 11, 13, 15, 17, 18]
                # 红方步骤: 1,3,5,7,8,10,12,14,16,19 (10步)
                red_steps = [1, 3, 5, 7, 8, 10, 12, 14, 16, 19]

                # 限制到实际序列长度
                blue_steps = [s for s in blue_steps if s < seq_len]
                red_steps = [s for s in red_steps if s < seq_len]

                # 池化: 对蓝/红步骤的隐状态取平均
                blue_pooled = bp_hidden[:, blue_steps, :].mean(dim=1)  # (B, h_dim)
                red_pooled = bp_hidden[:, red_steps, :].mean(dim=1)    # (B, h_dim)

                # 投影到 query_dim
                blue_latent = self.model.bert_proj(blue_pooled).detach()  # (B, query_dim)
                red_latent = self.model.bert_proj(red_pooled).detach()    # (B, query_dim)

                # 计算特征
                # 1. tf_win_logits: 蓝方隐向量的 L2 范数 * sign(均值)
                tf_win_logits = blue_latent.norm(dim=1) * torch.sign(blue_latent.mean(dim=1))

                # 2. tf_cosine_sim: 蓝红隐向量余弦相似度
                tf_cosine_sim = F.cosine_similarity(blue_latent, red_latent, dim=1)

                # 3. tf_blue_l2norm: 蓝方隐向量 L2 范数
                tf_blue_l2norm = blue_latent.norm(dim=1)

                # 4. tf_red_l2norm: 红方隐向量 L2 范数
                tf_red_l2norm = red_latent.norm(dim=1)

                batch_features = torch.stack([
                    tf_win_logits, tf_cosine_sim, tf_blue_l2norm, tf_red_l2norm
                ], dim=1).cpu().numpy()

                all_features.append(batch_features)
                all_gids.extend(batch_df["gameid"].values.tolist())

        # 合并所有 batch
        all_features = np.concatenate(all_features, axis=0)

        for gid, feat in zip(all_gids, all_features):
            results[gid] = {
                "tf_win_logits": float(feat[0]),
                "tf_cosine_sim": float(feat[1]),
                "tf_blue_l2norm": float(feat[2]),
                "tf_red_l2norm": float(feat[3]),
            }

        return results


# =====================================================================
# 主流程
# =====================================================================
def main():
    setup_logging(log_dir=Path(LOGS_DIR).parent.parent / "logs", app_name="tf_extract",
                  console_level=logging.INFO, file_level=logging.DEBUG)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, f"tf_extract_run_{run_ts}.log"), encoding="utf-8")
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(run_handler)

    parser = argparse.ArgumentParser(
        description="Extract Transformer Features with PIT Isolation"
    )
    parser.add_argument("--window", type=int, default=12, choices=[6, 9, 12],
                        help="Training window in months (default: 12)")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="数据截止日期 YYYY-MM-DD (默认自动检测最新数据)")
    parser.add_argument("--tf_epochs", type=int, default=15,
                        help="Max training epochs per fold (default: 15)")
    parser.add_argument("--tf_lr", type=float, default=3e-4,
                        help="Learning rate for fold-level training (default: 3e-4)")
    parser.add_argument("--tf_batch_size", type=int, default=32,
                        help="Batch size for fold-level training (default: 32)")
    parser.add_argument("--tf_patience", type=int, default=8,
                        help="Early stopping patience (default: 8)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto)")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Force retrain all fold snapshots, ignoring cached ones")
    args = parser.parse_args()

    # 设备选择
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"

    # 动态生成 OOT Folds
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import _training as _train_cfg, resolve_oot_folds
    _train_cfg.window_months = args.window
    OOT_FOLDS, resolved_cutoff = resolve_oot_folds(_train_cfg, cutoff=args.cutoff)

    log_info("=" * 70)
    log_info("Transformer Feature Extraction (PIT Isolation)")
    log_info(f"  Cutoff Date   : {resolved_cutoff}" + (" (auto-detected)" if args.cutoff is None else ""))
    log_info(f"  Window        : {args.window} months")
    log_info(f"  TF Epochs     : {args.tf_epochs}")
    log_info(f"  TF LR         : {args.tf_lr}")
    log_info(f"  TF Batch Size : {args.tf_batch_size}")
    log_info(f"  TF Patience   : {args.tf_patience}")
    log_info(f"  Device        : {device}")
    log_info(f"  Force Retrain : {args.force_retrain}")
    log_info(f"  PIT Isolation : Per-fold NoCS Transformer training")
    log_info(f"  Cache Check   : Fingerprint (window + params + data mtime)")
    log_info("=" * 70)

    # 加载宽表特征 (获取 gameid 和日期)
    df = pd.read_parquet(WIDE_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # 按 cutoff 过滤数据
    cutoff_dt = pd.Timestamp(resolved_cutoff)
    n_before = len(df)
    df = df[df["date"] <= cutoff_dt].copy().reset_index(drop=True)
    if len(df) < n_before:
        log_info(f"  [Cutoff] 过滤 cutoff_date={resolved_cutoff} 之后的数据: {n_before} -> {len(df)} (移除 {n_before - len(df)} 条)")

    # 加载 context parquet
    log_info("Loading context parquet...")
    context_df = pd.read_parquet(CONTEXT_PARQUET)
    log_info(f"  Context rows: {len(context_df)}, unique games: {context_df['gameid'].nunique()}")

    window_months = args.window

    log_info(f"\nOOT Fold Definitions ({window_months}m window):")
    for i, (ts, te, tst, tse) in enumerate(OOT_FOLDS):
        log_info(f"  Fold {i}: Train [{ts} ~ {te}] | Test: {tst} ~ {tse}")

    # 初始化 PIT 训练器
    trainer = PITNoCSTrainer(
        device=device,
        tf_epochs=args.tf_epochs,
        tf_lr=args.tf_lr,
        tf_batch_size=args.tf_batch_size,
        tf_patience=args.tf_patience,
        force_retrain=args.force_retrain,
    )

    # 对每折: 训练快照 + 提取特征
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(OOT_FOLDS):
        log_info(f"\n{'='*70}")
        log_info(f"Fold {fold_idx}: Train [{train_start} ~ {train_end}] | Test [{test_start} ~ {test_end}]")
        log_info(f"{'='*70}")

        # Step 1: 训练/加载该折的 NoCS Transformer 快照
        model, _ = trainer.train_fold(fold_idx, train_start, train_end)

        # Step 2: 筛选该折的训练集+测试集 gameid
        train_start_dt = pd.Timestamp(train_start)
        train_end_dt = pd.Timestamp(train_end)
        test_start_dt = pd.Timestamp(test_start)
        test_end_dt = pd.Timestamp(test_end)

        train_mask = (df["date"] >= train_start_dt) & (df["date"] <= train_end_dt)
        test_mask = (df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)

        train_gameids = df.loc[train_mask, "gameid"].unique().tolist()
        test_gameids = df.loc[test_mask, "gameid"].unique().tolist()

        log_info(f"  Train gameids: {len(train_gameids)} | Test gameids: {len(test_gameids)}")

        # Step 3: 提取特征
        extractor = TransformerFeatureExtractor(model, device=device)

        train_features = extractor.extract_features(train_gameids, context_df)
        test_features = extractor.extract_features(test_gameids, context_df)

        # Step 4: 构建 DataFrame 并保存
        rows = []
        for gid in train_gameids:
            if gid in train_features:
                feat = train_features[gid]
                rows.append({
                    "gameid": gid, "fold": fold_idx, "split": "train",
                    "tf_win_logits": feat["tf_win_logits"],
                    "tf_cosine_sim": feat["tf_cosine_sim"],
                    "tf_blue_l2norm": feat["tf_blue_l2norm"],
                    "tf_red_l2norm": feat["tf_red_l2norm"],
                })
        for gid in test_gameids:
            if gid in test_features:
                feat = test_features[gid]
                rows.append({
                    "gameid": gid, "fold": fold_idx, "split": "test",
                    "tf_win_logits": feat["tf_win_logits"],
                    "tf_cosine_sim": feat["tf_cosine_sim"],
                    "tf_blue_l2norm": feat["tf_blue_l2norm"],
                    "tf_red_l2norm": feat["tf_red_l2norm"],
                })

        fold_df = pd.DataFrame(rows)
        output_path = os.path.join(TF_FEATURES_DIR, f"{fold_idx}_tf_features.parquet")
        fold_df.to_parquet(output_path, index=False)

        log_info(f"  Saved {len(fold_df)} rows to {output_path}")

        # 打印特征统计
        for col in ["tf_win_logits", "tf_cosine_sim", "tf_blue_l2norm", "tf_red_l2norm"]:
            tr = fold_df[fold_df["split"] == "train"][col]
            te = fold_df[fold_df["split"] == "test"][col]
            drift = abs(te.mean() - tr.mean()) / (tr.std() + 1e-8)
            log_info(f"    {col:20s}: train mu={tr.mean():8.3f} sigma={tr.std():7.3f} | "
                     f"test mu={te.mean():8.3f} sigma={te.std():7.3f} | drift={drift:.2f}σ")

    log.info("运行日志已通过 FileHandler 保存")
    log.info("Done!")


if __name__ == "__main__":
    main()
