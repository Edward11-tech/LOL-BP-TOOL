"""
生产 Transformer 快照导出脚本
================================
从推荐模型 (bp_recommendation) 的 NoCS Transformer checkpoint 导出生产快照,
供胜率预测模型推理时提取 TF 特征使用。

核心思路:
  推荐模型训练时已经用全量数据训练了 NoCS Transformer,
  无需重复训练, 直接导出其 checkpoint 即可。

导出流程:
  1. 从 bp_recommendation/model_pick/checkpoints/best_model_nocs.pt 复制快照
  2. 添加生产标记 (production=True, cutoff_date 等)
  3. 保存到 tf_snapshots/production_nocs.pt
  4. 对全量数据提取 TF 特征, 保存到 tf_features/production_tf_features.parquet
  5. feature_builder.py 会自动优先加载 production_nocs.pt

用法:
  python export_production_transformer.py                     # 自动检测最新数据日期
  python export_production_transformer.py --cutoff 2026-06-07 # 指定截止日期
  python export_production_transformer.py --skip-extract      # 仅导出快照, 不提取特征
  python export_production_transformer.py --device mps        # 指定设备
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path

# 必须在 logger_config 导入前设置 sys.path
_PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logger_config import get_logger, setup_logging

FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

import torch
import torch.nn.functional as F

# =====================================================================
# 路径配置
# =====================================================================
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# 推荐模型 checkpoint 路径
MODEL_PICK_DIR = os.path.join(PROJECT_ROOT, "bp_recommendation", "model_pick")
NOCS_CKPT_PATH = os.path.join(MODEL_PICK_DIR, "checkpoints", "best_model_nocs.pt")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "cleaned_data", "champion_vocabulary.json")
POS_JSON = os.path.join(PROJECT_ROOT, "cleaned_data", "champion_position_mapping.json")

# Context parquet 路径
CONTEXT_PARQUET = os.path.join(PROJECT_ROOT, "bp_recommendation", "features", "ALL_context.parquet")

# 输出路径
FEATURES_DIR = os.path.join(MODEL_DIR, "features")
WIDE_FEATURES_PATH = os.path.join(FEATURES_DIR, "ALL_prediction_wide_features.parquet")
TF_FEATURES_DIR = os.path.join(MODEL_DIR, "tf_features")
TF_SNAPSHOTS_DIR = os.path.join(MODEL_DIR, "tf_snapshots")
LOGS_DIR = os.path.join(MODEL_DIR, "logs")

for d in [TF_FEATURES_DIR, TF_SNAPSHOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# 确保模型路径在 sys.path 中
if MODEL_PICK_DIR not in sys.path:
    sys.path.insert(0, MODEL_PICK_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bp_recommendation.model_pick.model_pick import BPTacticalTransformerPick
from bp_recommendation.feature_pipeline import load_champion_vocabulary

LEAGUES = ["LPL", "LCK", "LEC"]

log = get_logger(__name__)


def compute_file_md5(file_path, chunk_size=8192):
    """计算文件的 MD5 哈希值 (分块读取, 适用于大文件)。

    Args:
        file_path: 文件路径
        chunk_size: 每次读取的字节数

    Returns:
        str: 32 位十六进制 MD5 哈希值
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


# =====================================================================
# Step 1: 导出快照
# =====================================================================
def export_snapshot(cutoff_date):
    """从推荐模型 checkpoint 导出生产快照。

    直接复制推荐模型的 best_model_nocs.pt, 添加生产标记。
    """
    snapshot_path = os.path.join(TF_SNAPSHOTS_DIR, "production_nocs.pt")

    if not os.path.exists(NOCS_CKPT_PATH):
        log.error("推荐模型 checkpoint 不存在: %s", NOCS_CKPT_PATH)
        log.error("请先训练推荐模型 (bp_recommendation)")
        return False

    ckpt = torch.load(NOCS_CKPT_PATH, map_location="cpu", weights_only=False)
    log.info("加载推荐模型 checkpoint: %s", NOCS_CKPT_PATH)
    log.info("  epoch: %s", ckpt.get('epoch', 'N/A'))
    log.info("  context_dim: %s", ckpt.get('context_dim', 'N/A'))
    log.info("  candidate_dim: %s", ckpt.get('candidate_dim', 'N/A'))
    if "metrics" in ckpt:
        log.info("  metrics: %s", ckpt['metrics'])

    # 添加生产标记
    ckpt["production"] = True
    ckpt["cutoff_date"] = cutoff_date
    ckpt["source"] = "bp_recommendation/model_pick/checkpoints/best_model_nocs.pt"
    ckpt["export_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 计算源 checkpoint 的 MD5, 用于加载时校验版本一致性
    source_md5 = compute_file_md5(NOCS_CKPT_PATH)
    ckpt["source_md5"] = source_md5
    ckpt["source_file_size"] = os.path.getsize(NOCS_CKPT_PATH)
    log.info("  source_md5: %s", source_md5)
    log.info("  source_file_size: %d bytes", ckpt['source_file_size'])

    torch.save(ckpt, snapshot_path)
    log.info("生产快照已导出: %s", snapshot_path)

    from feature_builder import load_tf_extractor
    import feature_builder
    feature_builder._TF_EXTRACTOR = None

    extractor = load_tf_extractor(snapshot_path=snapshot_path)
    if extractor is not None:
        log.info("快照验证通过: TransformerFeatureExtractor 加载成功")
    else:
        log.error("快照验证失败")
        return False

    return True


# =====================================================================
# Step 2: 提取 TF 特征
# =====================================================================
class TransformerFeatureExtractor:
    """使用 Forward Hook 从 NoCS Transformer 提取 4 种深层特征。"""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self._captured_hidden = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            if hasattr(output, "last_hidden_state"):
                self._captured_hidden = output.last_hidden_state.detach()
            elif isinstance(output, tuple):
                self._captured_hidden = output[0].detach()
            else:
                self._captured_hidden = output.detach()
        self.model.bert.register_forward_hook(hook_fn)

    def _build_inputs_from_context(self, context_df):
        n = len(context_df)
        vocab_size = self.model.vocab_size
        # champion_start_idx 固定为 role_token_start + n_positions (v3 方案)
        role_start = getattr(self.model, 'role_token_start', 2)
        n_pos = getattr(self.model, 'n_positions', 5)
        cs = role_start + n_pos
        context_dim = self.model.context_mlp[0].in_features

        bp_cols = [f"bp_step{i}_champion_id" for i in range(20)]
        if all(col in context_df.columns for col in bp_cols):
            bp_sequence = context_df[bp_cols].values.astype(np.int64)
            bp_sequence = np.clip(bp_sequence, 0, vocab_size - 1)
        else:
            bp_sequence = np.zeros((n, 20), dtype=np.int64)
        bp_sequence = torch.as_tensor(bp_sequence, dtype=torch.long)

        global_context = np.zeros((n, context_dim), dtype=np.float32)
        league_map = {lg: idx for idx, lg in enumerate(LEAGUES)}
        if "league" in context_df.columns:
            league_indices = context_df["league"].map(league_map).fillna(0).astype(int).values
            global_context[np.arange(n), league_indices] = 1.0

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

        if "playoffs" in context_df.columns:
            global_context[:, 13] = context_df["playoffs"].fillna(0).values.astype(np.float32)
        if "first_pick_map_side" in context_df.columns:
            global_context[:, 14] = context_df["first_pick_map_side"].fillna(1).values.astype(np.float32)

        # 提取局数特征 (is_game_1 到 is_game_5)
        for i in range(1, 6):
            col_name = f"is_game_{i}"
            if col_name in context_df.columns:
                # 放在 15 到 19 的位置
                global_context[:, 14 + i] = context_df[col_name].fillna(0).values.astype(np.float32)
            else:
                # 默认第一局为 1
                if i == 1:
                    global_context[:, 15] = 1.0

        global_context = torch.as_tensor(global_context, dtype=torch.float32)

        candidate_dim = self.model.candidate_mlp[0].in_features
        candidate_matrix = torch.zeros(n, vocab_size, candidate_dim, dtype=torch.float32)
        available_mask = torch.ones(n, vocab_size, dtype=torch.float32)
        available_mask[:, :cs] = 0.0

        return {
            "bp_sequence": bp_sequence,
            "global_context": global_context,
            "candidate_matrix": candidate_matrix,
            "available_mask": available_mask,
        }

    def extract_features(self, gameids_list, context_df, batch_size=64):
        self.model.eval()
        gameids_list = list(gameids_list)

        ctx_gameids = [gid for gid in gameids_list if gid in context_df["gameid"].values]
        no_ctx_gameids = [gid for gid in gameids_list if gid not in context_df["gameid"].values]

        results = {}

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
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                _ = self.model(
                    inputs["bp_sequence"],
                    inputs["global_context"],
                    inputs["candidate_matrix"],
                    inputs["available_mask"],
                )

                hidden = self._captured_hidden
                seq_len = inputs["bp_sequence"].shape[1]
                bp_hidden = hidden[:, :seq_len, :]

                    
                BLUE_PICK_STEPS = [6, 9, 10, 17, 18]
                RED_PICK_STEPS = [7, 8, 11, 16, 19]
                blue_steps = [s for s in BLUE_PICK_STEPS if s < seq_len]
                red_steps = [s for s in RED_PICK_STEPS if s < seq_len]

                seq_mask = (inputs["bp_sequence"] != self.model.pad_idx).float()

                b_mask = seq_mask[:, blue_steps].unsqueeze(-1)
                b_hidden = bp_hidden[:, blue_steps, :] * b_mask
                blue_pooled = b_hidden.sum(dim=1) / b_mask.sum(dim=1).clamp(min=1e-9)
                
                # 红方
                r_mask = seq_mask[:, red_steps].unsqueeze(-1)
                r_hidden = bp_hidden[:, red_steps, :] * r_mask
                red_pooled = r_hidden.sum(dim=1) / r_mask.sum(dim=1).clamp(min=1e-9)

                blue_latent = self.model.bert_proj(blue_pooled).detach()
                red_latent = self.model.bert_proj(red_pooled).detach()

                tf_win_logits = blue_latent.norm(dim=1) * torch.sign(blue_latent.mean(dim=1))
                tf_cosine_sim = F.cosine_similarity(blue_latent, red_latent, dim=1)
                tf_blue_l2norm = blue_latent.norm(dim=1)
                tf_red_l2norm = red_latent.norm(dim=1)

                batch_features = torch.stack([
                    tf_win_logits, tf_cosine_sim, tf_blue_l2norm, tf_red_l2norm
                ], dim=1).cpu().numpy()

                all_features.append(batch_features)
                all_gids.extend(batch_df["gameid"].values.tolist())

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
def extract_tf_features(cutoff_date, device="cpu"):
    from feature_builder import load_tf_extractor

    log.info("\n%s", "=" * 70)
    log.info("Step 2: 提取 TF 特征")
    log.info("  Device: %s", device)
    log.info("%s", "=" * 70)

    snapshot_path = os.path.join(TF_SNAPSHOTS_DIR, "production_nocs.pt")
    import feature_builder
    feature_builder._TF_EXTRACTOR = None
    extractor = load_tf_extractor(snapshot_path=snapshot_path, device=device)
    if extractor is None:
        log.error("TF 特征提取器加载失败")
        return

    if not os.path.exists(WIDE_FEATURES_PATH):
        log.error("Wide features 不存在: %s", WIDE_FEATURES_PATH)
        return

    wide_df = pd.read_parquet(WIDE_FEATURES_PATH)
    gameids = wide_df["gameid"].unique().tolist()
    log.info("Wide features: %d 行, %d 个唯一 gameid", len(wide_df), len(gameids))

    if not os.path.exists(CONTEXT_PARQUET):
        log.error("Context parquet 不存在: %s", CONTEXT_PARQUET)
        return

    context_df = pd.read_parquet(CONTEXT_PARQUET)
    log.info("Context parquet: %d 行", len(context_df))

    import time
    t0 = time.time()
    results = extractor.extract_features(gameids, context_df, batch_size=64)
    elapsed = time.time() - t0
    log.info("TF 特征提取完成: %d 个 gameid, 耗时 %.1fs", len(results), elapsed)

    tf_rows = []
    for gid, feat in results.items():
        tf_rows.append({
            "gameid": gid,
            "tf_win_logits": feat["tf_win_logits"],
            "tf_cosine_sim": feat["tf_cosine_sim"],
            "tf_blue_l2norm": feat["tf_blue_l2norm"],
            "tf_red_l2norm": feat["tf_red_l2norm"],
        })
    tf_df = pd.DataFrame(tf_rows)
    tf_output_path = os.path.join(TF_FEATURES_DIR, "production_tf_features.parquet")
    tf_df.to_parquet(tf_output_path, index=False)
    log.info("TF 特征已保存: %s (%d 行)", tf_output_path, len(tf_df))

    matched = tf_df["tf_win_logits"].notna().sum()
    log.info("有效 TF 特征: %d/%d (%.1f%%)", matched, len(tf_df), 100*matched/len(tf_df))


def main():
    setup_logging()
    log_file = os.path.join(LOGS_DIR, "export_production_transformer.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)

    parser = argparse.ArgumentParser(
        description="Export Production Transformer Snapshot from Recommendation Model"
    )
    parser.add_argument("--cutoff", type=str, default=None,
                        help="数据截止日期 (default: 自动检测最新数据日期)")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (default: auto)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="跳过 TF 特征提取, 仅导出快照")
    args = parser.parse_args()

    from common.paths import get_latest_data_date
    if args.cutoff is None:
        latest_date = get_latest_data_date()
        args.cutoff = latest_date.strftime("%Y-%m-%d")
        log.info("[AutoDate] cutoff 未指定，自动检测到最新数据日期: %s", args.cutoff)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"

    log.info("%s", "=" * 70)
    log.info("Export Production Transformer Snapshot")
    log.info("  Source        : %s", NOCS_CKPT_PATH)
    log.info("  Cutoff Date   : %s", args.cutoff)
    log.info("  Device        : %s", device)
    log.info("%s", "=" * 70)

    success = export_snapshot(args.cutoff)
    if not success:
        log.error("\n导出失败, 请检查推荐模型 checkpoint 是否存在")
        return

    if args.skip_extract:
        log.info("\n跳过 TF 特征提取 (--skip-extract)")
        return

    extract_tf_features(args.cutoff, device=device)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
