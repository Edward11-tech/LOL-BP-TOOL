#!/usr/bin/env python3
"""
构建 BP 推荐模型与预测模型的 PSI 特征基线。

用法:
    cd <项目根目录>
    python build_feature_baselines.py

输出:
    - bp_recommendation/features/feature_baseline.json
    - bp_prediction/features/prediction_feature_baseline.json
"""
import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# 将项目根目录加入 sys.path，确保模块导入一致
PROJECT_ROOT = str(Path(__file__).parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logger_config import setup_logging, get_logger

log = get_logger(__name__)


def _check_file(path: str) -> bool:
    if not os.path.exists(path):
        log.error(f"文件不存在: {path}")
        return False
    return True


def build_recommendation_baseline():
    """
    基于推荐模型 (bp_recommendation) 的训练特征构建 PSI 基线。

    当前推荐模型的特征来源:
      - 候选矩阵 (candidate_matrix): CANDIDATE_FEAT_MAP 定义的 33 维特征
      - 全局上下文 (global_context): 20 维特征
      - LightGBM Cascade 输入: cascade_pick / cascade_ban 的 FEATURE_COLS

    由于训练时导出的是按 sample 组织的 logits/features npz 文件，
    这里直接加载 pick/ban 的训练 npz，采样并聚合候选特征，构建每个
    CANDIDATE_FEAT_MAP 特征以及 global_context 的分布基线。
    """
    from bp_recommendation.feature_pipeline import CANDIDATE_FEAT_MAP
    from bp_recommendation.feature_monitor import FeatureMonitor

    features_dir = os.path.join(PROJECT_ROOT, "bp_recommendation", "features")
    baseline_path = os.path.join(features_dir, "feature_baseline.json")

    # 候选矩阵训练特征来自 cascade_pick / cascade_ban 训练时使用的 candidates
    # 这里读取 npz 中的 candidates 数组 (shape: [n_samples, vocab_size, n_features])
    pick_npz = os.path.join(PROJECT_ROOT, "bp_recommendation", "model_pick", "features", "ALL_train_logits_cs.npz")
    ban_npz = os.path.join(PROJECT_ROOT, "bp_recommendation", "model_ban", "features", "ALL_train_logits_cs.npz")

    # 同时也尝试读取 context 文件获取全局上下文 (context 在 npz 中没有直接保存)
    context_parquet = os.path.join(features_dir, "ALL_context.parquet")

    feature_dict: dict[str, np.ndarray] = {}

    # ---- 1) 候选矩阵特征 ----
    sample_limit = 200_000  # 每个 npz 最多采样的 sample 数，控制内存
    for name, path in [("pick", pick_npz), ("ban", ban_npz)]:
        if not _check_file(path):
            continue
        try:
            data = np.load(path)
            candidates = data["candidates"]  # [N, vocab_size, n_features]
            masks = data.get("masks", None)
            n_samples = candidates.shape[0]
            if n_samples > sample_limit:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(n_samples, size=sample_limit, replace=False)
                candidates = candidates[idx]
                masks = masks[idx] if masks is not None else None

            # 只保留可用的英雄行 (mask == 1) 以及英雄索引区域
            # 但 PSI 基线需要所有特征值的分布，这里简单压平所有位置
            # 注意：候选矩阵中很多位置为 0 (非英雄或默认填充)，会干扰分布，
            # 因此仅保留 mask == 1 且为英雄索引的部分
            if masks is not None:
                valid_mask = (masks > 0.5).reshape(-1)
            else:
                valid_mask = np.ones(candidates.shape[0] * candidates.shape[1], dtype=bool)

            flat_candidates = candidates.reshape(-1, candidates.shape[-1])
            for feat_name, col_idx in CANDIDATE_FEAT_MAP.items():
                values = flat_candidates[valid_mask, col_idx]
                values = values[np.isfinite(values)]
                key = f"candidate_{feat_name}"
                if key not in feature_dict:
                    feature_dict[key] = values
                else:
                    feature_dict[key] = np.concatenate([feature_dict[key], values])
            log.info(f"[推荐] 已处理 {name} candidates: shape={candidates.shape}")
        except Exception:
            log.exception(f"[推荐] 处理 {name} npz 时出错")

    # ---- 2) 全局上下文特征 ----
    if _check_file(context_parquet):
        try:
            ctx_df = pd.read_parquet(context_parquet)
            # global_context 在 bp_predict.py 中按顺序拼接：
            # league_vec(3) + b_style(5) + r_style(5) + [playoffs_f, first_pick_f](2) + game_num_vec(5)
            ctx_cols = (
                [c for c in ctx_df.columns if c.startswith("league_")][:3]
                + [c for c in ctx_df.columns if c.startswith("blue_team_avg_")][:5]
                + [c for c in ctx_df.columns if c.startswith("red_team_avg_")][:5]
                + ["playoffs", "first_pick_map_side"]
                + [c for c in ctx_df.columns if c.startswith("is_game_")][:5]
            )
            # 兜底：如果列名对不上，直接使用数值列
            numeric_cols = ctx_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(ctx_cols) < 20:
                ctx_cols = numeric_cols[:20]

            for col in ctx_cols[:20]:
                if col in ctx_df.columns:
                    values = ctx_df[col].dropna().values.astype(np.float64)
                    values = values[np.isfinite(values)]
                    feature_dict[f"context_{col}"] = values
            log.info(f"[推荐] 已处理全局上下文: {len(ctx_cols[:20])} 列")
        except Exception:
            log.exception("[推荐] 处理 context parquet 时出错")

    # ---- 3) 使用 FeatureMonitor 构建并保存基线 ----
    monitor = FeatureMonitor()
    if not feature_dict:
        log.error("[推荐] 没有可用的特征数据，无法构建基线")
        return False

    monitor.build_baseline(feature_dict, baseline_path)
    log.info(f"[推荐] 基线已保存: {baseline_path}")
    return True


def build_prediction_baseline():
    """
    基于预测模型 (bp_prediction) 的训练宽表构建 PSI 基线。

    直接读取 ALL_prediction_wide_features.parquet，排除非特征列后
    使用 PredictionFeatureMonitor 构建基线。
    """
    from bp_prediction.feature_monitor import PredictionFeatureMonitor

    features_dir = os.path.join(PROJECT_ROOT, "bp_prediction", "features")
    baseline_path = os.path.join(features_dir, "prediction_feature_baseline.json")
    wide_path = os.path.join(features_dir, "ALL_prediction_wide_features.parquet")

    if not _check_file(wide_path):
        return False

    try:
        df = pd.read_parquet(wide_path)
        # 排除非特征列 (与训练脚本一致)
        label_cols = {"gameid", "date", "league", "result", "split",
                      "playoffs", "first_pick_map_side", "patch"}
        meta_cols = {"blue_team", "red_team"}
        leak_cols = {"match_seq_idx"}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols
                        if c not in label_cols and c not in meta_cols and c not in leak_cols]

        features_df = df[feature_cols].copy()
        monitor = PredictionFeatureMonitor(feature_cols=feature_cols)
        monitor.build_baseline(features_df, baseline_path)
        log.info(f"[预测] 基线已保存: {baseline_path} ({len(feature_cols)} features)")
        return True
    except Exception:
        log.exception("[预测] 构建基线失败")
        return False


def main():
    log.info("=" * 60)
    log.info("构建 PSI 特征基线")
    log.info("=" * 60)

    ok1 = build_recommendation_baseline()
    ok2 = build_prediction_baseline()

    log.info("=" * 60)
    if ok1 and ok2:
        log.info("全部基线构建完成")
    else:
        log.warning(f"部分基线构建失败: 推荐={ok1}, 预测={ok2}")
    log.info("=" * 60)


if __name__ == "__main__":
    setup_logging()
    main()
