"""
端到端推理一致性校验
====================
对比同一场比赛的线上（feature_builder）和线下（wide_features.parquet）推理结果。

包含两种模式的一致性检测：
  1. predict (胜率预测): 线上/线下胜率概率对比
  2. bp_delta (BP 影响量化): 线上/线下 post_prob、pre_prob、delta 对比

线上路径：feature_builder.py → model.predict_proba() → ensemble
线下路径：wide_features.parquet → model.predict_proba() → ensemble

采样策略：按联赛分层采样 (LPL / LCK / LEC)，确保三联赛均覆盖。

用法:
  python check_prediction_alignment.py [--samples 50] [--mode both|predict|bp_delta]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 必须在 logger_config 导入前设置 sys.path
_PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from logger_config import get_logger, setup_logging

FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ---- 必须在导入 LightGBM 之前设置，防止 macOS 上 OpenMP 死锁 ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- 路径 ----
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, MODEL_DIR)

from catboost import CatBoostClassifier
from bp_prediction.feature_builder import (
    POSITIONS, load_feature_cols, load_feature_stores, load_champion_tags,
    load_known_champions, resolve_team_name, get_team_roster,
    build_single_match_features, build_predraft_features,
    extract_tf_features_for_match, classify_features, TF_COLS,
)

log = get_logger(__name__)

# ---- 常量 ----
FEATURES_DIR = os.path.join(MODEL_DIR, "features")
PRODUCTION_DIR = os.path.join(MODEL_DIR, "models", "production")
WIDE_FEATURES_PATH = os.path.join(FEATURES_DIR, "ALL_prediction_wide_features.parquet")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"check_prediction_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def load_models():
    """加载生产模型 (7-seed bagging)。"""
    models = {}
    if not os.path.exists(PRODUCTION_DIR):
        log.error(f"生产模型目录不存在: {PRODUCTION_DIR}")
        return None

    for seed_idx in range(7):
        model_path = os.path.join(PRODUCTION_DIR, f"catboost_seed_{seed_idx}.cbm")
        if os.path.exists(model_path):
            model = CatBoostClassifier()
            model.load_model(model_path)
            models[seed_idx] = model
            log.info(f"  加载 catboost_seed_{seed_idx}.cbm")
        else:
            log.warning(f"  模型文件不存在: {model_path}")

    if not models:
        log.error("未找到任何生产模型")
        return None

    return {"production": list(models.values())}


def predict_online(match_info, stores, champion_tags, feature_cols, models):
    """
    线上推理路径：
    1. 提取 TF 特征
    2. 通过 feature_builder 构建特征
    3. 通过模型集成预测
    """
    tf_features = extract_tf_features_for_match(match_info)

    features_df, unknown_info = build_single_match_features(
        match_info, stores, champion_tags,
        feature_cols=feature_cols, tf_features=tf_features
    )

    if features_df is None:
        return None, None, None

    # 对齐列顺序 & NaN 处理 (与 PredictBackend._predict 一致)
    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    for c in missing_cols:
        features_df[c] = 0.0
    ordered_df = features_df[feature_cols]
    X_online = ordered_df.values.astype(np.float32)
    X_online = np.nan_to_num(X_online, nan=0.0, posinf=0.0, neginf=0.0)

    all_preds = []
    seed_preds = []
    for fold_key, fold_models in sorted(models.items()):
        fold_preds = []
        for seed_idx, model in enumerate(fold_models):
            pred = float(model.predict_proba(X_online)[0, 1])
            fold_preds.append(pred)
            seed_preds.append(pred)
        fold_mean = float(np.mean(fold_preds))
        all_preds.append(fold_mean)

    final_prob = float(np.mean(all_preds))
    return final_prob, seed_preds, ordered_df


def predict_offline(row, feature_cols, models, tf_features=None):
    """
    线下推理路径：
    1. 直接从 wide_features.parquet 提取特征列
    2. 用 tf_features 覆盖 TF 列（对齐线上实时提取的 TF 特征）
    3. 通过模型集成预测
    """
    X_offline = np.array([float(row.get(c, 0.0)) if not pd.isna(row.get(c)) else 0.0
                          for c in feature_cols], dtype=np.float32)
    X_offline = np.nan_to_num(X_offline, nan=0.0, posinf=0.0, neginf=0.0)

    # 【修复】：match_seq_idx 是行号索引，线上无法预知（默认 0），线下强制对齐为 0
    if "match_seq_idx" in feature_cols:
        X_offline[feature_cols.index("match_seq_idx")] = 0.0

    # 用实时提取的 TF 特征覆盖 parquet 中的 TF 列（parquet 中 TF 列可能为 0 / 不存在）
    if tf_features:
        for tc in TF_COLS:
            if tc in feature_cols:
                idx = feature_cols.index(tc)
                X_offline[idx] = float(tf_features.get(tc, 0.0))

    X_offline = X_offline.reshape(1, -1)

    all_preds = []
    seed_preds = []
    for fold_key, fold_models in sorted(models.items()):
        fold_preds = []
        for seed_idx, model in enumerate(fold_models):
            pred = float(model.predict_proba(X_offline)[0, 1])
            fold_preds.append(pred)
            seed_preds.append(pred)
        fold_mean = float(np.mean(fold_preds))
        all_preds.append(fold_mean)

    final_prob = float(np.mean(all_preds))
    return final_prob, seed_preds


def predict_bp_delta_online(match_info, stores, champion_tags, feature_cols, models):
    """
    线上 bp_delta 推理路径：
    1. 提取 TF 特征 (post/pre 共享)
    2. build_single_match_features → post 特征
    3. build_predraft_features → pre 特征 (draft 列置零)
    4. 分别集成预测，计算 delta = post_prob - pre_prob

    Returns:
        dict: {post_prob, pre_prob, delta, post_seeds, pre_seeds}
    """
    tf_features = extract_tf_features_for_match(match_info)

    # Post-Draft: 完整特征
    post_df, _ = build_single_match_features(
        match_info, stores, champion_tags,
        feature_cols=feature_cols, tf_features=tf_features
    )
    if post_df is None:
        return None

    # Pre-Draft: draft 特征置零
    pre_df = build_predraft_features(
        match_info, stores, champion_tags,
        feature_cols=feature_cols, tf_features=tf_features
    )

    # 对齐列顺序 & NaN 处理
    for df in (post_df, pre_df):
        missing = [c for c in feature_cols if c not in df.columns]
        for c in missing:
            df[c] = 0.0

    post_X = np.nan_to_num(post_df[feature_cols].values.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
    pre_X = np.nan_to_num(pre_df[feature_cols].values.astype(np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)

    # 集成预测
    def _ensemble(X):
        all_preds, seed_preds = [], []
        for fold_key, fold_models in sorted(models.items()):
            fold_preds = []
            for model in fold_models:
                pred = float(model.predict_proba(X)[0, 1])
                fold_preds.append(pred)
                seed_preds.append(pred)
            all_preds.append(np.mean(fold_preds))
        return float(np.mean(all_preds)), seed_preds

    post_prob, post_seeds = _ensemble(post_X)
    pre_prob, pre_seeds = _ensemble(pre_X)
    delta = post_prob - pre_prob

    return {
        "post_prob": post_prob,
        "pre_prob": pre_prob,
        "delta": delta,
        "post_seeds": post_seeds,
        "pre_seeds": pre_seeds,
    }


def predict_bp_delta_offline(row, feature_cols, models, tf_features=None):
    """
    线下 bp_delta 推理路径：
    1. 从 parquet 提取完整特征 → post 特征
    2. 将 draft 相关列置零 → pre 特征 (与 build_predraft_features 一致)
    3. 用 tf_features 覆盖 TF 列 (对齐线上实时提取)
    4. 分别集成预测，计算 delta

    Returns:
        dict: {post_prob, pre_prob, delta, post_seeds, pre_seeds}
    """
    # 构建完整特征向量
    post_X = np.array([float(row.get(c, 0.0)) if not pd.isna(row.get(c)) else 0.0
                       for c in feature_cols], dtype=np.float32)
    post_X = np.nan_to_num(post_X, nan=0.0, posinf=0.0, neginf=0.0)

    # 【修复】：match_seq_idx 是行号索引，线上无法预知（默认 0），线下强制对齐为 0
    if "match_seq_idx" in feature_cols:
        post_X[feature_cols.index("match_seq_idx")] = 0.0

    # 用实时 TF 特征覆盖 (对齐线上)
    if tf_features:
        for tc in TF_COLS:
            if tc in feature_cols:
                idx = feature_cols.index(tc)
                post_X[idx] = float(tf_features.get(tc, 0.0))

    # Pre-Draft: draft 列置零 (与 classify_features 逻辑一致)
    draft_cols, _ = classify_features(feature_cols)
    pre_X = post_X.copy()
    for col in draft_cols:
        if col in feature_cols:
            idx = feature_cols.index(col)
            pre_X[idx] = 0.0

    post_X = post_X.reshape(1, -1)
    pre_X = pre_X.reshape(1, -1)

    def _ensemble(X):
        all_preds, seed_preds = [], []
        for fold_key, fold_models in sorted(models.items()):
            fold_preds = []
            for model in fold_models:
                pred = float(model.predict_proba(X)[0, 1])
                fold_preds.append(pred)
                seed_preds.append(pred)
            all_preds.append(np.mean(fold_preds))
        return float(np.mean(all_preds)), seed_preds

    post_prob, post_seeds = _ensemble(post_X)
    pre_prob, pre_seeds = _ensemble(pre_X)
    delta = post_prob - pre_prob

    return {
        "post_prob": post_prob,
        "pre_prob": pre_prob,
        "delta": delta,
        "post_seeds": post_seeds,
        "pre_seeds": pre_seeds,
    }


def stratified_sample_by_league(df, num_samples, leagues=("LPL", "LCK", "LEC"),
                                 random_state=42):
    """按联赛分层采样，确保每个联赛均有覆盖。

    分配策略: 每个联赛均分 num_samples，不足则从其他联赛补足。
    """
    per_league = max(1, num_samples // len(leagues))
    samples = []

    for league in leagues:
        league_df = df[df["league"] == league]
        n = min(per_league, len(league_df))
        if n > 0:
            samples.append(league_df.sample(n=n, random_state=random_state))

    # 补足差额
    total_so_far = sum(len(s) for s in samples)
    if total_so_far < num_samples:
        remaining = num_samples - total_so_far
        already_idx = set()
        for s in samples:
            already_idx.update(s.index)
        pool = df[~df.index.isin(already_idx)]
        if len(pool) > 0:
            n = min(remaining, len(pool))
            samples.append(pool.sample(n=n, random_state=random_state))

    result = pd.concat(samples, ignore_index=False)
    return result.sample(frac=1, random_state=random_state)  # 打乱顺序


def build_match_info_from_row(row, known_champions, all_teams):
    """从 parquet 行重建 match_info (完全对齐 PredictBackend._build_match_info)。"""
    positions = ["top", "jng", "mid", "bot", "sup"]

    blue_team = resolve_team_name(row.get("blue_team", ""), all_teams)
    red_team = resolve_team_name(row.get("red_team", ""), all_teams)
    league = row.get("league", "LCK")
    
    # 保持精确到秒的时序对齐，防止指数衰减权重错位
    match_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d %H:%M:%S")
    is_playoff = bool(row.get("is_playoff", 0))

    game_num = 1
    for i in range(1, 6):
        if row.get(f"is_game_{i}", 0) == 1:
            game_num = i
            break
        
    match_info = {
        "league": league,
        "is_playoff": is_playoff,
        "is_blue_map_side": bool(row.get("is_blue_map_side", True)),
        "game_num": game_num, # 【修复点】：显式透传局数，防止时序特征打架
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_champions": [],
        "red_champions": [],
        "date": match_date,
        "mode": "full" if (blue_team or red_team) else "draft",
        # 【修复点】：初始化 unknown 列表
        "blue_unknown_positions": [],
        "red_unknown_positions": [],
    }

    for pos in positions:
        match_info["blue_champions"].append(row.get(f"blue_{pos}_champion", ""))
        match_info["red_champions"].append(row.get(f"red_{pos}_champion", ""))

    # 【修复点】：完全对齐线上的新秀解析逻辑
    for side in ["blue", "red"]:
        for pos in positions:
            player_col = f"{side}_{pos}_player_id"
            player_val = str(row.get(player_col, "")).strip()

            # 如果在离线数据集里是 NaN、空字符串或 unknown，则判定为新秀/未知
            if player_val.lower() in ("unknown", "unk", "?", "未知", "新秀", "", "nan"):
                match_info[f"{side}_unknown_positions"].append(pos)
                match_info[f"{side}_{pos}_player_id"] = ""
            else:
                match_info[f"{side}_{pos}_player_id"] = player_val

    return match_info

def run_prediction_comparison(num_samples=50, verbose=True):
    """对比线上/线下胜率预测结果。

    Args:
        num_samples: 采样数量
        verbose: 是否输出详细日志

    Returns:
        dict: {"passed": bool, "total": int, "mismatch": int, "max_diff": float, "errors": int}
    """
    if verbose:
        log.info("=" * 70)
        log.info("  端到端推理一致性校验 — 胜率预测 (predict)")
        log.info("  线上路径: feature_builder.py → model.predict_proba()")
        log.info("  线下路径: wide_features.parquet → model.predict_proba()")
        log.info("=" * 70)

    # 1. 加载模型
    if verbose:
        log.info("\n[1/4] 加载生产模型...")
    models = load_models()
    if models is None:
        log.error("模型加载失败，退出")
        return {"passed": False, "total": 0, "mismatch": 0, "max_diff": 0.0, "errors": 1}

    # 2. 加载数据
    if verbose:
        log.info("\n[2/4] 加载特征数据...")
    feature_cols = load_feature_cols(use_production=True)
    if verbose:
        log.info(f"  特征列数: {len(feature_cols)}")

    stores = load_feature_stores()
    champion_tags = load_champion_tags()
    known_champions = load_known_champions()
    all_teams = set()
    if stores and "team_profile" in stores:
        all_teams = set(stores["team_profile"]["team"].unique())

    offline_df = pd.read_parquet(WIDE_FEATURES_PATH)
    offline_df["date"] = pd.to_datetime(offline_df["date"])
    if verbose:
        log.info(f"  线下数据量: {len(offline_df)} 场")

    # 3. 按联赛分层采样
    if verbose:
        log.info(f"\n[3/4] 按联赛分层采样 {num_samples} 场比赛 (LPL/LCK/LEC)...")
    sample_df = stratified_sample_by_league(offline_df, num_samples)
    league_counts = sample_df["league"].value_counts()
    if verbose:
        log.info(f"  采样分布: {dict(league_counts)}")

    # 4. 逐场对比
    if verbose:
        log.info(f"\n[4/4] 逐场对比推理结果...")
        log.info("-" * 70)

    total_compared = 0
    total_mismatch = 0
    total_errors = 0
    max_diff = 0.0
    all_diffs = []

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        gameid = row.get("gameid", "Unknown")
        league = row.get("league", "LCK")
        match_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        blue_team = row.get("blue_team", "?")
        red_team = row.get("red_team", "?")
        actual_result = row.get("result", -1)

        log.info(f"\n--- 第 {idx}/{num_samples} 场 ---")
        log.info(f"  GameID: {gameid} | {league} | {match_date}")
        log.info(f"  {blue_team} vs {red_team} | 实际结果: {'蓝胜' if actual_result == 1 else '红胜' if actual_result == 0 else '未知'}")

        # 构建 match_info
        match_info = build_match_info_from_row(row, known_champions, all_teams)

        # 提取 TF 特征（线上/线下共用同一份，确保一致性）
        tf_features = extract_tf_features_for_match(match_info)

        # 线上推理
        try:
            online_prob, online_seeds, online_features_df = predict_online(
                match_info, stores, champion_tags, feature_cols, models
            )
        except Exception:
            if verbose:
                log.exception("  ❌ 线上推理失败")
            total_errors += 1
            continue

        if online_features_df is None:
            if verbose:
                log.error(f"  ❌ 线上特征构建失败")
            total_errors += 1
            continue

        # 直接对比特征矩阵 (诊断用)
        # 线下特征向量：从 parquet 提取，并用 TF 特征覆盖 TF 列
        online_X = np.array([float(v) if not pd.isna(v) else 0.0
                             for v in online_features_df.iloc[0].values], dtype=np.float32)
        offline_X = np.array([float(row.get(c, 0.0)) if not pd.isna(row.get(c)) else 0.0
                              for c in feature_cols], dtype=np.float32)
        offline_X = np.nan_to_num(offline_X, nan=0.0, posinf=0.0, neginf=0.0)
        # 【修复】：match_seq_idx 线下强制对齐为 0（线上无法预知，默认 0）
        if "match_seq_idx" in feature_cols:
            offline_X[feature_cols.index("match_seq_idx")] = 0.0
        # 用实时提取的 TF 特征覆盖（对齐线上）
        for tc in TF_COLS:
            if tc in feature_cols:
                idx = feature_cols.index(tc)
                offline_X[idx] = float(tf_features.get(tc, 0.0))

        feature_diff = np.abs(online_X - offline_X)
        feature_max_diff = np.max(feature_diff)
        feature_diff_count = np.sum(feature_diff > 1e-4)
        if feature_diff_count > 0:
            worst_idx = np.argmax(feature_diff)
            log.info(f"  ⚠️ 特征差异: {feature_diff_count}/{len(feature_cols)} 列不一致, "
                     f"最大差异={feature_max_diff:.6f} ({feature_cols[worst_idx]})")
            top_diff_idx = np.argsort(feature_diff)[-5:][::-1]
            for di in top_diff_idx:
                if feature_diff[di] > 1e-4:
                    log.info(f"    {feature_cols[di]}: online={online_X[di]:.6f} | offline={offline_X[di]:.6f} | diff={feature_diff[di]:.6f}")
        else:
            log.info(f"  ✅ 特征矩阵完全一致 ({len(feature_cols)} 列)")

        # 离线推理（使用相同的 TF 特征）
        try:
            offline_prob, offline_seeds = predict_offline(row, feature_cols, models, tf_features=tf_features)
        except Exception:
            if verbose:
                log.exception("  ❌ 线下推理失败")
            total_errors += 1
            continue

        total_compared += 1

        # 对比
        diff = abs(online_prob - offline_prob)
        all_diffs.append(diff)
        max_diff = max(max_diff, diff)

        if verbose:
            log.info(f"  线上概率: {online_prob:.6f} (seeds: {[f'{s:.4f}' for s in online_seeds]})")
            log.info(f"  线下概率: {offline_prob:.6f} (seeds: {[f'{s:.4f}' for s in offline_seeds]})")

        if diff < 1e-6:
            if verbose:
                log.info(f"  ✅ 完全一致 (diff={diff:.2e})")
        elif diff < 1e-4:
            if verbose:
                log.info(f"  ⚠️ 微小差异 (diff={diff:.6f}) — 浮点误差范围内")
            total_mismatch += 1
        else:
            if verbose:
                log.info(f"  ❌ 不一致! (diff={diff:.6f})")
                # 逐 seed 对比
                log.info(f"  逐 seed 对比:")
                for si, (on, off) in enumerate(zip(online_seeds, offline_seeds)):
                    sd = abs(on - off)
                    flag = "✅" if sd < 1e-6 else ("⚠️" if sd < 1e-4 else "❌")
                    log.info(f"    seed_{si}: online={on:.6f} | offline={off:.6f} | diff={sd:.2e} {flag}")
            total_mismatch += 1

    # 汇总
    mean_diff = float(np.mean(all_diffs)) if all_diffs else 0.0
    std_diff = float(np.std(all_diffs)) if all_diffs else 0.0
    passed = (total_mismatch == 0) or (max_diff < 1e-4 and total_mismatch <= total_compared * 0.2)

    if verbose:
        log.info("\n" + "=" * 70)
        log.info("  汇总")
        log.info("=" * 70)
        log.info(f"  对比场次: {total_compared}")
        log.info(f"  不一致场次: {total_mismatch}")
        log.info(f"  错误场次: {total_errors}")
        log.info(f"  最大差异: {max_diff:.6f}")
        if all_diffs:
            log.info(f"  平均差异: {mean_diff:.6f}")
            log.info(f"  差异标准差: {std_diff:.6f}")
        if total_mismatch == 0:
            log.info("\n  ✅ 结论: 线上/线下推理结果完全一致")
        elif max_diff < 1e-4 and total_mismatch <= total_compared * 0.2:
            log.info(f"\n  ⚠️ 结论: 存在微小差异 (max={max_diff:.2e})，在浮点误差范围内")
        else:
            log.info(f"\n  ❌ 结论: 线上/线下推理结果存在显著差异，需要排查")
        log.info(f"\n  详细日志: {LOG_FILE}")

    return {
        "mode": "predict",
        "passed": passed and total_errors == 0,
        "total": total_compared,
        "mismatch": total_mismatch,
        "errors": total_errors,
        "max_diff": float(max_diff),
        "mean_diff": mean_diff,
    }


def run_bp_delta_comparison(num_samples=50, verbose=True):
    """对比线上/线下 bp_delta 推理结果。

    bp_delta = post_prob - pre_prob
    - post_prob: 完整特征 (含 draft 信息) 的预测胜率
    - pre_prob: draft 特征置零后的预测胜率 (纸面硬实力)
    - delta: BP 选人带来的胜率增量

    Args:
        num_samples: 采样数量
        verbose: 是否输出详细日志

    Returns:
        dict: {"passed": bool, "total": int, "mismatch": int, "max_diff": float, "errors": int}
    """
    if verbose:
        log.info("=" * 70)
        log.info("  端到端推理一致性校验 — BP Delta (bp_delta)")
        log.info("  线上路径: feature_builder.build_predraft_features → predict")
        log.info("  线下路径: parquet + classify_features 置零 → predict")
        log.info("  检测项: post_prob / pre_prob / delta")
        log.info("=" * 70)

    # 1. 加载模型
    if verbose:
        log.info("\n[1/4] 加载生产模型...")
    models = load_models()
    if models is None:
        log.error("模型加载失败，退出")
        return {"mode": "bp_delta", "passed": False, "total": 0, "mismatch": 0,
                "max_diff": 0.0, "errors": 1}

    # 2. 加载数据
    if verbose:
        log.info("\n[2/4] 加载特征数据...")
    feature_cols = load_feature_cols(use_production=True)
    if verbose:
        log.info(f"  特征列数: {len(feature_cols)}")

    draft_cols, hard_cols = classify_features(feature_cols)
    if verbose:
        log.info(f"  draft 特征: {len(draft_cols)} 列 | hard 特征: {len(hard_cols)} 列")

    stores = load_feature_stores()
    champion_tags = load_champion_tags()
    known_champions = load_known_champions()
    all_teams = set()
    if stores and "team_profile" in stores:
        all_teams = set(stores["team_profile"]["team"].unique())

    offline_df = pd.read_parquet(WIDE_FEATURES_PATH)
    offline_df["date"] = pd.to_datetime(offline_df["date"])
    if verbose:
        log.info(f"  线下数据量: {len(offline_df)} 场")

    # 3. 按联赛分层采样
    if verbose:
        log.info(f"\n[3/4] 按联赛分层采样 {num_samples} 场比赛 (LPL/LCK/LEC)...")
    sample_df = stratified_sample_by_league(offline_df, num_samples)
    league_counts = sample_df["league"].value_counts()
    if verbose:
        log.info(f"  采样分布: {dict(league_counts)}")

    # 4. 逐场对比
    if verbose:
        log.info(f"\n[4/4] 逐场对比 bp_delta 推理结果...")
        log.info("-" * 70)

    total_compared = 0
    total_mismatch = 0
    total_errors = 0
    max_post_diff = 0.0
    max_pre_diff = 0.0
    max_delta_diff = 0.0
    all_post_diffs = []
    all_pre_diffs = []
    all_delta_diffs = []
    league_stats = {}

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        gameid = row.get("gameid", "Unknown")
        league = row.get("league", "LCK")
        match_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        blue_team = row.get("blue_team", "?")
        red_team = row.get("red_team", "?")

        if verbose:
            log.info(f"\n--- 第 {idx}/{len(sample_df)} 场 [bp_delta] ---")
            log.info(f"  GameID: {gameid} | {league} | {match_date}")
            log.info(f"  {blue_team} vs {red_team}")

        # 构建 match_info
        match_info = build_match_info_from_row(row, known_champions, all_teams)

        # 提取 TF 特征 (线上/线下共用)
        tf_features = extract_tf_features_for_match(match_info)

        # 线上 bp_delta
        try:
            online_result = predict_bp_delta_online(
                match_info, stores, champion_tags, feature_cols, models
            )
        except Exception:
            if verbose:
                log.exception("  ❌ 线上 bp_delta 推理失败")
            total_errors += 1
            continue

        if online_result is None:
            if verbose:
                log.error("  ❌ 线上特征构建失败")
            total_errors += 1
            continue

        # 线下 bp_delta
        try:
            offline_result = predict_bp_delta_offline(
                row, feature_cols, models, tf_features=tf_features
            )
        except Exception:
            if verbose:
                log.exception("  ❌ 线下 bp_delta 推理失败")
            total_errors += 1
            continue

        total_compared += 1

        # 对比
        post_diff = abs(online_result["post_prob"] - offline_result["post_prob"])
        pre_diff = abs(online_result["pre_prob"] - offline_result["pre_prob"])
        delta_diff = abs(online_result["delta"] - offline_result["delta"])

        all_post_diffs.append(post_diff)
        all_pre_diffs.append(pre_diff)
        all_delta_diffs.append(delta_diff)
        max_post_diff = max(max_post_diff, post_diff)
        max_pre_diff = max(max_pre_diff, pre_diff)
        max_delta_diff = max(max_delta_diff, delta_diff)
        overall_max = max(post_diff, pre_diff, delta_diff)

        if league not in league_stats:
            league_stats[league] = {"count": 0, "mismatch": 0,
                                     "post_diffs": [], "pre_diffs": [], "delta_diffs": []}
        league_stats[league]["count"] += 1
        league_stats[league]["post_diffs"].append(post_diff)
        league_stats[league]["pre_diffs"].append(pre_diff)
        league_stats[league]["delta_diffs"].append(delta_diff)

        if verbose:
            log.info(f"  Post-Draft: 线上={online_result['post_prob']:.6f} | "
                     f"线下={offline_result['post_prob']:.6f} | diff={post_diff:.2e}")
            log.info(f"  Pre-Draft:  线上={online_result['pre_prob']:.6f} | "
                     f"线下={offline_result['pre_prob']:.6f} | diff={pre_diff:.2e}")
            log.info(f"  Delta:      线上={online_result['delta']:+.6f} | "
                     f"线下={offline_result['delta']:+.6f} | diff={delta_diff:.2e}")

        has_mismatch = (post_diff >= 1e-4 or pre_diff >= 1e-4 or delta_diff >= 1e-4)
        if not has_mismatch:
            if verbose:
                log.info(f"  ✅ 完全一致 (post={post_diff:.2e}, pre={pre_diff:.2e}, delta={delta_diff:.2e})")
        elif overall_max < 1e-4:
            if verbose:
                log.info(f"  ⚠️ 微小差异 — 浮点误差范围内")
            total_mismatch += 1
            league_stats[league]["mismatch"] += 1
        else:
            if verbose:
                log.info(f"  ❌ 不一致! (post={post_diff:.6f}, pre={pre_diff:.6f}, delta={delta_diff:.6f})")
                log.info(f"  Post-Draft 逐 seed 对比:")
                for si, (on, off) in enumerate(zip(online_result["post_seeds"],
                                                    offline_result["post_seeds"])):
                    sd = abs(on - off)
                    flag = "✅" if sd < 1e-6 else ("⚠️" if sd < 1e-4 else "❌")
                    log.info(f"    seed_{si}: online={on:.6f} | offline={off:.6f} | diff={sd:.2e} {flag}")
            total_mismatch += 1
            league_stats[league]["mismatch"] += 1

    # 汇总
    overall_max_diff = max(max_post_diff, max_pre_diff, max_delta_diff)
    passed = (total_mismatch == 0) or (overall_max_diff < 1e-4 and total_mismatch <= total_compared * 0.2)

    if verbose:
        log.info("\n" + "=" * 70)
        log.info("  BP Delta 一致性汇总")
        log.info("=" * 70)
        log.info(f"  对比场次: {total_compared}")
        log.info(f"  不一致场次: {total_mismatch}")
        log.info(f"  错误场次: {total_errors}")
        log.info(f"  Post-Draft 最大差异: {max_post_diff:.6f}")
        log.info(f"  Pre-Draft  最大差异: {max_pre_diff:.6f}")
        log.info(f"  Delta       最大差异: {max_delta_diff:.6f}")
        if all_post_diffs:
            log.info(f"  Post-Draft 平均差异: {np.mean(all_post_diffs):.6f} (std={np.std(all_post_diffs):.6f})")
        if all_pre_diffs:
            log.info(f"  Pre-Draft  平均差异: {np.mean(all_pre_diffs):.6f} (std={np.std(all_pre_diffs):.6f})")
        if all_delta_diffs:
            log.info(f"  Delta       平均差异: {np.mean(all_delta_diffs):.6f} (std={np.std(all_delta_diffs):.6f})")
        if league_stats:
            log.info("\n  按联赛汇总:")
            log.info(f"  {'联赛':<6} {'场次':<6} {'不一致':<8} {'Post最大差':<12} {'Pre最大差':<12} {'Delta最大差':<12}")
            log.info("  " + "-" * 62)
            for league in sorted(league_stats.keys()):
                s = league_stats[league]
                log.info(f"  {league:<6} {s['count']:<6} {s['mismatch']:<8} "
                         f"{max(s['post_diffs']):<12.6f} {max(s['pre_diffs']):<12.6f} "
                         f"{max(s['delta_diffs']):<12.6f}")
        if total_mismatch == 0:
            log.info("\n  ✅ 结论: bp_delta 线上/线下推理结果完全一致")
        elif overall_max_diff < 1e-4 and total_mismatch <= total_compared * 0.2:
            log.info(f"\n  ⚠️ 结论: 存在微小差异 (max={overall_max_diff:.2e})，在浮点误差范围内")
        else:
            log.info(f"\n  ❌ 结论: bp_delta 线上/线下推理结果存在显著差异，需要排查")
        log.info(f"\n  详细日志: {LOG_FILE}")

    return {
        "mode": "bp_delta",
        "passed": passed and total_errors == 0,
        "total": total_compared,
        "mismatch": total_mismatch,
        "errors": total_errors,
        "max_diff": float(overall_max_diff),
        "max_post_diff": float(max_post_diff),
        "max_pre_diff": float(max_pre_diff),
        "max_delta_diff": float(max_delta_diff),
    }


def run_quick_feature_check(num_samples=5):
    """快速特征级对齐校验 (使用 PredictBackend, 无需重复加载模型)。

    适用于 CI 快速门禁，仅检查特征值一致性，不做端到端预测对比。

    Returns:
        dict: {"passed": bool, "total": int, "mismatch": int, "errors": int}
    """
    try:
        from predict_backend import PredictBackend
    except ImportError as e:
        return {"passed": False, "total": 0, "mismatch": 0, "errors": 1, "error_msg": str(e)}

    try:
        backend = PredictBackend()
        backend.load()
        feature_cols = backend.feature_cols
    except Exception as e:
        return {"passed": False, "total": 0, "mismatch": 0, "errors": 1, "error_msg": str(e)}

    if not os.path.exists(WIDE_FEATURES_PATH):
        return {"passed": False, "total": 0, "mismatch": 0, "errors": 1,
                "error_msg": f"特征文件不存在: {WIDE_FEATURES_PATH}"}

    offline_df = pd.read_parquet(WIDE_FEATURES_PATH)
    offline_df["date_str"] = pd.to_datetime(offline_df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    positions = ["top", "jng", "mid", "bot", "sup"]
    sample_df = offline_df.sample(n=min(num_samples, len(offline_df)), random_state=42)

    total_checked = 0
    critical_mismatches = 0
    errors = 0
    CRITICAL_THRESHOLD = 0.5

    for _, row in sample_df.iterrows():
        current_game_num = 1
        for i in range(1, 6):
            if row.get(f"is_game_{i}", 0) == 1:
                current_game_num = i
                break

        mock_request = {
            "date": row.get("date_str"),
            "league": row.get("league", "LCK"),
            "is_playoff": bool(row.get("is_playoff", 0)),
            "game_num": current_game_num,
            "first_pick": "blue" if row.get("is_blue_map_side", 1) == 1 else "red",
            "blue_team": row.get("blue_team", ""),
            "red_team": row.get("red_team", ""),
            "blue_champions": {pos: row.get(f"blue_{pos}_champion", "") for pos in positions},
            "red_champions": {pos: row.get(f"red_{pos}_champion", "") for pos in positions},
            "blue_players": {pos: row.get(f"blue_{pos}_player_id", "") for pos in positions},
            "red_players": {pos: row.get(f"red_{pos}_player_id", "") for pos in positions},
        }

        try:
            match_info = backend._build_match_info(mock_request)
            online_features_df, _ = backend._build_features(match_info)
        except Exception:
            errors += 1
            continue

        online_vector = online_features_df.iloc[0]
        for col in feature_cols:
            if col.startswith("tf_"):
                continue
            total_checked += 1
            off_val = float(row.get(col, 0.0)) if not pd.isna(row.get(col)) else 0.0
            on_val = float(online_vector.get(col, 0.0)) if not pd.isna(online_vector.get(col)) else 0.0
            if abs(off_val - on_val) >= CRITICAL_THRESHOLD:
                critical_mismatches += 1

    return {
        "mode": "quick_feature",
        "passed": critical_mismatches == 0 and errors == 0,
        "total": total_checked,
        "mismatch": critical_mismatches,
        "errors": errors,
    }


if __name__ == "__main__":
    setup_logging()
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)

    parser = argparse.ArgumentParser(description="端到端推理一致性校验")
    parser.add_argument("--samples", type=int, default=50,
                        help="采样数量 (默认 50, 按联赛分层)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "predict", "bp_delta", "quick"],
                        help="检测模式: both=全量, predict=仅胜率, bp_delta=仅BP影响, quick=快速特征检查 (CI推荐)")
    parser.add_argument("--ci", action="store_true",
                        help="CI模式: 静默运行，仅输出JSON结果，通过退出码表示结果 (0=通过, 1=失败, 2=错误)")
    args = parser.parse_args()

    if args.ci:
        log.setLevel(logging.WARNING)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.WARNING)

    results = []
    has_error = False

    if args.mode in ("both", "predict"):
        r = run_prediction_comparison(num_samples=args.samples, verbose=not args.ci)
        results.append(r)
        if r.get("errors", 0) > 0 and r.get("total", 0) == 0:
            has_error = True

    if args.mode in ("both", "bp_delta"):
        if args.mode == "both" and not args.ci:
            log.info("\n\n")
        r = run_bp_delta_comparison(num_samples=args.samples, verbose=not args.ci)
        results.append(r)
        if r.get("errors", 0) > 0 and r.get("total", 0) == 0:
            has_error = True

    if args.mode == "quick":
        r = run_quick_feature_check(num_samples=args.samples)
        results.append(r)
        if r.get("errors", 0) > 0 and r.get("total", 0) == 0:
            has_error = True

    if args.ci:
        all_passed = all(r.get("passed", False) for r in results) and not has_error
        summary = {
            "passed": all_passed,
            "results": results,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if has_error:
            sys.exit(2)
        elif not all_passed:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        all_passed = all(r.get("passed", False) for r in results)
        if not all_passed:
            log.warning("⚠️  校验发现不一致，请查看上方详细日志")
        log.info("详细日志: %s", LOG_FILE)