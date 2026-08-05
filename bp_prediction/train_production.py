"""
生产模型训练脚本
==================
使用截止到指定日期的全部数据训练生产模型，用于实际推理。

与 OOT 验证的区别:
  - OOT: 5折滚动窗口，每折仅用12个月数据，用于评估模型泛化能力
  - 生产: 使用全部历史数据 + 时间衰减权重，最大化利用数据

训练策略:
  1. 全量数据 (截止到 cutoff_date) + 指数时间衰减权重 (半衰期 180 天)
  2. 联赛自适应权重: LCK=1.2x, LPL=1.0x, LEC=0.8x
  3. 镜像增强: 交换红蓝方特征 + 翻转结果
  4. Label Smoothing = 0.05
  5. 7-Seed Bagging + Early Stopping
  6. 可选 TF 特征 (使用最近折的 Transformer 快照)

用法:
  python train_production.py                          # 自动检测最新数据日期
  python train_production.py --cutoff 2026-06-07      # 指定截止日期
  python train_production.py --no-tf                  # 不使用 TF 特征
  python train_production.py --window-only            # 仅使用1年窗口数据
"""

import os
import sys
import json
import logging
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# =====================================================================
# 路径配置 & 统一配置管理 (必须在 logger_config 导入前设置 sys.path)
# =====================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(MODEL_DIR).parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from logger_config import get_logger, setup_logging, log_context, timed

FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

warnings.filterwarnings("ignore")

from bp_prediction.config import (
    Mode, set_mode, get_config, print_config_summary,
    FEATURES_DIR, WIDE_FEATURES_PATH, PRODUCTION_DIR,
    TF_FEATURES_DIR, TF_SNAPSHOTS_DIR, LOGS_DIR,
)
from common.paths import PRODUCTION_METRICS_DIR, ensure_dirs as _ensure_common_dirs

# 生产训练脚本强制使用生产模式
set_mode(Mode.PRODUCTION)

# 共享数据异常检测工具
from data_checks import check_dataframe, check_array, check_labels, check_predictions

_ensure_common_dirs()
for d in [PRODUCTION_DIR, LOGS_DIR, str(PRODUCTION_METRICS_DIR)]:
    os.makedirs(d, exist_ok=True)

# =====================================================================
# 日志
# =====================================================================
log = get_logger(__name__)

# =====================================================================
# 特征工程 (从 config.py 获取 CS 特征前缀)
# =====================================================================
_shared_cfg, _production_cfg = get_config(Mode.PRODUCTION)
CS_FEATURE_PREFIXES = _shared_cfg.cs_feature_prefixes

def load_wide_features():
    df = pd.read_parquet(WIDE_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # === 数据加载异常检查 ===
    _df_logger = get_logger("data_checks")
    check_dataframe("wide_features", df, _df_logger, context="生产宽表特征加载")
    return df

def get_feature_columns(df, exclude_cs=True):
    label_cols = {"gameid", "date", "league", "result", "split",
                  "playoffs", "first_pick_map_side", "patch"}
    meta_cols = {"blue_team", "red_team"}
    # 【修复】：排除 match_seq_idx — 它是行号索引，不是真实特征，线上无法预知，会导致线上/线下不一致
    leak_cols = {"match_seq_idx"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in label_cols and c not in meta_cols and c not in leak_cols]
    if exclude_cs:
        feature_cols = [c for c in feature_cols
                        if not any(c.startswith(p) for p in CS_FEATURE_PREFIXES)]
    return feature_cols

# =====================================================================
# 镜像增强
# =====================================================================
def create_mirror_samples(X_df, y_arr):
    mirror_X = X_df.copy()
    mirror_y = 1.0 - y_arr.copy()
    
    # 1. 翻转基础 blue/red 字段
    blue_cols = [c for c in X_df.columns if c.startswith("blue_")]
    for b_col in blue_cols:
        r_col = b_col.replace("blue_", "red_", 1)
        if r_col in X_df.columns:
            mirror_X[b_col] = X_df[r_col].values
            mirror_X[r_col] = X_df[b_col].values
            
    # 2. 翻转差值字段
    diff_cols = [c for c in X_df.columns if c.startswith("diff_")]
    for d_col in diff_cols:
        mirror_X[d_col] = -X_df[d_col].values
        
    # 3. 【修复】：显式翻转 TF 特征
    if "tf_blue_l2norm" in X_df.columns and "tf_red_l2norm" in X_df.columns:
        mirror_X["tf_blue_l2norm"] = X_df["tf_red_l2norm"].values
        mirror_X["tf_red_l2norm"] = X_df["tf_blue_l2norm"].values
        
    if "tf_win_logits" in X_df.columns:
        # tf_win_logits 是蓝方的绝对优势分，镜像后必须取反
        mirror_X["tf_win_logits"] = -X_df["tf_win_logits"].values
        
    # 注: tf_cosine_sim 是余弦相似度，属于对称标量，不需要修改
        
    return mirror_X, mirror_y

# =====================================================================
# 联赛自适应权重 (从 config.py 获取默认配置)
# =====================================================================
def compute_league_weights(train_df, weight_config=None):
    if weight_config is None:
        weight_config = _shared_cfg.league_weights
    weights = np.ones(len(train_df), dtype=np.float32)
    for league, w in weight_config.items():
        mask = train_df["league"] == league
        weights[mask] = w
        log.info("    [Weight] %s: %.1fx (%s samples)", league, w, mask.sum())
    return weights


# =====================================================================
# 方案 B: 从 OOT 验证结果计算生产模式固定轮数
# =====================================================================
def compute_production_iterations(n_production_samples):
    """从 OOT 验证的 best_iteration 计算生产模式的固定训练轮数.

    方案 B 核心逻辑:
      1. 读取 OOT 最后 3 折的 best_iteration 均值作为 base_iterations
      2. 按 √n 补偿: production_iterations = base × (n_prod / n_fold) ^ 0.5
      3. 考虑时间衰减等效数据量: n_prod_effective = n_prod × 0.65
      4. 关闭 early stopping, 用 100% 数据训练固定轮数
      5. LR × 0.85 作为正则化补偿

    Args:
        n_production_samples: 生产模式训练样本数 (镜像增强后)

    Returns:
        dict: {
            "production_iterations": int,
            "production_learning_rate": float,
            "base_iterations": float,
            "oot_source_path": str,
            "compensation_ratio": float,
            "fallback_used": bool,
        }
    """
    source_path = _production_cfg.oot_iterations_source_path

    if not _production_cfg.use_oot_driven_iterations or not os.path.exists(source_path):
        # 回退到旧模式: 使用 BEST_PARAMS 默认 iterations + early stopping
        log.info("  [Iterations] OOT 驱动轮数未启用或参数文件不存在, 回退到 early stopping 模式")
        log.info("    use_oot_driven_iterations=%s", _production_cfg.use_oot_driven_iterations)
        log.info("    source_path=%s", source_path)
        return {
            "production_iterations": None,  # None 表示用默认 iterations + early stopping
            "production_learning_rate": _shared_cfg.catboost_params["learning_rate"],
            "base_iterations": None,
            "oot_source_path": source_path,
            "compensation_ratio": None,
            "fallback_used": True,
        }

    # 读取 OOT 参数文件
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            oot_params = json.load(f)
    except Exception as e:
        log.exception("  [Iterations] 读取 OOT 参数文件失败: %s, 回退到 early stopping 模式", e)
        return {
            "production_iterations": None,
            "production_learning_rate": _shared_cfg.catboost_params["learning_rate"],
            "base_iterations": None,
            "oot_source_path": source_path,
            "compensation_ratio": None,
            "fallback_used": True,
        }

    base_iterations = oot_params["base_iterations"]
    last_fold_train_samples = oot_params["last_fold_train_samples"]
    n_recent_folds = oot_params["n_recent_folds"]

    # 计算等效数据膨胀比 (考虑时间衰减)
    # n_production_samples 是镜像增强后的样本数, 需要除以 2 还原原始比赛数
    n_production_raw = n_production_samples / 2.0
    n_production_effective = n_production_raw * _production_cfg.effective_data_ratio

    # √n 补偿
    data_expansion_ratio = n_production_effective / max(last_fold_train_samples, 1)
    compensation_ratio = data_expansion_ratio ** _production_cfg.expansion_exponent
    production_iterations = int(base_iterations * compensation_ratio)

    # 安全边界
    production_iterations = max(
        _production_cfg.min_production_iterations,
        min(_production_cfg.max_production_iterations, production_iterations)
    )

    # LR 衰减
    base_lr = _shared_cfg.catboost_params["learning_rate"]
    production_lr = base_lr * _production_cfg.lr_decay_factor

    log.info("  [Iterations] OOT 驱动轮数计算 (方案 B):")
    log.info("    OOT source           : %s", source_path)
    log.info("    Base iterations      : %.1f (last %s folds mean)", base_iterations, n_recent_folds)
    log.info("    Last fold train samples : %s", last_fold_train_samples)
    log.info("    Production samples (raw) : %.0f", n_production_raw)
    log.info("    Effective ratio      : %s (time decay)", _production_cfg.effective_data_ratio)
    log.info("    Production samples (eff) : %.0f", n_production_effective)
    log.info("    Expansion ratio      : %.4f", data_expansion_ratio)
    log.info("    Compensation (√n)    : %.4f", compensation_ratio)
    log.info("    Production iterations: %s (clamped to [%s, %s])",
             production_iterations,
             _production_cfg.min_production_iterations,
             _production_cfg.max_production_iterations)
    log.info("    Base LR              : %s", base_lr)
    log.info("    Production LR        : %s (× %s)", production_lr, _production_cfg.lr_decay_factor)

    return {
        "production_iterations": production_iterations,
        "production_learning_rate": production_lr,
        "base_iterations": base_iterations,
        "oot_source_path": source_path,
        "compensation_ratio": compensation_ratio,
        "data_expansion_ratio": data_expansion_ratio,
        "n_production_raw": n_production_raw,
        "n_production_effective": n_production_effective,
        "last_fold_train_samples": last_fold_train_samples,
        "fallback_used": False,
    }

# =====================================================================
# 时间衰减权重 (从 config.py 获取默认半衰期)
# =====================================================================
def compute_time_decay_weights(train_df, cutoff_date, half_life_days=None):
    """指数时间衰减权重: 距 cutoff_date 越近权重越高。

    weight = 2^(-days_ago / half_life_days)
    half_life_days=180: 180天前的数据权重为当前的一半
    """
    # 从 config.py 获取默认半衰期
    if half_life_days is None:
        half_life_days = _production_cfg.time_decay_half_life_days
    cutoff_dt = pd.Timestamp(cutoff_date)
    days_ago = (cutoff_dt - train_df["date"]).dt.days.clip(lower=0)
    weights = np.power(2.0, -days_ago.values.astype(float) / half_life_days)
    weights = weights.astype(np.float32)
    # 归一化到均值=1
    weights = weights / weights.mean()
    log.info("    [TimeDecay] half_life=%sd, weight range: [%.3f, %.3f], mean: %.3f",
             half_life_days, weights.min(), weights.max(), weights.mean())
    return weights

# =====================================================================
# 主流程
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Production Model Training")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="数据截止日期 (含), 默认自动检测最新数据日期")
    parser.add_argument("--no-tf", action="store_true",
                        help="不使用 TF 特征")
    parser.add_argument("--window-only", action="store_true",
                        help="仅使用1年窗口数据 (而非全量)")
    parser.add_argument("--half-life", type=int, default=_production_cfg.time_decay_half_life_days,
                        help=f"时间衰减半衰期天数 (默认 {_production_cfg.time_decay_half_life_days})")
    parser.add_argument("--min-date", type=str, default=_production_cfg.min_date,
                        help="数据最早日期 (含), 早于此日期的数据将被排除, 默认无限制")
    args = parser.parse_args()

    setup_logging()

    from config import resolve_cutoff_date
    cutoff_date = args.cutoff if args.cutoff is not None else resolve_cutoff_date(_production_cfg.cutoff_date)
    use_tf = not args.no_tf
    window_only = args.window_only
    half_life_days = args.half_life
    min_date = args.min_date

    training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("%s", "=" * 70)
    log.info("=" * 70)
    log.info("PRODUCTION MODEL TRAINING - Version: %s", version_tag)
    log.info("=" * 70)
    log.info("  Training Date  : %s", training_date)
    log.info("  Mode           : PRODUCTION (Full Data + Time Decay)")
    log.info("  Cutoff Date    : %s", cutoff_date)
    if min_date:
        log.info("  Min Date       : %s", min_date)
    log.info("  Data Strategy  : %s", '1-Year Window' if window_only else 'Full Data + Time Decay')
    if not window_only:
        log.info("  Time Decay     : Exponential, half_life=%sd", half_life_days)
    log.info("  TF Features    : %s", 'Enabled' if use_tf else 'Disabled')
    log.info("  Architecture   : CatBoost-%sSeed-Bagging", _shared_cfg.n_seeds)
    log.info("  Iterations     : %s (default)", _shared_cfg.catboost_params['iterations'])
    log.info("  Learning Rate  : %s", _shared_cfg.catboost_params['learning_rate'])
    log.info("  League Weights : %s", _shared_cfg.league_weights)
    log.info("  Label Smoothing: %s", _shared_cfg.label_smoothing)
    log.info("  Optimizations  : Mirror augmentation (%s), League-adaptive weights",
             _shared_cfg.mirror_augmentation)
    log.info("  Version Tag    : %s", version_tag)
    log.info("%s", "=" * 70)
    print_config_summary(Mode.PRODUCTION)

    # =====================================================================
    # Step 1: 加载数据
    # =====================================================================
    log.info("\nLoading data...")
    df = load_wide_features()
    feature_cols = get_feature_columns(df, exclude_cs=True)

    # 截止到 cutoff_date
    cutoff_dt = pd.Timestamp(cutoff_date)
    df = df[df["date"] <= cutoff_dt].copy().reset_index(drop=True)

    # 排除早于 min_date 的数据
    if min_date:
        min_dt = pd.Timestamp(min_date)
        n_before = len(df)
        df = df[df["date"] >= min_dt].copy().reset_index(drop=True)
        log.info("  [MinDate] Filtered: %s -> %s (removed %s matches before %s)",
                 n_before, len(df), n_before - len(df), min_date)

    if window_only:
        window_start = cutoff_dt - pd.DateOffset(months=12)
        df = df[df["date"] >= window_start].copy().reset_index(drop=True)
        log.info("  [Window] Using 1-year window: %s ~ %s", window_start.strftime('%Y-%m-%d'), cutoff_date)

    log.info("Dataset: %s matches | LPL: %s | LCK: %s | LEC: %s",
             len(df),
             (df['league']=='LPL').sum(),
             (df['league']=='LCK').sum(),
             (df['league']=='LEC').sum())
    log.info("Date range: %s ~ %s",
             df['date'].min().strftime('%Y-%m-%d'),
             df['date'].max().strftime('%Y-%m-%d'))
    log.info("Feature dim: %s", len(feature_cols))

    # =====================================================================
    # Step 2: 加载 TF 特征 (优先使用生产快照提取的特征)
    # =====================================================================
    tf_cols = []
    if use_tf:
        log.info("  [TF] Building Out-Of-Fold (OOF) TF features to prevent target leakage...")
        all_tf_records = []
        
        # 遍历 5 个 OOT 折，提取无泄漏的预测特征
        for fold_idx in range(5):
            tf_path = os.path.join(TF_FEATURES_DIR, f"{fold_idx}_tf_features.parquet")
            if os.path.exists(tf_path):
                tf_df = pd.read_parquet(tf_path)
                
                # 核心逻辑: 优先收集各折的 test 集 (绝对无泄漏的时序 OOF 特征)
                test_tf = tf_df[tf_df["split"] == "test"]
                all_tf_records.append(test_tf)
                
                # 对于最古老的数据 (Fold 0 的训练集以前), 我们只能使用 Fold 0 的 train 集打底
                # 此时 Transformer 只看过 2025-05 之前的数据，绝未被后期的版本更迭污染
                if fold_idx == 0:
                    train_tf = tf_df[tf_df["split"] == "train"]
                    all_tf_records.append(train_tf)

        if all_tf_records:
            # 拼接所有历史数据
            tf_combined = pd.concat(all_tf_records, ignore_index=True)
            # drop_duplicates keep='last' 确保如果同一局比赛出现多次，优先使用靠后的(通常是 test 集)
            tf_combined = tf_combined.drop_duplicates(subset=["gameid"], keep="last")
            tf_combined = tf_combined[["gameid", "tf_win_logits", "tf_cosine_sim", "tf_blue_l2norm", "tf_red_l2norm"]]
            
            # 合并到主数据流
            df = df.merge(tf_combined, on="gameid", how="left")
            tf_cols = ["tf_win_logits", "tf_cosine_sim", "tf_blue_l2norm", "tf_red_l2norm"]
            
            for col in tf_cols:
                df[col] = df[col].fillna(df[col].median())
                
            matched = df["tf_win_logits"].notna().sum()
            log.info("  [TF] Merged OOF TF features: %s/%s matched (%.1f%%)",
                     matched, len(df), matched/len(df)*100)
        else:
            log.info("  [TF] No OOF TF features found, training without TF")

    # 合并特征列: 基础特征 + TF 特征
    all_feature_cols = feature_cols + tf_cols
    log.info("  Feature columns: %s base + %s TF = %s total",
             len(feature_cols), len(tf_cols), len(all_feature_cols))

    # =====================================================================
    # Step 3: 准备训练数据
    # =====================================================================
    log.info("\nPreparing training data...")

    y_train_raw = df["result"].values.astype(float)

    # 镜像增强
    log.info("  [Augmentation] Mirror augmentation...")
    X_train_features = df[all_feature_cols].copy()
    mirror_X, mirror_y = create_mirror_samples(X_train_features, y_train_raw)

    X_train_aug = pd.concat([X_train_features, mirror_X], ignore_index=True)
    y_train_aug = np.concatenate([y_train_raw, mirror_y])

    # 镜像后的 league 列和 date 列 (用于权重计算)
    train_leagues_aug = pd.concat([df["league"], df["league"]], ignore_index=True)
    train_dates_aug = pd.concat([df["date"], df["date"]], ignore_index=True)

    log.info("    [Mirror] Original: %s, Mirror: %s, Combined: %s",
             len(X_train_features), len(mirror_X), len(X_train_aug))

    # 联赛自适应权重
    log.info("  [Weights] League-adaptive sample weights...")
    train_df_for_weight = pd.DataFrame({"league": train_leagues_aug})
    league_weights = compute_league_weights(train_df_for_weight)

    # 时间衰减权重
    if not window_only:
        log.info("  [Weights] Time-decay sample weights...")
        train_df_for_time = pd.DataFrame({"date": train_dates_aug})
        time_weights = compute_time_decay_weights(train_df_for_time, cutoff_date, half_life_days)
        # 综合权重 = 联赛权重 × 时间衰减权重
        sample_weights = league_weights * time_weights
        log.info("    [Combined] Final weight range: [%.3f, %.3f]",
                 sample_weights.min(), sample_weights.max())
    else:
        sample_weights = league_weights

    # Label Smoothing (从 config.py 获取)
    LABEL_SMOOTHING = _shared_cfg.label_smoothing
    y_train_smooth = y_train_aug * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

    # 准备特征矩阵
    X_train = X_train_aug[all_feature_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    log.info("  Final training set: %s samples, %s features",
             len(X_train), len(all_feature_cols))

    # =====================================================================
    # Step 4: 训练 7-Seed Bagging CatBoost
    # =====================================================================
    from catboost import CatBoostClassifier, Pool

    # 从 config.py 获取训练参数
    N_SEEDS = _shared_cfg.n_seeds
    SEEDS = _shared_cfg.seeds
    BEST_PARAMS = _shared_cfg.catboost_params.copy()

    # ---------------------------------------------------------------------
    # 方案 B: 从 OOT 验证结果计算生产模式固定轮数
    # ---------------------------------------------------------------------
    # 生产模式从开发模式 (OOT) 读取 best_iteration, 按 √n 补偿计算固定轮数,
    # 关闭 early stopping, 用 100% 数据训练.
    # 与推荐模型 "开发模式定停止点 → 生产模式全量训练" 逻辑一致.
    iter_config = compute_production_iterations(n_production_samples=len(X_train))

    log.info("\n%s", "="*70)
    log.info("TRAINING %s-SEED BAGGING CATBOOST", N_SEEDS)
    if iter_config["fallback_used"]:
        log.info("  Mode: FALLBACK (80/20 early stopping, 80% data)")
    else:
        log.info("  Mode: OOT-DRIVEN (fixed iterations, 100% data, no early stopping)")
        log.info("  Iterations: %s", iter_config['production_iterations'])
        log.info("  Learning rate: %.6f", iter_config['production_learning_rate'])
    log.info("%s", "="*70)

    production_models = []
    production_best_iters = []
    seed_train_times = []

    for seed_idx, seed in enumerate(SEEDS):
        with log_context(Seed=seed):
            seed_start = time.time()
            log.info("\n  Seed %s/%s (seed=%s)...", seed_idx+1, N_SEEDS, seed)
            params = BEST_PARAMS.copy()
            params["random_seed"] = seed

            if iter_config["fallback_used"]:
                model = CatBoostClassifier(**params)

                n_orig = len(X_train_features)
                n_val_orig = int(n_orig * _shared_cfg.val_split_ratio)

                perm = np.random.RandomState(seed).permutation(n_orig)
                val_idx_orig = perm[:n_val_orig]
                tr_idx_orig = perm[n_val_orig:]

                val_idx = np.concatenate([val_idx_orig, val_idx_orig + n_orig])
                tr_idx = np.concatenate([tr_idx_orig, tr_idx_orig + n_orig])

                train_pool = Pool(
                    X_train[tr_idx], y_train_smooth[tr_idx],
                    weight=sample_weights[tr_idx]
                )
                val_pool = Pool(
                    X_train[val_idx], y_train_smooth[val_idx],
                    weight=sample_weights[val_idx]
                )

                model.fit(train_pool, eval_set=val_pool, verbose=0,
                          early_stopping_rounds=_shared_cfg.early_stopping_rounds, use_best_model=True)

                best_iter = model.get_best_iteration()
                evals_result = model.get_evals_result()
                train_loss = evals_result['learn']['Logloss'][-1] if 'learn' in evals_result else None
                val_loss = evals_result['validation']['Logloss'][-1] if 'validation' in evals_result else None
                log.info("    Best iteration: %s%s%s", best_iter,
                         f", train_loss={train_loss:.4f}" if train_loss else "",
                         f", val_loss={val_loss:.4f}" if val_loss else "")
                if best_iter < _shared_cfg.early_stopping_rounds * 0.3:
                    log.warning("    Early stopping triggered very early at iter %s", best_iter)
                production_best_iters.append(int(best_iter))

            else:
                params["iterations"] = iter_config["production_iterations"]
                params["learning_rate"] = iter_config["production_learning_rate"]

                model = CatBoostClassifier(**params)

                full_pool = Pool(
                    X_train, y_train_smooth,
                    weight=sample_weights
                )

                model.fit(full_pool, verbose=0)

                log.info("    Trained %s iterations (lr=%.6f, 100%% data, no early stopping)",
                         iter_config['production_iterations'],
                         iter_config['production_learning_rate'])

            production_models.append(model)
            seed_elapsed = time.time() - seed_start
            seed_train_times.append(seed_elapsed)
            log.info("    Seed training completed in %.1fs", seed_elapsed)

    log.info("\n  All %s seeds trained successfully. Avg seed time: %.1fs, Total: %.1fs",
             N_SEEDS, np.mean(seed_train_times), sum(seed_train_times))

    # =====================================================================
    # Step 5: 保存生产模型
    # =====================================================================
    log.info("\n%s", "="*70)
    log.info("SAVING PRODUCTION MODEL")
    log.info("%s", "="*70)

    total_model_size = 0
    for seed_idx, model in enumerate(production_models):
        model_path = os.path.join(PRODUCTION_DIR, f"catboost_seed_{seed_idx}.cbm")
        model.save_model(model_path)
        fsize = os.path.getsize(model_path) / (1024*1024)
        total_model_size += fsize
        log.info("  Saved: %s (%.2f MB)", model_path, fsize)

    with open(os.path.join(PRODUCTION_DIR, "feature_columns.json"), "w") as f:
        json.dump(all_feature_cols, f)
    fc_size = os.path.getsize(os.path.join(PRODUCTION_DIR, "feature_columns.json")) / 1024
    log.info("  Saved: feature_columns.json (%s features, %.2f KB)", len(all_feature_cols), fc_size)

    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cutoff_date": cutoff_date,
        "min_date": min_date,
        "data_strategy": "1_year_window" if window_only else "full_data_time_decay",
        "half_life_days": half_life_days if not window_only else None,
        "n_raw_matches": len(df) // 2,
        "n_augmented": len(X_train),
        "n_features": len(all_feature_cols),
        "tf_features": len(tf_cols) > 0,
        "tf_feature_count": len(tf_cols),
        "label_smoothing": LABEL_SMOOTHING,
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "best_params": BEST_PARAMS,
        "league_weights": _shared_cfg.league_weights,
        "mirror_augmentation": True,
        "date_range": {
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
        },
        "iterations_config": {
            "mode": "fallback_early_stopping" if iter_config["fallback_used"] else "oot_driven_fixed",
            "fallback_used": iter_config["fallback_used"],
            "production_iterations": iter_config.get("production_iterations"),
            "production_learning_rate": iter_config.get("production_learning_rate"),
            "base_iterations": iter_config.get("base_iterations"),
            "compensation_ratio": iter_config.get("compensation_ratio"),
            "data_expansion_ratio": iter_config.get("data_expansion_ratio"),
            "oot_source_path": iter_config.get("oot_source_path"),
            "production_best_iters": production_best_iters if iter_config["fallback_used"] else None,
        },
    }
    metadata["version_tag"] = version_tag
    metadata["training_date"] = training_date
    with open(os.path.join(PRODUCTION_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    meta_size = os.path.getsize(os.path.join(PRODUCTION_DIR, "metadata.json")) / 1024
    log.info("  Saved: metadata.json (version=%s, %.2f KB)", version_tag, meta_size)
    log.info("  Total model size: %.2f MB across %s seeds", total_model_size, N_SEEDS)

    # =====================================================================
    # Step 6: OOT 验证 (训练时排除最近2个月，仅用排除期做评估)
    # 注意: 自验证 (在训练集上评估) 会导致 AUC 虚高 (0.9999)，
    #       必须用 OOT 方式才能反映真实泛化性能
    # =====================================================================
    log.info("\n%s", "="*70)
    log.info("OOT VALIDATION (Hold-out recent 2 months)")
    log.info("%s", "="*70)

    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

    val_start = cutoff_dt - pd.DateOffset(months=2)
    val_mask = (df["date"] >= val_start) & (df["date"] <= cutoff_dt)
    val_df = df[val_mask].copy()

    if len(val_df) > 20:
        # OOT: 仅用 val_start 之前的数据训练临时模型
        oot_train_df = df[df["date"] < val_start].copy()
        log.info("  OOT train: %s ~ %s (%s matches)",
                 oot_train_df['date'].min().strftime('%Y-%m-%d'),
                 oot_train_df['date'].max().strftime('%Y-%m-%d'),
                 len(oot_train_df))
        log.info("  OOT test : %s ~ %s (%s matches)",
                 val_start.strftime('%Y-%m-%d'), cutoff_date, len(val_df))

        # 准备 OOT 训练数据 (简化版: 不做镜像增强, 仅用单 seed 快速评估)
        X_oot_train = np.nan_to_num(oot_train_df[all_feature_cols].values.astype(np.float32))
        y_oot_train = oot_train_df["result"].values.astype(float)
        X_val = np.nan_to_num(val_df[all_feature_cols].values.astype(np.float32))
        y_val = val_df["result"].values.astype(float)

        oot_model = CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.035,
            l2_leaf_reg=5.0, random_seed=42, verbose=0,
            eval_metric="AUC", loss_function="Logloss",
        )
        oot_model.fit(X_oot_train, y_oot_train, verbose=0)
        val_pred = oot_model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, val_pred)
        acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        ll = log_loss(y_val, val_pred)
        brier = brier_score_loss(y_val, val_pred)

        log.info("  OOT AUC     = %.4f", auc)
        log.info("  OOT ACC     = %.4f", acc)
        log.info("  OOT LogLoss = %.4f", ll)
        log.info("  OOT Brier   = %.4f", brier)

        league_metrics = {}
        for league in ["LPL", "LCK", "LEC"]:
            league_mask = val_df["league"] == league
            if league_mask.sum() >= 10:
                y_l = y_val[league_mask]
                p_l = val_pred[league_mask]
                try:
                    auc_l = roc_auc_score(y_l, p_l)
                    log.info("  %s OOT AUC = %.4f (N=%s)", league, auc_l, league_mask.sum())
                    league_metrics[league] = {"auc": float(auc_l), "n_samples": int(league_mask.sum())}
                except Exception:
                    log.info("  %s: insufficient label diversity", league)
        oot_metrics = {
            "auc": float(auc), "acc": float(acc), "logloss": float(ll), "brier": float(brier),
            "n_val": int(len(val_df)), "leagues": league_metrics,
        }
    else:
        log.info("  Not enough recent data for OOT validation (N=%s)", len(val_df))
        oot_metrics = {"auc": None, "note": f"insufficient OOT data (N={len(val_df)})"}

    prod_metrics = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "bp_prediction",
            "mode": "production",
            "version_tag": version_tag,
            "training_date": training_date,
            "n_seeds": N_SEEDS,
            "n_features": len(all_feature_cols),
            "n_base_features": len(feature_cols),
            "n_tf_features": len(tf_cols),
            "data_strategy": "1_year_window" if window_only else "full_data_time_decay",
            "half_life_days": half_life_days if not window_only else None,
            "total_model_size_mb": round(total_model_size, 2),
            "date_range": {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            },
            "n_raw_matches": len(df) // 2,
            "n_augmented_samples": len(X_train),
            "best_params": BEST_PARAMS,
            "league_weights": _shared_cfg.league_weights,
        },
        "oot_validation": oot_metrics,
    }
    metrics_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prod_metrics_path = os.path.join(str(PRODUCTION_METRICS_DIR), f"prediction_production_{version_tag}_{metrics_ts}.json")
    with open(prod_metrics_path, "w", encoding="utf-8") as f:
        json.dump(prod_metrics, f, indent=2, ensure_ascii=False)
    log.info("  Production metrics saved to: %s", prod_metrics_path)

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"production_{version_tag}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT)
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    log.info("\n%s", "="*70)
    log.info("="*70)
    log.info("PRODUCTION MODEL TRAINING COMPLETE - Version: %s", version_tag)
    log.info("="*70)
    log.info("  Training Date  : %s", training_date)
    log.info("  Model Directory: %s", PRODUCTION_DIR)
    log.info("  Total Model Size: %.2f MB (%s seeds)", total_model_size, N_SEEDS)
    log.info("  Dataset: %d raw matches, %d augmented samples", len(df)//2, len(X_train))
    log.info("  Date Range: %s ~ %s", df['date'].min().strftime('%Y-%m-%d'), df['date'].max().strftime('%Y-%m-%d'))
    log.info("  Features: %d (%d base + %d TF)", len(all_feature_cols), len(feature_cols), len(tf_cols))
    if len(val_df) > 20 and 'auc' in locals():
        log.info("  Consistency Check: OOT AUC=%.4f (N=%d) - %s", auc, len(val_df), "PASSED" if auc > 0.55 else "WARNING")
        log.info("  OOT Metrics: ACC=%.4f, LogLoss=%.4f, Brier=%.4f", acc, ll, brier)
    log.info("  Log File: %s", log_path)
    log.info("="*70)

    if file_handler:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    main()
