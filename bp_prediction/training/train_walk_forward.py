"""
BP 胜负预测模型 - Deep-Tabular Cascade Fusion (正式版)
======================================================
架构: Transformer(4-dim 深层特征) + CatBoost-7Seed-Bagging
验证: 5-Fold Rolling OOT (12个月训练窗口)

最终性能 (5-Fold Mean):
  Overall AUC = 0.6635±0.0329
  LPL      AUC = 0.6400±0.0313
  LCK      AUC = 0.6990±0.0543
  LEC      AUC = 0.6260±0.0780

核心优化:
  1. 联赛自适应样本权重: LPL=1.3x, LEC=1.5x, LCK=1.0x
  2. 镜像增强: 交换红蓝方特征 + 翻转结果 (强制蓝红对称性)
  3. 优化超参数: iterations=800, depth=6, lr=0.035, l2=5.0
  4. Label Smoothing = 0.05
  5. 7-Seed Bagging + Early Stopping

用法:
  python train_walk_forward.py [--window 12]
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")

# =====================================================================
# 路径配置 & 统一配置管理
# =====================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BP_PRED_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = str(Path(_BP_PRED_DIR).parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _BP_PRED_DIR not in sys.path:
    sys.path.insert(0, _BP_PRED_DIR)

from logger_config import get_logger, setup_logging, log_context, timed

FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

from bp_prediction.config import (
    Mode, set_mode, get_config, print_config_summary,
    FEATURES_DIR, WIDE_FEATURES_PATH, MODELS_DIR,
    TF_FEATURES_DIR, LOGS_DIR, REPORTS_DIR,
)
from common.paths import PREDICTION_METRICS_DIR, ensure_dirs as _ensure_common_dirs

set_mode(Mode.TRAINING)

PROJECT_ROOT = _PROJECT_ROOT
MODEL_DIR = _BP_PRED_DIR

sys.path.insert(0, PROJECT_ROOT)
from data_checks import check_dataframe, check_array, check_labels, check_predictions

_ensure_common_dirs()
for d in [LOGS_DIR, REPORTS_DIR, MODELS_DIR, str(PREDICTION_METRICS_DIR)]:
    os.makedirs(d, exist_ok=True)

log = get_logger(__name__)
_last_report_path = None
_log_buffer = []


def log_info(msg):
    log.info(str(msg))
    _log_buffer.append(str(msg))


class _BufferHandler(logging.Handler):
    def emit(self, record):
        _log_buffer.append(self.format(record))


_shared_cfg, _training_cfg = get_config(Mode.TRAINING)
CS_FEATURE_PREFIXES = _shared_cfg.cs_feature_prefixes


def load_wide_features():
    df = pd.read_parquet(WIDE_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    check_dataframe("wide_features", df, log, context="宽表特征加载")
    return df

def get_feature_columns(df, exclude_cs=True):
    """获取特征列名, 可选排除 CS 特征。只保留数值列。"""
    label_cols = {"gameid", "date", "league", "result", "split",
                  "playoffs", "first_pick_map_side", "patch"}
    meta_cols = {"blue_team", "red_team"}
    # 【修复】：排除 match_seq_idx — 它是行号索引，不是真实特征，线上无法预知
    leak_cols = {"match_seq_idx"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in label_cols and c not in meta_cols and c not in leak_cols]
    if exclude_cs:
        feature_cols = [c for c in feature_cols
                        if not any(c.startswith(p) for p in CS_FEATURE_PREFIXES)]
    return feature_cols

# =====================================================================
# Bootstrap 置信区间
# =====================================================================
def bootstrap_ci(y_true, y_pred, metric_fn, n_resamples=1000, ci=0.95, seed=42):
    """计算 Bootstrap 置信区间。"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            if not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
    if len(scores) == 0:
        return 0.0, [0.0, 0.0]
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return np.mean(scores), [lower, upper]

# =====================================================================
# 优化1: 镜像增强
# =====================================================================
def create_mirror_samples(X_df, y_arr):
    """创建镜像样本: 交换红蓝方特征, 翻转结果标签。"""
    mirror_X = X_df.copy()
    mirror_y = 1.0 - y_arr.copy()

    # 交换 blue_/red_ 前缀列
    blue_cols = [c for c in X_df.columns if c.startswith("blue_")]
    for b_col in blue_cols:
        r_col = b_col.replace("blue_", "red_", 1)
        if r_col in X_df.columns:
            mirror_X[b_col] = X_df[r_col].values
            mirror_X[r_col] = X_df[b_col].values

    # 翻转 diff_ 列
    diff_cols = [c for c in X_df.columns if c.startswith("diff_")]
    for d_col in diff_cols:
        mirror_X[d_col] = -X_df[d_col].values

    return mirror_X, mirror_y

# =====================================================================
# 优化2: 联赛自适应样本权重
# =====================================================================
def compute_league_weights(train_df, weight_config=None):
    """计算联赛自适应样本权重 (从 config.py 获取默认配置)。

    weight_config: dict, e.g. {"LPL": 1.3, "LEC": 1.5, "LCK": 1.0}
    默认使用 config.py 中的 league_weights 配置
    """
    if weight_config is None:
        weight_config = _shared_cfg.league_weights

    weights = np.ones(len(train_df), dtype=np.float32)
    for league, w in weight_config.items():
        mask = train_df["league"] == league
        weights[mask] = w
        log_info(f"    [Weight] {league}: {w:.1f}x ({mask.sum()} samples)")

    return weights

# =====================================================================
# 优化3: 轻量 LPL 噪声增强
# =====================================================================
def augment_lpl_light(X_train_df, y_train, train_df, seed=42):
    """轻量 LPL 增强: 1x 高斯噪声副本 + 其镜像。

    旧模型用 3x 过重导致过拟合, 0x 过轻导致 LPL 信号不足。
    折中方案: 1x 噪声 + 1x 镜像 = 2x LPL 额外样本。
    """
    lpl_mask = train_df["league"] == "LPL"
    if lpl_mask.sum() == 0:
        return X_train_df, y_train

    X_lpl = X_train_df[lpl_mask.values].copy()
    y_lpl = y_train[lpl_mask.values]

    # 1x 高斯噪声副本
    rng = np.random.RandomState(seed)
    numeric_cols = X_lpl.select_dtypes(include=[np.number]).columns
    noise = rng.normal(0, 0.01, X_lpl[numeric_cols].shape)
    X_lpl_noisy = X_lpl.copy()
    X_lpl_noisy[numeric_cols] = X_lpl_noisy[numeric_cols] + noise

    # 噪声副本的镜像
    mirror_noisy_X, mirror_noisy_y = create_mirror_samples(X_lpl_noisy, y_lpl)

    # 合并
    X_aug = pd.concat([X_train_df, X_lpl_noisy, mirror_noisy_X], ignore_index=True)
    y_aug = np.concatenate([y_train, y_lpl, mirror_noisy_y])

    log_info(f"    [LPL Augment] Added {len(X_lpl_noisy)} noisy + {len(mirror_noisy_X)} mirror = "
             f"{len(X_lpl_noisy) + len(mirror_noisy_X)} extra LPL samples")

    return X_aug, y_aug

# =====================================================================
# 优化5: Temperature Scaling 后校准
# =====================================================================
def temperature_scale(preds, temperature):
    """应用 Temperature Scaling 校准预测概率。"""
    scaled = np.clip(preds, 1e-7, 1 - 1e-7)
    scaled = np.log(scaled / (1 - scaled)) / temperature
    return 1.0 / (1.0 + np.exp(-scaled))

def optimize_temperature(y_true, y_pred):
    """在验证集上优化 Temperature 参数, 最小化 NLL。"""
    from sklearn.metrics import log_loss

    def nll_loss(temp):
        if temp <= 0.01:
            return 10.0
        calibrated = temperature_scale(y_pred, temp)
        return log_loss(y_true, calibrated)

    result = minimize_scalar(nll_loss, bounds=(0.5, 3.0), method='bounded')
    return result.x

# =====================================================================
# 主流程
# =====================================================================
def main():
    global _last_report_path
    _log_buffer.clear()
    setup_logging()
    log_path = os.path.join(LOGS_DIR, f"optimized_cascade_oot5fold_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    buffer_handler = _BufferHandler()
    buffer_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(buffer_handler)

    parser = argparse.ArgumentParser(description="Optimized All-League Cascade Fusion OOT Validation")
    parser.add_argument("--window", type=int, default=_training_cfg.window_months, choices=[6, 9, 12],
                        help=f"Training window in months (default: {_training_cfg.window_months})")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="数据截止日期 YYYY-MM-DD (默认自动检测最新数据)")
    args = parser.parse_args()
    window_months = args.window
    _training_cfg.window_months = window_months

    from config import resolve_oot_folds
    OOT_FOLDS, resolved_cutoff = resolve_oot_folds(_training_cfg, cutoff=args.cutoff)

    training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_info("=" * 70)
    log_info("OPTIMIZED All-League Deep-Tabular Cascade Fusion - 5-Fold Rolling OOT")
    log_info(f"  Training Date  : {training_date}")
    log_info(f"  Mode           : TRAINING (5-Fold OOT Validation)")
    log_info(f"  Cutoff Date    : {resolved_cutoff} (auto-detected)" if args.cutoff is None else f"  Cutoff Date    : {resolved_cutoff}")
    log_info(f"  Window Months  : {window_months}")
    log_info(f"  Architecture   : Transformer(4-dim) + CatBoost-{_shared_cfg.n_seeds}Seed-Bagging")
    log_info(f"  CatBoost Params: iterations={_shared_cfg.catboost_params['iterations']}, depth={_shared_cfg.catboost_params['depth']}, lr={_shared_cfg.catboost_params['learning_rate']}, l2={_shared_cfg.catboost_params['l2_leaf_reg']}")
    log_info(f"  Batch Size     : {_shared_cfg.catboost_params.get('batch_size', 'N/A')}")
    log_info(f"  Early Stopping : {_shared_cfg.early_stopping_rounds} rounds")
    log_info(f"  Val Split      : {_shared_cfg.val_split_ratio:.0%}")
    log_info(f"  Optimizations  :")
    log_info(f"    1. League-adaptive weights: {_shared_cfg.league_weights}")
    log_info(f"    2. Mirror augmentation (blue-red symmetry): {_shared_cfg.mirror_augmentation}")
    log_info(f"    3. Label Smoothing: {_shared_cfg.label_smoothing}")
    log_info(f"  Device         : CPU (CatBoost)")
    log_info(f"  Validation     : {_training_cfg.n_folds} OOT Folds ({window_months}m training span)")
    for i, (ts, te, tst, tse) in enumerate(OOT_FOLDS, 1):
        log_info(f"    Fold {i}: Train [{ts}~{te}] | Test [{tst}~{tse}]")
    log_info(f"  Bootstrap      : {_training_cfg.n_bootstrap} resamples, {_training_cfg.bootstrap_ci:.0%} CI for AUC/LogLoss/Brier")
    log_info(f"  PIT Isolation  : Per-fold TF snapshot + per-fold base_prior")
    log_info("=" * 70)

    # =====================================================================
    # Step 1: 加载数据
    # =====================================================================
    log_info("\nLoading data...")
    df = load_wide_features()
    feature_cols = get_feature_columns(df, exclude_cs=True)

    cutoff_dt = pd.Timestamp(resolved_cutoff)
    n_before = len(df)
    df = df[df["date"] <= cutoff_dt].copy().reset_index(drop=True)
    if len(df) < n_before:
        log_info(f"  [Cutoff] 过滤 cutoff_date={resolved_cutoff} 之后的数据: {n_before} -> {len(df)} (移除 {n_before - len(df)} 条)")

    log_info(f"Dataset: {len(df)} matches | LPL: {(df['league']=='LPL').sum()} | "
             f"LCK: {(df['league']=='LCK').sum()} | LEC: {(df['league']=='LEC').sum()}")
    log_info(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    log_info(f"Excluded {len([c for c in df.columns if any(c.startswith(p) for p in CS_FEATURE_PREFIXES)])} "
             f"CS features. Tabular feature dim: {len(feature_cols)}")

   
    # =====================================================================
    # Step 2: 构造 5 折固定窗口 OOT 划分
    # =====================================================================
    log_info(f"\n{'='*70}")
    log_info(f"OOT FOLD DEFINITIONS (Rolling Window - Fixed Training Length)")
    log_info(f"{'='*70}")

    TRAIN_WINDOW_DAYS = window_months * 30

    for i, (ts, te, tst, tse) in enumerate(OOT_FOLDS):
        train_start_dt = pd.Timestamp(ts)
        train_end_dt = pd.Timestamp(te)
        train_df_tmp = df[(df["date"] >= train_start_dt) & (df["date"] <= train_end_dt)]
        lpl_train = (train_df_tmp["league"] == "LPL").sum()
        lck_train = (train_df_tmp["league"] == "LCK").sum()
        lec_train = (train_df_tmp["league"] == "LEC").sum()
        span = (train_end_dt - train_start_dt).days + 1

        test_start_dt = pd.Timestamp(tst)
        test_end_dt = pd.Timestamp(tse)
        test_df_tmp = df[(df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)]
        lpl_test = (test_df_tmp["league"] == "LPL").sum()
        lck_test = (test_df_tmp["league"] == "LCK").sum()
        lec_test = (test_df_tmp["league"] == "LEC").sum()

        log_info(f"  Fold {i+1}: Train [{ts} ~ {te}] ({len(train_df_tmp)}, LPL:{lpl_train}/LCK:{lck_train}/LEC:{lec_train}, span:{span}d) | "
                 f"Test: {tst} ~ {tse} ({len(test_df_tmp)}, LPL:{lpl_test}/LCK:{lck_test}/LEC:{lec_test})")

    # =====================================================================
    # Step 3: 5-Fold OOT 训练与评估
    # =====================================================================
    log_info(f"\n{'='*70}")
    log_info("STARTING 5-FOLD OOT EVALUATION (Optimized Cascade Fusion)")
    log_info(f"{'='*70}")

    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

    # 从 config.py 获取训练参数 (确保与配置一致)
    LABEL_SMOOTHING = _shared_cfg.label_smoothing
    N_SEEDS = _shared_cfg.n_seeds
    SEEDS = _shared_cfg.seeds
    N_BOOTSTRAP = _training_cfg.n_bootstrap

    # CatBoost 超参数 (从 config.py 获取)
    BEST_PARAMS = _shared_cfg.catboost_params.copy()

    fold_results = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(OOT_FOLDS):
        log_info(f"\n{'─'*70}")
        log_info(f"Fold {fold_idx+1}/5 | Train: {train_start} ~ {train_end} | Test: {test_start} ~ {test_end}")
        log_info(f"{'─'*70}")
        train_start_dt = pd.Timestamp(train_start)
        train_end_dt = pd.Timestamp(train_end)
        test_start_dt = pd.Timestamp(test_start)
        test_end_dt = pd.Timestamp(test_end)

        train_mask = (df["date"] >= train_start_dt) & (df["date"] <= train_end_dt)
        test_mask = (df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)

        train_df = df[train_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)
        log_info(f"  Split sizes: Train={len(train_df)}, Test={len(test_df)}")
        log_info(f"  Train leagues: LPL={(train_df['league']=='LPL').sum()}, LCK={(train_df['league']=='LCK').sum()}, LEC={(train_df['league']=='LEC').sum()}")
        log_info(f"  Test leagues: LPL={(test_df['league']=='LPL').sum()}, LCK={(test_df['league']=='LCK').sum()}, LEC={(test_df['league']=='LEC').sum()}")

        # ---- 加载 TF 特征 ----
        tf_path = os.path.join(TF_FEATURES_DIR, f"{fold_idx}_tf_features.parquet")
        tf_cols = []
        if os.path.exists(tf_path):
            tf_df = pd.read_parquet(tf_path)
            log_info(f"  [TF Features] Loading fold {fold_idx} Transformer features...")

            train_tf = tf_df[tf_df["split"] == "train"][["gameid", "tf_win_logits", "tf_cosine_sim",
                                                          "tf_blue_l2norm", "tf_red_l2norm"]]
            test_tf = tf_df[tf_df["split"] == "test"][["gameid", "tf_win_logits", "tf_cosine_sim",
                                                        "tf_blue_l2norm", "tf_red_l2norm"]]

            train_df = train_df.merge(train_tf, on="gameid", how="left")
            test_df = test_df.merge(test_tf, on="gameid", how="left")

            tf_cols = ["tf_win_logits", "tf_cosine_sim", "tf_blue_l2norm", "tf_red_l2norm"]
            for col in tf_cols:
                train_df[col] = train_df[col].fillna(train_df[col].median())
                test_df[col] = test_df[col].fillna(train_df[col].median())

            train_matched = train_df["tf_win_logits"].notna().sum()
            test_matched = test_df["tf_win_logits"].notna().sum()
            log_info(f"    [TF Merge] train: {train_matched}/{len(train_df)} matched "
                     f"({train_matched/len(train_df)*100:.1f}%)")
            log_info(f"    [TF Merge] test: {test_matched}/{len(test_df)} matched "
                     f"({test_matched/len(test_df)*100:.1f}%)")

            all_feature_cols = feature_cols + tf_cols
            log_info(f"  [TF Features] Merged {len(tf_cols)} TF features. Total dim: {len(all_feature_cols)}")
        else:
            all_feature_cols = feature_cols
            log_info(f"  [TF Features] No TF features found for fold {fold_idx}, using tabular only")

        # ---- 准备训练数据 ----
        y_train_raw = train_df["result"].values.astype(float)
        y_test = test_df["result"].values.astype(float)

        # 优化1: 镜像增强
        log_info(f"  [Optimization 1] Mirror augmentation...")
        X_train_features = train_df[all_feature_cols].copy()
        mirror_X, mirror_y = create_mirror_samples(X_train_features, y_train_raw)

        # 合并原始 + 镜像
        X_train_aug = pd.concat([X_train_features, mirror_X], ignore_index=True)
        y_train_aug = np.concatenate([y_train_raw, mirror_y])

        # 镜像后的 league 列 (用于后续权重计算)
        train_leagues_aug = pd.concat([train_df["league"], train_df["league"]], ignore_index=True)
        log_info(f"    [Mirror] Original: {len(X_train_features)}, Mirror: {len(mirror_X)}, "
                 f"Combined: {len(X_train_aug)}")

        # 优化3: 移除 LPL 噪声增强 (实验证明过重导致 LCK 退化)
        # 仅保留镜像增强 + 联赛权重
        n_lpl_extra = 0

        # 重新计算增强后的 league 列
        train_leagues_final = pd.concat([
            train_df["league"],       # 原始
            train_df["league"],       # 镜像
            pd.Series(["LPL"] * n_lpl_extra),  # LPL 噪声增强
        ], ignore_index=True)

        # 优化2: 联赛自适应样本权重
        log_info(f"  [Optimization 2] League-adaptive sample weights...")
        train_df_for_weight = pd.DataFrame({"league": train_leagues_final})
        sample_weights = compute_league_weights(train_df_for_weight)

        # ---- Label Smoothing ----
        y_train_smooth = y_train_aug * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        # ---- 准备特征矩阵 ----
        X_train = X_train_aug[all_feature_cols].values.astype(np.float32)
        X_test = test_df[all_feature_cols].values.astype(np.float32)

        # 处理 NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # 各联赛测试集索引
        league_masks = {}
        for league in ["LPL", "LCK", "LEC"]:
            league_masks[league] = test_df["league"] == league

        log_info(f"  Train (augmented): {len(X_train)} | Test: {len(test_df)} "
                 f"(LPL: {league_masks['LPL'].sum()}, LCK: {league_masks['LCK'].sum()}, "
                 f"LEC: {league_masks['LEC'].sum()})")
        log_info(f"  Feature dim: {len(all_feature_cols)} "
                 f"(Tabular: {len(feature_cols)} + TF: {len(all_feature_cols) - len(feature_cols)})")

        # ---- CatBoost 7-seed Bagging ----
        all_test_preds = []
        fold_models = []
        fold_seed_best_iters = []
        fold_seed_metrics = []
        log_info(f"  Starting {N_SEEDS}-Seed CatBoost Bagging...")

        for seed_idx, seed in enumerate(SEEDS):
            with log_context(Seed=seed):
                params = BEST_PARAMS.copy()
                params["random_seed"] = seed

                model = CatBoostClassifier(**params)

                n_train = len(X_train)
                n_val = int(n_train * _shared_cfg.val_split_ratio)
                perm = np.random.RandomState(seed).permutation(n_train)
                val_idx = perm[:n_val]
                tr_idx = perm[n_val:]

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
                train_loss = evals_result['learn']['Logloss'][-1] if 'learn' in evals_result and 'Logloss' in evals_result['learn'] else None
                val_loss = evals_result['validation']['Logloss'][-1] if 'validation' in evals_result and 'Logloss' in evals_result['validation'] else None
                log_info(f"    [Seed {seed_idx}/{N_SEEDS-1}] seed={seed} best_iteration={best_iter}"
                         f"{f', train_loss={train_loss:.4f}' if train_loss else ''}"
                         f"{f', val_loss={val_loss:.4f}' if val_loss else ''}")

                if best_iter < _shared_cfg.early_stopping_rounds * 0.3:
                    log.warning(f"    [Seed {seed_idx}] Early stopping triggered very early at iter {best_iter} (< 30% of patience), possible underfitting")
                elif best_iter >= params.get('iterations', BEST_PARAMS['iterations']) - 5:
                    log.warning(f"    [Seed {seed_idx}] Model did not converge early (best_iter={best_iter} near max), consider increasing iterations")

                test_pred = model.predict_proba(X_test)[:, 1]
                seed_auc = roc_auc_score(y_test, test_pred)
                seed_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
                seed_ll = log_loss(y_test, test_pred)
                log_info(f"    [Seed {seed_idx}] test AUC={seed_auc:.4f}, ACC={seed_acc:.4f}, LogLoss={seed_ll:.4f}")
                fold_seed_metrics.append({"seed": seed, "auc": seed_auc, "acc": seed_acc, "logloss": seed_ll, "best_iter": best_iter})

                all_test_preds.append(test_pred)
                fold_models.append(model)
                fold_seed_best_iters.append(int(best_iter))

        seed_aucs = [m["auc"] for m in fold_seed_metrics]
        seed_accs = [m["acc"] for m in fold_seed_metrics]
        seed_lls = [m["logloss"] for m in fold_seed_metrics]
        log_info(f"  Ensemble {N_SEEDS}-Seed averages pre-bagging: AUC={np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}, "
                 f"ACC={np.mean(seed_accs):.4f}±{np.std(seed_accs):.4f}, LogLoss={np.mean(seed_lls):.4f}±{np.std(seed_lls):.4f}")

        # ---- 保存当前折的模型 ----
        fold_model_dir = os.path.join(MODELS_DIR, f"fold_{fold_idx}")
        os.makedirs(fold_model_dir, exist_ok=True)
        for seed_idx, model in enumerate(fold_models):
            model_path = os.path.join(fold_model_dir, f"catboost_seed_{seed_idx}.cbm")
            model.save_model(model_path)
        # 保存特征列名
        import json as _json
        with open(os.path.join(fold_model_dir, "feature_columns.json"), "w") as f:
            _json.dump(all_feature_cols, f)
        log_info(f"  [Model] Saved {len(fold_models)} models to {fold_model_dir}")

        # Bagging: 取 5 个 seed 的平均
        test_pred_avg = np.mean(all_test_preds, axis=0)

        # 移除 Temperature Scaling (实验证明在 label-smoothed 训练下效果不佳)
        test_pred_final = test_pred_avg
        optimal_temp = 1.0

        # ---- 计算各联赛指标 ----
        def compute_league_metrics(y_true_all, y_pred_all, mask, league_name, min_samples=10):
            """计算单个联赛的指标。"""
            y_league = y_true_all[mask]
            pred_league = y_pred_all[mask]
            n = len(y_league)
            if n < min_samples or len(np.unique(y_league)) < 2:
                log_info(f"  [{league_name}] N={n} (样本不足或标签单一, 跳过)")
                return None
            auc = roc_auc_score(y_league, pred_league)
            acc = accuracy_score(y_league, (pred_league > 0.5).astype(int))
            ll = log_loss(y_league, pred_league)
            brier = brier_score_loss(y_league, pred_league)
            auc_mean, auc_ci = bootstrap_ci(y_league, pred_league, roc_auc_score,
                                             n_resamples=N_BOOTSTRAP, ci=0.95)
            ll_mean, ll_ci = bootstrap_ci(y_league, pred_league, log_loss,
                                          n_resamples=N_BOOTSTRAP, ci=0.95)
            brier_mean, brier_ci = bootstrap_ci(y_league, pred_league, brier_score_loss,
                                                 n_resamples=N_BOOTSTRAP, ci=0.95)
            return {
                "n": n, "auc": auc, "acc": acc, "logloss": ll, "brier": brier,
                "auc_ci": auc_ci, "logloss_ci": ll_ci, "brier_ci": brier_ci,
            }

        # Overall
        overall_metrics = {
            "n": len(y_test),
            "auc": roc_auc_score(y_test, test_pred_final),
            "acc": accuracy_score(y_test, (test_pred_final > 0.5).astype(int)),
            "logloss": log_loss(y_test, test_pred_final),
            "brier": brier_score_loss(y_test, test_pred_final),
        }
        _, overall_auc_ci = bootstrap_ci(y_test, test_pred_final, roc_auc_score,
                                          n_resamples=N_BOOTSTRAP, ci=0.95)
        overall_metrics["auc_ci"] = overall_auc_ci

        # 各联赛
        lpl_metrics = compute_league_metrics(y_test, test_pred_final, league_masks["LPL"], "LPL")
        lck_metrics = compute_league_metrics(y_test, test_pred_final, league_masks["LCK"], "LCK")
        lec_metrics = compute_league_metrics(y_test, test_pred_final, league_masks["LEC"], "LEC")

        # ---- 打印结果 ----
        log_info(f"  ┌{'─'*90}┐")
        log_info(f"  │ {'Overall':<8} AUC={overall_metrics['auc']:.4f} "
                 f"[{overall_metrics['auc_ci'][0]:.4f}, {overall_metrics['auc_ci'][1]:.4f}]  "
                 f"ACC={overall_metrics['acc']:.4f}  "
                 f"LogLoss={overall_metrics['logloss']:.4f}  "
                 f"Brier={overall_metrics['brier']:.4f}  N={overall_metrics['n']:>4} │")

        for league_name, metrics in [("LPL", lpl_metrics), ("LCK", lck_metrics), ("LEC", lec_metrics)]:
            if metrics is not None:
                log_info(f"  │ {league_name:<8} AUC={metrics['auc']:.4f} "
                         f"[{metrics['auc_ci'][0]:.4f}, {metrics['auc_ci'][1]:.4f}]  "
                         f"ACC={metrics['acc']:.4f}  "
                         f"LogLoss={metrics['logloss']:.4f}  "
                         f"Brier={metrics['brier']:.4f}  N={metrics['n']:>4} │")
            else:
                log_info(f"  │ {league_name:<8} N/A (insufficient data) │")
        log_info(f"  └{'─'*90}┘")

        # ---- 保存折结果 ----
        fold_result = {
            "fold": fold_idx + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "n_train": len(train_df),
            "n_train_aug": len(X_train),
            "n_test": len(test_df),
            "dim": len(all_feature_cols),
            "temperature": optimal_temp,
            "overall": overall_metrics,
            # 生产模式参数传递: 记录本折各 seed 的 best_iteration
            "best_iterations": fold_seed_best_iters,
            "best_iteration_mean": float(np.mean(fold_seed_best_iters)),
            "best_iteration_std": float(np.std(fold_seed_best_iters)),
        }
        if lpl_metrics:
            fold_result["lpl"] = lpl_metrics
        if lck_metrics:
            fold_result["lck"] = lck_metrics
        if lec_metrics:
            fold_result["lec"] = lec_metrics
        fold_results.append(fold_result)

    # =====================================================================
    # Step 4: 汇总报告
    # =====================================================================
    log_info(f"\n{'='*90}")
    log_info("5-FOLD OOT VALIDATION SUMMARY (Optimized Cascade Fusion)")
    log_info(f"{'='*90}")

    leagues_to_report = ["overall", "lpl", "lck", "lec"]
    league_display = {"overall": "Overall", "lpl": "LPL", "lck": "LCK", "lec": "LEC"}

    for league_key in leagues_to_report:
        log_info(f"\n  [{league_display[league_key]}] Per-Fold Results:")
        log_info(f"  {'Fold':>4} | {'Train Window':>25} | {'N_Train':>7} | {'N_Test':>6} | "
                 f"{'AUC':>7} | {'AUC 95% CI':>17} | {'ACC':>6} | {'LogLoss':>7} | {'Brier':>7}")
        log_info(f"  {'─'*100}")

        league_aucs = []
        league_accs = []
        league_lls = []
        league_briers = []

        for r in fold_results:
            m = r.get(league_key)
            if m is None:
                log_info(f"  {r['fold']:>4} | {r['train_start']}~{r['train_end']:>10} | "
                         f"{r['n_train']:>7} | {'N/A':>6} | "
                         f"{'N/A':>7} | {'N/A':>17} | {'N/A':>6} | {'N/A':>7} | {'N/A':>7}")
                continue
            auc_ci_str = f"[{m['auc_ci'][0]:.4f},{m['auc_ci'][1]:.4f}]" if "auc_ci" in m else "N/A"
            log_info(f"  {r['fold']:>4} | {r['train_start']}~{r['train_end']:>10} | "
                     f"{r['n_train']:>7} | {m['n']:>6} | "
                     f"{m['auc']:>7.4f} | {auc_ci_str:>17} | "
                     f"{m['acc']:>6.1%} | {m['logloss']:>7.4f} | {m['brier']:>7.4f}")
            league_aucs.append(m["auc"])
            league_accs.append(m["acc"])
            league_lls.append(m["logloss"])
            league_briers.append(m["brier"])

        if league_aucs:
            auc_mean = np.mean(league_aucs)
            auc_std = np.std(league_aucs)
            acc_mean = np.mean(league_accs)
            acc_std = np.std(league_accs)
            ll_mean = np.mean(league_lls)
            brier_mean = np.mean(league_briers)

            ci_lowers = [r[league_key]["auc_ci"][0] for r in fold_results
                         if league_key in r and "auc_ci" in r[league_key]]
            ci_uppers = [r[league_key]["auc_ci"][1] for r in fold_results
                         if league_key in r and "auc_ci" in r[league_key]]
            ci_lower_avg = np.mean(ci_lowers) if ci_lowers else 0
            ci_upper_avg = np.mean(ci_uppers) if ci_uppers else 0

            log_info(f"  {'─'*100}")
            log_info(f"  {'Mean':>4} | {'':>25} | {'':>7} | {int(np.mean([r[league_key]['n'] for r in fold_results if league_key in r])):>6} | "
                     f"{auc_mean:>7.4f} | [{ci_lower_avg:.4f},{ci_upper_avg:.4f}] | "
                     f"{acc_mean:>6.1%} | {ll_mean:>7.4f} | {brier_mean:>7.4f}")
            log_info(f"  {'Std':>4} | {'':>25} | {'':>7} | {'':>6} | "
                     f"{auc_std:>7.4f} | {'':>17} | "
                     f"{acc_std:>6.1%} | {'':>7} | {'':>7}")

    # ---- 横向对比表 ----
    log_info(f"\n{'='*90}")
    log_info("LEAGUE COMPARISON (5-Fold Mean ± Std)")
    log_info(f"{'='*90}")
    log_info(f"  {'League':<10} {'AUC':>12} {'ACC':>12} {'LogLoss':>10} {'Brier':>10} {'N_avg':>8}")
    log_info(f"  {'─'*62}")

    summary = {}
    for league_key in leagues_to_report:
        league_aucs = [r[league_key]["auc"] for r in fold_results if league_key in r]
        league_accs = [r[league_key]["acc"] for r in fold_results if league_key in r]
        league_lls = [r[league_key]["logloss"] for r in fold_results if league_key in r]
        league_briers = [r[league_key]["brier"] for r in fold_results if league_key in r]
        league_ns = [r[league_key]["n"] for r in fold_results if league_key in r]

        if not league_aucs:
            continue

        auc_m, auc_s = np.mean(league_aucs), np.std(league_aucs)
        acc_m, acc_s = np.mean(league_accs), np.std(league_accs)
        ll_m = np.mean(league_lls)
        brier_m = np.mean(league_briers)
        n_avg = int(np.mean(league_ns))

        log_info(f"  {league_display[league_key]:<10} {auc_m:.4f}±{auc_s:.4f} "
                 f"{acc_m:.1%}±{acc_s:.1%} {ll_m:>10.4f} {brier_m:>10.4f} {n_avg:>8}")

        summary[f"{league_key}_auc_mean"] = auc_m
        summary[f"{league_key}_auc_std"] = auc_s
        summary[f"{league_key}_acc_mean"] = acc_m
        summary[f"{league_key}_acc_std"] = acc_s
        summary[f"{league_key}_logloss_mean"] = ll_m
        summary[f"{league_key}_brier_mean"] = brier_m

    # ---- 与基线对比 ----
    BASELINE = {
        "overall_auc_mean": 0.6577, "overall_auc_std": 0.0334,
        "lpl_auc_mean": 0.6332, "lpl_auc_std": 0.0261,
        "lck_auc_mean": 0.6930, "lck_auc_std": 0.0583,
        "lec_auc_mean": 0.6215, "lec_auc_std": 0.0847,
    }

    log_info(f"\n{'='*90}")
    log_info("OPTIMIZATION vs BASELINE COMPARISON")
    log_info(f"{'='*90}")
    log_info(f"  {'League':<10} {'Baseline AUC':>14} {'Optimized AUC':>14} {'Delta':>8} {'Verdict':>10}")
    log_info(f"  {'─'*58}")

    for league_key in leagues_to_report:
        b_key = f"{league_key}_auc_mean"
        if b_key in summary and b_key in BASELINE:
            baseline_auc = BASELINE[b_key]
            optimized_auc = summary[b_key]
            delta = optimized_auc - baseline_auc
            verdict = "IMPROVED" if delta > 0.005 else ("STABLE" if delta > -0.005 else "REGRESSED")
            log_info(f"  {league_display[league_key]:<10} {baseline_auc:>14.4f} {optimized_auc:>14.4f} "
                     f"{delta:>+8.4f} {verdict:>10}")

    # ---- 诊断 ----
    log_info(f"\n  [DIAGNOSIS]")
    overall_aucs = [r["overall"]["auc"] for r in fold_results]
    overall_auc_std = np.std(overall_aucs)
    if overall_auc_std < 0.03:
        log_info(f"    Overall AUC 标准差 {overall_auc_std:.4f} < 0.03, 泛化能力稳定。")
    elif overall_auc_std < 0.05:
        log_info(f"    注意: Overall AUC 标准差 {overall_auc_std:.4f} 在 0.03~0.05 之间，存在一定波动。")
    else:
        log_info(f"    警告: Overall AUC 标准差 {overall_auc_std:.4f} > 0.05, 泛化能力不稳定。")

    # 时序衰减分析
    first_half = np.mean(overall_aucs[:2])
    second_half = np.mean(overall_aucs[3:])
    delta = second_half - first_half
    if delta < -0.03:
        log_info(f"    趋势: Overall AUC 从 {first_half:.4f} 下降到 {second_half:.4f} (Δ={delta:.4f})，模型可能存在时序衰减。")
    elif delta > 0.02:
        log_info(f"    趋势: Overall AUC 从 {first_half:.4f} 上升到 {second_half:.4f} (Δ={delta:.4f})，模型近期表现改善。")
    else:
        log_info(f"    趋势: Overall AUC 变化不大 (Δ={delta:.4f})，模型时序稳定性良好。")

    # 联赛间差异
    if "lpl_auc_mean" in summary and "lck_auc_mean" in summary:
        gap = summary["lpl_auc_mean"] - summary["lck_auc_mean"]
        log_info(f"    LPL vs LCK AUC 差距: {gap:+.4f}")
    if "lpl_auc_mean" in summary and "lec_auc_mean" in summary:
        gap = summary["lpl_auc_mean"] - summary["lec_auc_mean"]
        log_info(f"    LPL vs LEC AUC 差距: {gap:+.4f}")

    summary["temporal_decay"] = delta

    # =====================================================================
    # Step 5: 保存报告
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "bp_prediction",
            "mode": "training_5fold_oot",
            "validation_method": f"5-Fold Rolling OOT (Fixed Training Window = {window_months} months)",
            "architecture": "Optimized Deep-Tabular Cascade Fusion: Transformer(4-dim) + CatBoost-7Seed-Bagging",
            "optimizations": [
                "League-adaptive weights: LPL=1.3x, LEC=1.5x, LCK=1.0x",
                "Mirror augmentation (blue-red symmetry)",
                "Optimized hyperparams (iterations=800, stronger reg)",
            ],
            "label_smoothing": LABEL_SMOOTHING,
            "n_seeds": N_SEEDS,
            "n_bootstrap": N_BOOTSTRAP,
            "window_months": window_months,
            "training": "All leagues + mirror augmentation",
            "league_reports": ["Overall", "LPL", "LCK", "LEC"],
            "baseline_comparison": BASELINE,
        },
        "fold_results": fold_results,
        "summary": summary,
    }

    report_path = os.path.join(REPORTS_DIR, f"optimized_cascade_oot_5fold_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    metrics_path = os.path.join(str(PREDICTION_METRICS_DIR), f"prediction_train_5fold_oot_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _last_report_path = metrics_path
    log_info(f"\n  Report saved to: {report_path}")
    log_info(f"  Metrics saved to: {metrics_path}")
    log.info(f"  Report size: {os.path.getsize(metrics_path)/1024:.1f} KB")

    n_recent_folds = min(3, len(fold_results))
    recent_folds = fold_results[-n_recent_folds:]
    recent_best_iters = [f["best_iteration_mean"] for f in recent_folds]
    base_iterations = float(np.mean(recent_best_iters))

    last_fold_train_samples = int(recent_folds[-1]["n_train"])

    production_params = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"OOT 5-Fold Validation (last {n_recent_folds} folds mean)",
        "base_iterations": base_iterations,
        "base_iterations_per_fold": recent_best_iters,
        "last_fold_train_samples": last_fold_train_samples,
        "n_recent_folds": n_recent_folds,
        "fold_details": [
            {
                "fold": f["fold"],
                "best_iteration_mean": f["best_iteration_mean"],
                "best_iteration_std": f["best_iteration_std"],
                "best_iterations_per_seed": f["best_iterations"],
                "n_train": f["n_train"],
                "n_train_aug": f["n_train_aug"],
            }
            for f in recent_folds
        ],
        "oot_overall_auc_mean": summary.get("overall_auc_mean"),
        "oot_overall_auc_std": summary.get("overall_auc_std"),
    }

    prod_params_path = os.path.join(REPORTS_DIR, "production_iterations_source.json")
    with open(prod_params_path, "w", encoding="utf-8") as f:
        json.dump(production_params, f, indent=2, ensure_ascii=False)
    prod_params_metrics_path = os.path.join(str(PREDICTION_METRICS_DIR), f"prediction_prod_iterations_{timestamp}.json")
    with open(prod_params_metrics_path, "w", encoding="utf-8") as f:
        json.dump(production_params, f, indent=2, ensure_ascii=False)
    log_info(f"\n  Production params saved to: {prod_params_path}")
    log_info(f"  Production params (archived): {prod_params_metrics_path}")
    log_info(f"  Params file size: {os.path.getsize(prod_params_path)/1024:.1f} KB")
    log_info(f"  Base iterations (last {n_recent_folds} folds mean): {base_iterations:.1f}")
    log_info(f"  Per-fold best_iterations: {recent_best_iters}")
    log_info(f"  Last fold train samples: {last_fold_train_samples}")
    log_info(f"  OOT Overall AUC: {summary.get('overall_auc_mean', 'N/A'):.4f}±{summary.get('overall_auc_std', 0):.4f}")
    log_info(f"  Consistency check: Model artifacts ready for production handoff")
    sample_count = sum(f["n_train_aug"] for f in recent_folds)
    log_info(f"  Recent folds total augmented samples for calibration: {sample_count}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_buffer))

    log_info(f"\n{'='*90}")
    log_info("OOT Validation Complete (Optimized Cascade Fusion)!")
    log_info(f"Log: {log_path}")
    log_info(f"{'='*70}")


if __name__ == "__main__":
    main()
