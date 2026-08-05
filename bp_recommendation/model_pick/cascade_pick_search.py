#!/usr/bin/env python3
"""
Cascade Pick 超参数搜索 (Optuna + TPE) — 统一 Phase-Aware Routing

对齐 cascade_pick.py 的单模型架构:
  - 统一 LGB 模型，通过 Phase-Aware 特征（p1_*/p2_*）实现阶段路由
  - 搜索 LGB 超参 + blend_alpha
  - 使用 GroupKFold 5-Fold CV

用法:
    cd <project_root>
    python -m bp_recommendation.model_pick.cascade_pick_search --n_trials 60
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bp_recommendation.model_pick.cascade_pick import (
    extract_pick_ranking_data,
    _evaluate_pick_at_k,
    _rank_normalize,
    FEATURE_COLS,
)
from logger_config import get_logger

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SEARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_results")
DB_DIR = os.path.join(SEARCH_DIR, "optuna_db")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

log = get_logger(__name__)

N_FOLDS = 5


def _fast_oof_cv(X_train, y_train, group_train, w_train, base_train, match_ids, params, num_round, early_stop):
    """5-Fold CV，接收 w_train 作为 LGBM 的样本权重，base_train 作为 init_score 残差训练"""
    # 【修复 1】：使用 match_id 作为 GroupKFold 的 groups，确保同一场比赛的所有 step 在同一 fold
    row_match_ids = np.repeat(match_ids, group_train)
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_pred = np.zeros(len(y_train), dtype=np.float64)
    fold_best_iters = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, row_match_ids)):
        fold_y_train = y_train[train_idx]
        fold_y_val = y_train[val_idx]
        fold_w_train = w_train[train_idx]
        fold_w_val = w_train[val_idx]

        train_group_ids = np.unique(row_match_ids[train_idx])
        val_group_ids = np.unique(row_match_ids[val_idx])

        # 【修复 3】：传入 init_score=base_train 做残差训练
        fold_train_ds = lgb.Dataset(
            X_train[train_idx], fold_y_train,
            group=group_train[train_group_ids],
            weight=fold_w_train,
            init_score=base_train[train_idx]
        )
        fold_val_ds = lgb.Dataset(
            X_train[val_idx], fold_y_val,
            group=group_train[val_group_ids],
            weight=fold_w_val,
            init_score=base_train[val_idx],
            reference=fold_train_ds,
        )

        model = lgb.train(
            params, fold_train_ds, num_boost_round=num_round,
            valid_sets=[fold_val_ds],
            callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)],
        )

        # 【修复 3】：预测 = 树残差 + TF base logits
        oof_pred[val_idx] = model.predict(X_train[val_idx]) + base_train[val_idx]
        fold_best_iters.append(model.best_iteration)

    avg_iter = int(np.mean(fold_best_iters))
    oof_metrics = _evaluate_pick_at_k(oof_pred, y_train, group_train)
    return oof_metrics, avg_iter, oof_pred


def create_objective(X_train, y_train, group_train, w_train, base_train, match_ids):
    """创建 Optuna objective 函数，match_ids 用于 GroupKFold 按比赛分组"""

    def objective(trial):
        num_leaves = trial.suggest_int("num_leaves", 31, 127)
        max_depth = trial.suggest_int("max_depth", 6, 12)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 50)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
        feature_fraction = trial.suggest_float("feature_fraction", 0.3, 0.8)
        # 【修复 2】：添加 feature_fraction_bynode 强迫节点级特征采样
        feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.4, 0.7)
        # 【修复 2】：极随机树模式，增加特征多样性
        extra_trees = trial.suggest_categorical("extra_trees", [True, False])
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 0.95)
        bagging_freq = trial.suggest_int("bagging_freq", 1, 7)
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True)
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-2, 20.0, log=True)
        num_round = trial.suggest_int("num_round", 3000, 6000, step=500)
        early_stop = trial.suggest_int("early_stop", 100, 250, step=50)

        params = {
            "objective": "rank_xendcg",
            "metric": "ndcg",
            "ndcg_at": [10],
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_data_in_leaf": min_data_in_leaf,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "feature_fraction_bynode": feature_fraction_bynode,
            "extra_trees": extra_trees,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "verbose": -1,
            "seed": 42,
        }

        try:
            # 【修复 1+3】：传入 match_ids 和 base_train 做按比赛分组 + 残差训练
            oof_metrics, avg_iter, oof_pred = _fast_oof_cv(
                X_train, y_train, group_train, w_train, base_train, match_ids,
                params, num_round, early_stop
            )
        except Exception as e:
            log.warning(f"  Trial {trial.number} failed: {e}")
            return 0.0

        # 【修复 3】：残差训练模式下，oof_pred 已包含 TF base + LGBM 残差，不再做 blend
        best_alpha = None
        best_blend_p10 = oof_metrics[10]

        log.info(f"  Trial {trial.number:03d}: "
                 f"OOF P@10={oof_metrics[10]:.2f}% (residual mode) "
                 f"avg_iter={avg_iter} | "
                 f"leaves={num_leaves} depth={max_depth} lr={learning_rate:.4f} "
                 f"ff={feature_fraction:.2f} ff_bynode={feature_fraction_bynode:.2f} "
                 f"extra_trees={extra_trees} l1={lambda_l1:.2f} l2={lambda_l2:.2f}")

        # 记录最佳 blend_alpha 和 avg_iter 到 trial，供 run_search 末尾写入配置文件
        trial.set_user_attr("best_alpha", best_alpha)
        trial.set_user_attr("avg_iter", int(avg_iter))
        trial.set_user_attr("best_metric", best_blend_p10)
        # 记录 fold 级别 best_iteration，便于生产模式精确复现
        trial.set_user_attr("fold_best_iterations", [int(i) for i in fold_best_iters])

        return best_blend_p10

    return objective


def run_search():
    parser = argparse.ArgumentParser(description="Cascade Pick Unified Hyperparameter Search")
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    total_t0 = time.time()
    log.info("=" * 70)
    log.info("  Cascade Pick Hyperparameter Search (Unified Phase-Aware)")
    log.info("=" * 70)

    log.info("\n  Loading data...")
    # 【修复 1】：extract_pick_ranking_data 现在返回 match_ids
    X_train_df, y_train, group_train, w_train, base_train, match_ids = extract_pick_ranking_data("train")

    X_train = X_train_df.values.astype(np.float32)
    from sklearn.preprocessing import StandardScaler
    X_train = StandardScaler().fit_transform(X_train)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="cascade_pick_unified",
        storage=f"sqlite:///{os.path.join(DB_DIR, 'cascade_pick_unified.db')}",
        load_if_exists=True,
    )

    # 【修复 1+3】：传入 match_ids 用于 GroupKFold，base_train 用于 init_score 残差训练
    obj = create_objective(X_train, y_train, group_train, w_train, base_train, match_ids)
    study.optimize(obj, n_trials=args.n_trials, timeout=args.timeout)

    log.info(f"\n  Unified Search Complete:")
    log.info(f"  Best Trial: {study.best_trial.number}")
    log.info(f"  Best Blend Pick@10: {study.best_value:.2f}%")
    log.info(f"  Best Params:")
    for k, v in study.best_params.items():
        log.info(f"    {k}: {v}")

    df = study.trials_dataframe()
    df.to_csv(os.path.join(SEARCH_DIR, f"pick_cascade_search_{_run_ts}.csv"), index=False)

    with open(os.path.join(SEARCH_DIR, f"pick_cascade_best_configs_{_run_ts}.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    # 将最佳参数写入正式配置文件，供生产模式加载
    best_trial = study.best_trial
    best_params = best_trial.params
    best_alpha = best_trial.user_attrs.get("best_alpha")
    avg_iter = best_trial.user_attrs.get("avg_iter")
    best_metric = best_trial.user_attrs.get("best_metric", study.best_value)
    fold_best_iters = best_trial.user_attrs.get("fold_best_iterations", [])
    try:
        from bp_recommendation.config import save_best_params as _save_best_params
        _save_best_params(
            model_type="pick",
            model_subtype="cascade",
            best_iteration=avg_iter,
            # 【修复 3】：残差模式下 best_alpha=None，save_best_params 会跳过写入
            blend_alpha=best_alpha,
            best_metric=best_metric / 100.0 if best_metric > 1.0 else best_metric,
            best_metric_name="P@10",
            architecture={
                "num_leaves": best_params.get("num_leaves"),
                "max_depth": best_params.get("max_depth"),
                "min_data_in_leaf": best_params.get("min_data_in_leaf"),
                "feature_fraction_bynode": best_params.get("feature_fraction_bynode"),
                "extra_trees": best_params.get("extra_trees"),
            },
            optimizer={
                "learning_rate": best_params.get("learning_rate"),
            },
            loss={
                "objective": "rank_xendcg",
                "metric": "ndcg",
            },
            training={
                "n_folds": 5,
                "num_boost_round": best_params.get("num_round"),
                "early_stop": best_params.get("early_stop"),
                "fold_best_iterations": fold_best_iters,
                "fusion_mode": "residual_init_score",
            },
        )
        log.info(f"  Best params for [pick/cascade] saved to training config")
    except Exception as e:
        log.warning(f"  Failed to save best params to config: {e}")

    log.info(f"\n  Total time: {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    from pathlib import Path
    from logger_config import setup_logging
    setup_logging(log_dir=Path(LOG_DIR))
    
    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _run_fh = logging.FileHandler(os.path.join(LOG_DIR, f"cascade_pick_search_{_run_ts}.log"), encoding="utf-8")
    _run_fh.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _run_fh.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_run_fh)
    
    run_search()