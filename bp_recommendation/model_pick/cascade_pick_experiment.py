#!/usr/bin/env python3
"""
Cascade Pick 超参数严谨对比实验
================================

实验目标:
    对比不同正则化强度的超参数配置，找出 cascade 模型能真正提升 transformer_base 的最优配置。

实验设计:
    1. 6 组超参配置 (从强正则到弱正则，梯度对比)
    2. 评估方式:
       - 主要指标: val 集上 cascade_final 相对 transformer_base 的提升 (Δ P@K)
       - 辅助指标: OOF 指标 (虽被 TF In-Sample 污染，但可观察过拟合趋势)
       - 稳定性: 5 个随机种子重复实验，取均值±标准差
    3. 公平对比: 所有配置使用相同的 TF logits、相同的特征矩阵、相同的 fold 划分

泄漏说明:
    - TF train logits 是 In-Sample 预测 (TF 模型见过训练集标签)
    - TF val logits 有模型选择偏差 (TF 通过 val early stopping 选模型)
    - 因此 val 指标有轻微乐观偏置，但 cascade_final vs transformer_base 的相对提升仍可靠
    - OOF 指标被严重高估 (In-Sample TF logits)，仅用于观察过拟合趋势

用法:
    cd /Users/siwentu/Desktop/LOL analysis
    conda run -n LOL python -m bp_recommendation.model_pick.cascade_pick_experiment
"""

import os
import sys
import time
import json
import copy
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from bp_recommendation.model_pick.cascade_pick import (
    extract_pick_ranking_data,
    _evaluate_pick_at_k,
    FEATURE_COLS,
    CS_TOP_K,
)
from logger_config import get_logger, setup_logging
from common.paths import RECOMMENDATION_METRICS_DIR, ensure_dirs as _ensure_common_dirs

_ensure_common_dirs()

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

log = get_logger(__name__)

N_FOLDS = 5
NUM_ROUND = 4500
EARLY_STOP = 250
N_SEEDS = 5  # 每组配置用 5 个随机种子重复实验
SEEDS = [42, 123, 456, 789, 2024]


# ============================================================
# 实验配置: 6 组超参，从强正则到弱正则
# ============================================================
# 所有配置共享的基础参数
BASE_CONFIG = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "ndcg_at": [10],
    "num_leaves": 48,
    "max_depth": 8,
    "min_data_in_leaf": 17,
    "bagging_fraction": 0.8859303492103779,
    "bagging_freq": 5,
    "lambda_l1": 0.0020744246659736253,
    "lambda_l2": 0.04360744274903356,
    "max_bin": 255,
    "verbose": -1,
}

# 实验配置组: key -> (description, overrides)
EXPERIMENT_CONFIGS = {
    "A_strong_reg": {
        "description": "强正则化 (当前生产配置): extra_trees + ff_bynode=0.5 + lr=0.008",
        "overrides": {
            "learning_rate": 0.007980833558993788,
            "feature_fraction": 0.32917339062527257,
            "feature_fraction_bynode": 0.5,
            "extra_trees": True,
        },
    },
    "B_medium_reg": {
        "description": "中等正则化: 移除 extra_trees, ff_bynode=0.7, lr=0.01",
        "overrides": {
            "learning_rate": 0.01,
            "feature_fraction": 0.5,
            "feature_fraction_bynode": 0.7,
            "extra_trees": False,
        },
    },
    "C_mild_reg": {
        "description": "轻度正则化: ff_bynode=0.85, lr=0.015, ff=0.7",
        "overrides": {
            "learning_rate": 0.015,
            "feature_fraction": 0.7,
            "feature_fraction_bynode": 0.85,
            "extra_trees": False,
        },
    },
    "D_weak_reg": {
        "description": "弱正则化: ff_bynode=1.0, lr=0.02, ff=0.8",
        "overrides": {
            "learning_rate": 0.02,
            "feature_fraction": 0.8,
            "feature_fraction_bynode": 1.0,
            "extra_trees": False,
        },
    },
    "E_minimal_reg": {
        "description": "极弱正则化: ff_bynode=1.0, lr=0.03, ff=1.0, 无 bagging",
        "overrides": {
            "learning_rate": 0.03,
            "feature_fraction": 1.0,
            "feature_fraction_bynode": 1.0,
            "extra_trees": False,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
        },
    },
    "F_no_cascade": {
        "description": "基线: 仅 transformer_base, 无 cascade (用于对比)",
        "overrides": None,  # 特殊处理，不训练 LGBM
    },
}


def _run_single_config(X_train, y_train, group_train, w_train, base_cs_train,
                       match_ids_train, X_val, y_val, group_val, w_val, base_cs_val,
                       match_ids_val, config, seed):
    """对单组配置 + 单种子运行完整训练和评估"""
    if config["overrides"] is None:
        # 基线: 仅 transformer_base
        val_metrics = _evaluate_pick_at_k(base_cs_val, y_val, group_val)
        oof_metrics = _evaluate_pick_at_k(base_cs_train, y_train, group_train)
        return {
            "val_metrics": val_metrics,
            "oof_metrics": oof_metrics,
            "best_iterations": [],
        }

    lgb_config = {**BASE_CONFIG, **config["overrides"], "seed": seed}

    # === GroupKFold CV (与生产代码一致) ===
    row_match_ids = np.repeat(match_ids_train, group_train)
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_pred = np.zeros(len(y_train), dtype=np.float64)
    fold_models = []
    best_iterations = []

    for fold_i, (t_idx, v_idx) in enumerate(gkf.split(X_train, y_train, row_match_ids)):
        t_match_ids = np.unique(row_match_ids[t_idx])
        t_query_mask = np.isin(match_ids_train, t_match_ids)
        v_match_ids = np.unique(row_match_ids[v_idx])
        v_query_mask = np.isin(match_ids_train, v_match_ids)

        train_ds = lgb.Dataset(
            X_train[t_idx], y_train[t_idx],
            group=group_train[t_query_mask],
            weight=w_train[t_idx],
            init_score=base_cs_train[t_idx],
        )
        val_ds = lgb.Dataset(
            X_train[v_idx], y_train[v_idx],
            group=group_train[v_query_mask],
            weight=w_train[v_idx],
            init_score=base_cs_train[v_idx],
            reference=train_ds,
        )

        callbacks = [lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            lgb_config, train_ds, num_boost_round=NUM_ROUND,
            valid_sets=[val_ds], callbacks=callbacks,
        )
        oof_pred[v_idx] = model.predict(X_train[v_idx]) + base_cs_train[v_idx]
        fold_models.append(model)
        best_iterations.append(model.best_iteration)

    # === OOF 指标 (被 TF In-Sample 污染，仅观察过拟合) ===
    oof_metrics = _evaluate_pick_at_k(oof_pred, y_train, group_train)

    # === Val 指标 (主要评估依据) ===
    val_preds = np.array([m.predict(X_val) + base_cs_val for m in fold_models])
    val_final_pred = val_preds.mean(axis=0)
    val_metrics = _evaluate_pick_at_k(val_final_pred, y_val, group_val)

    # === TF Base 在 val 上的指标 (对比基准) ===
    base_val_metrics = _evaluate_pick_at_k(base_cs_val, y_val, group_val)

    return {
        "val_metrics": val_metrics,
        "oof_metrics": oof_metrics,
        "base_val_metrics": base_val_metrics,
        "best_iterations": best_iterations,
    }


def run_experiment():
    """运行完整对比实验"""
    total_t0 = time.time()
    log.info("=" * 80)
    log.info("  Cascade Pick 超参数严谨对比实验")
    log.info(f"  配置数: {len(EXPERIMENT_CONFIGS)}, 种子数: {N_SEEDS}, 总运行次数: {len(EXPERIMENT_CONFIGS) * N_SEEDS}")
    log.info("=" * 80)

    # === 1. 加载数据 (所有配置共享) ===
    log.info("\n[1/3] 加载训练和验证数据...")
    X_train_df, y_train, group_train, w_train, base_cs_train, match_ids_train = extract_pick_ranking_data("train")
    X_val_df, y_val, group_val, w_val, base_cs_val, match_ids_val = extract_pick_ranking_data("val")

    # StandardScaler (与生产代码一致: 在 train 上 fit)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values.astype(np.float32))
    X_val = scaler.transform(X_val_df.values.astype(np.float32))

    log.info(f"  Train: {X_train.shape[0]} samples, {len(group_train)} queries, {len(np.unique(match_ids_train))} matches")
    log.info(f"  Val:   {X_val.shape[0]} samples, {len(group_val)} queries, {len(np.unique(match_ids_val))} matches")

    # === 2. 运行所有配置 × 所有种子 ===
    log.info("\n[2/3] 运行对比实验...")
    all_results = {}

    for config_name, config_info in EXPERIMENT_CONFIGS.items():
        log.info(f"\n{'='*60}")
        log.info(f"  配置: {config_name}")
        log.info(f"  描述: {config_info['description']}")
        log.info(f"{'='*60}")

        seed_results = []
        for seed_idx, seed in enumerate(SEEDS):
            log.info(f"\n  [种子 {seed_idx+1}/{N_SEEDS}] seed={seed}")
            t0 = time.time()

            result = _run_single_config(
                X_train, y_train, group_train, w_train, base_cs_train, match_ids_train,
                X_val, y_val, group_val, w_val, base_cs_val, match_ids_val,
                config_info, seed,
            )

            elapsed = time.time() - t0
            vm = result["val_metrics"]
            om = result["oof_metrics"]
            bvm = result.get("base_val_metrics", {})
            delta_p10 = vm.get(10, 0) - bvm.get(10, 0) if bvm else 0
            log.info(f"  [种子 {seed_idx+1}] 耗时={elapsed:.1f}s | "
                     f"Val P@1={vm.get(1,0):.2f}% P@10={vm.get(10,0):.2f}% | "
                     f"OOF P@10={om.get(10,0):.2f}% | "
                     f"Δ P@10(vs base)={delta_p10:+.2f}%")

            seed_results.append({
                "seed": seed,
                "val_metrics": vm,
                "oof_metrics": om,
                "base_val_metrics": bvm,
                "best_iterations": result["best_iterations"],
                "elapsed_sec": elapsed,
            })

        all_results[config_name] = {
            "description": config_info["description"],
            "overrides": config_info["overrides"],
            "seed_results": seed_results,
        }

    # === 3. 汇总和分析 ===
    log.info("\n[3/3] 生成汇总报告...")
    report = _generate_report(all_results)

    # 保存结果
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(EXPERIMENT_DIR, f"cascade_pick_experiment_{ts}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"\n实验结果已保存: {result_path}")

    # 打印汇总表
    _print_summary_table(report)

    log.info(f"\n总耗时: {time.time() - total_t0:.1f}s")
    return report


def _generate_report(all_results):
    """生成汇总报告"""
    report = {
        "experiment_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "configs": {},
    }

    for config_name, config_data in all_results.items():
        seed_results = config_data["seed_results"]

        # 收集各种子的指标
        val_p1, val_p3, val_p5, val_p10 = [], [], [], []
        oof_p10 = []
        base_p10 = []
        delta_p1, delta_p10 = [], []
        best_iters = []
        elapsed_secs = []

        for sr in seed_results:
            vm = sr["val_metrics"]
            om = sr["oof_metrics"]
            bvm = sr.get("base_val_metrics", {})

            val_p1.append(vm.get(1, 0))
            val_p3.append(vm.get(3, 0))
            val_p5.append(vm.get(5, 0))
            val_p10.append(vm.get(10, 0))
            oof_p10.append(om.get(10, 0))
            if bvm:
                base_p10.append(bvm.get(10, 0))
                delta_p1.append(vm.get(1, 0) - bvm.get(1, 0))
                delta_p10.append(vm.get(10, 0) - bvm.get(10, 0))
            if sr["best_iterations"]:
                best_iters.append(np.mean(sr["best_iterations"]))
            elapsed_secs.append(sr["elapsed_sec"])

        def mean_std(arr):
            arr = np.array(arr)
            return {"mean": float(arr.mean()), "std": float(arr.std()), "values": arr.tolist()}

        report["configs"][config_name] = {
            "description": config_data["description"],
            "overrides": config_data["overrides"],
            "val_p1": mean_std(val_p1),
            "val_p3": mean_std(val_p3),
            "val_p5": mean_std(val_p5),
            "val_p10": mean_std(val_p10),
            "oof_p10": mean_std(oof_p10),
            "base_val_p10": mean_std(base_p10) if base_p10 else None,
            "delta_p1": mean_std(delta_p1) if delta_p1 else None,
            "delta_p10": mean_std(delta_p10) if delta_p10 else None,
            "avg_best_iteration": float(np.mean(best_iters)) if best_iters else None,
            "avg_elapsed_sec": float(np.mean(elapsed_secs)),
        }

    return report


def _print_summary_table(report):
    """打印汇总对比表"""
    log.info("\n" + "=" * 100)
    log.info("  实验汇总报告: Cascade Pick 超参数对比")
    log.info("=" * 100)

    # 表头
    header = f"  {'配置':<20} {'Val P@1':>12} {'Val P@10':>12} {'Δ P@1':>12} {'Δ P@10':>12} {'OOF P@10':>12} {'Avg Iter':>10}"
    log.info(header)
    log.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for config_name, cfg in report["configs"].items():
        val_p1 = cfg["val_p1"]
        val_p10 = cfg["val_p10"]
        oof_p10 = cfg["oof_p10"]
        delta_p1 = cfg.get("delta_p1")
        delta_p10 = cfg.get("delta_p10")
        avg_iter = cfg.get("avg_best_iteration", 0)

        p1_str = f"{val_p1['mean']:.2f}±{val_p1['std']:.2f}"
        p10_str = f"{val_p10['mean']:.2f}±{val_p10['std']:.2f}"
        d_p1_str = f"{delta_p1['mean']:+.2f}±{delta_p1['std']:.2f}" if delta_p1 else "N/A"
        d_p10_str = f"{delta_p10['mean']:+.2f}±{delta_p10['std']:.2f}" if delta_p10 else "N/A"
        oof_str = f"{oof_p10['mean']:.2f}±{oof_p10['std']:.2f}"
        iter_str = f"{avg_iter:.0f}" if avg_iter else "N/A"

        log.info(f"  {config_name:<20} {p1_str:>12} {p10_str:>12} {d_p1_str:>12} {d_p10_str:>12} {oof_str:>12} {iter_str:>10}")

    log.info("")

    # 结论分析
    log.info("  关键分析:")
    log.info("  1. Δ P@K = cascade_final P@K - transformer_base P@K (正值表示 cascade 有提升)")
    log.info("  2. Val 指标为主要评估依据 (TF 未在 val 上训练)")
    log.info("  3. OOF 指标被 TF In-Sample 污染，仅观察过拟合趋势")
    log.info("  4. ±表示 5 个随机种子的标准差，标准差大说明不稳定")

    # 找出最优配置
    best_config = None
    best_delta_p10 = -1e9
    for config_name, cfg in report["configs"].items():
        if cfg["overrides"] is None:
            continue  # 跳过基线
        delta = cfg.get("delta_p10")
        if delta and delta["mean"] > best_delta_p10:
            best_delta_p10 = delta["mean"]
            best_config = config_name

    if best_config:
        best_cfg = report["configs"][best_config]
        log.info(f"\n  最优配置 (Δ P@10 最大): {best_config}")
        log.info(f"    描述: {best_cfg['description']}")
        log.info(f"    Val P@10: {best_cfg['val_p10']['mean']:.2f}±{best_cfg['val_p10']['std']:.2f}")
        log.info(f"    Δ P@10:  {best_cfg['delta_p10']['mean']:+.2f}±{best_cfg['delta_p10']['std']:.2f}")
        log.info(f"    Δ P@1:   {best_cfg['delta_p1']['mean']:+.2f}±{best_cfg['delta_p1']['std']:.2f}")
        if best_delta_p10 <= 0:
            log.info(f"    ⚠ 警告: 最优配置的 Δ P@10 ≤ 0，说明 cascade 未带来有效提升")
        else:
            log.info(f"    ✓ 最优配置的 Δ P@10 > 0，说明 cascade 带来了有效提升")

    log.info("=" * 100)


if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))

    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"cascade_pick_experiment_{_run_ts}.log"), encoding="utf-8"
    )
    _file_handler.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _file_handler.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_file_handler)

    run_experiment()
