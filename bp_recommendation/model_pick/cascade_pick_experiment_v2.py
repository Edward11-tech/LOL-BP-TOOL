#!/usr/bin/env python3
"""
Cascade Pick 融合模式与架构优化实验
======================================

实验背景:
    2026-06-27 (commit dce8c68) 将融合模式从 blend (rank normalize + alpha) 改为
    residual init_score 后，cascade 对 transformer_base 的提升从 2-3% 退化到 0.05%。

    根因分析:
    - Blend 模式: LGBM 独立学习排序信号，推理时 rank_normalize 后与 TF 分数线性混合
      → LGBM 能学到 TF 无法捕获的独立信号 (mastery/synergy/counter/phase-aware)
    - Residual 模式: LGBM 以 TF logits 为 init_score，只学习残差
      → TF 在训练集上 In-Sample 预测过强，残差几乎为 0，LGBM 学不到东西

实验设计:
    维度 1 - 融合模式:
      A. blend: LGBM 独立训练 + rank_normalize + alpha 混合 (旧模式)
      B. residual: LGBM init_score 残差训练 (当前模式)

    维度 2 - 超参数:
      1. blend_optuna: 旧 blend 时代的 Optuna 最优参数 (lr=0.015, ff=0.6, L1=0.5, L2=1.5)
      2. residual_optuna: residual 时代的 Optuna 参数 (lr=0.008, extra_trees, ff_bynode=0.5)
      3. medium_reg: 中等正则化 (lr=0.01, ff=0.5, ff_bynode=0.7)

    维度 3 - 评估:
      - Val 集 Δ P@K (cascade_final - transformer_base)
      - 5 个随机种子均值±标准差
      - 特征重要性分析

用法:
    cd /Users/siwentu/Desktop/LOL analysis
    conda run -n LOL python -m bp_recommendation.model_pick.cascade_pick_experiment_v2
"""

import os
import sys
import time
import json
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
    _rank_normalize,
    FEATURE_COLS,
    CS_TOP_K,
)
from logger_config import get_logger, setup_logging
from common.paths import ensure_dirs as _ensure_common_dirs

_ensure_common_dirs()

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

log = get_logger(__name__)

N_FOLDS = 5
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2024]

# ============================================================
# 超参数配置
# ============================================================

# 旧 blend 时代的 Optuna 最优参数 (commit 3ce9e50, 2026-06)
# 来自 cascade_pick_search.py 的 TPE 搜索
BLEND_OPTUNA_CONFIG = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "ndcg_at": [10],
    "num_leaves": 31,
    "max_depth": 7,
    "min_data_in_leaf": 20,
    "learning_rate": 0.015,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 1.5,
    "max_bin": 255,
    "verbose": -1,
}
BLEND_OPTUNA_NUM_ROUND = 3000
BLEND_OPTUNA_EARLY_STOP = 150

# Residual 时代的 Optuna 参数 (commit dce8c68, 2026-06-27)
RESIDUAL_OPTUNA_CONFIG = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "ndcg_at": [10],
    "num_leaves": 48,
    "max_depth": 8,
    "min_data_in_leaf": 17,
    "learning_rate": 0.007980833558993788,
    "feature_fraction": 0.32917339062527257,
    "feature_fraction_bynode": 0.5,
    "extra_trees": True,
    "bagging_fraction": 0.8859303492103779,
    "bagging_freq": 5,
    "lambda_l1": 0.0020744246659736253,
    "lambda_l2": 0.04360744274903356,
    "max_bin": 255,
    "verbose": -1,
}
RESIDUAL_OPTUNA_NUM_ROUND = 4500
RESIDUAL_OPTUNA_EARLY_STOP = 250

# 中等正则化参数 (实验 v1 选出的最优)
MEDIUM_REG_CONFIG = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "ndcg_at": [10],
    "num_leaves": 48,
    "max_depth": 8,
    "min_data_in_leaf": 17,
    "learning_rate": 0.01,
    "feature_fraction": 0.5,
    "feature_fraction_bynode": 0.7,
    "bagging_fraction": 0.8859303492103779,
    "bagging_freq": 5,
    "lambda_l1": 0.0020744246659736253,
    "lambda_l2": 0.04360744274903356,
    "max_bin": 255,
    "verbose": -1,
}
MEDIUM_REG_NUM_ROUND = 4500
MEDIUM_REG_EARLY_STOP = 250


# ============================================================
# 实验配置: 融合模式 × 超参数
# ============================================================
EXPERIMENT_CONFIGS = {
    "A_blend_optuna": {
        "description": "Blend + Optuna旧参数 (lr=0.015, L1=0.5, L2=1.5) — 2026-06版本",
        "fusion_mode": "blend",
        "lgb_config": BLEND_OPTUNA_CONFIG,
        "num_round": BLEND_OPTUNA_NUM_ROUND,
        "early_stop": BLEND_OPTUNA_EARLY_STOP,
    },
    "B_blend_medium_reg": {
        "description": "Blend + 中等正则化 (lr=0.01, ff_bynode=0.7) — 实验v1最优参数",
        "fusion_mode": "blend",
        "lgb_config": MEDIUM_REG_CONFIG,
        "num_round": MEDIUM_REG_NUM_ROUND,
        "early_stop": MEDIUM_REG_EARLY_STOP,
    },
    "C_residual_optuna": {
        "description": "Residual + Optuna残差参数 (extra_trees, lr=0.008) — 2026-06-27版本",
        "fusion_mode": "residual",
        "lgb_config": RESIDUAL_OPTUNA_CONFIG,
        "num_round": RESIDUAL_OPTUNA_NUM_ROUND,
        "early_stop": RESIDUAL_OPTUNA_EARLY_STOP,
    },
    "D_residual_medium_reg": {
        "description": "Residual + 中等正则化 (lr=0.01, ff_bynode=0.7) — 实验v1最优参数",
        "fusion_mode": "residual",
        "lgb_config": MEDIUM_REG_CONFIG,
        "num_round": MEDIUM_REG_NUM_ROUND,
        "early_stop": MEDIUM_REG_EARLY_STOP,
    },
    "E_no_cascade": {
        "description": "基线: 仅 transformer_base, 无 cascade",
        "fusion_mode": "none",
        "lgb_config": None,
        "num_round": 0,
        "early_stop": 0,
    },
}


def _run_single_experiment(X_train, y_train, group_train, w_train, base_cs_train,
                           match_ids_train, X_val, y_val, group_val, w_val, base_cs_val,
                           match_ids_val, config, seed):
    """运行单组实验配置 + 单种子"""
    fusion_mode = config["fusion_mode"]

    if fusion_mode == "none":
        # 基线
        val_metrics = _evaluate_pick_at_k(base_cs_val, y_val, group_val)
        return {
            "val_metrics": val_metrics,
            "base_val_metrics": val_metrics,
            "best_iterations": [],
            "feature_importance": {},
        }

    lgb_config = {**config["lgb_config"], "seed": seed}
    num_round = config["num_round"]
    early_stop = config["early_stop"]

    # === GroupKFold CV ===
    row_match_ids = np.repeat(match_ids_train, group_train)
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_pred = np.zeros(len(y_train), dtype=np.float64)
    fold_models = []
    best_iterations = []
    use_init_score = (fusion_mode == "residual")

    for fold_i, (t_idx, v_idx) in enumerate(gkf.split(X_train, y_train, row_match_ids)):
        t_match_ids = np.unique(row_match_ids[t_idx])
        t_query_mask = np.isin(match_ids_train, t_match_ids)
        v_match_ids = np.unique(row_match_ids[v_idx])
        v_query_mask = np.isin(match_ids_train, v_match_ids)

        if use_init_score:
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
        else:
            train_ds = lgb.Dataset(
                X_train[t_idx], y_train[t_idx],
                group=group_train[t_query_mask],
                weight=w_train[t_idx],
            )
            val_ds = lgb.Dataset(
                X_train[v_idx], y_train[v_idx],
                group=group_train[v_query_mask],
                weight=w_train[v_idx],
                reference=train_ds,
            )

        callbacks = [lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            lgb_config, train_ds, num_boost_round=num_round,
            valid_sets=[val_ds], callbacks=callbacks,
        )

        if use_init_score:
            oof_pred[v_idx] = model.predict(X_train[v_idx]) + base_cs_train[v_idx]
        else:
            oof_pred[v_idx] = model.predict(X_train[v_idx])

        fold_models.append(model)
        best_iterations.append(model.best_iteration)

    # === Val 评估 ===
    if use_init_score:
        # Residual: val_pred = LGBM residual + TF base
        val_preds = np.array([m.predict(X_val) + base_cs_val for m in fold_models])
        val_final_pred = val_preds.mean(axis=0)
    else:
        # Blend: LGBM 独立预测, 然后 rank_normalize + alpha 混合
        val_preds = np.array([m.predict(X_val) for m in fold_models])
        lgb_val_pred = val_preds.mean(axis=0)

        # 搜索最优 alpha
        cs_val_rn = _rank_normalize(base_cs_val, group_val)
        lgb_val_rn = _rank_normalize(lgb_val_pred, group_val)

        best_alpha, best_p10 = 0.0, 0.0
        for alpha_int in range(0, 101, 2):
            alpha = alpha_int / 100.0
            blend_scores = alpha * cs_val_rn + (1.0 - alpha) * lgb_val_rn
            p10 = _evaluate_pick_at_k(blend_scores, y_val, group_val)[10]
            if p10 > best_p10:
                best_p10 = p10
                best_alpha = alpha

        val_final_pred = best_alpha * cs_val_rn + (1.0 - best_alpha) * lgb_val_rn

    val_metrics = _evaluate_pick_at_k(val_final_pred, y_val, group_val)
    base_val_metrics = _evaluate_pick_at_k(base_cs_val, y_val, group_val)

    # === 特征重要性 ===
    importance = fold_models[0].feature_importance(importance_type="gain")
    feat_imp = {FEATURE_COLS[i]: float(importance[i]) for i in range(len(FEATURE_COLS))}

    return {
        "val_metrics": val_metrics,
        "base_val_metrics": base_val_metrics,
        "best_iterations": best_iterations,
        "feature_importance": feat_imp,
        "best_alpha": best_alpha if not use_init_score else None,
    }


def run_experiment():
    """运行完整对比实验"""
    total_t0 = time.time()
    log.info("=" * 80)
    log.info("  Cascade Pick 融合模式与架构优化实验 (v2)")
    log.info(f"  配置数: {len(EXPERIMENT_CONFIGS)}, 种子数: {N_SEEDS}")
    log.info(f"  总运行次数: {len(EXPERIMENT_CONFIGS) * N_SEEDS}")
    log.info("=" * 80)

    # === 1. 加载数据 ===
    log.info("\n[1/3] 加载训练和验证数据...")
    X_train_df, y_train, group_train, w_train, base_cs_train, match_ids_train = extract_pick_ranking_data("train")
    X_val_df, y_val, group_val, w_val, base_cs_val, match_ids_val = extract_pick_ranking_data("val")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values.astype(np.float32))
    X_val = scaler.transform(X_val_df.values.astype(np.float32))

    log.info(f"  Train: {X_train.shape[0]} samples, {len(group_train)} queries, {len(np.unique(match_ids_train))} matches")
    log.info(f"  Val:   {X_val.shape[0]} samples, {len(group_val)} queries, {len(np.unique(match_ids_val))} matches")

    # TF Base 在 val 上的表现
    base_val_metrics = _evaluate_pick_at_k(base_cs_val, y_val, group_val)
    log.info(f"  TF Base Val: P@1={base_val_metrics[1]:.2f}% P@3={base_val_metrics[3]:.2f}% "
             f"P@5={base_val_metrics[5]:.2f}% P@10={base_val_metrics[10]:.2f}%")

    # === 2. 运行所有配置 × 所有种子 ===
    log.info("\n[2/3] 运行对比实验...")
    all_results = {}

    for config_name, config_info in EXPERIMENT_CONFIGS.items():
        log.info(f"\n{'='*70}")
        log.info(f"  配置: {config_name}")
        log.info(f"  描述: {config_info['description']}")
        log.info(f"  融合: {config_info['fusion_mode']}")
        log.info(f"{'='*70}")

        seed_results = []
        for seed_idx, seed in enumerate(SEEDS):
            log.info(f"\n  [种子 {seed_idx+1}/{N_SEEDS}] seed={seed}")
            t0 = time.time()

            result = _run_single_experiment(
                X_train, y_train, group_train, w_train, base_cs_train, match_ids_train,
                X_val, y_val, group_val, w_val, base_cs_val, match_ids_val,
                config_info, seed,
            )

            elapsed = time.time() - t0
            vm = result["val_metrics"]
            bvm = result["base_val_metrics"]
            delta_p1 = vm.get(1, 0) - bvm.get(1, 0)
            delta_p10 = vm.get(10, 0) - bvm.get(10, 0)
            alpha_str = f" alpha={result.get('best_alpha', 'N/A')}" if result.get('best_alpha') is not None else ""
            log.info(f"  [种子 {seed_idx+1}] 耗时={elapsed:.1f}s | "
                     f"Val P@1={vm.get(1,0):.2f}% P@10={vm.get(10,0):.2f}% | "
                     f"Δ P@1={delta_p1:+.2f}% Δ P@10={delta_p10:+.2f}%{alpha_str}")

            seed_results.append({
                "seed": seed,
                "val_metrics": vm,
                "base_val_metrics": bvm,
                "best_iterations": result["best_iterations"],
                "feature_importance": result["feature_importance"],
                "best_alpha": result.get("best_alpha"),
                "elapsed_sec": elapsed,
            })

        all_results[config_name] = {
            "description": config_info["description"],
            "fusion_mode": config_info["fusion_mode"],
            "seed_results": seed_results,
        }

    # === 3. 汇总 ===
    log.info("\n[3/3] 生成汇总报告...")
    report = _generate_report(all_results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(EXPERIMENT_DIR, f"cascade_pick_experiment_v2_{ts}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"\n实验结果已保存: {result_path}")

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

        val_p1, val_p3, val_p5, val_p10 = [], [], [], []
        delta_p1, delta_p3, delta_p5, delta_p10 = [], [], [], []
        best_iters = []
        elapsed_secs = []
        alphas = []
        all_feat_imp = {}

        for sr in seed_results:
            vm = sr["val_metrics"]
            bvm = sr["base_val_metrics"]

            val_p1.append(vm.get(1, 0))
            val_p3.append(vm.get(3, 0))
            val_p5.append(vm.get(5, 0))
            val_p10.append(vm.get(10, 0))

            delta_p1.append(vm.get(1, 0) - bvm.get(1, 0))
            delta_p3.append(vm.get(3, 0) - bvm.get(3, 0))
            delta_p5.append(vm.get(5, 0) - bvm.get(5, 0))
            delta_p10.append(vm.get(10, 0) - bvm.get(10, 0))

            if sr["best_iterations"]:
                best_iters.append(np.mean(sr["best_iterations"]))
            elapsed_secs.append(sr["elapsed_sec"])
            if sr.get("best_alpha") is not None:
                alphas.append(sr["best_alpha"])

            # 累加特征重要性
            for fname, imp in sr["feature_importance"].items():
                if fname not in all_feat_imp:
                    all_feat_imp[fname] = []
                all_feat_imp[fname].append(imp)

        def mean_std(arr):
            arr = np.array(arr)
            return {"mean": float(arr.mean()), "std": float(arr.std()), "values": arr.tolist()}

        # 平均特征重要性
        avg_feat_imp = {fname: float(np.mean(imps)) for fname, imps in all_feat_imp.items()}
        sorted_feat_imp = dict(sorted(avg_feat_imp.items(), key=lambda x: -x[1]))

        report["configs"][config_name] = {
            "description": config_data["description"],
            "fusion_mode": config_data["fusion_mode"],
            "val_p1": mean_std(val_p1),
            "val_p3": mean_std(val_p3),
            "val_p5": mean_std(val_p5),
            "val_p10": mean_std(val_p10),
            "delta_p1": mean_std(delta_p1),
            "delta_p3": mean_std(delta_p3),
            "delta_p5": mean_std(delta_p5),
            "delta_p10": mean_std(delta_p10),
            "avg_best_iteration": float(np.mean(best_iters)) if best_iters else None,
            "avg_elapsed_sec": float(np.mean(elapsed_secs)),
            "avg_alpha": float(np.mean(alphas)) if alphas else None,
            "feature_importance": sorted_feat_imp,
        }

    return report


def _print_summary_table(report):
    """打印汇总对比表"""
    log.info("\n" + "=" * 110)
    log.info("  实验汇总报告: Cascade Pick 融合模式与架构优化 (v2)")
    log.info("=" * 110)

    # 主指标表
    header = f"  {'配置':<25} {'融合':>8} {'Val P@1':>12} {'Val P@10':>12} {'Δ P@1':>12} {'Δ P@10':>12} {'Alpha':>8} {'Iter':>8}"
    log.info(header)
    log.info(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    for config_name, cfg in report["configs"].items():
        val_p1 = cfg["val_p1"]
        val_p10 = cfg["val_p10"]
        delta_p1 = cfg["delta_p1"]
        delta_p10 = cfg["delta_p10"]
        avg_iter = cfg.get("avg_best_iteration", 0)
        avg_alpha = cfg.get("avg_alpha")

        p1_str = f"{val_p1['mean']:.2f}±{val_p1['std']:.2f}"
        p10_str = f"{val_p10['mean']:.2f}±{val_p10['std']:.2f}"
        d_p1_str = f"{delta_p1['mean']:+.2f}±{delta_p1['std']:.2f}"
        d_p10_str = f"{delta_p10['mean']:+.2f}±{delta_p10['std']:.2f}"
        alpha_str = f"{avg_alpha:.2f}" if avg_alpha is not None else "N/A"
        iter_str = f"{avg_iter:.0f}" if avg_iter else "N/A"

        log.info(f"  {config_name:<25} {cfg['fusion_mode']:>8} {p1_str:>12} {p10_str:>12} "
                 f"{d_p1_str:>12} {d_p10_str:>12} {alpha_str:>8} {iter_str:>8}")

    log.info("")

    # 找最优配置
    best_config = None
    best_delta_p10 = -1e9
    for config_name, cfg in report["configs"].items():
        if cfg["fusion_mode"] == "none":
            continue
        delta = cfg["delta_p10"]
        if delta["mean"] > best_delta_p10:
            best_delta_p10 = delta["mean"]
            best_config = config_name

    if best_config:
        best_cfg = report["configs"][best_config]
        log.info(f"  最优配置 (Δ P@10 最大): {best_config}")
        log.info(f"    描述: {best_cfg['description']}")
        log.info(f"    融合模式: {best_cfg['fusion_mode']}")
        log.info(f"    Val P@1:  {best_cfg['val_p1']['mean']:.2f}±{best_cfg['val_p1']['std']:.2f}")
        log.info(f"    Val P@10: {best_cfg['val_p10']['mean']:.2f}±{best_cfg['val_p10']['std']:.2f}")
        log.info(f"    Δ P@1:    {best_cfg['delta_p1']['mean']:+.2f}±{best_cfg['delta_p1']['std']:.2f}")
        log.info(f"    Δ P@10:   {best_cfg['delta_p10']['mean']:+.2f}±{best_cfg['delta_p10']['std']:.2f}")
        if best_cfg.get("avg_alpha") is not None:
            log.info(f"    Avg Alpha: {best_cfg['avg_alpha']:.2f}")

        if best_delta_p10 > 1.0:
            log.info(f"    ✓✓ 显著提升 (Δ P@10 > 1%)，cascade 发挥了重要作用")
        elif best_delta_p10 > 0.1:
            log.info(f"    ✓ 有效提升 (Δ P@10 > 0.1%)")
        elif best_delta_p10 > 0:
            log.info(f"    ~ 微弱提升 (Δ P@10 > 0 但 < 0.1%)")
        else:
            log.info(f"    ⚠ 无提升 (Δ P@10 ≤ 0)")

    # 特征重要性 Top-10
    log.info(f"\n  {'='*70}")
    log.info(f"  特征重要性 Top-10 (各配置对比)")
    log.info(f"  {'='*70}")
    log.info(f"  {'特征':<25}", end="")
    for config_name in report["configs"]:
        if report["configs"][config_name]["fusion_mode"] == "none":
            continue
        log.info(f" {config_name[:12]:>12}", end="")
    log.info("")

    # 收集所有特征名
    all_features = set()
    for config_name, cfg in report["configs"].items():
        if cfg["fusion_mode"] == "none":
            continue
        all_features.update(cfg["feature_importance"].keys())

    # 按第一个非基线配置的重要性排序
    first_config = None
    for config_name in report["configs"]:
        if report["configs"][config_name]["fusion_mode"] != "none":
            first_config = config_name
            break

    if first_config:
        sorted_features = sorted(
            all_features,
            key=lambda f: -report["configs"][first_config]["feature_importance"].get(f, 0)
        )
        for fname in sorted_features[:10]:
            log.info(f"  {fname:<25}", end="")
            for config_name in report["configs"]:
                cfg = report["configs"][config_name]
                if cfg["fusion_mode"] == "none":
                    continue
                imp = cfg["feature_importance"].get(fname, 0)
                log.info(f" {imp:>12.1f}", end="")
            log.info("")

    log.info(f"\n  {'='*70}")
    log.info(f"  关键分析:")
    log.info(f"  1. Δ P@K = cascade_final P@K - transformer_base P@K (正值=cascade有提升)")
    log.info(f"  2. Blend 模式: LGBM 独立训练 + rank_normalize + alpha 混合")
    log.info(f"  3. Residual 模式: LGBM init_score 残差训练")
    log.info(f"  4. Alpha: Blend 模式下 TF 分数权重 (1-alpha = LGBM 权重)")
    log.info(f"  5. ±表示 5 个随机种子的标准差")
    log.info(f"  {'='*110}")


if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))

    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"cascade_pick_experiment_v2_{_run_ts}.log"), encoding="utf-8"
    )
    _file_handler.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _file_handler.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_file_handler)

    run_experiment()
