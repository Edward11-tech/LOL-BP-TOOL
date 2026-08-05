"""
Pick Cascade LightGBM 级联模型训练
=============================================
使用 LightGBM LambdaMART 对 CS/NoCS 双 Transformer 模型输出进行级联融合，
通过 Transformer logit + 阶段感知特征（Phase-Aware Routing）构建更精准的 Pick 排序模型。

功能描述:
    - 从训练好的 CS/NoCS Transformer 模型提取 OOF 预测分数
    - 构建 Transformer 输出 + 原始统计特征 + 阶段感知特征的融合矩阵
    - 使用 5-Fold GroupKFold 交叉验证训练 LightGBM Ranker
    - 支持 StandardScaler 特征标准化
    - 支持残差训练（init_score）
    - 计算 Pick@K 评估指标
    - 保存融合模型和 blend_alpha 权重到配置文件

主要函数/常量:
    - FEATURE_COLS: Cascade 模型特征列定义
    - extract_pick_ranking_data(): 提取训练数据
    - _build_feature_matrix_batch(): 批量构建特征矩阵
    - _compute_group_features(): 计算组级特征
    - train_cascade_pick(): 训练 Cascade Pick 模型
    - _evaluate_pick_at_k(): 评估 Pick@K 指标

使用方法:
    cd <project_root>
    python -m bp_recommendation.model_pick.cascade_pick
    
    注意: 需要先训练好 Pick CS/NoCS Transformer 模型并生成 OOF 预测。
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

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(TEST_DIR)))

from bp_recommendation.config import (
    is_production_mode,
    get_production_num_boost_round,
    get_production_blend_alpha,
    save_best_params,
    record_scaler_coefficients,
    record_production_params,
    get_config,
)
from bp_recommendation.feature_pipeline import CANDIDATE_FEAT_MAP, load_champion_vocabulary, CHAMPION_VOCABULARY_JSON

# 共享数据异常检测工具
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(TEST_DIR))))
from data_checks import check_array, check_labels, check_groups, check_predictions
from logger_config import get_logger
from common.paths import RECOMMENDATION_METRICS_DIR, ensure_dirs as _ensure_common_dirs
_ensure_common_dirs()

LOG_DIR = os.path.join(TEST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log = get_logger(__name__)

FEATURES_DIR = os.path.join(TEST_DIR, "features")
CKPT_DIR = os.path.join(TEST_DIR, "checkpoints", "cascade_pick")

CS_TOP_K = 50
N_FOLDS = 5

# ============================================================
# 超参数: 融合模式实验选优 (cascade_pick_experiment_v2.py, 2026-07-28)
# ============================================================
# 实验背景:
#   2026-06-27 (commit dce8c68) 将融合模式从 blend 改为 residual init_score，
#   导致 cascade 对 transformer_base 的提升从 2-3% 退化到 0.05%。
#
# 实验方案:
#   5 组配置 (融合模式 × 超参数) × 5 个随机种子 (42/123/456/789/2024)
#
# 实验结果 (Val 集, 5 种子均值±标准差):
#   A_blend_optuna (blend+旧参数):     Δ P@10 = +2.66±0.15%  Δ P@1 = +0.31±0.11%
#   B_blend_medium (blend+本配置):     Δ P@10 = +8.30±0.39%  Δ P@1 = +7.63±0.38%  ← 最优
#   C_residual_optuna (residual+旧):   Δ P@10 = -0.03±0.08%  Δ P@1 = +0.07±0.04%
#   D_residual_medium (residual+本):   Δ P@10 = +0.05±0.03%  Δ P@1 = +0.15±0.06%
#
# 根因分析:
#   Residual 模式下，TF 在训练集上 In-Sample 预测过强，残差几乎为 0，LGBM 学不到东西。
#   Blend 模式下，LGBM 独立学习排序信号，推理时 rank_normalize + alpha 混合，
#   LGBM 能学到 TF 无法捕获的独立信号 (mastery/synergy/counter/phase-aware)。
#
# 选择理由:
#   1. Δ P@10 = +8.30% 远超其他配置，5 个种子标准差仅 0.39%
#   2. Alpha=0.00 说明 LGBM 完全替代了 TF base，学到了更强的排序模式
#   3. medium_reg 参数 (lr=0.01, ff_bynode=0.7) 比 blend_optuna (lr=0.015, L1=0.5) 更优
LGB_CONFIG = {
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
    "seed": 42,
}
NUM_ROUND = 4500
EARLY_STOP = 250


# ================= 精简且致命的特征列表 =================
FEATURE_COLS = [
    # 基础分数
    "cs_logit", "nocs_logit", "logit_diff",
    "cs_rank_pct", "nocs_rank_pct",
    
    # 原始基础特征
    "meta_presence", "meta_wr",
    "player_mastery", "player_recent_wr", "player_recent_games",
    "synergy_ally", "counter_enemy", "role_fit_ally",
    
    # 【核心】：阶段感知交叉特征 (Phase-Aware)
    "p1_meta_presence", "p1_mastery", "p1_cs_logit",
    "p2_role_fit", "p2_synergy", "p2_counter", "p2_mastery_x_role",
    
    # 【必杀技】：残局拼图 (仅 Phase 2 或连选位激活)
    "perfect_fit",         # 契合空缺 + 有队友配合
    "counter_kill",        # 契合空缺 + 克制对面
    "combo_synergy",       # 与上一手立刻选出的英雄有配合 (Tuple Pick 核心)
    "enemy_mastery_choke", # 对面绝活哥 + 我能克制他 (针对敌方绝活)
]

# 特征名 → 索引的映射，供外部模块查询
FEAT_IDX = {name: i for i, name in enumerate(FEATURE_COLS)}

def _compute_group_features(sample_logits, sample_mask, champion_start_idx, vocab_size):
    rank_map = np.argsort(np.argsort(-sample_logits)) + 1.0
    return {"rank_map": rank_map}

def _build_feature_matrix_batch(cs_logits_arr, cs_ranks_arr, cs_gf,
                                 nocs_logits_arr, nocs_ranks_arr, nocs_gf,
                                 cand_feats, total_valid_in_group, group_size, target_step):
    N = len(cs_logits_arr)
    FI = CANDIDATE_FEAT_MAP

    meta_presence = cand_feats[:, FI["meta_presence"]]
    meta_wr = cand_feats[:, FI["meta_wr"]]
    mastery = cand_feats[:, FI["player_mastery"]]
    recent_wr = cand_feats[:, FI["player_recent_wr"]]
    recent_games = cand_feats[:, FI["player_recent_games"]]
    
    synergy_ally = cand_feats[:, FI["ally_synergy"]]
    counter_enemy = cand_feats[:, FI["enemy_counter"]]
    role_fit_ally = cand_feats[:, FI["ally_role_fit"]]
    enemy_mastery_max = cand_feats[:, FI["enemy_mastery_max"]]
    
    n_ally_picked = cand_feats[:, FI["n_ally_picked"]]
    last_ally_synergy = cand_feats[:, FI["last_ally_synergy"]]

    logit_diff = cs_logits_arr - nocs_logits_arr
    cs_rank_pct = 1.0 - (cs_ranks_arr / max(total_valid_in_group, 1.0))
    nocs_rank_pct = 1.0 - (nocs_ranks_arr / max(total_valid_in_group, 1.0))

    # ---------------- 核心：阶段路由 ----------------
    # Step 6-11 是 Phase 1 (前三手); Step 16-19 是 Phase 2 (后两手)
    is_p1_scalar = 1.0 if target_step < 12 else 0.0
    is_p2_scalar = 1.0 - is_p1_scalar

    # P1 看重全局和基础直觉
    p1_meta_presence = meta_presence * is_p1_scalar
    p1_mastery = mastery * is_p1_scalar
    p1_cs_logit = cs_logits_arr * is_p1_scalar

    # P2 极度看重空缺和博弈
    p2_role_fit = role_fit_ally * is_p2_scalar
    p2_synergy = synergy_ally * is_p2_scalar
    p2_counter = counter_enemy * is_p2_scalar
    p2_mastery_x_role = mastery * role_fit_ally * is_p2_scalar

    # ---------------- 必杀技 ----------------
    perfect_fit = role_fit_ally * synergy_ally
    counter_kill = role_fit_ally * counter_enemy
    enemy_mastery_choke = enemy_mastery_max * counter_enemy
    
    # Tuple Pick 连选效应：如果有队友已经选出，放大最近一个队友的 synergy
    has_ally = (n_ally_picked >= 1.0).astype(np.float32)
    combo_synergy = last_ally_synergy * has_ally

    X = np.column_stack([
        cs_logits_arr, nocs_logits_arr, logit_diff,
        cs_rank_pct, nocs_rank_pct,
        meta_presence, meta_wr,
        mastery, recent_wr, recent_games,
        synergy_ally, counter_enemy, role_fit_ally,
        
        p1_meta_presence, p1_mastery, p1_cs_logit,
        p2_role_fit, p2_synergy, p2_counter, p2_mastery_x_role,
        
        perfect_fit, counter_kill, combo_synergy, enemy_mastery_choke
    ]).astype(np.float32)

    return X

def extract_pick_ranking_data(split="val", features_dir=None):
    t0 = time.time()
    fdir = features_dir or FEATURES_DIR
    log.info(f"Loading {split} logits from {fdir}...")

    cs_data = np.load(os.path.join(fdir, f"ALL_{split}_logits_cs.npz"))
    nocs_data = np.load(os.path.join(fdir, f"ALL_{split}_logits_nocs.npz"))

    cs_logits = cs_data["logits"]
    cs_masks = cs_data["masks"]
    cs_candidates = cs_data["candidates"]
    labels = cs_data["labels"]
    is_pick = cs_data["is_pick"]
    
    # 【修复】：必须使用你生成的真实 bp_steps！
    cs_bp_steps = cs_data["bp_steps"]

    cs_time_weights = cs_data.get("time_weights", np.ones(len(labels), dtype=np.float32))

    nocs_logits = nocs_data["logits"]
    nocs_masks = nocs_data["masks"]

    _, _, vocab_size, special_tokens, champion_start_idx = load_champion_vocabulary(str(CHAMPION_VOCABULARY_JSON))
    pick_indices = np.where(is_pick > 0.5)[0]

    # 【修复 1】：从 bp_steps 重建 match_id，确保 GroupKFold 按比赛分组
    # 每场比赛的 bp_step 从 0 递增到 19，bp_step 回退即为新比赛边界
    match_ids_all = np.zeros(len(cs_bp_steps), dtype=np.int64)
    for i in range(1, len(cs_bp_steps)):
        if cs_bp_steps[i] <= cs_bp_steps[i - 1]:
            match_ids_all[i] = match_ids_all[i - 1] + 1
        else:
            match_ids_all[i] = match_ids_all[i - 1]

    X_list, y_list, group_list, weight_list, base_cs_list, match_id_list = [], [], [], [], [], []

    for i in pick_indices:
        label = int(labels[i])
        if label <= 0: continue

        cs_l_arr = cs_logits[i].copy()
        cs_l_arr[cs_masks[i] == 0] = -1e9
        nocs_l_arr = nocs_logits[i].copy()
        nocs_l_arr[nocs_masks[i] == 0] = -1e9

        target_step = int(cs_bp_steps[i])

        cs_sorted = np.argsort(-cs_l_arr)
        nocs_sorted = np.argsort(-nocs_l_arr)

        # 取 CS 的 Top 50 作为候选池
        cs_top_set = set(cs_sorted[:CS_TOP_K].tolist())
        candidate_set = cs_top_set
        
        has_positive = (label in candidate_set)
        if split == "train" and not has_positive:
            continue
        # 【修复】：对于 validation，如果 label 漏了，强行补进去
        elif split == "val" and not has_positive:
            candidate_set.add(label)

        cs_gf = _compute_group_features(cs_l_arr, cs_masks[i], champion_start_idx, vocab_size)
        nocs_gf = _compute_group_features(nocs_l_arr, nocs_masks[i], champion_start_idx, vocab_size)
        total_valid_in_group = int(cs_masks[i][champion_start_idx:].sum())

        group_cids = sorted(candidate_set)
        group_size = len(group_cids)

        group_cids_arr = np.array(group_cids, dtype=np.int64)
        cs_logits_group = cs_l_arr[group_cids_arr].astype(np.float64)
        cs_ranks_group = cs_gf["rank_map"][group_cids_arr].astype(np.float64)
        nocs_logits_group = nocs_l_arr[group_cids_arr].astype(np.float64)
        nocs_ranks_group = nocs_gf["rank_map"][group_cids_arr].astype(np.float64)
        cand_feats_group = cs_candidates[i, group_cids_arr].astype(np.float64)

        X_group = _build_feature_matrix_batch(
            cs_logits_group, cs_ranks_group, cs_gf,
            nocs_logits_group, nocs_ranks_group, nocs_gf,
            cand_feats_group, total_valid_in_group, group_size, target_step # 传入 target_step
        )

        X_list.append(X_group)
        is_positive_arr = (group_cids_arr == label).astype(np.int32)
        y_list.append(is_positive_arr)
        base_cs_list.append(cs_logits_group)

        weight_list.append(np.full(group_size, float(cs_time_weights[i]), dtype=np.float64))
        group_list.append(group_size)
        match_id_list.append(match_ids_all[i])

    X_all = np.concatenate(X_list, axis=0)
    # === 断言：特征维度必须与FEATURE_COLS严格对齐 ===
    assert X_all.shape[1] == len(FEATURE_COLS), \
        f"[{split}] 特征维度不匹配! X_all: {X_all.shape[1]}, FEATURE_COLS: {len(FEATURE_COLS)}"
    df = pd.DataFrame(X_all, columns=FEATURE_COLS)
    y = np.concatenate(y_list)
    groups = np.array(group_list)
    weights = np.concatenate(weight_list)
    base_cs = np.concatenate(base_cs_list)
    match_ids = np.array(match_id_list, dtype=np.int64)

    # === 数据异常检查 ===
    log.info(f"  [{split}] 数据提取完成，开始异常值检查...")
    check_array(f"{split}_features", X_all, log, context="特征矩阵")
    check_labels(f"{split}_labels", y, log, context="排序标签")
    check_groups(f"{split}_groups", groups, log, context="LightGBM group")
    check_array(f"{split}_weights", weights, log, context="样本权重")
    check_array(f"{split}_base_cs", base_cs, log, context="TF base logits")
    # 校验 group sum == 数据行数 (cascade pick 之前的 bug 就是这里不匹配)
    group_sum = int(groups.sum())
    if group_sum != len(y):
        log.error(f"  [{split}] 严重错误: group sum({group_sum}) != 数据行数({len(y)})!")
    else:
        log.info(f"  [{split}] group sum({group_sum}) == 数据行数({len(y)}) 校验通过")

    elapsed = time.time() - t0
    log.info(f"  {split} extraction done. Extracted {len(group_list)} queries "
             f"({len(np.unique(match_ids))} matches) in {elapsed:.1f}s")

    return df, y, groups, weights, base_cs, match_ids

def _evaluate_pick_at_k(final_scores, y_val, group_val, ks=(1, 3, 5, 10, 20)):
    hits = {k: 0 for k in ks}
    total_queries = len(group_val)
    start_idx = 0
    for g_size in group_val:
        end_idx = start_idx + g_size
        g_scores = final_scores[start_idx:end_idx]
        g_labels = y_val[start_idx:end_idx]
        sorted_idx = np.argsort(-g_scores)
        for k in ks:
            if 1 in g_labels[sorted_idx[:k]]:
                hits[k] += 1
        start_idx = end_idx
    return {k: hits[k] / total_queries * 100 for k in ks}

def _rank_normalize(scores, groups):
    normalized = np.zeros_like(scores, dtype=np.float64)
    start_idx = 0
    for g_size in groups:
        end_idx = start_idx + g_size
        ranks = np.empty(g_size, dtype=np.float64)
        ranks[np.argsort(-scores[start_idx:end_idx])] = np.linspace(1.0, 0.0, g_size)
        normalized[start_idx:end_idx] = ranks
        start_idx = end_idx
    return normalized

def train_pick_cascade(override_config=None):
    total_t0 = time.time()
    is_production = is_production_mode()
    log.info("=" * 70)
    log.info("  Cascade Pick: Unified Model with Phase-Aware Routing")
    log.info(f"  Mode: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
    log.info("=" * 70)

    # 合并外部传入的 HPO 超参数
    current_config = LGB_CONFIG.copy()
    if override_config is not None:
        current_config.update(override_config)
        log.info(f"  Overrides applied: {override_config}")

    # === 生产模式：加载开发模式记录的最优参数 ===
    if is_production:
        config = get_config("pick", "cascade")
        num_boost_round = get_production_num_boost_round(config)
        best_alpha = get_production_blend_alpha(config)
        log.info(f"  Production params loaded: num_boost_round={num_boost_round}, blend_alpha={best_alpha} (blend mode)")
        use_early_stopping = False
    else:
        num_boost_round = NUM_ROUND
        best_alpha = None
        use_early_stopping = True

    X_train_df, y_train, group_train, w_train, base_cs_train, match_ids_train = extract_pick_ranking_data("train")
    X_val_df, y_val, group_val, w_val, base_cs_val, match_ids_val = extract_pick_ranking_data("val")

    # 【修复 2】：严格分离生产模式与开发模式的数据缩放 (Scaler Fitting)
    scaler = StandardScaler()
    X_train_raw = X_train_df.values.astype(np.float32)
    X_val_raw = X_val_df.values.astype(np.float32)

    if is_production:
        # 生产模式：合并原始数据 -> 在 100% 数据上 Fit Scaler
        X_all_raw = np.vstack([X_train_raw, X_val_raw])
        X_all = scaler.fit_transform(X_all_raw)

        y_all = np.concatenate([y_train, y_val])
        group_all = np.concatenate([group_train, group_val])
        w_all = np.concatenate([w_train, w_val])
        log.info(f"  Full training data scaled: {X_all.shape[0]} samples, {len(group_all)} groups")
    else:
        # 开发模式：只在 85% 的 Train 数据上 Fit Scaler，防止信息泄露
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_all = X_train
        y_all = y_train
        group_all = group_train
        w_all = w_train

    if is_production:
        log.info(f"  Training single production model with {num_boost_round} rounds...")
        # Blend 模式: LGBM 独立训练（不使用 init_score），推理时 rank_normalize + alpha 混合
        train_ds = lgb.Dataset(X_all, y_all, group=group_all, weight=w_all)
        callbacks = [lgb.log_evaluation(100)]
        model = lgb.train(
            current_config, train_ds, num_boost_round=num_boost_round,
            callbacks=callbacks
        )

        # 为防止残留投毒，复制 5 份相同的单模型覆盖旧 fold_0 ~ fold_4
        # 在线推理时 5 个模型求平均，数学上等价于 1 个最新模型
        import copy as _copy
        fold_models = [_copy.deepcopy(model) for _ in range(N_FOLDS)]

        # Blend 模式下 OOF 预测仅用于记录，不做评估
        oof_pred = model.predict(X_all)
        best_iterations = [num_boost_round]
        log.info(f"  Production model trained: {num_boost_round} rounds (Duplicated x{N_FOLDS} to overwrite old folds)")
    else:
        # 开发模式：N-Fold CV + Early Stopping
        log.info(f"  Training Unified Cascade Model with {N_FOLDS}-Fold CV...")
        # 【修复 1】：使用 match_id 作为 GroupKFold 的 groups，确保同一场比赛的所有
        # pick step 要么全在训练集，要么全在验证集，杜绝数据泄露
        row_match_ids = np.repeat(match_ids_train, group_train)
        gkf = GroupKFold(n_splits=N_FOLDS)

        oof_pred = np.zeros(len(y_train), dtype=np.float64)
        fold_models = []
        best_iterations = []

        for fold_i, (t_idx, v_idx) in enumerate(gkf.split(X_train, y_train, row_match_ids)):
            # Blend 模式: LGBM 独立训练（不使用 init_score），推理时 rank_normalize + alpha 混合
            # group 必须按 query 位置取子集，而非用 match_id 值当索引
            t_match_ids = np.unique(row_match_ids[t_idx])
            t_query_mask = np.isin(match_ids_train, t_match_ids)
            v_match_ids = np.unique(row_match_ids[v_idx])
            v_query_mask = np.isin(match_ids_train, v_match_ids)
            train_ds = lgb.Dataset(
                X_train[t_idx], y_train[t_idx],
                group=group_train[t_query_mask],
                weight=w_train[t_idx],
            )
            val_ds = lgb.Dataset(
                X_train[v_idx], y_train[v_idx],
                group=group_train[v_query_mask],
                weight=w_train[v_idx],
                reference=train_ds
            )

            callbacks = [lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)] if use_early_stopping else [lgb.log_evaluation(0)]
            model = lgb.train(
                current_config, train_ds, num_boost_round=num_boost_round,
                valid_sets=[val_ds], callbacks=callbacks
            )
            # Blend 模式: OOF 预测仅返回 LGBM 分数，alpha 混合在评估阶段做
            oof_pred[v_idx] = model.predict(X_train[v_idx])
            fold_models.append(model)
            best_iterations.append(model.best_iteration if use_early_stopping else num_boost_round)
            log.info(f"    Fold {fold_i+1} best_iter={model.best_iteration if use_early_stopping else num_boost_round}")
            # === Fold 级别数据检查 ===
            check_array(f"fold{fold_i}_train_X", X_train[t_idx], log, context=f"Fold{fold_i}训练特征")
            check_labels(f"fold{fold_i}_train_y", y_train[t_idx], log, context=f"Fold{fold_i}训练标签")
            check_predictions(f"fold{fold_i}_oof_pred", oof_pred[v_idx], log, context=f"Fold{fold_i}OOF预测")

    importance = fold_models[0].feature_importance(importance_type="gain")
    top_indices = np.argsort(-importance)[:15]
    log.info(f"\n  Top-15 Feature Importance:")
    for rank, idx in enumerate(top_indices):
        log.info(f"    {rank+1}. {FEATURE_COLS[idx]}: {importance[idx]:.1f}")

    # 生产模式：全量训练无独立验证集，跳过验证评估
    if is_production:
        log.info(f"  Production mode: blend (alpha={best_alpha} from config)")
        final_metrics = {}
        base_metrics = {}
    else:
        # Blend 模式: LGBM 独立预测 → rank_normalize → alpha 混合
        val_preds = np.array([m.predict(X_val) for m in fold_models])
        lgb_val_pred = val_preds.mean(axis=0)

        # 搜索最优 alpha (步长 0.02)
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
        final_metrics = _evaluate_pick_at_k(val_final_pred, y_val, group_val)
        base_metrics = _evaluate_pick_at_k(base_cs_val, y_val, group_val)

        log.info(f"\n  {'Method':<40} {'P@1':>6} {'P@3':>6} {'P@5':>6} {'P@10':>7}")
        log.info(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
        log.info(f"  {'Transformer Base':<40} {base_metrics[1]:>5.2f}% {base_metrics[3]:>5.2f}% {base_metrics[5]:>5.2f}% {base_metrics[10]:>6.2f}%")
        log.info(f"  {'Unified Routed Cascade (LGB)':<40} {final_metrics[1]:>5.2f}% {final_metrics[3]:>5.2f}% {final_metrics[5]:>5.2f}% {final_metrics[10]:>6.2f}%")
        log.info(f"  (Blend mode: alpha={best_alpha:.2f}, LGBM independent + rank_normalize)")

    os.makedirs(CKPT_DIR, exist_ok=True)
    import pickle
    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    for fi, m in enumerate(fold_models):
        m.save_model(os.path.join(CKPT_DIR, f"fold_{fi}_model.txt"))

    # 将 scaler 系数序列化保存到配置文件，供推理时校验
    record_scaler_coefficients("pick", scaler, model_subtype="cascade")

    # 生产模式：记录实际使用的参数
    if is_production:
        record_production_params(
            "pick", "cascade",
            best_iteration=get_production_num_boost_round(get_config("pick", "cascade")),
            blend_alpha=best_alpha if best_alpha is not None else 0.0,
            num_boost_round=num_boost_round,
            train_samples=len(X_all),
        )

    with open(os.path.join(CKPT_DIR, "routing_config.json"), "w") as f:
        # Blend 模式: 推理时 rank_normalize + alpha 混合 (LGBM 独立预测 + TF base)
        json.dump({"mode": "unified_phase_aware", "blend_alpha": best_alpha,
                   "fusion_mode": "blend"}, f)

    # 保存最终指标（生产模式跳过）
    if not is_production:
        final_metrics_data = {
            "blend_alpha": best_alpha,
            "transformer_base": {
                "P@1": base_metrics[1],
                "P@3": base_metrics[3],
                "P@5": base_metrics[5],
                "P@10": base_metrics[10],
            },
            "cascade_final": {
                "P@1": final_metrics[1],
                "P@3": final_metrics[3],
                "P@5": final_metrics[5],
                "P@10": final_metrics[10],
            }
        }
        metrics_path = os.path.join(CKPT_DIR, "cascade_final_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics_data, f, indent=2)
        log.info(f"  Final metrics saved to {metrics_path}")
        ts = time.strftime("%Y%m%d_%H%M%S")
        metrics_archive_path = os.path.join(str(RECOMMENDATION_METRICS_DIR), f"pick_cascade_final_{ts}.json")
        archive_data = dict(final_metrics_data)
        archive_data["metadata"] = {
            "model_type": "bp_recommendation_pick",
            "mode": "training",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(metrics_archive_path, 'w') as f:
            json.dump(archive_data, f, indent=2)
        log.info(f"  Metrics archived to {metrics_archive_path}")

    log.info(f"\n  Saved {len(fold_models)} models and scaler to {CKPT_DIR}")
    log.info(f"  Total time: {time.time() - total_t0:.1f}s")

    # === 开发模式：保存最佳参数到配置文件 ===
    if not is_production:
        avg_best_iteration = int(np.mean(best_iterations)) if best_iterations else num_boost_round
        save_best_params(
            model_type="pick",
            best_iteration=avg_best_iteration,
            blend_alpha=best_alpha,
            best_metric=final_metrics[10],
            best_metric_name="P@10",
            model_subtype="cascade",
            architecture={
                "num_leaves": current_config.get("num_leaves"),
                "max_depth": current_config.get("max_depth"),
                "min_data_in_leaf": current_config.get("min_data_in_leaf"),
            },
            optimizer={
                "learning_rate": current_config.get("learning_rate"),
            },
            loss={
                "objective": current_config.get("objective"),
                "metric": current_config.get("metric"),
            },
            training={
                "n_folds": N_FOLDS,
                "num_boost_round": num_boost_round,
                "early_stop": EARLY_STOP,
                "seed": current_config.get("seed"),
            },
        )
        log.info(f"  Best params saved: avg_best_iteration={avg_best_iteration}, blend_alpha={best_alpha}")

    return final_metrics.get(10, 0.0)

if __name__ == "__main__":
    from pathlib import Path
    from logger_config import setup_logging
    setup_logging(log_dir=Path(LOG_DIR))
    
    import sys
    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _file_handler = logging.FileHandler(os.path.join(LOG_DIR, f"cascade_pick_{_run_ts}.log"), encoding="utf-8")
    _file_handler.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _file_handler.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_file_handler)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true", default=None,
                        help="强制启用生产模式")
    args, _ = parser.parse_known_args()
    if args.production is not None:
        os.environ["BP_PRODUCTION_MODE"] = "true" if args.production else "false"
    
    train_pick_cascade()