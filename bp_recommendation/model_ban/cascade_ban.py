"""
Ban Cascade LightGBM 级联模型训练
=============================================
使用 LightGBM LambdaMART 对 Transformer 模型输出进行级联融合，
通过 Transformer logit + 手工特征构建更精准的 Ban 排序模型。

功能描述:
    - 从训练好的 Transformer 模型提取 OOF 预测分数
    - 构建 Transformer 输出 + 原始统计特征的融合特征矩阵
    - 使用 5-Fold GroupKFold 交叉验证训练 LightGBM Ranker
    - 支持 StandardScaler 特征标准化
    - 计算 Ban@K 评估指标
    - 保存融合模型和 blend_alpha 权重到配置文件

主要函数/常量:
    - FEATURE_COLS: Cascade 模型特征列定义
    - extract_ban_ranking_data(): 提取训练数据
    - _build_feature_matrix_batch(): 批量构建特征矩阵
    - _compute_group_features(): 计算组级特征
    - train_cascade_ban(): 训练 Cascade Ban 模型
    - _evaluate_ban_at_k(): 评估 Ban@K 指标

使用方法:
    cd /Users/siwentu/Desktop/LOL analysis
    python -m bp_recommendation.model_ban.cascade_ban
    
    注意: 需要先训练好 Ban Transformer 模型并生成 OOF 预测。
"""
import os
import sys
import time
import json
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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
from bp_recommendation.feature_pipeline import CANDIDATE_FEAT_MAP

# 共享数据异常检测工具
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(TEST_DIR))))
from data_checks import check_array, check_labels, check_groups, check_predictions
from logger_config import get_logger
from common.paths import RECOMMENDATION_METRICS_DIR, ensure_dirs as _ensure_common_dirs
_ensure_common_dirs()

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log = get_logger(__name__)

FEATURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "cascade_ban")
TOP_K_RECALL = 50
N_FOLDS = 5

# ============================================================
# 超参数: Optuna TPE 搜索最佳 (Trial 28, Blend B@10=82.03%)
# ============================================================
LGB_CONFIG = {
    "objective": "rank_xendcg", 
    "metric": "ndcg", 
    "ndcg_at": [5, 10, 20],
    "num_leaves": 53,
    "max_depth": 9,
    "min_data_in_leaf": 5,
    "learning_rate": 0.030418449585900214,
    "feature_fraction": 0.37275611567529676,
    "bagging_fraction": 0.7084019175123303,
    "bagging_freq": 4,
    "lambda_l1": 0.020293994280514982, 
    "lambda_l2": 0.5093134683763993,
    "verbose": -1, 
    "seed": 42,
}
NUM_ROUND = 3000
EARLY_STOP = 150


# ================= 精简且具有破坏性的战术特征 =================
FEATURE_COLS = [
    "transformer_logit", "logit_percentile", "logit_gap_top1",
    
    # 全局基础属性
    "meta_ban_rate", "meta_presence", 
    "enemy_mastery_max", "grudge", "respect", "hot_streak",
    "player_recent_games",
    
    # 【核心】：阶段感知特征 (Phase-Aware Routing)
    "p1_meta_ban", "p1_threat_respect", "p1_threat_hot_streak",
    "p2_choke_meta", "p2_choke_mastery", "p2_grudge", "p2_enemy_mastery",
    
    # 破坏性复合杀招
    "threat_meta_mastery",   # 版本强势 + 对面恰好是绝活
    "choke_composite",       # 综合卡脖子指数
]

def _compute_group_features(sample_logits, sample_mask, champion_start_idx, vocab_size):
    valid_indices = np.where(sample_mask > 0.5)[0]
    valid_champ_indices = valid_indices[valid_indices >= champion_start_idx]

    sorted_indices = np.argsort(-sample_logits)
    rank_map = np.zeros(vocab_size, dtype=np.float32)
    for rank_pos, idx in enumerate(sorted_indices):
        rank_map[idx] = rank_pos + 1

    if len(valid_champ_indices) > 0:
        valid_logits = sample_logits[valid_champ_indices]
        valid_mean = valid_logits.mean()
        valid_std = max(valid_logits.std(), 1e-6)
    else:
        valid_mean, valid_std = 0, 1.0

    top1_logit = sample_logits[sorted_indices[0]] if len(sorted_indices) > 0 else 0

    return {
        "rank_map": rank_map,
        "valid_mean": valid_mean,
        "valid_std": valid_std,
        "top1_logit": top1_logit,
    }

def _build_feature_matrix_batch(logits_arr, rank_map, gf, cand_feats, total_valid):
    # 【修复】：使用全局特征映射
    # rank_map: 每个候选英雄的排名（已切片到 eval_cids），预留用于扩展特征
    # total_valid: 当前步可用英雄总数，预留用于量纲对齐
    FI = CANDIDATE_FEAT_MAP

    meta_ban = cand_feats[:, FI["meta_ban"]]
    meta_presence = cand_feats[:, FI["meta_presence"]]
    meta_wr = cand_feats[:, FI["meta_wr"]]
    player_recent_games = cand_feats[:, FI["player_recent_games"]]
    
    # 注意：下面这两个 key 必须确保在 CANDIDATE_FEAT_MAP 里存在！
    role_fit_enemy = cand_feats[:, FI["enemy_role_fit"]]
    enemy_mastery_max = cand_feats[:, FI["enemy_mastery_max"]]
    
    ban_step = cand_feats[:, FI["ban_step"]]
    grudge = cand_feats[:, FI["grudge"]]
    respect = cand_feats[:, FI["respect"]]
    hot_streak = cand_feats[:, FI["hot_streak"]]

    logit_percentile = 0.5 * (1.0 + np.tanh((logits_arr - gf["valid_mean"]) / max(gf["valid_std"], 1e-6) * 0.5))
    logit_gap_top1 = logits_arr - gf["top1_logit"]

    # ---------------- 核心修复：严谨的 Phase 切分 ----------------
    # Ban_step = 1, 2, 3 是 Phase 1。 4, 5 是 Phase 2。
    is_p1 = (ban_step <= 3).astype(np.float32)
    is_p2 = 1.0 - is_p1

    # P1 看重大盘和版本大热绝活
    p1_meta_ban = meta_ban * is_p1
    p1_threat_respect = respect * enemy_mastery_max * is_p1
    p1_threat_hot_streak = hot_streak * enemy_mastery_max * is_p1

    # P2 看重针对性封锁和战术卡位
    choke_meta = role_fit_enemy * meta_wr
    choke_mastery = role_fit_enemy * enemy_mastery_max
    
    p2_choke_meta = choke_meta * is_p2
    p2_choke_mastery = choke_mastery * is_p2
    p2_grudge = grudge * is_p2
    p2_enemy_mastery = enemy_mastery_max * is_p2

    # 破坏性复合特征
    threat_meta_mastery = meta_ban * enemy_mastery_max
    choke_composite = (choke_meta + choke_mastery) * is_p2

    X = np.column_stack([
        logits_arr, logit_percentile, logit_gap_top1,
        meta_ban, meta_presence,
        enemy_mastery_max, grudge, respect, hot_streak,
        player_recent_games,
        
        p1_meta_ban, p1_threat_respect, p1_threat_hot_streak,
        p2_choke_meta, p2_choke_mastery, p2_grudge, p2_enemy_mastery,
        
        threat_meta_mastery, choke_composite
    ]).astype(np.float32)

    return X

def extract_ban_ranking_data(split="val", top_k=TOP_K_RECALL):
    t0 = time.time()
    log.info(f"Loading {split} logits and features...")
    data = np.load(os.path.join(FEATURES_DIR, f"ALL_{split}_logits_cs.npz"))

    logits = data["logits"]
    masks = data["masks"]
    candidates = data["candidates"]
    labels = data["labels"]
    is_pick = data["is_pick"]
    time_weights = data.get("time_weights", np.ones(len(labels), dtype=np.float32))
    bp_steps = data["bp_steps"]

    champion_start_idx = 1
    vocab_size = logits.shape[1]
    ban_indices = np.where(is_pick < 0.5)[0]

    # 【修复 1】：从 bp_steps 重建 match_id，确保 GroupKFold 按比赛分组
    match_ids_all = np.zeros(len(bp_steps), dtype=np.int64)
    for i in range(1, len(bp_steps)):
        if bp_steps[i] <= bp_steps[i - 1]:
            match_ids_all[i] = match_ids_all[i - 1] + 1
        else:
            match_ids_all[i] = match_ids_all[i - 1]

    X_list, y_list, group_list, base_logits_list, weight_list, match_id_list = [], [], [], [], [], []
    total_valid = 0

    for i in ban_indices:
        label = int(labels[i])
        if label <= 0: continue

        l_arr = logits[i].copy()
        l_arr[masks[i] == 0] = -1e9

        sorted_indices = np.argsort(-l_arr)
        top_k_indices = sorted_indices[:top_k]

        has_positive = (label in top_k_indices)
        if split == "train" and not has_positive:
            continue
        # 【修复】：如果验证集的召回漏了正确答案，强行将其挂在队尾
        elif split == "val" and not has_positive:
            # np.append 会返回一个新的数组，不影响原有数组
            top_k_indices = np.append(top_k_indices, label)

        # 【修复 group/weight 尺寸不匹配】：追加 label 后实际候选数为 len(top_k_indices)
        # group_list 和 weight_list 必须使用实际长度，否则 group sum != features 行数
        actual_k = len(top_k_indices)
        total_valid += 1

        gf = _compute_group_features(l_arr, masks[i], champion_start_idx, vocab_size)
        total_valid_in_group = int(masks[i][champion_start_idx:].sum())

        cand_feats_group = candidates[i, top_k_indices]
        logits_group = l_arr[top_k_indices]
        ranks_group = gf["rank_map"][top_k_indices]

        X_group = _build_feature_matrix_batch(
            logits_group, ranks_group, gf, cand_feats_group, total_valid_in_group
        )

        X_list.append(X_group)
        y_list.append((top_k_indices == label).astype(np.int32))
        base_logits_list.append(logits_group)
        group_list.append(actual_k)
        weight_list.append(np.full(actual_k, float(time_weights[i]), dtype=np.float64))
        match_id_list.append(match_ids_all[i])

    X_all = np.concatenate(X_list, axis=0)
    # === 断言：特征维度必须与FEATURE_COLS严格对齐 ===
    assert X_all.shape[1] == len(FEATURE_COLS), \
        f"[{split}] [Ban] 特征维度不匹配! X_all: {X_all.shape[1]}, FEATURE_COLS: {len(FEATURE_COLS)}"
    df = pd.DataFrame(X_all, columns=FEATURE_COLS)
    y = np.concatenate(y_list)
    groups = np.array(group_list)
    base_logits = np.concatenate(base_logits_list)
    weights = np.concatenate(weight_list)
    match_ids = np.array(match_id_list, dtype=np.int64)

    # === 数据异常检查 ===
    log.info(f"  [{split}] 数据提取完成，开始异常值检查...")
    check_array(f"{split}_features", df.values, log, context="特征矩阵")
    check_labels(f"{split}_labels", y, log, context="排序标签")
    check_groups(f"{split}_groups", groups, log, context="LightGBM group")
    check_array(f"{split}_weights", weights, log, context="样本权重")
    check_array(f"{split}_base_logits", base_logits, log, context="TF base logits")
    # 校验 group sum == 数据行数
    group_sum = int(groups.sum())
    if group_sum != len(y):
        log.error(f"  [{split}] 严重错误: group sum({group_sum}) != 数据行数({len(y)})!")
    else:
        log.info(f"  [{split}] group sum({group_sum}) == 数据行数({len(y)}) 校验通过")

    elapsed = time.time() - t0
    log.info(f"  {split} extraction done. Total Queries: {total_valid} "
             f"({len(np.unique(match_ids))} matches) in {elapsed:.1f}s")
    return df, y, groups, base_logits, weights, match_ids

def _evaluate_ban_at_k(final_scores, y_val, group_val, ks=(5, 10, 20)):
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

def train_ban_cascade(override_config=None):
    total_t0 = time.time()
    # 【修复 1】：添加括号 ()，正确调用动态判断函数
    is_production = is_production_mode() 
    
    log.info("=" * 70)
    log.info("  Cascade Ban: Unified Model with Phase-Aware Choking Routing")
    log.info(f"  Mode: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
    log.info("=" * 70)

    current_config = LGB_CONFIG.copy()
    if override_config is not None:
        current_config.update(override_config)
        log.info(f"  Overrides applied: {override_config}")

    # === 生产模式：加载开发模式记录的最优参数 ===
    if is_production:
        config = get_config("ban", "cascade")
        num_boost_round = get_production_num_boost_round(config)
        best_alpha = get_production_blend_alpha(config)
        log.info(f"  Production params loaded: num_boost_round={num_boost_round}, blend_alpha={best_alpha:.4f}")
        use_early_stopping = False
    else:
        num_boost_round = NUM_ROUND
        best_alpha = None
        use_early_stopping = True

    X_train_df, y_train, group_train, base_train, w_train, match_ids_train = extract_ban_ranking_data("train")
    X_val_df, y_val, group_val, base_val, w_val, match_ids_val = extract_ban_ranking_data("val")

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
        # 生产模式：全量数据单模型训练，无 Early Stopping
        log.info(f"  Training single production model with {num_boost_round} rounds...")
        train_ds = lgb.Dataset(X_all, y_all, group=group_all, weight=w_all)
        callbacks = [lgb.log_evaluation(100)]
        model = lgb.train(
            current_config, train_ds, num_boost_round=num_boost_round,
            callbacks=callbacks
        )
        
        # 【修复 3】：复制 5 份相同的单模型！
        # 完美覆盖硬盘上的旧模型，防止在线推理时被"残留投毒"
        # 使用深拷贝确保每个 fold 独立，避免后续修改影响所有 fold
        import copy as _copy
        fold_models = [_copy.deepcopy(model) for _ in range(N_FOLDS)]
        
        oof_pred = model.predict(X_all)
        best_iterations = [num_boost_round]
        log.info(f"  Production model trained: {num_boost_round} rounds (Duplicated x{N_FOLDS} to overwrite old folds)")
    else:
        # 开发模式：N-Fold CV + Early Stopping
        log.info(f"  Training Unified Ban Cascade Model with {N_FOLDS}-Fold CV...")
        # 【修复 1】：使用 match_id 作为 GroupKFold 的 groups，确保同一场比赛的所有
        # ban step 要么全在训练集，要么全在验证集，杜绝数据泄露
        row_match_ids = np.repeat(match_ids_train, group_train)
        gkf = GroupKFold(n_splits=N_FOLDS)

        oof_pred = np.zeros(len(y_train), dtype=np.float64)
        fold_models = []
        best_iterations = []

        for fold_i, (t_idx, v_idx) in enumerate(gkf.split(X_train, y_train, row_match_ids)):
            # 【修复 4】：group 必须按 query 位置取子集，而非用 match_id 值当索引。
            # np.unique(row_match_ids[t_idx]) 返回的是 match_id 值，不能直接作为
            # group_train 的位置索引（group_train 长度 = query 数，非 match 数）。
            # 正确做法：用 match_id 反查 query 位置掩码，再取 group_train[掩码]。
            t_match_ids = np.unique(row_match_ids[t_idx])
            t_query_mask = np.isin(match_ids_train, t_match_ids)
            v_match_ids = np.unique(row_match_ids[v_idx])
            v_query_mask = np.isin(match_ids_train, v_match_ids)
            train_ds = lgb.Dataset(
                X_train[t_idx], y_train[t_idx],
                group=group_train[t_query_mask],
                weight=w_train[t_idx]
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
            oof_pred[v_idx] = model.predict(X_train[v_idx])
            fold_models.append(model)
            best_iterations.append(model.best_iteration if use_early_stopping else num_boost_round)
            log.info(f"    Fold {fold_i+1} best_iter={model.best_iteration if use_early_stopping else num_boost_round}")
            # === Fold 级别数据检查 ===
            check_array(f"fold{fold_i}_train_X", X_train[t_idx], log, context=f"Fold{fold_i}训练特征")
            check_labels(f"fold{fold_i}_train_y", y_train[t_idx], log, context=f"Fold{fold_i}训练标签")
            check_predictions(f"fold{fold_i}_oof_pred", oof_pred[v_idx], log, context=f"Fold{fold_i}OOF预测")

    importance = fold_models[0].feature_importance(importance_type="gain")
    top_indices = np.argsort(-importance)[:10]
    log.info(f"\n  Top-10 Feature Importance:")
    for rank, idx in enumerate(top_indices):
        log.info(f"    {rank+1}. {FEATURE_COLS[idx]}: {importance[idx]:.1f}")

    # 生产模式：全量训练无独立验证集，跳过验证评估
    if is_production:
        log.info(f"  Using production blend_alpha={best_alpha:.4f}")
        final_metrics = {}
        base_metrics = {}
    else:
        val_preds = np.array([m.predict(X_val) for m in fold_models])
        val_final_pred = val_preds.mean(axis=0)

        cs_val_rn = _rank_normalize(base_val, group_val)
        lgb_val_rn = _rank_normalize(val_final_pred, group_val)

        best_alpha_dev, best_b10 = 0.0, 0.0
        for alpha_int in range(5, 101, 5):
            alpha = alpha_int / 100.0
            blend_scores = alpha * cs_val_rn + (1 - alpha) * lgb_val_rn
            b10 = _evaluate_ban_at_k(blend_scores, y_val, group_val)[10]
            if b10 > best_b10:
                best_b10 = b10
                best_alpha_dev = alpha
        best_alpha = best_alpha_dev
        best_blend_scores = best_alpha * cs_val_rn + (1 - best_alpha) * lgb_val_rn
        final_metrics = _evaluate_ban_at_k(best_blend_scores, y_val, group_val)

        base_metrics = _evaluate_ban_at_k(base_val, y_val, group_val)

        log.info(f"\n  {'Method':<40} {'B@5':>6} {'B@10':>7} {'B@20':>7}")
        log.info(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*7}")
        log.info(f"  {'Transformer Base':<40} {base_metrics[5]:>5.2f}% {base_metrics[10]:>6.2f}% {base_metrics[20]:>6.2f}%")
        log.info(f"  {'Unified Routed Cascade (LGB)':<40} {final_metrics[5]:>5.2f}% {final_metrics[10]:>6.2f}% {final_metrics[20]:>6.2f}%")
        log.info(f"  (Blend Alpha: {best_alpha:.4f})")

    os.makedirs(CKPT_DIR, exist_ok=True)
    for fi, m in enumerate(fold_models):
        m.save_model(os.path.join(CKPT_DIR, f"fold_{fi}_model.txt"))

    with open(os.path.join(CKPT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # 将 scaler 系数序列化保存到配置文件，供推理时校验
    record_scaler_coefficients("ban", scaler, model_subtype="cascade")

    # 生产模式：记录实际使用的参数
    if is_production:
        record_production_params(
            "ban", "cascade",
            best_iteration=get_production_num_boost_round(get_config("ban", "cascade")),
            blend_alpha=best_alpha,
            num_boost_round=num_boost_round,
            train_samples=len(X_all),
        )

    with open(os.path.join(CKPT_DIR, "routing_config.json"), "w") as f:
        json.dump({"mode": "unified_ban_phase_aware", "blend_alpha": best_alpha}, f)

    # 保存最终指标（生产模式跳过）
    if not is_production:
        final_metrics_data = {
            "blend_alpha": best_alpha,
            "transformer_base": {
                "B@5": base_metrics[5],
                "B@10": base_metrics[10],
                "B@20": base_metrics[20],
            },
            "cascade_final": {
                "B@5": final_metrics[5],
                "B@10": final_metrics[10],
                "B@20": final_metrics[20],
            }
        }
        metrics_path = os.path.join(CKPT_DIR, "cascade_final_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics_data, f, indent=2)
        log.info(f"  Final metrics saved to {metrics_path}")
        ts = time.strftime("%Y%m%d_%H%M%S")
        metrics_archive_path = os.path.join(str(RECOMMENDATION_METRICS_DIR), f"ban_cascade_final_{ts}.json")
        archive_data = dict(final_metrics_data)
        archive_data["metadata"] = {
            "model_type": "bp_recommendation_ban",
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
            model_type="ban",
            best_iteration=avg_best_iteration,
            blend_alpha=best_alpha,
            best_metric=final_metrics[10],
            best_metric_name="B@10",
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
        log.info(f"  Best params saved: avg_best_iteration={avg_best_iteration}, blend_alpha={best_alpha:.4f}")

    return final_metrics.get(10, 0.0)

if __name__ == "__main__":
    from pathlib import Path
    from logger_config import setup_logging
    setup_logging(log_dir=Path(LOG_DIR))
    
    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _file_handler = logging.FileHandler(os.path.join(LOG_DIR, f"cascade_ban_{_run_ts}.log"), encoding="utf-8")
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
    
    train_ban_cascade()