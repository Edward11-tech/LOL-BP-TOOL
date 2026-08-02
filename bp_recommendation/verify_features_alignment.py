#!/usr/bin/env python3
"""
线上线下特征一致性验证工具
=============================================
验证训练时（dataloader）构建的特征与线上推理时（bp_predict）构建的特征是否完全一致，
确保训练-推理特征对齐，避免训练服务偏差（Training-Serving Skew）。

功能描述:
    - 逐列对比 candidate_matrix 特征
    - 对比 global_context 全局上下文向量
    - 对比 player_matrix 选手特征矩阵
    - 支持模糊匹配（允许时间快照导致的微小差异）
    - 输出详细的不一致报告

主要函数:
    - compare_matrix(): 逐列对比两个矩阵
    - _build_online_ctx(): 构建线上风格的全局上下文
    - _parse_pre_unavail(): 解析前置局已用英雄（Fearless Draft）

使用方法:
    cd /Users/siwentu/Desktop/LOL analysis
    python -m bp_recommendation.verify_features_alignment
    
    用于模型上线前的特征一致性校验，确保训练和推理使用相同的特征计算逻辑。
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logger_config import get_logger, setup_logging

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log = get_logger(__name__)

from bp_recommendation.model_pick.dataloader_pick import create_train_val_dataloaders
from bp_recommendation.feature_pipeline import BP_SEQUENCE, CANDIDATE_FEAT_MAP, CANDIDATE_DIM

FEATURES_DIR = os.path.join(BASE_DIR, "bp_recommendation", "features")
CTX_PARQUET = os.path.join(FEATURES_DIR, "ALL_context.parquet")
META_PARQUET = os.path.join(FEATURES_DIR, "ALL_meta_store.parquet")
PLAYER_PARQUET = os.path.join(FEATURES_DIR, "ALL_player_store.parquet")
VOCAB_PATH = os.path.join(BASE_DIR, "cleaned_data", "champion_vocabulary.json")
POS_JSON = os.path.join(BASE_DIR, "cleaned_data", "champion_position_mapping.json")
LEAGUES = ["LPL", "LCK", "LEC"]

CTX_COL_NAMES = [
    "league_LPL", "league_LCK", "league_LEC",
    "blue_team_avg_ckpm", "blue_team_avg_golddiffat15",
    "blue_team_avg_gamelength", "blue_team_firstdragon_rate", "blue_team_firsttower_rate",
    "red_team_avg_ckpm", "red_team_avg_golddiffat15",
    "red_team_avg_gamelength", "red_team_firstdragon_rate", "red_team_firsttower_rate",
    "playoffs", "first_pick_map_side",
    "is_game_1", "is_game_2", "is_game_3", "is_game_4", "is_game_5",
]
# 动态生成候选列名用于打印
CAND_COL_NAMES = [""] * len(CANDIDATE_FEAT_MAP)
for name, idx in CANDIDATE_FEAT_MAP.items():
    if idx < len(CAND_COL_NAMES): CAND_COL_NAMES[idx] = name

def compare_matrix(name, off_mat, on_mat, col_names, fuzzy_cols=None, fuzzy_threshold=0.1):
    fuzzy_cols = fuzzy_cols or []
    log.info(f"\n{'='*60}\n  {name} column-by-column comparison\n{'='*60}")
    mismatches = []
    for col in range(off_mat.shape[1]):
        col_name = col_names[col] if col < len(col_names) else f"col_{col}"
        diff = np.abs(off_mat[:, col] - on_mat[:, col]).max()
        is_fuzzy = col in fuzzy_cols
        threshold = fuzzy_threshold if is_fuzzy else 1e-4

        if diff > threshold:
            off_mean = off_mat[:, col].mean()
            on_mean = on_mat[:, col].mean()
            status = "FUZZY" if is_fuzzy else "MISMATCH"
            log.warning(f"  [{col:2d}] {col_name:30s}  off={off_mean:10.4f} vs on={on_mean:10.4f}  max_diff={diff:.6f}  {status}")
            # FUZZY 不匹配（时间快照差异等）只记录警告，不计入失败
            if not is_fuzzy:
                mismatches.append((col, col_name, diff, off_mean, on_mean))
        else:
            log.info(f"  [{col:2d}] {col_name:30s}  MATCH (max_diff={diff:.6f})")
    return len(mismatches) == 0, mismatches

def _parse_pre_unavail(row, store):
    """从历史比赛数据的 prev_game_champs 字段解析前置局已用英雄 (Fearless Draft)。

    与训练时 dataloader 的逻辑一致：prev_game_champs 为 '|' 分隔的英雄名。
    """
    cs = store.champion_start_idx
    pre_unavail_list = []
    prev_champs_str = str(row.get("prev_game_champs", ""))
    if prev_champs_str and prev_champs_str.strip() and prev_champs_str != "nan":
        for champ_name in prev_champs_str.split("|"):
            champ_name = champ_name.strip()
            if champ_name:
                cid = store.name_to_idx.get(champ_name, -1)
                if cid >= cs:
                    pre_unavail_list.append(cid)
    return pre_unavail_list

def _build_online_ctx(store, target_match_row):
    """构建线上 global_context 向量，与训练 dataloader 的 20 维上下文对齐。"""
    league = target_match_row.get("league", "LPL")
    blue_team = target_match_row.get("blue_team", "")
    red_team = target_match_row.get("red_team", "")
    playoffs_f = 1.0 if target_match_row.get("playoffs", 0) else 0.0
    first_pick_f = float(target_match_row.get("first_pick_map_side", 1.0))
    game_num = next((i for i in range(1, 6) if target_match_row.get(f"is_game_{i}", 0) == 1), 1)

    league_vec = np.zeros(len(LEAGUES), dtype=np.float32)
    if league in LEAGUES: league_vec[LEAGUES.index(league)] = 1.0
    b_style = store.team_style_dict.get(blue_team, [0.7, 0.0, 1900.0, 0.5, 0.5])
    r_style = store.team_style_dict.get(red_team, [0.7, 0.0, 1900.0, 0.5, 0.5])
    game_num_vec = np.zeros(5, dtype=np.float32)
    if 1 <= game_num <= 5: game_num_vec[game_num - 1] = 1.0
    return np.concatenate([league_vec, b_style, r_style, [playoffs_f, first_pick_f], game_num_vec]).astype(np.float32)


def _safe_get_pid(row, pos_key):
    val = row.get(pos_key, "")
    return str(val) if pd.notna(val) and str(val).lower() != "nan" else "unknown"


def _verify_one_sample(offline_dataset, store, idx, action_filter=None):
    """对单个样本执行离线 vs 在线特征对齐检查。

    Args:
        offline_dataset: 离线 dataloader 的 dataset
        store: 在线 PredictFeatureStore
        idx: sample_list 中的索引
        action_filter: "pick" / "ban" / None，用于限定动作类型

    Returns:
        dict: {"ctx_ok": bool, "cand_ok": bool, "ctx_miss": list, "cand_miss": list}
    """
    match_idx, step = offline_dataset.sample_list[idx]
    action = BP_SEQUENCE[step][0]
    if action_filter and action != action_filter:
        return None

    target_match_row = offline_dataset.context[match_idx]
    offline_sample = offline_dataset[idx]
    off_ctx = offline_sample["global_context"].numpy()
    off_cand = offline_sample["candidate_matrix"].numpy()

    side = BP_SEQUENCE[step][1]
    bp_seq_ids = [int(target_match_row.get(f"bp_step{i}_champion_id", 0)) for i in range(step)]

    on_ally, on_enemy, last_ally_pos = [], [], -1
    for i, cid in enumerate(bp_seq_ids):
        if cid < store.champion_start_idx: continue
        if BP_SEQUENCE[i][0] == "pick":
            if (BP_SEQUENCE[i][1] == "blue" and side == "blue") or (BP_SEQUENCE[i][1] == "red" and side == "red"):
                on_ally.append(cid)
                last_ally_pos = i
            else: on_enemy.append(cid)

    blue_pids = [_safe_get_pid(target_match_row, f"blue_{pos}_player_id") for pos in ["top", "jng", "mid", "bot", "sup"]]
    red_pids = [_safe_get_pid(target_match_row, f"red_{pos}_player_id") for pos in ["top", "jng", "mid", "bot", "sup"]]
    on_ally_pids = blue_pids if side == "blue" else red_pids
    on_enemy_pids = red_pids if side == "blue" else blue_pids

    pre_unavail = _parse_pre_unavail(target_match_row, store)
    unavail_set = set(bp_seq_ids) | set(pre_unavail)
    blue_team = target_match_row.get("blue_team", "")
    red_team = target_match_row.get("red_team", "")

    on_ctx = _build_online_ctx(store, target_match_row)

    if action == "pick":
        on_cand, on_mask = store.get_pick_candidate_matrix(
            side, on_ally, on_enemy, unavail_set, on_ally_pids, on_enemy_pids, step,
            blue_team if side == "blue" else red_team,
            red_team if side == "blue" else blue_team,
            pre_unavail_list=pre_unavail,
        )
    else:
        on_cand, on_mask = store.get_ban_candidate_matrix(
            side, on_ally, on_enemy, unavail_set, on_ally_pids, on_enemy_pids, step,
            blue_team if side == "blue" else red_team,
            red_team if side == "blue" else blue_team,
            pre_unavail_list=pre_unavail,
        )

    ctx_ok, ctx_miss = compare_matrix(f"Global Context (20d) [{action} step={step}]",
                                       off_ctx.reshape(1, -1), on_ctx.reshape(1, -1), CTX_COL_NAMES,
                                       fuzzy_cols=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12], fuzzy_threshold=50.0)

    # Ban 模型训练时 use_extended_features=False，last_ally_synergy@idx30 恒为 0
    # 验证脚本使用 pick dataloader 构建离线特征（包含 last_ally_synergy），
    # 因此对 ban 步骤需手动置零以匹配 ban 模型的训练行为
    if action == "ban":
        off_cand[:, CANDIDATE_FEAT_MAP["last_ally_synergy"]] = 0.0

    # fuzzy_cols: 16-19=synergy/counter (Bayesian 平滑差异), 0-3=meta (PIT 快照时间差异)
    cand_ok, cand_miss = compare_matrix(f"Candidate Matrix ({CANDIDATE_DIM}d) [{action} step={step}]",
                                         off_cand, on_cand, CAND_COL_NAMES,
                                         fuzzy_cols=[0, 1, 2, 3, 16, 17, 18, 19], fuzzy_threshold=0.1)

    return {"ctx_ok": ctx_ok, "cand_ok": cand_ok, "ctx_miss": ctx_miss, "cand_miss": cand_miss,
            "gameid": target_match_row.get("gameid"), "step": step, "action": action}


def verify_alignment():
    log.info("=" * 70 + "\n  Feature Alignment Check: Offline vs Online (Stateless)\n" + "=" * 70)

    log.info("\n[1/3] Initializing offline DataLoader...")
    train_loader, _ = create_train_val_dataloaders(
        CTX_PARQUET, META_PARQUET, PLAYER_PARQUET, VOCAB_PATH, POS_JSON,
        batch_size=1, num_workers=0, val_ratio=0.0, force_unroll_train=True
    )
    offline_dataset = train_loader.dataset

    # 取训练集最后一天的全部比赛样本，接近线上最新快照信息
    # context 是 list of dict (to_dict("records")), 需转回 DataFrame
    context_df = pd.DataFrame(offline_dataset.context)
    all_dates = pd.to_datetime(context_df["date"], errors="coerce")
    max_date = all_dates.max()
    last_day_mask = all_dates == max_date
    last_day_indices = [i for i, (match_idx, step) in enumerate(offline_dataset.sample_list)
                        if last_day_mask.iloc[match_idx]]
    log.info(f"  Training set last day: {max_date.date()}, samples found: {len(last_day_indices)}")
    if not last_day_indices:
        return log.error("No samples found for the last training day!")

    log.info("\n[2/3] Initializing online PredictFeatureStore...")
    from bp_recommendation.bp_predict import BPRecommender
    recommender = BPRecommender()
    store = recommender.store

    log.info("\n[3/3] Comparing offline vs online features for all last-day samples...")
    total = 0
    pick_total = pick_pass = 0
    ban_total = ban_pass = 0
    all_ctx_miss = []
    all_cand_miss = []

    for idx in last_day_indices:
        result = _verify_one_sample(offline_dataset, store, idx)
        if result is None:
            continue
        total += 1
        ok = result["ctx_ok"] and result["cand_ok"]
        if result["action"] == "pick":
            pick_total += 1
            if ok: pick_pass += 1
        else:
            ban_total += 1
            if ok: ban_pass += 1
        if not result["ctx_ok"]: all_ctx_miss.extend(result["ctx_miss"])
        if not result["cand_ok"]: all_cand_miss.extend(result["cand_miss"])
        log.info(f"  [{'PASS' if ok else 'FAIL'}] {result['action']} step={result['step']} gameid={result['gameid']}")

    log.info("\n" + "=" * 70)
    log.info(f"  Summary: {total} samples checked (last day: {max_date.date()})")
    log.info(f"  Pick: {pick_pass}/{pick_total} passed")
    log.info(f"  Ban:  {ban_pass}/{ban_total} passed")
    log.info(f"  Total context mismatches: {len(all_ctx_miss)}")
    log.info(f"  Total candidate mismatches: {len(all_cand_miss)}")
    if pick_pass == pick_total and ban_pass == ban_total:
        log.info("  ALL MATCHED! No Online-Offline Skew detected.")
    else:
        log.warning("  Skew detected. Please review mismatches above.")

if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))
    
    _run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _run_log_path = os.path.join(LOG_DIR, f"verify_features_alignment_{_run_ts}.log")
    _run_fh = logging.FileHandler(_run_log_path, encoding="utf-8")
    _run_fh.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _run_fh.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_run_fh)
    
    verify_alignment()