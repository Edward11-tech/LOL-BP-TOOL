#!/usr/bin/env python3
"""
端到端预测一致性验证工具
=============================================
验证训练时模型预测结果与线上推理结果的一致性，通过 Spearman 相关系数、Top-K 重叠度、
标签排名等指标，确保训练-推理预测结果对齐。

功能描述:
    - 对比离线训练时的模型输出与线上推理的输出
    - 计算 Spearman 秩相关系数
    - 计算 Top-K 推荐结果的重叠度
    - 检查真实标签在推荐列表中的排名
    - 支持 Pick 和 Ban 两种模式的验证

主要函数:
    - compare_scores(): 对比两组预测分数，计算一致性指标
    - print_comparison(): 打印对比结果报告
    - _get_ally_enemy_from_bp(): 从 BP 序列解析友方/敌方英雄
    - _parse_pre_unavail_list(): 解析前置局已用英雄

使用方法:
    cd <project_root>
    python -m bp_recommendation.verify_predictions
    
    用于模型上线前的端到端验证，确保训练保存的模型与线上推理结果一致。
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logger_config import get_logger, setup_logging

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log = get_logger(__name__)

from bp_recommendation.model_pick.dataloader_pick import create_train_val_dataloaders
from bp_recommendation.feature_pipeline import BP_SEQUENCE
from bp_recommendation.bp_predict import BPRecommender
from bp_recommendation.model_pick.train_pick import CS_FEATURE_INDICES
from bp_recommendation.bp_recommendation_backend import BPRecommendationBackend, LEAGUES, POSITIONS

FEATURES_DIR = os.path.join(BASE_DIR, "bp_recommendation", "features")
CTX_PARQUET = os.path.join(FEATURES_DIR, "ALL_context.parquet")
META_PARQUET = os.path.join(FEATURES_DIR, "ALL_meta_store.parquet")
PLAYER_PARQUET = os.path.join(FEATURES_DIR, "ALL_player_store.parquet")
VOCAB_PATH = os.path.join(BASE_DIR, "cleaned_data", "champion_vocabulary.json")
POS_JSON = os.path.join(BASE_DIR, "cleaned_data", "champion_position_mapping.json")

DEVICE = "cpu"
TOP_K = 20

def _get_ally_enemy_from_bp(bp_seq, target_step, side, cs):
    ally_champs, enemy_champs = [], []
    for i in range(target_step):
        cid = int(bp_seq[i])
        if cid < cs: continue
        if BP_SEQUENCE[i][0] == "pick":
            if BP_SEQUENCE[i][1] == side: ally_champs.append(cid)
            else: enemy_champs.append(cid)
    return ally_champs, enemy_champs

def _parse_pre_unavail_list(row, name_to_idx, cs):
    """从历史比赛数据的 prev_game_champs 字段解析前置局已用英雄 (Fearless Draft)。

    与训练时 dataloader 的逻辑一致：prev_game_champs 为 '|' 分隔的英雄名。
    """
    pre_unavail_list = []
    prev_champs_str = str(row.get("prev_game_champs", ""))
    if prev_champs_str and prev_champs_str.strip() and prev_champs_str != "nan":
        for champ_name in prev_champs_str.split("|"):
            champ_name = champ_name.strip()
            if champ_name:
                cid = name_to_idx.get(champ_name, -1)
                if cid >= cs:
                    pre_unavail_list.append(cid)
    return pre_unavail_list

def compare_scores(off_scores, on_scores, champion_names, label, top_k=TOP_K):
    cs = 3
    valid_mask = ~np.isinf(off_scores) & ~np.isnan(off_scores) & (off_scores > -1e8) & \
                 ~np.isinf(on_scores) & ~np.isnan(on_scores) & (on_scores > -1e8)
    valid_mask[:cs] = False
    valid = np.where(valid_mask)[0]
    if len(valid) == 0: return {"spearman_r": 0, "spearman_p": 1, "top_k_overlap": 0, "top_k_denom": 1, "top1_match": False, "off_top1": "", "on_top1": "", "label_rank_off": -1, "label_rank_on": -1, "off_top_names": [], "on_top_names": []}

    off_valid, on_valid = off_scores[valid], on_scores[valid]
    rho, pval = spearmanr(off_valid, on_valid)
    
    off_top_k = set(valid[np.argsort(-off_valid)[:top_k]])
    on_top_k = set(valid[np.argsort(-on_valid)[:top_k]])
    
    return {
        "spearman_r": float(rho), "spearman_p": float(pval),
        "top_k_overlap": len(off_top_k & on_top_k), "top_k_denom": min(top_k, len(off_top_k)),
        "top1_match": valid[np.argmax(off_valid)] == valid[np.argmax(on_valid)],
        "off_top1": champion_names.get(valid[np.argmax(off_valid)], "???"),
        "on_top1": champion_names.get(valid[np.argmax(on_valid)], "???"),
        "off_top_names": [champion_names.get(c, "???") for c in valid[np.argsort(-off_valid)[:top_k]]],
        "on_top_names": [champion_names.get(c, "???") for c in valid[np.argsort(-on_valid)[:top_k]]],
        "label_rank_off": int(np.where(np.argsort(-off_valid) == np.where(valid == label)[0][0])[0][0] + 1) if label in valid else -1,
        "label_rank_on": int(np.where(np.argsort(-on_valid) == np.where(valid == label)[0][0])[0][0] + 1) if label in valid else -1,
    }

def print_comparison(title, result, top_k=TOP_K):
    log.info(f"\n  --- {title} ---")
    log.info(f"  Spearman r = {result['spearman_r']:.6f}  (p={result['spearman_p']:.2e})")
    log.info(f"  Top-{top_k} Overlap: {result['top_k_overlap']}/{result['top_k_denom']}")
    log.info(f"  Top-1 Match: {result['top1_match']}")
    if result['spearman_r'] > 0.99 and result['top1_match']: log.info(f"  => PASS")
    elif result['spearman_r'] > 0.95 and result['top_k_overlap'] >= result['top_k_denom'] * 0.8: log.info(f"  => ACCEPTABLE")
    else: log.warning(f"  => MISMATCH")

def build_on_ctx(store, match_row):
    """构建线上 Stateless Context"""
    league = match_row.get("league", "LPL")
    l_vec = np.zeros(len(LEAGUES), dtype=np.float32)
    if league in LEAGUES: l_vec[LEAGUES.index(league)] = 1.0
    bt, rt = match_row.get("blue_team", ""), match_row.get("red_team", "")
    bs = store.team_style_dict.get(bt, [0.7, 0.0, 1900.0, 0.5, 0.5])
    rs = store.team_style_dict.get(rt, [0.7, 0.0, 1900.0, 0.5, 0.5])
    p_f = 1.0 if match_row.get("playoffs", 0) else 0.0
    fp_f = float(match_row.get("first_pick_map_side", 1.0))
    gn = next((i for i in range(1, 6) if match_row.get(f"is_game_{i}", 0) == 1), 1)
    g_vec = np.zeros(5, dtype=np.float32)
    if 1 <= gn <= 5: g_vec[gn - 1] = 1.0
    return np.concatenate([l_vec, bs, rs, [p_f, fp_f], g_vec]).astype(np.float32)

def cascade_to_score_map(cascade_results, vocab_size):
    scores = np.full(vocab_size, -np.inf, dtype=np.float32)
    for cid, score, rank in cascade_results:
        scores[cid] = score
    return scores

def verify_predictions():
    log.info("=" * 70 + "\n  End-to-End Prediction Consistency (Stateless)\n" + "=" * 70)

    log.info("\n[1/7] Loading offline DataLoader...")
    train_loader, _ = create_train_val_dataloaders(CTX_PARQUET, META_PARQUET, PLAYER_PARQUET, VOCAB_PATH, POS_JSON, batch_size=1, num_workers=0, val_ratio=0.0, force_unroll_train=True)
    offline_dataset = train_loader.dataset
    cs = offline_dataset.champion_start_idx

    log.info("\n[2/7] Loading BPRecommender (via BPRecommendationBackend)...")
    backend = BPRecommendationBackend()
    load_res = backend.load()
    if not load_res.get("success"):
        log.error(f"Backend load failed: {load_res.get('message')}")
        return
    recommender = backend.recommender

    # 取训练集最后一天的全部比赛样本，接近线上最新快照信息
    # context 是 list of dict (to_dict("records")), 需转回 DataFrame
    context_df = pd.DataFrame(offline_dataset.context)
    all_dates = pd.to_datetime(context_df["date"], errors="coerce")
    max_date = all_dates.max()
    last_day_mask = all_dates == max_date
    last_day_indices = [i for i, (match_idx, step) in enumerate(offline_dataset.sample_list)
                        if last_day_mask.iloc[match_idx]]
    log.info(f"\n[3/7] Last training day: {max_date.date()}, samples found: {len(last_day_indices)}")
    if not last_day_indices:
        log.error("No samples found for the last training day!")
        return

    def _safe_get_pid(row, pos_key):
        val = row.get(pos_key, "")
        return str(val) if pd.notna(val) and str(val).lower() != "nan" else "unknown"

    pick_results = []
    ban_results = []

    log.info(f"\n[4/7] Comparing offline vs online predictions for all last-day samples...")
    for sample_idx, idx in enumerate(last_day_indices):
        match_idx, step = offline_dataset.sample_list[idx]
        action = BP_SEQUENCE[step][0]
        sample = offline_dataset[idx]
        match_row = offline_dataset.context[match_idx]
        label = int(sample["label"].numpy())

        if action == "pick":
            res = _compare_pick_sample(recommender, offline_dataset, sample, match_row, step, label, cs, _safe_get_pid)
            if res: pick_results.append(res)
        else:
            res = _compare_ban_sample(recommender, offline_dataset, sample, match_row, step, label, cs, _safe_get_pid)
            if res: ban_results.append(res)

        if (sample_idx + 1) % 20 == 0:
            log.info(f"  Progress: {sample_idx + 1}/{len(last_day_indices)}")

    log.info("\n[5/7] Summary")
    _print_summary("Pick", pick_results)
    _print_summary("Ban", ban_results)

    log.info("\n[6/7] Backend.recommend(payload) End-to-End Check (sampling)")
    # 对最后一天的样本抽样进行 E2E 检查（避免过多请求触发限流）
    e2e_sample_indices = last_day_indices[::max(1, len(last_day_indices) // 10)][:10]
    e2e_ok = True
    for idx in e2e_sample_indices:
        match_idx, step = offline_dataset.sample_list[idx]
        action = BP_SEQUENCE[step][0]
        sample = offline_dataset[idx]
        match_row = offline_dataset.context[match_idx]
        label = int(sample["label"].numpy())
        ok = verify_backend_recommend_e2e(backend, match_row if action == "pick" else None, step if action == "pick" else 0, label if action == "pick" else 0,
                                          match_row if action == "ban" else None, step if action == "ban" else 0, label if action == "ban" else 0, cs)
        e2e_ok = e2e_ok and ok
    if e2e_ok:
        log.info("Backend E2E: PASS")
    else:
        log.warning("Backend E2E: FAIL")

    log.info("\n[7/7] Verify Script Finished.")


def _compare_pick_sample(recommender, offline_dataset, sample, match_row, step, label, cs, _safe_get_pid):
    """对比单个 pick 样本的离线与在线预测。"""
    side = BP_SEQUENCE[step][1]
    off_bp_seq = sample["bp_sequence"].numpy().astype(np.int64)
    off_ctx = sample["global_context"].numpy().astype(np.float32)
    off_cand = sample["candidate_matrix"].numpy().astype(np.float32)
    off_mask = sample["available_mask"].numpy().astype(np.float32)
    off_ally, off_enemy = _get_ally_enemy_from_bp(off_bp_seq, step, side, cs)
    off_lap = -1
    for i in range(step):
        if off_bp_seq[i] >= cs and BP_SEQUENCE[i][0] == "pick" and BP_SEQUENCE[i][1] == side: off_lap = i

    on_bp_seq = [int(match_row.get(f"bp_step{i}_champion_id", 0)) for i in range(step)]
    on_ctx = build_on_ctx(recommender.store, match_row)
    on_ally, on_enemy = _get_ally_enemy_from_bp(on_bp_seq, step, side, cs)

    b_pids = [_safe_get_pid(match_row, f"blue_{pos}_player_id") for pos in ["top", "jng", "mid", "bot", "sup"]]
    r_pids = [_safe_get_pid(match_row, f"red_{pos}_player_id") for pos in ["top", "jng", "mid", "bot", "sup"]]
    on_ally_pids = b_pids if side == "blue" else r_pids
    on_enemy_pids = r_pids if side == "blue" else b_pids
    b_team, r_team = match_row.get("blue_team", ""), match_row.get("red_team", "")
    pick_pre_unavail = _parse_pre_unavail_list(match_row, recommender.store.name_to_idx, cs)

    on_cand, on_mask = recommender.store.get_pick_candidate_matrix(
        side, on_ally, on_enemy, set(on_bp_seq) | set(pick_pre_unavail), on_ally_pids, on_enemy_pids, step,
        b_team if side == "blue" else r_team, r_team if side == "blue" else b_team, pick_pre_unavail
    )

    bp_t = torch.as_tensor([off_bp_seq], dtype=torch.long, device=DEVICE)
    ctx_t, cand_t, mask_t, lap_t = [torch.as_tensor(x, device=DEVICE).unsqueeze(0) for x in [off_ctx, off_cand, off_mask, off_lap]]
    on_bp_padded = on_bp_seq + [recommender.store.PAD_IDX] * (20 - len(on_bp_seq))
    on_bp_t = torch.as_tensor([on_bp_padded], dtype=torch.long, device=DEVICE)
    on_ctx_t, on_cand_t, on_mask_t, on_lap_t = [torch.as_tensor(x, device=DEVICE).unsqueeze(0) for x in [on_ctx, on_cand, on_mask, off_lap]]

    with torch.no_grad():
        off_logits = recommender.pick_cs_model(bp_t, ctx_t, cand_t, mask_t, last_ally_pos=lap_t)["logits"].squeeze(0).cpu().numpy()
        on_logits = recommender.pick_cs_model(on_bp_t, on_ctx_t, on_cand_t, on_mask_t, last_ally_pos=on_lap_t)["logits"].squeeze(0).cpu().numpy()

    tf_res = compare_scores(off_logits, on_logits, offline_dataset.idx_to_name, label)

    off_cascade = recommender.predict_pick(off_bp_seq.tolist()[:step], off_ally, off_enemy, set(off_bp_seq), off_ctx, off_cand, off_mask, step, off_lap)
    on_cascade = recommender.predict_pick(on_bp_seq, on_ally, on_enemy, set(on_bp_seq), on_ctx, on_cand, on_mask, step, off_lap)
    cascade_res = compare_scores(cascade_to_score_map(off_cascade, offline_dataset.vocab_size), cascade_to_score_map(on_cascade, offline_dataset.vocab_size), offline_dataset.idx_to_name, label)

    return {"tf": tf_res, "cascade": cascade_res, "step": step, "gameid": match_row.get("gameid")}


def _compare_ban_sample(recommender, offline_dataset, sample, match_row, step, label, cs, _safe_get_pid):
    """对比单个 ban 样本的离线与在线预测。"""
    side = BP_SEQUENCE[step][1]
    off_b_bp = sample["bp_sequence"].numpy().astype(np.int64)
    off_b_ctx = sample["global_context"].numpy().astype(np.float32)
    off_b_cand = sample["candidate_matrix"].numpy().astype(np.float32)
    off_b_mask = sample["available_mask"].numpy().astype(np.float32)
    off_b_ally, off_b_enemy = _get_ally_enemy_from_bp(off_b_bp, step, side, cs)

    on_b_bp_seq = [int(match_row.get(f"bp_step{i}_champion_id", 0)) for i in range(step)]
    on_b_ctx = build_on_ctx(recommender.store, match_row)
    on_b_ally, on_b_enemy = _get_ally_enemy_from_bp(on_b_bp_seq, step, side, cs)

    b_pids = [_safe_get_pid(match_row, f"blue_{pos}_player_id") for pos in ["top", "jng", "mid", "bot", "sup"]]
    r_pids = [_safe_get_pid(match_row, f"red_{pos}_player_id") for pos in ["top", "jng", "mid", "bot", "sup"]]
    b_team, r_team = match_row.get("blue_team", ""), match_row.get("red_team", "")
    ban_pre_unavail = _parse_pre_unavail_list(match_row, recommender.store.name_to_idx, cs)

    on_b_cand, on_b_mask = recommender.store.get_ban_candidate_matrix(
        side, on_b_ally, on_b_enemy, set(on_b_bp_seq) | set(ban_pre_unavail),
        b_pids if side == "blue" else r_pids, r_pids if side == "blue" else b_pids, step,
        b_team if side == "blue" else r_team, r_team if side == "blue" else b_team, ban_pre_unavail
    )

    hist_pos = np.full(20, -1, dtype=np.int64)
    for i in range(min(len(off_b_bp), 20)):
        cid = off_b_bp[i]
        if cid >= cs and BP_SEQUENCE[i][0] == "pick": hist_pos[i] = int(np.argmax(recommender.store.pos_prior[cid]))

    off_b_bp_t, off_b_ctx_t, off_b_cand_t, off_b_mask_t, off_hist_t = [torch.as_tensor(x, device=DEVICE).unsqueeze(0) for x in [off_b_bp, off_b_ctx, off_b_cand, off_b_mask, hist_pos]]
    on_b_bp_t, on_b_ctx_t, on_b_cand_t, on_b_mask_t = [torch.as_tensor(np.asarray(x), device=DEVICE).unsqueeze(0) for x in [on_b_bp_seq + [recommender.store.PAD_IDX] * (20 - len(on_b_bp_seq)), on_b_ctx, on_b_cand, on_b_mask]]

    with torch.no_grad():
        off_logits = recommender.ban_model(off_b_bp_t, off_b_ctx_t, off_b_cand_t, off_b_mask_t, history_positions=off_hist_t)["logits"].squeeze(0).cpu().numpy()
        on_logits = recommender.ban_model(on_b_bp_t, on_b_ctx_t, on_b_cand_t, on_b_mask_t, history_positions=off_hist_t)["logits"].squeeze(0).cpu().numpy()

    tf_res = compare_scores(off_logits, on_logits, offline_dataset.idx_to_name, label)

    off_cascade = recommender.predict_ban(off_b_bp.tolist()[:step], off_b_ally, off_b_enemy, set(off_b_bp[:step]), off_b_ctx, off_b_cand, off_b_mask, step)
    on_cascade = recommender.predict_ban(on_b_bp_seq, on_b_ally, on_b_enemy, set(on_b_bp_seq), on_b_ctx, on_b_cand, on_b_mask, step)
    cascade_res = compare_scores(cascade_to_score_map(off_cascade, offline_dataset.vocab_size), cascade_to_score_map(on_cascade, offline_dataset.vocab_size), offline_dataset.idx_to_name, label)

    return {"tf": tf_res, "cascade": cascade_res, "step": step, "gameid": match_row.get("gameid")}


def _print_summary(label, results):
    """打印某类样本的汇总统计。"""
    if not results:
        log.info(f"\n  {label}: no samples")
        return
    tf_rhos = [r["tf"]["spearman_r"] for r in results]
    cascade_rhos = [r["cascade"]["spearman_r"] for r in results]
    tf_top1 = sum(1 for r in results if r["tf"]["top1_match"])
    cascade_top1 = sum(1 for r in results if r["cascade"]["top1_match"])
    log.info(f"\n  {label}: {len(results)} samples")
    log.info(f"    Transformer  Spearman r: mean={np.mean(tf_rhos):.6f} min={np.min(tf_rhos):.6f} max={np.max(tf_rhos):.6f}")
    log.info(f"    Cascade      Spearman r: mean={np.mean(cascade_rhos):.6f} min={np.min(cascade_rhos):.6f} max={np.max(cascade_rhos):.6f}")
    log.info(f"    Top-1 Match: TF={tf_top1}/{len(results)}  Cascade={cascade_top1}/{len(results)}")


def _parse_pre_unavail_list_for_backend(row, backend):
    """通过 backend.store 解析 prev_game_champs，与 _parse_pre_unavail_list 等价但无需 name_to_idx。"""
    store = backend.store
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


def _build_payload_from_match_row(row, step, backend):
    """从历史比赛数据构造与 bp_recommendation_backend.recommend() 一致的 payload。"""
    cs = backend.store.champion_start_idx
    action, side, slot = BP_SEQUENCE[step]

    bp_seq_ids = [int(row.get(f"bp_step{i}_champion_id", 0)) for i in range(step)]

    pre_unavail_list = _parse_pre_unavail_list_for_backend(row, backend)
    unavail_set = set(bp_seq_ids) | set(pre_unavail_list)

    blue_pids = [str(row.get(f"blue_{pos}_player_id", "")) for pos in POSITIONS]
    red_pids = [str(row.get(f"red_{pos}_player_id", "")) for pos in POSITIONS]
    for i in range(len(blue_pids)):
        if pd.isna(blue_pids[i]) or str(blue_pids[i]).lower() == "nan":
            blue_pids[i] = "unknown"
        if pd.isna(red_pids[i]) or str(red_pids[i]).lower() == "nan":
            red_pids[i] = "unknown"

    league = row.get("league", "LPL")
    if league not in LEAGUES:
        league = "LPL"

    game_num = next((i for i in range(1, 6) if row.get(f"is_game_{i}", 0) == 1), 1)

    payload = {
        "completed_steps": step,
        "league": league,
        "blue_team": row.get("blue_team", ""),
        "red_team": row.get("red_team", ""),
        "playoffs": bool(row.get("playoffs", 0)),
        "first_pick_map_side": float(row.get("first_pick_map_side", 1.0)),
        "game_num": game_num,
        "bp_seq_ids": bp_seq_ids,
        "unavail_set": list(unavail_set),
        "pre_unavail_list": pre_unavail_list,
        "blue_pids": blue_pids,
        "red_pids": red_pids,
    }

    if action == "pick" and slot is not None and 1 <= slot <= len(POSITIONS):
        payload["position_hint"] = POSITIONS[slot - 1]

    return payload


def verify_backend_recommend_e2e(backend, pick_row, pick_step, pick_label,
                                 ban_row, ban_step, ban_label, cs):
    """直接调用 backend.recommend(payload) 进行端到端验证。

    校验点：
    1. 后端返回结构包含 recommendations / step_info，无 error。
    2. 后端推荐结果与直接模型推理的 top1/top5 一致。
    3. 返回字段包含 champion / champion_idx / score / rank / reasons 等必要字段。
    """
    if pick_row is None and ban_row is None:
        log.warning("No pick/ban sample available for E2E check")
        return False

    all_ok = True

    def _check_one(row, step, label, action):
        nonlocal all_ok
        pre_unavail_list = _parse_pre_unavail_list_for_backend(row, backend)
        payload = _build_payload_from_match_row(row, step, backend)

        res = backend.recommend(payload)
        if "error" in res:
            log.error(f"Backend E2E [{action} step={step}] returned error: {res['error']}")
            all_ok = False
            return

        if "recommendations" not in res or "step_info" not in res:
            log.error(f"Backend E2E [{action} step={step}] missing recommendations/step_info")
            all_ok = False
            return

        recs = res["recommendations"]
        if not isinstance(recs, list) or len(recs) == 0:
            log.error(f"Backend E2E [{action} step={step}] empty recommendations")
            all_ok = False
            return

        for key in ("rank", "champion", "champion_idx", "score", "reasons"):
            if key not in recs[0]:
                log.error(f"Backend E2E [{action} step={step}] recommendation missing key: {key}")
                all_ok = False
                return

        backend_top5 = [r["champion_idx"] for r in recs[:5]]
        backend_top1 = backend_top5[0]

        # 与直接模型推理对比：复用 payload 中的字段构造特征矩阵并调用 predict_{action}
        side = BP_SEQUENCE[step][1]
        ally_champs, enemy_champs = _get_ally_enemy_from_bp(payload["bp_seq_ids"], step, side, cs)
        ally_pids = payload["blue_pids"] if side == "blue" else payload["red_pids"]
        enemy_pids = payload["red_pids"] if side == "blue" else payload["blue_pids"]
        team_name = payload["blue_team"] if side == "blue" else payload["red_team"]
        opp_team = payload["red_team"] if side == "blue" else payload["blue_team"]
        ctx = build_on_ctx(backend.store, row)
        unavail_set = set(payload["bp_seq_ids"]) | set(pre_unavail_list)

        if action == "pick":
            cand_np, mask_np = backend.store.get_pick_candidate_matrix(
                side, ally_champs, enemy_champs, unavail_set,
                ally_pids, enemy_pids, step, team_name, opp_team,
                pre_unavail_list=pre_unavail_list,
            )
            last_ally_pos = -1
            for i, cid in enumerate(payload["bp_seq_ids"]):
                if cid >= cs and BP_SEQUENCE[i][0] == "pick" and BP_SEQUENCE[i][1] == side:
                    last_ally_pos = i
            model_results = backend.recommender.predict_pick(
                payload["bp_seq_ids"], ally_champs, enemy_champs, unavail_set,
                ctx, cand_np, mask_np, step, last_ally_pos,
            )
        else:
            cand_np, mask_np = backend.store.get_ban_candidate_matrix(
                side, ally_champs, enemy_champs, unavail_set,
                ally_pids, enemy_pids, step, team_name, opp_team,
                pre_unavail_list=pre_unavail_list,
            )
            model_results = backend.recommender.predict_ban(
                payload["bp_seq_ids"], ally_champs, enemy_champs, unavail_set,
                ctx, cand_np, mask_np, step,
            )

        model_top5 = [int(cid) for cid, _, _ in model_results[:5]]
        model_top1 = model_top5[0]

        log.info(f"  Backend E2E [{action} step={step}] top1={backend_top1} model_top1={model_top1}")
        log.info(f"  Backend E2E [{action} step={step}] top5 overlap={len(set(backend_top5) & set(model_top5))}/5")

        if backend_top1 != model_top1:
            log.warning(f"  Backend E2E [{action} step={step}] top1 mismatch")
            all_ok = False

        overlap = len(set(backend_top5) & set(model_top5))
        if overlap < 4:
            log.warning(f"  Backend E2E [{action} step={step}] top5 overlap too low: {overlap}/5")
            all_ok = False

        if label in model_top5 and label not in backend_top5:
            log.warning(f"  Backend E2E [{action} step={step}] label not in backend top5")
            all_ok = False

    if pick_row is not None and pick_step > 0:
        _check_one(pick_row, pick_step, pick_label, "pick")
    if ban_row is not None and ban_step > 0:
        _check_one(ban_row, ban_step, ban_label, "ban")

    return all_ok


if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))
    
    _run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _run_log_path = os.path.join(LOG_DIR, f"verify_predictions_{_run_ts}.log")
    _run_fh = logging.FileHandler(_run_log_path, encoding="utf-8")
    _run_fh.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _run_fh.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_run_fh)
    
    verify_predictions()