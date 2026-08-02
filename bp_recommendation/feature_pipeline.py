"""
特征工程流水线
=============================================
从原始比赛数据中提取和构建 BP 推荐模型所需的各类特征，生成训练和推理用的特征文件。

功能描述:
    - 加载和处理原始比赛数据
    - 构建英雄词汇表和位置映射
    - 计算 Meta 特征（版本强势度、胜率、禁用率等）
    - 计算选手特征（熟练度、KDA、胜率等）
    - 计算英雄克制/协同关系（Counter/Synergy）
    - 计算战队风格特征
    - 计算恩怨/绝活/火热状态等高级特征
    - 生成 serving 快照供生产环境快速加载

主要常量/函数:
    - BP_SEQUENCE: BP 阶段顺序定义
    - CANDIDATE_FEAT_MAP: 候选特征索引映射
    - CANDIDATE_DIM: 候选特征维度
    - load_champion_vocabulary(): 加载英雄词汇表
    - run_feature_pipeline(): 运行完整特征流水线

使用方法:
    cd /Users/siwentu/Desktop/LOL analysis
    python -m bp_recommendation.feature_pipeline
    
    或在代码中调用:
    from bp_recommendation.feature_pipeline import run_feature_pipeline
    run_feature_pipeline()
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import json
import math
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from collections import Counter
from dateutil.relativedelta import relativedelta

BASE_DIR_LOADER = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, BASE_DIR_LOADER)
from logger_config import get_logger, setup_logging, log_context, timed
from common.paths import (
    MATCHES_CLEANED_CSV,
    PLAYER_CAREER_STATS_CSV,
    CHAMPION_VOCABULARY_JSON,
    CHAMPION_COUNTERS_CSV,
    CHAMPION_SYNERGY_CSV,
    get_latest_data_date,
)


def _sanitize_for_json(obj):
    """递归将 numpy 类型转换为 Python 原生类型，确保 JSON 序列化安全。

    处理以下类型:
    - np.integer (int32, int64, ...) -> int
    - np.floating (float32, float64, ...) -> float
    - np.ndarray -> list
    - float('nan') / np.nan -> None (标准 JSON 不支持 NaN)
    - dict -> 递归转换 key 和 value
    - list/tuple -> 递归转换元素
    """
    if isinstance(obj, dict):
        return {_sanitize_for_json(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    elif isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    else:
        return obj

BASE_DIR = str(Path(__file__).parent.parent.resolve())
CLEANED_DATA_DIR = MATCHES_CLEANED_CSV.parent
CLEANED_DIR = str(CLEANED_DATA_DIR)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")

# 共享数据异常检测工具
import sys as _sys
_sys.path.insert(0, BASE_DIR)
from data_checks import check_dataframe, check_array

BP_SEQUENCE = [
    ("ban", "blue", 1), ("ban", "red", 1),
    ("ban", "blue", 2), ("ban", "red", 2),
    ("ban", "blue", 3), ("ban", "red", 3),
    ("pick", "blue", 1), ("pick", "red", 1),
    ("pick", "red", 2), ("pick", "blue", 2),
    ("pick", "blue", 3), ("pick", "red", 3),
    ("ban", "red", 4), ("ban", "blue", 4),
    ("ban", "red", 5), ("ban", "blue", 5),
    ("pick", "red", 4), ("pick", "blue", 4),
    ("pick", "blue", 5), ("pick", "red", 5),
]

# 贝叶斯平滑先验权重，用于 synergy_lookup 的胜率平滑
BAYESIAN_PRIOR_WEIGHT = 2

POSITIONS_SHORT = ["top", "jng", "mid", "bot", "sup"]
POSITIONS_FULL = ["top", "jungle", "mid", "bot", "support"]
POS_SHORT2FULL = dict(zip(POSITIONS_SHORT, POSITIONS_FULL))

GAME_RESULT_COLS = [
    "gamelength", "ckpm",
    "blue_firstdragon", "red_firstdragon",
    "blue_firstherald", "red_firstherald",
    "blue_void_grubs", "red_void_grubs",
    "blue_firsttower", "red_firsttower",
    "blue_golddiffat15", "red_golddiffat15",
]
PLAYER_RESULT_COLS = []
for _s in ["blue", "red"]:
    for _p in POSITIONS_SHORT:
        for _st in ["kills", "deaths", "assists"]:
            PLAYER_RESULT_COLS.append(f"{_s}_{_p}_{_st}")
ALL_RESULT_COLS = GAME_RESULT_COLS + PLAYER_RESULT_COLS

META_WINDOW_DAYS = 30
META_DECAY_HALF_LIFE = 7
PLAYER_WINDOW_DAYS = 90
PLAYER_DECAY_HALF_LIFE = 21
PLAYER_WINDOW_GAMES = 15
MASTERY_DECAY_HALF_LIFE = 180
GLOBAL_KDA_PRIOR = 3.0
TEAM_PROFILE_WINDOW_DAYS = 90
TEAM_PROFILE_DECAY_HALF_LIFE = 45

MAX_HISTORY_MONTHS = 18

_pipeline_logger = None

CANDIDATE_FEAT_MAP = {
    "meta_pick": 0, "meta_ban": 1, "meta_presence": 2, "meta_wr": 3,
    "player_mastery": 4, "player_recent_kda": 5, "player_recent_wr": 6,
    "player_recent_games": 7,
    "player_overall_kda": 8, "player_overall_wr": 9, "player_overall_games": 10,
    "pos_top": 11, "pos_jng": 12, "pos_mid": 13, "pos_bot": 14, "pos_sup": 15,
    "ally_synergy": 16, "enemy_synergy": 17, "enemy_counter": 18, "ally_counter": 19,
    "ally_role_fit": 20, "enemy_role_fit": 21, "is_pick": 22,
    "enemy_mastery_max": 23, "enemy_mastery_mean": 24, "ban_step": 25,
    "grudge": 26, "respect": 27, "hot_streak": 28,
    "n_ally_picked": 29, "is_red_side": 30, "last_ally_synergy": 31,
    "is_fearless_banned": 32
}

CANDIDATE_DIM = len(CANDIDATE_FEAT_MAP)

def _get_logger():
    global _pipeline_logger
    if _pipeline_logger is not None:
        return _pipeline_logger
    _pipeline_logger = get_logger(__name__)
    return _pipeline_logger

def setup_pipeline_logger(output_dir=None):
    global _pipeline_logger
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"feature_pipeline_{timestamp}.log")
    logger = get_logger(__name__)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    _pipeline_logger = logger
    logger.info(f"Log file: {log_path}")
    return logger

def load_matches(league=None, max_history_months=None):
    log = _get_logger()
    if league is None:
        path = str(MATCHES_CLEANED_CSV)
    else:
        path = os.path.join(str(CLEANED_DATA_DIR), league, "matches_cleaned.csv")
    log.info(f"[数据加载] 加载比赛数据: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"patch": str})
    log.info(f"[数据加载] matches原始: {len(df)} 行, {len(df.columns)} 列")
    df["patch"] = df["patch"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    date_na = df["date"].isna().sum()
    if date_na > 0:
        log.warning(f"[数据加载] date列有 {date_na} 个空值")
    latest_date = df["date"].max()
    if max_history_months is None:
        max_history_months = MAX_HISTORY_MONTHS
    train_start = latest_date - relativedelta(months=max_history_months)
    n_before = len(df)
    df = df[df["date"] >= train_start].reset_index(drop=True)
    log.info(f"[数据加载] 时间窗口过滤: {train_start.date()} ~ {latest_date.date()}, {n_before} -> {len(df)} 行 (窗口: {max_history_months}个月)")
    leagues = df["league"].value_counts().to_dict() if "league" in df.columns else {}
    log.info(f"[数据加载] 联赛分布: {leagues}")
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    df["match_seq_idx"] = df.index
    log.info(f"[数据加载] matches加载完成")
    return df

def load_career_stats(league=None):
    log = _get_logger()
    if league is None:
        path = str(PLAYER_CAREER_STATS_CSV)
    else:
        path = os.path.join(CLEANED_DIR, league, "player_career_hero_stats_cleaned.csv")
    if not os.path.exists(path):
        log.warning(f"[数据加载] career_stats文件不存在: {path}, 返回空DataFrame")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info(f"[数据加载] player_career_stats: {len(df)} 行, {len(df.columns)} 列")
    unique_players = df["player_id"].nunique() if "player_id" in df.columns else 0
    unique_champs = df["champion"].nunique() if "champion" in df.columns else 0
    log.info(f"[数据加载] career_stats: {unique_players} 名选手, {unique_champs} 个英雄")
    return df

def enforce_pit(matches_df):
    df = matches_df.copy()
    keep_cols = [c for c in df.columns if c not in ALL_RESULT_COLS]
    target_df = df[keep_cols].copy()
    result_df = df[["gameid"] + ALL_RESULT_COLS].copy()
    return target_df, result_df

def melt_matches_to_player_rows(matches_df):
    records = []
    for side in ["blue", "red"]:
        result_val = matches_df["result"].values if side == "blue" else (1 - matches_df["result"].values)
        team_col = f"{side}_team"
        for pos_short in POSITIONS_SHORT:
            pos_full = POS_SHORT2FULL[pos_short]
            cols_needed = [
                "gameid", "league", "year", "split", "date", "patch", "match_seq_idx",
                team_col,
                f"{side}_{pos_short}_player_id",
                f"{side}_{pos_short}_champion",
                f"{side}_{pos_short}_kills",
                f"{side}_{pos_short}_deaths",
                f"{side}_{pos_short}_assists",
            ]
            sub = matches_df[cols_needed].copy()
            sub.columns = [
                "gameid", "league", "year", "split", "date", "patch", "match_seq_idx",
                "team", "player_id", "champion", "kills", "deaths", "assists",
            ]
            sub["side"] = side
            sub["position"] = pos_full
            sub["result"] = result_val
            sub["orig_pos_key"] = f"{side}_{pos_short}"
            records.append(sub)
    return pd.concat(records, ignore_index=True)

def melt_bans_from_matches(matches_df):
    ban_records = []
    for side in ["blue", "red"]:
        for i in range(1, 6):
            col = f"{side}_ban{i}"
            if col in matches_df.columns:
                sub = matches_df[["gameid", "date", "patch", "match_seq_idx", col]].copy()
                sub.columns = ["gameid", "date", "patch", "match_seq_idx", "champion"]
                sub["ban_side"] = side
                sub = sub[sub["champion"].notna() & (sub["champion"] != "") & (sub["champion"] != "Unknown")]
                ban_records.append(sub)
    if not ban_records:
        return pd.DataFrame(columns=["gameid", "date", "patch", "match_seq_idx", "champion", "ban_side"])
    return pd.concat(ban_records, ignore_index=True)

def build_base_prior(player_history, career_df):
    player_champ_agg = player_history.groupby(["player_id", "champion"]).agg(
        Total_Match_G=("gameid", "count"),
        Total_Match_W=("result", "sum"),
    ).reset_index()

    if career_df.empty:
        player_champ_agg["Base_G"] = 0
        player_champ_agg["Base_W"] = 0
        return player_champ_agg[["player_id", "champion", "Base_G", "Base_W"]]

    career_copy = career_df.copy()
    career_copy["Career_W"] = career_copy["games"] * career_copy["win_rate"]
    career_agg = career_copy.groupby(["player_id", "champion"]).agg(
        Career_G=("games", "sum"),
        Career_W=("Career_W", "sum"),
    ).reset_index()

    merged = player_champ_agg.merge(career_agg, on=["player_id", "champion"], how="left")
    merged["Career_G"] = merged["Career_G"].fillna(0)
    merged["Career_W"] = merged["Career_W"].fillna(0)

    merged["Base_G"] = (merged["Career_G"] - merged["Total_Match_G"]).clip(lower=0)
    merged["Base_W"] = (merged["Career_W"] - merged["Total_Match_W"]).clip(lower=0)

    extra_career = career_agg[
        ~career_agg.set_index(["player_id", "champion"]).index.isin(
            player_champ_agg.set_index(["player_id", "champion"]).index
        )
    ].copy()
    if not extra_career.empty:
        extra_career = extra_career.rename(columns={"Career_G": "Base_G", "Career_W": "Base_W"})
        result = pd.concat([
            merged[["player_id", "champion", "Base_G", "Base_W"]],
            extra_career[["player_id", "champion", "Base_G", "Base_W"]]
        ], ignore_index=True)
    else:
        result = merged[["player_id", "champion", "Base_G", "Base_W"]]

    return result

def parse_patch_to_index(patch_str):
    """将版本号 14.1, 14.10 转换为绝对索引，1大版本=24小版本（近似）"""
    try:
        parts = str(patch_str).strip().split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major * 24 + minor
    except:
        return 0

def compute_meta_features_pit(player_history, ban_history, matches_df):
    log = _get_logger()
    log.info("  [Meta] Computing champion meta features with patch distance penalty...")

    matches_sorted = matches_df.sort_values("match_seq_idx").reset_index(drop=True)
    decay_lambda = np.log(2) / META_DECAY_HALF_LIFE
    window_days = META_WINDOW_DAYS

    unique_champs = pd.concat([player_history["champion"], ban_history["champion"]]).unique()
    champ_to_int = {c: i for i, c in enumerate(unique_champs)}
    int_to_champ = {i: c for c, i in champ_to_int.items()}
    n_champs = len(unique_champs)

    match_dates = matches_sorted["date"].values.astype('datetime64[D]')
    base_date = match_dates.min()
    days_from_base = (match_dates - base_date).astype(float)

    match_seq_ids = matches_sorted["match_seq_idx"].values
    match_gameids = matches_sorted["gameid"].values

    ph = player_history.copy()
    ph["c_int"] = ph["champion"].map(champ_to_int)
    pick_dict = ph.groupby("match_seq_idx")["c_int"].apply(list).to_dict()
    win_dict = ph[ph["result"] == 1].groupby("match_seq_idx")["c_int"].apply(list).to_dict()

    bh = ban_history.copy()
    bh["c_int"] = bh["champion"].map(champ_to_int)
    ban_dict = bh.groupby("match_seq_idx")["c_int"].apply(list).to_dict()

    # ── Per-patch accumulators ──────────────────────────────────────────
    #   patch_key is str(patch) to handle float patches like 14.1, 14.10
    patch_A_pick: dict[str, np.ndarray] = {}
    patch_A_ban: dict[str, np.ndarray] = {}
    patch_A_win: dict[str, np.ndarray] = {}
    patch_A_slots: dict[str, float] = {}
    #   match_patches[i] = patch_key of match at original index i
    match_patches: list[str] = []

    results = []
    left_idx = 0
    n_matches = len(matches_sorted)

    for i in range(n_matches):
        target_seq_idx = match_seq_ids[i]  # noqa
        target_gameid = match_gameids[i]
        t_i = days_from_base[i]
        target_patch = matches_sorted.iloc[i]["patch"]
        target_pkey = str(target_patch)

        # ── 1. 将前一场比赛入库（按 patch 分组）────────────────────
        if i > 0:
            prev_seq_idx = match_seq_ids[i - 1]
            t_prev = days_from_base[i - 1]
            prev_pkey = str(matches_sorted.iloc[i - 1]["patch"])
            factor = np.exp(decay_lambda * t_prev)

            match_patches.append(prev_pkey)

            # 延迟初始化 per-patch 数组
            if prev_pkey not in patch_A_pick:
                patch_A_pick[prev_pkey] = np.zeros(n_champs, dtype=np.float64)
                patch_A_ban[prev_pkey] = np.zeros(n_champs, dtype=np.float64)
                patch_A_win[prev_pkey] = np.zeros(n_champs, dtype=np.float64)
                patch_A_slots[prev_pkey] = 0.0

            for c in pick_dict.get(prev_seq_idx, []):
                patch_A_pick[prev_pkey][c] += factor
            for c in win_dict.get(prev_seq_idx, []):
                patch_A_win[prev_pkey][c] += factor
            for c in ban_dict.get(prev_seq_idx, []):
                patch_A_ban[prev_pkey][c] += factor
            patch_A_slots[prev_pkey] += factor

        # ── 2. 剔除超过 window_days 的过期比赛 ───────────────────
        while left_idx < i and (match_dates[i] - match_dates[left_idx]).astype(int) >= window_days:
            old_pkey = match_patches[left_idx]
            old_seq_idx = match_seq_ids[left_idx]
            t_old = days_from_base[left_idx]
            factor_old = np.exp(decay_lambda * t_old)

            for c in pick_dict.get(old_seq_idx, []):
                patch_A_pick[old_pkey][c] -= factor_old
            for c in win_dict.get(old_seq_idx, []):
                patch_A_win[old_pkey][c] -= factor_old
            for c in ban_dict.get(old_seq_idx, []):
                patch_A_ban[old_pkey][c] -= factor_old
            patch_A_slots[old_pkey] -= factor_old

            # 如果某个 patch 的累积量为 0，清空释放
            if patch_A_slots[old_pkey] <= 0.0:
                del patch_A_pick[old_pkey]
                del patch_A_ban[old_pkey]
                del patch_A_win[old_pkey]
                del patch_A_slots[old_pkey]

            left_idx += 1

        # ── 3. 按 patch 距离惩罚加权计算当前比赛的 Meta ──────────
        w_picks = np.zeros(n_champs, dtype=np.float64)
        w_bans = np.zeros(n_champs, dtype=np.float64)
        w_wins = np.zeros(n_champs, dtype=np.float64)
        w_total_slots = 0.0

        current_factor = np.exp(-decay_lambda * t_i)

        target_patch_idx = parse_patch_to_index(target_pkey)

        for pkey in patch_A_slots:
            patch_idx = parse_patch_to_index(pkey)
            patch_dist = abs(patch_idx - target_patch_idx)
            # 版本差乘以 0.5 作为惩罚系数（每隔1个版本，权重衰减 60%）
            patch_penalty = float(np.exp(-patch_dist * 0.5))
            patch_penalty = max(patch_penalty, 0.1)  # 底限 0.1

            w_picks += patch_penalty * patch_A_pick[pkey] * current_factor
            w_bans += patch_penalty * patch_A_ban[pkey] * current_factor
            w_wins += patch_penalty * patch_A_win[pkey] * current_factor
            w_total_slots += patch_penalty * patch_A_slots[pkey] * current_factor

        total_denom = max(w_total_slots, 1.0)
        pick_rates = w_picks / total_denom
        ban_rates = w_bans / total_denom
        presences = (w_picks + w_bans) / total_denom
        win_rates = (w_wins + 1.0) / (w_picks + 2.0)

        active_indices = np.where(presences > 0.0)[0]
        for idx in active_indices:
            results.append({
                "gameid": target_gameid,
                "date": match_dates[i],
                "champion": int_to_champ[idx],
                "meta_pick_rate_pit": pick_rates[idx],
                "meta_ban_rate_pit": ban_rates[idx],
                "meta_presence_pit": presences[idx],
                "meta_win_rate_pit": win_rates[idx],
            })

    result_df = pd.DataFrame(results)
    log.info(f"  [Meta] Meta特征计算完成: {len(result_df)} 行, {len(result_df.columns)} 列, {result_df['champion'].nunique()} 个英雄, 4维特征(pick/ban/presence/win)")
    meta_na_rates = {}
    for c in result_df.select_dtypes(include=[np.number]).columns:
        na_pct = result_df[c].isna().mean() * 100
        if na_pct > 30:
            log.warning(f"  [Meta] {c} 缺失率 {na_pct:.1f}% > 30%")
        meta_na_rates[c] = f"{na_pct:.1f}%"
    log.info(f"  [Meta] Meta特征列缺失率: {meta_na_rates}")
    return result_df

def compute_player_features_pit(player_history, base_prior_df):
    log = _get_logger()
    log.info("  [Player] Computing DENSE Player Feature Store with chronological match_seq_idx isolation...")

    base_prior_dict = {}
    if not base_prior_df.empty:
        for _, row in base_prior_df.iterrows():
            base_prior_dict[(row["player_id"], row["champion"])] = (row["Base_G"], row["Base_W"])

    ph = player_history.sort_values("match_seq_idx").copy()
    hist_dict = {}
    for (pid, c), g in ph.groupby(["player_id", "champion"]):
        hist_dict[(pid, c)] = {
            "dates": g["date"].values.astype("datetime64[ns]"),
            "match_seq_indices": g["match_seq_idx"].values,
            "results": g["result"].values.astype(float),
            "kills": g["kills"].values.astype(float),
            "deaths": g["deaths"].values.astype(float),
            "assists": g["assists"].values.astype(float),
        }

    player_matches = {}
    for pid, g in ph.groupby("player_id"):
        matches = g[["gameid", "date", "match_seq_idx"]].drop_duplicates().sort_values("match_seq_idx")
        player_matches[pid] = matches.to_dict("records")

    player_pools = {}
    for pid in player_matches.keys():
        played = ph[ph["player_id"] == pid]["champion"].unique()
        priors = base_prior_df[base_prior_df["player_id"] == pid]["champion"].unique() if not base_prior_df.empty else []
        player_pools[pid] = set(played) | set(priors)

    mastery_decay_lambda = np.log(2) / MASTERY_DECAY_HALF_LIFE
    recent_decay_lambda = np.log(2) / PLAYER_DECAY_HALF_LIFE
    window_days = PLAYER_WINDOW_DAYS
    GLOBAL_KDA_PRIOR = 3.0

    all_player_hist = {}
    for pid, g in ph.groupby("player_id"):
        g_sorted = g.sort_values("match_seq_idx").drop_duplicates(subset=["gameid"])
        all_player_hist[pid] = {
            "dates": g_sorted["date"].values.astype("datetime64[ns]"),
            "match_seq_indices": g_sorted["match_seq_idx"].values,
            "results": g_sorted["result"].values.astype(float),
            "kills": g_sorted["kills"].values.astype(float),
            "deaths": g_sorted["deaths"].values.astype(float),
            "assists": g_sorted["assists"].values.astype(float),
        }

    dense_results = []

    for pid, matches in player_matches.items():
        pool = player_pools[pid]
        ap = all_player_hist[pid]
        ap_dates = ap["dates"]
        ap_seq_indices = ap["match_seq_indices"]
        ap_res, ap_k, ap_d, ap_a = ap["results"], ap["kills"], ap["deaths"], ap["assists"]
        
        # 【性能修复】：提前在外部解析本选手的比赛时间
        for m in matches:
            if "np_date" not in m:
                m["np_date"] = np.datetime64(pd.Timestamp(m["date"]))

        for champ in pool:
            h = hist_dict.get((pid, champ), None)
            if h is not None:
                h_dates, h_seq_indices = h["dates"], h["match_seq_indices"]
                h_res, h_k, h_d, h_a = h["results"], h["kills"], h["deaths"], h["assists"]
            else:
                h_dates = np.array([], dtype='datetime64[ns]')
                h_seq_indices = np.array([], dtype=int)
                h_res = h_k = h_d = h_a = np.array([])

            base_g, base_w = base_prior_dict.get((pid, champ), (0.0, 0.0))
            base_wr = base_w / base_g if base_g > 0 else 0.5

            for match in matches:
                target_date = match["np_date"]  # 直接取缓存
                target_seq_idx = match["match_seq_idx"]
                gameid = match["gameid"]
                ap_cutoff = target_date - np.timedelta64(window_days, "D")
                
                idx_end = np.searchsorted(ap_seq_indices, target_seq_idx, side='left')
                idx_start = np.searchsorted(ap_dates[:idx_end], ap_cutoff, side='left')
                
                if idx_start == idx_end:
                    overall_wr, overall_kda, overall_games = np.nan, np.nan, 0
                else:
                    ap_dates_win = ap_dates[idx_start:idx_end]
                    ap_res_win = ap_res[idx_start:idx_end]
                    ap_k_win = ap_k[idx_start:idx_end]
                    ap_d_win = ap_d[idx_start:idx_end]
                    ap_a_win = ap_a[idx_start:idx_end]
                    
                    ap_delta = (target_date - ap_dates_win) / np.timedelta64(1, "D")
                    ap_w = np.exp(-recent_decay_lambda * ap_delta)
                    ap_w_sum = ap_w.sum()
                    
                    overall_wr = float(np.sum(ap_w * ap_res_win) / ap_w_sum)
                    
                    # 【逻辑修复】：对 Death 进行保底处理，防止 KDA 萎缩
                    ap_d_safe = np.maximum(ap_d_win, 1.0)
                    ap_wd = np.sum(ap_w * ap_d_safe)
                    overall_kda = float((np.sum(ap_w * ap_k_win) + np.sum(ap_w * ap_a_win)) / ap_wd)
                    overall_games = int(idx_end - idx_start)

                # 【优化 1.2】：同样应用二分查找截取绝活 Mastery 记录
                h_end = np.searchsorted(h_seq_indices, target_seq_idx, side='left')
                
                if h_end == 0:
                    mastery_g = base_g
                    m_wr_score = base_wr * 50
                    m_exp_score = np.minimum(np.log1p(mastery_g) / np.log1p(50), 1.0) * 30
                    m_kda_score = np.minimum(GLOBAL_KDA_PRIOR, 8.0) / 8.0 * 20
                    mastery_score = m_wr_score + m_exp_score + m_kda_score
                    recent_kda, recent_wr = np.nan, np.nan
                    recent_games = 0
                else:
                    h_dates_valid = h_dates[:h_end]
                    h_res_valid = h_res[:h_end]
                    h_k_valid = h_k[:h_end]
                    h_d_valid = h_d[:h_end]
                    h_a_valid = h_a[:h_end]

                    delta_days_full = (target_date - h_dates_valid) / np.timedelta64(1, "D")
                    weights_full = np.exp(-mastery_decay_lambda * delta_days_full)
                    w_sum_full = weights_full.sum()

                    w_wins_full = np.sum(weights_full * h_res_valid)
                    mastery_decay_wr = (w_wins_full + base_wr * 3) / (w_sum_full + 3)

                    w_k_full = np.sum(weights_full * h_k_valid)
                    w_d_full = np.sum(weights_full * h_d_valid)
                    w_a_full = np.sum(weights_full * h_a_valid)
                    mastery_decay_kda = (w_k_full + w_a_full) / w_d_full if w_d_full > 0 else (w_k_full + w_a_full)

                    mastery_g = base_g + h_end
                    m_wr_score = mastery_decay_wr * 50
                    m_exp_score = np.minimum(np.log1p(mastery_g) / np.log1p(50), 1.0) * 30
                    m_kda_score = np.minimum(mastery_decay_kda, 8.0) / 8.0 * 20
                    mastery_score = m_wr_score + m_exp_score + m_kda_score

                    cutoff = target_date - np.timedelta64(window_days, "D")
                    h_start = np.searchsorted(h_dates_valid, cutoff, side='left')
                    
                    if h_start == h_end:
                        recent_kda, recent_wr = np.nan, np.nan
                        recent_games = 0
                    else:
                        h_dates_trunc = h_dates_valid[h_start:h_end]
                        h_res_trunc = h_res_valid[h_start:h_end]
                        h_k_trunc = h_k_valid[h_start:h_end]
                        h_d_trunc = h_d_valid[h_start:h_end]
                        h_a_trunc = h_a_valid[h_start:h_end]

                        delta_days_trunc = (target_date - h_dates_trunc) / np.timedelta64(1, "D")
                        weights_trunc = np.exp(-recent_decay_lambda * delta_days_trunc)
                        w_sum_trunc = weights_trunc.sum()
                        
                        w_wins_trunc = np.sum(weights_trunc * h_res_trunc)
                        w_k_trunc = np.sum(weights_trunc * h_k_trunc)
                        w_d_trunc = np.sum(weights_trunc * h_d_trunc)
                        w_a_trunc = np.sum(weights_trunc * h_a_trunc)
                        
                        recent_wr = w_wins_trunc / w_sum_trunc
                        recent_kda = (w_k_trunc + w_a_trunc) / w_d_trunc if w_d_trunc > 0 else (w_k_trunc + w_a_trunc)
                        recent_games = int(h_end - h_start)

                dense_results.append({
                    "player_id": pid,
                    "gameid": gameid,
                    "date": target_date,
                    "champion": champ,
                    "mastery_score": mastery_score,
                    "player_recent_kda_90d": recent_kda,
                    "player_recent_wr_90d": recent_wr,
                    "player_recent_games_90d": recent_games,
                    "player_overall_recent_wr": overall_wr,
                    "player_overall_recent_kda": overall_kda,
                    "player_overall_recent_games": overall_games,
                })

    dense_player_df = pd.DataFrame(dense_results)
    unique_players = dense_player_df["player_id"].nunique()
    unique_champs = dense_player_df["champion"].nunique()
    unique_games = dense_player_df["gameid"].nunique()
    log.info(f"  [Player] Dense Feature Store生成完成: {len(dense_player_df)} 行, {len(dense_player_df.columns)} 列")
    log.info(f"  [Player] 覆盖: {unique_players} 名选手, {unique_champs} 个英雄, {unique_games} 场比赛")
    player_na_rates = {}
    for c in dense_player_df.select_dtypes(include=[np.number]).columns:
        na_pct = dense_player_df[c].isna().mean() * 100
        if na_pct > 30:
            log.warning(f"  [Player] {c} 缺失率 {na_pct:.1f}% > 30%")
        player_na_rates[c] = f"{na_pct:.1f}%"
    log.info(f"  [Player] Player特征缺失率: {player_na_rates}")
    return dense_player_df

def compute_player_pick_distribution(player_history):
    log = _get_logger()
    log.info("  [PickDist] Computing player champion pick distributions (90d window, strict match_seq_idx)...")

    ph = player_history.sort_values(["player_id", "match_seq_idx"]).copy()

    results = []
    for player_id, group in ph.groupby("player_id"):
        group_sorted = group.reset_index(drop=True)
        dates = group_sorted["date"].values.astype('datetime64[D]')
        champions = group_sorted["champion"].values
        gameids = group_sorted["gameid"].values
        n = len(dates)

        left_idx = 0
        
        for i in range(n):
            t = dates[i]
            cutoff = t - np.timedelta64(PLAYER_WINDOW_DAYS, 'D')

            while left_idx < i and dates[left_idx] < cutoff:
                left_idx += 1

            # 由于已经按 match_seq_idx 排序，[:i] 自然就是严格之前的历史
            if left_idx == i:
                continue

            champs_in_window = champions[left_idx:i]
            
            # 【性能修复】：使用 Counter 替代 pd.Series.value_counts
            vc = Counter(champs_in_window)
            total = len(champs_in_window)
            top5 = vc.most_common(5)
            
            for champ, count in top5:
                results.append({
                    "gameid": gameids[i],
                    "player_id": player_id,
                    "date": pd.Timestamp(dates[i]),
                    "pick_dist_champion": champ,
                    "pick_dist_pct": count / total,
                })

    if not results:
        return pd.DataFrame(columns=["gameid", "player_id", "date", "pick_dist_champion", "pick_dist_pct"])

    return pd.DataFrame(results)

def compute_team_profile_pit(matches_df):
    log = _get_logger()
    log.info("  [TeamProfile] Computing dynamic team style features (match_seq_idx PIT)...")

    team_game_cols = [
        "gameid", "date", "match_seq_idx", "blue_team", "red_team", "result",
        "gamelength", "ckpm", "blue_golddiffat15", "red_golddiffat15",
        "blue_firstdragon", "red_firstdragon", "blue_firsttower", "red_firsttower",
    ]
    available_cols = [c for c in team_game_cols if c in matches_df.columns]
    m = matches_df[available_cols].copy()

    blue_records = m[["gameid", "date", "match_seq_idx", "blue_team", "result",
                       "gamelength", "ckpm", "blue_golddiffat15", "blue_firstdragon", "blue_firsttower"]].copy()
    blue_records.columns = ["gameid", "date", "match_seq_idx", "team", "result",
                             "gamelength", "ckpm", "golddiffat15", "firstdragon", "firsttower"]

    red_records = m[["gameid", "date", "match_seq_idx", "red_team", "result",
                      "gamelength", "ckpm", "red_golddiffat15", "red_firstdragon", "red_firsttower"]].copy()
    red_records.columns = ["gameid", "date", "match_seq_idx", "team", "result",
                            "gamelength", "ckpm", "golddiffat15", "firstdragon", "firsttower"]
    red_records["result"] = 1 - red_records["result"]

    team_history = pd.concat([blue_records, red_records], ignore_index=True)
    team_history = team_history.sort_values(["team", "match_seq_idx"]).reset_index(drop=True)

    # 防御性去重：原始数据中极少数比赛存在 blue_team == red_team 的异常记录，
    # 会导致同一 (gameid, team) 出现两次，后续 merge 时行数膨胀。
    before_dedup = len(team_history)
    team_history = team_history.drop_duplicates(subset=["gameid", "team"], keep="last")
    after_dedup = len(team_history)
    if before_dedup != after_dedup:
        _log = logging.getLogger(__name__)
        _log.warning(f"[TeamProfile] Dropped {before_dedup - after_dedup} duplicate (gameid, team) rows")

    numeric_cols = ["gamelength", "ckpm", "golddiffat15", "firstdragon", "firsttower"]
    for c in numeric_cols:
        team_history[c] = pd.to_numeric(team_history[c], errors="coerce")

    # 【重要修复】LPL 原始数据中 golddiffat15 100% 缺失（数据源不提供），
    # 旧逻辑用 0 填充导致模型误判为"经济平局"。
    # 正确做法：将 0 值（来自缺失填充）转为 NaN，下游 wavg 自动跳过 NaN。
    # 当 golddiffat15 缺失时，firsttower (一塔) 作为领先指标替代。
    zero_count = (team_history["golddiffat15"] == 0).sum()
    total_count = len(team_history)
    if zero_count > 0:
        _log = logging.getLogger(__name__)
        _log.info(f"[TeamProfile] golddiffat15 中有 {zero_count}/{total_count} 行为 0 (可能来自缺失填充)")
        # 将 0 转为 NaN (0 经济差在真实比赛中极罕见, 几乎都是缺失填充)
        team_history.loc[team_history["golddiffat15"] == 0, "golddiffat15"] = np.nan
        _log.info(f"[TeamProfile] 已将 golddiffat15 的 0 值转为 NaN, 下游用 firsttower 替代")
    else:
        _log = logging.getLogger(__name__)
        _log.info(f"[TeamProfile] golddiffat15 无 0 值, 数据正常")

    decay_lambda = np.log(2) / TEAM_PROFILE_DECAY_HALF_LIFE
    window_days = TEAM_PROFILE_WINDOW_DAYS

    results = []
    for team, group in team_history.groupby("team"):
        # 【优化 2】：提取底层 Numpy 数组，利用双指针构建 O(N) 滑动窗口
        dates = group["date"].values.astype('datetime64[D]')
        gids = group["gameid"].values
        gl_arr = group["gamelength"].values
        ckpm_arr = group["ckpm"].values
        gold_arr = group["golddiffat15"].values
        fd_arr = group["firstdragon"].values
        ft_arr = group["firsttower"].values
        
        n = len(group)
        left_idx = 0

        for i in range(n):
            target_date = dates[i]
            cutoff = target_date - np.timedelta64(window_days, "D")
            
            # 双指针向前推进
            while left_idx < i and dates[left_idx] < cutoff:
                left_idx += 1

            if left_idx == i: # 窗口为空
                results.append({
                    "gameid": gids[i], "team": team, "date": pd.Timestamp(dates[i]),
                    "team_avg_gamelength": np.nan, "team_avg_ckpm": np.nan,
                    "team_avg_golddiffat15": np.nan, "team_firstdragon_rate": np.nan,
                    "team_firsttower_rate": np.nan, "team_profile_games": 0,
                })
                continue
            
            # 纯 Numpy 切片聚合
            delta_days = (target_date - dates[left_idx:i]).astype(float)
            weights = np.exp(-decay_lambda * delta_days)

            def wavg(arr):
                win_arr = arr[left_idx:i]
                valid = ~np.isnan(win_arr)
                if not valid.any(): return np.nan
                return np.average(win_arr[valid], weights=weights[valid])

            results.append({
                "gameid": gids[i], "team": team, "date": pd.Timestamp(dates[i]),
                "team_avg_gamelength": wavg(gl_arr),
                "team_avg_ckpm": wavg(ckpm_arr),
                "team_avg_golddiffat15": wavg(gold_arr),
                "team_firstdragon_rate": wavg(fd_arr),
                "team_firsttower_rate": wavg(ft_arr),
                "team_profile_games": i - left_idx,
            })

    result_df = pd.DataFrame(results)
    team_profile_defaults = {
        "team_avg_gamelength": 1900, "team_avg_ckpm": 0.7,
        "team_avg_golddiffat15": 0, "team_firstdragon_rate": 0.5,
        "team_firsttower_rate": 0.5,
    }
    for c, default in team_profile_defaults.items():
        result_df[c] = result_df[c].fillna(default)

    unique_teams = result_df["team"].nunique() if "team" in result_df.columns else 0
    unique_games = result_df["gameid"].nunique() if "gameid" in result_df.columns else 0
    log.info(f"  [TeamProfile] TeamProfile特征计算完成: {len(result_df)} 行, {len(result_df.columns)} 列")
    log.info(f"  [TeamProfile] 覆盖: {unique_teams} 支战队, {unique_games} 场比赛")
    team_na_rates = {}
    for c in result_df.select_dtypes(include=[np.number]).columns:
        na_pct = result_df[c].isna().mean() * 100
        if na_pct > 30:
            log.warning(f"  [TeamProfile] {c} 缺失率 {na_pct:.1f}% > 30%")
        team_na_rates[c] = f"{na_pct:.1f}%"
    log.info(f"  [TeamProfile] Team特征缺失率: {team_na_rates}")
    return result_df

def load_champion_vocabulary(vocab_path=None):
    import json
    if vocab_path is None:
        vocab_path = str(CHAMPION_VOCABULARY_JSON)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"champion_vocabulary.json not found at {vocab_path}. "
            "Run _build_vocab.py first to generate it."
        )
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    name_to_idx = vocab["name_to_idx"]
    idx_to_name = {champ["idx"]: champ["name"] for champ in vocab["champions"]}
    vocab_size = vocab["vocab_size"]
    special_tokens = vocab["special_tokens"]
    champion_start_idx = vocab["champion_start_idx"]
    return name_to_idx, idx_to_name, vocab_size, special_tokens, champion_start_idx

def extract_bp_features(target_df):
    log = _get_logger()
    log.info("  [BP] Extracting strictly legal Ban/Pick sequence (No Post-Match Role Leakage)...")
    name_to_idx, _, vocab_size, special_tokens, champion_start_idx = load_champion_vocabulary()
    PAD_IDX = special_tokens["PAD"]
    UNK_IDX = special_tokens["UNK"]

    df = target_df.copy()

    # 【优化 3】：废弃耗时的 np.vectorize，直接使用 Pandas 的底层 C 映射 map().fillna()
    def _map_col(col_series):
        return col_series.astype(str).str.strip().map(name_to_idx).fillna(UNK_IDX).astype(np.int32)
    
    cols_to_drop = []
    for side in ["blue", "red"]:
        for pos in POSITIONS_SHORT:
            col = f"{side}_{pos}_champion"
            if col in df.columns:
                df[f"{side}_{pos}_champion_id"] = _map_col(df[col])
                # 名称列已编码为 _id，删除原始名称列减少 parquet 体积
                cols_to_drop.append(col)

        for i in range(1, 6):
            ban_col = f"{side}_ban{i}"
            if ban_col in df.columns:
                df[f"{side}_ban{i}_id"] = _map_col(df[ban_col])
                cols_to_drop.append(ban_col)
                
            pick_col = f"{side}_pick{i}"
            if pick_col in df.columns:
                df[f"{side}_pick{i}_id"] = _map_col(df[pick_col])
                cols_to_drop.append(pick_col)
                
    df.drop(columns=cols_to_drop, inplace=True)
    log.info(f"  [BP] Building BP sequence ({len(BP_SEQUENCE)} steps)...")

    bp_action_arr = np.zeros((len(df), len(BP_SEQUENCE)), dtype=np.int8)
    bp_side_arr = np.zeros((len(df), len(BP_SEQUENCE)), dtype=np.int8)
    bp_champ_arr = np.full((len(df), len(BP_SEQUENCE)), PAD_IDX, dtype=np.int32)

    for step_idx, (action_type, side, slot) in enumerate(BP_SEQUENCE):
        action_code = 0 if action_type == "ban" else 1
        side_code = 0 if side == "blue" else 1

        # 优先使用已编码的 _id 列（名称列已被删除），回退到名称列
        id_col = f"{side}_{action_type}{slot}_id"
        name_col = f"{side}_{action_type}{slot}"

        if id_col in df.columns:
            champ_ids = df[id_col].values.astype(np.int32)
        elif name_col in df.columns:
            champ_ids = _map_col(df[name_col]).values
        else:
            champ_ids = np.full(len(df), UNK_IDX, dtype=np.int32)

        bp_action_arr[:, step_idx] = action_code
        bp_side_arr[:, step_idx] = side_code
        bp_champ_arr[:, step_idx] = champ_ids

    for step_idx in range(len(BP_SEQUENCE)):
        df[f"bp_step{step_idx}_action"] = bp_action_arr[:, step_idx]
        df[f"bp_step{step_idx}_side"] = bp_side_arr[:, step_idx]
        df[f"bp_step{step_idx}_champion_id"] = bp_champ_arr[:, step_idx]

    return df

def assemble_features(target_df, team_profile_df):
    log = _get_logger()
    log.info("  [Assemble] Building Base Context DataFrame...")
    merged = target_df.copy()
    
    if team_profile_df is not None and not team_profile_df.empty:
        profile_cols = [c for c in team_profile_df.columns if c.startswith("team_")]
        # 防御性去重：确保 team_profile_df 中 (gameid, team) 唯一，避免 merge 后行数膨胀
        team_profile_df = team_profile_df.drop_duplicates(subset=["gameid", "team"], keep="last")
        for side in ["blue", "red"]:
            team_col = f"{side}_team"
            if team_col in merged.columns:
                merge_base = merged[["gameid", team_col]].rename(columns={team_col: "team"})
                profile_merged = merge_base.merge(team_profile_df[["gameid", "team"] + profile_cols], on=["gameid", "team"], how="left")
                # 再次防御：确保 merge 后行数与 merge_base 一致
                if len(profile_merged) != len(merge_base):
                    _log = logging.getLogger(__name__)
                    _log.warning(f"[Assemble] {side} profile merge length mismatch: {len(merge_base)} -> {len(profile_merged)}, dropping duplicates")
                    profile_merged = profile_merged.drop_duplicates(subset=["gameid", "team"], keep="last")
                    # 如果仍有缺失，用 merge_base 左连接保证长度
                    if len(profile_merged) != len(merge_base):
                        profile_merged = merge_base.merge(profile_merged, on=["gameid", "team"], how="left")
                
                for c in profile_cols:
                    merged[f"{side}_{c}"] = profile_merged[c].fillna(0).values
    


    # 删除 parquet 中两个模型均未使用的冗余列
    _drop_cols = []
    for c in ["result",
              "blue_golddiffat20", "blue_golddiffat25", "red_golddiffat20", "red_golddiffat25",
              "LCK", "LEC", "LPL", "league_LCK", "league_LEC", "league_LPL"]:
        if c in merged.columns:
            _drop_cols.append(c)
    if _drop_cols:
        merged.drop(columns=_drop_cols, inplace=True)

    # 【数据质量检测】特征拼接后空值检测
    _log = logging.getLogger(__name__)
    _log.info(f"[Assemble] 特征拼接完成, 形状={merged.shape}")
    # 检查 team_avg_golddiffat15 列 (LPL 应为 NaN, 已用 firsttower 替代)
    gold_cols = [c for c in merged.columns if 'golddiff' in c.lower()]
    for gc in gold_cols:
        na = merged[gc].isna().sum()
        total = len(merged)
        if na > 0:
            _log.info(f"  [Assemble] {gc}: {na}/{total} 空值 (LPL 缺失, firsttower 已作为替代特征)")
        else:
            _log.info(f"  [Assemble] {gc}: 0 空值 ✓")
    # 检查 firsttower 列 (应为 0/1 二值)
    ft_cols = [c for c in merged.columns if 'firsttower' in c.lower()]
    for fc in ft_cols:
        na = merged[fc].isna().sum()
        if na > 0:
            _log.warning(f"  [Assemble] {fc}: {na}/{len(merged)} 空值 (firsttower 不应缺失!)")

    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for c in numeric_cols:
        inf_mask = np.isinf(merged[c])
        inf_count += inf_mask.sum()
        if inf_mask.any():
            merged.loc[inf_mask, c] = np.nan
    
    if inf_count > 0:
        _log.warning(f"  [Assemble] 发现并处理 {inf_count} 个 Inf 值 (已转为NaN)")

    high_missing_cols = []
    for c in numeric_cols:
        na_pct = merged[c].isna().mean() * 100
        if na_pct > 30:
            high_missing_cols.append((c, f"{na_pct:.1f}%"))
    if high_missing_cols:
        _log.warning(f"  [Assemble] 高缺失率列 (>30%): {high_missing_cols}")

    team_feat_count = len([c for c in merged.columns if c.startswith("blue_team_") or c.startswith("red_team_")])
    bp_feat_count = len([c for c in merged.columns if c.startswith("bp_") or "pick" in c.lower() or "ban" in c.lower()])
    _log.info(f"  [Assemble] 特征分类统计: Team特征={team_feat_count}列, BP结构特征={bp_feat_count}列, 总列数={len(merged.columns)}")
    _log.info(f"  [Assemble] Context特征拼接完成: {merged.shape[0]} 行, {merged.shape[1]} 列")

    check_dataframe("merged_features", merged, _log, context="特征拼接后")

    return merged


def _parse_percentage_series(s):
    """将百分比字符串序列转换为 0-1 之间的小数，空值/无效值转为 NaN。"""
    if s is None:
        return pd.Series(dtype=float)
    s = s.astype(str).str.replace('%', '', regex=False).str.strip()
    s = pd.to_numeric(s, errors='coerce')
    # 自动识别百分制并转小数
    s = s.where(s <= 1.0, s / 100.0)
    return s


def _load_counter_synergy_dataframe(cleaned_dir, data_name):
    """加载 counter/synergy 数据，支持 cleaned 文件与 raw 文件回退。

    当前 cleaned_data 中的 champion_counters_cleaned.csv / champion_synergy_cleaned.csv
    存在 win_rate 列全部为空的问题，因此当 cleaned 文件 win_rate 无效时，
    自动回退到 raw_data 目录下的原始文件并解析百分比。

    Args:
        cleaned_dir: cleaned_data 目录路径
        data_name: "champion_counters" 或 "champion_synergy"

    Returns:
        pd.DataFrame: 标准化后的 DataFrame，列包括
            champion, opponent_name, games, win_rate, matchup_type
    """
    _log = logging.getLogger(__name__)
    cleaned_path = os.path.join(cleaned_dir, f"{data_name}_cleaned.csv")
    raw_dir = os.path.join(os.path.dirname(cleaned_dir), "raw_data")
    raw_subdir = "champion_counters" if data_name == "champion_counters" else "champion_duo_synergy"
    raw_path = os.path.join(raw_dir, raw_subdir, "champion_synergy.csv" if data_name == "champion_synergy" else "champion_counters.csv")

    df = None
    used_source = None
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path, low_memory=False)
        used_source = "cleaned"
        # 标准化列名
        col_map = {
            "champion_name": "champion",
            "opponent_games": "games",
            "opponent_win_rate": "win_rate",
        }
        df.columns = [col_map.get(c.lower().strip(), c.lower().strip()) for c in df.columns]
        if "win_rate" in df.columns:
            df["win_rate"] = _parse_percentage_series(df["win_rate"])
        # 检查 win_rate 是否有效
        wr_na = df["win_rate"].isna().sum() if "win_rate" in df.columns else len(df)
        wr_valid = df["win_rate"].notna().sum() if "win_rate" in df.columns else 0
        _log.info(f"[{data_name}] cleaned 数据加载: {len(df)} 行, win_rate 空值={wr_na}, 有效值={wr_valid}")
        if "win_rate" not in df.columns or wr_valid == 0:
            _log.error(f"[{data_name}] cleaned file win_rate 全部为空! 回退到 raw data: {raw_path}")
            df = None

    if df is None:
        if not os.path.exists(raw_path):
            _log.warning(f"[{data_name}] raw file not found at {raw_path}, returning empty DataFrame")
            return pd.DataFrame()
        df = pd.read_csv(raw_path, low_memory=False, encoding='utf-8-sig')
        used_source = "raw"
        col_map = {
            "champion_name": "champion",
            "opponent_games": "games",
            "opponent_win_rate": "win_rate",
        }
        df.columns = [col_map.get(c.lower().strip(), c.lower().strip()) for c in df.columns]
        if "win_rate" in df.columns:
            df["win_rate"] = _parse_percentage_series(df["win_rate"])

    # 确保必要列存在
    for col in ("champion", "opponent_name", "games", "win_rate"):
        if col not in df.columns:
            _log.warning(f"[{data_name}] missing column '{col}' in {used_source} data")
            return pd.DataFrame()

    # 过滤无效行：空值、games <= 0、win_rate 为空
    before = len(df)
    df = df.dropna(subset=["champion", "opponent_name", "games", "win_rate"])
    df = df[df["games"] > 0]
    df = df[df["win_rate"].between(0.0, 1.0)]
    after = len(df)
    if before != after:
        _log.info(f"[{data_name}] filtered {before - after} invalid rows from {used_source} data (remaining {after})")

    _log.info(f"[{data_name}] loaded from {used_source}: {len(df)} rows")
    return df


def build_counter_lookup_with_bayesian(cleaned_dir=None, bayesian_M=BAYESIAN_PRIOR_WEIGHT,
                                       clip_match_extreme=False):
    """从 champion_counters_cleaned.csv / 原始文件构建英雄克制关系查找表（贝叶斯平滑，先验固定 0.5）。

    处理 a-b / b-a 双向冲突：若两行 win_rate 之和不等于 1（超出容差），
    则保留 games 更大的那一行，反向关系由 1 - win_rate 推导。

    Returns:
        dict: {champion_name: {opponent_name: {"win_rate": float, "matchup_type": str}}}
    """
    _log = logging.getLogger(__name__)
    if cleaned_dir is None:
        cleaned_dir = CLEANED_DIR

    df = _load_counter_synergy_dataframe(cleaned_dir, "champion_counters")
    if df.empty:
        _log.warning("[Counter] no valid counter data available, returning empty dict")
        return {}

    PRIOR_WR = 0.5  # Counter 的先验胜率同样是 50-50 开
    CONFLICT_TOL = 0.02  # win_rate 之和与 1.0 的容差

    # ---- 第一遍：收集所有行，按 (champ, opponent) 建索引 ----
    raw_entries = {}  # key: (champ, opponent) -> dict
    for _, row in df.iterrows():
        champ = str(row["champion"]).strip()
        opponent = str(row["opponent_name"]).strip()
        raw_wr = float(row["win_rate"])
        matchup = str(row.get("matchup_type", "hard")).strip()
        games = int(row.get("games", 0))

        if clip_match_extreme and games < 5:
            continue  # 过滤低场次极端数据

        if clip_match_extreme:
            raw_wr = max(0.1, min(0.9, raw_wr))

        key = (champ, opponent)
        # 同方向重复时保留 games 更大的
        if key in raw_entries and raw_entries[key]["games"] >= games:
            continue
        raw_entries[key] = {
            "win_rate": raw_wr,
            "matchup_type": matchup,
            "games": games,
        }

    # ---- 第二遍：处理 a-b / b-a 冲突 ----
    resolved_entries = {}  # key: (champ, opponent) -> dict
    visited_pairs = set()
    conflict_count = 0

    for (champ, opponent), entry in raw_entries.items():
        pair_key = tuple(sorted([champ, opponent]))
        if pair_key in visited_pairs:
            continue
        visited_pairs.add(pair_key)

        fwd_key = (champ, opponent)
        rev_key = (opponent, champ)
        fwd = raw_entries.get(fwd_key)
        rev = raw_entries.get(rev_key)

        if fwd and rev:
            # 双向数据都存在，检查冲突
            wr_sum = fwd["win_rate"] + rev["win_rate"]
            if abs(wr_sum - 1.0) > CONFLICT_TOL:
                # 冲突：按 games 更大的行作为标准
                conflict_count += 1
                if fwd["games"] >= rev["games"]:
                    chosen, other = fwd, rev
                    chosen_key, other_key = fwd_key, rev_key
                else:
                    chosen, other = rev, fwd
                    chosen_key, other_key = rev_key, fwd_key

                resolved_entries[chosen_key] = {
                    "win_rate": chosen["win_rate"],
                    "matchup_type": chosen["matchup_type"],
                    "games": chosen["games"],
                }
                # 反向由 1 - win_rate 推导，matchup_type 翻转
                rev_matchup = "easy" if chosen["matchup_type"] == "hard" else "hard"
                resolved_entries[other_key] = {
                    "win_rate": round(1.0 - chosen["win_rate"], 6),
                    "matchup_type": rev_matchup,
                    "games": chosen["games"],
                }
            else:
                # 无冲突，双向都保留
                resolved_entries[fwd_key] = fwd
                resolved_entries[rev_key] = rev
        elif fwd:
            resolved_entries[fwd_key] = fwd
        elif rev:
            resolved_entries[rev_key] = rev

    # ---- 第三遍：贝叶斯平滑并构建 lookup ----
    lookup = {}
    for (champ, opponent), entry in resolved_entries.items():
        raw_wr = entry["win_rate"]
        matchup = entry["matchup_type"]
        games = entry["games"]

        if games > 0:
            wins = int(round(raw_wr * games))
            smoothed_wr = (wins + bayesian_M * PRIOR_WR) / (games + bayesian_M)
        else:
            smoothed_wr = raw_wr
            if clip_match_extreme:
                smoothed_wr = max(0.1, min(0.9, smoothed_wr))

        if champ not in lookup:
            lookup[champ] = {}

        lookup[champ][opponent] = {
            "win_rate": round(smoothed_wr, 6),
            "matchup_type": matchup
        }

    _log.info(f"  [Counter] Built counter_lookup: {len(lookup)} champions processed, "
             f"bayesian_M={bayesian_M}, clip={clip_match_extreme}, "
             f"conflicts_resolved={conflict_count}")
    return lookup

def build_synergy_lookup_with_bayesian(cleaned_dir=None, bayesian_M=BAYESIAN_PRIOR_WEIGHT,
                                        clip_match_extreme=False):
    """从 champion_synergy_cleaned.csv / 原始文件构建英雄协同关系查找表（贝叶斯平滑，先验固定 0.5）。"""
    _log = logging.getLogger(__name__)
    if cleaned_dir is None:
        cleaned_dir = CLEANED_DIR

    df = _load_counter_synergy_dataframe(cleaned_dir, "champion_synergy")
    if df.empty:
        _log.warning("[Synergy] no valid synergy data available, returning empty dict")
        return {}

    match_agg = {}
    for _, row in df.iterrows():
        c1 = str(row["champion"]).strip()
        c2 = str(row["opponent_name"]).strip()
        if not c1 or not c2:
            continue
        key = tuple(sorted([c1, c2]))
        games = int(row.get("games", 0))
        wr = float(row.get("win_rate", 0.5))
        if games <= 0 or not (0.0 <= wr <= 1.0):
            continue
        wins = int(round(wr * games))
        if key not in match_agg:
            match_agg[key] = {"wins": 0, "games": 0}
        match_agg[key]["wins"] += wins
        match_agg[key]["games"] += games

    # 极端值处理: 过滤 games < 5 的组合，clip win_rate 到 [0.1, 0.9]
    if clip_match_extreme:
        filtered = {}
        for key, info in match_agg.items():
            if info["games"] < 5:
                continue
            wr = info["wins"] / max(info["games"], 1)
            wr = max(0.1, min(0.9, wr))
            filtered[key] = {"wins": int(round(wr * info["games"])), "games": info["games"]}
        match_agg = filtered

    # 贝叶斯平滑：先验固定 0.5
    PRIOR_WR = 0.5
    bayesian_synergy = {}
    for key, info in match_agg.items():
        m_wins = info["wins"]
        m_games = info["games"]
        smoothed_wr = (m_wins + bayesian_M * PRIOR_WR) / (m_games + bayesian_M)
        bayesian_synergy[key] = round(smoothed_wr, 6)

    _log.info(f"  [Synergy] Built synergy_lookup: {len(bayesian_synergy)} pairs, "
             f"bayesian_M={bayesian_M}, clip={clip_match_extreme}")
    return bayesian_synergy


def build_team_grudge_store(matches_df, name_to_idx):
    from collections import defaultdict
    matches_sorted = matches_df.sort_values("match_seq_idx").reset_index(drop=True)

    ban_records = []
    game_pairs = []
    for _, row in matches_sorted.iterrows():
        ms_idx = row["match_seq_idx"]
        bt = str(row.get("blue_team", "")).strip()
        rt = str(row.get("red_team", "")).strip()
        game_pairs.extend([(ms_idx, bt, rt), (ms_idx, rt, bt)])
        for side in ["blue", "red"]:
            for i in range(1, 6):
                col = f"{side}_ban{i}"
                champ_name = row.get(col, "")
                if pd.isna(champ_name) or str(champ_name).strip() == "" or str(champ_name).strip() == "Unknown":
                    continue
                cid = name_to_idx.get(str(champ_name).strip(), -1)
                if cid >= 0:
                    banning_team = bt if side == "blue" else rt
                    opponent_team = rt if side == "blue" else bt
                    ban_records.append((ms_idx, banning_team, opponent_team, cid))

    ban_df = pd.DataFrame(ban_records, columns=["match_seq_idx", "banning_team", "opponent_team", "champion_id"])
    ban_df = ban_df.sort_values("match_seq_idx").reset_index(drop=True)

    game_df = pd.DataFrame(game_pairs, columns=["match_seq_idx", "team_a", "team_b"])
    game_df = game_df.sort_values("match_seq_idx").reset_index(drop=True)

    ban_counts = defaultdict(int)
    game_counts = defaultdict(int)
    pair_to_champs = defaultdict(set)
    
    ban_idx = 0
    game_idx = 0
    n_ban, n_game = len(ban_df), len(game_df)

    # 【优化 4】：将用于 while 扫描的数据彻底转为 Numpy Arrays
    ban_seq_ids = ban_df["match_seq_idx"].values
    ban_bt = ban_df["banning_team"].values
    ban_ot = ban_df["opponent_team"].values
    ban_cids = ban_df["champion_id"].values.astype(int)

    game_seq_ids = game_df["match_seq_idx"].values
    game_ta = game_df["team_a"].values
    game_tb = game_df["team_b"].values

    grudge_store = {}
    
    for _, match_row in matches_sorted.iterrows():
        target_seq_idx = match_row["match_seq_idx"]
        gameid = str(match_row["gameid"]) 
        current_bt = str(match_row.get("blue_team", "")).strip()
        current_rt = str(match_row.get("red_team", "")).strip()

        # Numpy 索引步进，速度提升百倍
        while ban_idx < n_ban and ban_seq_ids[ban_idx] < target_seq_idx:
            bt, ot, cid = ban_bt[ban_idx], ban_ot[ban_idx], ban_cids[ban_idx]
            ban_counts[(bt, ot, cid)] += 1
            pair_to_champs[(bt, ot)].add(cid)
            ban_idx += 1

        while game_idx < n_game and game_seq_ids[game_idx] < target_seq_idx:
            ta, tb = game_ta[game_idx], game_tb[game_idx]
            game_counts[(ta, tb)] += 1
            game_idx += 1

        date_dict = {}
        for bt, ot in [(current_bt, current_rt), (current_rt, current_bt)]:
            if not bt or not ot: 
                continue
            n_games = game_counts.get((bt, ot), 0)
            if n_games > 0:
                for cid in pair_to_champs.get((bt, ot), set()):
                    count = ban_counts.get((bt, ot, cid), 0)
                    rate = min(count / n_games, 1.0)
                    if rate > 0:
                        bt_dict = date_dict.setdefault(bt, {})
                        ot_dict = bt_dict.setdefault(ot, {})
                        ot_dict[int(cid)] = round(rate, 6)

        if date_dict:
            grudge_store[gameid] = date_dict

    # 提取所有队伍对的最新恩怨快照，供线上无 gameid 推理使用    
    latest_grudge_store = {}
    for (bt, ot), cids in pair_to_champs.items():
        n_games = game_counts.get((bt, ot), 0)
        if n_games > 0:
            for cid in cids:
                count = ban_counts.get((bt, ot, cid), 0)
                rate = min(count / n_games, 1.0)
                if rate > 0:
                    bt_dict = latest_grudge_store.setdefault(bt, {})
                    ot_dict = bt_dict.setdefault(ot, {})
                    ot_dict[int(cid)] = round(rate, 6)

    return grudge_store, latest_grudge_store

def build_player_respect_store(player_features_df, name_to_idx):
    pf = player_features_df.merge(load_matches()[["gameid", "match_seq_idx"]], on="gameid", how="left")
    pf = pf.sort_values("match_seq_idx").reset_index(drop=True)

    from collections import defaultdict
    mastery_accum = defaultdict(lambda: defaultdict(float))

    game_players = pf.groupby('gameid')['player_id'].unique().to_dict()
    respect_store = {}
    row_idx = 0
    n_rows = len(pf)

    unique_matches = pf[['gameid', 'match_seq_idx']].drop_duplicates().sort_values('match_seq_idx')
    
    # 【优化】：提前转换为 Numpy 数组，彻底干掉 .iloc 的性能瓶颈
    pf_seq_ids = pf["match_seq_idx"].values
    pf_pids = pf["player_id"].values
    pf_cids = pf["champion_id"].values.astype(int)
    pf_masteries = pf["mastery_score"].values.astype(float)

    for _, match_row in unique_matches.iterrows():
        target_seq_idx = match_row["match_seq_idx"]
        gameid = str(match_row["gameid"])
        current_players = game_players.get(gameid, [])

        while row_idx < n_rows and pf_seq_ids[row_idx] < target_seq_idx:
            pid = pf_pids[row_idx]
            cid = pf_cids[row_idx]
            ms = pf_masteries[row_idx]
            if ms > mastery_accum[pid][cid]:
                mastery_accum[pid][cid] = ms
            row_idx += 1

        date_dict = {}
        for pid in current_players:
            champ_dict = mastery_accum.get(pid, {})
            if not champ_dict:
                continue
            best_cid = max(champ_dict, key=champ_dict.get)
            date_dict[pid] = {
                "signature_champion_id": int(best_cid),
                "signature_mastery": float(min(champ_dict[best_cid], 100.0)),
            }

        if date_dict:
            respect_store[gameid] = date_dict
    
    # 提取每个选手的最新绝活快照
    latest_respect_store = {}
    for pid, champ_dict in mastery_accum.items():
        if not champ_dict:
            continue
        best_cid = max(champ_dict, key=champ_dict.get)
        latest_respect_store[pid] = {
            "signature_champion_id": int(best_cid),
            "signature_mastery": float(min(champ_dict[best_cid], 100.0)),
        }

    return respect_store, latest_respect_store

def build_enemy_hot_streak_store(player_history, name_to_idx, window_days=14, min_games=2):
    ph = player_history.sort_values("match_seq_idx").reset_index(drop=True)
    ph["date"] = pd.to_datetime(ph["date"])

    from collections import defaultdict
    streak_accum = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "games": 0, "kda_sum": 0.0}))

    game_players = ph.groupby('gameid')['player_id'].unique().to_dict()
    streak_store = {}

    unique_matches = ph[["gameid", "date", "match_seq_idx"]].drop_duplicates().sort_values("match_seq_idx")
    match_dates = unique_matches["date"].values.astype("datetime64[ns]")
    match_seq_ids = unique_matches["match_seq_idx"].values
    match_ids = unique_matches["gameid"].values

    ph_seq_ids = ph["match_seq_idx"].values
    ph_pids = ph["player_id"].values
    ph_champs = ph["champion"].values
    ph_results = ph["result"].values.astype(float)
    ph_kills = ph["kills"].values.astype(float)
    ph_deaths = ph["deaths"].values.astype(float)
    ph_assists = ph["assists"].values.astype(float)

    row_idx = 0
    n_rows = len(ph)

    for mi in range(len(match_ids)):
        target_seq_idx = match_seq_ids[mi]
        gameid = str(match_ids[mi])
        current_players = game_players.get(gameid, [])

        # 【修复 4】：防同日泄漏
        while row_idx < n_rows and ph_seq_ids[row_idx] < target_seq_idx:
            pid = ph_pids[row_idx]
            champ = ph_champs[row_idx]
            res = ph_results[row_idx]
            k = ph_kills[row_idx]
            d = ph_deaths[row_idx]
            a = ph_assists[row_idx]
            kda = (k + a) / max(d, 1)

            entry = streak_accum[pid][champ]
            entry["wins"] += res
            entry["games"] += 1
            entry["kda_sum"] += kda
            row_idx += 1

        date_dict = {}
        for pid in current_players:
            champ_dict = streak_accum.get(pid, {})
            hot_champs = []
            for champ, stats in champ_dict.items():
                if stats["games"] >= min_games:
                    wr = stats["wins"] / stats["games"]
                    avg_kda = stats["kda_sum"] / stats["games"]
                    hot_champs.append((champ, wr, avg_kda, stats["games"]))
            
            if hot_champs:
                hot_champs.sort(key=lambda x: (x[1], x[2]), reverse=True)
                best = hot_champs[0]
                cid = name_to_idx.get(str(best[0]).strip(), -1)
                if cid >= 0:
                    date_dict[pid] = {
                        "hot_champion_id": cid,
                        "hot_win_rate": round(best[1], 4),
                        "hot_avg_kda": round(best[2], 4),
                        "hot_games": int(best[3]),
                    }

        if date_dict:
            streak_store[gameid] = date_dict

    
    # 【新增】提取每个选手的最新火热状态快照
    latest_streak_store = {}
    for pid, champ_dict in streak_accum.items():
        hot_champs = []
        for champ, stats in champ_dict.items():
            if stats["games"] >= min_games:
                wr = stats["wins"] / stats["games"]
                avg_kda = stats["kda_sum"] / stats["games"]
                hot_champs.append((champ, wr, avg_kda, stats["games"]))
        
        if hot_champs:
            hot_champs.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best = hot_champs[0]
            cid = name_to_idx.get(str(best[0]).strip(), -1)
            if cid >= 0:
                latest_streak_store[pid] = {
                    "hot_champion_id": cid,
                    "hot_win_rate": round(best[1], 4),
                    "hot_avg_kda": round(best[2], 4),
                    "hot_games": int(best[3]),
                }
    
    return streak_store, latest_streak_store


def build_feature_pipeline(league=None, save_intermediate=True):
    label = league if league else "ALL"
    log = setup_pipeline_logger()

    t_total_start = time.time()
    log.info("=" * 60)
    log.info(f"[Pipeline] Building PIT feature pipeline for {label}")
    log.info("=" * 60)

    t0 = time.time()
    matches_df = load_matches(league)
    career_df = load_career_stats(league)

    # 【数据质量检测】检查原始数据的关键列空值情况
    log.info("[DataQuality] 原始数据空值检测:")
    # matches 关键列
    for col in ['gameid', 'league', 'blue_team', 'red_team', 'result', 'date',
                'blue_golddiffat15', 'red_golddiffat15', 'blue_firsttower', 'red_firsttower']:
        if col in matches_df.columns:
            na = matches_df[col].isna().sum()
            total = len(matches_df)
            if na > 0:
                log.warning(f"  matches.{col}: {na}/{total} 空值 ({na/total*100:.1f}%)")
            else:
                log.info(f"  matches.{col}: 0 空值 ✓")
    # career 关键列
    if not career_df.empty:
        for col in ['player_id', 'champion', 'win_rate', 'games']:
            if col in career_df.columns:
                na = career_df[col].isna().sum()
                total = len(career_df)
                if na > 0:
                    log.error(f"  career.{col}: {na}/{total} 空值 ({na/total*100:.1f}%) ← 严重问题!")
                else:
                    log.info(f"  career.{col}: 0 空值 ✓")

    t0 = time.time()
    target_df, result_df = enforce_pit(matches_df)

    t0 = time.time()
    target_df = extract_bp_features(target_df)

    t0 = time.time()
    player_history = melt_matches_to_player_rows(matches_df)
    player_history = player_history.sort_values("match_seq_idx").reset_index(drop=True)
    log.info(f"  player_history (选手视角): {len(player_history)} 行, 唯一选手={player_history['player_id'].nunique()}, 唯一英雄={player_history['champion'].nunique()}")

    t0 = time.time()
    ban_history = melt_bans_from_matches(matches_df)
    log.info(f"  ban_history (Ban记录): {len(ban_history)} 行")

    t0 = time.time()
    base_prior_df = build_base_prior(player_history, career_df)
    log.info(f"  base_prior (选手英雄先验): {len(base_prior_df)} 行")

    # ---- player_career 数据泄漏检查 ----
    # 验证剥离逆计算后 Base_G 占比合理（避免 career 数据包含测试期比赛）
    #
    # 注：career 数据包含所有历史赛季，训练数据仅包含2025+子集，
    # 因此 (Base+Train)/Career 可能 < 1.0 是正常的（职业生涯早于训练窗口）
    # 只有当 > 1.1 时才表明 career 数据可能缺少某些训练期选手
    total_base_g = base_prior_df["Base_G"].sum()
    total_career_g = career_df["games"].sum() if not career_df.empty else 0
    total_train_g = player_history.groupby(["player_id", "champion"])["gameid"].count().sum()
    if not career_df.empty:
        leakage_ratio = (total_base_g + total_train_g) / total_career_g if total_career_g > 0 else 0
        log.info(f"  [LeakageCheck] Career_G={total_career_g}, Training_PlayerChamp_G={total_train_g}, Base_G={total_base_g}, "
                 f"(Base+Train)/Career = {leakage_ratio:.4f}")
        if leakage_ratio > 1.10:
            log.warning(f"  [LeakageCheck] (Base+Train)/Career > 1.10，player_career 可能不包含全部训练期选手！")

    t0 = time.time()
    champ_meta_daily = compute_meta_features_pit(player_history, ban_history, matches_df)

    t0 = time.time()
    player_features_df = compute_player_features_pit(player_history, base_prior_df)

    t0 = time.time()
    pick_dist_df = compute_player_pick_distribution(player_history)
    log.info(f"  pick_dist (选手英雄池分布): {len(pick_dist_df)} 行")

    t0 = time.time()
    team_profile_df = compute_team_profile_pit(matches_df)

    t0 = time.time()
    context_df = assemble_features(target_df, team_profile_df)
    log.info(f"  context_df (基础上下文特征): {len(context_df)} 行, {len(context_df.columns)} 列")
    
    if save_intermediate:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        context_df.to_parquet(os.path.join(OUTPUT_DIR, f"{label}_context.parquet"), index=False)

        name_to_idx, _, _, _, _ = load_champion_vocabulary()
        _map_name = lambda n: name_to_idx.get(str(n).strip(), 1)
        champ_meta_daily["champion_id"] = champ_meta_daily["champion"].map(_map_name).astype(np.int32)
        player_features_df["champion_id"] = player_features_df["champion"].map(_map_name).astype(np.int32)

        champ_meta_daily.to_parquet(os.path.join(OUTPUT_DIR, f"{label}_meta_store.parquet"), index=False)
        player_features_df.to_parquet(os.path.join(OUTPUT_DIR, f"{label}_player_store.parquet"), index=False)

        grudge_store, latest_grudge_store = build_team_grudge_store(matches_df, name_to_idx)
        import json as _json
        with open(os.path.join(OUTPUT_DIR, f"{label}_grudge_store.json"), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json(grudge_store), f, ensure_ascii=False)
        with open(os.path.join(OUTPUT_DIR, f"{label}_serving_latest_grudge.json"), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json(latest_grudge_store), f, ensure_ascii=False)
        log.info(f"  [Store] grudge_store: {len(grudge_store)} 支战队, latest快照={len(latest_grudge_store)} 支战队")

        respect_store, latest_respect_store = build_player_respect_store(player_features_df, name_to_idx)
        with open(os.path.join(OUTPUT_DIR, f"{label}_respect_store.json"), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json(respect_store), f, ensure_ascii=False)
        with open(os.path.join(OUTPUT_DIR, f"{label}_serving_latest_respect.json"), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json(latest_respect_store), f, ensure_ascii=False)
        log.info(f"  [Store] respect_store: {len(respect_store)} 名选手, latest快照={len(latest_respect_store)} 名选手")

        hot_streak_store, latest_streak_store = build_enemy_hot_streak_store(player_history, name_to_idx)
        with open(os.path.join(OUTPUT_DIR, f"{label}_hot_streak_store.json"), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json(hot_streak_store), f, ensure_ascii=False)
        with open(os.path.join(OUTPUT_DIR, f"{label}_serving_latest_hot_streak.json"), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json(latest_streak_store), f, ensure_ascii=False)
        log.info(f"  [Store] hot_streak_store: {len(hot_streak_store)} 名选手, latest快照={len(latest_streak_store)} 名选手")

        counter_lookup = build_counter_lookup_with_bayesian(CLEANED_DIR, bayesian_M=BAYESIAN_PRIOR_WEIGHT, clip_match_extreme=True)
        with open(os.path.join(OUTPUT_DIR, f"{label}_counter_lookup.json"), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json(counter_lookup), f, ensure_ascii=False)
        counter_pairs = sum(len(v) for v in counter_lookup.values())
        log.info(f"  [Store] counter_lookup: {len(counter_lookup)} 个英雄, {counter_pairs} 对克制关系")

        synergy_lookup = build_synergy_lookup_with_bayesian(CLEANED_DIR, bayesian_M=BAYESIAN_PRIOR_WEIGHT,
                                                            clip_match_extreme=True)
        synergy_filename = f"{label}_synergy_lookup.json"
        with open(os.path.join(OUTPUT_DIR, synergy_filename), "w", encoding="utf-8") as f:
            _json.dump(_sanitize_for_json({f"{k[0]}||{k[1]}": v for k, v in synergy_lookup.items()}), f, ensure_ascii=False)
        log.info(f"  [Store] synergy_lookup: {len(synergy_lookup)} 对协同关系 (bayesian_M={BAYESIAN_PRIOR_WEIGHT}, clip=True)")

        _save_feature_manifest(label, OUTPUT_DIR, matches_df, context_df, champ_meta_daily, player_features_df)

    t_total = time.time() - t_total_start
    log.info("=" * 60)
    log.info(f"[Pipeline] 推荐特征管道构建完成! 总耗时: {t_total:.1f}s")
    log.info(f"[Pipeline] 输出统计: context={len(context_df)}行, meta={len(champ_meta_daily)}行, player={len(player_features_df)}行")
    log.info("=" * 60)
    return context_df, champ_meta_daily, player_features_df


def _save_feature_manifest(label, output_dir, matches_df, context_df, champ_meta_daily, player_features_df):
    """生成特征文件清单 manifest.json，记录所有输出文件及关键统计信息。"""
    import json as _json
    _log = logging.getLogger(__name__)
    manifest = {
        "label": label,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_window": {
            "start_date": str(matches_df["date"].min().date()) if not matches_df.empty else None,
            "end_date": str(matches_df["date"].max().date()) if not matches_df.empty else None,
            "total_matches": int(len(matches_df)),
        },
        "files": {},
        "statistics": {
            "context_rows": int(len(context_df)),
            "meta_rows": int(len(champ_meta_daily)),
            "player_rows": int(len(player_features_df)),
            "unique_champions": int(champ_meta_daily["champion_id"].nunique()) if "champion_id" in champ_meta_daily.columns else 0,
        },
    }

    # 记录所有输出文件及其大小
    expected_files = [
        f"{label}_context.parquet",
        f"{label}_meta_store.parquet",
        f"{label}_player_store.parquet",
        f"{label}_grudge_store.json",
        f"{label}_serving_latest_grudge.json",
        f"{label}_respect_store.json",
        f"{label}_serving_latest_respect.json",
        f"{label}_hot_streak_store.json",
        f"{label}_serving_latest_hot_streak.json",
        f"{label}_counter_lookup.json",
        f"{label}_synergy_lookup.json",
    ]
    for fname in expected_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            manifest["files"][fname] = {
                "size_bytes": os.path.getsize(fpath),
                "exists": True,
            }
        else:
            manifest["files"][fname] = {"exists": False}

    manifest_path = os.path.join(output_dir, f"{label}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        _json.dump(manifest, f, indent=2, ensure_ascii=False)
    _log.info(f"  Feature manifest saved -> {label}_manifest.json")

if __name__ == "__main__":
    setup_logging()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None, choices=["LPL", "LCK", "LEC", "ALL"])
    args = parser.parse_args()
    league_arg = None if args.league == "ALL" else args.league
    build_feature_pipeline(league=league_arg)
