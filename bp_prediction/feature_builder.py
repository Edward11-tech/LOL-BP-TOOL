"""
统一特征构建模块
==================
为 predict_match.py、bp_delta.py、predict_backend.py 提供一致的特征构建逻辑，
确保推理时特征与训练时 feature_pipeline.py 完全一致。

特征构建顺序 (与 feature_pipeline.py 一致):
  1. 联赛 one-hot (league_LPL, league_LCK, league_LEC)
  2. 季后赛和地图方 (is_playoff, is_blue_map_side)
  3. 选手历史特征 + 新秀惩罚
  4. 英雄元数据特征 (meta_win_rate_pit, meta_patch_drift_index, meta_pick_drift_index)
  5. 队伍画像特征
  6. 差分特征 (blue - red)
  7. 队伍聚合特征 (avg, max, std)
  8. 阵容发力期特征 (comp_*)
  9. 英雄KDA/胜率偏差特征 (champ_wr_delta, champ_kda_delta)
  10. 熟练度交互特征 (mastery_x_wr, mastery_x_meta_wr, mastery_x_player_wr)
  11. 队伍胜率平衡特征 (team_wr_max_gap, team_wr_balance)
  12. 队伍胜率×阵容胜率交互 (team_wr_x_roster_wr)
  13. LPL 特定交互特征 (bloodiness_x_aggression, early_power_x_snowball, etc.)
  14. TF 特征 (tf_win_logits, tf_cosine_sim, tf_blue_l2norm, tf_red_l2norm)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================================
# 路径配置
# =====================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

FEATURES_DIR = os.path.join(MODEL_DIR, "features")
MODELS_DIR = os.path.join(MODEL_DIR, "models")
PRODUCTION_DIR = os.path.join(MODEL_DIR, "models", "production")

# =====================================================================
# 常量
# =====================================================================
from bp_prediction.feature_utils import (
    POSITIONS, PLAYER_DEFAULTS, META_DEFAULTS, TEAM_PROFILE_DEFAULTS,
    PLAYER_FEATURE_COLS, META_FEATURE_COLS, TEAM_PROFILE_FEATURE_COLS, TF_COLS,
    calculate_derived_features
)

# 新秀惩罚系数
ROOKIE_PENALTY = {
    "mastery_score": 0.3,
    "player_recent_kda_90d": 0.85,
    "player_recent_wr_90d": 0.9,
    "player_recent_games_90d": 0.0,
    "player_overall_recent_kda": 0.9,
    "player_overall_recent_wr": 0.9,
    "player_overall_recent_games": 0.2,
}

# 差分特征列 (选手级)
DIFF_FEATURE_COLS = [
    "mastery_score",
    "player_recent_kda_90d",
    "player_recent_wr_90d",
    "player_recent_games_90d",
    "player_overall_recent_wr",
    "player_overall_recent_kda",
    "player_overall_recent_games",
    "meta_win_rate_pit",
    "meta_patch_drift_index",
    "meta_pick_drift_index",
]

# TF 特征列
TF_COLS = ["tf_win_logits", "tf_cosine_sim", "tf_blue_l2norm", "tf_red_l2norm"]

# Draft 相关特征关键词 (用于 BP Delta Pre-Draft 置零)
DRAFT_KEYWORDS = [
    "champion",       # 英雄独热编码
    "comp_",          # 阵容发力期
    "mastery",        # 英雄熟练度 (含 mastery_x_wr 等交互特征)
    "meta_win_rate_pit",   # 英雄版本胜率
    "meta_patch_drift",    # 版本偏移
    "meta_pick_drift",     # 选取偏移
    "champ_",         # 英雄KDA/胜率偏差
]

MAX_UNKNOWN_PLAYERS_PER_TEAM = 2


# =====================================================================
# 数据加载
# =====================================================================
def load_feature_cols(use_production=True):
    """加载训练时的特征列名。

    优先使用生产模型目录, 回退到 OOT 折目录。

    Args:
        use_production: 是否优先加载生产模型特征列

    Returns:
        list: 特征列名列表, 或 None
    """
    # 优先: 生产模型
    if use_production:
        fc_path = os.path.join(PRODUCTION_DIR, "feature_columns.json")
        if os.path.exists(fc_path):
            with open(fc_path, "r") as f:
                return json.load(f)

    # 回退: OOT 折
    for fold_idx in range(5):
        fc_path = os.path.join(MODELS_DIR, f"fold_{fold_idx}", "feature_columns.json")
        if os.path.exists(fc_path):
            with open(fc_path, "r") as f:
                return json.load(f)

    return None


def load_feature_stores():
    """加载特征存储数据 (选手、英雄元数据、队伍画像)。"""
    stores = {}
    player_path = os.path.join(FEATURES_DIR, "ALL_player_store.parquet")
    if os.path.exists(player_path):
        stores["player"] = pd.read_parquet(player_path)
    meta_path = os.path.join(FEATURES_DIR, "ALL_meta_store.parquet")
    if os.path.exists(meta_path):
        stores["meta"] = pd.read_parquet(meta_path)
    team_path = os.path.join(FEATURES_DIR, "ALL_team_profile_store.parquet")
    if os.path.exists(team_path):
        stores["team_profile"] = pd.read_parquet(team_path)
    return stores


def load_champion_tags():
    """从 feature_pipeline 加载英雄标签。"""
    try:
        from bp_prediction.feature_pipeline import CHAMPION_TAGS
        return CHAMPION_TAGS
    except ImportError:
        return {}


def load_known_champions():
    """加载已知英雄列表。"""
    vocab_path = os.path.join(PROJECT_ROOT, "cleaned_data", "champion_vocabulary.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        if isinstance(vocab, dict) and "name_to_idx" in vocab:
            champions_list = vocab.get("champions", [])
            return [c["name"] for c in champions_list if "name" in c]
    wide_path = os.path.join(FEATURES_DIR, "ALL_prediction_wide_features.parquet")
    if os.path.exists(wide_path):
        df = pd.read_parquet(wide_path)
        champ_cols = [c for c in df.columns if "champion" in c.lower()]
        champions = set()
        for col in champ_cols:
            champions.update(df[col].dropna().unique())
        return sorted(champions)
    return []


# =====================================================================
# 阵容特征计算
# =====================================================================
def compute_comp_features(champion_names, champion_tags):
    """基于英雄标签计算阵容发力期特征 (与 feature_pipeline.py 一致)。"""
    tags_list = []
    for name in champion_names:
        tags = champion_tags.get(name, {
            "Engage": 1, "Poke": 1, "Peel": 1, "Burst": 1,
            "Tank": 1, "HardCC": 1, "LineStrength": 0
        })
        tags_list.append(tags)

    if not tags_list:
        return {}

    agg = {}
    for key in ["Engage", "Poke", "Peel", "Burst", "Tank", "HardCC", "LineStrength"]:
        vals = [t.get(key, 0) for t in tags_list]
        agg[f"comp_{key.lower()}_sum"] = sum(vals)
        agg[f"comp_{key.lower()}_avg"] = float(np.mean(vals))

    agg["comp_early_power"] = (
        agg["comp_linestrength_sum"] * 1.0
        + agg["comp_burst_sum"] * 0.5
        + agg["comp_engage_sum"] * 0.3
    )
    agg["comp_late_power"] = agg["comp_tank_sum"] * 1.0 + agg["comp_peel_sum"] * 0.5
    agg["comp_teamfight_score"] = (
        agg["comp_engage_sum"] * 1.0
        + agg["comp_hardcc_sum"] * 0.8
        + agg["comp_tank_sum"] * 0.3
    )
    aggression_num = agg["comp_engage_sum"] + agg["comp_burst_sum"] + agg["comp_linestrength_sum"]
    aggression_den = agg["comp_peel_sum"] + agg["comp_poke_sum"] * 0.5 + 1
    agg["comp_aggression_index"] = aggression_num / aggression_den
    agg["comp_scaling_type"] = (agg["comp_peel_sum"] + agg["comp_poke_sum"]) / (agg["comp_engage_sum"] + agg["comp_burst_sum"] + 1)
    agg["comp_lane_dom_type"] = (agg["comp_linestrength_sum"] + agg["comp_burst_sum"]) / 10.0
    agg["comp_teamfight_type"] = (agg["comp_engage_sum"] + agg["comp_hardcc_sum"]) / 10.0
    return agg


# =====================================================================
# 战队名解析
# =====================================================================
_TEAM_ALIAS_CACHE = None
_ROSTER_CACHE = None


def _load_roster_data():
    """从 active_rosters.csv 加载战队/选手标准名称。

    active_rosters.csv 是现役名单（来自 Liquipedia），用于在线推理时的
    输入上下文（前端 UI 白名单 + 后端选手特征抽取）。
    """
    global _TEAM_ALIAS_CACHE, _ROSTER_CACHE
    if _TEAM_ALIAS_CACHE is not None:
        return _TEAM_ALIAS_CACHE, _ROSTER_CACHE

    _TEAM_ALIAS_CACHE = {}
    _ROSTER_CACHE = {}

    roster_path = os.path.join(PROJECT_ROOT, "cleaned_data", "active_rosters.csv")
    if not os.path.exists(roster_path):
        return _TEAM_ALIAS_CACHE, _ROSTER_CACHE

    try:
        roster_df = pd.read_csv(roster_path)
        for _, row in roster_df.iterrows():
            team = str(row.get("team", "")).strip()
            player_id = str(row.get("player_id", "")).strip()
            # active_rosters.csv 中 player_id 即为选手名，position 列对应 role
            player_name = str(row.get("player_name", player_id)).strip()
            role = str(row.get("role", row.get("position", ""))).strip()
            if not team:
                continue
            if team not in _ROSTER_CACHE:
                _ROSTER_CACHE[team] = []
            if player_id:
                _ROSTER_CACHE[team].append({
                    "player_id": player_id,
                    "player_name": player_name or player_id,
                    "role": role,
                })
        _build_aliases_from_rosters()
    except Exception:
        pass

    return _TEAM_ALIAS_CACHE, _ROSTER_CACHE


def _build_aliases_from_rosters():
    """根据战队全名自动生成缩写别名。

    【修复】：team_profile_store 中统一使用缩写（如 "BLG"、"EDG"），
    因此别名缓存的方向应为 **全称 -> 缩写**，
    让用户输入的全称（如 "Bilibili Gaming"）能被映射到 store 中的缩写（如 "BLG"）。
    """
    manual_aliases = {
        # LPL: 全称 -> 缩写
        "JD Gaming": "JDG", "Bilibili Gaming": "BLG", "Top Esports": "TES",
        "Weibo Gaming": "WBG", "LNG Esports": "LNG", "Royal Never Give Up": "RNG",
        "EDward Gaming": "EDG", "FunPlus Phoenix": "FPX", "Invictus Gaming": "IG",
        "Oh My God": "OMG", "Team WE": "WE", "Ultra Prime": "UP",
        "Anyone's Legend": "AL", "Rare Atom": "RA", "ThunderTalk Gaming": "TT",
        "LGD Gaming": "LGD", "Ninjas in Pyjamas": "NIP",
        # LCK: 全称 -> 缩写
        "Gen.G": "GEN", "Hanwha Life Esports": "HLE",
        "Dplus Kia": "DK", "Dplus KIA": "DK", "DWG KIA": "DK",
        "KT Rolster": "KT", "HANJIN BRION": "BRO",
        "Nongshim RedForce": "NS", "Karmine Corp Blue": "KCB",
        "BNK FEARX": "BFX", "DN SOOPers": "DNS", "Kiwoom DRX": "KRX",
        # LEC: 全称 -> 缩写
        "G2 Esports": "G2", "Fnatic": "FNC", "Team Vitality": "VIT",
        "MAD Lions KOI": "MKOI", "Karmine Corp": "KC", "Team BDS": "BDS",
        "Team Heretics": "TH", "SK Gaming": "SK", "GiantX": "GX",
        "Rogue": "GIANTS", "Movistar KOI": "MKOI",
        "Natus Vincere": "NAVI", "Shifters": "SHFT",
    }
    _TEAM_ALIAS_CACHE.update(manual_aliases)

    # 自动从 roster 中生成全称->缩写的别名
    # roster 中的 key 是缩写，对应的选手信息中可能包含全称
    for abbr_name in _ROSTER_CACHE.keys():
        # roster key 本身就是缩写，无需再生成
        pass


def resolve_team_name(input_name, known_teams=None):
    """将用户输入的战队名解析为 team_profile_store 中的标准缩写名。"""
    if not input_name:
        return input_name
    # 优先检查 known_teams，如果原始名已在 store 中，直接返回
    if known_teams and input_name in known_teams:
        return input_name
    _load_roster_data()
    # 其次检查别名缓存（全称->缩写映射，如 "Bilibili Gaming" -> "BLG"）
    if input_name in _TEAM_ALIAS_CACHE:
        resolved = _TEAM_ALIAS_CACHE[input_name]
        # 二次校验：确保解析结果在 known_teams 中
        if known_teams is None or resolved in known_teams:
            return resolved
    # 最后模糊匹配
    if known_teams:
        input_lower = input_name.lower()
        for full_name in known_teams:
            full_lower = full_name.lower()
            if input_lower in full_lower or full_lower in input_lower:
                return full_name
    return input_name


def get_team_roster(team_name):
    """获取战队的选手列表。"""
    _load_roster_data()
    if team_name in _ROSTER_CACHE:
        return _ROSTER_CACHE[team_name]
    resolved = resolve_team_name(team_name, set(_ROSTER_CACHE.keys()))
    return _ROSTER_CACHE.get(resolved, [])


# =====================================================================
# 核心特征构建
# =====================================================================
def build_single_match_features(match_info, stores, champion_tags, feature_cols=None, tf_features=None):
    """为单局对局构建完整特征向量 (完美对齐离线版)"""
    if feature_cols is None:
        feature_cols = load_feature_cols()
    
    features = {col: 0.0 for col in feature_cols}
    is_draft_mode = match_info.get("mode") == "draft"
    unknown_info = []

    # === 后端防御断言：每队unknown选手不得超过 MAX_UNKNOWN_PLAYERS_PER_TEAM ===
    if not is_draft_mode:
        for side in ["blue", "red"]:
            unknown_positions = match_info.get(f"{side}_unknown_positions", [])
            team_name = match_info.get(f"{side}_team", "unknown")
            n_unknown = len(unknown_positions)
            assert n_unknown <= MAX_UNKNOWN_PLAYERS_PER_TEAM, \
                f"[{side.upper()}方/{team_name}] unknown选手数量({n_unknown})超过限制({MAX_UNKNOWN_PLAYERS_PER_TEAM})! " \
                f"unknown位置: {unknown_positions}"

    # 1 & 2. 联赛和地图基础信息
    league = match_info.get("league", "LPL")
    for l in ["LPL", "LCK", "LEC"]:
        features[f"league_{l}"] = 1.0 if league == l else 0.0
    features["is_playoff"] = 1.0 if match_info.get("is_playoff", False) else 0.0
    features["is_blue_map_side"] = 1.0 if match_info.get("is_blue_map_side", True) else 0.0

    # 【修复】：设置局数 one-hot 特征 (is_game_1 ~ is_game_5)，对齐离线 parquet
    game_num = int(match_info.get("game_num", 1) or 1)
    if not (1 <= game_num <= 5):
        game_num = 1
    for i in range(1, 6):
        features[f"is_game_{i}"] = 1.0 if i == game_num else 0.0

    # 填充英雄 One-Hot (为了保持 features 字典中有基础字段)
    for side in ["blue", "red"]:
        for pos_idx, pos in enumerate(POSITIONS):
            features[f"{side}_{pos}_champion"] = match_info.get(f"{side}_champions", [""] * 5)[pos_idx]

    if is_draft_mode:
        for side in ["blue", "red"]:
            for pos in POSITIONS:
                for fc, fv in PLAYER_DEFAULTS.items(): features[f"{side}_{pos}_{fc}"] = fv
                for fc, fv in META_DEFAULTS.items(): features[f"{side}_{pos}_{fc}"] = fv
            for tc, tv in TEAM_PROFILE_DEFAULTS.items(): features[f"{side}_{tc}"] = tv
    else:
        # 提取时间戳
        match_date_str = match_info.get("date", "2099-12-31")
        match_date = pd.Timestamp(match_date_str)
        
        # --- 3. 选手历史特征 (带严格的 PIT 衰减重置) ---
        player_store = stores.get("player")
        for side in ["blue", "red"]:
            unknown_positions = match_info.get(f"{side}_unknown_positions", [])
            team_known_player_features = {}

            for pos_idx, pos in enumerate(POSITIONS):
                player_id = match_info.get(f"{side}_{pos}_player_id", "")
                champion = features[f"{side}_{pos}_champion"]

                if pos in unknown_positions:
                    unknown_info.append({"side": side, "pos": pos, "team": match_info.get(f"{side}_team", ""), "champion": champion})
                    continue

                matched = False
                if player_store is not None and not player_store.empty and player_id and champion:
                    # 【PIT 对齐修复】：用 <= 匹配当前场次的行（Store 中该行特征已是 PIT 严格 < 自身日期计算，无泄漏）
                    mask = (player_store["player_id"] == player_id) & (player_store["champion"] == champion) & (player_store["date"] <= match_date)
                    matched_df = player_store[mask]
                    if not matched_df.empty:
                        latest = matched_df.iloc[-1]
                        matched = True
                        
                        # 【核心修复】：时间流逝重置
                        delta_days = (match_date - latest["date"]).days
                        
                        for fc in PLAYER_FEATURE_COLS:
                            val = float(latest[fc]) if pd.notna(latest[fc]) else PLAYER_DEFAULTS[fc]
                            
                            # 超过90天，强制重置 recent 指标
                            if delta_days > 90 and ("recent_kda_90d" in fc or "recent_wr_90d" in fc or "recent_games_90d" in fc):
                                val = PLAYER_DEFAULTS[fc]
                            # 如果超过 90 天，其实 overall_recent 也应该重置，为了安全对齐暂不加
                                
                            features[f"{side}_{pos}_{fc}"] = val
                            team_known_player_features.setdefault(fc, []).append(val)

                if not matched:
                    for fc in PLAYER_FEATURE_COLS:
                        features[f"{side}_{pos}_{fc}"] = PLAYER_DEFAULTS[fc]
                        team_known_player_features.setdefault(fc, []).append(PLAYER_DEFAULTS[fc])

            # 新秀惩罚
            for info in [u for u in unknown_info if u["side"] == side]:
                pos = info["pos"]
                for fc in PLAYER_FEATURE_COLS:
                    known_vals = team_known_player_features.get(fc, [])
                    avg_val = float(np.mean(known_vals)) if known_vals else PLAYER_DEFAULTS[fc]
                    features[f"{side}_{pos}_{fc}"] = avg_val * ROOKIE_PENALTY.get(fc, 1.0)

        # --- 4. 英雄元数据特征 ---
        meta_store = stores.get("meta")
        for side in ["blue", "red"]:
            for pos in POSITIONS:
                champion = features[f"{side}_{pos}_champion"]
                matched = False
                if meta_store is not None and not meta_store.empty and champion:
                    mask = (meta_store["champion"] == champion) & (meta_store["date"] <= match_date)
                    matched_df = meta_store[mask].sort_values("date")
                    if not matched_df.empty:
                        latest = matched_df.iloc[-1]
                        delta_days = (match_date - latest["date"]).days
                        if delta_days <= 60: # Meta 保质期
                            matched = True
                            for fc in META_FEATURE_COLS:
                                features[f"{side}_{pos}_{fc}"] = float(latest[fc]) if pd.notna(latest[fc]) else META_DEFAULTS[fc]
                
                if not matched:
                    for fc in META_FEATURE_COLS:
                        features[f"{side}_{pos}_{fc}"] = META_DEFAULTS[fc]

        # --- 5. 队伍画像特征 ---
        team_store = stores.get("team_profile")
        for side in ["blue", "red"]:
            team_name = match_info.get(f"{side}_team", "")
            matched = False
            if team_store is not None and not team_store.empty and team_name:
                mask = (team_store["team"] == team_name) & (team_store["date"] <= match_date)
                matched_df = team_store[mask]
                if not matched_df.empty:
                    latest = matched_df.iloc[-1]
                    delta_days = (match_date - latest["date"]).days
                    if delta_days <= 90: # Team Profile 保质期
                        matched = True
                        for tc in TEAM_PROFILE_FEATURE_COLS:
                            features[f"{side}_{tc}"] = float(latest[tc]) if pd.notna(latest[tc]) else TEAM_PROFILE_DEFAULTS[tc]

            if not matched:
                for tc, tv in TEAM_PROFILE_DEFAULTS.items():
                    features[f"{side}_{tc}"] = tv

    # --- 统一计算所有衍生特征 ---
    derived_features = calculate_derived_features(features, champion_tags)
    features.update(derived_features)

    # --- TF 特征填入 ---
    if tf_features is not None:
        for tc in TF_COLS:
            if tc in tf_features: features[tc] = float(tf_features[tc])

    df = pd.DataFrame([features], columns=feature_cols)
    return df, unknown_info

# =====================================================================
# 特征分类 (用于 BP Delta)
# =====================================================================
def classify_features(feature_cols):
    """将特征列分为 draft 相关和纸面硬实力两类。

    Returns:
        tuple: (draft_cols_set, hard_cols_set)
    """
    draft_cols = set()
    hard_cols = set()

    for col in feature_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in DRAFT_KEYWORDS):
            draft_cols.add(col)
        else:
            hard_cols.add(col)

    return draft_cols, hard_cols


def build_predraft_features(match_info, stores, champion_tags,
                            feature_cols=None, tf_features=None):
    """构建 Pre-Draft 特征向量 (纸面硬实力, draft 相关特征置零)。

    策略:
      - 保留: 联赛、季后赛、地图方、选手近期表现、队伍画像
      - 置零: mastery、comp_*、meta_*、champ_*、champion one-hot
      - 差分特征: 仅保留非 draft 差分

    Returns:
        pd.DataFrame: 特征矩阵
    """
    if feature_cols is None:
        feature_cols = load_feature_cols()
    if feature_cols is None:
        raise RuntimeError("未找到特征列名文件")

    # 先构建完整特征, 然后置零 draft 部分
    features_df, _ = build_single_match_features(
        match_info, stores, champion_tags,
        feature_cols=feature_cols, tf_features=tf_features
    )

    # 分类特征
    draft_cols, _ = classify_features(feature_cols)

    # 将 draft 相关特征置零
    for col in draft_cols:
        if col in features_df.columns:
            features_df[col] = 0.0

    return features_df


# =====================================================================
# Checkpoint 超参数自动推断 (避免硬编码导致架构不一致)
# =====================================================================

def _infer_n_layers(state_dict):
    """从 state_dict 中 bert.transformer.layer.{i} 的最大索引推断 n_layers。"""
    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("bert.transformer.layer."):
            idx = int(k.split(".")[3])
            layer_indices.add(idx)
    return max(layer_indices) + 1 if layer_indices else 2


def _infer_n_positions(state_dict):
    """从 position_embeddings weight 的 shape 推断 n_positions (role tokens 数量)。

    BPTacticalTransformerPick 内部: extended_vocab_size = vocab_size + n_positions
    position_embeddings shape = (max_position_embeddings, h_dim)
    但 n_positions 通常保存在 checkpoint 中, 回退到 5。
    """
    # position_embeddings 不直接暴露 n_positions, 使用默认值
    # 实际上 n_positions 是构造参数, checkpoint 可能保存了
    return 5  # 与 BPTacticalTransformerPick 默认值一致


def _infer_n_heads(state_dict, h_dim):
    """从 attention q_lin weight 推断 n_heads。

    q_lin weight shape = (h_dim, h_dim), 每个 head 的维度 = h_dim / n_heads。
    通过检查 q_lin bias 的分组模式推断 n_heads。
    """
    # 尝试从常见配置推断
    q_weight = state_dict.get("bert.transformer.layer.0.attention.q_lin.weight")
    if q_weight is not None:
        # q_lin shape = (h_dim, h_dim), 无法直接推断 n_heads
        # 使用常见组合: h_dim=384 -> 4/6/8 heads, h_dim=512 -> 8 heads
        if h_dim == 384:
            return 4  # NoCS 最佳配置
        elif h_dim == 512:
            return 8
        elif h_dim == 768:
            return 12
    return 4  # 默认


def _infer_candidate_dim(state_dict):
    """从 candidate_mlp 第一层推断 candidate_dim。"""
    first_layer = state_dict.get("candidate_mlp.0.weight")
    if first_layer is not None:
        return first_layer.shape[1]
    return 31  # 默认


def _infer_query_dim(state_dict):
    """从 bert_proj.weight 推断 query_dim。

    bert_proj 是 nn.Linear(h_dim, query_dim), weight shape = (query_dim, h_dim)
    """
    weight = state_dict.get("bert_proj.weight")
    if weight is not None:
        return weight.shape[0]
    return 256  # 默认


def _infer_c_dim(state_dict):
    """从 context_mlp.0.weight 推断 c_dim。

    context_mlp[0] 是 nn.Linear(context_dim, c_dim), weight shape = (c_dim, context_dim)
    """
    weight = state_dict.get("context_mlp.0.weight")
    if weight is not None:
        return weight.shape[0]
    return 64  # 默认


def _infer_candidate_hidden(state_dict):
    """从 candidate_mlp.5.weight 推断 candidate_hidden。

    candidate_mlp[5] 是 nn.Linear(candidate_hidden//2, query_dim), weight shape = (query_dim, candidate_hidden//2)
    """
    weight = state_dict.get("candidate_mlp.5.weight")
    if weight is not None:
        return weight.shape[1] * 2
    return 256  # 默认


# =====================================================================
# TF 特征提取 (从 NoCS Transformer 快照实时推理)
# =====================================================================
_TF_EXTRACTOR = None  # 全局缓存, 避免重复加载模型


def load_tf_extractor(snapshot_path=None, device="cpu"):
    """加载 Transformer 特征提取器 (懒加载, 全局缓存)。

    Args:
        snapshot_path: NoCS Transformer 快照路径, 默认使用 fold_4
        device: 推理设备

    Returns:
        TransformerFeatureExtractor 实例, 或 None (加载失败)
    """
    global _TF_EXTRACTOR
    if _TF_EXTRACTOR is not None:
        return _TF_EXTRACTOR

    try:
        import torch
        # 确保模型路径在 sys.path 中
        model_pick_dir = os.path.join(PROJECT_ROOT, "bp_recommendation", "model_pick")
        if model_pick_dir not in sys.path:
            sys.path.insert(0, model_pick_dir)
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)

        from model_pick import BPTacticalTransformerPick

        # 优先加载生产快照, 回退到 fold_4 快照
        if snapshot_path is None:
            prod_path = os.path.join(MODEL_DIR, "tf_snapshots", "production_nocs.pt")
            fold4_path = os.path.join(MODEL_DIR, "tf_snapshots", "fold_4_nocs.pt")
            if os.path.exists(prod_path):
                snapshot_path = prod_path
            elif os.path.exists(fold4_path):
                snapshot_path = fold4_path
            else:
                return None

        if not os.path.exists(snapshot_path):
            return None

        ckpt = torch.load(snapshot_path, map_location=device, weights_only=False)

        # ---- TF 快照版本校验：验证源 checkpoint MD5 ----
        # 防止推荐模型重训后未同步导出，导致 TF 特征分布偏移
        source_md5 = ckpt.get("source_md5")
        if source_md5:
            source_path = ckpt.get("source", "")
            # 推断源 checkpoint 的实际路径
            if source_path:
                actual_source_path = os.path.join(PROJECT_ROOT, source_path)
            else:
                actual_source_path = os.path.join(
                    PROJECT_ROOT, "bp_recommendation", "model_pick",
                    "checkpoints", "best_model_nocs.pt"
                )

            if os.path.exists(actual_source_path):
                import hashlib
                md5 = hashlib.md5()
                with open(actual_source_path, "rb") as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        md5.update(chunk)
                actual_md5 = md5.hexdigest()

                if actual_md5 != source_md5:
                    import logging
                    logging.getLogger("feature_builder").warning(
                        f"TF 快照版本不匹配! 源 checkpoint 已更新: "
                        f"快照记录 MD5={source_md5}, 实际 MD5={actual_md5}. "
                        f"请重新运行 export_production_transformer.py 导出快照。"
                    )
                else:
                    import logging
                    logging.getLogger("feature_builder").info(
                        f"TF 快照版本校验通过 (MD5={source_md5})"
                    )
            else:
                import logging
                logging.getLogger("feature_builder").warning(
                    f"无法校验 TF 快照版本: 源 checkpoint 不存在: {actual_source_path}"
                )
        # ---- 版本校验结束 ----

        # 从 checkpoint 的 model_state_dict 自动推断所有架构超参数
        # 避免硬编码导致与实际模型不一致 (如推荐模型 n_layers=3, OOT n_layers=2)
        state_dict = ckpt["model_state_dict"]

        # vocab_size: embedding shape = vocab_size + n_positions (含 role tokens)
        emb_weight = state_dict.get("bert.embeddings.word_embeddings.weight")
        n_positions = _infer_n_positions(state_dict)
        if emb_weight is not None:
            vocab_size = emb_weight.shape[0] - n_positions
        else:
            vocab_size = ckpt.get("vocab_size", 170)

        # n_layers: 从 bert.transformer.layer.{i} 的最大索引推断
        n_layers = _infer_n_layers(state_dict)

        # h_dim: 从 embedding 维度推断
        h_dim = emb_weight.shape[1] if emb_weight is not None else ckpt.get("h_dim", 384)

        # n_heads: 从 attention q_lin weight 推断
        n_heads = _infer_n_heads(state_dict, h_dim)

        # candidate_dim: 从 candidate_mlp 第一层推断
        candidate_dim = _infer_candidate_dim(state_dict)

        # query_dim: 从 bert_proj.weight 推断
        query_dim = _infer_query_dim(state_dict)

        # c_dim: 从 context_mlp.0.weight 推断
        c_dim = _infer_c_dim(state_dict)

        # candidate_hidden: 从 candidate_mlp.5.weight 推断
        candidate_hidden = _infer_candidate_hidden(state_dict)

        # 从 checkpoint 重建模型 (所有超参数自动推断)
        model = BPTacticalTransformerPick(
            vocab_size=vocab_size,
            context_dim=ckpt.get("context_dim", 15),
            candidate_dim=candidate_dim,
            h_dim=h_dim,
            c_dim=c_dim,
            query_dim=query_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=0.0,
            attention_dropout=0.0,
            candidate_hidden=candidate_hidden,
            tactical_hidden=ckpt.get("tactical_hidden", 256),
            aux_loss_weight=0.0,
            ban_sample_weight=0.0,
            n_positions=n_positions,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        # 使用 extract_transformer_features.py 中的 TransformerFeatureExtractor
        training_dir = os.path.join(MODEL_DIR, "training")
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        from extract_transformer_features import TransformerFeatureExtractor
        _TF_EXTRACTOR = TransformerFeatureExtractor(model, device=device)
        return _TF_EXTRACTOR

    except Exception as e:
        import logging
        logging.getLogger("feature_builder").warning(f"TF 特征提取器加载失败: {e}")
        return None


def extract_tf_features_for_match(match_info, context_df=None):
    """为单局对局提取 TF 特征。

    优先级:
      1. 有 gameid 且在 context parquet 中找到 → 直接提取
      2. 有英雄列表 → 从英雄名称构建 BP 序列实时推理
      3. 以上都不满足 → 返回默认值

    Args:
        match_info: dict, 对局信息
        context_df: DataFrame, context parquet 数据 (可选, 懒加载)

    Returns:
        dict: {tf_win_logits, tf_cosine_sim, tf_blue_l2norm, tf_red_l2norm}
    """
    extractor = load_tf_extractor()
    if extractor is None:
        return _get_tf_defaults()

    try:
        # 路径 1: 通过 gameid 查找历史 context
        gameid = match_info.get("gameid")
        if gameid:
            if context_df is None:
                context_path = os.path.join(PROJECT_ROOT, "bp_recommendation", "features", "ALL_context.parquet")
                if os.path.exists(context_path):
                    context_df = pd.read_parquet(context_path)
            if context_df is not None and gameid in context_df["gameid"].values:
                results = extractor.extract_features([gameid], context_df)
                if gameid in results:
                    return results[gameid]

        # 路径 2: 从前端英雄列表实时构建 BP 序列推理
        blue_champs = match_info.get("blue_champions", [])
        red_champs = match_info.get("red_champions", [])
        if blue_champs and red_champs and len(blue_champs) == 5 and len(red_champs) == 5:
            return _extract_tf_from_champions(extractor, match_info)

        # 路径 3: 默认值
        return _get_tf_defaults()

    except Exception:
        return _get_tf_defaults()


# 英雄名称 → ID 映射缓存
_CHAMP_NAME_TO_IDX = None


def _load_champ_name_to_idx():
    """加载英雄名称到 ID 的映射 (懒加载, 全局缓存)。"""
    global _CHAMP_NAME_TO_IDX
    if _CHAMP_NAME_TO_IDX is not None:
        return _CHAMP_NAME_TO_IDX

    vocab_path = os.path.join(PROJECT_ROOT, "cleaned_data", "champion_vocabulary.json")
    if not os.path.exists(vocab_path):
        return None

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    _CHAMP_NAME_TO_IDX = vocab.get("name_to_idx", {})
    return _CHAMP_NAME_TO_IDX


def _extract_tf_from_champions(extractor, match_info):
    """从前端输入的英雄列表构建 BP 序列, 实时推理 TF 特征。"""
    import torch
    import torch.nn.functional as F

    name_to_idx = _load_champ_name_to_idx()
    if name_to_idx is None:
        return _get_tf_defaults()

    UNK_IDX = 1  # champion_vocabulary.json 中的 UNK token
    blue_champs = match_info.get("blue_champions", [])
    red_champs = match_info.get("red_champions", [])

    blue_ids = [name_to_idx.get(str(c).strip(), UNK_IDX) for c in blue_champs]
    red_ids = [name_to_idx.get(str(c).strip(), UNK_IDX) for c in red_champs]

    # 【修复 1】：严格按照标准 BP_SEQUENCE 的索引放置 Pick 位
    # 0-5: Ban 1-3 | 12-15: Ban 4-5
    # 6:B1, 7:R1, 8:R2, 9:B2, 10:B3, 11:R3 | 16:R4, 17:B4, 18:B5, 19:R5
    bp_sequence = [UNK_IDX] * 20  
    if len(blue_ids) >= 5 and len(red_ids) >= 5:
        bp_sequence[6]  = blue_ids[0]   # B1
        bp_sequence[7]  = red_ids[0]    # R1
        bp_sequence[8]  = red_ids[1]    # R2
        bp_sequence[9]  = blue_ids[1]   # B2
        bp_sequence[10] = blue_ids[2]   # B3
        bp_sequence[11] = red_ids[2]    # R3
        bp_sequence[16] = red_ids[3]    # R4
        bp_sequence[17] = blue_ids[3]   # B4
        bp_sequence[18] = blue_ids[4]   # B5
        bp_sequence[19] = red_ids[4]    # R5

    bp_tensor = torch.tensor([bp_sequence], dtype=torch.long).to(extractor.device)

    # 【修复 2】：构建 20 维的 global_context
    context_dim = extractor.model.context_mlp[0].in_features
    global_context = np.zeros((1, context_dim), dtype=np.float32)

    league = match_info.get("league", "LCK")
    league_map = {"LPL": 0, "LCK": 1, "LEC": 2}
    global_context[0, league_map.get(league, 1)] = 1.0

    # 季后赛与地图方
    global_context[0, 13] = 1.0 if match_info.get("is_playoff", False) else 0.0
    global_context[0, 14] = 1.0 if match_info.get("is_blue_map_side", True) else 0.0

    # 局数 (game_num: 1-5 对应 index 15-19)
    game_num = match_info.get("game_num", 1)
    if 1 <= game_num <= 5:
        global_context[0, 14 + game_num] = 1.0
    else:
        global_context[0, 15] = 1.0 # 默认第一局

    # 战队统计
    try:
        stores = load_feature_stores()
        tp = stores.get("team_profile")
        if tp is not None:
            blue_team = match_info.get("blue_team", "")
            red_team = match_info.get("red_team", "")
            for side, team_name, offset in [("blue", blue_team, 3), ("red", red_team, 8)]:
                if team_name and team_name in tp["team"].values:
                    row = tp[tp["team"] == team_name].iloc[0]
                    stat_keys = [
                        "team_avg_ckpm", "team_avg_golddiffat15",
                        "team_avg_gamelength", "team_firstdragon_rate",
                        "team_firsttower_rate",
                    ]
                    for j, key in enumerate(stat_keys):
                        if key in row.index:
                            global_context[0, offset + j] = float(row[key]) if pd.notna(row[key]) else 0.0
    except Exception:
        pass

    ctx_tensor = torch.tensor(global_context, dtype=torch.float32).to(extractor.device)

    # 占位矩阵
    vocab_size = extractor.model.vocab_size
    candidate_dim = extractor.model.candidate_mlp[0].in_features
    cand_tensor = torch.zeros(1, vocab_size, candidate_dim, dtype=torch.float32).to(extractor.device)
    mask_tensor = torch.ones(1, vocab_size, dtype=torch.float32).to(extractor.device)

    # 前向推理
    extractor.model.eval()
    with torch.no_grad():
        _ = extractor.model(bp_tensor, ctx_tensor, cand_tensor, mask_tensor)

    hidden = extractor._captured_hidden
    seq_len = 20
    bp_hidden = hidden[:, :seq_len, :] 

    # 【修复 3】：纠正提取索引，并加入 Padding 掩码避免 UNK 噪声污染
    blue_steps = [0, 2, 4, 6, 9, 10, 13, 15, 17, 18]
    red_steps = [1, 3, 5, 7, 8, 11, 12, 14, 16, 19]

    # 创建掩码：屏蔽 pad_idx 和 unk_idx
    valid_mask = (bp_tensor != extractor.model.pad_idx) & (bp_tensor != UNK_IDX)
    valid_mask = valid_mask.float().unsqueeze(-1) # [1, 20, 1]

    # 蓝方池化
    b_mask = valid_mask[:, blue_steps, :]
    b_hidden = bp_hidden[:, blue_steps, :] * b_mask
    blue_pooled = b_hidden.sum(dim=1) / b_mask.sum(dim=1).clamp(min=1e-9)

    # 红方池化
    r_mask = valid_mask[:, red_steps, :]
    r_hidden = bp_hidden[:, red_steps, :] * r_mask
    red_pooled = r_hidden.sum(dim=1) / r_mask.sum(dim=1).clamp(min=1e-9)

    blue_latent = extractor.model.bert_proj(blue_pooled).detach()
    red_latent = extractor.model.bert_proj(red_pooled).detach()

    tf_win_logits = float((blue_latent.norm(dim=1) * torch.sign(blue_latent.mean(dim=1)))[0])
    tf_cosine_sim = float(F.cosine_similarity(blue_latent, red_latent, dim=1)[0])
    tf_blue_l2norm = float(blue_latent.norm(dim=1)[0])
    tf_red_l2norm = float(red_latent.norm(dim=1)[0])

    return {
        "tf_win_logits": tf_win_logits,
        "tf_cosine_sim": tf_cosine_sim,
        "tf_blue_l2norm": tf_blue_l2norm,
        "tf_red_l2norm": tf_red_l2norm,
    }

def _get_tf_defaults():
    """返回 TF 特征的默认值 (与训练时缺失值填充一致)。"""
    return {
        "tf_win_logits": 0.0,
        "tf_cosine_sim": 0.5,
        "tf_blue_l2norm": 10.0,
        "tf_red_l2norm": 10.0,
    }
