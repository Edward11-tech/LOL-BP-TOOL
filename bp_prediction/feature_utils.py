"""
feature_utils.py — 特征工程公共计算库
将离线 pipeline 和在线 builder 的计算逻辑统一，杜绝线上线下不一致。
"""
import numpy as np
import pandas as pd

POSITIONS = ["top", "jng", "mid", "bot", "sup"]
POSITIONS_SHORT = POSITIONS
POSITIONS_FULL = ["top", "jungle", "mid", "bot", "support"]
POS_SHORT2FULL = dict(zip(POSITIONS_SHORT, POSITIONS_FULL))

# =====================================================================
# 1. 默认值与常量配置 (严格统一)
# =====================================================================
PLAYER_DEFAULTS = {
    "mastery_score": 32.5, "player_recent_kda_90d": 3.0, "player_recent_wr_90d": 0.5,
    "player_recent_games_90d": 0,
    "player_overall_recent_wr": 0.5, "player_overall_recent_kda": 3.0, "player_overall_recent_games": 0
}
META_DEFAULTS = {
    "meta_pick_rate_pit": 0.0, "meta_ban_rate_pit": 0.0, "meta_presence_pit": 0.0, "meta_win_rate_pit": 0.5,
    "meta_patch_drift_index": 1.0, "meta_pick_drift_index": 1.0
}
TEAM_PROFILE_DEFAULTS = {
    "team_avg_gamelength": 1954, "team_avg_ckpm": 0.7, "team_avg_golddiffat15": 0,
    "team_firstdragon_rate": 0.5, "team_firsttower_rate": 0.5, "team_recent_wr": 0.5,
    "team_recent_wr_5": 0.5, "team_recent_wr_10": 0.5, "team_side_wr": 0.5, "team_streak": 0,
    "team_profile_games": 0, "team_avg_kills": 25.0, "team_avg_deaths": 25.0,
    "team_avg_assists": 60.0, "team_bloodiness": 1.5, "team_snowball_rate": 0.7, "team_led_at_15_rate": 0.5,
}

PLAYER_FEATURE_COLS = list(PLAYER_DEFAULTS.keys())
META_FEATURE_COLS = ["meta_win_rate_pit", "meta_patch_drift_index", "meta_pick_drift_index"]
TEAM_PROFILE_FEATURE_COLS = list(TEAM_PROFILE_DEFAULTS.keys())
TF_COLS = ["tf_win_logits", "tf_cosine_sim", "tf_blue_l2norm", "tf_red_l2norm"]

# 英雄标签字典 (保留你的完整字典，这里为了精简展示折叠)
CHAMPION_TAGS = {
    "Ornn": {"Engage":3,"Poke":0,"Peel":1,"Burst":0,"Tank":3,"HardCC":2,"LineStrength":0},
    "Lee Sin": {"Engage":2,"Poke":0,"Peel":1,"Burst":2,"Tank":1,"HardCC":2,"LineStrength":0},
    "Ahri": {"Engage":2,"Poke":1,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Jinx": {"Engage":0,"Poke":1,"Peel":1,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":0},
    "Nautilus": {"Engage":3,"Poke":0,"Peel":2,"Burst":1,"Tank":2,"HardCC":3,"LineStrength":0},
    # ... 请把你原来的 CHAMPION_TAGS 完整复制粘贴到这里 ...
}

# =====================================================================
# 2. 共享纯数学计算核心 (The Single Source of Truth)
# =====================================================================
def calc_comp_features(champion_names, champion_tags=None):
    """
    基于 champion_tags 计算阵容发力期特征。
    
    Args:
        champion_names: 英雄名称列表
        champion_tags: 英雄标签字典，为 None 时使用模块级 CHAMPION_TAGS（向后兼容）
    """
    if champion_tags is None:
        champion_tags = CHAMPION_TAGS  # 向后兼容：未传入时用模块级字典
    
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

    agg["comp_early_power"] = agg["comp_linestrength_sum"] * 1.0 + agg["comp_burst_sum"] * 0.5 + agg["comp_engage_sum"] * 0.3
    agg["comp_late_power"] = agg["comp_tank_sum"] * 1.0 + agg["comp_peel_sum"] * 0.5
    agg["comp_teamfight_score"] = agg["comp_engage_sum"] * 1.0 + agg["comp_hardcc_sum"] * 0.8 + agg["comp_tank_sum"] * 0.3
    agg["comp_aggression_index"] = (agg["comp_engage_sum"] + agg["comp_burst_sum"] + agg["comp_linestrength_sum"]) / (agg["comp_peel_sum"] + agg["comp_poke_sum"] * 0.5 + 1)
    agg["comp_scaling_type"] = (agg["comp_peel_sum"] + agg["comp_poke_sum"]) / (agg["comp_engage_sum"] + agg["comp_burst_sum"] + 1)
    agg["comp_lane_dom_type"] = (agg["comp_linestrength_sum"] + agg["comp_burst_sum"]) / 10.0
    agg["comp_teamfight_type"] = (agg["comp_engage_sum"] + agg["comp_hardcc_sum"]) / 10.0
    return agg

def safe_std(vals):
    if len(vals) <= 1: return 0.0
    return float(np.std(vals, ddof=1))

def calculate_derived_features(f, champion_tags=None):
    """
    接收基础特征字典，计算所有的衍生、差值、交互特征。
    无论是离线的大 DataFrame apply，还是线上单条字典，全部过这里！
    
    Args:
        f: 基础特征字典
        champion_tags: 英雄标签字典，传给 calc_comp_features 使用
    """
    res = {}
    
    # 1. 基础差分特征 (Player & Meta)
    for pos in POSITIONS:
        for c in PLAYER_FEATURE_COLS + META_FEATURE_COLS:
            res[f"diff_{pos}_{c}"] = f.get(f"blue_{pos}_{c}", 0.0) - f.get(f"red_{pos}_{c}", 0.0)

    # 2. 队伍聚合特征 (avg, max, std)
    for side in ["blue", "red"]:
        for c in PLAYER_FEATURE_COLS:
            vals = [f.get(f"{side}_{pos}_{c}", 0.0) for pos in POSITIONS]
            res[f"{side}_team_avg_{c}"] = float(np.mean(vals))
            res[f"{side}_team_max_{c}"] = float(np.max(vals))
            res[f"{side}_team_std_{c}"] = safe_std(vals)
        for c in META_FEATURE_COLS:
            vals = [f.get(f"{side}_{pos}_{c}", 0.0) for pos in POSITIONS]
            res[f"{side}_team_avg_{c}"] = float(np.mean(vals))

    # 3. 队伍聚合差分
    for c in PLAYER_FEATURE_COLS + META_FEATURE_COLS:
        res[f"diff_team_avg_{c}"] = res.get(f"blue_team_avg_{c}", 0.0) - res.get(f"red_team_avg_{c}", 0.0)

    # 4. 队伍画像差分
    for c in TEAM_PROFILE_FEATURE_COLS:
        res[f"diff_{c}"] = f.get(f"blue_{c}", 0.0) - f.get(f"red_{c}", 0.0)

    # 5. 英雄偏差特征 & 熟练度交互
    for side in ["blue", "red"]:
        for pos in POSITIONS:
            res[f"{side}_{pos}_champ_wr_delta"] = f.get(f"{side}_{pos}_player_recent_wr_90d", 0.5) - f.get(f"{side}_{pos}_player_overall_recent_wr", 0.5)
            res[f"{side}_{pos}_champ_kda_delta"] = f.get(f"{side}_{pos}_player_recent_kda_90d", 3.0) - f.get(f"{side}_{pos}_player_overall_recent_kda", 3.0)
            res[f"{side}_{pos}_mastery_x_presence"] = f.get(f"{side}_{pos}_mastery_score", 0.0) * f.get(f"{side}_{pos}_meta_presence_pit", 0.0)
            res[f"{side}_{pos}_mastery_x_wr"] = f.get(f"{side}_{pos}_mastery_score", 0.0) * f.get(f"{side}_{pos}_meta_win_rate_pit", 0.5)

    # 6. 交互特征差分
    for pos in POSITIONS:
        for c in ["champ_wr_delta", "champ_kda_delta", "mastery_x_wr"]:
            res[f"diff_{pos}_{c}"] = res.get(f"blue_{pos}_{c}", 0.0) - res.get(f"red_{pos}_{c}", 0.0)
            
        d_mastery = res.get(f"diff_{pos}_mastery_score", 0.0)
        res[f"diff_{pos}_mastery_x_meta_wr"] = d_mastery * res.get(f"diff_{pos}_meta_win_rate_pit", 0.0)
        res[f"diff_{pos}_mastery_x_player_wr"] = d_mastery * res.get(f"diff_{pos}_player_overall_recent_wr", 0.0)

    # 7. 队伍胜率平衡与阵容折损
    for side in ["blue", "red"]:
        wr_vals = [f.get(f"{side}_{pos}_player_overall_recent_wr", 0.5) for pos in POSITIONS]
        res[f"{side}_team_wr_max_gap"] = max(wr_vals) - min(wr_vals)
        res[f"{side}_team_wr_balance"] = safe_std(wr_vals)
        
        team_wr = f.get(f"{side}_team_recent_wr", 0.5)
        roster_wr = res.get(f"{side}_team_avg_player_overall_recent_wr", 0.5)
        res[f"{side}_team_wr_x_roster_wr"] = team_wr * roster_wr

    for c in ["team_wr_max_gap", "team_wr_balance", "team_wr_x_roster_wr"]:
        res[f"diff_{c}"] = res.get(f"blue_{c}", 0.0) - res.get(f"red_{c}", 0.0)

    # 8. 阵容特征 & 差分
    blue_champs = [f.get(f"blue_{pos}_champion", "") for pos in POSITIONS]
    red_champs = [f.get(f"red_{pos}_champion", "") for pos in POSITIONS]
    
    blue_comp = calc_comp_features(blue_champs, champion_tags)
    red_comp = calc_comp_features(red_champs, champion_tags)
    
    res.update({f"blue_{k}": v for k, v in blue_comp.items()})
    res.update({f"red_{k}": v for k, v in red_comp.items()})
    
    for k in blue_comp.keys():
        res[f"diff_{k}"] = blue_comp.get(k, 0.0) - red_comp.get(k, 0.0)

    # 9. LPL 特定打架交互特征
    for side in ["blue", "red"]:
        blood = f.get(f"{side}_team_bloodiness", 1.5)
        aggr = res.get(f"{side}_comp_aggression_index", 0.0)
        early = res.get(f"{side}_comp_early_power", 0.0)
        snow = f.get(f"{side}_team_snowball_rate", 0.5)
        ckpm = f.get(f"{side}_team_avg_ckpm", 0.7)
        wr = f.get(f"{side}_team_recent_wr", 0.5)

        res[f"{side}_bloodiness_x_aggression"] = blood * aggr
        res[f"{side}_early_power_x_snowball"] = early * snow
        res[f"{side}_ckpm_x_aggression"] = ckpm * aggr
        res[f"{side}_wr_x_aggression"] = wr * aggr

    for c in ["bloodiness_x_aggression", "early_power_x_snowball", "ckpm_x_aggression", "wr_x_aggression"]:
        res[f"diff_{c}"] = res.get(f"blue_{c}", 0.0) - res.get(f"red_{c}", 0.0)

    return res