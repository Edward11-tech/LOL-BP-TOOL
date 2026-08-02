"""
预测模型可解释性模块
====================
提供以下功能:
1. FEATURE_EXPLAIN_DICT: 特征名→通俗中文解释的映射字典，按类别分组
2. compute_shap_values(): 计算单条样本的SHAP值（使用CatBoost内置TreeSHAP）
3. analyze_counters(): 分路counter关系分析
4. analyze_synergy(): 英雄组合synergy关系分析
5. get_champion_stats(): 英雄出场率/胜率数据
"""

import os
import json
import numpy as np
import pandas as pd
from catboost import Pool
from logger_config import get_logger

log = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")

POS_MAP = {"top": "top", "jungle": "jungle", "mid": "mid", "bot": "bot", "support": "support",
           "jng": "jungle", "sup": "support"}
POS_NAMES_CN = {"top": "上单", "jungle": "打野", "mid": "中单", "bot": "ADC", "support": "辅助"}

_counters_df = None
_synergy_df = None
_champion_ranks_df = None
_merged_stats_df = None
_champion_zh_cache = None  # 英雄英文名 -> 中文名的映射缓存


def _load_counters():
    """延迟加载counter数据"""
    global _counters_df
    if _counters_df is None:
        path = os.path.join(CLEANED_DIR, "champion_counters_cleaned.csv")
        if os.path.exists(path):
            _counters_df = pd.read_csv(path)
            _counters_df["position"] = _counters_df["position"].str.lower().map(
                lambda x: POS_MAP.get(x, x))
        else:
            _counters_df = pd.DataFrame()
            log.warning(f"Counter数据文件不存在: {path}")
    return _counters_df


def _load_synergy():
    """延迟加载synergy数据"""
    global _synergy_df
    if _synergy_df is None:
        path = os.path.join(CLEANED_DIR, "champion_synergy_cleaned.csv")
        if os.path.exists(path):
            _synergy_df = pd.read_csv(path)
            _synergy_df["position"] = _synergy_df["position"].str.lower().map(
                lambda x: POS_MAP.get(x, x))
        else:
            _synergy_df = pd.DataFrame()
            log.warning(f"Synergy数据文件不存在: {path}")
    return _synergy_df


def _load_champion_ranks():
    """延迟加载champion_ranks数据（lol.ps职业赛场统计，分位置）"""
    global _champion_ranks_df
    if _champion_ranks_df is None:
        path = os.path.join(CLEANED_DIR, "champion_ranks_cleaned.csv")
        if os.path.exists(path):
            _champion_ranks_df = pd.read_csv(path)
            _champion_ranks_df["position"] = _champion_ranks_df["position"].str.lower().map(
                lambda x: POS_MAP.get(x, x))
            for col in ["win_rate", "pick_rate", "ban_rate", "presence_rate"]:
                if col in _champion_ranks_df.columns:
                    _champion_ranks_df[col] = pd.to_numeric(_champion_ranks_df[col], errors="coerce")
        else:
            _champion_ranks_df = pd.DataFrame()
            log.warning(f"Champion Ranks数据文件不存在: {path}")
    return _champion_ranks_df


def _load_merged_stats():
    """延迟加载融合后的英雄统计数据（排位高分段先验 + 职业赛场观测，c=5）"""
    global _merged_stats_df
    if _merged_stats_df is None:
        path = os.path.join(CLEANED_DIR, "merged_champion_stats.csv")
        if os.path.exists(path):
            _merged_stats_df = pd.read_csv(path)
            for col in ["win_rate", "pick_rate", "ban_rate", "presence_rate"]:
                if col in _merged_stats_df.columns:
                    _merged_stats_df[col] = pd.to_numeric(_merged_stats_df[col], errors="coerce")
        else:
            log.warning(f"Merged Stats数据文件不存在: {path}, 回退到champion_ranks")
            _merged_stats_df = _load_champion_ranks()
    return _merged_stats_df


def _load_champion_zh_names():
    """延迟加载英雄英文名->中文名映射 (源自 champion_vocabulary.json)。

    用于在 counter_desc 等"自然语言描述"字段中将英文英雄名翻译为中文，
    确保中文版界面不出现英文英雄名 (e.g. "Aatrox 克制 Darius" -> "暗裔剑魔 克制 诺克萨斯之手")。

    Returns:
        dict: {english_name: chinese_name}；文件缺失时返回空 dict 并打印警告
    """
    global _champion_zh_cache
    if _champion_zh_cache is not None:
        return _champion_zh_cache

    _champion_zh_cache = {}
    path = os.path.join(CLEANED_DIR, "champion_vocabulary.json")
    try:
        if not os.path.exists(path):
            log.warning(f"英雄词汇表不存在: {path}, 中文翻译将回退到英文原名")
            return _champion_zh_cache
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # champion_vocabulary.json 结构: {"champions": [{"name": "Aatrox", "aliases": {"zh": "暗裔剑魔"}}, ...]}
        champions = vocab.get("champions", []) if isinstance(vocab, dict) else []
        for c in champions:
            en_name = c.get("name", "")
            zh_name = (c.get("aliases") or {}).get("zh", "")
            if en_name and zh_name:
                _champion_zh_cache[en_name] = zh_name
        log.info(f"英雄中文名映射加载完成: {len(_champion_zh_cache)} 个英雄")
    except Exception as e:
        log.warning(f"加载英雄中文名映射失败 (将回退到英文原名): {e}")
        _champion_zh_cache = {}
    return _champion_zh_cache


def _champ_to_zh(name):
    """将英雄英文名翻译为中文名，未找到时回退到原英文名。

    Args:
        name (str): 英雄英文名 (如 "Aatrox")

    Returns:
        str: 中文名 (如 "暗裔剑魔")；未找到时返回原英文名
    """
    if not name:
        return name
    zh_map = _load_champion_zh_names()
    return zh_map.get(name, name)


# ============================================================
# 1. 特征解释字典（中英双语）
# ============================================================
# 格式: feature_name -> { "cn": (cat_cn, desc_cn), "en": (cat_en, desc_en) }

POS_NAMES_EN = {"top": "Top", "jungle": "Jungle", "mid": "Mid", "bot": "Bot", "support": "Support"}
SIDE_CN = {"blue": "首选方", "red": "次选方"}
SIDE_EN = {"blue": "First Pick", "red": "Second Pick"}
POS_CN = {"top": "上单", "jungle": "打野", "mid": "中单", "bot": "ADC", "support": "辅助",
          "jng": "打野", "sup": "辅助"}
POS_EN = {"top": "Top", "jungle": "Jungle", "mid": "Mid", "bot": "Bot", "support": "Support",
          "jng": "Jungle", "sup": "Support"}
AGG_CN = {"avg": "平均", "max": "最高", "std": "标准差"}
AGG_EN = {"avg": "Avg", "max": "Max", "std": "Std"}

FEATURE_EXPLAIN_DICT = {
    "is_game_1": (("局数因素", "首局比赛，战队通常准备更充分"),
                  ("Game Factor", "Game 1: Teams usually better prepared")),
    "is_game_2": (("局数因素", "第二局"),
                  ("Game Factor", "Game 2")),
    "is_game_3": (("局数因素", "第三局（关键局）"),
                  ("Game Factor", "Game 3 (Key Game)")),
    "is_game_4": (("局数因素", "第四局"),
                  ("Game Factor", "Game 4")),
    "is_game_5": (("局数因素", "第五局（决胜局）"),
                  ("Game Factor", "Game 5 (Decider)")),
    "league_LPL": (("联赛特性", "LPL联赛（打架风格）"),
                   ("League", "LPL (Fight-heavy)")),
    "league_LCK": (("联赛特性", "LCK联赛（运营风格）"),
                   ("League", "LCK (Macro-oriented)")),
    "league_LEC": (("联赛特性", "LEC联赛（欧美风格）"),
                   ("League", "LEC (Western Style)")),
    "is_blue_map_side": (("地图阵营", "首选方（地图蓝色方优势）"),
                         ("Map Side", "First Pick (Blue Side Advantage)")),
    "is_playoff": (("赛事阶段", "季后赛（BO5淘汰赛）"),
                   ("Tournament", "Playoffs (BO5 Elimination)")),
}


def _build_feature_explain_dict():
    """构建完整的双语特征解释字典，支持前缀匹配"""
    d = {}
    for k, v in FEATURE_EXPLAIN_DICT.items():
        d[k] = v

    def _e(cn_cat, cn_desc, en_cat, en_desc):
        return ((cn_cat, cn_desc), (en_cat, en_desc))

    team_profile = {
        "avg_gamelength":   (("队伍节奏", "场均耗时 (反映偏前中期快攻还是后期莎士比亚)"),        ("Team Tempo", "Avg Game Length (Early Aggro vs Late Scale)")),
        "avg_ckpm":         (("打架频率", "血腥度 (每分钟爆发的人头数，打架队标志)"), ("Fight Frequency", "Bloodiness (Kills per minute)")),
        "avg_golddiffat15": (("前期能力", "15分钟经济差 (对线期和前期野区压制力)"),       ("Early Game", "Gold Diff @15 (Laning Dominance)")),
        "firstdragon_rate": (("资源控制", "首龙控制率 (下半区线权与控图倾向)"),          ("Resource Control", "First Dragon Rate (Bot Prio)")),
        "firsttower_rate":  (("推塔节奏", "一血塔率 (推进节奏与转线能力)"),          ("Tower Tempo", "First Tower Rate (Pushing Prio)")),
        "recent_wr":        (("近期手感", "近期比赛胜率 (队伍火热程度)"),            ("Recent Form", "Recent Winrate (Team Momentum)")),
        "recent_wr_5":      (("近期手感", "近5场胜率 (短期爆发状态)"),           ("Recent Form", "Last 5 Winrate (Short-term Form)")),
        "recent_wr_10":     (("近期手感", "近10场胜率 (中期稳定状态)"),          ("Recent Form", "Last 10 Winrate (Mid-term Form)")),
        "side_wr":          (("选边优势", "当前所在红/蓝色方胜率"),        ("Side Winrate", "Winrate on Current Side")),
        "streak":           (("势头", "当前连胜或连败场次 (士气影响)"),          ("Momentum", "Current Win/Loss Streak (Morale)")),
        "profile_games":    (("样本量", "战队数据样本数"),       ("Sample Size", "Team Sample Size")),
        "avg_kills":        (("进攻火力", "场均击杀数"),           ("Aggression", "Avg Kills")),
        "avg_deaths":       (("防守硬度", "场均死亡数 (防守与避战能力)"),           ("Defense", "Avg Deaths (Avoidance)")),
        "avg_assists":      (("团战配合", "场均助攻数 (反映团队协作与抱团紧密度)"),           ("Teamfight", "Avg Assists (Team Cohesion)")),
        "bloodiness":       (("血腥程度", "比赛血腥度"),         ("Bloodiness", "Game Bloodiness")),
        "snowball_rate":    (("滚雪球", "拿到优势后的终结比赛能力 (拒接翻盘)"), ("Snowball", "Closing out games with lead")),
        "led_at_15_rate":   (("前期优势率", "15分钟能打出经济领先的概率"),     ("Early Lead", "Probability of Leading @15")),
    }
    for side in ["blue", "red"]:
        for k, (cn, en) in team_profile.items():
            d[f"{side}_team_{k}"] = _e(
                f"{SIDE_CN[side]}{cn[0]}", cn[1],
                f"{SIDE_EN[side]} {en[0]}", en[1]
            )

    # 选手+Meta位置级特征
    player_keys = {
        "mastery_score":              (("绝活程度", "选手该英雄的招牌/熟练度得分"),        ("Champion Mastery", "Player's Signature Pick/Mastery")),
        "player_recent_kda_90d":      (("近期状态", "近90天KDA (生存与击杀贡献)"),                     ("Recent Form", "90d KDA (Survivability/Kills)")),
        "player_recent_wr_90d":       (("近期状态", "近90天胜率 (近期手感)"),                    ("Recent Form", "90d Winrate (Recent Form)")),
        "player_recent_games_90d":    (("近期状态", "近90天出场次数 (英雄池活跃度)"),                ("Recent Form", "90d Games (Hero Pool Activity)")),
        "player_overall_recent_wr":   (("综合状态", "近期所有英雄综合胜率"),                  ("Overall Form", "Recent Overall Winrate")),
        "player_overall_recent_kda":  (("综合状态", "近期综合KDA与发挥稳定性"),                   ("Overall Form", "Recent Overall KDA Stability")),
        "player_overall_recent_games":(("综合状态", "近期总登场局数 (比赛强度适应)"),                  ("Overall Form", "Recent Total Games Played")),
        "meta_win_rate_pit":          (("版本红利", "该英雄当前版本的绝对强度 (Meta胜率)"),                ("Meta Trend", "Champion Current Meta Winrate")),
        "meta_patch_drift_index":     (("版本契合", "选手对新版本的适应与练英雄速度"),     ("Patch Fit", "Adaptability to Meta Changes")),
        "meta_pick_drift_index":      (("BP博弈", "近期非Ban必选的热门BP趋势"),   ("Pick Trend", "Recent Priority Pick Trend")),
    }
    pos_keys = ["top", "jng", "mid", "bot", "sup"]

    for side in ["blue", "red"]:
        for pk in pos_keys:
            for k, (cn, en) in player_keys.items():
                d[f"{side}_{pk}_{k}"] = _e(
                    f"{SIDE_CN[side]}{POS_CN[pk]}-{cn[0]}", cn[1],
                    f"{SIDE_EN[side]} {POS_EN[pk]} {en[0]}", en[1]
                )

    # 位置级差分特征
    # 【Bug 修复】原代码错误使用 cn[0]/en[0]（分类名），导致同一分类下多个特征描述相同
    #   例如 diff_sup_player_overall_recent_wr/kda/games 三个特征都显示为
    #   "Support Overall Form Difference"，SHAP 图中表现为同名重复
    # 修复：使用 cn[1]/en[1]（描述名），确保每个特征有独立描述
    for pk in pos_keys:
        for k, (cn, en) in player_keys.items():
            d[f"diff_{pk}_{k}"] = _e(
                f"{POS_CN[pk]}对位差-{cn[0]}", f"双方{POS_CN[pk]}{cn[1]}差值",
                f"{POS_EN[pk]} Lane Diff - {en[0]}", f"{POS_EN[pk]} {en[1]} Difference"
            )

    # 队伍聚合特征 (avg/max/std)
    team_agg_keys = {
        "mastery_score":              (("英雄熟练度", "Champion Mastery")),
        "player_recent_kda_90d":      (("近90天KDA", "90d KDA")),
        "player_recent_wr_90d":       (("近90天胜率", "90d WR")),
        "player_recent_games_90d":    (("近90天出场", "90d Games")),
        "player_overall_recent_wr":   (("综合胜率", "Recent WR")),
        "player_overall_recent_kda":  (("综合KDA", "Recent KDA")),
        "player_overall_recent_games":(("总出场数", "Total Games")),
        "meta_win_rate_pit":          (("版本胜率", "Meta WR")),
        "meta_patch_drift_index":     (("版本漂移", "Patch Drift")),
        "meta_pick_drift_index":      (("选取漂移", "Pick Drift")),
    }
    for side in ["blue", "red"]:
        for agg_cn_key, agg_cn_val in AGG_CN.items():
            for k, (cn_name, en_name) in team_agg_keys.items():
                d[f"{side}_team_{agg_cn_key}_{k}"] = _e(
                    f"{SIDE_CN[side]}团队-{agg_cn_val}{cn_name}", f"全队{cn_name}的{agg_cn_val}",
                    f"{SIDE_EN[side]} Team {AGG_EN[agg_cn_key]} {en_name}", f"Team {en_name} {AGG_EN[agg_cn_key]}"
                )

    # 队伍聚合差分特征
    for agg_cn_key, agg_cn_val in AGG_CN.items():
        for k, (cn_name, en_name) in team_agg_keys.items():
            d[f"diff_team_{agg_cn_key}_{k}"] = _e(
                f"团队差-{agg_cn_val}{cn_name}", f"双方{cn_name}{agg_cn_val}的差值",
                f"Team Diff - {AGG_EN[agg_cn_key]} {en_name}", f"Team {en_name} {AGG_EN[agg_cn_key]} Diff"
            )

    # 队伍画像差分
    for k, (cn, en) in team_profile.items():
        d[f"diff_team_{k}"] = _e(
            f"队伍差-{cn[0]}", f"双方{cn[1]}差值",
            f"Team Diff - {en[0]}", f"{en[1]} Difference"
        )

    # 英雄偏差与熟练度交互特征
    interaction_keys = {
        "champ_wr_delta":  (("英雄偏差", "选手该英雄胜率偏差"),        ("Champion Delta", "Player Champion WR Delta")),
        "champ_kda_delta": (("英雄偏差", "选手该英雄KDA偏差"),         ("Champion Delta", "Player Champion KDA Delta")),
        "mastery_x_presence": (("熟练度×登场", "熟练度与登场率交互"),  ("Mastery×Presence", "Mastery×Presence Interaction")),
        "mastery_x_wr":      (("熟练度×胜率", "熟练度与版本胜率交互"), ("Mastery×Winrate", "Mastery×WR Interaction")),
    }
    for side in ["blue", "red"]:
        for pk in pos_keys:
            for k, (cn, en) in interaction_keys.items():
                d[f"{side}_{pk}_{k}"] = _e(
                    f"{SIDE_CN[side]}{POS_CN[pk]}-{cn[0]}", cn[1],
                    f"{SIDE_EN[side]} {POS_EN[pk]} {en[0]}", en[1]
                )

    # 交互差分特征
    diff_interaction = {
        "champ_wr_delta":        (("胜率偏差差", "Winrate Delta Diff")),
        "champ_kda_delta":       (("KDA偏差差", "KDA Delta Diff")),
        "mastery_x_wr":          (("熟练度×胜率差", "Mastery×WR Diff")),
        "mastery_x_meta_wr":     (("熟练度×Meta胜率差", "Mastery×Meta WR Diff")),
        "mastery_x_player_wr":   (("熟练度×选手胜率差", "Mastery×Player WR Diff")),
    }
    for pk in pos_keys:
        for k, (cn_name, en_name) in diff_interaction.items():
            d[f"diff_{pk}_{k}"] = _e(
                f"{POS_CN[pk]}对位交互差", cn_name,
                f"{POS_EN[pk]} Interaction Diff", en_name
            )

    # 胜率平衡特征
    balance_keys = {
        "wr_max_gap":      (("平衡", "队伍胜率最大差距（选手间不平衡度）"), ("Balance", "Max Player Winrate Gap")),
        "wr_balance":      (("平衡", "队伍胜率均衡度"),                  ("Balance", "Team Winrate Balance")),
        "wr_x_roster_wr":  (("平衡", "Meta胜率×选手胜率交互"),           ("Balance", "Meta×Player WR Interaction")),
    }
    for side in ["blue", "red"]:
        for k, (cn, en) in balance_keys.items():
            d[f"{side}_team_{k}"] = _e(
                f"{SIDE_CN[side]}{cn[0]}", cn[1],
                f"{SIDE_EN[side]} {en[0]}", en[1]
            )
    d["diff_team_wr_max_gap"]     = _e("平衡差", "双方胜率最大差距的差", "Balance Diff", "Max Winrate Gap Diff")
    d["diff_team_wr_balance"]     = _e("平衡差", "双方胜率均衡度差",     "Balance Diff", "Winrate Balance Diff")
    d["diff_team_wr_x_roster_wr"] = _e("平衡差", "双方Meta×选手胜率交互差", "Balance Diff", "Meta×Player WR Diff")

    # 阵容特征 comp_*
    comp_keys = {
        "engage_sum":           (("阵容-强开", "主动开团手段总和 (抓机会能力)"),       ("Comp-Engage", "Total Engage Tools")),
        "engage_avg":           (("阵容-强开", "平均强开能力"),     ("Comp-Engage", "Avg Engage")),
        "poke_sum":             (("阵容-消耗", "总Poke手段 (战前消耗与守塔能力)"),       ("Comp-Poke", "Total Poke/Siege Potential")),
        "poke_avg":             (("阵容-消耗", "平均Poke能力"),     ("Comp-Poke", "Avg Poke")),
        "disengage_sum":        (("阵容-反手", "拉扯与反打保护能力"),       ("Comp-Disengage", "Peel and Counter-engage")),
        "disengage_avg":        (("阵容-反手", "平均反手能力"),     ("Comp-Disengage", "Avg Disengage")),
        "sustain_sum":          (("阵容-续航", "回复与团战持久战能力"),       ("Comp-Sustain", "Healing and Fight Sustain")),
        "sustain_avg":          (("阵容-续航", "平均续航能力"),     ("Comp-Sustain", "Avg Sustain")),
        "burst_sum":            (("阵容-爆发", "瞬间秒杀C位的融化能力"),       ("Comp-Burst", "Instant Target Deletion")),
        "burst_avg":            (("阵容-爆发", "平均爆发能力"),     ("Comp-Burst", "Avg Burst")),
        "tankiness_sum":        (("阵容-坦度", "前排承伤与硬度"),           ("Comp-Tankiness", "Frontline Durability")),
        "tankiness_avg":        (("阵容-坦度", "平均坦度"),         ("Comp-Tankiness", "Avg Tankiness")),
        "mobility_sum":         (("阵容-机动", "位移与转线拉扯速度"),         ("Comp-Mobility", "Total Mobility/Dash")),
        "mobility_avg":         (("阵容-机动", "平均机动性"),       ("Comp-Mobility", "Avg Mobility")),
        "damage_type_balance":  (("阵容-伤害", "AD/AP伤害均衡度 (避免菜刀队被堆护甲)"), ("Comp-Damage", "AD/AP Damage Balance (Avoid Armor Stacking)")),
        "cc_score_sum":         (("阵容-控制", "硬控与连环控制链容错率"),     ("Comp-CC", "Hard CC Chain & Forgiveness")),
        "cc_score_avg":         (("阵容-控制", "平均控制得分"),     ("Comp-CC", "Avg CC Score")),
        "waveclear_sum":        (("阵容-清线", "兵线处理与防守拖后期能力"),       ("Comp-Wave Clear", "Minion Wave Clearing Power")),
        "waveclear_avg":        (("阵容-清线", "平均清线能力"),     ("Comp-Wave Clear", "Avg Wave Clear")),
        "early_power":          (("阵容-发力期", "前期强势度"),     ("Comp-Power Spike", "Early Game Strength")),
        "late_power":           (("阵容-发力期", "后期强势度"),     ("Comp-Power Spike", "Late Game Scaling")),
        "teamfight_score":      (("阵容-团战", "团战得分"),         ("Comp-Teamfight", "Teamfight Score")),
        "aggression_index":     (("阵容-风格", "进攻性指数"),       ("Comp-Style", "Aggression Index")),
        "scaling_type":         (("阵容-类型", "阵容成长类型"),     ("Comp-Type", "Comp Scaling Type")),
        "lane_dom_type":        (("阵容-线权", "线权主导类型"),     ("Comp-Lane Priority", "Lane Priority Type")),
        "teamfight_type":       (("阵容-团战类型", "团战类型"),     ("Comp-Teamfight Type", "Teamfight Type")),
    }
    for side in ["blue", "red"]:
        for k, (cn, en) in comp_keys.items():
            d[f"{side}_comp_{k}"] = _e(
                f"{SIDE_CN[side]}{cn[0]}", cn[1],
                f"{SIDE_EN[side]} {en[0]}", en[1]
            )
    for k, (cn, en) in comp_keys.items():
        cn_sub = cn[0].split("-")[-1]
        en_sub = en[0].split("-")[-1]
        d[f"diff_comp_{k}"] = _e(
            f"阵容差-{cn_sub}", f"双方{cn[1]}差值",
            f"Comp Diff - {en_sub}", f"{en[1]} Difference"
        )

    # LPL/风格交互特征
    lpl_keys = {
        "bloodiness_x_aggression":  (("血腥×进攻", "血腥度×进攻性交互"),    ("Blood×Aggression", "Bloodiness×Aggression")),
        "early_power_x_snowball":   (("前期×滚雪球", "前期强势×滚雪球能力"),("Early×Snowball", "Early Power×Snowball")),
        "ckpm_x_aggression":        (("节奏×进攻", "CKPM×进攻性交互"),      ("Tempo×Aggression", "CKPM×Aggression")),
        "wr_x_aggression":          (("胜率×进攻", "胜率×进攻性交互"),      ("WR×Aggression", "Winrate×Aggression")),
    }
    for side in ["blue", "red"]:
        for k, (cn, en) in lpl_keys.items():
            d[f"{side}_{k}"] = _e(
                f"{SIDE_CN[side]}风格-{cn[0]}", cn[1],
                f"{SIDE_EN[side]} Style - {en[0]}", en[1]
            )
    for k, (cn, en) in lpl_keys.items():
        d[f"diff_{k}"] = _e(
            f"风格差-{cn[0]}", f"双方{cn[1]}差值",
            f"Style Diff - {en[0]}", f"{en[1]} Difference"
        )

    # TF 特征 (Transformer 深度特征)
    d["tf_win_logits"]    = _e("深度学习", "Transformer阵容胜率logit",     "Deep Learning", "Transformer Comp Win Logit")
    d["tf_cosine_sim"]    = _e("深度学习", "双方阵容相似度",               "Deep Learning", "Comp Cosine Similarity")
    d["tf_blue_l2norm"]   = _e("深度学习", "首选方阵容embedding范数",      "Deep Learning", "First Pick Comp Embedding Norm")
    d["tf_red_l2norm"]    = _e("深度学习", "次选方阵容embedding范数",      "Deep Learning", "Second Pick Comp Embedding Norm")

    return d


FEATURE_EXPLAIN_DICT = _build_feature_explain_dict()


def explain_feature(feature_name):
    """将特征名映射为 ((cat_cn, desc_cn), (cat_en, desc_en))，支持前缀匹配"""
    if feature_name in FEATURE_EXPLAIN_DICT:
        return FEATURE_EXPLAIN_DICT[feature_name]
    # 前缀匹配（数字后缀如 _1, _2 等）
    for prefix, entry in FEATURE_EXPLAIN_DICT.items():
        if feature_name.startswith(prefix + "_") or feature_name == prefix:
            return entry
    cn = ("其他特征", feature_name)
    en = ("Other", feature_name)
    return (cn, en)


# ============================================================
# 2. SHAP值计算
# ============================================================
def compute_shap_values(model, X_infer, feature_cols, top_k_pos=5, top_k_neg=5):
    """
    计算单条样本的SHAP值，返回Top-K正向和负向特征。

    Args:
        model: CatBoost模型（已加载）
        X_infer: numpy array, shape (1, n_features)
        feature_cols: list, 特征名列表
        top_k_pos: 正向贡献Top-K
        top_k_neg: 负向贡献Top-K

    Returns:
        dict: {
            "base_value": float,        # 基准值（log-odds空间，0.5概率对应0）
            "final_logit": float,       # 最终logit值
            "positive": [...],          # Top正向贡献 [{feature, category, desc, shap_value, shap_pct}]
            "negative": [...],          # Top负向贡献
            "waterfall_data": [...]     # 瀑布图数据（有序）
        }
    """
    try:
        pool = Pool(X_infer)
        shap_values = model.get_feature_importance(type='ShapValues', data=pool)
        shap_vals = shap_values[0, :-1]  # shape (n_features,)
        expected_value = float(shap_values[0, -1])
    except Exception as e:
        log.error(f"SHAP计算失败: {e}")
        return {"error": str(e), "positive": [], "negative": [], "waterfall_data": []}

    final_logit = float(expected_value + shap_vals.sum())

    # 构建特征解释列表
    feature_contribs = []
    for i, (fname, sv) in enumerate(zip(feature_cols, shap_vals)):
        (cat_cn, desc_cn), (cat_en, desc_en) = explain_feature(fname)
        feature_contribs.append({
            "feature": fname,
            "category": cat_cn,
            "desc": desc_cn,
            "category_en": cat_en,
            "desc_en": desc_en,
            "shap_value": float(sv),
        })

    # 排序并取Top
    sorted_pos = sorted(feature_contribs, key=lambda x: -x["shap_value"])
    sorted_neg = sorted(feature_contribs, key=lambda x: x["shap_value"])

    positive = [f for f in sorted_pos[:top_k_pos] if f["shap_value"] > 1e-6]
    negative = [f for f in sorted_neg[:top_k_neg] if f["shap_value"] < -1e-6]

    # 瀑布图数据：先展示正向贡献（蓝色/利于首选方，按降序），再展示负向贡献（红色/利于次选方，按绝对值降序）
    waterfall = positive + negative  # 先大正→小正，再小负→大负
    max_abs = max((abs(f["shap_value"]) for f in waterfall), default=1.0)
    for f in waterfall:
        f["bar_pct"] = round(abs(f["shap_value"]) / max_abs * 100, 1) if max_abs > 0 else 0
        f["shap_text"] = f"{f['shap_value']:+.4f}"

    # 转换shap_value到概率空间近似值（用于理解）
    def logit_to_prob(l):
        return 1.0 / (1.0 + np.exp(-l))

    base_prob = logit_to_prob(expected_value)
    final_prob = logit_to_prob(final_logit)

    # 计算每个特征在概率空间的近似贡献（百分点）
    current_logit = expected_value
    for f in waterfall:
        before_p = logit_to_prob(current_logit)
        current_logit += f["shap_value"]
        after_p = logit_to_prob(current_logit)
        f["prob_delta"] = round((after_p - before_p) * 100, 2)  # 百分点

    return {
        "base_value": round(expected_value, 4),
        "base_prob": round(base_prob, 4),
        "final_logit": round(final_logit, 4),
        "final_prob": round(final_prob, 4),
        "positive": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "bar_pct": f["bar_pct"],
        } for f in positive],
        "negative": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "bar_pct": f["bar_pct"],
        } for f in negative],
        "waterfall_data": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "direction": "positive" if f["shap_value"] > 0 else "negative",
            "bar_pct": f["bar_pct"],
        } for f in waterfall],
    }


def _select_representative_models(models_dict, max_rep=5):
    """从完整模型集合中选取具有代表性的子集。

    选择策略:
    - 生产模式 (key="production"): 7个seed中均匀间隔取5个
    - 折模式 (key=0..4): 每个fold取第1个seed，共5个
    - 不足max_rep个模型时全取

    Args:
        models_dict: dict, {fold_key: [model, ...]} 或 {"production": [model, ...]}
        max_rep: 最多选取的代表模型数

    Returns:
        list of models
    """
    rep_models = []
    if "production" in models_dict:
        prod_models = models_dict["production"]
        n = len(prod_models)
        if n <= max_rep:
            rep_models = list(prod_models)
        else:
            step = n / max_rep
            indices = sorted(set(int(i * step) for i in range(max_rep)))
            indices = [min(i, n - 1) for i in indices]
            rep_models = [prod_models[i] for i in indices]
    else:
        for fk in sorted(models_dict.keys()):
            fold_models = models_dict[fk]
            if fold_models:
                rep_models.append(fold_models[0])
            if len(rep_models) >= max_rep:
                break
    return rep_models


def _prob_to_logit(p):
    """概率转log-odds，带数值截断防止溢出"""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(np.log(p / (1.0 - p)))


def compute_calibrated_shap_values(rep_models, X_infer, feature_cols,
                                    target_prob, top_k_pos=5, top_k_neg=5):
    """
    使用代表子集+基线校准计算SHAP值，性能提升5-7倍，同时保证解释与最终预测完全对齐。

    核心原理:
    1. 仅对N个代表模型（通常5个）计算SHAP值，而非全部35个
    2. TreeSHAP在log-odds空间满足加法公理: f(x) = φ0 + Σφi
    3. 计算代表模型平均预测Pred_rep和真实集成预测Pred_full
    4. 将残差Δ=logit(Pred_full)-logit(Pred_rep)加到基线φ0上
    5. 校准后 SHAP waterfall 精确累加至真实集成预测值

    Args:
        rep_models: 代表模型列表（建议5个）
        X_infer: numpy array, shape (1, n_features)
        feature_cols: list, 特征名列表
        target_prob: float, 完整集成模型的预测概率（0-1），用于校准基线
        top_k_pos: 正向贡献Top-K
        top_k_neg: 负向贡献Top-K

    Returns:
        dict: 校准后的SHAP结果，与compute_shap_values格式兼容
    """
    if not rep_models:
        return {"error": "No representative models provided", "positive": [], "negative": [], "waterfall_data": []}

    n_models = len(rep_models)
    n_features = X_infer.shape[1]
    all_shap_vals = np.zeros(n_features, dtype=np.float64)
    all_expected_vals = []
    rep_probs = []

    for i, model in enumerate(rep_models):
        try:
            pool = Pool(X_infer)
            shap_values = model.get_feature_importance(type='ShapValues', data=pool)
            shap_vals = shap_values[0, :-1]
            expected_val = float(shap_values[0, -1])
            all_shap_vals += shap_vals
            all_expected_vals.append(expected_val)
            pred_p = float(model.predict_proba(X_infer)[0, 1])
            rep_probs.append(pred_p)
        except Exception as e:
            log.warning(f"校准SHAP: 模型 {i}/{n_models} 计算失败: {e}")
            continue

    if not all_expected_vals:
        return {"error": "All models failed SHAP computation", "positive": [], "negative": [], "waterfall_data": []}

    n_used = len(all_expected_vals)
    avg_shap_vals = all_shap_vals / n_used
    avg_expected_val = float(np.mean(all_expected_vals))
    rep_avg_prob = float(np.mean(rep_probs))

    rep_logit = _prob_to_logit(rep_avg_prob)
    target_logit = _prob_to_logit(target_prob)
    delta_logit = target_logit - rep_logit
    calibrated_expected_val = avg_expected_val + delta_logit
    final_logit = float(calibrated_expected_val + avg_shap_vals.sum())

    feature_contribs = []
    for i, (fname, sv) in enumerate(zip(feature_cols, avg_shap_vals)):
        (cat_cn, desc_cn), (cat_en, desc_en) = explain_feature(fname)
        feature_contribs.append({
            "feature": fname,
            "category": cat_cn,
            "desc": desc_cn,
            "category_en": cat_en,
            "desc_en": desc_en,
            "shap_value": float(sv),
        })

    sorted_pos = sorted(feature_contribs, key=lambda x: -x["shap_value"])
    sorted_neg = sorted(feature_contribs, key=lambda x: x["shap_value"])

    positive = [f for f in sorted_pos[:top_k_pos] if f["shap_value"] > 1e-6]
    negative = [f for f in sorted_neg[:top_k_neg] if f["shap_value"] < -1e-6]

    waterfall = positive + negative
    max_abs = max((abs(f["shap_value"]) for f in waterfall), default=1.0)
    for f in waterfall:
        f["bar_pct"] = round(abs(f["shap_value"]) / max_abs * 100, 1) if max_abs > 0 else 0
        f["shap_text"] = f"{f['shap_value']:+.4f}"

    def logit_to_prob(l):
        return 1.0 / (1.0 + np.exp(-l))

    base_prob = logit_to_prob(calibrated_expected_val)
    final_prob_calibrated = logit_to_prob(final_logit)

    current_logit = calibrated_expected_val
    for f in waterfall:
        before_p = logit_to_prob(current_logit)
        current_logit += f["shap_value"]
        after_p = logit_to_prob(current_logit)
        f["prob_delta"] = round((after_p - before_p) * 100, 2)

    return {
        "base_value": round(calibrated_expected_val, 4),
        "base_prob": round(base_prob, 4),
        "final_logit": round(final_logit, 4),
        "final_prob": round(final_prob_calibrated, 4),
        "target_prob": round(target_prob, 4),
        "rep_avg_prob": round(rep_avg_prob, 4),
        "calibration_delta_logit": round(delta_logit, 4),
        "n_models_used": n_used,
        "calibrated": True,
        "positive": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "bar_pct": f["bar_pct"],
        } for f in positive],
        "negative": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "bar_pct": f["bar_pct"],
        } for f in negative],
        "waterfall_data": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "direction": "positive" if f["shap_value"] > 0 else "negative",
            "bar_pct": f["bar_pct"],
        } for f in waterfall],
    }


def compute_ensemble_shap_values(models, X_infer, feature_cols, top_k_pos=5, top_k_neg=5):
    """
    [已废弃，保留用于兼容] 请使用 compute_calibrated_shap_values 获得更好性能。

    计算集成模型的平均SHAP值。保留此函数是为了向后兼容，
    新代码应使用 compute_calibrated_shap_values 以获得5-7倍性能提升。
    """
    if not models:
        return {"error": "No models provided for ensemble SHAP", "positive": [], "negative": [], "waterfall_data": []}

    n_models = len(models)
    n_features = X_infer.shape[1]
    all_shap_vals = np.zeros(n_features, dtype=np.float64)
    all_expected_vals = []

    for i, model in enumerate(models):
        try:
            pool = Pool(X_infer)
            shap_values = model.get_feature_importance(type='ShapValues', data=pool)
            shap_vals = shap_values[0, :-1]
            expected_val = float(shap_values[0, -1])
            all_shap_vals += shap_vals
            all_expected_vals.append(expected_val)
        except Exception as e:
            log.warning(f"集成SHAP: 模型 {i}/{n_models} SHAP计算失败: {e}")
            continue

    if not all_expected_vals:
        return {"error": "All models failed SHAP computation", "positive": [], "negative": [], "waterfall_data": []}

    avg_shap_vals = all_shap_vals / len(all_expected_vals)
    avg_expected_val = float(np.mean(all_expected_vals))
    final_logit = float(avg_expected_val + avg_shap_vals.sum())

    feature_contribs = []
    for i, (fname, sv) in enumerate(zip(feature_cols, avg_shap_vals)):
        (cat_cn, desc_cn), (cat_en, desc_en) = explain_feature(fname)
        feature_contribs.append({
            "feature": fname,
            "category": cat_cn,
            "desc": desc_cn,
            "category_en": cat_en,
            "desc_en": desc_en,
            "shap_value": float(sv),
        })

    sorted_pos = sorted(feature_contribs, key=lambda x: -x["shap_value"])
    sorted_neg = sorted(feature_contribs, key=lambda x: x["shap_value"])

    positive = [f for f in sorted_pos[:top_k_pos] if f["shap_value"] > 1e-6]
    negative = [f for f in sorted_neg[:top_k_neg] if f["shap_value"] < -1e-6]

    waterfall = positive + negative
    max_abs = max((abs(f["shap_value"]) for f in waterfall), default=1.0)
    for f in waterfall:
        f["bar_pct"] = round(abs(f["shap_value"]) / max_abs * 100, 1) if max_abs > 0 else 0
        f["shap_text"] = f"{f['shap_value']:+.4f}"

    def logit_to_prob(l):
        return 1.0 / (1.0 + np.exp(-l))

    base_prob = logit_to_prob(avg_expected_val)
    final_prob = logit_to_prob(final_logit)

    current_logit = avg_expected_val
    for f in waterfall:
        before_p = logit_to_prob(current_logit)
        current_logit += f["shap_value"]
        after_p = logit_to_prob(current_logit)
        f["prob_delta"] = round((after_p - before_p) * 100, 2)

    return {
        "base_value": round(avg_expected_val, 4),
        "base_prob": round(base_prob, 4),
        "final_logit": round(final_logit, 4),
        "final_prob": round(final_prob, 4),
        "n_models_used": len(all_expected_vals),
        "positive": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "bar_pct": f["bar_pct"],
        } for f in positive],
        "negative": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "bar_pct": f["bar_pct"],
        } for f in negative],
        "waterfall_data": [{
            "feature": f["feature"],
            "category": f["category"],
            "desc": f["desc"],
            "category_en": f["category_en"],
            "desc_en": f["desc_en"],
            "shap_value": round(f["shap_value"], 4),
            "prob_delta": f["prob_delta"],
            "direction": "positive" if f["shap_value"] > 0 else "negative",
            "bar_pct": f["bar_pct"],
        } for f in waterfall],
    }


# ============================================================
# 3. 分路counter关系分析
# ============================================================
def analyze_counters(blue_champs, red_champs):
    """
    分析每条分路上的counter关系。

    Args:
        blue_champs: dict, {position: champion_name} e.g. {"top": "Aatrox", ...}
        red_champs: dict, {position: champion_name}

    Returns:
        list of dicts: [{
            position, position_cn, blue_champ, red_champ,
            counter_side: "blue"|"red"|"none",
            counter_desc: str,
            win_rate: float,  # counter方胜率
            games: int,
        }]
    """
    df = _load_counters()
    results = []

    for pos in ["top", "jungle", "mid", "bot", "support"]:
        blue_c = blue_champs.get(pos, "")
        red_c = red_champs.get(pos, "")
        pos_cn = POS_NAMES_CN.get(pos, pos)
        pos_en = POS_NAMES_EN.get(pos, pos)

        if not blue_c or not red_c:
            results.append({
                "position": pos, "position_cn": pos_cn, "position_en": pos_en,
                "blue_champ": blue_c, "red_champ": red_c,
                "counter_side": "none",
                "counter_desc": "双方英雄势均力敌",
                "counter_desc_en": "Even matchup",
                "blue_wr": 50.0, "win_rate": 50.0, "games": 0,
            })
            continue

        if df.empty:
            results.append({
                "position": pos, "position_cn": pos_cn, "position_en": pos_en,
                "blue_champ": blue_c, "red_champ": red_c,
                "counter_side": "none",
                "counter_desc": "双方英雄势均力敌",
                "counter_desc_en": "Even matchup",
                "blue_wr": 50.0, "win_rate": 50.0, "games": 0,
            })
            continue

        # 查找蓝方视角: blue vs red（先按位置精确查找）
        blue_persp = df[(df["champion"] == blue_c) & (df["opponent_name"] == red_c)
                        & (df["position"] == pos)]
        red_persp = df[(df["champion"] == red_c) & (df["opponent_name"] == blue_c)
                       & (df["position"] == pos)]

        # 如果精确位置没找到，fallback到全局查找（取对局数最多的记录，解决bot位置数据缺失问题）
        if blue_persp.empty and red_persp.empty:
            blue_persp_all = df[(df["champion"] == blue_c) & (df["opponent_name"] == red_c)]
            red_persp_all = df[(df["champion"] == red_c) & (df["opponent_name"] == blue_c)]
            if not blue_persp_all.empty:
                blue_persp = blue_persp_all.sort_values("games", ascending=False).head(1)
            if not red_persp_all.empty:
                red_persp = red_persp_all.sort_values("games", ascending=False).head(1)

        blue_wr = None
        games = 0
        if not blue_persp.empty:
            row = blue_persp.iloc[0]
            blue_wr = float(row["win_rate"])
            games = int(row["games"])
        elif not red_persp.empty:
            row = red_persp.iloc[0]
            blue_wr = 1.0 - float(row["win_rate"])
            games = int(row["games"])

        if blue_wr is None:
            results.append({
                "position": pos, "position_cn": pos_cn, "position_en": pos_en,
                "blue_champ": blue_c, "red_champ": red_c,
                "counter_side": "none",
                "counter_desc": "无对局数据，默认均势",
                "counter_desc_en": "No matchup data",
                "blue_wr": 50.0, "win_rate": 50.0, "games": 0,
            })
        elif blue_wr > 0.52:
            results.append({
                "position": pos, "position_cn": pos_cn, "position_en": pos_en,
                "blue_champ": blue_c, "red_champ": red_c,
                "counter_side": "blue",
                # 更具电竞风格的克制描述，带上显著压制感
                "counter_desc": f"{_champ_to_zh(blue_c)} 线上压制 {_champ_to_zh(red_c)}",
                "counter_desc_en": f"{blue_c} dominates {red_c} in lane",
                "blue_wr": round(blue_wr * 100, 1), "win_rate": round(blue_wr * 100, 1), "games": games,
            })
        elif blue_wr < 0.48:
            red_wr = 1.0 - blue_wr
            results.append({
                "position": pos, "position_cn": pos_cn, "position_en": pos_en,
                "blue_champ": blue_c, "red_champ": red_c,
                "counter_side": "red",
                "counter_desc": f"{_champ_to_zh(red_c)} 线上压制 {_champ_to_zh(blue_c)}",
                "counter_desc_en": f"{red_c} dominates {blue_c} in lane",
                "blue_wr": round(blue_wr * 100, 1), "win_rate": round(red_wr * 100, 1), "games": games,
            })
        else:
            results.append({
                "position": pos, "position_cn": pos_cn, "position_en": pos_en,
                "blue_champ": blue_c, "red_champ": red_c,
                "counter_side": "even",
                "counter_desc": "五五开 (均势对位)",
                "counter_desc_en": "Skill Matchup (Even)",
                "blue_wr": round(blue_wr * 100, 1), "win_rate": round(max(blue_wr, 1 - blue_wr) * 100, 1),
                "games": games,
            })

    return results


# ============================================================
# 4. 英雄组合synergy关系分析
# ============================================================
def analyze_synergy(blue_champs, red_champs, min_win_rate=0.45, min_games=3):
    """
    分析所有同队英雄组合的synergy关系。

    Args:
        blue_champs: dict {position: champion_name}
        red_champs: dict {position: champion_name}
        min_win_rate: 最低展示胜率阈值（>0.5才展示）
        min_games: 最低对局数阈值

    Returns:
        dict: {"blue": [...], "red": [...]} 每方的协同组合列表
    """
    df = _load_synergy()
    results = {"blue": [], "red": []}

    if df.empty:
        return results

    def _find_synergies(champs, side):
        pairs = []
        positions = [p for p in ["top", "jungle", "mid", "bot", "support"] if champs.get(p)]
        synergy_dict_cn = {
            ("top", "jungle"): "上野联动", ("mid", "jungle"): "中野代练", 
            ("bot", "support"): "下路双人组", ("jungle", "support"): "野辅双游控图",
            ("mid", "support"): "中辅游走", ("mid", "bot"): "双C核心", 
            ("top", "mid"): "单人路摇摆"
        }
        synergy_dict_en = {
            ("top", "jungle"): "Top-Jungle Synergy", ("mid", "jungle"): "Mid-Jungle Duo", 
            ("bot", "support"): "Botlane Duo", ("jungle", "support"): "Jungle-Sup Roam",
            ("mid", "support"): "Mid-Sup Roam", ("mid", "bot"): "Dual Carries"
        }

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pi, pj = positions[i], positions[j]
                ci, cj = champs[pi], champs[pj]
                if ci == cj:
                    continue

                # 查找 ci + cj synergy
                match = df[(df["champion"] == ci) & (df["opponent_name"] == cj)]
                if match.empty:
                    match = df[(df["champion"] == cj) & (df["opponent_name"] == ci)]

                if match.empty:
                    continue
                row = match.iloc[0]
                wr = float(row["win_rate"])
                games = int(row["games"])
                pos_cn_i = POS_NAMES_CN.get(pi, pi)
                pos_cn_j = POS_NAMES_CN.get(pj, pj)
                pos_en_i = POS_NAMES_EN.get(pi, pi)
                pos_en_j = POS_NAMES_EN.get(pj, pj)

                if wr > min_win_rate and games >= min_games:
                    # 匹配专业黑话
                    pair_key = tuple(sorted([pi, pj])) # 确保顺序无关可以命中字典
                    pair_key_alt = tuple(sorted([pi, pj], reverse=True))
                    
                    syn_cn = synergy_dict_cn.get(pair_key, synergy_dict_cn.get(pair_key_alt, f"{pos_cn_i}&{pos_cn_j}协同"))
                    syn_en = synergy_dict_en.get(pair_key, synergy_dict_en.get(pair_key_alt, f"{pos_en_i}&{pos_en_j} Synergy"))

                    pairs.append({
                        "champ1": ci, "champ2": cj,
                        "pos1": pos_cn_i, "pos2": pos_cn_j,
                        "pos1_en": pos_en_i, "pos2_en": pos_en_j,
                        "win_rate": round(wr * 100, 1),
                        "games": games,
                        "desc": f"【{syn_cn}】{ci} + {cj}",
                        "desc_en": f"[{syn_en}] {ci} + {cj}",
                    })
        pairs.sort(key=lambda x: -x["win_rate"])
        return pairs

    results["blue"] = _find_synergies(blue_champs, "blue")
    results["red"] = _find_synergies(red_champs, "red")
    return results


# ============================================================
# 5. 英雄数据统计
# ============================================================
def get_champion_stats(blue_champs, red_champs, meta_stats=None):
    """
    获取10个英雄的出场率和/登场率和胜率。
    按用户指定的分路查询该英雄在该位置上的数据，确保数据与分路匹配。
    查询顺序：
      1) champion_ranks_cleaned.csv（按位置分，lol.ps职业赛场统计）按 champion + position 匹配
      2) 回退到 merged_champion_stats.csv（按英雄主位置融合统计）
      3) 最后回退到 meta_stats（动态Meta特征）

    Args:
        blue_champs: dict {position: champion_name}
        red_champs: dict {position: champion_name}
        meta_stats: dict {champion_name: {meta_presence, meta_win_rate, ...}}

    Returns:
        list: [{side, position, position_cn, champion, presence_pct, win_rate_pct}]
    """
    ranks_df = _load_champion_ranks()  # 按位置分（position 列）
    merged_df = _load_merged_stats()   # 按英雄主位置融合
    stats = []
    for side, champs in [("blue", blue_champs), ("red", red_champs)]:
        for pos in ["top", "jungle", "mid", "bot", "support"]:
            name = champs.get(pos, "")
            presence = 0.0
            wr = 50.0
            found = False

            # 1) 优先在 champion_ranks_cleaned.csv 中按 champion + position 匹配
            if name and not ranks_df.empty and "position" in ranks_df.columns:
                match = ranks_df[(ranks_df["champion"] == name) & (ranks_df["position"] == pos)]
                if not match.empty:
                    row = match.iloc[0]
                    p = float(row.get("presence_rate", 0) or 0)
                    w = float(row.get("win_rate", 0.5) or 0.5)
                    if p > 0:
                        presence = min(p * 100, 99.9)
                        wr = w * 100
                        found = True

            # 2) 回退到 merged_champion_stats.csv（按英雄主位置）
            if not found and name and not merged_df.empty:
                match = merged_df[merged_df["champion"] == name]
                if not match.empty:
                    row = match.iloc[0]
                    p = float(row.get("presence_rate", 0) or 0)
                    w = float(row.get("win_rate", 0.5) or 0.5)
                    if p > 0:
                        presence = min(p * 100, 99.9)
                        wr = w * 100
                        found = True

            # 3) 最后回退到 meta_stats
            if not found and name and meta_stats and name in meta_stats:
                m = meta_stats[name]
                presence_val = m.get("meta_presence", m.get("meta_pick_rate", 0)) or 0
                presence = min(float(presence_val) * 100, 99.9)
                wr_val = m.get("meta_win_rate", 0.5) or 0.5
                wr = float(wr_val) * 100

            stats.append({
                "side": side,
                "position": pos,
                "position_cn": POS_NAMES_CN.get(pos, pos),
                "position_en": POS_NAMES_EN.get(pos, pos),
                "champion": name,
                "presence_pct": round(presence, 1),
                "win_rate_pct": round(wr, 1),
            })
    return stats
