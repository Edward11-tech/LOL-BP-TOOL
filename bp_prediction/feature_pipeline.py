"""
离线特征工程流水线
====================
构建 BP 胜负预测模型的训练特征数据集，严格遵循 Point-In-Time (PIT) 原则。

功能描述:
  从清洗后的比赛数据、选手生涯统计、英雄元数据等原始数据出发，
  计算以下特征类别并整合为宽表 parquet 供模型训练使用：
  1. Meta 特征: 英雄登场率、胜率（时间衰减 H=14 天，窗口 60 天）
  2. 选手近期状态特征: KDA、胜率等（时间衰减 H=45 天，窗口 90 天）
  3. 战队风格画像: 一血/一龙/一塔率、前期节奏指标
  4. BP 特征: Ban/Pick 顺序、counter/synergy 关系
  5. 阵容特征: 组合伤害类型、开团能力等团队构成指标

主要函数:
  - load_matches(league): 加载比赛数据
  - enforce_pit(matches_df): PIT 时间截断
  - compute_meta_features_pit(): 计算 Meta 特征（PIT 安全）
  - compute_player_features_pit(): 计算选手状态特征（PIT 安全）
  - compute_team_profile_pit(): 计算战队画像特征（PIT 安全）
  - extract_bp_features(): 提取 BP 序列特征
  - build_prediction_feature_pipeline(league, save_parquet): 端到端流水线入口

输出:
  bp_prediction/features/ALL_prediction_wide_features.parquet (主训练宽表)
  bp_prediction/features/ALL_meta_store.parquet (英雄元数据存储)
  bp_prediction/features/ALL_player_store.parquet (选手特征存储)
  bp_prediction/features/ALL_team_profile_store.parquet (战队画像存储)

用法:
  python bp_prediction/feature_pipeline.py
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# 必须在 logger_config 导入前设置 sys.path
BASE_DIR = str(Path(__file__).parent.parent.resolve())
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from logger_config import get_logger, setup_logging, log_context, timed

CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "bp_prediction", "features")

log = get_logger(__name__)

# 共享数据异常检测工具
from data_checks import check_dataframe, check_array

from bp_prediction.feature_utils import (
    POSITIONS_SHORT, POS_SHORT2FULL, 
    PLAYER_DEFAULTS, META_DEFAULTS, TEAM_PROFILE_DEFAULTS,
    PLAYER_FEATURE_COLS, META_FEATURE_COLS,
    calculate_derived_features
)

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

PICK_POSITIONS_BLUE = ["top", "jng", "bot", "mid", "sup"]
PICK_POSITIONS_RED = ["jng", "mid", "top", "bot", "sup"]


GAME_RESULT_COLS = [
    "gamelength", "ckpm",
    "blue_firstdragon", "red_firstdragon",
    "blue_firsttower", "red_firsttower",
    "blue_golddiffat15", "red_golddiffat15",
]
PLAYER_RESULT_COLS = []
for _s in ["blue", "red"]:
    for _p in POSITIONS_SHORT:
        for _st in ["kills", "deaths", "assists"]:
            PLAYER_RESULT_COLS.append(f"{_s}_{_p}_{_st}")
ALL_RESULT_COLS = GAME_RESULT_COLS + PLAYER_RESULT_COLS

META_WINDOW_DAYS = 60
META_DECAY_HALF_LIFE = 14
PLAYER_WINDOW_DAYS = 90
PLAYER_DECAY_HALF_LIFE = 45
PLAYER_WINDOW_GAMES = 15
MASTERY_DECAY_HALF_LIFE = 180
GLOBAL_KDA_PRIOR = 3.0
TEAM_PROFILE_WINDOW_DAYS = 90
TEAM_PROFILE_DECAY_HALF_LIFE = 45

BAYESIAN_PRIOR_WEIGHT = 2

# ===============================================
CHAMPION_TAGS = {
    # ================= 上单 (Top) =================
    "Ornn": {"Engage":3,"Poke":0,"Peel":1,"Burst":0,"Tank":3,"HardCC":2,"LineStrength":0},
    "Sion": {"Engage":3,"Poke":0,"Peel":0,"Burst":1,"Tank":3,"HardCC":2,"LineStrength":0},
    "K'Sante": {"Engage":2,"Poke":0,"Peel":1,"Burst":0,"Tank":3,"HardCC":2,"LineStrength":0},
    "Zac": {"Engage":3,"Poke":0,"Peel":1,"Burst":0,"Tank":3,"HardCC":3,"LineStrength":0},
    "Malphite": {"Engage":3,"Poke":1,"Peel":0,"Burst":1,"Tank":2,"HardCC":2,"LineStrength":0},
    "Poppy": {"Engage":2,"Poke":0,"Peel":3,"Burst":1,"Tank":2,"HardCC":2,"LineStrength":0}, 
    "Rumble": {"Engage":0,"Poke":2,"Peel":0,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 兰博线霸属性拉满
    "Jax": {"Engage":1,"Poke":0,"Peel":0,"Burst":1,"Tank":1,"HardCC":1,"LineStrength":0},
    "Camille": {"Engage":3,"Poke":0,"Peel":0,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":0}, 
    "Fiora": {"Engage":0,"Poke":0,"Peel":0,"Burst":1,"Tank":0,"HardCC":0,"LineStrength":0},
    "Gwen": {"Engage":0,"Poke":0,"Peel":0,"Burst":1,"Tank":1,"HardCC":0,"LineStrength":0},
    "Gnar": {"Engage":3,"Poke":2,"Peel":1,"Burst":1,"Tank":2,"HardCC":2,"LineStrength":0}, 
    "Renekton": {"Engage":1,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":1}, # 鳄鱼前期压制力
    "Riven": {"Engage":1,"Poke":0,"Peel":0,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Aatrox": {"Engage":1,"Poke":1,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":0},
    "Sett": {"Engage":2,"Poke":0,"Peel":1,"Burst":2,"Tank":2,"HardCC":1,"LineStrength":0},
    "Mordekaiser": {"Engage":1,"Poke":0,"Peel":2,"Burst":1,"Tank":2,"HardCC":1,"LineStrength":0},
    "Shen": {"Engage":2,"Poke":0,"Peel":3,"Burst":0,"Tank":3,"HardCC":1,"LineStrength":0},
    "Volibear": {"Engage":2,"Poke":0,"Peel":0,"Burst":1,"Tank":2,"HardCC":1,"LineStrength":1}, # 狗熊对线强劲
    "Gragas": {"Engage":2,"Poke":1,"Peel":3,"Burst":2,"Tank":2,"HardCC":2,"LineStrength":0},
    "Udyr": {"Engage":1,"Poke":0,"Peel":0,"Burst":0,"Tank":2,"HardCC":1,"LineStrength":0},
    "Yasuo": {"Engage":1,"Poke":0,"Peel":1,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":0},
    "Yone": {"Engage":2,"Poke":0,"Peel":0,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Irelia": {"Engage":2,"Poke":0,"Peel":0,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Jayce": {"Engage":0,"Poke":3,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":1}, # 杰斯强对线
    "Vladimir": {"Engage":1,"Poke":0,"Peel":0,"Burst":3,"Tank":1,"HardCC":0,"LineStrength":0},
    "Darius": {"Engage":1,"Poke":0,"Peel":0,"Burst":1,"Tank":1,"HardCC":1,"LineStrength":1}, # 诺手数值压制
    "Nasus": {"Engage":0,"Poke":0,"Peel":1,"Burst":0,"Tank":2,"HardCC":1,"LineStrength":0},
    "Trundle": {"Engage":1,"Poke":0,"Peel":0,"Burst":1,"Tank":1,"HardCC":1,"LineStrength":0},
    "Tryndamere": {"Engage":0,"Poke":0,"Peel":0,"Burst":0,"Tank":0,"HardCC":0,"LineStrength":0},
    "Dr. Mundo": {"Engage":1,"Poke":1,"Peel":0,"Burst":0,"Tank":3,"HardCC":0,"LineStrength":0},
    "Garen": {"Engage":1,"Poke":0,"Peel":0,"Burst":2,"Tank":2,"HardCC":0,"LineStrength":0},
    "Tahm Kench": {"Engage":1,"Poke":0,"Peel":3,"Burst":0,"Tank":3,"HardCC":1,"LineStrength":0},
    "Briar": {"Engage":3,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":0},
    "Ambessa": {"Engage":2,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":0},
    "Quinn": {"Engage":1,"Poke":1,"Peel":0,"Burst":2,"Tank":0,"HardCC":0,"LineStrength":1}, # 奎因上路长手压制
    "Illaoi": {"Engage":0,"Poke":1,"Peel":1,"Burst":2,"Tank":1,"HardCC":0,"LineStrength":0},
    "Olaf": {"Engage":2,"Poke":1,"Peel":0,"Burst":1,"Tank":1,"HardCC":0,"LineStrength":1}, # 奥拉夫Q到就砍
    "Kled": {"Engage":3,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":1}, # 克烈前期战神
    "Kennen": {"Engage":3,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":1}, # 凯南长手
    "Gangplank": {"Engage":1,"Poke":2,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0},
    "Teemo": {"Engage":0,"Poke":3,"Peel":0,"Burst":1,"Tank":0,"HardCC":0,"LineStrength":1}, # 提莫折磨长手
    "Cassiopeia": {"Engage":1,"Poke":1,"Peel":2,"Burst":1,"Tank":0,"HardCC":2,"LineStrength":1},

    # ================= 打野 (Jungle) =================
    # 打野由于没有固定的对线期，通常在野区算对拼或刷野速度，按对拼优势（野区霸主）给 1，抗压或发育给 0
    "Lee Sin": {"Engage":2,"Poke":0,"Peel":1,"Burst":2,"Tank":1,"HardCC":2,"LineStrength":0},
    "Vi": {"Engage":3,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":2,"LineStrength":0},
    "Maokai": {"Engage":3,"Poke":1,"Peel":2,"Burst":0,"Tank":3,"HardCC":3,"LineStrength":0},
    "Amumu": {"Engage":3,"Poke":0,"Peel":1,"Burst":1,"Tank":3,"HardCC":3,"LineStrength":0},
    "Elise": {"Engage":2,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":1}, # 蜘蛛入侵/越塔极强
    "Nidalee": {"Engage":0,"Poke":3,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":1}, # 豹女极强侵略性
    "Graves": {"Engage":0,"Poke":1,"Peel":0,"Burst":2,"Tank":1,"HardCC":0,"LineStrength":1}, # 男枪刷野侵略
    "Kindred": {"Engage":0,"Poke":1,"Peel":2,"Burst":1,"Tank":0,"HardCC":0,"LineStrength":1}, # 千厥高伤害入侵
    "Rek'Sai": {"Engage":2,"Poke":0,"Peel":1,"Burst":2,"Tank":2,"HardCC":2,"LineStrength":0},
    "Xin Zhao": {"Engage":2,"Poke":0,"Peel":1,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":1}, # 赵信前期野区战神
    "Jarvan IV": {"Engage":3,"Poke":0,"Peel":0,"Burst":2,"Tank":2,"HardCC":2,"LineStrength":0},
    "Hecarim": {"Engage":3,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":0},
    "Nocturne": {"Engage":3,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":0},
    "Viego": {"Engage":1,"Poke":0,"Peel":0,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Ekko": {"Engage":2,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0},
    "Taliyah": {"Engage":1,"Poke":2,"Peel":2,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Nunu & Willump": {"Engage":3,"Poke":0,"Peel":1,"Burst":1,"Tank":3,"HardCC":2,"LineStrength":0},
    "Warwick": {"Engage":2,"Poke":0,"Peel":0,"Burst":1,"Tank":2,"HardCC":2,"LineStrength":1}, # 狼人单挑强
    "Skarner": {"Engage":3,"Poke":1,"Peel":1,"Burst":1,"Tank":3,"HardCC":3,"LineStrength":0},
    "Diana": {"Engage":3,"Poke":1,"Peel":0,"Burst":3,"Tank":1,"HardCC":1,"LineStrength":0},
    "Karthus": {"Engage":0,"Poke":2,"Peel":0,"Burst":2,"Tank":0,"HardCC":0,"LineStrength":0},
    "Shaco": {"Engage":1,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":1},
    "Rammus": {"Engage":3,"Poke":0,"Peel":1,"Burst":1,"Tank":3,"HardCC":2,"LineStrength":0},
    "Ivern": {"Engage":1,"Poke":1,"Peel":3,"Burst":0,"Tank":1,"HardCC":2,"LineStrength":0},
    "Master Yi": {"Engage":0,"Poke":0,"Peel":0,"Burst":2,"Tank":0,"HardCC":0,"LineStrength":0},
    "Shyvana": {"Engage":2,"Poke":2,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":0},
    "Kha'Zix": {"Engage":1,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0},
    "Kayn": {"Engage":2,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0},
    "Bel'Veth": {"Engage":1,"Poke":0,"Peel":0,"Burst":1,"Tank":1,"HardCC":1,"LineStrength":1}, # 卑尔维斯前期单对单强
    "Lillia": {"Engage":2,"Poke":2,"Peel":0,"Burst":2,"Tank":1,"HardCC":2,"LineStrength":0},
    "Fiddlesticks": {"Engage":3,"Poke":1,"Peel":1,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":0},
    "Evelynn": {"Engage":1,"Poke":0,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0},
    "Rengar": {"Engage":2,"Poke":0,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":1}, # 狮子狗草丛
    "Wukong": {"Engage":3,"Poke":0,"Peel":1,"Burst":2,"Tank":1,"HardCC":2,"LineStrength":0},
    "Sejuani": {"Engage":3,"Poke":0,"Peel":2,"Burst":1,"Tank":3,"HardCC":3,"LineStrength":0},

    # ================= 中单 (Mid) =================
    "Ahri": {"Engage":2,"Poke":1,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Orianna": {"Engage":2,"Poke":2,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 发条传统线霸
    "Syndra": {"Engage":1,"Poke":2,"Peel":2,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":1}, # 辛德拉推线压制
    "Azir": {"Engage":2,"Poke":2,"Peel":2,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 沙皇手长折磨
    "LeBlanc": {"Engage":1,"Poke":2,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0},
    "Zoe": {"Engage":1,"Poke":3,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":1}, # 佐伊消耗能力强
    "Viktor": {"Engage":0,"Poke":2,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 维克托 E 技能推线
    "Tristana": {"Engage":1,"Poke":0,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 绝活中单小炮推线吃塔皮
    "Lucian": {"Engage":1,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":1}, # 中单卢锡安不解释
    "Galio": {"Engage":2,"Poke":1,"Peel":3,"Burst":2,"Tank":2,"HardCC":3,"LineStrength":0},
    "Annie": {"Engage":2,"Poke":1,"Peel":1,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":0},
    "Neeko": {"Engage":3,"Poke":1,"Peel":1,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":1}, # 妮蔻线上推线折磨
    "Akali": {"Engage":1,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0},
    "Sylas": {"Engage":2,"Poke":0,"Peel":0,"Burst":2,"Tank":1,"HardCC":1,"LineStrength":0},
    "Ryze": {"Engage":1,"Poke":1,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Twisted Fate": {"Engage":2,"Poke":2,"Peel":1,"Burst":1,"Tank":0,"HardCC":2,"LineStrength":0},
    "Aurelion Sol": {"Engage":1,"Poke":2,"Peel":1,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":0},
    "Vex": {"Engage":3,"Poke":1,"Peel":1,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":0},
    "Hwei": {"Engage":1,"Poke":3,"Peel":2,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":1}, # 彗技能形态极其夸张
    "Zed": {"Engage":0,"Poke":2,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0},
    "Talon": {"Engage":1,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0},
    "Katarina": {"Engage":1,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0},
    "Veigar": {"Engage":0,"Poke":2,"Peel":2,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0},
    "Malzahar": {"Engage":1,"Poke":2,"Peel":2,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":0},
    "Xerath": {"Engage":0,"Poke":3,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 泽拉斯无伤消耗
    "Vel'Koz": {"Engage":0,"Poke":3,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1},
    "Swain": {"Engage":2,"Poke":1,"Peel":1,"Burst":1,"Tank":2,"HardCC":1,"LineStrength":0},
    "Mel": {"Engage":0,"Poke":2,"Peel":2,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Lissandra": {"Engage":3,"Poke":1,"Peel":2,"Burst":2,"Tank":1,"HardCC":3,"LineStrength":0},
    "Akshan": {"Engage":1,"Poke":1,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":1}, # 阿克尚前期极强
    "Qiyana": {"Engage":2,"Poke":0,"Peel":0,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":0},
    "Aurora": {"Engage":2,"Poke":2,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Pantheon": {"Engage":2,"Poke":1,"Peel":1,"Burst":2,"Tank":1,"HardCC":2,"LineStrength":1},
    "Corki": {"Engage":1,"Poke":3,"Peel":0,"Burst":2,"Tank":0,"HardCC":0,"LineStrength":1}, # 库奇改动后极度依赖线上压制

    # ================= 下路 (ADC) =================
    "Jinx": {"Engage":0,"Poke":1,"Peel":1,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":0},
    "Kai'Sa": {"Engage":1,"Poke":2,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0}, 
    "Ezreal": {"Engage":0,"Poke":3,"Peel":0,"Burst":2,"Tank":0,"HardCC":0,"LineStrength":0},
    "Xayah": {"Engage":0,"Poke":1,"Peel":2,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":0}, 
    "Ashe": {"Engage":3,"Poke":2,"Peel":2,"Burst":1,"Tank":0,"HardCC":2,"LineStrength":1}, # 艾希手长带减速减甲
    "Senna": {"Engage":0,"Poke":3,"Peel":1,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":1}, # 赛娜手长成长消耗
    "Aphelios": {"Engage":1,"Poke":1,"Peel":1,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0},
    "Miss Fortune": {"Engage":0,"Poke":2,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":1}, # 女枪一箭双雕和推线
    "Draven": {"Engage":0,"Poke":1,"Peel":1,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":1}, # 德莱文纯攻击压制
    "Samira": {"Engage":1,"Poke":0,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0},
    "Zeri": {"Engage":1,"Poke":1,"Peel":0,"Burst":2,"Tank":0,"HardCC":0,"LineStrength":0},
    "Kalista": {"Engage":2,"Poke":1,"Peel":2,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":1}, # 滑板鞋经典线霸
    "Caitlyn": {"Engage":0,"Poke":3,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 女警标杆
    "Twitch": {"Engage":0,"Poke":0,"Peel":0,"Burst":2,"Tank":0,"HardCC":0,"LineStrength":0},
    "Kog'Maw": {"Engage":0,"Poke":3,"Peel":0,"Burst":1,"Tank":0,"HardCC":0,"LineStrength":0},
    "Sivir": {"Engage":2,"Poke":2,"Peel":1,"Burst":1,"Tank":0,"HardCC":0,"LineStrength":1}, # 轮子妈推线压制
    "Nilah": {"Engage":2,"Poke":0,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0},
    "Jhin": {"Engage":1,"Poke":2,"Peel":1,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":1}, # 烬长手配合
    "Varus": {"Engage":2,"Poke":3,"Peel":1,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":1}, # 韦鲁斯标杆
    "Smolder": {"Engage":0,"Poke":3,"Peel":0,"Burst":1,"Tank":0,"HardCC":0,"LineStrength":0},
    "Ziggs": {"Engage":0,"Poke":3,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 吉格斯下路推线怪
    "Vayne": {"Engage":0,"Poke":0,"Peel":1,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":0},

    # ================= 辅助 (Support) =================
    "Nautilus": {"Engage":3,"Poke":0,"Peel":2,"Burst":1,"Tank":2,"HardCC":3,"LineStrength":0},
    "Thresh": {"Engage":2,"Poke":0,"Peel":3,"Burst":1,"Tank":2,"HardCC":2,"LineStrength":0},
    "Leona": {"Engage":3,"Poke":0,"Peel":2,"Burst":1,"Tank":3,"HardCC":3,"LineStrength":0},
    "Alistar": {"Engage":3,"Poke":0,"Peel":2,"Burst":1,"Tank":3,"HardCC":3,"LineStrength":0},
    "Rakan": {"Engage":3,"Poke":1,"Peel":2,"Burst":1,"Tank":1,"HardCC":2,"LineStrength":0},
    "Lulu": {"Engage":0,"Poke":2,"Peel":3,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":0},
    "Janna": {"Engage":0,"Poke":2,"Peel":3,"Burst":0,"Tank":0,"HardCC":2,"LineStrength":0},
    "Nami": {"Engage":1,"Poke":2,"Peel":2,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":1}, # 娜美强线上消耗
    "Morgana": {"Engage":1,"Poke":2,"Peel":3,"Burst":1,"Tank":0,"HardCC":2,"LineStrength":0}, 
    "Braum": {"Engage":2,"Poke":1,"Peel":3,"Burst":0,"Tank":3,"HardCC":2,"LineStrength":0},
    "Karma": {"Engage":1,"Poke":3,"Peel":2,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":1}, # 卡尔玛长手
    "Renata Glasc": {"Engage":2,"Poke":1,"Peel":3,"Burst":0,"Tank":1,"HardCC":2,"LineStrength":0}, 
    "Blitzcrank": {"Engage":3,"Poke":0,"Peel":1,"Burst":2,"Tank":2,"HardCC":2,"LineStrength":0},
    "Pyke": {"Engage":2,"Poke":1,"Peel":1,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":0},
    "Seraphine": {"Engage":1,"Poke":2,"Peel":2,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":1},
    "Sona": {"Engage":1,"Poke":2,"Peel":2,"Burst":1,"Tank":0,"HardCC":1,"LineStrength":0},
    "Yuumi": {"Engage":0,"Poke":2,"Peel":3,"Burst":0,"Tank":0,"HardCC":0,"LineStrength":0},
    "Milio": {"Engage":0,"Poke":1,"Peel":3,"Burst":0,"Tank":0,"HardCC":1,"LineStrength":0},
    "Bard": {"Engage":2,"Poke":2,"Peel":2,"Burst":1,"Tank":1,"HardCC":2,"LineStrength":0},
    "Taric": {"Engage":1,"Poke":0,"Peel":3,"Burst":0,"Tank":2,"HardCC":2,"LineStrength":0},
    "Zilean": {"Engage":1,"Poke":2,"Peel":3,"Burst":1,"Tank":0,"HardCC":2,"LineStrength":0},
    "Rell": {"Engage":3,"Poke":0,"Peel":2,"Burst":1,"Tank":3,"HardCC":3,"LineStrength":0},
    "Heimerdinger": {"Engage":0,"Poke":3,"Peel":2,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":1}, # 大头推线
    "Zyra": {"Engage":1,"Poke":3,"Peel":2,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":1}, # 婕拉强压制
    "Brand": {"Engage":0,"Poke":3,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":1}, # 火男高输出

    # ================= 补充遗漏英雄 =================
    "Anivia": {"Engage":1,"Poke":2,"Peel":2,"Burst":2,"Tank":0,"HardCC":2,"LineStrength":1}, # 冰鸟推线墙
    "Cho'Gath": {"Engage":1,"Poke":1,"Peel":1,"Burst":1,"Tank":3,"HardCC":2,"LineStrength":0}, # 大虫子叠肉
    "Fizz": {"Engage":1,"Poke":0,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0}, # 小鱼人刺杀
    "Kassadin": {"Engage":1,"Poke":0,"Peel":0,"Burst":3,"Tank":0,"HardCC":0,"LineStrength":0}, # 卡萨丁后期法王
    "Kayle": {"Engage":0,"Poke":1,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0}, # 天使后期carry
    "Lux": {"Engage":1,"Poke":3,"Peel":2,"Burst":3,"Tank":0,"HardCC":2,"LineStrength":1}, # 拉克丝长手消耗
    "Naafiri": {"Engage":2,"Poke":0,"Peel":0,"Burst":3,"Tank":0,"HardCC":1,"LineStrength":0}, # 纳亚菲利刺杀
    "Singed": {"Engage":1,"Poke":0,"Peel":0,"Burst":0,"Tank":2,"HardCC":1,"LineStrength":0}, # 炼金跑图
    "Soraka": {"Engage":0,"Poke":1,"Peel":3,"Burst":0,"Tank":0,"HardCC":1,"LineStrength":0}, # 索拉卡纯奶
    "Urgot": {"Engage":2,"Poke":1,"Peel":0,"Burst":2,"Tank":2,"HardCC":1,"LineStrength":0}, # 厄加特近战斩杀
    "Yorick": {"Engage":0,"Poke":1,"Peel":0,"Burst":1,"Tank":1,"HardCC":1,"LineStrength":0}, # 约里克分推
    "Yunara": {"Engage":1,"Poke":1,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0}, # 新英雄默认
    "Zaahen": {"Engage":1,"Poke":1,"Peel":1,"Burst":2,"Tank":0,"HardCC":1,"LineStrength":0}, # 新英雄默认
}

#==================================================

def load_matches(league=None):
    if league is None:
        path = os.path.join(CLEANED_DIR, "matches_cleaned.csv")
    else:
        path = os.path.join(CLEANED_DIR, league, "matches_cleaned.csv")
    log.info(f"[数据加载] 加载比赛数据: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"[数据加载] matches: {len(df)} 行, {len(df.columns)} 列")
    leagues = df["league"].value_counts().to_dict() if "league" in df.columns else {}
    log.info(f"[数据加载] 联赛分布: {leagues}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    date_na = df["date"].isna().sum()
    if date_na > 0:
        log.warning(f"[数据加载] date列有 {date_na} 个空值")
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    df["match_seq_idx"] = df.index
    log.info(f"[数据加载] matches加载完成, 日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df

def load_career_stats(league=None):
    if league is None:
        path = os.path.join(CLEANED_DIR, "player_career_hero_stats_cleaned.csv")
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
    missing = [c for c in ALL_RESULT_COLS if c not in df.columns]
    if missing:
        log.warning(f"[enforce_pit] matches 缺少结果列，将用 NaN 填充: {missing}")
    result_df = df[["gameid"]].copy()
    for col in ALL_RESULT_COLS:
        result_df[col] = df[col] if col in df.columns else np.nan
    return target_df, result_df

def melt_matches_to_player_rows(matches_df):
    records = []
    for side in ["blue", "red"]:
        result_val = matches_df["result"].values if side == "blue" else (1 - matches_df["result"].values)
        team_col = f"{side}_team"
        for pos_short in POSITIONS_SHORT:
            cols_needed = [
                "gameid", "league", "year", "split", "date", "patch",
                team_col,
                f"{side}_{pos_short}_player_id",
                f"{side}_{pos_short}_champion",
                f"{side}_{pos_short}_kills",
                f"{side}_{pos_short}_deaths",
                f"{side}_{pos_short}_assists",
            ]
            sub = matches_df[cols_needed].copy()
            sub.columns = [
                "gameid", "league", "year", "split", "date", "patch",
                "team", "player_id", "champion", "kills", "deaths", "assists",
            ]
            sub["side"] = side
            sub["position"] = POS_SHORT2FULL[pos_short]
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
                sub = matches_df[["gameid", "date", "patch", col]].copy()
                sub.columns = ["gameid", "date", "patch", "champion"]
                sub["ban_side"] = side
                sub = sub[sub["champion"].notna() & (sub["champion"] != "") & (sub["champion"] != "Unknown")]
                ban_records.append(sub)
    if not ban_records:
        return pd.DataFrame(columns=["gameid", "date", "patch", "champion", "ban_side"])
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


def compute_meta_features_pit(player_history, ban_history, matches_df):
    log.info("  [Meta] Computing Patch-Aware Meta Drift (Patch-Relative Performance)...")
    
    # 基础聚合
    pick_agg = player_history.groupby(["date", "patch", "champion"]).agg(
        picks=("gameid", "count"),
        wins=("result", "sum"),
    ).reset_index()

    # 包含 Patch 信息的数据源
    all_data = pick_agg.sort_values("date")
    unique_dates = np.sort(matches_df["date"].dropna().unique())
    decay_lambda = np.log(2) / META_DECAY_HALF_LIFE
    window_days = META_WINDOW_DAYS

    results = []
    all_champions = all_data["champion"].unique()

    for champion in all_champions:
        champ_df = all_data[all_data["champion"] == champion].sort_values("date")
        
        for target_date in unique_dates:
            # 确定当前补丁
            curr_patch = matches_df[matches_df["date"] == target_date]["patch"].iloc[0]
            
            # 1. 历史 60 天基准窗口
            cutoff = target_date - np.timedelta64(window_days, "D")
            hist_mask = (champ_df["date"] >= cutoff) & (champ_df["date"] < target_date)
            hist_data = champ_df[hist_mask]
            
            # 2. 当前补丁窗口 (Patch-Specific)
            patch_mask = (champ_df["patch"] == curr_patch) & (champ_df["date"] < target_date)
            patch_data = champ_df[patch_mask]
            
            # 计算基准胜率
            hist_wr = 0.5
            if not hist_data.empty:
                hist_wr = (hist_data["wins"].sum() + 1) / (hist_data["picks"].sum() + 2)
            
            # 计算补丁特异性胜率 (如果当前补丁样本不足，回退到历史均值)
            patch_wr = hist_wr
            if not patch_data.empty and patch_data["picks"].sum() > 5:
                patch_wr = (patch_data["wins"].sum() + 1) / (patch_data["picks"].sum() + 2)
            
            # 🚀 核心指标：版本漂移指数 (Drift Index)
            # 大于 1 表示该英雄在当前版本强度显著高于过去 60 天平均水平
            patch_drift_index = patch_wr / (hist_wr + 1e-6)
            
            # 补丁登场率漂移
            hist_pick_rate = hist_data["picks"].sum() / (window_days * 0.5) # 简单归一化
            patch_pick_rate = patch_data["picks"].sum() / 7.0 # 假设 patch 活跃期
            pick_drift_index = (patch_pick_rate + 0.1) / (hist_pick_rate + 0.1)

            results.append({
                "champion": champion,
                "date": target_date,
                "meta_win_rate_pit": hist_wr,
                "meta_patch_drift_index": patch_drift_index,
                "meta_pick_drift_index": pick_drift_index,
            })

    result_df = pd.DataFrame(results)
    log.info(f"  [Meta] Meta特征计算完成: {len(result_df)} 行, {len(result_df.columns)} 列, {result_df['champion'].nunique()} 个英雄")
    meta_na_rates = {}
    for c in result_df.select_dtypes(include=[np.number]).columns:
        na_pct = result_df[c].isna().mean() * 100
        if na_pct > 30:
            log.warning(f"  [Meta] {c} 缺失率 {na_pct:.1f}% > 30%")
        meta_na_rates[c] = f"{na_pct:.1f}%"
    log.info(f"  [Meta] Meta特征列缺失率: {meta_na_rates}")
    return result_df

def compute_player_features_pit(player_history, base_prior_df):
    log.info("  [Player] Computing DENSE Player Feature Store (All Champions in Pool)...")
    base_prior_dict = {}
    if not base_prior_df.empty:
        for _, row in base_prior_df.iterrows():
            base_prior_dict[(row["player_id"], row["champion"])] = (row["Base_G"], row["Base_W"])

    ph = player_history.sort_values(["date"]).copy()
    hist_dict = {}
    for (pid, c), g in ph.groupby(["player_id", "champion"]):
        hist_dict[(pid, c)] = {
            "dates": g["date"].values.astype("datetime64[ns]"),
            "results": g["result"].values.astype(float),
            "kills": g["kills"].values.astype(float),
            "deaths": g["deaths"].values.astype(float),
            "assists": g["assists"].values.astype(float),
        }

    player_matches = {}
    for pid, g in ph.groupby("player_id"):
        matches = g[["gameid", "date"]].drop_duplicates().sort_values("date")
        player_matches[pid] = matches.to_dict("records")

    player_pools = {}
    for pid in player_matches.keys():
        played = ph[ph["player_id"] == pid]["champion"].unique()
        priors = base_prior_df[base_prior_df["player_id"] == pid]["champion"].unique() if not base_prior_df.empty else []
        player_pools[pid] = set(played) | set(priors)

    mastery_decay_lambda = np.log(2) / MASTERY_DECAY_HALF_LIFE
    recent_decay_lambda = np.log(2) / PLAYER_DECAY_HALF_LIFE
    window_days = PLAYER_WINDOW_DAYS

    all_player_hist = {}
    for pid, g in ph.groupby("player_id"):
        g_sorted = g.sort_values("date").drop_duplicates(subset=["gameid"])
        all_player_hist[pid] = {
            "dates": g_sorted["date"].values.astype("datetime64[ns]"),
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
        ap_res, ap_k, ap_d, ap_a = ap["results"], ap["kills"], ap["deaths"], ap["assists"]
        
        for champ in pool:
            h = hist_dict.get((pid, champ), None)
            if h is not None:
                h_dates = np.asarray(h["dates"], dtype="datetime64[ns]")
                h_res, h_k, h_d, h_a = h["results"], h["kills"], h["deaths"], h["assists"]
            else:
                h_dates = np.array([], dtype='datetime64[ns]')
                h_res = h_k = h_d = h_a = np.array([])

            base_g, base_w = base_prior_dict.get((pid, champ), (0.0, 0.0))
            base_wr = base_w / base_g if base_g > 0 else 0.5

            for match in matches:
                target_date = np.datetime64(pd.Timestamp(match["date"]))
                gameid = match["gameid"]

                ap_mask = ap_dates < target_date
                ap_cutoff = target_date - np.timedelta64(window_days, "D")
                ap_mask_trunc = ap_mask & (ap_dates >= ap_cutoff)
                if not ap_mask_trunc.any():
                    overall_wr, overall_kda, overall_games = np.nan, np.nan, 0
                else:
                    ap_delta = (target_date - ap_dates[ap_mask_trunc]) / np.timedelta64(1, "D")
                    ap_w = np.exp(-recent_decay_lambda * ap_delta)
                    ap_w_sum = ap_w.sum()
                    overall_wr = float(np.sum(ap_w * ap_res[ap_mask_trunc]) / ap_w_sum)
                    ap_wd = np.sum(ap_w * ap_d[ap_mask_trunc])
                    overall_kda = float(
                        (np.sum(ap_w * ap_k[ap_mask_trunc]) + np.sum(ap_w * ap_a[ap_mask_trunc])) / ap_wd
                        if ap_wd > 0 else
                        (np.sum(ap_w * ap_k[ap_mask_trunc]) + np.sum(ap_w * ap_a[ap_mask_trunc]))
                    )
                    overall_games = int(ap_mask_trunc.sum())

                mask_full = h_dates < target_date
                
                if not mask_full.any():
                    mastery_g = base_g
                    m_wr_score = base_wr * 50
                    m_exp_score = np.minimum(np.log1p(mastery_g) / np.log1p(50), 1.0) * 30
                    m_kda_score = np.minimum(GLOBAL_KDA_PRIOR, 8.0) / 8.0 * 20
                    mastery_score = m_wr_score + m_exp_score + m_kda_score
                    recent_kda, recent_wr = np.nan, np.nan
                    recent_games = 0
                else:
                    delta_days_full = (target_date - h_dates[mask_full]) / np.timedelta64(1, "D")
                    weights_full = np.exp(-mastery_decay_lambda * delta_days_full)
                    w_sum_full = weights_full.sum()

                    w_wins_full = np.sum(weights_full * h_res[mask_full])
                    mastery_decay_wr = (w_wins_full + base_wr * 3) / (w_sum_full + 3)

                    w_k_full = np.sum(weights_full * h_k[mask_full])
                    w_d_full = np.sum(weights_full * h_d[mask_full])
                    w_a_full = np.sum(weights_full * h_a[mask_full])
                    mastery_decay_kda = (w_k_full + w_a_full) / w_d_full if w_d_full > 0 else (w_k_full + w_a_full)

                    mastery_g = base_g + mask_full.sum()
                    
                    m_wr_score = mastery_decay_wr * 50
                    m_exp_score = np.minimum(np.log1p(mastery_g) / np.log1p(50), 1.0) * 30
                    m_kda_score = np.minimum(mastery_decay_kda, 8.0) / 8.0 * 20
                    mastery_score = m_wr_score + m_exp_score + m_kda_score

                    cutoff = target_date - np.timedelta64(window_days, "D")
                    mask_trunc = mask_full & (h_dates >= cutoff)

                    if not mask_trunc.any():
                        recent_kda, recent_wr = np.nan, np.nan
                        recent_games = 0
                    else:
                        delta_days_trunc = (target_date - h_dates[mask_trunc]) / np.timedelta64(1, "D")
                        weights_trunc = np.exp(-recent_decay_lambda * delta_days_trunc)
                        w_sum_trunc = weights_trunc.sum()

                        w_wins_trunc = np.sum(weights_trunc * h_res[mask_trunc])
                        w_k_trunc = np.sum(weights_trunc * h_k[mask_trunc])
                        w_d_trunc = np.sum(weights_trunc * h_d[mask_trunc])
                        w_a_trunc = np.sum(weights_trunc * h_a[mask_trunc])

                        recent_wr = w_wins_trunc / w_sum_trunc
                        recent_kda = (w_k_trunc + w_a_trunc) / w_d_trunc if w_d_trunc > 0 else (w_k_trunc + w_a_trunc)
                        recent_games = int(mask_trunc.sum())

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

def compute_team_profile_pit(matches_df):
    log.info("  [TeamProfile] Computing dynamic team style features (truncated decay, PIT)...")
    team_game_cols = [
        "gameid", "date", "blue_team", "red_team", "result",
        "gamelength", "ckpm",
        "blue_golddiffat15", "red_golddiffat15",
        "blue_firstdragon", "red_firstdragon",
        "blue_firsttower", "red_firsttower",
    ]
    # 添加击杀/助攻列用于血腥度计算
    for side in ["blue", "red"]:
        for pos in POSITIONS_SHORT:
            for stat in ["kills", "deaths", "assists"]:
                col = f"{side}_{pos}_{stat}"
                if col in matches_df.columns:
                    team_game_cols.append(col)
    available_cols = [c for c in team_game_cols if c in matches_df.columns]
    m = matches_df[available_cols].copy()
    m = m.sort_values("date").reset_index(drop=True)

    # 蓝方记录
    blue_base_cols = ["gameid", "date", "blue_team", "result",
                       "gamelength", "ckpm",
                       "blue_golddiffat15", "blue_firstdragon", "blue_firsttower"]
    for pos in POSITIONS_SHORT:
        for stat in ["kills", "deaths", "assists"]:
            col = f"blue_{pos}_{stat}"
            if col in m.columns:
                blue_base_cols.append(col)
    blue_base_cols = [c for c in blue_base_cols if c in m.columns]

    blue_records = m[blue_base_cols].copy()
    blue_records.columns = [c.replace("blue_", "") if c.startswith("blue_") else c for c in blue_base_cols]
    blue_records["side"] = "blue"

    # 红方记录
    red_base_cols = ["gameid", "date", "red_team", "result",
                      "gamelength", "ckpm",
                      "red_golddiffat15", "red_firstdragon", "red_firsttower"]
    for pos in POSITIONS_SHORT:
        for stat in ["kills", "deaths", "assists"]:
            col = f"red_{pos}_{stat}"
            if col in m.columns:
                red_base_cols.append(col)
    red_base_cols = [c for c in red_base_cols if c in m.columns]

    red_records = m[red_base_cols].copy()
    red_records.columns = [c.replace("red_", "") if c.startswith("red_") else c for c in red_base_cols]
    red_records["result"] = 1 - red_records["result"]
    red_records["side"] = "red"

    team_history = pd.concat([blue_records, red_records], ignore_index=True)
    team_history = team_history.sort_values(["team", "date"]).reset_index(drop=True)

    # 计算每场比赛的总击杀/死亡/助攻
    kill_cols = [f"{pos}_kills" for pos in POSITIONS_SHORT if f"{pos}_kills" in team_history.columns]
    death_cols = [f"{pos}_deaths" for pos in POSITIONS_SHORT if f"{pos}_deaths" in team_history.columns]
    assist_cols = [f"{pos}_assists" for pos in POSITIONS_SHORT if f"{pos}_assists" in team_history.columns]

    if kill_cols:
        team_history["total_kills"] = team_history[kill_cols].sum(axis=1)
    else:
        team_history["total_kills"] = 0
    if death_cols:
        team_history["total_deaths"] = team_history[death_cols].sum(axis=1)
    else:
        team_history["total_deaths"] = 0
    if assist_cols:
        team_history["total_assists"] = team_history[assist_cols].sum(axis=1)
    else:
        team_history["total_assists"] = 0

    # 血腥度 = (kills + assists) / gamelength_minutes
    team_history["gamelength_min"] = pd.to_numeric(team_history["gamelength"], errors="coerce") / 60.0
    team_history["bloodiness"] = np.where(
        team_history["gamelength_min"] > 0,
        (team_history["total_kills"] + team_history["total_assists"]) / team_history["gamelength_min"],
        0
    )

    # 滚雪球标记：15分钟经济领先 > 500 且最终获胜
    team_history["golddiffat15"] = pd.to_numeric(team_history.get("golddiffat15", 0), errors="coerce")
    team_history["led_at_15"] = (team_history["golddiffat15"] > 500).astype(int)
    team_history["snowball_win"] = ((team_history["golddiffat15"] > 500) & (team_history["result"] == 1)).astype(int)

    numeric_cols = ["gamelength", "ckpm", "golddiffat15", "firstdragon", "firsttower"]
    for c in numeric_cols:
        team_history[c] = pd.to_numeric(team_history[c], errors="coerce")

    # 【重要修复】LPL 原始数据中 golddiffat15 100% 缺失（数据源不提供），
    # 旧逻辑用 0 填充导致模型误判为"经济平局"。
    # 正确做法：将 0 值（来自缺失填充）转为 NaN，下游 wavg 自动跳过 NaN。
    # 当 golddiffat15 缺失时，firsttower (一塔) 作为领先指标替代。
    _log = logging.getLogger(__name__)
    zero_count = (team_history["golddiffat15"] == 0).sum()
    total_count = len(team_history)
    if zero_count > 0:
        _log.info(f"[TeamProfile] golddiffat15 中有 {zero_count}/{total_count} 行为 0 (可能来自缺失填充)")
        team_history.loc[team_history["golddiffat15"] == 0, "golddiffat15"] = np.nan
        _log.info(f"[TeamProfile] 已将 golddiffat15 的 0 值转为 NaN, 下游用 firsttower 替代")
    else:
        _log.info(f"[TeamProfile] golddiffat15 无 0 值, 数据正常")

    decay_lambda = np.log(2) / TEAM_PROFILE_DECAY_HALF_LIFE
    window_days = TEAM_PROFILE_WINDOW_DAYS

    results = []
    for team, group in team_history.groupby("team"):
        group = group.reset_index(drop=True)
        dates = group["date"].values
        sides = group["side"].values
        results_arr = group["result"].values
        n = len(group)

        for i in range(n):
            target_date = dates[i]
            current_side = sides[i]
            cutoff = target_date - np.timedelta64(window_days, "D")
            mask = (dates[:i] >= cutoff) & (dates[:i] < target_date)
            window = group.iloc[:i][mask]

            if len(window) == 0:
                results.append({
                    "gameid": group.iloc[i]["gameid"],
                    "team": team,
                    "date": dates[i],
                    "side": current_side,
                    "team_avg_gamelength": np.nan,
                    "team_avg_ckpm": np.nan,
                    "team_avg_golddiffat15": np.nan,
                    "team_firstdragon_rate": np.nan,
                    "team_firsttower_rate": np.nan,
                    "team_profile_games": 0,
                    "team_recent_wr": np.nan,
                    "team_recent_wr_5": np.nan,
                    "team_recent_wr_10": np.nan,
                    "team_side_wr": np.nan,
                    "team_streak": 0,
                    "team_avg_kills": np.nan,
                    "team_avg_deaths": np.nan,
                    "team_avg_assists": np.nan,
                    "team_bloodiness": np.nan,
                    "team_snowball_rate": np.nan,
                    "team_led_at_15_rate": np.nan,
                })
            else:
                delta_days = (target_date - window["date"].values) / np.timedelta64(1, "D")
                weights = np.exp(-decay_lambda * delta_days)

                def wavg(col, w=weights, win=window):
                    valid = win[col].notna()
                    if valid.sum() == 0:
                        return np.nan
                    return np.average(win.loc[valid, col], weights=w[valid.values])

                w_sum = weights.sum()
                decay_wr = np.sum(weights * window["result"].values) / w_sum if w_sum > 0 else np.nan

                last_n = min(5, len(window))
                wr_5 = window["result"].iloc[-last_n:].mean()

                last_n10 = min(10, len(window))
                wr_10 = window["result"].iloc[-last_n10:].mean()

                side_mask = window["side"].values == current_side
                side_games = side_mask.sum()
                side_wr = window["result"].values[side_mask].mean() if side_games >= 2 else np.nan

                streak = 0
                last_result = results_arr[i - 1] if i > 0 else None
                if last_result is not None:
                    streak_val = 1 if last_result == 1 else -1
                    streak = streak_val
                    for j in range(i - 2, -1, -1):
                        if results_arr[j] == last_result:
                            streak += streak_val
                        else:
                            break

                # 血腥度特征
                avg_kills = wavg("total_kills")
                avg_deaths = wavg("total_deaths")
                avg_assists = wavg("total_assists")
                avg_bloodiness = wavg("bloodiness")

                # 滚雪球率：15分钟领先时的胜率
                led_games = window[window["led_at_15"] == 1]
                if len(led_games) >= 2:
                    snowball_rate = led_games["result"].mean()
                else:
                    snowball_rate = np.nan
                led_at_15_rate = window["led_at_15"].mean()

                results.append({
                    "gameid": group.iloc[i]["gameid"],
                    "team": team,
                    "date": dates[i],
                    "side": current_side,
                    "team_avg_gamelength": wavg("gamelength"),
                    "team_avg_ckpm": wavg("ckpm"),
                    "team_avg_golddiffat15": wavg("golddiffat15"),
                    "team_firstdragon_rate": wavg("firstdragon"),
                    "team_firsttower_rate": wavg("firsttower"),
                    "team_profile_games": len(window),
                    "team_recent_wr": decay_wr,
                    "team_recent_wr_5": wr_5,
                    "team_recent_wr_10": wr_10,
                    "team_side_wr": side_wr,
                    "team_streak": streak,
                    "team_avg_kills": avg_kills,
                    "team_avg_deaths": avg_deaths,
                    "team_avg_assists": avg_assists,
                    "team_bloodiness": avg_bloodiness,
                    "team_snowball_rate": snowball_rate,
                    "team_led_at_15_rate": led_at_15_rate,
                })

    result_df = pd.DataFrame(results)
    team_profile_defaults = {
        "team_avg_gamelength": 1954,
        "team_avg_ckpm": 0.7,
        "team_avg_golddiffat15": 0,
        "team_firstdragon_rate": 0.5,
        "team_firsttower_rate": 0.5,
        "team_recent_wr": 0.5,
        "team_recent_wr_5": 0.5,
        "team_recent_wr_10": 0.5,
        "team_side_wr": 0.5,
        "team_streak": 0,
        "team_avg_kills": 25.0,
        "team_avg_deaths": 25.0,
        "team_avg_assists": 60.0,
        "team_bloodiness": 1.5,
        "team_snowball_rate": 0.7,
        "team_led_at_15_rate": 0.5,
    }
    for c, default in team_profile_defaults.items():
        result_df[c] = result_df[c].fillna(default)

    unique_teams = result_df["team"].nunique() if "team" in result_df.columns else 0
    unique_games = result_df["gameid"].nunique() if "gameid" in result_df.columns else 0
    log.info(f"  [TeamProfile] TeamProfile特征计算完成: {len(result_df)} 行, {len(result_df.columns)} 列")
    log.info(f"  [TeamProfile] 覆盖: {unique_teams} 支战队, {unique_games} 场比赛 (每场2个视角)")
    team_na_rates = {}
    for c in result_df.select_dtypes(include=[np.number]).columns:
        na_pct = result_df[c].isna().mean() * 100
        if na_pct > 30:
            log.warning(f"  [TeamProfile] {c} 缺失率 {na_pct:.1f}% > 30%")
        team_na_rates[c] = f"{na_pct:.1f}%"
    log.info(f"  [TeamProfile] Team特征缺失率: {team_na_rates}")
    return result_df

def extract_bp_features(target_df):
    return target_df.copy()




def _compute_comp_features(champion_names):
    """基于 champion_tags 计算阵容发力期特征"""
    tags_list = []
    for name in champion_names:
        tags = CHAMPION_TAGS.get(name, {
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

    # 衍生特征
    agg["comp_early_power"] = (
        agg["comp_linestrength_sum"] * 1.0
        + agg["comp_burst_sum"] * 0.5
        + agg["comp_engage_sum"] * 0.3
    )
    agg["comp_late_power"] = (
        agg["comp_tank_sum"] * 1.0
        + agg["comp_peel_sum"] * 0.5
    )
    agg["comp_teamfight_score"] = (
        agg["comp_engage_sum"] * 1.0
        + agg["comp_hardcc_sum"] * 0.8
        + agg["comp_tank_sum"] * 0.3
    )
    # 打架意愿综合指数
    aggression_num = agg["comp_engage_sum"] + agg["comp_burst_sum"] + agg["comp_linestrength_sum"]
    aggression_den = agg["comp_peel_sum"] + agg["comp_poke_sum"] * 0.5 + 1
    agg["comp_aggression_index"] = aggression_num / aggression_den

    # 阵容类型分类
    agg["comp_scaling_type"] = (agg["comp_peel_sum"] + agg["comp_poke_sum"]) / (agg["comp_engage_sum"] + agg["comp_burst_sum"] + 1)
    agg["comp_lane_dom_type"] = (agg["comp_linestrength_sum"] + agg["comp_burst_sum"]) / 10.0
    agg["comp_teamfight_type"] = (agg["comp_engage_sum"] + agg["comp_hardcc_sum"]) / 10.0

    return agg


def _get_comp_feature_names():
    """返回阵容特征名称列表"""
    sample = _compute_comp_features(["Ahri", "Lee Sin", "Orianna", "Jinx", "Nautilus"])
    return list(sample.keys())


def assemble_features_for_prediction(target_df, team_profile_df, player_features_df, champ_meta_daily):
    log.info("  [Assemble V2] Assembling baseline features using unified feature_utils...")
    merged = target_df.copy()
    
    # 1. 组合 Team Profile 基础特征
    if team_profile_df is not None and not team_profile_df.empty:
        profile_cols = [c for c in team_profile_df.columns if c.startswith("team_")]
        # 对 team_profile_df 按 (gameid, team) 去重，避免 merge 时产生重复行导致长度不匹配
        team_profile_dedup = team_profile_df.drop_duplicates(subset=["gameid", "team"], keep="first")
        for side in ["blue", "red"]:
            team_col = f"{side}_team"
            if team_col in merged.columns:
                merge_base = merged[["gameid", team_col]].rename(columns={team_col: "team"})
                profile_merged = merge_base.merge(team_profile_dedup[["gameid", "team"] + profile_cols], on=["gameid", "team"], how="left")
                for c in profile_cols:
                    merged[f"{side}_{c}"] = profile_merged[c].fillna(TEAM_PROFILE_DEFAULTS.get(c, 0)).values
    
    # 2. 联赛和地图方
    REQUIRED_LEAGUES = ['LPL', 'LCK', 'LEC']
    if 'league' in merged.columns:
        league_dummies = pd.get_dummies(merged['league'], prefix='league', dummy_na=False).astype(int)
        for l in REQUIRED_LEAGUES:
            col_name = f"league_{l}"
            if col_name not in league_dummies.columns:
                league_dummies[col_name] = 0
        league_dummies = league_dummies[[f"league_{l}" for l in REQUIRED_LEAGUES]]
        merged = pd.concat([merged, league_dummies], axis=1)
    else:
        for l in REQUIRED_LEAGUES:
            merged[f"league_{l}"] = 0

    if 'first_pick_map_side' in merged.columns:
        # 训练数据语义: first_pick_map_side=1 表示先 Pick 方位于地图蓝色方，
        # 默认值取 1（与训练数据清洗阶段 fillna(1) 保持一致，2026 年前规则）
        merged['is_blue_map_side'] = merged['first_pick_map_side'].fillna(1).astype(int)
        merged = merged.drop(columns=["first_pick_map_side"])
    else:
        # 列不存在时回退到 is_blue_map_side，默认值与上面分支保持一致
        merged['is_blue_map_side'] = merged.get('is_blue_map_side', 1).fillna(1).astype(int)

    if 'playoffs' in merged.columns:
        merged['is_playoff'] = merged['playoffs'].fillna(0).astype(int)
        merged = merged.drop(columns=["playoffs"])
    else:
        merged['is_playoff'] = merged.get('is_playoff', 0).fillna(0).astype(int)

    # 3. 组合 Player & Meta 基础特征
    for side in ["blue", "red"]:
        for pos in POSITIONS_SHORT:
            pid_col = f"{side}_{pos}_player_id"
            champ_col = f"{side}_{pos}_champion"
            
            if pid_col in merged.columns and champ_col in merged.columns:
                p_sub = player_features_df[["gameid", "player_id", "champion"] + PLAYER_FEATURE_COLS].copy()
                p_sub.columns = ["gameid", pid_col, champ_col] + [f"{side}_{pos}_{c}" for c in PLAYER_FEATURE_COLS]
                merged = merged.merge(p_sub, on=["gameid", pid_col, champ_col], how="left")
                
                # 填充 Player 默认值
                for c in PLAYER_FEATURE_COLS:
                    merged[f"{side}_{pos}_{c}"] = merged[f"{side}_{pos}_{c}"].fillna(PLAYER_DEFAULTS[c])
                    
                m_sub = champ_meta_daily[["date", "champion"] + META_FEATURE_COLS].copy()
                m_sub.columns = ["date", champ_col] + [f"{side}_{pos}_{c}" for c in META_FEATURE_COLS]
                merged = merged.merge(m_sub, on=["date", champ_col], how="left")
                
                # 填充 Meta 默认值
                for c in META_FEATURE_COLS:
                    merged[f"{side}_{pos}_{c}"] = merged[f"{side}_{pos}_{c}"].fillna(META_DEFAULTS[c])

    log.info("  [Assemble V2] Computing derived features via shared feature_utils...")
    
    records = merged.to_dict(orient="records")
    for row in records:
        derived_feats = calculate_derived_features(row, CHAMPION_TAGS)
        row.update(derived_feats)
        
    final_merged = pd.DataFrame(records)
    
    dropped = [c for c in ["year", "patch", "split"] if c in final_merged.columns]
    if dropped:
        final_merged = final_merged.drop(columns=dropped)

    numeric_cols = final_merged.select_dtypes(include=[np.number]).columns
    inf_count = 0
    nan_count_before = 0
    for c in numeric_cols:
        inf_mask = np.isinf(final_merged[c])
        inf_count += inf_mask.sum()
        if inf_mask.any():
            final_merged.loc[inf_mask, c] = np.nan
        nan_count_before += final_merged[c].isna().sum()
    
    if inf_count > 0:
        log.warning(f"  [Assemble V2] 发现并处理 {inf_count} 个 Inf 值 (已转为NaN)")

    high_missing_cols = []
    for c in numeric_cols:
        na_pct = final_merged[c].isna().mean() * 100
        if na_pct > 30:
            high_missing_cols.append((c, f"{na_pct:.1f}%"))
    if high_missing_cols:
        log.warning(f"  [Assemble V2] 高缺失率列 (>30%): {high_missing_cols}")

    log.info(f"  [Assemble V2] 特征合并完成: {final_merged.shape[0]} 行, {final_merged.shape[1]} 列")
    team_feat_count = len([c for c in final_merged.columns if c.startswith("team_")])
    player_feat_count = len([c for c in final_merged.columns if any(f"_{p}_" in c for p in POSITIONS_SHORT) and not c.startswith("blue_") and not c.startswith("red_")]) + len([c for c in final_merged.columns if c.startswith("blue_") or c.startswith("red_")])
    meta_feat_count = len([c for c in final_merged.columns if "meta_" in c])
    comp_feat_count = len([c for c in final_merged.columns if c.startswith("comp_")])
    log.info(f"  [Assemble V2] 特征分类: Team特征={team_feat_count}列, Player特征={player_feat_count}列, Meta特征={meta_feat_count}列, 阵容Comp特征={comp_feat_count}列")
    return final_merged

def build_prediction_feature_pipeline(league=None, save_parquet=True):
    label = league if league else "ALL"
    log.info("\n%s", "="*60)
    log.info("[Pipeline] Building PIT feature pipeline for WIN/LOSS PREDICTION MODEL (%s)", label)
    log.info("%s\n", "="*60)

    log.info("[Step 0] Loading raw cleaned match records and prior metrics...")
    matches_df = load_matches(league)
    career_df = load_career_stats(league)

    log.info("[DataQuality] 原始数据空值检测:")
    for col in ['gameid', 'league', 'blue_team', 'red_team', 'result', 'date',
                'blue_golddiffat15', 'red_golddiffat15', 'blue_firsttower', 'red_firsttower']:
        if col in matches_df.columns:
            na = matches_df[col].isna().sum()
            total = len(matches_df)
            if na > 0:
                log.warning("  [WARNING] matches.%s: %d/%d 空值 (%.1f%%)", col, na, total, na/total*100)
            else:
                log.info("  [OK] matches.%s: 0 空值", col)
    if not career_df.empty:
        for col in ['player_id', 'champion', 'win_rate', 'games']:
            if col in career_df.columns:
                na = career_df[col].isna().sum()
                total = len(career_df)
                if na > 0:
                    log.error("  [ERROR] career.%s: %d/%d 空值 (%.1f%%) ← 严重问题!", col, na, total, na/total*100)
                else:
                    log.info("  [OK] career.%s: 0 空值", col)

    log.info("\n[Step 1] Enforcing PIT: stripping current game process/result metrics to block data leakage...")
    target_df, _ = enforce_pit(matches_df)

    log.info("\n[Step 2] Extracting structural BP information (Multi-hot & Step Sequences)...")
    target_df = extract_bp_features(target_df)

    log.info("\n[Step 3] Transforming records to timeline formats...")
    player_history = melt_matches_to_player_rows(matches_df).sort_values("date").reset_index(drop=True)
    log.info(f"  player_history (选手视角): {len(player_history)} 行, 唯一选手={player_history['player_id'].nunique()}, 唯一英雄={player_history['champion'].nunique()}")
    ban_history = melt_bans_from_matches(matches_df)
    log.info(f"  ban_history (Ban记录): {len(ban_history)} 行")

    log.info("\n[Step 4] Crafting historical priors and reverse career stats...")
    base_prior_df = build_base_prior(player_history, career_df)
    log.info(f"  base_prior (选手英雄先验): {len(base_prior_df)} 行")

    log.info("\n[Step 5] Building independent time-decay Feature Stores (严格 PIT 隔离)...")
    champ_meta_daily = compute_meta_features_pit(player_history, ban_history, matches_df)
    player_features_df = compute_player_features_pit(player_history, base_prior_df)
    team_profile_df = compute_team_profile_pit(matches_df)

    log.info("\n[Step 6] Execution of rewritten feature assembler: static flattening to wide rows...")
    final_prediction_df = assemble_features_for_prediction(
        target_df, team_profile_df, player_features_df, champ_meta_daily
    )
    
    if "result" in final_prediction_df.columns:
        log.info("  [Verification] Label 'result' exists safely in the final matrix.")

    check_dataframe("final_prediction_df", final_prediction_df, log, context="预测宽表特征最终输出")

    if save_parquet:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_prediction_df.to_parquet(os.path.join(OUTPUT_DIR, f"{label}_prediction_wide_features.parquet"), index=False)
        log.info("\n[Success] Exported production wide-table for training to: %s", OUTPUT_DIR)

        champ_meta_daily.to_parquet(os.path.join(OUTPUT_DIR, f"{label}_meta_store.parquet"), index=False)
        log.info("  Exported meta_store for inference")

        player_features_df.to_parquet(os.path.join(OUTPUT_DIR, f"{label}_player_store.parquet"), index=False)
        log.info("  Exported player_store for inference")

        team_profile_df.to_parquet(os.path.join(OUTPUT_DIR, f"{label}_team_profile_store.parquet"), index=False)
        log.info("  Exported team_profile_store for inference")
        
    return final_prediction_df

if __name__ == "__main__":
    setup_logging()
    build_prediction_feature_pipeline(league=None)