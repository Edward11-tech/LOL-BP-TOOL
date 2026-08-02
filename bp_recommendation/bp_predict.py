#!/usr/bin/env python3
"""
bp_predict.py — 单场 BP 实时推荐

模拟真实 BP 场景，用户输入当前 BP 状态和赛前信息，模型输出下一步的 Pick/Ban 推荐 Top-20。

用法:
    cd /Users/siwentu/Desktop/LOL analysis
    python -m bp_recommendation.bp_predict

支持两种模式:
  1) 纯 Draft 模式: 不输入任何战队/选手信息，纯基于 draft 逻辑推荐
     - 所有 player 特征使用默认值 (mastery=0, kda=3.0, wr=0.5, games=0)
     - 不使用新秀惩罚
     - 适用场景: 快速模拟、娱乐局、无战队信息可用

  2) 完整模式: 双方都输入战队名 + 5 名选手 (选手可填 unknown，每队最多 2 名)
     - 已知选手: 使用该选手的真实 player mastery/kda/wr 等特征
     - unknown 选手 (战队已知): 使用该战队所有已知选手的平均特征 + 新秀惩罚
       新秀惩罚系数: mastery×0.3, recent_kda×0.85, recent_wr×0.9,
                      overall_kda×0.9, overall_wr×0.9, overall_games×0.2
     - unknown 选手 (战队未知): 回退到纯 draft 模式，使用默认值，不施加新秀惩罚
     - 适用场景: 职业联赛 BP 预测、有明确战队和选手信息的对局

注意: 不支持混合模式 (一方输入战队另一方不输入)，必须双方都输入或都不输入。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
import json
import pickle
import hashlib
from collections import OrderedDict
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, RECO_DIR)

from logger_config import get_logger, setup_logging
from bp_recommendation.feature_pipeline import (
    load_champion_vocabulary, BP_SEQUENCE,
    CANDIDATE_FEAT_MAP, CANDIDATE_DIM,
)
from bp_recommendation.model_pick.model_pick import BPTacticalTransformerPick
from bp_recommendation.model_ban.model_ban import BPTacticalTransformer as BPTacticalTransformerBan
from bp_recommendation.model_ban.dataloader_ban import BAN_CONTEXT_DIM
from bp_recommendation.model_pick.cascade_pick import _build_feature_matrix_batch, FEATURE_COLS
from bp_recommendation.model_pick.train_pick import CS_FEATURE_INDICES
from bp_recommendation.model_ban.cascade_ban import (
    _build_feature_matrix_batch as _build_ban_feature_matrix_batch,
    _compute_group_features as _compute_ban_group_features,
    FEATURE_COLS as BAN_FEATURE_COLS,
)
from bp_recommendation.feature_monitor import FeatureMonitor
from bp_recommendation.config import get_config, get_production_blend_alpha

PICK_CKPT_DIR = os.path.join(RECO_DIR, "model_pick", "checkpoints")
BAN_CKPT_DIR = os.path.join(RECO_DIR, "model_ban", "checkpoints")
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")
FEATURES_DIR = os.path.join(RECO_DIR, "features")

VOCAB_PATH = os.path.join(CLEANED_DIR, "champion_vocabulary.json")
POS_JSON = os.path.join(CLEANED_DIR, "champion_position_mapping.json")

DEVICE = torch.device("cpu")
POS_2_IDX = {"top": 0, "jungle": 1, "mid": 2, "bot": 3, "support": 4}
LEAGUES = ["LPL", "LCK", "LEC"]
FOLLOWER_STEPS = {8, 10, 18}

log = get_logger(__name__)

# ==================== 英雄名称查找工具 ====================

def build_name_lookup(vocab_path):
    """构建 英雄名称/别名 -> idx 的查找表"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for champ in data["champions"]:
        idx = champ["idx"]
        name = champ["name"].lower()
        lookup[name] = idx
        riot_id = champ.get("riot_id")
        if riot_id is not None:
            lookup[str(riot_id)] = idx
        aliases = champ.get("aliases", {})
        for lang, alias in aliases.items():
            lookup[alias.lower()] = idx
    return lookup, data


def resolve_champion(input_str, name_lookup, idx_to_name, champion_start_idx):
    """将用户输入解析为英雄 idx，支持英文名/中文名/riot_id"""
    if not input_str or input_str.strip() == "":
        return None
    key = input_str.strip().lower()
    if key in name_lookup:
        return name_lookup[key]
    # 尝试部分匹配
    matches = [k for k in name_lookup if k.startswith(key)]
    if len(matches) == 1:
        return name_lookup[matches[0]]
    if len(matches) > 1:
        # 多个匹配，返回 None 让调用者处理
        return None
    return None


# ==================== 特征存储 (轻量版) ====================

class PredictFeatureStore:
    """预测用特征存储，复用 inference_test.py 的逻辑"""

    # 候选矩阵 LRU 缓存配置
    _CANDIDATE_CACHE_SIZE = 64

    # 数据版本指纹相关的特征文件列表（mtime变化时自动失效缓存）
    _DATA_VERSION_FILES = [
        VOCAB_PATH, POS_JSON,
        os.path.join(FEATURES_DIR, "ALL_counter_lookup.json"),
        os.path.join(FEATURES_DIR, "ALL_synergy_lookup.json"),
        os.path.join(FEATURES_DIR, "ALL_meta_store.parquet"),
        os.path.join(FEATURES_DIR, "ALL_player_store.parquet"),
        os.path.join(FEATURES_DIR, "ALL_context.parquet"),
        os.path.join(FEATURES_DIR, "ALL_serving_latest_grudge.json"),
        os.path.join(FEATURES_DIR, "ALL_serving_latest_respect.json"),
        os.path.join(FEATURES_DIR, "ALL_serving_latest_hot_streak.json"),
    ]

    def __init__(self):
        # 候选矩阵 LRU 缓存，避免相同 BP 状态重复构建特征
        self._candidate_cache = OrderedDict()
        # 计算特征数据版本指纹（基于文件 mtime，特征更新时自动失效）
        self._data_version = self._compute_data_version()
        log.info(f"PredictFeatureStore data version: {self._data_version}")
        # 加载词汇表
        self.name_to_idx, self.idx_to_name, self.vocab_size, self.special_tokens, self.champion_start_idx = \
            load_champion_vocabulary(VOCAB_PATH)
        self.PAD_IDX = self.special_tokens["PAD"]
        self.UNK_IDX = self.special_tokens["UNK"]
        cs = self.champion_start_idx
        ve = self.vocab_size
        self.n_champs = ve - cs

        # 加载 counter/synergy
        def load_json(name):
            path = os.path.join(FEATURES_DIR, name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            # 关键特征文件缺失警告
            if name.endswith("_counter_lookup.json"):
                log.warning(f"⚠️  {name} not found at {path}! "
                              f"Counter features will use default value (0.5). "
                              f"CS model will be effectively equivalent to NoCS model. "
                              f"Run feature_pipeline to generate this file.")
            elif name.endswith("_synergy_lookup.json"):
                log.warning(f"⚠️  {name} not found at {path}! "
                              f"Synergy features will use default value (0.5). "
                              f"CS model will be effectively equivalent to NoCS model. "
                              f"Run feature_pipeline to generate this file.")
            elif name.endswith("_grudge_store.json"):
                log.warning(f"⚠️  {name} not found at {path}! Grudge features will be empty.")
            elif name.endswith("_respect_store.json"):
                log.warning(f"⚠️  {name} not found at {path}! Respect features will be empty.")
            elif name.endswith("_hot_streak_store.json"):
                log.warning(f"⚠️  {name} not found at {path}! Hot streak features will be empty.")
            return {}

        self.counter_dict = load_json("ALL_counter_lookup.json")
        self.synergy_dict = load_json("ALL_synergy_lookup.json")

        # 构建 synergy/counter 矩阵
        self.syn_mat = np.full((self.vocab_size, self.vocab_size), 0.5, dtype=np.float32)
        self.ctr_mat = np.full((self.vocab_size, self.vocab_size), 0.5, dtype=np.float32)
        for k, wr in self.synergy_dict.items():
            parts = k.split("||")
            if len(parts) == 2:
                c1 = self.name_to_idx.get(parts[0], -1)
                c2 = self.name_to_idx.get(parts[1], -1)
                if c1 >= 0 and c2 >= 0:
                    self.syn_mat[c1, c2] = self.syn_mat[c2, c1] = float(wr)
        for c_name, opps in self.counter_dict.items():
            c1 = self.name_to_idx.get(c_name, -1)
            if c1 < 0:
                continue
            for opp_name, stats in opps.items():
                c2 = self.name_to_idx.get(opp_name, -1)
                if c2 >= 0:
                    self.ctr_mat[c1, c2] = float(stats.get("win_rate", 0.5))

        # 位置先验
        self.pos_prior = np.zeros((self.vocab_size, 5), dtype=np.float32)
        with open(POS_JSON, "r", encoding="utf-8") as f:
            pos_data = json.load(f)
        for c_name, p_list in pos_data.items():
            cid = self.name_to_idx.get(c_name, -1)
            if cid >= cs:
                for item in p_list:
                    pos_name = item.get("position", "")
                    if pos_name in POS_2_IDX:
                        self.pos_prior[cid, POS_2_IDX[pos_name]] = float(item.get("probability", 0.0))

        # Meta 快照
        self.meta_matrix = np.zeros((self.vocab_size, 4), dtype=np.float32)
        self.meta_matrix[:, 3] = 0.5
        meta_path = os.path.join(FEATURES_DIR, "ALL_meta_store.parquet")
        if os.path.exists(meta_path):
            meta_df = pd.read_parquet(meta_path)
            self._meta_df = meta_df  # 保留原始 DataFrame 供 PIT 查询
            latest_meta = meta_df.drop_duplicates(subset=["champion_id"], keep="last")
            for _, row in latest_meta.iterrows():
                cid = int(row["champion_id"])
                if 0 <= cid < self.vocab_size:
                    self.meta_matrix[cid] = [
                        row["meta_pick_rate_pit"],
                        row["meta_ban_rate_pit"],
                        row["meta_presence_pit"],
                        row["meta_win_rate_pit"],
                    ]

        # Player 快照
        self.player_matrix_map = {}
        # 维度顺序: [mastery, recent_kda, recent_wr, recent_games, overall_kda, overall_wr, overall_games]
        self.default_player_vec = np.array([0.0, 3.0, 0.5, 0.0, 3.0, 0.5, 0.0], dtype=np.float32)
        player_path = os.path.join(FEATURES_DIR, "ALL_player_store.parquet")
        if os.path.exists(player_path):
            player_df = pd.read_parquet(player_path)
            self._player_df = player_df  # 保留原始 DataFrame 供 PIT 查询
            player_snapshot = {}
            latest_player = player_df.drop_duplicates(subset=["player_id", "champion_id"], keep="last")
            for _, row in latest_player.iterrows():
                pid = str(row["player_id"])
                cid = int(row["champion_id"])
                player_snapshot.setdefault(pid, {})[cid] = [
                    row["mastery_score"],
                    row["player_recent_kda_90d"] if pd.notna(row["player_recent_kda_90d"]) else 3.0,
                    row["player_recent_wr_90d"] if pd.notna(row["player_recent_wr_90d"]) else 0.5,
                    row.get("player_recent_games_90d", 0.0) if pd.notna(row.get("player_recent_games_90d", 0.0)) else 0.0,
                    row.get("player_overall_recent_kda", 3.0) if pd.notna(row.get("player_overall_recent_kda", 3.0)) else 3.0,
                    row.get("player_overall_recent_wr", 0.5) if pd.notna(row.get("player_overall_recent_wr", 0.5)) else 0.5,
                    row.get("player_overall_recent_games", 0.0) if pd.notna(row.get("player_overall_recent_games", 0.0)) else 0.0,
                ]
            for pid, champ_dict in player_snapshot.items():
                mat = np.tile(self.default_player_vec, (self.vocab_size, 1))
                for cid, feats in champ_dict.items():
                    if 0 <= cid < self.vocab_size:
                        mat[cid] = feats
                self.player_matrix_map[pid] = mat

        # Grudge/Respect/Hot Streak
        self.grudge_matrix_map = {}
        self.online_respect = {}
        self.online_hot_streak = {}
        self._load_grudge_respect_streak()

        # Team Style
        self.team_style_dict = {}
        context_path = os.path.join(FEATURES_DIR, "ALL_context.parquet")
        self._context_df = None  # 延迟加载
        if os.path.exists(context_path):
            self._context_df = pd.read_parquet(context_path)
            context_df = self._context_df.sort_values("match_seq_idx")
            for side in ["blue", "red"]:
                latest = context_df.drop_duplicates(subset=[f"{side}_team"], keep="last")
                for _, r in latest.iterrows():
                    team = r[f"{side}_team"]
                    if team not in self.team_style_dict:
                        self.team_style_dict[team] = [
                            r.get(f"{side}_team_avg_ckpm", 0.7),
                            r.get(f"{side}_team_avg_golddiffat15", 0),
                            r.get(f"{side}_team_avg_gamelength", 1900),
                            r.get(f"{side}_team_firstdragon_rate", 0.5),
                            r.get(f"{side}_team_firsttower_rate", 0.5),
                        ]

        # Team-Player 映射 & Team Average Player 特征
        self.team_players_map = {}    # team_name -> set of player_ids
        self.team_avg_player_mat = {}  # team_name -> (vocab_size, 6) 平均特征矩阵
        self._build_team_player_features()

        log.info("Feature store loaded.")

    def _build_team_player_features(self):
        """构建 team->players 映射和 team average player 特征矩阵"""
        if self._context_df is None:
            return

        # 1. 构建 team -> player_ids 映射
        for side in ["blue", "red"]:
            for pos in ["top", "jng", "mid", "bot", "sup"]:
                pid_col = f"{side}_{pos}_player_id"
                team_col = f"{side}_team"
                for _, row in self._context_df.iterrows():
                    pid = str(row[pid_col]).strip()
                    team = str(row[team_col]).strip()
                    if pid and pid != "nan":
                        self.team_players_map.setdefault(team, set()).add(pid)

        # 2. 为每个战队计算平均 player 特征矩阵
        #    取该战队所有已知选手的 player_matrix 的均值
        for team, pids in self.team_players_map.items():
            mats = []
            for pid in pids:
                pmat = self.player_matrix_map.get(pid)
                if pmat is not None:
                    mats.append(pmat)
            if mats:
                avg_mat = np.mean(mats, axis=0)  # (vocab_size, 6)
            else:
                avg_mat = np.tile(self.default_player_vec, (self.vocab_size, 1))
            self.team_avg_player_mat[team] = avg_mat.astype(np.float32)

    def get_unknown_player_mat(self, team_name, position=None):
        """获取 unknown 选手的特征矩阵

        战队已知: 使用该战队所有已知选手的平均特征 + 新秀惩罚
          新秀惩罚系数: mastery×0.3, recent_kda×0.85, recent_wr×0.9,
                         overall_kda×0.9, overall_wr×0.9, overall_games×0.2
        战队未知: 回退到纯 draft 模式，使用默认值，不施加新秀惩罚
        """
        # 维度顺序: [mastery, recent_kda, recent_wr, recent_games, overall_kda, overall_wr, overall_games]
        ROOKIE_PENALTY = np.array([0.3, 0.85, 0.9, 0.0, 0.9, 0.9, 0.2], dtype=np.float32)

        base_mat = self.team_avg_player_mat.get(team_name)
        if base_mat is not None:
            # 战队已知: 战队平均 + 新秀惩罚
            return base_mat * ROOKIE_PENALTY[np.newaxis, :]
        else:
            # 战队未知: 纯 draft 模式，使用默认值，不施加惩罚
            return np.tile(self.default_player_vec, (self.vocab_size, 1))

    # ========== PIT (Point-in-Time) 辅助方法 ==========
    # 这些方法仅用于特征一致性校验，不改动在线推理逻辑。
    # 它们按 match_date 过滤历史数据，取该时间点前的最新快照。

    def get_team_style_pit(self, team_name, side, match_date):
        """获取指定战队在 match_date 时的队伍风格特征。"""
        if self._context_df is None:
            return [0.7, 0.0, 1900.0, 0.5, 0.5]
        team_col = f"{side}_team"
        pit_mask = (self._context_df[team_col] == team_name) & (self._context_df["date"] <= match_date)
        pit_df = self._context_df[pit_mask].sort_values("match_seq_idx")
        if pit_df.empty:
            return [0.7, 0.0, 1900.0, 0.5, 0.5]
        latest = pit_df.iloc[-1]
        return [
            float(latest.get(f"{side}_team_avg_ckpm", 0.7)),
            float(latest.get(f"{side}_team_avg_golddiffat15", 0.0)),
            float(latest.get(f"{side}_team_avg_gamelength", 1900.0)),
            float(latest.get(f"{side}_team_firstdragon_rate", 0.5)),
            float(latest.get(f"{side}_team_firsttower_rate", 0.5)),
        ]

    def get_meta_matrix_pit(self, match_date):
        """获取 match_date 时点的 meta 矩阵 (vocab_size, 4)。"""
        meta_mat = np.zeros((self.vocab_size, 4), dtype=np.float32)
        meta_mat[:, 3] = 0.5
        if not hasattr(self, "_meta_df") or self._meta_df is None:
            return meta_mat
        pit_mask = self._meta_df["date"] <= match_date
        pit_df = self._meta_df[pit_mask]
        if pit_df.empty:
            return meta_mat
        if "date" in pit_df.columns:
            pit_df = pit_df.sort_values("date")
        latest_per_champ = pit_df.drop_duplicates(subset=["champion_id"], keep="last")
        for _, row in latest_per_champ.iterrows():
            cid = int(row["champion_id"])
            if 0 <= cid < self.vocab_size:
                meta_mat[cid] = [
                    float(row["meta_pick_rate_pit"]),
                    float(row["meta_ban_rate_pit"]),
                    float(row["meta_presence_pit"]),
                    float(row["meta_win_rate_pit"]),
                ]
        return meta_mat

    def get_player_mat_pit(self, player_id, match_date):
        """获取指定选手在 match_date 时点的 player matrix (vocab_size, 6)。"""
        if not hasattr(self, "_player_df") or self._player_df is None:
            return None
        pit_mask = (self._player_df["player_id"].astype(str) == str(player_id)) & (self._player_df["date"] <= match_date)
        pit_df = self._player_df[pit_mask]
        if pit_df.empty:
            return None
        if "date" in pit_df.columns:
            pit_df = pit_df.sort_values("date")
        latest_per_champ = pit_df.drop_duplicates(subset=["champion_id"], keep="last")
        mat = np.tile(self.default_player_vec, (self.vocab_size, 1)).astype(np.float32)
        for _, row in latest_per_champ.iterrows():
            cid = int(row["champion_id"])
            if 0 <= cid < self.vocab_size:
                mat[cid] = [
                    float(row["mastery_score"]),
                    float(row["player_recent_kda_90d"]) if pd.notna(row["player_recent_kda_90d"]) else 3.0,
                    float(row["player_recent_wr_90d"]) if pd.notna(row["player_recent_wr_90d"]) else 0.5,
                    float(row["player_recent_games_90d"]) if pd.notna(row.get("player_recent_games_90d", 0.0)) else 0.0,
                    float(row["player_overall_recent_kda"]) if pd.notna(row["player_overall_recent_kda"]) else 3.0,
                    float(row["player_overall_recent_wr"]) if pd.notna(row["player_overall_recent_wr"]) else 0.5,
                    float(row["player_overall_recent_games"]) if pd.notna(row["player_overall_recent_games"]) else 0.0,
                ]
        return mat

    def get_unknown_player_mat_pit(self, team_name, match_date):
        """获取 unknown 选手在 match_date 时点的特征矩阵。"""
        # 维度顺序: [mastery, recent_kda, recent_wr, recent_games, overall_kda, overall_wr, overall_games]
        ROOKIE_PENALTY = np.array([0.3, 0.85, 0.9, 0.0, 0.9, 0.9, 0.2], dtype=np.float32)
        if not hasattr(self, "_player_df") or self._player_df is None or self._context_df is None:
            return np.tile(self.default_player_vec * ROOKIE_PENALTY, (self.vocab_size, 1))

        # 从 context_df 中找到该战队在 match_date 前的所有选手
        player_ids = set()
        for side in ["blue", "red"]:
            team_col = f"{side}_team"
            for pos in ["top", "jng", "mid", "bot", "sup"]:
                pid_col = f"{side}_{pos}_player_id"
                pit_mask = ((self._context_df[team_col] == team_name) &
                            (self._context_df["date"] <= match_date))
                for pid in self._context_df.loc[pit_mask, pid_col]:
                    pid_str = str(pid).strip()
                    if pid_str and pid_str != "nan":
                        player_ids.add(pid_str)

        if not player_ids:
            return np.tile(self.default_player_vec * ROOKIE_PENALTY, (self.vocab_size, 1))

        # 聚合所有选手的 PIT 特征
        pit_mask = self._player_df["player_id"].astype(str).isin(player_ids) & (self._player_df["date"] <= match_date)
        pit_df = self._player_df[pit_mask]
        if pit_df.empty:
            return np.tile(self.default_player_vec * ROOKIE_PENALTY, (self.vocab_size, 1))

        if "date" in pit_df.columns:
            pit_df = pit_df.sort_values("date")
        latest_per_player = pit_df.drop_duplicates(subset=["player_id", "champion_id"], keep="last")

        agg = {}
        for _, row in latest_per_player.iterrows():
            cid = int(row["champion_id"])
            if not (0 <= cid < self.vocab_size):
                continue
            if cid not in agg:
                agg[cid] = []
            agg[cid].append([
                float(row["mastery_score"]),
                float(row["player_recent_kda_90d"]) if pd.notna(row["player_recent_kda_90d"]) else 3.0,
                float(row["player_recent_wr_90d"]) if pd.notna(row["player_recent_wr_90d"]) else 0.5,
                float(row["player_recent_games_90d"]) if pd.notna(row.get("player_recent_games_90d", 0.0)) else 0.0,
                float(row["player_overall_recent_kda"]) if pd.notna(row["player_overall_recent_kda"]) else 3.0,
                float(row["player_overall_recent_wr"]) if pd.notna(row["player_overall_recent_wr"]) else 0.5,
                float(row["player_overall_recent_games"]) if pd.notna(row["player_overall_recent_games"]) else 0.0,
            ])
        avg_mat = np.tile(self.default_player_vec, (self.vocab_size, 1)).astype(np.float32)
        for cid, rows in agg.items():
            avg_mat[cid] = np.mean(rows, axis=0)
        return avg_mat * ROOKIE_PENALTY[np.newaxis, :]

    def _load_grudge_respect_streak(self):
        """加载恩怨/绝活/火热状态特征。

        优先使用 serving_latest_*.json 快照文件（由 feature_pipeline 预聚合的最新状态），
        若快照不存在则回退到 per-game store + 时间衰减聚合。
        """
        def load_json(name):
            path = os.path.join(FEATURES_DIR, name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        # ---- 优先加载 serving_latest 快照 ----
        # 关键特征文件缺失时记录 error 并降级，避免静默失败导致线上特征全 0
        critical_snapshots = {
            "grudge": "ALL_serving_latest_grudge.json",
            "respect": "ALL_serving_latest_respect.json",
            "hot_streak": "ALL_serving_latest_hot_streak.json",
        }
        missing_critical = []
        for label_key, fname in critical_snapshots.items():
            fpath = os.path.join(FEATURES_DIR, fname)
            if not os.path.exists(fpath):
                missing_critical.append(label_key)
                log.error(f"[Serving] 关键特征文件缺失: {fname}，对应特征将降级为默认值")
        if missing_critical:
            log.error(f"[Serving] 共 {len(missing_critical)} 个关键快照缺失: {missing_critical}")

        snapshot_grudge = load_json("ALL_serving_latest_grudge.json")
        snapshot_respect = load_json("ALL_serving_latest_respect.json")
        snapshot_streak = load_json("ALL_serving_latest_hot_streak.json")

        if snapshot_grudge:
            # 快照结构: {team_a: {team_b: {cid: rate}}}
            for ta, opp_data in snapshot_grudge.items():
                for tb, champ_dict in opp_data.items():
                    mat = np.zeros(self.vocab_size, dtype=np.float32)
                    for cid_str, val in champ_dict.items():
                        cid = int(cid_str)
                        if 0 <= cid < self.vocab_size:
                            mat[cid] = float(val)
                    self.grudge_matrix_map[(ta, tb)] = mat
            log.info(f"[Serving] Loaded serving_latest_grudge: {len(snapshot_grudge)} teams")

        if snapshot_respect:
            # 快照结构: {pid: {signature_champion_id, signature_mastery}}
            for pid, rinfo in snapshot_respect.items():
                self.online_respect[str(pid)] = rinfo
            log.info(f"[Serving] Loaded serving_latest_respect: {len(snapshot_respect)} players")

        if snapshot_streak:
            # 快照结构: {pid: {hot_champion_id, hot_win_rate, hot_avg_kda, hot_games}}
            for pid, hs_info in snapshot_streak.items():
                self.online_hot_streak[str(pid)] = hs_info
            log.info(f"[Serving] Loaded serving_latest_hot_streak: {len(snapshot_streak)} players")

        # 若三种快照均已加载，跳过 per-game store 的冗余聚合
        if snapshot_grudge and snapshot_respect and snapshot_streak:
            return

        # ---- 回退: per-game store + 时间衰减聚合 ----
        log.warning("[Serving] serving_latest snapshots incomplete, falling back to per-game store aggregation")
        context_path = os.path.join(FEATURES_DIR, "ALL_context.parquet")
        if not os.path.exists(context_path):
            return
        context_df = pd.read_parquet(context_path)
        context_df = context_df.sort_values("match_seq_idx")
        gid2seq = dict(zip(context_df["gameid"].astype(str), context_df["match_seq_idx"]))
        max_seq = context_df["match_seq_idx"].max()

        # Grudge: 聚合 per-game store，仅填充尚未通过快照加载的 team pair
        raw_grudge = load_json("ALL_grudge_store.json")
        GRUDGE_HALF_LIFE = 500
        grudge_entries = {}
        for game_id, team_data in raw_grudge.items():
            seq = gid2seq.get(game_id, -1)
            if seq < 0:
                continue
            for team_a, opp_data in team_data.items():
                for team_b, champ_stats in opp_data.items():
                    for cid, val in champ_stats.items():
                        grudge_entries.setdefault((team_a, team_b, cid), []).append((seq, float(val)))

        # 先聚合到临时 dict，避免遍历时修改 self.grudge_matrix_map
        online_grudge = {}
        for (ta, tb, cid), entries in grudge_entries.items():
            if (ta, tb) in self.grudge_matrix_map:
                continue  # 已通过快照加载
            entries.sort(key=lambda x: x[0])
            w_sum, wv_sum = 0.0, 0.0
            for seq, val in entries:
                w = 2.0 ** (-(max_seq - seq) / GRUDGE_HALF_LIFE)
                wv_sum += w * val
                w_sum += w
            online_grudge.setdefault(ta, {}).setdefault(tb, {})[cid] = wv_sum / w_sum if w_sum > 0 else 0.0

        for ta, opp_data in online_grudge.items():
            for tb, champ_dict in opp_data.items():
                mat = np.zeros(self.vocab_size, dtype=np.float32)
                for cid_str, val in champ_dict.items():
                    cid = int(cid_str)
                    if 0 <= cid < self.vocab_size:
                        mat[cid] = val
                self.grudge_matrix_map[(ta, tb)] = mat

        # Respect (仅填充尚未通过快照加载的 player)
        raw_respect = load_json("ALL_respect_store.json")
        for game_id, player_data in raw_respect.items():
            seq = gid2seq.get(game_id, -1)
            if seq < 0:
                continue
            for pid, rinfo in player_data.items():
                if str(pid) not in self.online_respect:
                    self.online_respect[str(pid)] = rinfo

        # Hot Streak (仅填充尚未通过快照加载的 player)
        raw_streak = load_json("ALL_hot_streak_store.json")
        for game_id, player_data in raw_streak.items():
            seq = gid2seq.get(game_id, -1)
            if seq < 0:
                continue
            for pid, hs_info in player_data.items():
                if str(pid) not in self.online_hot_streak:
                    self.online_hot_streak[str(pid)] = hs_info

    def map_champ(self, name):
        if not name or name.lower() == "nan" or name == "<EMPTY_BAN>":
            return self.UNK_IDX
        return self.name_to_idx.get(name, self.UNK_IDX)

    def get_pick_candidate_matrix(self, side_str, ally_champs, enemy_champs, 
                                  unavail_set,ally_pids, enemy_pids, target_step, 
                                  team_name, opp_team, pre_unavail_list=None):
        """构建 Pick 候选矩阵 (CANDIDATE_DIM 维)

        ally_pids / enemy_pids: 每个元素为选手 ID 字符串，"unknown" 表示未知选手
        """
        # LRU 缓存：相同 BP 状态直接返回已构建的候选矩阵
        cache_key = self._make_cache_key("pick", side_str, ally_champs, enemy_champs,
                                         unavail_set, ally_pids, enemy_pids,
                                         target_step, team_name, opp_team, pre_unavail_list)
        if cache_key in self._candidate_cache:
            self._candidate_cache.move_to_end(cache_key)
            return self._candidate_cache[cache_key]

        cs = self.champion_start_idx
        ve = self.vocab_size
        n_champs = self.n_champs
        FI = CANDIDATE_FEAT_MAP

        cand = np.zeros((self.vocab_size, CANDIDATE_DIM), dtype=np.float32)
        cand[:, FI["meta_pick"]:FI["meta_wr"]+1] = self.meta_matrix
        cand[:, FI["pos_top"]:FI["pos_sup"]+1] = self.pos_prior

        # Player features (支持 unknown 选手)
        # 使用 default_player_vec 初始化，防止纯 Draft 模式 KDA/WR 变 0
        ally_feat_mat = np.tile(self.default_player_vec, (5, n_champs, 1)).astype(np.float32)
        
        for i, pid in enumerate(ally_pids[:5]):
            pmat = None
            if pid == "unknown":
                pmat = self.get_unknown_player_mat(team_name)
            elif pid:  # 只有 pid 非空才去查询，跳过纯 Draft 模式的 ""
                pmat = self.player_matrix_map.get(pid)
                
            if pmat is not None:
                ally_feat_mat[i] = pmat[cs:ve]

        ally_pos_sum = np.zeros(5, dtype=np.float32)
        for c in ally_champs:
            ally_pos_sum += self.pos_prior[c]
        enemy_pos_sum = np.zeros(5, dtype=np.float32)
        for c in enemy_champs:
            enemy_pos_sum += self.pos_prior[c]
        ally_missing_roles = np.clip(1.0 - ally_pos_sum, 0.0, 1.0)
        enemy_missing_roles = np.clip(1.0 - enemy_pos_sum, 0.0, 1.0)

        # 切片 4:11 包含全部 7 个 player 特征 (含 recent_games@7)
        cand[cs:ve, FI["player_mastery"]:FI["player_overall_games"]+1] = (ally_feat_mat * ally_missing_roles[:, None, None]).max(axis=0)

        # Synergy/Counter
        if ally_champs:
            ally_arr = np.array(ally_champs, dtype=np.int64)
            cand[cs:ve, FI["ally_synergy"]] = np.max(self.syn_mat[cs:ve, ally_arr], axis=1)
            cand[cs:ve, FI["ally_counter"]] = np.max(1.0 - self.ctr_mat[cs:ve, ally_arr], axis=1)
        if enemy_champs:
            enemy_arr = np.array(enemy_champs, dtype=np.int64)
            cand[cs:ve, FI["enemy_counter"]] = np.max(1.0 - self.ctr_mat[cs:ve, enemy_arr], axis=1)
            cand[cs:ve, FI["enemy_synergy"]] = np.max(self.syn_mat[cs:ve, enemy_arr], axis=1)

        # Role fit
        pos_block = cand[cs:ve, FI["pos_top"]:FI["pos_sup"]+1]
        cand[cs:ve, FI["ally_role_fit"]] = pos_block @ ally_missing_roles
        cand[cs:ve, FI["enemy_role_fit"]] = pos_block @ enemy_missing_roles
        cand[cs:ve, FI["is_pick"]] = 1.0

        # Enemy mastery (支持 unknown 选手)
        enemy_mastery_matrix = np.zeros((5, n_champs), dtype=np.float32)
        for i, epid in enumerate(enemy_pids[:5]):
            if epid == "unknown":
                pmat = self.get_unknown_player_mat(opp_team)
            else:
                pmat = self.player_matrix_map.get(epid)
            if pmat is not None:
                enemy_mastery_matrix[i] = pmat[cs:ve, 0]
        weighted_enemy_mastery = enemy_mastery_matrix * enemy_missing_roles[:, None]
        cand[cs:ve, FI["enemy_mastery_max"]] = weighted_enemy_mastery.max(axis=0)
        cand[cs:ve, FI["enemy_mastery_mean"]] = weighted_enemy_mastery.mean(axis=0)

        # Ban step count
        cand[cs:ve, FI["ban_step"]] = sum(1 for i in range(target_step + 1)
                              if BP_SEQUENCE[i][0] == "ban" and BP_SEQUENCE[i][1] == side_str)

        # Grudge
        grudge_vec = self.grudge_matrix_map.get((team_name, opp_team))
        if grudge_vec is not None:
            cand[cs:ve, FI["grudge"]] = grudge_vec[cs:ve]

        # Respect & Hot Streak
        enemy_respect_vec = np.zeros(self.vocab_size, dtype=np.float32)
        enemy_streak_vec = np.zeros(self.vocab_size, dtype=np.float32)
        for epid in enemy_pids[:5]:
            rinfo = self.online_respect.get(epid)
            if rinfo:
                sig_cid = int(rinfo.get("signature_champion_id", -1))
                if 0 <= sig_cid < self.vocab_size:
                    val = min(float(rinfo.get("signature_mastery", 0.0)) / 100.0, 1.0)
                    enemy_respect_vec[sig_cid] = max(enemy_respect_vec[sig_cid], val)
            hs_info = self.online_hot_streak.get(epid)
            if hs_info:
                hot_cid = int(hs_info.get("hot_champion_id", -1))
                if 0 <= hot_cid < self.vocab_size:
                    val = (float(hs_info.get("hot_win_rate", 0.0)) * 0.5 +
                           (min(float(hs_info.get("hot_avg_kda", 0.0)), 10.0) / 10.0) * 0.3 +
                           (min(int(hs_info.get("hot_games", 0)), 10) / 10.0) * 0.2)
                    enemy_streak_vec[hot_cid] = max(enemy_streak_vec[hot_cid], val)
        cand[cs:ve, FI["respect"]] = enemy_respect_vec[cs:ve]
        cand[cs:ve, FI["hot_streak"]] = enemy_streak_vec[cs:ve]

        # Misc
        cand[cs:ve, FI["n_ally_picked"]] = float(len(ally_champs))
        cand[cs:ve, FI["is_red_side"]] = 0.0 if side_str == "blue" else 1.0
        if ally_champs:
            cand[cs:ve, FI["last_ally_synergy"]] = self.syn_mat[cs:ve, ally_champs[-1]]
        else:
            cand[cs:ve, FI["last_ally_synergy"]] = 0.5

        if pre_unavail_list is not None:
            for uid in pre_unavail_list:
                if cs <= uid < ve:
                    cand[uid, FI["is_fearless_banned"]] = 1.0

        # Mask: 排除 special tokens (PAD/UNK/MASK/EMPTY_BAN) 和已用英雄
        mask = np.ones(self.vocab_size, dtype=np.float32)
        mask[:cs] = 0.0
        # 排除 EMPTY_BAN 等落在 champion 区间的 special token
        for sp_idx in self.special_tokens.values():
            if 0 <= sp_idx < self.vocab_size:
                mask[sp_idx] = 0.0
        for uid in unavail_set:
            if 0 <= uid < self.vocab_size:
                mask[uid] = 0.0

        # 写入 LRU 缓存
        self._put_cache(cache_key, (cand, mask))
        return cand, mask

    def get_ban_candidate_matrix(self, side_str, ally_champs, enemy_champs, 
                                 unavail_set,ally_pids, enemy_pids, target_step, 
                                 team_name, opp_team, pre_unavail_list=None):
        """构建 Ban 候选矩阵 (CANDIDATE_DIM=33 维)

        Ban 模型与 Pick 模型使用相同的 33 维特征布局，
        包含 last_ally_synergy@30 和 is_fearless_banned@31。
        ally_pids / enemy_pids: 每个元素为选手 ID 字符串，"unknown" 表示未知选手
        """
        # LRU 缓存：相同 BP 状态直接返回已构建的候选矩阵
        cache_key = self._make_cache_key("ban", side_str, ally_champs, enemy_champs,
                                         unavail_set, ally_pids, enemy_pids,
                                         target_step, team_name, opp_team, pre_unavail_list)
        if cache_key in self._candidate_cache:
            self._candidate_cache.move_to_end(cache_key)
            return self._candidate_cache[cache_key]

        cs = self.champion_start_idx
        ve = self.vocab_size
        n_champs = self.n_champs
        curr_side_code = 0 if side_str == "blue" else 1
        FI = CANDIDATE_FEAT_MAP

        cand = np.zeros((self.vocab_size, CANDIDATE_DIM), dtype=np.float32)
        cand[:, FI["meta_pick"]:FI["meta_wr"]+1] = self.meta_matrix
        cand[:, FI["pos_top"]:FI["pos_sup"]+1] = self.pos_prior

        # Player features (支持 unknown 选手)
        # 使用 default_player_vec 初始化，防止纯 Draft 模式 KDA/WR 变 0
        ally_feat_mat = np.tile(self.default_player_vec, (5, n_champs, 1)).astype(np.float32)
        
        for i, pid in enumerate(ally_pids[:5]):
            pmat = None
            if pid == "unknown":
                pmat = self.get_unknown_player_mat(team_name)
            elif pid:  # 只有 pid 非空才去查询，跳过纯 Draft 模式的 ""
                pmat = self.player_matrix_map.get(pid)
                
            if pmat is not None:
                ally_feat_mat[i] = pmat[cs:ve]

        ally_pos_sum = np.zeros(5, dtype=np.float32)
        for c in ally_champs:
            ally_pos_sum += self.pos_prior[c]
        enemy_pos_sum = np.zeros(5, dtype=np.float32)
        for c in enemy_champs:
            enemy_pos_sum += self.pos_prior[c]
        ally_missing_roles = np.clip(1.0 - ally_pos_sum, 0.0, 1.0)
        enemy_missing_roles = np.clip(1.0 - enemy_pos_sum, 0.0, 1.0)

        # 切片 4:11 包含全部 7 个 player 特征 (含 recent_games@7)
        cand[cs:ve, FI["player_mastery"]:FI["player_overall_games"]+1] = (ally_feat_mat * ally_missing_roles[:, None, None]).max(axis=0)

        # Synergy/Counter
        if ally_champs:
            ally_arr = np.array(ally_champs, dtype=np.int64)
            cand[cs:ve, FI["ally_synergy"]] = np.max(self.syn_mat[cs:ve, ally_arr], axis=1)
            cand[cs:ve, FI["ally_counter"]] = np.max(1.0 - self.ctr_mat[cs:ve, ally_arr], axis=1)
        if enemy_champs:
            enemy_arr = np.array(enemy_champs, dtype=np.int64)
            cand[cs:ve, FI["enemy_counter"]] = np.max(1.0 - self.ctr_mat[cs:ve, enemy_arr], axis=1)
            cand[cs:ve, FI["enemy_synergy"]] = np.max(self.syn_mat[cs:ve, enemy_arr], axis=1)

        # Role fit
        pos_block = cand[cs:ve, FI["pos_top"]:FI["pos_sup"]+1]
        cand[cs:ve, FI["ally_role_fit"]] = pos_block @ ally_missing_roles
        cand[cs:ve, FI["enemy_role_fit"]] = pos_block @ enemy_missing_roles

        # is_pick flag
        cand[cs:ve, FI["is_pick"]] = 0.0  # ban 步骤

        # Enemy mastery (支持 unknown 选手)
        enemy_mastery_matrix = np.zeros((5, n_champs), dtype=np.float32)
        for i, epid in enumerate(enemy_pids[:5]):
            pmat = None
            if epid == "unknown":
                pmat = self.get_unknown_player_mat(opp_team)
            elif epid:
                pmat = self.player_matrix_map.get(epid)
                
            if pmat is not None:
                enemy_mastery_matrix[i] = pmat[cs:ve, 0]
            else:
                enemy_mastery_matrix[i] = self.default_player_vec[0]
        weighted_enemy_mastery = enemy_mastery_matrix * enemy_missing_roles[:, None]
        cand[cs:ve, FI["enemy_mastery_max"]] = weighted_enemy_mastery.max(axis=0)
        cand[cs:ve, FI["enemy_mastery_mean"]] = weighted_enemy_mastery.mean(axis=0)

        # Ban step count
        cand[cs:ve, FI["ban_step"]] = sum(1 for i in range(target_step + 1)
                              if BP_SEQUENCE[i][0] == "ban" and BP_SEQUENCE[i][1] == side_str)

        # Grudge
        grudge_vec = self.grudge_matrix_map.get((team_name, opp_team))
        if grudge_vec is not None:
            cand[cs:ve, FI["grudge"]] = grudge_vec[cs:ve]

        # Respect & Hot Streak
        enemy_respect_vec = np.zeros(self.vocab_size, dtype=np.float32)
        enemy_streak_vec = np.zeros(self.vocab_size, dtype=np.float32)
        for epid in enemy_pids[:5]:
            rinfo = self.online_respect.get(epid)
            if rinfo:
                sig_cid = int(rinfo.get("signature_champion_id", -1))
                if 0 <= sig_cid < self.vocab_size:
                    val = min(float(rinfo.get("signature_mastery", 0.0)) / 100.0, 1.0)
                    enemy_respect_vec[sig_cid] = max(enemy_respect_vec[sig_cid], val)
            hs_info = self.online_hot_streak.get(epid)
            if hs_info:
                hot_cid = int(hs_info.get("hot_champion_id", -1))
                if 0 <= hot_cid < self.vocab_size:
                    val = (float(hs_info.get("hot_win_rate", 0.0)) * 0.5 +
                           (min(float(hs_info.get("hot_avg_kda", 0.0)), 10.0) / 10.0) * 0.3 +
                           (min(int(hs_info.get("hot_games", 0)), 10) / 10.0) * 0.2)
                    enemy_streak_vec[hot_cid] = max(enemy_streak_vec[hot_cid], val)
        cand[cs:ve, FI["respect"]] = enemy_respect_vec[cs:ve]
        cand[cs:ve, FI["hot_streak"]] = enemy_streak_vec[cs:ve]

        # Misc
        cand[cs:ve, FI["n_ally_picked"]] = float(len(ally_champs))
        cand[cs:ve, FI["is_red_side"]] = float(curr_side_code)

        # is_fearless_banned: 前置局已用英雄 (Fearless Draft)，与 Pick 模型一致 @idx31
        # 注意: ban 模型不使用 last_ally_synergy@idx30（与训练时 use_extended_features=False 一致），idx30 保持为 0
        if pre_unavail_list is not None:
            for uid in pre_unavail_list:
                if cs <= uid < ve:
                    cand[uid, FI["is_fearless_banned"]] = 1.0

        # Mask: 排除 special tokens (PAD/UNK/MASK/EMPTY_BAN) 和已用英雄
        mask = np.ones(self.vocab_size, dtype=np.float32)
        mask[:cs] = 0.0
        for sp_idx in self.special_tokens.values():
            if 0 <= sp_idx < self.vocab_size:
                mask[sp_idx] = 0.0
        for uid in unavail_set:
            if 0 <= uid < self.vocab_size:
                mask[uid] = 0.0

        # 写入 LRU 缓存
        self._put_cache(cache_key, (cand, mask))
        return cand, mask

    # ---- LRU 缓存辅助方法 ----

    def _compute_data_version(self):
        """计算特征数据版本指纹。

        收集所有关键特征文件的 mtime，拼接后取 MD5。
        当 feature_pipeline 重新生成特征文件时，mtime 变化，
        旧版本的候选矩阵缓存自动失效，无需重启服务。

        Returns:
            str: 12位hex版本指纹
        """
        mtime_parts = []
        for fpath in self._DATA_VERSION_FILES:
            try:
                if os.path.exists(fpath):
                    mtime_parts.append(f"{int(os.path.getmtime(fpath))}")
                else:
                    mtime_parts.append("0")
            except OSError:
                mtime_parts.append("0")
        raw = "|".join(mtime_parts)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def _make_cache_key(self, action, side_str, ally_champs, enemy_champs,
                        unavail_set, ally_pids, enemy_pids, target_step,
                        team_name, opp_team, pre_unavail_list):
        """根据 BP 上下文生成可哈希的缓存键（包含数据版本指纹）。"""
        key_parts = [
            self._data_version,
            action, side_str,
            tuple(ally_champs), tuple(enemy_champs),
            tuple(sorted(unavail_set)) if isinstance(unavail_set, set) else tuple(unavail_set),
            tuple(ally_pids), tuple(enemy_pids),
            target_step, team_name, opp_team,
            tuple(pre_unavail_list) if pre_unavail_list else (),
        ]
        return hashlib.md5(str(key_parts).encode("utf-8")).hexdigest()

    def _put_cache(self, key, value):
        """写入 LRU 缓存，超过容量时淘汰最久未使用的条目。"""
        self._candidate_cache[key] = value
        self._candidate_cache.move_to_end(key)
        while len(self._candidate_cache) > self._CANDIDATE_CACHE_SIZE:
            self._candidate_cache.popitem(last=False)

# ==================== 模型加载 ====================

def _infer_pick_params_from_state(state_dict):
    """从 BPTacticalTransformerPick 的 state_dict 推断模型参数。

    返回值:
        dict with keys: c_dim, query_dim, candidate_hidden, h_dim, n_layers, n_heads
    """
    params = {}

    # c_dim: context_mlp.0.weight shape = (c_dim, context_dim)
    w = state_dict.get("context_mlp.0.weight")
    if w is not None:
        params["c_dim"] = w.shape[0]

    # query_dim: bert_proj.weight shape = (query_dim, h_dim)
    w = state_dict.get("bert_proj.weight")
    if w is not None:
        params["query_dim"] = w.shape[0]
        params["h_dim"] = w.shape[1]

    # candidate_hidden: candidate_mlp.5.weight shape = (query_dim, candidate_hidden//2)
    w = state_dict.get("candidate_mlp.5.weight")
    if w is not None:
        params["candidate_hidden"] = w.shape[1] * 2

    # candidate_dim: candidate_mlp.0.weight shape = (candidate_hidden, candidate_dim)
    w = state_dict.get("candidate_mlp.0.weight")
    if w is not None:
        params["candidate_dim"] = w.shape[1]

    # n_layers: count bert.transformer.layer.N keys
    n_layers = 0
    while f"bert.transformer.layer.{n_layers}.attention.q_lin.weight" in state_dict:
        n_layers += 1
    params["n_layers"] = n_layers

    # n_heads: from distilbert config, infer from q_lin weight
    # q_lin.weight shape = (h_dim, h_dim), n_heads needs to divide h_dim evenly
    if "h_dim" in params:
        h_dim = params["h_dim"]
        # Common head counts that divide h_dim=384: 4, 6, 8, 12
        # Use the default from the model class
        for nh in [12, 8, 6, 4]:
            if h_dim % nh == 0:
                params["n_heads"] = nh
                break
        if "n_heads" not in params:
            params["n_heads"] = 4  # fallback

    return params


def _infer_ban_params_from_state(state_dict):
    """从 BPTacticalTransformer (Ban) 的 state_dict 推断模型参数。

    返回值:
        dict with keys: c_dim, query_dim, h_dim, n_layers, n_heads, candidate_dim
    """
    params = {}

    # c_dim: context_mlp.0.weight shape = (c_dim, context_dim)
    w = state_dict.get("context_mlp.0.weight")
    if w is not None:
        params["c_dim"] = w.shape[0]

    # query_dim: bert_proj.weight shape = (query_dim, h_dim)
    w = state_dict.get("bert_proj.weight")
    if w is not None:
        params["query_dim"] = w.shape[0]
        params["h_dim"] = w.shape[1]

    # candidate_dim: candidate_mlp.0.weight shape = (256, candidate_dim)
    w = state_dict.get("candidate_mlp.0.weight")
    if w is not None:
        params["candidate_dim"] = w.shape[1]

    # n_layers: count bert.transformer.layer.N keys
    n_layers = 0
    while f"bert.transformer.layer.{n_layers}.attention.q_lin.weight" in state_dict:
        n_layers += 1
    params["n_layers"] = n_layers

    # n_heads
    if "h_dim" in params:
        h_dim = params["h_dim"]
        for nh in [6, 8, 12, 4]:
            if h_dim % nh == 0:
                params["n_heads"] = nh
                break
        if "n_heads" not in params:
            params["n_heads"] = 6

    return params


class BPRecommender:
    """BP 推荐器：加载所有模型，提供 predict 方法"""

    def __init__(self):
        log.info("Loading models and feature store...")
        self.store = PredictFeatureStore()
        self._load_pick_models()
        self._load_ban_models()
        # 初始化特征监控器（PSI 基线可选）
        baseline_dir = os.path.join(RECO_DIR, "features")
        self.feature_monitor = FeatureMonitor(baseline_dir=baseline_dir)
        log.info("All models loaded.")

    def _get_device(self):
        """返回当前推理设备，供 FallbackManager 等外部组件使用"""
        return DEVICE

    def _load_pick_models(self):
        # CS Transformer
        cs_ckpt = torch.load(os.path.join(PICK_CKPT_DIR, "best_model_cs.pt"),
                             map_location=DEVICE, weights_only=False)
        self.pick_ctx_dim = cs_ckpt.get("context_dim", 15)
        cs_params = _infer_pick_params_from_state(cs_ckpt["model_state_dict"])
        self.pick_cs_model = BPTacticalTransformerPick(
            vocab_size=self.store.vocab_size, context_dim=self.pick_ctx_dim,
            candidate_dim=cs_params.get("candidate_dim", cs_ckpt.get("candidate_dim", CANDIDATE_DIM)),
            h_dim=cs_params.get("h_dim", 384),
            c_dim=cs_params.get("c_dim", 128),
            query_dim=cs_params.get("query_dim", 128),
            n_layers=cs_params.get("n_layers", 3),
            n_heads=cs_params.get("n_heads", 8),
            candidate_hidden=cs_params.get("candidate_hidden", 256),
            tactical_hidden=cs_ckpt.get("tactical_hidden", 256),
            dropout=cs_ckpt.get("dropout", 0.052),
            attention_dropout=cs_ckpt.get("attention_dropout", 0.106),
        ).to(DEVICE)
        self.pick_cs_model.load_state_dict(cs_ckpt["model_state_dict"])
        self.pick_cs_model.eval()
        # 释放 checkpoint 字典，降低峰值内存
        del cs_ckpt
        # NoCS Transformer
        nocs_path = os.path.join(PICK_CKPT_DIR, "best_model_nocs.pt")
        self.pick_nocs_model = None
        if os.path.exists(nocs_path):
            nocs_ckpt = torch.load(nocs_path, map_location=DEVICE, weights_only=False)
            nocs_params = _infer_pick_params_from_state(nocs_ckpt["model_state_dict"])
            self.pick_nocs_model = BPTacticalTransformerPick(
                vocab_size=self.store.vocab_size,
                context_dim=nocs_ckpt.get("context_dim", self.pick_ctx_dim),
                candidate_dim=nocs_params.get("candidate_dim", nocs_ckpt.get("candidate_dim", CANDIDATE_DIM)),
                h_dim=nocs_params.get("h_dim", 384),
                c_dim=nocs_params.get("c_dim", 128),
                query_dim=nocs_params.get("query_dim", 128),
                n_layers=nocs_params.get("n_layers", 3),
                n_heads=nocs_params.get("n_heads", 4),
                candidate_hidden=nocs_params.get("candidate_hidden", 256),
                tactical_hidden=nocs_ckpt.get("tactical_hidden", 256),
                dropout=nocs_ckpt.get("dropout", 0.198),
                attention_dropout=nocs_ckpt.get("attention_dropout", 0.114),
            ).to(DEVICE)
            self.pick_nocs_model.load_state_dict(nocs_ckpt["model_state_dict"])
            self.pick_nocs_model.eval()
            del nocs_ckpt

        # Cascade LGB
        cascade_dir = os.path.join(PICK_CKPT_DIR, "cascade_pick")
        routing_path = os.path.join(cascade_dir, "routing_config.json")
        # 默认值从配置文件读取，保持与训练阶段一致
        _pick_cfg = get_config("pick", "cascade")
        self.pick_blend_alpha = get_production_blend_alpha(_pick_cfg)
        # 【修复 3】：支持残差训练模式
        self.pick_fusion_mode = "blend"  # 默认回退
        if os.path.exists(routing_path):
            with open(routing_path, "r") as f:
                _routing = json.load(f)
                self.pick_blend_alpha = _routing.get("blend_alpha", self.pick_blend_alpha)
                self.pick_fusion_mode = _routing.get("fusion_mode", "blend")

        self.lgb_models = []
        for i in range(5):
            p = os.path.join(cascade_dir, f"fold_{i}_model.txt")
            if os.path.exists(p):
                self.lgb_models.append(lgb.Booster(model_file=p, params={"num_threads": 1}))

        with open(os.path.join(cascade_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

    def _load_ban_models(self):
        # Ban Transformer
        ban_ckpt = torch.load(os.path.join(BAN_CKPT_DIR, "best_model_cs.pt"),
                              map_location=DEVICE, weights_only=False)
        self.ban_ctx_dim = ban_ckpt.get("context_dim", BAN_CONTEXT_DIM)
        ban_params = _infer_ban_params_from_state(ban_ckpt["model_state_dict"])
        self.ban_model = BPTacticalTransformerBan(
            vocab_size=self.store.vocab_size, context_dim=self.ban_ctx_dim,
            candidate_dim=ban_params.get("candidate_dim", ban_ckpt.get("candidate_dim", CANDIDATE_DIM)),
        ).to(DEVICE)
        self.ban_model.load_state_dict(ban_ckpt["model_state_dict"])
        self.ban_model.eval()
        del ban_ckpt

        # Ban NoCS (外部 mask CS 特征)
        ban_nocs_path = os.path.join(BAN_CKPT_DIR, "best_model_nocs.pt")
        self.ban_nocs_model = None
        if os.path.exists(ban_nocs_path):
            nocs_ckpt = torch.load(ban_nocs_path, map_location=DEVICE, weights_only=False)
            nocs_params = _infer_ban_params_from_state(nocs_ckpt["model_state_dict"])
            self.ban_nocs_model = BPTacticalTransformerBan(
                vocab_size=self.store.vocab_size,
                context_dim=nocs_ckpt.get("context_dim", self.ban_ctx_dim),
                candidate_dim=nocs_params.get("candidate_dim", nocs_ckpt.get("candidate_dim", CANDIDATE_DIM)),
            ).to(DEVICE)
            self.ban_nocs_model.load_state_dict(nocs_ckpt["model_state_dict"])
            self.ban_nocs_model.eval()
            del nocs_ckpt

        # Ban Cascade (Unified 单模型)
        ban_cascade_dir = os.path.join(BAN_CKPT_DIR, "cascade_ban")
        ban_routing_path = os.path.join(ban_cascade_dir, "routing_config.json")
        # 默认值从配置文件读取，保持与训练阶段一致
        _ban_cfg = get_config("ban", "cascade")
        self.ban_blend_alpha = get_production_blend_alpha(_ban_cfg)
        if os.path.exists(ban_routing_path):
            with open(ban_routing_path, "r") as f:
                ban_routing = json.load(f)
            self.ban_blend_alpha = ban_routing.get("blend_alpha", self.ban_blend_alpha)

        self.ban_lgb_models = []
        for i in range(5):
            p = os.path.join(ban_cascade_dir, f"fold_{i}_model.txt")
            if os.path.exists(p):
                self.ban_lgb_models.append(lgb.Booster(model_file=p, params={"num_threads": 1}))

        with open(os.path.join(ban_cascade_dir, "scaler.pkl"), "rb") as f:
            self.ban_scaler = pickle.load(f)

    @staticmethod
    def _rank_normalize(scores):
        order = np.argsort(-scores)
        ranks = np.zeros_like(scores, dtype=np.float64)
        n = len(scores)
        for rank_pos, idx in enumerate(order):
            ranks[idx] = 1.0 - rank_pos / max(n - 1, 1)
        return ranks

    @staticmethod
    def _compute_group_features(logits, mask, champion_start_idx, vocab_size):
        desc_order = np.argsort(-logits)
        rank_map = np.empty_like(logits, dtype=np.float64)
        rank_map[desc_order] = np.arange(1, len(logits) + 1, dtype=np.float64)
        valid_mask = mask > 0.5
        valid_mask[:champion_start_idx] = False
        valid_logits = logits[valid_mask]
        if valid_logits.size > 0:
            valid_mean = valid_logits.mean()
            valid_std = max(valid_logits.std(), 1e-6)
        else:
            valid_mean, valid_std = 0.0, 1.0
        return {
            "rank_map": rank_map,
            "logit_min": float(valid_logits.min()) if valid_logits.size > 0 else 0.0,
            "logit_max": float(valid_logits.max()) if valid_logits.size > 0 else 1.0,
            "logit_range": float(valid_logits.max() - valid_logits.min()) if valid_logits.size > 0 else 1e-6,
            "valid_mean": float(valid_mean),
            "valid_std": float(valid_std),
            "valid_median": float(valid_mean),
            "valid_q75": float(valid_mean + 0.675 * valid_std),
            "valid_q25": float(valid_mean - 0.675 * valid_std),
            "valid_iqr": float(1.35 * valid_std),
            "top1_logit": float(logits[desc_order[0]]),
            "top3_logit": 0.0, "top5_logit": 0.0, "top10_logit": 0.0,
        }

    def predict_pick(self, bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                     global_context, cand_np, mask_np, target_step, last_ally_pos):
        """Pick 推荐: 返回 [(champion_idx, score, rank), ...]"""
        cs = self.store.champion_start_idx

        # === 断言 1: 确保输入的候选矩阵和掩码物理长度对齐 ===
        assert cand_np.shape[0] == mask_np.shape[0] == self.store.vocab_size, \
            f"候选矩阵与掩码维度不匹配! cand: {cand_np.shape}, mask: {mask_np.shape}, vocab: {self.store.vocab_size}"

        # === 断言 2: 确保已Pick/Ban的英雄绝不会出现在有效候选集里 ===
        already_selected = set(ally_champs) | set(enemy_champs) | set(unavail_set)
        valid_mask_check = mask_np > 0.5
        valid_cids_check = np.where(valid_mask_check)[0]
        valid_cids_check = valid_cids_check[valid_cids_check >= cs]
        leaked = [cid for cid in valid_cids_check if cid in already_selected]
        assert len(leaked) == 0, f"候选集包含已选/已禁英雄! 泄露英雄IDs: {leaked}"

        # 特征监控：推理前校验特征完整性与范围
        if hasattr(self, 'feature_monitor') and self.feature_monitor is not None:
            integrity_result = self.feature_monitor.validate_feature_integrity(
                cand_np, np.asarray(global_context), mask_np,
                expected_vocab_size=self.store.vocab_size,
            )
            if not integrity_result.is_valid:
                log.warning(f"Feature integrity check failed: {integrity_result.violations}")
            range_cm = self.feature_monitor.validate_candidate_matrix(cand_np)
            if not range_cm.is_valid:
                log.warning(f"Candidate matrix range check failed: {range_cm.violations}")
            range_gc = self.feature_monitor.validate_global_context(np.asarray(global_context))
            if not range_gc.is_valid:
                log.warning(f"Global context range check failed: {range_gc.violations}")

        # 推理特征日志 (供周度 PSI 漂移分析，失败不影响主业务)
        try:
            from common.inference_feature_logger import log_recommendation_features
            import uuid as _uuid
            log_recommendation_features(
                cand_np=cand_np,
                global_context=np.asarray(global_context),
                mask_np=mask_np,
                step_type="pick",
                request_id=_uuid.uuid4().hex[:12],
            )
        except Exception:
            pass  # 埋点失败不能影响主业务

        # Transformer 推理
        bp_padded = np.array(bp_seq_ids + [self.store.PAD_IDX] * (20 - len(bp_seq_ids)), dtype=np.int64)
        bp_t = torch.as_tensor(bp_padded[np.newaxis, :], dtype=torch.long, device=DEVICE)
        ctx_t = torch.as_tensor(np.array([global_context], dtype=np.float32), device=DEVICE)
        cand_t = torch.as_tensor(np.array([cand_np], dtype=np.float32), device=DEVICE)
        mask_t = torch.as_tensor(np.array([mask_np], dtype=np.float32), device=DEVICE)
        lap_t = torch.as_tensor(np.array([last_ally_pos], dtype=np.int64), dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            cs_logits_raw = self.pick_cs_model(bp_t, ctx_t, cand_t, mask_t,
                                           last_ally_pos=lap_t)["logits"].squeeze(0).cpu().numpy()
            if self.pick_nocs_model:
                cand_nocs_t = cand_t.clone()
                cand_nocs_t[:, :, CS_FEATURE_INDICES] = 0.0
                nocs_logits_raw = self.pick_nocs_model(bp_t, ctx_t, cand_nocs_t, mask_t,
                                                   last_ally_pos=lap_t)["logits"].squeeze(0).cpu().numpy()
            else:
                nocs_logits_raw = cs_logits_raw

        # 【核心大修 1】：彻底屏蔽幽灵英雄！把 mask 为 0 的位置设为 -1e9
        cs_logits = cs_logits_raw.copy()
        cs_logits[mask_np == 0] = -1e9
        nocs_logits = nocs_logits_raw.copy()
        nocs_logits[mask_np == 0] = -1e9

        # Cascade
        valid_cids = np.where(mask_np > 0.5)[0]
        valid_cids = valid_cids[valid_cids >= cs]
        
        # 【核心大修 2】：获取真实的可用英雄总数（约 150），对齐特征量纲
        total_valid = len(valid_cids)

        top_k_limit = min(50, total_valid)
        cs_valid_logits = cs_logits[valid_cids]
        top_k_local_indices = np.argsort(-cs_valid_logits)[:top_k_limit]
        eval_cids = valid_cids[top_k_local_indices]
        total_eval = len(eval_cids)
        
        cs_gf = self._compute_group_features(cs_logits, mask_np, cs, self.store.vocab_size)
        nocs_gf = self._compute_group_features(nocs_logits, mask_np, cs, self.store.vocab_size)

        X_arr = _build_feature_matrix_batch(
            cs_logits[eval_cids], cs_gf["rank_map"][eval_cids], cs_gf,
            nocs_logits[eval_cids], nocs_gf["rank_map"][eval_cids], nocs_gf,
            cand_np[eval_cids], total_valid, total_eval, target_step,
        )

        # === 断言 3: 确保精排输入的样本数与粗排Logit的行数绝对一致 ===
        assert len(X_arr) == len(eval_cids) == total_eval, \
            f"粗排Logit与精排特征行数错位! X_arr: {len(X_arr)}, eval_cids: {len(eval_cids)}, total_eval: {total_eval}"
        assert X_arr.shape[1] == len(FEATURE_COLS), \
            f"精排特征维度与FEATURE_COLS不匹配! X_arr: {X_arr.shape[1]}, FEATURE_COLS: {len(FEATURE_COLS)}"

        lgb_preds = np.zeros(total_eval, dtype=np.float64)
        X_scaled = self.scaler.transform(X_arr)
        for m in self.lgb_models:
            lgb_preds += m.predict(X_scaled)
        lgb_preds /= max(len(self.lgb_models), 1)

        # 【修复 3】：残差模式下，最终分数 = LGBM 残差 + TF base logits
        # 模型训练时 init_score=base_cs，所以预测时需要加回 base_cs
        if getattr(self, "pick_fusion_mode", "blend") == "residual_init_score":
            blend_scores = lgb_preds + cs_logits[eval_cids]
        else:
            cs_rn = self._rank_normalize(cs_logits[eval_cids])
            lgb_rn = self._rank_normalize(lgb_preds)
            blend_scores = self.pick_blend_alpha * cs_rn + (1.0 - self.pick_blend_alpha) * lgb_rn
        
        # 【核心大修 3】：已彻底删除 blend_scores -= 0.5 的外挂位置惩罚，交由模型自行决策。

        # 排序输出
        sorted_idx = np.argsort(-blend_scores)
        results = []
        # for rank, si in enumerate(sorted_idx[:20]):  comment for alignment test
        for rank, si in enumerate(sorted_idx):
            cid = eval_cids[si]
            results.append((cid, float(blend_scores[si]), rank + 1))
            
        return results
    

    def predict_ban(self, bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                    global_context, cand_np, mask_np, target_step):
        """Ban 推荐: 返回 [(champion_idx, score, rank), ...]"""
        cs = self.store.champion_start_idx

        # === 断言 1: 确保输入的候选矩阵和掩码物理长度对齐 ===
        assert cand_np.shape[0] == mask_np.shape[0] == self.store.vocab_size, \
            f"[Ban] 候选矩阵与掩码维度不匹配! cand: {cand_np.shape}, mask: {mask_np.shape}, vocab: {self.store.vocab_size}"

        # === 断言 2: 确保已Pick/Ban的英雄绝不会出现在有效候选集里 ===
        already_selected = set(ally_champs) | set(enemy_champs) | set(unavail_set)
        valid_mask_check = mask_np > 0.5
        valid_cids_check = np.where(valid_mask_check)[0]
        valid_cids_check = valid_cids_check[valid_cids_check >= cs]
        leaked = [cid for cid in valid_cids_check if cid in already_selected]
        assert len(leaked) == 0, f"[Ban] 候选集包含已选/已禁英雄! 泄露英雄IDs: {leaked}"

        # 特征监控：推理前校验特征完整性与范围
        if hasattr(self, 'feature_monitor') and self.feature_monitor is not None:
            integrity_result = self.feature_monitor.validate_feature_integrity(
                cand_np, np.asarray(global_context), mask_np,
                expected_vocab_size=self.store.vocab_size,
            )
            if not integrity_result.is_valid:
                log.warning(f"[Ban] Feature integrity check failed: {integrity_result.violations}")
            range_cm = self.feature_monitor.validate_candidate_matrix(cand_np)
            if not range_cm.is_valid:
                log.warning(f"[Ban] Candidate matrix range check failed: {range_cm.violations}")

        # 推理特征日志 (供周度 PSI 漂移分析，失败不影响主业务)
        try:
            from common.inference_feature_logger import log_recommendation_features
            import uuid as _uuid
            log_recommendation_features(
                cand_np=cand_np,
                global_context=np.asarray(global_context),
                mask_np=mask_np,
                step_type="ban",
                request_id=_uuid.uuid4().hex[:12],
            )
        except Exception:
            pass  # 埋点失败不能影响主业务

        # Transformer 推理
        bp_padded = np.array(bp_seq_ids + [self.store.PAD_IDX] * (20 - len(bp_seq_ids)), dtype=np.int64)
        bp_t = torch.as_tensor(bp_padded[np.newaxis, :], dtype=torch.long, device=DEVICE)
        ctx_t = torch.as_tensor(np.array([global_context], dtype=np.float32), device=DEVICE)
        cand_t = torch.as_tensor(np.array([cand_np], dtype=np.float32), device=DEVICE)
        mask_t = torch.as_tensor(np.array([mask_np], dtype=np.float32), device=DEVICE)

        hist_pos = np.full(20, -1, dtype=np.int64)
        for i in range(min(len(bp_seq_ids), 20)):
            cid = bp_seq_ids[i]
            if cid >= cs and BP_SEQUENCE[i][0] == "pick":
                hist_pos[i] = int(np.argmax(self.store.pos_prior[cid]))
        hist_t = torch.as_tensor(np.array([hist_pos], dtype=np.int64), dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            cs_logits_raw = self.ban_model(bp_t, ctx_t, cand_t, mask_t,
                                       history_positions=hist_t)["logits"].squeeze(0).cpu().numpy()

        # 【核心大修 1】：屏蔽幽灵英雄
        cs_logits = cs_logits_raw.copy()
        cs_logits[mask_np == 0] = -1e9

        # Cascade (Unified 单模型)
        valid_cids = np.where(mask_np > 0.5)[0]
        valid_cids = valid_cids[valid_cids >= cs]
        
        # 【核心大修 2】：传入真实的英雄总数对齐量纲
        total_valid = len(valid_cids)

        top_k_limit = min(50, total_valid)
        cs_valid_logits = cs_logits[valid_cids]
        top_k_local_indices = np.argsort(-cs_valid_logits)[:top_k_limit]
        eval_cids = valid_cids[top_k_local_indices]
        total_eval = len(eval_cids)

        cs_gf = _compute_ban_group_features(cs_logits, mask_np, cs, self.store.vocab_size)

        X_arr = _build_ban_feature_matrix_batch(
            cs_logits[eval_cids], cs_gf["rank_map"][eval_cids], cs_gf,
            cand_np[eval_cids], total_valid, 
        )

        # === 断言 3: 确保精排输入的样本数与粗排Logit的行数绝对一致 ===
        assert len(X_arr) == len(eval_cids) == total_eval, \
            f"[Ban] 粗排Logit与精排特征行数错位! X_arr: {len(X_arr)}, eval_cids: {len(eval_cids)}, total_eval: {total_eval}"
        assert X_arr.shape[1] == len(BAN_FEATURE_COLS), \
            f"[Ban] 精排特征维度与BAN_FEATURE_COLS不匹配! X_arr: {X_arr.shape[1]}, BAN_FEATURE_COLS: {len(BAN_FEATURE_COLS)}"

        lgb_preds = np.zeros(total_eval, dtype=np.float64)
        X_scaled = self.ban_scaler.transform(X_arr)
        for m in self.ban_lgb_models:
            lgb_preds += m.predict(X_scaled)
        lgb_preds /= max(len(self.ban_lgb_models), 1)

        base_rn = self._rank_normalize(cs_logits[eval_cids])
        lgb_rn = self._rank_normalize(lgb_preds)
        final_scores = self.ban_blend_alpha * base_rn + (1.0 - self.ban_blend_alpha) * lgb_rn

        sorted_idx = np.argsort(-final_scores)
        results = []
        # for rank, si in enumerate(sorted_idx[:20]):  comment for alignment test
        for rank, si in enumerate(sorted_idx):
            cid = eval_cids[si]
            results.append((cid, float(final_scores[si]), rank + 1))
        return results


# ==================== 交互式界面 ====================

def print_bp_board(completed_steps, bp_seq_ids, store):
    log.info("")
    log.info("=" * 60)
    log.info("  当前 BP 面板")
    log.info("=" * 60)

    blue_bans = []
    red_bans = []
    blue_picks = []
    red_picks = []
    for i in range(completed_steps):
        action, side, slot = BP_SEQUENCE[i]
        cid = bp_seq_ids[i] if i < len(bp_seq_ids) else store.UNK_IDX
        name = store.idx_to_name.get(str(cid), "???") if cid >= store.champion_start_idx else "---"
        if action == "ban":
            if side == "blue":
                blue_bans.append(name)
            else:
                red_bans.append(name)
        else:
            if side == "blue":
                blue_picks.append(name)
            else:
                red_picks.append(name)

    log.info(f"  Blue Bans:  {', '.join(blue_bans) if blue_bans else '(none)'}")
    log.info(f"  Red Bans:   {', '.join(red_bans) if red_bans else '(none)'}")
    log.info(f"  Blue Picks: {', '.join(blue_picks) if blue_picks else '(none)'}")
    log.info(f"  Red Picks:  {', '.join(red_picks) if red_picks else '(none)'}")
    log.info("=" * 60)


def interactive_predict():
    recommender = BPRecommender()
    store = recommender.store

    name_lookup, vocab_data = build_name_lookup(VOCAB_PATH)
    idx_to_name = {}
    for champ in vocab_data["champions"]:
        idx_to_name[champ["idx"]] = champ["name"]

    log.info("")
    log.info("=" * 60)
    log.info("  LOL BP 实时推荐系统")
    log.info("  输入英雄英文名/中文名/Riot ID 进行 Ban/Pick")
    log.info("=" * 60)

    log.info("")
    log.info("--- 选择模式 ---")
    log.info("  1) 纯 Draft 模式 (不输入战队/选手信息)")
    log.info("  2) 完整模式 (输入双方战队 + 选手信息)")
    while True:
        mode = input("  请选择 (1/2): ").strip()
        if mode in ("1", "2"):
            break
        log.info("  无效输入，请输入 1 或 2")

    is_full_mode = (mode == "2")

    log.info("")
    log.info("--- 赛前信息 ---")
    league = input(f"  联赛 ({'/'.join(LEAGUES)}), 默认LPL: ").strip() or "LPL"
    playoffs_str = input("  是否季后赛 (y/n, 默认n): ").strip().lower()
    playoffs = 1.0 if playoffs_str == "y" else 0.0
    fp_str = input("  先选方 (blue/red, 默认blue): ").strip().lower()
    first_pick = 1.0 if fp_str == "blue" else 0.0

    blue_team = ""
    red_team = ""
    blue_pids = [""] * 5
    red_pids = [""] * 5

    if is_full_mode:
        log.info("")
        log.info("--- 战队信息 ---")
        while True:
            blue_team = input("  蓝方队伍名: ").strip()
            if blue_team:
                break
            log.info("  完整模式必须输入蓝方队伍名!")

        while True:
            red_team = input("  红方队伍名: ").strip()
            if red_team:
                break
            log.info("  完整模式必须输入红方队伍名!")

        known_teams = sorted(store.team_players_map.keys())
        if known_teams:
            log.info(f"\n  已知战队 ({len(known_teams)}): {', '.join(known_teams[:10])}{'...' if len(known_teams) > 10 else ''}")

        for team_name, side_cn in [(blue_team, "蓝方"), (red_team, "红方")]:
            if team_name not in store.team_players_map:
                log.info(f"  ⚠ {side_cn}队伍 '{team_name}' 不在已知战队列表中，选手特征将使用默认值")

        positions = ["top", "jng", "mid", "bot", "sup"]
        log.info("")
        log.info("--- 选手信息 ---")
        log.info("  输入选手 ID，未知选手请输入 'unknown'")
        log.info("  每队最多允许 2 名 unknown 选手")
        log.info("")

        for side, side_cn, team_name in [("blue", "蓝方", blue_team), ("red", "红方", red_team)]:
            known_pids = store.team_players_map.get(team_name, set())
            if known_pids:
                log.info(f"  {side_cn} ({team_name}) 已知选手: {', '.join(sorted(known_pids)[:10])}{'...' if len(known_pids) > 10 else ''}")

            pids = []
            unknown_count = 0
            for pos in positions:
                while True:
                    pid = input(f"  {side_cn} {pos}: ").strip()
                    if not pid:
                        log.info("  完整模式必须输入选手 ID (未知选手请输入 'unknown')")
                        continue
                    if pid.lower() == "unknown":
                        if unknown_count >= 2:
                            log.info(f"  ⚠ 每队最多 2 名 unknown 选手! 已有 {unknown_count} 名，请输入已知选手 ID")
                            continue
                        unknown_count += 1
                        pids.append("unknown")
                        break
                    else:
                        if pid not in store.player_matrix_map:
                            close = [p for p in store.player_matrix_map if p.lower().startswith(pid.lower())]
                            if len(close) == 1:
                                log.info(f"  自动匹配: {close[0]}")
                                pids.append(close[0])
                                break
                            elif len(close) > 1:
                                log.info(f"  多个匹配: {', '.join(close[:5])}，请更精确输入")
                                continue
                            else:
                                log.info(f"  ⚠ 选手 '{pid}' 不在数据库中，将使用战队平均特征")
                                if unknown_count >= 2:
                                    log.info(f"  ⚠ 每队最多 2 名 unknown 选手! 请输入已知选手 ID")
                                    continue
                                unknown_count += 1
                                pids.append("unknown")
                                break
                        else:
                            pids.append(pid)
                            break

            if side == "blue":
                blue_pids = pids
            else:
                red_pids = pids

    # ---- 构建 Global Context (对齐 20 维) ----
    league_vec = np.zeros(len(LEAGUES), dtype=np.float32)
    if league in LEAGUES:
        league_vec[LEAGUES.index(league)] = 1.0
    b_style = store.team_style_dict.get(blue_team, [0.7, 0.0, 1900.0, 0.5, 0.5])
    r_style = store.team_style_dict.get(red_team, [0.7, 0.0, 1900.0, 0.5, 0.5])
    team_style = np.array(b_style + r_style, dtype=np.float32)
    
    # 增加 5 维的局数 One-Hot，由于是在线推荐预测第一局，将 is_game_1 设为 1
    game_number_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    global_context = np.concatenate([league_vec, team_style, [playoffs, first_pick], game_number_vec])

    # ---- BP 状态 ----
    bp_seq_ids = []
    completed_steps = 0
    unavail_set = set()

    # 纯 Draft 模式下使用空 pids
    if not is_full_mode:
        blue_pids = [""] * 5
        red_pids = [""] * 5

    mode_desc = "完整模式" if is_full_mode else "纯 Draft 模式"
    log.info("")
    log.info(f"--- 开始 BP ({mode_desc}) ---")
    log.info("  每步输入英雄名称后回车，输入 'q' 退出，输入 'skip' 跳过当前步")
    log.info("  输入 'undo' 撤销上一步")
    log.info("")

    while completed_steps < 20:
        step = completed_steps
        action, side, slot = BP_SEQUENCE[step]
        action_cn = "Ban" if action == "ban" else "Pick"
        side_cn = "蓝方" if side == "blue" else "红方"

        print_bp_board(completed_steps, bp_seq_ids, store)
        log.info("")
        log.info(f"  >>> 第 {step+1}/20 步: {side_cn} {action_cn}{slot}")

        curr_side_code = 0 if side == "blue" else 1
        ally_champs = []
        enemy_champs = []
        last_ally_pos = -1
        for i in range(len(bp_seq_ids)):
            cid = bp_seq_ids[i]
            if cid < store.champion_start_idx:
                continue
            if BP_SEQUENCE[i][0] == "pick":
                if (BP_SEQUENCE[i][1] == "blue" and curr_side_code == 0) or \
                   (BP_SEQUENCE[i][1] == "red" and curr_side_code == 1):
                    ally_champs.append(cid)
                    last_ally_pos = i
                else:
                    enemy_champs.append(cid)

        ally_pids = blue_pids if side == "blue" else red_pids
        enemy_pids = red_pids if side == "blue" else blue_pids
        team_name = blue_team if side == "blue" else red_team
        opp_team = red_team if side == "blue" else blue_team

        if action == "pick":
            cand_np, mask_np = store.get_pick_candidate_matrix(
                side, ally_champs, enemy_champs, unavail_set,
                ally_pids, enemy_pids, step, team_name, opp_team, pre_unavail_list=None
            )
            results = recommender.predict_pick(
                bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                global_context, cand_np, mask_np, step, last_ally_pos,
            )
        else:
            cand_np, mask_np = store.get_ban_candidate_matrix(
                side, ally_champs, enemy_champs, unavail_set,
                ally_pids, enemy_pids, step, team_name, opp_team, pre_unavail_list=None
            )
            results = recommender.predict_ban(
                bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                global_context, cand_np, mask_np, step,
            )

        log.info("")
        log.info(f"  Top-20 {action_cn} 推荐:")
        log.info(f"  {'Rank':>4}  {'Champion':<16}  {'Score':>8}")
        log.info(f"  {'----':>4}  {'--------':<16}  {'-----':>8}")
        for cid, score, rank in results[:20]:
            name = idx_to_name.get(cid, "???")
            marker = " <<<" if rank <= 3 else ""
            log.info(f"  {rank:>4}  {name:<16}  {score:>8.4f}{marker}")

        while True:
            user_input = input(f"\n  输入 {side_cn}{action_cn}{slot} 的英雄 (或 q/undo): ").strip()
            if user_input.lower() == "q":
                log.info("  退出。")
                return
            if user_input.lower() == "undo":
                if bp_seq_ids:
                    removed = bp_seq_ids.pop()
                    unavail_set.discard(removed)
                    completed_steps -= 1
                    log.info("  已撤销上一步。")
                else:
                    log.info("  没有可撤销的步骤。")
                break
            if user_input.lower() == "skip":
                bp_seq_ids.append(store.UNK_IDX)
                completed_steps += 1
                log.info("  已跳过。")
                break

            cid = resolve_champion(user_input, name_lookup, idx_to_name, store.champion_start_idx)
            if cid is None:
                key = user_input.lower()
                matches = sorted(set(k for k in name_lookup if k.startswith(key)))[:5]
                if matches:
                    log.info(f"  未找到 '{user_input}'，相似名称: {', '.join(matches)}")
                else:
                    log.info(f"  未找到英雄 '{user_input}'，请重新输入。")
                continue
            if cid in unavail_set:
                log.info(f"  {idx_to_name.get(cid, '???')} 已被 Ban/Pick，请选择其他英雄。")
                continue
            if cid < store.champion_start_idx:
                log.info(f"  无效英雄 ID，请重新输入。")
                continue

            bp_seq_ids.append(cid)
            unavail_set.add(cid)
            completed_steps += 1
            log.info(f"  已选择: {idx_to_name.get(cid, '???')}")
            break

    log.info("")
    log.info("=" * 60)
    log.info("  BP 结束!")
    print_bp_board(20, bp_seq_ids, store)


if __name__ == "__main__":
    setup_logging()
    interactive_predict()
