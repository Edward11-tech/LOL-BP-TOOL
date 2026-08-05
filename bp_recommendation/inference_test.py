"""
inference_test.py — Pick/Ban 模型推理测试

支持模式:
  --mode pick   : 仅测试 Pick 模型
  --mode ban    : 仅测试 Ban 模型
  --mode both   : 同时测试 Pick 和 Ban 模型 (默认)

核心优化策略:
1. 预计算 Numpy 矩阵: meta/player/grudge/respect/streak 全部转为连续内存数组
2. 向量化 get_candidate_matrix: 消除 list comprehension + dict lookup
3. 预分配 GPU Buffer: 避免 torch.tensor() 重复分配
4. 双模型并行推理: CS + No-CS 合并为单次 batch forward (已修复并优化逻辑)
5. 轻量级 _compute_group_features: 避免 np.percentile 等重计算
6. LightGBM 预测优化: 预构建 dataset 减少开销
"""
import os
# ⚠️ macOS workaround: torch + lightgbm OpenMP conflict
#   - OMP_NUM_THREADS=1 prevents the deadlock (lgb after torch) and segfault (torch after lgb)
#   - torch.set_num_threads(4) would be ignored in this config; OMP_NUM_THREADS=1 is respected
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import json
import time
import logging
import argparse
import torch
import lightgbm as lgb
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import faulthandler
faulthandler.enable()

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, TEST_DIR)

from logger_config import get_logger, setup_logging
from bp_recommendation.model_pick.model_pick import BPTacticalTransformerPick
from bp_recommendation.model_ban.model_ban import BPTacticalTransformer as BPTacticalTransformerBan
from bp_recommendation.model_ban.dataloader_ban import BAN_CANDIDATE_DIM, EXTENDED_CANDIDATE_DIM, BAN_CONTEXT_DIM
from bp_recommendation.feature_pipeline import load_champion_vocabulary, BP_SEQUENCE
from bp_recommendation.model_pick.cascade_pick import _build_feature_matrix_batch, FEATURE_COLS, FEAT_IDX
from bp_recommendation.model_pick.train_pick import CS_FEATURE_INDICES
from bp_recommendation.model_ban.cascade_ban import (
    _build_feature_matrix_batch as _build_ban_feature_matrix_batch,
    _compute_group_features as _compute_ban_group_features,
    FEATURE_COLS as BAN_FEATURE_COLS,
)

# ----------------- 路径配置 -----------------
PICK_DIR = os.path.join(TEST_DIR, "model_pick")
BAN_DIR = os.path.join(TEST_DIR, "model_ban")
TEST_CSV = os.path.join(BASE_DIR, "testdata", "matches_test2_with_game.csv")
TRAIN_FEATURES_DIR = os.path.join(TEST_DIR, "features")
PICK_CKPT_DIR = os.path.join(PICK_DIR, "checkpoints")
BAN_CKPT_DIR = os.path.join(BAN_DIR, "checkpoints")
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")

VOCAB_PATH = os.path.join(CLEANED_DIR, "champion_vocabulary.json")
POS_JSON = os.path.join(CLEANED_DIR, "champion_position_mapping.json")
BLEND_ALPHA = 0.38
DEVICE = torch.device("cpu")  # CPU 推理更快: 避免 MPS kernel launch overhead

POS_2_IDX = {"top": 0, "jungle": 1, "mid": 2, "bot": 3, "support": 4}
LEAGUES = ["LPL", "LCK", "LEC"]

LOG_DIR = os.path.join(TEST_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log = get_logger(__name__)


def _get_champion_from_raw_row(row, step):
    action_type, side, slot = BP_SEQUENCE[step]
    col_prefix = f"{side}_ban" if action_type == "ban" else f"{side}_pick"
    col_name = f"{col_prefix}{slot}"
    val = row.get(col_name, "")
    if pd.isna(val):
        return ""
    return str(val).strip()


# ==================== 预计算索引 ====================
BP_ACTION_ARR = np.array([0 if s[0] == "ban" else 1 for s in BP_SEQUENCE], dtype=np.int8)
BP_SIDE_ARR = np.array([0 if s[1] == "blue" else 1 for s in BP_SEQUENCE], dtype=np.int8)
PICK_STEP_INDICES = [i for i, (a, _, _) in enumerate(BP_SEQUENCE) if a == "pick"]
BAN_STEP_INDICES = [i for i, (a, _, _) in enumerate(BP_SEQUENCE) if a == "ban"]
TUPLE_START_STEPS = {7, 9, 17}


class FastOOTFeatureStore:
    """优化版特征存储: 所有查找表预转为连续 Numpy 数组"""

    def __init__(self):
        log.info("[1/5] Building Feature Store Snapshots (Optimized)...")

        def load_json(name):
            path = os.path.join(TRAIN_FEATURES_DIR, name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            if name.endswith("_counter_lookup.json"):
                log.warning(f"⚠️  {name} not found at {path}! Counter features will use default value (0.5).")
            elif name.endswith("_synergy_lookup.json"):
                log.warning(f"⚠️  {name} not found at {path}! Synergy features will use default value (0.5).")
            elif name.endswith("_grudge_store.json"):
                log.warning(f"⚠️  {name} not found at {path}! Grudge features will be empty.")
            elif name.endswith("_respect_store.json"):
                log.warning(f"⚠️  {name} not found at {path}! Respect features will be empty.")
            elif name.endswith("_hot_streak_store.json"):
                log.warning(f"⚠️  {name} not found at {path}! Hot streak features will be empty.")
            return {}

        self.counter_dict = load_json("ALL_counter_lookup.json")
        self.synergy_dict = load_json("ALL_synergy_lookup.json")

        context_df = pd.read_parquet(os.path.join(TRAIN_FEATURES_DIR, "ALL_context.parquet"))
        context_df = context_df.sort_values("match_seq_idx").reset_index(drop=True)
        gid2seq = dict(zip(context_df["gameid"].astype(str), context_df["match_seq_idx"]))
        max_seq = context_df["match_seq_idx"].max()

        raw_grudge = load_json("ALL_grudge_store.json")
        raw_respect = load_json("ALL_respect_store.json")
        raw_hot_streak = load_json("ALL_hot_streak_store.json")

        GRUDGE_HALF_LIFE = 500
        grudge_entries = defaultdict(list)
        for game_id, team_data in raw_grudge.items():
            seq = gid2seq.get(game_id, -1)
            if seq < 0: continue
            for team_a, opp_data in team_data.items():
                for team_b, champ_stats in opp_data.items():
                    for cid, val in champ_stats.items():
                        grudge_entries[(team_a, team_b, cid)].append((seq, float(val)))

        self.online_grudge = defaultdict(lambda: defaultdict(dict))
        for (ta, tb, cid), entries in grudge_entries.items():
            entries.sort(key=lambda x: x[0])
            w_sum, wv_sum = 0.0, 0.0
            for seq, val in entries:
                w = 2.0 ** (-(max_seq - seq) / GRUDGE_HALF_LIFE)
                wv_sum += w * val
                w_sum += w
            self.online_grudge[ta][tb][cid] = wv_sum / w_sum if w_sum > 0 else 0.0

        respect_by_player = defaultdict(list)
        for game_id, player_data in raw_respect.items():
            seq = gid2seq.get(game_id, -1)
            if seq < 0: continue
            for pid, rinfo in player_data.items():
                respect_by_player[str(pid)].append((seq, rinfo))

        self.online_respect = {}
        for pid, entries in respect_by_player.items():
            entries.sort(key=lambda x: x[0])
            self.online_respect[pid] = entries[-1][1]

        streak_by_player = defaultdict(list)
        for game_id, player_data in raw_hot_streak.items():
            seq = gid2seq.get(game_id, -1)
            if seq < 0: continue
            for pid, hs_info in player_data.items():
                streak_by_player[str(pid)].append((seq, hs_info))

        self.online_hot_streak = {}
        for pid, entries in streak_by_player.items():
            entries.sort(key=lambda x: x[0])
            self.online_hot_streak[pid] = entries[-1][1]

        self.name_to_idx, self.idx_to_name, self.vocab_size, self.special_tokens, self.champion_start_idx = \
            load_champion_vocabulary(VOCAB_PATH)
        self.PAD_IDX = self.special_tokens["PAD"]
        self.UNK_IDX = self.special_tokens["UNK"]
        cs = self.champion_start_idx
        ve = self.vocab_size
        self.n_champs = ve - cs 

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

        self.meta_matrix = np.zeros((self.vocab_size, 4), dtype=np.float32)
        self.meta_matrix[:, 3] = 0.5 
        if os.path.exists(os.path.join(TRAIN_FEATURES_DIR, "ALL_meta_store.parquet")):
            meta_df = pd.read_parquet(os.path.join(TRAIN_FEATURES_DIR, "ALL_meta_store.parquet"))
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

        self.player_snapshot = defaultdict(dict)
        self.player_matrix_map = {} 
        # 维度顺序: [mastery, recent_kda, recent_wr, recent_games, overall_kda, overall_wr, overall_games]
        self.default_player_vec = np.array([0.0, 3.0, 0.5, 0.0, 3.0, 0.5, 0.0], dtype=np.float32)
        if os.path.exists(os.path.join(TRAIN_FEATURES_DIR, "ALL_player_store.parquet")):
            player_df = pd.read_parquet(os.path.join(TRAIN_FEATURES_DIR, "ALL_player_store.parquet"))
            latest_player = player_df.drop_duplicates(subset=["player_id", "champion_id"], keep="last")
            for _, row in latest_player.iterrows():
                pid = str(row["player_id"])
                cid = int(row["champion_id"])
                self.player_snapshot[pid][cid] = [
                    row["mastery_score"],
                    row["player_recent_kda_90d"] if pd.notna(row["player_recent_kda_90d"]) else 3.0,
                    row["player_recent_wr_90d"] if pd.notna(row["player_recent_wr_90d"]) else 0.5,
                    row.get("player_recent_games_90d", 0.0) if pd.notna(row.get("player_recent_games_90d", 0.0)) else 0.0,
                    row.get("player_overall_recent_kda", 3.0) if pd.notna(row.get("player_overall_recent_kda", 3.0)) else 3.0,
                    row.get("player_overall_recent_wr", 0.5) if pd.notna(row.get("player_overall_recent_wr", 0.5)) else 0.5,
                    row.get("player_overall_recent_games", 0.0) if pd.notna(row.get("player_overall_recent_games", 0.0)) else 0.0,
                ]
            log.info(f"  Pre-building player matrices for {len(self.player_snapshot)} players...")
            for pid, champ_dict in self.player_snapshot.items():
                mat = np.tile(self.default_player_vec, (self.vocab_size, 1))
                for cid, feats in champ_dict.items():
                    if 0 <= cid < self.vocab_size:
                        mat[cid] = feats
                self.player_matrix_map[pid] = mat

        self.team_style_dict = {}
        blue_latest = context_df.sort_values("match_seq_idx").drop_duplicates(subset=["blue_team"], keep="last")
        red_latest = context_df.sort_values("match_seq_idx").drop_duplicates(subset=["red_team"], keep="last")
        for _, r in blue_latest.iterrows():
            self.team_style_dict[r["blue_team"]] = [r.get("blue_team_avg_ckpm", 0.7), r.get("blue_team_avg_golddiffat15", 0), r.get("blue_team_avg_gamelength", 1900), r.get("blue_team_firstdragon_rate", 0.5), r.get("blue_team_firsttower_rate", 0.5)]
        for _, r in red_latest.iterrows():
            if r["red_team"] not in self.team_style_dict:
                self.team_style_dict[r["red_team"]] = [r.get("red_team_avg_ckpm", 0.7), r.get("red_team_avg_golddiffat15", 0), r.get("red_team_avg_gamelength", 1900), r.get("red_team_firstdragon_rate", 0.5), r.get("red_team_firsttower_rate", 0.5)]

        self._build_cs_matrices()
        self.champ_range = np.arange(cs, ve)
        self.grudge_matrix_map = {}
        for ta, opp_data in self.online_grudge.items():
            for tb, champ_dict in opp_data.items():
                mat = np.zeros(self.vocab_size, dtype=np.float32)
                for cid_str, val in champ_dict.items():
                    cid = int(cid_str)
                    if 0 <= cid < self.vocab_size:
                        mat[cid] = val
                self.grudge_matrix_map[(ta, tb)] = mat

        log.info("  Feature Store built successfully.")

    def _build_cs_matrices(self):
        self.syn_mat = np.full((self.vocab_size, self.vocab_size), 0.5, dtype=np.float32)
        self.ctr_mat = np.full((self.vocab_size, self.vocab_size), 0.5, dtype=np.float32)
        for k, wr in self.synergy_dict.items():
            parts = k.split("||")
            if len(parts) == 2:
                c1, c2 = self.name_to_idx.get(parts[0], -1), self.name_to_idx.get(parts[1], -1)
                if c1 >= 0 and c2 >= 0:
                    self.syn_mat[c1, c2] = self.syn_mat[c2, c1] = float(wr)
        for c_name, opps in self.counter_dict.items():
            c1 = self.name_to_idx.get(c_name, -1)
            if c1 < 0: continue
            for opp_name, stats in opps.items():
                c2 = self.name_to_idx.get(opp_name, -1)
                if c2 >= 0:
                    self.ctr_mat[c1, c2] = float(stats.get("win_rate", 0.5))

    def _map_champ(self, name):
        # empty_ban 在训练侧已归并到 UNK（vocab v2 移除 EMPTY_BAN，统一回退到 UNK_IDX）
        if not name or name.lower() == "nan" or name == "<EMPTY_BAN>":
            return self.UNK_IDX
        return self.name_to_idx.get(name, self.UNK_IDX)

    def get_candidate_matrix_fast(self, row, target_step, ally_champs, enemy_champs, unavail_set):
        cs = self.champion_start_idx
        ve = self.vocab_size
        n_champs = self.n_champs
        current_action = BP_SEQUENCE[target_step]
        side_str = current_action[1]
        enemy_side_str = "red" if side_str == "blue" else "blue"
        
        cand = np.zeros((self.vocab_size, 33), dtype=np.float32)  # CANDIDATE_DIM=33
        cand[:, 0:4] = self.meta_matrix
        cand[:, 11:16] = self.pos_prior

        ally_pids = [str(row.get(f"{side_str}_{p}_player_id", "")).strip() for p in ["top", "jng", "mid", "bot", "sup"]]
        ally_feat_mat = np.zeros((5, n_champs, 7), dtype=np.float32)
        for i, pid in enumerate(ally_pids):
            pmat = self.player_matrix_map.get(pid)
            if pmat is not None:
                ally_feat_mat[i] = pmat[cs:ve]

        ally_pos_sum = np.zeros(5, dtype=np.float32)
        for c in ally_champs: ally_pos_sum += self.pos_prior[c]
        enemy_pos_sum = np.zeros(5, dtype=np.float32)
        for c in enemy_champs: enemy_pos_sum += self.pos_prior[c]
        
        ally_missing_roles = np.clip(1.0 - ally_pos_sum, 0.0, 1.0)
        enemy_missing_roles = np.clip(1.0 - enemy_pos_sum, 0.0, 1.0)

        # 切片 4:11 包含全部 7 个 player 特征 (含 recent_games@7)
        cand[cs:ve, 4:11] = (ally_feat_mat * ally_missing_roles[:, None, None]).max(axis=0)

        if ally_champs:
            ally_arr = np.array(ally_champs, dtype=np.int64)
            cand[cs:ve, 16] = np.max(self.syn_mat[cs:ve, ally_arr], axis=1)
            cand[cs:ve, 19] = np.max(1.0 - self.ctr_mat[cs:ve, ally_arr], axis=1)
        if enemy_champs:
            enemy_arr = np.array(enemy_champs, dtype=np.int64)
            cand[cs:ve, 18] = np.max(1.0 - self.ctr_mat[cs:ve, enemy_arr], axis=1)
            cand[cs:ve, 17] = np.max(self.syn_mat[cs:ve, enemy_arr], axis=1)

        pos_block = cand[cs:ve, 11:16]
        cand[cs:ve, 20] = pos_block @ ally_missing_roles
        cand[cs:ve, 21] = pos_block @ enemy_missing_roles
        cand[cs:ve, 22] = 1.0

        enemy_pids = [str(row.get(f"{enemy_side_str}_{p}_player_id", "")).strip() for p in ["top", "jng", "mid", "bot", "sup"]]
        enemy_mastery_matrix = np.zeros((5, n_champs), dtype=np.float32)
        for i, epid in enumerate(enemy_pids):
            pmat = self.player_matrix_map.get(epid)
            if pmat is not None:
                enemy_mastery_matrix[i] = pmat[cs:ve, 0] 
        weighted_enemy_mastery = enemy_mastery_matrix * enemy_missing_roles[:, None]
        cand[cs:ve, 23] = weighted_enemy_mastery.max(axis=0)
        cand[cs:ve, 24] = weighted_enemy_mastery.mean(axis=0)

        cand[cs:ve, 25] = sum(1 for i in range(target_step + 1) if BP_SEQUENCE[i][0] == "ban" and BP_SEQUENCE[i][1] == side_str)

        banning_team = str(row.get(f"{side_str}_team", "")).strip()
        opponent_team = str(row.get(f"{enemy_side_str}_team", "")).strip()
        grudge_vec = self.grudge_matrix_map.get((banning_team, opponent_team))
        if grudge_vec is not None:
            cand[cs:ve, 26] = grudge_vec[cs:ve]

        enemy_respect_vec = np.zeros(self.vocab_size, dtype=np.float32)
        enemy_streak_vec = np.zeros(self.vocab_size, dtype=np.float32)
        for epid in enemy_pids:
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
        cand[cs:ve, 27] = enemy_respect_vec[cs:ve]
        cand[cs:ve, 28] = enemy_streak_vec[cs:ve]

        cand[cs:ve, 29] = float(len(ally_champs))
        cand[cs:ve, 30] = 0.0 if side_str == "blue" else 1.0
        if ally_champs:
            cand[cs:ve, 31] = self.syn_mat[cs:ve, ally_champs[-1]]
        else:
            cand[cs:ve, 31] = 0.5

        # [32] is_fearless_banned: 标记前置局已使用/全局不可选英雄
        # 1) 从 prev_game_champs 获取前置局已使用英雄
        prev_champ_names = str(row.get("prev_game_champs", ""))
        if prev_champ_names and prev_champ_names.strip():
            for name in prev_champ_names.split("|"):
                cid = self.name_to_idx.get(name.strip(), -1)
                if cid >= cs:
                    cand[cid, 32] = 1.0
        # 2) 推理阶段 unavail_set 中的英雄（前端传入的不可选列表）
        for uid in unavail_set:
            if cs <= uid < ve:
                cand[uid, 32] = 1.0

        mask = np.ones(self.vocab_size, dtype=np.float32)
        mask[:cs] = 0.0
        for uid in unavail_set:
            if 0 <= uid < self.vocab_size:
                mask[uid] = 0.0

        return cand, mask

    def get_candidate_matrix_ban_fast(self, row, target_step, ally_champs, enemy_champs, unavail_set):
        cs = self.champion_start_idx
        ve = self.vocab_size
        n_champs = self.n_champs
        current_action = BP_SEQUENCE[target_step]
        side_str = current_action[1]
        enemy_side_str = "red" if side_str == "blue" else "blue"
        curr_side_code = 0 if side_str == "blue" else 1
        is_pick_action = 1.0 if current_action[0] == "pick" else 0.0

        cand = np.zeros((self.vocab_size, 33), dtype=np.float32)  # CANDIDATE_DIM=33
        cand[:, 0:4] = self.meta_matrix
        cand[:, 11:16] = self.pos_prior

        ally_pids = [str(row.get(f"{side_str}_{p}_player_id", "")).strip() for p in ["top", "jng", "mid", "bot", "sup"]]
        ally_feat_mat = np.zeros((5, n_champs, 7), dtype=np.float32)
        for i, pid in enumerate(ally_pids):
            pmat = self.player_matrix_map.get(pid)
            if pmat is not None:
                ally_feat_mat[i] = pmat[cs:ve]

        ally_pos_sum = np.zeros(5, dtype=np.float32)
        for c in ally_champs: ally_pos_sum += self.pos_prior[c]
        enemy_pos_sum = np.zeros(5, dtype=np.float32)
        for c in enemy_champs: enemy_pos_sum += self.pos_prior[c]
        
        ally_missing_roles = np.clip(1.0 - ally_pos_sum, 0.0, 1.0)
        enemy_missing_roles = np.clip(1.0 - enemy_pos_sum, 0.0, 1.0)

        # 切片 4:11 包含全部 7 个 player 特征 (含 recent_games@7)
        cand[cs:ve, 4:11] = (ally_feat_mat * ally_missing_roles[:, None, None]).max(axis=0)

        if ally_champs:
            ally_arr = np.array(ally_champs, dtype=np.int64)
            cand[cs:ve, 16] = np.max(self.syn_mat[cs:ve, ally_arr], axis=1)
            cand[cs:ve, 19] = np.max(1.0 - self.ctr_mat[cs:ve, ally_arr], axis=1)
        if enemy_champs:
            enemy_arr = np.array(enemy_champs, dtype=np.int64)
            cand[cs:ve, 18] = np.max(1.0 - self.ctr_mat[cs:ve, enemy_arr], axis=1)
            cand[cs:ve, 17] = np.max(self.syn_mat[cs:ve, enemy_arr], axis=1)

        pos_block = cand[cs:ve, 11:16]
        cand[cs:ve, 20] = pos_block @ ally_missing_roles
        cand[cs:ve, 21] = pos_block @ enemy_missing_roles
        cand[cs:ve, 22] = is_pick_action

        enemy_pids = [str(row.get(f"{enemy_side_str}_{p}_player_id", "")).strip() for p in ["top", "jng", "mid", "bot", "sup"]]
        enemy_mastery_matrix = np.zeros((5, n_champs), dtype=np.float32)
        for i, epid in enumerate(enemy_pids):
            pmat = self.player_matrix_map.get(epid)
            if pmat is not None:
                enemy_mastery_matrix[i] = pmat[cs:ve, 0]
        weighted_enemy_mastery = enemy_mastery_matrix * enemy_missing_roles[:, None]
        cand[cs:ve, 23] = weighted_enemy_mastery.max(axis=0)
        cand[cs:ve, 24] = weighted_enemy_mastery.mean(axis=0)

        cand[cs:ve, 25] = sum(1 for i in range(target_step + 1) if BP_SEQUENCE[i][0] == "ban" and BP_SEQUENCE[i][1] == side_str)

        banning_team = str(row.get(f"{side_str}_team", "")).strip()
        opponent_team = str(row.get(f"{enemy_side_str}_team", "")).strip()
        grudge_vec = self.grudge_matrix_map.get((banning_team, opponent_team))
        if grudge_vec is not None:
            cand[cs:ve, 26] = grudge_vec[cs:ve]

        enemy_respect_vec = np.zeros(self.vocab_size, dtype=np.float32)
        enemy_streak_vec = np.zeros(self.vocab_size, dtype=np.float32)
        for epid in enemy_pids:
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
        cand[cs:ve, 27] = enemy_respect_vec[cs:ve]
        cand[cs:ve, 28] = enemy_streak_vec[cs:ve]

        cand[cs:ve, 29] = float(len(ally_champs))
        cand[cs:ve, 30] = float(curr_side_code)

        # [32] is_fearless_banned: 标记前置局已使用/全局不可选英雄
        # 注意: ban 模型不使用 last_ally_synergy@idx31（与训练时 use_extended_features=False 一致），idx31 保持为 0
        # 1) 从 prev_game_champs 获取前置局已使用英雄
        prev_champ_names = str(row.get("prev_game_champs", ""))
        if prev_champ_names and prev_champ_names.strip():
            for name in prev_champ_names.split("|"):
                cid = self.name_to_idx.get(name.strip(), -1)
                if cid >= cs:
                    cand[cid, 32] = 1.0
        # 2) 推理阶段 unavail_set 中的英雄（前端传入的不可选列表）
        for uid in unavail_set:
            if cs <= uid < ve:
                cand[uid, 32] = 1.0

        mask = np.ones(self.vocab_size, dtype=np.float32)
        mask[:cs] = 0.0
        for uid in unavail_set:
            if 0 <= uid < self.vocab_size:
                mask[uid] = 0.0

        return cand, mask


def _compute_group_features_fast(sample_logits, sample_mask, champion_start_idx, vocab_size):
    desc_order = np.argsort(-sample_logits)
    rank_map = np.empty_like(sample_logits, dtype=np.float64)
    rank_map[desc_order] = np.arange(1, len(sample_logits) + 1, dtype=np.float64)

    valid_mask = sample_mask > 0.5
    valid_mask[:champion_start_idx] = False
    valid_logits = sample_logits[valid_mask]

    if valid_logits.size > 0:
        valid_mean = valid_logits.mean()
        valid_std = max(valid_logits.std(), 1e-6)
        logit_min = valid_logits.min()
        logit_max = valid_logits.max()
    else:
        valid_mean, valid_std = 0.0, 1.0
        logit_min, logit_max = 0.0, 1.0

    return {
        "rank_map": rank_map,
        "logit_min": float(logit_min),
        "logit_max": float(logit_max),
        "logit_range": float(logit_max - logit_min) if valid_logits.size > 0 else 1e-6,
        "valid_mean": float(valid_mean),
        "valid_std": float(valid_std),
        "valid_median": float(valid_mean),
        "valid_q75": float(valid_mean + 0.675 * valid_std),
        "valid_q25": float(valid_mean - 0.675 * valid_std),
        "valid_iqr": float(1.35 * valid_std),
        "top1_logit": float(sample_logits[desc_order[0]]),
        "top3_logit": 0.0,
        "top5_logit": 0.0,
        "top10_logit": 0.0,
    }

def _rank_normalize(scores):
    order = np.argsort(-scores)
    ranks = np.zeros_like(scores, dtype=np.float64)
    n = len(scores)
    for rank_pos, idx in enumerate(order):
        ranks[idx] = 1.0 - rank_pos / max(n - 1, 1)
    return ranks

def evaluate_oot_fast(skip_step6=False, override_alpha=None):
    t_total = time.time()
    log.info("=" * 70)
    log.info("  Pick Model Inference Test (OPTIMIZED) on Test Set")
    log.info(f"  Device: {DEVICE}")
    log.info(f"  Skip Step6: {skip_step6}")
    log.info("=" * 70)

    store = FastOOTFeatureStore()

    # ---- Load Models ----
    log.info("[2/5] Loading Transformer Models...")
    ckpt = torch.load(os.path.join(PICK_CKPT_DIR, "best_model_cs.pt"), map_location=DEVICE, weights_only=False)
    cand_dim = ckpt.get("candidate_dim", 33)
    ctx_dim = ckpt.get("context_dim", 15)
    
    # Infer correct vocab_size / n_positions from checkpoint's state_dict
    # (store.vocab_size may differ if champion vocabulary changed between training)
    sd = ckpt["model_state_dict"]
    extended_vocab = sd["bert.embeddings.word_embeddings.weight"].shape[0]
    ckpt_n_positions = sd["enemy_role_head.5.weight"].shape[0]
    base_vocab = extended_vocab - ckpt_n_positions
    log.info(f"  CS checkpoint: base_vocab={base_vocab}, n_positions={ckpt_n_positions}, "
             f"extended_vocab={extended_vocab}, store.vocab_size={store.vocab_size}")

    cs_h_dim = ckpt.get("h_dim", 384)
    cs_c_dim = ckpt.get("c_dim", 32)
    cs_query_dim = ckpt.get("query_dim", 128)
    cs_n_layers = ckpt.get("n_layers", 3)
    cs_n_heads = ckpt.get("n_heads", 8)
    cs_candidate_hidden = ckpt.get("candidate_hidden", 256)
    cs_tactical_hidden = ckpt.get("tactical_hidden", 256)
    cs_dropout = ckpt.get("dropout", 0.052)
    cs_attention_dropout = ckpt.get("attention_dropout", 0.106)
    
    nn_model = BPTacticalTransformerPick(
        vocab_size=base_vocab, n_positions=ckpt_n_positions, context_dim=ctx_dim,
        candidate_dim=cand_dim,
        h_dim=cs_h_dim, c_dim=cs_c_dim, query_dim=cs_query_dim,
        n_layers=cs_n_layers, n_heads=cs_n_heads,
        candidate_hidden=cs_candidate_hidden, tactical_hidden=cs_tactical_hidden,
        dropout=cs_dropout, attention_dropout=cs_attention_dropout,
    ).to(DEVICE)
    nn_model.load_state_dict(sd)
    nn_model.eval()
    log.info(f"  CS Transformer model loaded (h_dim={cs_h_dim}, query_dim={cs_query_dim})")

    has_nocs = os.path.exists(os.path.join(PICK_CKPT_DIR, "best_model_nocs.pt"))
    nn_model_nocs = None
    if has_nocs:
        nocs_ckpt = torch.load(os.path.join(PICK_CKPT_DIR, "best_model_nocs.pt"), map_location=DEVICE, weights_only=False)
        nocs_sd = nocs_ckpt["model_state_dict"]
        nocs_ext_voc = nocs_sd["bert.embeddings.word_embeddings.weight"].shape[0]
        nocs_n_pos = nocs_sd["enemy_role_head.5.weight"].shape[0]
        nocs_base_vocab = nocs_ext_voc - nocs_n_pos
        nocs_cand_dim = nocs_ckpt.get("candidate_dim", cand_dim)
        nocs_ctx_dim = nocs_ckpt.get("context_dim", ctx_dim)
        nocs_h_dim = nocs_ckpt.get("h_dim", 384)
        nocs_c_dim = nocs_ckpt.get("c_dim", 64)
        nocs_query_dim = nocs_ckpt.get("query_dim", 256)
        nocs_n_layers = nocs_ckpt.get("n_layers", 3)
        nocs_n_heads = nocs_ckpt.get("n_heads", 4)
        nocs_candidate_hidden = nocs_ckpt.get("candidate_hidden", 256)
        nocs_tactical_hidden = nocs_ckpt.get("tactical_hidden", 256)
        nocs_dropout = nocs_ckpt.get("dropout", 0.198)
        nocs_attention_dropout = nocs_ckpt.get("attention_dropout", 0.114)
        nn_model_nocs = BPTacticalTransformerPick(
            vocab_size=nocs_base_vocab, n_positions=nocs_n_pos, context_dim=nocs_ctx_dim,
            candidate_dim=nocs_cand_dim,
            h_dim=nocs_h_dim, c_dim=nocs_c_dim, query_dim=nocs_query_dim,
            n_layers=nocs_n_layers, n_heads=nocs_n_heads,
            candidate_hidden=nocs_candidate_hidden, tactical_hidden=nocs_tactical_hidden,
            dropout=nocs_dropout, attention_dropout=nocs_attention_dropout,
        ).to(DEVICE)
        nn_model_nocs.load_state_dict(nocs_ckpt["model_state_dict"])
        nn_model_nocs.eval()

    log.info("[3/5] Loading Cascade Models (Unified Phase-Aware)...")
    cascade_dir = os.path.join(PICK_CKPT_DIR, "cascade_pick")
    routing_config_path = os.path.join(cascade_dir, "routing_config.json")
    # 【修复 3】：支持残差训练模式
    fusion_mode = "blend"
    if os.path.exists(routing_config_path):
        with open(routing_config_path, "r") as f:
            routing_config = json.load(f)
        blend_alpha = routing_config.get("blend_alpha", BLEND_ALPHA)
        fusion_mode = routing_config.get("fusion_mode", "blend")
        log.info(f"  Loaded blend_alpha={blend_alpha}, fusion_mode={fusion_mode} from routing_config.json")
    else:
        blend_alpha = BLEND_ALPHA
    if override_alpha is not None:
        blend_alpha = override_alpha
        log.info(f"  Override blend_alpha={blend_alpha}")
    if fusion_mode == "residual_init_score":
        log.info(f"  Residual mode: final_score = LGBM_residual + TF_base_logits")
    else:
        log.info(f"  Blend Alpha: {blend_alpha}")

    lgb_models = []
    for i in range(5):
        m_path = os.path.join(cascade_dir, f"fold_{i}_model.txt")
        if os.path.exists(m_path):
            lgb_models.append(lgb.Booster(model_file=m_path, params={"num_threads": 1}))
            log.info(f"    fold_{i} OK ({len(lgb_models[-1].feature_name())} features)")

    with open(os.path.join(cascade_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    log.info(f"  Scaler loaded OK")

    vs = store.vocab_size
    bp_buf = torch.zeros(1, 20, dtype=torch.long, device=DEVICE)
    ctx_buf = torch.zeros(1, ctx_dim, dtype=torch.float32, device=DEVICE)
    cand_buf = torch.zeros(1, vs, 32, dtype=torch.float32, device=DEVICE)  # [31] is_fearless_banned
    mask_buf = torch.zeros(1, vs, dtype=torch.float32, device=DEVICE)
    lap_buf = torch.tensor([-1], dtype=torch.long, device=DEVICE)

    log.info(f"[4/5] Reading Test CSV: {TEST_CSV}")
    raw_df = pd.read_csv(TEST_CSV)
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df = raw_df[raw_df["date"] >= "2025-01-01"].reset_index(drop=True)

    # 每日指标追踪
    games_per_day_pick = raw_df["date"].dt.date.astype(str).value_counts().to_dict()
    from collections import defaultdict
    daily_pick = defaultdict(lambda: {
        "hits": {k: 0 for k in [1, 3, 5, 10, 20]},
        "total": 0, "mrr_sum": 0.0,
    })

    hits = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    hits_phase1 = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    hits_phase2 = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    total_pick_queries = 0
    total_pick_phase1 = 0
    total_pick_phase2 = 0
    mrr_sum = 0.0

    pick_step_labels = {s: f"{BP_SEQUENCE[s][1].capitalize()} Pick{BP_SEQUENCE[s][2]}" for s in PICK_STEP_INDICES}
    step_hits = {s: 0 for s in PICK_STEP_INDICES}
    step_totals = {s: 0 for s in PICK_STEP_INDICES}

    t_feat_total = 0.0
    t_nn_total = 0.0
    t_cascade_total = 0.0

    t0 = time.time()
    log.info("[5/5] Starting End-to-End Online Inference (Optimized)...")

    for idx in range(len(raw_df)):
        row = raw_df.iloc[idx]
        row_date = str(row["date"].date())

        for step in range(20):
            action_type, side, slot = BP_SEQUENCE[step]
            if action_type != "pick":
                continue

            true_label_name = _get_champion_from_raw_row(row, step)
            true_label = store._map_champ(true_label_name)
            if true_label < store.champion_start_idx:
                continue
            if skip_step6 and step == 6:
                continue

            valid_targets = [true_label]
            if step in TUPLE_START_STEPS:
                partner_name = _get_champion_from_raw_row(row, step + 1)
                partner_label = store._map_champ(partner_name)
                if partner_label >= store.champion_start_idx:
                    valid_targets.append(partner_label)

            total_pick_queries += 1
            is_phase1 = step < 12
            if is_phase1: total_pick_phase1 += 1
            else: total_pick_phase2 += 1

            t_feat_start = time.perf_counter()

            league_vec = np.zeros(len(LEAGUES), dtype=np.float32)
            league_str = str(row.get("league", "LPL")).strip()
            if league_str in LEAGUES: league_vec[LEAGUES.index(league_str)] = 1.0

            b_style = store.team_style_dict.get(str(row.get("blue_team", "")).strip(), [0.7, 0.0, 1900.0, 0.5, 0.5])
            r_style = store.team_style_dict.get(str(row.get("red_team", "")).strip(), [0.7, 0.0, 1900.0, 0.5, 0.5])
            team_style = np.array(b_style + r_style, dtype=np.float32)
            playoffs_val = float(row.get("playoffs", 0)) if pd.notna(row.get("playoffs")) else 0.0
            fp_val = float(row.get("first_pick_map_side", 1)) if pd.notna(row.get("first_pick_map_side")) else 1.0
            global_context = np.concatenate([league_vec, team_style, [playoffs_val, fp_val]])

            # 游戏局数 One-Hot 特征 (is_game_1 ~ is_game_5)
            game_num_features = np.array([
                float(row.get("is_game_1", 0)),
                float(row.get("is_game_2", 0)),
                float(row.get("is_game_3", 0)),
                float(row.get("is_game_4", 0)),
                float(row.get("is_game_5", 0)),
            ], dtype=np.float32)
            global_context = np.concatenate([global_context, game_num_features])

            bp_seq = []
            ally_champs = []
            enemy_champs = []
            unavail_set = set()
            curr_side_code = 0 if side == "blue" else 1
            last_ally_pos = -1

            for i in range(step):
                cid = store._map_champ(_get_champion_from_raw_row(row, i))
                bp_seq.append(cid)
                if cid < store.champion_start_idx: continue
                unavail_set.add(cid)
                if BP_SEQUENCE[i][0] == "pick":
                    if (BP_SEQUENCE[i][1] == "blue" and curr_side_code == 0) or \
                       (BP_SEQUENCE[i][1] == "red" and curr_side_code == 1):
                        ally_champs.append(cid)
                        last_ally_pos = i
                    else:
                        enemy_champs.append(cid)

            cand_np, mask_np = store.get_candidate_matrix_fast(row, step, ally_champs, enemy_champs, unavail_set)
            t_feat_total += time.perf_counter() - t_feat_start

            t_nn_start = time.perf_counter()

            bp_padded = bp_seq + [store.PAD_IDX] * (20 - len(bp_seq))
            bp_buf[0].copy_(torch.as_tensor(bp_padded, dtype=torch.long))
            ctx_buf[0].copy_(torch.as_tensor(global_context, dtype=torch.float32))
            cand_buf[0].copy_(torch.as_tensor(cand_np, dtype=torch.float32))
            mask_buf[0].copy_(torch.as_tensor(mask_np, dtype=torch.float32))
            lap_buf[0] = last_ally_pos

            with torch.no_grad():
                cs_logits = nn_model(bp_buf, ctx_buf, cand_buf, mask_buf,
                                     last_ally_pos=lap_buf)["logits"].squeeze(0).cpu().numpy()

                cand_nocs_buf = cand_buf.clone()
                cand_nocs_buf[:, :, CS_FEATURE_INDICES] = 0.0

                if nn_model_nocs:
                    nocs_logits = nn_model_nocs(bp_buf, ctx_buf, cand_nocs_buf, mask_buf,
                                                last_ally_pos=lap_buf)["logits"].squeeze(0).cpu().numpy()
                else:
                    nocs_logits = nn_model(bp_buf, ctx_buf, cand_nocs_buf, mask_buf,
                                           last_ally_pos=lap_buf)["logits"].squeeze(0).cpu().numpy()

            t_nn_total += time.perf_counter() - t_nn_start

            t_cascade_start = time.perf_counter()

            valid_cids = np.where(mask_np > 0.5)[0]
            valid_cids = valid_cids[valid_cids >= store.champion_start_idx]
            total_valid = len(valid_cids)

            cs_gf = _compute_group_features_fast(cs_logits, mask_np, store.champion_start_idx, store.vocab_size)
            nocs_gf = _compute_group_features_fast(nocs_logits, mask_np, store.champion_start_idx, store.vocab_size)

            cs_logits_group = cs_logits[valid_cids]
            cs_ranks_group = cs_gf["rank_map"][valid_cids]
            nocs_logits_group = nocs_logits[valid_cids]
            nocs_ranks_group = nocs_gf["rank_map"][valid_cids]
            cand_feats_group = cand_np[valid_cids]

            X_arr = _build_feature_matrix_batch(
                cs_logits_group, cs_ranks_group, cs_gf,
                nocs_logits_group, nocs_ranks_group, nocs_gf,
                cand_feats_group, total_valid, total_valid, step
            )

            lgb_preds = np.zeros(total_valid, dtype=np.float64)
            X_scaled = scaler.transform(X_arr)
            for m in lgb_models:
                lgb_preds += m.predict(X_scaled)
            lgb_preds /= len(lgb_models)

            # 【修复 3】：残差模式下，最终分数 = LGBM 残差 + TF base logits
            if fusion_mode == "residual_init_score":
                final_scores = lgb_preds + cs_logits[valid_cids]
            else:
                cs_rn = _rank_normalize(cs_logits[valid_cids])
                lgb_rn = _rank_normalize(lgb_preds)
                final_scores = (blend_alpha * cs_rn) + ((1.0 - blend_alpha) * lgb_rn)

            sorted_idx = np.argsort(-final_scores)
            top_cids = valid_cids[sorted_idx[:20]]

            t_cascade_total += time.perf_counter() - t_cascade_start

            best_rank_for_mrr = float('inf')
            for vt in valid_targets:
                rank_arr = np.where(top_cids == vt)[0]
                if len(rank_arr) > 0 and rank_arr[0] < best_rank_for_mrr:
                    best_rank_for_mrr = rank_arr[0]
            if best_rank_for_mrr != float('inf'):
                mrr_sum += 1.0 / (best_rank_for_mrr + 1)

            for k in hits.keys():
                if any(vt in top_cids[:k] for vt in valid_targets):
                    hits[k] += 1
                    daily_pick[row_date]["hits"][k] += 1
                    if is_phase1: hits_phase1[k] += 1
                    else: hits_phase2[k] += 1

            if best_rank_for_mrr != float('inf'):
                daily_pick[row_date]["mrr_sum"] += 1.0 / (best_rank_for_mrr + 1)
            daily_pick[row_date]["total"] += 1

            if any(vt in top_cids[:10] for vt in valid_targets):
                step_hits[step] += 1
            step_totals[step] += 1

        if (idx + 1) % 10 == 0:
            log.info(f"  Processed {idx + 1}/{len(raw_df)} matches. Current P@10: {hits[10]/max(total_pick_queries, 1)*100:.2f}%")

    elapsed = time.time() - t0
    mrr = mrr_sum / max(total_pick_queries, 1) * 100

    log.info("")
    log.info("=" * 70)
    log.info("  PICK MODEL TEST SET RESULTS (Optimized Inference)")
    log.info("=" * 70)
    log.info(f"  Total test matches: {len(raw_df)}")
    log.info(f"  Total Pick queries: {total_pick_queries}")
    log.info(f"    Phase1 (step 0-11): {total_pick_phase1}")
    log.info(f"    Phase2 (step 12-19): {total_pick_phase2}")
    log.info("")
    log.info(f"  {'Metric':<12} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'@20':>8} {'MRR':>8}")
    log.info(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    def fmt_metrics(h, total, mrr_val=None):
        parts = [f"{h[k]/max(total,1)*100:>7.2f}%" for k in [1, 3, 5, 10, 20]]
        if mrr_val is not None: parts.append(f"{mrr_val:>7.2f}%")
        return "  ".join(parts)

    log.info(f"  {'Overall':<12} {fmt_metrics(hits, total_pick_queries, mrr)}")
    log.info(f"  {'Phase1':<12} {fmt_metrics(hits_phase1, total_pick_phase1)}")
    log.info(f"  {'Phase2':<12} {fmt_metrics(hits_phase2, total_pick_phase2)}")

    log.info("")
    log.info(f"  {'Per-Step Pick@10 Breakdown':<40}")
    log.info(f"  {'-'*40}")
    step_p10 = {}
    for s in PICK_STEP_INDICES:
        if s == 6 and skip_step6: continue
        p10 = step_hits[s] / max(step_totals[s], 1) * 100
        step_p10[s] = p10
        log.info(f"  Step {s:2d} ({pick_step_labels[s]:<12}): {step_hits[s]:>3}/{step_totals[s]:<3}  P@10 = {p10:>6.2f}%")

    sorted_steps = sorted(step_p10.items(), key=lambda x: x[1])
    log.info("")
    log.info(f"  Lowest P@10 steps:")
    for rank, (s, p10) in enumerate(sorted_steps[:3], 1):
        log.info(f"    {rank}. Step {s:2d} ({pick_step_labels[s]:<12}): P@10 = {p10:.2f}%")

    # --- 每日指标输出 ---
    log.info("")
    log.info("=" * 70)
    log.info("  PICK DAILY METRICS")
    log.info(f"  {'Date':<12} {'Games':>6} {'Queries':>8} {'P@1':>7} {'P@3':>7} {'P@5':>7} {'P@10':>7} {'P@20':>7} {'MRR':>7}")
    log.info(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for d in sorted(daily_pick.keys()):
        d_info = daily_pick[d]
        t = max(d_info["total"], 1)
        dmrr = d_info["mrr_sum"] / t * 100
        n_g = games_per_day_pick.get(d, 0)
        p_vals = [d_info["hits"][k] / t * 100 for k in [1, 3, 5, 10, 20]]
        log.info(f"  {d:<12} {n_g:>6} {d_info['total']:>8} "
                 f"{p_vals[0]:>6.2f}% {p_vals[1]:>6.2f}% {p_vals[2]:>6.2f}% {p_vals[3]:>6.2f}% {p_vals[4]:>6.2f}% {dmrr:>6.2f}%")

    log.info("=" * 70)
    log.info(f"  Inference time: {elapsed:.1f}s ({elapsed/max(total_pick_queries,1)*1000:.1f}ms/query)")
    log.info(f"  Timing breakdown:")
    log.info(f"    Feature construction: {t_feat_total:.2f}s ({t_feat_total/max(total_pick_queries,1)*1000:.1f}ms/query)")
    log.info(f"    Transformer forward:  {t_nn_total:.2f}s ({t_nn_total/max(total_pick_queries,1)*1000:.1f}ms/query)")
    log.info(f"    Cascade (LGB+blend):  {t_cascade_total:.2f}s ({t_cascade_total/max(total_pick_queries,1)*1000:.1f}ms/query)")


def evaluate_ban_fast():
    t_total = time.time()
    log.info("=" * 70)
    log.info("  Ban Model Inference Test (OPTIMIZED) on Test Set")
    log.info(f"  Device: {DEVICE}")
    log.info("=" * 70)

    store = FastOOTFeatureStore()

    log.info("[2/5] Loading Ban Transformer Models...")
    ban_ckpt = torch.load(os.path.join(BAN_CKPT_DIR, "best_model_cs.pt"), map_location=DEVICE, weights_only=False)
    ban_cand_dim = ban_ckpt.get("candidate_dim", EXTENDED_CANDIDATE_DIM)
    ban_ctx_dim = ban_ckpt.get("context_dim", BAN_CONTEXT_DIM)
    ban_sd = ban_ckpt["model_state_dict"]
    ban_ext_voc = ban_sd["bert.embeddings.word_embeddings.weight"].shape[0]
    ban_n_pos = ban_sd.get("enemy_role_head.5.weight", ban_sd.get("enemy_role_head.3.weight")).shape[0]
    # Ban model uses vocab_size directly (not extended with role tokens)
    ban_base_vocab = ban_ext_voc
    log.info(f"  Ban CS checkpoint: vocab_size={ban_base_vocab}, n_positions={ban_n_pos}, "
             f"store.vocab_size={store.vocab_size}")
    ban_model = BPTacticalTransformerBan(
        vocab_size=ban_base_vocab, context_dim=ban_ctx_dim,
        candidate_dim=ban_cand_dim,
    ).to(DEVICE)
    ban_model.load_state_dict(ban_sd)
    ban_model.eval()

    ban_has_nocs = os.path.exists(os.path.join(BAN_CKPT_DIR, "best_model_nocs.pt"))
    ban_model_nocs = None
    if ban_has_nocs:
        nocs_ckpt = torch.load(os.path.join(BAN_CKPT_DIR, "best_model_nocs.pt"), map_location=DEVICE, weights_only=False)
        nocs_sd = nocs_ckpt["model_state_dict"]
        nocs_ext_voc = nocs_sd["bert.embeddings.word_embeddings.weight"].shape[0]
        nocs_n_pos = nocs_sd.get("enemy_role_head.5.weight", nocs_sd.get("enemy_role_head.3.weight")).shape[0]
        nocs_base_vocab = nocs_ext_voc
        nocs_cand_dim = nocs_ckpt.get("candidate_dim", ban_cand_dim)
        nocs_ctx_dim = nocs_ckpt.get("context_dim", ban_ctx_dim)
        ban_model_nocs = BPTacticalTransformerBan(
            vocab_size=nocs_base_vocab, context_dim=nocs_ctx_dim,
            candidate_dim=nocs_cand_dim,
        ).to(DEVICE)
        ban_model_nocs.load_state_dict(nocs_sd)
        ban_model_nocs.eval()

    log.info("[3/5] Loading Ban Cascade Models (Unified Phase-Aware)...")
    ban_cascade_dir = os.path.join(BAN_CKPT_DIR, "cascade_ban")
    ban_routing_path = os.path.join(ban_cascade_dir, "routing_config.json")
    BAN_BLEND_ALPHA = 0.1
    if os.path.exists(ban_routing_path):
        with open(ban_routing_path, "r") as f:
            ban_routing = json.load(f)
        BAN_BLEND_ALPHA = ban_routing.get("blend_alpha", 0.1)

    ban_lgb_models = []
    for i in range(5):
        m_path = os.path.join(ban_cascade_dir, f"fold_{i}_model.txt")
        if os.path.exists(m_path):
            ban_lgb_models.append(lgb.Booster(model_file=m_path, params={"num_threads": 1}))

    ban_scaler_path = os.path.join(ban_cascade_dir, "scaler.pkl")
    ban_scaler = None
    if os.path.exists(ban_scaler_path):
        with open(ban_scaler_path, "rb") as f:
            ban_scaler = pickle.load(f)

    vs = store.vocab_size
    ban_bp_buf = torch.zeros(1, 20, dtype=torch.long, device=DEVICE)
    ban_ctx_buf = torch.zeros(1, ban_ctx_dim, dtype=torch.float32, device=DEVICE)
    ban_cand_buf = torch.zeros(1, vs, ban_cand_dim, dtype=torch.float32, device=DEVICE)
    ban_mask_buf = torch.zeros(1, vs, dtype=torch.float32, device=DEVICE)
    ban_hist_buf = torch.zeros(1, 20, dtype=torch.long, device=DEVICE)

    log.info(f"[4/5] Reading Test CSV: {TEST_CSV}")
    raw_df = pd.read_csv(TEST_CSV)
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df = raw_df[raw_df["date"] >= "2025-01-01"].reset_index(drop=True)

    # 每日指标追踪 (Ban)
    games_per_day_ban = raw_df["date"].dt.date.astype(str).value_counts().to_dict()
    daily_ban = defaultdict(lambda: {
        "hits": {k: 0 for k in [1, 3, 5, 10, 20]},
        "total": 0, "mrr_sum": 0.0,
    })

    hits = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    hits_phase1 = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    hits_phase2 = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    total_ban_queries = 0
    total_ban_phase1 = 0
    total_ban_phase2 = 0
    mrr_sum = 0.0

    ban_step_labels = {s: f"{BP_SEQUENCE[s][1].capitalize()} Ban{BP_SEQUENCE[s][2]}" for s in BAN_STEP_INDICES}
    step_hits = {s: 0 for s in BAN_STEP_INDICES}
    step_totals = {s: 0 for s in BAN_STEP_INDICES}

    t_feat_total = 0.0
    t_nn_total = 0.0
    t_cascade_total = 0.0

    t0 = time.time()
    log.info("[5/5] Starting Ban End-to-End Online Inference (Optimized)...")

    for idx in range(len(raw_df)):
        row = raw_df.iloc[idx]
        row_date = str(row["date"].date())

        for step in range(20):
            action_type, side, slot = BP_SEQUENCE[step]
            if action_type != "ban": continue

            true_label_name = _get_champion_from_raw_row(row, step)
            true_label = store._map_champ(true_label_name)
            if true_label < store.champion_start_idx: continue

            valid_targets = [true_label]
            total_ban_queries += 1

            ban_step_count = sum(1 for i in range(step + 1) if BP_SEQUENCE[i][0] == "ban" and BP_SEQUENCE[i][1] == side)
            is_phase1 = ban_step_count < 3
            if is_phase1: total_ban_phase1 += 1
            else: total_ban_phase2 += 1

            t_feat_start = time.perf_counter()

            league_vec = np.zeros(len(LEAGUES), dtype=np.float32)
            league_str = str(row.get("league", "LPL")).strip()
            if league_str in LEAGUES: league_vec[LEAGUES.index(league_str)] = 1.0

            b_style = store.team_style_dict.get(str(row.get("blue_team", "")).strip(), [0.7, 0.0, 1900.0, 0.5, 0.5])
            r_style = store.team_style_dict.get(str(row.get("red_team", "")).strip(), [0.7, 0.0, 1900.0, 0.5, 0.5])
            team_style = np.array(b_style + r_style, dtype=np.float32)
            playoffs_val = float(row.get("playoffs", 0)) if pd.notna(row.get("playoffs")) else 0.0
            fp_val = float(row.get("first_pick_map_side", 1)) if pd.notna(row.get("first_pick_map_side")) else 1.0
            global_context = np.concatenate([league_vec, team_style, [playoffs_val, fp_val]])

            # 游戏局数 One-Hot 特征 (is_game_1 ~ is_game_5)
            game_num_features = np.array([
                float(row.get("is_game_1", 0)),
                float(row.get("is_game_2", 0)),
                float(row.get("is_game_3", 0)),
                float(row.get("is_game_4", 0)),
                float(row.get("is_game_5", 0)),
            ], dtype=np.float32)
            global_context = np.concatenate([global_context, game_num_features])

            bp_seq = []
            ally_champs = []
            enemy_champs = []
            unavail_set = set()
            curr_side_code = 0 if side == "blue" else 1
            last_ally_pos = -1
            hist_pos = np.full(20, -1, dtype=np.int64)

            for i in range(step):
                cid = store._map_champ(_get_champion_from_raw_row(row, i))
                bp_seq.append(cid)
                if cid < store.champion_start_idx: continue
                unavail_set.add(cid)
                if BP_SEQUENCE[i][0] == "pick":
                    pos_idx = int(np.argmax(store.pos_prior[cid]))
                    hist_pos[i] = pos_idx
                    if (BP_SEQUENCE[i][1] == "blue" and curr_side_code == 0) or \
                       (BP_SEQUENCE[i][1] == "red" and curr_side_code == 1):
                        ally_champs.append(cid)
                        last_ally_pos = i
                    else:
                        enemy_champs.append(cid)

            cand_np, mask_np = store.get_candidate_matrix_ban_fast(row, step, ally_champs, enemy_champs, unavail_set)
            t_feat_total += time.perf_counter() - t_feat_start

            t_nn_start = time.perf_counter()

            bp_padded = bp_seq + [store.PAD_IDX] * (20 - len(bp_seq))
            ban_bp_buf[0].copy_(torch.as_tensor(bp_padded, dtype=torch.long))
            ban_ctx_buf[0].copy_(torch.as_tensor(global_context, dtype=torch.float32))
            ban_cand_buf[0].copy_(torch.as_tensor(cand_np, dtype=torch.float32))
            ban_mask_buf[0].copy_(torch.as_tensor(mask_np, dtype=torch.float32))
            ban_hist_buf[0].copy_(torch.as_tensor(hist_pos, dtype=torch.long))

            with torch.no_grad():
                cs_logits = ban_model(ban_bp_buf, ban_ctx_buf, ban_cand_buf, ban_mask_buf,
                                      history_positions=ban_hist_buf)["logits"].squeeze(0).cpu().numpy()

                if ban_model_nocs:
                    cand_nocs_buf = ban_cand_buf.clone()
                    # 修复 Segfault 的核心：Ban 模型候选矩阵维度没有 30，直接定义专属的 BAN_CS_INDICES
                    BAN_CS_INDICES = [15, 16, 17, 18]
                    cand_nocs_buf[:, :, BAN_CS_INDICES] = 0.0
                    nocs_logits = ban_model_nocs(ban_bp_buf, ban_ctx_buf, cand_nocs_buf, ban_mask_buf,
                                                  history_positions=ban_hist_buf)["logits"].squeeze(0).cpu().numpy()
                else:
                    nocs_logits = cs_logits

            t_nn_total += time.perf_counter() - t_nn_start

            t_cascade_start = time.perf_counter()

            valid_cids = np.where(mask_np > 0.5)[0]
            valid_cids = valid_cids[valid_cids >= store.champion_start_idx]
            total_valid = len(valid_cids)

            cs_gf = _compute_ban_group_features(cs_logits, mask_np, store.champion_start_idx, store.vocab_size)
            X_arr = _build_ban_feature_matrix_batch(
                cs_logits[valid_cids], cs_gf["rank_map"][valid_cids], cs_gf,
                cand_np[valid_cids], total_valid,
            )

            lgb_preds = np.zeros(total_valid, dtype=np.float64)
            if ban_scaler is not None and len(ban_lgb_models) > 0:
                X_scaled = ban_scaler.transform(X_arr)
                for m in ban_lgb_models:
                    lgb_preds += m.predict(X_scaled)
                lgb_preds /= max(len(ban_lgb_models), 1)

            base_rn = _rank_normalize(cs_logits[valid_cids])
            lgb_rn = _rank_normalize(lgb_preds)
            final_scores = BAN_BLEND_ALPHA * base_rn + (1.0 - BAN_BLEND_ALPHA) * lgb_rn

            sorted_idx = np.argsort(-final_scores)
            top_cids = valid_cids[sorted_idx[:20]]

            t_cascade_total += time.perf_counter() - t_cascade_start

            best_rank_for_mrr = float('inf')
            for vt in valid_targets:
                rank_arr = np.where(top_cids == vt)[0]
                if len(rank_arr) > 0 and rank_arr[0] < best_rank_for_mrr:
                    best_rank_for_mrr = rank_arr[0]
            if best_rank_for_mrr != float('inf'):
                mrr_sum += 1.0 / (best_rank_for_mrr + 1)

            for k in hits.keys():
                if any(vt in top_cids[:k] for vt in valid_targets):
                    hits[k] += 1
                    daily_ban[row_date]["hits"][k] += 1
                    if is_phase1: hits_phase1[k] += 1
                    else: hits_phase2[k] += 1

            if best_rank_for_mrr != float('inf'):
                daily_ban[row_date]["mrr_sum"] += 1.0 / (best_rank_for_mrr + 1)
            daily_ban[row_date]["total"] += 1

            if any(vt in top_cids[:10] for vt in valid_targets):
                step_hits[step] += 1
            step_totals[step] += 1

        if (idx + 1) % 10 == 0:
            log.info(f"  Processed {idx + 1}/{len(raw_df)} matches. Current B@10: {hits[10]/max(total_ban_queries, 1)*100:.2f}%")

    elapsed = time.time() - t0
    mrr = mrr_sum / max(total_ban_queries, 1) * 100

    log.info("")
    log.info("=" * 70)
    log.info("  BAN MODEL TEST SET RESULTS (Optimized Inference)")
    log.info("=" * 70)
    log.info(f"  Total test matches: {len(raw_df)}")
    log.info(f"  Total Ban queries: {total_ban_queries}")
    log.info(f"    Phase1 (ban_step < 3): {total_ban_phase1}")
    log.info(f"    Phase2 (ban_step >= 3): {total_ban_phase2}")
    log.info("")
    log.info(f"  {'Metric':<12} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'@20':>8} {'MRR':>8}")
    log.info(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    def fmt_metrics(h, total, mrr_val=None):
        parts = [f"{h[k]/max(total,1)*100:>7.2f}%" for k in [1, 3, 5, 10, 20]]
        if mrr_val is not None: parts.append(f"{mrr_val:>7.2f}%")
        return "  ".join(parts)

    log.info(f"  {'Overall':<12} {fmt_metrics(hits, total_ban_queries, mrr)}")
    log.info(f"  {'Phase1':<12} {fmt_metrics(hits_phase1, total_ban_phase1)}")
    log.info(f"  {'Phase2':<12} {fmt_metrics(hits_phase2, total_ban_phase2)}")

    log.info("")
    log.info(f"  {'Per-Step Ban@10 Breakdown':<40}")
    log.info(f"  {'-'*40}")
    step_b10 = {}
    for s in BAN_STEP_INDICES:
        b10 = step_hits[s] / max(step_totals[s], 1) * 100
        step_b10[s] = b10
        log.info(f"  Step {s:2d} ({ban_step_labels[s]:<12}): {step_hits[s]:>3}/{step_totals[s]:<3}  B@10 = {b10:>6.2f}%")

    sorted_steps = sorted(step_b10.items(), key=lambda x: x[1])
    log.info("")
    log.info(f"  Lowest B@10 steps:")
    for rank, (s, b10) in enumerate(sorted_steps[:3], 1):
        log.info(f"    {rank}. Step {s:2d} ({ban_step_labels[s]:<12}): B@10 = {b10:.2f}%")

    # --- 每日指标输出 (Ban) ---
    log.info("")
    log.info("=" * 70)
    log.info("  BAN DAILY METRICS")
    log.info(f"  {'Date':<12} {'Games':>6} {'Queries':>8} {'B@1':>7} {'B@3':>7} {'B@5':>7} {'B@10':>7} {'B@20':>7} {'MRR':>7}")
    log.info(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for d in sorted(daily_ban.keys()):
        d_info = daily_ban[d]
        t = max(d_info["total"], 1)
        dmrr = d_info["mrr_sum"] / t * 100
        n_g = games_per_day_ban.get(d, 0)
        b_vals = [d_info["hits"][k] / t * 100 for k in [1, 3, 5, 10, 20]]
        log.info(f"  {d:<12} {n_g:>6} {d_info['total']:>8} "
                 f"{b_vals[0]:>6.2f}% {b_vals[1]:>6.2f}% {b_vals[2]:>6.2f}% {b_vals[3]:>6.2f}% {b_vals[4]:>6.2f}% {dmrr:>6.2f}%")

    log.info("=" * 70)
    log.info(f"  Inference time: {elapsed:.1f}s ({elapsed/max(total_ban_queries,1)*1000:.1f}ms/query)")
    log.info(f"  Timing breakdown:")
    log.info(f"    Feature construction: {t_feat_total:.2f}s ({t_feat_total/max(total_ban_queries,1)*1000:.1f}ms/query)")
    log.info(f"    Transformer forward:  {t_nn_total:.2f}s ({t_nn_total/max(total_ban_queries,1)*1000:.1f}ms/query)")
    log.info(f"    Cascade (LGB+blend):  {t_cascade_total:.2f}s ({t_cascade_total/max(total_ban_queries,1)*1000:.1f}ms/query)")


if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))
    
    parser = argparse.ArgumentParser(description="Pick/Ban Model Inference Test")
    parser.add_argument("--mode", type=str, default="both", choices=["pick", "ban", "both"],
                        help="推理模式: pick=仅Pick, ban=仅Ban, both=同时Pick+Ban (默认: both)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="覆盖 Pick blend alpha (0~1, 默认使用 routing_config.json 中的值)")
    args = parser.parse_args()

    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _run_log_path = os.path.join(LOG_DIR, f"inference_test_{_run_ts}.log")
    _run_fh = logging.FileHandler(_run_log_path, encoding="utf-8")
    _run_fh.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _run_fh.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_run_fh)

    log.info("=" * 70)
    log.info(f"  >>> INFERENCE TEST — Mode: {args.mode.upper()} <<<")
    log.info("=" * 70)

    if args.mode in ("pick", "both"):
        evaluate_oot_fast(skip_step6=False, override_alpha=args.alpha)

    if args.mode in ("ban", "both"):
        evaluate_ban_fast()