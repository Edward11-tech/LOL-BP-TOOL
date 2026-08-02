"""
Ban 阶段数据加载器
=============================================
PyTorch Dataset 和 DataLoader，用于 Ban 阶段 Transformer 模型的训练和验证，
负责从特征文件加载数据并构建模型输入张量。

功能描述:
    - 加载 context、player、meta、counter/synergy 等特征数据
    - 构建 BP 序列的上下文表示
    - 为每个 Ban 步骤生成候选英雄特征矩阵
    - 支持 Point-in-Time (PIT) 特征快照，避免数据泄露
    - 支持 Fearless Draft 前置局禁用英雄
    - 支持赛区权重采样
    - 提供训练/验证 DataLoader 创建接口

主要类/常量:
    - BPRecommendationDataset: Ban 阶段 PyTorch 数据集类
    - BAN_CANDIDATE_DIM: Ban 候选特征维度
    - EXTENDED_CANDIDATE_DIM: 扩展候选特征维度
    - BAN_CONTEXT_DIM: 全局上下文维度
    - create_train_val_dataloaders(): 创建训练和验证 DataLoader

使用方法:
    from bp_recommendation.model_ban.dataloader_ban import create_train_val_dataloaders
    
    train_loader, val_loader = create_train_val_dataloaders(
        batch_size=32, val_ratio=0.15, num_workers=4
    )
"""
import json
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())
sys.path.insert(0, _PROJECT_ROOT)

from logger_config import get_logger
log = get_logger(__name__)

from bp_recommendation.feature_pipeline import (
    BP_SEQUENCE,
    load_champion_vocabulary,
    CANDIDATE_FEAT_MAP
)

POS_2_IDX = {"top": 0, "jungle": 1, "mid": 2, "bot": 3, "support": 4}
LEAGUES = ["LPL", "LCK", "LEC"]

LEAGUE_WEIGHTS = {"LCK": 1.0, "LPL": 1.0, "LEC": 0.7}

# ============================================================================
# Ban 模型特征截断常量
# ============================================================================
BAN_CANDIDATE_DIM = 32
EXTENDED_CANDIDATE_DIM = 33
BAN_CONTEXT_DIM = 20

BAN_CONTEXT_COLS = (
    ["date", "gameid", "league", "match_seq_idx"]
    + [f"bp_step{i}_{suffix}" for i in range(20) for suffix in ("champion_id", "action", "side")]
    + [f"{side}_{pos}_{col}" for side in ["blue", "red"] for pos in ["top", "jng", "mid", "bot", "sup"] for col in ["champion_id", "player_id"]]
    + [f"{side}_team" for side in ["blue", "red"]]
    + [f"{side}_team_{metric}" for side in ["blue", "red"] for metric in ["avg_ckpm", "avg_golddiffat15", "avg_gamelength", "firstdragon_rate", "firsttower_rate"]]
    + ["playoffs", "first_pick_map_side"]
    + ["is_game_1", "is_game_2", "is_game_3", "is_game_4", "is_game_5"]
    + ["prev_game_champs"]
)

EXTENDED_CONTEXT_COLS = (
    list(BAN_CONTEXT_COLS)
    + ["patch", "split", "year", "blue_team_profile_games", "red_team_profile_games"]
)

BAN_META_COLS = ["gameid", "champion_id", "meta_pick_rate_pit", "meta_ban_rate_pit", "meta_presence_pit", "meta_win_rate_pit"]
BAN_PLAYER_COLS = ["player_id", "gameid", "champion_id", "mastery_score",
                   "player_recent_kda_90d", "player_recent_wr_90d", "player_recent_games_90d",
                   "player_overall_recent_kda", "player_overall_recent_wr", "player_overall_recent_games"]

def _truncate_columns(df, required_cols, name="DataFrame"):
    existing = set(df.columns)
    keep = [c for c in required_cols if c in existing]
    missing = [c for c in required_cols if c not in existing]
    dropped = [c for c in existing if c not in set(required_cols)]
    if missing:
        log.warning(f"  [Truncate] WARNING: {name} missing expected columns: {missing}")
    if dropped:
        log.info(f"  [Truncate] {name}: dropped {len(dropped)} pick-only columns.")
    return df[keep]

class BPRecommendationDataset(Dataset):
    def __init__(self, context_df, player_store_df, meta_store_df,
                 counter_dict, synergy_dict, vocab_path, position_json_path,
                 grudge_store=None, respect_store=None, hot_streak_store=None,
                 is_train=True, anchor_date=None, force_unroll=False,
                 use_extended_features=False):  # 32 维矩阵（is_fearless_banned@idx31），不写 last_ally_synergy@idx30
        
        self.name_to_idx, self.idx_to_name, self.vocab_size, self.special_tokens, self.champion_start_idx = \
            load_champion_vocabulary(vocab_path)

        self.PAD_IDX = self.special_tokens["PAD"]
        self.UNK_IDX = self.special_tokens["UNK"]

        self.context = context_df.to_dict("records")
        self.counter_dict = counter_dict
        self.synergy_dict = synergy_dict
        self.grudge_store = grudge_store or {}
        self.respect_store = respect_store or {}
        self.hot_streak_store = hot_streak_store or {}
        self.is_train = is_train
        self.use_extended_features = use_extended_features
        self.force_unroll = force_unroll

        # 训练模式下过滤含未知选手的比赛，避免假选手样本污染选手特征学习
        if self.is_train:
            original_count = len(self.context)
            PLAYER_POSITIONS = ["top", "jng", "mid", "bot", "sup"]
            UNKNOWN_TOKENS = {"", "unknown", "nan", "none", "null"}
            filtered_context = []
            for row in self.context:
                has_unknown = False
                for side in ["blue", "red"]:
                    for pos in PLAYER_POSITIONS:
                        pid = str(row.get(f"{side}_{pos}_player_id", "")).strip().lower()
                        if pid in UNKNOWN_TOKENS:
                            has_unknown = True
                            break
                    if has_unknown:
                        break
                if not has_unknown:
                    filtered_context.append(row)
            removed = original_count - len(filtered_context)
            if removed > 0:
                self.context = filtered_context
                log.info(f"Filtered {removed} matches with unknown players "
                         f"({original_count} -> {len(self.context)} remaining).")

        self._build_cs_matrices()

        if self.is_train and not self.force_unroll:
            self.sample_list = [(i, None) for i in range(len(self.context))]
            log.info(f"Initializing TRAIN Dataset with {len(self.context)} matches (Random Steps).")
        else:
            self.sample_list = [(i, step) for i in range(len(self.context)) for step in range(len(BP_SEQUENCE))]
            mode_str = "TRAIN (Forced Unroll)" if self.is_train else "VAL"
            log.info(f"Initializing {mode_str} Dataset: {len(self.context)} matches unrolled to {len(self.sample_list)} static steps.")

        if anchor_date is not None:
            self.anchor_date = anchor_date
        else:
            self.anchor_date = pd.to_datetime([r["date"] for r in self.context]).max()

        self._build_feature_stores(meta_store_df, player_store_df)
        self.position_prior_matrix = self._build_position_matrix(position_json_path)
        
        # 【核心性能优化】：全量 Numpy 矩阵预编译
        self._precompute_match_tensors()

    def _build_cs_matrices(self):
        self.synergy_matrix = np.full((self.vocab_size, self.vocab_size), 0.5, dtype=np.float32)
        self.counter_matrix = np.full((self.vocab_size, self.vocab_size), 0.5, dtype=np.float32)

        for key_str, wr in self.synergy_dict.items():
            parts = key_str.split("||")
            if len(parts) == 2:
                c1_idx = self.name_to_idx.get(parts[0], -1)
                c2_idx = self.name_to_idx.get(parts[1], -1)
                if c1_idx >= 0 and c2_idx >= 0:
                    self.synergy_matrix[c1_idx, c2_idx] = float(wr)
                    self.synergy_matrix[c2_idx, c1_idx] = float(wr)

        for champ_name, opponents in self.counter_dict.items():
            c1_idx = self.name_to_idx.get(champ_name, -1)
            if c1_idx < 0: continue
            for opp_name, stats in opponents.items():
                c2_idx = self.name_to_idx.get(opp_name, -1)
                if c2_idx >= 0:
                    self.counter_matrix[c1_idx, c2_idx] = float(stats.get("win_rate", 0.5))

    def _build_feature_stores(self, meta_df, player_df):
        unique_gameids = [str(r["gameid"]) for r in self.context]
        self.gameid_to_idx = {gid: i for i, gid in enumerate(unique_gameids)}
        n_games = len(unique_gameids)
        cs, ve = self.champion_start_idx, self.vocab_size
        
        self.meta_tensor = np.zeros((n_games, self.vocab_size, 4), dtype=np.float32)
        self.meta_tensor[:, :, 3] = 0.5
        
        # 【修复】：使用 itertuples 提速
        for row in meta_df.itertuples(index=False):
            gid = str(row.gameid)
            c = int(row.champion_id)
            if gid in self.gameid_to_idx and c < self.vocab_size:
                g_idx = self.gameid_to_idx[gid]
                self.meta_tensor[g_idx, c] = [
                    row.meta_pick_rate_pit, row.meta_ban_rate_pit,
                    row.meta_presence_pit, row.meta_win_rate_pit
                ]

        # 【修复】：使用 pandas 字符串拼接替代 np.char.add，避免 object/<U dtype 不一致问题
        pkey_series = player_df['player_id'].astype(str) + "_" + player_df['gameid'].astype(str)
        unique_pkeys = np.unique(pkey_series.values)
        
        self.pkey_to_idx = {pk: i for i, pk in enumerate(unique_pkeys)}
        self.UNKNOWN_PKEY_IDX = len(unique_pkeys)
        self.player_tensor = np.zeros((len(unique_pkeys) + 1, self.vocab_size, 7), dtype=np.float32)
        # 维度顺序: [mastery, recent_kda, recent_wr, recent_games, overall_kda, overall_wr, overall_games]
        self.player_tensor[:, :, 1] = 3.0   # recent_kda 中性先验
        self.player_tensor[:, :, 2] = 0.5   # recent_wr 中性先验
        self.player_tensor[:, :, 4] = 3.0   # overall_kda 中性先验
        self.player_tensor[:, :, 5] = 0.5   # overall_wr 中性先验
        
        # 【修复】：使用 itertuples 提速
        for row in player_df.itertuples(index=False):
            pk = f"{row.player_id}_{row.gameid}"
            c = int(row.champion_id)
            if pk in self.pkey_to_idx and c < self.vocab_size:
                p_idx = self.pkey_to_idx[pk]
                # 使用 getattr 容错提取
                m_score = row.mastery_score
                r_kda = getattr(row, 'player_recent_kda_90d', 3.0) if pd.notna(getattr(row, 'player_recent_kda_90d', np.nan)) else 3.0
                r_wr = getattr(row, 'player_recent_wr_90d', 0.5) if pd.notna(getattr(row, 'player_recent_wr_90d', np.nan)) else 0.5
                r_games = getattr(row, 'player_recent_games_90d', 0.0)
                r_games = r_games if pd.notna(r_games) else 0.0
                o_kda = getattr(row, 'player_overall_recent_kda', 3.0) if pd.notna(getattr(row, 'player_overall_recent_kda', np.nan)) else 3.0
                o_wr = getattr(row, 'player_overall_recent_wr', 0.5) if pd.notna(getattr(row, 'player_overall_recent_wr', np.nan)) else 0.5
                o_games = getattr(row, 'player_overall_recent_games', 0.0) if pd.notna(getattr(row, 'player_overall_recent_games', np.nan)) else 0.0
                
                self.player_tensor[p_idx, c] = [m_score, r_kda, r_wr, r_games, o_kda, o_wr, o_games]

        self.grudge_tensor_blue = np.zeros((n_games, self.vocab_size), dtype=np.float32)
        self.grudge_tensor_red = np.zeros((n_games, self.vocab_size), dtype=np.float32)
        self.respect_tensor_blue = np.zeros((n_games, self.vocab_size), dtype=np.float32)
        self.respect_tensor_red = np.zeros((n_games, self.vocab_size), dtype=np.float32)
        self.streak_tensor_blue = np.zeros((n_games, self.vocab_size), dtype=np.float32)
        self.streak_tensor_red = np.zeros((n_games, self.vocab_size), dtype=np.float32)

        for i, row_dict in enumerate(self.context):
            gid = str(row_dict.get("gameid", ""))
            bt, rt = row_dict.get("blue_team", ""), row_dict.get("red_team", "")
            
            blue_grudge = self.grudge_store.get(gid, {}).get(bt, {}).get(rt, {})
            if blue_grudge:
                for cid_str, val in blue_grudge.items():
                    cid = int(cid_str)
                    if cs <= cid < ve: self.grudge_tensor_blue[i, cid] = float(val)
            red_grudge = self.grudge_store.get(gid, {}).get(rt, {}).get(bt, {})
            if red_grudge:
                for cid_str, val in red_grudge.items():
                    cid = int(cid_str)
                    if cs <= cid < ve: self.grudge_tensor_red[i, cid] = float(val)

            for side_key, resp_tensor, strk_tensor in [("blue", self.respect_tensor_blue, self.streak_tensor_blue), 
                                                       ("red", self.respect_tensor_red, self.streak_tensor_red)]:
                enemy_side = "red" if side_key == "blue" else "blue"
                match_resp = self.respect_store.get(gid, {})
                match_strk = self.hot_streak_store.get(gid, {})
                for p_short in ["top", "jng", "mid", "bot", "sup"]:
                    eid = row_dict.get(f"{enemy_side}_{p_short}_player_id", "unknown")
                    
                    rinfo = match_resp.get(eid)
                    if rinfo:
                        r_cid = int(rinfo.get("signature_champion_id", -1))
                        if cs <= r_cid < ve:
                            resp_tensor[i, r_cid] = max(resp_tensor[i, r_cid], min(float(rinfo.get("signature_mastery", 0.0)) / 100.0, 1.0))
                            
                    hs_info = match_strk.get(eid)
                    if hs_info:
                        hs_cid = int(hs_info.get("hot_champion_id", -1))
                        if cs <= hs_cid < ve:
                            streak_val = float(hs_info.get("hot_win_rate", 0.0))*0.5 + (min(float(hs_info.get("hot_avg_kda", 0.0)), 10.0)/10.0)*0.3 + (min(int(hs_info.get("hot_games", 0)), 10)/10.0)*0.2
                            strk_tensor[i, hs_cid] = max(strk_tensor[i, hs_cid], streak_val)

    def _build_position_matrix(self, json_path):
        matrix = np.zeros((self.vocab_size, 5), dtype=np.float32)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for champ_name, pos_list in data.items():
            cid = self.name_to_idx.get(champ_name, self.UNK_IDX)
            if cid < self.champion_start_idx: continue
            for item in pos_list:
                pos_str = item["position"]
                if pos_str in POS_2_IDX:
                    matrix[cid, POS_2_IDX[pos_str]] = float(item["probability"])
        return matrix

    def _precompute_match_tensors(self):
        N = len(self.context)
        ctx_dim = 24 if self.use_extended_features else 20
        
        self.global_ctx_tensor = np.zeros((N, ctx_dim), dtype=np.float32)
        self.bp_raw_tensor = np.full((N, 20), self.PAD_IDX, dtype=np.int64)
        self.bp_action_tensor = np.zeros((N, 20), dtype=np.int32)
        self.bp_side_tensor = np.zeros((N, 20), dtype=np.int32)
        self.blue_p_idx = np.full((N, 5), self.UNKNOWN_PKEY_IDX, dtype=np.int32)
        self.red_p_idx = np.full((N, 5), self.UNKNOWN_PKEY_IDX, dtype=np.int32)
        self.time_weights_tensor = np.ones(N, dtype=np.float32)

        for i, row in enumerate(self.context):
            league = str(row.get("league", "LPL"))
            l_idx = LEAGUES.index(league) if league in LEAGUES else 0
            self.global_ctx_tensor[i, l_idx] = 1.0

            self.global_ctx_tensor[i, 3:13] = [
                row.get("blue_team_avg_ckpm", 0.7), row.get("blue_team_avg_golddiffat15", 0),
                row.get("blue_team_avg_gamelength", 1900), row.get("blue_team_firstdragon_rate", 0.5),
                row.get("blue_team_firsttower_rate", 0.5), row.get("red_team_avg_ckpm", 0.7),
                row.get("red_team_avg_golddiffat15", 0), row.get("red_team_avg_gamelength", 1900),
                row.get("red_team_firstdragon_rate", 0.5), row.get("red_team_firsttower_rate", 0.5),
            ]
            self.global_ctx_tensor[i, 13] = float(row.get("playoffs", 0))
            self.global_ctx_tensor[i, 14] = float(row.get("first_pick_map_side", 1))
            self.global_ctx_tensor[i, 15:20] = [
                float(row.get(f"is_game_{g}", 0)) for g in range(1, 6)
            ]

            if self.use_extended_features:
                patch_norm = (float(row.get("patch", 15.0)) - 14.0) / 2.0
                split_val = {"Spring": 0, "Split 1": 0, "Summer": 1, "Split 2": 1, "Split 3": 2}.get(str(row.get("split", "")), 0)
                year_norm = (float(row.get("year", 2025)) - 2023) / 3.0
                b_pg, r_pg = float(row.get("blue_team_profile_games", 0)), float(row.get("red_team_profile_games", 0))
                profile_norm = np.log1p(b_pg + r_pg) / 10.0
                self.global_ctx_tensor[i, 20:24] = [patch_norm, float(split_val), year_norm, profile_norm]

            for step in range(20):
                self.bp_raw_tensor[i, step] = row.get(f"bp_step{step}_champion_id", self.PAD_IDX)
                self.bp_action_tensor[i, step] = row.get(f"bp_step{step}_action", 0)
                self.bp_side_tensor[i, step] = row.get(f"bp_step{step}_side", 0)

            gid = str(row.get('gameid', ''))
            for p_i, p in enumerate(["top", "jng", "mid", "bot", "sup"]):
                self.blue_p_idx[i, p_i] = self.pkey_to_idx.get(f"{row.get(f'blue_{p}_player_id', 'unknown')}_{gid}", self.UNKNOWN_PKEY_IDX)
                self.red_p_idx[i, p_i] = self.pkey_to_idx.get(f"{row.get(f'red_{p}_player_id', 'unknown')}_{gid}", self.UNKNOWN_PKEY_IDX)

            # 【核心修复】：测试/验证集的时间权重强行锁定 1.0 杜绝泄露
            if not self.is_train:
                self.time_weights_tensor[i] = 1.0
            else:
                current_date = pd.to_datetime(row.get("date"))
                delta_days = max((self.anchor_date - current_date).days, 0)
                time_w = max(float(np.exp(-np.log(2) * (delta_days / 21.0))), 0.05)
                league_w = LEAGUE_WEIGHTS.get(league, 0.8)
                self.time_weights_tensor[i] = time_w * league_w

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        match_idx, fixed_step = self.sample_list[idx]
        
        if fixed_step is None:
            target_step = np.random.randint(0, len(BP_SEQUENCE))
        else:
            target_step = fixed_step

        raw_full_seq = self.bp_raw_tensor[match_idx].tolist()
        bp_seq = raw_full_seq[:target_step]
        padded_seq = np.array(bp_seq + [self.PAD_IDX] * (20 - len(bp_seq)), dtype=np.int64)
        target_label = raw_full_seq[target_step]
        
        global_context = self.global_ctx_tensor[match_idx]

        action_type, side_str, slot = BP_SEQUENCE[target_step]
        is_pick_action = 1.0 if action_type == "pick" else 0.0
        current_side_code = 0 if side_str == "blue" else 1

        ban_step_number = sum(1 for i in range(target_step + 1) if BP_SEQUENCE[i][0] == "ban" and BP_SEQUENCE[i][1] == side_str)

        banned_ids = set()
        ally_champs = []
        enemy_champs = []
        
        history_positions = np.full(20, -1, dtype=np.int64)
        champ_to_pos_idx = {}
        row = self.context[match_idx] 
        for s_side in ["blue", "red"]:
            for p_short, p_full in zip(["top", "jng", "mid", "bot", "sup"], ["top", "jungle", "mid", "bot", "support"]):
                col_name = f"{s_side}_{p_short}_champion_id" 
                if col_name in row:
                    c_id = row[col_name]
                    if c_id >= self.champion_start_idx:
                        champ_to_pos_idx[c_id] = POS_2_IDX[p_full]

        for i in range(target_step):
            cid = bp_seq[i]
            if cid < self.champion_start_idx: continue
            
            act = self.bp_action_tensor[match_idx, i]
            side = self.bp_side_tensor[match_idx, i]

            if act == 1 and cid in champ_to_pos_idx:
                history_positions[i] = champ_to_pos_idx[cid]
                
            if act == 0:
                banned_ids.add(cid)
            elif side == current_side_code:
                ally_champs.append(cid)
            else:
                enemy_champs.append(cid)

        # 安全防泄露检查
        if ban_step_number < 3:
            assert len(ally_champs) == 0, f"Data leakage! ban_step={ban_step_number} but ally_pick_count={len(ally_champs)} > 0."

        unavailable_ids = banned_ids | set(ally_champs) | set(enemy_champs)

        ally_pos_sum = np.sum([self.position_prior_matrix[c] for c in ally_champs], axis=0) if ally_champs else np.zeros(5)
        enemy_pos_sum = np.sum([self.position_prior_matrix[c] for c in enemy_champs], axis=0) if enemy_champs else np.zeros(5)
        ally_missing_roles = np.clip(1.0 - ally_pos_sum, 0.0, 1.0)
        enemy_missing_roles = np.clip(1.0 - enemy_pos_sum, 0.0, 1.0)

        
        FI = CANDIDATE_FEAT_MAP
        # 矩阵始终 33 维（CANDIDATE_DIM），player_recent_games@idx7, is_fearless_banned@idx32
        # use_extended_features 仅控制是否写入 last_ally_synergy@idx31
        N_FEAT = EXTENDED_CANDIDATE_DIM  # 33 维
        candidate_matrix = np.zeros((self.vocab_size, N_FEAT), dtype=np.float32)
        candidate_matrix[:, FI["pos_top"]:FI["pos_sup"]+1] = self.position_prior_matrix

        cs, ve = self.champion_start_idx, self.vocab_size
        
        g_idx = self.gameid_to_idx.get(str(row.get('gameid', '')), -1)
        if g_idx >= 0:
            candidate_matrix[cs:ve, FI["meta_pick"]:FI["meta_wr"]+1] = self.meta_tensor[g_idx, cs:ve]
        else:
            candidate_matrix[cs:ve, FI["meta_pick"]:FI["meta_wr"]+1] = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)

        ally_p_indices = self.blue_p_idx[match_idx] if side_str == "blue" else self.red_p_idx[match_idx]
        enemy_p_indices = self.red_p_idx[match_idx] if side_str == "blue" else self.blue_p_idx[match_idx]

        ally_features_matrix = self.player_tensor[ally_p_indices, cs:ve, :]
        weighted_ally_features = ally_features_matrix * ally_missing_roles[:, None, None]
        # 切片 4:11 包含全部 7 个 player 特征 (mastery, recent_kda, recent_wr, recent_games, overall_kda, overall_wr, overall_games)
        candidate_matrix[cs:ve, FI["player_mastery"]:FI["player_overall_games"]+1] = weighted_ally_features.max(axis=0)

        if ally_champs:
            candidate_matrix[cs:ve, FI["ally_synergy"]] = np.max(self.synergy_matrix[cs:ve, ally_champs], axis=1)
            candidate_matrix[cs:ve, FI["ally_counter"]] = np.max(1.0 - self.counter_matrix[cs:ve, ally_champs], axis=1)
        if enemy_champs:
            candidate_matrix[cs:ve, FI["enemy_counter"]] = np.max(1.0 - self.counter_matrix[cs:ve, enemy_champs], axis=1)
            candidate_matrix[cs:ve, FI["enemy_synergy"]] = np.max(self.synergy_matrix[cs:ve, enemy_champs], axis=1)

        pos_block = candidate_matrix[cs:ve, FI["pos_top"]:FI["pos_sup"]+1]
        candidate_matrix[cs:ve, FI["ally_role_fit"]] = pos_block @ ally_missing_roles
        candidate_matrix[cs:ve, FI["enemy_role_fit"]] = pos_block @ enemy_missing_roles
        candidate_matrix[cs:ve, FI["is_pick"]] = is_pick_action
        
        enemy_mastery_matrix = self.player_tensor[enemy_p_indices, cs:ve, 0]
        weighted_enemy_mastery = enemy_mastery_matrix * enemy_missing_roles[:, None]
        candidate_matrix[cs:ve, FI["enemy_mastery_max"]] = weighted_enemy_mastery.max(axis=0)
        candidate_matrix[cs:ve, FI["enemy_mastery_mean"]] = weighted_enemy_mastery.mean(axis=0)

        candidate_matrix[cs:ve, FI["ban_step"]] = float(ban_step_number)

        if g_idx >= 0:
            if side_str == "blue":
                candidate_matrix[cs:ve, FI["grudge"]] = self.grudge_tensor_blue[g_idx, cs:ve]
                candidate_matrix[cs:ve, FI["respect"]] = self.respect_tensor_blue[g_idx, cs:ve]
                candidate_matrix[cs:ve, FI["hot_streak"]] = self.streak_tensor_blue[g_idx, cs:ve]
            else:
                candidate_matrix[cs:ve, FI["grudge"]] = self.grudge_tensor_red[g_idx, cs:ve]
                candidate_matrix[cs:ve, FI["respect"]] = self.respect_tensor_red[g_idx, cs:ve]
                candidate_matrix[cs:ve, FI["hot_streak"]] = self.streak_tensor_red[g_idx, cs:ve]

        candidate_matrix[cs:ve, FI["n_ally_picked"]] = float(len(ally_champs))
        candidate_matrix[cs:ve, FI["is_red_side"]] = float(current_side_code)

        # [31] is_fearless_banned: 前置局已使用英雄标记为不可选 (Global BP / Fearless Draft)
        prev_champs_str = str(row.get('prev_game_champs', ''))
        if prev_champs_str and prev_champs_str != 'nan' and prev_champs_str != '':
            for champ_name in prev_champs_str.split('|'):
                champ_name = champ_name.strip()
                if champ_name:
                    cid = self.name_to_idx.get(champ_name, -1)
                    if cs <= cid < ve:
                        # 使用统一字典的 Key
                        candidate_matrix[cid, FI["is_fearless_banned"]] = 1.0
                        unavailable_ids.add(cid)

        if self.use_extended_features:
            if ally_champs:
                candidate_matrix[cs:ve, FI["last_ally_synergy"]] = self.synergy_matrix[cs:ve, ally_champs[-1]]
            else:
                candidate_matrix[cs:ve, FI["last_ally_synergy"]] = 0.5

        if self.is_train and not self.force_unroll:
            noise = np.random.normal(loc=0.0, scale=0.02, size=(self.vocab_size, 4)).astype(np.float32)
            # CS 特征列: ally_synergy(15) ~ ally_counter(18)
            cs_slice = slice(FI["ally_synergy"], FI["ally_counter"] + 1)
            candidate_matrix[:, cs_slice] = np.clip(candidate_matrix[:, cs_slice] + noise, 0.0, 1.0)

        available_mask = np.ones(self.vocab_size, dtype=np.float32)
        available_mask[:self.champion_start_idx] = 0.0
        for uid in unavailable_ids:
            if 0 <= uid < self.vocab_size: available_mask[uid] = 0.0

        if 0 <= target_label < self.vocab_size:
            available_mask[target_label] = 1.0

        # 【Zero-Copy】极速转化为张量
        return {
            "global_context": torch.as_tensor(global_context, dtype=torch.float32),
            "bp_sequence": torch.as_tensor(padded_seq, dtype=torch.long),
            "candidate_matrix": torch.as_tensor(candidate_matrix, dtype=torch.float32),
            "available_mask": torch.as_tensor(available_mask, dtype=torch.float32),
            "label": torch.as_tensor(target_label, dtype=torch.long),
            "history_positions": torch.as_tensor(history_positions, dtype=torch.long), 
            "is_pick": torch.as_tensor(is_pick_action, dtype=torch.float32),
            "time_weight": torch.as_tensor(self.time_weights_tensor[match_idx], dtype=torch.float32), 
            "bp_step": torch.as_tensor(target_step, dtype=torch.long),
        }

def create_train_val_dataloaders(context_parquet, meta_parquet, player_parquet,
                                 vocab_path, position_json_path,
                                 batch_size=32, num_workers=0, val_ratio=0.15,
                                 force_unroll_train=False,
                                 use_extended_features=False):  # 不写 last_ally_synergy，与修改前一致
    log.info(f"Loading Parquet files (force_unroll_train={force_unroll_train}, use_extended_features={use_extended_features})...")
    full_context_df = pd.read_parquet(context_parquet)
    meta_df = pd.read_parquet(meta_parquet)
    player_df = pd.read_parquet(player_parquet)

    ctx_cols = EXTENDED_CONTEXT_COLS if use_extended_features else BAN_CONTEXT_COLS
    full_context_df = _truncate_columns(full_context_df, ctx_cols, "Context")
    meta_df = _truncate_columns(meta_df, BAN_META_COLS, "MetaStore")
    player_df = _truncate_columns(player_df, BAN_PLAYER_COLS, "PlayerStore")

    full_context_df = full_context_df.sort_values(by="match_seq_idx").reset_index(drop=True)
    
    n_total = len(full_context_df)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_context_df = full_context_df.iloc[:n_train].reset_index(drop=True)
    val_context_df = full_context_df.iloc[n_train:].reset_index(drop=True)

    log.info(f"Data Split -> Train: {n_train} games ({100-val_ratio*100:.0f}%) | Val: {n_val} games ({val_ratio*100:.0f}%)")

    features_dir = os.path.dirname(context_parquet)
    league_prefix = os.path.basename(context_parquet).split('_')[0]

    def load_json_safe(filename):
        path = os.path.join(features_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    counter_dict = load_json_safe(f"{league_prefix}_counter_lookup.json")
    synergy_dict = load_json_safe(f"{league_prefix}_synergy_lookup.json")
    grudge_store = load_json_safe(f"{league_prefix}_grudge_store.json")
    respect_store = load_json_safe(f"{league_prefix}_respect_store.json")
    hot_streak_store = load_json_safe(f"{league_prefix}_hot_streak_store.json")

    train_dataset = BPRecommendationDataset(
        train_context_df, player_df, meta_df,
        counter_dict, synergy_dict, vocab_path, position_json_path,
        grudge_store=grudge_store, respect_store=respect_store, hot_streak_store=hot_streak_store,
        is_train=True, force_unroll=force_unroll_train,
        use_extended_features=use_extended_features
    )

    val_dataset = BPRecommendationDataset(
        val_context_df, player_df, meta_df,
        counter_dict, synergy_dict, vocab_path, position_json_path,
        grudge_store=grudge_store, respect_store=respect_store, hot_streak_store=hot_streak_store,
        is_train=False,
        anchor_date=train_dataset.anchor_date,
        use_extended_features=use_extended_features
    )
    
    shuffle_train = not force_unroll_train
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader