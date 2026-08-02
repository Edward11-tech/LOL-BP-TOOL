"""
rule_engine.py — Rule-based Fallback 纯规则引擎
================================================
当深度模型触发兜底条件时，使用纯规则引擎进行 BP 推荐。

核心公式:
  Fallback_Score = 0.6 * (Meta_Presence) + 0.4 * (Player_Mastery)

Pick 阶段:
  1. 取交集候选池: 当前小分路中 meta_presence > 0.05 且选手历史玩过的英雄
  2. 暴力打分: 加权求和
  3. 返回排序后的 Top-20 推荐

Ban 阶段:
  1. 优先 Ban 对方历史熟练度排名前 3 且大盘胜率 > 50% 的英雄
  2. 如果条件不满足，回退到对方 meta_presence 最高的英雄
"""

import sys
import logging
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
from logger_config import get_logger

from fallback.data_pipeline import load_cleaned_meta, load_cleaned_players

log = get_logger(__name__)

# 位置名称映射
POS_CN = {"top": "上单", "jng": "打野", "mid": "中单", "bot": "ADC", "sup": "辅助"}
POSITION_LIST = ["top", "jng", "mid", "bot", "sup"]

# 阈值
META_PRESENCE_THRESHOLD = 0.05   # 大盘登场率门槛
BAN_ENEMY_WR_THRESHOLD = 0.50    # Ban 对手英雄的胜率门槛
SCORE_WEIGHT_META = 0.6          # Meta Presence 权重
SCORE_WEIGHT_MASTERY = 0.4       # Player Mastery 权重


class RuleBasedEngine:
    """
    纯规则引擎 — 不依赖任何深度学习模型。

    数据来源:
      - meta_stats: 英雄大盘统计 (来自清洗后的比赛数据)
      - player_stats: 选手英雄熟练度 (来自清洗后的比赛数据)
      - feature_store: 备用，提供 pos_prior 和 champion vocabulary
    """

    def __init__(self, meta_stats=None, player_stats=None, feature_store=None):
        """
        Args:
            meta_stats: dict, 英雄 Meta 统计
            player_stats: dict, 选手英雄熟练度统计
            feature_store: PredictFeatureStore 实例, 提供 idx_to_name, name_to_idx, pos_prior
        """
        self.meta_stats = meta_stats or load_cleaned_meta()
        self.player_stats = player_stats or load_cleaned_players()
        self.store = feature_store

        log.info(f"规则引擎初始化: {len(self.meta_stats)} 英雄, {len(self.player_stats)} 选手")

    def refresh_data(self):
        """重新加载数据 (清空缓存后从 cleaned_data 重新读取)"""
        from fallback.data_pipeline import refresh_cache
        refresh_cache()
        self.meta_stats = load_cleaned_meta()
        self.player_stats = load_cleaned_players()
        log.info(f"规则引擎数据刷新: {len(self.meta_stats)} 英雄, {len(self.player_stats)} 选手")

    def recommend_pick(self, position, ally_pids, enemy_pids,
                       ally_champs, enemy_champs, unavail_set):
        # 移除之前的 target_pid 获取逻辑，直接把所有 ally_pids 传进底层计算
        if not ally_pids:
            log.warning("Pick 推荐: 没有友方选手信息")
            return self._fallback_pick_no_player(position, unavail_set)

        # 把真实的 ally_pids 列表传进去，让引擎自己找最高熟练度
        candidates = self._compute_pick_candidates(position, ally_pids, unavail_set)

        if not candidates:
            log.warning(f"Pick 推荐: 无候选英雄，使用全局回退")
            return self._fallback_pick_no_player(position, unavail_set)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(candidates[:50]):
            c["rank"] = i + 1
        return candidates[:50]

    def recommend_ban(self, position, ally_pids, enemy_pids,
                      ally_champs, enemy_champs, unavail_set):
        """
        Ban 阶段规则推荐。

        策略:
          1. 找到对方该位置选手 (enemy_pids[position])
          2. 找出对方熟练度最高的英雄
          3. 过滤: 大盘胜率 > 50%, 且不在 unavail_set 中
          4. 取 Top-3 作为推荐

        Args:
            position: str, 当前需要 Ban 的位置 (top/jng/mid/bot/sup)
            ally_pids: list[str], 友方选手 ID 列表
            enemy_pids: list[str], 敌方选手 ID 列表
            ally_champs: list[int], 已选友方英雄索引
            enemy_champs: list[int], 已选敌方英雄索引
            unavail_set: set[int], 不可选英雄索引集合

        Returns:
            list[dict]: 推荐列表
        """
        # 找到对方该位置选手
        target_pid = None
        if position in POSITION_LIST:
            pos_idx = POSITION_LIST.index(position)
            if pos_idx < len(enemy_pids):
                target_pid = enemy_pids[pos_idx]

        if target_pid is None:
            log.warning("Ban 推荐: 没有敌方选手信息，使用全局回退")
            return self._fallback_ban_global(unavail_set)

        # 获取对方选手的英雄熟练度
        player_heroes = self.player_stats.get(target_pid, {})
        if not player_heroes:
            log.warning(f"Ban 推荐: 选手 {target_pid} 无历史数据")
            return self._fallback_ban_global(unavail_set)

        # 计算 Ban 候选
        candidates = []
        for champ_name, stats in player_heroes.items():
            mastery = stats.get("mastery_score", 0)
            wr = stats.get("win_rate", 0)

            # 获取该英雄的 Meta 信息
            meta = self.meta_stats.get(champ_name, {})
            meta_wr = meta.get("meta_win_rate", 0.5)
            meta_presence = meta.get("meta_presence", 0)

            # 判断是否可 Ban
            champion_idx = self._get_champion_idx(champ_name)
            if champion_idx is None:
                continue
            if champion_idx in unavail_set:
                continue

            # Ban 条件: 对方熟练度高 + 大盘胜率 > 50%
            ban_score = 0.5 * mastery / 10.0 + 0.5 * meta_wr

            candidates.append({
                "champion": champ_name,
                "champion_idx": champion_idx,
                "score": round(ban_score, 4),
                "meta_presence": round(meta_presence, 4),
                "meta_win_rate": round(meta_wr, 4),
                "enemy_mastery": round(mastery, 2),
                "enemy_win_rate": round(wr, 4),
                "enemy_games": stats.get("games_played", 0),
                "reasons": [f"对方选手{target_pid}高熟练度({mastery:.1f})"],
            })

        # 排序: 按 ban_score 降序
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 取 Top-3 满足 Ban 条件的
        result = [c for c in candidates if c["meta_win_rate"] >= BAN_ENEMY_WR_THRESHOLD][:3]

        if not result:
            # 回退: 取对方熟练度最高的 3 个
            result = candidates[:3]

        for i, c in enumerate(result):
            c["rank"] = i + 1

        return result

    # ---- 内部方法 ----

    def _compute_pick_candidates(self, position, ally_pids, unavail_set):
        candidates = []
        
        for champ_name, meta in self.meta_stats.items():
            meta_presence = meta.get("meta_presence", 0)
            if meta_presence < META_PRESENCE_THRESHOLD:
                continue

            champion_idx = self._get_champion_idx(champ_name)
            if champion_idx is None or champion_idx in unavail_set:
                continue

            if self.store is not None and position in POSITION_LIST:
                pos_prior = self.store.pos_prior[champion_idx]
                pos_idx = POSITION_LIST.index(position)
                if pos_prior[pos_idx] < 0.05:  
                    continue

            # 【核心修复】：遍历全队，找出对这个英雄熟练度最高的人
            best_mastery_score = 0.0
            best_pid = None
            for pid in ally_pids:
                if pid == "unknown" or not pid: continue
                p_score = self.player_stats.get(pid, {}).get(champ_name, {}).get("mastery_score", 0)
                if p_score > best_mastery_score:
                    best_mastery_score = p_score
                    best_pid = pid

            # 如果全队都没玩过，给一个基础分
            if best_mastery_score == 0:
                best_mastery_score = min(meta_presence * 20, 5.0)

            mastery_norm = best_mastery_score / 10.0
            score = SCORE_WEIGHT_META * meta_presence + SCORE_WEIGHT_MASTERY * mastery_norm

            reasons = []
            if meta_presence > 0.2:
                reasons.append(f"版本强势({meta_presence*100:.0f}%)")
            if best_mastery_score > 5 and best_pid:
                reasons.append(f"选手绝活(熟练度{best_mastery_score:.1f})") # 甚至不需要说是谁，只要队里有人会玩就行
            if meta.get("meta_win_rate", 0) > 0.52:
                reasons.append(f"高胜率({meta.get('meta_win_rate', 0)*100:.0f}%)")

            candidates.append({
                "champion": champ_name,
                "champion_idx": champion_idx,
                "score": round(score, 4),
                "meta_presence": round(meta_presence, 4),
                "meta_win_rate": round(meta.get("meta_win_rate", 0.5), 4),
                "mastery_score": round(best_mastery_score, 2),
                "reasons": reasons,
                "rank": 0,
            })
        return candidates


    def _fallback_pick_no_player(self, position, unavail_set):
        """无选手信息时的 Pick 回退: 按 meta_presence 排序"""
        candidates = []
        for champ_name, meta in self.meta_stats.items():
            meta_presence = meta.get("meta_presence", 0)
            if meta_presence < META_PRESENCE_THRESHOLD:
                continue

            champion_idx = self._get_champion_idx(champ_name)
            if champion_idx is None or champion_idx in unavail_set:
                continue

            if self.store is not None:
                pos_prior = self.store.pos_prior[champion_idx]
                pos_idx = POSITION_LIST.index(position) if position in POSITION_LIST else 0
                if pos_prior[pos_idx] < 0.05:
                    continue

            candidates.append({
                "champion": champ_name,
                "champion_idx": champion_idx,
                "score": round(meta_presence, 4),
                "meta_presence": round(meta_presence, 4),
                "meta_win_rate": round(meta.get("meta_win_rate", 0.5), 4),
                "mastery_score": 0,
                "reasons": ["版本强势(无选手数据)"],
                "rank": 0,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(candidates[:50]):
            c["rank"] = i + 1
        return candidates[:50]

    def _fallback_ban_global(self, unavail_set):
        """无选手信息时的 Ban 回退: 按 meta_presence + meta_win_rate 排序"""
        candidates = []
        for champ_name, meta in self.meta_stats.items():
            meta_presence = meta.get("meta_presence", 0)
            meta_wr = meta.get("meta_win_rate", 0.5)

            champion_idx = self._get_champion_idx(champ_name)
            if champion_idx is None or champion_idx in unavail_set:
                continue

            score = 0.6 * meta_presence + 0.4 * meta_wr
            candidates.append({
                "champion": champ_name,
                "champion_idx": champion_idx,
                "score": round(score, 4),
                "meta_presence": round(meta_presence, 4),
                "meta_win_rate": round(meta_wr, 4),
                "enemy_mastery": 0,
                "enemy_win_rate": 0,
                "enemy_games": 0,
                "reasons": ["版本强势英雄(Ban全局回退)"],
                "rank": 0,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(candidates[:50]):
            c["rank"] = i + 1
        return candidates[:50]

    def _get_champion_idx(self, champ_name):
        """通过 store 获取英雄索引"""
        if self.store is None:
            return None
        idx = self.store.name_to_idx.get(champ_name)
        if idx is None or idx < self.store.champion_start_idx:
            return None
        return idx

    def _get_champion_name(self, champion_idx):
        """通过 store 获取英雄名称"""
        if self.store is None:
            return None
        return self.store.idx_to_name.get(str(champion_idx))