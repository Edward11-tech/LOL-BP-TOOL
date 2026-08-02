"""
data_pipeline.py — Fallback 数据加载器
======================================
从 cleaned_data/ 直接加载 Meta 统计和选手熟练度数据，供规则引擎使用。

设计原则:
  - 不从互联网下载任何数据
  - 不检测数据文件是否存在 (默认部署包内一定有 cleaned_data)
  - 直接复用清洗后的高质量数据 (含贝叶斯平滑, 优于 raw value_counts)
  - 模块级缓存避免重复 IO

数据来源:
  - cleaned_data/merged_champion_stats.csv → meta_stats (英雄大盘统计)
  - cleaned_data/player_career_hero_stats_cleaned.csv → player_stats (选手熟练度)
"""

import os
import sys
import threading
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))
from logger_config import get_logger
from common.paths import MERGED_CHAMPION_STATS_CSV, PLAYER_CAREER_STATS_CSV

log = get_logger(__name__)

# 模块级缓存 (首次调用后不再读 CSV)
_META_CACHE: dict = {}
_PLAYER_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _build_meta_stats_from_cleaned() -> dict:
    """从 merged_champion_stats.csv 构造 meta_stats 字典。

    优先使用贝叶斯平滑后的观测值 (xxx_obs 字段)，质量高于 raw value_counts。

    Returns:
        dict: {champion_name: {meta_pick_rate, meta_ban_rate, meta_presence, meta_win_rate, total_games}}
    """
    df = pd.read_csv(MERGED_CHAMPION_STATS_CSV)
    # 数值列填充 NaN
    num_cols = ["pick_rate_obs", "pick_rate", "ban_rate_obs", "ban_rate",
                "presence_rate_obs", "presence_rate", "win_rate_obs", "win_rate",
                "games_obs", "total_games_pro"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    meta = {}
    for _, row in df.iterrows():
        champ = str(row.get("champion", "")).strip()
        if not champ or champ.lower() == "nan":
            continue

        # 优先用贝叶斯平滑后的观测值，回退到原始值
        def _pick(obs_key, raw_key, default=0.0):
            v = row.get(obs_key, 0)
            if v == 0:
                v = row.get(raw_key, default)
            return float(v)

        meta[champ] = {
            "meta_pick_rate": round(_pick("pick_rate_obs", "pick_rate"), 6),
            "meta_ban_rate": round(_pick("ban_rate_obs", "ban_rate"), 6),
            "meta_presence": round(_pick("presence_rate_obs", "presence_rate"), 6),
            "meta_win_rate": round(_pick("win_rate_obs", "win_rate", 0.5), 6),
            "total_games": int(_pick("games_obs", "total_games_pro", 0)),
        }

    log.info(f"从 cleaned_data 加载 Meta 统计: {len(meta)} 英雄")
    return meta


def _build_player_stats_from_cleaned() -> dict:
    """从 player_career_hero_stats_cleaned.csv 构造 player_stats 字典。

    mastery_score 计算公式 (与原实现保持一致):
        games_factor = min(games / 20.0, 1.0)
        kda_factor = min(kda / 5.0, 1.0)
        mastery = 10 * (0.4 * games_factor + 0.3 * win_rate + 0.3 * kda_factor)

    Returns:
        dict: {player_id: {champion: {mastery_score, games_played, win_rate, recent_kda, ...}}}
    """
    df = pd.read_csv(PLAYER_CAREER_STATS_CSV)
    df["games"] = pd.to_numeric(df.get("games", 0), errors="coerce").fillna(0).astype(int)
    df["win_rate"] = pd.to_numeric(df.get("win_rate", 0), errors="coerce").fillna(0)
    df["kda"] = pd.to_numeric(df.get("kda", 3.0), errors="coerce").fillna(3.0)

    players = {}
    for _, row in df.iterrows():
        pid = str(row.get("player_id", "")).strip()
        champ = str(row.get("champion", "")).strip()
        if not pid or pid.lower() == "nan" or not champ or champ.lower() == "nan":
            continue

        games = int(row["games"])
        wr = float(row["win_rate"])
        kda = float(row["kda"])

        games_factor = min(games / 20.0, 1.0)
        kda_factor = min(kda / 5.0, 1.0)
        mastery = 10 * (0.4 * games_factor + 0.3 * wr + 0.3 * kda_factor)

        players.setdefault(pid, {})[champ] = {
            "mastery_score": round(mastery, 2),
            "games_played": games,
            "win_rate": round(wr, 4),
            "recent_kda": round(kda, 2),
            "avg_kills": 0.0,
            "avg_deaths": 0.0,
            "avg_assists": 0.0,
        }

    log.info(f"从 cleaned_data 加载选手统计: {len(players)} 选手")
    return players


def load_cleaned_meta() -> dict:
    """加载英雄 Meta 统计 (带模块级缓存)"""
    if _META_CACHE:
        return _META_CACHE
    with _CACHE_LOCK:
        if not _META_CACHE:
            _META_CACHE.update(_build_meta_stats_from_cleaned())
    return _META_CACHE


def load_cleaned_players() -> dict:
    """加载选手英雄熟练度统计 (带模块级缓存)"""
    if _PLAYER_CACHE:
        return _PLAYER_CACHE
    with _CACHE_LOCK:
        if not _PLAYER_CACHE:
            _PLAYER_CACHE.update(_build_player_stats_from_cleaned())
    return _PLAYER_CACHE


def refresh_cache() -> None:
    """清空缓存，下次调用 load_* 时重新从 CSV 读取"""
    with _CACHE_LOCK:
        _META_CACHE.clear()
        _PLAYER_CACHE.clear()
    log.info("Fallback 数据缓存已清空")
