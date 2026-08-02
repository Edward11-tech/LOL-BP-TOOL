"""
线上/线下特征强一致性校验工具
==============================
对比离线特征流水线 (feature_pipeline.py) 生成的 parquet 数据与
在线特征构建器 (feature_builder.py) 实时计算的特征值，确保两者完全一致。

校验逻辑:
  1. 从 ALL_prediction_wide_features.parquet 随机采样比赛样本
  2. 使用 PredictBackend 在线重建相同比赛的特征向量
  3. 逐特征对比线上/线下数值差异（跳过 TF 特征，因 TF 快照可能不同）
  4. 输出差异超过 1e-4 的特征列表

主要函数:
  - run_alignment_test(num_samples=5): 执行一致性校验

用法:
  python bp_prediction/check_feature_alignment.py
"""
import os
import sys

# ---- 必须在导入 LightGBM 之前设置，防止 macOS 上 OpenMP 死锁 ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from bp_prediction.predict_backend import PredictBackend
from logger_config import get_logger, setup_logging


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"check_feature_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

log = get_logger(__name__)


def run_alignment_test(num_samples=5):
    log.info("=" * 60)
    log.info("开始进行线上/线下特征强一致性校验 (v2 精确时间版)")
    log.info("=" * 60)

    backend = PredictBackend()
    backend.load()
    feature_cols = backend.feature_cols

    parquet_path = os.path.join("bp_prediction", "features", "ALL_prediction_wide_features.parquet")
    offline_df = pd.read_parquet(parquet_path)
    offline_df["date_str"] = pd.to_datetime(offline_df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    sample_df = offline_df.sample(n=num_samples, random_state=42)
    positions = ["top", "jng", "mid", "bot", "sup"]

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        gameid = row.get("gameid", "Unknown_GameID")
        match_date = row.get("date_str")
        log.info("--- 校验第 %d/%d 场 (GameID: %s, 日期: %s) ---",
                 idx, num_samples, gameid, match_date)

        current_game_num = 1
        for i in range(1, 6):
            if row.get(f"is_game_{i}", 0) == 1:
                current_game_num = i
                break

        mock_request = {
            "date": match_date,
            "league": row.get("league", "LCK"),
            "is_playoff": bool(row.get("is_playoff", 0)),
            "game_num": current_game_num,
            "first_pick": "blue" if row.get("is_blue_map_side", 1) == 1 else "red",
            "blue_team": row.get("blue_team", ""),
            "red_team": row.get("red_team", ""),
            "blue_champions": {pos: row.get(f"blue_{pos}_champion", "") for pos in positions},
            "red_champions": {pos: row.get(f"red_{pos}_champion", "") for pos in positions},
            "blue_players": {pos: row.get(f"blue_{pos}_player_id", "") for pos in positions},
            "red_players": {pos: row.get(f"red_{pos}_player_id", "") for pos in positions},
        }

        try:
            match_info = backend._build_match_info(mock_request)
            online_features_df, _ = backend._build_features(match_info)
        except Exception as e:
            log.exception("生成失败 (GameID: %s): %s", gameid, e)
            continue

        online_vector = online_features_df.iloc[0]

        mismatch_for_this_game = 0
        for col in feature_cols:
            if col.startswith("tf_"):
                continue

            off_val = float(row.get(col, 0.0)) if not pd.isna(row.get(col)) else 0.0
            on_val = float(online_vector.get(col, 0.0)) if not pd.isna(online_vector.get(col)) else 0.0

            diff = abs(off_val - on_val)
            if diff > 1e-4:
                log.info("  -> %-35s | 线下: %8.4f | 线上: %8.4f | 差值: %8.4f",
                         col, off_val, on_val, diff)
                mismatch_for_this_game += 1

        if mismatch_for_this_game == 0:
            log.info("完美对齐 (GameID: %s)", gameid)


if __name__ == "__main__":
    setup_logging(log_dir=LOG_DIR, app_name="check_feature_alignment",
                  console_level=logging.INFO, file_level=logging.DEBUG)
    run_alignment_test(num_samples=5)
