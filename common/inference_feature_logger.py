"""
推理特征日志组件
=================
记录每次推理请求的实际特征向量，供周度 PSI 漂移分析使用。

存储格式:
    logs/inference_features/prediction_YYYYMMDD.parquet
        列: request_id, timestamp, league, feature_1, feature_2, ...
    logs/inference_features/recommendation_YYYYMMDD.parquet
        列: request_id, timestamp, step_type,
            candidate_feature_1, ..., context_feature_1, ...

设计原则:
    1. 失败不影响主业务 (与 _log_tracking 一致，try/except 全包裹)
    2. 线程安全 (Flask 多线程，使用 threading.Lock)
    3. 采样率可配置 (环境变量 INFERENCE_FEATURE_SAMPLE_RATE，默认 1.0 全量记录)
    4. read-modify-write 追加策略 (低流量场景足够，单次 <50ms)

使用方式:
    from common.inference_feature_logger import (
        log_prediction_features,
        log_recommendation_features,
    )

    # 预测路径
    log_prediction_features(features_df=features_df, league="LCK")

    # 推荐路径
    log_recommendation_features(
        cand_np=cand_np,
        global_context=global_context_np,
        mask_np=mask_np,
        step_type="pick",
    )
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from logger_config import get_logger

_log = get_logger(__name__)

# 日志根目录：项目根目录下的 logs/inference_features/
_LOG_DIR = Path(__file__).parent.parent / "logs" / "inference_features"

# 全局锁：保护 parquet 文件的 read-modify-write 追加
_LOCK = threading.Lock()

# 采样率：环境变量控制，默认 1.0 (全量记录)
_SAMPLE_RATE = float(os.environ.get("INFERENCE_FEATURE_SAMPLE_RATE", "1.0"))


def _should_sample() -> bool:
    """根据采样率决定是否记录本次请求"""
    if _SAMPLE_RATE >= 1.0:
        return True
    if _SAMPLE_RATE <= 0.0:
        return False
    return np.random.random() < _SAMPLE_RATE


def _append_parquet(path: Path, new_df: pd.DataFrame) -> None:
    """线程安全地追加 parquet 文件 (read-modify-write)

    Args:
        path: parquet 文件路径
        new_df: 要追加的新数据
    """
    with _LOCK:
        try:
            if path.exists():
                old_df = pd.read_parquet(path)
                combined = pd.concat([old_df, new_df], ignore_index=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                combined = new_df
            combined.to_parquet(path, index=False)
        except Exception:
            # 日志失败不能影响主业务
            _log.debug("特征日志写入失败: %s", path, exc_info=True)


def log_prediction_features(
    features_df: pd.DataFrame,
    request_id: Optional[str] = None,
    league: str = "",
    timestamp: Optional[str] = None,
) -> None:
    """记录胜率预测特征

    Args:
        features_df: 单行特征 DataFrame (列名 = feature_cols)
        request_id: 请求标识，None 则自动生成 uuid
        league: 联赛 (LCK/LPL/LEC，用于分组分析)
        timestamp: ISO 格式时间戳，None 则用当前时间
    """
    try:
        if not _should_sample():
            return
        if features_df is None or len(features_df) == 0:
            return

        rid = request_id or uuid.uuid4().hex[:12]
        ts = timestamp or datetime.now().isoformat()

        # 取第一行 (单次推理)
        row = {"request_id": rid, "timestamp": ts, "league": league}
        first_row = features_df.iloc[0].to_dict()
        # 仅保留数值型特征 (跳过非数值列如 date/league/team_name)
        for k, v in first_row.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
            elif pd.isna(v):
                row[k] = float("nan")
            else:
                # 非数值列转字符串 (理论上不应出现在 features_df 中)
                continue

        df = pd.DataFrame([row])
        path = _LOG_DIR / f"prediction_{datetime.now().strftime('%Y%m%d')}.parquet"
        _append_parquet(path, df)
    except Exception:
        _log.debug("记录预测特征失败", exc_info=True)


def log_recommendation_features(
    cand_np: np.ndarray,
    global_context: np.ndarray,
    mask_np: np.ndarray,
    step_type: str,
    request_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> None:
    """记录 BP 推荐特征

    仅记录 mask==1 的有效英雄候选，避免大量 0 填充污染分布。
    每个有效英雄一行，全局上下文共享。

    Args:
        cand_np: [vocab_size, CANDIDATE_DIM] 候选特征矩阵
        global_context: [20] 全局上下文特征
        mask_np: [vocab_size] 可用英雄掩码 (>0.5 为有效)
        step_type: "pick" 或 "ban"
        request_id: 请求标识，None 则自动生成 uuid
        timestamp: ISO 时间戳，None 则用当前时间
    """
    try:
        if not _should_sample():
            return
        # 延迟导入避免循环依赖
        from bp_recommendation.feature_pipeline import CANDIDATE_FEAT_MAP

        rid = request_id or uuid.uuid4().hex[:12]
        ts = timestamp or datetime.now().isoformat()

        # 仅记录 mask==1 的英雄，避免大量 0 填充污染分布
        valid_mask = np.asarray(mask_np).flatten() > 0.5
        cand_np = np.asarray(cand_np)
        valid_rows = cand_np[valid_mask]

        records = []
        global_context = np.asarray(global_context).flatten()
        for row in valid_rows:
            rec = {
                "request_id": rid,
                "timestamp": ts,
                "step_type": step_type,
            }
            # 候选特征 (CANDIDATE_FEAT_MAP: {feat_name: col_idx})
            for feat_name, col_idx in CANDIDATE_FEAT_MAP.items():
                rec[f"candidate_{feat_name}"] = float(row[col_idx])
            # 全局上下文 20 维 (每个请求的同一 step 共享)
            for i in range(len(global_context)):
                rec[f"context_{i}"] = float(global_context[i])
            records.append(rec)

        if not records:
            return
        df = pd.DataFrame(records)
        path = _LOG_DIR / f"recommendation_{datetime.now().strftime('%Y%m%d')}.parquet"
        _append_parquet(path, df)
    except Exception:
        _log.debug("记录推荐特征失败", exc_info=True)
