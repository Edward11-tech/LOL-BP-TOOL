"""
data_checks.py — 共享的数据异常检测工具
=====================================
提供可复用的数据完整性检查函数，供所有模型训练文件和数据处理文件
在关键节点调用，统一打印异常值检测结果。

使用方式:
    from data_checks import check_array, check_dataframe, check_labels, check_groups

    check_array("X_train", X_train, log)
    check_labels("y_train", y_train, log)
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
from logger_config import get_logger

log = get_logger(__name__)


def check_array(name: str, arr, logger: logging.Logger, context: str = "") -> bool:
    """
    检查 numpy 数组的完整性：NaN、Inf、形状、数据范围。

    Args:
        name: 变量名 (用于日志标识)
        arr: 待检查的数组 (np.ndarray 或类似)
        logger: 日志记录器
        context: 可选的上下文描述

    Returns:
        bool: True 表示数据正常，False 表示发现异常
    """
    if arr is None:
        logger.error(f"  [数据检查] {name}: 为 None! {context}")
        return False

    arr = np.asarray(arr)
    prefix = f"  [数据检查] {name}" + (f" ({context})" if context else "")

    # 基本形状信息
    logger.info(f"{prefix}: shape={arr.shape}, dtype={arr.dtype}")

    if arr.size == 0:
        logger.error(f"{prefix}: 数组为空 (size=0)!")
        return False

    issues = []

    # NaN 检查
    nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
    if nan_count > 0:
        nan_pct = nan_count / arr.size * 100
        issues.append(f"NaN={nan_count} ({nan_pct:.2f}%)")

    # Inf 检查
    if np.issubdtype(arr.dtype, np.number) and np.issubdtype(arr.dtype, np.floating):
        inf_count = int(np.isinf(arr).sum())
        if inf_count > 0:
            issues.append(f"Inf={inf_count}")

    # 数值范围检查 (仅对数值型)
    if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
        try:
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            arr_mean = float(np.nanmean(arr)) if not np.isnan(arr_mean := np.nanmean(arr)) else float('nan')

            # 极端值检测：绝对值超过 1e6 可能是异常
            abs_max = max(abs(arr_min), abs(arr_max))
            if abs_max > 1e6:
                issues.append(f"极端值(max_abs={abs_max:.2e})")

            logger.info(f"{prefix}: range=[{arr_min:.4f}, {arr_max:.4f}], mean={arr_mean:.4f}")
        except (ValueError, TypeError):
            pass

    if issues:
        logger.warning(f"{prefix}: 发现异常 -> {', '.join(issues)}")
        return False
    else:
        logger.info(f"{prefix}: 正常")
        return True


def check_dataframe(name: str, df, logger: logging.Logger, context: str = "") -> bool:
    """
    检查 pandas DataFrame 的完整性：NaN、重复行、空列、形状。

    Args:
        name: 变量名
        df: 待检查的 DataFrame
        logger: 日志记录器
        context: 可选的上下文描述

    Returns:
        bool: True 表示数据正常，False 表示发现异常
    """
    if df is None:
        logger.error(f"  [数据检查] {name}: 为 None! {context}")
        return False

    prefix = f"  [数据检查] {name}" + (f" ({context})" if context else "")
    logger.info(f"{prefix}: shape={df.shape}")

    if df.empty:
        logger.error(f"{prefix}: DataFrame 为空!")
        return False

    issues = []

    # NaN 检查
    nan_total = int(df.isna().sum().sum())
    if nan_total > 0:
        nan_cols = df.isna().sum()
        nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
        top_nan = nan_cols.head(5)
        issues.append(f"NaN={nan_total} (主要列: {dict(top_nan)})")

    # 全空列检查
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        issues.append(f"全空列={empty_cols}")

    # 重复行检查
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(f"重复行={dup_count}")

    # Inf 检查 (数值列)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        inf_count = int(np.isinf(numeric_df.values).sum())
        if inf_count > 0:
            issues.append(f"Inf={inf_count}")

        # 极端值检查
        abs_max = float(np.nanmax(np.abs(numeric_df.values))) if numeric_df.size > 0 else 0
        if abs_max > 1e6:
            issues.append(f"极端值(max_abs={abs_max:.2e})")

    if issues:
        logger.warning(f"{prefix}: 发现异常 -> {'; '.join(issues)}")
        return False
    else:
        logger.info(f"{prefix}: 正常 ({len(df)} 行, {len(df.columns)} 列)")
        return True


def check_labels(name: str, y, logger: logging.Logger, context: str = "") -> bool:
    """
    检查标签数组的分布和有效性。

    Args:
        name: 变量名
        y: 标签数组
        logger: 日志记录器
        context: 可选的上下文描述

    Returns:
        bool: True 表示正常，False 表示异常
    """
    if y is None:
        logger.error(f"  [数据检查] {name}: 为 None! {context}")
        return False

    y = np.asarray(y)
    prefix = f"  [数据检查] {name}" + (f" ({context})" if context else "")

    if y.size == 0:
        logger.error(f"{prefix}: 标签数组为空!")
        return False

    unique_vals, counts = np.unique(y, return_counts=True)
    issues = []

    # 检查是否全为同一类别
    if len(unique_vals) == 1:
        issues.append(f"单一类别(value={unique_vals[0]})")

    # 检查类别不平衡 (正样本比例 < 1% 或 > 99%)
    if len(unique_vals) == 2:
        pos_ratio = counts[1] / y.size * 100 if counts[0] <= counts[1] else counts[0] / y.size * 100
        if pos_ratio < 1.0:
            issues.append(f"极端不平衡(少数类={pos_ratio:.2f}%)")
        elif pos_ratio > 99.0:
            issues.append(f"极端不平衡(多数类={pos_ratio:.2f}%)")

    # NaN 检查
    if np.issubdtype(y.dtype, np.number):
        nan_count = int(np.isnan(y).sum())
        if nan_count > 0:
            issues.append(f"NaN={nan_count}")

    # 打印分布
    dist_str = ", ".join(f"{v}:{c}" for v, c in zip(unique_vals, counts))
    logger.info(f"{prefix}: n={y.size}, 分布=[{dist_str}]")

    if issues:
        logger.warning(f"{prefix}: 发现异常 -> {', '.join(issues)}")
        return False
    else:
        logger.info(f"{prefix}: 正常")
        return True


def check_groups(name: str, groups, logger: logging.Logger, context: str = "") -> bool:
    """
    检查 LightGBM/Ranking 的 group 参数有效性。

    Args:
        name: 变量名
        groups: group 大小数组
        logger: 日志记录器
        context: 可选的上下文描述

    Returns:
        bool: True 表示正常，False 表示异常
    """
    if groups is None:
        logger.error(f"  [数据检查] {name}: 为 None! {context}")
        return False

    groups = np.asarray(groups)
    prefix = f"  [数据检查] {name}" + (f" ({context})" if context else "")

    if groups.size == 0:
        logger.error(f"{prefix}: group 数组为空!")
        return False

    total = int(groups.sum())
    issues = []

    # 检查 group 大小是否为正数
    non_positive = int((groups <= 0).sum())
    if non_positive > 0:
        issues.append(f"非正group={non_positive}")

    # 检查 group 大小是否合理 (过大可能是数据问题)
    max_group = int(groups.max()) if groups.size > 0 else 0
    if max_group > 500:
        issues.append(f"超大group(max={max_group})")

    min_group = int(groups.min()) if groups.size > 0 else 0
    logger.info(f"{prefix}: n_groups={groups.size}, sum={total}, "
                f"range=[{min_group}, {max_group}], mean={groups.mean():.1f}")

    if issues:
        logger.warning(f"{prefix}: 发现异常 -> {', '.join(issues)}")
        return False
    else:
        logger.info(f"{prefix}: 正常")
        return True


def check_predictions(name: str, preds, logger: logging.Logger, context: str = "") -> bool:
    """
    检查模型预测输出的有效性。

    Args:
        name: 变量名
        preds: 预测数组
        logger: 日志记录器
        context: 可选的上下文描述

    Returns:
        bool: True 表示正常，False 表示异常
    """
    if preds is None:
        logger.error(f"  [预测检查] {name}: 为 None! {context}")
        return False

    preds = np.asarray(preds)
    prefix = f"  [预测检查] {name}" + (f" ({context})" if context else "")

    if preds.size == 0:
        logger.error(f"{prefix}: 预测数组为空!")
        return False

    issues = []

    # NaN 检查
    nan_count = int(np.isnan(preds).sum()) if np.issubdtype(preds.dtype, np.number) else 0
    if nan_count > 0:
        issues.append(f"NaN={nan_count}")

    # Inf 检查
    if np.issubdtype(preds.dtype, np.floating):
        inf_count = int(np.isinf(preds).sum())
        if inf_count > 0:
            issues.append(f"Inf={inf_count}")

    # 常数预测检查 (所有预测值相同 = 模型可能没学到东西)
    if preds.size > 1 and np.issubdtype(preds.dtype, np.number):
        pred_std = float(np.std(preds))
        if pred_std < 1e-8:
            issues.append(f"常数预测(std={pred_std:.2e})")

    if np.issubdtype(preds.dtype, np.number) and preds.size > 0:
        logger.info(f"{prefix}: n={preds.size}, range=[{float(np.min(preds)):.4f}, "
                    f"{float(np.max(preds)):.4f}], mean={float(np.mean(preds)):.4f}, "
                    f"std={float(np.std(preds)):.4f}")

    if issues:
        logger.warning(f"{prefix}: 发现异常 -> {', '.join(issues)}")
        return False
    else:
        logger.info(f"{prefix}: 正常")
        return True


def check_file_exists(name: str, path, logger: logging.Logger) -> bool:
    """
    检查文件是否存在且非空。

    Args:
        name: 文件标识名
        path: 文件路径
        logger: 日志记录器

    Returns:
        bool: True 表示文件正常
    """
    import os
    prefix = f"  [文件检查] {name}"

    if not os.path.exists(path):
        logger.error(f"{prefix}: 文件不存在! path={path}")
        return False

    size = os.path.getsize(path)
    if size == 0:
        logger.error(f"{prefix}: 文件为空! path={path}")
        return False

    size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
    logger.info(f"{prefix}: 正常 ({size_str})")
    return True
