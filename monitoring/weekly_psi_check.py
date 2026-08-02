#!/usr/bin/env python3
"""
周度 PSI 漂移检查
==================
读取过去 N 天的推理特征日志 (parquet)，与训练基线对比计算 PSI。

本脚本仅用于监控漂移，不触发任何 fallback 或兜底逻辑，
检测报告保存在 monitoring/reports/ 供人工审阅。

用法:
    python monitoring/weekly_psi_check.py
    python monitoring/weekly_psi_check.py --days 7 --threshold 0.25
    python monitoring/weekly_psi_check.py --days 14 --threshold 0.2

退出码:
    0: 所有特征 PSI < threshold (或无数据)
    1: 存在显著漂移特征 (PSI >= threshold)，报告保存到 monitoring/reports/

Cron 调度建议:
    0 9 * * 1 cd /path/to/lol_serving && \\
        python monitoring/weekly_psi_check.py --days 7 >> monitoring/logs/psi_cron.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from logger_config import setup_logging, get_logger
from common.psi import PSIMonitor

log = get_logger(__name__)

# ---- 路径常量 ----
FEATURE_LOG_DIR = PROJECT_ROOT / "logs" / "inference_features"
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"
PREDICTION_BASELINE = (
    PROJECT_ROOT / "bp_prediction" / "features" / "prediction_feature_baseline.json"
)
RECOMMENDATION_BASELINE = (
    PROJECT_ROOT / "bp_recommendation" / "features" / "feature_baseline.json"
)


def _load_baseline(baseline_path: Path) -> Dict[str, PSIMonitor]:
    """加载基线文件，返回 {feature_name: PSIMonitor}

    兼容新旧格式：
        旧格式: {"feat": [c1, ..., c10]}
        新格式: {"feat": {"counts": [...], "bin_edges": [...]}}
    """
    if not baseline_path.exists():
        log.warning(f"基线文件不存在: {baseline_path}")
        return {}

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.error(f"读取基线文件失败 {baseline_path}: {e}")
        return {}

    monitors: Dict[str, PSIMonitor] = {}
    n_old_format = 0
    for feat_name, payload in data.items():
        if isinstance(payload, list):
            # 旧格式: [c1, ..., c10] —— 无 bin_edges
            counts = np.array(payload, dtype=np.float64)
            edges = None
            n_old_format += 1
        elif isinstance(payload, dict):
            # 新格式: {"counts": [...], "bin_edges": [...]}
            counts = np.array(payload["counts"], dtype=np.float64)
            edges = (
                np.array(payload["bin_edges"], dtype=np.float64)
                if "bin_edges" in payload
                else None
            )
        else:
            log.warning(f"基线条目 {feat_name} 类型异常 {type(payload)}，跳过")
            continue

        monitors[feat_name] = PSIMonitor(
            baseline_bins=counts,
            feature_name=feat_name,
            bin_edges=edges,
        )

    if n_old_format > 0:
        log.warning(
            f"{baseline_path.name}: {n_old_format} 个特征为旧格式 (无 bin_edges)，"
            f"将回退到动态分箱。请尽快重建基线。"
        )
    log.info(f"已加载基线 {baseline_path.name}: {len(monitors)} 个特征")
    return monitors


def _collect_feature_logs(prefix: str, days: int) -> pd.DataFrame:
    """读取过去 N 天的推理特征 parquet

    Args:
        prefix: "prediction" 或 "recommendation"
        days: 回溯天数
    """
    frames = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        path = FEATURE_LOG_DIR / f"{prefix}_{date}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                frames.append(df)
                log.info(f"  读取 {path.name}: {len(df)} 行")
            except Exception as e:
                log.warning(f"  读取失败 {path}: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"  {prefix} 合并后: {len(combined)} 行, {len(combined.columns)} 列")
    return combined


def _compute_psi_for_features(
    df: pd.DataFrame,
    monitors: Dict[str, PSIMonitor],
    threshold: float,
) -> dict:
    """对每列特征计算 PSI

    Returns:
        dict: {
            n_features_compared: int,
            n_drifted_features: int,
            max_psi: float,
            drifted_features: [{"name": str, "psi": float}, ...],
            all_psi: {feat_name: psi_value, ...},
        }
    """
    # 跳过非特征列
    skip_cols = {"request_id", "timestamp", "league", "step_type"}

    all_psi: Dict[str, float] = {}
    drifted = []

    for col in df.columns:
        if col in skip_cols:
            continue
        if col not in monitors:
            continue
        values = df[col].dropna().values.astype(np.float64)
        # 至少 10 个样本才计算 PSI (单次请求不可靠)
        if len(values) < 10:
            continue
        try:
            psi = monitors[col].compute_psi(values)
            all_psi[col] = round(psi, 4)
            if psi >= threshold:
                drifted.append({"name": col, "psi": round(psi, 4)})
        except Exception as e:
            log.debug(f"计算 PSI 失败 {col}: {e}")

    drifted.sort(key=lambda x: -x["psi"])
    return {
        "n_features_compared": len(all_psi),
        "n_drifted_features": len(drifted),
        "max_psi": max(all_psi.values()) if all_psi else 0.0,
        "drifted_features": drifted,
        "all_psi": all_psi,
    }


def main():
    parser = argparse.ArgumentParser(description="周度 PSI 漂移检查")
    parser.add_argument(
        "--days", type=int, default=7,
        help="回溯天数 (默认 7)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25,
        help="PSI 漂移阈值 (默认 0.25)"
    )
    args = parser.parse_args()

    setup_logging()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    report = {
        "generated_at": datetime.now().isoformat(),
        "period": {"start": start_date, "end": end_date, "days": args.days},
        "threshold": args.threshold,
        "models": {},
    }

    any_drifted = False

    # ---- 预测模型 ----
    log.info("=" * 60)
    log.info("  预测模型 PSI 检查")
    log.info("=" * 60)
    pred_df = _collect_feature_logs("prediction", args.days)
    if len(pred_df) > 0:
        pred_monitors = _load_baseline(PREDICTION_BASELINE)
        if pred_monitors:
            pred_result = _compute_psi_for_features(
                pred_df, pred_monitors, args.threshold
            )
            report["models"]["prediction"] = {
                "n_samples": len(pred_df),
                **pred_result,
            }
            log.info(
                f"  样本数: {len(pred_df)}, 比较特征数: {pred_result['n_features_compared']}, "
                f"漂移特征数: {pred_result['n_drifted_features']}, max PSI: {pred_result['max_psi']:.4f}"
            )
            for d in pred_result["drifted_features"][:5]:
                log.warning(f"    漂移特征: {d['name']} PSI={d['psi']}")
            if pred_result["n_drifted_features"] > 0:
                any_drifted = True
        else:
            report["models"]["prediction"] = {
                "n_samples": len(pred_df),
                "error": "baseline not loaded",
            }
            log.warning("  基线未加载，跳过预测模型 PSI 检查")
    else:
        report["models"]["prediction"] = {
            "n_samples": 0,
            "note": "no inference logs found",
        }
        log.info("  未找到预测推理日志")

    # ---- 推荐模型 ----
    log.info("=" * 60)
    log.info("  推荐模型 PSI 检查")
    log.info("=" * 60)
    rec_df = _collect_feature_logs("recommendation", args.days)
    if len(rec_df) > 0:
        rec_monitors = _load_baseline(RECOMMENDATION_BASELINE)
        if rec_monitors:
            rec_result = _compute_psi_for_features(
                rec_df, rec_monitors, args.threshold
            )
            report["models"]["recommendation"] = {
                "n_samples": len(rec_df),
                **rec_result,
            }
            log.info(
                f"  样本数: {len(rec_df)}, 比较特征数: {rec_result['n_features_compared']}, "
                f"漂移特征数: {rec_result['n_drifted_features']}, max PSI: {rec_result['max_psi']:.4f}"
            )
            for d in rec_result["drifted_features"][:5]:
                log.warning(f"    漂移特征: {d['name']} PSI={d['psi']}")
            if rec_result["n_drifted_features"] > 0:
                any_drifted = True
        else:
            report["models"]["recommendation"] = {
                "n_samples": len(rec_df),
                "error": "baseline not loaded",
            }
            log.warning("  基线未加载，跳过推荐模型 PSI 检查")
    else:
        report["models"]["recommendation"] = {
            "n_samples": 0,
            "note": "no inference logs found",
        }
        log.info("  未找到推荐推理日志")

    # ---- 保存报告 ----
    report_path = REPORTS_DIR / f"psi_weekly_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"PSI 周报已保存: {report_path}")

    # ---- 摘要输出 ----
    log.info("=" * 60)
    log.info("  摘要")
    log.info("=" * 60)
    for model_name, model_report in report["models"].items():
        n_drifted = model_report.get("n_drifted_features", 0)
        max_psi = model_report.get("max_psi", 0.0)
        n_samples = model_report.get("n_samples", 0)
        log.info(
            f"  [{model_name}] samples={n_samples}, drifted={n_drifted}, max_psi={max_psi:.4f}"
        )

    # ---- 退出码 (供 cron 告警) ----
    if any_drifted:
        log.warning(f"检测到显著漂移 (PSI >= {args.threshold})，退出码 1")
        sys.exit(1)
    else:
        log.info("未检测到显著漂移，退出码 0")
        sys.exit(0)


if __name__ == "__main__":
    main()
