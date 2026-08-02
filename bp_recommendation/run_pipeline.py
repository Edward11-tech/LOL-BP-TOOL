#!/usr/bin/env python3
"""
BP 推荐模型训练流水线（生产环境）
=============================================
端到端训练流水线入口，自动化执行特征工程、模型训练、模型融合、验证报告等完整流程。

功能描述:
    - 执行上线前预检，验证关键数据文件
    - 运行特征工程流水线生成最新特征
    - 训练 Pick CS/NoCS Transformer 模型
    - 训练 Pick Cascade LightGBM 融合模型
    - 训练 Ban CS Transformer 模型
    - 训练 Ban Cascade LightGBM 融合模型
    - 生成训练报告和指标汇总
    - 支持开发/生产双模式，通过 BP_PRODUCTION_MODE 环境变量切换

主要函数:
    - run_command(): 执行子进程命令并实时记录输出
    - _preflight_checks(): 上线前预检
    - run_full_pipeline(): 执行完整训练流水线

使用方法:
    cd /Users/siwentu/Desktop/LOL analysis
    
    开发模式（超参搜索用）:
    python -m bp_recommendation.run_pipeline
    
    生产模式（全量数据训练）:
    BP_PRODUCTION_MODE=true python -m bp_recommendation.run_pipeline
    
    可选参数:
    --skip-features: 跳过特征工程步骤
    --skip-pick: 跳过 Pick 模型训练
    --skip-ban: 跳过 Ban 模型训练
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import torch
from pathlib import Path

BASE_DIR = str(Path(__file__).parent.resolve())
ROOT_DIR = str(Path(BASE_DIR).parent.resolve())
sys.path.insert(0, ROOT_DIR)
MODEL_PKG = "bp_recommendation"

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

from logger_config import get_logger, setup_logging, log_context, timed
from common.paths import RECOMMENDATION_METRICS_DIR, ensure_dirs as _ensure_common_dirs
_ensure_common_dirs()
os.makedirs(str(RECOMMENDATION_METRICS_DIR), exist_ok=True)

log = get_logger(__name__)


def run_command(cmd, description, cwd=ROOT_DIR):
    """执行子进程命令并实时记录输出"""
    log.info("")
    log.info("=" * 70)
    log.info(f"  {description}")
    log.info("=" * 70)
    log.info(f"  Command: {' '.join(cmd)}")
    log.info("")

    start = time.time()
    try:
        process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding="utf-8", bufsize=1
        )
        
        for line in process.stdout:
            clean_line = line.strip()
            if clean_line:
                log.info(f"  > {clean_line}")

        process.wait()
        elapsed = time.time() - start

        if process.returncode != 0:
            log.error(f"  FAILED with exit code {process.returncode} after {elapsed:.1f}s")
            return False, elapsed

        log.info(f"  SUCCESS in {elapsed:.1f}s")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  EXCEPTION after {elapsed:.1f}s: {e}")
        return False, elapsed


# ======================================================================
# Full Pipeline（生产环境）
# ======================================================================
def _preflight_checks():
    """上线前预检：验证关键数据文件和配置是否存在，避免训练中途失败。"""
    log.info("")
    log.info("=" * 70)
    log.info("  PRE-FLIGHT CHECKS")
    log.info("=" * 70)

    checks = []

    # 1. 原始数据文件
    matches_csv = os.path.join(ROOT_DIR, "cleaned_data", "matches_cleaned.csv")
    checks.append(("matches_cleaned.csv", os.path.exists(matches_csv)))

    # 2. 英雄词表
    vocab_json = os.path.join(ROOT_DIR, "cleaned_data", "champion_vocabulary.json")
    checks.append(("champion_vocabulary.json", os.path.exists(vocab_json)))

    # 3. 位置映射
    pos_json = os.path.join(ROOT_DIR, "cleaned_data", "champion_position_mapping.json")
    checks.append(("champion_position_mapping.json", os.path.exists(pos_json)))

    # 4. 训练配置文件（生产模式需要）
    config_dir = os.path.join(BASE_DIR, "training_configs")
    if os.path.isdir(config_dir):
        for cfg in os.listdir(config_dir):
            if cfg.endswith(".json"):
                checks.append((f"training_configs/{cfg}", True))
    else:
        checks.append(("training_configs/ (directory)", False))

    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        log.info(f"  [{status:7s}] {name}")
        if not ok:
            all_ok = False

    if not all_ok:
        log.error("  Pre-flight checks FAILED. Aborting pipeline.")
        return False

    log.info("  Pre-flight checks PASSED.")
    return True


def _post_training_verification(args):
    """训练后验证：运行特征对齐和预测一致性检查。"""
    log.info("")
    log.info("=" * 70)
    log.info("  POST-TRAINING VERIFICATION")
    log.info("=" * 70)

    all_passed = True

    align_cmd = [sys.executable, "-m", f"{MODEL_PKG}.verify_features_alignment"]
    success, elapsed = run_command(align_cmd, "Feature Alignment Check")
    if success:
        log.info("  Feature Alignment Check: PASSED (%.1fs)", elapsed)
    else:
        log.warning("  Feature Alignment Check: FAILED (%.1fs) - Please review before deployment.", elapsed)
        all_passed = False

    pred_cmd = [sys.executable, "-m", f"{MODEL_PKG}.verify_predictions"]
    success, elapsed = run_command(pred_cmd, "Prediction Consistency Check")
    if success:
        log.info("  Prediction Consistency Check: PASSED (%.1fs)", elapsed)
    else:
        log.warning("  Prediction Consistency Check: FAILED (%.1fs) - Please review before deployment.", elapsed)
        all_passed = False

    log.info("")
    if all_passed:
        log.info("  Verification Summary: ALL CHECKS PASSED")
    else:
        log.warning("  Verification Summary: SOME CHECKS FAILED")
    log.info("=" * 70)


def run_full_pipeline(args):
    from datetime import datetime as _dt
    pipeline_start = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = args.run_name or time.strftime("full_pipeline_%Y%m%d_%H%M%S")
    pick_ckpt = os.path.join(BASE_DIR, "model_pick", "checkpoints")
    ban_ckpt = os.path.join(BASE_DIR, "model_ban", "checkpoints")

    results = {"run_name": run_name, "model_dir": MODEL_PKG,
               "start_time": time.time(), "pipeline_date": pipeline_start, "stages": {}}

    device_name = args.device if args.device else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    log.info("\n" + "=" * 70)
    log.info("=" * 70)
    log.info("  BP RECOMMENDATION FULL PIPELINE")
    log.info("=" * 70)
    log.info("  Start Date/Time: %s", pipeline_start)
    log.info("  Mode          : %s", "PRODUCTION (Blind Training)" if args.production else "DEVELOPMENT")
    log.info("  Device        : %s", device_name)
    log.info("  Seed          : %s", args.seed)
    log.info("  Run Name      : %s", run_name)
    log.info("=" * 70)
    log.info("  Pipeline Stages:")
    log.info("    Stage 0: Feature Pipeline")
    log.info("    Stage 1: Pick Training (CS + NoCS + Cascade)")
    log.info("    Stage 2: Ban Training (CS + Cascade)")
    log.info("    Stage 3: Validation Metrics Report")
    log.info("    Stage 4: Post-Training Verification")
    log.info("=" * 70 + "\n")

    if not _preflight_checks():
        return

    device_args = ["--device", device_name]
    
    # 动态传播 --production 信号给所有子脚本
    prod_args = ["--production"] if args.production else []

    # 硬件加速参数
    pick_accel = ["--num_workers", "0"]
    if device_name == "cuda": pick_accel.extend(["--amp", "--compile"])
    pick_args = device_args + pick_accel + prod_args

    ban_accel = ["--num_workers", "0", "--amp"]
    if device_name == "cuda": ban_accel.append("--compile")
    ban_args = device_args + ban_accel + prod_args

    # --- Stage 0: Feature Pipeline ---
    feat_cmd = [sys.executable, "-m", f"{MODEL_PKG}.feature_pipeline"]
    success, elapsed = run_command(feat_cmd, "STAGE 0: Feature Pipeline")
    results["stages"]["feature_pipeline"] = {"success": success, "elapsed_sec": round(elapsed, 1)}
    if not success: return

    # --- Stage 1: Pick Training ---
    # 【重构核心】：完全删去命令行超参数拼接，由脚本内部 config.py 自动接管！
    common_train_args = ["--seed", str(args.seed)]

    # 1a: CS Transformer (训练完成后会自动导出 logits)
    cs_cmd = [sys.executable, "-m", f"{MODEL_PKG}.model_pick.train_pick"] \
        + common_train_args + ["--run_name", f"{run_name}_cs", "--ckpt_dir", pick_ckpt] + pick_args
    success, elapsed = run_command(cs_cmd, "STAGE 1a: Pick CS Transformer")
    results["stages"]["pick_cs_transformer"] = {"success": success, "elapsed_sec": round(elapsed, 1)}
    if not success: return

    # 1b: NoCS Transformer (训练完成后会自动导出 logits)
    nocs_cmd = [sys.executable, "-m", f"{MODEL_PKG}.model_pick.train_pick"] \
        + common_train_args + ["--run_name", f"{run_name}_nocs", "--mask_cs", "--ckpt_dir", pick_ckpt] + pick_args
    success, elapsed = run_command(nocs_cmd, "STAGE 1b: Pick NoCS Transformer")
    results["stages"]["pick_nocs_transformer"] = {"success": success, "elapsed_sec": round(elapsed, 1)}
    if not success: return

    # 1c: Cascade Pick
    if not args.skip_cascade:
        cascade_cmd = [sys.executable, "-m", f"{MODEL_PKG}.model_pick.cascade_pick"] + prod_args
        success, elapsed = run_command(cascade_cmd, "STAGE 1c: Cascade Pick")
        results["stages"]["pick_cascade"] = {"success": success, "elapsed_sec": round(elapsed, 1)}
        if not success: return
    else:
        log.info("  [SKIP] STAGE 1c: Cascade Pick (--skip_cascade)")

    # --- Stage 2: Ban Training ---
    # 2a: Ban Transformer (训练完成后会自动导出 logits)
    ban_cmd = [sys.executable, "-m", f"{MODEL_PKG}.model_ban.train_ban"] \
        + common_train_args + ["--run_name", f"{run_name}_ban_cs"] + ban_args
    success, elapsed = run_command(ban_cmd, "STAGE 2a: Ban Transformer")
    results["stages"]["ban_transformer"] = {"success": success, "elapsed_sec": round(elapsed, 1)}
    if not success: return

    # 2b: Cascade Ban
    if not args.skip_cascade:
        cascade_ban_cmd = [sys.executable, "-m", f"{MODEL_PKG}.model_ban.cascade_ban"] + prod_args
        success, elapsed = run_command(cascade_ban_cmd, "STAGE 2b: Cascade Ban")
        results["stages"]["ban_cascade"] = {"success": success, "elapsed_sec": round(elapsed, 1)}
        if not success: return
    else:
        log.info("  [SKIP] STAGE 2b: Cascade Ban (--skip_cascade)")

    # --- Stage 3: Report Validation Metrics ---
    val_metrics = _collect_val_metrics(pick_ckpt, ban_ckpt)
    results["val_metrics"] = val_metrics
    _report_val_metrics(val_metrics)

    # --- Stage 4: Post-Training Verification ---
    if not args.skip_verification:
        _post_training_verification(args)
    else:
        log.info("  [SKIP] STAGE 4: Post-Training Verification (--skip_verification)")

    # Summary
    total_elapsed = time.time() - results["start_time"]
    results["total_elapsed_sec"] = round(total_elapsed, 1)
    results["status"] = "SUCCESS"
    _save_results(results, run_name)

    from datetime import datetime as _dt2
    pipeline_end = _dt2.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("\n" + "=" * 70)
    log.info("=" * 70)
    log.info("  PIPELINE COMPLETE - FINAL SUMMARY")
    log.info("=" * 70)
    log.info("  Start Time    : %s", pipeline_start)
    log.info("  End Time      : %s", pipeline_end)
    log.info("  Total Duration: %.1f minutes (%.1f seconds)", total_elapsed / 60, total_elapsed)
    log.info("  Mode          : %s", "PRODUCTION" if args.production else "DEVELOPMENT")
    log.info("  Device        : %s", device_name)
    log.info("")
    log.info("  Stage Results:")
    for stage_name, stage_data in results["stages"].items():
        status = "SUCCESS" if stage_data.get("success", False) else "FAILED"
        elapsed = stage_data.get("elapsed_sec", 0)
        log.info("    %-30s: %s (%.1fs)", stage_name, status, elapsed)
    log.info("")
    if val_metrics:
        pick_m = val_metrics.get("pick_cascade", {}).get("cascade_final", {})
        ban_m = val_metrics.get("ban_cascade", {}).get("cascade_final", {})
        if pick_m:
            log.info("  Pick Model Metrics:")
            for k, v in sorted(pick_m.items()):
                if isinstance(v, (int, float)):
                    log.info("    %-15s: %.4f", k, v)
        if ban_m:
            log.info("  Ban Model Metrics:")
            for k, v in sorted(ban_m.items()):
                if isinstance(v, (int, float)):
                    log.info("    %-15s: %.4f", k, v)
    log.info("=" * 70)
    log.info("=" * 70 + "\n")


def _collect_val_metrics(pick_ckpt, ban_ckpt):
    metrics = {}
    pick_metrics_path = os.path.join(pick_ckpt, "cascade_final_metrics.json")
    if os.path.exists(pick_metrics_path):
        with open(pick_metrics_path, 'r') as f:
            metrics["pick_cascade"] = json.load(f)
            
    ban_metrics_path = os.path.join(ban_ckpt, "cascade_final_metrics.json")
    if os.path.exists(ban_metrics_path):
        with open(ban_metrics_path, 'r') as f:
            metrics["ban_cascade"] = json.load(f)
    return metrics


def _report_val_metrics(val_metrics):
    log.info("")
    log.info("=" * 70)
    log.info("  FINAL CASCADE METRICS")
    log.info("=" * 70)

    def _log_model(name, m):
        if not m:
            log.info(f"  {name}: (no metrics found)")
            return
        cascade_final = m.get("cascade_final", {})
        if cascade_final:
            for key in sorted(cascade_final.keys()):
                val = cascade_final[key]
                if isinstance(val, (int, float)):
                    log.info(f"  {name}  {key}: {val:.4f}")
        else:
            log.info(f"  {name}: (no cascade_final metrics)")

    _log_model("Pick Cascade  ", val_metrics.get("pick_cascade", {}))
    _log_model("Ban Cascade   ", val_metrics.get("ban_cascade", {}))
    log.info("=" * 70)


def _save_results(results, run_name):
    results_path = os.path.join(LOG_DIR, f"{run_name}_results.json")
    if isinstance(results.get("start_time"), float):
        results["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results["start_time"]))
    if isinstance(results.get("end_time"), float):
        results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results["end_time"]))
    results["metadata"] = {
        "model_type": "bp_recommendation",
        "mode": results.get("mode", "training"),
        "run_name": run_name,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    ts = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(str(RECOMMENDATION_METRICS_DIR), f"recommendation_{run_name}_{ts}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"  Pipeline results saved to {results_path}")
    log.info(f"  Metrics archived to {metrics_path}")


if __name__ == "__main__":
    setup_logging(log_dir=Path(LOG_DIR))
    
    parser = argparse.ArgumentParser(description="BP 推荐模型 Pipeline（生产环境编排器）")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cpu/mps/cuda)")
    parser.add_argument("--run_name", type=str, default=None, help="自定义运行名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip_cascade", action="store_true", help="跳过 Cascade 训练")
    parser.add_argument("--skip_verification", action="store_true", help="跳过训练后验证")
    parser.add_argument("--production", action="store_true", help="强制启用生产模式")
    
    args = parser.parse_args()

    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _run_log_path = os.path.join(LOG_DIR, f"run_pipeline_{_run_ts}.log")
    _run_fh = logging.FileHandler(_run_log_path, encoding="utf-8")
    _run_fh.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _run_fh.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_run_fh)

    log.info("=" * 70)
    log.info(f"  BP Pipeline Orchestrator - {MODEL_PKG}")
    log.info(f"  Run: {args.run_name or 'auto'}")
    log.info("=" * 70)

    total_start = time.time()
    run_full_pipeline(args)

    total_elapsed = time.time() - total_start
    log.info("")
    log.info("=" * 70)
    log.info(f"  PIPELINE COMPLETE in {total_elapsed / 60:.1f} minutes")
    log.info("=" * 70)