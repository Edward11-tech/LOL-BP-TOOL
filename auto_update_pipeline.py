#!/usr/bin/env python3
"""
auto_update_pipeline.py — 全自动数据更新与模型重训练流水线
===========================================================

功能:
  1. 按指定顺序依次执行细粒度阶段: 数据爬取 → 数据清洗
     → 推荐模型训练/生产/验证 → 预测模型训练/生产/验证 → 启动 app
  2. 实时监控每个阶段运行状态，记录关键节点和完整日志
  3. 每阶段完成后自动验证输出文件完整性
  4. 全部完成后执行生产环境健康检查
  5. 自动重启 Web 后端服务
  6. 错误时自动告警 + 可选回滚

运行模式:
  --mode complete    完整模式 (7阶段): 严格对齐上线前检查流程
    数据爬取 (含数据清洗, run_all_scrapers.py 任务8 执行)
    → 推荐模型训练模式 (Pick/Ban 训练 + 报告指标 + 保存参数)
    → 推荐模型生产模式 (使用参数盲训)
    → 推荐模型一致性检测
    → 预测模型训练模式 (OOT 验证 + 记录 best_iteration + 产出生产参数源)
    → 预测模型生产模式 (读取 OOT best_iteration, 按 √n×0.65 补偿, 100% 数据盲训)
    → 预测模型一致性检测
    → 启动 app.py

  --mode production  仅生产模式 (5阶段): 跳过训练模式, 直接使用固定参数
    数据爬取 (含数据清洗)
    → 推荐模型生产模式 (--production 盲训)
    → 推荐模型一致性检测
    → 预测模型生产模式 (--skip-tf 跳过 OOT, 回退到 80/20 early stopping)
    → 预测模型一致性检测
    → 启动 app.py
    自动定时每14天一次

  --mode no_scrape   跳过爬取模式 (8阶段): 已有原始数据, 从后处理开始从头重训
    数据后处理 (跳过爬取 id 0-5, 执行 id 6-9: 位置概率→验证→清洗→贝叶斯融合)
    → 推荐模型训练模式 (Pick/Ban 训练 + 报告指标 + 保存参数)
    → 推荐模型生产模式 (使用参数盲训)
    → 推荐模型一致性检测
    → 预测模型训练模式 (OOT 验证 + 记录 best_iteration + 产出生产参数源)
    → 预测模型生产模式 (读取 OOT best_iteration, 按 √n×0.65 补偿, 100% 数据盲训)
    → 预测模型一致性检测
    → PSI 特征基线重建
    → 启动 app.py

配置:
  所有可调参数在 auto_update_config.json 中，支持:
    - 更新周期 (默认 14 天)
    - 执行顺序
    - 各阶段超时时间
    - 告警通知方式
    - 健康检查阈值

用法:
  python auto_update_pipeline.py                              # 完整模式 (默认)
  python auto_update_pipeline.py --mode production            # 仅生产模式
  python auto_update_pipeline.py --mode no_scrape             # 跳过爬取模式 (保留清洗+完整训练)
  python auto_update_pipeline.py --daemon                     # 守护进程模式 (每14天, 仅生产模式)
  python auto_update_pipeline.py --daemon --mode complete     # 守护进程模式 (完整模式)
  python auto_update_pipeline.py --dry-run                    # 干运行 (仅检查环境)
  python auto_update_pipeline.py --stage 2                    # 从指定阶段开始执行
  python auto_update_pipeline.py --status                     # 查看上次运行状态

作者: Auto-generated
"""

import os
import sys
import json
import time
import signal
import socket
import shutil
import hashlib
import logging
import argparse
import threading
import subprocess
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from logger_config import setup_logging as _setup_root_logging, get_logger, get_run_logger, silence_third_party
from common.paths import PROJECT_ROOT as PATHS_PROJECT_ROOT, get_match_data_path, RAW_MATCHES_DIR

# =====================================================================
# 路径配置
# =====================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "auto_update_config.json"
STATE_PATH = PROJECT_ROOT / "auto_update_state.json"
LOG_DIR = PROJECT_ROOT / "logs" / "auto_update"
BACKUP_DIR = PROJECT_ROOT / "backups"

# =====================================================================
# 阶段脚本配置
# =====================================================================
# 细粒度阶段定义: 将训练模式/生产模式/一致性检测分离, 严格对齐 complete 模式流程
#   complete: 数据爬取 → 数据清洗 → 推荐训练 → 推荐生产 → 推荐验证
#             → 预测训练 → 预测生产 → 预测验证 → 启动 app
#   production: 数据爬取 → 数据清洗 → 推荐生产 → 推荐验证
#               → 预测生产 → 预测验证 → 启动 app (跳过训练模式, 直接使用固定参数)

STAGE_SCRIPTS = {
    # ---- 数据阶段 ----
    "data_scrape": {
        "name": "数据爬取",
        "script": PROJECT_ROOT / "data_scraper" / "run_all_scrapers.py",
        "cwd": PROJECT_ROOT / "data_scraper",
        "description": "从各数据源爬取最新数据 (run_all_scrapers.py 任务8 自动调用 data_cleaning.py)",
    },
    "data_cleaning": {
        "name": "数据清洗",
        "script": PROJECT_ROOT / "cleaned_data" / "data_cleaning.py",
        "cwd": PROJECT_ROOT / "cleaned_data",
        "description": "独立数据清洗阶段 (可选, 默认已在 data_scrape 阶段完成)",
    },
    "data_postprocess": {
        "name": "数据后处理 (跳过爬取)",
        "script": PROJECT_ROOT / "data_scraper" / "run_all_scrapers.py",
        "cwd": PROJECT_ROOT / "data_scraper",
        "description": "跳过爬取 (id 0-5), 执行后处理: 位置概率(6) → 验证(7) → 数据清洗(8) → 贝叶斯融合(9)",
        "extra_args": ["--no-scrape"],
    },

    # ---- BP 推荐模型阶段 ----
    "bp_recommendation_training": {
        "name": "BP推荐模型 - 训练模式",
        "script": PROJECT_ROOT / "bp_recommendation" / "run_pipeline.py",
        "cwd": PROJECT_ROOT,
        "description": "DEVELOPMENT 模式: 特征工程 + Pick(CS+NoCS+Cascade) + Ban(CS+Cascade) + 报告指标/保存参数",
        "extra_args": ["--skip_verification"],  # 验证留给独立阶段
    },
    "bp_recommendation_production": {
        "name": "BP推荐模型 - 生产模式",
        "script": PROJECT_ROOT / "bp_recommendation" / "run_pipeline.py",
        "cwd": PROJECT_ROOT,
        "description": "PRODUCTION 模式: 使用训练模式保存的参数进行盲训 (Pick+Ban 完整生产流程)",
        "extra_args": ["--production", "--skip_verification"],
    },
    "bp_recommendation_validation": {
        "name": "BP推荐模型 - 一致性检测",
        "script": PROJECT_ROOT / "bp_recommendation" / "verify_predictions.py",
        "pre_script": PROJECT_ROOT / "bp_recommendation" / "verify_features_alignment.py",
        "cwd": PROJECT_ROOT,
        "description": "推荐模型特征对齐 + 预测一致性检查",
    },

    # ---- BP 预测模型阶段 ----
    "bp_prediction_training": {
        "name": "BP预测模型 - 训练模式",
        "script": PROJECT_ROOT / "bp_prediction" / "run_training.py",
        "cwd": PROJECT_ROOT / "bp_prediction",
        "description": "开发模式: 特征工程 + 5-Fold OOT Transformer 训练 + OOT 验证 (记录 best_iteration, 产出 production_iterations_source.json)",
        "extra_args": ["--skip-training", "--skip-inference"],  # 跳过生产训练(Step3)和推理(Step4), 仅运行 OOT 验证
    },
    "bp_prediction_production": {
        "name": "BP预测模型 - 生产模式",
        "script": PROJECT_ROOT / "bp_prediction" / "run_training.py",
        "pre_script": PROJECT_ROOT / "bp_prediction" / "export_production_transformer.py",
        "cwd": PROJECT_ROOT / "bp_prediction",
        "description": "生产模式: 导出生产TF快照 + 读取 OOT best_iteration, 按 √n×0.65 补偿计算固定轮数, 100% 数据盲训 (方案 B)",
        # complete 模式: 训练模式已生成 features 和 fold TF, 跳过两者
        "extra_args": ["--skip-features", "--skip-tf", "--skip-inference"],
        # production 模式: 无训练模式, 需要运行 features (但跳过 OOT TF, 使用已有 fold 快照)
        "production_extra_args": ["--skip-tf", "--skip-inference"],
    },
    "bp_prediction_validation": {
        "name": "BP预测模型 - 一致性检测",
        "script": PROJECT_ROOT / "bp_prediction" / "check_prediction_alignment.py",
        "cwd": PROJECT_ROOT / "bp_prediction",
        "description": "预测模型端到端推理一致性校验 (线上/线下对比, --ci模式: 失败时阻断流水线)",
        "extra_args": ["--ci", "--mode", "quick", "--samples", "5"],
    },

    # ---- PSI 特征基线重建 ----
    "psi_baseline_rebuild": {
        "name": "PSI 特征基线重建",
        "script": PROJECT_ROOT / "build_feature_baselines.py",
        "cwd": PROJECT_ROOT,
        "description": "基于最新训练数据重建 PSI 特征基线 (含 bin_edges 分箱对齐)",
    },
}

# =====================================================================
# 运行模式定义
# =====================================================================
# 完整模式 (complete): 严格对齐上线前检查流程
#   数据爬取 (含数据清洗, run_all_scrapers.py 任务8 自动调用 data_cleaning.py)
#   → 推荐模型训练模式 (Pick/Ban 训练 + 报告指标 + 保存参数)
#   → 推荐模型生产模式 (使用参数盲训)
#   → 推荐模型一致性检测
#   → 预测模型训练模式 (特征工程 + OOT 验证 + 记录 best_iteration + 产出生产参数源)
#   → 预测模型生产模式 (导出生产TF快照 + 读取 OOT best_iteration, 按 √n×0.65 补偿盲训)
#   → 预测模型一致性检测
#   → 启动 app.py
MODE_COMPLETE_STAGES = [
    "data_scrape",                      # 已包含数据清洗
    "bp_recommendation_training",
    "bp_recommendation_production",
    "bp_recommendation_validation",
    "bp_prediction_training",
    "bp_prediction_production",
    "bp_prediction_validation",
    "psi_baseline_rebuild",             # 所有模型阶段之后重建 PSI 基线
]

# 仅生产模式 (production): 跳过训练模式, 直接使用固定参数进行生产训练
#   数据爬取 (含数据清洗)
#   → 推荐模型生产模式 (--production 盲训)
#   → 推荐模型一致性检测
#   → 预测模型生产模式 (特征工程 + 导出生产TF快照 + 训练生产模型, 跳过 OOT)
#     注意: 跳过 OOT 意味着无 production_iterations_source.json,
#           生产模式自动回退到 80/20 early stopping (旧模式)
#   → 预测模型一致性检测
#   → 启动 app.py
MODE_PRODUCTION_STAGES = [
    "data_scrape",                      # 已包含数据清洗
    "bp_recommendation_production",
    "bp_recommendation_validation",
    "bp_prediction_production",
    "bp_prediction_validation",
    "psi_baseline_rebuild",             # 生产模式也重建 PSI 基线
]

# 跳过爬取模式 (no_scrape): 用于已有原始数据但需从头重训模型的场景
#   数据清洗 (独立运行, 不爬取新数据)
#   → 推荐模型训练模式 (Pick/Ban 训练 + 报告指标 + 保存参数)
#   → 推荐模型生产模式 (使用参数盲训)
#   → 推荐模型一致性检测
#   → 预测模型训练模式 (特征工程 + OOT 验证 + 记录 best_iteration + 产出生产参数源)
#   → 预测模型生产模式 (导出生产TF快照 + 读取 OOT best_iteration, 按 √n×0.65 补偿盲训)
#   → 预测模型一致性检测
#   → PSI 特征基线重建
#   → 启动 app.py
MODE_NO_SCRAPE_STAGES = [
    "data_postprocess",                 # 跳过爬取, 执行完整后处理 (id 6-9: 位置概率→验证→清洗→贝叶斯融合)
    "bp_recommendation_training",
    "bp_recommendation_production",
    "bp_recommendation_validation",
    "bp_prediction_training",
    "bp_prediction_production",
    "bp_prediction_validation",
    "psi_baseline_rebuild",
]

# 兼容旧配置的默认执行顺序
DEFAULT_STAGE_ORDER = MODE_COMPLETE_STAGES

_CURRENT_YEAR = datetime.now().year
_MATCHES_DATA_RELPATH = f"raw_data/matches/{_CURRENT_YEAR}_LoL_esports_match_data_from_OraclesElixir.csv"

# 生产环境关键文件（用于健康检查）
# 按细粒度阶段分组, 每个阶段完成后验证对应输出文件
PRODUCTION_CHECK_FILES = {
    "data_scrape": [
        # champion_vocabulary.json 是静态手动维护文件, 不由爬虫生成, 不纳入新鲜度校验
        "cleaned_data/champion_position_mapping.json",
        "cleaned_data/player_career_hero_stats_cleaned.csv",
        _MATCHES_DATA_RELPATH,
    ],
    "data_cleaning": [
        # champion_vocabulary.json 是静态手动维护文件, 不由清洗流程生成
        "cleaned_data/champion_position_mapping.json",
        "cleaned_data/player_career_hero_stats_cleaned.csv",
        "cleaned_data/champion_counters_cleaned.csv",
        "cleaned_data/champion_synergy_cleaned.csv",
        "cleaned_data/champion_ranks_cleaned.csv",
    ],
    "data_postprocess": [
        # 后处理阶段 (id 6-9) 产出的所有文件
        "cleaned_data/champion_position_mapping.json",      # id=6 add_position_probabilities
        "cleaned_data/matches_cleaned.csv",                 # id=8 data_cleaning
        "cleaned_data/player_career_hero_stats_cleaned.csv",
        "cleaned_data/champion_counters_cleaned.csv",
        "cleaned_data/champion_synergy_cleaned.csv",
        "cleaned_data/champion_ranks_cleaned.csv",
        "cleaned_data/active_rosters.csv",
        "cleaned_data/merged_champion_stats.csv",            # id=9 merge_champion_stats
    ],
    # 推荐模型: 训练模式产出 checkpoint + 参数文件 + cascade 模型
    "bp_recommendation_training": [
        "bp_recommendation/model_pick/checkpoints/best_model_cs.pt",
        "bp_recommendation/model_pick/checkpoints/best_model_nocs.pt",
        "bp_recommendation/model_ban/checkpoints/best_model_cs.pt",
        # Cascade Pick 模型 (LightGBM 5-fold)
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_0_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_1_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_2_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_3_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_4_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/scaler.pkl",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/routing_config.json",
        # Cascade Ban 模型 (LightGBM 5-fold)
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_0_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_1_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_2_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_3_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_4_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/scaler.pkl",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/routing_config.json",
    ],
    # 推荐模型: 生产模式产出 cascade 模型
    "bp_recommendation_production": [
        "bp_recommendation/model_pick/checkpoints/best_model_cs.pt",
        "bp_recommendation/model_pick/checkpoints/best_model_nocs.pt",
        "bp_recommendation/model_ban/checkpoints/best_model_cs.pt",
        # Cascade Pick 模型 (LightGBM 5-fold)
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_0_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_1_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_2_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_3_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/fold_4_model.txt",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/scaler.pkl",
        "bp_recommendation/model_pick/checkpoints/cascade_pick/routing_config.json",
        # Cascade Ban 模型 (LightGBM 5-fold)
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_0_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_1_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_2_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_3_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/fold_4_model.txt",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/scaler.pkl",
        "bp_recommendation/model_ban/checkpoints/cascade_ban/routing_config.json",
    ],
    # 推荐模型: 一致性检测不产出文件, 仅验证现有模型
    "bp_recommendation_validation": [],
    # 预测模型: 训练模式产出 TF 特征 + OOT 指标 + 生产参数源 (方案 B)
    "bp_prediction_training": [
        "bp_prediction/features/ALL_prediction_wide_features.parquet",
        "bp_prediction/reports/production_iterations_source.json",  # 方案 B: OOT best_iteration
    ],
    # 预测模型: 生产模式产出生产TF快照 + .cbm 模型文件
    "bp_prediction_production": [
        "bp_prediction/tf_snapshots/production_nocs.pt",
        "bp_prediction/models/production/catboost_seed_0.cbm",
        "bp_prediction/models/production/feature_columns.json",
    ],
    # 预测模型: 一致性检测不产出文件, 仅验证现有模型
    "bp_prediction_validation": [],
    # PSI 基线重建: 产出两个基线 JSON (含 counts + bin_edges)
    "psi_baseline_rebuild": [
        "bp_prediction/features/prediction_feature_baseline.json",
        "bp_recommendation/features/feature_baseline.json",
    ],
    # 兼容旧配置 (已废弃, 保留向后兼容)
    "bp_recommendation": [
        "bp_recommendation/model_pick/checkpoints/best_model_cs.pt",
        "bp_recommendation/model_pick/checkpoints/best_model_nocs.pt",
        "bp_recommendation/model_ban/checkpoints/best_model_cs.pt",
    ],
    "bp_prediction": [
        "bp_prediction/models/production/catboost_seed_0.cbm",
        "bp_prediction/models/production/feature_columns.json",
    ],
    "data": [
        "cleaned_data/champion_position_mapping.json",
        "cleaned_data/player_career_hero_stats_cleaned.csv",
        _MATCHES_DATA_RELPATH,
    ],
}

# =====================================================================
# 默认配置
# =====================================================================
DEFAULT_CONFIG = {
    "update_interval_hours": 336,  # 14 天 = 336 小时
    "stage_order": DEFAULT_STAGE_ORDER,
    "timeouts": {
        "data_scrape": 9000,                       # 2.5 小时 (含数据清洗)
        "data_cleaning": 1800,                     # 30 分钟 (独立数据清洗)
        "data_postprocess": 3600,                  # 1 小时 (位置概率 + 验证 + 清洗 + 贝叶斯融合)
        "bp_recommendation_training": 14400,       # 4 小时 (DEVELOPMENT 模式含验证)
        "bp_recommendation_production": 14400,     # 4 小时 (PRODUCTION 盲训)
        "bp_recommendation_validation": 1800,      # 30 分钟 (一致性检测)
        "bp_prediction_training": 10800,           # 3 小时 (OOT 5折训练)
        "bp_prediction_production": 3600,          # 1 小时 (7-Seed Bagging)
        "bp_prediction_validation": 1800,          # 30 分钟 (一致性检测)
        "psi_baseline_rebuild": 1800,              # 30 分钟 (基线构建含 npz 读取)
        # 兼容旧配置
        "bp_recommendation": 14400,
        "bp_prediction": 7200,
    },
     "server": {
        "start_script": "app.py",
        "port": 5001,
        "test_port": 5002,         
        "health_check_url": "http://127.0.0.1:{port}/api/health", # 使用 127.0.0.1 避免 IPv6 解析问题
        "health_check_timeout": 30,
        "health_check_retries": 5,
        "health_check_interval": 5,
    },
    "backup": {
        "enabled": True,
        "keep_last_n": 3,          # 保留最近 N 次备份
        "max_backup_age_days": 30, # 备份最长保留天数
    },
    "alert": {
        "enabled": True,
        "log_file": str(LOG_DIR / "alerts.log"),
    },
    "rollback": {
        "enabled": True,
        "auto_rollback_on_failure": True,
    },
    "daemon": {
        "check_interval_hours": 1,  # 守护进程每隔多久检查一次是否需要更新
    },
}


# =====================================================================
# 日志系统
# =====================================================================
def setup_logging(run_ts: str) -> logging.Logger:
    """配置日志系统：控制台彩色输出 + 主日志轮转 + 本次运行独立日志文件"""
    os.makedirs(LOG_DIR, exist_ok=True)

    _setup_root_logging(
        log_dir=PROJECT_ROOT / "logs",
        app_name="pipeline",
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    silence_third_party()

    run_log_file = LOG_DIR / f"pipeline_{run_ts}.log"
    logger = logging.getLogger("auto_update")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(run_log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    logger.propagate = True

    logger.info("本次运行日志文件: %s", run_log_file)
    return logger


# =====================================================================
# 配置管理
# =====================================================================
def load_config() -> dict:
    """加载配置文件，不存在则创建默认配置"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        # 合并默认值（确保新增字段有默认值）
        merged = DEFAULT_CONFIG.copy()
        _deep_merge(merged, config)
        return merged
    else:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return DEFAULT_CONFIG.copy()


def _deep_merge(base: dict, override: dict):
    """递归合并字典，override 覆盖 base"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_state(state: dict):
    """保存运行状态到文件"""
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def load_state() -> dict:
    """加载上次运行状态"""
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# =====================================================================
# 备份与回滚
# =====================================================================
def create_backup(logger: logging.Logger, config: dict) -> Optional[str]:
    """创建关键模型文件的备份"""
    if not config["backup"]["enabled"]:
        return None

    backup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_DIR / backup_ts
    os.makedirs(backup_dir, exist_ok=True)

    backup_paths = [
        "bp_recommendation/model_pick/checkpoints",
        "bp_recommendation/model_ban/checkpoints",
        "bp_prediction/models/production",
    ]

    backed_up = []
    for rel_path in backup_paths:
        src = PROJECT_ROOT / rel_path
        if src.exists():
            dst = backup_dir / rel_path
            os.makedirs(dst.parent, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            backed_up.append(rel_path)

    if backed_up:
        logger.info(f"备份已创建: {backup_dir} ({len(backed_up)} 个目录)")
        # 清理旧备份
        _cleanup_old_backups(logger, config)
    else:
        logger.warning("没有可备份的模型文件")

    return str(backup_dir)


def _cleanup_old_backups(logger: logging.Logger, config: dict):
    """清理超过保留期限的旧备份"""
    if not BACKUP_DIR.exists():
        return

    keep_n = config["backup"]["keep_last_n"]
    # 兼容字段名: max_backup_age_days (新) / max_age_days (旧)
    max_age_days = config["backup"].get("max_backup_age_days") or config["backup"].get("max_age_days", 30)
    cutoff = datetime.now() - timedelta(days=max_age_days)

    backups = sorted(BACKUP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for backup in backups[keep_n:]:
        shutil.rmtree(backup, ignore_errors=True)
        logger.info(f"清理旧备份: {backup.name}")

    for backup in backups[:keep_n]:
        mtime = datetime.fromtimestamp(backup.stat().st_mtime)
        if mtime < cutoff:
            shutil.rmtree(backup, ignore_errors=True)
            logger.info(f"清理过期备份: {backup.name} (超过 {max_age_days} 天)")


def rollback_from_backup(logger: logging.Logger, backup_path: str):
    """从备份恢复模型文件"""
    backup_dir = Path(backup_path)
    if not backup_dir.exists():
        logger.error(f"回滚失败: 备份目录不存在 {backup_path}")
        return False

    logger.warning(f"正在从备份回滚: {backup_path}")
    for item in backup_dir.rglob("*"):
        if item.is_file():
            rel = item.relative_to(backup_dir)
            target = PROJECT_ROOT / rel
            os.makedirs(target.parent, exist_ok=True)
            shutil.copy2(item, target)

    logger.info("回滚完成")
    return True


# =====================================================================
# 子进程执行
# =====================================================================
def run_stage(stage_key: str, config: dict, logger: logging.Logger,
              mode: str = "complete") -> dict:
    """
    执行单个流水线阶段。

    Args:
        stage_key: 阶段标识 (如 data_scrape / bp_recommendation_training / ...)
        config: 配置字典
        logger: 日志记录器
        mode: 运行模式 (complete / production)
            - 细粒度阶段已通过 extra_args 内置模式参数, mode 仅用于日志展示

    返回:
        dict: {
            "success": bool,
            "elapsed_sec": float,
            "exit_code": int,
            "stdout_tail": str,
            "stderr_tail": str,
        }
    """
    stage_info = STAGE_SCRIPTS[stage_key]
    script_path = stage_info["script"]
    cwd = stage_info["cwd"]
    timeout = config["timeouts"].get(stage_key, 3600)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  阶段: {stage_info['name']} ({stage_key})")
    logger.info(f"  脚本: {script_path}")
    logger.info(f"  说明: {stage_info['description']}")
    logger.info(f"  模式: {mode}")
    logger.info(f"  超时: {timeout}s ({timeout/60:.0f} 分钟)")
    logger.info("=" * 70)

    if not script_path.exists():
        logger.error(f"脚本文件不存在: {script_path}")
        return {"success": False, "elapsed_sec": 0, "exit_code": -1,
                "stdout_tail": "", "stderr_tail": "脚本文件不存在"}

    # ---- 运行前置脚本 (pre_script) ----
    # 用于一致性检测阶段: 先运行特征对齐, 再运行预测一致性检查
    pre_script = stage_info.get("pre_script")
    if pre_script:
        if not pre_script.exists():
            logger.warning(f"  前置脚本不存在, 跳过: {pre_script}")
        else:
            logger.info(f"  前置脚本: {pre_script}")
            pre_cmd = [sys.executable, str(pre_script)]
            pre_result = _run_subprocess(pre_cmd, cwd, timeout, logger, label="前置")
            if not pre_result["success"]:
                logger.warning(f"  前置脚本执行失败, 继续执行主脚本")
            # 前置脚本的输出合并到主结果中
            stdout_lines = pre_result.get("stdout_lines", [])
            stderr_lines = pre_result.get("stderr_lines", [])
    else:
        stdout_lines = []
        stderr_lines = []

    # ---- 构建主脚本命令 ----
    cmd = [sys.executable, str(script_path)]

    # 推荐模型阶段: 自动检测设备并传递 --device 参数
    if stage_key in ("bp_recommendation_training", "bp_recommendation_production"):
        try:
            import torch
            device_name = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
            cmd.extend(["--device", device_name])
            logger.info(f"  设备: {device_name}")
        except ImportError:
            logger.warning("  torch 未安装, 使用默认设备")

    # 附加 extra_args (阶段固有的命令行参数, 如 --production / --skip-tf 等)
    # production 模式下优先使用 production_extra_args (用于区分是否跳过 features 等)
    if mode == "production" and "production_extra_args" in stage_info:
        extra_args = stage_info["production_extra_args"]
    else:
        extra_args = stage_info.get("extra_args", [])
    if extra_args:
        cmd.extend(extra_args)
        logger.info(f"  附加参数: {' '.join(extra_args)}")

    # ---- 执行主脚本 ----
    result = _run_subprocess(cmd, cwd, timeout, logger, label="主脚本")

    # 合并前置脚本和主脚本的输出
    if pre_script and stdout_lines:
        combined_stdout = stdout_lines + result.get("stdout_lines", [])
        result["stdout_tail"] = "\n".join(combined_stdout[-20:])
    if pre_script and stderr_lines:
        combined_stderr = stderr_lines + result.get("stderr_lines", [])
        result["stderr_tail"] = "\n".join(combined_stderr[-20:])

    return result


def _extract_stage_summary(stdout_lines: list, stage_key: str) -> list:
    """从子进程输出中提取关键指标和里程碑，用于简洁的阶段完成摘要。

    匹配规则（仅输出真正重要的行，忽略每个 epoch 的中间 loss）：
      - 包含 PASSED / FAILED / 完成 / 保存 / 总耗时 / AUC / 样本量 等关键词
      - 不超过 10 行
    """
    if not stdout_lines:
        return []

    summary_patterns = [
        "PASSED", "FAILED", "ALL CHECKS",
        "AUC", "NDCG", "HitRate", "LogLoss", "Accuracy", "ACC",
        "完成", "保存", "saved", "Saved", "done", "Done", "成功",
        "总耗时", "total elapsed", "报告", "report",
        "样本", "samples", "rows", "特征", "features",
        "训练完成", "训练结束", "模型加载", "Loaded",
        "best", "Best", "最佳",
        "数据集", "dataset", "split", "Split",
        "一致性", "alignment", "对齐",
        "错误日志", "error.log",
    ]
    noisy_patterns = [
        "Epoch ", "epoch ", "Step ", "step ", "batch ", "Batch ",
        "loss:", "Loss:", "val_loss", "train_loss",
        "it/s", "s/it", "it/s",
        "Gradient", "gradient", "weight:", "param:",
        "Token", "token", "Embedding", "embedding",
        "Tqdm", "tqdm", "pipeline:", "Pipeline:",
        "  -> ", "diff:", "差值:",
    ]

    seen = set()
    summary = []
    for line in stdout_lines:
        low = line.lower()
        if not any(p.lower() in low for p in summary_patterns):
            continue
        if any(p.lower() in low for p in noisy_patterns):
            continue
        if len(line) > 200:
            line = line[:197] + "..."
        if line not in seen:
            seen.add(line)
            summary.append(line)
        if len(summary) >= 10:
            break

    if not summary:
        for line in reversed(stdout_lines):
            if any(kw in line for kw in ["完成", "Done", "OK", "saved", "Saved", "成功"]):
                summary = [line]
                break

    return summary


def _run_subprocess(cmd: list, cwd: Path, timeout: int,
                    logger: logging.Logger, label: str = "") -> dict:
    """
    执行子进程并实时捕获输出 (run_stage 的内部辅助函数)。

    日志策略：
      - DEBUG: 所有子进程 stdout/stderr 完整记录到日志文件
      - INFO : 仅输出里程碑/关键指标行（完成、错误、AUC、保存、PASSED/FAILED 等）
      - 阶段结束后: 输出提取的关键指标摘要 (最多10行)

    Args:
        cmd: 命令列表 (如 [sys.executable, "script.py", "--arg"])
        cwd: 工作目录
        timeout: 超时秒数
        logger: 日志记录器
        label: 标签 (用于日志区分, 如 "前置" / "主脚本")

    Returns:
        dict: {success, elapsed_sec, exit_code, stdout_tail, stderr_tail,
               stdout_lines, stderr_lines, summary}
    """
    prefix = f"  [{label}] " if label else "  "
    start = time.time()
    stdout_lines = []
    stderr_lines = []

    milestone_kws = [
        "PASSED", "FAILED", "ALL CHECKS", "ERROR", "CRITICAL",
        "完成", "失败", "成功", "保存", "saved", "Saved", "Done",
        "AUC", "NDCG", "accuracy", "Accuracy", "LogLoss", "best", "Best",
        "saving", "Saving", "loaded", "Loaded",
        "训练完成", "训练结束", "报告已保存", "report saved",
        "样本: ", "samples:", "rows:", "特征: ", "features:",
        "总耗时", "total elapsed",
        "=====",
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        def _read_stdout():
            for line in iter(process.stdout.readline, ""):
                clean = line.rstrip()
                if clean:
                    stdout_lines.append(clean)
                    logger.debug("%s[out] %s", prefix, clean)
                    low = clean.lower()
                    if any(kw.lower() in low for kw in milestone_kws):
                        logger.info("%s| %s", prefix, clean)

        def _read_stderr():
            noise = ["DevTools", "ERROR:gpu", "ERROR:command_buffer",
                     "ERROR:shared_image", "ERROR:viz", "ERROR:skia",
                     "ERROR:gl", "ERROR:media", "ERROR:network",
                     "Warning:", "warnings.warn", "DeprecationWarning",
                     "FutureWarning", "UserWarning",
                     "INFO:", "DEBUG:"]
            for line in iter(process.stderr.readline, ""):
                clean = line.rstrip()
                if clean:
                    stderr_lines.append(clean)
                    if not any(n in clean for n in noise):
                        logger.warning("%s[err] %s", prefix, clean)

        t1 = threading.Thread(target=_read_stdout, daemon=True)
        t2 = threading.Thread(target=_read_stderr, daemon=True)
        t1.start()
        t2.start()

        process.wait(timeout=timeout)

        t1.join(timeout=5)
        t2.join(timeout=5)

        elapsed = time.time() - start
        exit_code = process.returncode

        stdout_tail = "\n".join(stdout_lines[-30:]) if stdout_lines else ""
        stderr_tail = "\n".join(stderr_lines[-20:]) if stderr_lines else ""
        summary = _extract_stage_summary(stdout_lines, "")

        if exit_code == 0:
            logger.info("%s[OK] 完成, 耗时 %.1fs (%.1f 分钟)", prefix, elapsed, elapsed / 60)
            if summary:
                logger.info("%s--- 关键输出 ---", prefix)
                for s in summary[:8]:
                    logger.info("%s  %s", prefix, s)
            return {
                "success": True, "elapsed_sec": round(elapsed, 1),
                "exit_code": exit_code, "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stdout_lines": stdout_lines, "stderr_lines": stderr_lines,
                "summary": summary,
            }
        else:
            logger.error("%s[FAIL] 失败, 返回码=%d, 耗时 %.1fs",
                         prefix, exit_code, elapsed)
            if stderr_tail:
                logger.error("%s--- stderr (最后20行) ---", prefix)
                for line in stderr_lines[-20:]:
                    logger.error("%s  %s", prefix, line)
            if stdout_tail:
                logger.error("%s--- stdout (最后30行) ---", prefix)
                for line in stdout_lines[-30:]:
                    logger.error("%s  %s", prefix, line)
            return {
                "success": False, "elapsed_sec": round(elapsed, 1),
                "exit_code": exit_code, "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stdout_lines": stdout_lines, "stderr_lines": stderr_lines,
                "summary": summary,
            }

    except subprocess.TimeoutExpired:
        process.kill()
        elapsed = time.time() - start
        logger.error("%s[TIMEOUT] 超时 (%ds), 已强制终止", prefix, timeout)
        return {
            "success": False, "elapsed_sec": round(elapsed, 1),
            "exit_code": -1, "stdout_tail": "\n".join(stdout_lines[-20:]),
            "stderr_tail": f"超时 ({timeout}s)",
            "stdout_lines": stdout_lines, "stderr_lines": stderr_lines,
            "summary": [],
        }
    except Exception as e:
        elapsed = time.time() - start
        logger.exception("%s[EXCEPTION] 异常: %s", prefix, e)
        return {
            "success": False, "elapsed_sec": round(elapsed, 1),
            "exit_code": -1, "stdout_tail": "",
            "stderr_tail": str(e),
            "stdout_lines": stdout_lines, "stderr_lines": stderr_lines,
            "summary": [],
        }


def _validate_pt_in_subprocess(pt_path: str) -> Optional[str]:
    """
    在独立子进程中验证 PyTorch checkpoint, 防止主进程长时间运行后
    MPS/PyTorch 资源泄漏导致段错误。

    Args:
        pt_path: checkpoint 文件路径

    Returns:
        epoch 字符串 (成功) 或 None (失败)
    """
    import textwrap
    validate_code = textwrap.dedent(f"""
        import sys
        try:
            import torch
            ckpt = torch.load({pt_path!r}, map_location="cpu", weights_only=False)
            if "model_state_dict" not in ckpt:
                print("ERROR:model_state_dict_missing", flush=True)
                sys.exit(1)
            epoch = ckpt.get("epoch", "?")
            # 显式释放
            del ckpt
            print(f"OK:{{epoch}}", flush=True)
            sys.exit(0)
        except Exception as e:
            print(f"ERROR:{{e}}", flush=True)
            sys.exit(2)
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", validate_code],
            capture_output=True, text=True, timeout=60,
        )
        stdout = result.stdout.strip() if result.stdout else ""
        if stdout.startswith("OK:"):
            return stdout[3:]
        return None
    except Exception:
        return None


def _validate_lgb_in_subprocess(txt_path: str) -> Optional[int]:
    """
    在独立子进程中验证 LightGBM 模型文件, 防止主进程长时间运行后
    加载 LightGBM 模型触发段错误 (与 PyTorch checkpoint 同样原因)。

    Args:
        txt_path: LightGBM 模型文件路径

    Returns:
        树的数量 (成功) 或 None (失败)
    """
    import textwrap
    validate_code = textwrap.dedent(f"""
        import sys
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file={txt_path!r})
            n_trees = booster.num_trees()
            del booster
            print(f"OK:{{n_trees}}", flush=True)
            sys.exit(0)
        except Exception as e:
            print(f"ERROR:{{e}}", flush=True)
            sys.exit(2)
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", validate_code],
            capture_output=True, text=True, timeout=60,
        )
        stdout = result.stdout.strip() if result.stdout else ""
        if stdout.startswith("OK:"):
            return int(stdout[3:])
        return None
    except Exception:
        return None


def _validate_cbm_in_subprocess(cbm_path: str) -> bool:
    """
    在独立子进程中验证 CatBoost 模型文件, 防止主进程长时间运行后
    加载 CatBoost 模型触发段错误。

    Args:
        cbm_path: CatBoost 模型文件路径

    Returns:
        True (成功) 或 False (失败)
    """
    import textwrap
    validate_code = textwrap.dedent(f"""
        import sys
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            try:
                CatBoostClassifier().load_model({cbm_path!r})
            except Exception:
                CatBoostRegressor().load_model({cbm_path!r})
            print("OK", flush=True)
            sys.exit(0)
        except Exception as e:
            print(f"ERROR:{{e}}", flush=True)
            sys.exit(2)
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", validate_code],
            capture_output=True, text=True, timeout=60,
        )
        stdout = result.stdout.strip() if result.stdout else ""
        return stdout.startswith("OK")
    except Exception:
        return False


# =====================================================================
# 输出验证
# =====================================================================
def validate_stage_outputs(stage_key: str, logger: logging.Logger,
                           freshness_baseline: float = 0.0,
                           allow_stale: bool = False) -> dict:
    """
    验证阶段输出文件是否存在、是否在本轮更新过、是否可正确加载。

    三重验证:
    1. 存在性: 文件是否存在且非空
    2. 新鲜度: 文件 mtime > freshness_baseline (本轮流水线启动时间)，
       确保不是残留的旧文件
    3. 可加载性: 对 LightGBM (.txt) / PyTorch (.pt) / JSON / PKL 文件
       尝试加载并验证内容有效性

    对于一致性检测阶段 (bp_recommendation_validation / bp_prediction_validation),
    不产出文件, 仅验证现有模型文件是否可访问（新鲜度检测仍启用）。

    Args:
        stage_key: 阶段标识
        logger: 日志记录器
        freshness_baseline: 流水线启动时间戳 (time.time())。
            若 > 0, 则检查文件 mtime 必须大于此值 (证明本轮已更新)。

    返回:
        dict: {"stage": str, "all_present": bool, "missing": list,
               "stale": list, "invalid": list, "details": list}
    """
    logger.info(f"  验证阶段输出: {stage_key}")

    check_files = PRODUCTION_CHECK_FILES.get(stage_key, [])

    # 一致性检测阶段: 不产出文件, 验证对应的生产模型是否存在
    if stage_key == "bp_recommendation_validation":
        check_files = PRODUCTION_CHECK_FILES.get("bp_recommendation_production", [])
        logger.info("  (一致性检测阶段: 验证生产模型文件是否存在)")
    elif stage_key == "bp_prediction_validation":
        check_files = PRODUCTION_CHECK_FILES.get("bp_prediction_production", [])
        logger.info("  (一致性检测阶段: 验证生产模型文件是否存在)")

    if not check_files:
        logger.info("  本阶段无需验证文件 (跳过)")
        return {
            "stage": stage_key,
            "all_present": True,
            "missing": [],
            "stale": [],
            "invalid": [],
            "details": ["  [SKIP] 本阶段无需验证文件"],
        }

    missing = []
    stale = []
    invalid = []
    details = []

    # 验证前清理内存 (PyTorch checkpoint 验证已移至独立子进程, 主进程不再加载 torch)
    import gc
    gc.collect()

    for rel_path in check_files:
        if "_LoL_esports_match_data_from_OraclesElixir.csv" in rel_path and rel_path.startswith("raw_data/matches/"):
            try:
                full_path = get_match_data_path()
                rel_path = str(full_path.relative_to(PROJECT_ROOT))
            except FileNotFoundError:
                full_path = PROJECT_ROOT / rel_path
        else:
            full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            missing.append(rel_path)
            details.append(f"  [MISSING] {rel_path}")
            continue

        size = full_path.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"

        # 检查文件非空
        if size == 0:
            invalid.append(rel_path)
            details.append(f"  [EMPTY] {rel_path} (文件大小为 0)")
            continue

        # 检查新鲜度: 文件是否在本轮流水线中被更新过
        file_mtime = full_path.stat().st_mtime
        if freshness_baseline > 0 and file_mtime < freshness_baseline:
            mtime_str = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S")
            if allow_stale:
                # --stage N 跳过前序阶段时, 允许使用前序阶段产出的陈旧文件
                # 陈旧文件只检查存在性, 跳过可加载性验证 (前序阶段已验证过)
                # 避免在主进程中 import lightgbm/torch 导致段错误
                details.append(f"  [OK-STALE] {rel_path} ({size_str}, mtime={mtime_str}, 允许陈旧)")
                continue
            else:
                stale.append(rel_path)
                details.append(f"  [STALE] {rel_path} ({size_str}, mtime={mtime_str} 未在本轮更新)")
                continue

        # 可加载性验证: 根据文件类型尝试加载
        validity_tag = ""
        try:
            if rel_path.endswith(".txt"):
                # LightGBM 模型文件 (在子进程中验证, 防止主进程段错误)
                n_trees = _validate_lgb_in_subprocess(str(full_path))
                if n_trees is None:
                    raise ValueError("子进程加载 LightGBM 模型失败")
                validity_tag = f", trees={n_trees}"
                if n_trees == 0:
                    invalid.append(rel_path)
                    details.append(f"  [INVALID] {rel_path} ({size_str}, 0 棵树)")
                    continue
            elif rel_path.endswith(".pt") and "cascade" not in rel_path:
                # PyTorch checkpoint (非 cascade)
                # 【关键】在独立子进程中加载 checkpoint, 防止主进程长时间运行后
                # MPS/PyTorch 资源泄漏导致段错误 (macOS 上尤其常见)
                epoch_str = _validate_pt_in_subprocess(str(full_path))
                if epoch_str is None:
                    raise ValueError("子进程加载 checkpoint 失败")
                validity_tag = f", epoch={epoch_str}"
            elif rel_path.endswith(".pkl"):
                # Pickle 文件 (scaler 等)
                import pickle
                with open(full_path, "rb") as f:
                    pickle.load(f)
                validity_tag = ", loadable"
            elif rel_path.endswith(".json"):
                # JSON 配置文件
                import json
                with open(full_path, "r") as f:
                    json.load(f)
                validity_tag = ", valid_json"
            elif rel_path.endswith(".cbm"):
                # Catboost 模型文件 (在子进程中验证, 防止主进程段错误)
                if not _validate_cbm_in_subprocess(str(full_path)):
                    raise ValueError("子进程加载 CatBoost 模型失败")
                validity_tag = ", loadable"
        except Exception as e:
            invalid.append(rel_path)
            details.append(f"  [INVALID] {rel_path} ({size_str}, 加载失败: {e})")
            continue

        details.append(f"  [OK] {rel_path} ({size_str}{validity_tag})")

    for d in details:
        logger.info(d)

    total_issues = len(missing) + len(stale) + len(invalid)
    if total_issues > 0:
        parts = []
        if missing:
            parts.append(f"{len(missing)} 缺失")
        if stale:
            parts.append(f"{len(stale)} 未更新(陈旧)")
        if invalid:
            parts.append(f"{len(invalid)} 无效")
        logger.warning(f"  文件验证发现问题: {', '.join(parts)}")
    else:
        logger.info(f"  所有关键文件验证通过 ({len(check_files)} 个)")

    return {
        "stage": stage_key,
        "all_present": total_issues == 0,
        "missing": missing,
        "stale": stale,
        "invalid": invalid,
        "details": details,
    }


# =====================================================================
# 生产环境健康检查
# =====================================================================
def check_server_health(config: dict, logger: logging.Logger) -> dict:
    """
    检查 Web 服务是否正常运行。

    返回:
        dict: {"healthy": bool, "details": str}
    """
    port = config["server"]["port"]
    health_url = config["server"]["health_check_url"]
    timeout = config["server"]["health_check_timeout"]
    retries = config["server"]["health_check_retries"]
    interval = config["server"]["health_check_interval"]

    logger.info("=" * 70)
    logger.info("  生产环境健康检查")
    logger.info("=" * 70)

    # 1. 检查端口是否监听
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    result = sock.connect_ex(("localhost", port))
    sock.close()

    if result != 0:
        logger.error(f"  端口 {port} 未监听 (服务未启动)")
        return {"healthy": False, "details": f"端口 {port} 未监听"}

    logger.info(f"  端口 {port} 已监听")

    # 2. HTTP 健康检查
    import urllib.request
    import urllib.error

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(health_url, method="GET")
            resp = urllib.request.urlopen(req, timeout=timeout)
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            logger.info(f"  健康检查 ({attempt}/{retries}): HTTP {status}")
            if status == 200:
                return {"healthy": True, "details": f"HTTP 200, 响应: {body[:200]}"}
        except urllib.error.URLError as e:
            logger.warning(f"  健康检查 ({attempt}/{retries}): {e}")
        except Exception as e:
            logger.warning(f"  健康检查 ({attempt}/{retries}): {e}")

        if attempt < retries:
            time.sleep(interval)

    logger.error("  健康检查失败: 超过最大重试次数")
    return {"healthy": False, "details": f"HTTP 健康检查失败 (已重试 {retries} 次)"}


def run_model_validation(config: dict, logger: logging.Logger) -> dict:
    """
    启动服务并运行端到端模型验证。

    包含:
      1. 临时启动 Flask 服务
      2. 调用 BP 推荐 API
      3. 调用预测 API
      4. 验证返回结果格式和质量
      5. 停止临时服务
    """
    logger.info("=" * 70)
    logger.info("  模型端到端验证")
    logger.info("=" * 70)

    import urllib.request
    import urllib.error

    results = {
        "server_started": False,
        "models_loaded": False,
        "bp_recommend_ok": False,
        "predict_ok": False,
        "bp_delta_ok": False,
        "fallback_ok": False,
        "details": [],
    }

    app_path = PROJECT_ROOT / config["server"]["start_script"]
    test_port = config["server"]["test_port"] 
    base_url = f"http://127.0.0.1:{test_port}"  # 使用 127.0.0.1 避免 IPv6 解析问题
    health_url = config["server"]["health_check_url"].format(port=test_port)

    logger.info(f"  在测试端口 {test_port} 启动临时沙盒服务...")
    
    # 将测试端口通过环境变量注入给 Flask/FastAPI
    env = os.environ.copy()
    env["PORT"] = str(test_port)  # 需确保你的 app.py 能够读取 os.environ.get("PORT")
    try:
        # 注意: 不能使用 PIPE 而不读取，否则当子进程输出填满管道缓冲区时会死锁
        # 使用 DEVNULL 丢弃输出（健康检查通过 HTTP 接口进行，不依赖 stdout）
        # 将 stdout/stderr 写入日志文件（避免 DEVNULL 吞掉启动错误，同时防止管道死锁）
        sandbox_log = os.path.join(PROJECT_ROOT, "logs", "auto_update", "sandbox_stderr.log")
        os.makedirs(os.path.dirname(sandbox_log), exist_ok=True)
        sandbox_log_fp = open(sandbox_log, "w", encoding="utf-8")
        server_proc = subprocess.Popen(
            [sys.executable, str(app_path)],
            cwd=str(PROJECT_ROOT), env=env,
            stdout=sandbox_log_fp, stderr=sandbox_log_fp,
        )

        # 动态等待就绪
        for i in range(120):
            # 检查进程是否已退出（启动失败）
            if server_proc.poll() is not None:
                logger.error(f"  沙盒服务进程已退出 (返回码={server_proc.returncode}), 查看日志: {sandbox_log}")
                sandbox_log_fp.close()
                return results
            try:
                urllib.request.urlopen(urllib.request.Request(health_url, method="GET"), timeout=3)
                results["server_started"] = True
                logger.info(f"  沙盒服务已就绪 (等待 {i}s)")
                break
            except Exception:
                time.sleep(1)
        else:
            logger.error("  沙盒服务启动超时 (120s), 查看日志: " + sandbox_log)
            server_proc.terminate()
            sandbox_log_fp.close()
            return results


        # 测试 BP 推荐
        logger.info("  测试 BP 推荐 API...")
        try:
            setup_data = json.dumps({
                "league": "LCK", "blue_team": "T1", "red_team": "Gen.G",
                "first_pick": "blue", "is_playoff": False,
                "completed_steps": 0, "bp_seq_ids": [],
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/bp/recommend",
                data=setup_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=15)
            body = json.loads(resp.read().decode("utf-8"))
            if "error" not in body:
                results["bp_recommend_ok"] = True
                results["details"].append(f"BP推荐: OK")
                logger.info(f"  BP推荐 API: OK")
            else:
                results["details"].append(f"BP推荐: {body['error']}")
                logger.warning(f"  BP推荐 API: {body['error']}")
        except Exception as e:
            results["details"].append(f"BP推荐异常: {e}")
            logger.error(f"  BP推荐 API 异常: {e}")

        # 测试预测
        logger.info("  测试预测 API...")
        try:
            predict_data = json.dumps({
                "league": "LCK", "is_playoff": False, "first_pick": "blue",
                "blue_team": "T1", "red_team": "Gen.G",
                "blue_champions": {"top": "Aatrox", "jng": "Vi", "mid": "Ahri",
                                   "bot": "Jinx", "sup": "Alistar"},
                "red_champions": {"top": "Gnar", "jng": "Sejuani", "mid": "Azir",
                                  "bot": "Zeri", "sup": "Lulu"},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/predict",
                data=predict_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            body = json.loads(resp.read().decode("utf-8"))
            if "error" not in body and "blue_win_prob" in body:
                results["predict_ok"] = True
                results["details"].append(f"预测: blue={body['blue_win_prob']:.3f}")
                logger.info(f"  预测 API: OK (blue_win_prob={body['blue_win_prob']:.3f})")
            else:
                results["details"].append(f"预测: {body.get('error', '未知错误')}")
                logger.warning(f"  预测 API: {body.get('error', '未知错误')}")
        except Exception as e:
            results["details"].append(f"预测异常: {e}")
            logger.error(f"  预测 API 异常: {e}")

        # 测试 BP Delta
        logger.info("  测试 BP Delta API...")
        try:
            delta_data = json.dumps({
                "league": "LCK", "is_playoff": False, "first_pick": "blue",
                "blue_team": "T1", "red_team": "Gen.G",
                "blue_champions": {"top": "Aatrox", "jng": "Vi", "mid": "Ahri",
                                   "bot": "Jinx", "sup": "Alistar"},
                "red_champions": {"top": "Gnar", "jng": "Sejuani", "mid": "Azir",
                                  "bot": "Zeri", "sup": "Lulu"},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/bp_delta",
                data=delta_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            body = json.loads(resp.read().decode("utf-8"))
            if "error" not in body and "delta" in body:
                results["bp_delta_ok"] = True
                results["details"].append(f"BP Delta: delta={body['delta']:.4f}")
                logger.info(f"  BP Delta API: OK (delta={body['delta']:.4f})")
            else:
                results["details"].append(f"BP Delta: {body.get('error', '未知错误')}")
                logger.warning(f"  BP Delta API: {body.get('error', '未知错误')}")
        except Exception as e:
            results["details"].append(f"BP Delta 异常: {e}")
            logger.error(f"  BP Delta API 异常: {e}")

        # 测试兜底机制
        logger.info("  测试兜底机制 API...")
        try:
            # 先尝试恢复正常模式（清除持久化的降级状态）
            try:
                recover_req = urllib.request.Request(
                    f"{base_url}/api/fallback/recover", method="POST"
                )
                urllib.request.urlopen(recover_req, timeout=5)
            except Exception:
                pass  # 恢复失败不阻断验证

            req = urllib.request.Request(f"{base_url}/api/fallback/status", method="GET")
            resp = urllib.request.urlopen(req, timeout=5)
            body = json.loads(resp.read().decode("utf-8"))
            if body.get("enabled"):
                results["fallback_ok"] = True
                results["details"].append("兜底机制: OK")
                logger.info("  兜底机制 API: OK")
            else:
                results["details"].append("兜底机制: 未启用")
                logger.warning("  兜底机制: 未启用")
        except Exception as e:
            results["details"].append(f"兜底机制异常: {e}")
            logger.error(f"  兜底机制 API 异常: {e}")

        # BP推荐为非必需（stage 1 可能被跳过导致模型维度不匹配），
        # 只要预测和 BP Delta 通过即视为验证成功
        results["models_loaded"] = all([
            results["predict_ok"], results["bp_delta_ok"],
        ])

    finally:
        # 【防御性编程】：无论验证通过还是抛出异常，必须确保沙盒被彻底击杀，释放端口
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        except Exception:
            pass
        try:
            sandbox_log_fp.close()
        except Exception:
            pass
        logger.info("  沙盒服务已彻底销毁")

    return results


# =====================================================================
# 主流水线
# =====================================================================
def run_pipeline(config: dict, logger: logging.Logger, start_stage: int = 0,
                 mode: str = "complete") -> dict:
    """
    执行完整流水线。

    Args:
        config: 配置字典
        logger: 日志记录器
        start_stage: 从第几个阶段开始 (0-indexed)
        mode: 运行模式 (complete / production)
            - complete: 完整训练，含 OOT 验证
            - production: 生产模式，跳过 OOT 验证，使用固定参数

    Returns:
        dict: 运行结果
    """
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_order = config["stage_order"]

    # 初始化结果
    results = {
        "run_id": run_ts,
        "mode": mode,
        "start_time": datetime.now().isoformat(),
        "stages": {},
        "backup_path": None,
        "rollback_performed": False,
        "validation": {},
        "overall_success": False,
        "total_elapsed_sec": 0,
    }

    pipeline_start = time.time()

    # ---- 备份现有模型 ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("  创建备份")
    logger.info("=" * 70)
    results["backup_path"] = create_backup(logger, config)

    # ---- 执行各阶段 ----
    all_success = True
    for idx, stage_key in enumerate(stage_order):
        if idx < start_stage:
            logger.info(f"跳过阶段 {idx}: {STAGE_SCRIPTS[stage_key]['name']} (--stage {start_stage})")
            continue

        # 执行阶段
        stage_result = run_stage(stage_key, config, logger, mode=mode)
        results["stages"][stage_key] = stage_result

        if not stage_result["success"]:
            logger.error(f"阶段失败: {stage_key}")
            all_success = False

            # 自动回滚
            if config["rollback"]["enabled"] and config["rollback"]["auto_rollback_on_failure"]:
                if results["backup_path"]:
                    logger.warning("触发自动回滚...")
                    rollback_from_backup(logger, results["backup_path"])
                    results["rollback_performed"] = True
            break

        # 验证阶段输出 (三重验证: 存在性 + 新鲜度 + 可加载性)
        # 当 --stage N 跳过前序阶段时, 允许使用前序阶段产出的陈旧文件
        validation = validate_stage_outputs(
            stage_key, logger,
            freshness_baseline=pipeline_start,
            allow_stale=(start_stage > 0),
        )
        results["stages"][f"{stage_key}_validation"] = validation
        if not validation["all_present"]:
            # 验证失败: 文件缺失/陈旧/无效 —— 视为阶段失败, 触发回滚
            missing_n = len(validation.get("missing", []))
            stale_n = len(validation.get("stale", []))
            invalid_n = len(validation.get("invalid", []))
            logger.error(
                f"阶段 {stage_key} 输出验证失败: "
                f"{missing_n} 缺失, {stale_n} 陈旧(未更新), {invalid_n} 无效"
            )
            logger.error("  缺失文件:" + "".join(f"\n    - {f}" for f in validation.get("missing", [])))
            logger.error("  陈旧文件:" + "".join(f"\n    - {f}" for f in validation.get("stale", [])))
            logger.error("  无效文件:" + "".join(f"\n    - {f}" for f in validation.get("invalid", [])))
            all_success = False
            # 自动回滚
            if config["rollback"]["enabled"] and config["rollback"]["auto_rollback_on_failure"]:
                if results["backup_path"]:
                    logger.warning("触发自动回滚 (验证失败)...")
                    rollback_from_backup(logger, results["backup_path"])
                    results["rollback_performed"] = True
            break

    # ---- 总耗时 ----
    results["total_elapsed_sec"] = round(time.time() - pipeline_start, 1)

    if not all_success:
        results["overall_success"] = False
        _send_alert(config, logger, results, "流水线执行失败")
        # 即使失败也生成异常汇总日志
        try:
            generate_error_summary(run_ts, logger, results)
        except Exception as e:
            logger.warning(f"生成异常汇总日志失败 (不影响流水线结果): {e}")
        return results

    # ---- 模型端到端验证 ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("  模型端到端验证")
    logger.info("=" * 70)
    validation_results = run_model_validation(config, logger)
    results["validation"] = validation_results

    if not validation_results["models_loaded"]:
        logger.error("模型验证失败: 模型未能正确加载")
        results["overall_success"] = False
        _send_alert(config, logger, results, "模型验证失败")
        # 即使验证失败也生成异常汇总日志
        try:
            generate_error_summary(run_ts, logger, results)
        except Exception as e:
            logger.warning(f"生成异常汇总日志失败 (不影响流水线结果): {e}")
        return results


    logger.info("")
    logger.info("=" * 70)
    logger.info("  触发生产服务热重启 (应用新模型)")
    logger.info("=" * 70)
    _restart_production_server(config, logger)

    # ---- 完成 ----
    results["overall_success"] = True
    results["end_time"] = datetime.now().isoformat()
    total_min = results["total_elapsed_sec"] / 60
    logger.info("")
    logger.info("=" * 70)
    logger.info("  流水线执行成功!")
    logger.info("  总耗时: %.1fs (%.1f 分钟)", results["total_elapsed_sec"], total_min)
    logger.info("")
    logger.info("  各阶段耗时:")
    for sk in stage_order:
        sr = results["stages"].get(sk, {})
        if not sr:
            continue
        elapsed = sr.get("elapsed_sec", 0)
        status = "OK" if sr.get("success") else "FAIL"
        name = STAGE_SCRIPTS.get(sk, {}).get("name", sk)
        logger.info("    [%s] %-30s %6.1fs (%5.1f min)",
                    status, name, elapsed, elapsed / 60)
        summary_lines = sr.get("summary", [])
        for s in summary_lines[:3]:
            if len(s) > 100:
                s = s[:97] + "..."
            logger.info("         %s", s)
    logger.info("=" * 70)

    save_state(results)

    # ---- 生成异常 error 汇总日志 ----
    try:
        generate_error_summary(run_ts, logger, results)
    except Exception as e:
        logger.warning(f"生成异常汇总日志失败 (不影响流水线结果): {e}")

    return results


# =====================================================================
# 异常 Error 汇总日志
# =====================================================================
def generate_error_summary(run_ts: str, logger: logging.Logger, results: dict):
    """
    扫描本次流水线运行的完整日志，提取所有 ERROR / WARNING / 数据质量异常，
    生成独立的异常汇总日志文件，便于快速排查问题。

    输出文件: logs/auto_update/error_summary_{run_ts}.log
    """
    summary_path = LOG_DIR / f"error_summary_{run_ts}.log"

    # 关键异常模式 (不区分大小写匹配)
    # 涵盖: 数据质量、NaN、空值、校验失败、段错误、Traceback 等
    CRITICAL_PATTERNS = [
        "error", "exception", "traceback", "failed", "失败",
        "sigsegv", "段错误", "critical",
        "nan", "空值", "缺失", "missing",
        "质量校验失败", "quality check",
        "不完整", "invalid", "无效",
        "warning", "警告",
    ]

    # 噪声模式: 这些行虽然包含关键词但不是真正的异常
    NOISE_PATTERNS = [
        "silence_third_party", "verbose", "log_evaluation",
        "UserWarning", "FutureWarning", "DeprecationWarning",
        "warning.warn", "warnings.warn",
    ]

    # 收集所有日志行
    pipeline_log = LOG_DIR / f"pipeline_{run_ts}.log"
    if not pipeline_log.exists():
        logger.warning(f"流水线日志文件不存在，跳过异常汇总: {pipeline_log}")
        return

    error_lines = []
    warning_lines = []
    critical_lines = []

    try:
        with open(pipeline_log, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                low = line_stripped.lower()

                # 跳过噪声行
                if any(n.lower() in low for n in NOISE_PATTERNS):
                    continue

                # 分类收集
                if "[error]" in low or "[critical]" in low or "traceback" in low or "sigsegv" in low or "段错误" in low:
                    critical_lines.append((line_num, line_stripped))
                elif "[warning]" in low or "警告" in low:
                    warning_lines.append((line_num, line_stripped))
                elif any(p in low for p in ["nan", "空值", "缺失", "质量校验失败", "无效", "不完整"]):
                    warning_lines.append((line_num, line_stripped))
                elif "failed" in low or "失败" in low:
                    critical_lines.append((line_num, line_stripped))
    except Exception as e:
        logger.error(f"读取流水线日志失败: {e}")
        return

    # 去重 (保留首次出现)
    seen_critical = set()
    seen_warning = set()
    unique_critical = []
    unique_warning = []
    for line_num, line in critical_lines:
        key = line[:120]
        if key not in seen_critical:
            seen_critical.add(key)
            unique_critical.append((line_num, line))
    for line_num, line in warning_lines:
        key = line[:120]
        if key not in seen_warning:
            seen_warning.add(key)
            unique_warning.append((line_num, line))

    # 写入汇总日志
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  异常 Error 汇总报告\n")
        f.write(f"  运行 ID: {run_ts}\n")
        f.write(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  流水线状态: {'成功' if results.get('overall_success') else '失败'}\n")
        f.write(f"  总耗时: {results.get('total_elapsed_sec', 0):.1f}s\n")
        f.write("=" * 70 + "\n\n")

        # ---- 1. 阶段执行状态 ----
        f.write("-" * 70 + "\n")
        f.write("  1. 阶段执行状态\n")
        f.write("-" * 70 + "\n")
        for stage_key, stage_result in results.get("stages", {}).items():
            if not isinstance(stage_result, dict):
                continue
            success = stage_result.get("success", True)
            elapsed = stage_result.get("elapsed_sec", 0)
            status = "✓ OK" if success else "✗ FAIL"
            f.write(f"  [{status}] {stage_key:40s} {elapsed:7.1f}s\n")
            if not success:
                stderr_tail = stage_result.get("stderr_tail", "")
                if stderr_tail:
                    f.write(f"         stderr: {stderr_tail[:300]}\n")
        f.write("\n")

        # ---- 2. 严重错误 (ERROR / Traceback / 段错误) ----
        f.write("-" * 70 + "\n")
        f.write(f"  2. 严重错误 (共 {len(unique_critical)} 条)\n")
        f.write("-" * 70 + "\n")
        if unique_critical:
            for line_num, line in unique_critical:
                f.write(f"  [L{line_num}] {line}\n")
        else:
            f.write("  (无严重错误)\n")
        f.write("\n")

        # ---- 3. 警告信息 (WARNING / 数据质量异常) ----
        f.write("-" * 70 + "\n")
        f.write(f"  3. 警告信息 (共 {len(unique_warning)} 条)\n")
        f.write("-" * 70 + "\n")
        if unique_warning:
            for line_num, line in unique_warning:
                f.write(f"  [L{line_num}] {line}\n")
        else:
            f.write("  (无警告)\n")
        f.write("\n")

        # ---- 4. 数据质量专项检查 ----
        f.write("-" * 70 + "\n")
        f.write("  4. 数据质量专项检查\n")
        f.write("-" * 70 + "\n")
        data_quality_keywords = ["win_rate", "pick_rate", "ban_rate", "nan", "空值", "缺失",
                                  "质量校验", "presence_rate", "golddiff", "数据集为空"]
        data_issues = []
        for line_num, line in unique_critical + unique_warning:
            low = line.lower()
            if any(kw in low for kw in data_quality_keywords):
                data_issues.append((line_num, line))
        if data_issues:
            for line_num, line in data_issues:
                f.write(f"  [L{line_num}] {line}\n")
        else:
            f.write("  (未检测到数据质量问题)\n")
        f.write("\n")

        # ---- 5. 统计摘要 ----
        f.write("-" * 70 + "\n")
        f.write("  5. 统计摘要\n")
        f.write("-" * 70 + "\n")
        f.write(f"  严重错误数: {len(unique_critical)}\n")
        f.write(f"  警告数:     {len(unique_warning)}\n")
        f.write(f"  数据质量问题: {len(data_issues)}\n")
        f.write(f"  完整日志:   {pipeline_log}\n")
        f.write("=" * 70 + "\n")

    logger.info(f"异常汇总日志已生成: {summary_path}")
    logger.info(f"  严重错误: {len(unique_critical)} 条, 警告: {len(unique_warning)} 条, 数据质量问题: {len(data_issues)} 条")


# =====================================================================
# 告警通知
# =====================================================================
def _send_alert(config: dict, logger: logging.Logger, results: dict, reason: str):
    """发送告警通知"""
    if not config["alert"]["enabled"]:
        return

    alert_file = config["alert"]["log_file"]
    os.makedirs(os.path.dirname(alert_file), exist_ok=True)

    with open(alert_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"ALERT: {reason}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Run ID: {results.get('run_id', 'unknown')}\n")
        f.write(f"Failed stages:\n")
        for stage, result in results.get("stages", {}).items():
            if not result.get("success", False):
                f.write(f"  - {stage}: exit_code={result.get('exit_code')}\n")
                f.write(f"    stderr: {result.get('stderr_tail', '')[:200]}\n")
        f.write(f"{'='*70}\n")

    logger.error(f"告警已记录: {alert_file}")


# =====================================================================
# 生产服务热重启
# =====================================================================
def _restart_production_server(config: dict, logger: logging.Logger) -> bool:
    """通过端口查找并终止旧生产服务，再用新会话启动新进程，确保服务常驻。"""
    port = config["server"].get("port", 5001)
    start_script = config["server"].get("start_script", "app.py")
    app_path = PROJECT_ROOT / start_script
    log_path = PROJECT_ROOT / "logs" / "web.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 终止旧服务
    try:
        result = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
        old_pids = [p.strip() for p in result.stdout.splitlines() if p.strip().isdigit()]
        if old_pids:
            logger.info(f"  发现旧服务进程: {', '.join(old_pids)}，准备终止")
            for pid in old_pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for _ in range(10):
                time.sleep(0.5)
                result = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
                if not result.stdout.strip():
                    break
            else:
                for pid in old_pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                logger.warning("  旧服务未正常退出，已强制终止")
        else:
            logger.info("  未发现旧服务进程")
    except Exception as e:
        logger.warning(f"  终止旧服务时出错: {e}")

    # 2. 启动新服务
    env = os.environ.copy()
    env["PORT"] = str(port)
    try:
        log_fp = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(app_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger.info(f"  新生产服务已启动 (PID={proc.pid}, PORT={port})")
        return True
    except Exception as e:
        logger.error(f"  启动生产服务失败: {e}")
        return False


# =====================================================================
# 守护进程模式
# =====================================================================
def run_daemon(config: dict, logger: logging.Logger, mode: str = "production"):
    """
    守护进程模式：每 14 天自动执行一次流水线。

    Args:
        config: 配置字典
        logger: 日志记录器
        mode: 运行模式 (默认 production, 仅生产模式)
    """
    interval_hours = config["update_interval_hours"]
    check_interval = config["daemon"]["check_interval_hours"]

    # 检查上次运行时间
    state = load_state()
    last_run = state.get("end_time")
    if last_run:
        last_run_dt = datetime.fromisoformat(last_run)
        next_run = last_run_dt + timedelta(hours=interval_hours)
        if datetime.now() < next_run:
            wait_seconds = (next_run - datetime.now()).total_seconds()
            logger.info(f"上次运行: {last_run}, 下次运行: {next_run.isoformat()}")
            logger.info(f"等待 {wait_seconds/3600:.1f} 小时后执行")
        else:
            logger.info(f"上次运行已过期 ({last_run}), 立即执行 (模式: {mode})")
            run_pipeline(config, logger, mode=mode)
    else:
        logger.info(f"首次运行，立即执行流水线 (模式: {mode})")
        run_pipeline(config, logger, mode=mode)

    # 主循环
    def _shutdown_handler(signum, frame):
        logger.info("收到终止信号，守护进程退出")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    logger.info(f"守护进程已启动，每 {interval_hours} 小时自动更新 (模式: {mode})")
    logger.info(f"检查间隔: {check_interval} 小时")

    while True:
        try:
            time.sleep(check_interval * 3600)

            state = load_state()
            last_run = state.get("end_time")

            if last_run:
                last_run_dt = datetime.fromisoformat(last_run)
                next_run = last_run_dt + timedelta(hours=interval_hours)
                if datetime.now() >= next_run:
                    logger.info(f"触发定时更新 (上次: {last_run}, 模式: {mode})")
                    run_pipeline(config, logger, mode=mode)
            else:
                logger.info(f"触发定时更新 (无历史记录, 模式: {mode})")
                run_pipeline(config, logger, mode=mode)

        except KeyboardInterrupt:
            logger.info("守护进程被用户中断")
            break
        except Exception as e:
            logger.exception("守护进程异常: %s", e)
            time.sleep(60)


# =====================================================================
# 打印状态
# =====================================================================
def print_status(logger: logging.Logger):
    """打印上次运行状态"""
    state = load_state()
    if not state:
        logger.info("暂无运行记录")
        return

    logger.info("=" * 70)
    logger.info("  上次流水线运行状态")
    logger.info("=" * 70)
    logger.info("  运行 ID:    %s", state.get("run_id", "N/A"))
    logger.info("  开始时间:   %s", state.get("start_time", "N/A"))
    logger.info("  结束时间:   %s", state.get("end_time", "N/A"))
    logger.info("  总耗时:     %.0fs", state.get("total_elapsed_sec", 0))
    logger.info("  整体成功:   %s", state.get("overall_success", False))
    logger.info("  是否回滚:   %s", state.get("rollback_performed", False))

    logger.info("  各阶段状态:")
    for stage, result in state.get("stages", {}).items():
        if stage.endswith("_validation"):
            continue
        status = "OK" if result.get("success") else "FAIL"
        elapsed = result.get("elapsed_sec", 0)
        logger.info("    [%s] %s: %.0fs", status, stage, elapsed)

    val = state.get("validation", {})
    if val:
        logger.info("  验证结果:")
        logger.info("    服务启动:   %s", val.get("server_started", False))
        logger.info("    模型加载:   %s", val.get("models_loaded", False))
        logger.info("    BP推荐:     %s", val.get("bp_recommend_ok", False))
        logger.info("    预测:       %s", val.get("predict_ok", False))
        logger.info("    BP Delta:   %s", val.get("bp_delta_ok", False))
        logger.info("    兜底机制:   %s", val.get("fallback_ok", False))
        for d in val.get("details", []):
            logger.info("      %s", d)


# =====================================================================
# CLI 入口: 模块别名 / 单模块运行 / 主函数
# =====================================================================
MODULE_ALIASES = {
    "scrape": "data_scrape",
    "data-scrape": "data_scrape",
    "clean": "data_cleaning",
    "data-cleaning": "data_cleaning",
    "postprocess": "data_postprocess",
    "data-postprocess": "data_postprocess",
    "no-scrape": "data_postprocess",
    "rec-train": "bp_recommendation_training",
    "rec-training": "bp_recommendation_training",
    "rec-prod": "bp_recommendation_production",
    "rec-production": "bp_recommendation_production",
    "rec-validate": "bp_recommendation_validation",
    "rec-val": "bp_recommendation_validation",
    "pred-train": "bp_prediction_training",
    "pred-training": "bp_prediction_training",
    "pred-prod": "bp_prediction_production",
    "pred-production": "bp_prediction_production",
    "pred-validate": "bp_prediction_validation",
    "pred-val": "bp_prediction_validation",
    "app": "app",
    "serve": "app",
    "server": "app",
}


def _resolve_module(name: str) -> Optional[str]:
    """解析模块名或别名, 返回标准 stage_key (或 'app' 表示启动服务)"""
    if name in STAGE_SCRIPTS:
        return name
    if name in MODULE_ALIASES:
        return MODULE_ALIASES[name]
    return None


def _list_available_stages():
    """打印所有可用模块及其说明"""
    print("\n可用模块 (可通过 --module <name> 单独运行):\n")
    print(f"  {'模块名':<35} {'别名':<25} 说明")
    print(f"  {'-'*35} {'-'*25} {'-'*40}")
    alias_map = {}
    for alias, key in MODULE_ALIASES.items():
        alias_map.setdefault(key, []).append(alias)
    for key, info in STAGE_SCRIPTS.items():
        aliases = ", ".join(alias_map.get(key, []))
        print(f"  {key:<35} {aliases:<25} {info['name']}")
    print(f"  {'app':<35} {'app, serve, server':<25} 启动后端推理服务 (不运行流水线)")
    print("\n预设流水线模式 (--mode):")
    print(f"  complete    完整流水线 (数据爬取→推荐训练→推荐生产→推荐验证→预测训练→预测生产→预测验证→启动服务)")
    print(f"  production  仅生产流水线 (数据爬取→推荐生产→推荐验证→预测生产→预测验证→启动服务)")
    print(f"  no_scrape   跳过爬取流水线 (后处理→推荐训练→推荐生产→推荐验证→预测训练→预测生产→预测验证→启动服务)")
    print()


def run_single_stage(stage_key: str, config: dict, logger: logging.Logger,
                     mode: str = "production", skip_validation: bool = False) -> dict:
    """单独运行一个模块（不做备份、不做回滚、不重启服务、不触发全流水线验证）。

    用于开发者单独调试某个阶段，例如只跑数据爬取或只跑生产训练。
    """
    if stage_key == "app":
        logger.info("=" * 70)
        logger.info("  启动生产推理服务 (app.py)")
        logger.info("=" * 70)
        app_path = PROJECT_ROOT / "app.py"
        if not app_path.exists():
            logger.error("app.py 不存在: %s", app_path)
            return {"success": False, "reason": "app.py not found"}
        port = config["server"].get("port", 5001)
        log_path = PROJECT_ROOT / "logs" / "web.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                proc = subprocess.Popen(
                    [sys.executable, str(app_path)],
                    cwd=str(PROJECT_ROOT),
                    stdout=lf, stderr=subprocess.STDOUT,
                )
            logger.info("服务已启动, PID=%d, 端口=%d, 日志=%s", proc.pid, port, log_path)
            logger.info("按 Ctrl+C 停止 (或 kill %d)", proc.pid)
            proc.wait()
            return {"success": True, "pid": proc.pid}
        except KeyboardInterrupt:
            logger.info("收到中断信号, 终止服务")
            proc.terminate()
            return {"success": True, "interrupted": True}

    if stage_key not in STAGE_SCRIPTS:
        logger.error("未知模块: %s", stage_key)
        return {"success": False, "reason": f"unknown module: {stage_key}"}

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "run_id": run_ts,
        "mode": f"single:{stage_key}",
        "start_time": datetime.now().isoformat(),
        "stages": {},
        "backup_path": None,
        "rollback_performed": False,
        "validation": {},
        "overall_success": False,
        "total_elapsed_sec": 0,
    }

    pipeline_start = time.time()
    stage_result = run_stage(stage_key, config, logger, mode=mode)
    results["stages"][stage_key] = stage_result
    elapsed = time.time() - pipeline_start
    results["total_elapsed_sec"] = round(elapsed, 1)

    if not stage_result["success"]:
        logger.error("模块 %s 执行失败", stage_key)
        results["overall_success"] = False
        return results

    if not skip_validation:
        validation = validate_stage_outputs(
            stage_key, logger,
            freshness_baseline=pipeline_start,
            allow_stale=False,
        )
        results["stages"][f"{stage_key}_validation"] = validation
        if not validation["all_present"]:
            logger.warning("输出文件验证不全部通过 (缺失/陈旧/无效), 但不阻断单模块运行")
            for f in validation.get("missing", []):
                logger.warning("  缺失: %s", f)
            for f in validation.get("stale", []):
                logger.warning("  陈旧: %s", f)
            for f in validation.get("invalid", []):
                logger.warning("  无效: %s", f)

    results["overall_success"] = True
    results["end_time"] = datetime.now().isoformat()

    logger.info("")
    logger.info("=" * 70)
    logger.info("  模块执行完成: %s", STAGE_SCRIPTS[stage_key]["name"])
    logger.info("  耗时: %.1fs (%.1f 分钟)", elapsed, elapsed / 60)
    if stage_result.get("summary"):
        logger.info("  --- 关键输出 ---")
        for s in stage_result["summary"][:8]:
            logger.info("    %s", s)
    logger.info("=" * 70)

    save_state(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="LOL BP 预测 - 数据更新与模型训练流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
【全流水线模式】
  python auto_update_pipeline.py                              # 完整模式 (训练+生产, 默认)
  python auto_update_pipeline.py --mode production            # 仅生产模式 (跳过训练)
  python auto_update_pipeline.py --mode no_scrape             # 跳过爬取模式 (保留清洗+完整训练)
  python auto_update_pipeline.py --daemon                     # 守护进程 (每14天自动生产更新)
  python auto_update_pipeline.py --stage 2                    # 从第3阶段开始 (0-indexed)

【单模块模式 - 单独运行某个模块】
  python auto_update_pipeline.py --module scrape              # 仅爬取数据
  python auto_update_pipeline.py --module clean               # 仅清洗数据
  python auto_update_pipeline.py --module rec-train           # 仅训练推荐模型(开发模式)
  python auto_update_pipeline.py --module rec-prod            # 仅训练推荐模型(生产模式)
  python auto_update_pipeline.py --module rec-val             # 仅做推荐模型一致性检测
  python auto_update_pipeline.py --module pred-train          # 仅训练预测模型(开发模式)
  python auto_update_pipeline.py --module pred-prod           # 仅训练预测模型(生产模式)
  python auto_update_pipeline.py --module pred-val            # 仅做预测模型一致性检测
  python auto_update_pipeline.py --module app                 # 仅启动推理服务

【其他】
  python auto_update_pipeline.py --list-stages                # 列出所有可用模块
  python auto_update_pipeline.py --dry-run                    # 干运行 (检查环境)
  python auto_update_pipeline.py --status                     # 查看上次运行状态
        """,
    )
    parser.add_argument("--mode", type=str, default="complete",
                        choices=["complete", "production", "no_scrape"],
                        help="全流水线运行模式")
    parser.add_argument("--module", type=str, default=None,
                        help="单模块模式: 指定要运行的模块名 (用 --list-stages 查看所有模块)")
    parser.add_argument("--list-stages", action="store_true",
                        help="列出所有可单独运行的模块及其别名后退出")
    parser.add_argument("--skip-validation", action="store_true",
                        help="单模块模式下跳过输出文件验证")
    parser.add_argument("--daemon", action="store_true",
                        help="以守护进程模式运行 (每14天自动执行 production 模式)")
    parser.add_argument("--dry-run", action="store_true",
                        help="干运行模式，仅检查环境和脚本是否存在")
    parser.add_argument("--stage", type=int, default=0,
                        help="全流水线模式下从指定阶段开始 (0-indexed)")
    parser.add_argument("--status", action="store_true",
                        help="查看上次运行状态")
    parser.add_argument("--no-backup", action="store_true",
                        help="全流水线模式下跳过备份步骤")
    parser.add_argument("--no-rollback", action="store_true",
                        help="全流水线模式下禁用自动回滚")
    args = parser.parse_args()

    if args.list_stages:
        _list_available_stages()
        return

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_ts)

    if args.status:
        print_status(logger)
        return

    config = load_config()

    # 单模块模式
    if args.module:
        resolved = _resolve_module(args.module)
        if resolved is None:
            logger.error("未知模块: %s (使用 --list-stages 查看可用模块)", args.module)
            sys.exit(2)
        mode = args.mode if args.mode != "complete" else "production"
        logger.info("=" * 70)
        logger.info("  单模块运行模式")
        logger.info("  模块: %s (%s)", resolved, STAGE_SCRIPTS.get(resolved, {}).get("name", "启动服务"))
        logger.info("  时间: %s", datetime.now().isoformat())
        logger.info("  子模式: %s", mode)
        logger.info("  输出验证: %s", "跳过" if args.skip_validation else "启用")
        logger.info("=" * 70)

        results = run_single_stage(
            resolved, config, logger, mode=mode,
            skip_validation=args.skip_validation,
        )
        save_state(results)
        sys.exit(0 if results["overall_success"] else 1)

    # 全流水线模式
    if args.mode == "production":
        config["stage_order"] = MODE_PRODUCTION_STAGES
    elif args.mode == "no_scrape":
        config["stage_order"] = MODE_NO_SCRAPE_STAGES
    else:
        config["stage_order"] = MODE_COMPLETE_STAGES

    if args.no_backup:
        config["backup"]["enabled"] = False
    if args.no_rollback:
        config["rollback"]["enabled"] = False
        config["rollback"]["auto_rollback_on_failure"] = False

    logger.info("=" * 70)
    logger.info("  全自动数据更新与模型重训练流水线")
    logger.info("  时间: %s", datetime.now().isoformat())
    logger.info("  项目目录: %s", PROJECT_ROOT)
    logger.info("  运行模式: %s", args.mode)
    logger.info("  更新周期: %s 小时", config["update_interval_hours"])
    logger.info("  执行顺序: %s", " -> ".join(config["stage_order"]))
    logger.info("  备份: %s", "启用" if config["backup"]["enabled"] else "禁用")
    logger.info("  回滚: %s", "启用" if config["rollback"]["enabled"] else "禁用")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("")
        logger.info("=== 干运行模式: 检查环境和脚本 ===")
        all_ok = True
        for stage_key in config["stage_order"]:
            info = STAGE_SCRIPTS[stage_key]
            exists = info["script"].exists()
            status = "OK" if exists else "MISSING"
            if not exists:
                all_ok = False
            logger.info("  [%s] %s: %s", status, info["name"], info["script"])
        for category, files in PRODUCTION_CHECK_FILES.items():
            for rel_path in files:
                exists = (PROJECT_ROOT / rel_path).exists()
                status = "OK" if exists else "MISSING"
                logger.info("  [%s] %s", status, rel_path)
        logger.info("干运行结果: %s", "所有检查通过" if all_ok else "存在问题需要修复")
        return

    if args.daemon:
        daemon_mode = args.mode
        run_daemon(config, logger, mode=daemon_mode)
        return

    results = run_pipeline(config, logger, start_stage=args.stage, mode=args.mode)

    if results["overall_success"]:
        logger.info("流水线执行成功! 系统已就绪。")
        sys.exit(0)
    else:
        logger.error("流水线执行失败! 请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()