"""
BP 胜负预测模型 - 统一配置管理
================================
提供训练模式 (training) 和生产模式 (production) 的配置隔离。

模式说明:
  - training  : 5-Fold OOT 滚动窗口验证，用于评估模型泛化能力
  - production: 全量数据 + 时间衰减权重训练，用于线上推理部署

线上部署动态日期说明:
  - cutoff_date = None 时自动从 cleaned_data/matches_cleaned.csv 检测最新日期
  - oot_test_starts = None 时根据最新日期动态反向生成 OOT 折窗口
  - 新增 resolve_cutoff_date() / resolve_oot_folds() 函数获取实际值

使用方式:
  # 获取当前模式配置
  from config import get_config, set_mode, Mode
  set_mode(Mode.PRODUCTION)
  cfg = get_config()

  # 环境变量切换模式
  # export BP_PRED_MODE=production  (或 training)
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from logger_config import get_logger

log = get_logger(__name__)


# =====================================================================
# 路径常量
# =====================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

FEATURES_DIR = os.path.join(MODEL_DIR, "features")
WIDE_FEATURES_PATH = os.path.join(FEATURES_DIR, "ALL_prediction_wide_features.parquet")
MODELS_DIR = os.path.join(MODEL_DIR, "models")
PRODUCTION_DIR = os.path.join(MODELS_DIR, "production")
TF_FEATURES_DIR = os.path.join(MODEL_DIR, "tf_features")
TF_SNAPSHOTS_DIR = os.path.join(MODEL_DIR, "tf_snapshots")
LOGS_DIR = os.path.join(MODEL_DIR, "logs")
REPORTS_DIR = os.path.join(MODEL_DIR, "reports")
TRAINING_DIR = os.path.join(MODEL_DIR, "training")

CLEANED_DIR = os.path.join(PROJECT_ROOT, "cleaned_data")
VOCAB_PATH = os.path.join(CLEANED_DIR, "champion_vocabulary.json")


# =====================================================================
# 模式枚举
# =====================================================================
class Mode(str, Enum):
    TRAINING = "training"
    PRODUCTION = "production"


# =====================================================================
# 共享配置 (训练和生产共用)
# =====================================================================
@dataclass
class SharedConfig:
    """训练和生产模式共享的配置"""
    # CatBoost 超参数 (来自 train_walk_forward.py 的最优配置)
    catboost_params: Dict = field(default_factory=lambda: {
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.035,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.5,
        "bagging_temperature": 0.1,
        "border_count": 224,
        "verbose": 0,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
    })

    # 7-Seed Bagging
    n_seeds: int = 7
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024, 2026, 3141])

    # Label Smoothing
    label_smoothing: float = 0.05

    # 联赛自适应权重 (Round 2 OOT 实验最优: B_lck12_lec08)
    league_weights: Dict = field(default_factory=lambda: {"LCK": 1.2, "LPL": 1.0, "LEC": 0.8})

    # 镜像增强
    mirror_augmentation: bool = True

    # Early Stopping
    early_stopping_rounds: int = 50
    val_split_ratio: float = 0.2

    # CS 特征前缀 (排除项)
    cs_feature_prefixes: List[str] = field(default_factory=lambda: [
        "blue_counter_", "red_counter_",
        "blue_synergy_", "red_synergy_",
    ])

    # TF 特征列
    tf_cols: List[str] = field(default_factory=lambda: [
        "tf_win_logits", "tf_cosine_sim", "tf_blue_l2norm", "tf_red_l2norm"
    ])

    # 推理并发控制
    # max_concurrent_inferences: 根据压测结果调整，CatBoost 单次推理约 200-500ms，
    #   6 并发可支撑峰值 ~12 QPS (每请求 7-seed 集成)，兼顾 CPU 占用与响应延迟
    max_concurrent_inferences: int = 6
    # inference_timeout_seconds: 含 TF 特征实时推理 (Transformer 前向) + CatBoost 7-seed 集成，
    #   20s 覆盖 P99 延迟，避免误杀正常请求
    inference_timeout_seconds: float = 20.0
    rate_limit_window_seconds: float = 60.0
    # rate_limit_max_requests: 6 并发 × 单次 ~500ms = 理论 12 QPS，60s 内 120 次留 20% 余量
    rate_limit_max_requests: int = 120


# =====================================================================
# 训练模式配置 (5-Fold OOT 验证)
# =====================================================================
@dataclass
class TrainingConfig:
    """训练模式配置 - 5-Fold OOT 滚动窗口验证

    线上部署模式 (oot_test_starts=None):
      - 根据最新数据日期动态反向生成 5 折 OOT 窗口
      - window_months=12 训练窗口, test_duration_months=2 测试窗口

    本地开发模式 (oot_test_starts 显式指定):
      - 使用固定的测试起点列表，便于复现历史实验结果
    """
    # OOT 折数
    n_folds: int = 5

    # 训练窗口 (月)
    window_months: int = 12

    # 测试窗口 (月)
    test_duration_months: int = 2

    # OOT 测试起点: None = 自动根据最新日期动态生成 (线上部署模式)
    # 显式列表 = 固定窗口 (本地开发复现模式), 如 ["2025-06-01", "2025-08-01", ...]
    oot_test_starts: Optional[List[str]] = None

    # 相邻折测试窗口之间的间隔月数 (0 = 紧接无间隔)
    fold_gap_months: int = 0

    # Bootstrap 置信区间
    n_bootstrap: int = 1000
    bootstrap_ci: float = 0.95

    # 模型保存目录 (每折独立)
    fold_model_dir_pattern: str = "fold_{fold_idx}"

    # 是否使用 TF 特征
    use_tf_features: bool = True

    # LPL 轻量噪声增强 (已禁用，与 train_walk_forward.py 一致)
    lpl_noise_augmentation: bool = False


# =====================================================================
# 生产模式配置 (推理优化)
# =====================================================================
@dataclass
class ProductionConfig:
    """生产模式配置 - 全量数据训练 + 推理优化

    线上部署模式 (cutoff_date=None):
      - 自动从 cleaned_data/matches_cleaned.csv 检测最新比赛日期作为 cutoff
      - 使用全量历史数据 + 指数时间衰减权重训练

    本地开发模式 (cutoff_date 显式指定):
      - 使用固定截止日期，便于复现历史模型
    """
    # 数据截止日期 (None = 自动检测最新数据日期, 线上部署推荐)
    cutoff_date: Optional[str] = None

    # 时间衰减半衰期 (天)
    time_decay_half_life_days: int = 180

    # 是否使用全量数据 (False = 仅 1 年窗口)
    use_full_data: bool = True

    # 数据最早日期 (None = 无限制)
    min_date: Optional[str] = None

    # 是否使用 TF 特征
    use_tf_features: bool = True

    # 推理优化
    enable_feature_monitor: bool = True
    enable_fallback: bool = True
    enable_concurrency_control: bool = True

    # 模型加载优先级: production > fold_0~4
    prefer_production_models: bool = True

    # Temperature Scaling 后校准
    enable_temperature_scaling: bool = False
    temperature: float = 1.0

    # =====================================================================
    # 生产模式轮数计算 (方案 B: OOT best_iteration + √n 补偿)
    # =====================================================================
    # 生产模式从开发模式 (OOT) 读取 best_iteration, 按 √n 补偿计算固定轮数,
    # 关闭 early stopping, 用 100% 数据训练.
    # 与推荐模型 "开发模式定停止点 → 生产模式全量训练" 逻辑一致.

    # 是否启用 OOT 驱动的固定轮数 (False = 回退到旧的 80/20 early stopping)
    use_oot_driven_iterations: bool = True

    # OOT 参数文件路径 (train_walk_forward.py 产出)
    oot_iterations_source_path: str = os.path.join(REPORTS_DIR, "production_iterations_source.json")

    # 数据膨胀补偿系数: production_iterations = base_iterations × (n_prod / n_fold) ^ expansion_exponent
    # 理论值 0.5 (√n), 但时间衰减会让等效数据量减少, 因此引入 effective_data_ratio
    expansion_exponent: float = 0.5

    # 时间衰减等效数据比例: 全量数据 + 时间衰减后, 等效均匀数据量约为全量的 65%
    # (半衰期 180 天, 数据跨度约 2 年, 近期数据权重更高)
    effective_data_ratio: float = 0.65

    # LR 衰减系数: 关闭 early stopping 后, 降低 LR 作为正则化补偿
    lr_decay_factor: float = 0.85

    # 最小/最大生产轮数 (安全边界, 防止异常值)
    min_production_iterations: int = 100
    max_production_iterations: int = 800


# =====================================================================
# 全局配置管理
# =====================================================================
_shared = SharedConfig()
_training = TrainingConfig()
_production = ProductionConfig()

_current_mode: Mode = Mode.PRODUCTION  # 默认生产模式


def set_mode(mode: Mode):
    """切换全局模式"""
    global _current_mode
    _current_mode = mode


def get_mode() -> Mode:
    """获取当前模式"""
    return _current_mode


def get_config(mode: Optional[Mode] = None):
    """获取指定模式的配置 (默认返回当前模式)

    Returns:
        tuple: (shared_config, mode_config)
            - shared_config: SharedConfig
            - mode_config: TrainingConfig 或 ProductionConfig
    """
    target_mode = mode or _current_mode
    if target_mode == Mode.TRAINING:
        return _shared, _training
    else:
        return _shared, _production


def is_training_mode() -> bool:
    return _current_mode == Mode.TRAINING


def is_production_mode() -> bool:
    return _current_mode == Mode.PRODUCTION


# =====================================================================
# 环境变量初始化
# =====================================================================
_env_mode = os.environ.get("BP_PRED_MODE", "").lower()
if _env_mode == "training":
    _current_mode = Mode.TRAINING
elif _env_mode == "production":
    _current_mode = Mode.PRODUCTION


# =====================================================================
# 辅助函数
# =====================================================================
def resolve_cutoff_date(cfg_cutoff: Optional[str] = None) -> str:
    """解析数据截止日期：None 时自动检测最新数据日期。

    Args:
        cfg_cutoff: 配置中的 cutoff_date 值，None 表示自动检测。

    Returns:
        "YYYY-MM-DD" 格式的截止日期字符串。
    """
    if cfg_cutoff is not None:
        return cfg_cutoff
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from common.paths import get_latest_data_date
    latest = get_latest_data_date()
    resolved = latest.strftime("%Y-%m-%d")
    log.info("  [AutoDate] cutoff_date 未指定，自动检测到最新数据日期: %s", resolved)
    return resolved


def resolve_oot_folds(training_cfg: "TrainingConfig", cutoff: Optional[str] = None):
    """解析 OOT 折窗口：oot_test_starts=None 时动态生成。

    Args:
        training_cfg: TrainingConfig 实例。
        cutoff: 数据截止日期 (YYYY-MM-DD)，None 时自动检测。

    Returns:
        (folds, resolved_cutoff):
            folds: List of (train_start, train_end, test_start, test_end) 字符串元组。
            resolved_cutoff: 实际使用的截止日期字符串。
    """
    import pandas as pd
    resolved_cutoff = resolve_cutoff_date(cutoff)
    if training_cfg.oot_test_starts is not None:
        from dateutil.relativedelta import relativedelta
        folds = []
        for ts_str in training_cfg.oot_test_starts:
            ts = pd.Timestamp(ts_str)
            te = ts + relativedelta(months=training_cfg.test_duration_months) - pd.Timedelta(days=1)
            train_end = ts - pd.Timedelta(days=1)
            train_start = train_end - relativedelta(months=training_cfg.window_months) + pd.Timedelta(days=1)
            folds.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                ts.strftime("%Y-%m-%d"),
                te.strftime("%Y-%m-%d"),
            ))
        log.info("  [AutoDate] 使用固定 OOT 测试起点列表 (%d 折)", len(folds))
        return folds, resolved_cutoff
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from common.paths import generate_dynamic_oot_folds
    latest_dt = pd.Timestamp(resolved_cutoff)
    folds = generate_dynamic_oot_folds(
        latest_date=latest_dt,
        n_folds=training_cfg.n_folds,
        window_months=training_cfg.window_months,
        test_duration_months=training_cfg.test_duration_months,
        gap_between_folds_months=training_cfg.fold_gap_months,
    )
    log.info("  [AutoDate] 动态生成 OOT 折窗口 (%d 折, cutoff=%s)", len(folds), resolved_cutoff)
    return folds, resolved_cutoff


def get_model_dir(mode: Optional[Mode] = None) -> str:
    """获取模型目录"""
    target_mode = mode or _current_mode
    if target_mode == Mode.PRODUCTION:
        return PRODUCTION_DIR
    else:
        return MODELS_DIR  # OOT 折模型在 fold_0~4 子目录


def get_feature_cols_path(mode: Optional[Mode] = None) -> str:
    """获取特征列名文件路径"""
    target_mode = mode or _current_mode
    if target_mode == Mode.PRODUCTION:
        return os.path.join(PRODUCTION_DIR, "feature_columns.json")
    else:
        # 训练模式: 使用第一折的特征列
        return os.path.join(MODELS_DIR, "fold_0", "feature_columns.json")


def print_config_summary(mode: Optional[Mode] = None):
    """打印配置摘要"""
    target_mode = mode or _current_mode
    shared, cfg = get_config(target_mode)

    log.info("\n%s", "="*70)
    log.info("  BP Prediction Model - %s Mode", target_mode.value.upper())
    log.info("%s", "="*70)
    log.info("  CatBoost params: iterations=%d, depth=%d, lr=%s",
             shared.catboost_params['iterations'], shared.catboost_params['depth'], shared.catboost_params['learning_rate'])
    log.info("  Seeds: %d seeds, Label Smoothing: %s", shared.n_seeds, shared.label_smoothing)
    log.info("  Mirror Augmentation: %s", shared.mirror_augmentation)
    log.info("  League Weights: %s", shared.league_weights)

    if target_mode == Mode.TRAINING:
        folds, resolved_cutoff = resolve_oot_folds(cfg)
        log.info("  Cutoff Date (auto): %s", resolved_cutoff)
        log.info("  OOT Folds: %d (%s)", cfg.n_folds,
                 "dynamic" if cfg.oot_test_starts is None else "fixed list")
        log.info("  Training Window: %d months", cfg.window_months)
        log.info("  Test Window: %d months", cfg.test_duration_months)
        for i, (ts, te, tst, tse) in enumerate(folds, 1):
            log.info("    Fold %d: Train [%s~%s] | Test [%s~%s]", i, ts, te, tst, tse)
        log.info("  Bootstrap: %d resamples, %.0f%% CI", cfg.n_bootstrap, cfg.bootstrap_ci*100)
        log.info("  TF Features: %s", cfg.use_tf_features)
    else:
        resolved_cutoff = resolve_cutoff_date(cfg.cutoff_date)
        log.info("  Cutoff Date: %s%s", resolved_cutoff,
                 " (auto-detected)" if cfg.cutoff_date is None else "")
        log.info("  Data Strategy: %s", 'Full + Time Decay' if cfg.use_full_data else '1-Year Window')
        if cfg.use_full_data:
            log.info("  Time Decay: half_life=%dd", cfg.time_decay_half_life_days)
        log.info("  TF Features: %s", cfg.use_tf_features)
        log.info("  Feature Monitor: %s", cfg.enable_feature_monitor)
        log.info("  Fallback: %s", cfg.enable_fallback)
        log.info("  Concurrency Control: %s", cfg.enable_concurrency_control)

    log.info("%s\n", "="*70)
