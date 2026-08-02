"""
统一路径管理模块 (Single Source of Truth for all project paths)

功能描述:
    集中管理项目中所有目录和文件路径，作为路径配置的唯一真实来源，避免硬编码路径。
    所有路径均为 pathlib.Path 对象，可直接用于 open()/pandas.read_csv() 等操作。

主要函数:
    - find_latest_match_data(): 自动发现最新年份的比赛数据文件
    - get_match_data_path(year): 获取指定年份的比赛数据路径
    - ensure_dirs(): 确保所有必要目录存在
    - get_path(key): 通过名称动态获取路径常量

主要常量:
    - PROJECT_ROOT: 项目根目录
    - RAW_DATA_DIR/CLEANED_DATA_DIR: 原始数据/清洗后数据目录
    - BP_PREDICTION_DIR/BP_RECOMMENDATION_DIR: BP预测/推荐模块目录
    - 各类数据文件路径常量（CHAMPION_VOCABULARY_JSON等）

使用方式:
    from common.paths import PROJECT_ROOT, MATCHES_CSV, get_path, ensure_dirs

    # 获取项目根目录
    print(PROJECT_ROOT)

    # 获取最新比赛数据
    match_path = get_path("match_data")

    # 确保所有目录存在
    ensure_dirs()
"""

from __future__ import annotations

import glob as _glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

LOG_DIR: Path = PROJECT_ROOT / "logs"
BACKUP_DIR: Path = PROJECT_ROOT / "backups"

RAW_DATA_DIR: Path = PROJECT_ROOT / "raw_data"
RAW_MATCHES_DIR: Path = RAW_DATA_DIR / "matches"

CLEANED_DATA_DIR: Path = PROJECT_ROOT / "cleaned_data"
FALLBACK_DATA_DIR: Path = PROJECT_ROOT / "fallback" / "data"

BP_PREDICTION_DIR: Path = PROJECT_ROOT / "bp_prediction"
BP_RECOMMENDATION_DIR: Path = PROJECT_ROOT / "bp_recommendation"
DATA_SCRAPER_DIR: Path = PROJECT_ROOT / "data_scraper"
TESTDATA_DIR: Path = PROJECT_ROOT / "testdata"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

MODEL_METRICS_DIR: Path = PROJECT_ROOT / "model_metrics"
PREDICTION_METRICS_DIR: Path = MODEL_METRICS_DIR / "prediction"
RECOMMENDATION_METRICS_DIR: Path = MODEL_METRICS_DIR / "recommendation"
PRODUCTION_METRICS_DIR: Path = MODEL_METRICS_DIR / "production"

PREDICTION_FEATURES_DIR: Path = BP_PREDICTION_DIR / "features"
PREDICTION_MODELS_DIR: Path = BP_PREDICTION_DIR / "models"
PREDICTION_PRODUCTION_MODELS_DIR: Path = PREDICTION_MODELS_DIR / "production"
PREDICTION_TF_SNAPSHOTS_DIR: Path = BP_PREDICTION_DIR / "tf_snapshots"
PREDICTION_REPORTS_DIR: Path = BP_PREDICTION_DIR / "reports"
PREDICTION_TRAINING_DIR: Path = BP_PREDICTION_DIR / "training"

RECOMMENDATION_FEATURES_DIR: Path = BP_RECOMMENDATION_DIR / "features"
RECOMMENDATION_CONFIG_DIR: Path = BP_RECOMMENDATION_DIR / "training_configs"
PICK_MODEL_DIR: Path = BP_RECOMMENDATION_DIR / "model_pick"
BAN_MODEL_DIR: Path = BP_RECOMMENDATION_DIR / "model_ban"
PICK_CKPT_DIR: Path = PICK_MODEL_DIR / "checkpoints"
BAN_CKPT_DIR: Path = BAN_MODEL_DIR / "checkpoints"
PICK_CASCADE_DIR: Path = PICK_CKPT_DIR / "cascade_pick"
BAN_CASCADE_DIR: Path = BAN_CKPT_DIR / "cascade_ban"
PICK_SAVED_MODELS_DIR: Path = PICK_MODEL_DIR / "saved_models"
BAN_SAVED_MODELS_DIR: Path = BAN_MODEL_DIR / "saved_models"

MESSAGES_DIR: Path = CLEANED_DATA_DIR / "messages"

_champion_vocab = CLEANED_DATA_DIR / "champion_vocabulary.json"
_champion_position = CLEANED_DATA_DIR / "champion_position_mapping.json"
_player_career = CLEANED_DATA_DIR / "player_career_hero_stats_cleaned.csv"
_champion_counters = CLEANED_DATA_DIR / "champion_counters_cleaned.csv"
_champion_synergy = CLEANED_DATA_DIR / "champion_synergy_cleaned.csv"
_champion_ranks = CLEANED_DATA_DIR / "champion_ranks_cleaned.csv"
_merged_champion_stats = CLEANED_DATA_DIR / "merged_champion_stats.csv"
_active_rosters = CLEANED_DATA_DIR / "active_rosters.csv"
_team_player_mapping = CLEANED_DATA_DIR / "team_player_mapping.json"
_removed_rows = CLEANED_DATA_DIR / "removed_rows.csv"

CLEANED_KEY_FILES: dict[str, Path] = {
    "champion_vocabulary": _champion_vocab,
    "champion_position_mapping": _champion_position,
    "player_career_stats": _player_career,
    "champion_counters": _champion_counters,
    "champion_synergy": _champion_synergy,
    "champion_ranks": _champion_ranks,
    "merged_champion_stats": _merged_champion_stats,
    "active_rosters": _active_rosters,
    "team_player_mapping": _team_player_mapping,
}

CHAMPION_VOCABULARY_JSON: Path = _champion_vocab
CHAMPION_POSITION_MAPPING_JSON: Path = _champion_position
PLAYER_CAREER_STATS_CSV: Path = _player_career
CHAMPION_COUNTERS_CSV: Path = _champion_counters
CHAMPION_SYNERGY_CSV: Path = _champion_synergy
CHAMPION_RANKS_CSV: Path = _champion_ranks
MERGED_CHAMPION_STATS_CSV: Path = _merged_champion_stats
ACTIVE_ROSTERS_CSV: Path = _active_rosters
TEAM_PLAYER_MAPPING_JSON: Path = _team_player_mapping

MATCHES_CLEANED_CSV: Path = CLEANED_DATA_DIR / "matches_cleaned.csv"

PREDICTION_FEATURES_PARQUET: Path = PREDICTION_FEATURES_DIR / "ALL_prediction_wide_features.parquet"
PREDICTION_FEATURE_COLUMNS_JSON: Path = PREDICTION_PRODUCTION_MODELS_DIR / "feature_columns.json"
PREDICTION_PRODUCTION_ITERATIONS_JSON: Path = PREDICTION_REPORTS_DIR / "production_iterations_source.json"
PREDICTION_PRODUCTION_TF_SNAPSHOT: Path = PREDICTION_TF_SNAPSHOTS_DIR / "production_nocs.pt"

RECOMMENDATION_CONTEXT_PARQUET: Path = RECOMMENDATION_FEATURES_DIR / "ALL_context.parquet"
RECOMMENDATION_META_STORE_PARQUET: Path = RECOMMENDATION_FEATURES_DIR / "ALL_meta_store.parquet"
RECOMMENDATION_PLAYER_STORE_PARQUET: Path = RECOMMENDATION_FEATURES_DIR / "ALL_player_store.parquet"

PICK_BEST_CS: Path = PICK_CKPT_DIR / "best_model_cs.pt"
PICK_BEST_NOCS: Path = PICK_CKPT_DIR / "best_model_nocs.pt"
BAN_BEST_CS: Path = BAN_CKPT_DIR / "best_model_cs.pt"

FALLBACK_META_STATS_JSON: Path = FALLBACK_DATA_DIR / "meta_stats.json"
FALLBACK_PLAYER_STATS_JSON: Path = FALLBACK_DATA_DIR / "player_stats.json"
FALLBACK_MATCHES_PARQUET: Path = FALLBACK_DATA_DIR / "cleaned_matches.parquet"


def find_latest_match_data() -> Path:
    """自动发现 raw_data/matches/ 目录下最新年份的 OraclesElixir 比赛数据文件。

    文件名格式: YYYY_LoL_esports_match_data_from_OraclesElixir.csv
    按年份降序排序，返回第一个存在的文件。
    """
    pattern = str(RAW_MATCHES_DIR / "*_LoL_esports_match_data_from_OraclesElixir.csv")
    candidates = sorted(_glob.glob(pattern), reverse=True)
    if not candidates:
        alt_pattern = str(RAW_MATCHES_DIR / "*.csv")
        candidates = sorted(_glob.glob(alt_pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No match data CSV found in {RAW_MATCHES_DIR}. "
            "Expected pattern: YYYY_LoL_esports_match_data_from_OraclesElixir.csv"
        )
    return Path(candidates[0])


def get_match_data_path(year: Optional[int] = None) -> Path:
    """获取比赛数据文件路径。

    Args:
        year: 指定年份，如 2026。为 None 时自动返回最新文件。
    """
    if year is not None:
        p = RAW_MATCHES_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        if not p.exists():
            raise FileNotFoundError(f"Match data for year {year} not found: {p}")
        return p
    return find_latest_match_data()


def find_all_match_data_files() -> List[Path]:
    """自动发现 raw_data/matches/ 目录下所有年份的 OraclesElixir 比赛数据文件。

    文件名格式: YYYY_LoL_esports_match_data_from_OraclesElixir.csv
    按年份升序排序返回（从旧到新），便于按时间顺序合并。
    """
    import re
    pattern = str(RAW_MATCHES_DIR / "*_LoL_esports_match_data_from_OraclesElixir.csv")
    candidates = list(_glob.glob(pattern))
    if not candidates:
        alt_pattern = str(RAW_MATCHES_DIR / "*.csv")
        candidates = list(_glob.glob(alt_pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No match data CSV found in {RAW_MATCHES_DIR}. "
            "Expected pattern: YYYY_LoL_esports_match_data_from_OraclesElixir.csv"
        )
    def _year_key(p: str) -> int:
        m = re.search(r'(\d{4})_', os.path.basename(p))
        return int(m.group(1)) if m else 0
    candidates.sort(key=_year_key)
    return [Path(p) for p in candidates]


def get_data_date_range(matches_csv: Optional[Path] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """从清洗后的 matches CSV 文件中自动检测数据的最早和最晚日期。

    Args:
        matches_csv: matches_cleaned.csv 路径，为 None 时使用默认路径 MATCHES_CLEANED_CSV。

    Returns:
        (min_date, max_date): 数据集中日期范围的起止（Timestamp 对象）。
    """
    csv_path = matches_csv or MATCHES_CLEANED_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Matches cleaned CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["date"], low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        raise ValueError(f"No valid date column found in {csv_path}")
    return df["date"].min(), df["date"].max()


def get_latest_data_date(matches_csv: Optional[Path] = None) -> pd.Timestamp:
    """获取数据集中最新日期（用于自动设置 cutoff_date）。

    Args:
        matches_csv: matches_cleaned.csv 路径，为 None 时使用默认路径。

    Returns:
        最新比赛的日期 (Timestamp)。
    """
    _, max_date = get_data_date_range(matches_csv)
    return max_date


def generate_dynamic_oot_folds(
    latest_date: pd.Timestamp,
    n_folds: int = 5,
    window_months: int = 12,
    test_duration_months: int = 2,
    gap_between_folds_months: int = 0,
) -> List[Tuple[str, str, str, str]]:
    """根据数据最新日期动态生成 5-Fold Rolling OOT 的时间窗口划分。

    从最新日期往回倒推，自动生成 n_folds 个测试窗口，每个测试窗口前的 window_months 个月为训练窗口。
    折号从旧到新排列 (Fold 1 = 最早的测试窗口, Fold n_folds = 最近的测试窗口)。

    Args:
        latest_date: 数据集中最晚日期 (截止日期)。
        n_folds: OOT 折数，默认 5。
        window_months: 训练窗口长度（月），默认 12。
        test_duration_months: 测试窗口长度（月），默认 2。
        gap_between_folds_months: 相邻折测试窗口起点的间隔月数，0 表示紧接前一折。

    Returns:
        List of (train_start, train_end, test_start, test_end)，均为 "YYYY-MM-DD" 格式字符串。
    """
    from dateutil.relativedelta import relativedelta
    folds = []
    for i in range(n_folds - 1, -1, -1):
        months_back = i * (test_duration_months + gap_between_folds_months)
        test_end = latest_date - relativedelta(months=months_back)
        test_start = test_end - relativedelta(months=test_duration_months) + pd.Timedelta(days=1)
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - relativedelta(months=window_months) + pd.Timedelta(days=1)
        folds.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
    return folds


def ensure_dirs() -> None:
    """确保所有必要的目录存在。"""
    for d in [
        LOG_DIR, BACKUP_DIR, RAW_DATA_DIR, RAW_MATCHES_DIR,
        CLEANED_DATA_DIR, FALLBACK_DATA_DIR, MESSAGES_DIR,
        MODEL_METRICS_DIR, PREDICTION_METRICS_DIR, RECOMMENDATION_METRICS_DIR, PRODUCTION_METRICS_DIR,
        PREDICTION_FEATURES_DIR, PREDICTION_MODELS_DIR,
        PREDICTION_PRODUCTION_MODELS_DIR, PREDICTION_TF_SNAPSHOTS_DIR,
        PREDICTION_REPORTS_DIR, PREDICTION_TRAINING_DIR,
        RECOMMENDATION_FEATURES_DIR, RECOMMENDATION_CONFIG_DIR,
        PICK_MODEL_DIR, BAN_MODEL_DIR,
        PICK_CKPT_DIR, BAN_CKPT_DIR, PICK_CASCADE_DIR, BAN_CASCADE_DIR,
        PICK_SAVED_MODELS_DIR, BAN_SAVED_MODELS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_path(key: str) -> Path:
    """通过名称获取路径常量，便于动态引用。"""
    if key in CLEANED_KEY_FILES:
        return CLEANED_KEY_FILES[key]
    mapping = {
        "project_root": PROJECT_ROOT,
        "raw_data_dir": RAW_DATA_DIR,
        "raw_matches_dir": RAW_MATCHES_DIR,
        "cleaned_data_dir": CLEANED_DATA_DIR,
        "match_data": None,
        "matches_cleaned": MATCHES_CLEANED_CSV,
        "prediction_features": PREDICTION_FEATURES_PARQUET,
        "pick_best_cs": PICK_BEST_CS,
        "pick_best_nocs": PICK_BEST_NOCS,
        "ban_best_cs": BAN_BEST_CS,
        "log_dir": LOG_DIR,
    }
    if key == "match_data":
        return get_match_data_path()
    val = mapping.get(key)
    if val is None:
        raise KeyError(f"Unknown path key: {key}")
    return val


def __getattr__(name: str):
    if name == "MATCHES_CSV":
        return get_match_data_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
