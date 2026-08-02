"""
预测模型特征监控模块
=====================
为 bp_prediction (CatBoost 胜率预测模型) 提供三大特征监控机制：
1. 特征级漂移检测 (PSI)
2. 特征值范围校验 (基于 feature_cols 动态适配)
3. 特征完整性校验 (NaN/Inf、维度、全零检测)

与 bp_recommendation/feature_monitor.py 的区别：
- 预测模型使用 CatBoost，特征为 1D 向量 (feature_cols)
- 特征列名动态加载自 feature_columns.json
- 范围校验基于特征名前缀自动分类
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.psi import PSIMonitor, ValidationResult as PredictionFeatureValidationResult
from logger_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# 1. 特征范围定义 (按特征名关键字智能匹配)
# ============================================================================

# 特征名核心关键字 → (min, max) 范围映射
# 注意：我们适当放宽了部分极端情况下的业务上限，防止监控误杀
FEATURE_KEYWORD_RANGES: Dict[str, Tuple[float, float]] = {
    "league_": (0.0, 1.0),
    "is_playoff": (0.0, 1.0),
    "is_blue_map_side": (0.0, 1.0),
    
    "mastery_score": (0.0, 150.0), # 放宽熟练度上限
    "player_recent_kda": (0.0, 50.0),
    "player_recent_wr": (0.0, 1.0),
    "player_overall_recent_wr": (0.0, 1.0),
    "player_overall_recent_kda": (0.0, 50.0),
    "player_overall_recent_games": (0.0, 10000.0),
    
    "meta_win_rate_pit": (0.0, 1.0),
    "meta_pick_rate_pit": (0.0, 1.0),
    "meta_ban_rate_pit": (0.0, 1.0),
    "meta_presence_pit": (0.0, 1.0),
    "meta_patch_drift_index": (0.0, 10.0),
    "meta_pick_drift_index": (0.0, 10.0),
    
    "team_avg_gamelength": (900.0, 4000.0),
    "team_avg_ckpm": (0.0, 5.0),
    "team_avg_golddiffat15": (-10000.0, 10000.0),
    "team_firstdragon_rate": (0.0, 1.0),
    "team_firsttower_rate": (0.0, 1.0),
    "team_recent_wr": (0.0, 1.0),
    "team_side_wr": (0.0, 1.0),
    "team_streak": (-20.0, 20.0),
    "team_profile_games": (0.0, 10000.0),
    "team_avg_kills": (0.0, 100.0),
    "team_avg_deaths": (0.0, 100.0),
    "team_avg_assists": (0.0, 200.0),
    "team_bloodiness": (0.0, 5.0),
    "team_snowball_rate": (0.0, 1.0),
    "team_led_at_15_rate": (0.0, 1.0),
    
    "comp_": (0.0, 50.0),
    "champ_wr_delta": (-1.0, 1.0),
    "champ_kda_delta": (-50.0, 50.0),
    "mastery_x_": (0.0, 150.0),
    "team_wr_max_gap": (-1.0, 1.0),
    "team_wr_balance": (0.0, 1.0),
    "team_wr_x_roster_wr": (0.0, 1.0),
    
    "bloodiness_x_aggression": (-20.0, 20.0),
    "early_power_x_snowball": (-20.0, 20.0),
    "ckpm_x_aggression": (-20.0, 20.0),
    "wr_x_aggression": (-20.0, 20.0),
    
    "tf_win_logits": (-100.0, 100.0),
    "tf_cosine_sim": (-1.0, 1.0),
    "tf_blue_l2norm": (0.0, 100.0),
    "tf_red_l2norm": (0.0, 100.0),
    "champion_": (0.0, 1.0),
}

DEFAULT_RANGE: Tuple[float, float] = (-10000.0, 10000.0)

def get_feature_range(feature_name: str) -> Tuple[float, float]:
    """智能解析特征名，动态计算合法范围"""
    is_diff = feature_name.startswith("diff_")
    # 如果是差分特征，剥离前缀以寻找它底层的本体特征
    base_name = feature_name[5:] if is_diff else feature_name

    matched_rng = None
    # 【修复 1】：使用 in 关键字匹配，无视 blue_top_ 或 red_ 等前缀
    for key, rng in FEATURE_KEYWORD_RANGES.items():
        if key in base_name:
            matched_rng = rng
            break

    if matched_rng is None:
        return DEFAULT_RANGE

    # 【修复 2】：动态计算差分特征的合法范围
    # 数学原理：如果基础特征范围是 [A, B]，那么两者差值的极值范围就是 [-(B-A), B-A]
    # 例如胜率是 [0, 1]，差值就是 [-1, 1]。经济是 [-10000, 10000]，差值就是 [-20000, 20000]。
    if is_diff:
        span = matched_rng[1] - matched_rng[0]
        return (-span, span)
        
    return matched_rng

# ============================================================================
# 3. 预测特征监控器
# ============================================================================

class PredictionFeatureMonitor:
    """
    预测模型特征监控器

    集成三大监控机制：
    1. 特征级漂移检测 (PSI)
    2. 特征值范围校验 (基于特征名前缀自动适配)
    3. 特征完整性校验 (NaN/Inf、维度、全零检测)
    """

    def __init__(self, feature_cols: Optional[List[str]] = None,
                 baseline_dir: Optional[str] = None):
        """
        Args:
            feature_cols: 训练时的特征列名列表
            baseline_dir: PSI 基线文件目录
        """
        self.feature_cols = feature_cols or []
        self.psi_monitors: Dict[str, PSIMonitor] = {}
        self.baseline_dir = baseline_dir
        self._baseline_loaded = False

        if baseline_dir and os.path.exists(baseline_dir):
            self._load_baseline()

    # ----------------------------------------------------------------------
    # 3.1 特征级漂移检测 (PSI)
    # ----------------------------------------------------------------------

    def _load_baseline(self):
        """加载 PSI 基线分布

        支持两种格式：
            旧格式: {"feat": [c1, ..., c10]}（仅 counts，无 bin_edges）
            新格式: {"feat": {"counts": [...], "bin_edges": [...]}}（含分箱边界）

        旧格式会触发 warning 并回退到动态分箱（结果不可靠）。
        """
        baseline_path = os.path.join(self.baseline_dir, "prediction_feature_baseline.json")
        if not os.path.exists(baseline_path):
            logger.info(f"Baseline file not found at {baseline_path}, PSI monitoring disabled.")
            return

        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)

            n_old_format = 0
            for feat_name, payload in baseline_data.items():
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
                    logger.warning(
                        f"Baseline entry for {feat_name} has unexpected type {type(payload)}, skipping"
                    )
                    continue

                self.psi_monitors[feat_name] = PSIMonitor(
                    baseline_bins=counts,
                    feature_name=feat_name,
                    bin_edges=edges,
                )
            self._baseline_loaded = True
            if n_old_format > 0:
                logger.warning(
                    f"Loaded {n_old_format} features in OLD format (no bin_edges). "
                    f"PSI will fall back to dynamic binning. Rebuild baseline ASAP."
                )
            logger.info(f"Loaded PSI baseline for {len(self.psi_monitors)} prediction features.")
        except Exception as e:
            logger.warning(f"Failed to load baseline: {e}")

    def build_baseline(self, features_df: pd.DataFrame, output_path: str):
        """
        从训练数据构建 PSI 基线分布（新格式：保存 counts + bin_edges）

        Args:
            features_df: 训练特征 DataFrame
            output_path: 基线文件保存路径
        """
        baseline_data = {}
        n_bins = 10

        for col in features_df.columns:
            values = features_df[col].values.astype(np.float64)
            values = values[np.isfinite(values)]
            if len(values) < n_bins:
                continue

            counts, edges = np.histogram(values, bins=n_bins)
            baseline_data[col] = {
                "counts": counts.tolist(),
                "bin_edges": edges.tolist(),
            }

            self.psi_monitors[col] = PSIMonitor(
                baseline_bins=counts.astype(np.float64),
                feature_name=col,
                bin_edges=edges,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)

        self._baseline_loaded = True
        logger.info(f"Built and saved PSI baseline for {len(baseline_data)} features to {output_path}")

    def check_drift(self, features_df: pd.DataFrame,
                    threshold: float = 0.25) -> Tuple[bool, Dict[str, float]]:
        """检查特征漂移"""
        if not self._baseline_loaded:
            return False, {}

        psi_report = {}
        any_drifted = False

        for col in features_df.columns:
            if col not in self.psi_monitors:
                continue

            monitor = self.psi_monitors[col]
            psi = monitor.compute_psi(features_df[col].values.astype(np.float64))
            psi_report[col] = round(psi, 4)

            if monitor.is_drifted(threshold):
                any_drifted = True
                logger.warning(f"Feature drift detected: {col} PSI={psi:.4f} >= {threshold}")

        return any_drifted, psi_report

    # ----------------------------------------------------------------------
    # 3.2 特征值范围校验
    # ----------------------------------------------------------------------

    def validate_feature_ranges(self, features_df: pd.DataFrame) -> PredictionFeatureValidationResult:
        """
        校验特征 DataFrame 中每列的值范围

        Args:
            features_df: 特征 DataFrame，列名对应 feature_cols
        """
        result = PredictionFeatureValidationResult(is_valid=True)

        if self.feature_cols:
            expected_cols = set(self.feature_cols)
            actual_cols = set(features_df.columns)
            missing = expected_cols - actual_cols
            if missing:
                result.add_violation(
                    f"Missing {len(missing)} feature columns: {list(missing)[:5]}..."
                )
            extra = actual_cols - expected_cols
            if extra:
                result.add_warning(
                    f"Found {len(extra)} unexpected columns: {list(extra)[:5]}..."
                )

        for col in features_df.columns:
            col_values = features_df[col].values.astype(np.float64)

            # NaN/Inf 检查
            finite_mask = np.isfinite(col_values)
            if not finite_mask.all():
                nan_count = int((~finite_mask).sum())
                result.add_violation(f"{col}: {nan_count} NaN/Inf values")
                continue

            # 范围检查
            lo, hi = get_feature_range(col)
            out_of_range = (col_values < lo) | (col_values > hi)
            if out_of_range.any():
                n_violations = int(out_of_range.sum())
                min_val = col_values.min()
                max_val = col_values.max()
                result.add_violation(
                    f"{col}: {n_violations} values out of range [{lo}, {hi}], "
                    f"actual range [{min_val:.4f}, {max_val:.4f}]"
                )

        return result

    def validate_feature_array(self, X: np.ndarray) -> PredictionFeatureValidationResult:
        """
        校验 numpy 特征数组的值范围

        Args:
            X: shape [n_samples, n_features] 或 [n_features]
        """
        result = PredictionFeatureValidationResult(is_valid=True)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            result.add_violation(f"Feature array must be 2D, got {X.ndim}D")
            return result

        n_features = X.shape[1]
        if self.feature_cols and n_features != len(self.feature_cols):
            result.add_violation(
                f"Feature count mismatch: expected {len(self.feature_cols)}, got {n_features}"
            )

        # NaN/Inf 检查
        nan_count = int(np.sum(~np.isfinite(X)))
        if nan_count > 0:
            result.add_violation(f"Feature array contains {nan_count} NaN/Inf values")

        # 逐列范围检查
        for i in range(n_features):
            col_name = self.feature_cols[i] if i < len(self.feature_cols) else f"col_{i}"
            lo, hi = get_feature_range(col_name)
            col = X[:, i]
            finite_mask = np.isfinite(col)
            if not finite_mask.all():
                continue  # 已在上方汇总报告
            out_of_range = (col < lo) | (col > hi)
            if out_of_range.any():
                n_violations = int(out_of_range.sum())
                result.add_violation(
                    f"{col_name} (col {i}): {n_violations} values out of range [{lo}, {hi}]"
                )

        return result

    # ----------------------------------------------------------------------
    # 3.3 特征完整性校验
    # ----------------------------------------------------------------------

    def validate_integrity(self, features_df: pd.DataFrame,
                           expected_n_features: Optional[int] = None) -> PredictionFeatureValidationResult:
        """
        校验特征 DataFrame 的完整性

        Args:
            features_df: 特征 DataFrame
            expected_n_features: 期望的特征列数
        """
        result = PredictionFeatureValidationResult(is_valid=True)

        # 1. 维度检查
        if expected_n_features is not None and features_df.shape[1] != expected_n_features:
            result.add_violation(
                f"Feature count mismatch: expected {expected_n_features}, "
                f"got {features_df.shape[1]}"
            )

        if self.feature_cols:
            if features_df.shape[1] != len(self.feature_cols):
                result.add_violation(
                    f"Column count mismatch: feature_cols has {len(self.feature_cols)}, "
                    f"DataFrame has {features_df.shape[1]}"
                )

        # 2. NaN/Inf 检查
        values = features_df.values.astype(np.float64)
        nan_count = int(np.sum(~np.isfinite(values)))
        if nan_count > 0:
            result.add_violation(
                f"Feature matrix contains {nan_count} NaN/Inf values"
            )

        # 3. 全零行检测
        all_zero_rows = np.all(values == 0, axis=1)
        n_all_zero = int(all_zero_rows.sum())
        if n_all_zero > 0:
            result.add_warning(
                f"{n_all_zero} rows have all-zero feature vectors "
                f"(may indicate missing feature data)"
            )

        # 4. 全零列检测
        all_zero_cols = np.all(values == 0, axis=0)
        n_all_zero_cols = int(all_zero_cols.sum())
        if n_all_zero_cols > 0:
            zero_col_names = [features_df.columns[i] for i in np.where(all_zero_cols)[0]]
            result.add_warning(
                f"{n_all_zero_cols} columns are all-zero: {zero_col_names[:5]}..."
            )

        # 5. 特征列顺序检查
        if self.feature_cols and list(features_df.columns) != self.feature_cols:
            result.add_warning(
                "Feature column order does not match training feature_cols"
            )

        return result

    # ----------------------------------------------------------------------
    # 3.4 综合校验入口
    # ----------------------------------------------------------------------

    def full_check(
        self,
        features_df: pd.DataFrame,
        expected_n_features: Optional[int] = None,
        psi_threshold: float = 0.25,
        enable_psi: bool = False,
    ) -> Tuple[bool, Dict]:
        """
        执行完整的特征监控检查

        Args:
            features_df: 特征 DataFrame
            expected_n_features: 期望特征列数
            psi_threshold: PSI 漂移阈值
            enable_psi: 是否启用 PSI 检测（需要基线）

        Returns:
            (all_passed, report)
        """
        report = {
            "range_check": None,
            "integrity_check": None,
            "drift_check": None,
        }
        all_passed = True

        # 1. 范围校验
        range_result = self.validate_feature_ranges(features_df)
        report["range_check"] = {
            "is_valid": range_result.is_valid,
            "violations": range_result.violations,
            "warnings": range_result.warnings,
        }
        if not range_result.is_valid:
            all_passed = False

        # 2. 完整性校验
        integrity_result = self.validate_integrity(features_df, expected_n_features)
        report["integrity_check"] = {
            "is_valid": integrity_result.is_valid,
            "violations": integrity_result.violations,
            "warnings": integrity_result.warnings,
        }
        if not integrity_result.is_valid:
            all_passed = False

        # 3. 漂移检测
        if enable_psi and self._baseline_loaded:
            is_drifted, psi_report = self.check_drift(features_df, psi_threshold)
            report["drift_check"] = {
                "is_drifted": is_drifted,
                "psi_values": psi_report,
                "threshold": psi_threshold,
            }
            if is_drifted:
                all_passed = False
        else:
            report["drift_check"] = {
                "skipped": True,
                "reason": "PSI disabled or no baseline" if not enable_psi else "No baseline loaded"
            }

        return all_passed, report


# ============================================================================
# 4. 便捷函数
# ============================================================================

def quick_validate_prediction_features(
    features_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> PredictionFeatureValidationResult:
    """推理时快速校验预测特征（不含 PSI 漂移检测）"""
    monitor = PredictionFeatureMonitor(feature_cols=feature_cols)
    return monitor.validate_feature_ranges(features_df)
