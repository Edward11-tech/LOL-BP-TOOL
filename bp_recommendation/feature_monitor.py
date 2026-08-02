"""
特征监控模块

提供三大特征监控机制：
1. 特征级漂移检测 (PSI - Population Stability Index)
2. 特征值范围校验 (Range Validation)
3. 特征完整性校验 (Integrity Validation)

用于在训练和推理时避免无效特征进入模型。
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.psi import PSIMonitor, ValidationResult as FeatureValidationResult
from logger_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# 1. 特征值范围定义 (彻底消除硬编码，与 SSOT 对齐)
# ============================================================================

from bp_recommendation.feature_pipeline import CANDIDATE_FEAT_MAP, CANDIDATE_DIM

# 维护一个纯粹的“特征名 -> 范围”的字典，不包含任何位置索引信息
_BASE_CANDIDATE_RANGES: Dict[str, Tuple[float, float]] = {
    "meta_pick": (-1e-6, 1.0 + 1e-6),  # 允许浮点负零 -0.0000 和微小的精度误差
    "meta_ban": (-1e-6, 1.0 + 1e-6),
    "meta_presence": (-1e-6, 1.0 + 1e-6),
    "meta_wr": (-1e-6, 1.0 + 1e-6), # 允许 0.0 和 1.0（极端情况）及浮点精度误差
    
    "player_mastery": (0.0, 150.0), # 放宽熟练度上限，以防绝活哥打出超高分
    "player_recent_kda": (0.0, 100.0),  # 放宽上限：极端 KDA 可超过 50（实测最大 74.58）
    "player_recent_wr": (0.0, 1.0),
    "player_overall_kda": (0.0, 100.0),  # 同步放宽
    "player_overall_wr": (0.0, 1.0),
    "player_overall_games": (0.0, 10000.0),
    
    "pos_top": (0.0, 1.0), "pos_jng": (0.0, 1.0), "pos_mid": (0.0, 1.0), 
    "pos_bot": (0.0, 1.0), "pos_sup": (0.0, 1.0),
    
    "ally_synergy": (0.0, 1.0), "enemy_synergy": (0.0, 1.0),
    "enemy_counter": (0.0, 1.0), "ally_counter": (0.0, 1.0),
    "ally_role_fit": (0.0, 1.0), "enemy_role_fit": (0.0, 1.0),
    
    "is_pick": (0.0, 1.0),
    "enemy_mastery_max": (0.0, 150.0),
    "enemy_mastery_mean": (0.0, 150.0),
    "ban_step": (0.0, 20.0),
    
    "grudge": (0.0, 1.0),
    "respect": (0.0, 1.0),
    "hot_streak": (0.0, 10.0), # 热度综合分可能超过1
    
    "n_ally_picked": (0.0, 5.0),
    "is_red_side": (0.0, 1.0),
    "last_ally_synergy": (0.0, 1.0),
    "is_fearless_banned": (0.0, 1.0),
    "player_recent_games": (0.0, 1000.0)  # 90天窗口内该英雄对局数
}

# 动态构建 CANDIDATE_MATRIX_RANGES，索引永远与 feature_pipeline.py 保持 100% 同步
CANDIDATE_MATRIX_RANGES: Dict[int, Tuple[str, float, float]] = {}
for feat_name, idx in CANDIDATE_FEAT_MAP.items():
    if feat_name in _BASE_CANDIDATE_RANGES:
        lo, hi = _BASE_CANDIDATE_RANGES[feat_name]
        CANDIDATE_MATRIX_RANGES[idx] = (feat_name, lo, hi)
    else:
        # 如果新加了特征但没写范围，给一个极度宽松的默认兜底，防止直接报错
        CANDIDATE_MATRIX_RANGES[idx] = (feat_name, -10000.0, 10000.0)

# global_context 20 维特征：必须与 bp_predict.py 中的拼接逻辑 100% 对应！
# league_vec(3) + b_style(5) + r_style(5) + [playoffs_f, first_pick_f](2) + game_num_vec(5)
GLOBAL_CONTEXT_RANGES: Dict[int, Tuple[str, float, float]] = {
    0:  ("league_LPL", 0.0, 1.0),
    1:  ("league_LCK", 0.0, 1.0),
    2:  ("league_LEC", 0.0, 1.0),
    
    3:  ("blue_team_avg_ckpm", 0.0, 5.0),
    4:  ("blue_team_avg_golddiffat15", -10000.0, 10000.0),
    5:  ("blue_team_avg_gamelength", 900.0, 4000.0),
    6:  ("blue_team_firstdragon_rate", 0.0, 1.0),
    7:  ("blue_team_firsttower_rate", 0.0, 1.0),
    
    8:  ("red_team_avg_ckpm", 0.0, 5.0),
    9:  ("red_team_avg_golddiffat15", -10000.0, 10000.0),
    10: ("red_team_avg_gamelength", 900.0, 4000.0),
    11: ("red_team_firstdragon_rate", 0.0, 1.0),
    12: ("red_team_firsttower_rate", 0.0, 1.0),
    
    13: ("is_playoffs", 0.0, 1.0),
    14: ("first_pick_map_side", 0.0, 1.0),
    
    15: ("is_game_1", 0.0, 1.0),
    16: ("is_game_2", 0.0, 1.0),
    17: ("is_game_3", 0.0, 1.0),
    18: ("is_game_4", 0.0, 1.0),
    19: ("is_game_5", 0.0, 1.0),
}

# Cascade 特征范围（LightGBM 输入）
CASCADE_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "cs_logit": (-100.0, 100.0),
    "nocs_logit": (-100.0, 100.0),
    "logit_diff": (-200.0, 200.0),
    "cs_rank_pct": (0.0, 1.0),
    "nocs_rank_pct": (0.0, 1.0),
    "meta_presence": (-1e-6, 1.0 + 1e-6),
    "meta_wr": (-1e-6, 1.0 + 1e-6),
    "player_mastery": (0.0, 150.0),
    "player_recent_wr": (0.0, 1.0),
    "synergy_ally": (0.0, 1.0),
    "counter_enemy": (0.0, 1.0),
    "role_fit_ally": (0.0, 1.0),
}


# ============================================================================
# 3. 特征监控器
# ============================================================================

class FeatureMonitor:
    """
    特征监控器

    集成三大监控机制：
    1. 特征级漂移检测 (PSI)
    2. 特征值范围校验
    3. 特征完整性校验
    """

    def __init__(self, baseline_dir: Optional[str] = None):
        """
        Args:
            baseline_dir: 基线特征文件目录，用于加载 PSI 基线分布
        """
        self.psi_monitors: Dict[str, PSIMonitor] = {}
        self.baseline_dir = baseline_dir
        self._baseline_loaded = False

        if baseline_dir and os.path.exists(baseline_dir):
            self._load_baseline()

    # ----------------------------------------------------------------------
    # 3.1 特征级漂移检测 (PSI)
    # ----------------------------------------------------------------------

    def _load_baseline(self):
        """从基线文件加载 PSI 基线分布

        支持两种格式：
            旧格式: {"feat": [c1, ..., c10]}（仅 counts，无 bin_edges）
            新格式: {"feat": {"counts": [...], "bin_edges": [...]}}（含分箱边界）

        旧格式会触发 warning 并回退到动态分箱（结果不可靠）。
        """
        baseline_path = os.path.join(self.baseline_dir, "feature_baseline.json")
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
            logger.info(f"Loaded PSI baseline for {len(self.psi_monitors)} features.")
        except Exception as e:
            logger.warning(f"Failed to load baseline: {e}")

    def build_baseline(self, feature_dict: Dict[str, np.ndarray], output_path: str):
        """
        从训练数据构建 PSI 基线分布（新格式：保存 counts + bin_edges）

        Args:
            feature_dict: {feature_name: values_array}
            output_path: 基线文件保存路径
        """
        baseline_data = {}
        n_bins = 10

        for feat_name, values in feature_dict.items():
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) < n_bins:
                logger.warning(f"Feature {feat_name} has only {len(values)} samples, skipping.")
                continue

            counts, edges = np.histogram(values, bins=n_bins)
            baseline_data[feat_name] = {
                "counts": counts.tolist(),
                "bin_edges": edges.tolist(),
            }

            self.psi_monitors[feat_name] = PSIMonitor(
                baseline_bins=counts.astype(np.float64),
                feature_name=feat_name,
                bin_edges=edges,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)

        self._baseline_loaded = True
        logger.info(f"Built and saved PSI baseline for {len(baseline_data)} features to {output_path}")

    def check_drift(self, feature_dict: Dict[str, np.ndarray],
                    threshold: float = 0.25) -> Tuple[bool, Dict[str, float]]:
        """
        检查特征漂移

        Args:
            feature_dict: {feature_name: current_values}
            threshold: PSI 漂移阈值

        Returns:
            (is_drifted, psi_report)
        """
        if not self._baseline_loaded:
            return False, {}

        psi_report = {}
        any_drifted = False

        for feat_name, values in feature_dict.items():
            if feat_name not in self.psi_monitors:
                continue

            monitor = self.psi_monitors[feat_name]
            psi = monitor.compute_psi(np.asarray(values, dtype=np.float64))
            psi_report[feat_name] = round(psi, 4)

            if monitor.is_drifted(threshold):
                any_drifted = True
                logger.warning(
                    f"Feature drift detected: {feat_name} PSI={psi:.4f} >= {threshold}"
                )

        return any_drifted, psi_report

    # ----------------------------------------------------------------------
    # 3.2 特征值范围校验
    # ----------------------------------------------------------------------

    def validate_candidate_matrix(self, matrix: np.ndarray) -> FeatureValidationResult:
        """
        校验 candidate_matrix 的特征值范围

        Args:
            matrix: shape [vocab_size, CANDIDATE_DIM] 的候选矩阵
        """
        result = FeatureValidationResult(is_valid=True)

        if matrix.ndim != 2:
            result.add_violation(
                f"candidate_matrix must be 2D, got {matrix.ndim}D"
            )
            return result

        if matrix.shape[1] < CANDIDATE_DIM:
            result.add_violation(
                f"candidate_matrix must have {CANDIDATE_DIM} columns, got {matrix.shape[1]}"
            )
            return result

        # 检查每列特征值范围
        for col_idx, (feat_name, lo, hi) in CANDIDATE_MATRIX_RANGES.items():
            col = matrix[:, col_idx]
            finite_mask = np.isfinite(col)
            if not finite_mask.all():
                nan_count = (~finite_mask).sum()
                result.add_violation(
                    f"col[{col_idx}] {feat_name}: {nan_count} NaN/Inf values"
                )
                continue

            col_finite = col[finite_mask]
            if len(col_finite) == 0:
                result.add_violation(f"col[{col_idx}] {feat_name}: all values are NaN/Inf")
                continue

            out_of_range = (col_finite < lo) | (col_finite > hi)
            if out_of_range.any():
                n_violations = out_of_range.sum()
                min_val = col_finite.min()
                max_val = col_finite.max()
                result.add_violation(
                    f"col[{col_idx}] {feat_name}: {n_violations} values out of range "
                    f"[{lo}, {hi}], actual range [{min_val:.4f}, {max_val:.4f}]"
                )

        return result

    def validate_global_context(self, ctx: np.ndarray) -> FeatureValidationResult:
        """
        校验 global_context 的特征值范围

        Args:
            ctx: shape [20] 的全局上下文向量
        """
        result = FeatureValidationResult(is_valid=True)

        if ctx.ndim != 1:
            result.add_violation(
                f"global_context must be 1D, got {ctx.ndim}D"
            )
            return result

        if ctx.shape[0] < 20:
            result.add_violation(
                f"global_context must have 20 elements, got {ctx.shape[0]}"
            )
            return result

        for idx, (feat_name, lo, hi) in GLOBAL_CONTEXT_RANGES.items():
            val = ctx[idx]
            if not np.isfinite(val):
                result.add_violation(f"ctx[{idx}] {feat_name}: NaN/Inf value")
                continue
            if val < lo or val > hi:
                result.add_violation(
                    f"ctx[{idx}] {feat_name}: value {val:.4f} out of range [{lo}, {hi}]"
                )

        return result

    def validate_cascade_features(self, df, feature_cols: List[str]) -> FeatureValidationResult:
        """
        校验 Cascade (LightGBM) 特征值范围

        Args:
            df: 包含特征列的 DataFrame
            feature_cols: 特征列名列表
        """
        result = FeatureValidationResult(is_valid=True)

        for col_name in feature_cols:
            if col_name not in df.columns:
                result.add_violation(f"Missing cascade feature column: {col_name}")
                continue

            col = df[col_name].values
            finite_mask = np.isfinite(col)
            if not finite_mask.all():
                nan_count = int((~finite_mask).sum())
                result.add_violation(f"{col_name}: {nan_count} NaN/Inf values")
                continue

            if col_name in CASCADE_FEATURE_RANGES:
                lo, hi = CASCADE_FEATURE_RANGES[col_name]
                out_of_range = (col < lo) | (col > hi)
                if out_of_range.any():
                    n_violations = int(out_of_range.sum())
                    result.add_violation(
                        f"{col_name}: {n_violations} values out of range [{lo}, {hi}]"
                    )

        return result

    # ----------------------------------------------------------------------
    # 3.3 特征完整性校验
    # ----------------------------------------------------------------------

    def validate_feature_integrity(
        self,
        candidate_matrix: np.ndarray,
        global_context: np.ndarray,
        available_mask: np.ndarray,
        player_matrix: Optional[np.ndarray] = None,
        expected_vocab_size: int = 0,
    ) -> FeatureValidationResult:
        """
        校验特征矩阵的完整性

        Args:
            candidate_matrix: [vocab_size, CANDIDATE_DIM]
            global_context: [20]
            available_mask: [vocab_size]
            player_matrix: [vocab_size, 7] 或 None
            expected_vocab_size: 期望的 vocab_size
        """
        result = FeatureValidationResult(is_valid=True)

        # 1. 维度一致性
        if candidate_matrix.ndim != 2:
            result.add_violation(f"candidate_matrix must be 2D, got {candidate_matrix.ndim}D")
        else:
            actual_vocab = candidate_matrix.shape[0]
            if expected_vocab_size > 0 and actual_vocab != expected_vocab_size:
                result.add_violation(
                    f"candidate_matrix vocab_size mismatch: expected {expected_vocab_size}, "
                    f"got {actual_vocab}"
                )

            if candidate_matrix.shape[1] != CANDIDATE_DIM:
                result.add_violation(
                    f"candidate_matrix must have {CANDIDATE_DIM} columns, got {candidate_matrix.shape[1]}"
                )

        if available_mask.ndim != 1:
            result.add_violation(f"available_mask must be 1D, got {available_mask.ndim}D")
        elif candidate_matrix.ndim == 2 and available_mask.shape[0] != candidate_matrix.shape[0]:
            result.add_violation(
                f"available_mask length {available_mask.shape[0]} != "
                f"candidate_matrix rows {candidate_matrix.shape[0]}"
            )

        if global_context.shape != (20,):
            result.add_violation(
                f"global_context shape must be (20,), got {global_context.shape}"
            )

        # 2. NaN / Inf 检查
        nan_count_cm = int(np.sum(~np.isfinite(candidate_matrix)))
        if nan_count_cm > 0:
            result.add_violation(
                f"candidate_matrix contains {nan_count_cm} NaN/Inf values"
            )

        nan_count_gc = int(np.sum(~np.isfinite(global_context)))
        if nan_count_gc > 0:
            result.add_violation(
                f"global_context contains {nan_count_gc} NaN/Inf values"
            )

        # 3. available_mask 值域检查
        if available_mask.size > 0:
            unique_vals = np.unique(available_mask)
            invalid_vals = unique_vals[(unique_vals != 0.0) & (unique_vals != 1.0)]
            if len(invalid_vals) > 0:
                result.add_violation(
                    f"available_mask contains non-binary values: {invalid_vals[:5]}"
                )

        # 4. player_matrix 完整性
        if player_matrix is not None:
            if player_matrix.ndim != 2:
                result.add_violation(f"player_matrix must be 2D, got {player_matrix.ndim}D")
            else:
                if player_matrix.shape[1] != 6:
                    result.add_violation(
                        f"player_matrix must have 6 columns, got {player_matrix.shape[1]}"
                    )
                nan_count_pm = int(np.sum(~np.isfinite(player_matrix)))
                if nan_count_pm > 0:
                    result.add_violation(
                        f"player_matrix contains {nan_count_pm} NaN/Inf values"
                    )

        # 5. 全零行检测（candidate_matrix 中不应有全零的有效行）
        if candidate_matrix.ndim == 2 and available_mask.shape[0] == candidate_matrix.shape[0]:
            valid_rows = available_mask > 0.5
            if valid_rows.any():
                valid_matrix = candidate_matrix[valid_rows]
                all_zero_rows = np.all(valid_matrix == 0, axis=1)
                n_all_zero = int(all_zero_rows.sum())
                if n_all_zero > 0:
                    result.add_warning(
                        f"{n_all_zero} valid candidates have all-zero feature vectors "
                        f"(may indicate missing feature data)"
                    )

        return result

    # ----------------------------------------------------------------------
    # 3.4 综合校验入口
    # ----------------------------------------------------------------------

    def full_check(
        self,
        candidate_matrix: np.ndarray,
        global_context: np.ndarray,
        available_mask: np.ndarray,
        player_matrix: Optional[np.ndarray] = None,
        expected_vocab_size: int = 0,
        feature_dict_for_psi: Optional[Dict[str, np.ndarray]] = None,
        psi_threshold: float = 0.25,
    ) -> Tuple[bool, Dict]:
        """
        执行完整的特征监控检查

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
        cm_range = self.validate_candidate_matrix(candidate_matrix)
        gc_range = self.validate_global_context(global_context)
        report["range_check"] = {
            "candidate_matrix": {
                "is_valid": cm_range.is_valid,
                "violations": cm_range.violations,
                "warnings": cm_range.warnings,
            },
            "global_context": {
                "is_valid": gc_range.is_valid,
                "violations": gc_range.violations,
                "warnings": gc_range.warnings,
            },
        }
        if not cm_range.is_valid or not gc_range.is_valid:
            all_passed = False

        # 2. 完整性校验
        integrity = self.validate_feature_integrity(
            candidate_matrix, global_context, available_mask,
            player_matrix, expected_vocab_size,
        )
        report["integrity_check"] = {
            "is_valid": integrity.is_valid,
            "violations": integrity.violations,
            "warnings": integrity.warnings,
        }
        if not integrity.is_valid:
            all_passed = False

        # 3. 漂移检测
        if feature_dict_for_psi and self._baseline_loaded:
            is_drifted, psi_report = self.check_drift(feature_dict_for_psi, psi_threshold)
            report["drift_check"] = {
                "is_drifted": is_drifted,
                "psi_values": psi_report,
                "threshold": psi_threshold,
            }
            if is_drifted:
                all_passed = False
        else:
            report["drift_check"] = {"skipped": True, "reason": "No baseline or no feature_dict provided"}

        return all_passed, report


# ============================================================================
# 4. 便捷函数
# ============================================================================

def quick_validate_inference_inputs(
    candidate_matrix: np.ndarray,
    global_context: np.ndarray,
    available_mask: np.ndarray,
) -> FeatureValidationResult:
    """
    推理时快速校验输入特征（不含 PSI 漂移检测）

    用于 bp_predict.py 推理前的快速检查。
    """
    monitor = FeatureMonitor()
    return monitor.validate_feature_integrity(
        candidate_matrix, global_context, available_mask
    )


def quick_validate_range(
    candidate_matrix: np.ndarray,
    global_context: np.ndarray,
) -> Tuple[FeatureValidationResult, FeatureValidationResult]:
    """快速范围校验"""
    monitor = FeatureMonitor()
    return (
        monitor.validate_candidate_matrix(candidate_matrix),
        monitor.validate_global_context(global_context),
    )
