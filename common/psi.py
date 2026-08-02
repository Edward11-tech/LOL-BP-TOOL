"""
共享 PSI (Population Stability Index) 漂移监控组件

功能描述:
    实现PSI（群体稳定性指标）计算，用于监控特征数据分布是否发生漂移。
    被 bp_prediction.feature_monitor 和 bp_recommendation.feature_monitor 共同复用。
    PSI阈值标准：< 0.1 稳定，0.1-0.25 轻微漂移，>= 0.25 显著漂移。

主要类/函数:
    - ValidationResult: 通用特征校验结果数据类
    - PSIMonitor: PSI漂移监控器类，用于计算和监控PSI值
    - compute_psi_from_arrays(): 便捷函数，直接计算两组数据间的PSI
    - check_array_finite(): 检查数组中NaN/Inf值数量

使用方式:
    from common.psi import PSIMonitor, compute_psi_from_arrays, ValidationResult

    # 使用监控器
    monitor = PSIMonitor(baseline_bins=baseline_counts, feature_name="win_rate")
    psi_value = monitor.compute_psi(current_data)
    if monitor.is_drifted():
        print("特征发生漂移")

    # 便捷计算
    psi = compute_psi_from_arrays(baseline_data, current_data)

    # 校验结果
    result = ValidationResult(is_valid=True)
    result.add_violation("数据异常")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from logger_config import get_logger

_log = get_logger(__name__)


@dataclass
class ValidationResult:
    """通用特征校验结果（两个模型共用）"""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_violation(self, msg: str) -> None:
        self.violations.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class PSIMonitor:
    """
    Population Stability Index (PSI) 漂移监控器

    PSI < 0.1   : 稳定
    0.1 <= PSI < 0.25 : 轻微漂移
    PSI >= 0.25 : 显著漂移
    """

    def __init__(
        self,
        baseline_bins: np.ndarray,
        feature_name: str,
        n_bins: int = 10,
        bin_edges: Optional[np.ndarray] = None,
    ):
        """
        Args:
            baseline_bins: 基线分箱计数（来自 np.histogram 返回的 counts）
            feature_name: 特征名（用于日志标识）
            n_bins: 分箱数（仅当无 bin_edges 时用于动态分箱回退）
            bin_edges: 基线分箱边界（来自 np.histogram 返回的 edges）。
                       提供后 compute_psi 会用固定边界对当前数据分箱，
                       避免基线/当前数据使用不同分箱导致的 PSI 不可靠。
                       None 时回退到动态分箱（带 warning）。
        """
        self.baseline_bins = baseline_bins.astype(np.float64)
        self.feature_name = feature_name
        self.n_bins = n_bins
        self.bin_edges = (
            np.asarray(bin_edges, dtype=np.float64) if bin_edges is not None else None
        )
        self.psi_value: float = 0.0
        self._edges_warned = False  # 控制向后兼容警告只打印一次

    def compute_psi(self, current_values: np.ndarray) -> float:
        """计算当前样本相对于基线的 PSI。

        若构造时传入 bin_edges，使用固定边界对当前数据分箱（修复分箱对齐 BUG）；
        否则回退到动态分箱（np.histogram 基于 current_values 自身 min/max 分箱），
        并打印一次 warning 提示重建基线。
        """
        current_values = np.asarray(current_values, dtype=np.float64)
        current_values = current_values[np.isfinite(current_values)]
        if len(current_values) == 0:
            return 1.0

        baseline_total = self.baseline_bins.sum()
        if baseline_total == 0:
            return 0.0

        baseline_pct = self.baseline_bins / baseline_total

        try:
            if self.bin_edges is not None:
                # 修复：使用基线固定的 bin_edges 对当前数据分箱
                # np.histogram 会自动把超出 edges 范围的值计入首尾 bin
                current_counts, _ = np.histogram(current_values, bins=self.bin_edges)
            else:
                if not self._edges_warned:
                    _log.warning(
                        "PSIMonitor('%s') 缺少 bin_edges，回退到动态分箱 "
                        "(结果不可靠，请重建基线)",
                        self.feature_name,
                    )
                    self._edges_warned = True
                current_counts, _ = np.histogram(current_values, bins=self.n_bins)
            current_pct = current_counts / max(current_counts.sum(), 1)

            eps = 1e-6
            baseline_pct = np.clip(baseline_pct, eps, None)
            current_pct = np.clip(current_pct, eps, None)

            psi = float(
                np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            )
            self.psi_value = psi
            return psi
        except Exception as e:
            _log.warning("PSI computation failed for %s: %s", self.feature_name, e)
            return 0.0

    def is_drifted(self, threshold: float = 0.25) -> bool:
        return self.psi_value >= threshold


def compute_psi_from_arrays(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    return_edges: bool = False,
):
    """便捷函数：直接计算两组数据之间的 PSI（无需构建 PSIMonitor 对象）。

    Args:
        baseline: 基线数据
        current: 当前数据
        n_bins: 分箱数
        return_edges: 若为 True，返回 (psi, bin_edges)，便于调用方持久化 bin_edges
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    baseline = baseline[np.isfinite(baseline)]
    if len(baseline) < n_bins:
        return (0.0, None) if return_edges else 0.0
    counts, edges = np.histogram(baseline, bins=n_bins)
    monitor = PSIMonitor(
        baseline_bins=counts,
        feature_name="ad_hoc",
        n_bins=n_bins,
        bin_edges=edges,
    )
    psi = monitor.compute_psi(current)
    return (psi, edges) if return_edges else psi


def check_array_finite(arr: np.ndarray, name: str = "array") -> Tuple[bool, int, Optional[str]]:
    """检查数组中 NaN/Inf 数量，返回 (是否全部有限, 异常数量, 描述字符串)。"""
    arr = np.asarray(arr)
    finite_mask = np.isfinite(arr)
    n_bad = int((~finite_mask).sum())
    if n_bad == 0:
        return True, 0, None
    return False, n_bad, f"{name}: {n_bad} NaN/Inf values"
