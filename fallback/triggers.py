"""
triggers.py — 模型兜底触发器
=============================
触发器 A: Logit Variance Collapse (置信度坍塌)
  - 检测模型输出的 Top 1 Logit - Top 10 Logit < 0.5
  - 说明模型完全丧失区分度

触发器 B: 滑动窗口指标监控
  - 维护最近 30 场比赛的滑动队列
  - 监控 Pick@10, Ban@10, 预测 AUC
  - 任一指标跌破红线 → 降级模式

触发器 C: 极端冷启动检测
  - 检测是否遇到全新大版本 (大量英雄 meta 数据接近 0)
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
from collections import deque
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
from logger_config import get_logger

log = get_logger(__name__)

# ---- 阈值 ----
LOGIT_COLLAPSE_THRESHOLD = 0.5      # Top1 - Top10 logit 差值阈值
SLIDING_WINDOW_SIZE = 30            # 滑动窗口大小
PICK_AT_10_THRESHOLD = 0.55         # Pick@10 红线
BAN_AT_10_THRESHOLD = 0.52          # Ban@10 红线
AUC_THRESHOLD = 0.53                # 预测 AUC 红线

# 指标持久化路径
METRICS_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fallback", "logs", "rolling_metrics.json"
)


class LogitCollapseDetector:
    """
    触发器 A: 置信度坍塌检测。

    用法:
        detector = LogitCollapseDetector()
        if detector.is_collapsed(logits):
            # 触发兜底
    """

    def __init__(self, threshold=LOGIT_COLLAPSE_THRESHOLD):
        self.threshold = threshold
        self.collapse_count = 0
        self.total_checks = 0

    def is_collapsed(self, logits):
        """
        检测 Logit 是否坍塌。

        Args:
            logits: np.ndarray or torch.Tensor, 模型 raw logits (未经过 softmax)

        Returns:
            bool: True 表示置信度坍塌
        """
        self.total_checks += 1

        if logits is None:
            return False

        # 转换为 numpy
        if hasattr(logits, "cpu"):
            logits = logits.detach().cpu().numpy()
        logits = np.asarray(logits, dtype=np.float32).flatten()

        if len(logits) < 10:
            return False

        # 取 Top 1 和 Top 10
        sorted_logits = np.sort(logits)[::-1]
        top1 = sorted_logits[0]
        top10 = sorted_logits[min(9, len(sorted_logits) - 1)]

        diff = top1 - top10

        if diff < self.threshold:
            self.collapse_count += 1
            log.warning(f"Logit 坍塌检测: Top1={top1:.4f}, Top10={top10:.4f}, "
                        f"diff={diff:.4f} < {self.threshold}")
            return True

        return False

    def get_stats(self):
        """获取检测统计"""
        return {
            "total_checks": self.total_checks,
            "collapse_count": self.collapse_count,
            "collapse_rate": round(self.collapse_count / max(self.total_checks, 1), 4),
        }


class RollingMetricsMonitor:
    """
    触发器 B: 滑动窗口指标监控。

    维护最近 30 场比赛的指标队列，实时计算滚动均值。
    当任一指标跌破红线时，触发降级模式。

    用法:
        monitor = RollingMetricsMonitor()
        monitor.record_pick_result(pick_at_10, ban_at_10, auc)
        if monitor.is_degraded():
            # 触发兜底
    """

    def __init__(self, window_size=SLIDING_WINDOW_SIZE,
                 pick_threshold=PICK_AT_10_THRESHOLD,
                 ban_threshold=BAN_AT_10_THRESHOLD,
                 auc_threshold=AUC_THRESHOLD):
        self.window_size = window_size
        self.pick_threshold = pick_threshold
        self.ban_threshold = ban_threshold
        self.auc_threshold = auc_threshold

        # 三个滑动窗口
        self.pick_at_10_window = deque(maxlen=window_size)
        self.ban_at_10_window = deque(maxlen=window_size)
        self.auc_window = deque(maxlen=window_size)

        self._degraded = False
        self._lock = threading.Lock()
        self._load_state()

    def record_pick_result(self, pick_at_10=None, ban_at_10=None, auc=None):
        """
        记录一次推荐结果。

        Args:
            pick_at_10: float or None, 本次 Pick Top-10 命中率
            ban_at_10: float or None, 本次 Ban Top-10 命中率
            auc: float or None, 本次预测 AUC
        """
        with self._lock:
            if pick_at_10 is not None:
                self.pick_at_10_window.append(float(pick_at_10))
            if ban_at_10 is not None:
                self.ban_at_10_window.append(float(ban_at_10))
            if auc is not None:
                self.auc_window.append(float(auc))

            self._check_degradation()
            self._save_state()

    def record_batch_results(self, pick_at_10_list=None, ban_at_10_list=None, auc_list=None):
        """
        批量记录结果。

        Args:
            pick_at_10_list: list[float] or None
            ban_at_10_list: list[float] or None
            auc_list: list[float] or None
        """
        with self._lock:
            if pick_at_10_list:
                for v in pick_at_10_list:
                    self.pick_at_10_window.append(float(v))
            if ban_at_10_list:
                for v in ban_at_10_list:
                    self.ban_at_10_window.append(float(v))
            if auc_list:
                for v in auc_list:
                    self.auc_window.append(float(v))

            self._check_degradation()
            self._save_state()

    def is_degraded(self):
        """
        检查是否处于降级模式。

        使用 OR 逻辑: 任一指标跌破红线即触发。
        """
        return self._degraded

    def get_rolling_metrics(self):
        """获取当前滚动指标"""
        with self._lock:
            return {
                "pick_at_10": {
                    "rolling_mean": self._safe_mean(self.pick_at_10_window),
                    "threshold": self.pick_threshold,
                    "degraded": self._check_single(self.pick_at_10_window, self.pick_threshold),
                    "window_size": len(self.pick_at_10_window),
                },
                "ban_at_10": {
                    "rolling_mean": self._safe_mean(self.ban_at_10_window),
                    "threshold": self.ban_threshold,
                    "degraded": self._check_single(self.ban_at_10_window, self.ban_threshold),
                    "window_size": len(self.ban_at_10_window),
                },
                "auc": {
                    "rolling_mean": self._safe_mean(self.auc_window),
                    "threshold": self.auc_threshold,
                    "degraded": self._check_single(self.auc_window, self.auc_threshold),
                    "window_size": len(self.auc_window),
                },
                "is_degraded": self._degraded,
            }

    def get_metrics(self):
        """获取当前滚动指标 (get_rolling_metrics 的别名)"""
        return self.get_rolling_metrics()

    def reset(self):
        """重置所有指标 (全量重训后调用)"""
        with self._lock:
            self.pick_at_10_window.clear()
            self.ban_at_10_window.clear()
            self.auc_window.clear()
            self._degraded = False
            self._save_state()
            log.info("滑动窗口指标已重置")

    def force_degraded(self):
        """手动强制进入降级模式"""
        with self._lock:
            self._degraded = True
            log.warning("手动强制进入降级模式")

    def force_recovered(self):
        """手动强制恢复"""
        with self._lock:
            self._degraded = False
            log.info("手动恢复正常模式")

    # ---- 内部方法 ----

    def _check_degradation(self):
        """检查是否触发降级 (OR 逻辑)"""
        pick_degraded = self._check_single(self.pick_at_10_window, self.pick_threshold)
        ban_degraded = self._check_single(self.ban_at_10_window, self.ban_threshold)
        auc_degraded = self._check_single(self.auc_window, self.auc_threshold)

        was_degraded = self._degraded
        self._degraded = pick_degraded or ban_degraded or auc_degraded

        if not was_degraded and self._degraded:
            reasons = []
            if pick_degraded:
                reasons.append(f"Pick@10={self._safe_mean(self.pick_at_10_window):.3f} < {self.pick_threshold}")
            if ban_degraded:
                reasons.append(f"Ban@10={self._safe_mean(self.ban_at_10_window):.3f} < {self.ban_threshold}")
            if auc_degraded:
                reasons.append(f"AUC={self._safe_mean(self.auc_window):.3f} < {self.auc_threshold}")
            log.warning(f"触发降级模式: {'; '.join(reasons)}")

    def _check_single(self, window, threshold):
        """检查单个指标是否跌破红线 (需要窗口至少有一半数据)"""
        if len(window) < max(5, self.window_size // 2):
            return False  # 数据不足，不触发
        return self._safe_mean(window) < threshold

    @staticmethod
    def _safe_mean(window):
        """安全计算均值"""
        if not window:
            return 0.0
        return float(np.mean(list(window)))

    def _save_state(self):
        """持久化指标状态"""
        try:
            os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)
            state = {
                "pick_at_10_window": list(self.pick_at_10_window),
                "ban_at_10_window": list(self.ban_at_10_window),
                "auc_window": list(self.auc_window),
                "degraded": self._degraded,
                "updated_at": time.time(),
            }
            with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception:
            log.exception("保存指标状态失败")

    def _load_state(self):
        """从持久化恢复指标状态"""
        if not os.path.exists(METRICS_SAVE_PATH):
            return
        try:
            with open(METRICS_SAVE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.pick_at_10_window = deque(state.get("pick_at_10_window", []), maxlen=self.window_size)
            self.ban_at_10_window = deque(state.get("ban_at_10_window", []), maxlen=self.window_size)
            self.auc_window = deque(state.get("auc_window", []), maxlen=self.window_size)
            self._degraded = state.get("degraded", False)
            log.info(f"从持久化恢复指标状态: 降级={self._degraded}, "
                     f"Pick窗口={len(self.pick_at_10_window)}, "
                     f"Ban窗口={len(self.ban_at_10_window)}, "
                     f"AUC窗口={len(self.auc_window)}")
        except Exception:
            log.exception("加载指标状态失败")


class ExtremeColdStartDetector:
    """
    触发器: 极端冷启动检测。

    当模型遇到完全没见过的全新大版本时，检测数据中 meta 特征的覆盖率。
    如果大量英雄的 meta_presence 接近 0，说明数据不足，触发兜底。
    """

    def __init__(self, meta_coverage_threshold=0.3):
        """
        Args:
            meta_coverage_threshold: meta_presence > 0 的英雄比例低于此值则触发
        """
        self.meta_coverage_threshold = meta_coverage_threshold

    def is_cold_start(self, meta_matrix, champion_start_idx):
        """
        检测是否为极端冷启动。

        Args:
            meta_matrix: np.ndarray, shape (vocab_size, 4), meta 特征矩阵
            champion_start_idx: int, 英雄起始索引

        Returns:
            bool: True 表示冷启动
        """
        if meta_matrix is None:
            return True

        # 检查 meta_presence (index 2) 的覆盖率
        champion_meta = meta_matrix[champion_start_idx:]
        presence_values = champion_meta[:, 2]  # meta_presence
        non_zero_count = np.sum(presence_values > 0.01)
        total_champions = len(presence_values)

        if total_champions == 0:
            return True

        coverage = non_zero_count / total_champions
        if coverage < self.meta_coverage_threshold:
            log.warning(f"极端冷启动: meta 覆盖率={coverage:.2%} < {self.meta_coverage_threshold}")
            return True

        return False

    def is_cold_start_from_store(self, store):
        """从 PredictFeatureStore 检测冷启动"""
        if store is None:
            return True
        return self.is_cold_start(store.meta_matrix, store.champion_start_idx)


# 全局单例
_rolling_monitor = None
_logit_detector = None
_cold_start_detector = None


def get_rolling_monitor():
    """获取全局 RollingMetricsMonitor 单例"""
    global _rolling_monitor
    if _rolling_monitor is None:
        _rolling_monitor = RollingMetricsMonitor()
    return _rolling_monitor


def get_logit_detector():
    """获取全局 LogitCollapseDetector 单例"""
    global _logit_detector
    if _logit_detector is None:
        _logit_detector = LogitCollapseDetector()
    return _logit_detector


def get_cold_start_detector():
    """获取全局 ExtremeColdStartDetector 单例"""
    global _cold_start_detector
    if _cold_start_detector is None:
        _cold_start_detector = ExtremeColdStartDetector()
    return _cold_start_detector