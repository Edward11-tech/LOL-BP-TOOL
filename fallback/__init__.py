"""
fallback — 模型兜底机制模块
============================
当模型遇到极端冷启动、置信度坍塌或滚动指标跌破红线时，
自动降级为 Rule-based 纯规则引擎，确保服务稳定。

组件:
  - data_pipeline: 从 cleaned_data 加载 Meta/选手统计 (带模块级缓存)
  - rule_engine: 规则引擎 (Meta Presence + Player Mastery)
  - triggers: 触发器 A (Logit Collapse) + 触发器 B (滑动窗口)
  - fallback_manager: 兜底管理器，协调所有组件
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

from fallback.fallback_manager import FallbackManager

__all__ = ["FallbackManager"]