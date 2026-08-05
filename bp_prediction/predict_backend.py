"""
predict_backend.py — 胜率预测 & BP Delta 后端封装
===================================================
为 Flask app 提供:
  1. LPL/LCK/LEC 三联赛胜率预测 (基于 bp_prediction)
  2. LPL/LCK/LEC 三联赛 BP Delta 分析

所有模型加载/推理逻辑统一使用 feature_builder.py, 确保与训练时一致。
前端仅负责 UI 展示。
"""
import os
import sys
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import numpy as np
import pandas as pd
from logger_config import get_logger

# ---- 路径 ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "bp_prediction")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from pathlib import Path
FEATURES_DIR = os.path.join(MODEL_DIR, "features")
MODELS_DIR = os.path.join(MODEL_DIR, "models")
PRODUCTION_DIR = os.path.join(MODEL_DIR, "models", "production")
PROJECT_ROOT = BASE_DIR

# 统一特征构建模块 (使用绝对导入避免命名冲突)
from bp_prediction.feature_builder import (
    POSITIONS, MAX_UNKNOWN_PLAYERS_PER_TEAM,
    load_feature_cols, load_feature_stores, load_champion_tags, load_known_champions,
    resolve_team_name, get_team_roster,
    build_single_match_features, build_predraft_features, classify_features,
    extract_tf_features_for_match, resolve_match_info_date,
)

# 统一配置管理 (使用绝对导入避免与 bp_recommendation.config 冲突)
from bp_prediction.config import (
    Mode, get_mode, get_config, is_production_mode, is_training_mode,
    PRODUCTION_DIR, MODELS_DIR, FEATURES_DIR,
)

# 特征监控 (使用绝对导入避免与 bp_recommendation.feature_monitor 冲突)
from bp_prediction.feature_monitor import PredictionFeatureMonitor

# 可解释性模块
from bp_prediction.explainability import (
    compute_shap_values, compute_calibrated_shap_values, _select_representative_models,
    analyze_counters, analyze_synergy, get_champion_stats, explain_feature,
)

# 兜底机制
try:
    from fallback.triggers import get_rolling_monitor
    from fallback.data_pipeline import load_cleaned_meta
    HAS_FALLBACK = True
except ImportError:
    HAS_FALLBACK = False
    get_rolling_monitor = None
    load_cleaned_meta = None

POS_CN = {"top": "上单", "jng": "打野", "mid": "中单", "bot": "ADC", "sup": "辅助"}
POS_WEB = {"top": "top", "jng": "jungle", "mid": "mid", "bot": "bot", "sup": "support"}
SUPPORTED_PREDICT_LEAGUES = frozenset({"LPL", "LCK", "LEC"})

log = get_logger(__name__)


class PredictBackend:
    """胜率预测 & BP Delta 后端 (模式感知)"""

    def __init__(self):
        # 从配置加载并发控制参数 (与 config.py 保持一致)
        shared_cfg, _ = get_config()
        self.MAX_CONCURRENT_INFERENCES = shared_cfg.max_concurrent_inferences
        self.INFERENCE_TIMEOUT_SECONDS = shared_cfg.inference_timeout_seconds
        self.RATE_LIMIT_WINDOW_SECONDS = shared_cfg.rate_limit_window_seconds
        self.RATE_LIMIT_MAX_REQUESTS = shared_cfg.rate_limit_max_requests

        self.models = None
        self.stores = None
        self.champion_tags = None
        self.feature_cols = None
        self.known_champions = []
        self.league_teams = {}
        self._loaded = False
        # 兜底机制
        self.rolling_monitor = None  # 共享的滑动窗口监控器
        self.meta_stats = {}         # 英雄 Meta 统计 (用于规则引擎回退)
        self._display_stats = {}     # 融合统计数据 (用于前端展示)
        self._fallback_count = 0     # 触发兜底次数
        self._normal_count = 0       # 正常推理次数
        # 特征监控器
        self.feature_monitor = None

        # LEC 联赛兜底：AUC=0.6260±0.0780 泛化不稳定，
        # 当模型预测置信度 (max 胜率) 低于阈值时回退到规则引擎
        self.LEC_CONFIDENCE_THRESHOLD = 0.55

        # ====== 多并发应对机制 ======
        # 1. Per-seed 推理锁：CatBoost predict_proba 非线程安全，
        #    拆分为 per-seed 锁后，不同请求可并行预测不同 seed，提升吞吐量
        self._n_seed_locks = 7  # 与 config.n_seeds 一致
        self._seed_locks = [threading.Lock() for _ in range(self._n_seed_locks)]
        # 2. 推理信号量：限制同时执行的推理数量
        self._inference_semaphore = threading.Semaphore(self.MAX_CONCURRENT_INFERENCES)
        # 3. 限流：滑动窗口记录请求时间戳
        self._request_timestamps = deque()
        self._rate_limit_lock = threading.Lock()
        # 4. 推理线程池（资源隔离 + 超时中断）
        self._inference_executor = ThreadPoolExecutor(
            max_workers=self.MAX_CONCURRENT_INFERENCES,
            thread_name_prefix="predict_inference",
        )

        # ====== Pre-Draft 缓存 ======
        # bp_delta 需要计算 pre-draft (无英雄选择) 和 post-draft 两次预测。
        # pre-draft 结果仅依赖联赛/战队/赛制/选边，与英雄选择无关，
        # 同一对战组合在 BP 探索过程中会被反复查询，缓存可将响应时间减半。
        self._predraft_cache = {}  # key -> (pre_prob, pre_imp, timestamp)
        self._predraft_cache_lock = threading.Lock()
        self.PREDRAFT_CACHE_TTL = 300        # 缓存有效期 5 分钟
        self.PREDRAFT_CACHE_MAX_SIZE = 200   # 最大缓存条目数

    # ---- 加载 ----

    def load(self):
        """加载模型和数据 (耗时操作, 启动时调用一次, 模式感知)"""
        try:
            shared, mode_cfg = get_config()
            current_mode = get_mode()
            log.info(f"当前运行模式: {current_mode.value.upper()}")

            # 0. 清除 pre-draft 缓存 (模型/数据更新后旧缓存失效)
            with self._predraft_cache_lock:
                if self._predraft_cache:
                    log.info(f"清除 pre-draft 缓存 ({len(self._predraft_cache)} 条)")
                    self._predraft_cache.clear()

            # 1. 模型 (模式感知加载)
            self.models = self._load_models()
            if not self.models:
                log.warning("未找到训练好的模型")
                return {"success": False, "message": "未找到训练好的模型"}

            # 2. 特征列名 (模式感知: 生产用 production/, 训练用 fold_0/)
            use_production = is_production_mode()
            self.feature_cols = load_feature_cols(use_production=use_production)

            # 3. 数据存储 (统一使用 feature_builder)
            self.stores = load_feature_stores()

            # 4. 英雄标签
            self.champion_tags = load_champion_tags()

            # 5. 已知英雄列表
            self.known_champions = load_known_champions()

            # 6. 联赛-战队映射
            self.league_teams = self._load_league_teams()

            # 7. 特征监控器初始化 (仅生产模式启用)
            if is_production_mode() and getattr(mode_cfg, 'enable_feature_monitor', True):
                self.feature_monitor = PredictionFeatureMonitor(
                    feature_cols=self.feature_cols,
                    baseline_dir=FEATURES_DIR,
                )
                log.info(f"特征监控器已初始化 (生产模式): {len(self.feature_cols)} features")
            else:
                self.feature_monitor = None
                log.info("特征监控器已跳过 (训练模式)")

            # 8. 兜底机制初始化 (仅生产模式启用)
            if is_production_mode() and HAS_FALLBACK:
                try:
                    self.rolling_monitor = get_rolling_monitor()
                    self.meta_stats = load_cleaned_meta()
                    log.info(f"兜底机制已接入 (生产模式): {len(self.meta_stats)} 英雄 Meta 统计")
                except Exception as e:
                    log.warning(f"兜底机制初始化失败 (非致命): {e}")
            else:
                log.info("兜底机制已跳过 (训练模式)")

            # 9. 加载前端展示用融合统计数据
            self._load_merged_display_stats()

            self._loaded = True
            model_type = "Production" if "production" in self.models else "OOT 5-Fold"
            log.info(f"预测模型加载完成 [{current_mode.value}]: {model_type}, "
                     f"{len(self.feature_cols)} features, {len(self.known_champions)} champions")
            return {"success": True, "message": f"加载完成 [{current_mode.value}]: {model_type} 模型"}
        except Exception as e:
            log.error(f"模型加载失败: {e}")
            return {"success": False, "message": str(e)}

    def is_loaded(self):
        return self._loaded

    def _load_merged_display_stats(self):
        """加载融合后的英雄统计数据（排位先验 + 职业观测，c=5）用于前端展示"""
        merged_path = os.path.join(BASE_DIR, "cleaned_data", "merged_champion_stats.csv")
        self._display_stats = {}
        try:
            if os.path.exists(merged_path):
                df = pd.read_csv(merged_path)
                for _, row in df.iterrows():
                    champ = row["champion"]
                    self._display_stats[champ] = {
                        "win_rate": float(row.get("win_rate", 0.5) or 0.5),
                        "pick_rate": float(row.get("pick_rate", 0) or 0),
                        "ban_rate": float(row.get("ban_rate", 0) or 0),
                        "presence_rate": float(row.get("presence_rate", 0) or 0),
                    }
                log.info(f"前端展示融合统计加载完成: {len(self._display_stats)} 英雄")
            else:
                log.warning(f"融合统计文件不存在: {merged_path}")
        except Exception as e:
            log.warning(f"加载融合统计失败: {e}")
            self._display_stats = {}

    # ====== 限流机制 ======

    def _check_rate_limit(self) -> bool:
        """滑动窗口限流检查，返回 True 表示允许请求"""
        now = time.time()
        with self._rate_limit_lock:
            cutoff = now - self.RATE_LIMIT_WINDOW_SECONDS
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()
            if len(self._request_timestamps) >= self.RATE_LIMIT_MAX_REQUESTS:
                return False
            self._request_timestamps.append(now)
            return True

    def get_concurrency_status(self) -> dict:
        """获取当前并发状态（用于健康检查）"""
        return {
            "max_concurrent_inferences": self.MAX_CONCURRENT_INFERENCES,
            "available_inference_slots": self._inference_semaphore._value,
            "rate_limit_window_s": self.RATE_LIMIT_WINDOW_SECONDS,
            "rate_limit_max_requests": self.RATE_LIMIT_MAX_REQUESTS,
            "current_window_requests": len(self._request_timestamps),
            "inference_timeout_s": self.INFERENCE_TIMEOUT_SECONDS,
        }

    def get_status(self):
        return {
            "loaded": self._loaded,
            "model_type": "Production" if self.models and "production" in self.models else "OOT",
            "folds": len(self.models) if self.models else 0,
            "features": len(self.feature_cols) if self.feature_cols else 0,
            "champions": len(self.known_champions),
        }

    def get_champions(self):
        """返回英雄列表 (前端下拉用), 包含中文名和登场率（优先使用融合展示数据）"""
        vocab_path = os.path.join(BASE_DIR, "cleaned_data", "champion_vocabulary.json")
        pos_path = os.path.join(BASE_DIR, "cleaned_data", "champion_position_mapping.json")
        position_mapping = {}
        if os.path.exists(pos_path):
            with open(pos_path, "r", encoding="utf-8") as f:
                position_mapping = json.load(f)

        result = []
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                vocab = json.load(f)
            cs = int(vocab.get("champion_start_idx", 7))
            if isinstance(vocab, dict) and "champions" in vocab:
                for c in vocab["champions"]:
                    if c.get("idx", 0) < cs or "name" not in c:
                        continue
                    item = {"name": c["name"]}
                    if "aliases" in c and "zh" in c["aliases"]:
                        item["cn_name"] = c["aliases"]["zh"]
                    if "meta_presence" in c:
                        item["meta_presence"] = c["meta_presence"]
                    display = self._display_stats.get(c["name"])
                    if display:
                        item["meta_presence"] = round(float(display["presence_rate"]), 4)
                        item["meta_win_rate"] = round(float(display["win_rate"]), 4)
                    else:
                        meta = self.meta_stats.get(c["name"]) or {}
                        if "meta_presence" in meta and meta["meta_presence"] is not None:
                            item["meta_presence"] = round(float(meta["meta_presence"]), 4)
                        if "meta_win_rate" in meta and meta["meta_win_rate"] is not None:
                            item["meta_win_rate"] = round(float(meta["meta_win_rate"]), 4)
                    # 分路先验（供前端分路筛选）
                    pos_probs = {}
                    if c["name"] in position_mapping:
                        for entry in position_mapping[c["name"]]:
                            pos_name = entry.get("position", "")
                            prob = float(entry.get("probability", 0) or 0)
                            if prob > 0.05:
                                pos_probs[pos_name] = round(prob, 3)
                    if pos_probs:
                        item["positions"] = pos_probs
                    result.append(item)
        if not result:
            result = [{"name": c} for c in sorted(self.known_champions)]
        return result

    def get_teams(self, league="LCK"):
        """返回指定联赛的战队列表"""
        if self.league_teams and league in self.league_teams:
            return self.league_teams[league]
        if not self.stores or "team_profile" not in self.stores:
            return []
        tp = self.stores["team_profile"]
        if "league" in tp.columns:
            teams = tp[tp["league"] == league]["team"].unique().tolist()
        else:
            teams = tp["team"].unique().tolist()
        return sorted(teams)

    def get_team_players(self, team_name):
        """返回战队选手列表 (从 active_rosters.csv), 支持同位置替补"""
        resolved = resolve_team_name(team_name, set(self._get_all_teams()))
        roster = get_team_roster(resolved)
        result = []
        for p in roster:
            role = p["role"]
            web_role = POS_WEB.get(role, role)
            result.append({
                "player_id": p["player_name"],
                "player_name": p["player_name"],
                "role": web_role,
                "role_cn": POS_CN.get(role, role),
            })
        # 添加 unknown 选项到每个位置
        for web_role in ["top", "jungle", "mid", "bot", "support"]:
            has_unknown = any(p["player_name"] == "unknown" for p in result if p["role"] == web_role)
            if not has_unknown:
                internal_role = {v: k for k, v in POS_WEB.items()}.get(web_role, web_role)
                result.append({
                    "player_id": "unknown",
                    "player_name": "unknown",
                    "role": web_role,
                    "role_cn": POS_CN.get(internal_role, internal_role),
                })
        return result

    # ---- 胜率预测 ----

    def predict(self, request):
        """胜率预测 (公开接口，含限流 + 超时机制)"""
        # 1. 限流检查
        if not self._check_rate_limit():
            return {"error": "请求过于频繁，请稍后再试"}

        if not self._loaded:
            return {"error": "模型未加载"}

        # 2. 推理在独立线程执行（资源隔离 + 超时机制）
        try:
            future = self._inference_executor.submit(self._predict_impl, request)
            return future.result(timeout=self.INFERENCE_TIMEOUT_SECONDS)
        except FutureTimeout:
            log.error(f"预测推理超时 (>{self.INFERENCE_TIMEOUT_SECONDS}s)")
            return {"error": "推理超时，请重试"}
        except Exception as e:
            log.error(f"预测调度失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _predict_impl(self, request):
        """胜率预测实际实现（在信号量保护下执行）"""
        with self._inference_semaphore:
            try:
                match_info = self._build_match_info(request)

                # 提取 TF 特征
                tf_features = extract_tf_features_for_match(match_info)

                # 构建完整特征
                features_df, unknown_info = build_single_match_features(
                    match_info, self.stores, self.champion_tags,
                    feature_cols=self.feature_cols, tf_features=tf_features
                )
                if features_df is None:
                    return {"error": "特征构建失败"}

                # 预测
                blue_prob, fold_details, feature_importance = self._predict(features_df)

                # SHAP 可解释性计算（代表子集 + 基线校准，仅5个模型，性能提升~5倍）
                shap_data = {"positive": [], "negative": [], "waterfall_data": []}
                try:
                    if self.feature_cols:
                        ordered_df = features_df[self.feature_cols].copy()
                        for c in self.feature_cols:
                            if c not in ordered_df.columns:
                                ordered_df[c] = 0.0
                        ordered_df = ordered_df[self.feature_cols]
                        X_infer = ordered_df.values.astype(np.float32)
                        X_infer = np.nan_to_num(X_infer, nan=0.0, posinf=0.0, neginf=0.0)

                        rep_models = _select_representative_models(self.models, max_rep=5)
                        if len(rep_models) == 1:
                            shap_data = compute_shap_values(rep_models[0], X_infer, self.feature_cols)
                        elif len(rep_models) > 1:
                            shap_data = compute_calibrated_shap_values(
                                rep_models, X_infer, self.feature_cols,
                                target_prob=blue_prob
                            )
                            log.info(f"校准SHAP计算完成: {shap_data.get('n_models_used', 0)}个代表模型, "
                                     f"Δlogit={shap_data.get('calibration_delta_logit', 0):.4f}")
                except Exception as se:
                    log.warning(f"SHAP计算失败（非关键）: {se}")
                    shap_data = {"positive": [], "negative": [], "waterfall_data": [], "error": str(se)}

                # 模型输出语义：blue_* 列固定为 "BP 蓝方"（先Pick方），
                # is_blue_map_side 特征已经告诉模型 "BP 蓝方是否在地图蓝色方"，
                # 模型已经学会区分地图方差异，所以 blue_prob 直接是 "BP 蓝方胜率"，
                # 后续 _format_predict_result 会把概率和战队名重新对齐到 "用户视角的蓝红方"。

                # LEC 联赛置信度兜底：AUC 不稳定，低置信度时回退规则引擎
                league = match_info.get("league", "LCK")
                model_confidence = max(blue_prob, 1.0 - blue_prob)
                if league == "LEC" and model_confidence < self.LEC_CONFIDENCE_THRESHOLD:
                    log.warning(
                        f"LEC 联赛置信度不足 (confidence={model_confidence:.4f} < "
                        f"{self.LEC_CONFIDENCE_THRESHOLD}), 使用规则引擎回退"
                    )
                    self._fallback_count += 1
                    return self._rule_based_predict(request)

                # 检查兜底触发条件
                if self.rolling_monitor is not None and self.rolling_monitor.is_degraded():
                    log.warning("滑动窗口指标降级，使用规则引擎回退")
                    self._fallback_count += 1
                    return self._rule_based_predict(request)
                else:
                    self._normal_count += 1
                    return self._format_predict_result(
                        match_info, blue_prob, fold_details, feature_importance,
                        unknown_info, shap_data
                    )
            except Exception as e:
                log.error(f"预测失败: {e}", exc_info=True)
                # 异常时尝试规则引擎回退
                if self.meta_stats:
                    log.info("使用规则引擎回退预测")
                    self._fallback_count += 1
                    return self._rule_based_predict(request)
                return {"error": str(e)}

    def _predict(self, features_df):
        """模型预测 (线程安全：CatBoost 推理加锁 + 特征监控)

        统一推理入口，合并自原有两个重复的 _predict 方法。
        生产模式启用特征监控，训练模式跳过以加速验证。
        """
        # 特征监控：推理前校验特征范围与完整性 (仅生产模式)
        if self.feature_monitor is not None and is_production_mode():
            range_result = self.feature_monitor.validate_feature_ranges(features_df)
            if not range_result.is_valid:
                log.warning(f"特征范围校验失败: {range_result.violations[:3]}")
            integrity_result = self.feature_monitor.validate_integrity(
                features_df, expected_n_features=len(self.feature_cols) if self.feature_cols else None
            )
            if not integrity_result.is_valid:
                log.warning(f"特征完整性校验失败: {integrity_result.violations[:3]}")

        # 推理特征日志 (供周度 PSI 漂移分析，失败不影响主业务)
        try:
            from common.inference_feature_logger import log_prediction_features
            import uuid as _uuid
            league_val = ""
            if "league" in features_df.columns:
                league_val = str(features_df["league"].iloc[0])
            log_prediction_features(
                features_df=features_df,
                request_id=_uuid.uuid4().hex[:12],
                league=league_val,
            )
        except Exception:
            pass  # 埋点失败不能影响主业务

        all_preds = []
        fold_details = {}
        feature_importances = []

        # 1. 严格对齐特征列顺序 (防止 feature_builder 输出列顺序与训练时不一致)
        if self.feature_cols:
            missing_cols = [c for c in self.feature_cols if c not in features_df.columns]
            for c in missing_cols:
                features_df[c] = 0.0
            ordered_df = features_df[self.feature_cols]
        else:
            ordered_df = features_df

        # 2. NaN 填充 (与 train_production.py 保持完全一致)
        X_infer = ordered_df.values.astype(np.float32)
        X_infer = np.nan_to_num(X_infer, nan=0.0, posinf=0.0, neginf=0.0)
        
        # === 断言：确保预测特征维度与模型期望一致 ===
        if self.feature_cols:
            assert X_infer.shape[1] == len(self.feature_cols), \
                f"预测特征维度不匹配! X_infer: {X_infer.shape[1]}, feature_cols: {len(self.feature_cols)}"

        # 3. CatBoost predict_proba 非线程安全，使用 per-seed 锁保护
        #    不同请求可并行预测不同 seed，提升并发吞吐量
        global_seed_idx = 0
        for fold_key, fold_models in sorted(self.models.items()):
            fold_preds = []
            for seed_idx, model in enumerate(fold_models):
                # 按 global seed index 取锁，确保同一 seed 模型不会并发推理
                lock_idx = global_seed_idx % self._n_seed_locks
                with self._seed_locks[lock_idx]:
                    pred = model.predict_proba(X_infer)[0, 1]
                    importances = model.get_feature_importance()
                fold_preds.append(float(pred))
                feature_importances.append(importances)
                global_seed_idx += 1

            fold_mean = float(np.mean(fold_preds))
            fold_details[fold_key] = {
                "mean_prob": fold_mean,
                "seed_preds": fold_preds,
            }
            all_preds.append(fold_mean)

        final_prob = float(np.mean(all_preds))
        avg_importance = np.mean(feature_importances, axis=0) if feature_importances else np.zeros(len(self.feature_cols) if self.feature_cols else features_df.shape[1])

        return final_prob, fold_details, avg_importance
    # ---- BP Delta ----

    def bp_delta(self, request):
        """BP Delta 分析 (支持 LPL/LCK/LEC)

        request: 同 predict 的格式
        """
        # 1. 限流检查
        if not self._check_rate_limit():
            return {"error": "请求过于频繁，请稍后再试"}

        if not self._loaded:
            return {"error": "模型未加载"}

        # 2. 推理在独立线程执行（资源隔离 + 超时机制）
        try:
            future = self._inference_executor.submit(self._bp_delta_impl, request)
            return future.result(timeout=self.INFERENCE_TIMEOUT_SECONDS)
        except FutureTimeout:
            log.error(f"BP Delta 推理超时 (>{self.INFERENCE_TIMEOUT_SECONDS}s)")
            return {"error": "推理超时，请重试"}
        except Exception as e:
            log.error(f"BP Delta 调度失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _bp_delta_impl(self, request):
        """BP Delta 实际实现（在信号量保护下执行）"""
        with self._inference_semaphore:
            try:
                match_info = self._build_match_info(request)

                # 提取 TF 特征 (仅用于 Post-Draft)
                tf_features = extract_tf_features_for_match(match_info)

                # Post-Draft: 完整特征
                post_features, unknown_info = build_single_match_features(
                    match_info, self.stores, self.champion_tags,
                    feature_cols=self.feature_cols, tf_features=tf_features
                )
                if post_features is None:
                    return {"error": "特征构建失败"}

                # Pre-Draft: 使用缓存 (pre-draft 不依赖英雄选择)
                pre_prob, pre_imp = self._get_predraft_prediction(match_info)

                # Post-Draft 预测
                post_prob, _, post_imp = self._predict(post_features)

                # SHAP 可解释性计算（代表子集 + 基线校准，仅5个模型）
                shap_data = {"positive": [], "negative": [], "waterfall_data": []}
                try:
                    if self.feature_cols:
                        ordered_df = post_features[self.feature_cols].copy()
                        for c in self.feature_cols:
                            if c not in ordered_df.columns:
                                ordered_df[c] = 0.0
                        ordered_df = ordered_df[self.feature_cols]
                        X_infer = ordered_df.values.astype(np.float32)
                        X_infer = np.nan_to_num(X_infer, nan=0.0, posinf=0.0, neginf=0.0)

                        rep_models = _select_representative_models(self.models, max_rep=5)
                        if len(rep_models) == 1:
                            shap_data = compute_shap_values(rep_models[0], X_infer, self.feature_cols)
                        elif len(rep_models) > 1:
                            shap_data = compute_calibrated_shap_values(
                                rep_models, X_infer, self.feature_cols,
                                target_prob=post_prob
                            )
                except Exception as se:
                    log.warning(f"Delta SHAP计算失败（非关键）: {se}")
                    shap_data = {"positive": [], "negative": [], "waterfall_data": [], "error": str(se)}

                # 模型输出语义：post_prob/pre_prob 都是 "BP 蓝方胜率"，
                # is_blue_map_side 特征已经告诉模型地图阵营差异，不需要反转。
                # delta = post - pre，正值表示 "BP 对 BP 蓝方有利"，
                # _format_delta_result 会把概率和战队名重新对齐到 "用户视角的蓝红方"。
                delta = post_prob - pre_prob

                draft_cols, hard_cols = classify_features(self.feature_cols)

                result = self._format_delta_result(
                    match_info, pre_prob, post_prob, delta,
                    draft_cols, hard_cols, post_imp, unknown_info, shap_data
                )
                return result
            except Exception as e:
                log.error(f"BP Delta 计算失败: {e}", exc_info=True)
                return {"error": str(e)}

    def _get_predraft_prediction(self, match_info):
        """获取 pre-draft 预测结果（带缓存）。

        Pre-Draft 表示 BP 前的纸面实力基线，仅依赖联赛/战队/赛制/选边，
        与英雄选择无关。同一对战组合在 BP 探索过程中会被反复查询，
        缓存可将 bp_delta 响应时间减半。

        为确保 pre-draft 不受英雄选择影响:
          - 使用空英雄列表构建特征 (选手统计回退到默认值)
          - tf_features=None (不使用英雄组合 TF 特征)
        """
        # 缓存键: pre-draft 相关字段（含局数与 PIT 日期）
        cache_key = (
            match_info.get("league", "LCK"),
            match_info.get("blue_team", ""),
            match_info.get("red_team", ""),
            match_info.get("is_playoff", False),
            match_info.get("is_blue_map_side", True),
            int(match_info.get("game_num", 1)),
            match_info.get("date", ""),
        )

        # 1. 检查缓存
        now = time.time()
        with self._predraft_cache_lock:
            cached = self._predraft_cache.get(cache_key)
            if cached is not None:
                pre_prob, pre_imp, ts = cached
                if now - ts < self.PREDRAFT_CACHE_TTL:
                    log.debug(f"Pre-Draft 缓存命中: {cache_key[0]} {cache_key[1]} vs {cache_key[2]}")
                    return pre_prob, pre_imp
                else:
                    del self._predraft_cache[cache_key]

        # 2. 缓存未命中，计算 pre-draft
        #    使用空英雄确保 pre-draft 不依赖英雄选择
        predraft_match_info = dict(match_info)
        predraft_match_info["blue_champions"] = [""] * 5
        predraft_match_info["red_champions"] = [""] * 5

        pre_features = build_predraft_features(
            predraft_match_info, self.stores, self.champion_tags,
            feature_cols=self.feature_cols, tf_features=None
        )
        pre_prob, _, pre_imp = self._predict(pre_features)

        # 3. 写入缓存 (LRU 淘汰)
        with self._predraft_cache_lock:
            if len(self._predraft_cache) >= self.PREDRAFT_CACHE_MAX_SIZE:
                oldest_key = min(self._predraft_cache, key=lambda k: self._predraft_cache[k][2])
                del self._predraft_cache[oldest_key]
            self._predraft_cache[cache_key] = (pre_prob, pre_imp, now)
            log.debug(f"Pre-Draft 缓存写入: {cache_key[0]} {cache_key[1]} vs {cache_key[2]}")

        return pre_prob, pre_imp

    # ---- 内部方法 ----

    @staticmethod
    def _normalize_league(league, default="LCK"):
        value = (league or default).strip().upper()
        if value not in SUPPORTED_PREDICT_LEAGUES:
            raise ValueError(f"不支持的联赛: {league}，仅支持 LPL / LCK / LEC")
        return value

    def _parse_required_game_num(self, request):
        """解析并校验系列赛局数 (1-5)，推理必填。"""
        raw = request.get("game_num")
        if raw is None:
            raise ValueError("缺少必填字段: game_num (系列赛局数 1-5)")
        try:
            n = int(raw)
        except (TypeError, ValueError):
            raise ValueError("game_num 必须为 1-5 的整数")
        if not (1 <= n <= 5):
            raise ValueError("game_num 必须为 1-5 的整数")
        return n

    def _build_match_info(self, request):
        """从前端请求构建 match_info (feature_builder 格式)

        红蓝方语义约定（与训练数据一致）：
          - 前端传入的 blue_team / blue_champions / blue_players 永远是 "首选方"（BP 蓝方），
            red_* 永远是 "次选方"（BP 红方），不需要任何交换。
          - first_pick 字段表示 "首选方位于地图哪一侧"：
              "blue" -> 首选方在地图蓝色方 (first_pick_map_side=1)
              "red"  -> 首选方在地图红色方 (first_pick_map_side=0)
          - is_blue_map_side 特征记录 "首选方是否在地图蓝色方"，作为地图阵营特征输入模型，
            模型自己学会地图方差异，不涉及对换。
          - 模型输出 blue_prob 永远是 "首选方（BP 蓝方）胜率"，与用户查询语义一致，无需反转。
        """
        league = self._normalize_league(request.get("league", "LCK"))
        is_playoff = request.get("is_playoff", False)
        first_pick = request.get("first_pick", "red")
        game_num = self._parse_required_game_num(request)
        # is_blue_map_side = 首选方是否在地图蓝色方
        is_blue_map_side = 1 if first_pick == "blue" else 0

        # 解析战队名（前端 blue=首选方, red=次选方，直接使用，无需交换）
        all_teams = self._get_all_teams()
        blue_team = resolve_team_name(request.get("blue_team", ""), all_teams)
        red_team = resolve_team_name(request.get("red_team", ""), all_teams)

        # 英雄 / 选手（直接使用，无需交换）
        blue_champs_raw = request.get("blue_champions", {})
        red_champs_raw = request.get("red_champions", {})
        blue_players_raw = request.get("blue_players", {})
        red_players_raw = request.get("red_players", {})

        blue_champions = []
        red_champions = []
        blue_player_map = {}
        red_player_map = {}

        for pos in POSITIONS:
            web_pos = POS_WEB[pos]
            bc = blue_champs_raw.get(web_pos, "") or blue_champs_raw.get(pos, "")
            rc = red_champs_raw.get(web_pos, "") or red_champs_raw.get(pos, "")
            blue_champions.append(bc)
            red_champions.append(rc)

        for pos in POSITIONS:
            web_pos = POS_WEB[pos]
            bp = blue_players_raw.get(web_pos, "") or blue_players_raw.get(pos, "")
            rp = red_players_raw.get(web_pos, "") or red_players_raw.get(pos, "")
            if bp:
                blue_player_map[pos] = bp
            if rp:
                red_player_map[pos] = rp

        match_info = {
            "league": league,
            "is_playoff": is_playoff,
            # is_blue_map_side = 首选方是否在地图蓝色方（特征字段，非交换标志）
            "is_blue_map_side": is_blue_map_side,
            "game_num": game_num,
            "blue_team": blue_team,
            "red_team": red_team,
            "blue_champions": blue_champions,
            "red_champions": red_champions,
            "mode": "full" if (blue_team or red_team) else "draft",
        }
        if request.get("date"):
            match_info["date"] = request["date"]
        resolve_match_info_date(match_info)

        # 添加选手信息
        for pos in POSITIONS:
            if pos in blue_player_map:
                match_info[f"blue_{pos}_player_id"] = blue_player_map[pos]
            if pos in red_player_map:
                match_info[f"red_{pos}_player_id"] = red_player_map[pos]

        # 处理 unknown 选手
        for side in ["blue", "red"]:
            unknown_positions = []
            for pos in POSITIONS:
                player_id = match_info.get(f"{side}_{pos}_player_id", "")
                if player_id.lower() in ("unknown", "unk", "?", "未知", "新秀", ""):
                    if player_id.lower() in ("unknown", "unk", "?", "未知", "新秀"):
                        unknown_positions.append(pos)
                    match_info[f"{side}_{pos}_player_id"] = ""
            match_info[f"{side}_unknown_positions"] = unknown_positions

        return match_info

    def _build_features(self, match_info):
        """构建特征向量 (使用统一 feature_builder)"""
        # 尝试提取 TF 特征
        tf_features = extract_tf_features_for_match(match_info)

        return build_single_match_features(
            match_info, self.stores, self.champion_tags,
            feature_cols=self.feature_cols, tf_features=tf_features
        )

    # ---- 规则引擎回退预测 ----

    def _rule_based_predict(self, request):
        """
        基于 Meta 统计的规则引擎回退预测。

        当滑动窗口指标 (AUC < 0.53) 或模型推理异常时使用。
        公式: 蓝方胜率 = 蓝方英雄平均 Meta 胜率 / (蓝方 + 红方平均 Meta 胜率)

        返回格式与 _predict + _format_predict_result 一致。
        """
        blue_team = request.get("blue_team", "蓝方") or "蓝方"
        red_team = request.get("red_team", "红方") or "红方"
        blue_champs = request.get("blue_champions", {})
        red_champs = request.get("red_champions", {})

        # 计算双方英雄的 Meta 胜率
        def _team_avg_meta_wr(champs_dict):
            """计算一队英雄的平均 Meta 胜率"""
            wrs = []
            for pos in POSITIONS:
                web_pos = POS_WEB[pos]
                champ_name = champs_dict.get(web_pos, "") or champs_dict.get(pos, "")
                if champ_name:
                    # 尝试多种名称匹配
                    meta = self.meta_stats.get(champ_name) or self.meta_stats.get(
                        champ_name.lower(), {}) or self.meta_stats.get(
                        champ_name.title(), {})
                    if meta:
                        wrs.append(meta.get("meta_win_rate", 0.5))
                    else:
                        wrs.append(0.5)  # 未知英雄默认 50%
            return np.mean(wrs) if wrs else 0.5

        blue_avg_wr = _team_avg_meta_wr(blue_champs)
        red_avg_wr = _team_avg_meta_wr(red_champs)

        # 转换为蓝方胜率 (使用 logistic 校准)
        total = blue_avg_wr + red_avg_wr
        if total > 0:
            blue_prob = blue_avg_wr / total
        else:
            blue_prob = 0.5

        # 限制在合理范围
        blue_prob = max(0.25, min(0.75, blue_prob))
        red_prob = 1.0 - blue_prob

        log.info(f"规则引擎预测: {blue_team} vs {red_team}, "
                 f"蓝方 Meta WR={blue_avg_wr:.3f}, 红方 Meta WR={red_avg_wr:.3f}, "
                 f"蓝方胜率={blue_prob:.3f}")

        blue_champs_dict = {}
        red_champs_dict = {}
        pos_for_explain = ["top", "jungle", "mid", "bot", "support"]
        pos_web_order = ["top", "jungle", "mid", "bot", "support"]
        for i, pos in enumerate(pos_web_order):
            bc = blue_champs.get(pos, "") or blue_champs.get(POSITIONS[i] if i < len(POSITIONS) else "", "")
            rc = red_champs.get(pos, "") or red_champs.get(POSITIONS[i] if i < len(POSITIONS) else "", "")
            if bc:
                blue_champs_dict[pos] = bc
            if rc:
                red_champs_dict[pos] = rc

        counter_data = analyze_counters(blue_champs_dict, red_champs_dict)
        synergy_data = analyze_synergy(blue_champs_dict, red_champs_dict)
        champion_stats = get_champion_stats(blue_champs_dict, red_champs_dict, self.meta_stats)

        blue_counter_adv = sum(1 for c in counter_data if c["counter_side"] == "blue")
        red_counter_adv = sum(1 for c in counter_data if c["counter_side"] == "red")
        if blue_counter_adv > red_counter_adv:
            counter_verdict = f"蓝方阵容counter占优 ({blue_counter_adv}路优势)"
            counter_verdict_en = f"Blue Side has counter advantage ({blue_counter_adv} lanes)"
            counter_verdict_side = "blue"
        elif red_counter_adv > blue_counter_adv:
            counter_verdict = f"红方阵容counter占优 ({red_counter_adv}路优势)"
            counter_verdict_en = f"Red Side has counter advantage ({red_counter_adv} lanes)"
            counter_verdict_side = "red"
        else:
            counter_verdict = "双方阵容counter势均力敌"
            counter_verdict_en = "Even counter matchup on both sides"
            counter_verdict_side = "even"

        verdict_cn = "势均力敌" if abs(blue_prob - red_prob) <= 0.10 else (f"{blue_team} 优势" if blue_prob > red_prob else f"{red_team} 优势")
        verdict_en = "Even matchup" if abs(blue_prob - red_prob) <= 0.10 else (f"{blue_team} favored" if blue_prob > red_prob else f"{red_team} favored")

        return {
            "blue_team": blue_team,
            "red_team": red_team,
            "blue_win_prob": round(blue_prob, 4),
            "red_win_prob": round(red_prob, 4),
            "blue_score": round(blue_prob * 100, 1),
            "red_score": round(red_prob * 100, 1),
            "predicted_winner": blue_team if blue_prob > red_prob else red_team,
            "verdict": verdict_cn,
            "verdict_en": verdict_en,
            "verdict_side": "even" if abs(blue_prob - red_prob) <= 0.10 else ("blue" if blue_prob > red_prob else "red"),
            "counter_verdict": counter_verdict,
            "counter_verdict_en": counter_verdict_en,
            "counter_verdict_side": counter_verdict_side,
            "fold_std": 0,
            "fold_details": {},
            "shap_values": {"positive": [], "negative": [], "waterfall_data": [], "fallback": True},
            "counters": counter_data,
            "synergies": synergy_data,
            "champion_stats": champion_stats,
            "unknown_players": {},
            "fallback": True,
        }

    def record_match_outcome(self, blue_team, red_team, blue_win_prob, actual_blue_win):
        """
        记录比赛实际结果，用于更新 AUC 滑动窗口。

        应在比赛结束后调用，用于评估模型预测质量。

        Args:
            blue_team: str, 蓝方战队名
            red_team: str, 红方战队名
            blue_win_prob: float, 模型预测的蓝方胜率
            actual_blue_win: bool, 蓝方是否实际获胜
        """
        if self.rolling_monitor is None:
            return

        # 计算本次预测的 AUC 贡献 (简化: 使用二分类准确度作为代理)
        # 真正的 AUC 需要累积多个样本，这里用单样本的 squared error 的补数
        predicted_correct = (blue_win_prob > 0.5) == actual_blue_win
        single_auc_proxy = 1.0 if predicted_correct else 0.0

        self.rolling_monitor.record_pick_result(auc=single_auc_proxy)
        log.info(f"比赛结果已记录: {blue_team} vs {red_team}, "
                 f"预测={blue_win_prob:.3f}, 实际蓝方胜={actual_blue_win}, "
                 f"正确={predicted_correct}")

    def get_fallback_status(self):
        """获取预测模型的兜底状态"""
        if self.rolling_monitor is None:
            return {"enabled": False, "message": "兜底机制未启用"}
        metrics = self.rolling_monitor.get_metrics()
        return {
            "enabled": True,
            "is_degraded": self.rolling_monitor.is_degraded(),
            "rolling_metrics": metrics,
            "stats": {
                "normal_predictions": self._normal_count,
                "fallback_predictions": self._fallback_count,
            },
        }

    def _get_all_teams(self):
        """获取所有战队名"""
        if self.stores and "team_profile" in self.stores:
            return set(self.stores["team_profile"]["team"].unique())
        return set()

    def _format_predict_result(self, match_info, blue_prob, fold_details, feature_importance,
                               unknown_info, shap_data=None):
        """格式化预测结果（含可解释性信息）

        语义约定：
          - match_info.blue_team / blue_champions = 首选方（BP 蓝方）
          - match_info.red_team / red_champions = 次选方（BP 红方）
          - 模型输出 blue_prob = 首选方（BP 蓝方）胜率，直接对应用户查询语义，无需反转。
        """
        blue_team = match_info.get("blue_team", "蓝方") or "蓝方"
        red_team = match_info.get("red_team", "红方") or "红方"
        red_prob = 1 - blue_prob

        blue_champs_list = match_info.get("blue_champions", [])
        red_champs_list = match_info.get("red_champions", [])
        pos_order = ["top", "jng", "mid", "bot", "sup"]
        pos_for_explain = ["top", "jungle", "mid", "bot", "support"]
        blue_champs_dict = {}
        red_champs_dict = {}
        for i, pos in enumerate(pos_order):
            bc = blue_champs_list[i] if i < len(blue_champs_list) else ""
            rc = red_champs_list[i] if i < len(red_champs_list) else ""
            if bc:
                blue_champs_dict[pos_for_explain[i]] = bc
            if rc:
                red_champs_dict[pos_for_explain[i]] = rc

        shap_result = shap_data or {"positive": [], "negative": [], "waterfall_data": []}

        counter_data = analyze_counters(blue_champs_dict, red_champs_dict)
        synergy_data = analyze_synergy(blue_champs_dict, red_champs_dict)
        champion_stats = get_champion_stats(blue_champs_dict, red_champs_dict, self.meta_stats)

        fold_means = [fd["mean_prob"] for fd in fold_details.values()]
        fold_std = float(np.std(fold_means)) if fold_means else 0

        win_diff = abs(blue_prob - red_prob)
        if win_diff < 0.10:
            verdict = "势均力敌"
            verdict_en = "Even matchup"
            verdict_side = "even"
        elif blue_prob > red_prob:
            verdict = f"{blue_team} 优势"
            verdict_en = f"{blue_team} favored"
            verdict_side = "blue"
        else:
            verdict = f"{red_team} 优势"
            verdict_en = f"{red_team} favored"
            verdict_side = "red"

        blue_counter_adv = sum(1 for c in counter_data if c["counter_side"] == "blue")
        red_counter_adv = sum(1 for c in counter_data if c["counter_side"] == "red")
        if blue_counter_adv > red_counter_adv:
            counter_verdict = f"蓝方阵容counter占优 ({blue_counter_adv}路优势)"
            counter_verdict_en = f"Blue Side has counter advantage ({blue_counter_adv} lanes)"
            counter_verdict_side = "blue"
        elif red_counter_adv > blue_counter_adv:
            counter_verdict = f"红方阵容counter占优 ({red_counter_adv}路优势)"
            counter_verdict_en = f"Red Side has counter advantage ({red_counter_adv} lanes)"
            counter_verdict_side = "red"
        else:
            counter_verdict = "双方阵容counter势均力敌"
            counter_verdict_en = "Even counter matchup on both sides"
            counter_verdict_side = "even"

        return {
            "blue_team": blue_team,
            "red_team": red_team,
            "blue_win_prob": round(blue_prob, 4),
            "red_win_prob": round(red_prob, 4),
            "blue_score": round(blue_prob * 100, 1),
            "red_score": round(red_prob * 100, 1),
            "predicted_winner": blue_team if blue_prob > red_prob else red_team,
            "verdict": verdict,
            "verdict_en": verdict_en,
            "verdict_side": verdict_side,
            "counter_verdict": counter_verdict,
            "counter_verdict_en": counter_verdict_en,
            "counter_verdict_side": counter_verdict_side,
            "fold_std": round(fold_std, 4),
            "fold_details": {str(k): {"mean_prob": round(v["mean_prob"], 4),
                                       "seed_preds": [round(p, 4) for p in v["seed_preds"]]}
                             for k, v in fold_details.items()},
            "shap_values": shap_result,
            "counters": counter_data,
            "synergies": synergy_data,
            "champion_stats": champion_stats,
            "unknown_players": unknown_info,
        }

    def _format_delta_result(self, match_info, pre_prob, post_prob, delta,
                              draft_cols, hard_cols, post_imp, unknown_info, shap_data=None):
        """格式化 BP Delta 结果（含可解释性信息）

        语义约定：与 _format_predict_result 一致
          - match_info.blue_team / blue_champions = 首选方（BP 蓝方）
          - 模型输出 pre_prob/post_prob = 首选方（BP 蓝方）胜率，无需反转。
        """
        blue_team = match_info.get("blue_team", "蓝方") or "蓝方"
        red_team = match_info.get("red_team", "红方") or "红方"

        post_blue_prob = post_prob
        pre_blue_prob = pre_prob
        post_red_prob = 1 - post_blue_prob

        if delta > 0.005:
            direction = "blue"
            direction_text = f"BP 对蓝方 ({blue_team}) 有利"
            direction_text_en = f"BP favors Blue Side ({blue_team})"
        elif delta < -0.005:
            direction = "red"
            direction_text = f"BP 对红方 ({red_team}) 有利"
            direction_text_en = f"BP favors Red Side ({red_team})"
        else:
            direction = "even"
            direction_text = "BP 影响微弱, 双方阵容势均力敌"
            direction_text_en = "BP impact minimal, comps are evenly matched"

        abs_delta = abs(delta)
        if abs_delta >= 0.10:
            verdict = "极大影响 - BP 决定了比赛走向"
            verdict_en = "Huge impact - BP decides the game"
        elif abs_delta >= 0.05:
            verdict = "显著影响 - 阵容优劣明显"
            verdict_en = "Significant impact - clear comp advantage"
        elif abs_delta >= 0.02:
            verdict = "中等影响 - 阵容有一定优劣势"
            verdict_en = "Moderate impact - some comp advantages"
        else:
            verdict = "微弱影响 - 阵容基本均衡"
            verdict_en = "Slight impact - comps are balanced"

        win_diff = abs(post_blue_prob - post_red_prob)
        if win_diff <= 0.10:
            win_verdict = "势均力敌"
            win_verdict_en = "Even matchup"
            win_verdict_side = "even"
        elif post_blue_prob > post_red_prob:
            win_verdict = f"{blue_team} 优势"
            win_verdict_en = f"{blue_team} favored"
            win_verdict_side = "blue"
        else:
            win_verdict = f"{red_team} 优势"
            win_verdict_en = f"{red_team} favored"
            win_verdict_side = "red"

        shap_result = shap_data or {"positive": [], "negative": [], "waterfall_data": []}

        blue_champs_list = match_info.get("blue_champions", [])
        red_champs_list = match_info.get("red_champions", [])
        pos_order = ["top", "jng", "mid", "bot", "sup"]
        pos_for_explain = ["top", "jungle", "mid", "bot", "support"]
        blue_champs_dict = {}
        red_champs_dict = {}
        for i, pos in enumerate(pos_order):
            bc = blue_champs_list[i] if i < len(blue_champs_list) else ""
            rc = red_champs_list[i] if i < len(red_champs_list) else ""
            if bc:
                blue_champs_dict[pos_for_explain[i]] = bc
            if rc:
                red_champs_dict[pos_for_explain[i]] = rc

        counter_data = analyze_counters(blue_champs_dict, red_champs_dict)
        synergy_data = analyze_synergy(blue_champs_dict, red_champs_dict)
        champion_stats = get_champion_stats(blue_champs_dict, red_champs_dict, self.meta_stats)

        blue_counter_adv = sum(1 for c in counter_data if c["counter_side"] == "blue")
        red_counter_adv = sum(1 for c in counter_data if c["counter_side"] == "red")
        if blue_counter_adv > red_counter_adv:
            counter_verdict = f"蓝方阵容counter占优 ({blue_counter_adv}路优势)"
            counter_verdict_en = f"Blue Side has counter advantage ({blue_counter_adv} lanes)"
            counter_verdict_side = "blue"
        elif red_counter_adv > blue_counter_adv:
            counter_verdict = f"红方阵容counter占优 ({red_counter_adv}路优势)"
            counter_verdict_en = f"Red Side has counter advantage ({red_counter_adv} lanes)"
            counter_verdict_side = "red"
        else:
            counter_verdict = "双方阵容counter势均力敌"
            counter_verdict_en = "Even counter matchup on both sides"
            counter_verdict_side = "even"

        return {
            "blue_team": blue_team,
            "red_team": red_team,
            "league": match_info.get("league", "LCK"),
            "is_playoff": match_info.get("is_playoff", False),
            "predraft": {
                "blue_prob": round(pre_blue_prob, 4),
                "red_prob": round(1 - pre_blue_prob, 4),
            },
            "postdraft": {
                "blue_prob": round(post_blue_prob, 4),
                "red_prob": round(post_red_prob, 4),
                "blue_score": round(post_blue_prob * 100, 1),
                "red_score": round(post_red_prob * 100, 1),
            },
            "delta": round(delta, 4),
            "abs_delta": round(abs_delta, 4),
            "direction": direction,
            "direction_text": direction_text,
            "direction_text_en": direction_text_en,
            "verdict": verdict,
            "verdict_en": verdict_en,
            "win_verdict": win_verdict,
            "win_verdict_en": win_verdict_en,
            "win_verdict_side": win_verdict_side,
            "counter_verdict": counter_verdict,
            "counter_verdict_en": counter_verdict_en,
            "counter_verdict_side": counter_verdict_side,
            "shap_values": shap_result,
            "counters": counter_data,
            "synergies": synergy_data,
            "champion_stats": champion_stats,
            "unknown_players": unknown_info,
        }

    def _build_position_matchups(self, match_info):
        """构建位置对位信息"""
        blue_champs = match_info.get("blue_champions", [])
        red_champs = match_info.get("red_champions", [])
        matchups = {}

        for i, pos in enumerate(POSITIONS):
            bc = blue_champs[i] if i < len(blue_champs) else ""
            rc = red_champs[i] if i < len(red_champs) else ""
            if bc or rc:
                matchups[POS_WEB[pos]] = {
                    "blue_champ": bc,
                    "red_champ": rc,
                }
        return matchups

    # ---- 模型加载 ----

    def _load_models(self):
        """加载 CatBoost 模型 (模式感知)。

        生产模式: 优先加载 models/production/ 下的生产模型
        训练模式: 加载 models/fold_0~4/ 下的 OOT 折模型
        """
        from catboost import CatBoostClassifier
        shared, _ = get_config()
        n_seeds = shared.n_seeds

        # 生产模式: 优先加载生产模型
        if is_production_mode() and os.path.exists(PRODUCTION_DIR):
            prod_models = []
            for seed_idx in range(n_seeds):
                model_path = os.path.join(PRODUCTION_DIR, f"catboost_seed_{seed_idx}.cbm")
                if os.path.exists(model_path):
                    model = CatBoostClassifier()
                    model.load_model(model_path)
                    prod_models.append(model)
            if prod_models:
                log.info(f"生产模式: 加载 production 模型 ({len(prod_models)} seeds)")
                return {"production": prod_models}
            log.warning("生产模式但未找到生产模型, 回退到 OOT 折模型")

        # 训练模式 或 生产模型缺失: 加载 OOT 折模型
        models = {}
        for fold_idx in range(5):
            fold_dir = os.path.join(MODELS_DIR, f"fold_{fold_idx}")
            if not os.path.exists(fold_dir):
                continue
            fold_models = []
            for seed_idx in range(n_seeds):
                model_path = os.path.join(fold_dir, f"catboost_seed_{seed_idx}.cbm")
                if os.path.exists(model_path):
                    model = CatBoostClassifier()
                    model.load_model(model_path)
                    fold_models.append(model)
            if fold_models:
                models[fold_idx] = fold_models
        log.info(f"{'训练' if is_training_mode() else '回退'}模式: 加载 OOT 折模型 ({len(models)} folds)")
        return models

    def _load_league_teams(self):
        """从 active_rosters.csv 加载联赛-战队映射

        active_rosters.csv 是现役名单（来自 Liquipedia），用于前端 UI 白名单
        和推理时的输入上下文。
        """
        roster_path = os.path.join(BASE_DIR, "cleaned_data", "active_rosters.csv")
        if not os.path.exists(roster_path):
            return {}
        try:
            roster_df = pd.read_csv(roster_path)
            league_teams = {}
            for league in roster_df["league"].unique():
                teams = sorted(roster_df[roster_df["league"] == league]["team"].unique().tolist())
                league_teams[league] = teams
            log.info(f"联赛-战队映射加载完成: {', '.join(f'{k}:{len(v)}' for k,v in league_teams.items())}")
            return league_teams
        except Exception as e:
            log.warning(f"加载联赛-战队映射失败: {e}")
            return {}
