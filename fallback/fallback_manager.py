"""
fallback_manager.py — 兜底管理器
=================================
协调所有兜底组件，提供统一的推理入口。

推理流程:
  1. 检查前置规则: 极端冷启动检测
  2. 检查后置统计探针: 滑动窗口指标监控
  3. 走深度网络推理
  4. 检查置信度坍塌: Logit Collapse
  5. 走级联树模型
  6. 返回最终结果

用法:
    manager = FallbackManager(recommender, store, backend)
    results = manager.predict_pick(...)
    results = manager.predict_ban(...)
"""

import os
import sys
import logging
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logger_config import get_logger

from fallback.triggers import (
    LogitCollapseDetector,
    RollingMetricsMonitor,
    ExtremeColdStartDetector,
    get_rolling_monitor,
    get_logit_detector,
    get_cold_start_detector,
)
from fallback.rule_engine import RuleBasedEngine
from fallback.data_pipeline import load_cleaned_meta, load_cleaned_players

log = get_logger(__name__)


class FallbackManager:
    """
    兜底管理器 — 包裹 BPRecommender，提供带兜底的推理。

    设计原则:
      - 前置条件检测 (冷启动/指标降级) 在推理前执行
      - 后置条件检测 (Logit 坍塌) 在 Transformer 推理后执行
      - 任何触发条件满足 → 走规则引擎
      - 规则引擎输出格式与神经网络输出格式一致
    """

    def __init__(self, recommender, store, backend=None):
        """
        Args:
            recommender: BPRecommender 实例
            store: PredictFeatureStore 实例
            backend: BPRecommendationBackend 实例 (可选, 用于获取 idx_to_name)
        """
        self.recommender = recommender
        self.store = store
        self.backend = backend

        # 触发器
        self.logit_detector = get_logit_detector()
        self.rolling_monitor = get_rolling_monitor()
        self.cold_start_detector = get_cold_start_detector()

        # 规则引擎
        self.rule_engine = RuleBasedEngine(
            meta_stats=load_cleaned_meta(),
            player_stats=load_cleaned_players(),
            feature_store=store,
        )

        # 统计
        self._fallback_count = 0
        self._normal_count = 0
        self._fallback_reasons = {}  # reason -> count

        log.info("FallbackManager 初始化完成")


    def refresh_data(self):
        """刷新规则引擎数据"""
        self.rule_engine.refresh_data()

    # ==================== 推理入口 ====================

    def predict_pick(self, bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                     global_context, cand_np, mask_np, target_step, last_ally_pos,
                     position=None, ally_pids=None, enemy_pids=None):
        """
        带兜底的 Pick 推荐。

        参数与 BPRecommender.predict_pick 相同，额外接受:
          - position: str, 当前选位 (top/jng/mid/bot/sup)
          - ally_pids: list[str], 友方选手 ID
          - enemy_pids: list[str], 敌方选手 ID

        Returns:
            list[(champion_idx, score, rank)], 格式与神经网络输出一致
        """
        bp_context = {
            "bp_seq_ids": bp_seq_ids,
            "ally_champs": ally_champs,
            "enemy_champs": enemy_champs,
            "unavail_set": unavail_set,
            "global_context": global_context,
            "cand_np": cand_np,
            "mask_np": mask_np,
            "target_step": target_step,
            "last_ally_pos": last_ally_pos,
            "position": position,
            "ally_pids": ally_pids,
            "enemy_pids": enemy_pids,
        }

        # 1. 检查前置规则: 极端冷启动
        if self._check_cold_start():
            return self._rule_based_pick(bp_context, reason="cold_start")

        # 2. 检查后置统计探针: 滑动窗口
        if self._check_rolling_degraded():
            return self._rule_based_pick(bp_context, reason="rolling_degraded")

        # 3. 走深度网络推理
        try:
            # === Legacy v1 兼容: 翻译输入到 legacy idx ===
            recommender = self.recommender
            if getattr(recommender, 'legacy_mode', False):
                _bp_seq = recommender._translate_bp_seq_to_legacy(bp_seq_ids)
                _cand_np, _mask_np = recommender._translate_cand_to_legacy(cand_np, mask_np)
                _cs = 3  # legacy v1 champion_start_idx
                _ve = recommender.legacy_vocab_size
            else:
                _bp_seq = bp_seq_ids
                _cand_np = cand_np
                _mask_np = mask_np
                _cs = self.store.champion_start_idx
                _ve = self.store.vocab_size

            # Transformer 推理
            cs = _cs
            bp_padded = _bp_seq + [self.store.PAD_IDX] * (20 - len(_bp_seq))
            bp_t = self._to_tensor(np.array([bp_padded], dtype=np.int64), dtype="long")
            ctx_t = self._to_tensor(np.array([global_context], dtype=np.float32))
            cand_t = self._to_tensor(np.array([_cand_np], dtype=np.float32))
            mask_t = self._to_tensor(np.array([_mask_np], dtype=np.float32))
            lap_t = self._to_tensor(np.array([last_ally_pos], dtype=np.int64), dtype="long")

            with self._no_grad():
                cs_out = self.recommender.pick_cs_model(bp_t, ctx_t, cand_t, mask_t,
                                                        last_ally_pos=lap_t)
                cs_logits = cs_out["logits"].squeeze(0).cpu().numpy()

                if self.recommender.pick_nocs_model:
                    cand_nocs_t = cand_t.clone()
                    cand_nocs_t[:, :, CS_FEATURE_INDICES] = 0.0
                    nocs_logits = self.recommender.pick_nocs_model(
                        bp_t, ctx_t, cand_nocs_t, mask_t, last_ally_pos=lap_t
                    )["logits"].squeeze(0).cpu().numpy()
                else:
                    nocs_logits = cs_logits

            # 4. 检查置信度坍塌
            if self._check_logit_collapse(cs_logits):
                return self._rule_based_pick(bp_context, reason="logit_collapse")

            # 5. 走级联树模型 (截断 Top 50，与 predict_ban / BPRecommender.predict_pick 一致)
            valid_cids = np.where(_mask_np > 0.5)[0]
            valid_cids = valid_cids[valid_cids >= cs]
            total_valid = len(valid_cids)

            top_k_limit = min(50, total_valid)
            cs_valid_logits = cs_logits[valid_cids]
            top_k_local_indices = np.argsort(-cs_valid_logits)[:top_k_limit]
            eval_cids = valid_cids[top_k_local_indices]
            total_eval = len(eval_cids)

            cs_gf = self.recommender._compute_group_features(cs_logits, _mask_np, cs, _ve)
            nocs_gf = self.recommender._compute_group_features(nocs_logits, _mask_np, cs, _ve)

            X_arr = _build_feature_matrix_batch(
                cs_logits[eval_cids], cs_gf["rank_map"][eval_cids], cs_gf,
                nocs_logits[eval_cids], nocs_gf["rank_map"][eval_cids], nocs_gf,
                _cand_np[eval_cids], total_valid, total_eval, target_step,
            )

            lgb_preds = np.zeros(total_eval, dtype=np.float64)
            X_scaled = self.recommender.scaler.transform(X_arr)
            for m in self.recommender.lgb_models:
                lgb_preds += m.predict(X_scaled)
            lgb_preds /= max(len(self.recommender.lgb_models), 1)

            if getattr(self.recommender, "pick_fusion_mode", "blend") == "residual_init_score":
                final_scores = lgb_preds + cs_logits[eval_cids]
            else:
                cs_rn = self.recommender._rank_normalize(cs_logits[eval_cids])
                lgb_rn = self.recommender._rank_normalize(lgb_preds)
                final_scores = self.recommender.pick_blend_alpha * cs_rn + (1.0 - self.recommender.pick_blend_alpha) * lgb_rn

            sorted_idx = np.argsort(-final_scores)
            results = []
            for rank, si in enumerate(sorted_idx[:50]):
                cid = eval_cids[si]
                results.append((cid, float(final_scores[si]), rank + 1))

            # === Legacy v1 兼容: 翻译输出回 v2 idx ===
            if getattr(recommender, 'legacy_mode', False):
                results = recommender._translate_results_from_legacy(results)

            self._normal_count += 1
            return results

        except Exception:
            log.exception("Pick 深度推理失败")
            return self._rule_based_pick(bp_context, reason="inference_error")

    def predict_ban(self, bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                    global_context, cand_np, mask_np, target_step,
                    position=None, ally_pids=None, enemy_pids=None):
        """
        带兜底的 Ban 推荐。

        参数与 BPRecommender.predict_ban 相同，额外接受:
          - position: str, 当前 Ban 位
          - ally_pids: list[str], 友方选手 ID
          - enemy_pids: list[str], 敌方选手 ID

        Returns:
            list[(champion_idx, score, rank)], 格式与神经网络输出一致
        """
        bp_context = {
            "bp_seq_ids": bp_seq_ids,
            "ally_champs": ally_champs,
            "enemy_champs": enemy_champs,
            "unavail_set": unavail_set,
            "global_context": global_context,
            "cand_np": cand_np,
            "mask_np": mask_np,
            "target_step": target_step,
            "position": position,
            "ally_pids": ally_pids,
            "enemy_pids": enemy_pids,
        }

        # 1. 检查前置规则: 极端冷启动
        if self._check_cold_start():
            return self._rule_based_ban(bp_context, reason="cold_start")

        # 2. 检查后置统计探针: 滑动窗口
        if self._check_rolling_degraded():
            return self._rule_based_ban(bp_context, reason="rolling_degraded")

        # 3. 走深度网络推理
        try:
            # === Legacy v1 兼容: 翻译输入到 legacy idx ===
            recommender = self.recommender
            if getattr(recommender, 'legacy_mode', False):
                _bp_seq = recommender._translate_bp_seq_to_legacy(bp_seq_ids)
                _cand_np, _mask_np = recommender._translate_cand_to_legacy(cand_np, mask_np)
                _cs = 3  # legacy v1 champion_start_idx
                _ve = recommender.legacy_vocab_size
            else:
                _bp_seq = bp_seq_ids
                _cand_np = cand_np
                _mask_np = mask_np
                _cs = self.store.champion_start_idx
                _ve = self.store.vocab_size

            cs = _cs
            bp_padded = _bp_seq + [self.store.PAD_IDX] * (20 - len(_bp_seq))
            bp_t = self._to_tensor(np.array([bp_padded], dtype=np.int64), dtype="long")
            ctx_t = self._to_tensor(np.array([global_context], dtype=np.float32))
            cand_t = self._to_tensor(np.array([_cand_np], dtype=np.float32))
            mask_t = self._to_tensor(np.array([_mask_np], dtype=np.float32))

            hist_pos = np.full(20, -1, dtype=np.int64)
            for i in range(min(len(_bp_seq), 20)):
                cid = _bp_seq[i]
                if cid >= cs:
                    from bp_recommendation.feature_pipeline import BP_SEQUENCE
                    if i < len(BP_SEQUENCE) and BP_SEQUENCE[i][0] == "pick":
                        # Legacy 模式下 pos_prior 是 v2 形状, cid 是 legacy idx, 跳过
                        if not getattr(recommender, 'legacy_mode', False) and cid < self.store.pos_prior.shape[0]:
                            hist_pos[i] = int(np.argmax(self.store.pos_prior[cid]))
            hist_t = self._to_tensor(np.array([hist_pos], dtype=np.int64), dtype="long")

            with self._no_grad():
                ban_out = self.recommender.ban_model(bp_t, ctx_t, cand_t, mask_t, history_positions=hist_t)
                cs_logits = ban_out["logits"].squeeze(0).cpu().numpy()

            # 4. 检查置信度坍塌
            if self._check_logit_collapse(cs_logits):
                return self._rule_based_ban(bp_context, reason="logit_collapse")

            # 5. 走级联树模型 (安全修复版：必须截断 Top 50 防止外推幻觉)
            valid_cids = np.where(_mask_np > 0.5)[0]
            valid_cids = valid_cids[valid_cids >= cs]

            # 【截断 Top 50 逻辑】
            top_k_limit = min(50, len(valid_cids))
            cs_valid_logits = cs_logits[valid_cids]
            top_k_local_indices = np.argsort(-cs_valid_logits)[:top_k_limit]

            eval_cids = valid_cids[top_k_local_indices]
            total_eval = len(eval_cids)

            cs_gf = _compute_ban_group_features(cs_logits, _mask_np, cs, _ve)

            X_arr = _build_ban_feature_matrix_batch(
                cs_logits[eval_cids], cs_gf["rank_map"][eval_cids], cs_gf,
                _cand_np[eval_cids], total_eval,
            )

            lgb_preds = np.zeros(total_eval, dtype=np.float64)
            X_scaled = self.recommender.ban_scaler.transform(X_arr)
            for m in self.recommender.ban_lgb_models:
                lgb_preds += m.predict(X_scaled)
            lgb_preds /= max(len(self.recommender.ban_lgb_models), 1)

            base_rn = self.recommender._rank_normalize(cs_logits[eval_cids])
            lgb_rn = self.recommender._rank_normalize(lgb_preds)
            final_scores = self.recommender.ban_blend_alpha * base_rn + (1.0 - self.recommender.ban_blend_alpha) * lgb_rn

            sorted_idx = np.argsort(-final_scores)
            results = []
            # 返回 Top 50，前端在 'all' 下显示 Top 20，位置过滤时从 Top 50 中筛选
            for rank, si in enumerate(sorted_idx[:50]):
                cid = eval_cids[si]
                results.append((cid, float(final_scores[si]), rank + 1))

            # === Legacy v1 兼容: 翻译输出回 v2 idx ===
            if getattr(recommender, 'legacy_mode', False):
                results = recommender._translate_results_from_legacy(results)

            self._normal_count += 1
            return results

        except Exception:
            log.exception("Ban 深度推理失败")
            return self._rule_based_ban(bp_context, reason="inference_error")

    # ==================== 规则引擎兜底 ====================

    def _rule_based_pick(self, bp_context, reason="unknown"):
        """使用规则引擎进行 Pick 推荐"""
        self._fallback_count += 1
        self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
        log.warning(f"Pick 触发兜底 (原因: {reason})")

        position = bp_context.get("position", "top")
        ally_pids = bp_context.get("ally_pids", [])
        enemy_pids = bp_context.get("enemy_pids", [])
        ally_champs = bp_context.get("ally_champs", [])
        enemy_champs = bp_context.get("enemy_champs", [])
        unavail_set = bp_context.get("unavail_set", set())

        rule_results = self.rule_engine.recommend_pick(
            position=position,
            ally_pids=ally_pids,
            enemy_pids=enemy_pids,
            ally_champs=ally_champs,
            enemy_champs=enemy_champs,
            unavail_set=unavail_set,
        )

        # 转换为 (champion_idx, score, rank) 格式，与神经网络输出一致
        results = []
        for r in rule_results:
            results.append((r["champion_idx"], r["score"], r["rank"]))

        return results

    def _rule_based_ban(self, bp_context, reason="unknown"):
        """使用规则引擎进行 Ban 推荐"""
        self._fallback_count += 1
        self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
        log.warning(f"Ban 触发兜底 (原因: {reason})")

        position = bp_context.get("position", "top")
        ally_pids = bp_context.get("ally_pids", [])
        enemy_pids = bp_context.get("enemy_pids", [])
        ally_champs = bp_context.get("ally_champs", [])
        enemy_champs = bp_context.get("enemy_champs", [])
        unavail_set = bp_context.get("unavail_set", set())

        rule_results = self.rule_engine.recommend_ban(
            position=position,
            ally_pids=ally_pids,
            enemy_pids=enemy_pids,
            ally_champs=ally_champs,
            enemy_champs=enemy_champs,
            unavail_set=unavail_set,
        )

        # 转换为 (champion_idx, score, rank) 格式
        results = []
        for r in rule_results:
            results.append((r["champion_idx"], r["score"], r["rank"]))

        return results

    # ==================== 触发器检查 ====================

    def _check_cold_start(self):
        """检查极端冷启动"""
        return self.cold_start_detector.is_cold_start_from_store(self.store)

    def _check_rolling_degraded(self):
        """检查滑动窗口指标是否降级"""
        return self.rolling_monitor.is_degraded()

    def _check_logit_collapse(self, logits):
        """检查 Logit 置信度坍塌"""
        return self.logit_detector.is_collapsed(logits)

    # ==================== 辅助方法 ====================

    def _to_tensor(self, data, dtype="float"):
        """安全转换为 tensor"""
        import torch
        if dtype == "long":
            return torch.as_tensor(data, dtype=torch.long, device=self.recommender._get_device())
        return torch.as_tensor(data, dtype=torch.float32, device=self.recommender._get_device())

    def _no_grad(self):
        """获取 torch.no_grad 上下文"""
        import torch
        return torch.no_grad()

    def get_stats(self):
        """获取 FallbackManager 统计"""
        total = self._fallback_count + self._normal_count
        return {
            "total_inferences": total,
            "normal_count": self._normal_count,
            "fallback_count": self._fallback_count,
            "fallback_rate": round(self._fallback_count / max(total, 1), 4),
            "fallback_reasons": dict(self._fallback_reasons),
            "logit_detector": self.logit_detector.get_stats(),
            "rolling_metrics": self.rolling_monitor.get_rolling_metrics(),
        }

    def record_result(self, pick_at_10=None, ban_at_10=None, auc=None):
        """记录一次推荐结果（用于更新滑动窗口指标）"""
        self.rolling_monitor.record_pick_result(
            pick_at_10=pick_at_10, ban_at_10=ban_at_10, auc=auc
        )


# ==================== 需要从 bp_predict 引用的函数 ====================

# 这些函数在 bp_predict.py 中定义，但 fallback_manager 需要调用
# 为了避免循环引用，在模块级别声明为 None，运行时动态导入

CS_FEATURE_INDICES = None
_build_feature_matrix_batch = None
_build_ban_feature_matrix_batch = None
_compute_ban_group_features = None


def _lazy_import_bp_predict_helpers():
    """延迟导入 bp_predict 中的辅助函数"""
    global CS_FEATURE_INDICES, _build_feature_matrix_batch, _build_ban_feature_matrix_batch, _compute_ban_group_features
    if CS_FEATURE_INDICES is not None:
        return
    from bp_recommendation.bp_predict import (
        CS_FEATURE_INDICES as _cs,
        _build_feature_matrix_batch as _bfmb,
        _build_ban_feature_matrix_batch as _bbfmb,
        _compute_ban_group_features as _cbgf,
    )
    CS_FEATURE_INDICES = _cs
    _build_feature_matrix_batch = _bfmb
    _build_ban_feature_matrix_batch = _bbfmb
    _compute_ban_group_features = _cbgf


# Lazy import on module load
try:
    _lazy_import_bp_predict_helpers()
except Exception:
    log.exception("延迟导入 bp_predict 辅助函数失败")