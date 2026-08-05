"""
BP 推荐模型后端封装
=============================================
为 Flask Web 应用提供 BP 推荐 (Ban/Pick) 功能的生产环境后端。
从 bp_ui_server.py 中提取核心推理逻辑，与 HTTP 路由层解耦，支持多并发、限流、超时控制和兜底机制。

功能描述:
    - 加载 BP 推荐模型和特征存储
    - 提供无状态的推荐推理接口
    - 支持并发控制、请求限流、推理超时保护
    - 集成规则兜底机制，应对模型异常情况
    - 提供英雄列表、战队列表、选手列表等辅助查询

主要类:
    - BPRecommendationBackend: BP 推荐后端核心类

主要方法:
    - load(): 加载模型和特征数据
    - recommend(payload): 执行 BP 推荐推理
    - get_champions(): 获取英雄列表
    - get_teams(league): 获取战队列表
    - get_players(team_name): 获取选手列表
    - get_status(): 获取后端状态
    - get_fallback_status(): 获取兜底机制状态

使用方法:
    from bp_recommendation.bp_recommendation_backend import BPRecommendationBackend
    
    backend = BPRecommendationBackend()
    result = backend.load()
    if result["success"]:
        payload = {
            "blue_team": "JDG",
            "red_team": "T1",
            "completed_steps": 0,
            "bp_seq_ids": [],
            "league": "LPL"
        }
        recommendation = backend.recommend(payload)
"""
import os
import sys

# ---- 必须在导入 LightGBM 之前设置，防止 macOS 上 OpenMP 死锁 ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import uuid
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "bp_recommendation"))

from logger_config import get_logger
from bp_recommendation.bp_predict import BPRecommender, PredictFeatureStore, build_name_lookup, resolve_champion
from bp_recommendation.feature_pipeline import BP_SEQUENCE, load_champion_vocabulary

# 兜底机制
try:
    from fallback.fallback_manager import FallbackManager
    HAS_FALLBACK = True
except ImportError:
    HAS_FALLBACK = False
    FallbackManager = None

VOCAB_PATH = os.path.join(BASE_DIR, "cleaned_data", "champion_vocabulary.json")
POS_JSON = os.path.join(BASE_DIR, "cleaned_data", "champion_position_mapping.json")
MERGED_STATS_PATH = os.path.join(BASE_DIR, "cleaned_data", "merged_champion_stats.csv")
LEAGUES = ["LPL", "LCK", "LEC"]
POSITIONS = ["top", "jng", "mid", "bot", "sup"]
POS_CN = {"top": "上单", "jng": "打野", "mid": "中单", "bot": "ADC", "sup": "辅助"}
POS_FULL = {"top": "top", "jng": "jungle", "mid": "mid", "bot": "bot", "sup": "support",
            # 兼容数据文件中使用的全称
            "jungle": "jungle", "support": "support"}
# Pick阶段选人次序标签 (蓝方/红方)，不预设位置
PICK_POS_BLUE = ["pick1", "pick2", "pick3", "pick4", "pick5"]
PICK_POS_RED = ["pick1", "pick2", "pick3", "pick4", "pick5"]
PICK_ORDER_CN = {
    "pick1": "一选", "pick2": "二选", "pick3": "三选",
    "pick4": "四选", "pick5": "五选",
}

log = get_logger(__name__)


class BPRecommendationBackend:
    """BP 推荐模型后端"""

    # ====== 多并发应对预案配置 ======
    MAX_CONCURRENT_INFERENCES = 2        # 最大并发推理数（资源隔离）
    INFERENCE_TIMEOUT_SECONDS = 10.0     # 单次推理超时时间
    RATE_LIMIT_WINDOW_SECONDS = 60.0     # 限流时间窗口
    RATE_LIMIT_MAX_REQUESTS = 60         # 时间窗口内最大请求数 (60s 内最多 60 次, 覆盖 3 场完整 BP)

    def __init__(self):
        self.recommender = None
        self.store = None
        self.name_lookup = None
        self.idx_to_name = None
        self.champion_list = None
        self.position_mapping = None
        self.league_teams = {}
        self._loaded = False
        self.fallback_manager = None
        self.merged_stats = {}
        self._merged_stats_loaded = False

        # 2. 推理信号量：限制同时执行的推理数量，实现资源隔离
        self._inference_semaphore = threading.Semaphore(self.MAX_CONCURRENT_INFERENCES)
        # 3. 限流：滑动窗口记录请求时间戳
        self._request_timestamps = deque()
        self._rate_limit_lock = threading.Lock()
        # 4. 推理线程池（资源隔离：推理在独立线程执行，便于超时中断）
        self._inference_executor = ThreadPoolExecutor(
            max_workers=self.MAX_CONCURRENT_INFERENCES,
            thread_name_prefix="bp_inference",
        )

    def load(self):
        try:
            t0 = time.time()
            self.recommender = BPRecommender()
            self.store = self.recommender.store

            self.name_lookup, vocab_data = build_name_lookup(VOCAB_PATH)
            self.idx_to_name = self.store.idx_to_name

            self.position_mapping = {}
            if os.path.exists(POS_JSON):
                with open(POS_JSON, "r", encoding="utf-8") as f:
                    self.position_mapping = json.load(f)

            self._load_merged_stats()
            self.champion_list = self._build_champion_list(vocab_data)
            self.league_teams = self._load_league_teams()
            self._loaded = True

            # 初始化兜底管理器
            if HAS_FALLBACK:
                try:
                    self.fallback_manager = FallbackManager(
                        recommender=self.recommender,
                        store=self.store,
                        backend=self,
                    )
                    log.info("Fallback 兜底管理器已启动")
                except Exception as e:
                    log.warning(f"Fallback 兜底管理器初始化失败 (非致命): {e}")
                    self.fallback_manager = None

            elapsed = time.time() - t0
            log.info(f"BP推荐模型加载完成, 耗时 {elapsed:.1f}s, {len(self.champion_list)} 英雄")
            return {"success": True, "message": f"BP推荐模型加载完成, {len(self.champion_list)} 英雄"}
        except Exception as e:
            log.error(f"BP推荐模型加载失败: {e}")
            return {"success": False, "message": str(e)}

    def is_loaded(self):
        return self._loaded

    def _load_merged_stats(self):
        """加载融合后的英雄统计数据（Bayesian融合：排位先验 + 职业观测）"""
        import pandas as pd
        self.merged_stats = {}
        self._merged_stats_loaded = False
        try:
            if os.path.exists(MERGED_STATS_PATH):
                df = pd.read_csv(MERGED_STATS_PATH)
                is_new_format = "main_position" in df.columns
                for _, row in df.iterrows():
                    champ = row["champion"]
                    stats = {
                        "win_rate": float(row.get("win_rate", 0.5) or 0.5),
                        "pick_rate": float(row.get("pick_rate", 0) or 0),
                        "ban_rate": float(row.get("ban_rate", 0) or 0),
                        "presence_rate": float(row.get("presence_rate", 0) or 0),
                    }
                    if is_new_format:
                        self.merged_stats[champ] = {
                            "overall": stats,
                            "main_position": row.get("main_position", "mid"),
                            "by_position": {}
                        }
                    else:
                        pos = row.get("position", "")
                        if champ not in self.merged_stats:
                            self.merged_stats[champ] = {"by_position": {}}
                        self.merged_stats[champ]["by_position"][pos] = stats
                if not is_new_format:
                    for champ in self.merged_stats:
                        pos_data = self.merged_stats[champ]["by_position"]
                        if pos_data:
                            max_pr_pos = max(pos_data.keys(), key=lambda p: pos_data[p]["presence_rate"])
                            self.merged_stats[champ]["overall"] = pos_data[max_pr_pos]
                log.info(f"融合统计数据加载完成: {len(self.merged_stats)} 英雄")
                self._merged_stats_loaded = True
            else:
                log.warning(f"融合统计文件不存在: {MERGED_STATS_PATH}")
        except Exception as e:
            log.exception(f"加载融合统计数据失败: {e}")
            self.merged_stats = {}
            self._merged_stats_loaded = False

    def _get_display_stats(self, champ_name, position=None):
        """获取用于前端展示的英雄统计数据，优先使用融合数据"""
        if champ_name in self.merged_stats:
            data = self.merged_stats[champ_name]
            if position and position in data.get("by_position", {}):
                return data["by_position"][position]
            return data.get("overall", {"win_rate": 0.5, "pick_rate": 0, "ban_rate": 0, "presence_rate": 0})
        return None

    # ====== 限流机制 ======

    def _check_rate_limit(self) -> bool:
        """滑动窗口限流检查，返回 True 表示允许请求"""
        now = time.time()
        with self._rate_limit_lock:
            # 清理过期时间戳
            cutoff = now - self.RATE_LIMIT_WINDOW_SECONDS
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()
            # 检查是否超限
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
            "champion_count": len(self.champion_list) if self.champion_list else 0,
            "team_count": len(self.store.team_players_map) if self.store else 0,
        }

    def get_fallback_status(self):
        """获取兜底机制状态"""
        if self.fallback_manager is None:
            return {"enabled": False, "message": "兜底机制未启用"}
        stats = self.fallback_manager.get_stats()
        return {
            "enabled": True,
            "stats": stats,
        }

    def get_champions(self):
        return self.champion_list or []

    def get_teams(self, league=None):
        """返回指定联赛的战队列表，league=None返回全部"""
        if league and self.league_teams and league in self.league_teams:
            return self.league_teams[league]
        if not self.store:
            return []
        return sorted(self.store.team_players_map.keys())

    def get_players(self, team_name):
        """从 active_rosters.csv 返回战队选手（与 predict 后端格式一致）。"""
        if not self.store or not team_name:
            return []
        try:
            from bp_prediction.feature_builder import get_team_roster, resolve_team_name
            all_teams = set()
            for teams in (self.league_teams or {}).values():
                all_teams.update(teams)
            resolved = resolve_team_name(team_name, all_teams) or team_name
            roster = get_team_roster(resolved)
            if roster:
                pos_web = {
                    "top": "top", "jungle": "jungle", "mid": "mid",
                    "bot": "bot", "support": "support",
                    "jng": "jungle", "sup": "support",
                }
                result = []
                for p in roster:
                    role = str(p.get("role", "")).strip().lower()
                    web_role = pos_web.get(role, role)
                    pname = p.get("player_name") or p.get("player_id") or ""
                    if pname:
                        result.append({
                            "player_id": pname,
                            "player_name": pname,
                            "role": web_role,
                        })
                return result
        except Exception as e:
            log.warning(f"active_rosters 加载失败, 回退 context 快照: {e}")
        return sorted(self.store.team_players_map.get(team_name, []))

    def get_team_style(self, team_name):
        if not self.store:
            return None
        style = self.store.team_style_dict.get(team_name)
        if style is None:
            return None
        return {
            "team": team_name,
            "avg_ckpm": style[0],
            "avg_golddiffat15": style[1],
            "avg_gamelength": style[2],
            "firstdragon_rate": style[3],
            "firsttower_rate": style[4],
        }

    def _validate_payload(self, payload: dict) -> str:
        """校验推理请求 payload，返回错误描述字符串；返回 None 表示通过。"""
        if not isinstance(payload, dict):
            return "payload 必须为字典"
        # 必填字段（允许空字符串，表示 General 模式，使用默认统计特征）
        for key in ("blue_team", "red_team"):
            if key not in payload or payload[key] is None:
                return f"缺少必填字段: {key}"
        # completed_steps 范围校验
        step = payload.get("completed_steps", 0)
        if not isinstance(step, int) or step < 0:
            return "completed_steps 必须为非负整数"
        if step >= len(BP_SEQUENCE):
            return None  # 由后续逻辑返回 "BP已完成"
        # bp_seq_ids 长度需与 completed_steps 一致
        bp_seq_ids = payload.get("bp_seq_ids", [])
        if not isinstance(bp_seq_ids, list):
            return "bp_seq_ids 必须为列表"
        if len(bp_seq_ids) != step:
            return f"bp_seq_ids 长度({len(bp_seq_ids)})与 completed_steps({step})不一致"
        # 玩家 ID 列表长度（若提供）
        for pid_key in ("blue_pids", "red_pids"):
            pids = payload.get(pid_key)
            if pids is not None:
                if not isinstance(pids, list) or len(pids) != 5:
                    return f"{pid_key} 必须为长度 5 的列表"
        # game_num 范围
        game_num = payload.get("game_num", 1)
        if not isinstance(game_num, int) or not (1 <= game_num <= 5):
            return "game_num 必须为 1-5 的整数"
        return None

    def recommend(self, payload: dict):
        """
        纯粹的无状态推理接口。所有 BP 进度上下文均从 payload 中获取。
        """
        # 生成 request_id 用于全链路日志追踪
        request_id = payload.get("request_id") or uuid.uuid4().hex[:12]
        t_start = time.time()
        log.info(f"[req={request_id}] recommend start, payload keys={list(payload.keys())}")

        if not self._check_rate_limit():
            log.warning(f"[req={request_id}] rate limited")
            return {
                "error": "请求过于频繁，请稍后再试",
                "rate_limited": True,
                "request_id": request_id,
            }
        if not self._loaded:
            log.error(f"[req={request_id}] model not loaded")
            return {"error": "模型未加载", "request_id": request_id}

        # payload 基础校验，避免 KeyError 或类型异常导致推理失败
        validation_err = self._validate_payload(payload)
        if validation_err:
            log.warning(f"[req={request_id}] payload validation failed: {validation_err}")
            return {"error": validation_err, "request_id": request_id}

        step = payload.get("completed_steps", 0)
        if step >= len(BP_SEQUENCE):
            log.info(f"[req={request_id}] BP already completed")
            return {"error": "BP已完成", "request_id": request_id}

        action, side, slot = BP_SEQUENCE[step]
        log.info(f"[req={request_id}] step={step} action={action} side={side} slot={slot}")

        # 1. 解析全局上下文
        league = payload.get("league", "LPL")
        league_vec = np.zeros(len(LEAGUES), dtype=np.float32)
        if league in LEAGUES:
            league_vec[LEAGUES.index(league)] = 1.0

        b_team = payload.get("blue_team", "")
        r_team = payload.get("red_team", "")
        b_style = self.store.team_style_dict.get(b_team, [0.7, 0.0, 1900.0, 0.5, 0.5])
        r_style = self.store.team_style_dict.get(r_team, [0.7, 0.0, 1900.0, 0.5, 0.5])
        
        playoffs_f = 1.0 if payload.get("playoffs") else 0.0
        # first_pick_map_side 可能是 0/1 数值或 "red"/"blue" 字符串
        first_pick_raw = payload.get("first_pick_map_side", 1.0)
        if isinstance(first_pick_raw, str):
            first_pick_f = 1.0 if first_pick_raw.lower() == "blue" else 0.0
        else:
            first_pick_f = float(first_pick_raw)
        
        game_num = int(payload.get("game_num", 1))
        game_num_vec = np.zeros(5, dtype=np.float32)
        if 1 <= game_num <= 5:
            game_num_vec[game_num - 1] = 1.0

        global_context = np.concatenate([
            league_vec, b_style, r_style, [playoffs_f, first_pick_f], game_num_vec
        ]).astype(np.float32)

        # 2. 瞬时解析英雄状态
        bp_seq_ids = payload.get("bp_seq_ids", [])
        curr_side_code = 0 if side == "blue" else 1
        ally_champs, enemy_champs = [], []
        last_ally_pos = -1

        for i, cid in enumerate(bp_seq_ids):
            if cid < self.store.champion_start_idx: continue
            if BP_SEQUENCE[i][0] == "pick":
                if (BP_SEQUENCE[i][1] == "blue" and curr_side_code == 0) or \
                   (BP_SEQUENCE[i][1] == "red" and curr_side_code == 1):
                    ally_champs.append(cid)
                    last_ally_pos = i
                else:
                    enemy_champs.append(cid)

        unavail_set = set(payload.get("unavail_set", bp_seq_ids))
        pre_unavail_list = payload.get("pre_unavail_list", [])
        # 前置局已用英雄 (Fearless Draft) 需同时加入 unavail_set 用于掩码排除
        # 与训练时 dataloader 将 prev_game_champs 加入 unavailable_ids 的逻辑一致
        unavail_set |= set(pre_unavail_list)
        
        ally_pids = payload.get("blue_pids", [""]*5) if side == "blue" else payload.get("red_pids", [""]*5)
        enemy_pids = payload.get("red_pids", [""]*5) if side == "blue" else payload.get("blue_pids", [""]*5)
        team_name = b_team if side == "blue" else r_team
        opp_team = r_team if side == "blue" else b_team

        # 3. 生成特征矩阵
        try:
            if action == "pick":
                cand_np, mask_np = self.store.get_pick_candidate_matrix(
                    side, ally_champs, enemy_champs, unavail_set,
                    ally_pids, enemy_pids, step, team_name, opp_team,
                    pre_unavail_list=pre_unavail_list,
                )
            else:
                cand_np, mask_np = self.store.get_ban_candidate_matrix(
                    side, ally_champs, enemy_champs, unavail_set,
                    ally_pids, enemy_pids, step, team_name, opp_team,
                    pre_unavail_list=pre_unavail_list,
                )
        except Exception as e:
            log.error(f"[req={request_id}] 特征矩阵构建失败: {e}", exc_info=True)
            return {"error": f"特征构建失败: {str(e)}", "request_id": request_id}

        # 4. 线程池资源隔离推理
        try:
            position_hint = payload.get("position_hint")
            future = self._inference_executor.submit(
                self._run_inference, action, step, ally_champs, enemy_champs,
                ally_pids, enemy_pids, cand_np, mask_np, last_ally_pos, position_hint,
                bp_seq_ids, global_context, unavail_set,
            )
            results, is_fallback = future.result(timeout=self.INFERENCE_TIMEOUT_SECONDS)
        except FutureTimeout:
            log.error(f"[req={request_id}] 推理超时 (>{self.INFERENCE_TIMEOUT_SECONDS}s), step={step}")
            return {"error": "推理超时，请重试", "request_id": request_id}
        except Exception as e:
            log.error(f"[req={request_id}] 推荐失败: {e}", exc_info=True)
            return {"error": f"推荐失败: {str(e)}", "request_id": request_id}

        elapsed = time.time() - t_start

        # 4.5 Layer 3 后处理: 位置冲突硬过滤 (互斥去重)
        if action == "pick":
            results = self._filter_pick_duplicates(results, ally_champs)
        elif action == "ban":
            results = self._filter_ban_duplicates(results, enemy_champs)

        log.info(f"[req={request_id}] recommend done, elapsed={elapsed:.3f}s, fallback={is_fallback}, n_results={len(results)}")

        # 5. 返回推荐结果
        return {
            "request_id": request_id,
            "recommendations": self._format_recommendations(results, action, ally_champs, is_fallback),
            "step_info": {
                "step": step, "action": action, "side": side, "slot": slot,
                "is_fallback": is_fallback,
            }
        }

    def check_comp_feasibility(self, current_champs, candidate_cid, min_prob=0.10):
        """
        检查将 candidate_cid 加入 current_champs 后，是否能凑出合法的各司其职的阵容。
        使用二分图最大匹配 (匈牙利算法)。

        Args:
            current_champs: 己方已选英雄列表
            candidate_cid: 待推荐的候选英雄
            min_prob: 认为该英雄能打该位置的最低概率门槛 (低于此概率视为完全不能打)

        Returns:
            bool: True 表示不冲突，可以分配；False 表示扎堆导致死锁
        """
        test_roster = current_champs + [candidate_cid]
        n_heroes = len(test_roster)
        if n_heroes == 0:
            return True

        # 1. 构建邻接表: 记录每个英雄可以打的位置 [0, 1, 2, 3, 4]
        hero_possible_roles = []
        for cid in test_roster:
            if cid < 0 or cid >= len(self.store.pos_prior):
                # 越界英雄视为全能 (兜底)
                hero_possible_roles.append([0, 1, 2, 3, 4])
                continue
            pos_prior = self.store.pos_prior[cid]
            roles = [i for i, prob in enumerate(pos_prior) if float(prob) >= min_prob]
            # 若无任何位置 >= min_prob (如全零英雄)，视为全能避免误杀
            if not roles:
                roles = [0, 1, 2, 3, 4]
            hero_possible_roles.append(roles)

        # 2. 匈牙利算法 DFS 匹配
        role_match = {}  # role_idx -> hero_idx

        def dfs(hero_idx, visited_roles):
            for role in hero_possible_roles[hero_idx]:
                if role not in visited_roles:
                    visited_roles.add(role)
                    if role not in role_match or dfs(role_match[role], visited_roles):
                        role_match[role] = hero_idx
                        return True
            return False

        matches = 0
        for i in range(n_heroes):
            if dfs(i, set()):
                matches += 1

        # 最大匹配数等于英雄总数 => 可以排开，无死锁
        return matches == n_heroes

    def _filter_pick_duplicates(self, results, ally_champs, top_k=50):
        """Layer 3: 己方 Pick 推荐列表过滤器。

        双重拦截:
        1. 硬英雄位置冲突: 己方已选硬英雄 (>= HARD_HERO_THRESHOLD) 某位置，同位置硬英雄直接剔除
        2. 阵容可行性校验: 用匈牙利算法检查加入候选后是否能合法分配 K 个位置，防摇摆英雄扎堆死锁

        注意: 列表内部不去重，允许推荐多个同位置硬英雄供用户选择。
        """
        HARD_THRESH = self.store.HARD_HERO_THRESHOLD

        # 1. 记录己方已有阵容的纯硬位置
        occupied_hard_positions = set()
        for c in ally_champs:
            best_pos, _, is_hard = self.store.analyze_hero_pos_type(c, HARD_THRESH)
            if is_hard:
                occupied_hard_positions.add(best_pos)

        # 预计算己方位置占用（循环外一次，避免重复累加）
        ally_pos_sum = np.zeros(5, dtype=np.float32)
        ally_primary_counts = np.zeros(5, dtype=np.int32)
        for c in ally_champs:
            best_pos, _, _ = self.store.analyze_hero_pos_type(c, HARD_THRESH)
            ally_primary_counts[best_pos] += 1
            if 0 <= c < len(self.store.pos_prior):
                ally_pos_sum += self.store.pos_prior[c]

        # 2. 过滤: 硬英雄冲突 + 主位置拥挤 + 阵容可行性
        filtered = []
        n_filtered_hard = 0
        n_filtered_flex = 0

        for cid, score, rank in results:
            best_pos, best_pos_prob, is_hard = self.store.analyze_hero_pos_type(cid, HARD_THRESH)

            # 2a. 硬英雄位置冲突拦截 (防牛头+巴德)
            if is_hard and best_pos in occupied_hard_positions:
                n_filtered_hard += 1
                continue

            # 2a2. 主位置累计占用与生态位拥挤拦截
            if len(ally_champs) > 0:
                # 规则 1：不允许 3 个主位置相同
                if ally_primary_counts[best_pos] >= 2:
                    n_filtered_flex += 1
                    continue
                # 规则 2：概率累计期望拦截（阈值 2.0）
                if best_pos_prob >= 0.4 and (ally_pos_sum[best_pos] + best_pos_prob > 2.0):
                    n_filtered_flex += 1
                    continue
                # 规则 3：二分图可行性兜底
                if not self.check_comp_feasibility(ally_champs, cid, min_prob=0.15):
                    n_filtered_flex += 1
                    continue

            filtered.append((cid, score))
            if len(filtered) >= top_k:
                break

        if n_filtered_hard > 0 or n_filtered_flex > 0:
            log.info(f"  Pick position filter: {n_filtered_hard} hard, {n_filtered_flex} flex deadlocks filtered out")

        return [(cid, score, new_rank + 1) for new_rank, (cid, score) in enumerate(filtered)]

    def _filter_ban_duplicates(self, results, enemy_champs, top_k=50):
        """Layer 3: 敌方 Ban 推荐列表过滤器。

        规则: 若敌方阵容里已经选了纯硬英雄 (prob >= HARD_HERO_THRESHOLD) 某位置，
        敌方该位置需求已被封死，后续所有该位置的纯硬英雄 Ban 选项直接剔除。
        注意：Ban 列表内部【不需要】去重，因为集中 Ban 某一位置（如封锁上单池）是合理战术。
        """
        HARD_THRESH = self.store.HARD_HERO_THRESHOLD

        # 1. 记录敌方已选阵容的纯硬位置 (敌方选了诺手 -> 敌方 Top 位饱合)
        enemy_filled_positions = set()
        for c in enemy_champs:
            best_pos, _, is_hard = self.store.analyze_hero_pos_type(c, HARD_THRESH)
            if is_hard:
                enemy_filled_positions.add(best_pos)

        # 2. 仅过滤与敌方已确定位置冲突的候选，不限制推荐列表内部的重复
        filtered = []
        n_filtered = 0

        for cid, score, rank in results:
            best_pos, _, is_hard = self.store.analyze_hero_pos_type(cid, HARD_THRESH)

            if is_hard:
                # 冲突拦截: 敌方已经选了该位置纯硬英雄，无需再浪费 Ban 位
                if best_pos in enemy_filled_positions:
                    n_filtered += 1
                    continue
                # 注意: Ban 列表内部不去重，集中 Ban 某一位置是合理战术

            filtered.append((cid, score))
            if len(filtered) >= top_k:
                break

        if n_filtered > 0:
            log.info(f"  Ban position filter: {n_filtered} hard heroes filtered out to avoid wasted bans")

        return [(cid, score, new_rank + 1) for new_rank, (cid, score) in enumerate(filtered)]

    def _run_inference(self, action, step, ally_champs, enemy_champs,
                       ally_pids, enemy_pids, cand_np, mask_np, last_ally_pos, position,
                       bp_seq_ids, global_context, unavail_set):
        """在信号量保护下执行推理（资源隔离）

        使用传入的 session 快照数据，避免推理期间被其他线程修改。
        返回 (results, is_fallback) 元组，is_fallback 标识是否触发了规则兜底。
        """
        with self._inference_semaphore:
            if action == "pick":
                if self.fallback_manager is not None:
                    prev_fallback_count = self.fallback_manager._fallback_count
                    results = self.fallback_manager.predict_pick(
                        bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                        global_context, cand_np, mask_np, step, last_ally_pos,
                        position=position, ally_pids=ally_pids, enemy_pids=enemy_pids,
                    )
                    is_fallback = self.fallback_manager._fallback_count > prev_fallback_count
                    return results, is_fallback
                else:
                    return self.recommender.predict_pick(
                        bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                        global_context, cand_np, mask_np, step, last_ally_pos,
                    ), False
            else:
                if self.fallback_manager is not None:
                    prev_fallback_count = self.fallback_manager._fallback_count
                    results = self.fallback_manager.predict_ban(
                        bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                        global_context, cand_np, mask_np, step,
                        position=position, ally_pids=ally_pids, enemy_pids=enemy_pids,
                    )
                    is_fallback = self.fallback_manager._fallback_count > prev_fallback_count
                    return results, is_fallback
                else:
                    return self.recommender.predict_ban(
                        bp_seq_ids, ally_champs, enemy_champs, unavail_set,
                        global_context, cand_np, mask_np, step,
                    ), False

    # ---- 内部方法 ----

    def _build_champion_list(self, vocab_data):
        champion_list = []
        cs = self.store.champion_start_idx
        for champ in vocab_data["champions"]:
            if champ["idx"] < cs:
                continue
            name = champ["name"]
            cn_name = ""
            aliases = champ.get("aliases", {})
            if "zh" in aliases:
                cn_name = aliases["zh"]
            elif "cn" in aliases:
                cn_name = aliases["cn"]

            pos_probs = {}
            if name in self.position_mapping:
                for item in self.position_mapping[name]:
                    pos_name = item.get("position", "")
                    prob = item.get("probability", 0.0)
                    if pos_name in POS_FULL and prob > 0.05:
                        pos_probs[POS_FULL[pos_name]] = round(prob, 3)

            # 新英雄(如Locke)的idx可能超出meta_matrix范围，使用默认值兜底
            if champ["idx"] < len(self.store.meta_matrix):
                meta = self.store.meta_matrix[champ["idx"]]
                display_pr = float(meta[0])
                display_br = float(meta[1])
                display_presence = float(meta[2])
                display_wr = float(meta[3])
            else:
                display_pr = 0.0
                display_br = 0.0
                display_presence = 0.0
                display_wr = 0.5

            display_stats = self._get_display_stats(name)
            if display_stats:
                display_pr = display_stats["pick_rate"]
                display_br = display_stats["ban_rate"]
                display_presence = display_stats["presence_rate"]
                display_wr = display_stats["win_rate"]

            # 为中间英雄池构建简短理由标签
            pool_reasons = []
            if display_wr > 0.55:
                pool_reasons.append(f"版本强势({display_wr*100:.0f}%)")
            elif display_wr > 0.4:
                pool_reasons.append(f"高胜率({display_wr*100:.0f}%)")
            if display_presence >= 0.2:
                pool_reasons.append(f"高登场率({display_presence*100:.0f}%)")
            if display_br > 0.2:
                pool_reasons.append(f"高禁用率({display_br*100:.0f}%)")

            champion_list.append({
                "name": name,
                "cn_name": cn_name,
                "idx": champ["idx"],
                "positions": pos_probs,
                "meta_pick_rate": round(display_pr, 4),
                "meta_ban_rate": round(display_br, 4),
                "meta_presence": round(display_presence, 4),
                "meta_win_rate": round(display_wr, 4),
                "pool_reasons": pool_reasons,
            })

        champion_list.sort(key=lambda c: c["meta_presence"], reverse=True)
        return champion_list

    def _format_recommendations(self, results, action, ally_champs=None, is_fallback=False):
        """格式化推荐结果，含位置冲突检测和推荐理由"""
        if ally_champs is None:
            ally_champs = []

        # 计算友方已填充的位置（基于已选英雄的位置先验）
        ally_filled_positions = set()
        ally_pos_sum = np.zeros(5, dtype=np.float32)
        for c in ally_champs:
            # 新英雄(如Locke)的idx可能超出pos_prior范围，使用均匀分布兜底
            if c < len(self.store.pos_prior):
                pos_prior = self.store.pos_prior[c]
            else:
                pos_prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
            ally_pos_sum += pos_prior
            best_pos_idx = int(np.argmax(pos_prior))
            best_pos = ["top", "jungle", "mid", "bot", "support"][best_pos_idx]
            # 展示层位置冲突判定与 Layer 1/3 阈值一致
            if float(pos_prior[best_pos_idx]) >= self.store.HARD_HERO_THRESHOLD:
                ally_filled_positions.add(best_pos)

        recommendations = []
        for cid, score, rank in results:
            name = self.idx_to_name.get(cid, "???")
            # 新英雄(如Locke)的idx可能超出矩阵范围，使用默认值兜底
            if cid < len(self.store.meta_matrix):
                meta = self.store.meta_matrix[cid]
            else:
                meta = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)
            if cid < len(self.store.pos_prior):
                pos_prior = self.store.pos_prior[cid]
            else:
                pos_prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

            best_pos_idx = int(np.argmax(pos_prior))
            best_pos = ["top", "jungle", "mid", "bot", "support"][best_pos_idx]
            best_pos_prob = float(pos_prior[best_pos_idx])

            pos_cn_name = POS_CN.get(best_pos, best_pos)
            pos_conflict = False
            assigned_pos_prob = best_pos_prob

            if best_pos in ally_filled_positions:
                sorted_pos = np.argsort(-pos_prior)
                found_alt = False
                for pi in sorted_pos:
                    pn = ["top", "jungle", "mid", "bot", "support"][pi]
                    if pn not in ally_filled_positions and float(pos_prior[pi]) > 0.15:
                        assigned_pos_prob = float(pos_prior[pi])
                        pos_cn_name = POS_CN.get(pn, pn)
                        found_alt = True
                        break
                if not found_alt:
                    pos_conflict = True

            display_pr = float(meta[0])
            display_br = float(meta[1])
            display_presence = float(meta[2])
            display_wr = float(meta[3])
            display_stats = self._get_display_stats(name, best_pos)
            if display_stats:
                display_pr = display_stats["pick_rate"]
                display_br = display_stats["ban_rate"]
                display_presence = display_stats["presence_rate"]
                display_wr = display_stats["win_rate"]

            rec = {
                "rank": rank,
                "champion": name,
                "champion_idx": int(cid),
                "score": round(float(score), 4),
                "meta_pick_rate": round(display_pr, 4),
                "meta_ban_rate": round(display_br, 4),
                "meta_presence": round(display_presence, 4),
                "meta_win_rate": round(display_wr, 4),
                "best_position": best_pos,
                "best_position_prob": round(best_pos_prob, 3),
                "position_conflict": pos_conflict,
            }

            reasons = []
            if action == "pick":
                # 位置适配判定与硬英雄阈值一致
                if assigned_pos_prob >= self.store.HARD_HERO_THRESHOLD:
                    reasons.append(f"位置适配({pos_cn_name})")
                elif pos_conflict:
                    reasons.append("战术摇摆")
                # 胜率理由 (放宽至 >0.4, 让更多英雄有解释)
                if display_wr > 0.55:
                    reasons.append(f"版本强势({display_wr*100:.0f}%)")
                elif display_wr > 0.4:
                    reasons.append(f"高胜率({display_wr*100:.0f}%)")
                # 登场率理由 (>=0.2 即视为高登场)
                if display_presence >= 0.2:
                    reasons.append(f"高登场率({display_presence*100:.0f}%)")
                # 禁用率理由
                if display_br > 0.2:
                    reasons.append(f"高禁用率({display_br*100:.0f}%)")
            elif action == "ban":
                if display_br > 0.2: reasons.append(f"高禁用率({display_br*100:.0f}%)")
                if display_presence >= 0.2: reasons.append(f"高登场率({display_presence*100:.0f}%)")
                if display_wr > 0.55:
                    reasons.append(f"版本强势({display_wr*100:.0f}%)")
                elif display_wr > 0.4:
                    reasons.append(f"高胜率({display_wr*100:.0f}%)")

            rec["reasons"] = reasons
            recommendations.append(rec)
        return recommendations

    def _load_league_teams(self):
        """从 active_rosters.csv 加载联赛-战队映射

        active_rosters.csv 是现役名单（来自 Liquipedia），用于前端 UI 白名单
        和推理时的输入上下文。
        """
        import pandas as pd
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


