"""
BP Delta 计算脚本
==================
量化 BP (Ban/Pick) 对比赛胜负的影响程度。

核心思路:
  1. Pre-Draft 胜率: 仅依赖战队/选手纸面硬实力, 将所有 draft 相关特征置零
  2. Post-Draft 胜率: 输入双方完整阵容, 使用全部特征
  3. BP Delta = Post-Draft - Pre-Draft

特征构建逻辑统一使用 feature_builder.py, 确保与训练时 feature_pipeline.py 完全一致。

用法:
  python bp_delta.py

依赖:
  - models/ 目录下有训练好的 CatBoost 模型
  - features/ 目录下有特征数据
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from logger_config import get_logger, setup_logging

FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

from bp_prediction.feature_builder import (
    POSITIONS, PLAYER_DEFAULTS, TEAM_PROFILE_DEFAULTS,
    MAX_UNKNOWN_PLAYERS_PER_TEAM, ROOKIE_PENALTY,
    DRAFT_KEYWORDS,
    load_feature_cols, load_feature_stores, load_champion_tags, load_known_champions,
    resolve_team_name, get_team_roster,
    build_single_match_features, build_predraft_features, classify_features,
    extract_tf_features_for_match,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

FEATURES_DIR = os.path.join(MODEL_DIR, "features")
MODELS_DIR = os.path.join(MODEL_DIR, "models")
PRODUCTION_DIR = os.path.join(MODEL_DIR, "models", "production")

POSITION_NAMES = {"top": "上单", "jng": "打野", "mid": "中单", "bot": "ADC", "sup": "辅助"}

log = get_logger(__name__)

def load_models(use_production=True):
    from catboost import CatBoostClassifier

    if use_production and os.path.exists(PRODUCTION_DIR):
        prod_models = []
        for seed_idx in range(7):
            model_path = os.path.join(PRODUCTION_DIR, f"catboost_seed_{seed_idx}.cbm")
            if os.path.exists(model_path):
                model = CatBoostClassifier()
                model.load_model(model_path)
                prod_models.append(model)
        if prod_models:
            return {"production": prod_models}

    models = {}
    for fold_idx in range(5):
        fold_dir = os.path.join(MODELS_DIR, f"fold_{fold_idx}")
        if not os.path.exists(fold_dir):
            continue
        fold_models = []
        for seed_idx in range(7):
            model_path = os.path.join(fold_dir, f"catboost_seed_{seed_idx}.cbm")
            if os.path.exists(model_path):
                model = CatBoostClassifier()
                model.load_model(model_path)
                fold_models.append(model)
        if fold_models:
            models[fold_idx] = fold_models
    return models


def predict_with_models(models, features_df):
    all_preds = []
    fold_details = {}
    feature_importances = []
    
    X_infer = features_df.values.astype(np.float32)
    X_infer = np.nan_to_num(X_infer, nan=0.0, posinf=0.0, neginf=0.0)

    for fold_key, fold_models in sorted(models.items()):
        fold_preds = []
        for model in fold_models:
            pred = model.predict_proba(X_infer)[0, 1]
            fold_preds.append(pred)
        fold_mean = float(np.mean(fold_preds))
        fold_details[fold_key] = {"mean_prob": fold_mean, "seed_preds": fold_preds}
        all_preds.append(fold_mean)

        if fold_models:
            importances = fold_models[-1].get_feature_importance()
            feature_importances.append(importances)

    final_prob = float(np.mean(all_preds))
    avg_importance = np.mean(feature_importances, axis=0) if feature_importances else np.zeros(features_df.shape[1])
    return final_prob, fold_details, avg_importance


def validate_league(league_str):
    league_str = league_str.strip().upper()
    valid = {"LPL", "LCK", "LEC", "LCS", "PCS", "VCS", "CBLOL", "WORLDS", "MSI"}
    if league_str not in valid:
        raise ValueError(f"无效联赛: {league_str}")
    return league_str


def validate_champion(champ_str, known_champions=None):
    champ_str = champ_str.strip()
    if not champ_str:
        raise ValueError("英雄名称不能为空")
    if known_champions and champ_str not in known_champions:
        close = [c for c in known_champions if c.lower().startswith(champ_str.lower())]
        if close:
            raise ValueError(f"未找到英雄 '{champ_str}', 你是否指: {', '.join(close[:3])}?")
        raise ValueError(f"未找到英雄 '{champ_str}', 请检查拼写")
    return champ_str


def validate_yes_no(input_str):
    input_str = input_str.strip().lower()
    if input_str in ("y", "yes", "是", "1"):
        return True
    elif input_str in ("n", "no", "否", "0"):
        return False
    raise ValueError("请输入 y/n")


def get_input(prompt, validator=None, retry_msg=None):
    while True:
        try:
            value = input(prompt)
            if validator:
                return validator(value)
            return value.strip()
        except ValueError as e:
            log.error("  [错误] %s", e)
            if retry_msg:
                log.info("  %s", retry_msg)
        except KeyboardInterrupt:
            log.info("\n  已取消")
            sys.exit(0)


def collect_match_info(known_champions):
    log.info("\n%s", "="*70)
    log.info("  BP Delta 计算 - 输入对局信息")
    log.info("%s", "="*70)

    match_info = {"mode": "full"}

    log.info("\n  --- 基本信息 ---")
    match_info["league"] = get_input(
        "  联赛 (LPL/LCK/LEC): ",
        validator=validate_league,
        retry_msg="支持: LPL, LCK, LEC, LCS, PCS, VCS, CBLOL, WORLDS, MSI"
    )
    match_info["is_playoff"] = get_input("  是否季后赛? (y/n): ", validator=validate_yes_no)
    match_info["is_blue_map_side"] = get_input("  蓝方是否为先选方? (y/n): ", validator=validate_yes_no)

    log.info("\n  --- 队伍信息 ---")
    match_info["blue_team"] = input("  蓝方队伍名称: ").strip() or "蓝方"
    match_info["red_team"] = input("  红方队伍名称: ").strip() or "红方"

    log.info("\n  --- 阵容选择 ---")
    log.info("  位置顺序: 上单 → 打野 → 中单 → ADC → 辅助")

    def champ_validator(s):
        return validate_champion(s, known_champions if known_champions else None)

    for side in ["blue", "red"]:
        side_name = "蓝方" if side == "blue" else "红方"
        team_name = match_info.get(f"{side}_team", side_name)
        log.info("\n  [%s (%s) 阵容]", team_name, side_name)
        champions = []
        for pos in POSITIONS:
            pos_name = POSITION_NAMES[pos]
            champ = get_input(f"    {pos_name}: ", validator=champ_validator if known_champions else None)
            champions.append(champ)
        match_info[f"{side}_champions"] = champions

    log.info("\n  --- 选手信息 ---")
    log.info("  输入选手ID获取历史特征; 输入 unknown 标记未知选手 (每队最多%d名)", MAX_UNKNOWN_PLAYERS_PER_TEAM)

    for side in ["blue", "red"]:
        side_name = "蓝方" if side == "blue" else "红方"
        team_name = match_info.get(f"{side}_team", side_name)
        unknown_positions = []

        log.info("\n  [%s (%s) 选手]", team_name, side_name)
        for pos_idx, pos in enumerate(POSITIONS):
            pos_name = POSITION_NAMES[pos]
            champ = match_info[f"{side}_champions"][pos_idx]
            player_id = input(f"    {pos_name} ({champ}) 选手ID: ").strip()

            if player_id.lower() in ("unknown", "unk", "?", "未知", "新秀"):
                unknown_positions.append(pos)
                player_id = ""
            match_info[f"{side}_{pos}_player_id"] = player_id

        if len(unknown_positions) > MAX_UNKNOWN_PLAYERS_PER_TEAM:
            log.warning("  [警告] %s有 %d 名未知选手, 超过限制 (%d), 将截断", side_name, len(unknown_positions), MAX_UNKNOWN_PLAYERS_PER_TEAM)
            unknown_positions = unknown_positions[:MAX_UNKNOWN_PLAYERS_PER_TEAM]

        match_info[f"{side}_unknown_positions"] = unknown_positions
        if unknown_positions:
            pos_names = [POSITION_NAMES[p] for p in unknown_positions]
            log.info("  [%s] 未知选手位置: %s → 将施加新秀惩罚", side_name, ', '.join(pos_names))

    return match_info


def display_bp_delta(match_info, predraft_prob, postdraft_prob, draft_cols, hard_cols, feature_cols,
                     predraft_importance, postdraft_importance, unknown_info=None):
    blue_team = match_info.get("blue_team", "蓝方") or "蓝方"
    red_team = match_info.get("red_team", "红方") or "红方"

    delta = postdraft_prob - predraft_prob

    if delta >= 0.005:
        delta_direction = f"BP 对蓝方 ({blue_team}) 有利"
        delta_color = "+"
    elif delta <= -0.005:
        delta_direction = f"BP 对红方 ({red_team}) 有利"
        delta_color = ""
    else:
        delta_direction = "BP 影响微弱, 双方阵容势均力敌"
        delta_color = ""

    log.info("\n%s", "="*70)
    log.info("  BP Delta 分析结果")
    log.info("%s", "="*70)

    blue_champs = match_info.get("blue_champions", [])
    red_champs = match_info.get("red_champions", [])
    log.info("\n  %s (蓝方): %s", blue_team, ' / '.join(blue_champs))
    log.info("  %s (红方): %s", red_team, ' / '.join(red_champs))
    league = match_info.get("league", "LPL")
    playoff = "季后赛" if match_info.get("is_playoff") else "常规赛"
    log.info("  联赛: %s | %s", league, playoff)

    if unknown_info:
        log.info("\n  [新秀惩罚] %d 位未知选手使用战队平均 × 惩罚系数", len(unknown_info))

    log.info("\n%s", "─"*70)
    log.info("  %-30s %12s %12s", "指标", "蓝方胜率", "红方胜率")
    log.info("  %s", "─"*70)
    log.info("  %-30s %11.1f%% %11.1f%%", "Pre-Draft (纸面硬实力)", predraft_prob*100, (1-predraft_prob)*100)
    log.info("  %-30s %11.1f%% %11.1f%%", "Post-Draft (含阵容)", postdraft_prob*100, (1-postdraft_prob)*100)
    log.info("  %s", "─"*70)
    log.info("  %-30s %s%10.1f%%", "BP Delta", delta_color, delta*100)
    log.info("  %s", "─"*70)

    log.info("\n  >>> %s", delta_direction)
    log.info("  >>> BP 价值: |Δ| = %.1%%", abs(delta))

    if abs(delta) >= 0.10:
        verdict = "极大影响 - BP 决定了比赛走向"
    elif abs(delta) >= 0.05:
        verdict = "显著影响 - 阵容优劣明显"
    elif abs(delta) >= 0.02:
        verdict = "中等影响 - 阵容有一定优劣势"
    else:
        verdict = "微弱影响 - 阵容基本均衡"
    log.info("  >>> 判定: %s", verdict)

    _display_feature_contribution(predraft_importance, postdraft_importance, feature_cols, draft_cols, hard_cols)

    _display_position_contribution(match_info)


def _display_feature_contribution(predraft_imp, postdraft_imp, feature_cols, draft_cols, hard_cols):
    if postdraft_imp is None or len(postdraft_imp) == 0:
        return

    total_imp = np.sum(postdraft_imp)
    if total_imp == 0:
        return

    draft_total = 0.0
    hard_total = 0.0
    for idx, col in enumerate(feature_cols):
        if idx < len(postdraft_imp):
            if col in draft_cols:
                draft_total += postdraft_imp[idx]
            else:
                hard_total += postdraft_imp[idx]

    draft_pct = draft_total / total_imp * 100
    hard_pct = hard_total / total_imp * 100

    log.info("\n%s", "="*70)
    log.info("  特征贡献分解")
    log.info("%s", "="*70)
    log.info("\n  %-25s %10s %8s", "类别", "贡献占比", "特征数")
    log.info("  %s", "─"*45)
    log.info("  %-25s %9.1f%% %8d", "Draft 相关 (阵容)", draft_pct, len(draft_cols))
    log.info("  %-25s %9.1f%% %8d", "纸面硬实力 (战队/选手)", hard_pct, len(hard_cols))

    draft_imp_list = []
    for idx, col in enumerate(feature_cols):
        if col in draft_cols and idx < len(postdraft_imp):
            draft_imp_list.append((col, postdraft_imp[idx]))

    draft_imp_list.sort(key=lambda x: x[1], reverse=True)

    log.info("\n  Top Draft 特征贡献:")
    log.info("  %4s %-45s %8s", "排名", "特征名", "贡献")
    log.info("  %s", "─"*60)
    for rank, (col, imp) in enumerate(draft_imp_list[:10]):
        imp_pct = imp / total_imp * 100
        log.info("  %4d %-45s %7.2f%%", rank+1, col, imp_pct)


def _display_position_contribution(match_info):
    champion_tags = load_champion_tags()
    blue_champs = match_info.get("blue_champions", [])
    red_champs = match_info.get("red_champions", [])

    if not blue_champs or not red_champs:
        return

    log.info("\n%s", "="*70)
    log.info("  各位置阵容强度对比")
    log.info("%s", "="*70)

    log.info("\n  %-6s %-15s %-15s %8s %8s %8s", "位置", "蓝方", "红方", "蓝方强度", "红方强度", "差值")
    log.info("  %s", "─"*60)

    blue_total = 0
    red_total = 0
    for pos_idx, pos in enumerate(POSITIONS):
        b_champ = blue_champs[pos_idx] if pos_idx < len(blue_champs) else "?"
        r_champ = red_champs[pos_idx] if pos_idx < len(red_champs) else "?"
        b_tags = champion_tags.get(b_champ, {})
        r_tags = champion_tags.get(r_champ, {})
        b_power = sum(b_tags.values()) if b_tags else 0
        r_power = sum(r_tags.values()) if r_tags else 0
        diff = b_power - r_power
        blue_total += b_power
        red_total += r_power
        pos_name = POSITION_NAMES[pos]
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        log.info("  %-6s %-15s %-15s %8d %8d %8s", pos_name, b_champ, r_champ, b_power, r_power, diff_str)

    total_diff = blue_total - red_total
    total_str = f"+{total_diff}" if total_diff > 0 else str(total_diff)
    log.info("  %s", "─"*60)
    log.info("  %-6s %-15s %-15s %8d %8d %8s", "合计", "", "", blue_total, red_total, total_str)


def main():
    setup_logging()
    os.makedirs(os.path.join(MODEL_DIR, "logs"), exist_ok=True)
    log_path = os.path.join(MODEL_DIR, "logs", f"bp_delta_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT)
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    log.info("%s", "="*70)
    log.info("  BP Delta 计算器")
    log.info("  量化 Ban/Pick 对比赛胜负的影响")
    log.info("%s", "="*70)

    log.info("\n  加载模型...")
    models = load_models(use_production=True)
    if not models:
        log.error("  [错误] 未找到训练好的模型, 请先运行训练脚本")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        sys.exit(1)
    model_type = "Production" if "production" in models else "OOT 5-Fold"
    total_models = sum(len(fm) for fm in models.values())
    log.info("  已加载 %s 模型, 共 %d 个 seed", model_type, total_models)

    log.info("\n  加载特征数据...")
    stores = load_feature_stores()
    for name, store in stores.items():
        log.info("    %s: %d 条记录", name, len(store))

    known_champions = load_known_champions()
    champion_tags = load_champion_tags()
    log.info("  已加载 %d 个英雄名称", len(known_champions))
    log.info("  已加载 %d 个英雄标签", len(champion_tags))

    feature_cols = load_feature_cols()
    if feature_cols is None:
        log.error("  [错误] 未找到特征列名文件")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        sys.exit(1)
    draft_cols, hard_cols = classify_features(feature_cols)
    log.info("\n  特征分类: Draft=%d | 硬实力=%d | 总计=%d", len(draft_cols), len(hard_cols), len(feature_cols))

    while True:
        match_info = collect_match_info(known_champions)

        tf_features = extract_tf_features_for_match(match_info)

        log.info("\n  [Step 1/2] 计算 Pre-Draft 胜率 (纸面硬实力)...")
        predraft_df = build_predraft_features(match_info, stores, champion_tags,
                                              feature_cols=feature_cols,
                                              tf_features=tf_features)
        if predraft_df is None:
            log.error("  [错误] Pre-Draft 特征构建失败")
            continue

        predraft_prob, predraft_details, predraft_imp = predict_with_models(models, predraft_df)
        log.info("  Pre-Draft: 蓝方 %.1f%% | 红方 %.1f%%", predraft_prob*100, (1-predraft_prob)*100)

        log.info("  [Step 2/2] 计算 Post-Draft 胜率 (含阵容)...")
        postdraft_df, unknown_info = build_single_match_features(
            match_info, stores, champion_tags, feature_cols=feature_cols,
            tf_features=tf_features
        )
        if postdraft_df is None:
            log.error("  [错误] Post-Draft 特征构建失败")
            continue

        postdraft_prob, postdraft_details, postdraft_imp = predict_with_models(models, postdraft_df)
        log.info("  Post-Draft: 蓝方 %.1f%% | 红方 %.1f%%", postdraft_prob*100, (1-postdraft_prob)*100)

        delta = postdraft_prob - predraft_prob
        log.info("  BP Delta: %+.1f%%", delta*100)

        display_bp_delta(
            match_info, predraft_prob, postdraft_prob,
            draft_cols, hard_cols, feature_cols,
            predraft_imp, postdraft_imp, unknown_info
        )

        again = input(f"\n  是否计算下一局? (y/n): ").strip().lower()
        if again not in ("y", "yes", "是"):
            break

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    main()
