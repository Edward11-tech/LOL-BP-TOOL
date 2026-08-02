"""
BP 胜负预测模型 - 单局验证测试脚本
====================================
交互式输入对局信息, 调用训练好的级联模型进行预测, 输出预测结果及特征权重分析。

支持两种运行模式:
  --mode production : 生产模式 (默认), 优先加载 models/production/ 生产模型
  --mode training   : 训练模式, 加载 models/fold_0~4/ OOT 折模型, 跳过特征监控

支持两种输入模式:
  1) 纯 Draft 模式: 不输入战队/选手信息, 纯基于 draft 阵容预测
  2) 完整模式: 输入战队名 + 阵容 + 选手 (选手可填 unknown, 每队最多 2 名)

特征构建逻辑统一使用 feature_builder.py, 确保与训练时 feature_pipeline.py 完全一致。

用法:
  python predict_match.py                          # 默认生产模式
  python predict_match.py --mode production        # 显式指定生产模式
  python predict_match.py --mode training          # 训练模式 (OOT 折模型)
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from logger_config import get_logger, setup_logging

FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

from bp_prediction.feature_builder import (
    POSITIONS, PLAYER_DEFAULTS, TEAM_PROFILE_DEFAULTS,
    MAX_UNKNOWN_PLAYERS_PER_TEAM, ROOKIE_PENALTY,
    load_feature_cols, load_feature_stores, load_champion_tags, load_known_champions,
    resolve_team_name, get_team_roster,
    build_single_match_features,
)

from bp_prediction.config import (
    Mode, get_mode, set_mode, get_config, is_production_mode, is_training_mode,
    PRODUCTION_DIR, MODELS_DIR, FEATURES_DIR, print_config_summary,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())

FEATURES_DIR = os.path.join(MODEL_DIR, "features")
MODELS_DIR = os.path.join(MODEL_DIR, "models")
PRODUCTION_DIR = os.path.join(MODEL_DIR, "models", "production")

POSITION_NAMES = {"top": "上单", "jng": "打野", "mid": "中单", "bot": "ADC", "sup": "辅助"}

log = get_logger(__name__)

def validate_league(league_str):
    league_str = league_str.strip().upper()
    valid = {"LPL", "LCK", "LEC", "LCS", "PCS", "VCS", "CBLOL", "WORLDS", "MSI"}
    if league_str not in valid:
        raise ValueError(f"无效联赛: {league_str}, 支持: {', '.join(sorted(valid))}")
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

def get_input_with_validation(prompt, validator=None, retry_msg=None):
    while True:
        try:
            value = input(prompt)
            if validator:
                return validator(value)
            return value.strip()
        except ValueError as e:
            if retry_msg:
                log.error("  [错误] %s", e)
                log.info("  %s", retry_msg)
            else:
                log.error("  [错误] %s, 请重新输入", e)
        except KeyboardInterrupt:
            log.info("\n  已取消输入")
            sys.exit(0)

def load_models(use_production=None):
    from catboost import CatBoostClassifier
    shared_cfg, _ = get_config()
    n_seeds = shared_cfg.n_seeds

    if use_production is None:
        use_production = is_production_mode()

    if use_production and os.path.exists(PRODUCTION_DIR):
        prod_models = []
        for seed_idx in range(n_seeds):
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
        for seed_idx in range(n_seeds):
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

    for fold_key, fold_models in sorted(models.items()):
        fold_preds = []
        for seed_idx, model in enumerate(fold_models):
            pred = model.predict_proba(features_df.values)[0, 1]
            fold_preds.append(pred)

        fold_mean = float(np.mean(fold_preds))
        fold_details[fold_key] = {
            "mean_prob": fold_mean,
            "seed_preds": fold_preds,
        }
        all_preds.append(fold_mean)

        if fold_models:
            importances = fold_models[-1].get_feature_importance()
            feature_importances.append(importances)

    final_prob = float(np.mean(all_preds))

    if feature_importances:
        avg_importance = np.mean(feature_importances, axis=0)
    else:
        avg_importance = np.zeros(features_df.shape[1])

    return final_prob, fold_details, avg_importance

def display_prediction_result(match_info, blue_prob, red_prob, fold_details, feature_importance, feature_cols, unknown_info=None):
    log.info("\n%s", "="*70)
    log.info("  预测结果")
    log.info("%s", "="*70)

    is_draft = match_info.get("mode") == "draft"
    if is_draft:
        log.info("\n  [纯 Draft 模式] 仅基于阵容预测, 无战队/选手历史信息")

    blue_team = match_info.get("blue_team", "蓝方") or "蓝方"
    red_team = match_info.get("red_team", "红方") or "红方"
    winner = blue_team if blue_prob > red_prob else red_team
    confidence = abs(blue_prob - red_prob)

    log.info("\n  %s (蓝方) 胜率: %.1%%", blue_team, blue_prob * 100)
    log.info("  %s (红方) 胜率: %.1%%", red_team, red_prob * 100)
    log.info("\n  >>> 预测胜方: %s (置信度: %.1%%)", winner, confidence * 100)

    if unknown_info:
        log.info("\n  [新秀惩罚] 以下位置使用了战队平均特征 × 惩罚系数:")
        for info in unknown_info:
            side_name = "蓝方" if info["side"] == "blue" else "红方"
            pos_name = POSITION_NAMES.get(info["pos"], info["pos"])
            team = info["team"] or "未知战队"
            if team:
                log.info("    %s %s (%s) - 战队 %s 已知选手平均 × 新秀惩罚", side_name, pos_name, info['champion'], team)
            else:
                log.info("    %s %s (%s) - 使用默认值 (无新秀惩罚)", side_name, pos_name, info['champion'])

    fold_preds = [d["mean_prob"] for d in fold_details.values()]
    if len(fold_preds) > 1:
        fold_std = np.std(fold_preds)
        log.info("\n  各折预测标准差: %.4f (%s)", fold_std, '一致' if fold_std < 0.03 else '分歧较大')

    log.info("\n  各模型预测详情:")
    for fold_key, detail in sorted(fold_details.items()):
        seed_str = ", ".join([f"{p:.3f}" for p in detail["seed_preds"]])
        label = "Production" if fold_key == "production" else f"Fold {fold_key+1}"
        log.info("    %s: %.3f (seeds: [%s])", label, detail['mean_prob'], seed_str)

    log.info("\n%s", "="*70)
    log.info("  特征权重分析 (Top 20)")
    log.info("%s", "="*70)

    if feature_importance is not None and len(feature_importance) > 0:
        total_imp = np.sum(feature_importance)
        if total_imp > 0:
            norm_importance = feature_importance / total_imp * 100
        else:
            norm_importance = feature_importance

        indices = np.argsort(norm_importance)[::-1]
        feature_values = features_df.values[0] if features_df is not None else None

        log.info("\n  %4s %-45s %7s %10s", "排名", "特征名", "权重%", "当前值")
        log.info("  %s", "─"*70)

        for rank, idx in enumerate(indices[:20]):
            feat_name = feature_cols[idx] if idx < len(feature_cols) else f"feat_{idx}"
            imp = norm_importance[idx]
            val = feature_values[idx] if feature_values is not None and idx < len(feature_values) else 0.0
            val_str = f"{val:.4f}" if abs(val) < 1000 else f"{val:.0f}"
            log.info("  %4d %-45s %6.2f%% %10s", rank+1, feat_name, imp, val_str)

    log.info("\n%s", "="*70)
    log.info("  阵容对比分析")
    log.info("%s", "="*70)

    blue_champs = match_info.get("blue_champions", [])
    red_champs = match_info.get("red_champions", [])

    log.info("\n  蓝方阵容: %s", ' / '.join(blue_champs))
    log.info("  红方阵容: %s", ' / '.join(red_champs))

    champion_tags = load_champion_tags()
    log.info("\n  %-6s %-15s %-15s %8s %8s %8s", "位置", "蓝方", "红方", "蓝方强度", "红方强度", "差值")
    log.info("  %s", "─"*60)

    for pos_idx, pos in enumerate(POSITIONS):
        b_champ = blue_champs[pos_idx] if pos_idx < len(blue_champs) else "?"
        r_champ = red_champs[pos_idx] if pos_idx < len(red_champs) else "?"
        b_tags = champion_tags.get(b_champ, {})
        r_tags = champion_tags.get(r_champ, {})
        b_power = sum(b_tags.values()) if b_tags else 0
        r_power = sum(r_tags.values()) if r_tags else 0
        diff = b_power - r_power
        pos_name = POSITION_NAMES.get(pos, pos)
        log.info("  %-6s %-15s %-15s %8d %8d %+8d", pos_name, b_champ, r_champ, b_power, r_power, diff)

    log.info("\n%s", "="*70)

def collect_match_info(known_champions):
    log.info("\n%s", "="*70)
    log.info("  请输入对局信息")
    log.info("%s", "="*70)

    match_info = {}

    log.info("\n  --- 预测模式 ---")
    log.info("  1) 纯 Draft 模式: 仅输入阵容, 无战队/选手信息")
    log.info("  2) 完整模式: 输入战队+阵容+选手 (选手可填 unknown, 每队最多2名)")
    mode_choice = get_input_with_validation(
        "  选择模式 (1/2): ",
        validator=lambda x: "draft" if x.strip() in ("1", "draft") else "full",
        retry_msg="请输入 1 或 2"
    )
    match_info["mode"] = mode_choice

    log.info("\n  --- 基本信息 ---")
    match_info["league"] = get_input_with_validation(
        "  联赛 (LPL/LCK/LEC): ",
        validator=validate_league,
        retry_msg="支持: LPL, LCK, LEC, LCS, PCS, VCS, CBLOL, WORLDS, MSI"
    )
    match_info["is_playoff"] = get_input_with_validation(
        "  是否季后赛? (y/n): ",
        validator=validate_yes_no,
        retry_msg="请输入 y 或 n"
    )
    match_info["is_blue_map_side"] = get_input_with_validation(
        "  蓝方是否为先选方? (y/n): ",
        validator=validate_yes_no,
        retry_msg="请输入 y 或 n"
    )

    log.info("\n  --- 阵容选择 ---")
    log.info("  位置顺序: 上单 → 打野 → 中单 → ADC → 辅助")

    def champ_validator(champ_str):
        return validate_champion(champ_str, known_champions if known_champions else None)

    for side in ["blue", "red"]:
        side_name = "蓝方" if side == "blue" else "红方"
        log.info("\n  [%s 阵容]", side_name)
        champions = []
        for pos in POSITIONS:
            pos_name = POSITION_NAMES[pos]
            champ = get_input_with_validation(
                f"    {pos_name}: ",
                validator=champ_validator if known_champions else None,
                retry_msg="请输入有效的英雄名称"
            )
            champions.append(champ)
        match_info[f"{side}_champions"] = champions

    if mode_choice == "draft":
        match_info["blue_team"] = ""
        match_info["red_team"] = ""
        return match_info

    log.info("\n  --- 队伍信息 ---")
    match_info["blue_team"] = input("  蓝方队伍名称: ").strip() or ""
    match_info["red_team"] = input("  红方队伍名称: ").strip() or ""

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
            log.warning("\n  [错误] %s有 %d 名未知选手, 超过限制 (%d 名)", side_name, len(unknown_positions), MAX_UNKNOWN_PLAYERS_PER_TEAM)
            retry_unknown = []
            log.info("\n  [%s (%s) 重新输入选手]", team_name, side_name)
            for pos_idx, pos in enumerate(POSITIONS):
                pos_name = POSITION_NAMES[pos]
                champ = match_info[f"{side}_champions"][pos_idx]
                player_id = input(f"    {pos_name} ({champ}) 选手ID: ").strip()
                if player_id.lower() in ("unknown", "unk", "?", "未知", "新秀"):
                    retry_unknown.append(pos)
                    player_id = ""
                match_info[f"{side}_{pos}_player_id"] = player_id

            if len(retry_unknown) > MAX_UNKNOWN_PLAYERS_PER_TEAM:
                log.warning("  [错误] 仍然超过限制, 将强制将多余未知选手设为默认值")
                retry_unknown = retry_unknown[:MAX_UNKNOWN_PLAYERS_PER_TEAM]

            unknown_positions = retry_unknown

        match_info[f"{side}_unknown_positions"] = unknown_positions

        if unknown_positions:
            pos_names = [POSITION_NAMES[p] for p in unknown_positions]
            log.info("  [%s] 未知选手位置: %s → 将施加新秀惩罚", side_name, ', '.join(pos_names))

    return match_info

features_df = None

def main():
    global features_df

    parser = argparse.ArgumentParser(
        description="BP 胜负预测模型 - 单局验证测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, choices=["training", "production"], default="production",
        help="运行模式: training=OOT折模型验证, production=生产模型推理 (默认: production)",
    )
    args = parser.parse_args()

    setup_logging()
    os.makedirs(os.path.join(MODEL_DIR, "logs"), exist_ok=True)
    log_path = os.path.join(MODEL_DIR, "logs", f"predict_match_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT)
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    set_mode(Mode(args.mode))
    current_mode = get_mode()

    log.info("\n%s", "="*70)
    log.info("  BP 胜负预测模型 - 单局验证测试")
    log.info("  运行模式: %s", current_mode.value.upper())
    log.info("  模型: CatBoost-7Seed-Bagging")
    log.info("%s", "="*70)

    print_config_summary(current_mode)

    if not os.path.exists(MODELS_DIR):
        log.error("\n  [错误] 模型目录不存在, 请先运行训练脚本")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        sys.exit(1)

    models = load_models()
    if not models:
        log.error("\n  [错误] 未找到训练好的模型文件")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        sys.exit(1)

    model_type = "Production" if "production" in models else "OOT 5-Fold"
    n_seeds = len(next(iter(models.values())))
    log.info("\n  已加载 %s 模型, %d 个 seed [%s]", model_type, n_seeds, current_mode.value)

    log.info("\n  加载特征数据...")
    stores = load_feature_stores()
    for name, df in stores.items():
        log.info("    %s: %d 条记录", name, len(df))

    known_champions = load_known_champions()
    if known_champions:
        log.info("  已加载 %d 个英雄名称", len(known_champions))

    champion_tags = load_champion_tags()
    log.info("  已加载 %d 个英雄标签", len(champion_tags))

    while True:
        match_info = collect_match_info(known_champions)

        log.info("\n  构建特征向量...")
        features_df, unknown_info = build_single_match_features(match_info, stores, champion_tags)
        if features_df is None:
            log.error("  [错误] 特征构建失败")
            continue

        mode_str = "纯 Draft" if match_info.get("mode") == "draft" else "完整"
        log.info("  模式: %s | 特征维度: %d", mode_str, features_df.shape[1])
        if unknown_info:
            log.info("  新秀惩罚: %d 位未知选手", len(unknown_info))

        log.info("  进行预测...")
        blue_prob, fold_details, feature_importance = predict_with_models(models, features_df)
        red_prob = 1.0 - blue_prob

        feature_cols = features_df.columns.tolist()
        display_prediction_result(match_info, blue_prob, red_prob, fold_details, feature_importance, feature_cols, unknown_info)

        try:
            cont = input("\n  是否预测下一局? (y/n): ").strip().lower()
            if cont not in ("y", "yes", "是", "1"):
                break
        except KeyboardInterrupt:
            break

    log.info("\n  感谢使用 BP 胜负预测模型!")
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

if __name__ == "__main__":
    main()
