"""
BP 胜负预测模型 - 生产模型更新脚本
====================================
更新原始数据后运行此脚本, 自动完成生产模型的全量更新。

生产更新流程:
  Step 0: 环境检查与依赖验证
  Step 1: 特征工程 (feature_pipeline.py → features/)
  Step 2: 训练 5-Fold OOT Transformer 快照 (extract_production_transformer.py → tf_features/)
  Step 2.5: OOT 验证训练 (train_walk_forward.py → reports/production_iterations_source.json)
            ↑ 开发模式: 记录每折 best_iteration, 取最后 3 折均值
  Step 3: 生产模型训练 (train_production.py → models/production/)
            ↑ 生产模式: 读取 OOT best_iteration, 按 √n×0.65 补偿计算固定轮数,
              关闭 early stopping, 用 100% 数据训练 (方案 B)
  Step 4: 推理验证 (predict_backend.py 端到端测试)

开发/生产模式参数传递 (方案 B, 与推荐模型逻辑一致):
  开发模式 (Step 2.5):
    - 5-Fold OOT 验证, 评估泛化性能
    - 记录每折各 seed 的 best_iteration
    - 取最后 3 折均值作为 base_iterations
    - 保存到 reports/production_iterations_source.json
  生产模式 (Step 3):
    - 读取 base_iterations
    - 按 √n 补偿: production_iterations = base × (n_prod_eff / n_fold) ^ 0.5
    - 考虑时间衰减: n_prod_eff = n_prod × 0.65
    - LR × 0.85 作为正则化补偿
    - 关闭 early stopping, 用 100% 数据训练固定轮数

重要说明:
  - TF 特征使用 5-Fold OOT 快照提取, 防止数据泄漏
  - 每折独立训练 NoCS Transformer, 仅使用训练窗口内的数据
  - 生产模型训练时合并各折的 OOF (Out-Of-Fold) TF 特征
  - 自验证使用 OOT 方式 (训练时排除测试期), 避免 AUC 虚高

目录结构:
  bp_prediction/
  ├── run_training.py                  ← 本脚本 (生产更新入口)
  ├── feature_pipeline.py              ← 特征工程
  ├── export_production_transformer.py ← 从推荐模型导出 TF 快照
  ├── train_production.py              ← 生产模型训练 (方案 B: OOT 驱动固定轮数)
  ├── feature_builder.py               ← 统一特征构建 (推理用)
  ├── predict_backend.py               ← 后端推理服务
  ├── predict_match.py                 ← 命令行推理
  ├── bp_delta.py                      ← BP 影响分析
  ├── features/                        ← 特征数据 (训练+推理共用)
  │   ├── ALL_prediction_wide_features.parquet  (主训练数据)
  │   ├── ALL_meta_store.parquet                (英雄元数据)
  │   ├── ALL_player_store.parquet              (选手历史特征)
  │   └── ALL_team_profile_store.parquet        (队伍画像)
  ├── tf_snapshots/                    ← Transformer 快照 (推理用)
  │   └── production_nocs.pt           (生产快照, 从推荐模型导出)
  ├── tf_features/                     ← TF 特征 (训练用)
  │   └── production_tf_features.parquet
  ├── models/
  │   └── production/                  ← 生产模型 (推理用)
  │       ├── catboost_seed_0~6.cbm
  │       └── feature_columns.json
  ├── reports/                         ← OOT 验证报告 + 生产参数源
  │   └── production_iterations_source.json  (方案 B: OOT best_iteration)
  └── training/                        ← OOT 训练验证 (开发模式)
      ├── train_walk_forward.py        ← 记录 best_iteration 并保存参数源
      └── extract_transformer_features.py

用法:
  python run_training.py                  # 完整生产更新 (自动检测最新数据日期)
  python run_training.py --skip-features  # 跳过特征工程
  python run_training.py --skip-tf        # 跳过TF快照导出
  python run_training.py --skip-training  # 跳过模型训练
  python run_training.py --skip-all       # 仅运行推理验证
  python run_training.py --cutoff 2026-06-07  # 指定数据截止日期
"""

import os
import sys
import logging
import subprocess
import argparse
import importlib
from datetime import datetime
from pathlib import Path

# =====================================================================
# 路径配置 (必须在 logger_config 导入前设置 sys.path)
# =====================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(Path(__file__).parent.parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logger_config import get_logger, setup_logging, log_context, timed

FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

FEATURES_DIR = os.path.join(MODEL_DIR, "features")
TF_FEATURES_DIR = os.path.join(MODEL_DIR, "tf_features")
TF_SNAPSHOTS_DIR = os.path.join(MODEL_DIR, "tf_snapshots")
PRODUCTION_DIR = os.path.join(MODEL_DIR, "models", "production")
LOGS_DIR = os.path.join(MODEL_DIR, "logs")

# =====================================================================
# 日志
# =====================================================================
log = get_logger(__name__)

# =====================================================================
# Step 0: 环境检查
# =====================================================================
def check_environment():
    """检查 Python 版本和关键依赖包。"""
    log.info("\n%s", "="*70)
    log.info("  Step 0: 环境检查与依赖验证")
    log.info("%s", "="*70)

    py_version = sys.version_info
    log.info("Python 版本: %s.%s.%s", py_version.major, py_version.minor, py_version.micro)

    required_packages = {
        "pandas": "1.5", "numpy": "1.23", "catboost": "1.2",
        "sklearn": "1.2", "torch": "2.0", "scipy": "1.10",
    }
    missing = []
    for pkg, min_ver in required_packages.items():
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            ver = getattr(mod, "__version__", "unknown")
            log.info("  %s: %s", pkg, ver)
        except ImportError:
            missing.append(pkg)
            log.error("  %s: 未安装!", pkg)

    if missing:
        log.error("缺少依赖包: %s", ', '.join(missing))
        return False

    # 数据目录检查
    cleaned_data_dir = os.path.join(PROJECT_ROOT, "cleaned_data")
    if os.path.exists(cleaned_data_dir):
        log.info("清洗数据目录: %s (存在)", cleaned_data_dir)
    else:
        log.error("清洗数据目录: %s (不存在)", cleaned_data_dir)
        return False

    # bp_recommendation 检查 (TF 快照导出依赖)
    bp_rec_dir = os.path.join(PROJECT_ROOT, "bp_recommendation")
    nocs_ckpt = os.path.join(bp_rec_dir, "model_pick", "checkpoints", "best_model_nocs.pt")
    if os.path.exists(nocs_ckpt):
        log.info("推荐模型 NoCS checkpoint: (存在)")
    else:
        log.warning("推荐模型 NoCS checkpoint: (不存在, TF 快照导出将失败)")

    log.info("环境检查完成")
    return True

# =====================================================================
# Step 1: 特征工程
# =====================================================================
def run_feature_pipeline():
    """运行 feature_pipeline.py 生成 features/ 目录下的全部特征文件。"""
    log.info("\n%s", "="*70)
    log.info("  Step 1: 特征工程 (feature_pipeline.py)")
    log.info("%s", "="*70)

    wide_features_path = os.path.join(FEATURES_DIR, "ALL_prediction_wide_features.parquet")
    if os.path.exists(wide_features_path):
        import pandas as pd
        df = pd.read_parquet(wide_features_path)
        log.info("已有特征文件: %s 条记录, 日期范围: %s ~ %s",
            len(df),
            pd.to_datetime(df['date']).min().strftime('%Y-%m-%d'),
            pd.to_datetime(df['date']).max().strftime('%Y-%m-%d'))

    pipeline_script = os.path.join(MODEL_DIR, "feature_pipeline.py")
    if not os.path.exists(pipeline_script):
        log.error("feature_pipeline.py 不存在: %s", pipeline_script)
        return False

    log.info("运行 feature_pipeline.py (可能需要几分钟)...")
    try:
        result = subprocess.run(
            [sys.executable, pipeline_script],
            cwd=MODEL_DIR, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            log.error("特征工程失败:\n%s", result.stderr[-500:])
            return False
        log.info("特征工程完成")
        return True
    except subprocess.TimeoutExpired:
        log.error("特征工程超时 (>30分钟)")
        return False
    except Exception as e:
        log.exception("特征工程异常: %s", e)
        return False

# =====================================================================
# Step 2: 训练 5-Fold OOT Transformer 快照 + 提取 TF 特征
# =====================================================================
def run_oot_transformer_training(cutoff_date=None):
    """运行 extract_transformer_features.py 训练 5-Fold OOT 快照并提取 TF 特征。

    这是防止 TF 特征数据泄漏的关键步骤:
      - 每折独立训练 NoCS Transformer, 仅使用该折训练窗口内的数据
      - 对训练集和测试集分别提取 TF 特征
      - 生产模型训练时合并各折的 OOF (Out-Of-Fold) 特征
    """
    log.info("\n%s", "="*70)
    log.info("  Step 2: 训练 5-Fold OOT Transformer 快照 (extract_transformer_features.py)")
    log.info("%s", "="*70)

    extract_script = os.path.join(MODEL_DIR, "training", "extract_transformer_features.py")
    if not os.path.exists(extract_script):
        log.error("extract_transformer_features.py 不存在: %s", extract_script)
        return False

    # 检查推荐模型 checkpoint (NoCS Transformer 初始化权重)
    bp_rec_dir = os.path.join(PROJECT_ROOT, "bp_recommendation")
    nocs_ckpt = os.path.join(bp_rec_dir, "model_pick", "checkpoints", "best_model_nocs.pt")
    if not os.path.exists(nocs_ckpt):
        log.error("推荐模型 NoCS checkpoint 不存在, TF 快照训练将失败")
        return False

    cmd = [sys.executable, extract_script]
    if cutoff_date:
        cmd.extend(["--cutoff", cutoff_date])

    log.info("运行 extract_transformer_features.py (5折 PIT 训练, 可能需要较长时间)...")
    try:
        result = subprocess.run(
            cmd,
            cwd=MODEL_DIR, capture_output=True, text=True, timeout=7200,
        )
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-20:]:
                log.info("  %s", line)

        if result.returncode != 0:
            log.error("OOT 快照训练失败:\n%s", result.stderr[-500:])
            return False

        # 验证 TF 特征文件
        all_ok = True
        for fold_idx in range(5):
            tf_path = os.path.join(TF_FEATURES_DIR, f"{fold_idx}_tf_features.parquet")
            if os.path.exists(tf_path):
                import pandas as pd
                tf_df = pd.read_parquet(tf_path)
                n_train = (tf_df["split"] == "train").sum() if "split" in tf_df.columns else 0
                n_test = (tf_df["split"] == "test").sum() if "split" in tf_df.columns else 0
                log.info("  Fold %s: %s train + %s test TF features", fold_idx, n_train, n_test)
            else:
                log.warning("  Fold %s: TF 特征文件缺失", fold_idx)
                all_ok = False

        # 验证快照文件
        for fold_idx in range(5):
            snap_path = os.path.join(TF_SNAPSHOTS_DIR, f"fold_{fold_idx}_nocs.pt")
            if os.path.exists(snap_path):
                log.info("  Fold %s 快照: 存在", fold_idx)
            else:
                log.warning("  Fold %s 快照: 缺失", fold_idx)

        return all_ok
    except subprocess.TimeoutExpired:
        log.error("OOT 快照训练超时 (>120分钟)")
        return False
    except Exception as e:
        log.exception("OOT 快照训练异常: %s", e)
        return False

# =====================================================================
# Step 3: 生产模型训练
# =====================================================================
def run_production_training(cutoff_date=None, min_date=None):
    """运行 train_production.py 训练生产 CatBoost 模型 (方案 B 生产模式).

    方案 B 生产模式:
      - 读取 OOT 验证产出的 reports/production_iterations_source.json
      - 按 √n×0.65 补偿计算固定训练轮数
      - 关闭 early stopping, 用 100% 数据训练
      - LR × 0.85 作为正则化补偿
      - 若 OOT 参数文件不存在, 自动回退到 80/20 early stopping 模式
    """
    log.info("\n%s", "="*70)
    log.info("  Step 3: 生产模型训练 (train_production.py - 方案 B)")
    log.info("%s", "="*70)

    wide_features_path = os.path.join(FEATURES_DIR, "ALL_prediction_wide_features.parquet")
    if not os.path.exists(wide_features_path):
        log.error("训练数据不存在: %s", wide_features_path)
        return False

    train_script = os.path.join(MODEL_DIR, "train_production.py")
    if not os.path.exists(train_script):
        log.error("train_production.py 不存在: %s", train_script)
        return False

    # 检查 OOT 参数源文件 (方案 B)
    oot_params_path = os.path.join(MODEL_DIR, "reports", "production_iterations_source.json")
    if os.path.exists(oot_params_path):
        log.info("OOT 参数源存在: %s (方案 B 启用)", oot_params_path)
    else:
        log.warning("OOT 参数源不存在: %s", oot_params_path)
        log.warning("  生产模式将回退到 80/20 early stopping (旧模式)")
        log.warning("  建议先运行 OOT 验证 (Step 2.5) 以启用方案 B")

    cmd = [sys.executable, train_script]
    if cutoff_date:
        cmd.extend(["--cutoff", cutoff_date])
    if min_date:
        cmd.extend(["--min-date", min_date])

    log.info("运行 train_production.py (7-Seed Bagging, 可能需要较长时间)...")
    try:
        result = subprocess.run(
            cmd, cwd=MODEL_DIR, capture_output=True, text=True, timeout=3600,
        )
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-30:]:
                log.info("  %s", line)

        if result.returncode != 0:
            log.error("模型训练失败:\n%s", result.stderr[-500:])
            return False

        # 验证模型文件
        model_files = [f for f in os.listdir(PRODUCTION_DIR) if f.endswith(".cbm")] if os.path.exists(PRODUCTION_DIR) else []
        if len(model_files) >= 7:
            log.info("生产模型已生成: %s 个 .cbm 文件", len(model_files))
        else:
            log.warning("生产模型不完整: 仅 %s 个 .cbm 文件", len(model_files))

        # 验证 feature_columns.json
        fc_path = os.path.join(PRODUCTION_DIR, "feature_columns.json")
        if os.path.exists(fc_path):
            import json
            with open(fc_path, "r") as f:
                cols = json.load(f)
            log.info("特征列文件: %s 维", len(cols))
        else:
            log.warning("feature_columns.json 未生成")

        return True
    except subprocess.TimeoutExpired:
        log.error("模型训练超时 (>60分钟)")
        return False
    except Exception as e:
        log.exception("模型训练异常: %s", e)
        return False

# =====================================================================
# Step 2.5: OOT 验证训练 (训练模式)
# =====================================================================
def run_oot_validation(cutoff_date=None):
    """运行 train_walk_forward.py 进行 5-Fold Rolling OOT 验证。

    这是"训练模式"的核心步骤 (方案 B 开发模式):
      - 5-Fold Rolling OOT 验证 (12个月训练窗口)
      - 输出 AUC/LogLoss/Brier 等验证指标
      - 记录每折各 seed 的 best_iteration
      - 取最后 3 折均值, 保存到 reports/production_iterations_source.json
      - 生产模式 (Step 3) 读取此文件计算固定训练轮数

    与 Step 3 (生产模型训练) 的区别:
      - Step 2.5 (开发模式): OOT 验证, 评估模型性能 + 产出 best_iteration 参数
      - Step 3 (生产模式): 读取 best_iteration, 按 √n×0.65 补偿, 100% 数据盲训
    """
    log.info("\n%s", "="*70)
    log.info("  Step 2.5: OOT 验证训练 (train_walk_forward.py)")
    log.info("%s", "="*70)

    oot_script = os.path.join(MODEL_DIR, "training", "train_walk_forward.py")
    if not os.path.exists(oot_script):
        log.error("train_walk_forward.py 不存在: %s", oot_script)
        return False

    cmd = [sys.executable, oot_script]
    if cutoff_date:
        cmd.extend(["--cutoff", cutoff_date])

    log.info("运行 train_walk_forward.py (5-Fold OOT 验证, 可能需要较长时间)...")
    try:
        result = subprocess.run(
            cmd,
            cwd=MODEL_DIR, capture_output=True, text=True, timeout=7200,
        )
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-30:]:
                log.info("  %s", line)

        if result.returncode != 0:
            log.error("OOT 验证训练失败:\n%s", result.stderr[-500:])
            return False

        log.info("OOT 验证训练完成")
        return True
    except subprocess.TimeoutExpired:
        log.error("OOT 验证训练超时 (>120分钟)")
        return False
    except Exception as e:
        log.exception("OOT 验证训练异常: %s", e)
        return False

# =====================================================================
# Step 4: 推理验证
# =====================================================================
def run_inference_test():
    """端到端推理验证: 确保生产模型和特征构建正常工作。"""
    log.info("\n%s", "="*70)
    log.info("  Step 4: 推理验证 (端到端测试)")
    log.info("%s", "="*70)

    try:
        # predict_backend.py 使用绝对导入 (from bp_prediction.xxx import)，
        # 需要同时将项目根目录和 bp_prediction 目录加入 sys.path
        sys.path.insert(0, PROJECT_ROOT)
        sys.path.insert(0, MODEL_DIR)
        from predict_backend import PredictBackend

        backend = PredictBackend()
        load_result = backend.load()
        # load() 返回 dict {"success": True/False, ...} 或 bool
        if isinstance(load_result, dict):
            load_ok = load_result.get("success", False)
        else:
            load_ok = bool(load_result)
        if not load_ok:
            log.error("Backend 加载失败: %s", load_result)
            return False
        log.info("Backend 加载成功")

        # 测试预测
        request = {
            "league": "LCK", "is_playoff": False, "first_pick": "blue",
            "blue_team": "Gen.G", "red_team": "T1",
            "blue_champions": {"top": "Ornn", "jungle": "Lee Sin", "mid": "Azir",
                               "bot": "Jinx", "support": "Nautilus"},
            "red_champions": {"top": "Renekton", "jungle": "Viego", "mid": "Orianna",
                              "bot": "Varus", "support": "Rakan"},
            "blue_players": {}, "red_players": {},
        }
        pred = backend.predict(request)
        if "error" in pred:
            log.error("预测失败: %s", pred['error'])
            return False
        log.info("预测结果: blue=%s, red=%s", pred['blue_win_prob'], pred['red_win_prob'])

        # 测试 BP Delta
        delta = backend.bp_delta(request)
        if "error" in delta:
            log.error("BP Delta 失败: %s", delta['error'])
            return False
        log.info("BP Delta: pre=%s, post=%s, delta=%s",
            delta['predraft']['blue_prob'],
            delta['postdraft']['blue_prob'],
            delta['delta'])

        # 检查 TF 特征是否非默认值
        from feature_builder import extract_tf_features_for_match
        match_info = {
            "league": "LCK", "is_playoff": False,
            "blue_team": "Gen.G", "red_team": "T1",
            "blue_champions": ["Ornn", "Lee Sin", "Azir", "Jinx", "Nautilus"],
            "red_champions": ["Renekton", "Viego", "Orianna", "Varus", "Rakan"],
        }
        tf_features = extract_tf_features_for_match(match_info)
        defaults = {"tf_win_logits": 0.0, "tf_cosine_sim": 0.5, "tf_blue_l2norm": 10.0, "tf_red_l2norm": 10.0}
        tf_is_default = all(abs(tf_features[k] - defaults[k]) < 0.01 for k in defaults)
        if tf_is_default:
            log.warning("TF 特征为默认值 (Transformer 快照可能未加载)")
        else:
            log.info("TF 特征已实时推理: win_logits=%.2f, cosine_sim=%.4f",
                tf_features['tf_win_logits'],
                tf_features['tf_cosine_sim'])

        return True
    except Exception as e:
        log.exception("推理验证异常: %s", e)
        return False

# =====================================================================
# 主流程
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BP 胜负预测模型 - 生产模型更新脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_training.py                       # 完整生产更新 (自动检测最新数据日期)
  python run_training.py --skip-features       # 跳过特征工程
  python run_training.py --cutoff 2026-06-07   # 指定数据截止日期
  python run_training.py --skip-all            # 仅运行推理验证
        """,
    )
    parser.add_argument("--skip-features", action="store_true",
                        help="跳过特征工程步骤 (features/ 已存在)")
    parser.add_argument("--skip-tf", action="store_true",
                        help="跳过 TF 快照导出步骤")
    parser.add_argument("--skip-training", action="store_true",
                        help="跳过模型训练步骤")
    parser.add_argument("--skip-all", action="store_true",
                        help="跳过所有训练步骤, 仅运行推理验证")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跳过推理验证步骤 (Step 4), 用于流水线中将推理验证作为独立阶段执行")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="数据截止日期 (格式: YYYY-MM-DD, 默认自动检测最新数据日期)")
    parser.add_argument("--min-date", type=str, default=None,
                        help="数据最早日期 (格式: YYYY-MM-DD, 早于此日期的数据将被排除)")
    args = parser.parse_args()

    setup_logging()
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"production_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FMT)
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    from config import resolve_cutoff_date
    resolved_cutoff = args.cutoff if args.cutoff is not None else resolve_cutoff_date()

    training_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("\n%s", "="*70)
    log.info("="*70)
    log.info("  BP 胜负预测模型 - 生产模型更新流水线")
    log.info("="*70)
    log.info("  Start Time: %s", training_start)
    if args.cutoff is None:
        log.info("  Data Cutoff: %s (自动检测最新数据)", resolved_cutoff)
    else:
        log.info("  Data Cutoff: %s (用户指定)", resolved_cutoff)
    log.info("  Min Date: %s", args.min_date or 'No limit')
    log.info("  Output Directory: %s", PRODUCTION_DIR)
    log.info("  Skip Features: %s", args.skip_features or args.skip_all)
    log.info("  Skip TF Snapshot: %s", args.skip_tf or args.skip_all)
    log.info("  Skip Training: %s", args.skip_training or args.skip_all)
    log.info("  Skip Inference: %s", args.skip_inference)
    log.info("%s", "="*70)

    # Step 0: 环境检查
    if not check_environment():
        log.error("环境检查失败, 请修复后重试")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        sys.exit(1)

    # Step 1: 特征工程
    if not (args.skip_features or args.skip_all):
        if not run_feature_pipeline():
            log.error("特征工程失败")
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            sys.exit(1)
    else:
        log.info("跳过特征工程步骤")

    # Step 2: 训练 5-Fold OOT Transformer 快照
    if not (args.skip_tf or args.skip_all):
        if not run_oot_transformer_training(cutoff_date=resolved_cutoff):
            log.warning("OOT 快照训练失败, 将不使用 TF 特征")
    else:
        log.info("跳过 OOT 快照训练步骤")

    # Step 2.5: OOT 验证训练 (训练模式)
    #   先进行 OOT 验证评估模型泛化性能, 再进行生产模型训练
    #   --skip-tf 时跳过 (生产模式快速更新)
    if not (args.skip_tf or args.skip_all):
        if not run_oot_validation(cutoff_date=resolved_cutoff):
            log.warning("OOT 验证训练失败, 继续生产模型训练")
    else:
        log.info("跳过 OOT 验证训练步骤")

    # Step 3: 生产模型训练
    if not (args.skip_training or args.skip_all):
        if not run_production_training(cutoff_date=resolved_cutoff, min_date=args.min_date):
            log.error("模型训练失败")
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            sys.exit(1)
    else:
        log.info("跳过模型训练步骤")

    # Step 4: 推理验证
    #   --skip-inference 时跳过, 用于流水线中将推理验证作为独立阶段执行
    if args.skip_inference:
        log.info("跳过推理验证步骤 (--skip-inference)")
    elif not run_inference_test():
        log.error("推理验证失败")
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        sys.exit(1)

    total_elapsed = (datetime.now() - datetime.strptime(training_start, "%Y-%m-%d %H:%M:%S")).total_seconds()
    log.info("\n%s", "="*70)
    log.info("="*70)
    log.info("  生产模型更新流水线完成!")
    log.info("="*70)
    log.info("  Total Time: %.1f minutes", total_elapsed / 60)
    log.info("  Model Directory: %s", PRODUCTION_DIR)
    if os.path.exists(PRODUCTION_DIR):
        model_files = [f for f in os.listdir(PRODUCTION_DIR) if f.endswith('.cbm')]
        log.info("  Model Files: %s CatBoost models saved", len(model_files))
        fc_path = os.path.join(PRODUCTION_DIR, "feature_columns.json")
        if os.path.exists(fc_path):
            log.info("  Feature columns: %s", fc_path)
        meta_path = os.path.join(PRODUCTION_DIR, "metadata.json")
        if os.path.exists(meta_path):
            log.info("  Training metadata: %s", meta_path)
    oot_params = os.path.join(MODEL_DIR, "reports", "production_iterations_source.json")
    if os.path.exists(oot_params):
        log.info("  OOT iterations source: %s", oot_params)
    log.info("  Log File: %s", log_path)
    log.info("%s", "="*70)

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

if __name__ == "__main__":
    main()
