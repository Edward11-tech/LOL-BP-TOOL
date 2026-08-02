"""
全局配置模块：开发/生产双模式开关 + 训练参数持久化

用法:
    from bp_recommendation.config import (
        is_production_mode, get_config, save_config,
        get_best_params, save_best_params,
        get_model_config, save_model_config,
    )
"""

import os
import sys
import json
import time
import platform
import logging
from pathlib import Path
from datetime import datetime

# ============================================================
# 全局开关：生产模式 (默认 True)
# ============================================================
def is_production_mode():
    """动态检查当前是否处于生产模式"""
    return os.environ.get("BP_PRODUCTION_MODE", "false").lower() in ("true", "1", "yes")

# ============================================================
# 路径配置
# ============================================================
_PROJECT_ROOT = str(Path(__file__).parent.resolve())
CONFIG_DIR = os.path.join(_PROJECT_ROOT, "training_configs")
os.makedirs(CONFIG_DIR, exist_ok=True)


def _get_config_path(model_type, model_subtype=None):
    """
    获取指定模型类型的配置文件路径。
    支持按 subtype 区分，避免 CS/NoCS/cascade 互相覆盖。
    
    Args:
        model_type: "pick" 或 "ban"
        model_subtype: "CS", "NoCS", "cascade" 等 (可选)
    
    Returns:
        str: 配置文件路径
    """
    if model_subtype:
        filename = f"{model_type}_{model_subtype.lower()}_training_config.json"
    else:
        filename = f"{model_type}_training_config.json"
    return os.path.join(CONFIG_DIR, filename)

log = logging.getLogger("BPConfig")
log.setLevel(logging.INFO)

# ============================================================
# 默认配置模板
# ============================================================
DEFAULT_CONFIG_TEMPLATE = {
    # --- 数据预处理参数 ---
    "data_preprocessing": {
        "feature_normalizer": "StandardScaler",  # 标准化方法
        "scaler_coefficients": None,             # 将在 dev 模式训练后填充
        "categorical_encoding": "identity",      # 类别特征编码方式
        "feature_dimensions": {
            "context_dim": None,                 # 全局上下文维度
            "candidate_dim": None,               # 候选矩阵特征维度
        },
    },

    # --- Transformer 模型参数 ---
    "transformer": {
        "best_epoch": None,                      # 超参搜索记录的最佳 epoch（开发模式验证集最优）
        "num_epochs": None,                      # 生产模式实际训练轮数（= best_epoch，由 save_best_params 同步写入）
        "best_metric": None,                     # 最佳验证指标值
        "best_metric_name": None,                # 最佳验证指标名称 (e.g., "Pick@10", "Ban@10")

        # 架构参数
        "architecture": {
            "h_dim": None,
            "c_dim": None,
            "query_dim": None,
            "n_layers": None,
            "n_heads": None,
            "dropout": None,
            "attention_dropout": None,
            # Pick 特有
            "candidate_hidden": None,
            "tactical_hidden": None,
        },

        # 优化器参数
        "optimizer": {
            "optimizer_type": "AdamW",
            "learning_rate": None,
            "weight_decay": None,
            "warmup_ratio": None,
            "grad_clip": None,
            "scheduler_type": "cosine_with_warmup",
        },

        # 正则化参数
        "regularization": {
            "l1_coefficient": 0.0,
            "l2_coefficient": None,              # 由 weight_decay 体现
            "dropout": None,
            "attention_dropout": None,
            "label_smoothing": 0.0,
        },

        # 损失函数参数
        "loss": {
            "aux_loss_weight": None,             # 辅助损失权重
            "ban_sample_weight": None,           # Ban 样本权重 (Pick 模型)
            "step6_downweight": None,            # Step 6 降权 (Pick 模型)
        },

        # 训练配置
        "training": {
            "batch_size": None,
            "patience": None,                    # Early Stopping 耐心值
            "val_ratio": 0.15,                   # 验证集比例 (dev 模式)
            "seed": 42,
            "use_amp": False,                    # 混合精度
            "use_compile": False,                # torch.compile
            "num_workers": 0,                    # DataLoader 工作进程数
        },
    },

    # --- 树模型 (LightGBM/CatBoost) 参数 ---
    "tree_model": {
        "best_iteration": None,                  # 开发模式记录的最佳迭代次数
        "num_boost_round": None,                 # 生产模式硬编码的迭代次数
        "blend_alpha": None,                     # 模型融合权重系数
        "n_folds": 5,                            # 交叉验证折数

        # LightGBM 超参数
        "lgb_config": {
            "objective": "rank_xendcg",
            "metric": "ndcg",
            "num_leaves": None,
            "max_depth": None,
            "min_data_in_leaf": None,
            "learning_rate": None,
            "feature_fraction": None,
            "bagging_fraction": None,
            "bagging_freq": None,
            "lambda_l1": None,
            "lambda_l2": None,
            "seed": 42,
        },

        "training": {
            "num_round": None,                   # 开发模式最大迭代轮数
            "early_stop": None,                  # 开发模式 Early Stopping 轮数
            "top_k_recall": None,                # 候选池大小
        },
    },

    # --- 训练环境参数 ---
    "environment": {
        "training_timestamp": None,              # 训练时间戳
        "python_version": None,                  # Python 版本
        "pytorch_version": None,                 # PyTorch 版本
        "lightgbm_version": None,                # LightGBM 版本
        "numpy_version": None,                   # NumPy 版本
        "hardware": {
            "platform": None,                    # 操作系统
            "cpu_count": None,                   # CPU 核心数
            "gpu_type": None,                    # GPU 类型
            "gpu_memory_gb": None,              # GPU 显存 (GB)
            "total_ram_gb": None,               # 总内存 (GB)
        },
        "training_duration_sec": None,           # 训练总耗时 (秒)
        "dataset_size": {
            "train_samples": None,               # 训练样本数
            "val_samples": None,                 # 验证样本数 (dev 模式)
            "total_samples": None,               # 总样本数 (生产模式)
        },
    },

    # --- 元信息 ---
    "meta": {
        "config_version": "1.0.0",
        "model_type": None,                      # "pick" or "ban"
        "model_subtype": None,                   # "CS" or "NoCS"
        "production_mode": None,                 # 是否处于生产模式
        "last_updated": None,                    # 最后更新时间
        "description": "",                       # 描述信息
    },
}


def _detect_hardware():
    """自动检测硬件配置信息"""
    info = {
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "gpu_type": "None",
        "gpu_memory_gb": 0,
        "total_ram_gb": 0,
    }

    # 尝试检测 GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_type"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_type"] = "Apple Silicon (MPS)"
    except ImportError:
        pass

    # 尝试检测内存
    try:
        import psutil
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    return info


def _detect_versions():
    """自动检测依赖库版本"""
    versions = {
        "python_version": sys.version.split()[0],
    }
    try:
        import torch
        versions["pytorch_version"] = torch.__version__
    except ImportError:
        pass
    try:
        import lightgbm
        versions["lightgbm_version"] = lightgbm.__version__
    except ImportError:
        pass
    try:
        import numpy
        versions["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    return versions


def _deep_merge(base, override):
    """深度合并两个字典，override 中的值会覆盖 base 中的值"""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _validate_config(config):
    """校验配置参数的合法性和一致性"""
    errors = []

    # 检查关键参数是否存在
    transformer = config.get("transformer", {})
    if transformer.get("num_epochs") is not None and transformer["num_epochs"] <= 0:
        errors.append("transformer.num_epochs 必须大于 0")

    tree = config.get("tree_model", {})
    if tree.get("num_boost_round") is not None and tree["num_boost_round"] <= 0:
        errors.append("tree_model.num_boost_round 必须大于 0")

    if tree.get("blend_alpha") is not None and not (0.0 <= tree["blend_alpha"] <= 1.0):
        errors.append("tree_model.blend_alpha 必须在 [0.0, 1.0] 范围内")

    # 检查优化器参数
    opt = transformer.get("optimizer", {})
    if opt.get("learning_rate") is not None and opt["learning_rate"] <= 0:
        errors.append("transformer.optimizer.learning_rate 必须大于 0")

    if errors:
        error_msg = "配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors)
        log.error(error_msg)
        raise ValueError(error_msg)

    return True


# ============================================================
# 公共 API
# ============================================================

def get_config(model_type, model_subtype=None):
    """
    获取指定模型的训练配置。
    
    Args:
        model_type: "pick" 或 "ban"
        model_subtype: "CS" 或 "NoCS" (可选)
    
    Returns:
        dict: 配置字典
    """
    if model_type == "pick":
        config_file = _get_config_path("pick", model_subtype)
    elif model_type == "ban":
        config_file = _get_config_path("ban", model_subtype)
    else:
        raise ValueError(f"未知的 model_type: {model_type}，支持 'pick' 或 'ban'")

    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        log.info(f"已加载 {model_type} 模型配置: {config_file}")
        return config
    else:
        log.warning(f"配置文件不存在: {config_file}，返回默认模板")
        config = json.loads(json.dumps(DEFAULT_CONFIG_TEMPLATE))
        config["meta"]["model_type"] = model_type
        config["meta"]["model_subtype"] = model_subtype
        return config


def save_config(config, model_type, model_subtype=None):
    """
    保存模型训练配置到持久化文件。
    
    Args:
        config: 配置字典
        model_type: "pick" 或 "ban"
        model_subtype: "CS" 或 "NoCS" 等 (可选)
    """
    if model_subtype is None:
        model_subtype = config.get("meta", {}).get("model_subtype")

    if model_type == "pick":
        config_file = _get_config_path("pick", model_subtype)
    elif model_type == "ban":
        config_file = _get_config_path("ban", model_subtype)
    else:
        raise ValueError(f"未知的 model_type: {model_type}")

    # 更新元信息
    config["meta"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["meta"]["production_mode"] = is_production_mode()

    # 校验配置
    _validate_config(config)

    # 写入文件
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log.info(f"配置已保存到: {config_file}")
    return config_file


def get_best_params(model_type, model_subtype=None):
    """
    获取开发模式记录的最佳训练参数。
    用于生产模式加载。
    
    Returns:
        dict: {best_epoch, best_iteration, blend_alpha, ...}
    """
    config = get_config(model_type, model_subtype)
    transformer = config.get("transformer", {})
    tree_model = config.get("tree_model", {})

    best_params = {
        "best_epoch": transformer.get("best_epoch"),
        "best_iteration": tree_model.get("best_iteration"),
        "blend_alpha": tree_model.get("blend_alpha"),
        "best_metric": transformer.get("best_metric"),
        "best_metric_name": transformer.get("best_metric_name"),
    }

    # 校验关键参数
    missing = [k for k, v in best_params.items() if v is None]
    if missing:
        log.warning(f"最佳参数不完整，缺失字段: {missing}")

    return best_params


def save_best_params(
    model_type,
    best_epoch=None,
    best_iteration=None,
    blend_alpha=None,
    best_metric=None,
    best_metric_name=None,
    model_subtype=None,
    **extra_params,
):
    """
    保存开发模式训练得到的最佳参数。
    
    Args:
        model_type: "pick" 或 "ban"
        best_epoch: Transformer 最佳训练轮次
        best_iteration: LightGBM 最佳迭代次数
        blend_alpha: 模型融合权重系数
        best_metric: 最佳验证指标值
        best_metric_name: 最佳验证指标名称
        model_subtype: "CS" 或 "NoCS"
        **extra_params: 其他需要持久化的参数
    """
    config = get_config(model_type, model_subtype)

    # 更新 Transformer 最佳参数
    if best_epoch is not None:
        config["transformer"]["best_epoch"] = int(best_epoch)
        config["transformer"]["num_epochs"] = int(best_epoch)  # 生产模式硬编码值
    if best_metric is not None:
        config["transformer"]["best_metric"] = round(float(best_metric), 4)
    if best_metric_name is not None:
        config["transformer"]["best_metric_name"] = str(best_metric_name)

    # 更新树模型最佳参数
    if best_iteration is not None:
        config["tree_model"]["best_iteration"] = int(best_iteration)
        # 生产模式：按数据量增加比例 (+10%) 调整
        production_rounds = int(best_iteration * 1.1)
        config["tree_model"]["num_boost_round"] = production_rounds
    if blend_alpha is not None:
        config["tree_model"]["blend_alpha"] = round(float(blend_alpha), 4)

    # 更新额外参数
    for key, value in extra_params.items():
        if value is None:
            continue
        if key in config["transformer"]:
            if isinstance(config["transformer"][key], dict):
                config["transformer"][key].update(value if isinstance(value, dict) else {})
            else:
                config["transformer"][key] = value
        elif key in config["tree_model"]:
            if isinstance(config["tree_model"][key], dict):
                config["tree_model"][key].update(value if isinstance(value, dict) else {})
            else:
                config["tree_model"][key] = value

    save_config(config, model_type)
    return config


def get_model_config(model_type, model_subtype=None):
    """获取完整的模型配置（包括架构参数、优化器参数等）"""
    config = get_config(model_type, model_subtype)
    return config.get("transformer", {})


def save_model_config(model_type, model_config, model_subtype=None):
    """保存模型架构和训练配置"""
    config = get_config(model_type, model_subtype)
    config["transformer"] = _deep_merge(config["transformer"], model_config)
    save_config(config, model_type)
    return config


def record_training_environment(config, model_type, train_samples, val_samples,
                                 training_duration_sec):
    """记录训练环境信息（训练完成后调用）"""
    config["environment"].update(_detect_hardware())
    config["environment"].update(_detect_versions())
    config["environment"]["training_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["environment"]["training_duration_sec"] = round(training_duration_sec, 1)
    config["environment"]["dataset_size"]["train_samples"] = int(train_samples)
    config["environment"]["dataset_size"]["val_samples"] = int(val_samples)
    config["environment"]["dataset_size"]["total_samples"] = int(train_samples) + int(val_samples)

    save_config(config, model_type)
    return config


def record_feature_dimensions(model_type, context_dim, candidate_dim, model_subtype=None):
    """记录特征矩阵维度到配置文件，供推理时校验一致性。

    Args:
        model_type: "pick" 或 "ban"
        context_dim: 全局上下文向量维度
        candidate_dim: 候选矩阵特征维度
        model_subtype: "CS" / "NoCS" / "cascade"
    """
    config = get_config(model_type, model_subtype)
    config["data_preprocessing"]["feature_dimensions"]["context_dim"] = int(context_dim)
    config["data_preprocessing"]["feature_dimensions"]["candidate_dim"] = int(candidate_dim)
    save_config(config, model_type)
    log.info(f"  Feature dimensions recorded: context_dim={context_dim}, candidate_dim={candidate_dim}")
    return config


def record_scaler_coefficients(model_type, scaler, model_subtype=None):
    """将 StandardScaler 的系数序列化保存到配置文件。

    Args:
        model_type: "pick" 或 "ban"
        scaler: sklearn.preprocessing.StandardScaler 实例
        model_subtype: "cascade" 等
    """
    config = get_config(model_type, model_subtype)
    config["data_preprocessing"]["scaler_coefficients"] = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    save_config(config, model_type)
    log.info(f"  Scaler coefficients recorded: n_features={scaler.n_features_in_}")
    return config


def record_production_params(model_type, model_subtype=None, **kwargs):
    """记录生产模式训练实际使用的参数到配置文件的 production 字段。

    与开发模式的 best_params 分离，便于审计生产环境实际运行配置。

    Args:
        model_type: "pick" 或 "ban"
        model_subtype: "CS" / "NoCS" / "cascade"
        **kwargs: 实际使用的参数，如 best_epoch, num_epochs, best_iteration, blend_alpha, val_ratio, train_samples
    """
    config = get_config(model_type, model_subtype)
    if "production" not in config:
        config["production"] = {}
    config["production"].update({
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_subtype": model_subtype,
    })
    config["production"].update(kwargs)
    save_config(config, model_type)
    log.info(f"  Production params recorded for [{model_type}/{model_subtype}]: {list(kwargs.keys())}")
    return config


def get_production_val_ratio():
    """
    获取生产模式下的 val_ratio。
    生产模式：val_ratio = 0.001（仅使用最后一天的一场比赛作为最小验证集）
    """
    if is_production_mode():
        return 0.001
    return 0.15


def get_production_num_epochs(config):
    """
    获取生产模式下的 Transformer 训练轮数。

    语义说明:
    - best_epoch: 超参搜索阶段验证集最优 epoch，由 save_best_params 写入
    - num_epochs: 生产模式实际训练轮数，由 save_best_params 同步设置为 best_epoch
    生产模式优先读取 best_epoch（权威源），开发模式读取 training.num_epochs（默认 80）。
    """
    if not is_production_mode():
        return config.get("transformer", {}).get("training", {}).get("num_epochs", 80)

    best_epoch = config.get("transformer", {}).get("best_epoch")
    if best_epoch is None:
        log.warning("生产模式下 best_epoch 未配置，使用默认值 80")
        return 80
    return int(best_epoch)


def get_production_num_boost_round(config):
    """
    获取生产模式下的树模型迭代次数。
    从配置中读取 best_iteration，按 1.1 倍调整。
    """
    if not is_production_mode():
        return config.get("tree_model", {}).get("training", {}).get("num_round", 4500)

    best_iteration = config.get("tree_model", {}).get("best_iteration")
    if best_iteration is None:
        log.warning("生产模式下 best_iteration 未配置，使用默认值 4500")
        return 4500
    return int(best_iteration * 1.1)


def get_production_blend_alpha(config):
    """
    获取生产模式下的融合权重系数。
    从配置中读取 blend_alpha，若不存在则使用默认值。
    """
    if not is_production_mode():
        return config.get("tree_model", {}).get("blend_alpha", 0.18)

    blend_alpha = config.get("tree_model", {}).get("blend_alpha")
    if blend_alpha is None:
        log.warning("生产模式下 blend_alpha 未配置，使用默认值 0.18")
        return 0.18
    return float(blend_alpha)


def print_config_summary(config):
    """打印配置摘要信息"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  训练配置摘要")
    lines.append("=" * 70)

    meta = config.get("meta", {})
    lines.append(f"  模式: {'生产模式' if meta.get('production_mode') else '开发模式'}")
    lines.append(f"  模型类型: {meta.get('model_type', 'N/A')} ({meta.get('model_subtype', 'N/A')})")

    transformer = config.get("transformer", {})
    lines.append(f"  Transformer epochs: {transformer.get('num_epochs', 'N/A')} "
                 f"(best: {transformer.get('best_epoch', 'N/A')})")
    lines.append(f"  最佳指标: {transformer.get('best_metric_name', 'N/A')} = "
                 f"{transformer.get('best_metric', 'N/A')}")

    tree = config.get("tree_model", {})
    lines.append(f"  Tree boost rounds: {tree.get('num_boost_round', 'N/A')} "
                 f"(best_iter: {tree.get('best_iteration', 'N/A')})")
    lines.append(f"  Blend alpha: {tree.get('blend_alpha', 'N/A')}")

    env = config.get("environment", {})
    lines.append(f"  硬件: {env.get('hardware', {}).get('gpu_type', 'N/A')} "
                 f"({env.get('hardware', {}).get('platform', 'N/A')})")
    dataset = env.get("dataset_size", {})
    lines.append(f"  数据量: train={dataset.get('train_samples', 'N/A')}, "
                 f"val={dataset.get('val_samples', 'N/A')}")

    lines.append("=" * 70)

    summary = "\n".join(lines)
    log.info("\n" + summary)
    return summary


def generate_production_report(config, model_type, output_dir=None):
    """
    生成生产模式训练报告。
    包含训练时长、数据量、参数配置等信息。
    
    Args:
        config: 配置字典
        model_type: "pick" 或 "ban"
        output_dir: 报告输出目录，默认为 CONFIG_DIR
    
    Returns:
        str: 报告文件路径
    """
    if output_dir is None:
        output_dir = CONFIG_DIR

    report_path = os.path.join(output_dir, f"production_report_{model_type}.json")

    report = {
        "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "mode": "PRODUCTION",
        "timestamp": datetime.now().isoformat(),
    }

    # 元信息
    meta = config.get("meta", {})
    report["meta"] = {
        "model_type": meta.get("model_type"),
        "model_subtype": meta.get("model_subtype"),
        "last_updated": meta.get("last_updated"),
    }

    # Transformer 参数
    transformer = config.get("transformer", {})
    report["transformer"] = {
        "best_epoch": transformer.get("best_epoch"),
        "num_epochs": transformer.get("num_epochs"),
        "best_metric": transformer.get("best_metric"),
        "best_metric_name": transformer.get("best_metric_name"),
        "architecture": transformer.get("architecture", {}),
        "optimizer": transformer.get("optimizer", {}),
        "loss": transformer.get("loss", {}),
        "training": transformer.get("training", {}),
    }

    # 树模型参数
    tree = config.get("tree_model", {})
    report["tree_model"] = {
        "best_iteration": tree.get("best_iteration"),
        "num_boost_round": tree.get("num_boost_round"),
        "blend_alpha": tree.get("blend_alpha"),
        "architecture": tree.get("architecture", {}),
        "optimizer": tree.get("optimizer", {}),
        "loss": tree.get("loss", {}),
        "training": tree.get("training", {}),
    }

    # 数据预处理参数
    report["preprocessing"] = config.get("data_preprocessing", {})

    # 训练环境
    env = config.get("environment", {})
    report["environment"] = {
        "hardware": env.get("hardware", {}),
        # 将散落的版本信息收集起来
        "software": {
            "python": env.get("python_version"),
            "pytorch": env.get("pytorch_version"),
            "lightgbm": env.get("lightgbm_version"),
            "numpy": env.get("numpy_version"),
        },
        "training_timestamp": env.get("training_timestamp"),
        "training_duration_sec": env.get("training_duration_sec"),
        "dataset_size": env.get("dataset_size", {}),
    }

    # 参数校验
    report["validation"] = {
        "config_complete": _validate_config(config),
        "warnings": [],
    }

    # 写入文件
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(f"生产模式训练报告已生成: {report_path}")
    return report_path