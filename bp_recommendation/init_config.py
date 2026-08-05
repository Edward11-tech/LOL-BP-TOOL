"""
初始化配置模块
=============================================
用于首次部署或重置时，将开发模式超参搜索得到的最佳参数写入配置文件，
为生产环境训练提供初始的最佳超参数。

功能描述:
    - 初始化 Pick CS 模型最佳参数
    - 初始化 Pick NoCS 模型最佳参数
    - 初始化 Cascade Pick 模型最佳参数（LightGBM 融合权重）
    - 初始化 Ban CS 模型最佳参数
    - 初始化 Cascade Ban 模型最佳参数

主要函数:
    - seed_configs(): 执行配置初始化，灌入所有模型的最佳超参数

使用方法:
    cd <project_root>
    python -m bp_recommendation.init_config
    
    注意: 此脚本会覆盖 training_configs/ 目录下的配置文件，
    仅在首次部署或需要重置为已知最佳参数时运行。
"""
# init_config.py
import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT_DIR)

from logger_config import get_logger, setup_logging
from bp_recommendation.config import save_best_params

log = get_logger(__name__)

def seed_configs():
    log.info("开始初始化生产环境配置文件...")
    
    # ================================================================
    # 1. 灌入 Pick CS 模型的最佳参数
    #    来源: model_pick/logs/train_pick_cs_20260618_133044.log
    #    Best Pick@10=0.7389 @ Epoch 39, Early Stopping @ Epoch 64 (patience=25)
    # ================================================================
    save_best_params(
        model_type="pick", model_subtype="CS",
        best_epoch=39,
        best_metric=0.7389,
        best_metric_name="Pick@10",
        architecture={
            "h_dim": 384, "n_layers": 3, "n_heads": 16,
            "query_dim": 128, "c_dim": 128,
            "dropout": 0.1811891079, "attention_dropout": 0.1363890037,
            "candidate_hidden": 256, "tactical_hidden": 256,
        },
        optimizer={
            "learning_rate": 0.000353846126, "weight_decay": 0.009093929526,
            "warmup_ratio": 0.1528468877, "grad_clip": 1.388621853,
        },
        loss={
            "aux_loss_weight": 0.7090268572,
            "ban_sample_weight": 0.04050837781,
            "step6_downweight": 0.2534717113,
        }
    )
    
    # ================================================================
    # 2. 灌入 Pick NoCS 模型的最佳参数
    #    来源: model_pick/logs/train_pick_nocs_20260618_135318.log
    #    Best Pick@10=0.7419 @ Epoch 40, 训练完成 50 轮
    # ================================================================
    save_best_params(
        model_type="pick", model_subtype="NoCS",
        best_epoch=40,
        best_metric=0.7419,
        best_metric_name="Pick@10",
        architecture={
            "h_dim": 384, "n_layers": 3, "n_heads": 12,
            "query_dim": 128, "c_dim": 128,
            "dropout": 0.188211676, "attention_dropout": 0.1231182369,
            "candidate_hidden": 256, "tactical_hidden": 256,
        },
        optimizer={
            "learning_rate": 0.0004296709541, "weight_decay": 0.04650497671,
            "warmup_ratio": 0.2285677671, "grad_clip": 1.448441962,
        },
        loss={
            "aux_loss_weight": 1.543410388,
            "ban_sample_weight": 0.08233083651,
            "step6_downweight": 0.1831447945,
        }
    )
    
    # ================================================================
    # 3. 灌入 Cascade Pick 的最佳参数
    #    来源: model_pick/logs/cascade_pick_20260618_145708.log
    #    5-Fold CV best_iter: 1381, 1339, 1809, 1522, 1089 (avg=1428)
    #    Blend Alpha: 0.16
    # ================================================================
    save_best_params(
        model_type="pick", model_subtype="cascade",
        best_iteration=1428, blend_alpha=0.16,
    )
    
    # ================================================================
    # 4. 灌入 Ban CS 模型的最佳参数
    #    来源: model_ban/logs/prod_20260618_ban_cs_20260618_155845.log
    #    Best Ban@10=0.7966 @ Epoch 42, 训练完成 50 轮
    # ================================================================
    save_best_params(
        model_type="ban", model_subtype="CS",
        best_epoch=42,
        best_metric=0.7966,
        best_metric_name="Ban@10",
        architecture={
            "h_dim": 384, "n_layers": 6, "n_heads": 6,
            "query_dim": 256, "c_dim": 64,
            "dropout": 0.10041555, "attention_dropout": 0.1336216089,
        },
        optimizer={
            "learning_rate": 0.0002964419919, "weight_decay": 0.009017448969,
            "warmup_ratio": 0.1948678137, "grad_clip": 1.41486906,
        },
        loss={
            "aux_loss_weight": 1.242151319,
        }
    )
    
    # ================================================================
    # 5. 灌入 Cascade Ban 的最佳参数
    #    来源: model_ban/logs/cascade_ban_20260618_162505.log
    #    5-Fold CV best_iter: 33, 17, 113, 48, 50 (avg=52)
    #    Blend Alpha: 0.05
    # ================================================================
    save_best_params(
        model_type="ban", model_subtype="cascade",
        best_iteration=52, blend_alpha=0.05,
    )
    
    log.info("配置文件初始化成功！现在可以直接运行 python run_pipeline.py --production 了！")

if __name__ == "__main__":
    setup_logging()
    seed_configs()