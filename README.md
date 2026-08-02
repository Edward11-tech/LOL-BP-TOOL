# LOL BP 预测与推荐系统

英雄联盟（League of Legends）职业赛事 BP（Ban/Pick）预测与推荐系统。

本仓库**分享模型架构与工程思路**，不提供可复刻的完整运行环境（不含数据文件、模型权重、前端页面）。读者可通过阅读代码了解：

- 如何用 **Transformer + LightGBM 级联排序** 做英雄推荐
- 如何用 **CatBoost 集成 + PIT 隔离** 做比赛胜负预测
- 如何设计 **规则引擎兜底机制** 保证线上稳定性
- 如何用 **PSI 漂移检测** 监控模型性能

---

## 系统架构

```
                         ┌─────────────────────────────────────────────┐
                         │              数据来源 (外部)                  │
                         │                                             │
                         │  OraclesElixir                              │
                         │    └─ 比赛选手级长表 (gameid/champion/kills  │
                         │       /deaths/assists/result 等)            │
                         │                                             │
                         │  官方赛事公开数据                             │
                         │    └─ 选手职业生涯统计 / 英雄克制协同关系     │
                         │       / 英雄定位 / 战队阵容 / 选手归属        │
                         └────────────────────┬────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────┐
                    │         数据清洗 (PIT 隔离)              │
                    │  去除异常值 / 去重 / 标准化              │
                    │  贝叶斯平滑 / 时间衰减 / 实体解析         │
                    └────────────────────┬────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
   │  bp_recommendation  │  │   bp_prediction     │  │      fallback       │
   │  (BP 推荐)          │  │   (胜负预测)        │  │    (规则兜底)       │
   │                     │  │                     │  │                     │
   │ Transformer 召回    │  │ CatBoost 集成       │  │ Meta Presence       │
   │   (Pick/Ban 各一套) │  │   (7 seed × 5 fold) │  │ + Player Mastery     │
   │        ↓            │  │        +            │  │   规则引擎           │
   │ LightGBM 级联精排   │  │ Transformer 特征    │  │                     │
   │   (CS / NoCS 双模)  │  │   (NoCS 快照)       │  │ 触发器 A: Logit 坍塌 │
   │        ↓            │  │        ↓            │  │ 触发器 B: 滑动窗口   │
   │   Blend 融合输出    │  │   端到端概率输出    │  │                     │
   └─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │       后端推理接口 (Python 直接调用)      │
                    │                                         │
                    │  PredictBackend.predict()   比赛胜负预测 │
                    │  BPRecommendationBackend.recommend()  推荐│
                    │                                         │
                    │  ※ 本仓库仅提供后端推理接口              │
                    │    不含 Flask 服务 和 前端页面            │
                    │    调用方通过 Python 实例化后端获取结果   │
                    └────────────────────┬────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
               ┌──────────────────────┐  ┌──────────────────────┐
               │  monitoring/         │  │  logs/               │
               │  PSI 漂移检测        │  │  推理特征日志        │
               │  (每周一 cron)       │  │  (parquet, 按天滚动) │
               └──────────────────────┘  └──────────────────────┘
```

### 架构要点

| 层级 | 说明 |
|------|------|
| **数据层** | 原始数据来自 OraclesElixir 和官方赛事公开数据，经清洗后存入 `cleaned_data/`。所有特征工程严格遵循 Point-in-Time (PIT) 隔离，避免未来信息泄漏 |
| **模型层** | 推荐 = Transformer 召回 + LightGBM 级联精排；预测 = CatBoost 集成 + Transformer 深层特征。两个模块独立训练、独立推理 |
| **兜底层** | 当模型置信度坍塌或滚动指标跌破红线时，自动降级为 Rule-based 规则引擎，保证服务不中断 |
| **推理层** | 通过 Python 直接实例化后端类调用，不含 Flask 服务和前端页面 |
| **监控层** | PSI 特征漂移检测每周一自动运行，对比训练基线与线上推理特征分布 |

---

## 后端推理接口

本仓库不含 Flask 服务入口（`app.py`），调用方通过 Python 直接实例化后端类进行推理。

### 1. 胜负预测 — PredictBackend

**接口文件**：[bp_prediction/predict_backend.py](bp_prediction/predict_backend.py)

**使用方法**：

```python
from logger_config import setup_logging
from bp_prediction.predict_backend import PredictBackend

# 初始化
backend = PredictBackend()
backend.load()  # 加载模型 (耗时操作, 启动时调用一次)

# 构建请求
request = {
    "league": "LPL",              # 联赛
    "is_playoff": False,           # 是否季后赛
    "first_pick": "red",           # 首选方位于地图哪侧: "blue" 或 "red"
    "blue_team": "BLG",            # 首选方战队
    "red_team": "T1",             # 次选方战队
    "blue_champions": {            # 首选方阵容 (位置 -> 英雄名)
        "top": "Ornn",
        "jng": "Vi",
        "mid": "Azir",
        "bot": "Jinx",
        "sup": "Lulu",
    },
    "red_champions": {
        "top": "Gnar",
        "jng": "Sejuani",
        "mid": "Sylas",
        "bot": "Aphelios",
        "sup": "Thresh",
    },
    "blue_players": {              # 首选方选手 (可填 "unknown", 每队最多 2 名)
        "top": "Bin",
        "jng": "Xun",
        "mid": "Knight",
        "bot": "Viper",
        "sup": "ON",
    },
    "red_players": {
        "top": "Doran",
        "jng": "Oner",
        "mid": "Faker",
        "bot": "Peyz",
        "sup": "Keria",
    },
}

# 推理
result = backend.predict(request)
print(result)
# {
#     "blue_prob": 0.6234,        # 首选方 (BP 蓝方) 胜率
#     "fold_details": [...],      # 各折预测明细
#     "feature_importance": [...], # SHAP 特征重要性
#     ...
# }
```

**关键方法**：

| 方法 | 说明 |
|------|------|
| `PredictBackend()` | 实例化，配置并发控制、限流、兜底机制 |
| `.load()` | 加载模型和数据（耗时，启动时调用一次） |
| `.predict(request)` | 胜率预测，返回首选方胜率 + SHAP 可解释性 |

**内置机制**：
- 限流：滑动窗口控制（默认 60 秒内最多 N 次请求）
- 超时：独立线程池执行推理，超时自动中断
- 并发安全：Per-seed 推理锁（CatBoost predict_proba 非线程安全）
- 兜底：联赛置信度不足时回退规则引擎
- 缓存：Pre-Draft 结果缓存（BP 探索阶段减少重复计算）

### 2. BP 推荐 — BPRecommendationBackend

**接口文件**：[bp_recommendation/bp_recommendation_backend.py](bp_recommendation/bp_recommendation_backend.py)

**使用方法**：

```python
from logger_config import setup_logging
from bp_recommendation.bp_recommendation_backend import BPRecommendationBackend

# 初始化
backend = BPRecommendationBackend()
backend.load()  # 加载模型 (耗时操作, 启动时调用一次)

# 构建 payload
payload = {
    "league": "LPL",               # 联赛
    "blue_team": "BLG",             # 首选方战队
    "red_team": "T1",              # 次选方战队
    "playoffs": False,              # 是否季后赛
    "first_pick_map_side": 1.0,     # 首选方地图选边: 1.0=蓝色方, 0.0=红色方
    "game_num": 1,                  # 全局bp局次 (1-5)
    "completed_steps": 0,           # 已完成的 BP 步数 (0=第一步)
    "bp_seq_ids": [],               # 已选英雄 ID 序列 (按 BP_SEQUENCE 顺序)
    "unavail_set": [],              # 不可用英雄 ID 集合
    "pre_unavail_list": [],         # 前置局已用英雄 (Fearless Draft)
    "blue_pids": ["", "", "", "", ""],  # 首选方选手 ID (5个, 可空)
    "red_pids": ["", "", "", "", ""],   # 次选方选手 ID
}

# 推理
result = backend.recommend(payload)
print(result)
# {
#     "recommendations": [          # 推荐英雄列表 (Top-N)
#         {"champion": "Vi", "score": 0.92, ...},
#         ...
#     ],
#     "request_id": "a1b2c3d4e5f6",
#     ...
# }
```

**关键方法**：

| 方法 | 说明 |
|------|------|
| `BPRecommendationBackend()` | 实例化，配置并发控制、限流 |
| `.load()` | 加载 Transformer + LightGBM 模型、特征存储、兜底管理器 |
| `.recommend(payload)` | 无状态推理，输出当前 BP 步骤的 Pick/Ban 推荐 |

**内置机制**：
- 无状态：所有 BP 进度上下文从 payload 获取，便于水平扩展
- 限流 + 超时 + 资源隔离（同 PredictBackend）
- 兜底：FallbackManager 在模型异常时自动降级为规则引擎

### 交互式测试脚本

除后端类外，还提供两个交互式命令行脚本，便于快速验证：

| 脚本 | 说明 |
|------|------|
| [bp_prediction/predict_match.py](bp_prediction/predict_match.py) | 交互式输入对局信息，输出胜率预测 + 特征权重 |
| [bp_recommendation/bp_predict.py](bp_recommendation/bp_predict.py) | 模拟真实 BP 流程，逐步输出 Pick/Ban 推荐 Top-20 |

---

## 核心模块说明

### bp_recommendation/ — BP 推荐模块

英雄推荐（Pick/Ban）采用 **两阶段架构**：

1. **Transformer 召回**：`model_pick/` 和 `model_ban/` 各训练一个 BPTacticalTransformer，输入当前 BP 状态（已ban/pick的英雄、位置约束、选手信息），输出候选英雄的 logits
2. **LightGBM 级联精排**：`cascade_pick.py` / `cascade_ban.py` 对 Transformer 输出的候选集做重排序，使用 CS（Context-Aware Split）和 NoCS（No Context Split）两种模式
3. **Blend 融合**：最终通过 `routing_config.json` 配置的权重融合 CS/NoCS 结果

关键文件：
- [bp_recommendation_backend.py](bp_recommendation/bp_recommendation_backend.py) — **推理后端入口**
- [model_pick/model_pick.py](bp_recommendation/model_pick) — Pick 模型定义
- [model_ban/model_ban.py](bp_recommendation/model_ban) — Ban 模型定义
- [feature_pipeline.py](bp_recommendation/feature_pipeline.py) — 特征工程（含克制/协同/选手熟练度）

### bp_prediction/ — 胜负预测模块

比赛胜负预测采用 **CatBoost 集成 + Transformer 深层特征**：

1. **CatBoost 集成**：7 个随机种子 × 5 折 OOT（Out-of-Time）验证 = 35 个基模型，取平均
2. **Transformer 深层特征**：用 NoCS 快照提取 4 种深层特征（注意力权重、候选隐状态等），拼接到 CatBoost 的 wide features 上
3. **生产模型**：从 5 折中选出最优折，用全量数据重训，存入 `models/production/`

关键文件：
- [predict_backend.py](bp_prediction/predict_backend.py) — **推理后端入口**
- [training/](bp_prediction/training/) — 预测模型训练模式脚本
- [train_production.py](bp_prediction/train_production.py) — 预测模型生产模型脚本
- [feature_builder.py](bp_prediction/feature_builder.py) — 特征构建

### fallback/ — 规则兜底模块

当模型不可用或置信度不足时，自动降级为规则引擎：

- [rule_engine.py](fallback/rule_engine.py) — 基于 Meta Presence（英雄登场率）和 Player Mastery（选手熟练度）的规则排序
- [triggers.py](fallback/triggers.py) — 两个触发器：Logit 坍塌检测（瞬时）+ 滑动窗口指标监控（持续）
- [fallback_manager.py](fallback/fallback_manager.py) — 协调模型推理与规则兜底的切换
- [data_pipeline.py](fallback/data_pipeline.py) — 从 `cleaned_data/` 加载统计数据（带模块级缓存）

### monitoring/ — PSI 漂移检测

- [weekly_psi_check.py](monitoring/weekly_psi_check.py) — 对比训练基线与线上推理特征，计算 PSI（Population Stability Index）
- [setup_psi_cron.sh](monitoring/setup_psi_cron.sh) — cron 定时任务安装脚本（每周一 09:00）

### common/ — 共享模块

- [paths.py](common/paths.py) — 统一路径管理（Single Source of Truth）
- [psi.py](common/psi.py) — PSI 计算工具
- [inference_feature_logger.py](common/inference_feature_logger.py) — 推理特征日志记录（用于漂移检测）

---

## 数据来源

本系统不含任何数据文件。所需数据来自以下两类外部来源：

### 1. OraclesElixir（比赛数据）

| 项 | 说明 |
|----|------|
| **来源** | [OraclesElixir](https://oracleselixir.com/tools/downloads) |
| **格式** | CSV，选手级长表（每行 = 一个选手在一场比赛中的表现） |
| **字段** | gameid, date, league, teamname, playername, champion, position, kills, deaths, assists, result, damagetochampions, earnedgold, wardsplaced 等 |
| **覆盖** | 全球主要联赛（LPL / LCK / LEC / LCS 等），按年归档 |
| **用途** | 比赛胜负预测的特征工程、英雄 Meta 统计、选手英雄熟练度 |

### 2. 官方赛事公开数据（选手/英雄/战队）

| 项 | 说明 |
|----|------|
| **来源** | 赛事官方公开信息、游戏官方 Wiki |
| **字段** | 选手职业生涯统计（games / win_rate / KDA）、英雄克制关系、英雄协同关系、英雄定位映射、英雄评级、战队现役选手名单、战队-选手映射 |
| **用途** | 推荐模型的特征工程、规则引擎的 Meta 统计、推理时根据战队查询选手 |

### 数据清洗流程

```
外部数据源 → 去除异常值 / 去重 / 标准化
           → 英雄统计聚合 + 贝叶斯平滑
           → 选手/战队实体解析
           → PIT 隔离的特征文件
```

> **注意**：数据爬虫脚本和清洗脚本未包含在本仓库中。本仓库仅提供模型训练和推理代码。

---

## 快速了解

如果你想快速理解系统设计，建议按以下顺序阅读：

1. **整体架构** → 本 README 的架构图
2. **路径管理** → [common/paths.py](common/paths.py)（了解所有数据/模型路径定义）
3. **推荐模型** → [bp_recommendation/bp_recommendation_backend.py](bp_recommendation/bp_recommendation_backend.py) → [model_pick/model_pick.py](bp_recommendation/model_pick/model_pick.py) → [cascade_pick.py](bp_recommendation/model_pick/cascade_pick.py) → [model_ban/model_ban.py](bp_recommendation/model_ban/model_ban.py) → [model_ban/cascade_ban.py](bp_recommendation/model_ban/cascade_ban.py) 
4. **预测模型** → [bp_prediction/predict_backend.py](bp_prediction/predict_backend.py) → [training/train_walk_forward.py](bp_prediction/training/train_walk_forward.py) → [train_production.py](bp_prediction/train_production.py)
5. **兜底机制** → [fallback/fallback_manager.py](fallback/fallback_manager.py) → [rule_engine.py](fallback/rule_engine.py) → [triggers.py](fallback/triggers.py)
6. **漂移监控** → [monitoring/weekly_psi_check.py](monitoring/weekly_psi_check.py)

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 深度学习 | PyTorch, HuggingFace Transformers (DistilBert 骨架) |
| 树模型 | LightGBM (级联精排), CatBoost (胜负预测集成) |
| 特征工程 | 贝叶斯平滑, 时间衰减模型, PIT 隔离 |
| 超参搜索 | Optuna (TPE 采样) |
| 监控 | PSI (Population Stability Index) 特征漂移检测 |

---

## 目录结构

```
lol_public/
├── README.md                       # 本文件
├── .gitignore
├── logger_config.py                # 统一日志配置
├── data_checks.py                  # 数据质量检查工具
├── auto_update_pipeline.py         # 自动更新流水线
├── build_feature_baselines.py      # PSI 基线构建
│
├── bp_prediction/                  # 胜负预测模块
│   ├── predict_backend.py          #   ★ 推理后端入口
│   ├── predict_match.py            #   交互式测试脚本
│   ├── feature_builder.py          #   特征构建
│   ├── explainability.py           #   SHAP 可解释性
│   ├── bp_delta.py                 #   BP 价值差
│   ├── config.py                   #   配置
│   └── training/                   #   训练脚本
│       ├── train_walk_forward.py   #     Walk-forward 训练
│       ├── extract_transformer_features.py  # Transformer 特征提取
│       └── train_production.py     #     生产模型训练
│
├── bp_recommendation/              # BP 推荐模块
│   ├── bp_recommendation_backend.py#   ★ 推理后端入口
│   ├── bp_predict.py               #   交互式测试脚本
│   ├── feature_pipeline.py         #   特征工程
│   ├── config.py                   #   配置
│   ├── model_pick/                 #   Pick 模型
│   │   ├── model_pick.py           #     Transformer 定义
│   │   ├── cascade_pick.py         #     LightGBM 级联
│   │   ├── dataloader_pick.py      #     数据加载
│   │   └── train_pick.py           #     训练脚本
│   └── model_ban/                  #   Ban 模型 (结构同 model_pick)
│
├── fallback/                       # 规则兜底模块
│   ├── fallback_manager.py         #   兜底管理器
│   ├── rule_engine.py              #   规则引擎
│   ├── triggers.py                 #   触发器 (Logit 坍塌 + 滑动窗口)
│   └── data_pipeline.py            #   数据加载
│
├── common/                         # 共享模块
│   ├── paths.py                    #   统一路径管理
│   ├── psi.py                      #   PSI 计算
│   └── inference_feature_logger.py #   推理特征日志
│
├── monitoring/                     # 模型监控
│   ├── weekly_psi_check.py         #   PSI 漂移检测
│   └── setup_psi_cron.sh           #   cron 安装脚本
│
└── exploratory data analysis/      # EDA 分析
    └── LPL_2026_split2_analysis.ipynb
```

---

## License

本项目仅用于学习和交流目的，分享模型架构与工程思路。
