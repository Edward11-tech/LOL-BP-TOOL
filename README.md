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

## 推理接口

本仓库不含 Flask 服务入口（`app.py`）和前端页面，提供两个**交互式命令行脚本**作为推理接口，便于快速验证模型效果。

### 1. 胜负预测 — predict_match.py

**接口文件**：[bp_prediction/predict_match.py](bp_prediction/predict_match.py)

**启动命令**：

```bash
cd lol_public
python -m bp_prediction.predict_match --mode production
# --mode production : 加载生产模型 (默认)
# --mode training   : 加载 OOT 5-Fold 模型
```

**交互流程**（启动后逐步 `input()`，支持循环预测多局）：

```
$ python -m bp_prediction.predict_match --mode production

  BP 胜负预测模型 - 单局验证测试
  运行模式: PRODUCTION
  模型: CatBoost-7Seed-Bagging

  --- 预测模式 ---
  1) 纯 Draft 模式: 仅输入阵容, 无战队/选手信息
  2) 完整模式: 输入战队+阵容+选手 (选手可填 unknown, 每队最多2名)
  选择模式 (1/2): 2

  --- 基本信息 ---
  联赛 (LPL/LCK/LEC): LPL
  是否季后赛? (y/n): n
  蓝方是否为先选方? (y/n): y

  --- 阵容选择 ---
  位置顺序: 上单 → 打野 → 中单 → ADC → 辅助

  [蓝方阵容]
    上单: Ornn
    打野: Vi
    中单: Azir
    ADC:  Jinx
    辅助: Lulu

  [红方阵容]
    上单: Gnar
    打野: Sejuani
    中单: Sylas
    ADC:  Aphelios
    辅助: Thresh

  --- 队伍信息 ---
  蓝方队伍名称: BLG
  红方队伍名称: T1

  --- 选手信息 ---
  输入选手ID获取历史特征; 输入 unknown 标记未知选手 (每队最多2名)

  [BLG (蓝方) 选手]
    上单 (Ornn) 选手ID: Bin
    打野 (Vi) 选手ID: Xun
    中单 (Azir) 选手ID: Knight
    ADC (Jinx) 选手ID: Elk
    辅助 (Lulu) 选手ID: ON

  [T1 (红方) 选手]
    上单 (Gnar) 选手ID: Zeus
    打野 (Sejuani) 选手ID: Oner
    中单 (Sylas) 选手ID: Faker
    ADC (Aphelios) 选手ID: Gumayusi
    辅助 (Thresh) 选手ID: Keria

  构建特征向量...
  模式: 完整 | 特征维度: 352
  进行预测...

  (输出预测结果, 见下方)

  是否预测下一局? (y/n): n
```

**输出示例**：

```
======================================================================
  预测结果
======================================================================

  BLG (蓝方) 胜率: 62.3%
  T1  (红方) 胜率: 37.7%

  >>> 预测胜方: BLG (置信度: 24.6%)

  各模型预测详情:
    Production: 0.623 (seeds: [0.631, 0.618, 0.625, ...])

======================================================================
  特征权重分析 (Top 20)
======================================================================
  排名 特征名                                          权重%      当前值
  ─────────────────────────────────────────────────────────────────────
     1 blue_pick_power_diff                             8.42%     0.3210
     2 red_ban_target_value                             6.15%    -0.1540
     ...

======================================================================
  阵容对比分析
======================================================================
  位置   蓝方             红方             蓝方强度  红方强度    差值
  ───────────────────────────────────────────────────────────────────
  上单   Ornn             Gnar                  12        10      +2
  ...
```

**核心函数**：

| 函数 | 说明 |
|------|------|
| `load_models()` | 加载 CatBoost 集成模型 (生产 / OOT 折) |
| `build_single_match_features()` | 从 `feature_builder.py` 构建 350+ 维特征向量 |
| `predict_with_models()` | 7 seed × N fold 集成预测, 输出胜率 + 特征重要性 |

### 2. BP 推荐 — bp_predict.py

**接口文件**：[bp_recommendation/bp_predict.py](bp_recommendation/bp_predict.py)

**启动命令**：

```bash
cd lol_public
python -m bp_recommendation.bp_predict
```

**交互流程**（启动后逐步 `input()`，共 20 步 BP 流程）：

```
$ python -m bp_recommendation.bp_predict

  LOL BP 实时推荐系统
  输入英雄英文名/中文名/Riot ID 进行 Ban/Pick

--- 选择模式 ---
  1) 纯 Draft 模式 (不输入战队/选手信息)
  2) 完整模式 (输入双方战队 + 选手信息)
  请选择 (1/2): 2

--- 赛前信息 ---
  联赛 (LPL/LCK/LEC), 默认LPL: LPL
  是否季后赛 (y/n, 默认n): n
  先选方 (blue/red, 默认blue): blue

--- 战队信息 ---
  蓝方队伍名: BLG
  红方队伍名: T1

--- 选手信息 ---
  输入选手 ID，未知选手请输入 'unknown'
  每队最多允许 2 名 unknown 选手

  蓝方 (BLG) 已知选手: Bin, Xun, Knight, Elk, ON, ...
  蓝方 top: Bin
  蓝方 jng: Xun
  蓝方 mid: Knight
  蓝方 bot: Elk
  蓝方 sup: ON

  红方 (T1) 已知选手: Zeus, Oner, Faker, Gumayusi, Keria, ...
  红方 top: Zeus
  红方 jng: Oner
  红方 mid: Faker
  红方 bot: Gumayusi
  红方 sup: Keria

--- 开始 BP (完整模式) ---
  每步输入英雄名称后回车，输入 'q' 退出，输入 'skip' 跳过当前步
  输入 'undo' 撤销上一步

  >>> 第 1/20 步: 蓝方 Ban1

  Top-20 Ban 推荐:
  Rank  Champion            Score
  ----  --------            -----
     1  Xayah              0.8421 <<<
     2  Aphelios           0.7893 <<<
     3  Caitlyn            0.7654 <<<
     ...

  输入 蓝方Ban1 的英雄 (或 q/undo): Xayah
  已选择: Xayah

  >>> 第 2/20 步: 红方 Ban1
  (模型输出红方 Ban 推荐 Top-20, 用户输入...)
  ...
```

**输出示例**（每一步的推荐）：

```
  第 1/20 步: 蓝方 Ban1

  Top-20 Ban 推荐:
  Rank  Champion            Score
  ----  --------            -----
     1  Xayah              0.8421 <<<
     2  Aphelios           0.7893 <<<
     3  Caitlyn            0.7654 <<<
     4  Lucian             0.7102
     ...

  输入 蓝方Ban1 的英雄 (或 q/undo): Xayah
```

**核心函数**：

| 函数 | 说明 |
|------|------|
| `BPRecommender()` | 加载 Transformer (Pick/Ban) + LightGBM 级联 + 特征存储 |
| `recommender.predict_pick()` | 单步 Pick 推荐, 输出候选英雄 + 排序分数 |
| `recommender.predict_ban()` | 单步 Ban 推荐, 输出候选英雄 + 排序分数 |
| `interactive_predict()` | 交互式入口, 管理 20 步 BP 流程状态 |

**BP 步骤序列**（`BP_SEQUENCE`，共 20 步）：

```
Ban1蓝 → Ban1红 → Ban2蓝 → Ban2红 → Ban3蓝 → Ban3红
Pick1蓝 → Pick1红 → Pick1红 → Pick1蓝
Ban4蓝 → Ban4红 → Ban5红 → Ban5蓝
Pick2红 → Pick2蓝 → Pick2蓝 → Pick2红 → Pick2红 → Pick2蓝
```

### 推理流程说明

两个脚本的内部数据流一致：

```
用户交互输入
    ↓
build_single_match_features() / get_pick_candidate_matrix()
    ↓
构建特征向量 (350+ 维 wide features + Transformer 深层特征)
    ↓
模型推理 (CatBoost 集成 / Transformer + LightGBM 级联)
    ↓
FallbackManager 监控置信度 (异常时降级为规则引擎)
    ↓
输出胜率 / 推荐列表 + 特征权重
```

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
├── requirements.txt                # Python 依赖
├── .gitignore
├── logger_config.py                # 统一日志配置
├── data_checks.py                  # 数据质量检查工具
├── auto_update_pipeline.py         # 自动更新流水线
├── build_feature_baselines.py      # PSI 基线构建
│
├── bp_prediction/                  # 胜负预测模块
│   ├── predict_backend.py          #   ★ 推理后端入口
│   ├── predict_match.py            #   交互式测试脚本
│   ├── run_training.py             #   训练入口 (Walk-forward + OOT)
│   ├── train_production.py         #   生产模型训练
│   ├── feature_builder.py          #   特征构建
│   ├── feature_pipeline.py         #   特征流水线
│   ├── feature_utils.py            #   特征工具函数
│   ├── feature_monitor.py          #   推理特征监控
│   ├── explainability.py           #   SHAP 可解释性
│   ├── bp_delta.py                 #   BP 价值差
│   ├── config.py                   #   配置
│   ├── check_feature_alignment.py  #   特征对齐检查
│   ├── check_prediction_alignment.py # 预测对齐检查
│   ├── export_production_transformer.py  # Transformer 导出
│   └── training/                   #   训练子模块
│       ├── train_walk_forward.py   #     Walk-forward 训练
│       └── extract_transformer_features.py  # Transformer 特征提取
│
├── bp_recommendation/              # BP 推荐模块
│   ├── bp_recommendation_backend.py#   ★ 推理后端入口
│   ├── bp_predict.py               #   交互式测试脚本
│   ├── run_pipeline.py             #   训练流水线入口
│   ├── feature_pipeline.py         #   特征工程
│   ├── feature_monitor.py          #   推理特征监控
│   ├── config.py                   #   配置
│   ├── init_config.py              #   初始化配置
│   ├── inference_test.py           #   推理冒烟测试
│   ├── verify_features_alignment.py#   特征对齐验证
│   ├── verify_predictions.py       #   预测结果验证
│   ├── model_pick/                 #   Pick 模型
│   │   ├── model_pick.py           #     Transformer 定义
│   │   ├── cascade_pick.py         #     LightGBM 级联
│   │   ├── cascade_pick_search.py  #     级联超参搜索
│   │   ├── cascade_pick_experiment.py    # 级联实验 v1
│   │   ├── cascade_pick_experiment_v2.py # 级联实验 v2
│   │   ├── transformer_pick_search.py    # Transformer 超参搜索
│   │   ├── dataloader_pick.py      #     数据加载
│   │   └── train_pick.py           #     训练脚本
│   └── model_ban/                  #   Ban 模型 (结构同 model_pick)
│       ├── model_ban.py
│       ├── cascade_ban.py
│       ├── cascade_ban_search.py
│       ├── transformer_ban_search.py
│       ├── dataloader_ban.py
│       └── train_ban.py
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
