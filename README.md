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
                         │  OraclesElixir (比赛选手级长表)              │
                         │    └─ 每年 CSV, 含 gameid/playerid/champion  │
                         │       /kills/deaths/assists/result 等       │
                         │                                             │
                         │  ScoreGG (选手职业生涯英雄统计)              │
                         │    └─ games / win_rate / KDA                │
                         │                                             │
                         │  LoL Fandom (英雄克制/协同/定位)             │
                         │    └─ champion_counters / synergy / position│
                         │                                             │
                         │  官方赛事数据 (战队阵容/选手归属)             │
                         │    └─ active_rosters / team_player_mapping   │
                         └────────────────────┬────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────┐
                    │         数据清洗 (PIT 隔离)              │
                    │  cleaned_data/                           │
                    │    ├─ matches_cleaned.csv (比赛级宽表)    │
                    │    ├─ merged_champion_stats.csv (英雄大盘)│
                    │    ├─ player_career_hero_stats.csv       │
                    │    ├─ champion_vocabulary.json           │
                    │    └─ champion_counters/synergy/ranks.csv│
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
                    │            app.py (Flask API)           │
                    │                                         │
                    │  POST /api/predict   比赛胜负预测       │
                    │  POST /api/recommend BP 推荐            │
                    │  GET  /api/health    健康检查           │
                    │                                         │
                    │  ※ 本仓库仅提供后端 API, 不含前端页面   │
                    │    调用方通过 HTTP 请求获取 JSON 结果    │
                    └────────────────────┬────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
               ┌──────────────────────┐  ┌──────────────────────┐
               │  monitoring/         │  │  logs/               │
               │  PSI 漂移检测        │  │  推理特征日志        │
               │  (每周一 cron)       │  │  (parquet, 按天滚动) │
               │  weekly_psi_check.py │  │                      │
               └──────────────────────┘  └──────────────────────┘
```

### 架构要点

| 层级 | 说明 |
|------|------|
| **数据层** | 原始数据来自 4 个外部源（见下文），经清洗后存入 `cleaned_data/`。所有特征工程严格遵循 Point-in-Time (PIT) 隔离，避免未来信息泄漏 |
| **模型层** | 推荐 = Transformer 召回 + LightGBM 级联精排；预测 = CatBoost 集成 + Transformer 深层特征。两个模块独立训练、独立推理 |
| **兜底层** | 当模型置信度坍塌或滚动指标跌破红线时，自动降级为 Rule-based 规则引擎，保证服务不中断 |
| **服务层** | Flask 提供 RESTful API，不含前端页面。调用方通过 HTTP 请求获取 JSON 结果 |
| **监控层** | PSI 特征漂移检测每周一自动运行，对比训练基线与线上推理特征分布 |

---

## 核心模块说明

### bp_recommendation/ — BP 推荐模块

英雄推荐（Pick/Ban）采用 **两阶段架构**：

1. **Transformer 召回**：`model_pick/` 和 `model_ban/` 各训练一个 BPTacticalTransformer，输入当前 BP 状态（已ban/pick的英雄、位置约束、选手信息），输出候选英雄的_logits
2. **LightGBM 级联精排**：`cascade_pick.py` / `cascade_ban.py` 对 Transformer 输出的候选集做重排序，使用 CS（Context-Aware Split）和 NoCS（No Context Split）两种模式
3. **Blend 融合**：最终通过 `routing_config.json` 配置的权重融合 CS/NoCS 结果

关键文件：
- [model_pick/model_pick.py](bp_recommendation/model_pick/model_pick.py) — Transformer 模型定义
- [model_pick/cascade_pick.py](bp_recommendation/model_pick/cascade_pick.py) — LightGBM 级联排序
- [bp_predict.py](bp_recommendation/bp_predict.py) — 推理入口
- [feature_pipeline.py](bp_recommendation/feature_pipeline.py) — 特征工程（含克制/协同/选手熟练度）

### bp_prediction/ — 胜负预测模块

比赛胜负预测采用 **CatBoost 集成 + Transformer 深层特征**：

1. **CatBoost 集成**：7 个随机种子 × 5 折 OOT（Out-of-Time）验证 = 35 个基模型，取平均
2. **Transformer 深层特征**：用 NoCS 快照提取 4 种深层特征（注意力权重、候选隐状态等），拼接到 CatBoost 的 wide features 上
3. **生产模型**：从 5 折中选出最优折，用全量数据重训，存入 `models/production/`

关键文件：
- [predict_backend.py](bp_prediction/predict_backend.py) — 推理后端
- [training/train_walk_forward.py](bp_prediction/training/train_walk_forward.py) — Walk-forward 训练
- [training/extract_transformer_features.py](bp_prediction/training/extract_transformer_features.py) — PIT 隔离的 Transformer 特征提取
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

本系统不含任何数据文件。以下为所需数据的外部来源：

### 1. 比赛数据（核心）

| 项 | 说明 |
|----|------|
| **来源** | [OraclesElixir](https://oracleselixir.com/tools/downloads) |
| **格式** | CSV，选手级长表（每行 = 一个选手在一场比赛中的表现） |
| **字段** | gameid, date, league, teamname, playername, champion, position, kills, deaths, assists, result, damagetochampions, earnedgold, wardsplaced 等 |
| **覆盖** | 全球主要联赛（LPL / LCK / LEC / LCS 等），按年归档 |
| **用途** | 比赛胜负预测的特征工程、英雄 Meta 统计、选手英雄熟练度 |

### 2. 选手职业生涯统计

| 项 | 说明 |
|----|------|
| **来源** | [ScoreGG](https://www.scoregg.com/) |
| **格式** | 网页爬取后清洗为 CSV |
| **字段** | player_id, champion, games, win_rate, KDA |
| **用途** | fallback 规则引擎的选手熟练度评分 |

### 3. 英雄属性关系

| 项 | 说明 |
|----|------|
| **来源** | [LoL Fandom Wiki](https://leagueoflegends.fandom.com/) |
| **格式** | 爬取后清洗为 CSV |
| **字段** | champion_counters（克制关系）、champion_synergy（协同关系）、champion_position_mapping（位置映射）、champion_ranks（英雄评级） |
| **用途** | 推荐模型的特征工程、规则引擎的 Meta 统计 |

### 4. 战队阵容数据

| 项 | 说明 |
|----|------|
| **来源** | 官方赛事公开信息 |
| **格式** | CSV / JSON |
| **字段** | active_rosters（现役选手名单）、team_player_mapping（战队-选手映射） |
| **用途** | 推理时根据战队查询选手，构建选手级特征 |

### 数据处理流程

```
外部数据源 → data_cleaning.py (清洗/去重/标准化)
           → merge_champion_stats.py (英雄统计聚合 + 贝叶斯平滑)
           → global_entity_builder.py (选手/战队实体解析)
           → cleaned_data/*.csv (PIT 隔离的特征文件)
```

> **注意**：数据爬虫脚本（`data_scraper/`）和清洗脚本（`data_cleaning.py` 等）未包含在本仓库中。本仓库仅提供模型训练和推理代码。

---

## 快速了解

如果你想快速理解系统设计，建议按以下顺序阅读：

1. **整体架构** → 本 README 的架构图
2. **路径管理** → [common/paths.py](common/paths.py)（了解所有数据/模型路径定义）
3. **推荐模型** → [bp_recommendation/bp_predict.py](bp_recommendation/bp_predict.py) → [model_pick/model_pick.py](bp_recommendation/model_pick/model_pick.py) → [cascade_pick.py](bp_recommendation/model_pick/cascade_pick.py)
4. **预测模型** → [bp_prediction/predict_backend.py](bp_prediction/predict_backend.py) → [training/train_walk_forward.py](bp_prediction/training/train_walk_forward.py)
5. **兜底机制** → [fallback/fallback_manager.py](fallback/fallback_manager.py) → [rule_engine.py](fallback/rule_engine.py) → [triggers.py](fallback/triggers.py)
6. **API 入口** → [app.py](app.py)
7. **漂移监控** → [monitoring/weekly_psi_check.py](monitoring/weekly_psi_check.py)

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 深度学习 | PyTorch, HuggingFace Transformers (DistilBert 骨架) |
| 树模型 | LightGBM (级联精排), CatBoost (胜负预测集成) |
| 特征工程 | 贝叶斯平滑, 时间衰减模型, PIT 隔离 |
| 超参搜索 | Optuna (TPE 采样) |
| 服务框架 | Flask + Flask-CORS |
| 监控 | PSI (Population Stability Index) 特征漂移检测 |

完整依赖见 [requirements.txt](requirements.txt)。

---

## 目录结构

```
lol_public/
├── app.py                          # Flask API 入口
├── logger_config.py                # 日志配置
├── auto_update_pipeline.py         # 自动更新流水线 (训练+部署)
├── deploy_package.py               # 部署打包脚本
├── smoke_test.py                   # 冒烟测试
├── data_checks.py                  # 数据质量检查
├── build_feature_baselines.py      # PSI 基线构建
├── cleanup_training_artifacts.sh   # 训练中间文件清理
├── pyproject.toml                  # 项目配置
├── requirements.txt                # 依赖清单
│
├── bp_prediction/                  # 胜负预测模块
│   ├── predict_backend.py          #   推理后端
│   ├── feature_builder.py          #   特征构建
│   ├── explainability.py           #   可解释性
│   ├── bp_delta.py                 #   BP 价值差
│   ├── config.py                   #   配置
│   └── training/                   #   训练脚本
│       ├── train_walk_forward.py   #     Walk-forward 训练
│       ├── extract_transformer_features.py  # Transformer 特征提取
│       └── train_production.py     #     生产模型训练
│
├── bp_recommendation/              # BP 推荐模块
│   ├── bp_predict.py               #   推理入口
│   ├── bp_recommendation_backend.py#   推荐后端
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
│   └── data_pipeline.py            #   数据加载 (从 cleaned_data)
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
    ├── LPL_2026_split2_analysis.py
    ├── deep_analysis.py
    └── eda_breakthrough.py
```

---

## License

本项目仅用于学习和交流目的，分享模型架构与工程思路。
