"""
BP 推荐系统包

LOL 英雄联盟 Ban/Pick 推荐系统核心模块，提供 Ban 和 Pick 阶段的英雄推荐功能。
包含特征工程、模型训练、推理预测、生产后端等完整功能。

主要模块:
    - bp_predict: 单场 BP 实时推荐
    - bp_recommendation_backend: 生产环境后端封装
    - config: 全局配置管理
    - feature_pipeline: 特征工程流水线
    - feature_monitor: 特征监控与验证
    - model_ban: Ban 阶段模型
    - model_pick: Pick 阶段模型
"""
