"""
BP 胜负预测模型包
==================
提供英雄联盟职业比赛胜率预测和 BP Delta 分析功能。

主要模块:
- config: 统一配置管理，支持 training/production 双模式切换
- feature_builder: 在线特征构建器，确保与离线训练特征一致
- feature_pipeline: 离线特征工程流水线
- feature_utils: 特征计算公共工具函数
- feature_monitor: 特征漂移监控（PSI）
- predict_backend: 生产推理后端封装（7-seed bagging）
- predict_match: 单场比赛预测接口
- bp_delta: BP 影响量化分析（Pre/Post Draft 胜率差）
- explainability: SHAP 模型可解释性与 counter/synergy 分析
- train_production: 生产模式模型训练
- training/: 滚动窗口训练和 Transformer 特征提取子模块

使用方法:
    from bp_prediction.predict_backend import PredictBackend
    backend = PredictBackend()
    backend.load()
    result = backend.predict(blue_team, red_team, ...)
"""
