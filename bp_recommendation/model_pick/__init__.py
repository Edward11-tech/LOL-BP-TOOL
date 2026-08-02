"""
Pick 阶段模型包

BP 推荐系统中 Pick 阶段相关的模型、数据加载、训练和超参搜索模块。

主要模块:
    - model_pick: Pick 阶段 Transformer 模型定义
    - dataloader_pick: Pick 阶段数据加载器
    - train_pick: Pick Transformer 模型训练脚本（支持 CS/NoCS 双模型）
    - cascade_pick: Pick Cascade LightGBM 模型训练
    - transformer_pick_search: Pick Transformer 超参数搜索
    - cascade_pick_search: Pick Cascade 超参数搜索
"""
