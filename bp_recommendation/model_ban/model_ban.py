"""
Ban 阶段 Transformer 模型定义
=============================================
基于 DistilBERT 的 BP 战术 Transformer 模型，用于 Ban 阶段的英雄排序推荐。
采用 Late Fusion 架构融合 BP 序列上下文和全局上下文信息。

功能描述:
    - 基于 DistilBERT 编码已选/已禁英雄序列
    - Late Fusion 方式融合全局上下文（联赛、战队风格等）
    - 候选英雄特征通过 MLP 映射到查询空间
    - 点积注意力计算候选英雄得分
    - 支持辅助损失（位置预测等）
    - 支持 Ban 样本加权和温度缩放

主要类:
    - BPTacticalTransformer: Ban 阶段 Transformer 模型

使用方法:
    import torch
    from bp_recommendation.model_ban.model_ban import BPTacticalTransformer
    
    model = BPTacticalTransformer(
        candidate_dim=32,
        context_dim=20,
        vocab_size=175
    )
    # 输入: bp_seq, global_ctx, candidate_matrix, candidate_mask
    # 输出: scores, aux_loss
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertModel

class BPTacticalTransformer(nn.Module):
    """
    BAN 专用 Transformer 模型
    最佳超参数 (Optuna TPE 搜索 2026-06-17): Ban@10=0.8298
    """
    def __init__(
        self,
        candidate_dim: int,       # 强制必传，通常为 32 或 33
        context_dim: int,         # 强制必传，通常为 20 或 24
        vocab_size: int = 175,
        h_dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        c_dim: int = 64,
        query_dim: int = 256,
        n_positions: int = 5,
        aux_loss_weight: float = 1.242,
        pad_idx: int = 0,
        temperature: float = None,
        ban_sample_weight: float = 2.0,
        dropout: float = 0.100,
        attention_dropout: float = 0.134,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.query_dim = query_dim
        self.aux_loss_weight = aux_loss_weight
        self.pad_idx = pad_idx
        self.n_positions = n_positions
        self.ban_sample_weight = ban_sample_weight
        self.temperature = temperature if temperature is not None else math.sqrt(query_dim)

        # 确保配置正确对齐
        assert context_dim in (15, 19, 20, 24), f"Got {context_dim}."
        assert candidate_dim in (30, 31, 32, 33), f"Got {candidate_dim}."

        bert_config = DistilBertConfig(
            vocab_size=vocab_size,
            dim=h_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=h_dim * 4,
            max_position_embeddings=32,
            pad_token_id=pad_idx,
            dropout=dropout,
            attention_dropout=attention_dropout,
            attn_implementation="eager",  # 避免 SDPA 在某些输入形状下 IndexError
        )
        self.bert = DistilBertModel(bert_config)
        
        # 将 Team Context 映射
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )

        self.bert_proj = nn.Linear(h_dim, query_dim)
        
        # 【修复 1：Late Fusion】将特征交叉放在这里，极其稳定且安全
        self.fusion_proj = nn.Sequential(
            nn.Linear(query_dim + c_dim, query_dim),
            nn.GELU(),
            nn.LayerNorm(query_dim),
        )

        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, query_dim),
        )
        self.candidate_norm = nn.LayerNorm(query_dim)

        self.enemy_role_head = nn.Sequential(
            nn.Linear(h_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_positions),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.context_mlp, self.fusion_proj, self.candidate_mlp, self.enemy_role_head]:
            for layer in module.modules(): # 【修复 2】：使用 .modules() 递归初始化，避免遗漏
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        bp_sequence: torch.Tensor,
        global_context: torch.Tensor,
        candidate_matrix: torch.Tensor,
        available_mask: torch.Tensor,
        history_positions: torch.Tensor = None, 
    ):
        B = bp_sequence.size(0)

        # 常规传递给 BERT，确保 Position Embedding 完全正确
        seq_mask = (bp_sequence != self.pad_idx).long()
        bert_out = self.bert(
            input_ids=bp_sequence,
            attention_mask=seq_mask,
            return_dict=False,  # 【修复 ONNX】：返回纯 tuple，避免 ModelOutput 导致 "tuple index out of range"
        )
        bp_hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state

        mask_expanded = seq_mask.unsqueeze(-1).expand(bp_hidden_states.size()).float()
        sum_hidden = torch.sum(bp_hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_hidden = sum_hidden / sum_mask

        # 映射到 Query 空间
        query_bert = self.bert_proj(pooled_hidden)

        # 【修复 1 落地】：在这里执行 Fusion
        ctx_embed = self.context_mlp(global_context)
        query_fused = self.fusion_proj(torch.cat([query_bert, ctx_embed], dim=-1))

        # 候选集嵌入
        cand_embed = self.candidate_mlp(candidate_matrix)
        cand_embed = self.candidate_norm(cand_embed)

        # 打分计算
        raw_logits = torch.bmm(
            query_fused.unsqueeze(1),
            cand_embed.transpose(1, 2),
        ).squeeze(1) / self.temperature

        clamped_logits = raw_logits.clamp(-30, 30)
        masked_logits = clamped_logits + (1.0 - available_mask) * (-1e9)

        # 【优化 3】：安全的 Aux Loss 初始化
        aux_loss = torch.tensor(0.0, device=bp_sequence.device, requires_grad=True)
        role_logits = None

        if history_positions is not None:
            role_logits = self.enemy_role_head(bp_hidden_states)
            valid_logits = role_logits.view(-1, self.n_positions)
            valid_labels = history_positions.view(-1)
            # ONNX-safe: 移除数据依赖控制流 if (valid_labels != -1).any()
            # 改用 sum 归约 + 手动求均值，避免全 -1 时 mean 返回 NaN
            valid_count = (valid_labels != -1).sum().clamp(min=1)
            aux_loss = F.cross_entropy(
                valid_logits, valid_labels, ignore_index=-1, reduction='sum'
            ) / valid_count

        # 【修复】移除 requires_grad 判断：验证阶段使用 @torch.no_grad()，
        # 此时 aux_loss.requires_grad=False 会导致验证 aux_loss 恒为 0，无法监控辅助任务表现
        total_aux_loss = self.aux_loss_weight * aux_loss if isinstance(aux_loss, torch.Tensor) else 0.0

        return {
            "logits": masked_logits,
            "aux_loss": total_aux_loss,
            "role_logits": role_logits,
        }

    def compute_loss(self, logits, labels, aux_loss, is_pick=None, time_weight=None):
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        
        # Focal Loss (安全、兼容 ONNX 的写法)
        # ce_loss = -log(target_prob)，所以 target_prob = exp(-ce_loss)
        target_probs = torch.exp(-ce_loss)
        
        gamma = 1.0 
        focal_weight = torch.pow(1.0 - target_probs + 1e-6, gamma)
        per_sample_loss = ce_loss * focal_weight

        if is_pick is not None:
            action_weights = torch.where(
                is_pick < 0.5,
                torch.tensor(self.ban_sample_weight, device=logits.device),
                torch.tensor(1.0, device=logits.device),
            )
            per_sample_loss = per_sample_loss * action_weights
            
        if time_weight is not None:
            per_sample_loss = per_sample_loss * time_weight

        main_loss = per_sample_loss.mean()
        total_loss = main_loss + aux_loss
        return main_loss, total_loss