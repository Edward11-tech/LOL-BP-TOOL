"""
Pick 阶段 Transformer 模型定义
=============================================
基于 DistilBERT 的 BP 战术 Transformer 模型，用于 Pick 阶段的英雄排序推荐。
采用位置 Token、战术偏置、Late Fusion 等机制，支持 CS/NoCS 双模型架构。

功能描述:
    - 基于 DistilBERT 编码已选/已禁英雄序列（含位置 Token）
    - 扩展词表包含 5 个位置 Token（top/jng/mid/bot/sup）
    - Late Fusion 方式融合全局上下文（联赛、战队风格等）
    - 候选英雄特征通过 MLP 映射到查询空间
    - 战术偏置 MLP 提供位置感知的候选打分
    - 点积注意力 + 战术偏置计算最终候选得分
    - 支持辅助损失（位置预测、组合伙伴预测等）
    - 支持 Ban 样本加权、Step 6 降权和温度缩放

主要类:
    - BPTacticalTransformerPick: Pick 阶段 Transformer 模型

使用方法:
    import torch
    from bp_recommendation.model_pick.model_pick import BPTacticalTransformerPick
    
    model = BPTacticalTransformerPick(
        vocab_size=180,
        context_dim=20,
        candidate_dim=33
    )
    # 输入: bp_seq, global_ctx, candidate_matrix, candidate_mask, last_ally_pos
    # 输出: scores, aux_losses
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertModel

class BPTacticalTransformerPick(nn.Module):
    def __init__(
        self,
        vocab_size: int = 180,
        context_dim: int = 20,
        candidate_dim: int = 33,
        h_dim: int = 512,
        c_dim: int = 64,
        query_dim: int = 384,
        n_layers: int = 2,
        n_heads: int = 16,
        dropout: float = 0.155,
        attention_dropout: float = 0.117,
        candidate_hidden: int = 256,
        tactical_hidden: int = 128,
        n_positions: int = 5,
        aux_loss_weight: float = 3.01,
        pad_idx: int = 0,
        temperature: float = None,
        ban_sample_weight: float = 0.023,
        role_token_start: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.query_dim = query_dim
        self.aux_loss_weight = aux_loss_weight
        self.pad_idx = pad_idx
        self.n_positions = n_positions
        self.ban_sample_weight = ban_sample_weight
        self.temperature = temperature if temperature is not None else math.sqrt(query_dim)

        # role_token_start: 位置 token 的起始 idx (v3 方案固定为 2)
        self.role_token_start = role_token_start
        self.extended_vocab_size = max(vocab_size, self.role_token_start + n_positions)

        bert_config = DistilBertConfig(
            vocab_size=self.extended_vocab_size,
            dim=h_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=h_dim * 4,
            max_position_embeddings=40,
            pad_token_id=pad_idx,
            dropout=dropout,
            attention_dropout=attention_dropout,
            attn_implementation="eager",  # 避免 SDPA 在某些输入形状下 IndexError
        )
        self.bert = DistilBertModel(bert_config)
        self.bert_proj = nn.Linear(h_dim, query_dim)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, c_dim),
            nn.GELU(),
            nn.Linear(c_dim, c_dim),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(query_dim + c_dim, query_dim),
            nn.GELU(),
            nn.LayerNorm(query_dim),
        )

        self.query_norm = nn.LayerNorm(query_dim)
        
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(candidate_hidden, candidate_hidden // 2),
            nn.GELU(),
            nn.Linear(candidate_hidden // 2, query_dim),
        )
        self.candidate_norm = nn.LayerNorm(query_dim)

        self.tactical_bias_mlp = nn.Sequential(
            nn.Linear(candidate_dim, tactical_hidden),
            nn.GELU(),
            nn.Linear(tactical_hidden, 1)
        )

        self.enemy_role_head = nn.Sequential(
            nn.Linear(h_dim, candidate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(candidate_hidden, candidate_hidden // 2),
            nn.GELU(),
            nn.Linear(candidate_hidden // 2, n_positions),
        )

        self.combo_proj = nn.Sequential(
            nn.Linear(h_dim, query_dim),
            nn.GELU(),
        )
        self.combo_gate = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        # 【清理】移除了 team_repr_head 和 rhythm_head 的初始化
        for module in [self.context_mlp, self.fusion_proj, self.candidate_mlp, 
                       self.tactical_bias_mlp, self.enemy_role_head]:
            for layer in module.modules():
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
        last_ally_pos: torch.Tensor = None,
    ):
        B, seq_len = bp_sequence.size()


        role_tokens = torch.arange(self.role_token_start, self.role_token_start + self.n_positions, device=bp_sequence.device)
        role_tokens = role_tokens.unsqueeze(0).expand(B, -1)
        
        extended_seq = torch.cat([bp_sequence, role_tokens], dim=1)
        
        seq_mask = (bp_sequence != self.pad_idx).long()
        role_mask = torch.ones((B, self.n_positions), device=bp_sequence.device, dtype=torch.long)
        attention_mask = torch.cat([seq_mask, role_mask], dim=1)

        bert_out = self.bert(
            input_ids=extended_seq,
            attention_mask=attention_mask,
            return_dict=False,  # 【修复 ONNX】：返回纯 tuple，避免 ModelOutput 导致 "tuple index out of range"
        )
        hidden_states = bert_out[0] if isinstance(bert_out, tuple) else bert_out.last_hidden_state
        
        bp_hidden_states = hidden_states[:, :seq_len, :] 
        
        mask_expanded = seq_mask.unsqueeze(-1).expand(bp_hidden_states.size()).float()
        sum_hidden = torch.sum(bp_hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_hidden = sum_hidden / sum_mask

        query_bert = self.bert_proj(pooled_hidden)

        ctx_embed = self.context_mlp(global_context)
        query_fused = self.fusion_proj(torch.cat([query_bert, ctx_embed], dim=-1))
        query_fused = self.query_norm(query_fused)

        cand_embed = self.candidate_mlp(candidate_matrix)
        cand_embed = self.candidate_norm(cand_embed)

        base_logits = torch.bmm(
            query_fused.unsqueeze(1),
            cand_embed.transpose(1, 2),
        ).squeeze(1) / self.temperature

        tactical_bias = self.tactical_bias_mlp(candidate_matrix).squeeze(-1)
        
        # 【极致优化】：彻底废除 Python for 循环和 .item()，采用纯 Tensor 批量抽取与计算
        combo_bias = torch.zeros(B, self.vocab_size, device=bp_sequence.device)
        if last_ally_pos is not None:
            # 1. 建立合法性掩码 (B,)
            valid_mask = (last_ally_pos >= 0) & (last_ally_pos < seq_len)
            
            # 2. 安全阻断非法索引，防止 gather 越界
            safe_pos = torch.where(valid_mask, last_ally_pos, torch.zeros_like(last_ally_pos))
            
            # 3. O(1) 批量提取所有 Batch 的 last_ally_hidden -> (B, h_dim)
            expanded_pos = safe_pos.unsqueeze(1).unsqueeze(2).expand(-1, -1, bp_hidden_states.size(2))
            last_ally_hidden = bp_hidden_states.gather(1, expanded_pos).squeeze(1)
            
            # 4. 批量计算 Combo Query -> (B, query_dim)
            combo_query = self.combo_proj(last_ally_hidden)
            
            # 5. 批量计算 Combo Score，使用 BMM -> (B, vocab_size)
            combo_score = torch.bmm(cand_embed, combo_query.unsqueeze(-1)).squeeze(-1) / self.temperature
            
            # 6. 利用掩码过滤掉无效的 Batch 项
            combo_bias = torch.where(valid_mask.unsqueeze(-1), combo_score, combo_bias)
        
        raw_logits = base_logits + tactical_bias + self.combo_gate * combo_bias

        clamped_logits = raw_logits.clamp(-30, 30)
        masked_logits = clamped_logits + (1.0 - available_mask) * (-1e9)

        # -------------------------------------------------------------
        # 辅助任务计算区 (已清理实验性代码)
        # -------------------------------------------------------------
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

    def compute_loss(self, logits, labels, aux_loss, is_pick=None, time_weight=None, sample_weight=None,
                     tuple_partners=None, bp_steps=None):
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")

        if tuple_partners is not None and bp_steps is not None:
            is_tuple_start = (bp_steps == 7) | (bp_steps == 9) | (bp_steps == 17)
            valid_partner = (tuple_partners >= 0)
            soft_mask = is_tuple_start & valid_partner

            if soft_mask.any():
                # 全局计算 partner_loss，利用 ignore_index=-1 自动忽略无效项
                partner_loss = F.cross_entropy(
                    logits, tuple_partners, reduction="none", ignore_index=-1
                )
                # 安全融合
                per_sample_loss = torch.where(
                    soft_mask,
                    0.5 * per_sample_loss + 0.5 * partner_loss,
                    per_sample_loss
                )
                
        if is_pick is not None:
            action_weights = torch.where(
                is_pick < 0.5,
                torch.tensor(self.ban_sample_weight, device=logits.device),
                torch.tensor(1.0, device=logits.device),
            )
            per_sample_loss = per_sample_loss * action_weights
            
        if time_weight is not None:
            per_sample_loss = per_sample_loss * time_weight

        if sample_weight is not None:
            per_sample_loss = per_sample_loss * sample_weight

        main_loss = per_sample_loss.mean()
        total_loss = main_loss + aux_loss
        return main_loss, total_loss
