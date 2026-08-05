#!/usr/bin/env python3
"""
BPTacticalTransformerPick 超参数搜索 (Optuna + TPE)

适配当前模型架构:
  - 包含 combo_proj/combo_gate (last_ally_pos 机制)
  - 包含 role tokens (extended_vocab_size)
  - 包含 tuple_partners loss
  - candidate_dim=33 (含 is_fearless_banned 和 player_recent_games)
  - context_dim=20 (含 is_game_X)
  - step6_downweight
  - 增加对 PyTorch AMP 和 torch.compile 的支持以加速搜索

用法:
    cd <project_root>
    python -m bp_recommendation.model_pick.transformer_pick_search --n_trials 30 --amp --compile
"""

import os
import sys
import time
import json
import logging
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertModel, get_cosine_schedule_with_warmup
import optuna
from optuna.samplers import TPESampler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bp_recommendation.model_pick.dataloader_pick import create_train_val_dataloaders
from bp_recommendation.feature_pipeline import load_champion_vocabulary
from logger_config import get_logger

from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())
SHARED_FEATURES_DIR = os.path.join(_PROJECT_ROOT, "bp_recommendation", "features")
CLEANED_DIR = os.path.join(_PROJECT_ROOT, "cleaned_data")
VOCAB_PATH = os.path.join(CLEANED_DIR, "champion_vocabulary.json")
POS_JSON = os.path.join(CLEANED_DIR, "champion_position_mapping.json")
PICK_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PICK_DIR, "checkpoints")
LOG_DIR = os.path.join(PICK_DIR, "logs")
SEARCH_DIR = os.path.join(PICK_DIR, "search_results")
DB_DIR = os.path.join(SEARCH_DIR, "optuna_db")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

log = get_logger(__name__)

# --- 同步 CS 特征索引 ---
CS_FEATURE_INDICES = [15, 16, 17, 18]

class SearchableTransformerPick(nn.Module):
    def __init__(
        self,
        candidate_dim: int,
        context_dim: int,
        vocab_size: int = 180,
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

        self.role_token_start = role_token_start
        self.extended_vocab_size = max(vocab_size, role_token_start + n_positions)

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
            nn.Linear(tactical_hidden, 1),
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
        for module in [self.context_mlp, self.fusion_proj, self.candidate_mlp,
                       self.tactical_bias_mlp, self.enemy_role_head]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        bp_sequence: torch.LongTensor,
        global_context: torch.FloatTensor,
        candidate_matrix: torch.FloatTensor,
        available_mask: torch.FloatTensor,
        history_positions: torch.LongTensor = None,
        last_ally_pos: torch.LongTensor = None,
    ):
        B, seq_len = bp_sequence.size()

        role_tokens = torch.arange(
            self.role_token_start, self.role_token_start + self.n_positions,
            device=bp_sequence.device,
        )
        role_tokens = role_tokens.unsqueeze(0).expand(B, -1)

        extended_seq = torch.cat([bp_sequence, role_tokens], dim=1)

        seq_mask = (bp_sequence != self.pad_idx).long()
        role_mask = torch.ones((B, self.n_positions), device=bp_sequence.device, dtype=torch.long)
        attention_mask = torch.cat([seq_mask, role_mask], dim=1)

        bert_out = self.bert(input_ids=extended_seq, attention_mask=attention_mask)
        hidden_states = bert_out.last_hidden_state

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

        combo_bias = torch.zeros(B, self.vocab_size, device=bp_sequence.device)
        if last_ally_pos is not None:
            valid_mask = (last_ally_pos >= 0) & (last_ally_pos < seq_len)
            safe_pos = torch.where(valid_mask, last_ally_pos, torch.zeros_like(last_ally_pos))
            expanded_pos = safe_pos.unsqueeze(1).unsqueeze(2).expand(-1, -1, bp_hidden_states.size(2))
            last_ally_hidden = bp_hidden_states.gather(1, expanded_pos).squeeze(1)
            combo_query = self.combo_proj(last_ally_hidden)
            combo_score = torch.bmm(cand_embed, combo_query.unsqueeze(-1)).squeeze(-1) / self.temperature
            combo_bias = torch.where(valid_mask.unsqueeze(-1), combo_score, combo_bias)

        raw_logits = base_logits + tactical_bias + self.combo_gate * combo_bias

        clamped_logits = raw_logits.clamp(-30, 30)
        masked_logits = clamped_logits + (1.0 - available_mask) * (-1e9)

        aux_loss = torch.tensor(0.0, device=bp_sequence.device, requires_grad=True)
        role_logits = None

        if history_positions is not None:
            role_logits = self.enemy_role_head(bp_hidden_states)
            valid_logits = role_logits.view(-1, self.n_positions)
            valid_labels = history_positions.view(-1)
            if (valid_labels != -1).any():
                aux_loss = F.cross_entropy(valid_logits, valid_labels, ignore_index=-1)

        total_aux_loss = self.aux_loss_weight * aux_loss if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad else 0.0

        return {
            "logits": masked_logits,
            "aux_loss": total_aux_loss,
            "role_logits": role_logits,
        }

    def compute_loss(self, logits, labels, aux_loss, is_pick=None, time_weight=None,
                     sample_weight=None, tuple_partners=None, bp_steps=None):
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")

        if tuple_partners is not None and bp_steps is not None:
            is_tuple_start = (bp_steps == 7) | (bp_steps == 9) | (bp_steps == 17)
            valid_partner = (tuple_partners >= 0)
            soft_mask = is_tuple_start & valid_partner

            if soft_mask.any():
                partner_loss = F.cross_entropy(
                    logits, tuple_partners, reduction="none", ignore_index=-1
                )
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


def recall_at_k(logits, labels, available_mask, k, is_pick=None):
    masked = logits.clone()
    masked[available_mask == 0] = -1e9
    _, topk_indices = masked.topk(k, dim=-1)
    hit = (topk_indices == labels.unsqueeze(1)).any(dim=1)
    if is_pick is not None:
        hit = hit[is_pick.bool()] if is_pick.sum() > 0 else hit[:0]
    if hit.numel() == 0:
        return 0.0
    return hit.float().mean().item()


@torch.no_grad()
def evaluate(model, val_loader, device, mask_cs, use_amp):
    model.eval()
    all_logits, all_labels, all_masks, all_is_pick = [], [], [], []
    total_main_loss = 0.0
    n_batches = 0
    device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

    for batch in val_loader:
        bp_seq = batch["bp_sequence"].to(device)
        ctx = batch["global_context"].to(device)
        cand = batch["candidate_matrix"].to(device)
        mask = batch["available_mask"].to(device)
        labels = batch["label"].to(device)
        is_pick = batch["is_pick"].to(device)
        history_positions = batch.get("history_positions")
        if history_positions is not None:
            history_positions = history_positions.to(device)
        time_weight = batch.get("time_weight")
        if time_weight is not None:
            time_weight = time_weight.to(device)
        last_ally_pos = batch.get("last_ally_pos")
        if last_ally_pos is not None:
            last_ally_pos = last_ally_pos.to(device)

        if mask_cs:
            mask_tensor = torch.ones_like(cand)
            mask_tensor[:, :, CS_FEATURE_INDICES] = 0.0
            cand = cand * mask_tensor

        with torch.autocast(device_type=device_type, enabled=use_amp):
            out = model(bp_seq, ctx, cand, mask, history_positions, last_ally_pos=last_ally_pos)
            main_loss, _ = model.compute_loss(out["logits"], labels, out["aux_loss"], is_pick, time_weight)

        all_logits.append(out["logits"].float().cpu())
        all_labels.append(labels.cpu())
        all_masks.append(mask.cpu())
        all_is_pick.append(is_pick.cpu())
        total_main_loss += main_loss.item()
        n_batches += 1

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)
    is_pick = torch.cat(all_is_pick)

    is_pick_flag = (is_pick > 0.5).float()
    metrics = {
        "main_loss": total_main_loss / max(n_batches, 1),
        "Pick@1": recall_at_k(logits, labels, masks, 1, is_pick_flag),
        "Pick@3": recall_at_k(logits, labels, masks, 3, is_pick_flag),
        "Pick@5": recall_at_k(logits, labels, masks, 5, is_pick_flag),
        "Pick@10": recall_at_k(logits, labels, masks, 10, is_pick_flag),
        "Pick@20": recall_at_k(logits, labels, masks, 20, is_pick_flag),
    }
    model.train()
    return metrics


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, grad_clip, step6_downweight, mask_cs, use_amp):
    model.train()
    total_loss = 0.0
    n_batches = 0
    device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

    for batch in train_loader:
        bp_seq = batch["bp_sequence"].to(device)
        ctx = batch["global_context"].to(device)
        cand = batch["candidate_matrix"].to(device)
        mask = batch["available_mask"].to(device)
        labels = batch["label"].to(device)
        is_pick = batch["is_pick"].to(device)
        history_positions = batch.get("history_positions")
        if history_positions is not None:
            history_positions = history_positions.to(device)
        time_weight = batch.get("time_weight")
        if time_weight is not None:
            time_weight = time_weight.to(device)
        last_ally_pos = batch.get("last_ally_pos")
        if last_ally_pos is not None:
            last_ally_pos = last_ally_pos.to(device)

        if mask_cs:
            mask_tensor = torch.ones_like(cand)
            mask_tensor[:, :, CS_FEATURE_INDICES] = 0.0
            cand = cand * mask_tensor

        sample_weight = None
        if step6_downweight < 1.0:
            bp_step = batch.get("bp_step")
            if bp_step is not None:
                bp_step = bp_step.to(device)
                sample_weight = torch.ones(labels.shape[0], device=device)
                step6_mask = (bp_step == 6)
                sample_weight[step6_mask] = step6_downweight

        tuple_partners = batch.get("tuple_partner")
        if tuple_partners is not None:
            tuple_partners = tuple_partners.to(device)
        bp_step_for_loss = batch.get("bp_step")
        if bp_step_for_loss is not None:
            bp_step_for_loss = bp_step_for_loss.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            out = model(bp_seq, ctx, cand, mask, history_positions, last_ally_pos=last_ally_pos)
            _, loss = model.compute_loss(
                out["logits"], labels, out["aux_loss"], is_pick, time_weight,
                sample_weight=sample_weight, tuple_partners=tuple_partners, bp_steps=bp_step_for_loss,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def create_dataloaders(batch_size=32, val_ratio=0.15, num_workers=0):
    context_parquet = os.path.join(SHARED_FEATURES_DIR, "ALL_context.parquet")
    meta_parquet = os.path.join(SHARED_FEATURES_DIR, "ALL_meta_store.parquet")
    player_parquet = os.path.join(SHARED_FEATURES_DIR, "ALL_player_store.parquet")

    train_loader, val_loader = create_train_val_dataloaders(
        context_parquet, meta_parquet, player_parquet,
        VOCAB_PATH, POS_JSON,
        batch_size=batch_size, num_workers=num_workers, val_ratio=val_ratio,
        force_unroll_train=False,
    )
    return train_loader, val_loader


def create_objective(model_tag, device, vocab_size, context_dim, candidate_dim, use_amp=False, use_compile=False):
    def objective(trial):
        # ---- 架构参数 ----
        h_dim = trial.suggest_categorical("h_dim", [256, 384, 512, 768])
        n_heads = trial.suggest_categorical("n_heads", [4, 8, 12, 16])
        if h_dim % n_heads != 0:
            raise optuna.TrialPruned()
            
        n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
        query_dim = trial.suggest_categorical("query_dim", [128, 256, 384])
        c_dim = trial.suggest_categorical("c_dim", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        attention_dropout = trial.suggest_float("attention_dropout", 0.05, 0.25)
        candidate_hidden = trial.suggest_categorical("candidate_hidden", [128, 256, 384])
        tactical_hidden = trial.suggest_categorical("tactical_hidden", [64, 128, 256])

        # ---- 训练参数 ----
        lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.005, 0.1, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.25)
        grad_clip = trial.suggest_float("grad_clip", 0.5, 2.0)
        # NoCS 模型 aux_loss_weight 上限压到 1.0 以下，避免辅助任务过拟合
        # (实测 1.543 时 train aux 收敛至 0.015，val aux 停滞在 0.27，存在严重过拟合)
        if model_tag == "nocs":
            aux_loss_weight = trial.suggest_float("aux_loss_weight", 0.3, 1.0)
        else:
            aux_loss_weight = trial.suggest_float("aux_loss_weight", 0.5, 5.0)
        ban_sample_weight = trial.suggest_float("ban_sample_weight", 0.01, 0.1, log=True)
        step6_downweight = trial.suggest_float("step6_downweight", 0.1, 1.0)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        n_epochs = trial.suggest_int("n_epochs", 40, 80, step=10)
        patience = trial.suggest_int("patience", 10, 25, step=5)

        mask_cs = (model_tag == "nocs")

        torch.manual_seed(42)
        np.random.seed(42)

        model = SearchableTransformerPick(
            vocab_size=vocab_size,
            context_dim=context_dim,
            candidate_dim=candidate_dim,
            h_dim=h_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            query_dim=query_dim,
            c_dim=c_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            candidate_hidden=candidate_hidden,
            tactical_hidden=tactical_hidden,
            aux_loss_weight=aux_loss_weight,
            ban_sample_weight=ban_sample_weight,
        ).to(device)

        if use_compile and device.type == "cuda":
            model = torch.compile(model)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_loader, val_loader = create_dataloaders(batch_size=batch_size, num_workers=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * n_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None

        best_metric = 0.0
        patience_counter = 0
        best_metrics = None
        best_epoch = 0

        try:
            for epoch in range(1, n_epochs + 1):
                t0 = time.time()
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, scheduler, scaler, device, 
                    grad_clip, step6_downweight, mask_cs, use_amp
                )
                val_metrics = evaluate(model, val_loader, device, mask_cs, use_amp)
                elapsed = time.time() - t0

                current_metric = val_metrics["Pick@10"]
                log.info(
                    f"  Trial {trial.number:03d} [{model_tag.upper()}] Epoch {epoch:02d} | "
                    f"P@1={val_metrics['Pick@1']:.4f} P@5={val_metrics['Pick@5']:.4f} "
                    f"P@10={current_metric:.4f} | {elapsed:.1f}s"
                )
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    best_metrics = val_metrics.copy()
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

                trial.report(current_metric, epoch)
                if trial.should_prune():
                    # 在 prune 前记录当前最佳 epoch 和指标，防止 best_epoch=0 写入配置
                    trial.set_user_attr("best_epoch", best_epoch)
                    trial.set_user_attr("best_metric", best_metric)
                    trial.set_user_attr("best_metric_name", "Pick@10")
                    raise optuna.TrialPruned()
        except optuna.TrialPruned:
            log.info(
                f"  Trial {trial.number:03d} [{model_tag.upper()}] PRUNED at epoch {epoch} "
                f"P@10={current_metric:.4f} | h={h_dim} L={n_layers} H={n_heads} "
                f"best_epoch_so_far={best_epoch}"
            )
            return best_metric
        except Exception as e:
            log.warning(f"  Trial {trial.number} FAILED: {e}")
            return 0.0

        if best_metrics is None:
            return 0.0

        p10 = best_metrics["Pick@10"]
        p5 = best_metrics["Pick@5"]
        p1 = best_metrics["Pick@1"]
        log.info(
            f"  Trial {trial.number:03d} [{model_tag.upper()}]: "
            f"P@1={p1:.4f} P@5={p5:.4f} P@10={p10:.4f} "
            f"best_ep={best_epoch} params={n_params:,} | "
            f"h={h_dim} L={n_layers} H={n_heads} q={query_dim} c={c_dim} "
            f"drop={dropout:.3f} adrop={attention_dropout:.3f} "
            f"ch={candidate_hidden} th={tactical_hidden} | "
            f"lr={lr:.6f} wd={weight_decay:.4f} warmup={warmup_ratio:.2f} "
            f"gc={grad_clip:.2f} aux_w={aux_loss_weight:.2f} "
            f"ban_w={ban_sample_weight:.3f} s6dw={step6_downweight:.2f} "
            f"bs={batch_size} ep={n_epochs} pat={patience}"
        )

        # 记录最佳 epoch 和指标到 trial，供 run_search 末尾写入配置文件
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_metric", p10)
        trial.set_user_attr("best_metric_name", "Pick@10")

        return p10

    return objective


def run_search():
    parser = argparse.ArgumentParser(description="BPTacticalTransformerPick Hyperparameter Search (Optuna)")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials per model")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds per model")
    parser.add_argument("--model", type=str, default="cs", choices=["both", "cs", "nocs"])
    parser.add_argument("--resume", action="store_true", help="Resume from existing Optuna SQLite DB")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    args = parser.parse_args()

    total_t0 = time.time()
    log.info("=" * 80)
    log.info("  BPTacticalTransformerPick Hyperparameter Search (Optuna + TPE)")
    log.info(f"  [Acceleration] AMP: {args.amp} | Compile: {args.compile}")
    log.info("=" * 80)
    log.info(f"  Trials per model: {args.n_trials}")
    log.info(f"  Model: {args.model}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"  Device: {device}")

    _, _, vocab_size, _, _ = load_champion_vocabulary(VOCAB_PATH)

    train_loader_tmp, _ = create_dataloaders(batch_size=32, num_workers=0)
    sample_batch = next(iter(train_loader_tmp))
    context_dim = sample_batch["global_context"].shape[-1]
    candidate_dim = sample_batch["candidate_matrix"].shape[-1]
    log.info(f"  Dynamic dims detected: vocab_size={vocab_size}, context_dim={context_dim}, candidate_dim={candidate_dim}")
    del train_loader_tmp

    best_configs = {}

    for model_tag in (["cs", "nocs"] if args.model == "both" else [args.model]):
        log.info(f"\n{'=' * 80}")
        log.info(f"  Searching {model_tag.upper()} Model")
        log.info(f"{'=' * 80}")

        sampler = TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
        db_path = os.path.join(DB_DIR, f"transformer_pick_{model_tag}.db")
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner,
            study_name=f"transformer_pick_{model_tag}", storage=storage, load_if_exists=True,
        )
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, args.n_trials - completed)
        log.info(f"  DB: {db_path} | Completed: {completed} | Remaining: {remaining}")

        obj = create_objective(model_tag, device, vocab_size, context_dim, candidate_dim, args.amp, args.compile)
        if remaining > 0:
            study.optimize(obj, n_trials=remaining, timeout=args.timeout, n_jobs=1)
        else:
            log.info(f"  All {args.n_trials} trials already completed, skipping.")

        log.info(f"\n  {model_tag.upper()} Search Complete:")
        log.info(f"  Best Trial: {study.best_trial.number}")
        log.info(f"  Best Pick@10: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            log.info(f"    {k}: {v}")

        best_configs[model_tag] = {"best_trial": study.best_trial.number, "best_pick10": study.best_value, "params": study.best_params}

        df = study.trials_dataframe()
        df_path = os.path.join(SEARCH_DIR, f"search_transformer_{model_tag}_{_run_ts}.csv")
        df.to_csv(df_path, index=False)

        # 将最佳参数写入正式配置文件，供生产模式加载
        best_trial = study.best_trial
        best_params = best_trial.params
        best_epoch = best_trial.user_attrs.get("best_epoch")
        best_metric = best_trial.user_attrs.get("best_metric", study.best_value)
        model_subtype = "CS" if model_tag == "cs" else "NoCS"
        # 校验 best_epoch 有效，防止 pruned trial 写入 best_epoch=0 导致生产模式训练 0 轮
        if not best_epoch or best_epoch <= 0:
            log.warning(f"  Best trial has invalid best_epoch={best_epoch}, skipping config save for [pick/{model_subtype}]")
            continue
        try:
            from bp_recommendation.config import save_best_params as _save_best_params
            _save_best_params(
                model_type="pick",
                model_subtype=model_subtype,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_metric_name="Pick@10",
                architecture={
                    "h_dim": best_params.get("h_dim"),
                    "c_dim": best_params.get("c_dim"),
                    "query_dim": best_params.get("query_dim"),
                    "n_layers": best_params.get("n_layers"),
                    "n_heads": best_params.get("n_heads"),
                    "dropout": best_params.get("dropout"),
                    "attention_dropout": best_params.get("attention_dropout"),
                    "candidate_hidden": best_params.get("candidate_hidden"),
                    "tactical_hidden": best_params.get("tactical_hidden"),
                },
                optimizer={
                    "learning_rate": best_params.get("lr"),
                    "weight_decay": best_params.get("weight_decay"),
                    "warmup_ratio": best_params.get("warmup_ratio"),
                    "grad_clip": best_params.get("grad_clip"),
                },
                loss={
                    "aux_loss_weight": best_params.get("aux_loss_weight"),
                    "ban_sample_weight": best_params.get("ban_sample_weight"),
                    "step6_downweight": best_params.get("step6_downweight"),
                },
                training={
                    "batch_size": best_params.get("batch_size"),
                    "patience": best_params.get("patience"),
                },
            )
            log.info(f"  Best params for [pick/{model_subtype}] saved to training config")
        except Exception as e:
            log.warning(f"  Failed to save best params to config: {e}")

    config_path = os.path.join(SEARCH_DIR, f"transformer_best_configs_{_run_ts}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2, ensure_ascii=False)

    log.info(f"\n  Total search time: {time.time() - total_t0:.1f}s")
    log.info("=" * 80)


if __name__ == "__main__":
    from logger_config import setup_logging
    setup_logging(log_dir=Path(LOG_DIR))
    
    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _run_fh = logging.FileHandler(os.path.join(LOG_DIR, f"transformer_pick_search_{_run_ts}.log"), encoding="utf-8")
    _run_fh.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _run_fh.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_run_fh)
    
    run_search()