#!/usr/bin/env python3
"""
BPTacticalTransformer BAN 超参数搜索 (Optuna + TPE)

适配当前 Ban 模型架构:
  - Late Fusion (后融合 Context)
  - Focal Loss (焦点损失挖掘难样本)
  - candidate_dim=32/33, context_dim=20/24
  - 支持 AMP 和 torch.compile 加速

用法:
    python -m bp_recommendation.model_ban.transformer_ban_search --n_trials 40 --amp --compile
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
from transformers import DistilBertConfig, DistilBertModel, get_cosine_schedule_with_warmup
import optuna
from optuna.samplers import TPESampler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bp_recommendation.model_ban.dataloader_ban import create_train_val_dataloaders
from bp_recommendation.feature_pipeline import load_champion_vocabulary
from logger_config import get_logger

from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())
SHARED_FEATURES_DIR = os.path.join(_PROJECT_ROOT, "bp_recommendation", "features")
CLEANED_DIR = os.path.join(_PROJECT_ROOT, "cleaned_data")
VOCAB_PATH = os.path.join(CLEANED_DIR, "champion_vocabulary.json")
POS_JSON = os.path.join(CLEANED_DIR, "champion_position_mapping.json")
BAN_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BAN_DIR, "checkpoints")
LOG_DIR = os.path.join(BAN_DIR, "logs")
SEARCH_DIR = os.path.join(BAN_DIR, "search_results")
DB_DIR = os.path.join(SEARCH_DIR, "optuna_db")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

log = get_logger(__name__)

class SearchableTransformerBan(nn.Module):
    """提取自 model_ban.py 的结构，开放超参供 Optuna 搜索"""
    def __init__(
        self, candidate_dim: int, context_dim: int, vocab_size: int = 180,
        h_dim: int = 768, c_dim: int = 64, query_dim: int = 256,
        n_layers: int = 6, n_heads: int = 12, dropout: float = 0.15,
        attention_dropout: float = 0.15, candidate_hidden: int = 256,
        n_positions: int = 5, aux_loss_weight: float = 2.0,
        pad_idx: int = 0, temperature: float = None, ban_sample_weight: float = 2.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.query_dim = query_dim
        self.aux_loss_weight = aux_loss_weight
        self.pad_idx = pad_idx
        self.n_positions = n_positions
        self.ban_sample_weight = ban_sample_weight
        self.temperature = temperature if temperature is not None else math.sqrt(query_dim)

        bert_config = DistilBertConfig(
            vocab_size=vocab_size, dim=h_dim, n_layers=n_layers, n_heads=n_heads,
            hidden_dim=h_dim * 4, max_position_embeddings=32, pad_token_id=pad_idx,
            dropout=dropout, attention_dropout=attention_dropout,
            attn_implementation="eager",  # 避免 SDPA 在某些输入形状下 IndexError
        )
        self.bert = DistilBertModel(bert_config)
        
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, c_dim), nn.GELU(), nn.Linear(c_dim, c_dim),
        )
        self.bert_proj = nn.Linear(h_dim, query_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(query_dim + c_dim, query_dim), nn.GELU(), nn.LayerNorm(query_dim),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(candidate_hidden, candidate_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(candidate_hidden // 2, query_dim),
        )
        self.candidate_norm = nn.LayerNorm(query_dim)
        self.enemy_role_head = nn.Sequential(
            nn.Linear(h_dim, candidate_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(candidate_hidden, 128), nn.GELU(), nn.Linear(128, n_positions),
        )
        self._init_weights()

    def _init_weights(self):
        for module in [self.context_mlp, self.fusion_proj, self.candidate_mlp, self.enemy_role_head]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, bp_sequence, global_context, candidate_matrix, available_mask, history_positions=None):
        seq_mask = (bp_sequence != self.pad_idx).long()
        bert_out = self.bert(input_ids=bp_sequence, attention_mask=seq_mask)
        bp_hidden_states = bert_out.last_hidden_state

        mask_expanded = seq_mask.unsqueeze(-1).expand(bp_hidden_states.size()).float()
        sum_hidden = torch.sum(bp_hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_hidden = sum_hidden / sum_mask

        query_bert = self.bert_proj(pooled_hidden)
        ctx_embed = self.context_mlp(global_context)
        query_fused = self.fusion_proj(torch.cat([query_bert, ctx_embed], dim=-1))

        cand_embed = self.candidate_mlp(candidate_matrix)
        cand_embed = self.candidate_norm(cand_embed)

        raw_logits = torch.bmm(query_fused.unsqueeze(1), cand_embed.transpose(1, 2)).squeeze(1) / self.temperature
        masked_logits = raw_logits.clamp(-30, 30) + (1.0 - available_mask) * (-1e9)

        aux_loss = torch.tensor(0.0, device=bp_sequence.device, requires_grad=True)
        if history_positions is not None:
            role_logits = self.enemy_role_head(bp_hidden_states)
            valid_logits = role_logits.view(-1, self.n_positions) 
            valid_labels = history_positions.view(-1)             
            if (valid_labels != -1).any():
                aux_loss = F.cross_entropy(valid_logits, valid_labels, ignore_index=-1)

        total_aux_loss = self.aux_loss_weight * aux_loss if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad else 0.0

        return {"logits": masked_logits, "aux_loss": total_aux_loss}

    def compute_loss(self, logits, labels, aux_loss, is_pick=None, time_weight=None):
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        target_probs = probs[torch.arange(labels.size(0)), labels]
        
        focal_weight = torch.pow(1.0 - target_probs + 1e-6, 1.0)
        per_sample_loss = ce_loss * focal_weight

        if is_pick is not None:
            action_weights = torch.where(is_pick < 0.5, torch.tensor(self.ban_sample_weight, device=logits.device), torch.tensor(1.0, device=logits.device))
            per_sample_loss = per_sample_loss * action_weights
            
        if time_weight is not None:
            per_sample_loss = per_sample_loss * time_weight

        return per_sample_loss.mean(), per_sample_loss.mean() + aux_loss


def recall_at_k(logits, labels, available_mask, k, is_pick=None):
    masked = logits.clone()
    masked[available_mask == 0] = -1e9
    _, topk_indices = masked.topk(k, dim=-1)
    hit = (topk_indices == labels.unsqueeze(1)).any(dim=1)
    if is_pick is not None:
        hit = hit[is_pick.bool()] if is_pick.sum() > 0 else hit[:0]
    if hit.numel() == 0: return 0.0
    return hit.float().mean().item()


def create_objective(device, vocab_size, context_dim, candidate_dim, use_amp=False, use_compile=False):
    def objective(trial):
        h_dim = trial.suggest_categorical("h_dim", [384, 512, 768])
        n_heads = trial.suggest_categorical("n_heads", [6, 8, 12])
        if h_dim % n_heads != 0: raise optuna.TrialPruned()
            
        n_layers = trial.suggest_categorical("n_layers", [3, 4, 6])
        query_dim = trial.suggest_categorical("query_dim", [128, 256, 384])
        c_dim = trial.suggest_categorical("c_dim", [32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.25)
        attention_dropout = trial.suggest_float("attention_dropout", 0.05, 0.2)
        candidate_hidden = trial.suggest_categorical("candidate_hidden", [128, 256, 512])

        lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.005, 0.05, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
        grad_clip = trial.suggest_float("grad_clip", 0.5, 1.5)
        aux_loss_weight = trial.suggest_float("aux_loss_weight", 0.5, 1.2)  # 压缩自 [1.0, 4.0]，防止 aux 过拟合
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        n_epochs = trial.suggest_int("n_epochs", 50, 80, step=10)
        patience = trial.suggest_int("patience", 15, 25, step=5)

        torch.manual_seed(42)
        np.random.seed(42)

        model = SearchableTransformerBan(
            candidate_dim=candidate_dim, context_dim=context_dim, vocab_size=vocab_size,
            h_dim=h_dim, c_dim=c_dim, query_dim=query_dim, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, attention_dropout=attention_dropout, candidate_hidden=candidate_hidden,
            aux_loss_weight=aux_loss_weight,
        ).to(device)

        if use_compile and device.type == "cuda":
            model = torch.compile(model)

        train_loader, val_loader = create_train_val_dataloaders(
            os.path.join(SHARED_FEATURES_DIR, "ALL_context.parquet"),
            os.path.join(SHARED_FEATURES_DIR, "ALL_meta_store.parquet"),
            os.path.join(SHARED_FEATURES_DIR, "ALL_player_store.parquet"),
            VOCAB_PATH, POS_JSON, batch_size=batch_size, num_workers=0
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * n_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)
        scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
        device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'

        best_b10 = 0.0
        patience_counter = 0
        best_epoch = 0

        try:
            for epoch in range(1, n_epochs + 1):
                t0 = time.time()
                # Train
                model.train()
                for batch in train_loader:
                    bp_seq = batch["bp_sequence"].to(device)
                    ctx = batch["global_context"].to(device)
                    cand = batch["candidate_matrix"].to(device)
                    mask = batch["available_mask"].to(device)
                    labels = batch["label"].to(device)
                    is_pick = batch["is_pick"].to(device)
                    history_positions = batch.get("history_positions", None)
                    if history_positions is not None: history_positions = history_positions.to(device)
                    time_weight = batch.get("time_weight", None)
                    if time_weight is not None: time_weight = time_weight.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device_type, enabled=use_amp):
                        out = model(bp_seq, ctx, cand, mask, history_positions)
                        _, loss = model.compute_loss(out["logits"], labels, out["aux_loss"], is_pick, time_weight)

                    if scaler:
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

                # Eval
                model.eval()
                all_logits, all_labels, all_masks, all_is_pick = [], [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        bp_seq = batch["bp_sequence"].to(device)
                        ctx = batch["global_context"].to(device)
                        cand = batch["candidate_matrix"].to(device)
                        mask = batch["available_mask"].to(device)
                        all_logits.append(model(bp_seq, ctx, cand, mask)["logits"].cpu())
                        all_labels.append(batch["label"])
                        all_masks.append(mask.cpu())
                        all_is_pick.append(batch["is_pick"])

                logits = torch.cat(all_logits)
                is_ban_flag = 1.0 - torch.cat(all_is_pick).float()
                current_b10 = recall_at_k(logits, torch.cat(all_labels), torch.cat(all_masks), 10, is_ban_flag)
                elapsed = time.time() - t0
                log.info(f"  Trial {trial.number:03d} Epoch {epoch:02d} | B@10={current_b10:.4f} | {elapsed:.1f}s")

                if current_b10 > best_b10:
                    best_b10 = current_b10
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience: break

                trial.report(current_b10, epoch)
                if trial.should_prune():
                    # 在 prune 前记录当前最佳 epoch 和指标，防止 best_epoch=0 写入配置
                    trial.set_user_attr("best_epoch", best_epoch)
                    trial.set_user_attr("best_metric", best_b10)
                    trial.set_user_attr("best_metric_name", "Ban@10")
                    raise optuna.TrialPruned()

        except optuna.TrialPruned:
            return best_b10
        except Exception as e:
            log.warning(f"  Trial {trial.number} FAILED: {e}")
            return 0.0

        # 记录最佳 epoch 和指标到 trial，供 run_search 末尾写入配置文件
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_metric", best_b10)
        trial.set_user_attr("best_metric_name", "Ban@10")

        return best_b10

    return objective


def run_search():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("  BPTacticalTransformer BAN Hyperparameter Search")
    log.info("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    _, _, vocab_size, _, _ = load_champion_vocabulary(VOCAB_PATH)

    tmp_loader, _ = create_train_val_dataloaders(
        os.path.join(SHARED_FEATURES_DIR, "ALL_context.parquet"),
        os.path.join(SHARED_FEATURES_DIR, "ALL_meta_store.parquet"),
        os.path.join(SHARED_FEATURES_DIR, "ALL_player_store.parquet"),
        VOCAB_PATH, POS_JSON, batch_size=32, num_workers=0
    )
    sample = next(iter(tmp_loader))
    context_dim, candidate_dim = sample["global_context"].shape[-1], sample["candidate_matrix"].shape[-1]
    del tmp_loader

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name="transformer_ban", storage=f"sqlite:///{os.path.join(DB_DIR, 'transformer_ban.db')}", load_if_exists=True,
    )
    
    obj = create_objective(device, vocab_size, context_dim, candidate_dim, args.amp, args.compile)
    study.optimize(obj, n_trials=args.n_trials, timeout=args.timeout)

    log.info(f"\n  Best Ban@10: {study.best_value:.4f}")
    for k, v in study.best_params.items(): log.info(f"    {k}: {v}")
    
    with open(os.path.join(SEARCH_DIR, f"ban_tf_best_configs_{_run_ts}.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    # 将最佳参数写入正式配置文件，供生产模式加载
    best_trial = study.best_trial
    best_params = best_trial.params
    best_epoch = best_trial.user_attrs.get("best_epoch")
    best_metric = best_trial.user_attrs.get("best_metric", study.best_value)
    # 校验 best_epoch 有效，防止 pruned trial 写入 best_epoch=0 导致生产模式训练 0 轮
    if not best_epoch or best_epoch <= 0:
        log.warning(f"  Best trial has invalid best_epoch={best_epoch}, skipping config save for [ban/CS]")
    else:
        try:
            from bp_recommendation.config import save_best_params as _save_best_params
            _save_best_params(
                model_type="ban",
                model_subtype="CS",
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_metric_name="Ban@10",
                architecture={
                    "h_dim": best_params.get("h_dim"),
                    "c_dim": best_params.get("c_dim"),
                    "query_dim": best_params.get("query_dim"),
                    "n_layers": best_params.get("n_layers"),
                    "n_heads": best_params.get("n_heads"),
                    "dropout": best_params.get("dropout"),
                    "attention_dropout": best_params.get("attention_dropout"),
                    "candidate_hidden": best_params.get("candidate_hidden"),
                },
                optimizer={
                    "learning_rate": best_params.get("lr"),
                    "weight_decay": best_params.get("weight_decay"),
                    "warmup_ratio": best_params.get("warmup_ratio"),
                    "grad_clip": best_params.get("grad_clip"),
                },
                loss={
                    "aux_loss_weight": best_params.get("aux_loss_weight"),
                },
                training={
                    "batch_size": best_params.get("batch_size"),
                    "patience": best_params.get("patience"),
                },
            )
            log.info(f"  Best params for [ban/CS] saved to training config")
        except Exception as e:
            log.warning(f"  Failed to save best params to config: {e}")

if __name__ == "__main__":
    from logger_config import setup_logging
    setup_logging(log_dir=Path(LOG_DIR))
    
    _run_ts = time.strftime("%Y%m%d_%H%M%S")
    _run_fh = logging.FileHandler(os.path.join(LOG_DIR, f"transformer_ban_search_{_run_ts}.log"), encoding="utf-8")
    _run_fh.setLevel(logging.INFO)
    _run_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _run_fh.setFormatter(_run_fmt)
    logging.getLogger().addHandler(_run_fh)
    
    run_search()