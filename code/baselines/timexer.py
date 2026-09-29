# -*- coding: utf-8 -*-
"""
Task-adapted TimeXer for direct electricity-price mapping
=========================================================
适配当前任务：用“价格/anchor 历史线索”作为 endogenous branch，
用未来 48 步市场结构特征作为 exogenous branch，再做 cross-attention。

这不是照搬 THUML 原版，而是面向“未来协变量已知”的日前电价任务重构。
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_feature_indices(feature_names: Sequence[str]):
    names = list(feature_names or [])
    lower = [n.lower() for n in names]
    name_to_idx = {n: i for i, n in enumerate(lower)}

    def exact(keys):
        out = []
        for k in keys:
            if k in name_to_idx:
                out.append(name_to_idx[k])
        return out

    anchor = exact([
        "price_day_ahead", "price_day_ahead_load", "price_real_lag48",
        "price_real_lag96", "price_day_ahead_lag48",
    ])
    if not anchor and names:
        anchor = [0]

    endogenous = exact([
        "price_real_lag48", "price_real_lag49", "price_real_lag96", "price_real_lag144", "price_real_lag336",
        "price_day_ahead", "price_day_ahead_lag48", "price_day_ahead_lag96",
        "price_real_rollmean24h", "price_real_rollstd24h", "price_real_peakvalley24h",
        "price_day_ahead_rollmean24h", "price_day_ahead_rollstd24h",
    ])
    exogenous = list(dict.fromkeys(anchor + exact([
        "bidding_space_real", "bidding_space_pred", "demand_real", "demand_pred",
        "supply_real", "supply_pred", "renewable_penetration_real", "renewable_penetration_pred",
        "coal_gas_nuclear_real", "coal_gas_nuclear_pred", "load_real", "load_day_ahead_pred",
        "elec_exter_plan", "elec_fix_out_plan", "energy_hydro_renewable",
        "photo_gene_total_pred", "wind_gene_total_pred", "water_gene_total_pred",
        "price_day_ahead_cong", "elec_day_ahead_all",
    ])))

    if not endogenous:
        endogenous = anchor[:]
    if not exogenous:
        exogenous = anchor[:]
    return {
        "anchor_idx": list(dict.fromkeys(anchor)),
        "endo_idx": list(dict.fromkeys(endogenous)),
        "exo_idx": list(dict.fromkeys(exogenous)),
    }


class PatchEmbed1D(nn.Module):
    def __init__(self, patch_len: int, in_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        pad = (self.patch_len - (T % self.patch_len)) % self.patch_len
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad), mode="replicate")
        T2 = x.size(1)
        n_patch = T2 // self.patch_len
        x = x.reshape(B, n_patch, self.patch_len * C)
        pos = self.pos[:, :n_patch, :]
        return self.dropout(self.proj(x) + pos)


class CrossBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross):
        h, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(h))
        h, _ = self.cross_attn(x, cross, cross, need_weights=False)
        x = self.norm2(x + self.drop(h))
        h = self.ffn(x)
        x = self.norm3(x + self.drop(h))
        return x


class TimeXerModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.patch_len = max(2, int(getattr(configs, "patch_len", 4)))
        self.feature_names: List[str] = list(getattr(configs, "feature_names", []) or [])
        groups = _pick_feature_indices(self.feature_names)
        self.anchor_idx = groups["anchor_idx"]
        self.endo_idx = groups["endo_idx"]
        self.exo_idx = groups["exo_idx"]

        endo_dim = len(self.endo_idx) + 3
        exo_dim = len(self.exo_idx) + 3

        self.endo_patch = PatchEmbed1D(self.patch_len, endo_dim, self.d_model, self.dropout)
        self.exo_embed = nn.Sequential(
            nn.Linear(exo_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.horizon_queries = nn.Parameter(torch.randn(1, self.pred_len, self.d_model) * 0.02)
        self.encoder = nn.ModuleList([
            CrossBlock(self.d_model, configs.n_heads, self.dropout) for _ in range(configs.e_layers)
        ])
        self.query_refine = nn.ModuleList([
            CrossBlock(self.d_model, configs.n_heads, self.dropout) for _ in range(max(1, configs.d_layers))
        ])
        self.out_norm = nn.LayerNorm(self.d_model)

        self.anchor_mixer = nn.Sequential(
            nn.Linear(len(self.anchor_idx), self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
        )
        self.anchor_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        endo = torch.cat([x_enc[:, :, self.endo_idx], x_mark_enc], dim=-1)
        exo = torch.cat([x_enc[:, :, self.exo_idx], x_mark_enc], dim=-1)

        endo_tokens = self.endo_patch(endo)
        exo_tokens = self.exo_embed(exo)

        latent = endo_tokens
        for blk in self.encoder:
            latent = blk(latent, exo_tokens)

        q = self.horizon_queries.repeat(x_enc.size(0), 1, 1)
        for blk in self.query_refine:
            q = blk(q, latent)
        q = self.out_norm(q)

        delta = self.delta_head(q)
        anchor_base = self.anchor_mixer(x_enc[:, :, self.anchor_idx])
        gate = self.anchor_gate(q)
        return gate * anchor_base + delta
