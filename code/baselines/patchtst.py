# -*- coding: utf-8 -*-
"""
PatchTST — task-adapted single-target version
=============================================
Based on THUML PatchTST, but adapted for this project:
  1. use a curated subset of economically meaningful features;
  2. map multivariate patch forecasts to one target price trajectory;
  3. add an anchor branch to stabilize the price level.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


def _pick_feature_indices(feature_names: List[str], max_features: int = 20) -> List[int]:
    if not feature_names:
        return []
    names = [str(x).lower() for x in feature_names]
    selected: List[int] = []
    groups = [
        ["price_day_ahead_load", "price_day_ahead", "price_real_lag48", "price_real_lag96", "lag48", "lag96"],
        ["bidding_space", "demand", "supply", "renewable_penetration", "coal_gas_nuclear", "load", "wind", "photo", "water"],
        ["rollmean", "rollstd", "peakvalley", "range", "shock", "tight", "risk", "regime", "vol"],
    ]
    for group in groups:
        for kw in group:
            for i, name in enumerate(names):
                if i in selected:
                    continue
                if kw in name:
                    selected.append(i)
                    if len(selected) >= max_features:
                        return selected
    if len(selected) < min(max_features, len(feature_names)):
        for i in range(len(feature_names)):
            if i not in selected:
                selected.append(i)
                if len(selected) >= max_features:
                    break
    return selected


def _pick_anchor_index(feature_names: List[str]) -> int:
    names = [str(x).lower() for x in feature_names]
    for kw in ["price_day_ahead_load", "price_day_ahead", "price_real_lag48", "price_real_lag96", "lag48"]:
        for i, name in enumerate(names):
            if kw in name:
                return i
    return 0


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTSTModel(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        feature_names = list(getattr(configs, "feature_names", []))
        selected = _pick_feature_indices(feature_names, max_features=min(20, len(feature_names) if feature_names else 20))
        if not selected:
            selected = list(range(configs.enc_in))
        self.selected_feature_indices = selected
        self.anchor_feature_idx = _pick_anchor_index(feature_names) if feature_names else 0
        self.n_selected = len(self.selected_feature_indices)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.n_selected, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        hidden = max(8, min(64, self.n_selected + 3))
        self.target_head = nn.Sequential(
            nn.Linear(self.n_selected + 3, hidden),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden, 1),
        )
        self.anchor_proj = nn.Linear(1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_use = x_enc[:, :, self.selected_feature_indices]
        anchor = x_enc[:, :, self.anchor_feature_idx:self.anchor_feature_idx + 1]

        means = x_use.mean(1, keepdim=True).detach()
        x_use = x_use - means
        stdev = torch.sqrt(torch.var(x_use, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_use = x_use / stdev

        x_use = x_use.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_use)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        multi_forecast = self.head(enc_out).permute(0, 2, 1)
        multi_forecast = multi_forecast * stdev[:, 0, :].unsqueeze(1)
        multi_forecast = multi_forecast + means[:, 0, :].unsqueeze(1)

        if x_mark_enc is not None:
            fusion_in = torch.cat([multi_forecast, x_mark_enc], dim=-1)
        else:
            zeros = torch.zeros(multi_forecast.size(0), multi_forecast.size(1), 3, device=multi_forecast.device, dtype=multi_forecast.dtype)
            fusion_in = torch.cat([multi_forecast, zeros], dim=-1)
        residual = self.target_head(fusion_in)
        anchor_term = self.anchor_proj(anchor)
        return anchor_term + self.residual_scale * residual

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
