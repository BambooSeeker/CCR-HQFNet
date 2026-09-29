# -*- coding: utf-8 -*-
"""
Task-adapted TFT for direct electricity-price mapping
=====================================================
适配当前任务：已知未来 48 步市场结构特征，直接映射未来 48 步价格。

改造重点：
  1) 不再把 TFT 当成“只有时间特征的 seq2seq”；
  2) 显式拆分 history-like anchors / future-known covariates / regime context；
  3) 预测 residual，再叠加可学习 anchor，减轻 level 拟合压力；
  4) 保留 TFT 的核心味道：变量选择 + LSTM + attention + gating。
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# helpers
# =========================

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
    history_like = exact([
        "price_real_lag48", "price_real_lag49", "price_real_lag96", "price_real_lag144", "price_real_lag336",
        "price_day_ahead", "price_day_ahead_lag48", "price_day_ahead_lag96", "price_day_ahead_lag144",
        "price_real_rollmean24h", "price_real_rollstd24h", "price_real_peakvalley24h",
        "price_day_ahead_rollmean24h", "price_day_ahead_rollstd24h",
    ])
    regime_like = exact([
        "bidding_space_real", "bidding_space_pred", "demand_real", "demand_pred",
        "supply_real", "supply_pred", "renewable_penetration_real", "renewable_penetration_pred",
        "coal_gas_nuclear_real", "coal_gas_nuclear_pred", "load_real", "load_day_ahead_pred",
        "elec_exter_plan", "elec_fix_out_plan", "energy_hydro_renewable",
        "photo_gene_total_pred", "wind_gene_total_pred", "water_gene_total_pred",
    ])

    future_known = list(dict.fromkeys(anchor + regime_like + exact([
        "price_day_ahead_cong", "elec_day_ahead_all", "price_day_ahead_load",
        "bidding_space_real_lag48", "demand_real_lag48", "supply_real_lag48",
        "renewable_penetration_real_lag48",
    ])))

    if not anchor:
        anchor = [0] if names else []
    if not history_like:
        history_like = anchor[:]
    if not regime_like:
        regime_like = anchor[:]
    if not future_known:
        future_known = anchor + regime_like

    return {
        "anchor_idx": list(dict.fromkeys(anchor)),
        "history_idx": list(dict.fromkeys(history_like)),
        "future_idx": list(dict.fromkeys(future_known)),
        "regime_idx": list(dict.fromkeys(regime_like)),
    }


class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        return self.glu(torch.cat([self.fc1(x), self.fc2(x)], dim=-1))


class GateAddNorm(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.glu = GLU(input_size, input_size)
        self.proj = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x, skip):
        x = self.glu(x)
        x = x + skip
        return self.norm(self.proj(x))


class GRN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, context_size=None, dropout=0.0):
        super().__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.lin_a = nn.Linear(input_size, hidden_size)
        self.lin_c = nn.Linear(context_size, hidden_size) if context_size is not None else None
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.skip_proj = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate = GateAddNorm(hidden_size, output_size)

    def forward(self, a, c=None):
        x = self.lin_a(a)
        if c is not None and self.lin_c is not None:
            x = x + (self.lin_c(c).unsqueeze(1) if c.dim() == 2 else self.lin_c(c))
        x = F.elu(x)
        x = self.dropout(self.lin_i(x))
        return self.gate(x, self.skip_proj(a))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, variable_num, dropout=0.0):
        super().__init__()
        self.variable_num = variable_num
        self.joint_grn = GRN(d_model * variable_num, variable_num, hidden_size=d_model,
                             context_size=d_model, dropout=dropout)
        self.variable_grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(variable_num)])

    def forward(self, x, context=None):
        # x: (B, T, C, d)
        flat = torch.flatten(x, start_dim=-2)
        weights = F.softmax(self.joint_grn(flat, context), dim=-1)  # (B, T, C)
        processed = torch.stack([grn(x[..., i, :]) for i, grn in enumerate(self.variable_grns)], dim=-2)
        return torch.sum(processed * weights.unsqueeze(-1), dim=-2)


class TFTModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.feature_names: List[str] = list(getattr(configs, "feature_names", []) or [])
        groups = _pick_feature_indices(self.feature_names)
        self.anchor_idx = groups["anchor_idx"]
        self.history_idx = groups["history_idx"]
        self.future_idx = groups["future_idx"]
        self.regime_idx = groups["regime_idx"]
        self.n_time = 3

        self.history_embeddings = nn.ModuleList([nn.Linear(1, self.d_model) for _ in self.history_idx])
        self.future_embeddings = nn.ModuleList([nn.Linear(1, self.d_model) for _ in self.future_idx])
        self.time_embeddings_hist = nn.ModuleList([nn.Linear(1, self.d_model) for _ in range(self.n_time)])
        self.time_embeddings_fut = nn.ModuleList([nn.Linear(1, self.d_model) for _ in range(self.n_time)])

        self.history_vsn = VariableSelectionNetwork(self.d_model, len(self.history_idx) + self.n_time, self.dropout)
        self.future_vsn = VariableSelectionNetwork(self.d_model, len(self.future_idx) + self.n_time, self.dropout)

        self.history_lstm = nn.LSTM(self.d_model, self.d_model, batch_first=True)
        self.future_lstm = nn.LSTM(self.d_model, self.d_model, batch_first=True)
        self.gate_after_lstm = GateAddNorm(self.d_model, self.d_model)

        self.static_context = nn.Sequential(
            nn.Linear(len(self.regime_idx) + self.n_time, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.enrichment_grn = GRN(self.d_model, self.d_model, context_size=self.d_model, dropout=self.dropout)

        self.attn = nn.MultiheadAttention(self.d_model, configs.n_heads, dropout=self.dropout, batch_first=True)
        self.attn_gate = GateAddNorm(self.d_model, self.d_model)
        self.ffn = GRN(self.d_model, self.d_model, hidden_size=2 * self.d_model, dropout=self.dropout)
        self.final_gate = GateAddNorm(self.d_model, self.d_model)

        # residual over anchor
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

    def _embed_selected(self, x, indices, emb_layers):
        return [emb_layers[k](x[:, :, i:i+1]) for k, i in enumerate(indices)]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, F = x_enc.shape
        hist_embeds = self._embed_selected(x_enc, self.history_idx, self.history_embeddings)
        fut_embeds = self._embed_selected(x_enc, self.future_idx, self.future_embeddings)
        hist_time = [emb(x_mark_enc[:, :, i:i+1]) for i, emb in enumerate(self.time_embeddings_hist)]
        fut_time = [emb(x_mark_enc[:, :, i:i+1]) for i, emb in enumerate(self.time_embeddings_fut)]

        history_input = torch.stack(hist_embeds + hist_time, dim=-2)
        future_input = torch.stack(fut_embeds + fut_time, dim=-2)

        regime_raw = torch.cat([
            x_enc[:, :, self.regime_idx].mean(dim=1),
            x_mark_enc.mean(dim=1),
        ], dim=-1)
        static_ctx = self.static_context(regime_raw)

        history_selected = self.history_vsn(history_input, static_ctx)
        future_selected = self.future_vsn(future_input, static_ctx)

        hist_out, state = self.history_lstm(history_selected)
        fut_out, _ = self.future_lstm(future_selected, state)

        temporal_input = torch.cat([history_selected, future_selected], dim=1)
        temporal_features = torch.cat([hist_out, fut_out], dim=1)
        temporal_features = self.gate_after_lstm(temporal_features, temporal_input)
        enriched = self.enrichment_grn(temporal_features, static_ctx)

        attn_out, _ = self.attn(enriched, enriched, enriched, need_weights=False)
        attn_out = self.attn_gate(attn_out, enriched)
        future_latent = attn_out[:, -T:, :]
        out = self.ffn(future_latent)
        out = self.final_gate(out, temporal_features[:, -T:, :])

        delta = self.delta_head(out)
        anchor_stack = x_enc[:, :, self.anchor_idx]
        anchor_base = self.anchor_mixer(anchor_stack)
        gate = self.anchor_gate(out)
        return gate * anchor_base + delta
