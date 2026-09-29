# -*- coding: utf-8 -*-
"""
Transformer — Vanilla Encoder-Only Transformer for Time Series Forecasting
===========================================================================
Architecture:
  Input projection → (Positional Enc + Time Feature Enc) → Transformer Encoder
  → Output projection

Integrates with dl_trainer.py via the standard forward signature:
    forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    → (B, pred_len, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    Vanilla Transformer Encoder for electricity price forecasting.

    Parameters (from DLConfigs)
    ---------------------------
    seq_len  : input sequence length
    pred_len : output prediction length
    enc_in   : number of input features
    d_model  : model dimension
    n_heads  : number of attention heads
    e_layers : number of encoder layers
    d_ff     : feedforward dimension
    dropout  : dropout rate
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        d_model = configs.d_model
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        d_ff = configs.d_ff
        dropout = configs.dropout

        # Input projection: enc_in → d_model
        self.input_proj = nn.Linear(configs.enc_in, d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, configs.seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time feature projection: 3 → d_model (adds to x)
        self.time_proj = nn.Linear(3, d_model)

        # Transformer encoder (Post-LN for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection: d_model → 1
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc      : (B, T, enc_in)
        # x_mark_enc : (B, T, 3)  — [sin_slot, cos_slot, slot_norm]

        x = self.input_proj(x_enc)          # (B, T, d_model)
        x = x + self.pos_embed              # learnable positional encoding

        if x_mark_enc is not None:
            x = x + self.time_proj(x_mark_enc)  # fuse time features

        x = self.transformer_encoder(x)     # (B, T, d_model)
        out = self.output_proj(x)           # (B, T, 1)

        return out[:, -self.pred_len:, :]   # (B, pred_len, 1)
