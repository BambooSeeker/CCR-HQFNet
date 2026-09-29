# -*- coding: utf-8 -*-
"""
CNN-Transformer Hybrid Model
=============================
改进版：
  1) 残差卷积前端提取局部模式（local branch）
  2) Transformer Encoder 建模全局依赖（global branch）
  3) 门控融合 local/global，提升稳定性与峰值拟合能力
  4) 训练损失支持 weighted_huber（对剧烈波动更鲁棒）

输入:  (B, T, D)
输出:  (B, T)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
from tqdm import tqdm


class CNNTransformerModel(nn.Module):
    """
    CNN-Transformer: Conv1D局部特征 + Transformer全局建模 + 门控融合。

    Parameters
    ----------
    input_dim      : 输入特征维度（enc_in + tfeat）
    d_model        : 隐层维度（同时作为 Transformer d_model）
    kernel_size    : 卷积核大小
    n_heads        : Transformer 多头数
    n_layers       : Transformer 编码器层数
    d_ff           : Feedforward 维度
    dropout        : Dropout 概率
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        kernel_size: int = 5,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()

        # 输入投影（残差支路）
        self.in_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)

        # 局部分支：两层卷积提取短期模式
        self.local_cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer 编码器（pre-norm 在小数据上更稳）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 门控融合：在 local/global 间自适应权衡
        self.gate = nn.Linear(d_model * 2, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x_conv = x.permute(0, 2, 1)                      # (B, D, T)

        # local: 残差卷积
        local = self.local_cnn(x_conv) + self.in_proj(x_conv)  # (B, d_model, T)
        local = local.permute(0, 2, 1)                  # (B, T, d_model)

        # global: Transformer
        T = local.size(1)
        h = local + self.pos_embed[:, :T, :]
        h = self.pos_drop(h)
        global_h = self.transformer_encoder(h)           # (B, T, d_model)

        # gate fusion
        g = torch.sigmoid(self.gate(torch.cat([local, global_h], dim=-1)))
        fused = g * global_h + (1.0 - g) * local

        out = self.fc(self.out_norm(fused)).squeeze(-1) # (B, T)
        return out


def _compute_seq_loss(pred: torch.Tensor, yb: torch.Tensor, kind: str = "weighted_huber") -> torch.Tensor:
    if kind == "mse":
        return nn.functional.mse_loss(pred, yb)
    if kind == "weighted_huber":
        base = nn.functional.smooth_l1_loss(pred, yb, reduction="none", beta=0.5)
        level = torch.abs(yb)
        slope = torch.zeros_like(yb)
        slope[:, 1:] = torch.abs(yb[:, 1:] - yb[:, :-1])
        w = 1.0 + 0.15 * level + 0.25 * slope
        return (base * w).mean()
    raise ValueError(f"Unknown criterion kind: {kind}")


def run_cnn_transformer(
    bundle: Dict[str, Any],
    d_model: int = 128,
    kernel_size: int = 5,
    n_heads: int = 8,
    n_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.15,
    lr: float = 8e-4,
    weight_decay: float = 1e-4,
    epochs: int = 220,
    patience: int = 20,
    batch_size: int = 64,
    criterion_kind: str = "weighted_huber",
    device: str = "auto",
) -> np.ndarray:
    """
    训练 CNN-Transformer 并返回原始尺度预测值。

    Returns
    -------
    y_pred_raw : np.ndarray, shape (n_test, seq_len)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    y_mu = float(bundle["scalers"]["y_mu"])
    y_sd = float(bundle["scalers"]["y_sd"])

    def make_input(split):
        X = np.concatenate([split["X_z"], split["tfeat"]], axis=-1)
        return torch.tensor(X, dtype=torch.float32)

    X_tr = make_input(bundle["train"])
    y_tr = torch.tensor(bundle["train"]["y_z"], dtype=torch.float32)
    X_val = make_input(bundle["val"])
    y_val = torch.tensor(bundle["val"]["y_z"], dtype=torch.float32)
    X_te = make_input(bundle["test"])

    input_dim = X_tr.shape[-1]

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = CNNTransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        kernel_size=kernel_size,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(
            train_loader,
            desc=f"[CNN-Transformer] Epoch {epoch:3d}/{epochs}",
            leave=False,
            ncols=100,
        ):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = _compute_seq_loss(pred, yb, criterion_kind)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += _compute_seq_loss(pred, yb, criterion_kind).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [CNN-Transformer] Epoch {epoch:3d} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"best={best_val:.4f} | lr={cur_lr:.2e} | patience={wait}/{patience}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [CNN-Transformer] Early stopping at epoch {epoch}")
                break

    assert best_state is not None, "[CNN-Transformer] 训练未得到有效模型权重"

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_z = model(X_te.to(device)).cpu().numpy()

    y_pred_raw = y_pred_z * y_sd + y_mu
    return y_pred_raw
