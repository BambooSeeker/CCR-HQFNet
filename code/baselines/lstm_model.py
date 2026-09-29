# -*- coding: utf-8 -*-
"""
LSTM Sequence-to-Sequence Forecasting
======================================
标准 2 层 LSTM + 全连接输出头。
输入：(B, 48, F+3)  —— 标准化特征 + 时间特征
输出：(B, 48)        —— 标准化价格

参考：
  - Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
  - 在电价预测中 LSTM 是最常用的 DL baseline
    (Lago et al., 2021; Tschora et al., 2022)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
from tqdm import tqdm


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1, seq_len: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)       # (B, T, H)
        out = self.fc(out).squeeze(-1)  # (B, T)
        return out


def run_lstm(bundle: Dict[str, Any],
             hidden_dim: int = 128,
             num_layers: int = 2,
             dropout: float = 0.1,
             lr: float = 1e-3,
             epochs: int = 200,
             patience: int = 15,
             batch_size: int = 64,
             device: str = "auto") -> np.ndarray:
    """
    Returns
    -------
    y_pred_raw : (n_test, 48) 原始尺度预测值
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    SEQ_LEN = bundle["config"]["SEQ_LEN"]
    y_mu = float(bundle["scalers"]["y_mu"])
    y_sd = float(bundle["scalers"]["y_sd"])

    # 拼接 X_z + tfeat
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
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    model = LSTMModel(input_dim, hidden_dim, num_layers, dropout, SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"[LSTM] Epoch {epoch:3d}/{epochs}",
                           leave=False, ncols=100):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
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
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [LSTM] Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val:.4f} | patience={wait}/{patience}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [LSTM] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        X_te_dev = X_te.to(device)
        y_pred_z = model(X_te_dev).cpu().numpy()

    y_pred_raw = y_pred_z * y_sd + y_mu
    return y_pred_raw
