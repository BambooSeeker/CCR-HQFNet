# -*- coding: utf-8 -*-
"""
CNN-LSTM Hybrid Model
=====================
Conv1D 提取局部模式 → LSTM 捕捉时序依赖 → FC 输出。

参考：
  - Kim & Won (2018): "Forecasting stock prices with a feature fusion
    LSTM-CNN model using different representations of the same data"
  - Lago et al. (2018): "Forecasting spot electricity prices: Deep learning
    approaches and empirical comparison of traditional algorithms"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any
from tqdm import tqdm


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int, cnn_filters: int = 64,
                 kernel_size: int = 3, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (B, T, D) → CNN expects (B, D, T)
        c = self.cnn(x.permute(0, 2, 1))  # (B, C, T)
        c = c.permute(0, 2, 1)            # (B, T, C)
        out, _ = self.lstm(c)             # (B, T, H)
        out = self.fc(out).squeeze(-1)    # (B, T)
        return out


def run_cnn_lstm(bundle: Dict[str, Any],
                 cnn_filters: int = 64,
                 kernel_size: int = 3,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 epochs: int = 200,
                 patience: int = 15,
                 batch_size: int = 64,
                 device: str = "auto") -> np.ndarray:

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

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = CNNLSTMModel(input_dim, cnn_filters, kernel_size, lstm_hidden, lstm_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"[CNN-LSTM] Epoch {epoch:3d}/{epochs}",
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
            print(f"  [CNN-LSTM] Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val:.4f} | patience={wait}/{patience}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [CNN-LSTM] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_z = model(X_te.to(device)).cpu().numpy()

    y_pred_raw = y_pred_z * y_sd + y_mu
    return y_pred_raw
