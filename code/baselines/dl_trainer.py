# -*- coding: utf-8 -*-
"""
Deep Learning Trainer — 统一训练循环（滚动预测版 v2）
=====================================================

核心变更（相对于 v1）：
  1. train_dl_model() 新增 warm_start_state 参数
     → 接收上一滚动步训练好的模型权重
     → 以较小学习率（warm_lr = lr * 0.1）微调，节省 ~80% 训练时间
     → 若 warm_start_state=None，则随机初始化（第一步或不使用 Warm Start）
  2. 训练超参数通过函数参数传入（lr / batch_size / epochs / patience）
     → 不再使用硬编码的通用默认值
     → 由 run_benchmarks.py 中的 MODEL_HPARAMS 字典控制
  3. epochs 默认从 200 降至 100，patience 从 20 降至 10
     → 与 THUML 官方 electricity 脚本对齐
  4. 新增 train_dl_model_and_return_state() 便捷函数
     → 同时返回预测结果和模型权重，供外层滚动循环缓存 Warm Start 状态

所有 TSL 官方模型的 forward 签名：
    forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)

本项目中：
  - x_enc      = (B, seq_len, enc_in)  标准化特征
  - x_mark_enc = (B, seq_len, 3)       时间特征 [sin_slot, cos_slot, slot_norm]
  - x_dec      = None（encoder-only）或 decoder 输入
  - x_mark_dec = None 或 decoder 时间标记

输出：(B, pred_len, c_out)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ============================================================
# Configs 对象（模拟 TSL 的 argparse Namespace）
# ============================================================
class DLConfigs:
    """模拟 TSL 的 configs 对象，支持任意关键字参数。"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ============================================================
# 内部工具函数
# ============================================================
def _make_tensors(split: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 bundle split 转为 PyTorch 张量。"""
    X = torch.tensor(split["X_z"],    dtype=torch.float32)   # (N, T, F)
    t = torch.tensor(split["tfeat"],  dtype=torch.float32)   # (N, T, 3)
    y = torch.tensor(split["y_z"],    dtype=torch.float32)   # (N, T)
    return X, t, y


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device




def _reduce_pred_to_seq(pred: torch.Tensor) -> torch.Tensor:
    """Normalize model outputs to shape (B, T) for single-target training."""
    if pred.dim() == 2:
        return pred
    if pred.dim() == 3:
        if pred.shape[-1] == 1:
            return pred.squeeze(-1)
        return pred[:, :, -1]
    raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")


def _compute_seq_loss(pred: torch.Tensor, yb: torch.Tensor, kind: str = "mse") -> torch.Tensor:
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


# ============================================================
# 核心训练函数
# ============================================================
def train_dl_model(
    model: nn.Module,
    bundle: Dict[str, Any],
    model_name: str = "DL",
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 64,
    device: str = "auto",
    needs_dec: bool = False,
    warm_start_state: Optional[Dict[str, torch.Tensor]] = None,
    criterion_kind: str = "mse",
) -> np.ndarray:
    """
    统一训练循环（滚动预测版）。

    Parameters
    ----------
    model            : nn.Module，TSL 官方模型实例（已实例化但未移至 device）
    bundle           : 当前滚动步的数据字典
    model_name       : 显示名称，用于日志
    lr               : 初始学习率（由 MODEL_HPARAMS 传入）
    epochs           : 最大训练轮数，默认 100
    patience         : Early Stopping 耐心值，默认 10
    batch_size       : 批大小（由 MODEL_HPARAMS 传入）
    device           : 训练设备
    needs_dec        : 是否需要 decoder 输入（TFT 等 Seq2Seq 模型）
    warm_start_state : 上一滚动步的最优模型权重字典（state_dict）。
                       若提供，则以 lr * 0.1 的学习率微调；
                       若为 None，则随机初始化正常训练。

    Returns
    -------
    y_pred_raw : np.ndarray, shape (n_test, seq_len)，原始尺度预测值
    """
    device = _resolve_device(device)
    SEQ_LEN = bundle["config"]["SEQ_LEN"]
    y_mu = float(bundle["scalers"]["y_mu"])
    y_sd = float(bundle["scalers"]["y_sd"])

    # ── 数据准备 ──────────────────────────────────────────────
    X_tr, t_tr, y_tr = _make_tensors(bundle["train"])
    X_val, t_val, y_val = _make_tensors(bundle["val"])
    X_te, t_te, _ = _make_tensors(bundle["test"])

    train_loader = DataLoader(
        TensorDataset(X_tr, t_tr, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, t_val, y_val),
        batch_size=batch_size, shuffle=False,
    )

    # ── Warm Start ────────────────────────────────────────────
    if warm_start_state is not None:
        try:
            model.load_state_dict(warm_start_state, strict=True)
            effective_lr = lr * 0.1   # 微调学习率
            print(f"  [{model_name}] Warm Start: 加载上一步权重，使用微调 lr={effective_lr:.2e}")
        except RuntimeError as e:
            print(f"  [{model_name}] Warm Start 加载失败（架构变化？），随机初始化。原因: {e}")
            effective_lr = lr
    else:
        effective_lr = lr

    model = model.to(device)

    # ── 优化器与调度器 ────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    # ReduceLROnPlateau：验证 loss 不再下降时，lr 乘以 0.5，耐心 5 轮
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0

    # ── 训练循环 ──────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[{model_name}] Epoch {epoch:3d}/{epochs}",
            leave=False, ncols=120,
        )
        for xb, tb, yb in pbar:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)

            if needs_dec:
                dec_inp = torch.zeros_like(yb).unsqueeze(-1).to(device)
                pred = model(xb, tb, dec_inp, tb)
            else:
                pred = model(xb, tb, None, None)

            pred = _reduce_pred_to_seq(pred)

            loss = _compute_seq_loss(pred, yb, criterion_kind)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader.dataset)

        # ── 验证 ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, tb, yb in val_loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                if needs_dec:
                    dec_inp = torch.zeros_like(yb).unsqueeze(-1).to(device)
                    pred = model(xb, tb, dec_inp, tb)
                else:
                    pred = model(xb, tb, None, None)
                pred = _reduce_pred_to_seq(pred)
                val_loss += _compute_seq_loss(pred, yb, criterion_kind).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{model_name}] Epoch {epoch:3d} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"best={best_val:.4f} | lr={cur_lr:.2e} | "
                f"patience={wait}/{patience}"
            )

        # ── Early Stopping ────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch}")
                break

    # ── 推理 ──────────────────────────────────────────────────
    assert best_state is not None, f"[{model_name}] 训练未产生有效权重，请检查数据或模型。"
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_loader = DataLoader(
        TensorDataset(X_te, t_te),
        batch_size=batch_size, shuffle=False,
    )

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            xb, tb = batch[0].to(device), batch[1].to(device)
            if needs_dec:
                dec_inp = torch.zeros(xb.size(0), SEQ_LEN, 1, device=device)
                pred = model(xb, tb, dec_inp, tb)
            else:
                pred = model(xb, tb, None, None)
            pred = _reduce_pred_to_seq(pred)
            preds.append(pred.cpu().numpy())

    y_pred_z = np.concatenate(preds, axis=0)          # (N_test, T)
    y_pred_raw = y_pred_z * y_sd + y_mu               # 反标准化
    return y_pred_raw


# ============================================================
# 便捷函数：同时返回预测值和模型权重（供 Warm Start 缓存）
# ============================================================
def train_dl_model_and_return_state(
    model: nn.Module,
    bundle: Dict[str, Any],
    model_name: str = "DL",
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 64,
    device: str = "auto",
    needs_dec: bool = False,
    warm_start_state: Optional[Dict[str, torch.Tensor]] = None,
    criterion_kind: str = "mse",
) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
    """
    与 train_dl_model() 完全相同，但额外返回训练好的最优模型权重。

    Returns
    -------
    y_pred_raw   : np.ndarray, shape (n_test, seq_len)
    best_state   : dict，最优模型的 state_dict（CPU 张量），可直接传给下一步的 warm_start_state
    """
    device = _resolve_device(device)
    SEQ_LEN = bundle["config"]["SEQ_LEN"]
    y_mu = float(bundle["scalers"]["y_mu"])
    y_sd = float(bundle["scalers"]["y_sd"])

    X_tr, t_tr, y_tr = _make_tensors(bundle["train"])
    X_val, t_val, y_val = _make_tensors(bundle["val"])
    X_te, t_te, _ = _make_tensors(bundle["test"])

    train_loader = DataLoader(
        TensorDataset(X_tr, t_tr, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, t_val, y_val),
        batch_size=batch_size, shuffle=False,
    )

    if warm_start_state is not None:
        try:
            model.load_state_dict(warm_start_state, strict=True)
            effective_lr = lr * 0.1
            print(f"  [{model_name}] Warm Start: lr={effective_lr:.2e}")
        except RuntimeError:
            effective_lr = lr
    else:
        effective_lr = lr

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, tb, yb in train_loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            pred = model(xb, tb, None, None) if not needs_dec else \
                   model(xb, tb, torch.zeros_like(yb).unsqueeze(-1).to(device), tb)
            pred = _reduce_pred_to_seq(pred)
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
            for xb, tb, yb in val_loader:
                xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
                pred = model(xb, tb, None, None) if not needs_dec else \
                       model(xb, tb, torch.zeros_like(yb).unsqueeze(-1).to(device), tb)
                pred = _reduce_pred_to_seq(pred)
                val_loss += _compute_seq_loss(pred, yb, criterion_kind).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_loader = DataLoader(TensorDataset(X_te, t_te), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, tb in test_loader:
            xb, tb = xb.to(device), tb.to(device)
            pred = model(xb, tb, None, None) if not needs_dec else \
                   model(xb, tb, torch.zeros(xb.size(0), SEQ_LEN, 1, device=device), tb)
            pred = _reduce_pred_to_seq(pred)
            preds.append(pred.cpu().numpy())

    y_pred_z = np.concatenate(preds, axis=0)
    y_pred_raw = y_pred_z * y_sd + y_mu
    return y_pred_raw, best_state
