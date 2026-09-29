"""
评估指标计算
"""

import numpy as np
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict:
    """
    计算所有评估指标

    参数：
    - y_true: (n_samples, 48) 真实值
    - y_pred: (n_samples, 48) 预测值

    返回：字典，包含所有指标
    """
    # 展平
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # MAE
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))

    # RMSE
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))

    # =========================
    # 标准版 MAPE
    # MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    # 对 y_true == 0 的点，标准 MAPE 无定义，这里跳过
    # =========================
    mape_mask = (y_true_flat != 0)
    if mape_mask.sum() > 0:
        mape = np.mean(
            np.abs((y_true_flat[mape_mask] - y_pred_flat[mape_mask]) / y_true_flat[mape_mask])
        ) * 100
    else:
        mape = np.nan

    # =========================
    # 标准版 sMAPE
    # sMAPE = mean(2 * |y_pred - y_true| / (|y_true| + |y_pred|)) * 100
    # 对分母为 0 的点（即 y_true=0 且 y_pred=0），跳过
    # =========================
    smape_denom = np.abs(y_true_flat) + np.abs(y_pred_flat)
    smape_mask = (smape_denom != 0)
    if smape_mask.sum() > 0:
        smape = np.mean(
            2.0 * np.abs(y_pred_flat[smape_mask] - y_true_flat[smape_mask]) / smape_denom[smape_mask]
        ) * 100
    else:
        smape = np.nan

    # R²
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # =========================
    # 极端价格评估（改为双侧分位数定义）
    # 下5% 或 上5% 视为 extreme
    # =========================
    low_thr = np.quantile(y_true_flat, 0.05)
    high_thr = np.quantile(y_true_flat, 0.95)

    extreme_mask = (y_true_flat <= low_thr) | (y_true_flat >= high_thr)
    if extreme_mask.sum() > 0:
        mae_extreme = np.mean(np.abs(y_true_flat[extreme_mask] - y_pred_flat[extreme_mask]))
        rmse_extreme = np.sqrt(np.mean((y_true_flat[extreme_mask] - y_pred_flat[extreme_mask]) ** 2))
        extreme_ratio = extreme_mask.sum() / len(extreme_mask)
    else:
        mae_extreme = 0
        rmse_extreme = 0
        extreme_ratio = 0

    # 正常价格评估
    normal_mask = ~extreme_mask
    if normal_mask.sum() > 0:
        mae_normal = np.mean(np.abs(y_true_flat[normal_mask] - y_pred_flat[normal_mask]))
        rmse_normal = np.sqrt(np.mean((y_true_flat[normal_mask] - y_pred_flat[normal_mask]) ** 2))
    else:
        mae_normal = 0
        rmse_normal = 0

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
        "MAE_extreme": mae_extreme,
        "RMSE_extreme": rmse_extreme,
        "MAE_normal": mae_normal,
        "RMSE_normal": rmse_normal,
        "extreme_ratio": extreme_ratio,
    }
