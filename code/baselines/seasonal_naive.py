# -*- coding: utf-8 -*-
"""
Seasonal Naive-48 Baseline
==========================
预测值 = 训练集中最后一个完整周期（lag=48*7=336 步）的历史值。
即：y_hat(t) = y(t - 336)

这是电价预测文献中最常见的 naive baseline（Lago et al., 2021; Uniejewski et al., 2019）。
无需训练，仅需从训练集末尾获取历史参考。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any


def run_seasonal_naive(bundle: Dict[str, Any]) -> np.ndarray:
    """
    Parameters
    ----------
    bundle : 统一数据字典（由 data_pipeline 生成）

    Returns
    -------
    y_pred_raw : (n_test, 48) 原始尺度预测值
    """
    import pandas as pd

    ref_df = bundle["reference_df"]
    cfg = bundle["config"]
    test_meta = bundle["test"]["meta"]
    y_test_raw = bundle["test"]["y_raw"]

    from data_pipeline import TIME_COL, CODE_COL, Y_COL
    SEQ_LEN = cfg.get("SEQ_LEN", 48)

    # 周期 lag = 7 天 * 48 步/天 = 336
    LAG = 7 * SEQ_LEN  # 336

    n_test = y_test_raw.shape[0]
    y_pred = np.zeros_like(y_test_raw)

    for i in range(n_test):
        code = test_meta[i][0]
        t_start = pd.Timestamp(test_meta[i][1])

        # 获取该 code 的历史序列
        code_df = ref_df[ref_df[CODE_COL] == code].sort_values(TIME_COL).reset_index(drop=True)
        code_times = pd.to_datetime(code_df[TIME_COL])
        code_prices = code_df[Y_COL].values

        # 找到 t_start 在历史中的位置
        idx = code_times.searchsorted(t_start)

        # 回溯 LAG 步
        lag_start = idx - LAG
        lag_end = lag_start + SEQ_LEN

        if lag_start >= 0 and lag_end <= len(code_prices):
            y_pred[i] = code_prices[lag_start:lag_end]
        else:
            # fallback: 使用训练集均值
            y_pred[i] = bundle["train"]["y_raw"].mean()

    return y_pred
