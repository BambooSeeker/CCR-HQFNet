# -*- coding: utf-8 -*-
"""
LightGBM Point-wise Forecasting
================================
将每个时间点作为一个独立样本，训练单个 LightGBM 回归器。

设计思路（与旧版的区别）：
  旧版：每天一个样本（stride=48），特征 = 当天全部 48 步展平（48*F 维）
        → 90 个样本，3840 维特征，严重欠采样，极易过拟合
  新版：每个时间点一个样本，特征 = 当前时间点的 F 维外生特征 + 3 维时间特征
        → 4320 个训练样本（90天×48步），~79 维特征，样本/特征比 ≈ 55，合理

特征构成（共 F+3 维）：
  - 外生特征 (F)：数据集中的市场结构特征和滞后特征（已包含 lag_48, lag_168 等）
  - 时间特征 (3)：sin(slot), cos(slot), slot_norm

预测流程：
  1. 用训练集（~4320 样本）训练单个 LightGBM 模型
  2. 用验证集做 Early Stopping
  3. 对测试集（336 个时间点）逐点预测
  4. 将预测结果 reshape 为 (7, 48) 以匹配 ground truth 格式

数据来源：bundle["ml_train_pt"] / bundle["ml_val_pt"] / bundle["ml_test_pt"]
  - 由 data_pipeline._build_ml_samples_pointwise() 构建
  - 每个 split 包含 X_z (N, F), tfeat (N, 3), y_z (N,), y_raw (N,)

参考：
  - Ke et al. (2017): "LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree" (NeurIPS)
  - Lago et al. (2021): "Forecasting day-ahead electricity prices" (Applied Energy)
  - Tschora et al. (2022): "Electricity price forecasting on the day-ahead
    market using machine learning" (Applied Energy)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

import lightgbm as lgb


def run_lightgbm(bundle: Dict[str, Any]) -> np.ndarray:
    """
    Parameters
    ----------
    bundle : 统一数据字典（需包含 ml_train_pt / ml_val_pt / ml_test_pt）

    Returns
    -------
    y_pred_raw : (n_test_days, 48) 原始尺度预测值
                 n_test_days = ml_test_pt["n_samples"] // 48（严格等于测试天数）
    """
    SEQ_LEN = bundle["config"]["SEQ_LEN"]   # 48

    # ── 使用逐时间点 ML 样本 ──
    ml_tr  = bundle["ml_train_pt"]
    ml_val = bundle["ml_val_pt"]
    ml_te  = bundle["ml_test_pt"]

    if ml_tr["n_samples"] == 0:
        raise RuntimeError("[LightGBM] ml_train_pt 样本数为 0，请检查时间边界或数据完整性。")
    if ml_te["n_samples"] == 0:
        raise RuntimeError("[LightGBM] ml_test_pt 样本数为 0，请检查时间边界或数据完整性。")

    X_train  = ml_tr["X_z"]    # (N_tr, F)
    t_train  = ml_tr["tfeat"]  # (N_tr, 3)
    y_train  = ml_tr["y_z"]    # (N_tr,)

    X_val    = ml_val["X_z"]   # (N_val, F)
    t_val    = ml_val["tfeat"] # (N_val, 3)
    y_val    = ml_val["y_z"]   # (N_val,)

    X_test   = ml_te["X_z"]    # (N_te, F)
    t_test   = ml_te["tfeat"]  # (N_te, 3)

    y_mu = float(bundle["scalers"]["y_mu"])
    y_sd = float(bundle["scalers"]["y_sd"])

    N_tr  = X_train.shape[0]
    N_val = X_val.shape[0]
    N_te  = X_test.shape[0]

    # ── 拼接特征：[外生特征 (F) | 时间特征 (3)] ──
    Xh_tr  = np.concatenate([X_train, t_train], axis=1)   # (N_tr,  F+3)
    Xh_val = np.concatenate([X_val,   t_val],   axis=1)   # (N_val, F+3)
    Xh_te  = np.concatenate([X_test,  t_test],  axis=1)   # (N_te,  F+3)

    params = {
        "objective":        "regression",
        "metric":           "mae",
        "boosting_type":    "gbdt",
        "num_leaves":       31,    # 降低复杂度，防止过拟合
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_data_in_leaf": 50,    # 4320 样本时适当增大，增强稳健性
        "verbose":          -1,
        "n_jobs":           -1,
        "seed":             42,
        "deterministic":    True,
    }

    print(
        f"  [LightGBM] Training single point-wise model "
        f"(N_train={N_tr} pts = {N_tr // SEQ_LEN} days x {SEQ_LEN} steps, "
        f"N_val={N_val} pts, N_test={N_te} pts, "
        f"feat_dim={Xh_tr.shape[1]}) ..."
    )

    dtrain = lgb.Dataset(Xh_tr,  label=y_train)
    dval   = lgb.Dataset(Xh_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred_z = model.predict(Xh_te)   # (N_te,)

    # 反标准化
    y_pred_raw = y_pred_z * y_sd + y_mu   # (N_te,)

    # reshape 为 (n_test_days, 48) 以匹配 ground truth 格式
    # 注意：ml_test_pt 的样本按 [code, time] 排序，每 48 个连续点为一天
    n_test_days = N_te // SEQ_LEN
    if N_te % SEQ_LEN != 0:
        # 若不能整除，截断到整天（理论上不应发生）
        print(
            f"  [LightGBM] WARNING: N_te={N_te} 不能被 SEQ_LEN={SEQ_LEN} 整除，"
            f"截断到 {n_test_days} 天。"
        )
        y_pred_raw = y_pred_raw[:n_test_days * SEQ_LEN]

    y_pred_raw = y_pred_raw.reshape(n_test_days, SEQ_LEN).astype(np.float32)

    return y_pred_raw
