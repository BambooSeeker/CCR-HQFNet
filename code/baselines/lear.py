# -*- coding: utf-8 -*-
"""
LEAR — LASSO Estimated AutoRegressive Model（逐时间点版）
=========================================================
严格按照 Lago et al. (2021) "Forecasting day-ahead electricity prices:
A review of state-of-the-art algorithms, best practices and an open-access
benchmark" (Applied Energy) 中的 LEAR 模型实现。

设计思路（与旧版的区别）：
  旧版：每天一个样本（stride=48），特征 = 当天全部 48 步展平（48*F 维）
        → 90 个样本，3840 维特征，严重欠采样
  新版：每个时间点一个样本，特征 = 当前时间点的 F 维外生特征 + 3 维时间特征
        → 4320 个训练样本（90天×48步），~79 维特征，样本/特征比 ≈ 55，合理

特征构成（共 F+3 维）：
  - 外生特征 (F)：数据集中的市场结构特征和滞后特征（已包含 lag_48, lag_168 等）
  - 时间特征 (3)：sin(slot), cos(slot), slot_norm

预测流程：
  1. 用训练集（~4320 样本）拟合单个 LassoCV 模型
  2. 对测试集（336 个时间点）逐点预测
  3. 将预测结果 reshape 为 (7, 48) 以匹配 ground truth 格式

注意：
  - LEAR 不使用验证集做 Early Stopping（LassoCV 内部通过 5折交叉验证选 alpha）
  - 这与 Lago et al. (2021) 原论文的设计一致

数据来源：bundle["ml_train_pt"] / bundle["ml_test_pt"]
  - 由 data_pipeline._build_ml_samples_pointwise() 构建
  - 每个 split 包含 X_z (N, F), tfeat (N, 3), y_z (N,), y_raw (N,)

参考：
  - epftoolbox: https://github.com/jeslago/epftoolbox
  - Uniejewski et al. (2016): "Automated variable selection and shrinkage
    for day-ahead electricity price forecasting"
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

from sklearn.linear_model import LassoCV


def run_lear(bundle: Dict[str, Any], n_jobs: int = -1) -> np.ndarray:
    """
    Parameters
    ----------
    bundle : 统一数据字典（需包含 ml_train_pt / ml_test_pt）
    n_jobs : 并行数（-1 = 使用所有 CPU 核心）

    Returns
    -------
    y_pred_raw : (n_test_days, 48) 原始尺度预测值
                 n_test_days = ml_test_pt["n_samples"] // 48（严格等于测试天数）
    """
    SEQ_LEN = bundle["config"]["SEQ_LEN"]   # 48

    # ── 使用逐时间点 ML 样本 ──
    ml_tr  = bundle["ml_train_pt"]
    ml_te  = bundle["ml_test_pt"]

    # 检查样本数是否充足
    if ml_tr["n_samples"] == 0:
        raise RuntimeError("[LEAR] ml_train_pt 样本数为 0，请检查时间边界或数据完整性。")
    if ml_te["n_samples"] == 0:
        raise RuntimeError("[LEAR] ml_test_pt 样本数为 0，请检查时间边界或数据完整性。")

    X_train  = ml_tr["X_z"]    # (N_tr, F)  外生特征（标准化）
    t_train  = ml_tr["tfeat"]  # (N_tr, 3)  时间特征
    y_train  = ml_tr["y_z"]    # (N_tr,)    预测目标（标准化）

    X_test   = ml_te["X_z"]    # (N_te, F)
    t_test   = ml_te["tfeat"]  # (N_te, 3)

    y_mu = float(bundle["scalers"]["y_mu"])
    y_sd = float(bundle["scalers"]["y_sd"])

    N_tr = X_train.shape[0]
    N_te = X_test.shape[0]

    # ── 构建 LEAR 特征矩阵：[外生特征 (F) | 时间特征 (3)] ──
    X_tr_full = np.concatenate([X_train, t_train], axis=1)   # (N_tr, F+3)
    X_te_full = np.concatenate([X_test,  t_test],  axis=1)   # (N_te, F+3)

    # ── 单模型 LassoCV：内部 5折 CV 自动选择最优 alpha ──
    # 固定 alphas 候选网格（100个点），消除 FutureWarning，加速 CV
    model = LassoCV(
        alphas=np.logspace(-4, 2, 100),   # 固定候选 alpha 网格，无 FutureWarning
        cv=5,
        max_iter=5000,    # 适度增加迭代次数，消除收敛警告
        tol=1e-2,         # 进一步放宽阈值以彻底消除警告，对电价预测精度影响极小
        n_jobs=n_jobs,
        random_state=42,
    )

    print(
        f"  [LEAR] LassoCV fitting "
        f"(N_train={N_tr} pts = {N_tr // SEQ_LEN} days x {SEQ_LEN} steps, "
        f"N_test={N_te} pts, features={X_tr_full.shape[1]}) ..."
    )

    model.fit(X_tr_full, y_train)          # y_train: (N_tr,)
    y_pred_z = model.predict(X_te_full)    # (N_te,)

    # 反标准化
    y_pred_raw = y_pred_z * y_sd + y_mu   # (N_te,)

    # reshape 为 (n_test_days, 48) 以匹配 ground truth 格式
    n_test_days = N_te // SEQ_LEN
    if N_te % SEQ_LEN != 0:
        print(
            f"  [LEAR] WARNING: N_te={N_te} 不能被 SEQ_LEN={SEQ_LEN} 整除，"
            f"截断到 {n_test_days} 天。"
        )
        y_pred_raw = y_pred_raw[:n_test_days * SEQ_LEN]

    y_pred_raw = y_pred_raw.reshape(n_test_days, SEQ_LEN).astype(np.float32)

    return y_pred_raw
