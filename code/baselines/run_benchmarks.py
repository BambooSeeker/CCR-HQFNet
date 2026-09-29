# -*- coding: utf-8 -*-
"""
基准模型统一运行脚本（滚动预测版 v2 - 修复版）
======================================

核心逻辑：
  1. 深度学习模型（DL）：在测试集上预测 578 个重叠窗口（stride=1）。
  2. 评估对齐：在汇总评估前，从 DL 的 578 个预测中抽取出 7 个起始时间为 00:30 的不重叠窗口。
  3. 统一评估：所有模型（DL, ML, Naive）最终都以 (7, 48) 的 shape 进行 MAE/MAPE 等指标计算。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_pipeline import (
    load_raw_dataframe,
    infer_feature_cols,
    generate_rolling_bundles,
    build_single_bundle_by_endtime,
    infer_halfhour_slot,
)
from metrics import compute_metrics

# ============================================================
# 模型注册表
# ============================================================
ALL_MODELS = ["SeasonalNaive", "LEAR", "LightGBM", "LSTM", "CNN_LSTM", "CNN_Transformer", "DLinear", "PatchTST", "Transformer", "TFT", "TimeXer"]

# ============================================================
# 单模型调度函数
# ============================================================
def run_model(model_name: str, bundle: Dict[str, Any], device: str = "auto", warm_start_state: Optional[Dict] = None) -> np.ndarray:
    if model_name == "SeasonalNaive":
        from seasonal_naive import run_seasonal_naive
        return run_seasonal_naive(bundle)
    elif model_name == "LEAR":
        from lear import run_lear
        return run_lear(bundle)
    elif model_name == "LightGBM":
        from lightgbm_model import run_lightgbm
        return run_lightgbm(bundle)
    elif model_name == "LSTM":
        from lstm_model import run_lstm
        return run_lstm(bundle, device=device)
    elif model_name == "CNN_LSTM":
        from cnn_lstm import run_cnn_lstm
        return run_cnn_lstm(bundle, device=device)
    elif model_name == "CNN_Transformer":
        from cnn_transformer import run_cnn_transformer
        return run_cnn_transformer(bundle, device=device)
    else:
        # 深度学习模型 (DLinear, PatchTST, Transformer, TFT, TimeXer)
        from dl_trainer import DLConfigs, train_dl_model
        SEQ_LEN = bundle["config"]["SEQ_LEN"]
        feature_names = bundle["config"].get("feature_cols", [])
        n_features = bundle["train"]["X_z"].shape[-1]

        model_cfgs = {
            "DLinear":     dict(d_model=128, n_heads=8, e_layers=2, d_layers=1, d_ff=256, dropout=0.1,  patch_len=16, batch_size=64, lr=1e-3, epochs=200, patience=15, criterion_kind="mse"),
            "PatchTST":    dict(d_model=96,  n_heads=4, e_layers=2, d_layers=1, d_ff=192, dropout=0.15, patch_len=4,  batch_size=32, lr=8e-4, epochs=160, patience=20, criterion_kind="weighted_huber"),
            "Transformer": dict(d_model=128, n_heads=8, e_layers=2, d_layers=1, d_ff=256, dropout=0.1,  patch_len=4,  batch_size=32, lr=8e-4, epochs=160, patience=20, criterion_kind="weighted_huber"),
            "TFT":         dict(d_model=96,  n_heads=4, e_layers=2, d_layers=1, d_ff=192, dropout=0.15, patch_len=4,  batch_size=32, lr=8e-4, epochs=160, patience=20, criterion_kind="weighted_huber"),
            "TimeXer":     dict(d_model=96,  n_heads=4, e_layers=2, d_layers=1, d_ff=192, dropout=0.15, patch_len=4,  batch_size=32, lr=8e-4, epochs=160, patience=20, criterion_kind="weighted_huber"),
        }
        hp = model_cfgs[model_name]

        cfg_dict = dict(
            task_name="short_term_forecast", seq_len=SEQ_LEN, pred_len=SEQ_LEN,
            enc_in=n_features, dec_in=n_features, c_out=1, d_model=hp["d_model"], n_heads=hp["n_heads"],
            e_layers=hp["e_layers"], d_layers=hp["d_layers"], d_ff=hp["d_ff"], factor=3, dropout=hp["dropout"], activation="gelu",
            embed="timeF", freq="t", moving_avg=25, features="MS", use_norm=True,
            patch_len=hp["patch_len"], label_len=0, feature_names=feature_names,
        )
        configs = DLConfigs(**cfg_dict)

        if model_name == "DLinear":
            from dlinear import DLinearModel
            model = DLinearModel(configs)
        elif model_name == "PatchTST":
            from patchtst import PatchTSTModel
            model = PatchTSTModel(configs, patch_len=hp["patch_len"], stride=max(2, hp["patch_len"] // 2))
        elif model_name == "Transformer":
            from transformer import TransformerModel
            model = TransformerModel(configs)
        elif model_name == "TFT":
            from tft import TFTModel
            model = TFTModel(configs)
        elif model_name == "TimeXer":
            from timexer import TimeXerModel
            model = TimeXerModel(configs)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return train_dl_model(
            model=model, bundle=bundle, model_name=model_name,
            lr=hp["lr"], epochs=hp["epochs"], patience=hp["patience"], batch_size=hp["batch_size"],
            device=device, warm_start_state=warm_start_state, criterion_kind=hp["criterion_kind"],
            needs_dec=False,
        )

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/dataset.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--end-time", type=str, default=None)
    args = parser.parse_args()

    models_to_run = list(ALL_MODELS) if args.models.lower() == "all" else [m.strip() for m in args.models.split(",")]
    df = load_raw_dataframe(args.csv)
    feat_cols = infer_feature_cols(df)

    all_y_true_final: List[np.ndarray] = [] # 统一存储 (7, 48) 的真实值
    all_y_pred_final: Dict[str, List[np.ndarray]] = {m: [] for m in models_to_run}
    all_timestamps_final: List[List] = []   # 每个对齐窗口的 48 个时间戳（展平）
    all_codes_final: List[List] = []        # 对应的 code（展平）

    bundle_iter = generate_rolling_bundles(df, feat_cols, args.train_days, args.val_days, args.test_days, args.step_days)
    if args.end_time:
        bundle_iter = iter([build_single_bundle_by_endtime(df, feat_cols, args.end_time, args.train_days, args.val_days, args.test_days)])

    total_steps = 0
    for bundle in bundle_iter:
        # ── 关键对齐：从测试集中提取 00:30 起点的窗口 ──
        test_meta = bundle["test"]["meta"] # [code, start_time, end_time, start_idx]
        test_y_raw = bundle["test"]["y_raw"]

        # 获取所有唯一的 code
        unique_codes = np.unique(test_meta[:, 0])
        n_codes = len(unique_codes)
        expected_aligned = n_codes * args.test_days

        aligned_indices = []
        for i in range(len(test_meta)):
            t_s = pd.Timestamp(test_meta[i, 1])
            if infer_halfhour_slot(t_s) == 0:
                aligned_indices.append(i)

        if len(aligned_indices) != expected_aligned:
            print(f"  [WARNING] Step {bundle['roll_step']}: 预期 {expected_aligned} 个对齐天 ({n_codes} codes * {args.test_days} days)，实际找到 {len(aligned_indices)} 个。")

        y_true_aligned = test_y_raw[aligned_indices]
        all_y_true_final.append(y_true_aligned)

        # 收集每个对齐窗口的时间戳和 code（展平为 n_aligned*48 行）
        step_timestamps, step_codes = [], []
        for idx in aligned_indices:
            code = test_meta[idx, 0]
            t_start = pd.Timestamp(test_meta[idx, 1])
            for j in range(48):
                step_timestamps.append(t_start + pd.Timedelta(minutes=30 * j))
                step_codes.append(code)
        all_timestamps_final.extend(step_timestamps)
        all_codes_final.extend(step_codes)

        total_steps += 1

        for model_name in models_to_run:
            try:
                y_pred_step = run_model(model_name, bundle, device=args.device)

                # 如果是 DL 模型，输出是 (578, 48)，需要抽样对齐到 (7, 48)
                if y_pred_step.shape[0] == test_y_raw.shape[0]:
                    y_pred_aligned = y_pred_step[aligned_indices]
                else:
                    # ML 模型输出已经是 (7, 48)
                    y_pred_aligned = y_pred_step

                all_y_pred_final[model_name].append(y_pred_aligned)
                mae = np.mean(np.abs(y_true_aligned - y_pred_aligned))
                print(f"  [{model_name:12s}] Step {bundle['roll_step']:03d} | MAE={mae:.2f}")
            except Exception as e:
                print(f"  [{model_name:12s}] Step {bundle['roll_step']:03d} FAILED: {e}")
                traceback.print_exc()
                all_y_pred_final[model_name].append(np.full_like(y_true_aligned, np.nan))

    # ── 汇总评估 ──
    if total_steps == 0: return
    y_true_all = np.concatenate(all_y_true_final, axis=0)
    results = []
    for model_name in models_to_run:
        preds = all_y_pred_final[model_name]
        if not preds: continue
        y_pred_all = np.concatenate(preds, axis=0)
        if np.all(np.isnan(y_pred_all)): continue

        metrics = compute_metrics(y_true_all, y_pred_all, model_name=model_name)
        results.append(metrics)
        print(f"  {model_name:12s} | MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  sMAPE={metrics['sMAPE']:.2f}%")

    # 保存结果
    out_dir = ROOT / "results"; out_dir.mkdir(parents=True, exist_ok=True)

    # ── 导出详细预测结果 (真实值 vs 预测值) ──
    y_true_flat = y_true_all.flatten()
    detailed_df = pd.DataFrame({
        "time": all_timestamps_final,
        "code": all_codes_final,
        "True": y_true_flat,
    })
    for model_name in models_to_run:
        preds = all_y_pred_final[model_name]
        if preds:
            y_pred_all = np.concatenate(preds, axis=0)
            detailed_df[model_name] = y_pred_all.flatten()

    detailed_csv = out_dir / "detailed_predictions.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"\n[保存] 详细预测数据 → {detailed_csv}")

    if results:
        results_df = pd.DataFrame(results).sort_values("MAE")
        results_df.to_csv(out_dir / "rolling_benchmark_results.csv", index=False)
        print(f"[保存] 汇总指标表格 → {out_dir / 'rolling_benchmark_results.csv'}")

if __name__ == "__main__":
    main()
