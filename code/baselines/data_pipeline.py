# -*- coding: utf-8 -*-
"""
统一基准模型数据管线（滚动预测版 v2 - 修复版）
=====================================

核心逻辑：
  1. 深度学习模型（DL）：训练/验证集使用 stride=1 的滑动窗口（增加样本量）。
  2. 机器学习模型（ML）：使用逐时间点样本（90*48 样本，76 维特征）。
  3. 评估阶段：所有模型在测试集上必须严格对齐到 00:30 ~ 次日 00:00 的 7 个完整周期。
  4. 强制对齐：generate_rolling_bundles 确保 test_start 是 00:30，test_end 是 00:00。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import random
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

def set_seed(seed: int = 42):
    """固定全局随机种子，确保论文实验的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# 初始化时自动设置种子
set_seed(42)

# ============================================================
# 0) Global config
# ============================================================
SEQ_LEN = 48          # 每个样本的时间步数（30min × 48 = 1天）
STRIDE = 1            # 滑窗步长（样本级别，DL 模型使用）
STEP_MINUTES = 30     # 数据分辨率

TRAIN_DAYS = 90       # 训练窗口长度（天）
ROLL_VAL_DAYS = 7     # 验证窗口长度（天）
ROLL_TEST_DAYS = 7    # 每次滚动的测试窗口长度（天）
ROLL_STEP_DAYS = 7    # 每次滚动向前推进的天数

TIME_COL = "time"
CODE_COL = "code"
Y_COL = "price_real"


# ============================================================
# 1) Basic utilities
# ============================================================
def is_contiguous(time_arr: np.ndarray, step_minutes: int = 30) -> bool:
    """检查窗口时间是否严格连续（30min 间隔）。"""
    if len(time_arr) <= 1:
        return True
    t = pd.to_datetime(time_arr).values.astype("datetime64[ns]").astype("int64")
    step = int(step_minutes * 60 * 1e9)
    return bool(np.all(np.diff(t) == step))


def infer_halfhour_slot(ts: pd.Timestamp) -> int:
    """半小时时间槽 0~47 (00:30 -> 0, 01:00 -> 1, ..., 00:00 -> 47)。"""
    # 注意：00:30 是 slot 0，00:00 是 slot 47
    h, m = ts.hour, ts.minute
    if h == 0 and m == 0: return 47
    slot = int(h * 2 + (m // 30)) - 1
    return slot if slot >= 0 else 47


def make_time_feats(tg: pd.Series) -> np.ndarray:
    """时间特征：sin(slot), cos(slot), slot_norm"""
    slots = np.array([infer_halfhour_slot(pd.Timestamp(x)) for x in tg], dtype=np.float32)
    ang = 2.0 * np.pi * (slots / 48.0)
    sinv = np.sin(ang).astype(np.float32)
    cosv = np.cos(ang).astype(np.float32)
    slot_norm = (slots / 47.0).astype(np.float32)
    return np.stack([sinv, cosv, slot_norm], axis=-1)


def fit_scaler_3d(A: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mu = A.reshape(-1, A.shape[-1]).mean(axis=0)
    sd = A.reshape(-1, A.shape[-1]).std(axis=0)
    sd = np.maximum(sd, eps)
    return mu.astype(np.float32), sd.astype(np.float32)


def zscore_3d(A: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((A - mu) / sd).astype(np.float32)


def fit_y_scaler(ytr: np.ndarray, eps: float = 1e-6) -> Tuple[np.float32, np.float32]:
    mu = ytr.reshape(-1).mean()
    sd = ytr.reshape(-1).std()
    sd = max(sd, eps)
    return np.float32(mu), np.float32(sd)


def zscore_y(y: np.ndarray, mu: np.float32, sd: np.float32) -> np.ndarray:
    return ((y - mu) / sd).astype(np.float32)


# ============================================================
# 2) Feature selection / raw dataframe loading
# ============================================================
def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {TIME_COL, CODE_COL, Y_COL}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feat_cols


def load_raw_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.drop_duplicates().dropna(subset=[TIME_COL, CODE_COL, Y_COL]).copy()
    df[CODE_COL] = df[CODE_COL].astype(str)
    df = df.sort_values([CODE_COL, TIME_COL]).reset_index(drop=True)
    return df


# ============================================================
# 3) Window builder (DL 模型使用 stride=1)
# ============================================================
def build_windows_for_split(
    df: pd.DataFrame, feat_cols: List[str],
    train_start, train_end, val_start, val_end, test_start, test_end,
    seq_len=48, stride=1, step_minutes=30
) -> Tuple:
    packs = {k: {"X": [], "t": [], "y": [], "meta": []} for k in ["train", "val", "test"]}
    for code, g in df.groupby(CODE_COL, sort=True):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        Xg, yg, tg = g[feat_cols].to_numpy(dtype=np.float32), g[Y_COL].to_numpy(dtype=np.float32), pd.to_datetime(g[TIME_COL])
        T = len(g)
        if T < seq_len: continue
        train_mask = (tg >= train_start) & (tg <= train_end)
        val_mask   = (tg >= val_start)   & (tg <= val_end)
        test_mask  = (tg >= test_start)  & (tg <= test_end)
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            t_slice = tg.iloc[start:end]
            if not is_contiguous(t_slice.to_numpy(), step_minutes=step_minutes): continue
            t_s, t_e = pd.Timestamp(tg.iloc[start]), pd.Timestamp(tg.iloc[end - 1])
            if train_mask.iloc[start] and train_mask.iloc[end - 1]: key = "train"
            elif val_mask.iloc[start] and val_mask.iloc[end - 1]: key = "val"
            elif test_mask.iloc[start] and test_mask.iloc[end - 1]: key = "test"
            else: continue
            packs[key]["X"].append(Xg[start:end])
            packs[key]["t"].append(make_time_feats(t_slice))
            packs[key]["y"].append(yg[start:end])
            packs[key]["meta"].append([str(code), str(t_s), str(t_e), int(start)])
    def stack(pack, name):
        if not pack["X"]: raise RuntimeError(f"Split '{name}' is empty.")
        return np.stack(pack["X"]), np.stack(pack["t"]), np.stack(pack["y"]), np.array(pack["meta"], dtype=object)
    return stack(packs["train"], "train"), stack(packs["val"], "val"), stack(packs["test"], "test")


# ============================================================
# 4) ML Sample Builders (逐时间点 & 逐天)
# ============================================================
def _build_ml_samples(df, feat_cols, start, end, X_mu, X_sd, y_mu, y_sd) -> Dict:
    """构建逐天样本 (stride=48)，严格对齐 00:30-00:00。"""
    packs = {"X": [], "t": [], "y": [], "ar": [], "meta": []}
    for code, g in df.groupby(CODE_COL):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        times, Xg, yg = pd.to_datetime(g[TIME_COL]), g[feat_cols].to_numpy(dtype=np.float32), g[Y_COL].to_numpy(dtype=np.float32)
        for i in range(len(g) - 48 + 1):
            t_s = pd.Timestamp(times.iloc[i])
            if infer_halfhour_slot(t_s) != 0: continue # 必须是 00:30
            t_e = pd.Timestamp(times.iloc[i+47])
            if not (t_s >= start and t_e <= end): continue
            if not is_contiguous(times.iloc[i:i+48].to_numpy()): continue
            # 自回归项 (lag-48)
            if i < 48 or not is_contiguous(times.iloc[i-48:i].to_numpy()): continue
            packs["X"].append(Xg[i:i+48]); packs["t"].append(make_time_feats(times.iloc[i:i+48]))
            packs["y"].append(yg[i:i+48]); packs["ar"].append(yg[i-48:i])
            packs["meta"].append([str(code), str(t_s), str(t_e)])
    if not packs["X"]: return {"n_samples": 0}
    X_raw = np.stack(packs["X"])
    return {
        "X_z": zscore_3d(X_raw, X_mu, X_sd), "tfeat": np.stack(packs["t"]),
        "y_raw": np.stack(packs["y"]), "y_z": zscore_y(np.stack(packs["y"]), y_mu, y_sd),
        "ar_z": zscore_y(np.stack(packs["ar"]), y_mu, y_sd), "n_samples": len(X_raw)
    }

def _build_ml_samples_pointwise(df, feat_cols, start, end, X_mu, X_sd, y_mu, y_sd) -> Dict:
    """构建逐时间点样本，确保每个点属于一个完整的 00:30-00:00 周期。"""
    X_list, t_list, y_list, y_raw_list = [], [], [], []
    for code, g in df.groupby(CODE_COL):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        times, Xg, yg = pd.to_datetime(g[TIME_COL]), g[feat_cols].to_numpy(dtype=np.float32), g[Y_COL].to_numpy(dtype=np.float32)
        for i in range(len(g)):
            t = pd.Timestamp(times.iloc[i])
            if not (t >= start and t <= end): continue
            slot = infer_halfhour_slot(t)
            day_start = i - slot
            if day_start < 0 or day_start + 48 > len(g): continue
            if not is_contiguous(times.iloc[day_start:day_start+48].to_numpy()): continue
            X_list.append(Xg[i]); t_list.append(make_time_feats(times.iloc[i:i+1])[0])
            y_raw_list.append(yg[i]); y_list.append((yg[i] - y_mu) / y_sd)
    if not X_list: return {"n_samples": 0}
    return {
        "X_z": (np.stack(X_list) - X_mu) / X_sd, "tfeat": np.stack(t_list),
        "y_z": np.array(y_list, dtype=np.float32), "y_raw": np.array(y_raw_list, dtype=np.float32),
        "n_samples": len(X_list)
    }


# ============================================================
# 5) Rolling Generator
# ============================================================
def generate_rolling_bundles(df, feat_cols, train_days=90, val_days=7, test_days=7, step_days=7):
    all_times = pd.to_datetime(df[TIME_COL])
    # 强制对齐到第一个 00:30
    curr_start = all_times.min()
    while infer_halfhour_slot(curr_start) != 0: curr_start += pd.Timedelta(minutes=30)

    global_end = all_times.max()
    roll_step = 0
    while True:
        train_start = curr_start
        train_end   = train_start + pd.Timedelta(days=train_days) - pd.Timedelta(minutes=30)
        val_start   = train_end   + pd.Timedelta(minutes=30)
        val_end     = val_start   + pd.Timedelta(days=val_days)   - pd.Timedelta(minutes=30)
        test_start  = val_end     + pd.Timedelta(minutes=30)
        test_end    = test_start  + pd.Timedelta(days=test_days)  - pd.Timedelta(minutes=30)

        if test_end > global_end: break

        print(f"\n[滚动步 {roll_step}] Test: {test_start} ~ {test_end}")
        try:
            train_data, val_data, test_data = build_windows_for_split(
                df, feat_cols, train_start, train_end, val_start, val_end, test_start, test_end
            )
            X_mu, X_sd = fit_scaler_3d(train_data[0])
            y_mu, y_sd = fit_y_scaler(train_data[2])

            bundle = {
                "train": {"X_z": zscore_3d(train_data[0], X_mu, X_sd), "tfeat": train_data[1], "y_z": zscore_y(train_data[2], y_mu, y_sd), "y_raw": train_data[2], "n_samples": len(train_data[0]), "meta": train_data[3]},
                "val":   {"X_z": zscore_3d(val_data[0], X_mu, X_sd),   "tfeat": val_data[1],   "y_z": zscore_y(val_data[2], y_mu, y_sd),   "y_raw": val_data[2],   "n_samples": len(val_data[0]), "meta": val_data[3]},
                "test":  {"X_z": zscore_3d(test_data[0], X_mu, X_sd),  "tfeat": test_data[1],  "y_z": zscore_y(test_data[2], y_mu, y_sd),  "y_raw": test_data[2],  "n_samples": len(test_data[0]), "meta": test_data[3]},
                "scalers": {"X_mu": X_mu, "X_sd": X_sd, "y_mu": y_mu, "y_sd": y_sd},
                "config": {"SEQ_LEN": 48, "feature_cols": feat_cols}, "roll_step": roll_step,
                "reference_df": df.copy()
            }
            bundle["ml_train"] = _build_ml_samples(df, feat_cols, train_start, train_end, X_mu, X_sd, y_mu, y_sd)
            bundle["ml_test"]  = _build_ml_samples(df, feat_cols, test_start, test_end, X_mu, X_sd, y_mu, y_sd)
            bundle["ml_train_pt"] = _build_ml_samples_pointwise(df, feat_cols, train_start, train_end, X_mu, X_sd, y_mu, y_sd)
            bundle["ml_test_pt"]  = _build_ml_samples_pointwise(df, feat_cols, test_start, test_end, X_mu, X_sd, y_mu, y_sd)
            yield bundle
        except Exception as e:
            print(f"  跳过步 {roll_step}: {e}")

        curr_start += pd.Timedelta(days=step_days)
        roll_step += 1

def build_single_bundle_by_endtime(df, feat_cols, end_time, train_days=90, val_days=7, test_days=7):
    # 逻辑同上，仅运行一次
    end_ts = pd.Timestamp(end_time)
    test_end = end_ts - pd.Timedelta(minutes=30)
    while infer_halfhour_slot(test_end) != 47: test_end -= pd.Timedelta(minutes=30)
    test_start = test_end - pd.Timedelta(days=test_days) + pd.Timedelta(minutes=30)
    val_end = test_start - pd.Timedelta(minutes=30)
    val_start = val_end - pd.Timedelta(days=val_days) + pd.Timedelta(minutes=30)
    train_end = val_start - pd.Timedelta(minutes=30)
    train_start = train_end - pd.Timedelta(days=train_days) + pd.Timedelta(minutes=30)

    train_data, val_data, test_data = build_windows_for_split(df, feat_cols, train_start, train_end, val_start, val_end, test_start, test_end)
    X_mu, X_sd = fit_scaler_3d(train_data[0]); y_mu, y_sd = fit_y_scaler(train_data[2])
    bundle = {
        "train": {"X_z": zscore_3d(train_data[0], X_mu, X_sd), "tfeat": train_data[1], "y_z": zscore_y(train_data[2], y_mu, y_sd), "y_raw": train_data[2], "n_samples": len(train_data[0]), "meta": train_data[3]},
        "val":   {"X_z": zscore_3d(val_data[0], X_mu, X_sd),   "tfeat": val_data[1],   "y_z": zscore_y(val_data[2], y_mu, y_sd),   "y_raw": val_data[2],   "n_samples": len(val_data[0]), "meta": val_data[3]},
        "test":  {"X_z": zscore_3d(test_data[0], X_mu, X_sd),  "tfeat": test_data[1],  "y_z": zscore_y(test_data[2], y_mu, y_sd),  "y_raw": test_data[2],  "n_samples": len(test_data[0]), "meta": test_data[3]},
        "scalers": {"X_mu": X_mu, "X_sd": X_sd, "y_mu": y_mu, "y_sd": y_sd},
        "config": {"SEQ_LEN": 48, "feature_cols": feat_cols}, "roll_step": 0,
        "reference_df": df.copy()
    }
    bundle["ml_train"] = _build_ml_samples(df, feat_cols, train_start, train_end, X_mu, X_sd, y_mu, y_sd)
    bundle["ml_test"]  = _build_ml_samples(df, feat_cols, test_start, test_end, X_mu, X_sd, y_mu, y_sd)
    bundle["ml_train_pt"] = _build_ml_samples_pointwise(df, feat_cols, train_start, train_end, X_mu, X_sd, y_mu, y_sd)
    bundle["ml_val_pt"]   = _build_ml_samples_pointwise(df, feat_cols, val_start, val_end, X_mu, X_sd, y_mu, y_sd)
    bundle["ml_test_pt"]  = _build_ml_samples_pointwise(df, feat_cols, test_start, test_end, X_mu, X_sd, y_mu, y_sd)
    return bundle
