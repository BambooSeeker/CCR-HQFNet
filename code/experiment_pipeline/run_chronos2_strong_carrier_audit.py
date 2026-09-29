from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import wilcoxon

from run_physical_soft_route_price_audit import add_features, day_mask, error_metrics


MODES = ("chronos2_zero_shot", "chronos2_residual_blind", "chronos2_residual_full")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def model_params(config: dict) -> dict:
    params = dict(config["lightgbm"])
    params.update(
        objective="regression_l1",
        random_state=int(config["seed"]),
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
    )
    return params


def fit_residual(x_train, y_train, x_val, y_val, config):
    model = LGBMRegressor(**model_params(config))
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return model


def build_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows = []
    target_rows = []
    for delivery_day, daily in frame.groupby("delivery_day", sort=True):
        daily = daily.sort_values("time")
        if len(daily) != 48:
            raise ValueError(f"{delivery_day.date()} has {len(daily)} rows, expected 48")
        item_id = delivery_day.strftime("%Y-%m-%d")
        older = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": daily["time"].to_numpy() - pd.Timedelta(hours=72),
                "target": daily["price_real_lag144"].to_numpy(float),
            }
        )
        newer = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": daily["time"].to_numpy() - pd.Timedelta(hours=48),
                "target": daily["price_real_lag96"].to_numpy(float),
            }
        )
        context = pd.concat([older, newer], ignore_index=True)
        delta = context["timestamp"].diff().dropna()
        if not (delta == pd.Timedelta(minutes=30)).all():
            raise ValueError(f"non-regular reconstructed context for {item_id}")
        context_rows.append(context)
        local_target = daily[["time"]].copy()
        local_target.insert(0, "item_id", item_id)
        target_rows.append(local_target)
    return pd.concat(context_rows, ignore_index=True), pd.concat(target_rows, ignore_index=True)


def chronos_predictions(frame: pd.DataFrame, config: dict, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["time"])
        expected = frame[["time"]].sort_values("time").reset_index(drop=True)
        actual = cached[["time"]].sort_values("time").reset_index(drop=True)
        if expected.equals(actual):
            return cached
        raise ValueError("Chronos cache timestamps do not match locked data")

    context, targets = build_context(frame)
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
    pipeline = Chronos2Pipeline.from_pretrained(
        config["model_id"], device_map=config["device_map"], dtype=dtype
    )
    forecast = pipeline.predict_df(
        context,
        prediction_length=int(config["prediction_length"]),
        quantile_levels=list(config["quantile_levels"]),
        batch_size=int(config["batch_size"]),
        context_length=int(config["context_length"]),
        cross_learning=False,
        freq="30min",
    )
    start = int(config["target_horizon_start"])
    length = int(config["target_horizon_length"])
    selected = (
        forecast.sort_values(["item_id", "timestamp"])
        .groupby("item_id", sort=False)
        .nth(list(range(start, start + length)))
        .reset_index()
    )
    selected = selected.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    targets = targets.sort_values(["item_id", "time"]).reset_index(drop=True)
    if len(selected) != len(targets) or not (selected["item_id"] == targets["item_id"]).all():
        raise ValueError("Chronos forecast rows do not align with target rows")
    result = targets[["time"]].copy()
    result["chronos2_prediction"] = selected[str(config["point_column"])].to_numpy(float)
    result["chronos2_q10"] = selected["0.1"].to_numpy(float)
    result["chronos2_q50"] = selected["0.5"].to_numpy(float)
    result["chronos2_q90"] = selected["0.9"].to_numpy(float)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    return result


def comparison(metrics: pd.DataFrame, daily: pd.DataFrame, candidate: str, baseline: str) -> dict:
    result = {}
    names = (
        "mae",
        "extreme_low_mae",
        "negative_price_mae",
        "negative_sign_recall",
        "negative_congestion_mae",
        "positive_congestion_mae",
    )
    for metric_name in names:
        pivot = metrics.pivot(index="fold", columns="mode", values=metric_name).dropna()
        gain = pivot[baseline] - pivot[candidate]
        if metric_name == "negative_sign_recall":
            gain = -gain
        result[metric_name] = {
            "valid_folds": int(len(gain)),
            "improved_fold_count": int((gain > 0).sum()),
            "mean_improvement": float(gain.mean()) if len(gain) else None,
            "improvement_by_fold": {str(int(k)): float(v) for k, v in gain.items()},
        }
    pivot = daily[daily["mode"].isin([candidate, baseline])].pivot(
        index=["fold", "delivery_day"], columns="mode", values="mae"
    ).dropna()
    delta = pivot[candidate] - pivot[baseline]
    result["daily_mae"] = {
        "n_days": int(len(delta)),
        "candidate_minus_baseline": float(delta.mean()),
        "candidate_better_day_fraction": float((delta < 0).mean()),
        "wilcoxon_two_sided_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))

    frame = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels, usecols=["time", "congestion_state"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    labels["time"] = pd.to_datetime(labels["time"])
    frame = add_features(frame.merge(labels, on="time", how="left"))
    carrier = chronos_predictions(frame, config, args.cache)
    frame = frame.merge(carrier, on="time", how="left", validate="one_to_one")
    if frame["chronos2_prediction"].isna().any():
        raise ValueError("missing Chronos predictions after merge")

    congestion = manifest["congestion_features"]
    energy_columns = [name for name in manifest["features"] if name not in congestion and name != "price_day_ahead"]
    energy_columns += ["slot_sin", "slot_cos", "slot_norm", "price_day_ahead_energy_proxy", "chronos2_prediction"]
    full_columns = energy_columns + congestion
    x_blind = frame[energy_columns].to_numpy(float)
    x_full = frame[full_columns].to_numpy(float)
    actual = frame["price_real"].to_numpy(float)
    base = frame["chronos2_prediction"].to_numpy(float)
    residual = actual - base
    state = frame["congestion_state"].fillna(-1).to_numpy(int)

    fold_rows = []
    prediction_rows = []
    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (day_mask(frame, fold[name]) for name in ("train", "val", "test"))
        blind = fit_residual(x_blind[train], residual[train], x_blind[val], residual[val], config)
        full = fit_residual(x_full[train], residual[train], x_full[val], residual[val], config)
        predictions = {
            "chronos2_zero_shot": base[test],
            "chronos2_residual_blind": base[test] + blind.predict(x_blind[test]),
            "chronos2_residual_full": base[test] + full.predict(x_full[test]),
        }
        low_threshold = float(np.quantile(actual[train], 0.10))
        low_mask = actual[test] <= low_threshold
        for mode, predicted in predictions.items():
            fold_rows.append(
                {
                    "fold": fold_index,
                    "mode": mode,
                    "low_threshold_train_q10": low_threshold,
                    **error_metrics(actual[test], predicted, low_mask, state[test]),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_index,
                        "delivery_day": frame.loc[test, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy(),
                        "time": frame.loc[test, "time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                        "mode": mode,
                        "actual": actual[test],
                        "predicted": predicted,
                        "absolute_error": np.abs(predicted - actual[test]),
                        "low_price": low_mask,
                        "congestion_state": state[test],
                    }
                )
            )

    metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = predictions.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(mae=("absolute_error", "mean"))
    primary = comparison(metrics, daily, "chronos2_residual_full", "chronos2_residual_blind")
    full_vs_zero = comparison(metrics, daily, "chronos2_residual_full", "chronos2_zero_shot")
    main_ok = primary["mae"]["improved_fold_count"] >= 3 and primary["mae"]["mean_improvement"] > 0
    tail_ok = primary["extreme_low_mae"]["improved_fold_count"] >= 2 and primary["extreme_low_mae"]["mean_improvement"] > 0
    report = {
        "status": "complete",
        "strong_carrier_congestion_gate": "PASS" if main_ok and tail_ok else "FAIL",
        "gate_rule": "Full improves overall MAE in >=3/4 folds and Extreme-Low MAE in >=2 valid folds, both with positive mean gain.",
        "information_boundary": "Chronos-2 receives contiguous D-3 and D-2 RT trajectories, forecasts 96 steps, and only D horizon steps 49-96 are scored.",
        "input_hashes": {
            "data": sha256(args.data),
            "labels": sha256(args.labels),
            "manifest": sha256(args.manifest),
            "folds": sha256(args.folds),
            "config": sha256(args.config),
            "code": sha256(Path(__file__)),
            "chronos_cache": sha256(args.cache),
        },
        "features": {"blind": energy_columns, "congestion_increment": congestion},
        "comparisons": {
            "chronos2_residual_full_vs_blind": primary,
            "chronos2_residual_full_vs_zero_shot": full_vs_zero,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    (args.output / "chronos2_strong_carrier_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"gate": report["strong_carrier_congestion_gate"], "primary": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
