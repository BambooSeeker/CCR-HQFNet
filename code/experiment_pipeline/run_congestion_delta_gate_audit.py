from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_chronos2_strong_carrier_audit import fit_residual
from run_physical_soft_route_price_audit import add_features, day_mask, error_metrics, fit_router


MODES = ("carrier_blind", "carrier_full", "state_gated_congestion_delta")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def summarize(metrics: pd.DataFrame, daily: pd.DataFrame, candidate: str, baseline: str) -> dict:
    result = {}
    for metric_name in (
        "mae", "extreme_low_mae", "negative_price_mae", "negative_sign_recall",
        "negative_congestion_mae", "positive_congestion_mae",
    ):
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
    parser.add_argument("--chronos-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels, usecols=["time", "congestion_state"])
    carrier = pd.read_csv(args.chronos_cache)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    labels["time"] = pd.to_datetime(labels["time"])
    carrier["time"] = pd.to_datetime(carrier["time"])
    frame = add_features(frame.merge(labels, on="time", how="left").merge(carrier, on="time", how="left"))

    congestion = manifest["congestion_features"]
    blind_columns = [name for name in manifest["features"] if name not in congestion and name != "price_day_ahead"]
    blind_columns += ["slot_sin", "slot_cos", "slot_norm", "price_day_ahead_energy_proxy", "chronos2_prediction"]
    full_columns = blind_columns + congestion
    x_blind = frame[blind_columns].to_numpy(float)
    x_full = frame[full_columns].to_numpy(float)
    actual = frame["price_real"].to_numpy(float)
    carrier_prediction = frame["chronos2_prediction"].to_numpy(float)
    residual = actual - carrier_prediction
    state = frame["congestion_state"].fillna(-1).to_numpy(int)

    fold_rows, prediction_rows, route_rows = [], [], []
    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (day_mask(frame, fold[name]) for name in ("train", "val", "test"))
        labeled_train = train & (state >= 0)
        labeled_val = val & (state >= 0)
        if labeled_train.sum() < 64 or labeled_val.sum() < 16:
            raise ValueError(f"fold {fold_index} lacks router labels in train/validation")

        blind = fit_residual(x_blind[train], residual[train], x_blind[val], residual[val], config)
        full = fit_residual(x_full[train], residual[train], x_full[val], residual[val], config)
        pred_blind = carrier_prediction[test] + blind.predict(x_blind[test])
        pred_full = carrier_prediction[test] + full.predict(x_full[test])

        router = fit_router(
            x_full[labeled_train], state[labeled_train],
            x_full[labeled_val], state[labeled_val], config,
        )
        probability = router.predict_proba(x_full[test])
        nonneutral_gate = 1.0 - probability[:, 1]
        pred_gated = pred_blind + nonneutral_gate * (pred_full - pred_blind)
        predictions = {
            "carrier_blind": pred_blind,
            "carrier_full": pred_full,
            "state_gated_congestion_delta": pred_gated,
        }

        low_threshold = float(np.quantile(actual[train], 0.10))
        low_mask = actual[test] <= low_threshold
        test_indices = np.flatnonzero(test)
        route_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_index,
                    "time": frame.loc[test, "time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                    "p_negative": probability[:, 0],
                    "p_neutral": probability[:, 1],
                    "p_positive": probability[:, 2],
                    "nonneutral_gate": nonneutral_gate,
                    "full_minus_blind": pred_full - pred_blind,
                    "actual_state": state[test_indices],
                }
            )
        )
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
    routes = pd.concat(route_rows, ignore_index=True)
    daily = predictions.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(mae=("absolute_error", "mean"))
    routed_vs_blind = summarize(metrics, daily, "state_gated_congestion_delta", "carrier_blind")
    routed_vs_full = summarize(metrics, daily, "state_gated_congestion_delta", "carrier_full")
    overall_ok = routed_vs_blind["mae"]["improved_fold_count"] >= 3 and routed_vs_blind["mae"]["mean_improvement"] > 0
    tail_ok = routed_vs_blind["extreme_low_mae"]["improved_fold_count"] >= 2 and routed_vs_blind["extreme_low_mae"]["mean_improvement"] > 0
    report = {
        "status": "complete",
        "state_gated_delta_gate": "PASS" if overall_ok and tail_ok else "FAIL",
        "gate_rule": "State-gated delta improves Blind overall in >=3/4 folds and Extreme-Low in >=2 valid folds, both with positive mean gain.",
        "mechanism": "Blind carrier plus (1 - predicted neutral-congestion probability) times the Full-minus-Blind congestion response.",
        "input_hashes": {
            "data": sha256(args.data), "labels": sha256(args.labels), "manifest": sha256(args.manifest),
            "folds": sha256(args.folds), "config": sha256(args.config), "chronos_cache": sha256(args.chronos_cache),
            "code": sha256(Path(__file__)),
        },
        "comparisons": {
            "state_gated_delta_vs_blind": routed_vs_blind,
            "state_gated_delta_vs_full": routed_vs_full,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    routes.to_csv(args.output / "route_probabilities.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    (args.output / "state_gated_delta_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report["state_gated_delta_gate"], "vs_blind": routed_vs_blind, "vs_full": routed_vs_full}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
