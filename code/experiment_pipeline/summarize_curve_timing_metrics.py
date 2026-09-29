from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def nearest_extreme_time_error(actual: np.ndarray, predicted: np.ndarray, kind: str) -> float:
    actual_extreme = np.min(actual) if kind == "minimum" else np.max(actual)
    predicted_extreme = np.min(predicted) if kind == "minimum" else np.max(predicted)
    actual_slots = np.flatnonzero(np.isclose(actual, actual_extreme))
    predicted_slots = np.flatnonzero(np.isclose(predicted, predicted_extreme))
    distance = min(abs(int(i) - int(j)) for i in actual_slots for j in predicted_slots)
    return 0.5 * distance


def negative_interval_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual_negative = actual < 0.0
    predicted_negative = predicted < 0.0
    tp = int(np.sum(actual_negative & predicted_negative))
    fp = int(np.sum(~actual_negative & predicted_negative))
    fn = int(np.sum(actual_negative & ~predicted_negative))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if actual_negative.any() and predicted_negative.any():
        onset_error = 0.5 * abs(
            int(np.flatnonzero(actual_negative)[0]) - int(np.flatnonzero(predicted_negative)[0])
        )
        end_error = 0.5 * abs(
            int(np.flatnonzero(actual_negative)[-1]) - int(np.flatnonzero(predicted_negative)[-1])
        )
    elif actual_negative.any() or predicted_negative.any():
        onset_error = 24.0
        end_error = 24.0
    else:
        onset_error = 0.0
        end_error = 0.0
    return {
        "negative_interval_precision": precision,
        "negative_interval_recall": recall,
        "negative_interval_f1": f1,
        "negative_onset_error_hours": onset_error,
        "negative_end_error_hours": end_error,
    }


def build_daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, day, mode), group in predictions.groupby(["fold", "delivery_day", "mode"]):
        group = group.sort_values("time")
        actual = group["actual"].to_numpy(float)
        predicted = group["predicted"].to_numpy(float)
        if len(actual) != 48:
            continue
        row = {
            "fold": int(fold),
            "delivery_day": day,
            "mode": mode,
            "valley_time_error_hours": nearest_extreme_time_error(actual, predicted, "minimum"),
            "peak_time_error_hours": nearest_extreme_time_error(actual, predicted, "maximum"),
            "peak_valley_spread_error": abs(
                (np.max(predicted) - np.min(predicted)) - (np.max(actual) - np.min(actual))
            ),
            "curve_mae": float(np.mean(np.abs(predicted - actual))),
            "actual_has_negative": bool(np.any(actual < 0.0)),
        }
        row.update(negative_interval_metrics(actual, predicted))
        rows.append(row)
    return pd.DataFrame(rows)


def compare(daily: pd.DataFrame, candidate: str, baseline: str) -> dict:
    result = {}
    lower_is_better = (
        "valley_time_error_hours",
        "peak_time_error_hours",
        "peak_valley_spread_error",
        "curve_mae",
        "negative_onset_error_hours",
        "negative_end_error_hours",
    )
    higher_is_better = (
        "negative_interval_precision",
        "negative_interval_recall",
        "negative_interval_f1",
    )
    for metric in lower_is_better + higher_is_better:
        pivot = daily.pivot(index=["fold", "delivery_day"], columns="mode", values=metric).dropna()
        delta = pivot[baseline] - pivot[candidate] if metric in lower_is_better else pivot[candidate] - pivot[baseline]
        result[metric] = {
            "candidate_mean": float(pivot[candidate].mean()),
            "baseline_mean": float(pivot[baseline].mean()),
            "mean_improvement": float(delta.mean()),
            "improved_day_fraction": float((delta > 0).mean()),
            "wilcoxon_two_sided_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0,
            "n_days": int(len(delta)),
        }
    negative_days = daily[daily["actual_has_negative"]]
    result["actual_negative_day_count"] = int(
        negative_days[["fold", "delivery_day"]].drop_duplicates().shape[0]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    daily = build_daily_metrics(predictions)
    report = {
        "status": "complete",
        "candidate": args.candidate,
        "baseline": args.baseline,
        "definitions": {
            "valley_peak_timing": "Distance from the predicted extreme slot to the nearest tied actual extreme slot, in hours.",
            "negative_interval_missing_penalty": "24 hours when exactly one of actual or predicted trajectories contains negative prices.",
        },
        "comparison": compare(daily, args.candidate, args.baseline),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / "daily_curve_metrics.csv", index=False)
    (args.output / "curve_timing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
