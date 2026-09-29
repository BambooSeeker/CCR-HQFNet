from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon


LOWER_IS_BETTER = {
    "curve_mae",
    "ramp_mae",
    "peak_valley_spread_error",
    "peak_time_error_hours",
    "valley_time_error_hours",
    "negative_onset_error_hours",
    "negative_end_error_hours",
}

PRIMARY_METRICS = (
    "curve_mae",
    "pearson_r",
    "spearman_r",
    "ramp_mae",
    "ramp_direction_accuracy",
    "peak_top4_f1",
    "valley_bottom4_f1",
    "peak_valley_spread_error",
)

NEGATIVE_METRICS = (
    "negative_interval_f1",
    "negative_onset_error_hours",
    "negative_end_error_hours",
)

SECONDARY_METRICS = (
    "peak_time_error_hours",
    "valley_time_error_hours",
)


def safe_correlation(actual: np.ndarray, predicted: np.ndarray, kind: str) -> float:
    if np.std(actual) == 0.0 or np.std(predicted) == 0.0:
        return 0.0
    if kind == "pearson":
        return float(pearsonr(actual, predicted).statistic)
    return float(spearmanr(actual, predicted).statistic)


def stable_extreme_slots(values: np.ndarray, k: int, largest: bool) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    selected = order[-k:] if largest else order[:k]
    return np.sort(selected)


def set_f1(actual_slots: np.ndarray, predicted_slots: np.ndarray) -> float:
    intersection = len(set(actual_slots.tolist()) & set(predicted_slots.tolist()))
    precision = intersection / len(predicted_slots)
    recall = intersection / len(actual_slots)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def nearest_extreme_time_error(actual: np.ndarray, predicted: np.ndarray, largest: bool) -> float:
    actual_extreme = np.max(actual) if largest else np.min(actual)
    predicted_extreme = np.max(predicted) if largest else np.min(predicted)
    actual_slots = np.flatnonzero(np.isclose(actual, actual_extreme))
    predicted_slots = np.flatnonzero(np.isclose(predicted, predicted_extreme))
    distance = min(abs(int(i) - int(j)) for i in actual_slots for j in predicted_slots)
    return 0.5 * distance


def negative_event_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_negative = actual < 0.0
    predicted_negative = predicted < 0.0
    tp = int(np.sum(actual_negative & predicted_negative))
    fp = int(np.sum(~actual_negative & predicted_negative))
    fn = int(np.sum(actual_negative & ~predicted_negative))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if actual_negative.any() and predicted_negative.any():
        onset = 0.5 * abs(
            int(np.flatnonzero(actual_negative)[0]) - int(np.flatnonzero(predicted_negative)[0])
        )
        end = 0.5 * abs(
            int(np.flatnonzero(actual_negative)[-1]) - int(np.flatnonzero(predicted_negative)[-1])
        )
    elif actual_negative.any() or predicted_negative.any():
        onset = 24.0
        end = 24.0
    else:
        onset = 0.0
        end = 0.0
    return {
        "negative_interval_precision": precision,
        "negative_interval_recall": recall,
        "negative_interval_f1": f1,
        "negative_onset_error_hours": onset,
        "negative_end_error_hours": end,
    }


def build_daily_metrics(predictions: pd.DataFrame, top_k: int) -> pd.DataFrame:
    predictions = predictions.copy()
    if "fold_name" not in predictions.columns:
        predictions["fold_name"] = predictions["fold"].map(lambda value: f"fold_{value}")
    rows: list[dict] = []
    keys = ["fold", "fold_name", "delivery_day", "mode"]
    for (fold, fold_name, day, mode), group in predictions.groupby(keys, sort=False):
        group = group.sort_values("time")
        actual = group["actual"].to_numpy(float)
        predicted = group["predicted"].to_numpy(float)
        if len(actual) != 48:
            raise ValueError(f"{fold_name} {day} {mode} has {len(actual)} slots, expected 48.")
        actual_ramp = np.diff(actual)
        predicted_ramp = np.diff(predicted)
        actual_peak = stable_extreme_slots(actual, top_k, largest=True)
        predicted_peak = stable_extreme_slots(predicted, top_k, largest=True)
        actual_valley = stable_extreme_slots(actual, top_k, largest=False)
        predicted_valley = stable_extreme_slots(predicted, top_k, largest=False)
        row = {
            "fold": int(fold),
            "fold_name": fold_name,
            "delivery_day": day,
            "mode": mode,
            "actual_has_negative": bool(np.any(actual < 0.0)),
            "curve_mae": float(np.mean(np.abs(predicted - actual))),
            "pearson_r": safe_correlation(actual, predicted, "pearson"),
            "spearman_r": safe_correlation(actual, predicted, "spearman"),
            "ramp_mae": float(np.mean(np.abs(predicted_ramp - actual_ramp))),
            "ramp_direction_accuracy": float(
                np.mean(np.sign(predicted_ramp) == np.sign(actual_ramp))
            ),
            "peak_top4_f1": set_f1(actual_peak, predicted_peak),
            "valley_bottom4_f1": set_f1(actual_valley, predicted_valley),
            "peak_valley_spread_error": float(
                abs((np.max(predicted) - np.min(predicted)) - (np.max(actual) - np.min(actual)))
            ),
            "peak_time_error_hours": nearest_extreme_time_error(actual, predicted, largest=True),
            "valley_time_error_hours": nearest_extreme_time_error(actual, predicted, largest=False),
        }
        row.update(negative_event_metrics(actual, predicted))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    m = len(ordered)
    for rank, (metric, p_value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[metric] = running_max
    return adjusted


def compare_modes(
    daily: pd.DataFrame,
    candidate: str,
    baseline: str,
    metrics: tuple[str, ...],
    seed: int,
    negative_days_only: bool = False,
) -> dict:
    source = daily[daily["actual_has_negative"]].copy() if negative_days_only else daily
    result: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for metric in metrics:
        pivot = source.pivot(index=["fold", "delivery_day"], columns="mode", values=metric)
        pivot = pivot[[candidate, baseline]].dropna()
        if metric in LOWER_IS_BETTER:
            improvement = pivot[baseline] - pivot[candidate]
        else:
            improvement = pivot[candidate] - pivot[baseline]
        values = improvement.to_numpy(float)
        p_value = float(wilcoxon(values).pvalue) if np.any(values != 0.0) else 1.0
        raw_p[metric] = p_value
        result[metric] = {
            "candidate_mean": float(pivot[candidate].mean()),
            "baseline_mean": float(pivot[baseline].mean()),
            "mean_improvement": float(np.mean(values)),
            "mean_improvement_ci95": bootstrap_mean_ci(values, seed=seed),
            "improved_day_fraction": float(np.mean(values > 0.0)),
            "wilcoxon_two_sided_p": p_value,
            "n_days": int(len(values)),
        }
    adjusted = holm_adjust(raw_p)
    for metric, value in adjusted.items():
        result[metric]["holm_adjusted_p"] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate", default="energy_consistent_dual_route")
    parser.add_argument(
        "--baselines",
        default="carrier_blind,chronos2_lora,energy_consistent_supply_route",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    required_modes = {args.candidate, *baselines}
    missing = required_modes - set(predictions["mode"].unique())
    if missing:
        raise ValueError(f"Missing modes: {sorted(missing)}")

    daily = build_daily_metrics(predictions[predictions["mode"].isin(required_modes)], args.top_k)
    comparisons = {}
    for baseline_index, baseline in enumerate(baselines):
        comparisons[baseline] = {
            "primary_all_days": compare_modes(
                daily,
                args.candidate,
                baseline,
                PRIMARY_METRICS,
                seed=args.seed + baseline_index,
            ),
            "negative_event_days": compare_modes(
                daily,
                args.candidate,
                baseline,
                NEGATIVE_METRICS,
                seed=args.seed + 100 + baseline_index,
                negative_days_only=True,
            ),
            "single_extreme_diagnostics": compare_modes(
                daily,
                args.candidate,
                baseline,
                SECONDARY_METRICS,
                seed=args.seed + 200 + baseline_index,
            ),
        }

    report = {
        "status": "complete",
        "candidate": args.candidate,
        "baselines": baselines,
        "n_days": int(daily[["fold", "delivery_day"]].drop_duplicates().shape[0]),
        "n_negative_days": int(
            daily[daily["actual_has_negative"]][["fold", "delivery_day"]]
            .drop_duplicates()
            .shape[0]
        ),
        "definitions": {
            "peak_top4_f1": "F1 overlap between the four highest actual and predicted half-hour slots (a two-hour peak window).",
            "valley_bottom4_f1": "F1 overlap between the four lowest actual and predicted half-hour slots (a two-hour valley window).",
            "ramp_direction_accuracy": "Fraction of the 47 within-day first differences with matching signs.",
            "negative_event_subset": "Negative-price metrics are averaged only across days containing at least one realized negative-price slot.",
            "single_extreme_status": "Argmax/argmin timing is retained as a secondary diagnostic because isolated extrema are unstable under near-ties and price plateaus.",
            "inference": "Paired Wilcoxon tests use delivery days as blocks; 95% confidence intervals use 10,000 day-block bootstrap resamples; Holm adjustment is applied within each metric family.",
        },
        "comparisons": comparisons,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / "daily_curve_shape_metrics.csv", index=False)
    (args.output / "curve_shape_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
