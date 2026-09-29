from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "03_results"
OUTPUT = RESULTS / "r15_experiment_closure_v1.0"

ALIGNED = (
    RESULTS
    / "r13_unified_fixed_window_comparison_v1.0"
    / "aligned_predictions.csv"
)
CHRONOS_LHFDC = RESULTS / "r13_chronos2_lora_lhfdc_v0.1" / "predictions.csv"
CHRONOS_YQFDC = RESULTS / "r15_chronos2_lora_yqfdc_fixed_v1.0" / "predictions.csv"
ROUTE_LHFDC = RESULTS / "r15_four_season_route_lhfdc_seed42_v1.0" / "predictions.csv"
ROUTE_YQFDC = RESULTS / "r15_four_season_route_yqfdc_seed42_v1.0" / "predictions.csv"
STRESS_ORIGINAL = RESULTS / "r10_chronos2_lora_joint_route_v0.1" / "predictions.csv"
STRESS_SEED42 = RESULTS / "r15_stress_route_seed42_v1.0" / "predictions.csv"
LIGHTGBM_PROB = (
    RESULTS
    / "r13_unified_fixed_window_comparison_v1.0"
    / "probability_metrics.csv"
)
YQ_LABELS = (
    ROOT
    / "02_experiments/data_locked/r13_fixed_seasonal_yqfdc_congestion_labels.csv"
)

MODE_NAMES = {
    "chronos2_lora": "Chronos-2",
    "carrier_blind": "Strict Blind",
    "physical_congestion_gate": "Physical congestion only",
    "energy_consistent_supply_route": "Supply-price only",
    "ungated_dual_route": "Without risk gate",
    "energy_consistent_dual_route": "Complete route",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def season_from_fold(fold: str) -> str:
    lower = fold.lower()
    for season in ("spring", "summer", "autumn", "winter"):
        if season in lower:
            return season.title()
    raise ValueError(f"Cannot infer season from {fold}.")


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    keep = np.abs(actual) >= 50.0
    negative = actual < 0.0
    predicted_negative = predicted < 0.0
    tp = int(np.sum(negative & predicted_negative))
    fp = int(np.sum(~negative & predicted_negative))
    fn = int(np.sum(negative & ~predicted_negative))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {
        "targets": int(len(actual)),
        "mape50_targets": int(keep.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape50_pct": (
            float(100.0 * np.mean(np.abs(error[keep] / actual[keep])))
            if keep.any()
            else np.nan
        ),
        "smape50_pct": (
            float(
                100.0
                * np.mean(
                    2.0
                    * np.abs(error[keep])
                    / np.maximum(
                        np.abs(actual[keep]) + np.abs(predicted[keep]), 1.0
                    )
                )
            )
            if keep.any()
            else np.nan
        ),
        "negative_count": int(negative.sum()),
        "negative_mae": (
            float(np.mean(np.abs(error[negative]))) if negative.any() else np.nan
        ),
        "negative_precision": precision,
        "negative_recall": recall,
        "negative_false_positive": fp,
    }


def pinball(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    residual = actual - predicted
    return float(np.mean(np.maximum(q * residual, (q - 1.0) * residual)))


def probability_metrics(
    actual: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
) -> dict[str, float | int]:
    lower = np.minimum(q05, q95)
    upper = np.maximum(q05, q95)
    width = upper - lower
    alpha = 0.10
    score = width.copy()
    below = actual < lower
    above = actual > upper
    score[below] += 2.0 / alpha * (lower[below] - actual[below])
    score[above] += 2.0 / alpha * (actual[above] - upper[above])
    values = [
        pinball(actual, lower, 0.05),
        pinball(actual, q50, 0.50),
        pinball(actual, upper, 0.95),
    ]
    return {
        "targets": int(len(actual)),
        "pinball_q05": values[0],
        "pinball_q50": values[1],
        "pinball_q95": values[2],
        "mean_pinball": float(np.mean(values)),
        "picp90_pct": float(100.0 * np.mean((actual >= lower) & (actual <= upper))),
        "mpiw": float(np.mean(width)),
        "winkler90": float(np.mean(score)),
    }


def read_chronos() -> pd.DataFrame:
    pieces = []
    for path in (CHRONOS_LHFDC, CHRONOS_YQFDC):
        frame = pd.read_csv(path)
        frame = frame[frame["mode"].eq("Chronos2_covariate_LoRA")].copy()
        frame["time"] = pd.to_datetime(frame["time"])
        frame["season"] = frame["fold"].map(season_from_fold)
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != 6048 or result["time"].duplicated().any():
        raise RuntimeError("Chronos four-season predictions are incomplete or duplicated.")
    return result.sort_values("time").reset_index(drop=True)


def read_routes() -> pd.DataFrame:
    pieces = []
    for path in (ROUTE_LHFDC, ROUTE_YQFDC):
        frame = pd.read_csv(path)
        frame["time"] = pd.to_datetime(frame["time"])
        frame["season"] = frame["fold_name"].map(season_from_fold)
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    expected = 6048 * len(MODE_NAMES)
    if len(result[result["mode"].isin(MODE_NAMES)]) != expected:
        raise RuntimeError("Four-season route output does not contain every required mode.")
    return result


def summarize_points(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = routes[routes["mode"].isin(MODE_NAMES)].copy()
    for (mode, season), part in selected.groupby(["mode", "season"], sort=False):
        rows.append(
            {
                "scope": season,
                "mode": mode,
                "model": MODE_NAMES[mode],
                **point_metrics(part["actual"].to_numpy(), part["predicted"].to_numpy()),
            }
        )
    for mode, part in selected.groupby("mode", sort=False):
        rows.append(
            {
                "scope": "Overall",
                "mode": mode,
                "model": MODE_NAMES[mode],
                **point_metrics(part["actual"].to_numpy(), part["predicted"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def summarize_probability(
    chronos: pd.DataFrame, routes: pd.DataFrame
) -> pd.DataFrame:
    complete = routes[routes["mode"].eq("energy_consistent_dual_route")][
        ["time", "actual", "predicted"]
    ].copy()
    carrier = routes[routes["mode"].eq("chronos2_lora")][
        ["time", "predicted"]
    ].rename(columns={"predicted": "route_carrier"})
    aligned = (
        chronos[
            ["time", "price_real", "q05", "prediction", "q95"]
        ]
        .merge(complete, on="time", how="inner", validate="one_to_one")
        .merge(carrier, on="time", how="inner", validate="one_to_one")
    )
    if len(aligned) != 6048:
        raise RuntimeError("Probability alignment is incomplete.")
    if np.max(np.abs(aligned["prediction"] - aligned["route_carrier"])) > 1e-3:
        raise RuntimeError("Route carrier does not match Chronos in-memory predictions.")
    actual = aligned["actual"].to_numpy(float)
    carrier_q50 = aligned["prediction"].to_numpy(float)
    carrier_metrics = probability_metrics(
        actual,
        aligned["q05"].to_numpy(float),
        carrier_q50,
        aligned["q95"].to_numpy(float),
    )
    correction = aligned["predicted"].to_numpy(float) - carrier_q50
    route_metrics = probability_metrics(
        actual,
        aligned["q05"].to_numpy(float) + correction,
        aligned["predicted"].to_numpy(float),
        aligned["q95"].to_numpy(float) + correction,
    )
    rows = [
        {"model": "Chronos-2", **carrier_metrics},
        {"model": "Complete route", **route_metrics},
    ]
    baseline = pd.read_csv(LIGHTGBM_PROB)
    baseline = baseline[baseline["model"].eq("LightGBMQuantile")]
    rows.extend(baseline.to_dict(orient="records"))
    return pd.DataFrame(rows)


def daily_significance(routes: pd.DataFrame) -> pd.DataFrame:
    selected = routes[routes["mode"].isin(MODE_NAMES)].copy()
    daily = (
        selected.groupby(["delivery_day", "mode"], as_index=False)["absolute_error"]
        .mean()
        .pivot(index="delivery_day", columns="mode", values="absolute_error")
    )
    complete = daily["energy_consistent_dual_route"]
    rows = []
    for mode, label in MODE_NAMES.items():
        if mode == "energy_consistent_dual_route":
            continue
        difference = daily[mode] - complete
        statistic, pvalue = wilcoxon(difference, alternative="two-sided")
        rows.append(
            {
                "complete_vs": label,
                "days": int(len(difference)),
                "mean_mae_reduction": float(difference.mean()),
                "median_mae_reduction": float(difference.median()),
                "improved_day_share_pct": float(100.0 * np.mean(difference > 0.0)),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_pvalue": float(pvalue),
            }
        )
    return pd.DataFrame(rows)


def seed_sensitivity() -> pd.DataFrame:
    rows = []
    for label, path in (
        ("seed_20260721", STRESS_ORIGINAL),
        ("seed_42", STRESS_SEED42),
    ):
        frame = pd.read_csv(path)
        for mode in (
            "carrier_blind",
            "physical_congestion_gate",
            "energy_consistent_supply_route",
            "ungated_dual_route",
            "energy_consistent_dual_route",
        ):
            part = frame[frame["mode"].eq(mode)]
            rows.append(
                {
                    "seed_run": label,
                    "mode": mode,
                    **point_metrics(
                        part["actual"].to_numpy(), part["predicted"].to_numpy()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    pivot = result.pivot(index="mode", columns="seed_run", values="mae")
    result = result.merge(
        (
            pivot["seed_42"] - pivot["seed_20260721"]
        ).rename("mae_seed42_minus_original"),
        on="mode",
    )
    return result


def autumn_diagnostic(routes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    autumn = routes[routes["season"].eq("Autumn")].copy()
    pivot = autumn.pivot(
        index="time", columns="mode", values=["actual", "predicted"]
    )
    actual = pivot["actual"]["energy_consistent_dual_route"]
    errors = pd.DataFrame(
        {
            MODE_NAMES[mode]: np.abs(pivot["predicted"][mode] - actual)
            for mode in MODE_NAMES
        }
    )
    labels = pd.read_csv(YQ_LABELS, usecols=["time", "congestion_state"])
    labels["time"] = pd.to_datetime(labels["time"])
    detail = (
        autumn[autumn["mode"].eq("energy_consistent_dual_route")][
            [
                "time",
                "delivery_day",
                "actual",
                "predicted",
                "candidate_activation",
                "energy_consistency_selected",
            ]
        ]
        .merge(labels, on="time", how="left", validate="one_to_one")
        .set_index("time")
        .join(errors)
        .reset_index()
    )
    detail["price_state"] = np.select(
        [detail["actual"] < 0.0, detail["actual"] < 200.0, detail["actual"] > 700.0],
        ["Negative", "Low nonnegative", "Extreme high"],
        default="Regular",
    )
    detail["congestion_state"] = detail["congestion_state"].map(
        {0: "Negative", 1: "Neutral", 2: "Positive"}
    )
    rows = []
    for (price_state, congestion_state), part in detail.groupby(
        ["price_state", "congestion_state"], dropna=False
    ):
        rows.append(
            {
                "price_state": price_state,
                "congestion_state": congestion_state,
                "points": int(len(part)),
                "share_pct": float(100.0 * len(part) / len(detail)),
                "actual_mean": float(part["actual"].mean()),
                "actual_std": float(part["actual"].std(ddof=0)),
                "activation_mean": float(part["candidate_activation"].mean()),
                "selected_share_pct": float(
                    100.0 * part["energy_consistency_selected"].mean()
                ),
                **{
                    f"{name}_mae": float(part[name].mean())
                    for name in MODE_NAMES.values()
                },
            }
        )
    summary = {
        "points": int(len(detail)),
        "days": int(detail["delivery_day"].nunique()),
        "negative_points": int((detail["actual"] < 0.0).sum()),
        "negative_share_pct": float(100.0 * (detail["actual"] < 0.0).mean()),
        "activation_mean": float(detail["candidate_activation"].mean()),
        "selected_share_pct": float(
            100.0 * detail["energy_consistency_selected"].mean()
        ),
        "complete_mae": float(detail["Complete route"].mean()),
        "chronos_mae": float(detail["Chronos-2"].mean()),
        "blind_mae": float(detail["Strict Blind"].mean()),
        "supply_only_mae": float(detail["Supply-price only"].mean()),
        "congestion_only_mae": float(
            detail["Physical congestion only"].mean()
        ),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    required = [
        ALIGNED,
        CHRONOS_LHFDC,
        CHRONOS_YQFDC,
        ROUTE_LHFDC,
        ROUTE_YQFDC,
        STRESS_ORIGINAL,
        STRESS_SEED42,
        LIGHTGBM_PROB,
        YQ_LABELS,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing closure inputs:\n" + "\n".join(missing))

    chronos = read_chronos()
    routes = read_routes()
    point = summarize_points(routes)
    probability = summarize_probability(chronos, routes)
    significance = daily_significance(routes)
    seeds = seed_sensitivity()
    autumn, autumn_summary = autumn_diagnostic(routes)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    point.to_csv(OUTPUT / "four_season_ablation_metrics.csv", index=False)
    probability.to_csv(OUTPUT / "four_season_probability_metrics.csv", index=False)
    significance.to_csv(OUTPUT / "four_season_daily_significance.csv", index=False)
    seeds.to_csv(OUTPUT / "stress_seed_sensitivity.csv", index=False)
    autumn.to_csv(OUTPUT / "autumn_state_diagnostic.csv", index=False)
    (OUTPUT / "autumn_summary.json").write_text(
        json.dumps(autumn_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "status": "complete",
        "four_season_points": 6048,
        "four_season_days": 126,
        "stress_test_role": (
            "The frozen 140-day registry is retained as a tail-heavy mechanism "
            "stress test; the 6,048-point registry now carries the primary ablation."
        ),
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "code_hash": sha256(Path(__file__)),
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(point[point["scope"].eq("Overall")].to_string(index=False))
    print(probability.to_string(index=False))
    print(significance.to_string(index=False))
    print(json.dumps(autumn_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
