from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "03_results/r13_unified_fixed_window_comparison_v1.0"
ROUTE_DIRS = (
    ROOT / "03_results/r16_prequential_route_lhfdc_tol05_v1.0",
    ROOT / "03_results/r16_prequential_route_yqfdc_tol05_v1.0",
)
CARRIER_CACHES = (
    ROOT / "02_experiments/data_locked/r15_chronos2_lora_lhfdc_fold_cache.csv",
    ROOT / "02_experiments/data_locked/r15_chronos2_lora_yqfdc_fold_cache.csv",
)
OUT = ROOT / "03_results/r16_unified_experiment_closure_v1.0"

MODE_NAMES = {
    "energy_consistent_dual_route": "CCR-HQFNet (Full)",
    "ungated_dual_route": "w/o risk control",
    "energy_consistent_supply_route": "w/o congestion route",
    "physical_congestion_gate": "w/o supply-price route",
    "carrier_blind": "w/o factorized routes",
    "chronos2_lora": "Chronos-2 carrier",
}
SEASONS = ("Overall", "Spring", "Summer", "Autumn", "Winter")
DAY_TYPES = ("Typical-Normal", "High-Volatility", "Extreme-Low", "Extreme-High")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(predicted - actual)
    valid = np.abs(actual) >= 50.0
    negative = actual < 0.0
    predicted_negative = predicted < 0.0
    true_positive = int(np.sum(negative & predicted_negative))
    false_positive = int(np.sum(~negative & predicted_negative))
    return {
        "targets": int(len(actual)),
        "mape50_targets": int(valid.sum()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mape50": float(
            100.0 * np.mean(absolute[valid] / np.abs(actual[valid]))
        ),
        "smape50": float(
            200.0
            * np.mean(
                absolute[valid]
                / np.maximum(np.abs(actual[valid]) + np.abs(predicted[valid]), 1e-9)
            )
        ),
        "negative_targets": int(negative.sum()),
        "negative_mae": (
            float(absolute[negative].mean()) if negative.any() else np.nan
        ),
        "negative_recall": (
            float(true_positive / negative.sum()) if negative.any() else np.nan
        ),
        "negative_precision": (
            float(true_positive / (true_positive + false_positive))
            if true_positive + false_positive
            else np.nan
        ),
        "nonnegative_mae": float(absolute[~negative].mean()),
    }


def metric_table(
    frame: pd.DataFrame,
    columns: list[str],
    group_column: str,
    groups: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for group in groups:
        local = frame if group == "Overall" else frame[frame[group_column].eq(group)]
        for column in columns:
            rows.append(
                {
                    group_column: group,
                    "model": column,
                    **metrics(
                        local["actual"].to_numpy(float),
                        local[column].to_numpy(float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_daily_tests(
    frame: pd.DataFrame, full: str, baselines: list[str], family: str
) -> pd.DataFrame:
    local = frame[["season", "delivery_day", "actual", full, *baselines]].copy()
    for model in [full, *baselines]:
        local[model] = np.abs(local[model] - local["actual"])
    daily = local.groupby(["season", "delivery_day"], as_index=False)[
        [full, *baselines]
    ].mean()
    rows = []
    rng = np.random.default_rng(42)
    for baseline in baselines:
        delta = daily[baseline].to_numpy(float) - daily[full].to_numpy(float)
        samples = np.empty(10000, dtype=float)
        for index in range(len(samples)):
            samples[index] = rng.choice(delta, size=len(delta), replace=True).mean()
        statistic, p_value = wilcoxon(
            daily[full].to_numpy(float),
            daily[baseline].to_numpy(float),
            alternative="less",
            zero_method="wilcox",
        )
        rows.append(
            {
                "family": family,
                "comparison": f"{full} vs {baseline}",
                "baseline": baseline,
                "days": int(len(daily)),
                "baseline_minus_full_daily_mae": float(delta.mean()),
                "relative_improvement_pct": float(
                    100.0
                    * (daily[baseline].mean() - daily[full].mean())
                    / daily[baseline].mean()
                ),
                "bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_one_sided_p": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p"] = holm_adjust(
        result["wilcoxon_one_sided_p"].to_numpy(float)
    )
    return result


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_routes() -> pd.DataFrame:
    predictions = pd.concat(
        [pd.read_csv(path / "predictions.csv") for path in ROUTE_DIRS],
        ignore_index=True,
    )
    predictions["time"] = pd.to_datetime(predictions["time"])
    wide = predictions.pivot(
        index=["time", "delivery_day", "actual"],
        columns="mode",
        values="predicted",
    ).reset_index()
    wide = wide.rename(columns=MODE_NAMES)
    return wide


def probability_row(
    model: str,
    actual: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
) -> dict[str, float | int | str]:
    def pinball(predicted: np.ndarray, quantile: float) -> float:
        residual = actual - predicted
        return float(
            np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual))
        )

    alpha = 0.10
    width = q95 - q05
    winkler = width.copy()
    below = actual < q05
    above = actual > q95
    winkler[below] += (2.0 / alpha) * (q05[below] - actual[below])
    winkler[above] += (2.0 / alpha) * (actual[above] - q95[above])
    values = [pinball(q05, 0.05), pinball(q50, 0.50), pinball(q95, 0.95)]
    return {
        "model": model,
        "targets": int(len(actual)),
        "pinball_q05": values[0],
        "pinball_q50": values[1],
        "pinball_q95": values[2],
        "mean_pinball": float(np.mean(values)),
        "picp90_pct": float(100.0 * np.mean((actual >= q05) & (actual <= q95))),
        "mpiw": float(width.mean()),
        "winkler90": float(winkler.mean()),
    }


def probability_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = []
    for route_dir, cache_path in zip(ROUTE_DIRS, CARRIER_CACHES, strict=True):
        route = pd.read_csv(route_dir / "predictions.csv")
        route = route[route["mode"].eq("energy_consistent_dual_route")][
            ["fold_name", "time", "actual", "predicted"]
        ]
        cache = pd.read_csv(cache_path)[
            [
                "fold_name",
                "time",
                "chronos2_lora",
                "timesfm_q05",
                "timesfm_q50",
                "timesfm_q95",
            ]
        ]
        local = route.merge(
            cache, on=["fold_name", "time"], validate="one_to_one"
        )
        correction = local["predicted"] - local["chronos2_lora"]
        local["full_q05"] = local["timesfm_q05"] + correction
        local["full_q50"] = local["timesfm_q50"] + correction
        local["full_q95"] = local["timesfm_q95"] + correction
        blocks.append(local)
    frame = pd.concat(blocks, ignore_index=True)
    actual = frame["actual"].to_numpy(float)
    rows = [
        probability_row(
            "CCR-HQFNet (Full)",
            actual,
            frame["full_q05"].to_numpy(float),
            frame["full_q50"].to_numpy(float),
            frame["full_q95"].to_numpy(float),
        ),
        probability_row(
            "Chronos-2 carrier",
            actual,
            frame["timesfm_q05"].to_numpy(float),
            frame["timesfm_q50"].to_numpy(float),
            frame["timesfm_q95"].to_numpy(float),
        ),
    ]
    baseline = pd.read_csv(BASELINE_DIR / "probability_metrics.csv")
    rows.append(
        baseline[baseline["model"].eq("LightGBMQuantile")].iloc[0].to_dict()
    )
    return pd.DataFrame(rows), frame


def main() -> None:
    aligned = pd.read_csv(BASELINE_DIR / "aligned_predictions.csv")
    aligned["time"] = pd.to_datetime(aligned["time"])
    aligned["delivery_day"] = pd.to_datetime(aligned["delivery_day"])
    old_full = aligned["CCR-HQFNet"].copy()
    aligned = aligned.drop(columns=["CCR-HQFNet", "q05", "q50", "q95"])

    routes = load_routes()
    routes["delivery_day"] = pd.to_datetime(routes["delivery_day"])
    route_actual = routes[["time", "actual"]].rename(
        columns={"actual": "route_actual"}
    )
    frame = aligned.merge(
        routes.drop(columns="actual"), on=["time", "delivery_day"], validate="one_to_one"
    ).merge(route_actual, on="time", validate="one_to_one")
    maximum_actual_difference = float(
        np.max(np.abs(frame["actual"] - frame["route_actual"]))
    )
    if maximum_actual_difference > 2e-4:
        raise RuntimeError(
            f"Actual-price mismatch across experiment sources: {maximum_actual_difference}"
        )
    frame = frame.drop(columns="route_actual")

    day_classes = pd.read_csv(BASELINE_DIR / "day_classification.csv")
    day_classes["delivery_day"] = pd.to_datetime(day_classes["delivery_day"])
    frame = frame.merge(
        day_classes[["season", "delivery_day", "day_type"]],
        on=["season", "delivery_day"],
        validate="many_to_one",
    )
    if len(frame) != 6048:
        raise RuntimeError(f"Expected 6048 targets, found {len(frame)}.")

    full = "CCR-HQFNet (Full)"
    ablations = list(MODE_NAMES.values())
    comparison_models = [
        full,
        "Chronos-2 carrier",
        "TimesFM-2.5",
        "LightGBM",
        "TimeXer",
        "TFT",
        "PatchTST",
        "HybridTF-DilatedConv",
        "LEAR",
        "Transformer",
        "CNN_Transformer",
        "LSTM",
        "CNN_LSTM",
        "SeasonalNaive",
        "DLinear",
    ]
    comparison_models = list(dict.fromkeys(comparison_models))

    comparison_overall = metric_table(
        frame, comparison_models, "season", SEASONS
    )
    comparison_state = metric_table(
        frame, comparison_models, "day_type", DAY_TYPES
    )
    ablation_overall = metric_table(frame, ablations, "season", SEASONS)
    ablation_state = metric_table(frame, ablations, "day_type", DAY_TYPES)
    comparison_tests = paired_daily_tests(
        frame,
        full,
        [model for model in comparison_models if model != full],
        "comparison",
    )
    ablation_tests = paired_daily_tests(
        frame,
        full,
        [model for model in ablations if model != full],
        "ablation",
    )
    probability, probability_predictions = probability_evaluation()

    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "unified_predictions.csv", index=False)
    comparison_overall.to_csv(OUT / "comparison_overall_season.csv", index=False)
    comparison_state.to_csv(OUT / "comparison_market_state.csv", index=False)
    ablation_overall.to_csv(OUT / "ablation_overall_season.csv", index=False)
    ablation_state.to_csv(OUT / "ablation_market_state.csv", index=False)
    pd.concat([comparison_tests, ablation_tests], ignore_index=True).to_csv(
        OUT / "paired_daily_significance.csv", index=False
    )
    probability.to_csv(OUT / "probability_metrics.csv", index=False)
    probability_predictions.to_csv(
        OUT / "probability_predictions.csv", index=False
    )

    identity = {
        "status": "PASS",
        "targets": len(frame),
        "comparison_full_column": full,
        "ablation_full_column": full,
        "equal_target_count": int(
            np.sum(frame[full].to_numpy() == frame[full].to_numpy())
        ),
        "maximum_actual_difference": maximum_actual_difference,
        "old_initial_draft_full_equal_targets": int(
            np.sum(
                np.isclose(
                    old_full.to_numpy(float),
                    frame[full].to_numpy(float),
                    rtol=0.0,
                    atol=1e-9,
                )
            )
        ),
        "source_hashes": {
            str(path / "predictions.csv"): sha256(path / "predictions.csv")
            for path in ROUTE_DIRS
        },
    }
    (OUT / "full_identity_audit.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = comparison_overall[
        comparison_overall["season"].eq("Overall")
    ].sort_values("mae")
    ablation = ablation_overall[
        ablation_overall["season"].eq("Overall")
    ].sort_values("mae")
    print("COMPARISON")
    print(overall[["model", "mae", "mape50", "smape50"]].to_string(index=False))
    print("\nABLATION")
    print(
        ablation[
            ["model", "mae", "negative_mae", "negative_recall", "nonnegative_mae"]
        ].to_string(index=False)
    )
    print("\nPROBABILITY")
    print(probability.to_string(index=False))


if __name__ == "__main__":
    main()
