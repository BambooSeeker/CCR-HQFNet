from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon


FULL = "energy_consistent_dual_route"
BLIND = "carrier_blind"
NO_GATE = "ungated_dual_route"
NO_CONGESTION = "energy_consistent_supply_route"


def aggregate(frame: pd.DataFrame) -> dict:
    actual = frame["actual"].to_numpy(float)
    predicted = frame["predicted"].to_numpy(float)
    absolute = np.abs(predicted - actual)
    negative = actual < 0.0
    predicted_negative = predicted < 0.0
    true_positive = int(np.sum(negative & predicted_negative))
    false_positive = int(np.sum(~negative & predicted_negative))
    false_negative = int(np.sum(negative & ~predicted_negative))
    return {
        "points": int(len(frame)),
        "mae": float(absolute.mean()),
        "negative_price_mae": (
            float(absolute[negative].mean()) if negative.any() else None
        ),
        "nonnegative_price_mae": float(absolute[~negative].mean()),
        "negative_points": int(negative.sum()),
        "negative_recall": (
            true_positive / int(negative.sum()) if negative.any() else None
        ),
        "negative_precision": (
            true_positive / int(predicted_negative.sum())
            if predicted_negative.any()
            else 0.0
        ),
        "negative_true_positive": true_positive,
        "negative_false_positive": false_positive,
        "negative_false_negative": false_negative,
    }


def effect(candidate: dict, baseline: dict) -> dict:
    result = {}
    for name in ("mae", "negative_price_mae", "nonnegative_price_mae"):
        candidate_value = candidate[name]
        baseline_value = baseline[name]
        if candidate_value is None or baseline_value is None:
            result[name] = None
            continue
        absolute = baseline_value - candidate_value
        result[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "absolute_improvement": absolute,
            "relative_improvement_pct": 100.0 * absolute / baseline_value,
        }
    return result


def hac_dm(daily: pd.DataFrame, max_lag: int = 7) -> dict:
    values = daily["loss_difference"].to_numpy(float)
    centered = values - values.mean()
    n = len(values)
    long_run = float(np.dot(centered, centered) / n)
    fold = daily["fold"].to_numpy(int)
    day = pd.to_datetime(daily["delivery_day"]).to_numpy()
    for lag in range(1, min(max_lag, n - 1) + 1):
        valid = (fold[lag:] == fold[:-lag]) & (
            (day[lag:] - day[:-lag]) <= np.timedelta64(lag + 3, "D")
        )
        if not valid.any():
            continue
        covariance = float(
            np.mean(centered[lag:][valid] * centered[:-lag][valid])
        )
        long_run += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    variance_mean = max(long_run / n, 1e-12)
    statistic = float(values.mean() / np.sqrt(variance_mean))
    return {
        "n_days": n,
        "candidate_minus_baseline_daily_mae": float(values.mean()),
        "hac_lag": max_lag,
        "dm_hac_statistic": statistic,
        "dm_hac_two_sided_p": float(2.0 * norm.sf(abs(statistic))),
    }


def paired_tests(daily: pd.DataFrame, candidate: str, baseline: str) -> dict:
    wide = daily[daily["mode"].isin([candidate, baseline])].pivot(
        index=["fold", "fold_name", "delivery_day"], columns="mode", values="mae"
    ).dropna()
    difference = wide[candidate] - wide[baseline]
    test_frame = wide.reset_index()[["fold", "delivery_day"]]
    test_frame["loss_difference"] = difference.to_numpy(float)
    result = hac_dm(test_frame)
    result.update(
        {
            "candidate": candidate,
            "baseline": baseline,
            "candidate_better_day_fraction": float((difference < 0.0).mean()),
            "wilcoxon_two_sided_p": (
                float(wilcoxon(difference).pvalue)
                if np.any(difference.to_numpy() != 0.0)
                else 1.0
            ),
        }
    )
    return result


def plot_summary(metrics: pd.DataFrame, report: dict, output: Path) -> None:
    modes = [BLIND, NO_GATE, NO_CONGESTION, FULL]
    labels = ["Strict Blind", "Without gate", "Without congestion", "Full"]
    colors = ["#64748B", "#D97706", "#2563EB", "#047857"]
    pivot = metrics[metrics["mode"].isin(modes)].pivot(
        index="fold_name", columns="mode", values="mae"
    )
    order = metrics.drop_duplicates("fold").sort_values("fold")["fold_name"].tolist()
    pivot = pivot.loc[order]

    fig, (ax_fold, ax_effect) = plt.subplots(
        1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.65, 1.0]},
        constrained_layout=True,
    )
    x = np.arange(len(order))
    width = 0.19
    for index, (mode, label, color) in enumerate(zip(modes, labels, colors)):
        ax_fold.bar(
            x + (index - 1.5) * width,
            pivot[mode].to_numpy(float),
            width=width,
            label=label,
            color=color,
        )
    ax_fold.set_xticks(x)
    ax_fold.set_xticklabels(
        ["2024 trial\nlate summer", "2025 revised\nspring", "2025 revised\nsummer", "2026 formal\nwinter", "2026 formal\nspring"]
    )
    ax_fold.set_ylabel("MAE (RMB/MWh)")
    ax_fold.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax_fold.legend(frameon=False, ncol=2, loc="upper left")

    groups = ["all_windows", "revised_and_formal_2025plus", "revised_2025", "formal_2026"]
    group_labels = ["All windows", "2025 onward", "Revised 2025", "Formal 2026"]
    overall = [
        report["strata"][group]["full_vs_blind"]["mae"]["relative_improvement_pct"]
        for group in groups
    ]
    negative = [
        report["strata"][group]["full_vs_blind"]["negative_price_mae"]["relative_improvement_pct"]
        for group in groups
    ]
    y = np.arange(len(groups))
    ax_effect.barh(y + 0.16, overall, height=0.30, color="#047857", label="Overall MAE")
    ax_effect.barh(y - 0.16, negative, height=0.30, color="#2563EB", label="Negative-price MAE")
    ax_effect.axvline(0.0, color="#9CA3AF", linewidth=0.8)
    ax_effect.axvline(3.0, color="#DC2626", linewidth=0.9, linestyle="--", label="3% overall gate")
    ax_effect.set_yticks(y)
    ax_effect.set_yticklabels(group_labels)
    ax_effect.invert_yaxis()
    ax_effect.set_xlabel("Relative improvement over Strict Blind (%)")
    ax_effect.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax_effect.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    metrics = pd.read_csv(args.metrics)
    predictions["delivery_day"] = pd.to_datetime(predictions["delivery_day"])
    daily = predictions.groupby(
        ["fold", "fold_name", "delivery_day", "mode"], as_index=False
    ).agg(mae=("absolute_error", "mean"))

    strata_masks = {
        "all_windows": np.ones(len(predictions), dtype=bool),
        "revised_and_formal_2025plus": predictions["fold"].to_numpy(int) > 1,
        "revised_2025": predictions["fold"].isin([2, 3]).to_numpy(),
        "formal_2026": predictions["fold"].isin([4, 5]).to_numpy(),
    }
    strata = {}
    for name, mask in strata_masks.items():
        subset = predictions.loc[mask]
        modes = {
            mode: aggregate(local)
            for mode, local in subset.groupby("mode", sort=False)
        }
        strata[name] = {
            "modes": modes,
            "full_vs_blind": effect(modes[FULL], modes[BLIND]),
            "full_vs_no_gate": effect(modes[FULL], modes[NO_GATE]),
            "full_vs_no_congestion": effect(modes[FULL], modes[NO_CONGESTION]),
        }

    tests = {
        "full_vs_blind": paired_tests(daily, FULL, BLIND),
        "full_vs_no_gate": paired_tests(daily, FULL, NO_GATE),
        "full_vs_no_congestion": paired_tests(daily, FULL, NO_CONGESTION),
    }
    all_effect = strata["all_windows"]["full_vs_blind"]
    report = {
        "status": "complete",
        "r10_all_window_gate": {
            "overall_mae_improvement_at_least_3pct": bool(
                all_effect["mae"]["relative_improvement_pct"] >= 3.0
            ),
            "negative_mae_improvement_at_least_10pct": bool(
                all_effect["negative_price_mae"]["relative_improvement_pct"] >= 10.0
            ),
            "congestion_independent_gain": bool(
                strata["all_windows"]["full_vs_no_congestion"]["mae"][
                    "absolute_improvement"
                ]
                > 0.0
                and strata["all_windows"]["full_vs_no_congestion"][
                    "negative_price_mae"
                ]["absolute_improvement"]
                > 0.0
            ),
        },
        "strata": strata,
        "paired_tests": tests,
    }
    report["r10_all_window_gate"]["pass"] = all(
        report["r10_all_window_gate"].values()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "r10_seasonal_evidence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(tests).T.to_csv(args.output / "paired_tests.csv")
    plot_summary(metrics, report, args.output / "r10_seasonal_summary.png")
    print(json.dumps(report["r10_all_window_gate"], ensure_ascii=False, indent=2))
    print(json.dumps(tests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
