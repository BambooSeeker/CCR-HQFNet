from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def effect(frame: pd.DataFrame, candidate: str, baseline: str, mask: np.ndarray) -> dict:
    subset = frame.loc[mask]
    baseline_mae = float(subset[f"ae_{baseline}"].mean())
    candidate_mae = float(subset[f"ae_{candidate}"].mean())
    absolute = baseline_mae - candidate_mae
    return {
        "n": int(len(subset)),
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "absolute_improvement": absolute,
        "relative_improvement_pct": float(100.0 * absolute / baseline_mae),
        "total_absolute_error_reduction": float(
            (subset[f"ae_{baseline}"] - subset[f"ae_{candidate}"]).sum()
        ),
    }


def summarize_one(name: str, path: Path) -> tuple[dict, pd.DataFrame]:
    predictions = pd.read_csv(path / "predictions.csv")
    index_columns = ["fold", "delivery_day", "time", "actual", "low_price", "congestion_state"]
    wide = None
    for mode in ("carrier_blind", "physical_congestion_gate", "physical_gate_low_expert"):
        local = predictions[predictions["mode"] == mode][index_columns + ["absolute_error"]].copy()
        local = local.rename(columns={"absolute_error": f"ae_{mode}"})
        wide = local if wide is None else wide.merge(local, on=index_columns, how="inner", validate="one_to_one")
    if wide is None or len(wide) == 0:
        raise ValueError(f"no aligned predictions in {path}")
    low = wide["low_price"].astype(bool).to_numpy()
    negative = (wide["actual"] < 0).to_numpy()
    groups = {
        "all": np.ones(len(wide), dtype=bool),
        "extreme_low": low,
        "non_low": ~low,
        "negative_price": negative,
        "nonnegative_price": ~negative,
        "negative_congestion": (wide["congestion_state"] == 0).to_numpy(),
        "neutral_congestion": (wide["congestion_state"] == 1).to_numpy(),
        "positive_congestion": (wide["congestion_state"] == 2).to_numpy(),
    }
    comparisons = {}
    for comparison_name, candidate, baseline in (
        ("full_mechanism_vs_blind", "physical_gate_low_expert", "carrier_blind"),
        ("physical_congestion_gate_vs_blind", "physical_congestion_gate", "carrier_blind"),
        ("low_expert_vs_physical_gate", "physical_gate_low_expert", "physical_congestion_gate"),
    ):
        comparisons[comparison_name] = {
            group_name: effect(wide, candidate, baseline, mask)
            for group_name, mask in groups.items()
            if mask.any()
        }
    all_effect = comparisons["full_mechanism_vs_blind"]["all"]
    low_effect = comparisons["full_mechanism_vs_blind"]["extreme_low"]
    nonlow_effect = comparisons["full_mechanism_vs_blind"]["non_low"]
    attribution = {
        "low_sample_share_pct": float(100.0 * low.sum() / len(wide)),
        "low_total_error_reduction": low_effect["total_absolute_error_reduction"],
        "nonlow_total_error_reduction": nonlow_effect["total_absolute_error_reduction"],
        "overall_total_error_reduction": all_effect["total_absolute_error_reduction"],
        "low_share_of_net_reduction_pct": float(
            100.0 * low_effect["total_absolute_error_reduction"] / all_effect["total_absolute_error_reduction"]
        ),
    }
    fold_rows = []
    for fold, fold_frame in wide.groupby("fold"):
        row = effect(
            fold_frame,
            "physical_gate_low_expert",
            "carrier_blind",
            np.ones(len(fold_frame), dtype=bool),
        )
        fold_rows.append({"carrier": name, "fold": int(fold), **row})
    report = json.loads((path / "low_price_expert_report.json").read_text(encoding="utf-8"))
    result = {
        "carrier": name,
        "carrier_column": report.get("carrier_column"),
        "n_test_points": int(len(wide)),
        "comparisons": comparisons,
        "error_reduction_attribution": attribution,
        "daily_mae_test": report["comparisons"]["final_vs_blind"]["daily_mae"],
    }
    return result, pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", action="append", nargs=2, metavar=("NAME", "RESULT_DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    fold_tables = []
    for name, directory in args.carrier:
        report, folds = summarize_one(name, Path(directory))
        reports.append(report)
        fold_tables.append(folds)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "carrier_effect_sizes.json").write_text(
        json.dumps({"status": "complete", "carriers": reports}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(fold_tables, ignore_index=True).to_csv(args.output / "fold_relative_improvements.csv", index=False)
    print(json.dumps({"status": "complete", "carriers": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
