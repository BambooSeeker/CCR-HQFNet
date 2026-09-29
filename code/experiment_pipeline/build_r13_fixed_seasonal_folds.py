from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ALIGNED = (
    ROOT
    / "03_results"
    / "r13_fixed_seasonal_comparison_v0.1"
    / "aligned_predictions.csv"
)
DATASETS = {
    "lhfdc": {
        "path": ROOT / "02_experiments/data_locked/r13_fixed_seasonal_lhfdc_data.csv",
        "seasons": {"Spring", "Summer"},
    },
    "yqfdc": {
        "path": ROOT / "02_experiments/data_locked/r13_fixed_seasonal_yqfdc_data.csv",
        "seasons": {"Autumn", "Winter"},
    },
}
REPORT = ROOT / "00_control" / "r13_fixed_seasonal_folds_report.json"


def iso_days(values: list[pd.Timestamp]) -> list[str]:
    return [str(pd.Timestamp(value).date()) for value in values]


def main() -> None:
    targets = pd.read_csv(ALIGNED, usecols=["time", "actual", "season"])
    targets["time"] = pd.to_datetime(targets["time"])
    all_folds: list[dict[str, object]] = []
    season_report: list[dict[str, object]] = []
    node_reports = {}
    for node_key, config in DATASETS.items():
        frame = pd.read_csv(config["path"], usecols=["time", "delivery_day", "price_real"])
        frame["time"] = pd.to_datetime(frame["time"])
        frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
        selected_targets = targets[targets["season"].isin(config["seasons"])].copy()
        merged = selected_targets.merge(
            frame, on="time", how="left", validate="one_to_one"
        )
        if merged[["delivery_day", "price_real"]].isna().any().any():
            missing = int(merged["delivery_day"].isna().sum())
            raise RuntimeError(f"{node_key}: {missing} target timestamps are absent.")
        max_actual_delta = float(
            np.max(np.abs(merged["actual"].to_numpy() - merged["price_real"].to_numpy()))
        )
        if max_actual_delta > 1e-3:
            raise RuntimeError(
                f"{node_key}: target mismatch, max delta={max_actual_delta:.6f}."
            )
        available_counts = frame.groupby("delivery_day").size()
        complete_days = sorted(
            pd.Timestamp(day)
            for day in available_counts[available_counts.eq(48)].index
        )
        folds: list[dict[str, object]] = []
        for season, season_frame in merged.groupby("season", sort=False):
            target_counts = season_frame.groupby("delivery_day").size()
            if not bool(target_counts.eq(48).all()):
                raise RuntimeError(f"{season} contains incomplete delivery days.")
            test_days = sorted(pd.Timestamp(day) for day in target_counts.index)
            for block_index, start in enumerate(range(0, len(test_days), 7), start=1):
                block = test_days[start : start + 7]
                prior = [day for day in complete_days if day < block[0]]
                if len(prior) < 14:
                    raise RuntimeError(
                        f"Only {len(prior)} prior complete days for {block[0]}."
                    )
                val = prior[-7:]
                train = prior[max(0, len(prior) - 97) : -7]
                folds.append(
                    {
                        "name": f"fixed_{season.lower()}_{block_index:02d}",
                        "market_phase": (
                            "revised"
                            if block[0] < pd.Timestamp("2025-10-01")
                            else "formal"
                        ),
                        "season": season.lower(),
                        "train": iso_days(train),
                        "val": iso_days(val),
                        "test": iso_days(block),
                        "test_points": int(len(block) * 48),
                        "train_days_actual": len(train),
                    }
                )
            season_report.append(
                {
                    "season": season,
                    "test_days": len(test_days),
                    "targets": int(len(season_frame)),
                    "folds": int(np.ceil(len(test_days) / 7)),
                    "first_day": str(test_days[0].date()),
                    "last_day": str(test_days[-1].date()),
                    "node_key": node_key,
                }
            )
        output = ROOT / "00_control" / f"r13_fixed_seasonal_folds_{node_key}.json"
        output.write_text(json.dumps(folds, indent=2), encoding="utf-8")
        all_folds.extend(folds)
        node_reports[node_key] = {
            "fold_count": len(folds),
            "targets": sum(int(fold["test_points"]) for fold in folds),
            "max_actual_delta": max_actual_delta,
        }

    if sum(int(fold["test_points"]) for fold in all_folds) != len(targets):
        raise RuntimeError("Fold target count does not equal the 6,048-point benchmark.")
    REPORT.write_text(
        json.dumps(
            {
                "targets": int(len(targets)),
                "fold_count": len(all_folds),
                "nodes": node_reports,
                "seasons": season_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
