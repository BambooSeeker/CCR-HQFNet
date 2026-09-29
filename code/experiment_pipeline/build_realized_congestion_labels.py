from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_congestion_state_audit import load_realized_labels, state_label


def choose_folds(days: list[pd.Timestamp], count: int = 4) -> list[dict[str, list[str]]]:
    width = 104
    candidates = []
    for end in range(width, len(days) + 1, 30):
        segment = days[end - width : end]
        candidates.append({
            "train": segment[:90], "val": segment[90:97], "test": segment[97:104],
        })
    if candidates[-1]["test"][-1] != days[-1]:
        segment = days[-width:]
        candidates.append({"train": segment[:90], "val": segment[90:97], "test": segment[97:104]})
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    selected = [candidates[int(index)] for index in dict.fromkeys(indices.tolist())]
    return [
        {split: [str(day.date()) for day in fold[split]] for split in ("train", "val", "test")}
        for fold in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-data", type=Path, required=True)
    parser.add_argument("--raw-source-dir", type=Path, required=True)
    parser.add_argument("--plant-code", default="REPRESENTATIVE_SERIES")
    parser.add_argument("--deadband", type=float, default=0.01)
    parser.add_argument("--label-output", type=Path, required=True)
    parser.add_argument("--fold-output", type=Path, required=True)
    args = parser.parse_args()

    locked = pd.read_csv(args.locked_data, usecols=["time", "delivery_day"])
    locked["time"] = pd.to_datetime(locked["time"])
    locked["delivery_day"] = pd.to_datetime(locked["delivery_day"])
    labels = load_realized_labels(args.raw_source_dir, args.plant_code)
    merged = locked.merge(labels, on="time", how="left")
    merged["congestion_state"] = state_label(merged["price_real_cong"], args.deadband)
    output_columns = [
        "time", "delivery_day", "source_month", "price_real_energy",
        "price_real_cong", "price_real_load", "congestion_state",
    ]
    args.label_output.parent.mkdir(parents=True, exist_ok=True)
    merged[output_columns].to_csv(args.label_output, index=False)

    complete_days = (
        merged.assign(label_available=merged["congestion_state"].ge(0))
        .groupby("delivery_day")["label_available"].agg(["sum", "size"])
    )
    days = [pd.Timestamp(day) for day in complete_days.index[(complete_days["sum"] == 48) & (complete_days["size"] == 48)]]
    folds = choose_folds(days)
    args.fold_output.write_text(json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(merged), "labeled_rows": int(merged["congestion_state"].ge(0).sum()), "complete_label_days": len(days), "folds": len(folds)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
