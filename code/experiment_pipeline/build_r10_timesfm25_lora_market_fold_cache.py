from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODE = "TimesFM25_LoRA_market_conditioned"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path,
        default=ROOT / "03_results/r10_timesfm25_lora_market_v0.1/predictions_all_splits.csv",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "02_experiments/data_locked/lhfdc01_forecast_available_202405_202607.csv",
    )
    parser.add_argument(
        "--folds", type=Path, default=ROOT / "00_control/r10_seasonal_long_windows.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "02_experiments/data_locked/r10_timesfm25_lora_market_fold_cache.csv",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.predictions)
    source = source[source["mode"].eq(MODE)].copy()
    source["time"] = pd.to_datetime(source["time"])
    source["delivery_day"] = pd.to_datetime(source["delivery_day"]).dt.normalize()
    cache = source[
        ["fold_name", "delivery_day", "time", "prediction", "q10", "q90"]
    ].rename(
        columns={
            "prediction": "timesfm25_lora_market",
            "q10": "timesfm_q10",
            "q90": "timesfm_q90",
        }
    )
    cache["timesfm_q50"] = cache["timesfm25_lora_market"]
    cache["carrier_source"] = "in_memory_lora_market_prediction"

    frame = pd.read_csv(args.data)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fallback_rows: list[pd.DataFrame] = []
    for fold in folds:
        present = set(cache.loc[cache["fold_name"].eq(fold["name"]), "time"])
        relevant_days = pd.to_datetime(fold["train"] + fold["val"] + fold["test"])
        expected = frame[frame["delivery_day"].isin(relevant_days)].copy()
        missing = expected[~expected["time"].isin(present)].copy()
        if missing.empty:
            continue
        point = missing["price_real_lag96"].to_numpy(float)
        scale = float(np.std(point))
        fallback_rows.append(
            pd.DataFrame(
                {
                    "fold_name": fold["name"],
                    "delivery_day": missing["delivery_day"].to_numpy(),
                    "time": missing["time"].to_numpy(),
                    "timesfm25_lora_market": point,
                    "timesfm_q10": point - scale,
                    "timesfm_q90": point + scale,
                    "timesfm_q50": point,
                    "carrier_source": "safe_D_minus_2_fallback",
                }
            )
        )
    if fallback_rows:
        cache = pd.concat([cache, *fallback_rows], ignore_index=True)
    cache = cache.sort_values(["fold_name", "time"]).reset_index(drop=True)
    if cache.duplicated(["fold_name", "time"]).any():
        raise ValueError("Fold cache contains duplicate fold/time rows.")
    expected_rows = sum(
        len(fold["train"] + fold["val"] + fold["test"]) * 48 for fold in folds
    )
    if len(cache) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(cache)}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(args.output, index=False)
    report = {
        "rows": int(len(cache)),
        "folds": int(cache["fold_name"].nunique()),
        "fallback_rows": int(cache["carrier_source"].eq("safe_D_minus_2_fallback").sum()),
        "source_mode": MODE,
        "information_boundary": "Training-only LoRA and market conditioner; no explicit congestion carrier input.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
