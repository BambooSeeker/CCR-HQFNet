from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def delivery_day(times: pd.Series) -> pd.Series:
    day = times.dt.normalize()
    midnight = (times.dt.hour == 0) & (times.dt.minute == 0)
    return day.where(~midnight, day - pd.Timedelta(days=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input)
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    frame["delivery_day"] = delivery_day(frame["time"])
    congestion = frame["price_day_ahead_cong"].astype(float)
    frame["price_day_ahead_cong_abs"] = congestion.abs()
    for lag in (1, 2, 4):
        frame[f"price_day_ahead_cong_diff{lag}"] = congestion.diff(lag).fillna(0.0)
    frame["price_day_ahead_cong_shock"] = congestion.diff().abs().fillna(0.0)
    required = ["time", "delivery_day", manifest["target"], *manifest["features"]]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    clean = frame[required].replace([np.inf, -np.inf], np.nan).dropna()
    counts = clean.groupby("delivery_day").size()
    clean = clean[clean["delivery_day"].isin(counts[counts.eq(48)].index)]
    clean.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
