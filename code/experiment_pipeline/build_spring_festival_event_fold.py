from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--validation-end", type=str, default="2026-02-14")
    args = parser.parse_args()
    frame = pd.read_csv(args.data, usecols=["delivery_day", "price_real"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    counts = frame.groupby("delivery_day").size()
    complete_days = counts[counts == 48].index.sort_values()
    validation_end = pd.Timestamp(args.validation_end)
    validation = pd.date_range(
        validation_end - pd.Timedelta(days=args.validation_days - 1),
        validation_end,
        freq="D",
    )
    test = pd.date_range("2026-02-15", "2026-02-23", freq="D")
    for name, days in (("validation", validation), ("test", test)):
        missing = days.difference(complete_days)
        if len(missing):
            raise ValueError(f"{name} has incomplete days: {missing.strftime('%Y-%m-%d').tolist()}")
    eligible_train = complete_days[complete_days < validation.min()]
    train = eligible_train[-90:]
    if len(train) != 90:
        raise ValueError("fewer than 90 eligible training days")
    registry = [
        {
            "name": f"spring_festival_2026_val{args.validation_days}d",
            "train": train.strftime("%Y-%m-%d").tolist(),
            "val": validation.strftime("%Y-%m-%d").tolist(),
            "test": test.strftime("%Y-%m-%d").tolist(),
        }
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    train_rows = frame[frame["delivery_day"].isin(train)]
    val_rows = frame[frame["delivery_day"].isin(validation)]
    test_rows = frame[frame["delivery_day"].isin(test)]
    report = {
        "status": "complete",
        "event": "Chinese New Year holiday, 2026-02-15 through 2026-02-23",
        "train_days": len(train), "validation_days": len(validation), "test_days": len(test),
        "train_negative_points": int((train_rows["price_real"] < 0).sum()),
        "validation_negative_points": int((val_rows["price_real"] < 0).sum()),
        "test_negative_points": int((test_rows["price_real"] < 0).sum()),
        "test_floor_points": int((test_rows["price_real"] <= -199.99).sum()),
        "input_hash": sha256(args.data), "output_hash": sha256(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
