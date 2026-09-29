from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TRAIN_DAYS = 90
VALIDATION_DAYS = 7
TEST_DAYS = 28

# Calendar anchors are fixed before any model is evaluated on R10.
WINDOWS = (
    ("trial_late_summer_2024", "2024-08-13", "trial", "late_summer"),
    ("revised_spring_2025", "2025-04-06", "revised", "spring"),
    (
        "revised_summer_negative_2025",
        "2025-06-08",
        "revised",
        "summer_negative",
    ),
    ("formal_winter_event_2026", "2026-02-08", "formal", "winter_event"),
    ("formal_spring_2026", "2026-04-06", "formal", "spring"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def date_strings(values: pd.Index) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    counts = frame.groupby("delivery_day").size()
    if not counts.eq(48).all():
        invalid = counts[counts.ne(48)].to_dict()
        raise ValueError(f"R10 requires complete 48-point delivery days: {invalid}")

    days = pd.Index(sorted(frame["delivery_day"].unique()))
    folds: list[dict] = []
    audits: list[dict] = []
    for name, validation_start, phase, season in WINDOWS:
        anchor = pd.Timestamp(validation_start)
        if anchor not in days:
            raise ValueError(f"missing validation anchor: {validation_start}")
        position = days.get_loc(anchor)
        train = days[position - TRAIN_DAYS : position]
        validation = days[position : position + VALIDATION_DAYS]
        test = days[
            position + VALIDATION_DAYS : position + VALIDATION_DAYS + TEST_DAYS
        ]
        if (len(train), len(validation), len(test)) != (
            TRAIN_DAYS,
            VALIDATION_DAYS,
            TEST_DAYS,
        ):
            raise ValueError(f"insufficient complete days for {name}")

        folds.append(
            {
                "name": name,
                "market_phase": phase,
                "season": season,
                "train": date_strings(train),
                "val": date_strings(validation),
                "test": date_strings(test),
            }
        )

        validation_frame = frame[frame["delivery_day"].isin(validation)]
        test_frame = frame[frame["delivery_day"].isin(test)]
        test_by_day = test_frame.groupby("delivery_day")["price_real"]
        audits.append(
            {
                "name": name,
                "train_start": date_strings(train)[0],
                "train_end": date_strings(train)[-1],
                "validation_start": date_strings(validation)[0],
                "validation_end": date_strings(validation)[-1],
                "test_start": date_strings(test)[0],
                "test_end": date_strings(test)[-1],
                "test_calendar_span_days": int((test[-1] - test[0]).days + 1),
                "validation_negative_points": int(
                    (validation_frame["price_real"] < 0.0).sum()
                ),
                "test_negative_points": int((test_frame["price_real"] < 0.0).sum()),
                "test_negative_days": int(
                    test_by_day.apply(lambda values: (values < 0.0).any()).sum()
                ),
                "test_price_floor_points": int(
                    (test_frame["price_real"] <= -199.999).sum()
                ),
            }
        )

    args.folds_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.folds_output.write_text(
        json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "status": "locked_before_r10_model_evaluation",
        "selection_rule": (
            "Five calendar-anchored market-phase and seasonal windows; each uses the "
            "preceding 90 complete delivery days for training, the next 7 for recent "
            "validation, and the following 28 for testing. Model errors do not select windows."
        ),
        "grouping_key": "delivery_day",
        "train_days": TRAIN_DAYS,
        "recent_validation_days": VALIDATION_DAYS,
        "test_days": TEST_DAYS,
        "window_audits": audits,
        "input_hashes": {
            "data": sha256(args.data),
            "code": sha256(Path(__file__)),
        },
    }
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
