from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def season_for_month(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def market_phase(day: pd.Timestamp) -> str:
    return "formal" if day >= pd.Timestamp("2025-08-01") else "revised"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT
        / "02_experiments/data_locked/lhfdc01_forecast_available_202405_202607.csv",
    )
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "00_control/r12_continuous_year_monthly_folds.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "00_control/r12_continuous_year_monthly_folds_report.json",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.data, usecols=["delivery_day", "price_real"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    counts = frame.groupby("delivery_day").size().sort_index()
    complete_days = counts[counts.eq(48)].index

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    month_starts = pd.date_range(
        start=start.to_period("M").start_time,
        end=end.to_period("M").start_time,
        freq="MS",
    )

    folds: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    skipped_months: list[str] = []
    all_tests: list[pd.Timestamp] = []
    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(0)
        test_days = complete_days[
            (complete_days >= max(month_start, start))
            & (complete_days <= min(month_end, end))
        ]
        month_name = month_start.strftime("%Y-%m")
        if len(test_days) == 0:
            skipped_months.append(month_name)
            continue

        prior_days = complete_days[complete_days < test_days.min()]
        if len(prior_days) < 97:
            raise RuntimeError(
                f"{month_name} has only {len(prior_days)} preceding complete days."
            )
        validation_days = prior_days[-7:]
        training_days = prior_days[-97:-7]

        fold = {
            "name": f"continuous_{month_start.strftime('%Y_%m')}",
            "month": month_name,
            "market_phase": market_phase(test_days.min()),
            "season": season_for_month(month_start.month),
            "train": [day.date().isoformat() for day in training_days],
            "val": [day.date().isoformat() for day in validation_days],
            "test": [day.date().isoformat() for day in test_days],
        }
        folds.append(fold)
        all_tests.extend(test_days)

        calendar_days = (min(month_end, end) - max(month_start, start)).days + 1
        audits.append(
            {
                "name": fold["name"],
                "month": month_name,
                "season": fold["season"],
                "market_phase": fold["market_phase"],
                "train_start": training_days.min().date().isoformat(),
                "train_end": training_days.max().date().isoformat(),
                "validation_start": validation_days.min().date().isoformat(),
                "validation_end": validation_days.max().date().isoformat(),
                "test_start": test_days.min().date().isoformat(),
                "test_end": test_days.max().date().isoformat(),
                "test_complete_days": int(len(test_days)),
                "test_calendar_days": int(calendar_days),
                "test_coverage_pct": 100.0 * len(test_days) / calendar_days,
                "validation_to_test_gap_days": int(
                    (test_days.min() - validation_days.max()).days - 1
                ),
                "negative_points": int(
                    (
                        frame[
                            frame["delivery_day"].isin(test_days)
                        ]["price_real"]
                        < 0.0
                    ).sum()
                ),
            }
        )

    if len(all_tests) != len(set(all_tests)):
        raise RuntimeError("Monthly test sets overlap.")

    report = {
        "status": "preregistered_before_r12_model_evaluation",
        "selection_rule": (
            "Use every complete REPRESENTATIVE_SERIES delivery day in each calendar month "
            "from June 2025 through June 2026. For each nonempty month, use "
            "the preceding 90 complete days for training and the next seven "
            "complete days for validation. No model outcome selects dates."
        ),
        "requested_start": start.date().isoformat(),
        "requested_end": end.date().isoformat(),
        "fold_count": len(folds),
        "test_complete_days": len(all_tests),
        "test_points": 48 * len(all_tests),
        "skipped_months_without_lhfdc01_data": skipped_months,
        "source_data_sha256": sha256(args.data),
        "fold_audits": audits,
    }

    args.output.write_text(
        json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
