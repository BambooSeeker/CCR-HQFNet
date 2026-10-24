"""Recompute Table B.1 calibration-negative counts from saved inputs only.

Requires numpy and pandas. No model loading, fitting, inference or policy replay.
The frozen archive is read-only; only the two requested audit CSVs are written.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nearest_days(signatures, days, target):
    # Same pool scaling, missing-value rules and stable ties as r36 nearest.
    pool = signatures.loc[days].replace([np.inf, -np.inf], np.nan)
    mean = pool.mean()
    scale = pool.std().replace(0, 1).fillna(1)
    pool = pool.fillna(mean).fillna(0)
    target_signature = signatures.loc[target].fillna(mean).fillna(0)
    distance = np.square((pool - target_signature) / scale).sum(axis=1)
    return distance.sort_values(kind="stable").index[:14]


def reconstruct(archive):
    revision = archive / "r2_experiment_plan_20261018"
    trial = revision / "joint_routing_trial_20261022"
    selection_path = trial / "all_selection_frozen.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["folds"]
    primary_path = trial / "trial_predictions_LOCAL_ONLY.csv"
    full = pd.read_csv(primary_path, usecols=["time", "delivery_day", "model", "key"],
                       parse_dates=["time", "delivery_day"])
    full = full.loc[full.model.eq("Full")]
    require(len(full) == full.time.nunique() == 6048, "Expected 6,048 unique Full targets.")
    require(full.delivery_day.nunique() == 126, "Expected 126 target delivery days.")
    require(len(selection) == 20, "Expected 20 frozen fold selections.")
    require({rec["key"] for rec in selection} == set(full.key), "Fold selections differ from Full.")
    records = []
    for rec in selection:
        key, selected = rec["key"], rec["full"]
        candidate = trial / "folds" / key / f"candidate_{selected['candidate']}"
        suffix = f"shrink{selected['shrink']}_LOCAL_ONLY.csv"
        oof_path = candidate / ("oof_" + suffix)
        validation_path = candidate / ("validation_" + suffix)
        signature_path = revision / "congestion_validation_20261022" / key / "daily_signatures_LOCAL_ONLY.csv"
        inputs = [pd.read_csv(path, usecols=["time", "delivery_day", "actual"],
                              parse_dates=["time", "delivery_day"])
                  for path in (oof_path, validation_path)]
        oof, validation = inputs
        for frame in inputs:
            require(frame.time.is_unique and frame.notna().all().all(), f"Invalid saved inputs: {key}")
            require(np.isfinite(frame.actual).all(), f"Nonfinite calibration targets: {key}")
        require(validation.delivery_day.nunique() == 7, f"Expected seven validation days: {key}")
        signatures = pd.read_csv(signature_path, index_col=0, parse_dates=True)
        require(signatures.index.is_unique, f"Duplicate signature days: {key}")
        pool_days = pd.Index(sorted(oof.delivery_day.unique()))
        targets = full.loc[full.key.eq(key)]
        fold = "-".join(key.split("_")[-2:]).title()
        for target in sorted(targets.delivery_day.unique()):
            day = pd.Timestamp(target)
            near = nearest_days(signatures, pool_days, day)
            require(len(near) == 14, f"Insufficient OOF days: {key}/{day}")
            cal = pd.concat([oof.loc[oof.delivery_day.isin(near)], validation], ignore_index=True)
            hold = targets.loc[targets.delivery_day.eq(day)]
            require(len(hold) == 48, f"Incomplete target day: {key}/{day}")
            require(len(cal) == cal.time.nunique() == 1008, f"Calibration size/overlap: {key}/{day}")
            require(cal.delivery_day.nunique() == 21 and cal.groupby("delivery_day").size().eq(48).all(),
                    f"Calibration day counts: {key}/{day}")
            require(cal.delivery_day.max() < day and cal.time.max() < hold.time.min(),
                    f"Calibration chronology: {key}/{day}")
            require(set(cal.time).isdisjoint(hold.time), f"Target overlap: {key}/{day}")
            records.append({
                "fold": fold, "key": key, "target_day": day.date().isoformat(),
                "calibration_n": len(cal), "calibration_days": 21, "oof_days": 14,
                "validation_days": 7, "calibration_negatives": int(cal.actual.lt(0).sum()),
                "calibration_latest_time": cal.time.max().isoformat(),
                "target_first_time": hold.time.min().isoformat(),
                "chronology_passed": True, "target_overlap_n": 0,
                "selected_oof_days": json.dumps([pd.Timestamp(d).date().isoformat() for d in near]),
                "validation_delivery_days": json.dumps([pd.Timestamp(d).date().isoformat()
                                                        for d in sorted(validation.delivery_day.unique())]),
                "oof_source": oof_path.relative_to(archive).as_posix(),
                "validation_source": validation_path.relative_to(archive).as_posix(),
                "signatures_source": signature_path.relative_to(archive).as_posix(),
                "selection_source": selection_path.relative_to(archive).as_posix(),
                "target_registry_source": primary_path.relative_to(archive).as_posix(),
            })
    daily = pd.DataFrame(records)
    require(len(daily) == 126 and not daily.duplicated(["key", "target_day"]).any(),
            "Expected 126 unique fold/day records.")
    folds = daily.groupby(["fold", "key"], sort=False).agg(
        target_days=("target_day", "size"),
        calibration_negative_min=("calibration_negatives", "min"),
        calibration_negative_max=("calibration_negatives", "max"),
        calibration_n_min=("calibration_n", "min"),
        calibration_n_max=("calibration_n", "max"),
        chronology_passed=("chronology_passed", "all"),
    ).reset_index()
    folds["calibration_negative_range"] = (folds.calibration_negative_min.astype(str)
                                            + "-" + folds.calibration_negative_max.astype(str))
    return daily, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Extracted research_archive directory")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/final_revision_audit")
    args = parser.parse_args()
    archive, output = args.archive.expanduser().resolve(), args.output.expanduser().resolve()
    require(archive != output and archive not in output.parents and output not in archive.parents,
            "Output must be separate from the frozen archive.")
    daily, folds = reconstruct(archive)
    output.mkdir(parents=True, exist_ok=True)
    daily_path, fold_path = output / "B1_calibration_day_counts.csv", output / "B1_calibration_fold_counts.csv"
    daily.to_csv(daily_path, index=False)
    folds.to_csv(fold_path, index=False)
    print(folds[["fold", "target_days", "calibration_negative_range"]].to_string(index=False))
    print(f"PASS: {len(daily)} target days, {len(folds)} folds; 14+7 days, 1008 unique observations, chronology.")
    print(f"Per-day source records: {daily_path}")
    print(f"Fold ranges: {fold_path}")


if __name__ == "__main__":
    main()
