from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_r10_joint_calibrated_route as frozen
import run_r10_joint_route_fold_carrier as runner


MIN_SELECTED = 4
NONNEGATIVE_TOLERANCE = 1.025
MIN_NEGATIVE_CALIBRATION = 4


def all_pretest_oof_days(
    signatures: pd.DataFrame,
    candidates: pd.Index,
    target_day: pd.Timestamp,
) -> tuple[pd.Index, dict]:
    del signatures, target_day
    return candidates, {
        "calibration_scope": "all_pretest_oof_days",
        "candidate_days": int(len(candidates)),
    }


def select_prequential_policy(calibration: pd.DataFrame) -> tuple[dict, dict]:
    actual = calibration["actual"].to_numpy(float)
    negative = actual < 0.0
    energy = calibration["price_day_ahead_energy_proxy"].to_numpy(float)
    baseline = calibration["physical_congestion_gate"].to_numpy(float)
    baseline_error = np.abs(baseline - actual)
    baseline_mae = float(baseline_error.mean())
    baseline_nonnegative = float(baseline_error[~negative].mean())
    baseline_negative = (
        float(baseline_error[negative].mean()) if negative.any() else None
    )

    candidates: list[dict] = []
    for supply_weight in frozen.SUPPLY_WEIGHT_GRID:
        for activation_threshold in frozen.ACTIVATION_THRESHOLD_GRID:
            activation, _, dual = frozen.route_from_policy(
                calibration, supply_weight, activation_threshold
            )
            active = activation > 0.0
            if not active.any():
                continue
            thresholds = np.unique(
                np.quantile(energy[active], np.linspace(0.0, 1.0, 41))
            )
            for energy_threshold in thresholds:
                selected = active & (energy <= energy_threshold)
                selected_count = int(selected.sum())
                if selected_count < MIN_SELECTED:
                    continue
                predicted = baseline.copy()
                predicted[selected] = dual[selected]
                absolute = np.abs(predicted - actual)
                mae = float(absolute.mean())
                nonnegative_mae = float(absolute[~negative].mean())
                negative_mae = (
                    float(absolute[negative].mean()) if negative.any() else None
                )
                enough_negative = int(negative.sum()) >= MIN_NEGATIVE_CALIBRATION
                negative_ok = (
                    negative_mae < baseline_negative
                    if enough_negative
                    and negative_mae is not None
                    and baseline_negative is not None
                    else True
                )
                feasible = bool(
                    mae <= baseline_mae
                    and nonnegative_mae
                    <= NONNEGATIVE_TOLERANCE * baseline_nonnegative
                    and negative_ok
                )
                objective = mae
                if enough_negative and negative_mae is not None:
                    objective += 0.20 * negative_mae
                candidates.append(
                    {
                        "supply_weight": float(supply_weight),
                        "activation_threshold": float(activation_threshold),
                        "energy_threshold": float(energy_threshold),
                        "selected_points": selected_count,
                        "calibration_negative_points": int(negative.sum()),
                        "mae": mae,
                        "negative_mae": negative_mae,
                        "nonnegative_mae": nonnegative_mae,
                        "objective": float(objective),
                        "feasible": feasible,
                    }
                )

    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        return {
            "supply_weight": frozen.SUPPLY_WEIGHT,
            "activation_threshold": frozen.ACTIVATION_THRESHOLD,
            "energy_threshold": None,
        }, {
            "status": "DISABLED_NO_PREQUENTIAL_IMPROVEMENT",
            "calibration_points": int(len(calibration)),
            "calibration_negative_points": int(negative.sum()),
            "baseline_mae": baseline_mae,
            "baseline_negative_mae": baseline_negative,
            "baseline_nonnegative_mae": baseline_nonnegative,
        }

    selected = min(
        feasible,
        key=lambda row: (
            row["objective"],
            row["mae"],
            row["nonnegative_mae"],
            -row["selected_points"],
        ),
    )
    return {
        "supply_weight": selected["supply_weight"],
        "activation_threshold": selected["activation_threshold"],
        "energy_threshold": selected["energy_threshold"],
    }, {
        "status": "PASS",
        "calibration_points": int(len(calibration)),
        "baseline_mae": baseline_mae,
        "baseline_negative_mae": baseline_negative,
        "baseline_nonnegative_mae": baseline_nonnegative,
        **selected,
    }


def output_argument() -> Path:
    index = sys.argv.index("--output")
    return Path(sys.argv[index + 1])


def main() -> None:
    frozen.nearest_oof_days = all_pretest_oof_days
    frozen.select_joint_policy = select_prequential_policy
    runner.main()

    report_path = output_argument() / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["route_implementation"] = "R16 pooled prequential risk calibration"
    report["prequential_policy"] = {
        "calibration": "45 pretest OOF days plus the 7-day validation block",
        "minimum_selected_points": MIN_SELECTED,
        "minimum_negative_calibration_points": MIN_NEGATIVE_CALIBRATION,
        "nonnegative_mae_tolerance": NONNEGATIVE_TOLERANCE,
        "rejection_action": "preserve the physical-congestion carrier prediction",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
