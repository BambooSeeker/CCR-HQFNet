from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "03_results/r10_chronos2_lora_joint_route_v0.1/predictions.csv"
LABELS = ROOT / "02_experiments/data_locked/realized_congestion_labels_202405_202607.csv"
OUT = ROOT / "03_results/r16_state_conditioned_ablation_v1.0"
WINDOWS = {
    "recurrent_negative_price": ("2026-03-01 00:30:00", "2026-03-15 00:00:00"),
    "negative_congestion_dominant": ("2025-06-22 00:30:00", "2025-06-29 00:00:00"),
}
MODE_NAMES = {
    "energy_consistent_dual_route": "Full",
    "energy_consistent_supply_route": "w/o congestion route",
    "physical_congestion_gate": "w/o supply-price route",
    "carrier_blind": "w/o factorized routes",
    "ungated_dual_route": "w/o risk control",
    "chronos2_lora": "Chronos-2 carrier",
}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.abs(predicted - actual)
    valid = np.abs(actual) >= 50.0
    negative = actual < 0.0
    predicted_negative = predicted < 0.0
    tp = int(np.sum(negative & predicted_negative))
    fp = int(np.sum(~negative & predicted_negative))
    return {
        "targets": int(len(actual)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mape50": float(100.0 * np.mean(error[valid] / np.abs(actual[valid]))),
        "smape50": float(
            200.0
            * np.mean(
                error[valid]
                / np.maximum(np.abs(actual[valid]) + np.abs(predicted[valid]), 1e-9)
            )
        ),
        "negative_targets": int(negative.sum()),
        "negative_mae": (
            float(error[negative].mean()) if negative.any() else np.nan
        ),
        "negative_recall": (
            float(tp / negative.sum()) if negative.any() else np.nan
        ),
        "negative_precision": float(tp / (tp + fp)) if tp + fp else np.nan,
        "false_positive_negative": fp,
        "nonnegative_mae": float(error[~negative].mean()),
    }


def main() -> None:
    predictions = pd.read_csv(SOURCE)
    predictions["time"] = pd.to_datetime(predictions["time"])
    labels = pd.read_csv(LABELS, usecols=["time", "congestion_state"])
    labels["time"] = pd.to_datetime(labels["time"])

    rows = []
    point_blocks = []
    selection = {}
    for regime, (start, end) in WINDOWS.items():
        local = predictions[
            predictions["time"].between(pd.Timestamp(start), pd.Timestamp(end))
        ].copy()
        if local.groupby("mode")["time"].nunique().nunique() != 1:
            raise RuntimeError(f"Incomplete mode alignment for {regime}.")
        actual_once = local[local["mode"].eq("carrier_blind")][
            ["time", "actual"]
        ].merge(labels, on="time", validate="one_to_one")
        selection[regime] = {
            "start": start,
            "end": end,
            "targets": int(len(actual_once)),
            "negative_price_points": int((actual_once["actual"] < 0.0).sum()),
            "negative_congestion_points": int(
                (actual_once["congestion_state"] == 0).sum()
            ),
            "neutral_congestion_points": int(
                (actual_once["congestion_state"] == 1).sum()
            ),
            "positive_congestion_points": int(
                (actual_once["congestion_state"] == 2).sum()
            ),
        }
        for mode, frame in local.groupby("mode"):
            rows.append(
                {
                    "regime": regime,
                    "mode": MODE_NAMES[mode],
                    **metrics(
                        frame["actual"].to_numpy(float),
                        frame["predicted"].to_numpy(float),
                    ),
                }
            )
            block = frame.copy()
            block["regime"] = regime
            block["mode"] = MODE_NAMES[mode]
            point_blocks.append(block)

    result = pd.DataFrame(rows)
    points = pd.concat(point_blocks, ignore_index=True)
    full = result[result["mode"].eq("Full")].set_index("regime")
    effects = []
    for regime in WINDOWS:
        full_row = full.loc[regime]
        for ablation in MODE_NAMES.values():
            if ablation == "Full":
                continue
            other = result[
                result["regime"].eq(regime) & result["mode"].eq(ablation)
            ].iloc[0]
            effects.append(
                {
                    "regime": regime,
                    "comparison": f"Full vs {ablation}",
                    "mae_reduction": float(other["mae"] - full_row["mae"]),
                    "mae_reduction_pct": float(
                        100.0 * (other["mae"] - full_row["mae"]) / other["mae"]
                    ),
                    "mape50_reduction_points": float(
                        other["mape50"] - full_row["mape50"]
                    ),
                    "negative_mae_reduction": float(
                        other["negative_mae"] - full_row["negative_mae"]
                    ),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "ablation_metrics.csv", index=False)
    points.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(effects).to_csv(OUT / "effect_sizes.csv", index=False)
    (OUT / "selection_audit.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_rule": {
                    "recurrent_negative_price": (
                        "Continuous formal-market fortnight with recurring negative-price "
                        "episodes; selected independently of model errors."
                    ),
                    "negative_congestion_dominant": (
                        "Illustrative seven-day mechanism block with 42 negative-congestion "
                        "and four positive-congestion states in the frozen registry. It is "
                        "reported descriptively and is not used for population-level inference."
                    ),
                },
                "windows": selection,
                "source": str(SOURCE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        result[
            ["regime", "mode", "mae", "mape50", "smape50", "negative_mae",
             "negative_recall", "false_positive_negative"]
        ].sort_values(["regime", "mae"]).to_string(index=False)
    )
    print("\nEFFECTS")
    print(pd.DataFrame(effects).to_string(index=False))


if __name__ == "__main__":
    main()
