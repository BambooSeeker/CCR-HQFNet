from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_supply_congestion_dual_route_audit import classification_metrics


RISK_BUDGET = 0.33
RISK_Z_90 = 1.281552
REGISTERED_MAE = 210.96577374954444
REGISTERED_NONNEGATIVE_MAE = 262.33698440917453


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def calibrate_threshold(validation: pd.DataFrame) -> tuple[dict, list[dict]]:
    actual_negative = validation["price_real"].to_numpy(float) < 0.0
    energy = validation["price_day_ahead_energy_proxy"].to_numpy(float)
    rows: list[dict] = []
    for threshold in np.unique(energy):
        selected = energy <= threshold
        selected_count = int(selected.sum())
        true_selected = int(np.sum(selected & actual_negative))
        false_selected = int(np.sum(selected & ~actual_negative))
        risk = false_selected / selected_count if selected_count else 1.0
        recall = true_selected / int(actual_negative.sum()) if actual_negative.any() else 0.0
        standard_error = np.sqrt(
            max(risk * (1.0 - risk), 0.0) / max(selected_count, 1)
        )
        upper = min(1.0, risk + RISK_Z_90 * standard_error)
        rows.append(
            {
                "energy_threshold": float(threshold),
                "selected_points": selected_count,
                "selected_negative_points": true_selected,
                "selected_nonnegative_points": false_selected,
                "selection_recall": float(recall),
                "selection_precision": float(
                    true_selected / selected_count if selected_count else 0.0
                ),
                "hard_false_route_risk": float(risk),
                "risk_normal_90pct_upper": float(upper),
                "feasible": bool(
                    selected_count >= 8 and risk <= RISK_BUDGET and upper <= RISK_BUDGET
                ),
            }
        )
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("no energy-consistency threshold satisfies the validation risk budget")
    selected = min(
        feasible,
        key=lambda row: (
            -row["selection_recall"],
            row["risk_normal_90pct_upper"],
            row["energy_threshold"],
        ),
    )
    return selected, rows


def mode_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    absolute = np.abs(predicted - actual)
    negative = actual < 0.0
    metrics = {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "negative_price_mae": float(absolute[negative].mean()),
        "negative_price_count": int(negative.sum()),
        "nonnegative_price_mae": float(absolute[~negative].mean()),
    }
    metrics.update(classification_metrics(actual, predicted))
    return metrics


def plot_event(predictions: pd.DataFrame, threshold: float, output: Path) -> None:
    wide = predictions.pivot(index="time", columns="mode", values="predicted")
    unique = predictions.drop_duplicates("time").set_index("time")
    time = pd.to_datetime(wide.index)
    actual = unique["actual"].to_numpy(float)
    energy = unique["price_day_ahead_energy_proxy"].to_numpy(float)

    fig, (ax_price, ax_gate) = plt.subplots(
        2,
        1,
        figsize=(14, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    ax_price.plot(time, actual, color="#111827", linewidth=1.7, label="Actual")
    ax_price.plot(
        time,
        wide["carrier_blind"],
        color="#64748B",
        linewidth=1.0,
        label="Blind",
    )
    ax_price.plot(
        time,
        wide["risk_supply_congestion_route"],
        color="#D97706",
        linewidth=1.0,
        label="Before consistency gate",
    )
    ax_price.plot(
        time,
        wide["energy_consistent_dual_route"],
        color="#047857",
        linewidth=1.4,
        label="Energy-consistent dual route",
    )
    ax_price.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    ax_price.set_ylabel("Price (RMB/MWh)")
    ax_price.legend(ncol=4, frameon=False, loc="upper left")
    ax_price.grid(axis="y", color="#E5E7EB", linewidth=0.7)

    ax_gate.plot(time, energy, color="#2563EB", linewidth=1.1, label="Day-ahead energy component")
    ax_gate.axhline(
        threshold,
        color="#DC2626",
        linewidth=1.0,
        linestyle="--",
        label=f"Validation risk threshold ({threshold:.1f})",
    )
    ax_gate.fill_between(
        time,
        energy.min(),
        energy.max(),
        where=actual < 0.0,
        color="#111827",
        alpha=0.08,
        step="mid",
        label="Actual negative price",
    )
    ax_gate.set_ylabel("RMB/MWh")
    ax_gate.set_xlabel("Delivery time")
    ax_gate.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax_gate.legend(ncol=3, frameon=False, loc="upper left")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    frame["price_day_ahead_energy_proxy"] = (
        frame["price_day_ahead"] - frame["price_day_ahead_cong"]
    )
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if len(folds) != 1:
        raise ValueError("event reject-gate audit expects one fold")
    fold = folds[0]
    validation = frame[frame["delivery_day"].isin(pd.to_datetime(fold["val"]))]
    selected, candidates = calibrate_threshold(validation)

    source = pd.read_csv(args.predictions)
    source["time"] = pd.to_datetime(source["time"])
    wide = source.pivot(index="time", columns="mode", values="predicted")
    unique = source.drop_duplicates("time").set_index("time")
    energy = frame.set_index("time")["price_day_ahead_energy_proxy"].reindex(wide.index)
    if energy.isna().any():
        raise ValueError("prediction timestamps do not align with locked data")
    rejected = energy.to_numpy(float) > selected["energy_threshold"]

    supply = wide["risk_supply_route_only"].to_numpy(float).copy()
    dual = wide["risk_supply_congestion_route"].to_numpy(float).copy()
    blind = wide["carrier_blind"].to_numpy(float)
    physical = wide["physical_congestion_gate"].to_numpy(float)
    supply[rejected] = np.maximum(blind[rejected], 0.0)
    dual[rejected] = np.maximum(physical[rejected], 0.0)

    modes = {
        "carrier_blind": blind,
        "physical_congestion_gate": physical,
        "risk_supply_route_only": wide["risk_supply_route_only"].to_numpy(float),
        "risk_supply_congestion_route": wide["risk_supply_congestion_route"].to_numpy(float),
        "energy_consistent_supply_route": supply,
        "energy_consistent_dual_route": dual,
    }
    actual = unique["actual"].to_numpy(float)
    rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for mode, predicted in modes.items():
        rows.append({"fold": 1, "mode": mode, **mode_metrics(actual, predicted)})
        prediction_rows.append(
            pd.DataFrame(
                {
                    "fold": 1,
                    "delivery_day": unique["delivery_day"].to_numpy(),
                    "time": wide.index.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": mode,
                    "actual": actual,
                    "predicted": predicted,
                    "absolute_error": np.abs(predicted - actual),
                    "price_day_ahead_energy_proxy": energy.to_numpy(float),
                    "energy_consistency_selected": ~rejected,
                }
            )
        )
    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = predictions.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(
        mae=("absolute_error", "mean")
    )

    final = metrics[metrics["mode"] == "energy_consistent_dual_route"].iloc[0]
    supply_final = metrics[metrics["mode"] == "energy_consistent_supply_route"].iloc[0]
    gates = {
        "validation_risk_upper_at_most_33pct": selected["risk_normal_90pct_upper"] <= RISK_BUDGET,
        "negative_recall_at_least_70pct": final["negative_recall"] >= 0.70,
        "false_positive_below_69": int(final["negative_false_positive"]) < 69,
        "precision_above_63_87pct": final["negative_precision"] > 0.6387,
        "nonnegative_mae_below_registered_262_337": final["nonnegative_price_mae"]
        < REGISTERED_NONNEGATIVE_MAE,
        "overall_mae_below_registered_210_966": final["mae"] < REGISTERED_MAE,
        "congestion_independent_gain": (
            final["mae"] < supply_final["mae"]
            and final["negative_price_mae"] < supply_final["negative_price_mae"]
        ),
    }
    report = {
        "status": "complete",
        "r9_event_gate": "PASS" if all(gates.values()) else "FAIL",
        "test_gates": {key: bool(value) for key, value in gates.items()},
        "mechanism": (
            "A validation-calibrated energy-consistency reject gate suppresses negative-expert outputs "
            "when the congestion-free day-ahead energy component lies outside its risk-controlled low tail. "
            "The gate is calibrated without test labels; rejected points return to a nonnegative Blind or "
            "physical-congestion baseline. The congestion correction remains independently testable."
        ),
        "equation": (
            "a_final = a_negative * I(lambda_DA_energy <= c); "
            "c = argmax_c Recall_val(c) subject to UCB_0.90(R_FP(c)) <= 0.33"
        ),
        "selected_validation_policy": selected,
        "validation_candidates": candidates,
        "input_hashes": {
            "data": sha256(args.data),
            "folds": sha256(args.folds),
            "source_predictions": sha256(args.predictions),
            "code": sha256(Path(__file__)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    (args.output / "energy_consistency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_event(
        predictions,
        selected["energy_threshold"],
        args.output / "spring_festival_energy_consistency_gate.png",
    )
    print(
        json.dumps(
            {
                "r9_event_gate": report["r9_event_gate"],
                "selected_validation_policy": selected,
                "test_gates": report["test_gates"],
                "final_metrics": final.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
