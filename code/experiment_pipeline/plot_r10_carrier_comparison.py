from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def pooled(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[frame["scope"].eq("pooled")].copy()


def main() -> None:
    tabular = pooled(ROOT / "03_results/r10_carrier_baselines_v0.1/point_metrics.csv")
    neural = pooled(ROOT / "03_results/r10_neural_baselines_v0.1/point_metrics.csv")
    chronos = pd.read_csv(
        ROOT / "03_results/r10_chronos2_covariate_lora_v0.1/point_metrics.csv"
    )
    chronos = chronos[chronos["fold"].eq("pooled")]
    route = pd.read_csv(
        ROOT / "03_results/r10_chronos2_lora_joint_route_v0.1/predictions.csv"
    )

    rows: list[dict] = []
    selections = [
        (tabular, "SeasonalNaive", "Seasonal naive"),
        (tabular, "LightGBM", "LightGBM"),
        (tabular, "LEAR", "LEAR"),
        (tabular, "LightGBMQuantile", "LightGBM-Q"),
        (neural, "TFT", "TFT"),
        (neural, "TimeXer", "TimeXer"),
    ]
    for frame, model, label in selections:
        row = frame[frame["model"].eq(model)].iloc[0]
        rows.append(
            {
                "model": label,
                "mae": float(row["mae"]),
                "negative_mae": float(row["negative_price_mae"]),
            }
        )
    for mode, label in (
        ("Chronos2_covariate_zero_shot", "Chronos-2 cov. zero-shot"),
        ("Chronos2_covariate_LoRA", "Chronos-2 cov. LoRA"),
    ):
        row = chronos[chronos["mode"].eq(mode)].iloc[0]
        rows.append(
            {
                "model": label,
                "mae": float(row["mae"]),
                "negative_mae": float(row["negative_price_mae"]),
            }
        )
    full = route[route["mode"].eq("energy_consistent_dual_route")]
    rows.append(
        {
            "model": "Full routed model",
            "mae": float(full["absolute_error"].mean()),
            "negative_mae": float(full.loc[full["actual"] < 0, "absolute_error"].mean()),
        }
    )
    metrics = pd.DataFrame(rows)

    colors = ["#94A3B8"] * 6 + ["#F59E0B", "#2563EB", "#047857"]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), constrained_layout=True)
    y = np.arange(len(metrics))
    panels = (
        ("mae", "Overall MAE (RMB/MWh)"),
        ("negative_mae", "Negative-price MAE (RMB/MWh)"),
    )
    for axis, (column, title) in zip(axes, panels, strict=True):
        values = metrics[column].to_numpy(float)
        axis.barh(y, values, color=colors, height=0.68)
        axis.set_yticks(y)
        axis.set_yticklabels(metrics["model"])
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        for index, value in enumerate(values):
            axis.text(value + max(values) * 0.012, index, f"{value:.1f}", va="center", fontsize=9)
        axis.set_xlim(0, max(values) * 1.14)
    axes[0].set_title("Five frozen windows, 140 test days", loc="left", fontsize=12)
    axes[1].set_title("451 realized negative-price points", loc="left", fontsize=12)
    output = ROOT / "03_results/r10_chronos2_lora_joint_route_v0.1/evidence/r10_carrier_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
