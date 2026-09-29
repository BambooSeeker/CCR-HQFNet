from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "03_results/r10_chronos2_lora_joint_route_v0.1/predictions.csv"
OUT = ROOT / "03_results/r14_extreme_negative_event_20260308_0314_v0.1"
FIG_OUT = ROOT / "04_manuscript_after_gate/word_revision_v1/figures"
START = "2026-03-08"
END = "2026-03-14"

COLORS = {
    "actual": "#24272B",
    "blind": "#2F6FAE",
    "complete": "#C64745",
    "candidate": "#D3A23A",
    "accepted": "#3A7D62",
    "negative": "#B9BDC3",
    "grid": "#D9DDE3",
}


def point_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    negative = frame["actual"] < 0
    predicted_negative = frame["predicted"] < 0
    true_positive = negative & predicted_negative
    return {
        "targets": int(len(frame)),
        "mae": float(frame["absolute_error"].mean()),
        "negative_targets": int(negative.sum()),
        "negative_mae": float(frame.loc[negative, "absolute_error"].mean()),
        "negative_recall_pct": float(
            100 * true_positive.sum() / max(1, negative.sum())
        ),
        "negative_precision_pct": float(
            100 * true_positive.sum() / max(1, predicted_negative.sum())
        ),
        "false_negative_price_predictions": int((predicted_negative & ~negative).sum()),
    }


def main() -> None:
    route = pd.read_csv(SOURCE)
    event = route[route["delivery_day"].between(START, END)].copy()
    event["time"] = pd.to_datetime(event["time"])
    event = event.sort_values(["mode", "time"])

    modes = {
        "strict_blind": "carrier_blind",
        "supply_only": "energy_consistent_supply_route",
        "complete": "energy_consistent_dual_route",
        "ungated": "ungated_dual_route",
    }
    summary = {
        name: point_summary(event[event["mode"].eq(mode)])
        for name, mode in modes.items()
    }

    complete = event[event["mode"].eq(modes["complete"])].copy()
    blind = event[event["mode"].eq(modes["strict_blind"])].copy()
    supply = event[event["mode"].eq(modes["supply_only"])].copy()
    actual_negative = complete["actual"] < 0
    candidate = complete["candidate_activation"].gt(0)
    accepted = complete["energy_consistency_selected"].astype(bool)
    summary["activation"] = {
        "candidate_points": int(candidate.sum()),
        "accepted_points": int(accepted.sum()),
        "candidate_negative_recall_pct": float(
            100 * (candidate & actual_negative).sum() / actual_negative.sum()
        ),
        "candidate_precision_pct": float(
            100 * (candidate & actual_negative).sum() / candidate.sum()
        ),
        "accepted_negative_recall_pct": float(
            100 * (accepted & actual_negative).sum() / actual_negative.sum()
        ),
        "accepted_precision_pct": float(
            100 * (accepted & actual_negative).sum() / accepted.sum()
        ),
    }
    summary["effect"] = {
        "mae_reduction_vs_blind": float(
            summary["strict_blind"]["mae"] - summary["complete"]["mae"]
        ),
        "mae_reduction_vs_blind_pct": float(
            100
            * (summary["strict_blind"]["mae"] - summary["complete"]["mae"])
            / summary["strict_blind"]["mae"]
        ),
        "negative_mae_reduction_vs_blind": float(
            summary["strict_blind"]["negative_mae"]
            - summary["complete"]["negative_mae"]
        ),
        "negative_mae_reduction_vs_blind_pct": float(
            100
            * (
                summary["strict_blind"]["negative_mae"]
                - summary["complete"]["negative_mae"]
            )
            / summary["strict_blind"]["negative_mae"]
        ),
        "congestion_incremental_mae": float(
            summary["supply_only"]["mae"] - summary["complete"]["mae"]
        ),
        "congestion_incremental_negative_mae": float(
            summary["supply_only"]["negative_mae"]
            - summary["complete"]["negative_mae"]
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    event.to_csv(OUT / "event_predictions.csv", index=False)
    (OUT / "event_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.4,
            "legend.fontsize": 8.7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    actual = complete.sort_values("time").reset_index(drop=True)
    blind = blind.sort_values("time").reset_index(drop=True)
    supply = supply.sort_values("time").reset_index(drop=True)
    x = np.arange(1, len(actual) + 1)
    accepted_activation = np.where(
        actual["energy_consistency_selected"].astype(bool),
        actual["candidate_activation"],
        0.0,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 5.7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.18},
    )
    ax = axes[0]
    ax.plot(x, actual["actual"], color=COLORS["actual"], lw=1.35, label="Actual")
    ax.plot(
        x,
        blind["predicted"],
        color=COLORS["blind"],
        lw=1.05,
        linestyle="--",
        label="Strict Blind",
    )
    ax.plot(
        x,
        actual["predicted"],
        color=COLORS["complete"],
        lw=1.2,
        label="Complete route",
    )
    ax.axhline(0, color="#777777", lw=0.65)
    ax.set_ylabel("RTLMP (CNY/MWh)")
    ax.set_title(
        "a  Seven-day continuous extreme-negative-price forecast",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.fill_between(
        x,
        0,
        actual_negative.astype(float),
        step="mid",
        color=COLORS["negative"],
        alpha=0.42,
        label="Observed negative-price interval",
    )
    ax.plot(
        x,
        actual["candidate_activation"],
        color=COLORS["candidate"],
        lw=1.05,
        linestyle="--",
        label="Candidate tail activation",
    )
    ax.plot(
        x,
        accepted_activation,
        color=COLORS["accepted"],
        lw=1.2,
        label="Risk-controlled activation",
    )
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(1, 336)
    ax.set_xticks([1, 48, 96, 144, 192, 240, 288, 336])
    ax.set_ylabel("Activation")
    ax.set_xlabel("Half-hourly delivery interval")
    ax.set_title(
        "b  Negative-price routing and selective-risk control",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_OUT / "fig_extreme_negative_route_20260308_0314.png", dpi=420)
    fig.savefig(FIG_OUT / "fig_extreme_negative_route_20260308_0314.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
