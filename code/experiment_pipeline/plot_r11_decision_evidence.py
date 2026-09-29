from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "03_results/r11_lhfdc01_probabilistic_commitment_v0.1"
LAMBDA005 = ROOT / "03_results/r11_lhfdc01_probabilistic_commitment_lambda005_v0.1"
STARTUP05 = ROOT / "03_results/r11_lhfdc01_probabilistic_commitment_startup05_v0.1"
STARTUP20 = ROOT / "03_results/r11_lhfdc01_probabilistic_commitment_startup20_v0.1"
ROUTE = ROOT / "03_results/r10_chronos2_lora_joint_route_v0.1/predictions.csv"
CACHE = ROOT / "02_experiments/data_locked/r10_chronos2_lora_fold_cache.csv"
EVENT_DAY = pd.Timestamp("2026-02-17")


def bootstrap_interval(difference: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260722)
    samples = rng.choice(
        difference, size=(10000, len(difference)), replace=True
    ).mean(axis=1)
    return (
        float(difference.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def paired_table(daily: pd.DataFrame) -> pd.DataFrame:
    wide = daily.pivot(
        index=["fold_name", "delivery_day"], columns="policy", values="realized_profit"
    )
    comparisons = [
        ("Full point", "Chronos-2 point"),
        ("Full point", "Day-ahead energy"),
        ("Full probabilistic", "Chronos-2 point"),
        ("Full probabilistic", "Full point"),
        ("Full probabilistic", "Day-ahead energy"),
    ]
    rows = []
    for candidate, baseline in comparisons:
        difference = (wide[candidate] - wide[baseline]).to_numpy(float)
        mean, lower, upper = bootstrap_interval(difference)
        baseline_mean = float(wide[baseline].mean())
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "mean_difference_rmb_per_day": mean,
                "relative_difference_percent": 100.0 * mean / abs(baseline_mean),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "sample_total_difference_rmb": float(difference.sum()),
                "mechanical_annualized_difference_rmb": 365.0 * mean,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    daily = pd.read_csv(BASE / "daily_results.csv")
    schedules = pd.read_csv(BASE / "schedules.csv")
    summary = pd.read_csv(BASE / "summary.csv")
    summary005 = pd.read_csv(LAMBDA005 / "summary.csv")
    startup_summaries = {
        0.5: pd.read_csv(STARTUP05 / "summary.csv"),
        1.0: summary,
        2.0: pd.read_csv(STARTUP20 / "summary.csv"),
    }
    route = pd.read_csv(ROUTE)
    cache = pd.read_csv(CACHE)
    offer = pd.read_csv(BASE / "ten_segment_offer.csv")
    daily["delivery_day"] = pd.to_datetime(daily["delivery_day"])
    schedules["delivery_day"] = pd.to_datetime(schedules["delivery_day"])
    schedules["time"] = pd.to_datetime(schedules["time"])
    route["delivery_day"] = pd.to_datetime(route["delivery_day"])
    route["time"] = pd.to_datetime(route["time"])
    cache["delivery_day"] = pd.to_datetime(cache["delivery_day"])
    cache["time"] = pd.to_datetime(cache["time"])

    paired = paired_table(daily)
    paired.to_csv(BASE / "paired_profit_difference_ci95.csv", index=False)

    fold_name = "formal_winter_event_2026"
    event_route = route[
        route["fold_name"].eq(fold_name)
        & route["delivery_day"].eq(EVENT_DAY)
        & route["mode"].isin(["chronos2_lora", "energy_consistent_dual_route"])
    ]
    route_wide = event_route.pivot(
        index=["time", "actual"], columns="mode", values="predicted"
    ).reset_index()
    event_cache = cache[
        cache["fold_name"].eq(fold_name) & cache["delivery_day"].eq(EVENT_DAY)
    ][["time", "timesfm_q10", "timesfm_q50", "timesfm_q90"]]
    event = route_wide.merge(event_cache, on="time", validate="one_to_one")
    scale = float(
        daily[
            daily["fold_name"].eq(fold_name) & daily["delivery_day"].eq(EVENT_DAY)
        ]["calibration_scale"].iloc[0]
    )
    center = event["timesfm_q50"].to_numpy(float)
    full = event["energy_consistent_dual_route"].to_numpy(float)
    event["calibrated_q10"] = full + scale * (
        event["timesfm_q10"].to_numpy(float) - center
    )
    event["calibrated_q90"] = full + scale * (
        event["timesfm_q90"].to_numpy(float) - center
    )
    event_schedule = schedules[
        schedules["fold_name"].eq(fold_name)
        & schedules["delivery_day"].eq(EVENT_DAY)
        & schedules["policy"].isin(
            ["Chronos-2 point", "Full point", "Full probabilistic"]
        )
    ]
    event_schedule = event_schedule.copy()
    event_schedule["display_policy"] = event_schedule["policy"].replace(
        {"Full probabilistic": "Full probabilistic (lambda=0.20)"}
    )

    risk_rows = []
    for label, source, policy in [
        ("Day-ahead", summary, "Day-ahead energy"),
        ("Chronos-2", summary, "Chronos-2 point"),
        ("Full point", summary, "Full point"),
        ("Full prob. lambda=0.05", summary005, "Full probabilistic"),
        ("Full prob. lambda=0.20", summary, "Full probabilistic"),
    ]:
        row = source[source["policy"].eq(policy)].iloc[0]
        risk_rows.append(
            {
                "label": label,
                "mean_profit": row["mean_daily_profit"],
                "lower_cvar": row["profit_lower_cvar10"],
                "negative_output": row["negative_price_output_mwh"],
            }
        )
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(BASE / "risk_return_points.csv", index=False)

    sensitivity_rows = []
    for multiplier, source in startup_summaries.items():
        values = source.set_index("policy")
        point_gain = (
            values.loc["Full point", "mean_daily_profit"]
            - values.loc["Chronos-2 point", "mean_daily_profit"]
        )
        probability_gain = (
            values.loc["Full probabilistic", "mean_daily_profit"]
            - values.loc["Chronos-2 point", "mean_daily_profit"]
        )
        cvar_gain = (
            values.loc["Full probabilistic", "profit_lower_cvar10"]
            - values.loc["Chronos-2 point", "profit_lower_cvar10"]
        )
        negative_reduction = 1.0 - (
            values.loc["Full probabilistic", "negative_price_output_mwh"]
            / values.loc["Chronos-2 point", "negative_price_output_mwh"]
        )
        sensitivity_rows.append(
            {
                "startup_cost_hours_of_no_load_cost": multiplier,
                "full_point_vs_chronos_profit_gain_rmb_per_day": point_gain,
                "full_point_vs_chronos_profit_gain_percent": 100.0
                * point_gain
                / values.loc["Chronos-2 point", "mean_daily_profit"],
                "full_probability_vs_chronos_profit_gain_rmb_per_day": probability_gain,
                "full_probability_vs_chronos_cvar10_gain_rmb_per_day": cvar_gain,
                "full_probability_vs_chronos_negative_output_reduction_percent": 100.0
                * negative_reduction,
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(BASE / "startup_cost_sensitivity.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 12.5), constrained_layout=True)
    x = np.arange(48) / 2.0
    axes[0].fill_between(
        x,
        event["calibrated_q10"],
        event["calibrated_q90"],
        color="#7C3AED",
        alpha=0.18,
        label="Validation-calibrated q10-q90",
    )
    axes[0].plot(x, event["actual"], color="#111827", linewidth=2.0, label="Actual RT price")
    axes[0].plot(x, event["chronos2_lora"], color="#2563EB", linewidth=1.5, label="Chronos-2")
    axes[0].plot(x, event["energy_consistent_dual_route"], color="#047857", linewidth=1.8, label="Full point")
    axes[0].axhline(0.0, color="#9CA3AF", linewidth=0.8)
    axes[0].set_ylabel("Price (RMB/MWh)")
    axes[0].set_title("A. First decision-disagreement day in the locked 2026 Spring Festival window")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.7)

    colors = {
        "Chronos-2 point": "#2563EB",
        "Full point": "#047857",
        "Full probabilistic": "#7C3AED",
    }
    for policy, part in event_schedule.groupby("policy", sort=False):
        part = part.sort_values("time")
        axes[1].step(
            x,
            part["output_mw"],
            where="mid",
            color=colors[policy],
            linewidth=1.8,
            label=part["display_policy"].iloc[0],
        )
    axes[1].set_ylim(-20, 1080)
    axes[1].set_ylabel("Committed output (MW)")
    axes[1].set_title("B. Non-anticipative thermal-unit decisions")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.7)

    scatter = axes[2].scatter(
        risk["mean_profit"] / 1e6,
        risk["lower_cvar"] / 1e6,
        s=80 + 220 * risk["negative_output"] / risk["negative_output"].max(),
        c=["#64748B", "#2563EB", "#047857", "#A16207", "#7C3AED"],
        edgecolor="white",
        linewidth=0.9,
    )
    del scatter
    for _, row in risk.iterrows():
        axes[2].annotate(
            row["label"],
            (row["mean_profit"] / 1e6, row["lower_cvar"] / 1e6),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[2].set_xlabel("Mean realized profit (million RMB/day)")
    axes[2].set_ylabel("Lower-tail CVaR10 (million RMB/day; higher is better)")
    axes[2].set_title("C. Profit-risk trade-off; marker size denotes negative-price generation")
    axes[2].grid(color="#E5E7EB", linewidth=0.7)
    for axis in axes[:2]:
        axis.set_xlim(0, 23.5)
        axis.set_xlabel("Hour of day")
    fig.savefig(BASE / "decision_evidence.png", dpi=220)
    plt.close(fig)

    bound_violations = 0
    state_violations = 0
    minimum_time_violations = 0
    maximum_online_ramp_up = 0.0
    maximum_online_ramp_down = 0.0
    for _, part in schedules.groupby(["fold_name", "delivery_day", "policy"]):
        part = part.sort_values("time")
        output = part["output_mw"].to_numpy(float)
        commitment = part["commitment"].to_numpy(int)
        bound_violations += int(
            np.any(
                (output < -1e-6)
                | (output > 1030.0 + 1e-6)
                | ((commitment == 1) & (output < 400.0 - 1e-6))
                | ((commitment == 0) & (output > 1e-6))
            )
        )
        state_violations += int(np.any((commitment == 1) != (output > 1e-6)))
        previous_output = 400.0
        previous_commitment = 1
        for time in range(48):
            if previous_commitment == 1 and commitment[time] == 1:
                maximum_online_ramp_up = max(
                    maximum_online_ramp_up, output[time] - previous_output
                )
                maximum_online_ramp_down = max(
                    maximum_online_ramp_down, previous_output - output[time]
                )
            previous_output = output[time]
            previous_commitment = commitment[time]
        starts = np.flatnonzero(np.r_[True, commitment[1:] != commitment[:-1]])
        ends = np.r_[starts[1:], len(commitment)]
        for start, end in zip(starts, ends):
            if start > 0 and end < len(commitment) and end - start < 2:
                minimum_time_violations += 1

    checks = {
        "segments": int(len(offer)),
        "first_endpoint_mw": float(offer["endpoint_mw"].iloc[0]),
        "last_endpoint_mw": float(offer["endpoint_mw"].iloc[-1]),
        "quantity_endpoints_strictly_increasing": bool(
            np.all(np.diff(offer["endpoint_mw"]) > 0.0)
        ),
        "offer_prices_monotone_non_decreasing": bool(
            np.all(np.diff(offer["offer_price_rmb_per_mwh"]) >= -1e-9)
        ),
        "offer_prices_within_configured_bounds": bool(
            offer["offer_price_rmb_per_mwh"].between(-200.0, 800.0).all()
        ),
        "test_days": int(daily["delivery_day"].nunique()),
        "schedule_groups": int(
            schedules.groupby(["fold_name", "delivery_day", "policy"]).ngroups
        ),
        "output_or_commitment_bound_violations": bound_violations,
        "state_consistency_violations": state_violations,
        "internal_minimum_up_down_time_violations": minimum_time_violations,
        "maximum_online_ramp_up_mw_per_half_hour": maximum_online_ramp_up,
        "maximum_online_ramp_down_mw_per_half_hour": maximum_online_ramp_down,
        "realized_price_used_only_for_settlement": True,
        "interval_scale_source": "fold validation only",
    }
    (BASE / "compliance_audit.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(paired.to_string(index=False))
    print("\nStartup-cost sensitivity:\n" + sensitivity.to_string(index=False))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
