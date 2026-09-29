from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[2]
T = 48
DT = 0.5


def solve_schedule(
    scenarios: np.ndarray,
    weights: np.ndarray,
    power_mw: float,
    energy_mwh: float,
    eta_round_trip: float,
    soc_min_fraction: float,
    soc_max_fraction: float,
    degradation_cost: float,
    risk_aversion: float,
    cvar_alpha: float,
) -> dict[str, np.ndarray | float]:
    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim == 1:
        scenarios = scenarios[None, :]
    if scenarios.shape[1] != T:
        raise ValueError(f"Expected {T} prices per scenario, got {scenarios.shape}.")
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n_scenarios = len(scenarios)
    eta = float(np.sqrt(eta_round_trip))
    soc_initial = 0.5 * energy_mwh
    soc_min = soc_min_fraction * energy_mwh
    soc_max = soc_max_fraction * energy_mwh

    c_slice = slice(0, T)
    d_slice = slice(T, 2 * T)
    e_slice = slice(2 * T, 3 * T + 1)
    u_slice = slice(3 * T + 1, 4 * T + 1)
    has_risk = risk_aversion > 0.0 and n_scenarios > 1
    zeta_index = 4 * T + 1 if has_risk else None
    xi_slice = (
        slice(zeta_index + 1, zeta_index + 1 + n_scenarios)
        if has_risk and zeta_index is not None
        else None
    )
    n_variables = 4 * T + 1 + (1 + n_scenarios if has_risk else 0)

    expected_price = weights @ scenarios
    objective = np.zeros(n_variables, dtype=float)
    objective[c_slice] = DT * (expected_price + degradation_cost)
    objective[d_slice] = -DT * (expected_price - degradation_cost)
    if has_risk and zeta_index is not None and xi_slice is not None:
        objective[zeta_index] = risk_aversion
        objective[xi_slice] = risk_aversion * weights / (1.0 - cvar_alpha)

    lower = np.full(n_variables, -np.inf)
    upper = np.full(n_variables, np.inf)
    lower[c_slice], upper[c_slice] = 0.0, power_mw
    lower[d_slice], upper[d_slice] = 0.0, power_mw
    lower[e_slice], upper[e_slice] = soc_min, soc_max
    lower[u_slice], upper[u_slice] = 0.0, 1.0
    if has_risk and xi_slice is not None:
        lower[xi_slice] = 0.0
    integrality = np.zeros(n_variables, dtype=int)
    integrality[u_slice] = 1

    equality = lil_matrix((T + 2, n_variables), dtype=float)
    equality_rhs = np.zeros(T + 2, dtype=float)
    equality[0, e_slice.start] = 1.0
    equality_rhs[0] = soc_initial
    for step in range(T):
        row = step + 1
        equality[row, e_slice.start + step + 1] = 1.0
        equality[row, e_slice.start + step] = -1.0
        equality[row, c_slice.start + step] = -DT * eta
        equality[row, d_slice.start + step] = DT / eta
    equality[-1, e_slice.stop - 1] = 1.0
    equality_rhs[-1] = soc_initial

    n_inequalities = 2 * T + (n_scenarios if has_risk else 0)
    inequality = lil_matrix((n_inequalities, n_variables), dtype=float)
    inequality_upper = np.zeros(n_inequalities, dtype=float)
    for step in range(T):
        inequality[step, c_slice.start + step] = 1.0
        inequality[step, u_slice.start + step] = -power_mw
        inequality[T + step, d_slice.start + step] = 1.0
        inequality[T + step, u_slice.start + step] = power_mw
        inequality_upper[T + step] = power_mw
    if has_risk and zeta_index is not None and xi_slice is not None:
        for scenario_index, prices in enumerate(scenarios):
            row = 2 * T + scenario_index
            inequality[row, c_slice] = DT * (prices + degradation_cost)
            inequality[row, d_slice] = -DT * (prices - degradation_cost)
            inequality[row, zeta_index] = -1.0
            inequality[row, xi_slice.start + scenario_index] = -1.0

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=[
            LinearConstraint(equality.tocsr(), equality_rhs, equality_rhs),
            LinearConstraint(
                inequality.tocsr(),
                np.full(n_inequalities, -np.inf),
                inequality_upper,
            ),
        ],
        options={"time_limit": 20.0, "mip_rel_gap": 1e-8},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Storage MILP failed: {result.message}")
    charge = result.x[c_slice]
    discharge = result.x[d_slice]
    soc = result.x[e_slice]
    scenario_profit = np.array(
        [
            DT * np.sum(prices * (discharge - charge))
            - DT * degradation_cost * np.sum(charge + discharge)
            for prices in scenarios
        ]
    )
    return {
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "scenario_profit": scenario_profit,
        "objective": float(result.fun),
    }


def realized_profit(
    prices: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    degradation_cost: float,
) -> float:
    return float(
        DT * np.sum(prices * (discharge - charge))
        - DT * degradation_cost * np.sum(charge + discharge)
    )


def validation_spread_scale(
    cache: pd.DataFrame,
    frame: pd.DataFrame,
    fold: dict[str, Any],
) -> float:
    val_days = pd.to_datetime(fold["val"])
    selected = cache[
        cache["fold_name"].eq(fold["name"])
        & cache["delivery_day"].isin(val_days)
    ].merge(frame[["time", "price_real"]], on="time", how="left", validate="one_to_one")
    center = selected["timesfm_q50"].to_numpy(float)
    lower_width = np.maximum(center - selected["timesfm_q10"].to_numpy(float), 1e-6)
    upper_width = np.maximum(selected["timesfm_q90"].to_numpy(float) - center, 1e-6)
    actual = selected["price_real"].to_numpy(float)
    score = np.maximum((center - actual) / lower_width, (actual - center) / upper_width)
    return float(max(1.0, np.quantile(score, 0.80, method="higher")))


def lower_tail_cvar(values: np.ndarray, probability: float = 0.10) -> float:
    count = max(1, int(np.ceil(len(values) * probability)))
    return float(np.mean(np.sort(values)[:count]))


def plot_results(summary: pd.DataFrame, seasonal: pd.DataFrame, output: Path) -> None:
    order = [
        "Day-ahead energy",
        "Chronos-2 point",
        "Full point",
        "Full risk-aware",
        "Route-adjusted DA point",
        "Route-adjusted DA risk",
        "Oracle",
    ]
    summary = summary.set_index("policy").loc[order].reset_index()
    colors = ["#64748B", "#2563EB", "#047857", "#7C3AED", "#0F766E", "#A16207", "#111827"]
    fig, (ax_profit, ax_season) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.0, 1.55]},
        constrained_layout=True,
    )
    ax_profit.barh(summary["policy"], summary["mean_daily_profit"], color=colors)
    ax_profit.invert_yaxis()
    ax_profit.set_xlabel("Mean realized profit (RMB per MW-day)")
    ax_profit.grid(axis="x", color="#E5E7EB", linewidth=0.7)

    pivot = seasonal.pivot(index="fold_name", columns="policy", values="mean_daily_profit")
    fold_order = seasonal.drop_duplicates("fold").sort_values("fold")["fold_name"].tolist()
    pivot = pivot.loc[fold_order]
    x = np.arange(len(fold_order))
    width = 0.18
    for index, policy in enumerate(order[:-1]):
        ax_season.bar(
            x + (index - 1.5) * width,
            pivot[policy].to_numpy(float),
            width,
            label=policy,
            color=colors[index],
        )
    ax_season.set_xticks(x)
    ax_season.set_xticklabels(
        ["2024 trial\nlate summer", "2025 revised\nspring", "2025 revised\nsummer", "2026 formal\nwinter", "2026 formal\nspring"]
    )
    ax_season.set_ylabel("Mean realized profit (RMB per MW-day)")
    ax_season.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax_season.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route-predictions", type=Path,
        default=ROOT / "03_results/r10_chronos2_lora_joint_route_v0.1/predictions.csv",
    )
    parser.add_argument(
        "--carrier-cache", type=Path,
        default=ROOT / "02_experiments/data_locked/r10_chronos2_lora_fold_cache.csv",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "02_experiments/data_locked/lhfdc01_forecast_available_202405_202607.csv",
    )
    parser.add_argument(
        "--folds", type=Path, default=ROOT / "00_control/r10_seasonal_long_windows.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "03_results/r11_storage_arbitrage_v0.1"
    )
    parser.add_argument("--power-mw", type=float, default=1.0)
    parser.add_argument("--energy-mwh", type=float, default=2.0)
    parser.add_argument("--eta-round-trip", type=float, default=0.90)
    parser.add_argument("--soc-min-fraction", type=float, default=0.10)
    parser.add_argument("--soc-max-fraction", type=float, default=0.90)
    parser.add_argument("--degradation-cost", type=float, default=10.0)
    parser.add_argument("--risk-aversion", type=float, default=0.20)
    parser.add_argument("--cvar-alpha", type=float, default=0.80)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    route = pd.read_csv(args.route_predictions)
    route["time"] = pd.to_datetime(route["time"])
    route["delivery_day"] = pd.to_datetime(route["delivery_day"]).dt.normalize()
    cache = pd.read_csv(args.carrier_cache)
    cache["time"] = pd.to_datetime(cache["time"])
    cache["delivery_day"] = pd.to_datetime(cache["delivery_day"]).dt.normalize()
    frame = pd.read_csv(args.data)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    frame["price_day_ahead_energy_proxy"] = (
        frame["price_day_ahead"] - frame["price_day_ahead_cong"]
    )

    raw = route[route["mode"].eq("chronos2_lora")][
        ["fold", "fold_name", "delivery_day", "time", "actual", "predicted"]
    ].rename(columns={"predicted": "chronos_point"})
    full = route[route["mode"].eq("energy_consistent_dual_route")][
        ["fold_name", "time", "predicted"]
    ].rename(columns={"predicted": "full_point"})
    test = raw.merge(full, on=["fold_name", "time"], validate="one_to_one")
    test = test.merge(
        cache[
            ["fold_name", "time", "timesfm_q10", "timesfm_q50", "timesfm_q90"]
        ],
        on=["fold_name", "time"],
        validate="one_to_one",
    )
    test = test.merge(
        frame[["time", "price_day_ahead_energy_proxy"]],
        on="time",
        validate="one_to_one",
    )

    scale_by_fold = {
        fold["name"]: validation_spread_scale(cache, frame, fold) for fold in folds
    }
    policy_rows: list[dict[str, Any]] = []
    dispatch_rows: list[pd.DataFrame] = []
    for (fold_name, day), daily in test.groupby(["fold_name", "delivery_day"], sort=True):
        daily = daily.sort_values("time")
        if len(daily) != T:
            raise ValueError(f"{fold_name}/{day.date()} has {len(daily)} points.")
        actual = daily["actual"].to_numpy(float)
        full_point = daily["full_point"].to_numpy(float)
        day_ahead = daily["price_day_ahead_energy_proxy"].to_numpy(float)
        chronos_point = daily["chronos_point"].to_numpy(float)
        route_adjusted_day_ahead = day_ahead + (full_point - chronos_point)
        center = daily["timesfm_q50"].to_numpy(float)
        scale = scale_by_fold[fold_name]
        scenarios = np.stack(
            [
                full_point + scale * (daily["timesfm_q10"].to_numpy(float) - center),
                full_point,
                full_point + scale * (daily["timesfm_q90"].to_numpy(float) - center),
            ]
        )
        adjusted_scenarios = np.stack(
            [
                route_adjusted_day_ahead
                + scale * (daily["timesfm_q10"].to_numpy(float) - center),
                route_adjusted_day_ahead,
                route_adjusted_day_ahead
                + scale * (daily["timesfm_q90"].to_numpy(float) - center),
            ]
        )
        definitions = {
            "Day-ahead energy": (
                day_ahead[None, :],
                np.array([1.0]),
                0.0,
            ),
            "Chronos-2 point": (
                chronos_point[None, :], np.array([1.0]), 0.0
            ),
            "Full point": (full_point[None, :], np.array([1.0]), 0.0),
            "Full risk-aware": (scenarios, np.array([0.25, 0.50, 0.25]), args.risk_aversion),
            "Route-adjusted DA point": (
                route_adjusted_day_ahead[None, :], np.array([1.0]), 0.0
            ),
            "Route-adjusted DA risk": (
                adjusted_scenarios, np.array([0.25, 0.50, 0.25]), args.risk_aversion
            ),
            "Oracle": (actual[None, :], np.array([1.0]), 0.0),
        }
        for policy, (price_scenarios, weights, risk_aversion) in definitions.items():
            solution = solve_schedule(
                price_scenarios,
                weights,
                args.power_mw,
                args.energy_mwh,
                args.eta_round_trip,
                args.soc_min_fraction,
                args.soc_max_fraction,
                args.degradation_cost,
                risk_aversion,
                args.cvar_alpha,
            )
            charge = np.asarray(solution["charge"])
            discharge = np.asarray(solution["discharge"])
            profit = realized_profit(
                actual, charge, discharge, args.degradation_cost
            )
            negative = actual < 0.0
            policy_rows.append(
                {
                    "fold": int(daily["fold"].iloc[0]),
                    "fold_name": fold_name,
                    "delivery_day": day,
                    "policy": policy,
                    "realized_profit": profit,
                    "negative_price_count": int(negative.sum()),
                    "negative_charge_count": int(np.sum(negative & (charge > 0.05))),
                    "throughput_mwh": float(DT * np.sum(charge + discharge)),
                    "calibration_scale": scale,
                }
            )
            dispatch_rows.append(
                pd.DataFrame(
                    {
                        "fold_name": fold_name,
                        "delivery_day": day,
                        "time": daily["time"].to_numpy(),
                        "policy": policy,
                        "actual_price": actual,
                        "charge_mw": charge,
                        "discharge_mw": discharge,
                        "soc_mwh_end": np.asarray(solution["soc"])[1:],
                    }
                )
            )

    daily_results = pd.DataFrame(policy_rows)
    oracle = daily_results[daily_results["policy"].eq("Oracle")][
        ["fold_name", "delivery_day", "realized_profit"]
    ].rename(columns={"realized_profit": "oracle_profit"})
    daily_results = daily_results.merge(
        oracle, on=["fold_name", "delivery_day"], validate="many_to_one"
    )
    daily_results["regret"] = daily_results["oracle_profit"] - daily_results["realized_profit"]
    summary_rows = []
    for policy, part in daily_results.groupby("policy", sort=False):
        profits = part["realized_profit"].to_numpy(float)
        negative_count = int(part["negative_price_count"].sum())
        summary_rows.append(
            {
                "policy": policy,
                "mean_daily_profit": float(profits.mean()),
                "total_profit": float(profits.sum()),
                "profit_std": float(profits.std(ddof=1)),
                "profit_lower_cvar10": lower_tail_cvar(profits),
                "mean_regret": float(part["regret"].mean()),
                "negative_price_charge_recall": (
                    float(part["negative_charge_count"].sum() / negative_count)
                    if negative_count else np.nan
                ),
                "mean_throughput_mwh": float(part["throughput_mwh"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    seasonal = (
        daily_results.groupby(["fold", "fold_name", "policy"], as_index=False)
        .agg(
            mean_daily_profit=("realized_profit", "mean"),
            mean_regret=("regret", "mean"),
        )
    )
    daily_results.to_csv(args.output / "daily_results.csv", index=False)
    pd.concat(dispatch_rows, ignore_index=True).to_csv(
        args.output / "dispatch.csv", index=False
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    seasonal.to_csv(args.output / "seasonal_summary.csv", index=False)
    plot_results(summary, seasonal, args.output / "storage_arbitrage_summary.png")
    report = {
        "status": "complete",
        "decision_timing": "All schedules fixed day-ahead; realized RT prices used only for settlement.",
        "storage": {
            "power_mw": args.power_mw,
            "energy_mwh": args.energy_mwh,
            "duration_hours": args.energy_mwh / args.power_mw,
            "round_trip_efficiency": args.eta_round_trip,
            "soc_min_fraction": args.soc_min_fraction,
            "soc_max_fraction": args.soc_max_fraction,
            "initial_and_terminal_soc_fraction": 0.50,
            "degradation_cost_rmb_per_mwh_throughput": args.degradation_cost,
            "simultaneous_charge_discharge": "forbidden by binary mode",
        },
        "risk": {
            "scenario_weights": [0.25, 0.50, 0.25],
            "cvar_alpha": args.cvar_alpha,
            "risk_aversion": args.risk_aversion,
            "interval_calibration": "fold validation-only multiplicative conformal spread",
            "scale_by_fold": scale_by_fold,
        },
        "test_days": int(daily_results["delivery_day"].nunique()),
        "normalization": "Profit is reported per 1 MW / 2 MWh storage unit.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.sort_values("mean_daily_profit", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
