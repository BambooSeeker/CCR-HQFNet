from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_chronos2_strong_carrier_audit import fit_residual
from run_physical_soft_route_price_audit import add_features, day_mask, error_metrics, fit_router
from run_risk_controlled_negative_route import (
    activation_from_score,
    apply_risk_route,
    fused_negative_score,
    zero_price_probability,
)
from run_supply_congestion_dual_route_audit import (
    add_supply_features,
    classification_metrics,
    fit_low_nonnegative_expert,
    fit_supply_router,
    supply_labels,
)


RISK_BUDGET = 0.33
RISK_Z_90 = 1.281552
OOF_DAYS = 45
OOF_BLOCK_DAYS = 15
SIMILAR_DAYS = 14
SUPPLY_WEIGHT = 0.90
ACTIVATION_THRESHOLD = 0.45
BETA_NEGATIVE = 1.0
QUANTILE_SCALE = 1.0
BETA_LOW = 0.50

SIGNATURE_SOURCES = (
    "price_day_ahead_energy_proxy",
    "price_day_ahead_load",
    "load_day_ahead_pred",
    "elec_gene_total_pred",
    "market_bid_space_proxy",
    "supply_adequacy_proxy",
    "renewable_supply_index",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def add_nonnegative_mae(actual: np.ndarray, predicted: np.ndarray, metrics: dict) -> dict:
    mask = actual >= 0.0
    metrics["nonnegative_price_mae"] = (
        float(np.mean(np.abs(predicted[mask] - actual[mask]))) if mask.any() else None
    )
    return metrics


def frame_arrays(frame: pd.DataFrame, manifest: dict, carrier_column: str) -> dict:
    congestion_columns = manifest["congestion_features"]
    blind_columns = [
        name
        for name in manifest["features"]
        if name not in congestion_columns and name != "price_day_ahead"
    ]
    blind_columns += [
        "slot_sin",
        "slot_cos",
        "slot_norm",
        "price_day_ahead_energy_proxy",
        carrier_column,
    ]
    full_columns = blind_columns + congestion_columns
    supply_columns = [
        "price_day_ahead_energy_proxy",
        "price_day_ahead_load",
        "load_day_ahead_pred",
        "elec_exter_plan",
        "elec_fix_out_plan",
        "energy_hydro_renewable",
        "elec_gene_total_pred",
        "elec_day_ahead_all",
        "market_bid_space_proxy",
        "supply_adequacy_proxy",
        "external_plan_load_ratio",
        "fixed_output_load_ratio",
        "generation_load_ratio",
        "renewable_supply_index",
        "price_real_safe48_mean",
        "price_real_safe48_std",
        "price_real_safe48_min",
        "price_real_safe48_max",
        "slot_sin",
        "slot_cos",
        "slot_norm",
        "market_phase_trial_2024",
        "market_phase_revised_2025",
        "market_phase_formal",
        carrier_column,
    ]
    return {
        "x_blind": frame[blind_columns].to_numpy(float),
        "x_full": frame[full_columns].to_numpy(float),
        "x_supply": frame[supply_columns].to_numpy(float),
    }


def daily_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for day, daily in frame.groupby("delivery_day", sort=True):
        row: dict[str, float | pd.Timestamp] = {"delivery_day": day}
        for source in SIGNATURE_SOURCES:
            values = daily[source].to_numpy(float)
            row[f"{source}_mean"] = float(np.mean(values))
            row[f"{source}_std"] = float(np.std(values))
            row[f"{source}_min"] = float(np.min(values))
            row[f"{source}_max"] = float(np.max(values))
            row[f"{source}_q10"] = float(np.quantile(values, 0.10))
            row[f"{source}_q90"] = float(np.quantile(values, 0.90))
        energy = daily["price_day_ahead_energy_proxy"].to_numpy(float)
        row["energy_mean_abs_ramp"] = float(np.mean(np.abs(np.diff(energy))))
        row["market_phase_trial_2024"] = float(daily["market_phase_trial_2024"].max())
        row["market_phase_revised_2025"] = float(daily["market_phase_revised_2025"].max())
        row["market_phase_formal"] = float(daily["market_phase_formal"].max())
        angle = 2.0 * np.pi * pd.Timestamp(day).dayofyear / 365.25
        row["day_of_year_sin"] = float(np.sin(angle))
        row["day_of_year_cos"] = float(np.cos(angle))
        rows.append(row)
    return pd.DataFrame(rows).set_index("delivery_day").sort_index()


def phase_column(row: pd.Series) -> str:
    columns = (
        "market_phase_trial_2024",
        "market_phase_revised_2025",
        "market_phase_formal",
    )
    return max(columns, key=lambda name: float(row[name]))


def nearest_oof_days(
    signatures: pd.DataFrame,
    oof_days: pd.Index,
    target_day: pd.Timestamp,
) -> tuple[list[pd.Timestamp], dict[str, float]]:
    target = signatures.loc[target_day]
    target_phase = phase_column(target)
    eligible = pd.Index(
        [day for day in oof_days if signatures.loc[day, target_phase] > 0.5]
    )
    if len(eligible) < SIMILAR_DAYS:
        eligible = oof_days
    feature_columns = [
        name
        for name in signatures.columns
        if not name.startswith("market_phase_")
    ]
    reference = signatures.loc[eligible, feature_columns]
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    distances = np.sqrt(
        np.mean(((reference - target[feature_columns]) / scale) ** 2, axis=1)
    ).sort_values()
    selected = list(pd.to_datetime(distances.index[:SIMILAR_DAYS]))
    diagnostics = {
        "nearest_distance": float(distances.iloc[0]),
        "furthest_selected_distance": float(distances.iloc[min(SIMILAR_DAYS, len(distances)) - 1]),
        "eligible_days": int(len(eligible)),
    }
    return selected, diagnostics


def fit_components(
    frame: pd.DataFrame,
    arrays: dict,
    actual: np.ndarray,
    carrier: np.ndarray,
    congestion_state: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    fit_mask: np.ndarray,
    early_mask: np.ndarray,
    predict_mask: np.ndarray,
    low_threshold: float,
    config: dict,
) -> pd.DataFrame:
    residual = actual - carrier
    blind_model = fit_residual(
        arrays["x_blind"][fit_mask],
        residual[fit_mask],
        arrays["x_blind"][early_mask],
        residual[early_mask],
        config,
    )
    full_model = fit_residual(
        arrays["x_full"][fit_mask],
        residual[fit_mask],
        arrays["x_full"][early_mask],
        residual[early_mask],
        config,
    )
    blind = carrier[predict_mask] + blind_model.predict(arrays["x_blind"][predict_mask])
    full = carrier[predict_mask] + full_model.predict(arrays["x_full"][predict_mask])

    labeled_fit = fit_mask & (congestion_state >= 0)
    labeled_early = early_mask & (congestion_state >= 0)
    congestion_router = fit_router(
        arrays["x_full"][labeled_fit],
        congestion_state[labeled_fit],
        arrays["x_full"][labeled_early],
        congestion_state[labeled_early],
        config,
    )
    congestion_probability = congestion_router.predict_proba(
        arrays["x_full"][predict_mask]
    )
    physical = blind + (1.0 - congestion_probability[:, 1]) * (full - blind)

    fit_states = supply_labels(actual[fit_mask], low_threshold)
    early_states = supply_labels(actual[early_mask], low_threshold)
    supply_router = fit_supply_router(
        arrays["x_supply"][fit_mask],
        fit_states,
        arrays["x_supply"][early_mask],
        early_states,
        config,
    )
    supply_probability = supply_router.predict_proba(arrays["x_supply"][predict_mask])

    low_fit = fit_mask & (actual >= 0.0) & (actual <= low_threshold)
    low_early = early_mask & (actual >= 0.0) & (actual <= low_threshold)
    if int(low_fit.sum()) >= 8:
        low_model = fit_low_nonnegative_expert(
            arrays["x_supply"][low_fit],
            actual[low_fit],
            arrays["x_supply"][low_early],
            actual[low_early],
            config,
        )
        low_prediction = low_model.predict(arrays["x_supply"][predict_mask])
    else:
        low_prediction = blind.copy()

    p_timesfm, _, _, _ = zero_price_probability(
        blind, q10[predict_mask], q50[predict_mask], q90[predict_mask]
    )
    score = fused_negative_score(
        supply_probability[:, 0], p_timesfm, SUPPLY_WEIGHT
    )
    activation = activation_from_score(score, ACTIVATION_THRESHOLD)
    supply_route = apply_risk_route(
        blind,
        supply_probability,
        low_prediction,
        q10[predict_mask],
        q50[predict_mask],
        activation,
        BETA_NEGATIVE,
        QUANTILE_SCALE,
        BETA_LOW,
    )
    dual_route = apply_risk_route(
        physical,
        supply_probability,
        low_prediction,
        q10[predict_mask],
        q50[predict_mask],
        activation,
        BETA_NEGATIVE,
        QUANTILE_SCALE,
        BETA_LOW,
    )
    positions = np.flatnonzero(predict_mask)
    return pd.DataFrame(
        {
            "position": positions,
            "delivery_day": frame.loc[predict_mask, "delivery_day"].to_numpy(),
            "time": frame.loc[predict_mask, "time"].to_numpy(),
            "actual": actual[predict_mask],
            "timesfm_zero_shot": carrier[predict_mask],
            "carrier_blind": blind,
            "physical_congestion_gate": physical,
            "ungated_supply_route": supply_route,
            "ungated_dual_route": dual_route,
            "candidate_activation": activation,
            "negative_score": score,
            "price_day_ahead_energy_proxy": frame.loc[
                predict_mask, "price_day_ahead_energy_proxy"
            ].to_numpy(float),
        }
    )


def oof_components(
    frame: pd.DataFrame,
    fold: dict,
    arrays: dict,
    actual: np.ndarray,
    carrier: np.ndarray,
    congestion_state: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    low_threshold: float,
    config: dict,
) -> pd.DataFrame:
    train_days = pd.Index(pd.to_datetime(fold["train"]))
    calibration_days = train_days[-OOF_DAYS:]
    blocks: list[pd.DataFrame] = []
    for start in range(0, OOF_DAYS, OOF_BLOCK_DAYS):
        block_days = calibration_days[start : start + OOF_BLOCK_DAYS]
        prefix = train_days[train_days < block_days[0]]
        if len(prefix) < 30:
            raise ValueError(f"insufficient OOF prefix before {block_days[0]}")
        early_days = prefix[-7:]
        fit_days = prefix[:-7]
        blocks.append(
            fit_components(
                frame,
                arrays,
                actual,
                carrier,
                congestion_state,
                q10,
                q50,
                q90,
                day_mask(frame, date_strings(fit_days)),
                day_mask(frame, date_strings(early_days)),
                day_mask(frame, date_strings(block_days)),
                low_threshold,
                config,
            )
        )
    return pd.concat(blocks, ignore_index=True).sort_values("time")


def date_strings(values: pd.Index) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in values]


def select_energy_threshold(calibration: pd.DataFrame) -> tuple[float | None, dict]:
    actual_negative = calibration["actual"].to_numpy(float) < 0.0
    activation = calibration["candidate_activation"].to_numpy(float) > 0.0
    energy = calibration["price_day_ahead_energy_proxy"].to_numpy(float)
    candidates: list[dict] = []
    for threshold in np.unique(energy[activation]):
        selected = activation & (energy <= threshold)
        count = int(selected.sum())
        true_selected = int(np.sum(selected & actual_negative))
        false_selected = int(np.sum(selected & ~actual_negative))
        risk = false_selected / count if count else 1.0
        standard_error = np.sqrt(max(risk * (1.0 - risk), 0.0) / max(count, 1))
        upper = min(1.0, risk + RISK_Z_90 * standard_error)
        recall = true_selected / int(actual_negative.sum()) if actual_negative.any() else 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "selected_points": count,
                "true_selected": true_selected,
                "false_selected": false_selected,
                "risk": float(risk),
                "risk_90pct_upper": float(upper),
                "recall": float(recall),
                "feasible": bool(count >= 8 and upper <= RISK_BUDGET),
            }
        )
    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        return None, {
            "status": "DISABLED_NO_FEASIBLE_SIMILAR_DAY_POLICY",
            "calibration_points": int(len(calibration)),
            "calibration_negative_points": int(actual_negative.sum()),
            "candidate_active_points": int(activation.sum()),
        }
    selected = min(
        feasible,
        key=lambda row: (-row["recall"], row["risk_90pct_upper"], row["threshold"]),
    )
    return float(selected["threshold"]), {"status": "PASS", **selected}


def mode_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    low_mask: np.ndarray,
    congestion_state: np.ndarray,
) -> dict:
    metrics = error_metrics(actual, predicted, low_mask, congestion_state)
    metrics.update(classification_metrics(actual, predicted))
    return add_nonnegative_mae(actual, predicted, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--carrier-cache", type=Path, required=True)
    parser.add_argument("--carrier-column", default="timesfm_prediction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels, usecols=["time", "congestion_state"])
    carrier_frame = pd.read_csv(args.carrier_cache)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    labels["time"] = pd.to_datetime(labels["time"])
    carrier_frame["time"] = pd.to_datetime(carrier_frame["time"])
    frame = add_supply_features(
        add_features(
            frame.merge(labels, on="time", how="left").merge(
                carrier_frame, on="time", how="left"
            )
        )
    )
    required = {args.carrier_column, "timesfm_q10", "timesfm_q50", "timesfm_q90"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("TimesFM cache is incomplete or misaligned")

    arrays = frame_arrays(frame, manifest, args.carrier_column)
    actual = frame["price_real"].to_numpy(float)
    carrier = frame[args.carrier_column].to_numpy(float)
    q10 = frame["timesfm_q10"].to_numpy(float)
    q50 = frame["timesfm_q50"].to_numpy(float)
    q90 = frame["timesfm_q90"].to_numpy(float)
    congestion_state = frame["congestion_state"].fillna(-1).to_numpy(int)
    signatures = daily_signatures(frame)

    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict] = []
    for fold_index, fold in enumerate(folds, start=1):
        train = day_mask(frame, fold["train"])
        val = day_mask(frame, fold["val"])
        test = day_mask(frame, fold["test"])
        low_threshold = float(np.quantile(actual[train][actual[train] >= 0.0], 0.10))

        oof = oof_components(
            frame,
            fold,
            arrays,
            actual,
            carrier,
            congestion_state,
            q10,
            q50,
            q90,
            low_threshold,
            config,
        )
        recent = fit_components(
            frame,
            arrays,
            actual,
            carrier,
            congestion_state,
            q10,
            q50,
            q90,
            train,
            val,
            val,
            low_threshold,
            config,
        )
        target = fit_components(
            frame,
            arrays,
            actual,
            carrier,
            congestion_state,
            q10,
            q50,
            q90,
            train,
            val,
            test,
            low_threshold,
            config,
        )

        thresholds: dict[pd.Timestamp, float | None] = {}
        oof_days = pd.Index(sorted(oof["delivery_day"].unique()))
        for target_day in pd.Index(pd.to_datetime(fold["test"])):
            similar_days, distance = nearest_oof_days(signatures, oof_days, target_day)
            similar = oof[oof["delivery_day"].isin(similar_days)]
            calibration = pd.concat([similar, recent], ignore_index=True)
            threshold, policy = select_energy_threshold(calibration)
            thresholds[target_day] = threshold
            calibration_rows.append(
                {
                    "fold": fold_index,
                    "fold_name": fold["name"],
                    "target_day": target_day.strftime("%Y-%m-%d"),
                    "selected_similar_days": date_strings(pd.Index(similar_days)),
                    "energy_threshold": threshold,
                    **distance,
                    **policy,
                }
            )

        supply_final = target["ungated_supply_route"].to_numpy(float).copy()
        dual_final = target["ungated_dual_route"].to_numpy(float).copy()
        selected_final = np.zeros(len(target), dtype=bool)
        for target_day, threshold in thresholds.items():
            local = target["delivery_day"].eq(target_day).to_numpy()
            if threshold is None:
                accepted = np.zeros(local.sum(), dtype=bool)
            else:
                accepted = (
                    target.loc[local, "candidate_activation"].to_numpy(float) > 0.0
                ) & (
                    target.loc[local, "price_day_ahead_energy_proxy"].to_numpy(float)
                    <= threshold
                )
            local_positions = np.flatnonzero(local)
            selected_final[local_positions] = accepted
            rejected_positions = local_positions[~accepted]
            supply_final[rejected_positions] = np.maximum(
                target.loc[rejected_positions, "carrier_blind"].to_numpy(float), 0.0
            )
            dual_final[rejected_positions] = np.maximum(
                target.loc[rejected_positions, "physical_congestion_gate"].to_numpy(float),
                0.0,
            )

        modes = {
            "timesfm_zero_shot": target["timesfm_zero_shot"].to_numpy(float),
            "carrier_blind": target["carrier_blind"].to_numpy(float),
            "physical_congestion_gate": target[
                "physical_congestion_gate"
            ].to_numpy(float),
            "ungated_dual_route": target["ungated_dual_route"].to_numpy(float),
            "energy_consistent_supply_route": supply_final,
            "energy_consistent_dual_route": dual_final,
        }
        fold_actual = target["actual"].to_numpy(float)
        fold_state = congestion_state[target["position"].to_numpy(int)]
        low_mask = fold_actual <= low_threshold
        for mode, predicted in modes.items():
            metric_rows.append(
                {
                    "fold": fold_index,
                    "fold_name": fold["name"],
                    "mode": mode,
                    "low_threshold_train_q10": low_threshold,
                    **mode_metrics(fold_actual, predicted, low_mask, fold_state),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_index,
                        "fold_name": fold["name"],
                        "delivery_day": target["delivery_day"].dt.strftime(
                            "%Y-%m-%d"
                        ),
                        "time": target["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": mode,
                        "actual": fold_actual,
                        "predicted": predicted,
                        "absolute_error": np.abs(predicted - fold_actual),
                        "negative_price": fold_actual < 0.0,
                        "candidate_activation": target[
                            "candidate_activation"
                        ].to_numpy(float),
                        "energy_consistency_selected": selected_final,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = predictions.groupby(
        ["fold", "fold_name", "delivery_day", "mode"], as_index=False
    ).agg(mae=("absolute_error", "mean"))
    report = {
        "status": "complete",
        "protocol": {
            "oof_days": OOF_DAYS,
            "oof_block_days": OOF_BLOCK_DAYS,
            "similar_days_per_target": SIMILAR_DAYS,
            "similarity_uses_realized_price": False,
            "recent_validation_days": 7,
            "risk_budget": RISK_BUDGET,
            "risk_confidence_level": 0.90,
            "fixed_r9_route_policy": {
                "supply_weight": SUPPLY_WEIGHT,
                "activation_threshold": ACTIVATION_THRESHOLD,
                "beta_negative": BETA_NEGATIVE,
                "quantile_scale": QUANTILE_SCALE,
                "beta_low": BETA_LOW,
            },
        },
        "input_hashes": {
            "data": sha256(args.data),
            "labels": sha256(args.labels),
            "manifest": sha256(args.manifest),
            "folds": sha256(args.folds),
            "config": sha256(args.config),
            "carrier_cache": sha256(args.carrier_cache),
            "code": sha256(Path(__file__)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_json(
        args.output / "similar_day_calibration.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    (args.output / "r10_similar_day_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
