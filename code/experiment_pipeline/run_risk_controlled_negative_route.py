from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from run_chronos2_strong_carrier_audit import fit_residual, model_params
from run_physical_soft_route_price_audit import add_features, day_mask, error_metrics, fit_router
from run_supply_congestion_dual_route_audit import (
    PRICE_FLOOR,
    SUPPLY_STATE_NAMES,
    add_supply_features,
    apply_supply_route,
    classification_metrics,
    fit_low_nonnegative_expert,
    fit_supply_router,
    select_betas,
    sha256,
    summarize,
    supply_labels,
)


# Pre-registered R9 calibration grid. Test data never enter this selection.
FUSION_SUPPLY_WEIGHTS = (0.65, 0.80, 0.90)
ENERGY_PRIOR_WEIGHTS = (0.65, 0.80, 0.90, 0.95, 0.99)
ACTIVATION_THRESHOLDS = tuple(np.round(np.arange(0.10, 0.96, 0.05), 2))
NEGATIVE_BETAS = (0.25, 0.50, 0.75, 1.00)
LOWER_QUANTILE_SCALES = (0.25, 0.50, 0.75, 1.00)
LOW_BETAS = (0.00, 0.25, 0.50)
STRUCTURED_WINDOWS = (1, 3, 5)
SELECTION_RISK_BUDGET = 0.33
RECENT_SELECTION_RISK_BUDGET = 0.35
NONNEGATIVE_MAE_BUDGET_RATIO = 1.05
MIN_VALIDATION_NEGATIVE_RECALL = 0.70
MIN_RECENT_NEGATIVE_RECALL = 0.50
MIN_VALIDATION_NEGATIVE_POINTS = 8
SOFT_ACTIVATION_FLOOR = 0.50
OOF_CALIBRATION_DAYS = 45
OOF_BLOCK_DAYS = 15
REGISTERED_LEGACY_EVENT_MAE = 210.96577374954444
REGISTERED_LEGACY_NONNEGATIVE_MAE = 262.33698440917453
EPS = 1e-9


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def zero_price_probability(
    baseline: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate P(price < 0) from residual-aligned TimesFM quantiles."""
    aligned = np.column_stack(
        [
            baseline + (q10 - q50),
            baseline,
            baseline + (q90 - q50),
        ]
    )
    aligned.sort(axis=1)
    a10, a50, a90 = aligned.T
    probability = np.empty(len(baseline), dtype=float)

    below_q10 = 0.0 <= a10
    q10_to_q50 = (a10 < 0.0) & (0.0 <= a50)
    q50_to_q90 = (a50 < 0.0) & (0.0 <= a90)
    above_q90 = a90 < 0.0

    lower_scale = np.maximum(a50 - a10, 1.0)
    upper_scale = np.maximum(a90 - a50, 1.0)
    probability[below_q10] = 0.10 * np.exp(-a10[below_q10] / lower_scale[below_q10])
    probability[q10_to_q50] = 0.10 + 0.40 * (
        -a10[q10_to_q50] / np.maximum(a50[q10_to_q50] - a10[q10_to_q50], 1.0)
    )
    probability[q50_to_q90] = 0.50 + 0.40 * (
        -a50[q50_to_q90] / np.maximum(a90[q50_to_q90] - a50[q50_to_q90], 1.0)
    )
    probability[above_q90] = 1.0 - 0.10 * np.exp(
        a90[above_q90] / upper_scale[above_q90]
    )
    return np.clip(probability, 1e-5, 1.0 - 1e-5), a10, a50, a90


def fused_negative_score(
    p_supply_negative: np.ndarray,
    p_timesfm_negative: np.ndarray,
    supply_weight: float,
) -> np.ndarray:
    return sigmoid(
        supply_weight * logit(p_supply_negative)
        + (1.0 - supply_weight) * logit(p_timesfm_negative)
    )


def empirical_low_price_score(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=float))
    rank = np.searchsorted(sorted_reference, values, side="left")
    score = (len(sorted_reference) - rank + 0.5) / (len(sorted_reference) + 1.0)
    return np.clip(score, 1e-5, 1.0 - 1e-5)


def market_consistent_negative_score(
    p_energy_low: np.ndarray,
    p_supply_negative: np.ndarray,
    p_timesfm_negative: np.ndarray,
    energy_weight: float,
    supply_weight: float,
) -> np.ndarray:
    state_score = fused_negative_score(
        p_supply_negative, p_timesfm_negative, supply_weight
    )
    return sigmoid(
        energy_weight * logit(p_energy_low)
        + (1.0 - energy_weight) * logit(state_score)
    )


def add_negative_selector_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    base_columns = [
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
    ]
    trajectory_sources = [
        "price_day_ahead_energy_proxy",
        "market_bid_space_proxy",
        "supply_adequacy_proxy",
        "renewable_supply_index",
        "load_day_ahead_pred",
    ]
    trajectory_columns: list[str] = []
    grouped = result.groupby("delivery_day", sort=False)
    for name in trajectory_sources:
        daily_mean = f"{name}_day_mean"
        daily_min = f"{name}_day_min"
        result[daily_mean] = grouped[name].transform("mean")
        result[daily_min] = grouped[name].transform("min")
        trajectory_columns.extend([daily_mean, daily_min])
        for window in (3, 5):
            feature = f"{name}_center{window}_mean"
            result[feature] = grouped[name].transform(
                lambda values, w=window: values.rolling(
                    w, center=True, min_periods=1
                ).mean()
            )
            trajectory_columns.append(feature)
    columns = base_columns + trajectory_columns
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0), columns


def fit_negative_selector(x_train, y_train, x_val, y_val, config):
    params = model_params(config)
    params.update(objective="binary", class_weight="balanced")
    model = LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return model


def expanding_oof_supply_probabilities(
    frame: pd.DataFrame,
    x_supply: np.ndarray,
    actual: np.ndarray,
    train_mask: np.ndarray,
    low_threshold: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    train_days = np.array(
        sorted(frame.loc[train_mask, "delivery_day"].dt.normalize().unique())
    )
    if len(train_days) < OOF_CALIBRATION_DAYS + 14:
        raise ValueError("too few training days for expanding-window OOF calibration")
    calibration_days = train_days[-OOF_CALIBRATION_DAYS:]
    all_states = supply_labels(actual, low_threshold)
    positions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for block_start in range(0, len(calibration_days), OOF_BLOCK_DAYS):
        block_days = calibration_days[block_start : block_start + OOF_BLOCK_DAYS]
        first_block_day = block_days[0]
        prefix_days = train_days[train_days < first_block_day]
        if len(prefix_days) < 14:
            raise ValueError("OOF prefix lacks fit and early-stopping days")
        early_days = prefix_days[-7:]
        fit_days = prefix_days[:-7]
        fit_mask = train_mask & frame["delivery_day"].isin(fit_days).to_numpy()
        early_mask = train_mask & frame["delivery_day"].isin(early_days).to_numpy()
        block_mask = train_mask & frame["delivery_day"].isin(block_days).to_numpy()
        model = fit_supply_router(
            x_supply[fit_mask],
            all_states[fit_mask],
            x_supply[early_mask],
            all_states[early_mask],
            config,
        )
        positions.append(np.flatnonzero(block_mask))
        probabilities.append(model.predict_proba(x_supply[block_mask]))
    position = np.concatenate(positions)
    probability = np.vstack(probabilities)
    order = np.argsort(position)
    return position[order], probability[order]


def detector_metrics(actual: np.ndarray, activation: np.ndarray) -> dict:
    negative = actual < 0.0
    selected = activation > 0.0
    risk = selection_risk(actual, activation)
    true_selected = int(np.sum(negative & selected))
    false_selected = int(np.sum(~negative & selected))
    recall = true_selected / int(negative.sum()) if negative.any() else 0.0
    precision = true_selected / int(selected.sum()) if selected.any() else 0.0
    return {
        **risk,
        "selection_recall": float(recall),
        "selection_precision": float(precision),
        "selected_points": int(selected.sum()),
        "selected_negative_points": true_selected,
        "selected_nonnegative_points": false_selected,
        "negative_points": int(negative.sum()),
    }


def select_detector_policy(
    actual: np.ndarray,
    p_market_negative: np.ndarray,
    p_timesfm_negative: np.ndarray,
    delivery_days: np.ndarray,
    recent_mask: np.ndarray,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for supply_weight, threshold, structured_window in product(
        FUSION_SUPPLY_WEIGHTS, ACTIVATION_THRESHOLDS, STRUCTURED_WINDOWS
    ):
        raw_score = fused_negative_score(
            p_market_negative, p_timesfm_negative, supply_weight
        )
        score = smooth_score_by_day(raw_score, delivery_days, structured_window)
        activation = activation_from_score(score, threshold)
        pooled = detector_metrics(actual, activation)
        recent = detector_metrics(actual[recent_mask], activation[recent_mask])
        constraints = {
            "pooled_risk_ok": pooled["weighted_false_route_risk"]
            <= SELECTION_RISK_BUDGET + EPS,
            "pooled_recall_ok": pooled["selection_recall"]
            >= MIN_VALIDATION_NEGATIVE_RECALL - EPS,
            "recent_risk_ok": recent["weighted_false_route_risk"]
            <= RECENT_SELECTION_RISK_BUDGET + EPS,
            "recent_recall_ok": recent["selection_recall"]
            >= MIN_RECENT_NEGATIVE_RECALL - EPS,
            "selection_ok": pooled["effective_selected_points"]
            >= MIN_VALIDATION_NEGATIVE_POINTS,
        }
        violation = (
            max(pooled["weighted_false_route_risk"] - SELECTION_RISK_BUDGET, 0.0)
            + max(MIN_VALIDATION_NEGATIVE_RECALL - pooled["selection_recall"], 0.0)
            + max(
                recent["weighted_false_route_risk"] - RECENT_SELECTION_RISK_BUDGET,
                0.0,
            )
            + max(MIN_RECENT_NEGATIVE_RECALL - recent["selection_recall"], 0.0)
        )
        rows.append(
            {
                "supply_weight": supply_weight,
                "activation_threshold": threshold,
                "structured_window": structured_window,
                "feasible": all(constraints.values()),
                "constraint_violation": float(violation),
                **constraints,
                **{f"pooled_{key}": value for key, value in pooled.items()},
                **{f"recent_{key}": value for key, value in recent.items()},
            }
        )
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                row["pooled_weighted_false_route_risk"],
                row["recent_weighted_false_route_risk"],
                -row["pooled_selection_recall"],
                -row["activation_threshold"],
            ),
        )
        gate = "PASS"
    else:
        selected = min(
            rows,
            key=lambda row: (
                row["constraint_violation"],
                row["pooled_weighted_false_route_risk"],
            ),
        )
        gate = "FAIL_NO_FEASIBLE_DETECTOR"
    policy = {
        "supply_weight": selected["supply_weight"],
        "activation_threshold": selected["activation_threshold"],
        "structured_window": selected["structured_window"],
        "detector_calibration_gate": gate,
    }
    return policy, rows


def select_market_consistent_detector(
    actual: np.ndarray,
    p_energy_low: np.ndarray,
    p_supply_negative: np.ndarray,
    p_timesfm_negative: np.ndarray,
    delivery_days: np.ndarray,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for energy_weight, supply_weight, threshold, structured_window in product(
        ENERGY_PRIOR_WEIGHTS,
        FUSION_SUPPLY_WEIGHTS,
        ACTIVATION_THRESHOLDS,
        STRUCTURED_WINDOWS,
    ):
        raw_score = market_consistent_negative_score(
            p_energy_low,
            p_supply_negative,
            p_timesfm_negative,
            energy_weight,
            supply_weight,
        )
        score = smooth_score_by_day(raw_score, delivery_days, structured_window)
        activation = activation_from_score(score, threshold)
        metrics = detector_metrics(actual, activation)
        hard_risk = 1.0 - metrics["selection_precision"] if metrics["selected_points"] else 1.0
        constraints = {
            "upper_risk_ok": metrics["risk_normal_90pct_upper"]
            <= SELECTION_RISK_BUDGET + EPS,
            "hard_risk_ok": hard_risk <= SELECTION_RISK_BUDGET + EPS,
            "selection_ok": metrics["effective_selected_points"]
            >= MIN_VALIDATION_NEGATIVE_POINTS,
        }
        rows.append(
            {
                "energy_weight": energy_weight,
                "supply_weight": supply_weight,
                "activation_threshold": threshold,
                "structured_window": structured_window,
                "feasible": all(constraints.values()),
                "hard_false_route_risk": float(hard_risk),
                **constraints,
                **metrics,
            }
        )
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                -row["energy_weight"],
                -row["selection_recall"],
                row["hard_false_route_risk"],
                row["weighted_false_route_risk"],
            ),
        )
        gate = "PASS"
    else:
        selected = min(
            rows,
            key=lambda row: (
                max(row["hard_false_route_risk"] - SELECTION_RISK_BUDGET, 0.0)
                + max(
                    row["weighted_false_route_risk"] - SELECTION_RISK_BUDGET,
                    0.0,
                ),
                -row["selection_recall"],
            ),
        )
        gate = "FAIL_NO_FEASIBLE_MARKET_CONSISTENT_DETECTOR"
    policy = {
        "energy_weight": selected["energy_weight"],
        "supply_weight": selected["supply_weight"],
        "activation_threshold": selected["activation_threshold"],
        "structured_window": selected["structured_window"],
        "detector_calibration_gate": gate,
    }
    return policy, rows


def select_amplitude_policy(
    actual: np.ndarray,
    physical_baseline: np.ndarray,
    nonnegative_reference: np.ndarray,
    p_supply: np.ndarray,
    low_prediction: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    activation: np.ndarray,
) -> tuple[dict, list[dict], dict]:
    baseline = policy_metrics(actual, physical_baseline, np.zeros(len(actual)))
    nonnegative = actual >= 0.0
    reference_nonnegative_mae = float(
        np.mean(np.abs(nonnegative_reference[nonnegative] - actual[nonnegative]))
    )
    baseline["nonnegative_budget_reference_mae"] = reference_nonnegative_mae
    rows: list[dict] = []
    for beta_negative, quantile_scale, beta_low in product(
        NEGATIVE_BETAS, LOWER_QUANTILE_SCALES, LOW_BETAS
    ):
        predicted = apply_risk_route(
            physical_baseline,
            p_supply,
            low_prediction,
            q10,
            q50,
            activation,
            beta_negative,
            quantile_scale,
            beta_low,
        )
        metrics = policy_metrics(actual, predicted, activation)
        constraints = {
            "overall_ok": metrics["mae"] <= baseline["mae"] + EPS,
            "nonnegative_ok": metrics["nonnegative_mae"]
            <= reference_nonnegative_mae * NONNEGATIVE_MAE_BUDGET_RATIO + EPS,
            "negative_ok": metrics["negative_mae"] < baseline["negative_mae"] - EPS,
        }
        violation = (
            max(metrics["mae"] / baseline["mae"] - 1.0, 0.0)
            + max(
                metrics["nonnegative_mae"]
                / (reference_nonnegative_mae * NONNEGATIVE_MAE_BUDGET_RATIO)
                - 1.0,
                0.0,
            )
        )
        rows.append(
            {
                "beta_negative": beta_negative,
                "quantile_scale": quantile_scale,
                "beta_low": beta_low,
                "feasible": all(constraints.values()),
                "constraint_violation": float(violation),
                **constraints,
                **metrics,
            }
        )
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        selected = min(feasible, key=lambda row: (row["mae"], row["nonnegative_mae"]))
        gate = "PASS"
    else:
        selected = min(rows, key=lambda row: (row["constraint_violation"], row["mae"]))
        gate = "FAIL_NO_FEASIBLE_AMPLITUDE"
    policy = {
        "beta_negative": selected["beta_negative"],
        "quantile_scale": selected["quantile_scale"],
        "beta_low": selected["beta_low"],
        "amplitude_calibration_gate": gate,
    }
    return policy, rows, baseline


def smooth_score_by_day(
    score: np.ndarray, delivery_days: np.ndarray, window: int
) -> np.ndarray:
    if window == 1:
        return score.copy()
    local = pd.DataFrame({"day": delivery_days, "score": score})
    return local.groupby("day", sort=False)["score"].transform(
        lambda values: values.rolling(window, center=True, min_periods=1).mean()
    ).to_numpy(float)


def activation_from_score(score: np.ndarray, threshold: float) -> np.ndarray:
    selected = score >= threshold
    strength = SOFT_ACTIVATION_FLOOR + (1.0 - SOFT_ACTIVATION_FLOOR) * np.clip(
        (score - threshold) / max(1.0 - threshold, EPS), 0.0, 1.0
    )
    return selected.astype(float) * strength


def apply_risk_route(
    baseline: np.ndarray,
    p_supply: np.ndarray,
    low_prediction: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    activation: np.ndarray,
    beta_negative: float,
    quantile_scale: float,
    beta_low: float,
) -> np.ndarray:
    market_prediction = baseline + beta_low * p_supply[:, 1] * (low_prediction - baseline)
    conditional_lower = np.minimum(
        np.clip(baseline + quantile_scale * (q10 - q50), PRICE_FLOOR, 0.0), -1.0
    )
    total_strength = beta_negative * activation
    selected = activation > 0.0
    sign_crossing_strength = np.clip(
        (market_prediction + 1.0) / np.maximum(market_prediction - conditional_lower, EPS),
        0.0,
        1.0,
    )
    total_strength = np.where(
        selected & (market_prediction >= 0.0),
        np.maximum(total_strength, sign_crossing_strength),
        total_strength,
    )
    prediction = market_prediction + total_strength * (
        conditional_lower - market_prediction
    )
    return np.where(selected, prediction, np.maximum(market_prediction, 0.0))


def selection_risk(actual: np.ndarray, activation: np.ndarray) -> dict:
    total_weight = float(activation.sum())
    if total_weight <= EPS:
        return {
            "weighted_false_route_risk": 1.0,
            "effective_selected_points": 0.0,
            "risk_normal_90pct_upper": 1.0,
        }
    risk = float(np.sum(activation * (actual >= 0.0)) / total_weight)
    effective_n = float(total_weight**2 / max(float(np.sum(activation**2)), EPS))
    standard_error = np.sqrt(max(risk * (1.0 - risk), 0.0) / max(effective_n, 1.0))
    return {
        "weighted_false_route_risk": risk,
        "effective_selected_points": effective_n,
        "risk_normal_90pct_upper": float(min(1.0, risk + 1.281552 * standard_error)),
    }


def policy_metrics(actual: np.ndarray, predicted: np.ndarray, activation: np.ndarray) -> dict:
    absolute = np.abs(predicted - actual)
    negative = actual < 0.0
    nonnegative = ~negative
    result = {
        "mae": float(absolute.mean()),
        "negative_mae": float(absolute[negative].mean()) if negative.any() else None,
        "nonnegative_mae": float(absolute[nonnegative].mean()) if nonnegative.any() else None,
        "selected_points": int(np.sum(activation > 0.0)),
        "mean_activation": float(activation.mean()),
    }
    result.update(selection_risk(actual, activation))
    result.update(classification_metrics(actual, predicted))
    return result


def select_policy(
    actual: np.ndarray,
    physical_baseline: np.ndarray,
    p_supply: np.ndarray,
    p_market_negative: np.ndarray,
    low_prediction: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    p_timesfm_negative: np.ndarray,
    delivery_days: np.ndarray,
) -> tuple[dict, list[dict], dict]:
    negative = actual < 0.0
    if int(negative.sum()) < MIN_VALIDATION_NEGATIVE_POINTS:
        disabled = {
            "supply_weight": FUSION_SUPPLY_WEIGHTS[-1],
            "activation_threshold": 1.0,
            "beta_negative": 0.0,
            "quantile_scale": 0.0,
            "beta_low": 0.0,
            "structured_window": 1,
            "calibration_gate": "DISABLED_INSUFFICIENT_NEGATIVES",
        }
        return disabled, [], policy_metrics(actual, physical_baseline, np.zeros(len(actual)))

    baseline = policy_metrics(actual, physical_baseline, np.zeros(len(actual)))
    rows: list[dict] = []
    for supply_weight, threshold, beta_negative, quantile_scale, beta_low, structured_window in product(
        FUSION_SUPPLY_WEIGHTS,
        ACTIVATION_THRESHOLDS,
        NEGATIVE_BETAS,
        LOWER_QUANTILE_SCALES,
        LOW_BETAS,
        STRUCTURED_WINDOWS,
    ):
        raw_score = fused_negative_score(
            p_market_negative, p_timesfm_negative, supply_weight
        )
        score = smooth_score_by_day(raw_score, delivery_days, structured_window)
        activation = activation_from_score(score, threshold)
        predicted = apply_risk_route(
            physical_baseline,
            p_supply,
            low_prediction,
            q10,
            q50,
            activation,
            beta_negative,
            quantile_scale,
            beta_low,
        )
        metrics = policy_metrics(actual, predicted, activation)
        constraints = {
            "risk_ok": metrics["weighted_false_route_risk"] <= SELECTION_RISK_BUDGET + EPS,
            "recall_ok": metrics["negative_recall"] >= MIN_VALIDATION_NEGATIVE_RECALL - EPS,
            "overall_ok": metrics["mae"] <= baseline["mae"] + EPS,
            "nonnegative_ok": metrics["nonnegative_mae"]
            <= baseline["nonnegative_mae"] * NONNEGATIVE_MAE_BUDGET_RATIO + EPS,
            "negative_ok": metrics["negative_mae"] < baseline["negative_mae"] - EPS,
            "selection_ok": metrics["effective_selected_points"] >= MIN_VALIDATION_NEGATIVE_POINTS,
        }
        feasible = all(constraints.values())
        violation = (
            max(metrics["weighted_false_route_risk"] - SELECTION_RISK_BUDGET, 0.0)
            + max(MIN_VALIDATION_NEGATIVE_RECALL - metrics["negative_recall"], 0.0)
            + max(metrics["mae"] / baseline["mae"] - 1.0, 0.0)
            + max(
                metrics["nonnegative_mae"]
                / (baseline["nonnegative_mae"] * NONNEGATIVE_MAE_BUDGET_RATIO)
                - 1.0,
                0.0,
            )
        )
        rows.append(
            {
                "supply_weight": supply_weight,
                "activation_threshold": threshold,
                "beta_negative": beta_negative,
                "quantile_scale": quantile_scale,
                "beta_low": beta_low,
                "structured_window": structured_window,
                "feasible": feasible,
                "constraint_violation": float(violation),
                **constraints,
                **metrics,
            }
        )

    feasible_rows = [row for row in rows if row["feasible"]]
    if feasible_rows:
        selected = min(
            feasible_rows,
            key=lambda row: (
                row["mae"],
                row["nonnegative_mae"],
                row["negative_false_positive"],
                row["beta_negative"] + row["beta_low"],
                row["structured_window"],
            ),
        )
        gate = "PASS"
    else:
        selected = min(
            rows,
            key=lambda row: (
                row["constraint_violation"],
                row["mae"],
                row["negative_false_positive"],
            ),
        )
        gate = "FAIL_NO_FEASIBLE_POLICY"
    policy = {
        key: selected[key]
        for key in (
            "supply_weight",
            "activation_threshold",
            "beta_negative",
            "quantile_scale",
            "beta_low",
            "structured_window",
        )
    }
    policy["calibration_gate"] = gate
    return policy, rows, baseline


def add_nonnegative_mae(actual: np.ndarray, predicted: np.ndarray, metrics: dict) -> dict:
    nonnegative = actual >= 0.0
    metrics["nonnegative_price_mae"] = (
        float(np.mean(np.abs(predicted[nonnegative] - actual[nonnegative])))
        if nonnegative.any()
        else None
    )
    return metrics


def plot_event(predictions: pd.DataFrame, output: Path) -> None:
    if predictions["fold"].nunique() != 1:
        return
    wide = predictions.pivot(index="time", columns="mode", values="predicted")
    actual = predictions.drop_duplicates("time").set_index("time")["actual"]
    score = predictions[predictions["mode"] == "risk_supply_congestion_route"].set_index("time")
    time = pd.to_datetime(wide.index)

    fig, (ax_price, ax_route) = plt.subplots(
        2,
        1,
        figsize=(14, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    ax_price.plot(time, actual.to_numpy(), color="#111827", linewidth=1.7, label="Actual")
    ax_price.plot(time, wide["carrier_blind"], color="#64748B", linewidth=1.0, label="Blind")
    ax_price.plot(
        time,
        wide["legacy_floor_dual_route"],
        color="#D97706",
        linewidth=1.0,
        alpha=0.9,
        label="Legacy floor route",
    )
    ax_price.plot(
        time,
        wide["risk_supply_congestion_route"],
        color="#047857",
        linewidth=1.4,
        label="Risk-controlled dual route",
    )
    ax_price.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    ax_price.set_ylabel("Price (RMB/MWh)")
    ax_price.legend(ncol=4, frameon=False, loc="upper left")
    ax_price.grid(axis="y", color="#E5E7EB", linewidth=0.7)

    ax_route.plot(
        time,
        score["p_supply_negative"].to_numpy(),
        color="#2563EB",
        linewidth=1.0,
        label="Supply-state probability",
    )
    ax_route.plot(
        time,
        score["p_timesfm_negative"].to_numpy(),
        color="#7C3AED",
        linewidth=1.0,
        label="TimesFM distribution evidence",
    )
    ax_route.plot(
        time,
        score["negative_activation"].to_numpy(),
        color="#DC2626",
        linewidth=1.2,
        label="Expert activation",
    )
    ax_route.fill_between(
        time,
        0.0,
        1.0,
        where=actual.to_numpy() < 0.0,
        color="#111827",
        alpha=0.08,
        step="mid",
        label="Actual negative price",
    )
    ax_route.set_ylim(-0.02, 1.02)
    ax_route.set_ylabel("Probability")
    ax_route.set_xlabel("Delivery time")
    ax_route.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax_route.legend(ncol=4, frameon=False, loc="upper left")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--carrier-cache", type=Path, required=True)
    parser.add_argument("--carrier-column", type=str, default="timesfm_prediction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels, usecols=["time", "congestion_state"])
    carrier = pd.read_csv(args.carrier_cache)
    required_quantiles = {"timesfm_q10", "timesfm_q50", "timesfm_q90"}
    if not required_quantiles.issubset(carrier.columns):
        raise ValueError(f"carrier cache lacks quantiles: {sorted(required_quantiles - set(carrier.columns))}")

    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    labels["time"] = pd.to_datetime(labels["time"])
    carrier["time"] = pd.to_datetime(carrier["time"])
    frame = add_supply_features(
        add_features(frame.merge(labels, on="time", how="left").merge(carrier, on="time", how="left"))
    )

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
        args.carrier_column,
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
        args.carrier_column,
    ]

    x_blind = frame[blind_columns].to_numpy(float)
    x_full = frame[full_columns].to_numpy(float)
    x_supply = frame[supply_columns].to_numpy(float)
    actual = frame["price_real"].to_numpy(float)
    carrier_prediction = frame[args.carrier_column].to_numpy(float)
    q10 = frame["timesfm_q10"].to_numpy(float)
    q50 = frame["timesfm_q50"].to_numpy(float)
    q90 = frame["timesfm_q90"].to_numpy(float)
    residual = actual - carrier_prediction
    congestion_state = frame["congestion_state"].fillna(-1).to_numpy(int)

    fold_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict] = []
    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (day_mask(frame, fold[name]) for name in ("train", "val", "test"))
        labeled_train = train & (congestion_state >= 0)
        labeled_val = val & (congestion_state >= 0)
        nonnegative_train_prices = actual[train][actual[train] >= 0.0]
        low_threshold = float(np.quantile(nonnegative_train_prices, 0.10))
        train_supply_state = supply_labels(actual[train], low_threshold)
        val_supply_state = supply_labels(actual[val], low_threshold)
        low_train = train & (actual >= 0.0) & (actual <= low_threshold)
        low_val = val & (actual >= 0.0) & (actual <= low_threshold)

        blind = fit_residual(x_blind[train], residual[train], x_blind[val], residual[val], config)
        full = fit_residual(x_full[train], residual[train], x_full[val], residual[val], config)
        blind_val = carrier_prediction[val] + blind.predict(x_blind[val])
        full_val = carrier_prediction[val] + full.predict(x_full[val])
        blind_test = carrier_prediction[test] + blind.predict(x_blind[test])
        full_test = carrier_prediction[test] + full.predict(x_full[test])

        congestion_router = fit_router(
            x_full[labeled_train],
            congestion_state[labeled_train],
            x_full[labeled_val],
            congestion_state[labeled_val],
            config,
        )
        p_congestion_val = congestion_router.predict_proba(x_full[val])
        p_congestion_test = congestion_router.predict_proba(x_full[test])
        physical_val = blind_val + (1.0 - p_congestion_val[:, 1]) * (full_val - blind_val)
        physical_test = blind_test + (1.0 - p_congestion_test[:, 1]) * (full_test - blind_test)

        supply_router = fit_supply_router(
            x_supply[train], train_supply_state, x_supply[val], val_supply_state, config
        )
        p_supply_val = supply_router.predict_proba(x_supply[val])
        p_supply_test = supply_router.predict_proba(x_supply[test])
        low_expert = fit_low_nonnegative_expert(
            x_supply[low_train], actual[low_train], x_supply[low_val], actual[low_val], config
        )
        low_prediction_val = low_expert.predict(x_supply[val])
        low_prediction_test = low_expert.predict(x_supply[test])

        p_tf_val, _, _, _ = zero_price_probability(q50[val], q10[val], q50[val], q90[val])
        p_tf_test, aligned_q10, aligned_q50, aligned_q90 = zero_price_probability(
            q50[test], q10[test], q50[test], q90[test]
        )
        energy_reference = frame.loc[train, "price_day_ahead_energy_proxy"].to_numpy(float)
        p_energy_val = empirical_low_price_score(
            energy_reference,
            frame.loc[val, "price_day_ahead_energy_proxy"].to_numpy(float),
        )
        p_energy_test = empirical_low_price_score(
            energy_reference,
            frame.loc[test, "price_day_ahead_energy_proxy"].to_numpy(float),
        )
        detector_policy, detector_candidates = select_market_consistent_detector(
            actual[val],
            p_energy_val,
            p_supply_val[:, 0],
            p_tf_val,
            frame.loc[val, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy(),
        )
        raw_score_val = market_consistent_negative_score(
            p_energy_val,
            p_supply_val[:, 0],
            p_tf_val,
            detector_policy["energy_weight"],
            detector_policy["supply_weight"],
        )
        score_val = smooth_score_by_day(
            raw_score_val,
            frame.loc[val, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy(),
            detector_policy["structured_window"],
        )
        activation_val = activation_from_score(
            score_val, detector_policy["activation_threshold"]
        )
        amplitude_policy, amplitude_candidates, validation_baseline = select_amplitude_policy(
            actual[val],
            physical_val,
            blind_val,
            p_supply_val,
            low_prediction_val,
            q10[val],
            q50[val],
            activation_val,
        )
        policy = {**detector_policy, **amplitude_policy}
        policy["calibration_gate"] = (
            "PASS"
            if detector_policy["detector_calibration_gate"] == "PASS"
            and amplitude_policy["amplitude_calibration_gate"] == "PASS"
            else "FAIL"
        )
        raw_score_test = market_consistent_negative_score(
            p_energy_test,
            p_supply_test[:, 0],
            p_tf_test,
            policy["energy_weight"],
            policy["supply_weight"],
        )
        score_test = smooth_score_by_day(
            raw_score_test,
            frame.loc[test, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy(),
            policy["structured_window"],
        )
        activation_test = activation_from_score(score_test, policy["activation_threshold"])

        risk_supply_only = apply_risk_route(
            blind_test,
            p_supply_test,
            low_prediction_test,
            q10[test],
            q50[test],
            activation_test,
            policy["beta_negative"],
            policy["quantile_scale"],
            policy["beta_low"],
        )
        risk_dual = apply_risk_route(
            physical_test,
            p_supply_test,
            low_prediction_test,
            q10[test],
            q50[test],
            activation_test,
            policy["beta_negative"],
            policy["quantile_scale"],
            policy["beta_low"],
        )

        legacy_beta_negative, legacy_beta_low, legacy_threshold, legacy_candidates = select_betas(
            actual[val], physical_val, p_supply_val, low_prediction_val, low_threshold
        )
        legacy_dual = apply_supply_route(
            physical_test,
            p_supply_test,
            low_prediction_test,
            legacy_beta_negative,
            legacy_beta_low,
            legacy_threshold,
        )

        calibration_rows.append(
            {
                "fold": fold_index,
                "train_supply_state_counts": {
                    SUPPLY_STATE_NAMES[i]: int(np.sum(train_supply_state == i)) for i in range(3)
                },
                "val_supply_state_counts": {
                    SUPPLY_STATE_NAMES[i]: int(np.sum(val_supply_state == i)) for i in range(3)
                },
                "selected_policy": policy,
                "calibration_coverage": {
                    "validation_points": int(val.sum()),
                    "validation_days": int(frame.loc[val, "delivery_day"].nunique()),
                    "validation_negative_points": int(np.sum(actual[val] < 0.0)),
                    "training_energy_reference_points": int(train.sum()),
                },
                "validation_physical_baseline": validation_baseline,
                "detector_candidates": detector_candidates,
                "amplitude_candidates": amplitude_candidates,
                "legacy_policy": {
                    "beta_negative": legacy_beta_negative,
                    "beta_low": legacy_beta_low,
                    "activation_threshold": legacy_threshold,
                },
                "legacy_validation_candidates": legacy_candidates,
            }
        )

        predictions = {
            "carrier_blind": blind_test,
            "physical_congestion_gate": physical_test,
            "legacy_floor_dual_route": legacy_dual,
            "risk_supply_route_only": risk_supply_only,
            "risk_supply_congestion_route": risk_dual,
        }
        test_positions = np.flatnonzero(test)
        for mode, predicted in predictions.items():
            metrics = error_metrics(
                actual[test], predicted, actual[test] <= low_threshold, congestion_state[test]
            )
            metrics.update(classification_metrics(actual[test], predicted))
            add_nonnegative_mae(actual[test], predicted, metrics)
            fold_rows.append(
                {"fold": fold_index, "mode": mode, "low_threshold_train_q10": low_threshold, **metrics}
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_index,
                        "delivery_day": frame.loc[test, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy(),
                        "time": frame.loc[test, "time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                        "mode": mode,
                        "actual": actual[test],
                        "predicted": predicted,
                        "absolute_error": np.abs(predicted - actual[test]),
                        "low_price": actual[test] <= low_threshold,
                        "congestion_state": congestion_state[test_positions],
                        "p_supply_negative": p_supply_test[:, 0],
                        "p_energy_low_prior": p_energy_test,
                        "p_supply_low_nonnegative": p_supply_test[:, 1],
                        "p_timesfm_negative": p_tf_test,
                        "p_negative_fused": score_test,
                        "negative_activation": activation_test,
                        "p_congestion_non_neutral": 1.0 - p_congestion_test[:, 1],
                        "timesfm_aligned_q10": aligned_q10,
                        "timesfm_aligned_q50": aligned_q50,
                        "timesfm_aligned_q90": aligned_q90,
                    }
                )
            )

    metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = predictions.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(
        mae=("absolute_error", "mean")
    )
    comparisons = {
        "risk_dual_vs_blind": summarize(
            metrics, daily, "risk_supply_congestion_route", "carrier_blind"
        ),
        "risk_dual_vs_legacy": summarize(
            metrics, daily, "risk_supply_congestion_route", "legacy_floor_dual_route"
        ),
        "risk_dual_vs_supply_only": summarize(
            metrics, daily, "risk_supply_congestion_route", "risk_supply_route_only"
        ),
        "physical_vs_blind": summarize(
            metrics, daily, "physical_congestion_gate", "carrier_blind"
        ),
    }

    final_rows = metrics[metrics["mode"] == "risk_supply_congestion_route"]
    legacy_rows = metrics[metrics["mode"] == "legacy_floor_dual_route"]
    supply_rows = metrics[metrics["mode"] == "risk_supply_route_only"]
    calibration_pass = all(
        row["selected_policy"]["calibration_gate"] == "PASS" for row in calibration_rows
    )
    test_gates = {
        "calibration_pass": calibration_pass,
        "negative_recall_at_least_70pct": bool((final_rows["negative_recall"] >= 0.70).all()),
        "false_positive_below_69": bool((final_rows["negative_false_positive"] < 69).all()),
        "precision_above_63_87pct": bool((final_rows["negative_precision"] > 0.6387).all()),
        "nonnegative_mae_below_legacy": bool(
            (final_rows["nonnegative_price_mae"].to_numpy() < legacy_rows["nonnegative_price_mae"].to_numpy()).all()
        ),
        "nonnegative_mae_below_registered_262_337": bool(
            (final_rows["nonnegative_price_mae"] < REGISTERED_LEGACY_NONNEGATIVE_MAE).all()
        ),
        "overall_mae_below_legacy": bool(
            (final_rows["mae"].to_numpy() < legacy_rows["mae"].to_numpy()).all()
        ),
        "overall_mae_below_registered_210_966": bool(
            (final_rows["mae"] < REGISTERED_LEGACY_EVENT_MAE).all()
        ),
        "congestion_independent_gain": bool(
            (
                (final_rows["mae"].to_numpy() < supply_rows["mae"].to_numpy())
                | (
                    final_rows["negative_price_mae"].to_numpy()
                    < supply_rows["negative_price_mae"].to_numpy()
                )
            ).all()
        ),
    }
    report = {
        "status": "complete",
        "r9_gate": "PASS" if all(test_gates.values()) else "FAIL",
        "test_gates": test_gates,
        "mechanism": (
            "The activation set is calibrated separately from expert amplitude. A monotone empirical low-tail "
            "prior from the day-ahead energy component is fused with the congestion-free three-state market "
            "probability and the TimesFM probability below zero. The validation selector maximizes negative-state "
            "coverage under weighted and hard false-route budgets. Unselected points are projected to the "
            "nonnegative state; selected amplitudes use the TimesFM lower quantile. Congestion independently "
            "gates Full-minus-Blind, and -200 RMB/MWh remains only the legal lower bound."
        ),
        "equations": {
            "distribution_evidence": "p_tf = F_TimesFM(0 | q10_aligned, q50_aligned, q90_aligned)",
            "fusion": "p_neg = sigmoid(w*logit(p_supply_neg) + (1-w)*logit(p_tf))",
            "selection_risk": "R_FP = sum(a_t*I(y_t>=0))/sum(a_t)",
            "negative_target": "q_neg = clip(y_base + gamma*(q10-q50), -200, 0)",
            "prediction": "y_final = y_market + beta_neg*a_t*(q_neg-y_market)",
            "congestion": "y_physical = y_blind + (1-p_congestion_neutral)*(y_full-y_blind)",
        },
        "pre_registered_budgets": {
            "selection_false_route_risk": SELECTION_RISK_BUDGET,
            "validation_nonnegative_mae_ratio": NONNEGATIVE_MAE_BUDGET_RATIO,
            "soft_activation_floor": SOFT_ACTIVATION_FLOOR,
            "parameter_grid": {
                "energy_prior_weights": ENERGY_PRIOR_WEIGHTS,
                "supply_weights": FUSION_SUPPLY_WEIGHTS,
                "activation_thresholds": ACTIVATION_THRESHOLDS,
                "negative_betas": NEGATIVE_BETAS,
                "lower_quantile_scales": LOWER_QUANTILE_SCALES,
                "low_betas": LOW_BETAS,
                "structured_windows": STRUCTURED_WINDOWS,
            },
        },
        "features": {
            "strict_blind": blind_columns,
            "congestion_increment": congestion_columns,
            "three_state_supply": supply_columns,
        },
        "calibration": calibration_rows,
        "comparisons": comparisons,
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
    (args.output / "risk_route_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_event(predictions, args.output / "spring_festival_risk_route.png")
    print(
        json.dumps(
            {
                "r9_gate": report["r9_gate"],
                "test_gates": test_gates,
                "selected_policy": [row["selected_policy"] for row in calibration_rows],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
