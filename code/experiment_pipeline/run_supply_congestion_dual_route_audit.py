from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import wilcoxon

from run_chronos2_strong_carrier_audit import fit_residual, model_params
from run_physical_soft_route_price_audit import add_features, day_mask, error_metrics, fit_router


BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NEGATIVE_ACTIVATION_THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)
PRICE_FLOOR = -200.0
SUPPLY_STATE_NAMES = ("negative_price", "low_nonnegative", "regular")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def add_supply_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    load_scale = result["load_day_ahead_pred"].abs().clip(lower=1000.0)
    result["market_bid_space_proxy"] = (
        result["load_day_ahead_pred"] - result["elec_exter_plan"] - result["elec_fix_out_plan"]
    )
    result["supply_adequacy_proxy"] = (
        result["elec_gene_total_pred"] - result["market_bid_space_proxy"]
    )
    result["external_plan_load_ratio"] = result["elec_exter_plan"] / load_scale
    result["fixed_output_load_ratio"] = result["elec_fix_out_plan"] / load_scale
    result["generation_load_ratio"] = result["elec_gene_total_pred"] / load_scale
    result["renewable_supply_index"] = result["energy_hydro_renewable"] / load_scale
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_supply_router(x_train, y_train, x_val, y_val, config):
    params = model_params(config)
    params.update(objective="multiclass", num_class=3, class_weight="balanced")
    model = LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return model


def fit_low_nonnegative_expert(x_train, y_train, x_val, y_val, config):
    params = model_params(config)
    params["objective"] = "regression_l1"
    model = LGBMRegressor(**params)
    if len(y_val) >= 8:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
    else:
        model.fit(x_train, y_train, callbacks=[log_evaluation(0)])
    return model


def supply_labels(actual: np.ndarray, low_threshold: float) -> np.ndarray:
    labels = np.full(len(actual), 2, dtype=int)
    labels[(actual >= 0.0) & (actual <= low_threshold)] = 1
    labels[actual < 0.0] = 0
    return labels


def apply_supply_route(
    baseline: np.ndarray,
    probabilities: np.ndarray,
    low_prediction: np.ndarray,
    beta_negative: float,
    beta_low: float,
    negative_activation_threshold: float,
) -> np.ndarray:
    denominator = max(1.0 - negative_activation_threshold, 1e-6)
    negative_activation = np.clip(
        (probabilities[:, 0] - negative_activation_threshold) / denominator, 0.0, 1.0
    )
    negative_response = negative_activation * (PRICE_FLOOR - baseline)
    low_response = probabilities[:, 1] * (low_prediction - baseline)
    return baseline + beta_negative * negative_response + beta_low * low_response


def select_betas(
    actual: np.ndarray,
    baseline: np.ndarray,
    probabilities: np.ndarray,
    low_prediction: np.ndarray,
    low_threshold: float,
) -> tuple[float, float, float, list[dict]]:
    negative = actual < 0.0
    low_nonnegative = (actual >= 0.0) & (actual <= low_threshold)
    negative_betas = BETAS if negative.sum() >= 8 else (0.0,)
    low_betas = BETAS if low_nonnegative.sum() >= 8 else (0.0,)
    rows = []
    baseline_mae = float(np.mean(np.abs(baseline - actual)))
    thresholds = NEGATIVE_ACTIVATION_THRESHOLDS if negative.sum() >= 8 else (0.0,)
    for beta_negative, beta_low, negative_threshold in product(
        negative_betas, low_betas, thresholds
    ):
        prediction = apply_supply_route(
            baseline, probabilities, low_prediction, beta_negative, beta_low, negative_threshold
        )
        absolute = np.abs(prediction - actual)
        overall_mae = float(absolute.mean())
        negative_mae = float(absolute[negative].mean()) if negative.any() else overall_mae
        low_mae = (
            float(absolute[low_nonnegative].mean()) if low_nonnegative.any() else overall_mae
        )
        score = overall_mae
        if negative.sum() >= 8:
            score += 0.15 * negative_mae
        if low_nonnegative.sum() >= 8:
            score += 0.10 * low_mae
        rows.append(
            {
                "beta_negative": beta_negative,
                "beta_low": beta_low,
                "negative_activation_threshold": negative_threshold,
                "score": score,
                "mae": overall_mae,
                "overall_non_degrading": overall_mae <= baseline_mae + 1e-9,
                "negative_mae": negative_mae,
                "low_nonnegative_mae": low_mae,
            }
        )
    feasible = [row for row in rows if row["overall_non_degrading"]]
    selected = min(
        feasible,
        key=lambda row: (
            row["score"],
            row["beta_negative"] + row["beta_low"],
            row["negative_activation_threshold"],
        ),
    )
    return (
        float(selected["beta_negative"]),
        float(selected["beta_low"]),
        float(selected["negative_activation_threshold"]),
        rows,
    )


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual_negative = actual < 0.0
    predicted_negative = predicted < 0.0
    true_positive = int(np.sum(actual_negative & predicted_negative))
    false_positive = int(np.sum(~actual_negative & predicted_negative))
    false_negative = int(np.sum(actual_negative & ~predicted_negative))
    true_negative = int(np.sum(~actual_negative & ~predicted_negative))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    return {
        "negative_precision": precision,
        "negative_recall": recall,
        "negative_f1": f1,
        "negative_false_positive_rate": false_positive_rate,
        "negative_true_positive": true_positive,
        "negative_false_positive": false_positive,
        "negative_false_negative": false_negative,
        "negative_true_negative": true_negative,
    }


def summarize(metrics: pd.DataFrame, daily: pd.DataFrame, candidate: str, baseline: str) -> dict:
    result = {}
    lower_is_better = (
        "mae",
        "extreme_low_mae",
        "negative_price_mae",
        "negative_congestion_mae",
        "positive_congestion_mae",
        "negative_false_positive_rate",
    )
    higher_is_better = ("negative_recall", "negative_precision", "negative_f1")
    for name in lower_is_better + higher_is_better:
        pivot = metrics.pivot(index="fold", columns="mode", values=name).dropna()
        gain = pivot[baseline] - pivot[candidate] if name in lower_is_better else pivot[candidate] - pivot[baseline]
        result[name] = {
            "valid_folds": int(len(gain)),
            "improved_fold_count": int((gain > 0).sum()),
            "mean_improvement": float(gain.mean()) if len(gain) else None,
            "improvement_by_fold": {str(int(k)): float(v) for k, v in gain.items()},
        }
    pivot = daily[daily["mode"].isin([candidate, baseline])].pivot(
        index=["fold", "delivery_day"], columns="mode", values="mae"
    ).dropna()
    delta = pivot[candidate] - pivot[baseline]
    result["daily_mae"] = {
        "n_days": int(len(delta)),
        "candidate_minus_baseline": float(delta.mean()),
        "candidate_better_day_fraction": float((delta < 0).mean()),
        "wilcoxon_two_sided_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--carrier-cache", type=Path, required=True)
    parser.add_argument("--carrier-column", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels, usecols=["time", "congestion_state"])
    carrier = pd.read_csv(args.carrier_cache)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    labels["time"] = pd.to_datetime(labels["time"])
    carrier["time"] = pd.to_datetime(carrier["time"])
    frame = add_supply_features(
        add_features(frame.merge(labels, on="time", how="left").merge(carrier, on="time", how="left"))
    )

    congestion_columns = manifest["congestion_features"]
    blind_columns = [
        name for name in manifest["features"]
        if name not in congestion_columns and name != "price_day_ahead"
    ]
    blind_columns += [
        "slot_sin", "slot_cos", "slot_norm", "price_day_ahead_energy_proxy", args.carrier_column
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
    if args.carrier_column not in carrier.columns:
        raise ValueError(f"missing carrier column: {args.carrier_column}")

    x_blind = frame[blind_columns].to_numpy(float)
    x_full = frame[full_columns].to_numpy(float)
    x_supply = frame[supply_columns].to_numpy(float)
    actual = frame["price_real"].to_numpy(float)
    carrier_prediction = frame[args.carrier_column].to_numpy(float)
    residual = actual - carrier_prediction
    congestion_state = frame["congestion_state"].fillna(-1).to_numpy(int)

    fold_rows, prediction_rows, calibration_rows = [], [], []
    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (day_mask(frame, fold[name]) for name in ("train", "val", "test"))
        labeled_train = train & (congestion_state >= 0)
        labeled_val = val & (congestion_state >= 0)
        nonnegative_train_prices = actual[train][actual[train] >= 0.0]
        if len(nonnegative_train_prices) < 8:
            raise ValueError(f"fold {fold_index} has too few nonnegative training prices")
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
            x_full[labeled_train], congestion_state[labeled_train],
            x_full[labeled_val], congestion_state[labeled_val], config
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
        beta_negative, beta_low, negative_threshold, beta_candidates = select_betas(
            actual[val], physical_val, p_supply_val, low_prediction_val, low_threshold
        )

        supply_only_test = apply_supply_route(
            blind_test, p_supply_test, low_prediction_test, beta_negative, beta_low, negative_threshold
        )
        dual_route_test = apply_supply_route(
            physical_test, p_supply_test, low_prediction_test, beta_negative, beta_low, negative_threshold
        )
        calibration_rows.append(
            {
                "fold": fold_index,
                "low_threshold_train_q10": low_threshold,
                "train_supply_state_counts": {
                    SUPPLY_STATE_NAMES[i]: int(np.sum(train_supply_state == i)) for i in range(3)
                },
                "val_supply_state_counts": {
                    SUPPLY_STATE_NAMES[i]: int(np.sum(val_supply_state == i)) for i in range(3)
                },
                "selected_beta_negative": beta_negative,
                "selected_beta_low": beta_low,
                "selected_negative_activation_threshold": negative_threshold,
                "validation_candidates": beta_candidates,
            }
        )

        predictions = {
            "carrier_blind": blind_test,
            "physical_congestion_gate": physical_test,
            "supply_state_route_only": supply_only_test,
            "supply_congestion_dual_route": dual_route_test,
        }
        test_positions = np.flatnonzero(test)
        for mode, predicted in predictions.items():
            metrics = error_metrics(
                actual[test], predicted, actual[test] <= low_threshold, congestion_state[test]
            )
            metrics.update(classification_metrics(actual[test], predicted))
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
                        "p_supply_low_nonnegative": p_supply_test[:, 1],
                        "p_supply_regular": p_supply_test[:, 2],
                        "p_congestion_non_neutral": 1.0 - p_congestion_test[:, 1],
                    }
                )
            )

    metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = predictions.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(
        mae=("absolute_error", "mean")
    )
    comparisons = {
        "dual_vs_blind": summarize(metrics, daily, "supply_congestion_dual_route", "carrier_blind"),
        "dual_vs_physical": summarize(
            metrics, daily, "supply_congestion_dual_route", "physical_congestion_gate"
        ),
        "dual_vs_supply_only": summarize(
            metrics, daily, "supply_congestion_dual_route", "supply_state_route_only"
        ),
        "physical_vs_blind": summarize(metrics, daily, "physical_congestion_gate", "carrier_blind"),
    }
    primary = comparisons["dual_vs_blind"]
    overall_ok = primary["mae"]["mean_improvement"] > 0
    negative_ok = (
        primary["negative_price_mae"]["valid_folds"] == 0
        or primary["negative_price_mae"]["mean_improvement"] > 0
    )
    report = {
        "status": "complete",
        "dual_route_gate": "PASS" if overall_ok and negative_ok else "FAIL",
        "mechanism": (
            "A congestion-free three-state supply router separates negative-price, low-nonnegative, "
            "and regular market states. The negative state is anchored to the Zhejiang market floor "
            "of -200 RMB/MWh; the low-nonnegative state uses a learned conditional expert. An "
            "independent non-neutral congestion probability gates the Full-minus-Blind nodal response."
        ),
        "equation": (
            "y_physical = y_blind + (1-p_congestion_neutral)*(y_full-y_blind); "
            "g_negative = max(p_supply_negative-tau,0)/(1-tau); "
            "y_final = y_physical + beta_negative*g_negative*(-200-y_physical) + "
            "beta_low*p_supply_low*(y_low-y_physical)"
        ),
        "calibration_rule": (
            "Both beta values are selected on validation data only. The negative branch is disabled "
            "when validation contains fewer than 8 negative-price points; the low branch follows the "
            "same minimum-count rule."
        ),
        "features": {
            "strict_blind": blind_columns,
            "congestion_increment": congestion_columns,
            "supply_state": supply_columns,
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
        "carrier_column": args.carrier_column,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    (args.output / "dual_route_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"gate": report["dual_route_gate"], "calibration": calibration_rows, "comparisons": comparisons},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
