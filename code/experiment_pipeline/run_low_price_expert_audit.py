from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import wilcoxon

from run_chronos2_strong_carrier_audit import fit_residual, model_params
from run_physical_soft_route_price_audit import add_features, day_mask, error_metrics, fit_router


BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def fit_low_classifier(x_train, y_train, x_val, y_val, config):
    params = model_params(config)
    params.update(objective="binary", class_weight="balanced")
    model = LGBMClassifier(**params)
    model.fit(
        x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="binary_logloss",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return model


def fit_low_expert(x_train, y_train, x_val, y_val, config):
    params = model_params(config)
    params["objective"] = "regression_l1"
    model = LGBMRegressor(**params)
    use_val = len(y_val) >= 8
    if use_val:
        model.fit(
            x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
    else:
        model.fit(x_train, y_train, callbacks=[log_evaluation(0)])
    return model


def select_beta(actual, baseline, low_prediction, probability, low_mask):
    rows = []
    for beta in BETAS:
        prediction = baseline + beta * probability * (low_prediction - baseline)
        overall = float(np.mean(np.abs(prediction - actual)))
        low_mae = float(np.mean(np.abs(prediction[low_mask] - actual[low_mask]))) if low_mask.any() else overall
        score = overall + 0.25 * low_mae
        rows.append({"beta": beta, "score": score, "mae": overall, "low_mae": low_mae})
    selected = min(rows, key=lambda row: (row["score"], row["beta"]))
    return float(selected["beta"]), rows


def summarize(metrics: pd.DataFrame, daily: pd.DataFrame, candidate: str, baseline: str) -> dict:
    result = {}
    for name in (
        "mae", "extreme_low_mae", "negative_price_mae", "negative_sign_recall",
        "negative_congestion_mae", "positive_congestion_mae",
    ):
        pivot = metrics.pivot(index="fold", columns="mode", values=name).dropna()
        gain = pivot[baseline] - pivot[candidate]
        if name == "negative_sign_recall":
            gain = -gain
        result[name] = {
            "valid_folds": int(len(gain)), "improved_fold_count": int((gain > 0).sum()),
            "mean_improvement": float(gain.mean()) if len(gain) else None,
            "improvement_by_fold": {str(int(k)): float(v) for k, v in gain.items()},
        }
    pivot = daily[daily["mode"].isin([candidate, baseline])].pivot(
        index=["fold", "delivery_day"], columns="mode", values="mae"
    ).dropna()
    delta = pivot[candidate] - pivot[baseline]
    result["daily_mae"] = {
        "n_days": int(len(delta)), "candidate_minus_baseline": float(delta.mean()),
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
    frame = add_features(frame.merge(labels, on="time", how="left").merge(carrier, on="time", how="left"))

    congestion = manifest["congestion_features"]
    blind_columns = [name for name in manifest["features"] if name not in congestion and name != "price_day_ahead"]
    if args.carrier_column not in carrier.columns:
        raise ValueError(f"missing carrier column: {args.carrier_column}")
    blind_columns += ["slot_sin", "slot_cos", "slot_norm", "price_day_ahead_energy_proxy", args.carrier_column]
    full_columns = blind_columns + congestion
    x_blind = frame[blind_columns].to_numpy(float)
    x_full = frame[full_columns].to_numpy(float)
    actual = frame["price_real"].to_numpy(float)
    carrier_prediction = frame[args.carrier_column].to_numpy(float)
    residual = actual - carrier_prediction
    state = frame["congestion_state"].fillna(-1).to_numpy(int)

    fold_rows, prediction_rows, calibration = [], [], []
    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (day_mask(frame, fold[name]) for name in ("train", "val", "test"))
        labeled_train = train & (state >= 0)
        labeled_val = val & (state >= 0)
        threshold = float(np.quantile(actual[train], 0.10))
        low_train = train & (actual <= threshold)
        low_val = val & (actual <= threshold)
        low_test = test & (actual <= threshold)

        blind = fit_residual(x_blind[train], residual[train], x_blind[val], residual[val], config)
        full = fit_residual(x_full[train], residual[train], x_full[val], residual[val], config)
        blind_val = carrier_prediction[val] + blind.predict(x_blind[val])
        full_val = carrier_prediction[val] + full.predict(x_full[val])
        blind_test = carrier_prediction[test] + blind.predict(x_blind[test])
        full_test = carrier_prediction[test] + full.predict(x_full[test])

        state_router = fit_router(
            x_full[labeled_train], state[labeled_train], x_full[labeled_val], state[labeled_val], config
        )
        p_state_val = state_router.predict_proba(x_full[val])
        p_state_test = state_router.predict_proba(x_full[test])
        physical_val = blind_val + (1.0 - p_state_val[:, 1]) * (full_val - blind_val)
        physical_test = blind_test + (1.0 - p_state_test[:, 1]) * (full_test - blind_test)

        low_label_train = (actual[train] <= threshold).astype(int)
        low_label_val = (actual[val] <= threshold).astype(int)
        low_router = fit_low_classifier(x_full[train], low_label_train, x_full[val], low_label_val, config)
        p_low_val = low_router.predict_proba(x_full[val])[:, 1]
        p_low_test = low_router.predict_proba(x_full[test])[:, 1]
        low_expert = fit_low_expert(
            x_full[low_train], actual[low_train], x_full[low_val], actual[low_val], config
        )
        low_pred_val = low_expert.predict(x_full[val])
        low_pred_test = low_expert.predict(x_full[test])
        beta, candidates = select_beta(actual[val], physical_val, low_pred_val, p_low_val, actual[val] <= threshold)
        if low_val.sum() < 8:
            beta = 0.0
        final_test = physical_test + beta * p_low_test * (low_pred_test - physical_test)
        calibration.append(
            {
                "fold": fold_index, "threshold": threshold, "train_low_count": int(low_train.sum()),
                "val_low_count": int(low_val.sum()), "test_low_count": int(low_test.sum()),
                "selected_beta": beta, "validation_candidates": candidates,
            }
        )

        predictions = {
            "carrier_blind": blind_test,
            "physical_congestion_gate": physical_test,
            "physical_gate_low_expert": final_test,
        }
        test_mask = np.flatnonzero(test)
        for mode, predicted in predictions.items():
            fold_rows.append(
                {
                    "fold": fold_index, "mode": mode, "low_threshold_train_q10": threshold,
                    **error_metrics(actual[test], predicted, actual[test] <= threshold, state[test]),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_index,
                        "delivery_day": frame.loc[test, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy(),
                        "time": frame.loc[test, "time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                        "mode": mode, "actual": actual[test], "predicted": predicted,
                        "absolute_error": np.abs(predicted - actual[test]),
                        "low_price": actual[test] <= threshold, "congestion_state": state[test_mask],
                    }
                )
            )

    metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = predictions.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(mae=("absolute_error", "mean"))
    final_vs_blind = summarize(metrics, daily, "physical_gate_low_expert", "carrier_blind")
    final_vs_physical = summarize(metrics, daily, "physical_gate_low_expert", "physical_congestion_gate")
    overall_ok = final_vs_blind["mae"]["improved_fold_count"] >= 3 and final_vs_blind["mae"]["mean_improvement"] > 0
    tail_ok = final_vs_blind["extreme_low_mae"]["improved_fold_count"] >= 2 and final_vs_blind["extreme_low_mae"]["mean_improvement"] > 0
    report = {
        "status": "complete", "low_price_expert_gate": "PASS" if overall_ok and tail_ok else "FAIL",
        "gate_rule": "Final route improves Blind overall in >=3/4 folds and Extreme-Low in >=2 valid folds with positive mean gains.",
        "beta_selection": "Validation-only grid minimizes overall MAE + 0.25 * low-state MAE; ties prefer the smaller beta. The low expert is disabled when validation has fewer than 8 low-state points.",
        "calibration": calibration,
        "input_hashes": {
            "data": sha256(args.data), "labels": sha256(args.labels), "manifest": sha256(args.manifest),
            "folds": sha256(args.folds), "config": sha256(args.config), "carrier_cache": sha256(args.carrier_cache),
            "code": sha256(Path(__file__)),
        },
        "carrier_column": args.carrier_column,
        "comparisons": {
            "final_vs_blind": final_vs_blind,
            "final_vs_physical_congestion_gate": final_vs_physical,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    (args.output / "low_price_expert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report["low_price_expert_gate"], "calibration": calibration, "vs_blind": final_vs_blind, "vs_physical": final_vs_physical}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
