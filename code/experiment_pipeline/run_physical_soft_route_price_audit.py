from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold


MODES = ("energy_base", "direct_full", "route_blind", "route_full")
LABELS = (0, 1, 2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    slot = ((result["time"].dt.hour * 2 + result["time"].dt.minute // 30 - 1) % 48).to_numpy()
    result["slot"] = slot
    result["slot_sin"] = np.sin(2 * np.pi * slot / 48.0)
    result["slot_cos"] = np.cos(2 * np.pi * slot / 48.0)
    result["slot_norm"] = slot / 47.0
    result["price_day_ahead_energy_proxy"] = result["price_day_ahead"] - result["price_day_ahead_cong"]
    return result


def model_params(config: dict) -> dict:
    params = dict(config["lightgbm"])
    params.update(random_state=int(config["seed"]), n_jobs=-1, deterministic=True, force_col_wise=True)
    return params


def fit_regressor(x_train, y_train, x_val, y_val, config, use_early_stopping=True):
    params = model_params(config)
    params["objective"] = "regression_l1"
    model = LGBMRegressor(**params)
    if use_early_stopping:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="l1", callbacks=[early_stopping(40, verbose=False), log_evaluation(0)])
    else:
        model.fit(x_train, y_train, callbacks=[log_evaluation(0)])
    return model


def fit_router(x_train, y_train, x_val, y_val, config):
    params = model_params(config)
    params.update(objective="multiclass", num_class=3, class_weight="balanced")
    model = LGBMClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="multi_logloss", callbacks=[early_stopping(40, verbose=False), log_evaluation(0)])
    return model


def oof_base_predictions(x_train, y_train, config):
    prediction = np.empty(len(y_train), dtype=float)
    for fit_idx, hold_idx in KFold(n_splits=5, shuffle=False).split(x_train):
        model = fit_regressor(x_train[fit_idx], y_train[fit_idx], x_train[hold_idx], y_train[hold_idx], config, use_early_stopping=False)
        prediction[hold_idx] = model.predict(x_train[hold_idx])
    return prediction


def error_metrics(actual, predicted, low_mask, state):
    absolute = np.abs(predicted - actual)
    negative = actual < 0.0
    result = {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "smape": float(np.mean(200.0 * absolute / (np.abs(actual) + np.abs(predicted) + 1e-6))),
        "extreme_low_mae": float(absolute[low_mask].mean()) if low_mask.any() else None,
        "extreme_low_count": int(low_mask.sum()),
        "negative_price_mae": float(absolute[negative].mean()) if negative.any() else None,
        "negative_price_count": int(negative.sum()),
        "negative_sign_recall": float(np.mean(predicted[negative] < 0.0)) if negative.any() else None,
    }
    for label, name in ((0, "negative_congestion"), (1, "neutral_congestion"), (2, "positive_congestion")):
        mask = state == label
        result[f"{name}_mae"] = float(absolute[mask].mean()) if mask.any() else None
        result[f"{name}_count"] = int(mask.sum())
    return result


def day_mask(frame, days):
    return frame["delivery_day"].isin(pd.to_datetime(days)).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels, usecols=["time", "price_real_cong", "congestion_state"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    labels["time"] = pd.to_datetime(labels["time"])
    frame = add_features(frame.merge(labels, on="time", how="left"))

    congestion = manifest["congestion_features"]
    energy_columns = [name for name in manifest["features"] if name not in congestion and name != "price_day_ahead"]
    energy_columns += ["slot_sin", "slot_cos", "slot_norm", "price_day_ahead_energy_proxy"]
    full_columns = energy_columns + congestion
    x_energy = frame[energy_columns].to_numpy(float)
    x_full = frame[full_columns].to_numpy(float)
    y = frame["price_real"].to_numpy(float)
    state = frame["congestion_state"].fillna(-1).to_numpy(int)

    fold_rows, prediction_rows, daily_rows = [], [], []
    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (day_mask(frame, fold[name]) for name in ("train", "val", "test"))
        labeled_train = train & (state >= 0)
        labeled_val = val & (state >= 0)

        base_oof = oof_base_predictions(x_energy[train], y[train], config)
        base = fit_regressor(x_energy[train], y[train], x_energy[val], y[val], config)
        base_val = base.predict(x_energy[val])
        base_test = base.predict(x_energy[test])
        direct = fit_regressor(x_full[train], y[train], x_full[val], y[val], config)
        direct_test = direct.predict(x_full[test])

        train_positions = np.flatnonzero(train)
        labeled_train_positions = np.flatnonzero(labeled_train)
        position_lookup = {global_index: local_index for local_index, global_index in enumerate(train_positions)}
        labeled_local = np.array([position_lookup[index] for index in labeled_train_positions], dtype=int)
        train_residual = y[train] - base_oof
        val_residual = y[val] - base_val
        expert_test = []
        for label in LABELS:
            local_mask = state[labeled_train] == label
            expert_train_x = x_energy[labeled_train][local_mask]
            expert_train_y = train_residual[labeled_local][local_mask]
            val_mask = state[labeled_val] == label
            if val_mask.sum() >= 16:
                expert = fit_regressor(expert_train_x, expert_train_y, x_energy[labeled_val][val_mask], val_residual[state[val] == label], config)
            else:
                expert = fit_regressor(expert_train_x, expert_train_y, expert_train_x, expert_train_y, config, use_early_stopping=False)
            expert_test.append(expert.predict(x_energy[test]))
        expert_test = np.column_stack(expert_test)

        router_blind = fit_router(x_energy[labeled_train], state[labeled_train], x_energy[labeled_val], state[labeled_val], config)
        router_full = fit_router(x_full[labeled_train], state[labeled_train], x_full[labeled_val], state[labeled_val], config)
        p_blind = router_blind.predict_proba(x_energy[test])
        p_full = router_full.predict_proba(x_full[test])
        predictions = {
            "energy_base": base_test,
            "direct_full": direct_test,
            "route_blind": base_test + np.sum(p_blind * expert_test, axis=1),
            "route_full": base_test + np.sum(p_full * expert_test, axis=1),
        }

        low_threshold = float(np.quantile(y[train], 0.10))
        low_mask = y[test] <= low_threshold
        test_state = state[test]
        test_days = frame.loc[test, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy()
        test_times = frame.loc[test, "time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        for mode, predicted in predictions.items():
            result = error_metrics(y[test], predicted, low_mask, test_state)
            fold_rows.append({"fold": fold_index, "mode": mode, "low_threshold_train_q10": low_threshold, **result})
            local = pd.DataFrame({
                "fold": fold_index, "delivery_day": test_days, "time": test_times, "mode": mode,
                "actual": y[test], "predicted": predicted, "absolute_error": np.abs(predicted - y[test]),
                "low_price": low_mask, "congestion_state": test_state,
            })
            prediction_rows.append(local)
            daily_rows.append(local.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(mae=("absolute_error", "mean")))

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = pd.concat(daily_rows, ignore_index=True)
    comparisons = {}
    for candidate, baseline in (("route_full", "route_blind"), ("route_full", "energy_base"), ("direct_full", "energy_base")):
        name = f"{candidate}_vs_{baseline}"
        comparisons[name] = {}
        for metric_name in ("mae", "extreme_low_mae", "negative_price_mae", "negative_sign_recall", "negative_congestion_mae", "positive_congestion_mae"):
            pivot = fold_metrics.pivot(index="fold", columns="mode", values=metric_name).dropna()
            improvement = pivot[baseline] - pivot[candidate]
            if metric_name == "negative_sign_recall":
                improvement = -improvement
            comparisons[name][metric_name] = {
                "valid_folds": int(len(improvement)), "improved_fold_count": int((improvement > 0).sum()),
                "mean_improvement": float(improvement.mean()) if len(improvement) else None,
                "improvement_by_fold": {str(int(k)): float(v) for k, v in improvement.items()},
            }
        subset = daily[daily["mode"].isin([candidate, baseline])].pivot(index=["fold", "delivery_day"], columns="mode", values="mae").dropna()
        delta = subset[candidate] - subset[baseline]
        comparisons[name]["daily_mae"] = {
            "n_days": int(len(delta)), "candidate_minus_baseline": float(delta.mean()),
            "candidate_better_day_fraction": float((delta < 0).mean()),
            "wilcoxon_two_sided_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0,
        }

    primary = comparisons["route_full_vs_route_blind"]
    main_pass = any(primary[name]["improved_fold_count"] >= 3 and primary[name]["mean_improvement"] > 0 for name in ("mae", "extreme_low_mae"))
    negative_valid = primary["negative_price_mae"]["valid_folds"]
    negative_ok = negative_valid == 0 or primary["negative_price_mae"]["improved_fold_count"] >= max(1, negative_valid - 1)
    report = {
        "status": "complete", "physical_route_pilot_gate": "PASS" if main_pass and negative_ok else "FAIL",
        "gate_rule": "Route-Full improves overall or Extreme-Low MAE in >=3/4 folds with positive mean gain, without majority negative-price MAE degradation.",
        "input_hashes": {"data": sha256(args.data), "labels": sha256(args.labels), "manifest": sha256(args.manifest), "folds": sha256(args.folds), "config": sha256(args.config), "code": sha256(Path(__file__))},
        "features": {"energy": energy_columns, "congestion": congestion}, "comparisons": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    (args.output / "physical_route_price_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"physical_route_pilot_gate": report["physical_route_pilot_gate"], "primary": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
