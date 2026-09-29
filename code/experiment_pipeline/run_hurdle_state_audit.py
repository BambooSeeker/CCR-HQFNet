from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, recall_score


LABELS = (0, 1, 2)
METHODS = ("flat", "hurdle")
VARIANTS = ("energy_only", "energy_plus_congestion")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    slot = ((result["time"].dt.hour * 2 + result["time"].dt.minute // 30 - 1) % 48).to_numpy()
    result["slot_sin"] = np.sin(2 * np.pi * slot / 48.0)
    result["slot_cos"] = np.cos(2 * np.pi * slot / 48.0)
    result["slot_norm"] = slot / 47.0
    result["price_day_ahead_energy_proxy"] = result["price_day_ahead"] - result["price_day_ahead_cong"]
    return result


def model_params(config: dict) -> dict:
    params = dict(config["lightgbm"])
    params.update(class_weight="balanced", random_state=int(config["seed"]), n_jobs=-1, deterministic=True, force_col_wise=True)
    return params


def fit_binary(x_train, y_train, x_val, y_val, x_test, config):
    model = LGBMClassifier(objective="binary", **model_params(config))
    callbacks = [log_evaluation(0)]
    if len(np.unique(y_val)) > 1:
        callbacks.insert(0, early_stopping(40, verbose=False))
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)
    else:
        model.fit(x_train, y_train, callbacks=callbacks)
    return model.predict_proba(x_test)[:, 1]


def fit_flat(x_train, y_train, x_val, y_val, x_test, config):
    model = LGBMClassifier(objective="multiclass", num_class=3, **model_params(config))
    model.fit(
        x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="multi_logloss",
        callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
    )
    return model.predict_proba(x_test)


def fit_hurdle(x_train, y_train, x_val, y_val, x_test, config):
    train_active = y_train != 1
    val_active = y_val != 1
    p_active = fit_binary(x_train, train_active.astype(int), x_val, val_active.astype(int), x_test, config)
    p_positive_given_active = fit_binary(
        x_train[train_active], (y_train[train_active] == 2).astype(int),
        x_val[val_active], (y_val[val_active] == 2).astype(int), x_test, config,
    )
    probability = np.column_stack([
        p_active * (1.0 - p_positive_given_active),
        1.0 - p_active,
        p_active * p_positive_given_active,
    ])
    return probability / probability.sum(axis=1, keepdims=True)


def evaluate(actual, probability):
    prediction = probability.argmax(axis=1)
    recalls = recall_score(actual, prediction, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(actual, prediction, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "log_loss": float(log_loss(actual, probability, labels=LABELS)),
        "recall_negative": float(recalls[0]), "recall_neutral": float(recalls[1]), "recall_positive": float(recalls[2]),
    }, prediction


def mask_days(frame, days):
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
    common = [name for name in manifest["features"] if name not in congestion and name != "price_day_ahead"]
    common += ["slot_sin", "slot_cos", "slot_norm", "price_day_ahead_energy_proxy"]
    columns = {"energy_only": common, "energy_plus_congestion": common + congestion}
    y = frame["congestion_state"].to_numpy(int)
    rows, predictions = [], []

    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (mask_days(frame, fold[name]) for name in ("train", "val", "test"))
        for variant in VARIANTS:
            features = frame[columns[variant]].to_numpy(float)
            for method in METHODS:
                probability = (
                    fit_flat(features[train], y[train], features[val], y[val], features[test], config)
                    if method == "flat"
                    else fit_hurdle(features[train], y[train], features[val], y[val], features[test], config)
                )
                result, prediction = evaluate(y[test], probability)
                rows.append({"fold": fold_index, "variant": variant, "method": method, **result})
                local = frame.loc[test, ["time", "delivery_day", "price_real", "price_real_cong"]].copy()
                local["fold"], local["variant"], local["method"] = fold_index, variant, method
                local["actual_state"], local["predicted_state"] = y[test], prediction
                predictions.append(local)

    metrics = pd.DataFrame(rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    comparisons = {}
    pairs = [
        (("energy_plus_congestion", "hurdle"), ("energy_only", "hurdle")),
        (("energy_plus_congestion", "hurdle"), ("energy_plus_congestion", "flat")),
    ]
    for candidate, baseline in pairs:
        name = f"{candidate[0]}_{candidate[1]}_vs_{baseline[0]}_{baseline[1]}"
        comparisons[name] = {}
        for metric_name, higher_better in (("macro_f1", True), ("balanced_accuracy", True), ("log_loss", False), ("recall_negative", True), ("recall_positive", True)):
            cand = metrics[(metrics.variant == candidate[0]) & (metrics.method == candidate[1])].set_index("fold")[metric_name]
            base = metrics[(metrics.variant == baseline[0]) & (metrics.method == baseline[1])].set_index("fold")[metric_name]
            improvement = cand - base if higher_better else base - cand
            comparisons[name][metric_name] = {
                "improved_fold_count": int((improvement > 0).sum()), "mean_improvement": float(improvement.mean()),
                "improvement_by_fold": {str(int(k)): float(v) for k, v in improvement.items()},
            }

    primary = comparisons["energy_plus_congestion_hurdle_vs_energy_only_hurdle"]
    gate = (
        primary["macro_f1"]["improved_fold_count"] >= 3
        and primary["log_loss"]["improved_fold_count"] >= 3
        and (primary["recall_negative"]["improved_fold_count"] >= 3 or primary["recall_positive"]["improved_fold_count"] >= 3)
    )
    report = {
        "status": "complete", "hurdle_signal_gate": "PASS" if gate else "FAIL",
        "gate_rule": "Energy+Congestion hurdle routing improves macro-F1, log loss, and at least one directional recall in >=3/4 complete-label folds.",
        "input_hashes": {"data": sha256(args.data), "labels": sha256(args.labels), "manifest": sha256(args.manifest), "folds": sha256(args.folds), "config": sha256(args.config), "code": sha256(Path(__file__))},
        "comparisons": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    prediction_frame.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "hurdle_state_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"hurdle_signal_gate": report["hurdle_signal_gate"], "primary": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
