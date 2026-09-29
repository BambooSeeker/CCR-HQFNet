from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss, recall_score


VARIANTS = ("current_blind", "energy_only", "energy_plus_congestion")
LABELS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest.upper()


def load_realized_labels(source_dir: Path, plant_code: str) -> pd.DataFrame:
    columns = ["time", "code", "price_real_energy", "price_real_cong", "price_real_load"]
    parts = []
    for path in sorted(source_dir.glob("input_price_predict_*.csv")):
        month = path.stem.rsplit("_", 1)[-1]
        if not ("202405" <= month <= "202607"):
            continue
        frame = pd.read_csv(path, usecols=columns)
        selected = frame.loc[frame["code"].eq(plant_code), columns].copy()
        selected["source_month"] = month
        parts.append(selected)
    labels = pd.concat(parts, ignore_index=True)
    labels["time"] = pd.to_datetime(labels["time"])
    return labels.drop(columns="code").drop_duplicates("time")


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    slot = ((result["time"].dt.hour * 2 + result["time"].dt.minute // 30 - 1) % 48).to_numpy()
    result["slot_sin"] = np.sin(2 * np.pi * slot / 48.0)
    result["slot_cos"] = np.cos(2 * np.pi * slot / 48.0)
    result["slot_norm"] = slot / 47.0
    result["price_day_ahead_energy_proxy"] = result["price_day_ahead"] - result["price_day_ahead_cong"]
    return result


def state_label(values: pd.Series, deadband: float) -> np.ndarray:
    output = np.full(len(values), -1, dtype=int)
    available = values.notna().to_numpy()
    raw = values.fillna(0.0).to_numpy(float)
    output[available & (raw < -deadband)] = 0
    output[available & (np.abs(raw) <= deadband)] = 1
    output[available & (raw > deadband)] = 2
    return output


def row_mask(frame: pd.DataFrame, days: list[str]) -> np.ndarray:
    return frame["delivery_day"].isin(pd.to_datetime(days)).to_numpy()


def fit_predict(x_train, y_train, x_val, y_val, x_test, config):
    params = dict(config["lightgbm"])
    params.update(
        objective="multiclass", num_class=3, class_weight="balanced",
        random_state=int(config["seed"]), n_jobs=-1, deterministic=True, force_col_wise=True,
    )
    model = LGBMClassifier(**params)
    callbacks = [log_evaluation(0)]
    if len(np.unique(y_val)) > 1:
        callbacks.insert(0, early_stopping(40, verbose=False))
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="multi_logloss", callbacks=callbacks)
    else:
        model.fit(x_train, y_train, callbacks=callbacks)
    probability = model.predict_proba(x_test)
    prediction = np.argmax(probability, axis=1)
    return prediction, probability, int(getattr(model, "best_iteration_", 0) or 0)


def metrics(actual, prediction, probability):
    recalls = recall_score(actual, prediction, labels=LABELS, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, labels=LABELS, average="macro", zero_division=0)),
        "log_loss": float(log_loss(actual, probability, labels=LABELS)),
        "recall_negative": float(recalls[0]),
        "recall_neutral": float(recalls[1]),
        "recall_positive": float(recalls[2]),
        "confusion_matrix": confusion_matrix(actual, prediction, labels=LABELS).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    frame = frame.merge(load_realized_labels(args.raw_source_dir, config["plant_code"]), on="time", how="left")
    frame = add_features(frame)
    frame["congestion_state"] = state_label(frame["price_real_cong"], float(config["state_deadband"]))

    congestion = manifest["congestion_features"]
    common = [name for name in manifest["features"] if name not in congestion and name != "price_day_ahead"]
    common += ["slot_sin", "slot_cos", "slot_norm"]
    feature_sets = {
        "current_blind": common + ["price_day_ahead"],
        "energy_only": common + ["price_day_ahead_energy_proxy"],
        "energy_plus_congestion": common + ["price_day_ahead_energy_proxy"] + congestion,
    }

    fold_rows = []
    prediction_rows = []
    for fold_index, fold in enumerate(folds, start=1):
        masks = {name: row_mask(frame, fold[name]) & frame["congestion_state"].ge(0).to_numpy() for name in ("train", "val", "test")}
        coverage = {
            name: float(masks[name].sum() / max(row_mask(frame, fold[name]).sum(), 1))
            for name in masks
        }
        y = frame["congestion_state"].to_numpy(int)
        if masks["train"].sum() == 0 or masks["test"].sum() == 0:
            fold_rows.append({"fold": fold_index, "variant": "unavailable", "labeled_test_points": int(masks["test"].sum()), **{f"{k}_label_coverage": v for k, v in coverage.items()}})
            continue

        for variant in VARIANTS:
            columns = feature_sets[variant]
            prediction, probability, best_iteration = fit_predict(
                frame.loc[masks["train"], columns].to_numpy(float), y[masks["train"]],
                frame.loc[masks["val"], columns].to_numpy(float), y[masks["val"]],
                frame.loc[masks["test"], columns].to_numpy(float), config,
            )
            actual = y[masks["test"]]
            result = metrics(actual, prediction, probability)
            fold_rows.append({
                "fold": fold_index, "variant": variant, "labeled_test_points": int(len(actual)),
                **{f"{k}_label_coverage": v for k, v in coverage.items()},
                **result, "best_iteration": best_iteration,
            })
            test_frame = frame.loc[masks["test"], ["time", "delivery_day", "price_real", "price_real_cong"]].copy()
            test_frame["fold"] = fold_index
            test_frame["variant"] = variant
            test_frame["actual_state"] = actual
            test_frame["predicted_state"] = prediction
            for index, label in enumerate(("negative", "neutral", "positive")):
                test_frame[f"probability_{label}"] = probability[:, index]
            prediction_rows.append(test_frame)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    valid = fold_metrics[fold_metrics["variant"].isin(VARIANTS) & fold_metrics["labeled_test_points"].ge(int(config["min_labeled_test_points"]))]

    comparisons = {}
    for candidate, baseline in (("energy_plus_congestion", "energy_only"), ("current_blind", "energy_only")):
        comparisons[f"{candidate}_vs_{baseline}"] = {}
        for metric_name, higher_better in (("macro_f1", True), ("balanced_accuracy", True), ("log_loss", False), ("recall_negative", True), ("recall_positive", True)):
            pivot = valid.pivot(index="fold", columns="variant", values=metric_name).dropna()
            improvement = pivot[candidate] - pivot[baseline]
            if not higher_better:
                improvement = -improvement
            comparisons[f"{candidate}_vs_{baseline}"][metric_name] = {
                "valid_folds": int(len(improvement)),
                "improved_fold_count": int((improvement > 0).sum()),
                "mean_improvement": float(improvement.mean()) if len(improvement) else None,
                "improvement_by_fold": {str(int(k)): float(v) for k, v in improvement.items()},
            }

    primary = comparisons["energy_plus_congestion_vs_energy_only"]
    valid_folds = primary["macro_f1"]["valid_folds"]
    required = 3 if valid_folds >= 4 else max(2, valid_folds)
    gate = (
        valid_folds >= 3
        and primary["macro_f1"]["improved_fold_count"] >= required
        and primary["log_loss"]["improved_fold_count"] >= required
        and (primary["recall_negative"]["improved_fold_count"] >= required or primary["recall_positive"]["improved_fold_count"] >= required)
    )

    available = frame[frame["congestion_state"].ge(0)].copy()
    q10 = float(available["price_real"].quantile(0.10))
    available["low_price"] = available["price_real"].le(q10)
    available["negative_price"] = available["price_real"].lt(0.0)
    conditional = {
        "rows": int(len(frame)),
        "realized_congestion_label_coverage": float(frame["congestion_state"].ge(0).mean()),
        "state_counts": {str(int(k)): int(v) for k, v in available["congestion_state"].value_counts().sort_index().items()},
        "state_shares": {str(int(k)): float(v) for k, v in available["congestion_state"].value_counts(normalize=True).sort_index().items()},
        "low_price_q10": q10,
        "low_price_state_shares": {str(int(k)): float(v) for k, v in available.loc[available["low_price"], "congestion_state"].value_counts(normalize=True).sort_index().items()},
        "negative_price_count": int(available["negative_price"].sum()),
        "negative_price_state_shares": {str(int(k)): float(v) for k, v in available.loc[available["negative_price"], "congestion_state"].value_counts(normalize=True).sort_index().items()},
        "correlation": available[["price_real", "price_day_ahead", "price_day_ahead_cong", "price_real_cong"]].corr().to_dict(),
    }

    report = {
        "status": "complete", "state_predictability_gate": "PASS" if gate else "FAIL",
        "gate_rule": "Energy+Congestion must improve macro-F1, log loss, and at least one tail recall in the required number of valid folds.",
        "valid_folds": valid_folds, "required_improved_folds": required,
        "input_hashes": {"data": sha256(args.data), "manifest": sha256(args.manifest), "folds": sha256(args.folds), "config": sha256(args.config), "code": sha256(Path(__file__))},
        "feature_sets": feature_sets, "conditional_audit": conditional, "comparisons": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "congestion_state_audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state_predictability_gate": report["state_predictability_gate"], "valid_folds": valid_folds, "required": required, "primary": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
