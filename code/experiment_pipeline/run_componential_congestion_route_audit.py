from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_physical_soft_route_price_audit import add_features, day_mask, error_metrics, fit_regressor, fit_router


MODES = (
    "energy_total_baseline", "direct_full_total", "component_direct_blind",
    "component_direct_full", "component_route_blind", "component_route_full",
)
LABELS = (0, 1, 2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def daily_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["fold", "delivery_day", "mode"], as_index=False).agg(mae=("absolute_error", "mean"))


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
    total = frame["price_real"].to_numpy(float)
    congestion_target = frame["price_real_cong"].to_numpy(float)
    state = frame["congestion_state"].fillna(-1).to_numpy(int)

    fold_rows, prediction_rows = [], []
    for fold_index, fold in enumerate(folds, start=1):
        train, val, test = (day_mask(frame, fold[name]) for name in ("train", "val", "test"))
        labeled_train = train & (state >= 0) & np.isfinite(congestion_target)
        labeled_val = val & (state >= 0) & np.isfinite(congestion_target)
        noncongestion_target = total - congestion_target

        energy_total = fit_regressor(x_energy[train], total[train], x_energy[val], total[val], config)
        direct_total = fit_regressor(x_full[train], total[train], x_full[val], total[val], config)
        component_base = fit_regressor(
            x_energy[labeled_train], noncongestion_target[labeled_train],
            x_energy[labeled_val], noncongestion_target[labeled_val], config,
        )
        component_base_test = component_base.predict(x_energy[test])

        congestion_blind = fit_regressor(
            x_energy[labeled_train], congestion_target[labeled_train],
            x_energy[labeled_val], congestion_target[labeled_val], config,
        )
        congestion_full = fit_regressor(
            x_full[labeled_train], congestion_target[labeled_train],
            x_full[labeled_val], congestion_target[labeled_val], config,
        )
        direct_congestion_blind = congestion_blind.predict(x_energy[test])
        direct_congestion_full = congestion_full.predict(x_full[test])

        router_blind = fit_router(x_energy[labeled_train], state[labeled_train], x_energy[labeled_val], state[labeled_val], config)
        router_full = fit_router(x_full[labeled_train], state[labeled_train], x_full[labeled_val], state[labeled_val], config)
        p_blind = router_blind.predict_proba(x_energy[test])
        p_full = router_full.predict_proba(x_full[test])
        expert_blind, expert_full = [], []
        for label in LABELS:
            train_state = labeled_train & (state == label)
            val_state = labeled_val & (state == label)
            blind = fit_regressor(
                x_energy[train_state], congestion_target[train_state],
                x_energy[val_state], congestion_target[val_state], config,
                use_early_stopping=bool(val_state.sum() >= 16),
            )
            full = fit_regressor(
                x_full[train_state], congestion_target[train_state],
                x_full[val_state], congestion_target[val_state], config,
                use_early_stopping=bool(val_state.sum() >= 16),
            )
            expert_blind.append(blind.predict(x_energy[test]))
            expert_full.append(full.predict(x_full[test]))
        routed_congestion_blind = np.sum(p_blind * np.column_stack(expert_blind), axis=1)
        routed_congestion_full = np.sum(p_full * np.column_stack(expert_full), axis=1)

        predictions = {
            "energy_total_baseline": energy_total.predict(x_energy[test]),
            "direct_full_total": direct_total.predict(x_full[test]),
            "component_direct_blind": component_base_test + direct_congestion_blind,
            "component_direct_full": component_base_test + direct_congestion_full,
            "component_route_blind": component_base_test + routed_congestion_blind,
            "component_route_full": component_base_test + routed_congestion_full,
        }
        low_threshold = float(np.quantile(total[train], 0.10))
        low_mask = total[test] <= low_threshold
        test_state = state[test]
        for mode, predicted in predictions.items():
            result = error_metrics(total[test], predicted, low_mask, test_state)
            fold_rows.append({"fold": fold_index, "mode": mode, "low_threshold_train_q10": low_threshold, **result})
            local = pd.DataFrame({
                "fold": fold_index,
                "delivery_day": frame.loc[test, "delivery_day"].dt.strftime("%Y-%m-%d").to_numpy(),
                "time": frame.loc[test, "time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                "mode": mode, "actual": total[test], "predicted": predicted,
                "absolute_error": np.abs(predicted - total[test]), "low_price": low_mask,
                "congestion_state": test_state,
            })
            prediction_rows.append(local)

    metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = daily_metrics(predictions)
    pairs = [
        ("component_route_full", "component_route_blind"),
        ("component_route_full", "component_direct_full"),
        ("component_route_full", "direct_full_total"),
        ("component_direct_full", "component_direct_blind"),
        ("direct_full_total", "energy_total_baseline"),
    ]
    comparisons = {}
    for candidate, baseline in pairs:
        name = f"{candidate}_vs_{baseline}"
        comparisons[name] = {}
        for metric_name in ("mae", "extreme_low_mae", "negative_price_mae", "negative_sign_recall", "negative_congestion_mae", "positive_congestion_mae"):
            pivot = metrics.pivot(index="fold", columns="mode", values=metric_name).dropna()
            improvement = pivot[baseline] - pivot[candidate]
            if metric_name == "negative_sign_recall":
                improvement = -improvement
            comparisons[name][metric_name] = {
                "valid_folds": int(len(improvement)), "improved_fold_count": int((improvement > 0).sum()),
                "mean_improvement": float(improvement.mean()) if len(improvement) else None,
                "improvement_by_fold": {str(int(k)): float(v) for k, v in improvement.items()},
            }
        pivot_day = daily[daily["mode"].isin([candidate, baseline])].pivot(index=["fold", "delivery_day"], columns="mode", values="mae").dropna()
        delta = pivot_day[candidate] - pivot_day[baseline]
        comparisons[name]["daily_mae"] = {
            "n_days": int(len(delta)), "candidate_minus_baseline": float(delta.mean()),
            "candidate_better_day_fraction": float((delta < 0).mean()),
            "wilcoxon_two_sided_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0,
        }

    signal = comparisons["component_route_full_vs_component_route_blind"]
    route = comparisons["component_route_full_vs_component_direct_full"]
    signal_pass = any(signal[m]["improved_fold_count"] >= 3 and signal[m]["mean_improvement"] > 0 for m in ("mae", "extreme_low_mae"))
    route_pass = any(route[m]["improved_fold_count"] >= 3 and route[m]["mean_improvement"] > 0 for m in ("mae", "extreme_low_mae"))
    report = {
        "status": "complete", "componential_congestion_signal_gate": "PASS" if signal_pass else "FAIL",
        "componential_route_increment_gate": "PASS" if route_pass else "FAIL",
        "input_hashes": {"data": sha256(args.data), "labels": sha256(args.labels), "manifest": sha256(args.manifest), "folds": sha256(args.folds), "config": sha256(args.config), "code": sha256(Path(__file__))},
        "comparisons": comparisons,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    (args.output / "componential_congestion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "componential_congestion_signal_gate": report["componential_congestion_signal_gate"],
        "componential_route_increment_gate": report["componential_route_increment_gate"],
        "signal": signal, "route": route,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
