from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from run_physical_soft_route_price_audit import add_features, day_mask
from run_supply_congestion_dual_route_audit import add_supply_features


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260729
warnings.filterwarnings("ignore", message="X does not have valid feature names")

SOURCES = (
    {
        "data": ROOT / "02_experiments/data_locked/r15_fixed_seasonal_lhfdc_route_data.csv",
        "manifest": ROOT / "02_experiments/data_locked/r13_fixed_seasonal_lhfdc_manifest.json",
        "folds": ROOT / "00_control/r13_fixed_seasonal_folds_lhfdc.json",
        "chronos": ROOT / "02_experiments/data_locked/r15_chronos2_lora_lhfdc_fold_cache.csv",
    },
    {
        "data": ROOT / "02_experiments/data_locked/r15_fixed_seasonal_yqfdc_route_data.csv",
        "manifest": ROOT / "02_experiments/data_locked/r13_fixed_seasonal_yqfdc_manifest.json",
        "folds": ROOT / "00_control/r13_fixed_seasonal_folds_yqfdc.json",
        "chronos": ROOT / "02_experiments/data_locked/r15_chronos2_lora_yqfdc_fold_cache.csv",
    },
)

PARAM_GRID = (
    {
        "name": "balanced",
        "n_estimators": 1600,
        "learning_rate": 0.018,
        "num_leaves": 31,
        "min_child_samples": 24,
        "max_depth": -1,
        "reg_alpha": 0.10,
        "reg_lambda": 0.40,
        "feature_fraction": 0.90,
    },
)


def model(params: dict) -> LGBMRegressor:
    return LGBMRegressor(
        objective="huber",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
        bagging_fraction=0.90,
        bagging_freq=1,
        **{key: value for key, value in params.items() if key != "name"},
    )


def engineered(frame: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    result = add_supply_features(add_features(frame.copy()))
    result["da_minus_chronos"] = result["price_day_ahead"] - result["chronos2_lora"]
    result["da_energy_minus_chronos"] = (
        result["price_day_ahead_energy_proxy"] - result["chronos2_lora"]
    )
    for source in (
        "price_day_ahead",
        "price_day_ahead_energy_proxy",
        "price_day_ahead_cong",
        "chronos2_lora",
    ):
        grouped = result.groupby("delivery_day", sort=False)[source]
        result[f"{source}_day_mean"] = grouped.transform("mean")
        result[f"{source}_day_std"] = grouped.transform("std").fillna(0.0)
        result[f"{source}_day_min"] = grouped.transform("min")
        result[f"{source}_day_max"] = grouped.transform("max")
        result[f"{source}_ramp1"] = grouped.diff().fillna(0.0)
    congestion = list(manifest["congestion_features"])
    base = [name for name in manifest["features"] if name not in congestion]
    extras = [
        "slot_sin",
        "slot_cos",
        "slot_norm",
        "price_day_ahead_energy_proxy",
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
        "chronos2_lora",
        "timesfm_q10",
        "timesfm_q50",
        "timesfm_q90",
        "da_minus_chronos",
        "da_energy_minus_chronos",
    ]
    extras += [
        column
        for column in result.columns
        if column.endswith(("_day_mean", "_day_std", "_day_min", "_day_max", "_ramp1"))
    ]
    blind = list(dict.fromkeys(base + extras))
    full = list(dict.fromkeys(blind + congestion))
    for column in full:
        if result[column].isna().any():
            result[column] = result[column].fillna(result[column].median()).fillna(0.0)
    return result, blind, full


def merge_carriers(frame: pd.DataFrame, fold_name: str, source: dict) -> pd.DataFrame:
    chronos = pd.read_csv(source["chronos"], parse_dates=["time"])
    chronos = chronos[chronos["fold_name"].eq(fold_name)].copy()
    chronos = chronos.rename(columns={"chronos2_lora": "chronos2_lora"})
    keep = [
        "time",
        "chronos2_lora",
        "timesfm_q10",
        "timesfm_q50",
        "timesfm_q90",
    ]
    chronos = chronos[keep]
    return frame.merge(chronos, on="time", how="left")


def fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    params: dict,
    formulation: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = y[train] if formulation == "direct" else y[train] - baseline[train]
    val_target = y[val] if formulation == "direct" else y[val] - baseline[val]
    fitted = model(params)
    fitted.fit(
        x[train],
        target,
        eval_set=[(x[val], val_target)],
        eval_metric="l1",
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    val_pred = fitted.predict(x[val])
    test_pred = fitted.predict(x[test])
    if formulation != "direct":
        val_pred = baseline[val] + val_pred
        test_pred = baseline[test] + test_pred
    return val_pred, test_pred


def main() -> None:
    output = ROOT / "03_results/r16_safe_full_probe_v1.0"
    output.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    selections: list[dict] = []
    for source in SOURCES:
        manifest = json.loads(source["manifest"].read_text(encoding="utf-8"))
        folds = json.loads(source["folds"].read_text(encoding="utf-8"))
        raw = pd.read_csv(source["data"], parse_dates=["time", "delivery_day"])
        for fold in folds:
            local = merge_carriers(raw, fold["name"], source)
            relevant = local["delivery_day"].isin(
                pd.to_datetime(fold["train"] + fold["val"] + fold["test"])
            )
            local = local.loc[relevant].copy().reset_index(drop=True)
            local, blind_cols, full_cols = engineered(local, manifest)
            train = day_mask(local, fold["train"])
            val = day_mask(local, fold["val"])
            test = day_mask(local, fold["test"])
            y = local["price_real"].to_numpy(float)
            baselines = {
                "chronos": local["chronos2_lora"].to_numpy(float),
                "day_ahead": local["price_day_ahead"].to_numpy(float),
            }
            candidates: list[dict] = []
            for feature_name, columns in (("full", full_cols),):
                x = local[columns].to_numpy(float)
                for params in PARAM_GRID:
                    for formulation, baseline_name in (
                        ("direct", "chronos"),
                        ("residual", "chronos"),
                        ("residual", "day_ahead"),
                    ):
                        val_pred, test_pred = fit_predict(
                            x,
                            y,
                            baselines[baseline_name],
                            train,
                            val,
                            test,
                            params,
                            formulation,
                        )
                        candidates.append(
                            {
                                "name": (
                                    f"{feature_name}_{params['name']}_"
                                    f"{formulation}_{baseline_name}"
                                ),
                                "feature_set": feature_name,
                                "params": params["name"],
                                "formulation": formulation,
                                "baseline": baseline_name,
                                "val_pred": val_pred,
                                "test_pred": test_pred,
                                "val_mae": float(np.mean(np.abs(val_pred - y[val]))),
                            }
                        )
            candidates.sort(key=lambda row: row["val_mae"])
            selected = candidates[0]
            selections.append(
                {
                    "fold_name": fold["name"],
                    "season": fold["season"],
                    "selected": selected["name"],
                    "validation_mae": selected["val_mae"],
                    "runner_up": candidates[1]["name"],
                    "runner_up_validation_mae": candidates[1]["val_mae"],
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "fold_name": fold["name"],
                        "season": fold["season"],
                        "delivery_day": local.loc[test, "delivery_day"].dt.strftime("%Y-%m-%d"),
                        "time": local.loc[test, "time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "actual": y[test],
                        "chronos2_lora": baselines["chronos"][test],
                        "safe_full_probe": selected["test_pred"],
                        "selected_candidate": selected["name"],
                    }
                )
            )
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "validation_selections.csv", index=False)
    summary = {
        column: float(np.mean(np.abs(pred[column] - pred["actual"])))
        for column in ("chronos2_lora", "safe_full_probe")
    }
    summary["targets"] = int(len(pred))
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
