from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


ROOT = Path(__file__).resolve().parents[2]
SAFE_LAGS = (96, 144, 336)


def add_common_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("time").reset_index(drop=True).copy()
    slot = (
        result["time"].dt.hour * 2 + result["time"].dt.minute // 30 - 1
    ) % 48
    result["slot_sin"] = np.sin(2.0 * np.pi * slot / 48.0)
    result["slot_cos"] = np.cos(2.0 * np.pi * slot / 48.0)
    result["slot_norm"] = slot / 47.0

    load_scale = result["load_day_ahead_pred"].abs().clip(lower=1000.0)
    result["price_day_ahead_energy_proxy"] = (
        result["price_day_ahead"] - result["price_day_ahead_cong"]
    )
    result["market_bid_space_proxy"] = (
        result["load_day_ahead_pred"]
        - result["elec_exter_plan"]
        - result["elec_fix_out_plan"]
    )
    result["supply_adequacy_proxy"] = (
        result["elec_gene_total_pred"] - result["market_bid_space_proxy"]
    )
    result["external_plan_load_ratio"] = result["elec_exter_plan"] / load_scale
    result["fixed_output_load_ratio"] = result["elec_fix_out_plan"] / load_scale
    result["generation_load_ratio"] = result["elec_gene_total_pred"] / load_scale
    result["renewable_supply_index"] = result["energy_hydro_renewable"] / load_scale

    # Strict pre-bid variables cannot use the formal external plan released
    # after the 10:15 offer deadline.
    result["prebid_residual_load"] = (
        result["load_day_ahead_pred"]
        - result["elec_fix_out_plan"]
        - result["energy_hydro_renewable"]
    )
    result["prebid_supply_margin"] = (
        result["elec_gene_total_pred"] - result["prebid_residual_load"]
    )
    result["prebid_fixed_output_ratio"] = result["elec_fix_out_plan"] / load_scale
    result["prebid_generation_ratio"] = result["elec_gene_total_pred"] / load_scale
    result["prebid_renewable_ratio"] = result["energy_hydro_renewable"] / load_scale

    # D-2, D-3 and D-7 day-ahead outcomes are complete before D-1 10:15.
    historical_day_ahead = (
        "price_day_ahead",
        "price_day_ahead_cong",
        "price_day_ahead_load",
        "elec_day_ahead_all",
        "elec_exter_plan",
    )
    for name in historical_day_ahead:
        for lag in SAFE_LAGS:
            result[f"{name}_safe_lag{lag}"] = result[name].shift(lag)
    for lag in SAFE_LAGS:
        result[f"price_day_ahead_cong_abs_safe_lag{lag}"] = result[
            f"price_day_ahead_cong_safe_lag{lag}"
        ].abs()
    return result.replace([np.inf, -np.inf], np.nan)


def feature_sets(manifest: dict[str, Any]) -> dict[str, list[str]]:
    derived_post = [
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
    ]
    post_da = list(dict.fromkeys(manifest["features"] + derived_post))

    target_boundary = [
        "load_day_ahead_pred",
        "elec_fix_out_plan",
        "energy_hydro_renewable",
        "elec_gene_total_pred",
        "prebid_residual_load",
        "prebid_supply_margin",
        "prebid_fixed_output_ratio",
        "prebid_generation_ratio",
        "prebid_renewable_ratio",
    ]
    safe_history = [
        name
        for name in manifest["features"]
        if any(token in name for token in ("_lag96", "_lag144", "_lag336", "_safe48_"))
    ]
    calendar = [
        "slot_sin",
        "slot_cos",
        "slot_norm",
        "market_phase_trial_2024",
        "market_phase_revised_2025",
        "market_phase_formal",
    ]
    historical_market = [
        f"{name}_safe_lag{lag}"
        for name in (
            "price_day_ahead",
            "price_day_ahead_load",
            "elec_day_ahead_all",
            "elec_exter_plan",
        )
        for lag in SAFE_LAGS
    ]
    historical_congestion = [
        f"price_day_ahead_cong_safe_lag{lag}" for lag in SAFE_LAGS
    ] + [f"price_day_ahead_cong_abs_safe_lag{lag}" for lag in SAFE_LAGS]
    prebid_blind = list(
        dict.fromkeys(target_boundary + safe_history + calendar + historical_market)
    )
    prebid_with_congestion = prebid_blind + historical_congestion
    return {
        "post_da_full": post_da,
        "prebid_blind": prebid_blind,
        "prebid_historical_congestion": prebid_with_congestion,
    }


def model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=24,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_predict(
    frame: pd.DataFrame,
    train_days: list[str],
    val_days: list[str],
    test_days: list[str],
    features: list[str],
    target: str,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    train = frame["delivery_day"].isin(pd.to_datetime(train_days))
    val = frame["delivery_day"].isin(pd.to_datetime(val_days))
    test = frame["delivery_day"].isin(pd.to_datetime(test_days))
    usable = frame[features + [target]].notna().all(axis=1)
    train &= usable
    val &= usable
    if bool((test & ~usable).any()):
        missing = int((test & ~usable).sum())
        raise ValueError(f"{missing} test rows have unavailable features for {target}.")
    estimator = model(seed)
    estimator.fit(
        frame.loc[train, features],
        frame.loc[train, target],
        eval_set=[(frame.loc[val, features], frame.loc[val, target])],
        eval_metric="l1",
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    prediction = estimator.predict(frame.loc[test, features])
    importance = pd.DataFrame(
        {
            "feature": features,
            "gain": estimator.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    return prediction, importance


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    error = np.abs(predicted - actual)
    negative = actual < 0.0
    return {
        "count": int(len(actual)),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "negative_count": int(negative.sum()),
        "negative_mae": float(error[negative].mean()) if negative.any() else None,
        "negative_sign_recall": (
            float(np.mean(predicted[negative] < 0.0)) if negative.any() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "02_experiments/data_locked/lhfdc01_forecast_available_202405_202607.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_experiments/data_locked/dataset_manifest.json",
    )
    parser.add_argument(
        "--folds", type=Path, default=ROOT / "00_control/r10_seasonal_long_windows.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "03_results/r11_prebid_information_audit_v0.1",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    frame = add_common_features(frame)
    sets = feature_sets(manifest)

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(folds, start=1):
        test_mask = frame["delivery_day"].isin(pd.to_datetime(fold["test"]))
        test_meta = frame.loc[
            test_mask,
            ["time", "delivery_day", "price_real", "price_day_ahead"],
        ].reset_index(drop=True)
        for mode, features in sets.items():
            rt_prediction, rt_importance = fit_predict(
                frame,
                fold["train"],
                fold["val"],
                fold["test"],
                features,
                "price_real",
                args.seed + fold_index,
            )
            da_prediction, da_importance = fit_predict(
                frame,
                fold["train"],
                fold["val"],
                fold["test"],
                features,
                "price_day_ahead",
                args.seed + 100 + fold_index,
            )
            local = test_meta.copy()
            local["fold"] = fold_index
            local["fold_name"] = fold["name"]
            local["mode"] = mode
            local["rt_predicted"] = rt_prediction
            local["da_predicted"] = da_prediction
            local["spread_actual"] = local["price_real"] - local["price_day_ahead"]
            local["spread_predicted"] = rt_prediction - da_prediction
            rows.append(local)

            rt_metrics = metrics(local["price_real"].to_numpy(), rt_prediction)
            da_metrics = metrics(local["price_day_ahead"].to_numpy(), da_prediction)
            spread_metrics = metrics(
                local["spread_actual"].to_numpy(), local["spread_predicted"].to_numpy()
            )
            spread_actual = local["spread_actual"].to_numpy()
            spread_predicted = local["spread_predicted"].to_numpy()
            metric_rows.append(
                {
                    "fold": fold_index,
                    "fold_name": fold["name"],
                    "mode": mode,
                    "feature_count": len(features),
                    **{f"rt_{key}": value for key, value in rt_metrics.items()},
                    **{f"da_{key}": value for key, value in da_metrics.items()},
                    **{f"spread_{key}": value for key, value in spread_metrics.items()},
                    "spread_sign_accuracy": float(
                        np.mean(np.sign(spread_actual) == np.sign(spread_predicted))
                    ),
                    "spread_correlation": float(
                        np.corrcoef(spread_actual, spread_predicted)[0, 1]
                    ),
                }
            )
            for target_name, importance in (
                ("real_time", rt_importance),
                ("day_ahead", da_importance),
            ):
                importance = importance.copy()
                importance["fold"] = fold_index
                importance["fold_name"] = fold["name"]
                importance["mode"] = mode
                importance["target"] = target_name
                importance_rows.append(importance)

    predictions = pd.concat(rows, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    importances = pd.concat(importance_rows, ignore_index=True)
    pooled_rows = []
    for mode, part in predictions.groupby("mode", sort=False):
        rt = metrics(part["price_real"].to_numpy(), part["rt_predicted"].to_numpy())
        da = metrics(
            part["price_day_ahead"].to_numpy(), part["da_predicted"].to_numpy()
        )
        spread = metrics(
            part["spread_actual"].to_numpy(), part["spread_predicted"].to_numpy()
        )
        pooled_rows.append(
            {
                "mode": mode,
                "feature_count": len(sets[mode]),
                **{f"rt_{key}": value for key, value in rt.items()},
                **{f"da_{key}": value for key, value in da.items()},
                **{f"spread_{key}": value for key, value in spread.items()},
                "spread_sign_accuracy": float(
                    np.mean(
                        np.sign(part["spread_actual"].to_numpy())
                        == np.sign(part["spread_predicted"].to_numpy())
                    )
                ),
                "spread_correlation": float(
                    np.corrcoef(
                        part["spread_actual"].to_numpy(),
                        part["spread_predicted"].to_numpy(),
                    )[0, 1]
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows)

    predictions.to_csv(args.output / "predictions.csv", index=False)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    pooled.to_csv(args.output / "pooled_metrics.csv", index=False)
    importances.to_csv(args.output / "feature_importance.csv", index=False)
    report = {
        "status": "complete",
        "decision_time": "D-1 10:15 for prebid modes; D-1 20:00 for post_da_full",
        "prebid_exclusions": [
            "target-day day-ahead prices and congestion",
            "target-day day-ahead cleared quantity",
            "formal external plan without a pre-10:15 version timestamp",
            "all target-day real-time realizations",
        ],
        "feature_sets": sets,
        "pooled_metrics": pooled.to_dict(orient="records"),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
