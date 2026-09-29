from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "03_results/r16_ridge_carrier_probe_v1.0"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SITES = {
    "LHFDC": {
        "data": ROOT / "02_experiments/data_locked/r15_fixed_seasonal_lhfdc_route_data.csv",
        "cache": ROOT / "02_experiments/data_locked/r15_chronos2_lora_lhfdc_fold_cache.csv",
        "folds": ROOT / "00_control/r13_fixed_seasonal_folds_lhfdc.json",
    },
    "YQFDC": {
        "data": ROOT / "02_experiments/data_locked/r15_fixed_seasonal_yqfdc_route_data.csv",
        "cache": ROOT / "02_experiments/data_locked/r15_chronos2_lora_yqfdc_fold_cache.csv",
        "folds": ROOT / "00_control/r13_fixed_seasonal_folds_yqfdc.json",
    },
}
FEATURES = (
    "chronos2_lora",
    "price_day_ahead",
    "price_day_ahead_cong",
    "price_day_ahead_load",
    "load_day_ahead_pred",
    "elec_gene_total_pred",
    "bidding_space_pred",
    "demand_pred",
    "supply_pred",
    "renewable_penetration_pred",
    "price_real_lag96",
    "price_real_lag144",
    "price_real_lag336",
)


def mask(frame: pd.DataFrame, days: list[str]) -> np.ndarray:
    return frame["delivery_day"].isin(pd.to_datetime(days)).to_numpy()


def main() -> None:
    predictions: list[pd.DataFrame] = []
    selections: list[dict] = []
    for site, paths in SITES.items():
        frame = pd.read_csv(paths["data"])
        cache = pd.read_csv(paths["cache"])
        folds = json.loads(paths["folds"].read_text(encoding="utf-8"))
        frame["time"] = pd.to_datetime(frame["time"])
        frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
        cache["time"] = pd.to_datetime(cache["time"])
        for fold in folds:
            local_cache = cache[cache["fold_name"].eq(fold["name"])][
                ["time", "chronos2_lora"]
            ]
            local = frame.merge(local_cache, on="time", how="left", validate="one_to_one")
            train = mask(local, fold["train"])
            val = mask(local, fold["val"])
            test = mask(local, fold["test"])
            x = local[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
            fill = x.loc[train].median()
            x = x.fillna(fill).fillna(0.0).to_numpy(float)
            y = local["price_real"].to_numpy(float)

            candidates: list[tuple[float, float, object]] = []
            for alpha in ALPHAS:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(x[train], y[train])
                val_pred = model.predict(x[val])
                candidates.append(
                    (float(np.mean(np.abs(val_pred - y[val]))), alpha, model)
                )
            val_mae, alpha, _ = min(candidates, key=lambda row: row[0])
            final = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            final.fit(x[train | val], y[train | val])
            predicted = final.predict(x[test])
            positions = np.flatnonzero(test)
            predictions.append(
                pd.DataFrame(
                    {
                        "site": site,
                        "fold_name": fold["name"],
                        "time": local.loc[positions, "time"].to_numpy(),
                        "delivery_day": local.loc[positions, "delivery_day"].to_numpy(),
                        "actual": y[test],
                        "chronos2_lora": local.loc[
                            positions, "chronos2_lora"
                        ].to_numpy(float),
                        "ridge_carrier": predicted,
                    }
                )
            )
            selections.append(
                {
                    "site": site,
                    "fold_name": fold["name"],
                    "alpha": alpha,
                    "validation_mae": val_mae,
                }
            )

    result = pd.concat(predictions, ignore_index=True)
    rows = []
    for column in ("chronos2_lora", "ridge_carrier"):
        error = np.abs(result[column] - result["actual"])
        rows.append({"mode": column, "targets": len(result), "mae": float(error.mean())})
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "fold_selection.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
