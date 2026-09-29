from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_r10_joint_calibrated_route as frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--carrier-cache", type=Path, required=True)
    parser.add_argument("--carrier-column", default="chronos2_lora")
    parser.add_argument("--preserve-carrier-sign-on-reject", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    labels = pd.read_csv(args.labels, usecols=["time", "congestion_state"])
    cache = pd.read_csv(args.carrier_cache)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    labels["time"] = pd.to_datetime(labels["time"])
    cache["time"] = pd.to_datetime(cache["time"])
    base_frame = frozen.add_supply_features(
        frozen.add_features(frame.merge(labels, on="time", how="left"))
    )
    signatures = frozen.daily_signatures(base_frame)

    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict] = []
    for fold_index, fold in enumerate(folds, start=1):
        fold_cache = cache[cache["fold_name"].eq(fold["name"])].drop(
            columns=["fold_name", "delivery_day", "carrier_source"], errors="ignore"
        )
        local_frame = base_frame.merge(
            fold_cache, on="time", how="left", validate="one_to_one"
        )
        relevant_days = fold["train"] + fold["val"] + fold["test"]
        relevant = local_frame["delivery_day"].isin(pd.to_datetime(relevant_days))
        required = [args.carrier_column, "timesfm_q10", "timesfm_q50", "timesfm_q90"]
        if local_frame.loc[relevant, required].isna().any().any():
            raise ValueError(f"Incomplete carrier cache for {fold['name']}.")

        arrays = frozen.frame_arrays(local_frame, manifest, args.carrier_column)
        actual = local_frame["price_real"].to_numpy(float)
        carrier = local_frame[args.carrier_column].fillna(0.0).to_numpy(float)
        q10 = local_frame["timesfm_q10"].fillna(0.0).to_numpy(float)
        q50 = local_frame["timesfm_q50"].fillna(0.0).to_numpy(float)
        q90 = local_frame["timesfm_q90"].fillna(0.0).to_numpy(float)
        congestion_state = local_frame["congestion_state"].fillna(-1).to_numpy(int)
        train = frozen.day_mask(local_frame, fold["train"])
        val = frozen.day_mask(local_frame, fold["val"])
        test = frozen.day_mask(local_frame, fold["test"])
        low_threshold = float(np.quantile(actual[train][actual[train] >= 0.0], 0.10))

        oof = frozen.oof_components(
            local_frame, fold, arrays, actual, carrier, congestion_state,
            q10, q50, q90, low_threshold, config
        )
        recent = frozen.fit_components(
            local_frame, arrays, actual, carrier, congestion_state,
            q10, q50, q90, train, val, val, low_threshold, config
        )
        target = frozen.fit_components(
            local_frame, arrays, actual, carrier, congestion_state,
            q10, q50, q90, train, val, test, low_threshold, config
        )

        policies: dict[pd.Timestamp, dict] = {}
        oof_days = pd.Index(sorted(oof["delivery_day"].unique()))
        for target_day in pd.Index(pd.to_datetime(fold["test"])):
            similar_days, distance = frozen.nearest_oof_days(
                signatures, oof_days, target_day
            )
            similar = oof[oof["delivery_day"].isin(similar_days)]
            calibration = pd.concat([similar, recent], ignore_index=True)
            policy, diagnostics = frozen.select_joint_policy(calibration)
            policies[target_day] = policy
            calibration_rows.append(
                {
                    "fold": fold_index,
                    "fold_name": fold["name"],
                    "target_day": target_day.strftime("%Y-%m-%d"),
                    "selected_similar_days": frozen.date_strings(pd.Index(similar_days)),
                    **policy,
                    **distance,
                    **diagnostics,
                }
            )

        ungated_dual = np.empty(len(target), dtype=float)
        supply_final = np.empty(len(target), dtype=float)
        dual_final = np.empty(len(target), dtype=float)
        activation_all = np.empty(len(target), dtype=float)
        selected_all = np.zeros(len(target), dtype=bool)
        for target_day, policy in policies.items():
            mask = target["delivery_day"].eq(target_day).to_numpy()
            positions = np.flatnonzero(mask)
            activation, supply_route, dual_route = frozen.route_from_policy(
                target.loc[mask],
                policy["supply_weight"],
                policy["activation_threshold"],
            )
            activation_all[positions] = activation
            ungated_dual[positions] = dual_route
            threshold = policy["energy_threshold"]
            accepted = (
                np.zeros(mask.sum(), dtype=bool)
                if threshold is None
                else (activation > 0.0)
                & (
                    target.loc[mask, "price_day_ahead_energy_proxy"].to_numpy(float)
                    <= float(threshold)
                )
            )
            selected_all[positions] = accepted
            supply_final[positions] = supply_route
            dual_final[positions] = dual_route
            rejected = positions[~accepted]
            if args.preserve_carrier_sign_on_reject:
                # Rejection suppresses the routed residual; it must not erase a
                # negative-price signal already learned by the foundation model.
                supply_final[rejected] = target.loc[
                    rejected, "carrier_blind"
                ].to_numpy(float)
                dual_final[rejected] = target.loc[
                    rejected, "physical_congestion_gate"
                ].to_numpy(float)
            else:
                supply_final[rejected] = np.maximum(
                    target.loc[rejected, "carrier_blind"].to_numpy(float), 0.0
                )
                dual_final[rejected] = np.maximum(
                    target.loc[rejected, "physical_congestion_gate"].to_numpy(float), 0.0
                )

        modes = {
            "chronos2_lora": target["timesfm_zero_shot"].to_numpy(float),
            "carrier_blind": target["carrier_blind"].to_numpy(float),
            "physical_congestion_gate": target["physical_congestion_gate"].to_numpy(float),
            "ungated_dual_route": ungated_dual,
            "energy_consistent_supply_route": supply_final,
            "energy_consistent_dual_route": dual_final,
        }
        fold_actual = target["actual"].to_numpy(float)
        fold_state = congestion_state[target["position"].to_numpy(int)]
        low_mask = fold_actual <= low_threshold
        for mode, predicted in modes.items():
            metric_rows.append(
                {
                    "fold": fold_index,
                    "fold_name": fold["name"],
                    "mode": mode,
                    "low_threshold_train_q10": low_threshold,
                    **frozen.mode_metrics(fold_actual, predicted, low_mask, fold_state),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_index,
                        "fold_name": fold["name"],
                        "delivery_day": target["delivery_day"].dt.strftime("%Y-%m-%d"),
                        "time": target["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": mode,
                        "actual": fold_actual,
                        "predicted": predicted,
                        "absolute_error": np.abs(predicted - fold_actual),
                        "negative_price": fold_actual < 0.0,
                        "candidate_activation": activation_all,
                        "energy_consistency_selected": selected_all,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    daily = predictions.groupby(
        ["fold", "fold_name", "delivery_day", "mode"], as_index=False
    ).agg(mae=("absolute_error", "mean"))
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    daily.to_csv(args.output / "daily_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_json(
        args.output / "similar_day_calibration.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    report = {
        "status": "complete",
        "route_implementation": "Imported unchanged functions from run_r10_joint_calibrated_route.py",
        "carrier": args.carrier_column,
        "fold_specific_carrier": True,
        "preserve_carrier_sign_on_reject": args.preserve_carrier_sign_on_reject,
        "input_hashes": {
            "frozen_route_code": frozen.sha256(Path(frozen.__file__)),
            "carrier_cache": frozen.sha256(args.carrier_cache),
            "code": frozen.sha256(Path(__file__)),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
