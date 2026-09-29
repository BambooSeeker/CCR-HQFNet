from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "03_results"
OUT = RESULTS / "r15_experiment_closure_v1.0"
REFERENCE = (
    RESULTS
    / "r13_unified_fixed_window_comparison_v1.0"
    / "aligned_predictions.csv"
)
REFERENCE_METRICS = (
    RESULTS
    / "r13_unified_fixed_window_comparison_v1.0"
    / "overall_and_season_metrics.csv"
)
CHRONOS_FILES = (
    RESULTS / "r13_chronos2_lora_lhfdc_v0.1" / "predictions.csv",
    RESULTS / "r15_chronos2_lora_yqfdc_fixed_v1.0" / "predictions.csv",
)
ROUTE_FILES = (
    RESULTS / "r15_four_season_route_lhfdc_seed42_v1.0" / "predictions.csv",
    RESULTS / "r15_four_season_route_yqfdc_seed42_v1.0" / "predictions.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def vector_sha256(values: pd.Series) -> str:
    numeric = np.asarray(values, dtype="<f8")
    return hashlib.sha256(numeric.tobytes()).hexdigest().upper()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = np.abs(actual - predicted)
    retained = np.abs(actual) >= 50.0
    denominator = np.maximum(np.abs(actual) + np.abs(predicted), 1e-9)
    return {
        "targets": int(len(actual)),
        "mape50_targets": int(retained.sum()),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "mape50": float(100.0 * np.mean(error[retained] / np.abs(actual[retained]))),
        "smape50": float(
            np.mean(200.0 * error[retained] / denominator[retained])
        ),
    }


def read_chronos() -> pd.DataFrame:
    parts = []
    for path in CHRONOS_FILES:
        frame = pd.read_csv(path)
        frame = frame[frame["mode"].eq("Chronos2_covariate_LoRA")].copy()
        frame["time"] = pd.to_datetime(frame["time"])
        parts.append(
            frame[["time", "price_real", "prediction"]].rename(
                columns={"price_real": "actual", "prediction": "predicted"}
            )
        )
    result = pd.concat(parts, ignore_index=True).sort_values("time")
    if len(result) != 6048 or result["time"].duplicated().any():
        raise RuntimeError("Chronos-2 does not contain 6,048 unique targets.")
    return result


def read_route_full() -> pd.DataFrame:
    parts = []
    for path in ROUTE_FILES:
        frame = pd.read_csv(path)
        frame = frame[frame["mode"].eq("energy_consistent_dual_route")].copy()
        frame["time"] = pd.to_datetime(frame["time"])
        parts.append(frame[["time", "actual", "predicted"]])
    result = pd.concat(parts, ignore_index=True).sort_values("time")
    if len(result) != 6048 or result["time"].duplicated().any():
        raise RuntimeError("Candidate route does not contain 6,048 unique targets.")
    return result


def paired_test(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    merged = reference.merge(candidate, on="time", validate="one_to_one")
    daily = merged.groupby("delivery_day").apply(
        lambda group: pd.Series(
            {
                "reference_mae": np.mean(
                    np.abs(group["actual_x"] - group["CCR-HQFNet"])
                ),
                "candidate_mae": np.mean(
                    np.abs(group["actual_x"] - group["predicted"])
                ),
            }
        ),
        include_groups=False,
    )
    difference = daily["candidate_mae"] - daily["reference_mae"]
    statistic, pvalue = wilcoxon(
        difference,
        zero_method="pratt",
        alternative="two-sided",
        method="approx",
    )
    rng = np.random.default_rng(42)
    sampled = rng.integers(0, len(difference), size=(10000, len(difference)))
    bootstrap = difference.to_numpy()[sampled].mean(axis=1)
    return {
        "days": int(len(difference)),
        "candidate_minus_reference_daily_mae": float(difference.mean()),
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_pvalue": float(pvalue),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE, parse_dates=["time", "delivery_day"])
    if len(reference) != 6048 or reference["time"].duplicated().any():
        raise RuntimeError("Reference full prediction is not the frozen 6,048-point set.")
    chronos = read_chronos()
    route = read_route_full()

    chronos_joined = reference[["time", "season", "actual"]].merge(
        chronos, on="time", validate="one_to_one", suffixes=("_reference", "_chronos")
    )
    actual_delta = np.max(
        np.abs(
            chronos_joined["actual_reference"].to_numpy(float)
            - chronos_joined["actual_chronos"].to_numpy(float)
        )
    )
    if actual_delta > 1e-3:
        raise RuntimeError(f"Chronos-2 target alignment failed: {actual_delta}")

    rows = []
    for season, group in chronos_joined.groupby("season", sort=False):
        rows.append(
            {
                **metrics(
                    group["actual_reference"].to_numpy(float),
                    group["predicted"].to_numpy(float),
                ),
                "season": season,
                "model": "Chronos-2",
            }
        )
    rows.append(
        {
            **metrics(
                chronos_joined["actual_reference"].to_numpy(float),
                chronos_joined["predicted"].to_numpy(float),
            ),
            "season": "Overall",
            "model": "Chronos-2",
        }
    )
    comparison = pd.concat(
        [pd.read_csv(REFERENCE_METRICS), pd.DataFrame(rows)], ignore_index=True
    )
    comparison.to_csv(OUT / "comparison_metrics_with_chronos.csv", index=False)

    identity = pd.DataFrame(
        [
            {
                "artifact": "Frozen comparison Full",
                "role": "ACCEPTED_FULL",
                "points": len(reference),
                "mae": metrics(
                    reference["actual"].to_numpy(float),
                    reference["CCR-HQFNet"].to_numpy(float),
                )["mae"],
                "prediction_vector_sha256": vector_sha256(reference["CCR-HQFNet"]),
                "source_file": str(REFERENCE.relative_to(ROOT)),
                "source_file_sha256": sha256(REFERENCE),
            },
            {
                "artifact": "R15 candidate four-season route",
                "role": "REJECTED_AS_FULL_IDENTITY_MISMATCH",
                "points": len(route),
                "mae": metrics(
                    route["actual"].to_numpy(float),
                    route["predicted"].to_numpy(float),
                )["mae"],
                "prediction_vector_sha256": vector_sha256(route["predicted"]),
                "source_file": " + ".join(
                    str(path.relative_to(ROOT)) for path in ROUTE_FILES
                ),
                "source_file_sha256": " + ".join(sha256(path) for path in ROUTE_FILES),
            },
        ]
    )
    identity.to_csv(OUT / "full_prediction_identity.csv", index=False)

    route_aligned = reference[["time", "delivery_day", "actual", "CCR-HQFNet"]].merge(
        route, on="time", validate="one_to_one", suffixes=("_reference", "_route")
    )
    route_delta = np.abs(
        route_aligned["CCR-HQFNet"].to_numpy(float)
        - route_aligned["predicted"].to_numpy(float)
    )
    audit = {
        "status": "FULL_IDENTITY_CONFLICT",
        "hard_rule": (
            "The Full row in comparison, ablation, probabilistic, and stratified "
            "tables must resolve to one frozen 6,048-point prediction artifact."
        ),
        "accepted_full": "Frozen comparison Full",
        "accepted_full_mae": float(identity.iloc[0]["mae"]),
        "candidate_route_mae": float(identity.iloc[1]["mae"]),
        "pointwise_equal_count": int(np.sum(route_delta <= 1e-9)),
        "pointwise_total": int(len(route_delta)),
        "maximum_absolute_prediction_difference": float(route_delta.max()),
        "candidate_route_disposition": (
            "Internal mechanism audit only. It cannot populate the manuscript "
            "ablation table because its Full prediction is not the accepted Full."
        ),
        "chronos_comparison": paired_test(reference, chronos),
    }
    (OUT / "full_prediction_identity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison[comparison["season"].eq("Overall")].sort_values("mae").to_string(index=False))
    print(identity.to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
