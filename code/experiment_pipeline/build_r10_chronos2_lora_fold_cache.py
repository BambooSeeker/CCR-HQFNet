from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from peft import PeftModel

from run_r10_chronos2_covariate_lora import (
    ROOT,
    carrier_covariates,
    forecast,
    prediction_tasks,
    prepare_frame,
    validate_covariates,
)


def load_adapter(model_id: str, adapter_path: Path) -> Chronos2Pipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = Chronos2Pipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    adapted = PeftModel.from_pretrained(base.model, adapter_path)
    return Chronos2Pipeline(model=adapted)


def fallback_rows(
    frame: pd.DataFrame, fold_name: str, day: pd.Timestamp
) -> pd.DataFrame:
    future = frame[frame["delivery_day"].eq(day)].sort_values("time").copy()
    if len(future) != 48:
        raise ValueError(f"Fallback day {day.date()} is incomplete.")
    point = future["price_real_lag96"].to_numpy(float)
    scale = float(np.std(point))
    return pd.DataFrame(
        {
            "fold_name": fold_name,
            "delivery_day": day,
            "time": future["time"].to_numpy(),
            "chronos2_lora": point,
            "timesfm_q05": point - 1.645 * scale,
            "timesfm_q10": point - scale,
            "timesfm_q50": point,
            "timesfm_q90": point + scale,
            "timesfm_q95": point + 1.645 * scale,
            "carrier_source": "safe_D_minus_2_fallback",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "02_experiments/data_locked/lhfdc01_forecast_available_202405_202607.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "02_experiments/data_locked/realized_congestion_labels_202405_202607.csv",
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
        "--checkpoint-root",
        type=Path,
        default=ROOT / "03_results/r10_chronos2_covariate_lora_v0.1/checkpoints",
    )
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        default=ROOT / "03_results/r10_chronos2_covariate_lora_v0.1/predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "02_experiments/data_locked/r10_chronos2_lora_fold_cache.csv",
    )
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fold-limit", type=int)
    args = parser.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]
    frame = prepare_frame(args.data, args.labels)
    covariates = carrier_covariates(manifest)
    validate_covariates(frame, covariates)
    output_rows: list[pd.DataFrame] = []

    for index, fold in enumerate(folds, start=1):
        print(f"[{index}/{len(folds)}] {fold['name']}", flush=True)
        pipeline = load_adapter(args.model_id, args.checkpoint_root / fold["name"] / "final")
        for split in ("train", "val", "test"):
            split_tasks: list[dict[str, Any]] = []
            split_metadata: list[pd.DataFrame] = []
            for day_text in fold[split]:
                day = pd.Timestamp(day_text)
                try:
                    tasks, metadata = prediction_tasks(
                        frame, [day_text], covariates, args.context_length
                    )
                except ValueError as exc:
                    if "contiguous context" not in str(exc):
                        raise
                    output_rows.append(fallback_rows(frame, fold["name"], day))
                    continue
                split_tasks.extend(tasks)
                split_metadata.append(metadata)
            if not split_tasks:
                continue
            values = forecast(
                pipeline, split_tasks, args.context_length, args.batch_size
            )
            metadata = pd.concat(split_metadata, ignore_index=True)
            local = pd.DataFrame(
                {
                    "fold_name": fold["name"],
                    "delivery_day": metadata["delivery_day"].to_numpy(),
                    "time": metadata["time"].to_numpy(),
                    "chronos2_lora": values[:, 2, :].reshape(-1),
                    "timesfm_q05": values[:, 0, :].reshape(-1),
                    "timesfm_q10": values[:, 1, :].reshape(-1),
                    "timesfm_q50": values[:, 2, :].reshape(-1),
                    "timesfm_q90": values[:, 3, :].reshape(-1),
                    "timesfm_q95": values[:, 4, :].reshape(-1),
                    "carrier_source": "chronos2_covariate_lora",
                }
            )
            output_rows.append(local)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    output = pd.concat(output_rows, ignore_index=True)
    if output.duplicated(["fold_name", "time"]).any():
        raise ValueError("Fold cache has duplicate fold/time rows.")

    reference = pd.read_csv(args.reference_predictions)
    reference = reference[reference["mode"].eq("Chronos2_covariate_LoRA")].copy()
    reference["time"] = pd.to_datetime(reference["time"])
    check = reference.merge(
        output,
        left_on=["fold", "time"],
        right_on=["fold_name", "time"],
        how="left",
        validate="one_to_one",
    )
    check["reload_absolute_difference"] = np.abs(
        check["prediction"].to_numpy(float) - check["chronos2_lora"]
    )
    max_difference = float(
        check["reload_absolute_difference"].max()
    )
    reload_by_fold = (
        check.groupby("fold")["reload_absolute_difference"]
        .agg(["max", "mean", "median"])
        .reset_index()
        .to_dict(orient="records")
    )
    if max_difference > 1e-3:
        audit_path = args.output.with_name(args.output.stem + "_reload_audit.csv")
        check.sort_values("reload_absolute_difference", ascending=False).to_csv(
            audit_path, index=False
        )
        print(
            check.groupby("fold")["reload_absolute_difference"]
            .agg(["max", "mean", "median"])
            .to_string(),
            flush=True,
        )

    # Preserve the exact test predictions produced by the in-memory fitted
    # pipeline. Reloaded adapters are needed only for causal train/validation
    # carrier values; sparse bfloat16 boundary drift must not alter test evidence.
    exact = reference[
        ["fold", "time", "q05", "q10", "prediction", "q90", "q95"]
    ].rename(
        columns={
            "fold": "fold_name",
            "q05": "reference_q05",
            "q10": "reference_q10",
            "prediction": "reference_q50",
            "q90": "reference_q90",
            "q95": "reference_q95",
        }
    )
    output = output.merge(exact, on=["fold_name", "time"], how="left")
    test_reference = output["reference_q50"].notna()
    output.loc[test_reference, "timesfm_q05"] = output.loc[
        test_reference, "reference_q05"
    ]
    output.loc[test_reference, "timesfm_q10"] = output.loc[
        test_reference, "reference_q10"
    ]
    output.loc[test_reference, "chronos2_lora"] = output.loc[
        test_reference, "reference_q50"
    ]
    output.loc[test_reference, "timesfm_q50"] = output.loc[
        test_reference, "reference_q50"
    ]
    output.loc[test_reference, "timesfm_q90"] = output.loc[
        test_reference, "reference_q90"
    ]
    output.loc[test_reference, "timesfm_q95"] = output.loc[
        test_reference, "reference_q95"
    ]
    output.loc[test_reference, "carrier_source"] = "in_memory_test_reference"
    output = output.drop(
        columns=[
            "reference_q05",
            "reference_q10",
            "reference_q50",
            "reference_q90",
            "reference_q95",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = {
        "rows": int(len(output)),
        "folds": int(output["fold_name"].nunique()),
        "safe_lag_fallback_rows": int(
            output["carrier_source"].eq("safe_D_minus_2_fallback").sum()
        ),
        "in_memory_test_reference_rows": int(
            output["carrier_source"].eq("in_memory_test_reference").sum()
        ),
        "max_test_reload_difference": max_difference,
        "reload_difference_by_fold": reload_by_fold,
        "test_values_restored_from_in_memory_predictions": True,
        "information_boundary": (
            "Each fold uses its own training-only LoRA adapter. Forecasts receive observed "
            "history plus the target day's 52 day-ahead-available non-congestion market covariates."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
