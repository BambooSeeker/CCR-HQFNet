from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import TimesFm2_5ModelForPrediction

from run_r10_timesfm25_lora_market_audit import (
    CONTEXT,
    apply_market_conditioner,
    build_day_pack,
    carrier_covariates,
    fit_market_conditioner,
    forecast,
    point_metrics,
    prediction_frame,
    prepare_frame,
    probability_metrics,
    set_seed,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for fold, mode in predictions[["fold_name", "mode"]].drop_duplicates().itertuples(index=False):
        part = predictions[
            predictions["fold_name"].eq(fold) & predictions["mode"].eq(mode)
        ]
        actual = part["price_real"].to_numpy(float)
        predicted = part["prediction"].to_numpy(float)
        point_rows.append({"fold": fold, "mode": mode, **point_metrics(actual, predicted)})
        probability_rows.append(
            {
                "fold": fold,
                "mode": mode,
                **probability_metrics(
                    actual,
                    part["q10"].to_numpy(float),
                    predicted,
                    part["q90"].to_numpy(float),
                ),
            }
        )
    for mode, part in predictions.groupby("mode", sort=False):
        actual = part["price_real"].to_numpy(float)
        predicted = part["prediction"].to_numpy(float)
        point_rows.append({"fold": "pooled", "mode": mode, **point_metrics(actual, predicted)})
        probability_rows.append(
            {
                "fold": "pooled",
                "mode": mode,
                **probability_metrics(
                    actual,
                    part["q10"].to_numpy(float),
                    predicted,
                    part["q90"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(point_rows), pd.DataFrame(probability_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "02_experiments/data_locked/lhfdc01_forecast_available_202405_202607.csv",
    )
    parser.add_argument(
        "--labels", type=Path,
        default=ROOT / "02_experiments/data_locked/realized_congestion_labels_202405_202607.csv",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "02_experiments/data_locked/dataset_manifest.json",
    )
    parser.add_argument(
        "--folds", type=Path, default=ROOT / "00_control/r10_seasonal_long_windows.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "03_results/r10_timesfm25_zeroshot_market_v0.1",
    )
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    frame = prepare_frame(args.data, args.labels)
    covariates = carrier_covariates(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for index, fold in enumerate(folds, start=1):
        print(f"[{index}/{len(folds)}] {fold['name']}", flush=True)
        packs = {
            split: build_day_pack(
                frame, fold[split], covariates, CONTEXT, allow_missing_context=split == "train"
            )
            for split in ("train", "val", "test")
        }
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=str(device),
        )
        raw = {
            split: forecast(model, contexts, device, args.batch_size)
            for split, (contexts, _) in packs.items()
        }
        conditioner = fit_market_conditioner(
            packs["train"][1], packs["val"][1], raw["train"], raw["val"],
            covariates, args.seed,
        )
        conditioned = apply_market_conditioner(
            conditioner, packs["test"][1], raw["test"], covariates
        )
        rows.append(
            prediction_frame(
                fold, "test", "TimesFM25_zero_shot_raw", packs["test"][1], raw["test"]
            )
        )
        rows.append(
            prediction_frame(
                fold, "test", "TimesFM25_zero_shot_market_conditioned",
                packs["test"][1], conditioned,
            )
        )
        diagnostics.append(
            {"fold": fold["name"], "conditioner_best_iteration": conditioner.best_iteration_}
        )
        del model, conditioner
        gc.collect()
        torch.cuda.empty_cache()

    predictions = pd.concat(rows, ignore_index=True)
    point, probability = summarize(predictions)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    point.to_csv(args.output / "point_metrics.csv", index=False)
    probability.to_csv(args.output / "probability_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "diagnostics.csv", index=False)
    report = {
        "status": "fixed_ablation_control",
        "model_id": args.model_id,
        "purpose": "Separate target-day market-information gain from TimesFM LoRA gain.",
        "protocol": "Same five folds, 192-point maximum context, 48-point horizon, and 51 non-congestion covariates as the LoRA audit.",
        "seed": args.seed,
        "input_hashes": {
            "data": sha256(args.data), "labels": sha256(args.labels),
            "manifest": sha256(args.manifest), "folds": sha256(args.folds),
            "code": sha256(Path(__file__)),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(point[point["fold"].eq("pooled")].to_string(index=False), flush=True)
    print(probability[probability["fold"].eq("pooled")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
