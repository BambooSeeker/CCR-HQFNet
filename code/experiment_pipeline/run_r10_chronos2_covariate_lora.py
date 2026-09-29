from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
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

from run_physical_soft_route_price_audit import add_features
from run_supply_congestion_dual_route_audit import add_supply_features


ROOT = Path(__file__).resolve().parents[2]
HORIZON = 48
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
LORA_TARGET_MODULES = (
    "self_attention.q",
    "self_attention.v",
    "self_attention.k",
    "self_attention.o",
    "output_patch_embedding.output_layer",
)
DERIVED_COVARIATES = (
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
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_frame(data_path: Path, labels_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path, usecols=["time", "price_real_cong", "congestion_state"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    labels["time"] = pd.to_datetime(labels["time"])
    frame = frame.merge(labels, on="time", how="left", validate="one_to_one")
    frame = add_supply_features(add_features(frame))
    frame = frame.sort_values("time").reset_index(drop=True)
    if frame["time"].duplicated().any():
        raise ValueError("Duplicate timestamps in locked data.")
    return frame


def carrier_covariates(manifest: dict[str, Any]) -> list[str]:
    congestion = set(manifest["congestion_features"])
    # The day-ahead total price embeds congestion. The strict carrier receives
    # its energy proxy; explicit congestion remains isolated in the route.
    columns = [
        name
        for name in manifest["features"]
        if name not in congestion and name != "price_day_ahead"
    ]
    columns.extend(DERIVED_COVARIATES)
    return list(dict.fromkeys(columns))


def validate_covariates(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing covariates: {missing}")
    bad = [name for name in columns if not np.isfinite(frame[name].to_numpy(float)).all()]
    if bad:
        raise ValueError(f"Non-finite covariates: {bad}")


def split_contiguous(part: pd.DataFrame) -> list[pd.DataFrame]:
    if part.empty:
        return []
    part = part.sort_values("time").reset_index(drop=True)
    breaks = part["time"].diff().ne(pd.Timedelta(minutes=30)).to_numpy()
    breaks[0] = False
    groups = np.cumsum(breaks)
    return [group.reset_index(drop=True) for _, group in part.groupby(groups, sort=True)]


def training_tasks(
    frame: pd.DataFrame,
    days: list[str],
    covariates: list[str],
    min_length: int,
) -> list[dict[str, Any]]:
    selected = frame[frame["delivery_day"].isin(pd.to_datetime(days))].copy()
    tasks: list[dict[str, Any]] = []
    for segment in split_contiguous(selected):
        if len(segment) < min_length:
            continue
        target = segment["price_real"].to_numpy(np.float32)
        past = {name: segment[name].to_numpy(np.float32) for name in covariates}
        tasks.append(
            {
                "target": target,
                "past_covariates": past,
                "future_covariates": {
                    name: np.asarray([], dtype=np.float32) for name in covariates
                },
            }
        )
    if not tasks:
        raise RuntimeError("No contiguous training tasks survived the length gate.")
    return tasks


def previous_contiguous_context(
    frame: pd.DataFrame, start_index: int, context_length: int
) -> pd.DataFrame:
    start = max(0, start_index - context_length)
    context = frame.iloc[start:start_index].copy()
    if context.empty:
        return context
    times = context["time"].to_numpy(dtype="datetime64[ns]")
    delta = np.diff(times.astype("int64"))
    bad = np.flatnonzero(delta != int(pd.Timedelta(minutes=30).value))
    if len(bad):
        context = context.iloc[bad[-1] + 1 :].copy()
    return context


def prediction_tasks(
    frame: pd.DataFrame,
    days: list[str],
    covariates: list[str],
    context_length: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    tasks: list[dict[str, Any]] = []
    metadata: list[pd.DataFrame] = []
    for day in pd.to_datetime(days):
        future = frame[frame["delivery_day"].eq(day)].sort_values("time").copy()
        if len(future) != HORIZON:
            raise ValueError(f"{day.date()} has {len(future)} rows, expected {HORIZON}.")
        start_index = int(future.index.min())
        context = previous_contiguous_context(frame, start_index, context_length)
        if len(context) < HORIZON:
            raise ValueError(f"Only {len(context)} contiguous context points before {day.date()}.")
        tasks.append(
            {
                "target": context["price_real"].to_numpy(np.float32),
                "past_covariates": {
                    name: context[name].to_numpy(np.float32) for name in covariates
                },
                "future_covariates": {
                    name: future[name].to_numpy(np.float32) for name in covariates
                },
            }
        )
        metadata.append(
            future[
                [
                    "delivery_day",
                    "time",
                    "price_real",
                    "price_real_cong",
                    "congestion_state",
                ]
            ].copy()
        )
    return tasks, pd.concat(metadata, ignore_index=True)


def unpack_quantiles(items: list[torch.Tensor], n_quantiles: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for item in items:
        value = item.detach().float().cpu().numpy()
        value = np.squeeze(value)
        if value.shape == (n_quantiles, HORIZON):
            rows.append(value)
        elif value.shape == (HORIZON, n_quantiles):
            rows.append(value.T)
        else:
            raise ValueError(f"Unexpected Chronos quantile shape: {value.shape}")
    return np.stack(rows)


def forecast(
    pipeline: Chronos2Pipeline,
    tasks: list[dict[str, Any]],
    context_length: int,
    batch_size: int,
) -> np.ndarray:
    quantile_items, _ = pipeline.predict_quantiles(
        inputs=tasks,
        prediction_length=HORIZON,
        quantile_levels=list(QUANTILES),
        context_length=context_length,
        batch_size=batch_size,
        cross_learning=False,
        limit_prediction_length=False,
    )
    return unpack_quantiles(quantile_items, len(QUANTILES))


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    error = np.abs(actual - predicted)
    negative = actual < 0.0
    nonnegative = ~negative
    return {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "negative_price_mae": float(error[negative].mean()) if negative.any() else None,
        "negative_price_count": int(negative.sum()),
        "negative_sign_recall": float(np.mean(predicted[negative] < 0.0)) if negative.any() else None,
        "nonnegative_price_mae": float(error[nonnegative].mean()) if nonnegative.any() else None,
    }


def probability_metrics(
    actual: np.ndarray,
    q05: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
) -> dict[str, float]:
    pinballs = []
    for level, predicted in zip(
        QUANTILES, (q05, q10, q50, q90, q95), strict=True
    ):
        delta = actual - predicted
        pinballs.append(float(np.mean(np.maximum(level * delta, (level - 1.0) * delta))))
    covered = (actual >= q05) & (actual <= q95)
    width = q95 - q05
    alpha = 0.10
    winkler = width.copy()
    winkler += (2.0 / alpha) * (q05 - actual) * (actual < q05)
    winkler += (2.0 / alpha) * (actual - q95) * (actual > q95)
    return {
        "mean_pinball_q05_q10_q50_q90_q95": float(np.mean(pinballs)),
        "picp90": float(covered.mean()),
        "mpiw90": float(width.mean()),
        "winkler90": float(winkler.mean()),
    }


def add_predictions(
    rows: list[pd.DataFrame],
    fold: dict[str, Any],
    metadata: pd.DataFrame,
    values: np.ndarray,
    mode: str,
) -> None:
    local = metadata.copy()
    local["fold"] = fold["name"]
    local["mode"] = mode
    local["q05"] = values[:, 0, :].reshape(-1)
    local["q10"] = values[:, 1, :].reshape(-1)
    local["prediction"] = values[:, 2, :].reshape(-1)
    local["q90"] = values[:, 3, :].reshape(-1)
    local["q95"] = values[:, 4, :].reshape(-1)
    local["absolute_error"] = np.abs(local["prediction"] - local["price_real"])
    rows.append(local)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for (fold, mode), part in predictions.groupby(["fold", "mode"], sort=False):
        actual = part["price_real"].to_numpy(float)
        predicted = part["prediction"].to_numpy(float)
        point_rows.append({"fold": fold, "mode": mode, **point_metrics(actual, predicted)})
        probability_rows.append(
            {
                "fold": fold,
                "mode": mode,
                **probability_metrics(
                    actual,
                    part["q05"].to_numpy(float),
                    part["q10"].to_numpy(float),
                    predicted,
                    part["q90"].to_numpy(float),
                    part["q95"].to_numpy(float),
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
                    part["q05"].to_numpy(float),
                    part["q10"].to_numpy(float),
                    predicted,
                    part["q90"].to_numpy(float),
                    part["q95"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(point_rows), pd.DataFrame(probability_rows)


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
        "--output", type=Path, default=ROOT / "03_results/r10_chronos2_covariate_lora_v0.1"
    )
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--skip-zero-shot", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]
    frame = prepare_frame(args.data, args.labels)
    covariates = carrier_covariates(manifest)
    validate_covariates(frame, covariates)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = Chronos2Pipeline.from_pretrained(
        args.model_id, device_map=device, torch_dtype=dtype
    )
    prediction_rows: list[pd.DataFrame] = []
    fold_runtime: list[dict[str, Any]] = []

    for index, fold in enumerate(folds, start=1):
        print(f"[{index}/{len(folds)}] {fold['name']}", flush=True)
        set_seed(args.seed)
        train_tasks = training_tasks(
            frame,
            fold["train"],
            covariates,
            min_length=HORIZON * 2,
        )
        validation_tasks = training_tasks(
            frame,
            fold["val"],
            covariates,
            min_length=HORIZON * 2,
        )
        test_tasks, test_meta = prediction_tasks(
            frame, fold["test"], covariates, args.context_length
        )

        if not args.skip_zero_shot:
            zero_values = forecast(
                base, test_tasks, args.context_length, args.predict_batch_size
            )
            add_predictions(
                prediction_rows, fold, test_meta, zero_values, "Chronos2_covariate_zero_shot"
            )

        checkpoint_dir = args.output / "checkpoints" / fold["name"]
        started = time.time()
        finetuned = base.fit(
            inputs=train_tasks,
            validation_inputs=validation_tasks,
            prediction_length=HORIZON,
            finetune_mode="lora",
            lora_config={
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": list(LORA_TARGET_MODULES),
            },
            context_length=args.context_length,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            batch_size=args.train_batch_size,
            min_past=HORIZON,
            output_dir=checkpoint_dir,
            finetuned_ckpt_name="final",
            remove_printer_callback=True,
            logging_steps=max(1, min(25, args.num_steps)),
            eval_steps=max(1, min(100, args.num_steps)),
            save_steps=max(1, args.num_steps),
            report_to="none",
        )
        train_seconds = time.time() - started
        started = time.time()
        values = forecast(
            finetuned, test_tasks, args.context_length, args.predict_batch_size
        )
        predict_seconds = time.time() - started
        add_predictions(
            prediction_rows, fold, test_meta, values, "Chronos2_covariate_LoRA"
        )
        fold_runtime.append(
            {
                "fold": fold["name"],
                "train_tasks": len(train_tasks),
                "validation_tasks": len(validation_tasks),
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        del finetuned
        gc.collect()
        torch.cuda.empty_cache()

    predictions = pd.concat(prediction_rows, ignore_index=True)
    point, probability = summarize(predictions)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    point.to_csv(args.output / "point_metrics.csv", index=False)
    probability.to_csv(args.output / "probability_metrics.csv", index=False)
    pd.DataFrame(fold_runtime).to_csv(args.output / "runtime.csv", index=False)
    report = {
        "status": "screening" if args.num_steps < 600 or args.fold_limit else "fixed_protocol",
        "model_id": args.model_id,
        "finetune_mode": "lora",
        "target": "price_real",
        "information_boundary": (
            "The carrier receives observed target history and every locked target-day "
            "day-ahead-available non-congestion market covariate. Day-ahead total price "
            "and explicit congestion variables are excluded; congestion remains isolated "
            "in the frozen physical route."
        ),
        "covariates": covariates,
        "covariate_count": len(covariates),
        "fold_count": len(folds),
        "context_length": args.context_length,
        "prediction_length": HORIZON,
        "num_steps": args.num_steps,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "seed": args.seed,
        "device": device,
        "input_hashes": {
            "data": sha256(args.data),
            "labels": sha256(args.labels),
            "manifest": sha256(args.manifest),
            "folds": sha256(args.folds),
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
