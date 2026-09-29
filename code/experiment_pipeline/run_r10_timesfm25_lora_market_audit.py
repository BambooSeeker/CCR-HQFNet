from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, TensorDataset
from transformers import TimesFm2_5ModelForPrediction

from run_physical_soft_route_price_audit import add_features
from run_supply_congestion_dual_route_audit import add_supply_features


ROOT = Path(__file__).resolve().parents[2]
HORIZON = 48
CONTEXT = 192
QUANTILES = (0.10, 0.50, 0.90)
QUANTILE_INDEX = {0.10: 1, 0.50: 5, 0.90: 9}
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
    labels = pd.read_csv(
        labels_path, usecols=["time", "price_real_cong", "congestion_state"]
    )
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"]).dt.normalize()
    labels["time"] = pd.to_datetime(labels["time"])
    frame = frame.merge(labels, on="time", how="left", validate="one_to_one")
    frame = add_supply_features(add_features(frame))
    return frame.sort_values("time").reset_index(drop=True)


def carrier_covariates(manifest: dict[str, Any]) -> list[str]:
    congestion = set(manifest["congestion_features"])
    columns = [
        name
        for name in manifest["features"]
        if name not in congestion and name != "price_day_ahead"
    ]
    columns.extend(DERIVED_COVARIATES)
    return list(dict.fromkeys(columns))


def contiguous(values: pd.Series) -> bool:
    times = pd.to_datetime(values).to_numpy(dtype="datetime64[ns]").astype("int64")
    return bool(len(times) < 2 or np.all(np.diff(times) == pd.Timedelta(minutes=30).value))


def build_training_windows(
    frame: pd.DataFrame,
    days: list[str],
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    allowed = set(pd.to_datetime(days).to_numpy(dtype="datetime64[ns]"))
    selected = frame[frame["delivery_day"].isin(allowed)].copy()
    selected = selected.sort_values("time").reset_index(drop=True)
    total = context_length + HORIZON
    y = selected["price_real"].to_numpy(np.float32)
    day_values = selected["delivery_day"].to_numpy(dtype="datetime64[ns]")
    times = pd.to_datetime(selected["time"])
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    for start in range(0, len(selected) - total + 1):
        end = start + total
        if not all(day in allowed for day in day_values[start:end]):
            continue
        if not contiguous(times.iloc[start:end]):
            continue
        contexts.append(y[start : start + context_length])
        futures.append(y[start + context_length : end])
    if not contexts:
        raise RuntimeError("No contiguous TimesFM training windows were built.")
    return torch.from_numpy(np.stack(contexts)), torch.from_numpy(np.stack(futures))


def previous_context(
    frame: pd.DataFrame, start_index: int, context_length: int
) -> np.ndarray | None:
    start = max(0, start_index - context_length)
    part = frame.iloc[start:start_index].copy()
    if part.empty:
        return None
    times = pd.to_datetime(part["time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    breaks = np.flatnonzero(np.diff(times) != pd.Timedelta(minutes=30).value)
    if len(breaks):
        part = part.iloc[breaks[-1] + 1 :]
    if len(part) < HORIZON:
        return None
    return part["price_real"].to_numpy(np.float32)


def build_day_pack(
    frame: pd.DataFrame,
    days: list[str],
    covariates: list[str],
    context_length: int,
    allow_missing_context: bool,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    contexts: list[torch.Tensor] = []
    metadata: list[pd.DataFrame] = []
    for day in pd.to_datetime(days):
        future = frame[frame["delivery_day"].eq(day)].sort_values("time").copy()
        if len(future) != HORIZON:
            raise ValueError(f"{day.date()} has {len(future)} rows, expected {HORIZON}.")
        context = previous_context(frame, int(future.index.min()), context_length)
        if context is None:
            if allow_missing_context:
                continue
            raise ValueError(f"No {context_length}-point context before {day.date()}.")
        contexts.append(torch.from_numpy(context))
        metadata.append(
            future[
                [
                    "delivery_day",
                    "time",
                    "price_real",
                    "price_real_cong",
                    "congestion_state",
                    *covariates,
                ]
            ].copy()
        )
    if not contexts:
        raise RuntimeError("No day-level TimesFM forecast tasks were built.")
    return contexts, pd.concat(metadata, ignore_index=True)


def validation_loss(
    model: torch.nn.Module,
    contexts: torch.Tensor,
    futures: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    loader = DataLoader(TensorDataset(contexts, futures), batch_size=batch_size)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for past, future in loader:
            output = model(
                past_values=past.to(device),
                future_values=future.to(device),
                forecast_context_len=contexts.shape[1],
                truncate_negative=False,
            )
            losses.append(float(output.loss.detach().float().cpu()))
    return float(np.mean(losses))


def fit_lora(
    model_id: str,
    train_contexts: torch.Tensor,
    train_futures: torch.Tensor,
    val_contexts: torch.Tensor,
    val_futures: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_steps: int,
    seed: int,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    set_seed(seed)
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=str(device),
    )
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(base, config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=0.01,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_contexts, train_futures),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    step = 0
    while step < steps:
        for past, future in loader:
            if step >= steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(
                    past_values=past.to(device),
                    future_values=future.to(device),
                    forecast_context_len=train_contexts.shape[1],
                    truncate_negative=False,
                )
                loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0,
            )
            optimizer.step()
            step += 1
            if step == 1 or step % eval_steps == 0 or step == steps:
                val_loss = validation_loss(
                    model, val_contexts, val_futures, device, batch_size
                )
                row = {
                    "step": float(step),
                    "train_loss": float(loss.detach().float().cpu()),
                    "validation_loss": val_loss,
                }
                history.append(row)
                print(
                    f"  step={step:4d} train={row['train_loss']:.5f} val={val_loss:.5f}",
                    flush=True,
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in get_peft_model_state_dict(model).items()
                    }
    if best_state is None:
        raise RuntimeError("TimesFM LoRA did not produce a validation checkpoint.")
    set_peft_model_state_dict(model, best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    return model, history


def forecast(
    model: torch.nn.Module,
    contexts: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for start in range(0, len(contexts), batch_size):
            past = [item.to(device) for item in contexts[start : start + batch_size]]
            output = model(
                past_values=past,
                forecast_context_len=CONTEXT,
                truncate_negative=False,
            )
            full = output.full_predictions[:, :HORIZON, :]
            selected = torch.stack(
                [full[..., QUANTILE_INDEX[level]] for level in QUANTILES], dim=-1
            )
            rows.append(selected.detach().float().cpu().numpy())
    values = np.concatenate(rows, axis=0)
    values.sort(axis=-1)
    return values


def fit_market_conditioner(
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    covariates: list[str],
    seed: int,
) -> lgb.LGBMRegressor:
    def matrix(meta: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                meta[covariates].to_numpy(np.float32),
                raw[:, :, 1].reshape(-1),
                (raw[:, :, 2] - raw[:, :, 0]).reshape(-1),
            ]
        )

    x_train = matrix(train_meta, train_raw)
    x_val = matrix(val_meta, val_raw)
    y_train = train_meta["price_real"].to_numpy(float) - train_raw[:, :, 1].reshape(-1)
    y_val = val_meta["price_real"].to_numpy(float) - val_raw[:, :, 1].reshape(-1)
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        deterministic=True,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )
    return model


def apply_market_conditioner(
    model: lgb.LGBMRegressor,
    meta: pd.DataFrame,
    raw: np.ndarray,
    covariates: list[str],
) -> np.ndarray:
    matrix = np.column_stack(
        [
            meta[covariates].to_numpy(np.float32),
            raw[:, :, 1].reshape(-1),
            (raw[:, :, 2] - raw[:, :, 0]).reshape(-1),
        ]
    )
    correction = model.predict(matrix).reshape(raw.shape[0], HORIZON, 1)
    return raw + correction


def prediction_frame(
    fold: dict[str, Any],
    split: str,
    mode: str,
    meta: pd.DataFrame,
    values: np.ndarray,
) -> pd.DataFrame:
    result = meta[
        ["delivery_day", "time", "price_real", "price_real_cong", "congestion_state"]
    ].copy()
    result.insert(0, "fold_name", fold["name"])
    result.insert(1, "market_phase", fold["market_phase"])
    result.insert(2, "season", fold["season"])
    result.insert(3, "split", split)
    result.insert(4, "mode", mode)
    result["q10"] = values[:, :, 0].reshape(-1)
    result["prediction"] = values[:, :, 1].reshape(-1)
    result["q90"] = values[:, :, 2].reshape(-1)
    return result


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    error = np.abs(actual - predicted)
    negative = actual < 0.0
    return {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "negative_price_mae": float(error[negative].mean()) if negative.any() else None,
        "negative_price_count": int(negative.sum()),
        "negative_sign_recall": float(np.mean(predicted[negative] < 0.0)) if negative.any() else None,
        "nonnegative_price_mae": float(error[~negative].mean()) if (~negative).any() else None,
    }


def probability_metrics(
    actual: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
) -> dict[str, float]:
    pinballs = []
    for level, predicted in zip(QUANTILES, (q10, q50, q90), strict=True):
        delta = actual - predicted
        pinballs.append(float(np.mean(np.maximum(level * delta, (level - 1.0) * delta))))
    width = q90 - q10
    covered = (actual >= q10) & (actual <= q90)
    winkler = width.copy()
    winkler += 10.0 * (q10 - actual) * (actual < q10)
    winkler += 10.0 * (actual - q90) * (actual > q90)
    return {
        "mean_pinball_q10_q50_q90": float(np.mean(pinballs)),
        "picp80": float(covered.mean()),
        "mpiw80": float(width.mean()),
        "winkler80": float(winkler.mean()),
    }


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    test = predictions[predictions["split"].eq("test")]
    for fold, mode in test[["fold_name", "mode"]].drop_duplicates().itertuples(index=False):
        part = test[(test["fold_name"].eq(fold)) & (test["mode"].eq(mode))]
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
    for mode, part in test.groupby("mode", sort=False):
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
        "--output", type=Path, default=ROOT / "03_results/r10_timesfm25_lora_market_v0.1"
    )
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers")
    parser.add_argument("--context-length", type=int, default=CONTEXT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--predict-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--fold-limit", type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    folds = folds[args.fold_start :]
    if args.fold_limit is not None:
        folds = folds[: args.fold_limit]
    frame = prepare_frame(args.data, args.labels)
    covariates = carrier_covariates(manifest)
    missing = [name for name in covariates if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing market covariates: {missing}")
    if not np.isfinite(frame[covariates].to_numpy(float)).all():
        raise ValueError("Non-finite market covariates found.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}

    for index, fold in enumerate(folds, start=1):
        print(f"[{index}/{len(folds)}] {fold['name']}", flush=True)
        train_x, train_y = build_training_windows(frame, fold["train"], args.context_length)
        val_x, val_y = build_training_windows(frame, fold["val"], args.context_length)
        packs = {
            split: build_day_pack(
                frame,
                fold[split],
                covariates,
                args.context_length,
                allow_missing_context=split == "train",
            )
            for split in ("train", "val", "test")
        }
        started = time.time()
        model, history = fit_lora(
            args.model_id,
            train_x,
            train_y,
            val_x,
            val_y,
            args.output / "checkpoints" / fold["name"] / "final",
            device,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.eval_steps,
            args.seed,
        )
        train_seconds = time.time() - started
        histories[fold["name"]] = history
        raw: dict[str, np.ndarray] = {}
        predict_started = time.time()
        for split, (contexts, _) in packs.items():
            raw[split] = forecast(model, contexts, device, args.predict_batch_size)
        predict_seconds = time.time() - predict_started
        conditioner = fit_market_conditioner(
            packs["train"][1],
            packs["val"][1],
            raw["train"],
            raw["val"],
            covariates,
            args.seed,
        )
        for split, (_, meta) in packs.items():
            conditioned = apply_market_conditioner(conditioner, meta, raw[split], covariates)
            prediction_rows.append(
                prediction_frame(fold, split, "TimesFM25_LoRA_raw", meta, raw[split])
            )
            prediction_rows.append(
                prediction_frame(
                    fold,
                    split,
                    "TimesFM25_LoRA_market_conditioned",
                    meta,
                    conditioned,
                )
            )
        runtime_rows.append(
            {
                "fold": fold["name"],
                "train_windows": len(train_x),
                "validation_windows": len(val_x),
                "train_forecast_days": len(packs["train"][0]),
                "conditioner_best_iteration": conditioner.best_iteration_,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        del model, conditioner
        gc.collect()
        torch.cuda.empty_cache()

    predictions = pd.concat(prediction_rows, ignore_index=True)
    point, probability = summarize(predictions)
    predictions.to_csv(args.output / "predictions_all_splits.csv", index=False)
    predictions[predictions["split"].eq("test")].to_csv(
        args.output / "predictions.csv", index=False
    )
    point.to_csv(args.output / "point_metrics.csv", index=False)
    probability.to_csv(args.output / "probability_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(args.output / "runtime.csv", index=False)
    (args.output / "training_history.json").write_text(
        json.dumps(histories, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "status": "screening" if args.steps < 600 or args.fold_limit else "fixed_protocol",
        "purpose": "Fair TimesFM architecture audit after the historical-only zero-shot failure.",
        "model_id": args.model_id,
        "target": "price_real",
        "fold_count": len(folds),
        "context_length": args.context_length,
        "prediction_length": HORIZON,
        "lora": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "r": 4,
            "alpha": 8,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "checkpoint_selection": "minimum validation loss",
        },
        "market_conditioner": {
            "type": "LightGBM L1 residual conditioner",
            "fit_boundary": "fold training rows only; validation used only for early stopping",
            "inputs": "51 non-congestion target-day market covariates + TimesFM median + interval width",
            "quantile_handling": "one market residual shifts q10/q50/q90 equally",
        },
        "information_boundary": (
            "The carrier excludes price_day_ahead total and all six explicit congestion variables. "
            "Realized congestion is retained only as an evaluation label. The physical congestion "
            "route remains independently ablatable."
        ),
        "covariates": covariates,
        "covariate_count": len(covariates),
        "seed": args.seed,
        "device": str(device),
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
