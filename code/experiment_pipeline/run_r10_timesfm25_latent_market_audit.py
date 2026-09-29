from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import TimesFm2_5ModelForPrediction

from run_r10_timesfm25_lora_market_audit import (
    CONTEXT,
    HORIZON,
    QUANTILES,
    QUANTILE_INDEX,
    carrier_covariates,
    point_metrics,
    prediction_frame,
    prepare_frame,
    probability_metrics,
    build_day_pack,
    set_seed,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]


def forecast_with_latent(
    model: torch.nn.Module,
    contexts: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    forecasts: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for start in range(0, len(contexts), batch_size):
            past = [item.to(device) for item in contexts[start : start + batch_size]]
            output = model(
                past_values=past,
                forecast_context_len=CONTEXT,
                truncate_negative=False,
            )
            full = output.full_predictions[:, :HORIZON, :]
            values = torch.stack(
                [full[..., QUANTILE_INDEX[level]] for level in QUANTILES], dim=-1
            )
            hidden = output.last_hidden_state.float()
            latent = torch.cat([hidden.mean(dim=1), hidden[:, -1, :]], dim=-1)
            forecasts.append(values.float().cpu().numpy())
            embeddings.append(latent.cpu().numpy())
    forecast_values = np.concatenate(forecasts, axis=0)
    forecast_values.sort(axis=-1)
    return forecast_values, np.concatenate(embeddings, axis=0)


def day_latent_features(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    components: int,
) -> tuple[dict[str, np.ndarray], StandardScaler, PCA]:
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    pca = PCA(n_components=min(components, len(train) - 1), random_state=42).fit(train_scaled)
    return (
        {
            "train": pca.transform(train_scaled),
            "val": pca.transform(scaler.transform(val)),
            "test": pca.transform(scaler.transform(test)),
        },
        scaler,
        pca,
    )


def matrix(
    meta: pd.DataFrame,
    raw: np.ndarray,
    day_latent: np.ndarray,
    covariates: list[str],
) -> np.ndarray:
    repeated = np.repeat(day_latent, HORIZON, axis=0)
    return np.column_stack(
        [
            meta[covariates].to_numpy(np.float32),
            raw[:, :, 1].reshape(-1),
            (raw[:, :, 2] - raw[:, :, 0]).reshape(-1),
            repeated,
        ]
    )


def fit_conditioner(
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    train_latent: np.ndarray,
    val_latent: np.ndarray,
    covariates: list[str],
    seed: int,
) -> lgb.LGBMRegressor:
    x_train = matrix(train_meta, train_raw, train_latent, covariates)
    x_val = matrix(val_meta, val_raw, val_latent, covariates)
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


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for fold, part in predictions.groupby("fold_name", sort=False):
        actual = part["price_real"].to_numpy(float)
        predicted = part["prediction"].to_numpy(float)
        point_rows.append({"fold": fold, **point_metrics(actual, predicted)})
        probability_rows.append(
            {
                "fold": fold,
                **probability_metrics(
                    actual,
                    part["q10"].to_numpy(float),
                    predicted,
                    part["q90"].to_numpy(float),
                ),
            }
        )
    actual = predictions["price_real"].to_numpy(float)
    predicted = predictions["prediction"].to_numpy(float)
    point_rows.append({"fold": "pooled", **point_metrics(actual, predicted)})
    probability_rows.append(
        {
            "fold": "pooled",
            **probability_metrics(
                actual,
                predictions["q10"].to_numpy(float),
                predicted,
                predictions["q90"].to_numpy(float),
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
        "--adapter-root",
        type=Path,
        default=ROOT / "03_results/r10_timesfm25_lora_market_v0.1/checkpoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "03_results/r10_timesfm25_latent_market_v0.1",
    )
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-transformers")
    parser.add_argument("--latent-components", type=int, default=16)
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
                frame,
                fold[split],
                covariates,
                CONTEXT,
                allow_missing_context=split == "train",
            )
            for split in ("train", "val", "test")
        }
        base = TimesFm2_5ModelForPrediction.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=str(device),
        )
        model = PeftModel.from_pretrained(
            base, args.adapter_root / fold["name"] / "final"
        )
        raw: dict[str, np.ndarray] = {}
        latent: dict[str, np.ndarray] = {}
        for split, (contexts, _) in packs.items():
            raw[split], latent[split] = forecast_with_latent(
                model, contexts, device, args.batch_size
            )
        latent_features, _, pca = day_latent_features(
            latent["train"], latent["val"], latent["test"], args.latent_components
        )
        conditioner = fit_conditioner(
            packs["train"][1],
            packs["val"][1],
            raw["train"],
            raw["val"],
            latent_features["train"],
            latent_features["val"],
            covariates,
            args.seed,
        )
        test_matrix = matrix(
            packs["test"][1], raw["test"], latent_features["test"], covariates
        )
        correction = conditioner.predict(test_matrix).reshape(-1, HORIZON, 1)
        conditioned = raw["test"] + correction
        rows.append(
            prediction_frame(
                fold,
                "test",
                "TimesFM25_LoRA_latent_market_conditioned",
                packs["test"][1],
                conditioned,
            )
        )
        diagnostics.append(
            {
                "fold": fold["name"],
                "latent_input_dim": int(latent["train"].shape[1]),
                "latent_components": int(pca.n_components_),
                "latent_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "conditioner_best_iteration": int(conditioner.best_iteration_),
            }
        )
        del model, base, conditioner
        gc.collect()
        torch.cuda.empty_cache()

    predictions = pd.concat(rows, ignore_index=True)
    point, probability = summarize(predictions)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    point.to_csv(args.output / "point_metrics.csv", index=False)
    probability.to_csv(args.output / "probability_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "diagnostics.csv", index=False)
    report = {
        "status": "fixed_interface_audit",
        "purpose": "Test whether a shallow output-only interface suppresses TimesFM encoder value.",
        "model_id": args.model_id,
        "adapters": str(args.adapter_root),
        "latent_summary": "mean and final patch states (2560 dimensions), train-only standardized PCA",
        "latent_components": args.latent_components,
        "conditioner": "same LightGBM L1 residual protocol with 51 market covariates",
        "information_boundary": "No day-ahead total price, explicit congestion input, test fitting, or adapter retraining.",
        "seed": args.seed,
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
    print(point.to_string(index=False), flush=True)
    print(probability.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
