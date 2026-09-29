from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from run_r10_chronos2_covariate_lora import (
    HORIZON,
    ROOT,
    carrier_covariates,
    prepare_frame,
    validate_covariates,
)


CONTEXT = 192
TIME_FEATURES = ("slot_sin", "slot_cos", "slot_norm")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DailyDataset(Dataset):
    def __init__(self, x: np.ndarray, time_features: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.time_features = torch.from_numpy(time_features).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.time_features[index], self.y[index]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = CONTEXT + HORIZON):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).float().unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=-1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.unsqueeze(-1)


class DilatedDepthwiseBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        se_reduction: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=d_model,
        )
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.se = SqueezeExcitation1D(d_model, se_reduction)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value = self.norm(x).transpose(1, 2)
        value = self.depthwise(F.pad(value, (self.left_padding, 0)))
        left, gate = torch.chunk(self.pointwise_in(value), 2, dim=1)
        value = left * torch.sigmoid(gate)
        value = self.se(self.pointwise_out(value)).transpose(1, 2)
        return residual + self.dropout(value)


class HybridTransformerDilatedEncoder(nn.Module):
    """Original global-Transformer/local-dilated-convolution carrier."""

    def __init__(
        self,
        x_dim: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.10,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4),
        se_reduction: int = 8,
        fusion_hidden: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_projection = nn.Linear(x_dim + len(TIME_FEATURES), d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.local = nn.ModuleList(
            [
                DilatedDepthwiseBlock(
                    d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    se_reduction=se_reduction,
                )
                for dilation in dilations
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model + len(TIME_FEATURES), fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, 1),
        )
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        initial = self.position(
            self.input_projection(torch.cat([x, time_features], dim=-1))
        )
        global_state = self.transformer(initial)
        local_state = global_state
        for block in self.local:
            local_state = block(local_state)
        gate_input = torch.cat(
            [
                global_state,
                local_state,
                torch.abs(global_state - local_state),
                time_features,
            ],
            dim=-1,
        )
        weight = torch.sigmoid(self.fusion(gate_input) / self.temperature)
        fused = weight * global_state + (1.0 - weight) * local_state
        state = self.norm(initial + self.dropout(fused))
        return self.output(state[:, -HORIZON:]).squeeze(-1)


def fit_scalers(
    frame: pd.DataFrame, train_days: list[str], covariates: list[str]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = frame[frame["delivery_day"].isin(pd.to_datetime(train_days))]
    x_mean = train[covariates].mean().to_numpy(np.float32)
    x_std = train[covariates].std(ddof=0).clip(lower=1e-6).to_numpy(np.float32)
    y_mean = float(train["price_real"].mean())
    y_std = max(float(train["price_real"].std(ddof=0)), 1e-6)
    return x_mean, x_std, y_mean, y_std


def previous_complete_day_context(
    frame: pd.DataFrame, day: pd.Timestamp, length: int
) -> pd.DataFrame:
    prior = frame[frame["delivery_day"].lt(day)]
    complete_days = (
        prior.groupby("delivery_day", sort=True)
        .size()
        .loc[lambda count: count.eq(HORIZON)]
        .index
    )
    required_days = length // HORIZON
    selected_days = complete_days[-required_days:]
    if len(selected_days) < required_days:
        return prior.iloc[0:0].copy()
    return (
        prior[prior["delivery_day"].isin(selected_days)]
        .sort_values("time")
        .iloc[-length:]
        .copy()
    )


def build_samples(
    frame: pd.DataFrame,
    days: list[str],
    covariates: list[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    require_all: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for day_text in days:
        day = pd.Timestamp(day_text)
        future = frame[frame["delivery_day"].eq(day)].sort_values("time").copy()
        if len(future) != HORIZON:
            if require_all:
                raise ValueError(f"{day.date()} has {len(future)} rows.")
            continue
        context = previous_complete_day_context(frame, day, CONTEXT)
        if len(context) < CONTEXT:
            if require_all:
                raise ValueError(f"{day.date()} has only {len(context)} context rows.")
            continue
        context = context.iloc[-CONTEXT:]
        tokens = pd.concat([context, future], ignore_index=True)
        market = (tokens[covariates].to_numpy(np.float32) - x_mean) / x_std
        price_channel = np.zeros((CONTEXT + HORIZON, 1), dtype=np.float32)
        price_channel[:CONTEXT, 0] = (
            context["price_real"].to_numpy(np.float32) - y_mean
        ) / y_std
        observed_mask = np.zeros((CONTEXT + HORIZON, 1), dtype=np.float32)
        observed_mask[:CONTEXT, 0] = 1.0
        x_rows.append(np.concatenate([market, price_channel, observed_mask], axis=1))
        time_rows.append(tokens[list(TIME_FEATURES)].to_numpy(np.float32))
        y_rows.append(
            ((future["price_real"].to_numpy(np.float32) - y_mean) / y_std)
        )
        metadata.append(
            future[["delivery_day", "time", "price_real"]].copy()
        )
    if not x_rows:
        raise RuntimeError("No daily samples were constructed.")
    return (
        np.stack(x_rows),
        np.stack(time_rows),
        np.stack(y_rows),
        pd.concat(metadata, ignore_index=True),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    for x, time_features, _ in loader:
        values = model(x.to(device), time_features.to(device))
        rows.append(values.float().cpu().numpy())
    return np.concatenate(rows) * y_std + y_mean


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    error = predicted - actual
    negative = actual < 0.0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape": float(
            100
            * np.mean(
                2 * np.abs(error) / (np.abs(actual) + np.abs(predicted) + 1e-6)
            )
        ),
        "negative_price_count": int(negative.sum()),
        "negative_price_mae": (
            float(np.mean(np.abs(error[negative]))) if negative.any() else None
        ),
        "negative_sign_recall": (
            float(np.mean(predicted[negative] < 0.0)) if negative.any() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT
        / "02_experiments/data_locked/lhfdc01_forecast_available_202405_202607.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "02_experiments/data_locked/realized_congestion_labels_202405_202607.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "02_experiments/data_locked/dataset_manifest.json",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=ROOT / "00_control/r10_seasonal_weekly20_windows.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "03_results/r10_hybrid_tf_dilated_weekly20_v0.1",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-limit", type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if args.fold_limit:
        folds = folds[: args.fold_limit]
    frame = prepare_frame(args.data, args.labels)
    all_covariates = carrier_covariates(manifest)
    validate_covariates(frame, all_covariates)
    covariates = [name for name in all_covariates if name not in TIME_FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    parameter_count: int | None = None
    for fold_index, fold in enumerate(folds, start=1):
        print(f"[{fold_index}/{len(folds)}] {fold['name']}", flush=True)
        set_seed(args.seed)
        x_mean, x_std, y_mean, y_std = fit_scalers(
            frame, fold["train"], covariates
        )
        train = build_samples(
            frame,
            fold["train"],
            covariates,
            x_mean,
            x_std,
            y_mean,
            y_std,
            require_all=False,
        )
        validation = build_samples(
            frame,
            fold["val"],
            covariates,
            x_mean,
            x_std,
            y_mean,
            y_std,
            require_all=True,
        )
        test = build_samples(
            frame,
            fold["test"],
            covariates,
            x_mean,
            x_std,
            y_mean,
            y_std,
            require_all=True,
        )
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            DailyDataset(*train[:3]),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            DailyDataset(*validation[:3]), batch_size=args.batch_size
        )
        test_loader = DataLoader(
            DailyDataset(*test[:3]), batch_size=args.batch_size
        )
        model = HybridTransformerDilatedEncoder(
            x_dim=train[0].shape[-1]
        ).to(device)
        if parameter_count is None:
            parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        checkpoint = args.output / "checkpoints" / fold["name"] / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_mae = float("inf")
        best_epoch = 0
        stale = 0
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, time_features, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                forecast = model(x.to(device), time_features.to(device))
                loss = F.mse_loss(forecast, y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            validation_prediction = predict(
                model, validation_loader, device, y_mean, y_std
            )
            validation_actual = validation[3]["price_real"].to_numpy(float).reshape(
                validation_prediction.shape
            )
            validation_mae = float(
                np.mean(np.abs(validation_prediction - validation_actual))
            )
            if validation_mae < best_mae - 1e-6:
                best_mae = validation_mae
                best_epoch = epoch
                stale = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    break
        train_seconds = time.time() - started
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        test_prediction = predict(model, test_loader, device, y_mean, y_std)
        actual = test[3]["price_real"].to_numpy(float)
        predicted = test_prediction.reshape(-1)
        local_metrics = metrics(actual, predicted)
        metric_rows.append(
            {
                "fold": fold_index,
                "fold_name": fold["name"],
                "model": "HybridTF-DilatedConv",
                **local_metrics,
            }
        )
        local = test[3].copy()
        local["fold"] = fold_index
        local["fold_name"] = fold["name"]
        local["model"] = "HybridTF-DilatedConv"
        local["predicted"] = predicted
        local["absolute_error"] = np.abs(predicted - actual)
        prediction_rows.append(local)
        runtime_rows.append(
            {
                "fold": fold_index,
                "fold_name": fold["name"],
                "train_samples": len(train[0]),
                "best_epoch": best_epoch,
                "best_validation_mae": best_mae,
                "train_seconds": train_seconds,
            }
        )
        del model
        torch.cuda.empty_cache()

    predictions = pd.concat(prediction_rows, ignore_index=True)
    pooled = metrics(
        predictions["price_real"].to_numpy(float),
        predictions["predicted"].to_numpy(float),
    )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame = pd.concat(
        [
            metrics_frame,
            pd.DataFrame(
                [
                    {
                        "fold": "pooled",
                        "fold_name": "pooled",
                        "model": "HybridTF-DilatedConv",
                        **pooled,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics_frame.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(args.output / "runtime.csv", index=False)
    report = {
        "status": "screening" if args.fold_limit else "fixed_protocol",
        "architecture": (
            "Three-layer global Transformer plus causal dilated depthwise TCN "
            "(kernel 7; dilations 1,2,4; GLU; SE) with adaptive fusion."
        ),
        "task": "Direct 48-step target-day forecast from 192 observed history points.",
        "information_boundary": (
            "The same target history and target-day non-congestion covariates used "
            "by the Chronos-2 carrier; no realized target-day price or congestion."
        ),
        "folds": len(folds),
        "parameter_count": parameter_count,
        "covariate_count": len(covariates),
        "seed": args.seed,
        "hyperparameters": {
            "d_model": 192,
            "heads": 8,
            "transformer_layers": 3,
            "kernel_size": 7,
            "dilations": [1, 2, 4],
            "dropout": 0.10,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
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
    print(metrics_frame[metrics_frame["fold_name"].eq("pooled")].to_string(index=False))


if __name__ == "__main__":
    main()
