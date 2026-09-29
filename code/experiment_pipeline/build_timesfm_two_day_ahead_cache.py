from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.data)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["delivery_day"] = pd.to_datetime(frame["delivery_day"])
    inputs = []
    target_rows = []
    days = []
    for delivery_day, daily in frame.groupby("delivery_day", sort=True):
        daily = daily.sort_values("time")
        if len(daily) != 48:
            raise ValueError(f"{delivery_day.date()} has {len(daily)} rows, expected 48")
        context = np.concatenate(
            [daily["price_real_lag144"].to_numpy(float), daily["price_real_lag96"].to_numpy(float)]
        ).astype(np.float32)
        if len(context) != int(config["context_length"]) or not np.isfinite(context).all():
            raise ValueError(f"invalid context for {delivery_day.date()}")
        inputs.append(context)
        target_rows.append(daily["time"].to_numpy())
        days.append(delivery_day.strftime("%Y-%m-%d"))

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config["model_id"])
    model.compile(
        timesfm.ForecastConfig(
            max_context=int(config["max_context"]),
            max_horizon=int(config["max_horizon"]),
            normalize_inputs=bool(config["normalize_inputs"]),
            use_continuous_quantile_head=bool(config["use_continuous_quantile_head"]),
            force_flip_invariance=bool(config["force_flip_invariance"]),
            infer_is_positive=bool(config["infer_is_positive"]),
            fix_quantile_crossing=bool(config["fix_quantile_crossing"]),
        )
    )
    point, quantile = model.forecast(horizon=int(config["prediction_length"]), inputs=inputs)
    start = int(config["target_horizon_start"])
    stop = start + int(config["target_horizon_length"])
    q_index = int(config["point_quantile_index"])
    selected = quantile[:, start:stop, q_index]
    if selected.shape != (len(inputs), 48):
        raise ValueError(f"unexpected TimesFM output shape: {selected.shape}")

    result = pd.DataFrame(
        {
            "time": np.concatenate(target_rows),
            "timesfm_prediction": selected.reshape(-1),
            "timesfm_mean": point[:, start:stop].reshape(-1),
            "timesfm_q10": quantile[:, start:stop, 1].reshape(-1),
            "timesfm_q50": quantile[:, start:stop, 5].reshape(-1),
            "timesfm_q90": quantile[:, start:stop, 9].reshape(-1),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    report = {
        "status": "complete",
        "model_id": config["model_id"],
        "n_delivery_days": len(days),
        "n_target_points": len(result),
        "information_boundary": "Contiguous D-3 and D-2 RT prices form 96-step context; TimesFM forecasts 96 steps and only target-day steps 49-96 are retained.",
        "point_output": "continuous quantile head q50",
        "cuda_available": bool(torch.cuda.is_available()),
        "input_hashes": {
            "data": sha256(args.data), "config": sha256(args.config), "code": sha256(Path(__file__))
        },
        "output_hash": sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
