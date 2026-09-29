from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "demo_data/march_2026_reference_predictions.csv"
REFERENCE = ROOT / "outputs/reference_metrics.json"


def calculate(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["actual"].to_numpy(float)
    predicted = frame["predicted"].to_numpy(float)
    error = predicted - actual
    retained = np.abs(actual) >= 50.0
    negative = actual < 0.0
    predicted_negative = predicted < 0.0
    tp = int(np.sum(negative & predicted_negative))
    return {
        "n": int(len(frame)),
        "mape_n": int(retained.sum()),
        "negative_n": int(negative.sum()),
        "predicted_negative_n": int(predicted_negative.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(100.0 * np.mean(np.abs(error[retained]) / np.abs(actual[retained]))),
        "negative_mae": float(np.mean(np.abs(error[negative]))),
        "negative_recall_pct": float(100.0 * tp / negative.sum()),
        "negative_precision_pct": float(100.0 * tp / predicted_negative.sum()),
        "false_positives": int(np.sum(~negative & predicted_negative)),
    }


def main() -> None:
    observed = calculate(pd.read_csv(PREDICTIONS))
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if isinstance(value, int):
            assert observed[key] == value, (key, observed[key], value)
        else:
            assert np.isclose(observed[key], value, rtol=0.0, atol=1e-10), (key, observed[key], value)
    print(json.dumps(observed, indent=2))
    print("PASS: March 2026 test-period metrics match the released reference.")


if __name__ == "__main__":
    main()
