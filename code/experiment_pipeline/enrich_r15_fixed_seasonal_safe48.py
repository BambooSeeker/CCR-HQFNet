from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SAFE_COLUMNS = {
    "price_real_safe48_mean": "mean",
    "price_real_safe48_std": "std",
    "price_real_safe48_min": "min",
    "price_real_safe48_max": "max",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the frozen safe-48 history features omitted from R13 seasonal slices."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--continuous-source", required=True, type=Path)
    parser.add_argument("--node-code", required=True)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, parse_dates=["time"])
    if not frame["time"].is_monotonic_increasing:
        raise ValueError("input rows must be chronologically ordered")

    source = pd.read_csv(
        args.continuous_source,
        usecols=["time", "code", "price_real_lag96"],
        parse_dates=["time"],
    )
    source = (
        source[source["code"].eq(args.node_code)]
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )
    if source.empty:
        raise ValueError(f"node {args.node_code!r} is absent from the continuous source")
    rolling = source["price_real_lag96"].rolling(48, min_periods=48)
    for column, statistic in SAFE_COLUMNS.items():
        source[column] = getattr(rolling, statistic)()

    safe = source[["time", *SAFE_COLUMNS]].copy()
    frame = frame.merge(safe, on="time", how="left", validate="one_to_one")
    missing = {column: int(frame[column].isna().sum()) for column in SAFE_COLUMNS}
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    registered_days = {
        day
        for fold in folds
        for part in ("train", "val", "test")
        for day in fold[part]
    }
    registered = frame["delivery_day"].astype(str).isin(registered_days)
    registered_missing = {
        column: int(frame.loc[registered, column].isna().sum()) for column in SAFE_COLUMNS
    }
    if any(registered_missing.values()):
        raise RuntimeError(
            f"registered train/validation/test rows are incomplete: {registered_missing}"
        )

    # The continuous source starts 47 rows before a complete 48-point history exists.
    # Those rows precede every registered fold; expanding values keep the exported
    # carrier table finite without affecting any fitted or evaluated sample.
    expanding = source["price_real_lag96"].expanding(min_periods=1)
    fallback_values = {
        column: getattr(expanding, statistic)()
        for column, statistic in SAFE_COLUMNS.items()
    }
    fallback = source[["time"]].copy()
    for column, values in fallback_values.items():
        fallback[column] = values
    fallback = fallback.set_index("time")
    for column in SAFE_COLUMNS:
        frame[column] = frame[column].fillna(frame["time"].map(fallback[column]))
    frame["price_real_safe48_std"] = frame["price_real_safe48_std"].fillna(0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    audit = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "continuous_source": str(args.continuous_source.resolve()),
        "continuous_source_sha256": sha256(args.continuous_source),
        "node_code": args.node_code,
        "folds": str(args.folds.resolve()),
        "folds_sha256": sha256(args.folds),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "rows": int(len(frame)),
        "definition": "rolling 48-point statistics of price_real_lag96, including the current row",
        "causal_boundary": "price_real_lag96 is 48 hours old; the rolling window contains no target-day realization",
        "columns_added": list(SAFE_COLUMNS),
        "pre_fold_missing_values": missing,
        "registered_missing_values": registered_missing,
        "final_missing_values": {
            column: int(frame[column].isna().sum()) for column in SAFE_COLUMNS
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
