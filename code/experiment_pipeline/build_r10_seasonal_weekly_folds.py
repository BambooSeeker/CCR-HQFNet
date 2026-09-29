from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    source_folds = json.loads(args.source.read_text(encoding="utf-8"))
    folds = []
    audits = []
    for fold in source_folds:
        ordered_days = fold["train"] + fold["val"] + fold["test"]
        if len(ordered_days) != 125 or len(set(ordered_days)) != 125:
            raise ValueError(f"Invalid long window: {fold.get('name', '<unnamed>')}")
        for week_index in range(4):
            offset = 7 * week_index
            weekly = {
                **{key: value for key, value in fold.items() if key not in {"train", "val", "test"}},
                "name": f"{fold['name']}_wk{week_index + 1}",
                "seasonal_origin": fold["name"],
                "weekly_block": week_index + 1,
                "train": ordered_days[offset : offset + 90],
                "val": ordered_days[offset + 90 : offset + 97],
                "test": ordered_days[offset + 97 : offset + 104],
            }
            if (len(weekly["train"]), len(weekly["val"]), len(weekly["test"])) != (90, 7, 7):
                raise ValueError(f"Invalid weekly fold: {weekly['name']}")
            folds.append(weekly)
            audits.append(
                {
                    "name": weekly["name"],
                    "seasonal_origin": fold["name"],
                    "weekly_block": week_index + 1,
                    "train_start": weekly["train"][0],
                    "train_end": weekly["train"][-1],
                    "validation_start": weekly["val"][0],
                    "validation_end": weekly["val"][-1],
                    "test_start": weekly["test"][0],
                    "test_end": weekly["test"][-1],
                }
            )

    args.output.write_text(
        json.dumps(folds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "status": "locked_rolling_weekly_retraining_protocol",
        "selection_rule": (
            "Split every preregistered 28-day seasonal test window into four "
            "consecutive non-overlapping weeks. Before each week, roll the "
            "90-day training and 7-day validation windows forward by seven "
            "complete delivery days and retrain all learned components. No "
            "errors or realized future prices select dates."
        ),
        "train_days": 90,
        "validation_days": 7,
        "test_days": 7,
        "fold_count": len(folds),
        "seasonal_origins": len(source_folds),
        "total_test_days": sum(len(fold["test"]) for fold in folds),
        "folds": audits,
        "source_hash": sha256(args.source),
        "code_hash": sha256(Path(__file__)),
    }
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
