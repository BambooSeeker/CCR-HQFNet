"""Regenerate final Figures 7 and 8 from released predictions, without fitting.

Requires numpy, pandas and matplotlib. Pass the extracted research_archive
directory to --archive. Outputs are reference renderings; font/platform changes
can change their appearance without changing the underlying plotted values.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager


SEASONS = {"spring": 1392, "summer": 1488, "autumn": 1680, "winter": 1488}
STATES = {
    "Typical-Normal": 3792,
    "High-Volatility": 624,
    "Extreme-Low": 720,
    "Extreme-High": 912,
}
MODELS = [
    "CCR-HQFNet",
    "Fine-tuned Chronos-2",
    "LightGBM",
    "TimesFM-2.5",
    "HybridTF-DilatedConv",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_sources(archive):
    revision = archive / "r2_experiment_plan_20261018"
    primary_path = revision / "joint_routing_trial_20261022/trial_predictions_LOCAL_ONLY.csv"
    require(primary_path.is_file(), "--archive must point to the extracted research_archive directory.")
    candidates = sorted(archive.glob("CCR_HQFNet_*/03_results/r13_unified_fixed_window_comparison_v1.0"))
    require(len(candidates) == 1, "Expected exactly one archived primary baseline directory.")
    comparison = candidates[0]
    saved = pd.read_csv(primary_path, parse_dates=["time", "delivery_day"])
    full = saved.loc[saved.model.eq("Full")].set_index("time").sort_index()
    native = saved.loc[saved.model.eq("Chronos_native")].set_index("time").sort_index()
    require(len(full) == 6048 and full.index.is_unique, "Full must contain 6,048 unique targets.")
    require(len(native) == 6048 and native.index.is_unique, "Carrier must contain 6,048 unique targets.")
    require(full.index.equals(native.index), "Carrier and Full timestamps differ.")
    require(np.allclose(full.actual, native.actual, rtol=0, atol=1e-8), "Carrier and Full targets differ.")
    require(full.delivery_day.nunique() == 126, "Expected 126 primary delivery days.")
    require(full.groupby("delivery_day").size().eq(48).all(), "Each delivery day must contain 48 intervals.")
    require(full.season.value_counts().to_dict() == SEASONS, "Unexpected seasonal target counts.")
    require(np.isfinite(full[["actual", "predicted", "q05", "q50", "q95"]]).all().all(), "Nonfinite Full predictions.")
    require(np.isfinite(native.predicted).all(), "Nonfinite carrier predictions.")
    require(np.allclose(full.predicted, full.q50, rtol=0, atol=1e-8), "Point forecast must equal final q50.")
    require(full.q05.le(full.q50 + 1e-8).all() and full.q50.le(full.q95 + 1e-8).all(), "Quantiles cross.")
    require(full.q05.ge(-200 - 1e-8).all(), "Final quantiles violate the market floor.")
    require(abs((full.actual - full.predicted).abs().mean() - 59.55745866961517) < 1e-8,
            "Full does not match the final-revision primary forecast version.")

    baseline_path = comparison / "aligned_predictions.csv"
    baseline = pd.read_csv(baseline_path, parse_dates=["time"]).set_index("time")
    require(baseline.index.is_unique and len(baseline) == len(full), "Baseline targets are duplicated or incomplete.")
    require(baseline.index.difference(full.index).empty and full.index.difference(baseline.index).empty,
            "Baseline and Full target sets differ.")
    baseline = baseline.reindex(full.index)
    # The older baseline CSV stores targets as float32; keep final Full's targets.
    require((baseline.actual - full.actual).abs().max() < 2e-4, "Baseline targets differ beyond storage precision.")
    labels_path = comparison / "day_classification.csv"
    labels = pd.read_csv(labels_path, parse_dates=["delivery_day"])
    require(labels.delivery_day.is_unique, "Day classifications contain duplicate delivery days.")
    frame = full.reset_index().merge(labels[["delivery_day", "day_type"]], on="delivery_day",
                                    how="left", validate="many_to_one")
    require(frame.day_type.value_counts().to_dict() == STATES, "Incomplete or changed market-state partition.")
    frame = frame.set_index("time").sort_index()
    frame["CCR-HQFNet"] = frame.q50
    frame["Fine-tuned Chronos-2"] = native.predicted
    for name in MODELS[2:]:
        frame[name] = baseline[name]
    require(np.isfinite(frame[MODELS]).all().all(), "Nonfinite or missing plotted model predictions.")
    return frame, [primary_path, baseline_path, labels_path]


def validate_reference_tables(frame, archive):
    paths = [archive / "manuscript_reference" / f"table_object_{n:02}.csv" for n in (21, 22)]
    for path in paths:
        require(path.is_file(), f"Missing final table reference: {path}")
    overall, conditional = [pd.read_csv(path) for path in paths]
    conditional["Market state"] = conditional["Market state"].ffill()
    for name in MODELS:
        row = overall.loc[overall.Model.eq(name)]
        require(len(row) == 1, f"Missing/ambiguous Table 7 reference for {name}.")
        value = float((frame.actual - frame[name]).abs().mean())
        require(f"{value:.3f}" == f"{float(row.iloc[0].MAE):.3f}", f"Table 7 MAE mismatch for {name}.")
        for state in STATES:
            group = frame.loc[frame.day_type.eq(state)]
            row = conditional.loc[conditional.Model.eq(name) & conditional["Market state"].str.casefold().eq(state.casefold())]
            require(len(row) == 1, f"Missing/ambiguous Table 8 reference for {state}/{name}.")
            value = float((group.actual - group[name]).abs().mean())
            require(f"{value:.3f}" == f"{float(row.iloc[0].MAE):.3f}", f"Table 8 MAE mismatch for {state}/{name}.")
    return paths


def figure7(frame):
    fig, axes = plt.subplots(4, 1, figsize=(9.4, 7.1), layout="constrained")
    for ax, season in zip(axes, SEASONS):
        subset = frame.loc[frame.season.eq(season)].reset_index().sort_values("time")
        # Do not draw curves or interval bands across unevaluated calendar gaps.
        segments = subset.time.diff().gt(pd.Timedelta(minutes=30)).cumsum()
        for number, (_, group) in enumerate(subset.groupby(segments)):
            ax.fill_between(group.time, group.q05, group.q95, color="#64A7AF", alpha=.25,
                            label="90% interval" if number == 0 else None)
            ax.plot(group.time, group.actual, color="#303030", lw=.65,
                    label="Observed" if number == 0 else None)
            ax.plot(group.time, group.q50, color="#B34D42", lw=.65,
                    label="CCR-HQFNet" if number == 0 else None)
        ax.set_title(season.title(), loc="left", fontsize=10)
        ax.set_ylabel("CNY/MWh")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].legend(ncol=3, loc="upper right", frameon=False, fontsize=8)
    return fig


def figure8(frame):
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.9), layout="constrained")
    for ax, state in zip(axes.flat, STATES):
        group = frame.loc[frame.day_type.eq(state)]
        values = [(group.actual - group[name]).abs().mean() for name in MODELS]
        ax.barh(np.arange(len(MODELS)), values, color=["#B34D42"] + ["#628F9B"] * 4, height=.65)
        ax.set_yticks(np.arange(len(MODELS)), MODELS, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(state, fontsize=10)
        ax.set_xlabel("MAE (CNY/MWh)")
    return fig


def select_font():
    for name in ("Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "serif"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Extracted v2.0.0 research_archive directory")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "outputs/final_revision_figures")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--check-only", action="store_true", help="Validate plotted sources without writing figures")
    args = parser.parse_args()
    require(72 <= args.dpi <= 1200, "--dpi must be between 72 and 1200.")
    archive = args.archive.expanduser().resolve()
    frame, sources = load_sources(archive)
    sources.extend(validate_reference_tables(frame, archive))
    output = args.output.expanduser().resolve()
    # Reject destinations in the source archive even when a parent path is used.
    require(output != archive and archive not in output.parents and output not in archive.parents,
            "Output must be outside the source archive and its parent directories.")
    summary = {
        "targets": len(frame), "delivery_days": int(frame.delivery_day.nunique()),
        "MAE": float((frame.actual - frame.q50).abs().mean()),
        "season_MAE": {s: float((g.actual - g.q50).abs().mean()) for s, g in frame.groupby("season")},
        "state_MAE": {s: {m: float((g.actual - g[m]).abs().mean()) for m in MODELS}
                      for s, g in frame.groupby("day_type")},
        "table_7_and_8_MAE_checks": "PASS", "sources": [str(path) for path in sources],
    }
    if not args.check_only:
        output.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"font.family": select_font(), "font.size": 10,
                             "pdf.fonttype": 42, "svg.fonttype": "none",
                             "svg.hashsalt": "CCR-HQFNet-final-primary",
                             "axes.spines.top": False, "axes.spines.right": False}):
            files = []
            for name, plot in [("Fig7_four_season_forecasts", figure7), ("Fig8_market_state_mae", figure8)]:
                fig = plot(frame)
                try:
                    for ext in ("png", "pdf", "svg"):
                        target = output / f"{name}.{ext}"
                        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
                        fig.savefig(target, dpi=args.dpi, facecolor="white", metadata=metadata)
                        files.append(str(target))
                finally:
                    plt.close(fig)
            summary["generated_reference_figures"] = files
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
