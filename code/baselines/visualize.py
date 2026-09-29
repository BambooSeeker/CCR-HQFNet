# -*- coding: utf-8 -*-
"""
可视化绘图脚本 (论文定稿版)
==========================
自动读取 results/detailed_predictions.csv 并生成预测对比图。
支持多模型对比，x 轴使用真实时间戳，按 code 分图。
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_predictions(csv_path: str, output_dir: str = "results/plots",
                     plot_days: int = 7, code_filter: str = None):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"错误: 未找到文件 {csv_path}，请先运行 run_benchmarks.py")
        return

    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cols = [c for c in df.columns if c not in ("time", "code", "True")]

    # 确定要绘制的 code 列表
    if "code" in df.columns:
        codes = [code_filter] if code_filter else sorted(df["code"].unique())
    else:
        codes = [None]

    plot_points = plot_days * 48

    print(f"\n[可视化] {len(model_cols)} 个模型 × {len(codes)} 个 code")

    for code in codes:
        if code is not None:
            sub = df[df["code"] == code].copy()
            code_tag = f"_{code}"
        else:
            sub = df.copy()
            code_tag = ""

        # 只取前 plot_days 天
        sub = sub.iloc[:plot_points].reset_index(drop=True)
        if len(sub) == 0:
            continue

        x = sub["time"] if "time" in sub.columns else sub.index

        # ── 每个模型单独出图 ──
        for model in model_cols:
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(x, sub["True"], label="True", color="#1f77b4",
                    linewidth=1.5, alpha=0.85)
            ax.plot(x, sub[model], label=f"Predicted ({model})",
                    color="#ff7f0e", linestyle="--", linewidth=1.5, alpha=0.9)

            title = f"{model} — Prediction vs True"
            if code:
                title += f"  [{code}]"
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel("Price (CNY/MWh)", fontsize=11)
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.5)

            if "time" in sub.columns:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax.xaxis.set_major_locator(mdates.DayLocator())
                fig.autofmt_xdate(rotation=30)
            else:
                ax.set_xlabel("Time Steps (30min intervals)", fontsize=11)

            fig.tight_layout()
            save_path = output_dir / f"{model}{code_tag}_pred_{plot_days}d.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [保存] {save_path}")

        # ── 所有模型汇总对比图 ──
        if len(model_cols) > 1:
            fig, ax = plt.subplots(figsize=(18, 6))
            ax.plot(x, sub["True"], label="True", color="black",
                    linewidth=2.0, alpha=0.75)
            for model in model_cols:
                ax.plot(x, sub[model], label=model, linestyle="--",
                        linewidth=1.0, alpha=0.8)

            title = f"Multi-Model Comparison ({plot_days}-Day Test Set)"
            if code:
                title += f"  [{code}]"
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel("Price (CNY/MWh)", fontsize=11)
            ax.legend(loc="upper right", ncol=3, fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            if "time" in sub.columns:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax.xaxis.set_major_locator(mdates.DayLocator())
                fig.autofmt_xdate(rotation=30)
            else:
                ax.set_xlabel("Time Steps (30min intervals)", fontsize=11)

            fig.tight_layout()
            overview_path = output_dir / f"all_models{code_tag}_comparison_{plot_days}d.png"
            fig.savefig(overview_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  [保存] {overview_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       type=str, default="results/detailed_predictions.csv")
    parser.add_argument("--output",    type=str, default="results/plots")
    parser.add_argument("--days",      type=int, default=7, help="展示天数")
    parser.add_argument("--code",      type=str, default=None, help="只绘制指定 code，默认全部")
    args = parser.parse_args()

    plot_predictions(args.csv, args.output, args.days, args.code)
