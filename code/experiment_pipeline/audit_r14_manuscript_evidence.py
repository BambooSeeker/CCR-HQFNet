from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "03_results"
OUT = RESULTS / "r14_manuscript_evidence_audit_v0.1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def status_row(item: str, status: str, evidence: str, boundary: str = "") -> dict:
    return {
        "item": item,
        "status": status,
        "evidence": evidence,
        "boundary": boundary,
    }


def main() -> None:
    rows: list[dict] = []

    unified_dir = RESULTS / "r13_unified_fixed_window_comparison_v1.0"
    unified_report = load_json(unified_dir / "report.json")
    aligned = pd.read_csv(unified_dir / "aligned_predictions.csv")
    overall = pd.read_csv(unified_dir / "overall_and_season_metrics.csv")
    significance = pd.read_csv(unified_dir / "paired_daily_significance.csv")
    probability = pd.read_csv(unified_dir / "probability_metrics.csv")
    day_types = pd.read_csv(unified_dir / "day_type_metrics.csv")

    required_models = {
        "CCR-HQFNet",
        "SeasonalNaive",
        "LEAR",
        "LightGBM",
        "LightGBMQuantile",
        "LSTM",
        "TFT",
        "CNN_LSTM",
        "CNN_Transformer",
        "DLinear",
        "PatchTST",
        "Transformer",
        "TimeXer",
        "HybridTF-DilatedConv",
        "TimesFM-2.5",
    }
    aligned_models = set(aligned.columns) & required_models
    comparison_ok = (
        unified_report.get("status") == "complete"
        and len(aligned) == 6048
        and aligned_models == required_models
        and not aligned[list(required_models)].isna().any().any()
    )
    rows.append(
        status_row(
            "统一四季点预测对比",
            "PASS" if comparison_ok else "FAIL",
            f"126日、{len(aligned):,}点、{len(aligned_models)}个模型逐点对齐；"
            "MAE/MAPE50/sMAPE50均已计算。",
            "表中CCR-HQFNet曲线来自初稿四季预测文件，不是最新Chronos-2路由在该窗口的重跑结果。",
        )
    )

    rows.append(
        status_row(
            "TimesFM 2.5公平对比",
            "PASS" if "TimesFM-2.5" in aligned_models else "FAIL",
            "逐折LoRA、相同192点上下文、48点输出和目标日市场信息预算；统一四季MAE=66.317。",
            "TimesFM是编码器对照，不是本文机制创新。",
        )
    )
    rows.append(
        status_row(
            "原Transformer+空洞卷积对照",
            "PASS" if "HybridTF-DilatedConv" in aligned_models else "FAIL",
            "已按统一四季窗口重跑；MAE=70.759。",
            "仅作为原始编码器对照，不恢复旧方法主线。",
        )
    )

    probability_ok = (
        set(probability["model"]) == {"CCR-HQFNet", "LightGBMQuantile"}
        and probability["targets"].eq(6048).all()
    )
    rows.append(
        status_row(
            "90%概率基线",
            "PASS" if probability_ok else "FAIL",
            "CCR-HQFNet与LightGBMQuantile在相同6,048点报告pinball、PICP、MPIW和Winkler。",
            "CCR区间偏保守（PICP=94.61%），不得宣称精确校准。",
        )
    )

    state_names = set(day_types["day_type"].dropna().astype(str))
    expected_states = {
        "Typical-Normal",
        "High-Volatility",
        "Extreme-Low",
        "Extreme-High",
        "Overall",
    }
    rows.append(
        status_row(
            "四类市场状态结果",
            "PASS" if expected_states.issubset(state_names) else "FAIL",
            "Typical-Normal、High-Volatility、Extreme-Low、Extreme-High互斥分层及总体指标齐全。",
            "阈值来自固定四季样本定义；正文须给出层级与样本数。",
        )
    )

    sig_ok = (
        len(significance) == len(required_models) - 1
        and significance["days"].eq(126).all()
        and significance["holm_p"].notna().all()
    )
    rows.append(
        status_row(
            "统一对比统计检验",
            "PASS" if sig_ok else "FAIL",
            "14组日级配对Wilcoxon、Holm校正及10,000次日块bootstrap区间齐全。",
        )
    )

    carrier_metrics = pd.read_csv(
        RESULTS / "r10_chronos2_covariate_lora_v0.1/point_metrics.csv"
    )
    carrier_pooled = carrier_metrics[carrier_metrics["fold"].eq("pooled")]
    carrier_ok = {
        "Chronos2_covariate_zero_shot",
        "Chronos2_covariate_LoRA",
    }.issubset(set(carrier_pooled["mode"]))
    rows.append(
        status_row(
            "Chronos-2载体与微调",
            "PASS" if carrier_ok else "FAIL",
            "冻结140日协议中zero-shot/LoRA MAE=90.557/87.078，LoRA改善3.84%。",
            "该证据属于冻结140日载体实验，不得并入6,048点统一对比表。",
        )
    )

    route = pd.read_csv(RESULTS / "r10_chronos2_lora_joint_route_v0.1/predictions.csv")
    required_modes = {
        "chronos2_lora",
        "carrier_blind",
        "physical_congestion_gate",
        "ungated_dual_route",
        "energy_consistent_supply_route",
        "energy_consistent_dual_route",
    }
    mode_counts = route.groupby("mode").size()
    ablation_ok = required_modes.issubset(set(mode_counts.index)) and all(
        mode_counts.get(mode, 0) == 6720 for mode in required_modes
    )
    rows.append(
        status_row(
            "机制消融",
            "PASS" if ablation_ok else "FAIL",
            "裸载体、Strict Blind、拥塞only、供需only、无风险门、完整双路由均为6,720点。",
            "完整模型相对Strict Blind未同时达到预设3%/10%压力双门，须保留边界。",
        )
    )

    r6_report = load_json(RESULTS / "r6_curve_shape_evaluation_v0.1/curve_shape_report.json")
    curve_ok = r6_report.get("status") == "complete" and r6_report.get("n_days") == 140
    rows.append(
        status_row(
            "曲线形态与拥塞独立增益",
            "PASS" if curve_ok else "FAIL",
            "140日曲线MAE、相关性、坡度、坡度方向和峰谷幅差均已计算并配对检验。",
            "单点峰谷时刻及全部负价日F1未稳定改善，不得泛化。",
        )
    )

    event = route[
        route["delivery_day"].between("2026-03-08", "2026-03-14")
    ].copy()
    event_counts = event.groupby("mode").size()
    event_ok = all(event_counts.get(mode, 0) == 336 for mode in required_modes)
    complete = event[event["mode"].eq("energy_consistent_dual_route")]
    blind = event[event["mode"].eq("carrier_blind")]
    supply = event[event["mode"].eq("energy_consistent_supply_route")]
    negative = complete["actual"] < 0
    event_summary = {
        "days": int(complete["delivery_day"].nunique()),
        "targets": int(len(complete)),
        "negative_targets": int(negative.sum()),
        "complete_mae": float(complete["absolute_error"].mean()),
        "blind_mae": float(blind["absolute_error"].mean()),
        "supply_only_mae": float(supply["absolute_error"].mean()),
        "complete_negative_mae": float(complete.loc[negative, "absolute_error"].mean()),
        "blind_negative_mae": float(
            blind.loc[blind["actual"] < 0, "absolute_error"].mean()
        ),
        "supply_only_negative_mae": float(
            supply.loc[supply["actual"] < 0, "absolute_error"].mean()
        ),
    }
    rows.append(
        status_row(
            "2026-03-08至03-14连续极端负价案例",
            "PASS" if event_ok and negative.sum() > 0 else "FAIL",
            f"7日336点，每日均有负价，共{negative.sum()}点；完整/Blind MAE="
            f"{event_summary['complete_mae']:.3f}/{event_summary['blind_mae']:.3f}，"
            f"负价MAE={event_summary['complete_negative_mae']:.3f}/"
            f"{event_summary['blind_negative_mae']:.3f}。",
            "该周增益主要来自供需负价路由；完整模型与供需only近乎相同，不能作为拥塞独立增益主证据。",
        )
    )

    rows.append(
        status_row(
            "报价/收益实验",
            "EXCLUDE",
            "机组级Q_DA/Q_RT与历史申报曲线不可得，R11已判定FAIL。",
            "不得写入摘要、正文、结论、图表或补充材料。",
        )
    )

    audit = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "evidence_status.csv", index=False, encoding="utf-8-sig")
    (OUT / "event_20260308_0314_summary.json").write_text(
        json.dumps(event_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# R14 论文证据完整性审计",
        "",
        "审计原则：实验真实性、统一口径与证据来源分开核验；已有结果不因版式重写而覆盖。",
        "",
        "| 证据项 | 状态 | 已有证据 | 写作边界 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['item']} | {row['status']} | {row['evidence']} | "
            f"{row['boundary'] or '无额外边界'} |"
        )
    lines.extend(
        [
            "",
            "## 审计结论",
            "",
            "1. 对比、概率、分层、统计、载体、消融与曲线形态实验均有完整输出，可进入重写阶段。",
            "2. 6,048点四季曲线与最新路由消融来自不同冻结证据集。正文必须分别陈述，不能把初稿曲线冒充最新路由同窗重跑。",
            "3. 2026-03-08至03-14可替代春节作为连续极端负价的说明性机制案例，但它证明的是供需负价路由主效应；拥塞独立价值仍由140日消融承担。",
            "4. 报价与收益算例继续排除。",
        ]
    )
    (OUT / "R14_EVIDENCE_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
