# Final Revision Output Index

This index covers all 12 captioned figures and 29 captioned tables. The v2.0.2 repository uses the unchanged v2.0.0 research archive and v2.0.1 experimental predictions. Archive reference exports are manuscript snapshots, not proof of independent recalculation. Current point/tail provenance is in `outputs/final_revision_audit`, not the historical r21 traceability files. A source listed below establishes available evidence; it does not imply a portable renderer exists for every output.

## Locations and Source Versions

Paths below are relative to these locations:

| Alias | Location |
| --- | --- |
| `R` | This repository root |
| `A` | User-supplied extracted `research_archive` directory |
| `P` | The single `A/CCR_HQFNet_*/` project directory |
| `B` | `A/r2_experiment_plan_20261018` |
| `J` | `B/joint_routing_trial_20261022` |
| `C` | `B/four_comment_experiment_closeout_20261023` |
| `H` | `B/response_readiness_fresh_20261023` |
| `V` | `A/manuscript_reference` |
| `F` | `R/outputs/final_revision_figures` |
| `S` | `R/outputs/final_revision_audit` |

| Source | Exact saved input or output | Version boundary |
| --- | --- | --- |
| Primary | `J/trial_predictions_LOCAL_ONLY.csv` | Current Full and matched component predictions; use `model=Full` for primary CCR-HQFNet. |
| Baselines | `P/03_results/r13_unified_fixed_window_comparison_v1.0/aligned_predictions.csv` | Baseline columns only. Its CCR-HQFNet and quantile columns are historical. Use current Primary targets; historical targets have float32 storage differences. |
| States | `P/03_results/r13_unified_fixed_window_comparison_v1.0/day_classification.csv` | Fixed daily market-state partition. |
| Probability | `C/consistent_probability_predictions_LOCAL_ONLY.csv` | Current probability comparison; summaries in `consistent_probability_tables.csv` and `consistent_quantile_reliability.csv` in the same directory. |
| Controller | `C/current_daily_controller_summary.csv`, `C/current_test_actions_LOCAL_ONLY.csv` | Current primary policies/actions. |
| B.1 corrected counts | `S/B1_calibration_day_counts.csv`, `S/B1_calibration_fold_counts.csv` | Reconstructed from the frozen selected OOF/validation inputs and daily signatures; authoritative corrected negative counts, 126 day records and 20 fold ranges. |
| Episodes/stress | `P/03_results/r15_stress_route_seed42_v1.0/predictions.csv` | Archived seed42 implementation, deliberately distinct from current primary Full; do not pool their action denominators. Scope: `C/stress_source_scope.json`. |
| Sensitivity | `P/03_results/r18_reviewer1_negative_dense_controller_audit_v1.0/` | Archived March controller audit: `daily_policy_audit.csv`, `sensitivity_metrics.csv`, `sensitivity_predictions.csv`. |
| Seeds | `P/03_results/r17_reviewer1_statistical_inference_v1.0/seed_stability_detail.csv` | Saved baseline repeated-fit results. |
| Capacity | `C/matched_capacity_predictions_LOCAL_ONLY.csv`, `C/matched_capacity_comparison.csv` | Matched-capacity comparison, separate from ordinary residual baseline. |
| Context | `P/03_results/r38_context_length_sensitivity_yq_v1.0/predictions_context{96,192,384}.csv` | Saved context-length outputs; five-quantile mean pinball convention. |
| Current traceability | `S/point_metric_traceability.csv`, `S/tail_metric_traceability.csv`, `S/march_version_comparison.csv` | Revised P01 and complete point/tail records. Historical archives remain retained. |
| A.1 final inference | `S/paired_statistics.csv` | Definitive final 20,000-resample baseline comparisons. `H/historical_point_benchmark_check.csv` uses 10,000 resamples and is not its exact CI source. |

The final A.1 calculation came from the author-side `r2_experiment_plan_20261018/deliver_comment1_20261023.py`, lines 143-161: segment-aware circular seven-day blocks, 20,000 draws, seed 20260929, one continuously advancing RNG across model comparisons. That integration script is not included in the frozen archive. `A/r17_build_segment_aware_inference.py` supplies the original helper definitions, but its original entry point uses historical Full predictions and is not a turnkey final A.1 generator.

Other historical Full sources include `P/03_results/r16_state_conditioned_ablation_v1.0/predictions.csv`, the Full fields in `P/03_results/r19_reviewer1_probabilistic_audit_v1.0/aligned_probability_predictions.csv`, and `P/03_results/r21_reviewer1_numerical_traceability_v1.0/point_metric_traceability.csv`. Their presence is historical retention, not authority for current primary metrics. No new figure generator reads those historical Full metrics.

## Supported Figure Generation

Run from `R`, with numpy, pandas and matplotlib installed:

```sh
python -B regenerate_primary_figures.py --archive /path/to/research_archive --check-only
python -B regenerate_primary_figures.py --archive /path/to/research_archive
```

The generator reads saved predictions only. It validates 6,048 unique targets, 126 complete delivery days, season/state populations, target alignment, point/median identity, ordered floor-respecting quantiles and the final Full forecast version. It compares all 25 plotted model MAEs with archived Tables 7/8 at three decimals. It writes Fig. 7/8 PNG (600 dpi), PDF and editable SVG to `F`; `--output` and `--dpi` are configurable. It prefers the open-source Liberation Serif font, then DejaVu Serif. Fonts can vary across platforms, so these are reference renderings rather than byte-identical manuscript image replacements. Both figures were generated and visually inspected. Existing manuscript figure assets are not replaced.

Fig. 2 has its own `R/regenerate_routing_diagram.py` and existing `F/Fig2_factorized_routing.{png,pdf,svg}`. It is an editable conceptual illustration of the method, not a numerical recalculation.

Fig. 6 is generated by `python -B regenerate_rolling_protocol.py`. The PNG, PDF and editable SVG in `F/Fig6_rolling_protocol.*` show the 90/7/7-day rolling protocol using relative delivery days. This current diagram supersedes the historical `V/image6.png` snapshot.

The existing `R/reproduce_revision.py --archive /path/to/research_archive --mode verify --output /path/outside/archive` recalculates primary component summaries, consistent probability summaries and action-risk intervals, and renders Fig. C.1. It is not an all-output regeneration command. Archived experiment scripts can contain author-specific paths and fitting entry points; they are not automatically portable saved-data renderers.

## All Figures

Historical raster references below are under `V`. The archive scan found no SVG, Draw.io, Mermaid, PowerPoint or Visio source files; therefore editable conceptual originals are not established for Figs. 1/3. Current Figs. 2/6 and new Figs. 7/8 have verified SVG outputs in this repository.

| Figure | Subject | Reference raster | Source and regeneration coverage |
| --- | --- | --- | --- |
| 1 | Overall architecture/information boundary | `image1.png` | Conceptual method illustration. Raster available; original editable artwork/portable renderer not established. |
| 2 | Factorized routing and selective control | `image2.png` | Historical raster retained. Current conceptual source and PNG/PDF/SVG: `regenerate_routing_diagram.py`, `F/Fig2_factorized_routing.*`. |
| 3 | Market timeline and seasonal registry | `image3.png` | Conceptual timeline plus registry specification. Raster available; exact editable layout not established. |
| 4 | Supply, renewable and net-load inputs | `image4.png` | Inputs available under `A/raw_inputs/processed_features_hybrid.csv` and `A/raw_inputs/YQFDC-01.csv`; exact plotting population/transforms and renderer not established. |
| 5 | Annual price/congestion and distributions | `image5.png` | Raw/processed inputs available under `A/raw_inputs`; exact artwork selection/renderer not established. |
| 6 | Rolling 90/7/7-day protocol | `F/Fig6_rolling_protocol.png` | Current conceptual source: `regenerate_rolling_protocol.py`; PNG/PDF/SVG available. Relative delivery days follow Table 5(a). |
| 7 | Seasonal forecasts and 90% intervals | `image13.png` | Current Primary, `model=Full`. Portable generator provided; `F/Fig7_four_season_forecasts.{png,pdf,svg}`. No historical Full values used. |
| 8 | MAE in four market states | `image14.png` | Current Primary Full/Chronos_native, Baselines and States. Portable generator provided; `F/Fig8_market_state_mae.{png,pdf,svg}`. |
| 9 | Empirical supply/congestion factorization | `image9.png` | Fold-level saved states/components exist, but exact figure population and final renderer not established. Raster is the reference. |
| 10 | Recurrent-negative-price mechanism | `image10.png` | Episodes/stress saved predictions, 8-14 March 2026. Data available for recalculation; standalone final renderer not established. |
| B.1 | Activation-threshold sensitivity | `image11.png` | Sensitivity CSVs and `threshold_sensitivity.{png,pdf}` available. Numerical inputs and archived rendering available; no new portable renderer claimed. |
| C.1 | Quantile reliability and coverage | `image15.png` | Probability inputs and summaries; `C/Fig_C1_consistent_reliability.{png,pdf}`. Existing verify mode renders this figure. |

`V/image7.png`, `V/image8.png` and `V/image12.png` are retained older media. They are not the final Fig. 7/8/C.1 references.

## All Captioned Tables

The 42 Word table objects comprise 29 captioned tables and 13 equation-layout objects. Objects 02-14 contain numbered equations (1)-(15), not numerical result tables or additional algorithm-result tables. Their ordinary-text CSV exports omit Office Math and do not preserve those equations. Table 5(a) is protocol specification and Table 5(b) is metric definitions.

The object column maps to `V/table_object_NN.csv`. Repository `outputs/final_manuscript_tables/table_NN.csv` files, where present, are current document transcriptions, not independent numerical recalculations. Conceptual/specification tables should be checked against method definitions and configurations; result tables should be checked against saved numerical inputs.

| Table | Object | Type | Authoritative evidence and coverage/gap |
| --- | --- | --- | --- |
| 1 | 01 | Editorial literature mapping | Manuscript references; transcription only, no numerical experiment to regenerate. |
| 2 | 15 | Information-boundary specification | Manuscript and archived dataset/protocol definitions; transcription, not a recalculated result. |
| 3 | 16 | Descriptive statistics | Primary actual values and seasons; sufficient saved inputs, no standalone final table formatter supplied here. |
| 4 | 17 | State definitions/counts | States; partition counts checked by new figure generator, threshold definitions remain specification. |
| 5(a) | 18 | Evaluation protocol | `J/protocol.json`, frozen fold selections and archived fold metadata; specification, not an accuracy table. |
| 5(b) | 19 | Metric definitions | Manuscript metric definitions; not an empirical result table. |
| 6 | 20 | Model-setting specification | Archived model configurations/reports; transcription, no fitting required to inspect settings. |
| 7 | 21 | Primary point accuracy | Primary Full/Chronos_native plus Baselines. Generator checks the five plotted model MAEs; it does not regenerate the entire table. Current Full provenance: `S/point_metric_traceability.csv`, P01. |
| 8 | 22 | Conditional point accuracy | Primary, Baselines, States. Generator checks all 20 plotted state/model MAEs; full table formatter not supplied. |
| 9 | 23 | Episode pathway results | Episodes/stress, March 1-14 and June 22-28. Revised point/tail records in `S`; archived event implementation is distinct from current primary Full. |
| 10 | 24 | Pooled probability results | Probability; `C/consistent_probability_tables.csv`. Existing verify mode recalculates summaries. |
| A.1 | 25 | Paired statistics/seed summaries | `S/paired_statistics.csv` (20k final comparisons) and Seeds. Exact final integration entry point not archived; do not substitute the 10k historical benchmark check. |
| B.1 | 26 | Calibration support/decisions | Corrected negative ranges: `S/B1_calibration_fold_counts.csv`; full day/source records: `S/B1_calibration_day_counts.csv`. Portable saved-input checker: `R/verify_calibration_counts.py`. Decisions remain sourced from Controller; the 11 old archive-snapshot ranges are historical, not current authority. |
| B.2 | 27 | March calibration decisions | Sensitivity `daily_policy_audit.csv`; archived episode policy, not current primary controller. |
| B.3 | 28 | March screening/sensitivity | Sensitivity metrics/predictions; saved numerical evidence available, final table renderer not supplied. |
| B.4 | 29 | Action-risk denominators | Controller and Episodes/stress, `C/current_heldout_action_risk.csv`, `C/archived_stress_action_risk.csv`. Keep primary, stress and March subset denominators separate. |
| C.1 | 30 | Seasonal probability results | Probability grouped by season; existing verify mode recalculates summaries. |
| C.2 | 31 | Conditional probability results | Probability and States; existing verify mode recalculates summaries. |
| C.3 | 32 | Autumn upper-tail diagnostics | Probability autumn observations and highest-5% selection; existing verify mode recalculates summaries. |
| C.4 | 33 | Shift/floor probability audit | Probability, `H/interval_stage_metrics.csv`; saved stages support recalculation, no standalone final table formatter supplied. |
| D.1 | 34 | Point provenance records | `S/point_metric_traceability.csv`, all records including revised P01. Historical r21 CSV is not current authority. |
| D.2 | 35 | Tail provenance records | `S/tail_metric_traceability.csv`, all records. Snapshot is a transcription; source predictions remain the numerical authority. |
| E.1 | 36 | Predictor/availability specification | Raw input schemas, fold feature lists and manuscript availability definitions; not a result table. |
| E.2 | 37 | Preprocessing/software/settings | Frozen selections, protocol, environment and saved model configurations; not a recalculated result. |
| F.1 | 38 | Context sensitivity | Context saved predictions; no fitting needed to recompute metrics. Mean pinball uses five quantiles, unlike the three-quantile WIS comparison. |
| G.1 | 39 | Stress window registry | Episodes/stress dates/actuals and `C/stress_source_scope.json`; counts recalculable from saved predictions. |
| G.2 | 40 | Paired stress pathways | Episodes/stress saved modes. Point metrics recalculable; full paired-inference table generator not supplied here. |
| G.3 | 41 | Matched primary components | Primary, Capacity, `J/paired_analysis.json`, `H/fresh_paired_component_statistics.csv`, `C/matched_capacity_comparison.csv`. Existing verify mode writes component point summaries, not the whole formatted inference table. |
| G.4 | 42 | Conditional primary components | Primary/States and `J/conditional_metrics.csv`; point estimates recalculable, standalone final table formatter not supplied. |

## Corrected B.1 Counts

The portable `R/verify_calibration_counts.py` reads `J/all_selection_frozen.json`, the selected `J/folds/<key>/candidate_<candidate>/oof_shrink<shrink>_LOCAL_ONLY.csv` and `validation_shrink<shrink>_LOCAL_ONLY.csv`, and `B/congestion_validation_20261022/<key>/daily_signatures_LOCAL_ONLY.csv`. Target days come from current Primary Full. Its pure nearest-day helper preserves the rules in `P/03_results/r47_timestamp_lags_v1.0/finalization_source/r36_canonical_finalize.py`, lines 107-112: pool-standardized squared distance, missing-value handling, stable sorting, 14 nearest OOF days. These are combined with seven saved validation days, following `B/evaluate_joint_routing_trial.py:18` and `B/complete_four_comment_evidence_20261023.py:178-192`. No models are fitted/loaded and no policy or prediction is rerun.

```sh
python -B verify_calibration_counts.py --archive /path/to/research_archive
```

This command writes `S/B1_calibration_day_counts.csv` (126 rows, selected OOF dates, validation dates, source paths, chronology checks and per-day negatives) and `S/B1_calibration_fold_counts.csv` (20 rows, target-day counts and corrected minimum/maximum negatives). Use these corrected counts for the revised final Table B.1; the frozen archive reference retains the historical ranges below.

The checker was run successfully. All 126 windows had 21 distinct days, 1,008 unique timestamps, 48 observations/day, no target overlap and calibration timestamps strictly before the target. Counting `actual < 0` gives this correction mapping for Table B.1 column 3 (Word table object 26); header-inclusive row numbers are provided. The clean DOCX and archive CSV had the same historical ranges at the initial check; this task does not edit the manuscript.

| Fold | Word row | Historical range | Corrected range |
| --- | --- | --- | --- |
| Spring-01 | 2 | 13-56 | 15-55 |
| Spring-02 | 3 | 19-46 | 14-46 |
| Spring-03 | 4 | 22-35 | 22-32 |
| Spring-04 | 5 | 20-31 | 20-46 |
| Spring-05 | 6 | 40-40 | 39-39 |
| Summer-01 | 7 | 3-8 | 5-16 |
| Summer-02 | 8 | 1-6 | 3-8 |
| Autumn-05 | 16 | 0-0 | 0-6 |
| Winter-01 | 17 | 0-8 | 0-6 |
| Winter-02 | 18 | 2-7 | 2-5 |
| Winter-03 | 19 | 2-6 | 1-2 |

The other nine ranges agree. These corrections concern integer counts, not rounding. The portable helper was cross-checked against the exact archived `nearest` function for all 126 target days: every selected OOF-day list agreed. Both written CSVs matched a fresh reconstruction. The current repository transcription `outputs/final_manuscript_tables/table_26.csv` was then checked and all 20 negative ranges agreed with the corrected counts.

The negative-range verification is complete and its corrected numerical sources are now reproducible. It remains conditional on the released saved OOF/validation labels and signatures; upstream label/signature generation and policy evaluation are outside this check. Manuscript corrections are handled separately, and historical archive snapshots remain unchanged.

## Remaining Coverage Limits

All captioned outputs have a reference snapshot. All numerical result tables have mapped saved numerical sources; editorial/specification tables have manuscript or configuration sources. This is a source inventory, not a claim that every cell has been independently recomputed by one portable command.

Portable standalone renderers are supplied here for Figs. 2/6/7/8, with Fig. C.1 supported by existing verify mode. The B.1 count checker supplies numerical CSVs, not a figure renderer. Exact editable originals for other conceptual figures, exact plotting transformations/populations for Figs. 4/5/9, and standalone final renderers for Fig. 10/B.1 are not established. Thus Figs. 4/5/9 still lack a fully established data-to-artwork mapping; other conceptual figures have raster references but no verified editable originals. Saved numerical inputs support substantially more recalculation than these renderers cover; no claim of complete figure/table regeneration is made.
