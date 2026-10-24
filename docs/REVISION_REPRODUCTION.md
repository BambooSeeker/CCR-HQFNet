# Reproducing the second revision

## Archive layout

Extract the release archive to a short path, such as `D:/CCR/research_archive`
on Windows. Long checkpoint filenames can exceed legacy Windows path limits.

The original project directory retains its archive identifier
`CCR_HQFNet_评审修正_物理拥塞路由_20260721`. In that directory:

- `02_experiments/data_locked/` contains processed inputs, labels and fold-specific foundation-model predictions.
- `02_experiments/code/` contains data construction, adaptation, routing and experiment sources.
- `00_control/` contains chronological split registries.
- `03_results/` contains fitted checkpoints, prediction outputs and experimental summaries.

`raw_inputs/` supplies historical observations used by timestamp-based lags.
`r2_experiment_plan_20261018/` contains the second-revision implementation,
selected residual models, calibration records, matched predictions and
evaluation code. `manuscript_reference/` holds table-cell exports and embedded
figure assets for comparison with the submitted document.

## Numerical reference

Use the v2.0.1 repository with the complete v2.0.0 data archive. The downloader
reads the archive's own release tag, so both versions are selected explicitly.
The authoritative final point and negative-price indices are the repository's
`outputs/final_revision_audit/point_metric_traceability.csv` and
`tail_metric_traceability.csv`. The P01 primary row reports MAE 59.55745867.
Older r21 indices inside the archive describe earlier analyses and are retained
as provenance. Final Table A.1 uses `outputs/final_revision_audit/paired_statistics.csv`
with 20,000 resamples; the 10,000-resample historical diagnostic is a separate output.

See [the output index](../OUTPUT_INDEX.md) for figure and table source mappings,
including the final diagram source and point-comparison plotting commands.

| Analysis | Authoritative archive |
| --- | --- |
| Primary full model and matched components | `r2_experiment_plan_20261018/joint_routing_trial_20261022/trial_predictions_LOCAL_ONLY.csv` |
| Primary fitting selections and residual models | `r2_experiment_plan_20261018/joint_routing_trial_20261022/folds/` |
| Primary classifiers and historical calibration | `03_results/r47_timestamp_lags_v1.0/folds/` |
| Probability comparisons and Fig. C.1 | `r2_experiment_plan_20261018/four_comment_experiment_closeout_20261023/consistent_probability_predictions_LOCAL_ONLY.csv` |
| Five-window stress and March subset | `03_results/r15_stress_route_seed42_v1.0/predictions.csv` |
| Baseline predictions and market-state labels | `03_results/r13_unified_fixed_window_comparison_v1.0/` |
| Repeated baseline fits | `03_results/r17_reviewer1_seed_7/` and `r17_reviewer1_seed_2026/` |
| Context-length sensitivity | `03_results/r38_context_length_sensitivity_yq_v1.0/` |

Paths beginning `03_results/` are relative to the original project directory.
The historical aligned comparison file supplies baseline columns; the current
primary `Full` column is supplied by the joint-routing file above. The verifier
uses the latter for all current primary component and probability results.

## Execution

Run the commands in the repository README. `verify` regenerates numerical
outputs and the reliability figure from saved predictions. `replay` also
reconstructs primary pathways from the saved estimators. Both require the full
archive. No original author directory or drive mapping is required for these
entry points.

Replay writes regenerated predictions and its path-transformation audit under
`--output`, which must be separate from the frozen archive. It retains source,
protocol and fitted-artifact integrity checks. The historical guard that read
the author's local Word documents is omitted because those documents are not
forecasting inputs. The audited replay covers 13 configurations and all 20
folds with zero numerical difference from the saved reference.

`train-downstream` starts a fresh residual-selection run in the requested
output directory. Historical classifier outputs and Chronos predictions are
retained as inputs. Its settings and three chronological out-of-fold blocks
are defined in `joint_routing_trial.py`.

For supply and congestion classifier rebuilding, see
[CLASSIFIER_REBUILD.md](CLASSIFIER_REBUILD.md). Its default preflight checks
the 20-fold, three-holdout construction without fitting. An explicit fitting
option is provided; it rebuilds the classifiers only. The complete forecasting
pathway replay uses the released fitted artifacts.

Foundation-model adaptation is implemented in
`02_experiments/code/run_r10_chronos2_covariate_lora.py`; the corresponding
fold-cache builder is `build_r10_chronos2_lora_fold_cache.py`. Inspect their
`--help` for dataset, fold-registry, model and output arguments. Install the
full environment and obtain the third-party base model under its license.
The baseline sources are under the repository's `code/baselines/` directory.
Archived experiment scripts can contain original paths in provenance records;
use the supplied inputs and specify local paths when rerunning those scripts.

## Interpretation and integrity

Primary and stress evaluations have distinct timestamp registries and fitting
artifacts. Their action counts and metric denominators remain separate in the
released tables. The controller's calibration statistic is an empirical
screening criterion. Held-out action frequencies and day-block uncertainty are
reported directly in the audit outputs.

The archive's `SHA256SUMS.txt` covers extracted files; the repository's checksum
file covers repository files. Model serialization files should be loaded only
from this trusted release after checking their integrity.
