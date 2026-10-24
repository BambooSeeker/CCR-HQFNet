# CCR-HQFNet

Research materials for **Supply-price and congestion routing for time-series
foundation models in real-time electricity price forecasting: Evidence from
Zhejiang**, submitted to *Applied Energy*.

CCR-HQFNet combines adapted Chronos-2 forecasts, supply-price and congestion
pathways, and validation-calibrated selective control of negative-price
corrections.

## Second-revision research archive

The [v2.0.1 release](https://github.com/BambooSeeker/CCR-HQFNet/releases/tag/v2.0.1)
provides the final numerical indices and portable reproduction entry points.
It uses the complete, checksum-verified v2.0.0 research archive, containing
research inputs, chronological split definitions, fitted model
artifacts, source code, configurations and prediction outputs for the revised
analyses. Run `python download_research_archive.py` from this repository to
download, verify and assemble `CCR-HQFNet-v2.0.0-research.zip`. The release
distributes the archive in numbered parts; the script checks each part and
the assembled ZIP against the release manifest. Extract the ZIP to a short
local path. The extracted directory is named `research_archive`.
The release checksums identify the exact files supplied.
The data archive is unchanged by the v2.0.1 code and documentation update;
the downloader selects its original release automatically.

The archive includes the 6,048-observation primary evaluation, matched component
comparisons, the five-window stress evaluation, probability forecasts,
controller diagnostics, repeated baseline fits and context-length sensitivity
outputs. Its numerical registries are documented in
[the reproduction guide](docs/REVISION_REPRODUCTION.md). The earlier March
example remains available as a small entry point.

## Verify the revised results

Python 3.12 is recommended. From a checkout of this repository:

```bash
python -m pip install -r environment/requirements-routing.txt
python -m pip install matplotlib python-docx joblib
python reproduce_revision.py --archive /path/to/research_archive --mode verify
```

This recomputes primary component errors, probability metrics, quantile
reliability and action-risk intervals from the released predictions. The full
primary model has MAE 59.55745867 CNY/MWh on 6,048 targets. The command checks
the equality of point predictions and forecast medians, and the relationship
between MAE, interval score and WIS.

```bash
python reproduce_revision.py --archive /path/to/research_archive --mode replay
```

Replay loads the fitted downstream models and reconstructs all primary
pathways using the archived Chronos outputs. It compares the reconstructed
predictions with the released reference before repeating the metric checks.

```bash
python reproduce_revision.py --archive /path/to/research_archive --mode train-downstream --output outputs/refit
```

This refits and selects downstream residual models using the released
chronological fitting inputs. `--limit 1` restricts the run to the first fold.
Foundation-model adaptation and the other experiment entry points are
documented in the reproduction guide. Stochastic refitting can vary with
software and hardware; saved predictions and fitted artifacts provide the
numerical reference for the manuscript.

## Final figures, tables and numerical indices

The [output index](OUTPUT_INDEX.md) connects manuscript outputs to their
data, archived sources and regeneration commands. Current point and subgroup
indices are in `outputs/final_revision_audit/point_metric_traceability.csv`
and `tail_metric_traceability.csv`; the primary P01 row has MAE 59.55745867.
These final indices supersede the historical r21 traceability files retained
inside the immutable research archive. `paired_statistics.csv` records the
20,000-resample paired comparisons in Table A.1.

```bash
python regenerate_routing_diagram.py
python regenerate_primary_figures.py --archive /path/to/research_archive
python export_submission_assets.py --archive /path/to/research_archive
```

The first command renders the final routing diagram as PNG, PDF and editable
SVG. The second reconstructs the primary point-comparison figures from saved
forecasts. The final table-cell exports are in `outputs/final_manuscript_tables/`.
The asset-export command copies the deposited reference figures and table
transcriptions. It preserves their original appearance; numerical recalculation
and fresh plotting have the separate commands described in the output index.

## March example

```bash
python run_demo.py
python run_march_training.py
```

The March example contains 672 targets for 1-14 March 2026. Its full-model MAE
is 34.67387633 CNY/MWh. `run_march_training.py` refits six pathways using the
released 90-day training and seven-day validation records, conditional on
archived Chronos outputs. See [March training](docs/MARCH_TRAINING.md).

## Files and citation

- `reproduce_revision.py`: verification, fitted-model replay and downstream fitting.
- `config/revision_release_manifest.json`: manuscript identity and artifact scope.
- `config/four_season_fold_registry.json`: primary chronological folds.
- `outputs/final_revision_audit/`: machine-readable revision checks.
- `code/`: baseline and forecasting implementation sources.
- `environment/`: dependency specifications.
- `SHA256SUMS.txt`: checksums of repository files as distributed by Git.

[Data availability](DATA_AVAILABILITY.md) defines the research release.
[CITATION.cff](CITATION.cff) provides citation metadata.
Author-written code is covered by the [MIT License](LICENSE); base-model and
third-party conditions are described in [third-party notices](THIRD_PARTY_NOTICES.md).
Scientific enquiries: Peng Hou, `houpeng2026@tju.edu.cn`.
