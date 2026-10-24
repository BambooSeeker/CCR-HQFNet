# CCR-HQFNet

Research materials for **Supply-price and congestion routing for time-series
foundation models in real-time electricity price forecasting: Evidence from
Zhejiang**, submitted to *Applied Energy*.

CCR-HQFNet combines adapted Chronos-2 forecasts, supply-price and congestion
pathways, and validation-calibrated selective control of negative-price
corrections.

## Second-revision research archive

The [v2.0.0 release](https://github.com/BambooSeeker/CCR-HQFNet/releases/tag/v2.0.0)
provides research inputs, chronological split definitions, fitted model
artifacts, source code, configurations and prediction outputs for the revised
analyses. Download `CCR-HQFNet-v2.0.0-research.zip` from the release and extract
it to a short local path. The extracted directory is named `research_archive`.
The release checksums identify the exact files supplied.

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
