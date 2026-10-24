# Classifier reconstruction

Run `rebuild_classifiers.py` from the repository root with the extracted
research archive and a separate output directory. Python 3.12 and the routing
dependencies are required. The default operation checks inputs and fitting
boundaries without fitting a model:

```bash
python rebuild_classifiers.py --archive D:/CCR/research_archive --output outputs/classifier_preflight
```

Replace the archive path with its local location. Add
`--key lhfdc_fixed_spring_01` to restrict the check to one fold. Output paths
inside the archive, equal to it, or containing it are rejected.

The wrapper initializes the archived r36 source with the supplied project root,
replacing its recursive search for a traceability CSV. It uses the released
timestamp-lag restoration and locked feature lists, and extracts the archived
classifier fitter without running the original training drivers. Each source
transformation and source hash is recorded in `classifier_rebuild_audit.json`.
The original sources and archived artifacts are not edited.

For each fold, the plan contains supply-price and congestion classifiers for
the outer training/validation split and three downstream chronological
holdouts. The inner blocks use 38, 53 or 68 fitting days, seven early-stopping
days and 15 held-out days. Carrier-derived fields are excluded from classifier
inputs. The low-price label threshold is estimated from the outer 90-day
training block and shared across the inner blocks. Carrier forecasts and
upstream scaling also remain conditional on the archived outer-fold fit;
this operation does not establish full upstream cross-fitting.

## Optional classifier fitting

The following command explicitly requests fitting. It is unnecessary for
saved-model replay and was not executed during the portability verification:

```bash
python rebuild_classifiers.py --archive D:/CCR/research_archive --output outputs/classifier_refit --fit-classifiers
```

The same `--key` option can restrict fitting to one fold. New classifier pairs
are written to `folds/<key>/outer/classifiers.joblib` and
`folds/<key>/oof_<0-2>/classifiers.joblib` beneath the output directory. These
are classifier-only bundles, not replacements for the archive's canonical
model bundles. Residual forecasters, the low-price expert, probability
mappings, controller policies and foundation-model adaptation are outside
this wrapper's rebuilding scope. New-fit numerical equivalence has not been
tested.

## Verified scope

The no-training preflight passed all 20 folds and 160 classifier plans. The
saved-model revision replay separately passed all 13 pathways on 6,048 targets
each, with zero numerical difference across 78,624 rows. Replay uses saved
residual estimators and archived carrier, router and calibration outputs; it
does not refit the forecasting models. The test harness blocked model fitting,
author-local document access, network access and writes outside its test
output directory. All 1,795 existing archive files retained their hashes and
modification times after both checks.
