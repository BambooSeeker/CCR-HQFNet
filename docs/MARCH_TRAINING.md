# March pathway training and verification

This demonstration refits the residual models and state classifiers, selects
the daily tail-acceptance policies, and generates all six pathway forecasts
for 1-14 March 2026. The Chronos-2 outputs are held fixed at their archived
values. This isolates the trainable routing and acceptance mechanisms examined
in Table 9 and provides a reproducible comparison on identical target intervals.

## Run

Use Python 3.12 in a separate environment. A GPU is not required for this entry point.

```bash
python -m pip install -r environment/requirements-routing.txt
python run_march_training.py
```

The program checks input hashes and split sizes before fitting. It then compares
all 4,032 generated pathway predictions against the reference, using an absolute
tolerance of 1e-8 CNY/MWh. Outputs include predictions, daily policy selections,
metrics, software versions and a verification report. The complete-model MAE is
34.67387633016999 CNY/MWh.

## Data scope

| Partition | Explicit date selection | Days | Half-hour intervals |
| --- | --- | ---: | ---: |
| Training | Listed dates between 28 August 2025 and 7 February 2026 | 90 | 4,320 |
| Validation | 8-14 February 2026 | 7 | 336 |
| Test | 1-14 March 2026 | 14 | 672 |

`fold.json` is the authoritative date list. The training interval bounds contain
gaps; only the listed training days are released. No earlier February test
window is included. The 5,328 rows contain the feature fields required by the
archived pathway implementation, the price target and corresponding state labels.
Lagged predictors retain their historical information. Thus the dates of source
values inside lagged columns can precede the dates of the rows that contain them.

| File | Contents |
| --- | --- |
| `inputs.csv` | Timestamp, delivery day, price target and required predictors |
| `labels.csv` | Supervised congestion states; -1 denotes unavailable labels |
| `carrier_cache.csv` | Fixed fold-specific Chronos-2 median and internal quantiles |
| `fold.json` | Exact training, validation and test dates |
| `features.json` | Predictor lists and explicit congestion descriptors |
| `route_config.json` | Random seed and tree-model settings |
| `reference_pathways.csv` | Reference forecasts for all six March pathways |
| `checksums.json` | SHA-256 checksums of the demonstration inputs |

Feature definitions are provided in `config/feature_dictionary.csv`. The
historical `timesfm_q10`, `timesfm_q50` and `timesfm_q90` cache field names are
retained for compatibility with the archived routing code; in this demonstration
they contain Chronos-2 outputs. `route_config.json` specifies the downstream
models. The fixed carrier was adapted with a 192-interval context; the separate
legacy 96-interval configuration is not used by this entry point.

## Interpretation

This command trains the routing components from the released training records.
It does not repeat Chronos-2 pretraining or LoRA adaptation. The supplied carrier
cache is an explicit input, with the representative adapter retained separately
in `checkpoint/`. Independent carrier adaptation can introduce numerical
differences and requires its own prediction-cache validation.

The released records are limited to this demonstration and its fitting inputs.
Other industrial source records remain restricted under the research agreement.
See `DATA_AVAILABILITY.md` for access and permitted-use conditions. Repartitioning
public records is technically possible; the supplied split defines the reported
scientific comparison.

## Package conventions

The documentation follows the single-entry-point and explicit-data-scope
principles of [Horiuchi's replication-package guide](https://github.com/yhoriuchi/replication-package-guide).
