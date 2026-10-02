# Data availability and provenance

## Full study registry

The full market records used in the study were supplied directly by Zhejiang
Energy Group as Excel files under a research cooperation agreement. They cover
the price, schedule, load, generation, renewable, and congestion-related fields
needed to construct the rolling-origin experiments. The agreement does not
permit redistribution of the complete records.

Eligible organizations may enquire about access through the Zhejiang
Electricity Power Exchange Platform at <https://www.zjpx.com.cn/>. Registration,
eligibility, data scope, and authorization are determined by the platform and
the relevant data provider. This repository does not imply that unrestricted
public download or API access is available.

## Released demonstration data

`demo_data/` contains a permitted subset for 1-14 March 2026:

| File | Rows | Purpose |
| --- | ---: | --- |
| `march_2026_test_features.csv` | 672 | Documents the released half-hourly model-input and target schema |
| `march_2026_reference_predictions.csv` | 672 | Complete-model predictions used by the metric verification |

The files permit inspection of the schema and exact reproduction of the
released March test-period metrics.

The `demo_data/march_training/` directory additionally provides 4,320 training
rows from the exact 90-day list, 336 validation rows for 8-14 February 2026,
and 672 test rows for 1-14 March 2026. It includes the required predictors,
training labels, fixed carrier outputs, settings and all six pathway references.
Training-day bounds extend from 28 August 2025 to 7 February 2026; only the dates
listed in `fold.json` are included. Historical values within lagged predictors
remain part of these inputs. The complete industrial registry is not released.

`run_march_training.py` refits the downstream routing components with the fixed
carrier outputs and verifies their March forecasts. It does not repeat LoRA
adaptation. See [the demonstration specification](docs/MARCH_TRAINING.md).

The demonstration files are provided for non-commercial scholarly verification
of the associated manuscript. Redistribution or use outside that purpose may
require permission from the corresponding author and the original data
provider. The files must not be used to identify or infer confidential market
participants or operational records.

## Reproduction with authorized records

Researchers holding authorized records can use
`code/experiment_pipeline/prepare_authorized_dataset.py`,
`config/feature_dictionary.csv`, and the released fold registries to construct
the analysis inputs. The source-file hashes in `config/dataset_manifest.json`
identify the version used by the authors without redistributing the records.

Questions concerning scientific reproduction may be directed to Peng Hou at
`houpeng2026@tju.edu.cn`.
