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
released March test-period metrics. They do not reproduce the confidential
training and validation records or the complete four-season experiment.

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
