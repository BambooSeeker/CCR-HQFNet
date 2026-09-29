# CCR-HQFNet

Reproducibility materials for **Real-Time Electricity Price Forecasting with
Time-Series Foundation-Model Adaptation and Supply-Price and Congestion Routing:
A Zhejiang Case Study**, submitted to *Applied Energy*.

CCR-HQFNet adapts probabilistic trajectories from Chronos-2 to the Zhejiang
real-time electricity market through separately estimated supply-price and
congestion pathways and validation-calibrated control of negative-tail
corrections.

## Release scope

This repository contains:

- the experiment, baseline, routing, ablation, inference, and evaluation code;
- frozen feature definitions, fold registries, and training/search settings;
- a representative Chronos-2 LoRA adapter;
- permitted half-hourly demonstration data for 1-14 March 2026 (672 targets);
- frozen complete-model predictions and reference metrics for that period.

The March release supports direct verification of the reported test-period
metrics and documents the input/output schema. It is not a substitute for the
confidential registry used for the four-season evaluation. Reproducing those
experiments requires authorized source records prepared with the released
schema and fold definitions.

## Repository layout

```text
checkpoint/                 Representative Chronos-2 LoRA adapter
code/baselines/             Statistical, machine-learning, and deep baselines
code/experiment_pipeline/   Data, routing, ablation, inference, and evaluation
config/                     Feature dictionary, folds, and frozen settings
demo_data/                  Permitted March 2026 data and reference predictions
environment/                Pinned Python environments and runtime information
outputs/                    Reference metrics verified by run_demo.py
run_demo.py                 Metric-only reproducibility check
```

## Quick verification

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r environment/requirements-demo.txt
python run_demo.py
```

The check reads the released predictions, recomputes all metrics, and compares
them with `outputs/reference_metrics.json`. Expected complete-model results are
MAE 34.674 CNY/MWh, RMSE 72.662 CNY/MWh, and MAPE 15.970% on the 653 observations
with an absolute realized price of at least 50 CNY/MWh. It also verifies the
negative-price count, recall, precision, and false-positive count. A successful
run ends with:

```text
PASS: March 2026 test-period metrics match the released reference.
```

This command verifies the released test-period result; it does not retrain the
foundation model.

## Full pipeline with authorized data

1. Obtain the authorized source records described in
   [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
2. Create the analysis dataset using
   `code/experiment_pipeline/prepare_authorized_dataset.py` and the schema in
   `config/feature_dictionary.csv`.
3. Use `config/four_season_fold_registry.json` for the reported rolling-origin
   splits and `config/training_and_search_settings.json` for fixed settings.
4. Install the full environment from `environment/requirements.txt` and run the
   relevant carrier, route, ablation, baseline, and evaluation scripts under
   `code/`.

The complete experiments require a CUDA-capable environment and access to the
Chronos-2 and TimesFM base models. Runtime details are recorded in
`environment/runtime.txt`.

## Data access

The full Excel records were supplied directly by Zhejiang Energy Group under a
research cooperation agreement and cannot be redistributed. The Zhejiang
Electricity Power Exchange Platform is available at
<https://www.zjpx.com.cn/>. Eligibility and access are determined by the
platform's registration and authorization procedures. See
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the precise boundary between
the public demonstration files and the restricted source registry.

## Integrity

`SHA256SUMS.txt` records SHA-256 checksums for the released research artifacts.
From PowerShell, they can be checked with:

```powershell
Get-Content SHA256SUMS.txt | ForEach-Object {
  $hash, $path = $_ -split '  ', 2
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower() -ne $hash) {
    throw "Checksum mismatch: $path"
  }
}
```

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The repository
will be updated with the final bibliographic record if the manuscript is
accepted.

## License and contact

The authors' source code is released under the MIT License. Separate conditions
apply to the demonstration data, third-party base models, and their associated
licenses; see [`LICENSE`](LICENSE), [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md),
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

For scientific and data-access enquiries, contact the corresponding author,
Peng Hou (`houpeng2026@tju.edu.cn`).
