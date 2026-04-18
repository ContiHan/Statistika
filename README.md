# Time Series Forecasting Framework

Shared benchmark framework for comparing statistical models, deep learning models, and foundation models on economic and financial time series.

The repository contains:

- dataset-specific preprocessing notebooks
- exploratory data analysis notebooks
- forecasting notebooks with a shared workflow
- reusable source modules in `src/`
- exported figures and machine-readable artifacts used in the thesis

## What The Project Does

The benchmark compares three model families:

- statistical models
- deep learning models
- foundation models

It evaluates them across datasets with different domains and frequencies:

- yearly
- quarterly
- monthly
- daily multiseries
- hourly high-frequency

The framework tracks:

- validation RMSE and MAPE
- final test-set RMSE and MAPE
- tuning time and best-config training time
- dedicated Diebold-Mariano backtest summaries
- pairwise DM significance results

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .mac-venv
source .mac-venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

`TimeGPT` requires a Nixtla API key:

```bash
cp config/api_keys.template.py config/api_keys.py
```

Then edit:

```python
NIXTLA_API_KEY = "your_api_key_here"
```

`config/api_keys.py` is gitignored.

### 3. Run one dataset

Recommended order for each dataset:

1. run the preprocessing notebook
2. run the EDA notebook
3. run the forecasting notebook

Example for dataset `01`:

- `preprocessing/01_preprocessing_wb_usa_real_gdp_yearly.ipynb`
- `exploratory_data_analysis/01_eda_wb_usa_real_gdp_yearly.ipynb`
- `forecasting/01_forecasting_wb_usa_real_gdp_yearly.ipynb`

## Platform note

The project was developed and validated primarily on macOS. Linux and Windows were not systematically tested during the final thesis runs, so small environment-specific adjustments can still be needed.

Potential differences can include:

- virtual-environment activation and shell commands
- file-path conventions
- GPU backends (`MPS` on macOS vs. `CUDA` on Linux/Windows)
- SSL / trust-store behavior when downloading checkpoints or calling external services

## Repository Structure

- `src/`: shared forecasting logic
- `src/data_loader.py`: loading, split handling, scaling, and dataset preparation
- `src/tuning.py`: tuning loops for single-series and multiseries setups
- `src/pipeline.py`: final retraining, prediction generation, and foundation-model execution
- `src/evaluation.py`: DM shortlist selection, dedicated backtests, and pairwise statistical comparison
- `src/visualization.py`: plots, tables, and PNG export
- `src/export_data.py`: machine-readable export bundle (`CSV` / `JSON`)
- `src/notebook_setup.py`: shared notebook imports and setup helpers
- `datasets/`: prepared benchmark CSVs
- `preprocessing/`: preprocessing notebooks
- `exploratory_data_analysis/`: EDA notebooks
- `forecasting/`: benchmark notebooks
- `images/forecasting/`: exported figures
- `artifacts/forecasting/`: machine-readable per-dataset output bundles
- `artifacts/thesis_tables/`: thesis-oriented derived CSV tables
- `config/`: local API-key config template

## Dataset Portfolio

| ID | Dataset | Domain | Frequency | Notes |
| :-- | :-- | :-- | :-- | :-- |
| `01` | World Bank USA Real GDP | macroeconomics | yearly | compact low-frequency benchmark |
| `02` | FRED GPDIC1 Investments | macro / finance | quarterly | quarterly US investments series |
| `03` | ECB EUR/CZK | forex | monthly | exchange-rate forecasting |
| `04` | M5 Walmart Hobbies | retail | daily | 5-series multiseries setup with covariates |
| `05` | Kaggle BTC/USD | crypto | hourly | final benchmark kept on the most recent `50k` points |

## Model Families

### Statistical

- Holt-Winters
- AutoARIMA
- Prophet

### Deep Learning

- TiDE
- N-BEATS
- TFT

### Foundation

- Chronos2
- GraniteTTM
- TimeGPT

## Shared Benchmark Logic

### Data split

- each dataset uses a hold-out test set defined by `test_periods`
- tuning and validation are performed only on training data
- final reported forecasts are generated after retraining on the full train split

### Target preprocessing

The repository does not apply one global preprocessing rule to all model families.

- deep learning models use scaled targets
- foundation models use raw targets
- statistical models use a dataset-level `raw` vs `log` decision

For deep learning models, the scaling is performed with Darts `Scaler()`:

- default `MinMaxScaler(feature_range=(0, 1))`
- fitted on the training split
- in multiseries mode, scaling is fitted separately per series

For statistical models, the transform decision is based on Box-Cox diagnostics computed on the training split:

- `log` is used when the project decision rule supports a log-like transform
- otherwise the raw scale is kept
- AutoARIMA handles differencing internally; there is no single manual differencing rule applied to all statistical models

All transformed or scaled outputs are inverted back to the original target scale before final comparison.

### Validation and tuning

- validation uses rolling backtesting
- RMSE is the main model-selection metric
- MAPE is tracked as an auxiliary metric
- tuning identifies one winning configuration per model variant

### Dedicated DM layer

The repository uses a separate inferential layer for statistical comparison:

- shortlist models are selected from validation performance
- shortlisted models are re-evaluated on a dedicated rolling backtest
- pairwise DM comparisons are computed from those dedicated artifacts
- Holm correction is available for multiple pairwise comparisons

Important distinction:

- `Selected Model Configurations` summarize the winning settings from tuning
- `DM Backtest Summary` summarizes a separate dedicated backtest used only for inferential comparison
- these values are not expected to match numerically
- model ranking can change between validation, final forecast, and dedicated DM backtest

Also note:

- some DM pairs can be skipped if the compared models do not share enough overlapping backtest points
- therefore `DM Backtest Summary` and `dm_pairwise` do not always contain the same effective set of comparable models

## Final Experimental Status

### Source of truth

The main experimental logic lives in:

- `src/`
- `forecasting/*.ipynb`

### Notebook status

| Notebook | Status | Comment |
| :-- | :-- | :-- |
| `01_forecasting_wb_usa_real_gdp_yearly.ipynb` | usable | yearly benchmark |
| `02_forecasting_fred_gpdic1_investments_quarterly.ipynb` | usable | quarterly macro benchmark |
| `03_forecasting_ecb_eurczk_monthly.ipynb` | usable | monthly FX benchmark |
| `04_forecasting_m5_walmart_daily.ipynb` | usable but expensive | multiseries workflow with covariates and heavy TimeGPT backtests |
| `05_forecasting_kaggle_btcusd_hourly.ipynb` | usable with reduced scope | final benchmark kept on the most recent `50k` points because full AutoARIMA rolling validation was not computationally practical |

### Current known limits

- runtime comparisons should be interpreted mainly within each dataset run, not as hardware-independent global timing benchmarks
- some foundation-model DM pairs can be skipped when overlapping backtest points are not available
- `05` BTC is intentionally reported as a reduced run on the most recent `50k` points
- the final reduced BTC protocol uses:
  - tuning `cv_start_ratio = 0.95`
  - dedicated DM `cv_start_ratio = 0.90`

## TimeGPT Runtime Protection

`TimeGPT` cost is driven by the number of API requests, not by local runtime alone.

The repository currently uses a conservative defensive throttle:

- `40 requests -> 90s sleep`

This is intentionally slower than the provider limit to reduce reruns caused by:

- `429`
- transient `500`
- transient `502`
- transient `503`
- transient `504`
- malformed transient API responses

In practice:

- the main TimeGPT bottleneck is the multiseries `04` notebook
- `05` BTC is relatively cheap from the API perspective
- `05` is instead limited mainly by local compute cost of non-API models, especially AutoARIMA

## Outputs

### Figure exports

PNG outputs are exported to `images/forecasting/`, including:

- train/test split plots
- comparison plots
- forecast plots
- DM heatmaps
- selected-configuration tables
- Box-Cox tables and charts

### Machine-readable exports

Per-dataset bundles are written to `artifacts/forecasting/<dataset_slug>/`:

- `tables/`
  - `comparison_metrics.csv`
  - `selected_params.csv`
  - `transform_diagnostics.csv`
  - `dm_backtest_summary.csv`
  - `dm_pairwise.csv`
  - optional: `tracker_results.csv`, `dm_shortlist.csv`, `params_long.csv`
- `series/`
  - `reference_series.csv`
  - `final_forecasts.csv`
  - optional: `validation_points.csv`, `dm_backtest_points.csv`
- `run_metadata.json`

Thesis-oriented derived tables are stored in:

- `artifacts/thesis_tables/`

### Shared export helper

The helper `export_forecasting_data(...)` is available from `src/notebook_setup.py`.

Default behavior:

- keeps PNG outputs in `images/forecasting/`
- writes machine-readable outputs to `artifacts/forecasting/`
- exports the full bundle including optional audit-friendly CSVs

Core-only export:

```python
export_forecasting_data(
    ...,
    include_optional=False,
)
```

Custom optional export:

```python
export_forecasting_data(
    ...,
    include_optional=False,
    export_tracker_results=True,
    export_validation_points=True,
    export_dm_points=True,
)
```

## Running Notes

- start each notebook from a clean kernel when changing shared code
- `04` can run for a long time because of the multiseries TimeGPT path
- `05` final results are expected to come from the reduced `50k` setup, not from older full-history attempts
- if `final_predictions` are exported from notebooks, pass the actual notebook variable used in that notebook (`final_predictions` or `final_preds`)

## Documentation Policy

This repository keeps two maintained markdown documents:

- `README.md`: project overview, setup, methodology, and current status
- `todos.md`: active follow-up list

Older scattered notes were consolidated into these files.

