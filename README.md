# Time Series Forecasting Framework

Framework for benchmarking statistical models, deep learning models, and zero-shot foundation models on multiple time-series datasets with a shared workflow for preprocessing, exploratory analysis, tuning, evaluation, visualization, and export.

## Documentation Policy

This repository currently keeps two maintained markdown documents:

- `README.md`: project overview, methodology, repo structure, and current status
- `todos.md`: active work list and short project follow-up notes

Older scattered notes were consolidated into these files.

## Project Goal

The project compares three model families on datasets with different domains and frequencies:

- statistical models
- deep learning models
- foundation models

The benchmark focuses on:

- predictive accuracy
- compute cost
- behavior across yearly, quarterly, monthly, daily, and hourly data
- statistical significance of selected pairwise performance differences

## Dataset Portfolio

| Notebook | Dataset                 | Domain          | Frequency | Notes                                                        |
| :------- | :---------------------- | :-------------- | :-------- | :----------------------------------------------------------- |
| `01`     | World Bank USA Real GDP | macroeconomics  | yearly    | scaled to billions USD                                       |
| `02`     | FRED GPDIC1 Investments | macro / finance | quarterly | US investments series                                        |
| `03`     | ECB EUR/CZK             | forex           | monthly   | exchange-rate forecasting                                    |
| `04`     | M5 Walmart Hobbies      | retail          | daily     | 5-series multiseries setup                                   |
| `05`     | Kaggle BTC/USD          | crypto          | hourly    | final run kept on the most recent `50k` points in smoke mode |

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

## Notebook Workflow

Each dataset follows the same three-stage notebook flow:

1. `preprocessing/`
2. `exploratory_data_analysis/`
3. `forecasting/`

The forecasting notebooks use shared code from `src/`.

## Methodology

### Data Split

- each dataset uses a hold-out test set defined by `test_periods`
- tuning and validation are performed only on training data
- final reported forecasts are generated after retraining on the full train split

### Target Transform Strategy

The project does not apply one global preprocessing rule to all model families.

- deep learning models use scaled targets
- foundation models use raw targets
- statistical models use a dataset-level `raw` vs `log` decision

The statistical transform decision is based on Box-Cox diagnostics computed on the train split:

- `log` is used only when the diagnostics support a log-like transform
- otherwise the raw scale is kept
- AutoARIMA handles differencing internally; there is no global manual differencing step for all statistical models

### Exploratory Analysis vs Final Pipeline

The repository also contains exploratory notebooks for data understanding and diagnostics.

- preprocessing notebooks prepare the cleaned benchmark datasets
- EDA notebooks inspect trend, seasonality, volatility, and stationarity-related properties
- ADF tests belong to the exploratory diagnostic phase, not to the final global preprocessing rule used in forecasting

### Tuning and Validation

- single-series datasets use shared single-series tuning logic
- the multiseries M5 dataset separates local statistical tuning from global DL tuning
- validation metrics are based on rolling backtesting
- RMSE is the main model-selection metric
- MAPE is tracked as an auxiliary relative-error metric
- the experiment tracker stores the selected winning configuration for each model variant, including the final parameter set used after tuning

### Statistical Comparison

The repository uses dedicated Diebold-Mariano comparisons on backtest artifacts rather than selecting winners directly on the final test set.

Current logic:

- shortlist models are selected from validation performance after tuning has already identified one winning configuration per model variant
- pairwise DM comparisons run on dedicated rolling backtest artifacts
- Holm correction is available for multiple pairwise comparisons
- some pairs can be skipped if they do not share overlapping validation points

Important distinction:

- the `Selected Model Configurations` table reports the validation RMSE/MAPE stored by the experiment tracker during tuning
- the `DM Backtest Summary` table reports RMSE/MAPE recomputed on a separate dedicated backtest used only for statistical comparison
- these values are therefore not expected to match numerically, and the model ranking can change between the two tables
- this is especially visible when the DM backtest uses a different protocol than tuning, for example a dense one-step setup (`forecast_horizon=1`, `stride=1`) versus a sparser tuning setup

Interpretation:

- use the selected-configuration table as evidence of which parameter setting won during tuning
- use the DM backtest summary as evidence of how the shortlisted winning models behaved under the dedicated inferential backtest

## Current Repo Status

### Source Of Truth

The current source of truth is:

- `src/`
- `forecasting/*.ipynb`

The preprocessing and EDA notebooks are supporting stages of the workflow, but the main experimental logic lives in `src/` and the forecasting notebooks.

### Notebook Status

| Notebook                                                 | Status                    | Comment                                                                                                                          |
| :------------------------------------------------------- | :------------------------ | :------------------------------------------------------------------------------------------------------------------------------- |
| `01_forecasting_wb_usa_real_gdp_yearly.ipynb`            | usable                    | compact yearly benchmark                                                                                                         |
| `02_forecasting_fred_gpdic1_investments_quarterly.ipynb` | usable                    | quarterly macro benchmark                                                                                                        |
| `03_forecasting_ecb_eurczk_monthly.ipynb`                | usable                    | monthly FX benchmark                                                                                                             |
| `04_forecasting_m5_walmart_daily.ipynb`                  | usable but expensive      | multiseries workflow with covariates and longer runtimes                                                                         |
| `05_forecasting_kaggle_btcusd_hourly.ipynb`              | usable with reduced scope | final benchmark kept on the most recent `50k` points because full AutoARIMA rolling validation was not computationally practical |

### Current Known Limits

- some foundation-model DM pairs can be skipped when they are not comparable on overlapping validation points
- runtime comparisons should be interpreted within each dataset run, not across different hardware sessions
- `05` BTC is intentionally reported as a reduced / smoke-style run on the most recent `50k` points because the full AutoARIMA setup was not computationally feasible in a reasonable runtime
- the final reduced BTC protocol uses a denser DM setup than tuning:
  - tuning on the reduced `50k` segment uses `cv_start_ratio = 0.95`
  - dedicated DM backtest on the same reduced segment uses `cv_start_ratio = 0.90`
  - this was done to keep tuning computationally feasible while still obtaining more usable DM backtest points

### TimeGPT API Budgeting

`TimeGPT` cost is driven by the number of `client.forecast(...)` calls, not by local runtime alone. The practical bottleneck is mainly the multiseries `04` DM backtest, not the `05` BTC notebook.

The table below summarizes the approximate request budget for one notebook run under the current settings, assuming that:

- `TimeGPT` is enabled
- final forecast is generated
- `TimeGPT` reaches the DM shortlist, so the dedicated DM backtest is also executed

| Dataset            |                    Train len | Tuning requests | Final forecast | DM requests | Total with DM |
| :----------------- | ---------------------------: | --------------: | -------------: | ----------: | ------------: |
| `01 GDP`           |                           59 |              13 |              1 |          18 |            32 |
| `02 Investments`   |                          307 |              22 |              1 |          93 |           116 |
| `03 EUR/CZK`       |                          361 |               8 |              1 |         109 |           118 |
| `04 M5 full`       | 1885 per series (`5` series) |             385 |              1 |        2830 |          3216 |
| `05 BTC full`      |                       122318 |              36 |              1 |          36 |            73 |
| `05 BTC smoke 50k` |                        49832 |              14 |              1 |          14 |            29 |

Interpretation:

- one clean full run of `04` + one reduced `05 smoke 50k` run consumes about `3245` requests
- one clean run of all forecasting notebooks with the current reduced `05` setup consumes about `3511` requests
- the main API-risk notebook is `04`, so avoid unnecessary reruns once the setup is frozen
- `05` is relatively cheap from the `TimeGPT` API perspective; its main bottleneck is local compute time of the non-API models, especially AutoARIMA

Current conservative runtime protection:

- the repository currently uses a defensive `TimeGPT` throttle of `40 requests -> 90s sleep`
- this is intentionally slower than the provider limit to reduce reruns caused by burst failures, `429`, and transient `500` responses
- approximate added waiting time from throttling alone:
  - `04` M5 DM with dense `stride=1`: about `105` extra minutes
  - `04` full notebook TimeGPT path: about `120` extra minutes
  - `05` reduced `50k` notebook TimeGPT path: well under `1` extra minute
- these are upper-bound style estimates assuming sustained request flow; actual added time can be lower if the run naturally pauses between phases

## Repository Structure

- `src/`: shared forecasting logic
- `src/data_loader.py`: loading, split handling, scaling, and dataset preparation
- `src/tuning.py`: tuning loops for single-series and multiseries setups
- `src/pipeline.py`: final retraining, prediction generation, and foundation-model execution
- `src/evaluation.py`: validation summaries, pairwise DM analysis, and statistical comparison logic
- `src/visualization.py`: plots, tables, and PNG export
- `src/export_data.py`: machine-readable export bundle (`CSV` / `JSON`) for post-processing
- `src/notebook_setup.py`: shared notebook imports and setup helpers
- `datasets/`: prepared input CSVs
- `preprocessing/`: preprocessing notebooks
- `exploratory_data_analysis/`: EDA notebooks
- `forecasting/`: benchmark notebooks
- `images/forecasting/`: exported plots and tables
- `artifacts/forecasting/`: machine-readable per-dataset export bundles
- `config/`: local API-key config template

## Environment Setup

### Prerequisites

- Python `3.10+`
- virtual environment recommended

### Install

```bash
python3 -m venv .mac-venv
source .mac-venv/bin/activate
pip install -r requirements.txt
```

### API Keys

`TimeGPT` requires a Nixtla API key:

```bash
cp config/api_keys.template.py config/api_keys.py
```

Then set:

```python
NIXTLA_API_KEY = "your_api_key_here"
```

`config/api_keys.py` is gitignored.

## Running The Project

Recommended order for each dataset:

1. run the preprocessing notebook
2. run the EDA notebook
3. run the forecasting notebook

Examples:

- `preprocessing/01_preprocessing_wb_usa_real_gdp_yearly.ipynb`
- `exploratory_data_analysis/01_eda_wb_usa_real_gdp_yearly.ipynb`
- `forecasting/01_forecasting_wb_usa_real_gdp_yearly.ipynb`

## Outputs

Generated artifacts are exported to `images/forecasting/`, including:

- train/test split plots
- forecast plots
- comparison plots
- selected model configuration tables
- Box-Cox diagnostic tables and charts
- DM backtest summary tables
- DM pairwise tables
- DM heatmaps

Machine-readable exports can also be written to `artifacts/forecasting/<dataset_slug>/`:

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

The shared helper `export_forecasting_data(...)` is available from `src/notebook_setup.py`.

Default behavior:

- exports the full bundle, including the optional audit-friendly CSVs
- keeps PNG outputs in `images/forecasting/`
- keeps machine-readable outputs in `artifacts/forecasting/`

If a notebook run should save only the core bundle, the optional exports can be disabled with:

```python
export_forecasting_data(
    ...,
    include_optional=False,
)
```

If needed, the optional sections can also be toggled individually:

- `export_tracker_results`
- `export_dm_shortlist`
- `export_params_long`
- `export_validation_points`
- `export_dm_points`

Examples:

```python
# default: save the full bundle
export_forecasting_data(...)
```

```python
# core bundle only
export_forecasting_data(
    ...,
    include_optional=False,
)
```

```python
# custom bundle: keep only selected optional exports
export_forecasting_data(
    ...,
    include_optional=False,
    export_tracker_results=True,
    export_validation_points=True,
    export_dm_points=True,
)
```

## Language Policy

- code: English
- documentation: English
- user interaction with the agent: Czech

## Project Follow-Up

Active work is tracked in:

- `todos.md`
