# Time Series Forecasting Framework

Framework for benchmarking classical statistical models, deep learning models, and zero-shot foundation models on multiple time-series datasets with a shared workflow for preprocessing, tuning, evaluation, visualization, and export.

## Documentation Policy

This repository should keep only two maintained markdown documents:

- `README.md`: project overview, methodology, repo structure, and current status
- `todos.md`: active work list and short project follow-up notes

Older one-off notes were consolidated here and into `todos.md`.

## Project Goal

The project compares three model families on datasets with different domains and frequencies:

- statistical models
- deep learning models
- foundation models

The main purpose is not just to generate forecasts, but to compare:

- predictive accuracy
- compute cost
- behavior across yearly, quarterly, monthly, daily, and hourly data

## Dataset Portfolio

| Notebook | Dataset | Domain | Frequency | Notes |
| :-- | :-- | :-- | :-- | :-- |
| `01` | World Bank USA Real GDP | macroeconomics | yearly | scaled to billions USD |
| `02` | FRED GPDIC1 Investments | macro / finance | quarterly | US investments series |
| `03` | ECB EUR/CZK | forex | monthly | exchange-rate forecasting |
| `04` | M5 Walmart Hobbies | retail | daily | 5-series multiseries setup |
| `05` | Kaggle BTC/USD | crypto | hourly | currently still smoke-test oriented |

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

## Workflow

Each dataset follows the same three-step notebook flow:

1. `preprocessing/`
2. `exploration_data_analyses/`
3. `forecasting/`

The forecasting notebooks use shared code from `src/`.

## Methodology

### Data Split

- each dataset uses a hold-out test set defined by `test_periods`
- tuning is done only on training data
- final reported forecasts are generated after retraining on the full train split

### Tuning Strategy

- single-series datasets use `run_tuning_and_eval(...)`
- multiseries daily dataset uses:
  - `run_tuning_local_and_eval(...)` for local statistical models
  - `run_tuning_global_and_eval(...)` for global DL models

### Metrics

- RMSE is the main selection metric
- MAPE is kept as an auxiliary relative-error metric
- tuning time is tracked for efficiency comparisons

### Statistical Comparison

The repo uses the Diebold-Mariano test with Harvey-Leybourne-Newbold correction logic in `src/evaluation.py`.

Current intended logic:

- single-series macro / monthly datasets: conservative `h=1`
- pooled multiseries datasets: full horizon can be used if sample size is sufficient

Note:

- notebook `05` still has a mismatch between the explanatory comment and the actual DM call; this is tracked in `todos.md`

## Current Repo Status

### Source Of Truth

The current source of truth is:

- `src/`
- `forecasting/*.ipynb`

Legacy artifacts:

- `raw_py_scripts/*.py`

Those scripts still compile, but they are not fully aligned with the refactored notebook workflow and should not be treated as canonical.

### Notebook Status

| Notebook | Status | Comment |
| :-- | :-- | :-- |
| `01_forecasting_wb_usa_real_gdp_yearly.ipynb` | usable | refactored, executed, recently touched |
| `02_forecasting_fred_gpdic1_investments_quarterly.ipynb` | usable | refactored, executed |
| `03_forecasting_ecb_eurczk_monthly.ipynb` | usable | refactored, executed |
| `04_forecasting_m5_walmart_daily.ipynb` | usable but expensive | refactored multiseries workflow, long runtimes |
| `05_forecasting_kaggle_btcusd_hourly.ipynb` | unfinished | still uses `smoke_test_points` |

### Current Known Gaps

- `05` BTC notebook is not yet a final full-data run
- `src/evaluation.py` currently does not include `GraniteTTM` in foundation-category winner selection
- `raw_py_scripts/` should be archived, deleted, or resynced

## Repository Structure

- `src/`: shared forecasting logic
- `src/data_loader.py`: loading, split, scaling, smoke-test handling
- `src/tuning.py`: tuning loops for single-series and multiseries setups
- `src/pipeline.py`: final retraining, prediction generation, foundation-model execution
- `src/evaluation.py`: Diebold-Mariano test and comparison logic
- `src/visualization.py`: plots and PNG export
- `src/wrappers/granite_ttm.py`: Granite TTM wrapper
- `datasets/`: prepared input CSVs
- `preprocessing/`: preprocessing notebooks
- `exploration_data_analyses/`: EDA notebooks
- `forecasting/`: benchmark notebooks
- `images/forecasting/`: exported plots and DM tables
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

Recommended order:

1. run preprocessing notebook for a dataset
2. run EDA notebook for the same dataset
3. run forecasting notebook for the same dataset

Examples:

- `forecasting/01_forecasting_wb_usa_real_gdp_yearly.ipynb`
- `forecasting/04_forecasting_m5_walmart_daily.ipynb`
- `forecasting/05_forecasting_kaggle_btcusd_hourly.ipynb`

## Outputs

Generated artifacts are exported to `images/forecasting/`:

- forecast comparison plots
- validation vs test comparison plots
- DM test result tables

## Language Policy

- code: English
- documentation: English
- user interaction with the agent: Czech

## Project Follow-Up

Active work is tracked only in `todos.md`.
