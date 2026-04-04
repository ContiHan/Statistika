# Time Series Forecasting Framework

Framework for benchmarking statistical models, deep learning models, and zero-shot foundation models on multiple time-series datasets with a shared workflow for preprocessing, exploratory analysis, tuning, evaluation, visualization, and export.

## Documentation Policy

This repository currently keeps three maintained markdown documents:

- `README.md`: project overview, methodology, repo structure, and current status
- `todos.md`: active work list and short project follow-up notes
- `practical_part_outline.md`: working outline for the thesis practical chapter

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

| Notebook | Dataset | Domain | Frequency | Notes |
| :-- | :-- | :-- | :-- | :-- |
| `01` | World Bank USA Real GDP | macroeconomics | yearly | scaled to billions USD |
| `02` | FRED GPDIC1 Investments | macro / finance | quarterly | US investments series |
| `03` | ECB EUR/CZK | forex | monthly | exchange-rate forecasting |
| `04` | M5 Walmart Hobbies | retail | daily | 5-series multiseries setup |
| `05` | Kaggle BTC/USD | crypto | hourly | currently kept as a reduced / smoke-style run |

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

### Statistical Comparison

The repository uses dedicated Diebold-Mariano comparisons on backtest artifacts rather than selecting winners directly on the final test set.

Current logic:

- shortlist models are selected from validation performance
- pairwise DM comparisons run on dedicated rolling backtest artifacts
- Holm correction is available for multiple pairwise comparisons
- some pairs can be skipped if they do not share overlapping validation points

## Current Repo Status

### Source Of Truth

The current source of truth is:

- `src/`
- `forecasting/*.ipynb`

The preprocessing and EDA notebooks are supporting stages of the workflow, but the main experimental logic lives in `src/` and the forecasting notebooks.

### Notebook Status

| Notebook | Status | Comment |
| :-- | :-- | :-- |
| `01_forecasting_wb_usa_real_gdp_yearly.ipynb` | usable | compact yearly benchmark |
| `02_forecasting_fred_gpdic1_investments_quarterly.ipynb` | usable | quarterly macro benchmark |
| `03_forecasting_ecb_eurczk_monthly.ipynb` | usable | monthly FX benchmark |
| `04_forecasting_m5_walmart_daily.ipynb` | usable but expensive | multiseries workflow with covariates and longer runtimes |
| `05_forecasting_kaggle_btcusd_hourly.ipynb` | reduced-run setup | computationally heavy hourly benchmark |

### Current Known Limits

- `05` BTC is still intentionally run in a reduced / smoke-style setup because of compute cost
- some foundation-model DM pairs can be skipped when they are not comparable on overlapping validation points
- runtime comparisons should be interpreted within each dataset run, not across different hardware sessions

## Repository Structure

- `src/`: shared forecasting logic
- `src/data_loader.py`: loading, split handling, scaling, and dataset preparation
- `src/tuning.py`: tuning loops for single-series and multiseries setups
- `src/pipeline.py`: final retraining, prediction generation, and foundation-model execution
- `src/evaluation.py`: validation summaries, pairwise DM analysis, and statistical comparison logic
- `src/visualization.py`: plots, tables, and PNG export
- `src/notebook_setup.py`: shared notebook imports and setup helpers
- `datasets/`: prepared input CSVs
- `preprocessing/`: preprocessing notebooks
- `exploratory_data_analysis/`: EDA notebooks
- `forecasting/`: benchmark notebooks
- `images/forecasting/`: exported plots and tables
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

- forecast plots
- comparison plots
- Box-Cox diagnostic tables and charts
- DM backtest summary tables
- DM pairwise tables
- DM heatmaps

## Language Policy

- code: English
- documentation: English
- user interaction with the agent: Czech

## Project Follow-Up

Active work is tracked in:

- `todos.md`
- `practical_part_outline.md` for thesis-writing structure
