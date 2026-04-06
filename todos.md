# TODO

Only active and recent work should live here.

## Current Priorities

- [ ] Finalize `forecasting/05_forecasting_kaggle_btcusd_hourly.ipynb`
  - [ ] keep `test_periods = 168` (`week-ahead`) and use the same horizon consistently
  - [ ] decide the final BTC subset size (`smoke_test_points`) for the thesis run
  - [ ] rerun the notebook with the chosen BTC setup and refresh exported images
  - [ ] record the exact BTC runtime setup used in the thesis text

- [ ] Freeze final experimental settings for all datasets
  - [ ] confirm `test_periods`, `seasonal_period`, `cv_start_ratio`, and optional `smoke_test_points`
  - [ ] confirm the final dataset-level statistical transform (`raw` vs `log`)
  - [ ] keep one final result set per dataset and stop changing configs afterwards

- [ ] Regenerate final notebook outputs and exported figures
  - [ ] rerun `01` to `04` from a clean kernel if any setup or plotting code changed
  - [ ] verify `images/forecasting/` contains only the final PNGs you want to keep
  - [ ] remove old duplicate M5 exports after the final rerun
  - [ ] export the final selected-configuration tables showing which params won for each model

- [ ] Start the thesis-writing phase
  - [ ] finalize the practical-part outline
  - [ ] prepare a chronological implementation log from repo notes, code changes, and notebook evolution
  - [ ] draft the practical-part text from the final code and outputs
  - [ ] make sure every exported table and chart used in the thesis has an explicit interpretation in the practical part

## Secondary Tasks

- [ ] Optionally test Google Colab again only if BTC runtime becomes a blocker
- [ ] Consider a lightweight script / CLI wrapper for non-notebook execution
- [ ] Optionally add a short appendix note explaining why some DM pairs can be skipped

## Current Baseline

- [x] Shared data loading moved into `src/data_loader.py`
- [x] Shared notebook imports and setup exist in `src/notebook_setup.py`
- [x] Shared plotting and export logic moved into `src/visualization.py`
- [x] Forecast notebooks `01` to `04` were executed after the refactor
- [x] Multiseries workflow for notebook `04` is integrated into shared code
- [x] Foundation models are wired in: Chronos2, GraniteTTM, TimeGPT
- [x] `GraniteTTM` is included in the foundation category used by DM comparisons
- [x] Statistical preprocessing is now dataset-level (`raw` vs `log`) instead of model-level mixing
- [x] Diebold-Mariano testing now uses dedicated rolling backtest artifacts instead of test-set selection
- [x] Selected winning model configurations can be exported as a separate parameter-summary table
- [ ] Notebook `05` is still the only dataset that remains in reduced / smoke-test state
