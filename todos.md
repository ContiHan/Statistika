# TODO

Only active and recent work should live here.

## Current Priorities

- [ ] Finalize `forecasting/05_forecasting_kaggle_btcusd_hourly.ipynb`
  - [ ] keep `test_periods = 168` (`week-ahead`) and use the same horizon consistently
  - [x] keep the final BTC setup on the most recent `50k` points in smoke mode
  - [x] confirm that the full run is not practical because AutoARIMA rolling validation remains too slow
  - [ ] rerun the notebook with the frozen reduced BTC setup and refresh exported images if needed
  - [ ] record the exact reduced BTC runtime setup used in the thesis text

- [ ] Freeze final experimental settings for all datasets
  - [ ] confirm `test_periods`, `seasonal_period`, `cv_start_ratio`, and final `smoke_test_points` only where actually used
  - [ ] confirm the final dataset-level statistical transform (`raw` vs `log`)
  - [ ] keep one final result set per dataset and stop changing configs afterwards

- [ ] Regenerate final notebook outputs and exported figures
  - [ ] rerun `01` to `04` from a clean kernel if any setup or plotting code changed
  - [ ] keep `04` DM backtest on the practical setting `forecast_horizon = 1`, `stride = 2` unless there is a strong reason to revert it
  - [ ] remember that the current conservative `TimeGPT` throttle is `40 requests -> 90s sleep`, so `04` can be intentionally much slower but more stable
  - [ ] verify `images/forecasting/` contains only the final PNGs you want to keep
  - [ ] remove old duplicate M5 exports after the final rerun
  - [ ] export the final selected-configuration tables showing which params won for each model

- [ ] Start the thesis-writing phase
  - [ ] finalize the practical-part outline
  - [ ] prepare a chronological implementation log from repo notes, code changes, and notebook evolution
  - [ ] draft the practical-part text from the final code and outputs
  - [ ] make sure every exported table and chart used in the thesis has an explicit interpretation in the practical part
  - [ ] explicitly state that scaling and log transforms are inverted back to the original target scale before final comparison of models
  - [ ] explicitly explain that `05` is reported as a reduced `50k` smoke run because the full AutoARIMA-based setup was not computationally practical

- [ ] Plan structured export of run data for post-processing
  - [ ] decide which final artifacts should also be stored as CSV / tidy tables
  - [ ] discuss which structures are worth persisting after each forecast run and which are not
  - [ ] make it possible to rebuild thesis tables from saved data instead of only from PNG artifacts
  - [ ] consider storing comparison metrics, selected params, DM summaries, DM pairwise results, transform diagnostics, and forecast values in machine-readable form
  - [ ] keep open the option of a lightweight interactive web view for comparing RMSE, MAPE, timing, DM, and forecasts across datasets

## Secondary Tasks

- [ ] Optionally test Google Colab again only if BTC runtime becomes a blocker
- [ ] Consider a lightweight script / CLI wrapper for non-notebook execution
- [ ] Optionally add a short appendix note explaining why some DM pairs can be skipped
- [ ] If `04` TimeGPT keeps failing on API burst limits, add a lightweight client-side rate limiter / throttling wrapper before the final rerun
- [ ] When writing the thesis, explicitly explain why DM uses `mse` while model selection and final reporting use `RMSE`

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
- [x] Notebook `05` was finalized as a reduced `50k` smoke run after the last runtime test
