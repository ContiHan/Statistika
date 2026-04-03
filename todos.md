# TODO

Only active and recent work should live here.

## Current Priorities

- [ ] Finalize `forecasting/05_forecasting_kaggle_btcusd_hourly.ipynb`
  - [ ] remove `smoke_test_points` for the final run
  - [ ] decide the intended final horizon and keep it consistent
    - current notebook uses `test_periods = 24`
    - older notes referenced `168`
  - [ ] rerun the notebook on full data and refresh exported images

- [ ] Fix foundation-model category comparison in `src/evaluation.py`
  - [ ] include `GraniteTTM` in the foundation category used for DM comparisons

- [ ] Clean repo truth sources
  - [ ] decide whether to delete `raw_py_scripts/` or keep them as explicit legacy copies
  - [ ] if kept, mark them clearly as non-canonical

- [ ] Align notebook comments with actual code
  - [ ] notebook `05` comment says DM uses full horizon
  - [ ] actual current call falls back to default `h=1`

- [ ] Review the small 2026-04-02 changes
  - [ ] confirm whether `src/config.py` and notebook `01` need a fresh rerun or a clearer commit message

## Secondary Tasks

- [ ] Try Google Colab
- [ ] Unify "Cell 2" patterns if any forecast notebook still diverges from the common loader/setup
- [ ] Consider adding a lightweight script or CLI wrapper for non-notebook execution

## Current Baseline

- [x] Shared data loading moved into `src/data_loader.py`
- [x] Shared notebook imports and setup exist in `src/notebook_setup.py`
- [x] Shared plotting and export logic moved into `src/visualization.py`
- [x] Forecast notebooks `01` to `04` were executed after the refactor
- [x] Multiseries workflow for notebook `04` is integrated into shared code
- [x] Foundation models are wired in: Chronos2, GraniteTTM, TimeGPT
- [ ] Notebook `05` is still only in smoke-test state
