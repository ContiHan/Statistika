# TODO

Only active follow-up items should live here.

## Current Priorities

- [ ] Finish final thesis artifact placement
  - [ ] verify all figure and table references inside the thesis text
  - [ ] verify appendix numbering after the last layout pass
  - [ ] verify that all inserted thesis tables come from the final CSV exports

- [ ] Keep one final benchmark result set per dataset
  - [ ] verify `images/forecasting/` contains only the final PNGs to keep
  - [ ] verify `artifacts/forecasting/` contains the final kept CSV bundles for `01` to `05`
  - [ ] avoid further notebook reruns unless a real issue is discovered

- [ ] Finish README / repo hygiene
  - [ ] keep `README.md` aligned with the final benchmark setup
  - [ ] keep only active items in `todos.md`
  - [ ] avoid introducing new scattered markdown planning files

- [ ] Optional thesis-support polish
  - [ ] decide whether to add a short appendix note explaining why some DM pairs can be skipped
  - [ ] decide whether to add one more runtime-oriented thesis table based on `Best Config Time (s)`
  - [ ] decide whether to build a lightweight interactive result browser from `artifacts/forecasting/`

## Current Baseline

- [x] Shared data loading moved into `src/data_loader.py`
- [x] Shared notebook imports and setup exist in `src/notebook_setup.py`
- [x] Shared plotting and export logic moved into `src/visualization.py`
- [x] Shared machine-readable export helper exists in `src/export_data.py`
- [x] Per-dataset export bundles are written under `artifacts/forecasting/<dataset_slug>/`
- [x] Thesis-oriented CSV tables are written under `artifacts/thesis_tables/`
- [x] Multiseries workflow for notebook `04` is integrated into shared code
- [x] Foundation models are wired in: Chronos2, GraniteTTM, TimeGPT
- [x] Statistical preprocessing is dataset-level (`raw` vs `log`)
- [x] Dedicated Diebold-Mariano comparisons use separate rolling backtest artifacts
- [x] Notebook `05` was finalized as a reduced `50k` run after runtime testing
