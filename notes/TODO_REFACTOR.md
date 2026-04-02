# Refactoring TODO & Status

## ✅ Completed (Refactored)
1. **Created `src/data_loader.py`**:
   - Functions `load_dataset` and `get_prepared_data`.
   - Handles CSV loading, scaling, `smoke_test`, splitting (Train/Test), and multiseries logic.
   - Returns a complete dictionary of variables (`train`, `test`, `scaler`, etc.).
2. **Updated `src/visualization.py`**:
   - Added `plot_data_split` function for visualizing data distribution (replaces manual matplotlib code).
3. **Refactored `raw_py_scripts/*.py`**:
   - All scripts (01-05) now use `get_prepared_data` in Section 2.
4. **Refactored `forecasting/*.ipynb`**:
   - All notebooks (01-05) updated to match scripts (boilerplate code removed).

## 📝 TODO (Post-Restart Verification)
1. **Restart Kernel**: Ensure notebooks reload the updated `src/` modules.
2. **Verify 05 (BTC)**: Quick test to see if "Cell 2" loads and plots correctly.
3. **Verify 04 (Walmart)**: Test the most complex multiseries case.
4. **Smoke Test Toggle**: Remember that `smoke_test` is now auto-enabled by `smoke_test_points` in `DATASET_CONFIG`.

---
*Stored on Saturday, January 31, 2026*
