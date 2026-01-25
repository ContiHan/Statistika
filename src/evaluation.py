import numpy as np
import pandas as pd
from scipy.stats import t, norm
from darts import TimeSeries

def diebold_mariano_test(
    target: TimeSeries,
    pred1: TimeSeries,
    pred2: TimeSeries,
    h: int = 1,
    criterion: str = "mse",
):
    """
    Performs the Diebold-Mariano test to compare the forecast accuracy of two models.
    Includes the Harvey-Leybourne-Newbold correction for small sample sizes.

    Args:
        target: Ground truth series.
        pred1: Predictions from the first model.
        pred2: Predictions from the second model.
        h: Forecast horizon.
        criterion: Loss function to use ('mse' or 'mae').

    Returns:
        dict: DM statistic, p-value, and interpretation.
    """
    # Ensure we are working with numpy arrays of the same length
    y = target.values().flatten()
    y_hat1 = pred1.values().flatten()
    y_hat2 = pred2.values().flatten()

    n = len(y)
    if len(y_hat1) != n or len(y_hat2) != n:
        raise ValueError("All series must have the same length.")

    # Calculate errors
    e1 = y - y_hat1
    e2 = y - y_hat2

    # Define loss differential
    if criterion.lower() == "mse":
        d = e1**2 - e2**2
    elif criterion.lower() == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Criterion must be 'mse' or 'mae'.")

    d_bar = np.mean(d)

    # Autocovariance estimation (HAC - Heteroskedasticity and Autocorrelation Consistent)
    def autocovariance(xi, k):
        n_xi = len(xi)
        if k == 0:
            return np.var(xi)
        if k >= n_xi:
            return 0.0
        # Manual autocovariance to be more stable on tiny samples than np.cov
        xi_mean = np.mean(xi)
        return np.sum((xi[:n_xi-k] - xi_mean) * (xi[k:] - xi_mean)) / n_xi

    # Variance of the mean differential
    # We sum autocovariances up to h-1 lags, but never more than n-1
    max_lags = min(h, n)
    gamma = np.array([autocovariance(d, i) for i in range(max_lags)])
    
    # Standard DM variance formula
    if len(gamma) > 1:
        v_d = (gamma[0] + 2 * np.sum(gamma[1:])) / n
    else:
        v_d = gamma[0] / n

    # Ensure v_d is positive and not tiny to avoid division issues
    if v_d < 1e-12:
        v_d = np.var(d) / n if np.var(d) > 0 else 1e-12

    # Standard Diebold-Mariano Statistic
    dm_stat = d_bar / np.sqrt(v_d)

    # Harvey-Leybourne-Newbold (HLN) Correction for small samples
    # Crucial for yearly/quarterly data. 
    # Fallback to standard DM if n is too small relative to h to prevent zeroing out.
    if n > (2*h - 1):
        hln_correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1) / n)) / n)
        dm_stat_corrected = hln_correction * dm_stat
        applied_hln = True
    else:
        # If n is too small, HLN correction is unstable/zero. Use standard DM or simple scale.
        dm_stat_corrected = dm_stat
        applied_hln = False

    # P-value calculation (using t-distribution with n-1 degrees of freedom)
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat_corrected), df=n - 1))

    return {
        "dm_stat": dm_stat_corrected,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "better_model": "Model 1" if dm_stat_corrected < 0 else "Model 2",
        "criterion": criterion,
        "n": n,
        "h": h,
        "hln_applied": applied_hln
    }


def run_statistical_comparison(tracker, final_predictions, test_series, h=1):
    """
    Runs key Diebold-Mariano tests and returns a DataFrame for visualization.
    Refactored to select best models based on TEST SET performance (final_predictions).
    """
    if not final_predictions:
        return pd.DataFrame()

    results_tracker = tracker.get_results_df() # Still needed for Tuning Time lookups

    # 1. Identify Best Models based on TEST set RMSE
    # final_predictions structure: {'ModelName': {'rmse': float, 'tuning_time': float, ...}}
    
    # Filter out special keys like 'best_rmse' or 'fastest' if they duplicate model entries
    # We want to iterate over actual model names.
    # Assuming standard model names are keys in final_predictions.
    model_stats = []
    for m_name, info in final_predictions.items():
        if m_name in ["best_rmse", "fastest"]:
            continue
        model_stats.append({
            "Model": m_name,
            "RMSE": info["rmse"],
            "Time": info.get("tuning_time", float("inf"))
        })
    
    df_test_stats = pd.DataFrame(model_stats)
    
    if df_test_stats.empty:
        return pd.DataFrame()

    # Helper to find best model in a category (by Test RMSE)
    def get_best_test(models_list):
        subset = df_test_stats[df_test_stats["Model"].isin(models_list)]
        if subset.empty:
            return None
        return subset.sort_values("RMSE").iloc[0]["Model"]

    # Categories
    dl_models = ["TiDE", "N-BEATS", "TFT"]
    stat_models = ["AutoARIMA", "Prophet", "Holt-Winters"]
    found_models = ["Chronos", "TimeGPT"]

    comparisons = []

    # 1. Best vs 2nd Best (on Test Set)
    if len(df_test_stats) >= 2:
        sorted_models = df_test_stats.sort_values("RMSE")
        m1 = sorted_models.iloc[0]["Model"]
        m2 = sorted_models.iloc[1]["Model"]
        comparisons.append(("Best vs 2nd Best", m1, m2))

    # 2. Best DL vs Best Stat (on Test Set)
    bdl = get_best_test(dl_models)
    bstat = get_best_test(stat_models)
    if bdl and bstat:
        comparisons.append(("DL vs Statistical", bdl, bstat))

    # 3. Best Foundation vs Best DL (on Test Set)
    bfound = get_best_test(found_models)
    if bfound and bdl:
        comparisons.append(("Foundation vs DL", bfound, bdl))

    # 4. Best Accuracy vs Fastest
    # We use the fastest model from the TRACKER (tuning time), but compare it against the Best RMSE on TEST.
    best_rmse_test = df_test_stats.sort_values("RMSE").iloc[0]["Model"]
    if not results_tracker.empty:
        fastest_tracker = results_tracker.sort_values("Tuning Time (s)").iloc[0]["Model"]
    else:
        fastest_tracker = df_test_stats.sort_values("Time").iloc[0]["Model"]

    comparisons.append(("Best vs Fastest", best_rmse_test, fastest_tracker))

    # Run Tests
    data = []
    seen = set()
    
    for label, m1, m2 in comparisons:
        # Check for identical models (e.g. Best is also Fastest)
        if m1 == m2:
             data.append({
                "Comparison": label,
                "Model A": m1,
                "Model B": m2,
                "DM Stat": 0.0,
                "P-Value": 1.0,
                "Significant": "N/A",
                "Winner": "Same Model"
            })
             continue

        pair_key = tuple(sorted((m1, m2)))
        # We allow duplicates if the label is different (e.g. might want to see Best vs 2nd Best AND DL vs Stat even if same pair)
        # But if you strictly want unique pairs, keep 'seen'. 
        # For clarity in the report, it is better to show all 4 categories even if pairs repeat.
        # So I will remove the 'seen' check to ensure 4 rows.
        
        if m1 not in final_predictions or m2 not in final_predictions:
            continue
            
        try:
            pred1 = final_predictions[m1]['prediction']
            pred2 = final_predictions[m2]['prediction']
            
            # Check for identical predictions
            vals1 = pred1.values().flatten() if not isinstance(pred1, list) else np.concatenate([p.values().flatten() for p in pred1])
            vals2 = pred2.values().flatten() if not isinstance(pred2, list) else np.concatenate([p.values().flatten() for p in pred2])

            if np.allclose(vals1, vals2):
                 data.append({
                    "Comparison": label,
                    "Model A": m1,
                    "Model B": m2,
                    "DM Stat": 0.0,
                    "P-Value": 1.0,
                    "Significant": "No",
                    "Winner": "Identical predictions"
                })
                 continue

            # Run DM Test
            res = diebold_mariano_test(test_series, pred1, pred2, h=h)
            
            winner = m1 if res["better_model"] == "Model 1" else m2
            
            data.append({
                "Comparison": label,
                "Model A": m1,
                "Model B": m2,
                "DM Stat": res["dm_stat"],
                "P-Value": res["p_value"],
                "Significant": "Yes" if res["is_significant"] else "No",
                "Winner": winner if res["is_significant"] else "-"
            })
        except Exception as e:
            print(f"Error comparing {m1} vs {m2}: {e}")

    return pd.DataFrame(data)