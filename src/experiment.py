import pandas as pd
from typing import Dict, Any, Optional
from src.evaluation import diebold_mariano_test


class ExperimentTracker:
    def __init__(self):
        self.results = []

    def log(
        self,
        model_name: str,
        rmse_val: float,
        mape_val: float,
        tuning_time: float,
        best_config_time: float,
        params: Optional[Dict[str, Any]] = None,
        n_combinations: int = 1,
    ):
        """
        Saves experiment result and prints it to console.
        """
        entry = {
            "Model": model_name,
            "RMSE": rmse_val,
            "MAPE": mape_val,
            "Tuning Time (s)": tuning_time,
            "Best Config Time (s)": best_config_time,
            "Combinations": n_combinations,
            "Params": params,
        }
        self.results.append(entry)

        print(
            f"{model_name}: RMSE={rmse_val:.4f} | MAPE={mape_val:.2f}% | "
            f"Time={tuning_time:.1f}s ({n_combinations} combinations)"
        )

    def compare_models(self, target, model_a_name, model_b_name, predictions_dict, h=1):
        """
        Statistical comparison using Diebold-Mariano test.
        """
        if model_a_name not in predictions_dict or model_b_name not in predictions_dict:
            print(f"Predictions for {model_a_name} or {model_b_name} not found.")
            return None

        p1 = predictions_dict[model_a_name]
        p2 = predictions_dict[model_b_name]

        # If it's a dict from pipeline (containing 'prediction' key)
        if isinstance(p1, dict): p1 = p1['prediction']
        if isinstance(p2, dict): p2 = p2['prediction']

        res = diebold_mariano_test(target, p1, p2, h=h)

        print(f"\n--- Diebold-Mariano Test: {model_a_name} vs {model_b_name} ---")
        print(f"DM Statistic: {res['dm_stat']:.4f}")
        print(f"P-value:      {res['p_value']:.4f}")
        print(f"Significant:  {res['is_significant']}")
        print(f"Better Model: {model_a_name if res['better_model'] == 'Model 1' else model_b_name}")

        return res

    def get_results_df(self) -> pd.DataFrame:
        """Returns results as DataFrame sorted by RMSE."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results).sort_values(by="RMSE")

    def get_best_model(self):
        """Returns row (Series) of the best model (lowest RMSE)."""
        df = self.get_results_df()
        if df.empty:
            return None
        return df.iloc[0]

    def get_fastest_model(self):
        """Returns row (Series) of the fastest model."""
        df = self.get_results_df()
        if df.empty:
            return None
        return df.sort_values("Tuning Time (s)").iloc[0]
