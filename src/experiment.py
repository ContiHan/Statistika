import pandas as pd
from typing import Dict, Any, Optional


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
