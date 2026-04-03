import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from sklearn.preprocessing import LabelEncoder
import os

from src.runtime_context import (
    set_current_dataset_config,
    set_current_target_diagnostics,
)
from src.statistical_transforms import analyze_series_for_log

def load_dataset(config, smoke_test=False, smoke_test_points=2000):
    """
    Base loader for loading raw data and creating TimeSeries objects.
    """
    name = config.get("name", "Unknown Dataset")
    print(f"Loading dataset: {name}")
    set_current_dataset_config(config)
    
    # 1. Load File
    file_path = config["file_path"]
    if not os.path.exists(file_path):
        if file_path.startswith("../") and os.getcwd().endswith("Forecasting models"):
             file_path = file_path.replace("../", "")
    df = pd.read_csv(file_path)
    
    # 2. Date Conversion
    df[config["time_column"]] = pd.to_datetime(df[config["time_column"]])
    
    # 3. Unit Scaling
    if "value_scale" in config and config["value_scale"] != 1:
        df[config["target_column"]] = df[config["target_column"]] / config["value_scale"]

    # 4. Create TimeSeries
    extras = {}
    is_multiseries = "id_column" in config and config["id_column"]
    
    if is_multiseries:
        id_col = config["id_column"]
        if "static_covariates" in config and config["static_covariates"]:
            encoders = {}
            for col in config["static_covariates"]:
                if col == id_col:
                    continue
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            extras["encoders"] = encoders

        series_static_cols = config.get("static_covariates")
        if series_static_cols and id_col in series_static_cols:
            series_static_cols = [c for c in series_static_cols if c != id_col]

        series = TimeSeries.from_group_dataframe(
            df, time_col=config["time_column"], group_cols=id_col,
            value_cols=config["target_column"], static_cols=series_static_cols, freq=config["frequency"]
        )
        series = [s.astype(np.float32) for s in series]
        
        if smoke_test:
            series = [s[-smoke_test_points:] for s in series]
            
        if "future_covariates" in config and config["future_covariates"]:
            f_covs = TimeSeries.from_group_dataframe(
                df, time_col=config["time_column"], group_cols=id_col,
                value_cols=config["future_covariates"], freq=config["frequency"]
            )
            extras["future_covariates"] = [s.astype(np.float32) for s in f_covs]
            if smoke_test: extras["future_covariates"] = [s[-smoke_test_points:] for s in extras["future_covariates"]]

        if "past_covariates" in config and config["past_covariates"]:
            p_covs = TimeSeries.from_group_dataframe(
                df, time_col=config["time_column"], group_cols=id_col,
                value_cols=config["past_covariates"], freq=config["frequency"]
            )
            extras["past_covariates"] = [s.astype(np.float32) for s in p_covs]
            if smoke_test: extras["past_covariates"] = [s[-smoke_test_points:] for s in extras["past_covariates"]]
    else:
        series = TimeSeries.from_dataframe(
            df, time_col=config["time_column"], value_cols=config["target_column"], freq=config.get("frequency")
        ).astype(np.float32)
        if smoke_test:
            series = series[-smoke_test_points:]

    extras["dataframe"] = df
    return series, extras

def get_prepared_data(config):
    """
    Ultra-refactored loader that returns a full dictionary of variables 
    needed for the forecasting pipeline.
    """
    # Smoke Test Logic: Active if smoke_test_points is defined and > 0
    smoke_test_points = config.get("smoke_test_points")
    smoke_test = bool(smoke_test_points and smoke_test_points > 0)
    
    test_periods = config["test_periods"]
    freq_str = config["frequency"]
    
    series, extras = load_dataset(config, smoke_test, smoke_test_points)
    
    # 1. Splitting
    offset_map = {
        "MS": pd.DateOffset(months=test_periods),
        "QS": pd.DateOffset(months=test_periods * 3),
        "YS": pd.DateOffset(years=test_periods),
        "D": pd.DateOffset(days=test_periods),
        "H": pd.DateOffset(hours=test_periods),
    }
    offset = offset_map.get(freq_str, pd.DateOffset(days=test_periods))
    
    is_multi = isinstance(series, list)
    if is_multi:
        split_time = pd.Timestamp(series[0].end_time()) - offset
        train = [s.split_after(split_time)[0] for s in series]
        test = [s.split_after(split_time)[1] for s in series]
    else:
        split_time = pd.Timestamp(series.end_time()) - offset
        train, test = series.split_after(split_time)

    target_diagnostics = analyze_series_for_log(train, frequency=freq_str)
    set_current_target_diagnostics(target_diagnostics)

    # 2. Scaling Target
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    series_scaled = scaler.transform(series)
    
    # Static Covariates for Embeddings
    if is_multi:
        static_cov_transformer = StaticCovariatesTransformer()
        train_scaled = static_cov_transformer.fit_transform(train_scaled)

    data = {
        "train": train, "test": test, 
        "train_scaled": train_scaled, "test_scaled": test_scaled, 
        "series_scaled": series_scaled, "scaler": scaler,
        "series": series, "all_series": series, # Alias for consistency
        "extras": extras,
        "split_time": split_time, # Add this for plotting
        "target_diagnostics": target_diagnostics,
    }
    
    # Handle Covariates Naming for 04 (Multi-series compatible)
    data["train_series"] = train
    data["test_series"] = test
    data["target_scaler"] = scaler

    # 3. Handle Covariates Scaling
    if "future_covariates" in extras:
        f_covs = extras["future_covariates"]
        f_scaler = Scaler()
        if is_multi:
            split_time = pd.Timestamp(f_covs[0].end_time()) - offset
            train_f = [c.split_after(split_time)[0] for c in f_covs]
            data["train_future_covs_scaled"] = f_scaler.fit_transform(train_f)
            data["all_future_covs_scaled"] = f_scaler.transform(f_covs)
            data["all_future_covs"] = f_covs
        else:
            # Single series future covs not fully tested here but logic would match
            pass

    if "past_covariates" in extras:
        p_covs = extras["past_covariates"]
        p_scaler = Scaler()
        if is_multi:
            split_time = pd.Timestamp(p_covs[0].end_time()) - offset
            train_p = [c.split_after(split_time)[0] for c in p_covs]
            data["train_past_covs_scaled"] = p_scaler.fit_transform(train_p)
            data["all_past_covs_scaled"] = p_scaler.transform(p_covs)
            data["all_past_covs"] = p_covs

    # Final Info Print
    msg = f"Dataset: {config['name']}" + (" [SMOKE TEST]" if smoke_test else "")
    print(msg)
    if is_multi:
        print(f"Total: {len(series[0])} | Train: {len(train[0])} | Test: {len(test[0])}")
    else:
        print(f"Total: {len(series)} | Train: {len(train)} | Test: {len(test)}")
        
    return data
