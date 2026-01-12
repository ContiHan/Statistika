# soubor: src/tuning.py
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
import random
import traceback
from darts.metrics import rmse, mape
from darts.utils.utils import ModelMode


def grid_search_all(param_grid):
    """Generates all combinations of parameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations


def random_grid_search(param_grid, n_iter=10):
    """Generates random sample of parameter combinations."""
    all_combinations = grid_search_all(param_grid)
    if len(all_combinations) <= n_iter:
        return all_combinations
    return random.sample(all_combinations, n_iter)


# === MULTI-SERIES TUNING (04) ===


def evaluate_local_model(
    model_cls,
    params,
    train_list,
    future_covs_list=None,
    supports_covariates=False,
    test_periods=None,
    stride=1,
    cv_start_ratio=0.7,
):
    start_time = time.time()
    all_backtest, all_actual = [], []
    errors = []

    for i, train_s in enumerate(train_list):
        try:
            model = model_cls(**params)
            future_cov = (
                future_covs_list[i]
                if (supports_covariates and future_covs_list)
                else None
            )

            # Local models (stat) are fast, so we can use retrain=True
            backtest = model.historical_forecasts(
                series=train_s,
                future_covariates=future_cov,
                start=cv_start_ratio,
                forecast_horizon=test_periods,
                stride=stride,
                retrain=True,
                verbose=False,
                last_points_only=True,
            )
            all_backtest.append(backtest)
            all_actual.append(train_s.slice_intersect(backtest))
        except Exception as e:
            errors.append(f"series_{i}: {str(e)[:100]}")

    if not all_backtest:
        return float("inf"), 0, 0, f"All failed: {errors[0] if errors else 'Unknown'}"

    try:
        avg_rmse = np.mean([rmse(a, b) for a, b in zip(all_actual, all_backtest)])
    except Exception:
        avg_rmse = float("inf")

    try:
        avg_mape = np.mean([mape(a, b) for a, b in zip(all_actual, all_backtest)])
    except Exception:
        avg_mape = 0.0

    return avg_rmse, avg_mape, time.time() - start_time, None


def evaluate_global_model(
    model_cls,
    params,
    train_scaled_list,
    original_train_list,
    scaler,
    future_covs=None,
    past_covs=None,
    stride=1,
    cv_start_ratio=0.7,
    test_periods=None,
):
    start_time = time.time()
    try:
        model = model_cls(**params)
        # Global models train ONCE on the full dataset (or subset)
        model.fit(
            series=train_scaled_list,
            future_covariates=future_covs,
            past_covariates=past_covs,
            verbose=False,
        )

        all_backtest, all_actual = [], []
        for i, (train_s, orig_train) in enumerate(
            zip(train_scaled_list, original_train_list)
        ):
            backtest_scaled = model.historical_forecasts(
                series=train_s,
                future_covariates=future_covs[i] if future_covs else None,
                past_covariates=past_covs[i] if past_covs else None,
                start=cv_start_ratio,
                forecast_horizon=test_periods,
                stride=stride,
                retrain=False,
                verbose=False,
                last_points_only=True,
            )
            all_backtest.append(scaler.inverse_transform(backtest_scaled))
            all_actual.append(
                orig_train.slice_intersect(scaler.inverse_transform(backtest_scaled))
            )

        avg_rmse = np.mean([rmse(a, b) for a, b in zip(all_actual, all_backtest)])
        try:
            avg_mape = np.mean([mape(a, b) for a, b in zip(all_actual, all_backtest)])
        except Exception:
            avg_mape = 0.0

        return avg_rmse, avg_mape, time.time() - start_time, None
    except Exception as e:
        return float("inf"), 0, 0, str(e)[:100]


def run_tuning_local_and_eval(
    tracker,
    model_name,
    model_cls,
    param_grid,
    train_list,
    future_covs_list=None,
    supports_covariates=False,
    use_full_grid=True,
    n_iter=5,
    test_periods=None,
    seasonal_period=1,
    cv_start_ratio=0.7,
):
    tuning_start = time.time()
    combinations = (
        grid_search_all(param_grid)
        if use_full_grid
        else random_grid_search(param_grid, n_iter=n_iter)
    )

    if "Holt" in model_name:
        combinations = [
            p
            for p in combinations
            if not (p.get("damped", False) and p.get("trend") == ModelMode.NONE)
        ]

    best_rmse, best_params, best_mape, best_cfg_time = float("inf"), None, 0, 0

    loop = tqdm(combinations, desc=model_name)
    for params in loop:
        rmse_val, mape_val, cfg_time, err = evaluate_local_model(
            model_cls,
            params,
            train_list,
            future_covs_list,
            supports_covariates,
            test_periods=test_periods,
            stride=seasonal_period,
            cv_start_ratio=cv_start_ratio,
        )

        if rmse_val < float("inf"):
            loop.set_postfix(rmse=f"{rmse_val:.2f}", best=f"{best_rmse:.2f}")
        else:
            loop.set_postfix(status="Err")

        if rmse_val < best_rmse:
            best_rmse, best_params, best_mape, best_cfg_time = (
                rmse_val,
                params,
                mape_val,
                cfg_time,
            )

    if best_params and best_rmse != float("inf"):
        tracker.log(
            model_name,
            best_rmse,
            best_mape,
            time.time() - tuning_start,
            best_cfg_time,
            best_params,
            len(combinations),
        )

    return best_params


def run_tuning_global_and_eval(
    tracker,
    model_name,
    model_cls,
    param_grid,
    train_scaled_list,
    original_train_list,
    scaler,
    future_covs=None,
    past_covs=None,
    use_full_grid=False,
    n_iter=10,
    seasonal_period=1,
    cv_start_ratio=0.7,
    test_periods=None,
):
    tuning_start = time.time()
    combinations = (
        grid_search_all(param_grid)
        if use_full_grid
        else random_grid_search(param_grid, n_iter=n_iter)
    )

    best_rmse, best_params, best_mape, best_cfg_time = float("inf"), None, 0, 0

    loop = tqdm(combinations, desc=model_name)
    for params in loop:
        rmse_val, mape_val, cfg_time, err = evaluate_global_model(
            model_cls,
            params,
            train_scaled_list,
            original_train_list,
            scaler,
            future_covs,
            past_covs,
            stride=seasonal_period,
            cv_start_ratio=cv_start_ratio,
            test_periods=test_periods,
        )

        if rmse_val < float("inf"):
            loop.set_postfix(rmse=f"{rmse_val:.2f}", best=f"{best_rmse:.2f}")
        else:
            loop.set_postfix(status="Err")

        if rmse_val < best_rmse:
            best_rmse, best_params, best_mape, best_cfg_time = (
                rmse_val,
                params,
                mape_val,
                cfg_time,
            )

    if best_params and best_rmse != float("inf"):
        tracker.log(
            model_name,
            best_rmse,
            best_mape,
            time.time() - tuning_start,
            best_cfg_time,
            best_params,
            len(combinations),
        )

    return best_params


# === SINGLE SERIES TUNING (Legacy for 01-03, 05) ===


def evaluate_model(
    model_cls,
    params,
    train_series,
    test_periods=None,
    is_dl=False,
    scaler=None,
    original_train=None,
    cv_start_ratio=0.7,
    stride=1,
):
    """
    Single series evaluation (Legacy for 01-03).
    """
    start_time = time.time()
    try:
        model = model_cls(**params)
        if is_dl:
            # === OPRAVA PRO DL MODELY ===
            # Pokud je retrain=False, model musí být předem natrénován.
            # Natrénujeme ho na "historické" části dat (prvních 70%), kterou pak nebude předpovídat.
            # Tím zabráníme chybě "model not fitted" a zároveň Data Leakage.

            split_idx = int(len(train_series) * cv_start_ratio)
            train_subset = train_series[:split_idx]
            model.fit(train_subset, verbose=False)
            # ============================

            # DL models use scaled data for training
            backtest_scaled = model.historical_forecasts(
                series=train_series,
                start=cv_start_ratio,
                forecast_horizon=test_periods,
                stride=stride,
                retrain=False,  # Nyní validní, model je natrénován
                verbose=False,
                last_points_only=True,
            )
            # Inverse transform to get real values
            backtest = scaler.inverse_transform(backtest_scaled)
            # Compare with ORIGINAL train data (slice_intersect ensures alignment)
            actual = original_train.slice_intersect(backtest)
        else:
            # Statistical models use original data
            backtest = model.historical_forecasts(
                series=train_series,
                start=cv_start_ratio,
                forecast_horizon=test_periods,
                stride=stride,
                retrain=True,
                verbose=False,
                last_points_only=True,
            )
            actual = train_series.slice_intersect(backtest)

        rmse_val = rmse(actual, backtest)
        mape_val = mape(actual, backtest)

        return rmse_val, mape_val, time.time() - start_time, None

    except Exception as e:
        # print(f"\n[DEBUG] Error in {model_cls.__name__}: {str(e)[:200]}")
        # tqdm.write(f"Error in {model_cls.__name__}: {str(e)[:150]}")
        return float("inf"), float("inf"), 0, str(e)


def run_tuning_and_eval(
    tracker,
    model_name,
    model_cls,
    param_grid,
    train_series,
    use_full_grid=True,
    n_iter=10,
    test_periods=None,
    is_dl=False,
    scaler=None,
    original_train=None,
    cv_start_ratio=0.7,
):
    """
    Main tuning function for Single Series (Legacy for 01-03).
    """
    tuning_start = time.time()
    combinations = (
        grid_search_all(param_grid)
        if use_full_grid
        else random_grid_search(param_grid, n_iter=n_iter)
    )

    if "Holt" in model_name:
        combinations = [
            p
            for p in combinations
            if not (p.get("damped", False) and p.get("trend") == ModelMode.NONE)
        ]

    best_rmse, best_params, best_mape, best_cfg_time = float("inf"), None, 0, 0

    loop = tqdm(combinations, desc=model_name)
    for params in loop:
        rmse_val, mape_val, cfg_time, err = evaluate_model(
            model_cls,
            params,
            train_series,
            test_periods=test_periods,
            is_dl=is_dl,
            scaler=scaler,
            original_train=original_train,
            cv_start_ratio=cv_start_ratio,
        )
        if rmse_val < float("inf"):
            loop.set_postfix(rmse=f"{rmse_val:.2f}", best=f"{best_rmse:.2f}")
        else:
            loop.set_postfix(status="Err")

        if rmse_val < best_rmse:
            best_rmse, best_params, best_mape, best_cfg_time = (
                rmse_val,
                params,
                mape_val,
                cfg_time,
            )

    if best_params and best_rmse != float("inf"):
        tracker.log(
            model_name,
            best_rmse,
            best_mape,
            time.time() - tuning_start,
            best_cfg_time,
            best_params,
            len(combinations),
        )

    return best_params
