import time
import random
import numpy as np
from itertools import product
from tqdm import tqdm
from darts.metrics import rmse, mape

# Nastavení seedu pro reprodukovatelnost
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def grid_search_all(param_grid):
    """Vygeneruje všechny kombinace parametrů pro Grid Search."""
    if not param_grid:
        return [{}]
    keys, values = list(param_grid.keys()), list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def random_grid_search(param_grid, n_iter=5, seed=RANDOM_SEED):
    """Vygeneruje náhodné kombinace parametrů."""
    rng = random.Random(seed)
    keys = list(param_grid.keys())
    if not keys:
        return [{}]

    combinations = [
        {key: rng.choice(param_grid[key]) for key in keys} for _ in range(n_iter)
    ]
    seen = set()
    # Odstranění duplicit
    return [
        d
        for d in combinations
        if not (key := frozenset((k, str(v)) for k, v in d.items())) in seen
        and not seen.add(key)
    ]


def evaluate_model(
    model_cls,
    params,
    series_train,
    is_dl=False,
    forecast_horizon=12,
    stride=1,
    start_p=0.7,
    scaler=None,
    original_train=None,
):
    """
    Vyhodnotí jeden model pomocí rolling window cross-validation na trénovacích datech.
    """
    start_time = time.time()
    try:
        model = model_cls(**params)

        if is_dl:
            # Deep Learning modely: fit jednou, pak historical_forecasts bez retrain
            # (vyžaduje scaler a original_train pro inverzní transformaci)
            model.fit(series_train, verbose=False)
            backtest = model.historical_forecasts(
                series=series_train,
                start=start_p,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,
                verbose=False,
                last_points_only=True,
            )
            # Inverzní transformace
            backtest_unscaled = scaler.inverse_transform(backtest)
            actual = original_train.slice_intersect(backtest_unscaled)
            metric_pred = backtest_unscaled
        else:
            # Statistické modely: retrain=True
            backtest = model.historical_forecasts(
                series=series_train,
                start=start_p,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=True,
                verbose=False,
                last_points_only=True,
            )
            actual = series_train.slice_intersect(backtest)
            metric_pred = backtest

        # Výpočet metrik
        rmse_val = rmse(actual, metric_pred)
        mape_val = mape(actual, metric_pred)

        return rmse_val, mape_val, time.time() - start_time

    except Exception as e:
        # V případě chyby (např. špatné parametry) vrátíme nekonečné error metriky
        # print(f"Error evaluating {model_cls.__name__}: {e}") # Odkomentovat pro debug
        return float("inf"), float("inf"), 0


def run_tuning_and_eval(
    tracker,  # <--- DŮLEŽITÉ: Předáváme instanci trackeru
    model_name,
    model_cls,
    param_grid,
    train_series,
    is_dl=False,
    n_iter=5,
    scaler=None,
    original_train=None,
    use_full_grid=False,
    forecast_horizon=None,  # Pokud None, bere se z configu notebooku
    test_periods=None,  # Pro určení horizon, pokud není zadán explicitně
):
    """
    Řídí proces ladění hyperparametrů a loguje výsledky do trackeru.
    """
    tuning_start = time.time()

    # Pokud není zadán horizon, zkusíme ho odvodit (fallback na 12)
    horizon = (
        forecast_horizon if forecast_horizon else (test_periods if test_periods else 12)
    )

    # 1. Base run (pokud je grid prázdný, spustíme defaultní nastavení)
    if not param_grid:
        rmse_val, mape_val, cfg_time = evaluate_model(
            model_cls,
            {},
            train_series,
            is_dl=is_dl,
            forecast_horizon=horizon,
            scaler=scaler,
            original_train=original_train,
        )
        if rmse_val != float("inf"):
            tracker.log(
                model_name,
                rmse_val,
                mape_val,
                time.time() - tuning_start,
                cfg_time,
                {},
                1,
            )
        return {}

    # 2. Generování kombinací
    combinations = (
        grid_search_all(param_grid)
        if use_full_grid
        else random_grid_search(param_grid, n_iter=n_iter, seed=RANDOM_SEED)
    )

    best_rmse, best_params, best_mape, best_cfg_time = float("inf"), None, 0, 0

    # 3. Iterace přes parametry
    # Používáme tqdm pro progress bar
    for params in tqdm(combinations, desc=model_name):
        rmse_val, mape_val, cfg_time = evaluate_model(
            model_cls,
            params,
            train_series,
            is_dl=is_dl,
            forecast_horizon=horizon,
            scaler=scaler,
            original_train=original_train,
        )

        if rmse_val < best_rmse:
            best_rmse, best_params, best_mape, best_cfg_time = (
                rmse_val,
                params,
                mape_val,
                cfg_time,
            )

    # 4. Logování nejlepšího výsledku
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


# === MULTI-SERIES TUNING ===


def evaluate_local_model(
    model_cls,
    params,
    train_list,
    future_covs_list=None,
    supports_covariates=False,
    forecast_horizon=7,
    stride=1,
):
    """Evaluates local model on a list of series (trains separate model for each)."""
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

            backtest = model.historical_forecasts(
                series=train_s,
                future_covariates=future_cov,
                start=0.7,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=True,
                verbose=False,
                last_points_only=True,
            )
            all_backtest.append(backtest)
            all_actual.append(train_s.slice_intersect(backtest))
        except Exception as e:
            errors.append(f"series_{i}: {str(e)[:50]}")

    if not all_backtest:
        return float("inf"), float("inf"), 0, f"All failed"

    # Compute average metrics across all series
    avg_rmse = np.mean([rmse(a, b) for a, b in zip(all_actual, all_backtest)])
    avg_mape = np.mean([mape(a, b) for a, b in zip(all_actual, all_backtest)])

    return avg_rmse, avg_mape, time.time() - start_time, None


def evaluate_global_model(
    model_cls,
    params,
    train_scaled_list,
    original_train_list,
    scaler,
    future_covs=None,
    past_covs=None,
    forecast_horizon=7,
    stride=1,
):
    """Evaluates global model (trains once on all series)."""
    start_time = time.time()
    try:
        model = model_cls(**params)
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
                start=0.7,
                forecast_horizon=forecast_horizon,
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
        avg_mape = np.mean([mape(a, b) for a, b in zip(all_actual, all_backtest)])
        return avg_rmse, avg_mape, time.time() - start_time, None
    except Exception as e:
        return float("inf"), float("inf"), 0, str(e)[:100]


def run_tuning_local(
    tracker,
    model_name,
    model_cls,
    param_grid,
    train_list,
    future_covs_list=None,
    supports_covariates=False,
    use_full_grid=True,
    n_iter=5,
    seasonal_period=1,
):
    """Tuning loop for Local models."""
    tuning_start = time.time()
    combinations = (
        grid_search_all(param_grid)
        if use_full_grid
        else random_grid_search(param_grid, n_iter=n_iter)
    )

    # Filter HW invalid params
    if "Holt" in model_name:
        combinations = [
            p
            for p in combinations
            if not (p.get("damped", False) and p.get("trend") == ModelMode.NONE)
        ]

    best_rmse, best_params, best_mape, best_cfg_time = float("inf"), None, 0, 0

    for params in tqdm(combinations, desc=model_name):
        rmse_val, mape_val, cfg_time, err = evaluate_local_model(
            model_cls,
            params,
            train_list,
            future_covs_list,
            supports_covariates,
            stride=seasonal_period,
        )
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


def run_tuning_global(
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
):
    """Tuning loop for Global models."""
    tuning_start = time.time()
    combinations = (
        grid_search_all(param_grid)
        if use_full_grid
        else random_grid_search(param_grid, n_iter=n_iter)
    )

    best_rmse, best_params, best_mape, best_cfg_time = float("inf"), None, 0, 0

    for params in tqdm(combinations, desc=model_name):
        rmse_val, mape_val, cfg_time, err = evaluate_global_model(
            model_cls,
            params,
            train_scaled_list,
            original_train_list,
            scaler,
            future_covs,
            past_covs,
            stride=seasonal_period,
        )
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
