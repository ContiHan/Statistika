import time
import random
import numpy as np
from itertools import product
from tqdm.auto import tqdm
from darts.metrics import rmse, mape


# Enum pro režimy (aby se nepletly stringy)
class TuningMode:
    LOCAL = "local"  # Statistické modely (ARIMA, Prophet, HW) - retrain=True
    GLOBAL = "global"  # DL modely (TFT, TiDE, N-BEATS) - retrain=False


def generate_grid(param_grid, n_iter=5, seed=42, use_full_grid=False):
    """Generuje seznam unikátních kombinací parametrů."""
    if not param_grid:
        return [{}]

    keys, values = list(param_grid.keys()), list(param_grid.values())

    if use_full_grid:
        return [dict(zip(keys, combo)) for combo in product(*values)]

    rng = random.Random(seed)
    combinations = [
        {key: rng.choice(param_grid[key]) for key in keys} for _ in range(n_iter)
    ]

    # Odstranění duplicit
    seen = set()
    unique_combos = []
    for d in combinations:
        key = frozenset((k, str(v)) for k, v in d.items())
        if key not in seen:
            seen.add(key)
            unique_combos.append(d)
    return unique_combos


def _eval_local(
    model_cls, params, series_list, future_covs, past_covs, horizon, stride, start
):
    """Vyhodnocení LOCAL modelů (vrací tuple: avg_rmse, avg_mape)."""
    errors_rmse = []
    errors_mape = []

    if not isinstance(series_list, list):
        series_list = [series_list]
        future_covs = [future_covs] if future_covs else None
        past_covs = [past_covs] if past_covs else None

    for i, series in enumerate(series_list):
        try:
            model = model_cls(**params)
            fc = future_covs[i] if future_covs else None
            pc = past_covs[i] if past_covs else None

            backtest = model.historical_forecasts(
                series=series,
                future_covariates=fc,
                past_covariates=pc,
                start=start,
                forecast_horizon=horizon,
                stride=stride,
                retrain=True,
                verbose=False,
                last_points_only=True,
            )

            # Počítáme obě metriky
            errors_rmse.append(rmse(series, backtest))
            errors_mape.append(mape(series, backtest))
        except Exception:
            return float("inf"), float("inf")

    return np.mean(errors_rmse), np.mean(errors_mape)


def _eval_global(
    model_cls,
    params,
    train_scaled,
    val_original,
    scaler,
    future_covs,
    past_covs,
    horizon,
    stride,
    start,
):
    """Vyhodnocení GLOBAL modelů (vrací tuple: avg_rmse, avg_mape)."""
    try:
        model = model_cls(**params)
        model.fit(
            series=train_scaled,
            future_covariates=future_covs,
            past_covariates=past_covs,
            verbose=False,
        )

        errors_rmse = []
        errors_mape = []

        # Zajištění list formátu
        ts_list = train_scaled if isinstance(train_scaled, list) else [train_scaled]
        orig_list = val_original if isinstance(val_original, list) else [val_original]
        fc_list = (
            future_covs if isinstance(future_covs, list) else [None] * len(ts_list)
        )
        pc_list = past_covs if isinstance(past_covs, list) else [None] * len(ts_list)

        for i, (ts, orig) in enumerate(zip(ts_list, orig_list)):
            backtest_scaled = model.historical_forecasts(
                series=ts,
                future_covariates=fc_list[i],
                past_covariates=pc_list[i],
                start=start,
                forecast_horizon=horizon,
                stride=stride,
                retrain=False,
                verbose=False,
                last_points_only=True,
            )

            # Inverse transform pro správné metriky
            backtest = scaler.inverse_transform(backtest_scaled)
            actual = orig.slice_intersect(backtest)

            errors_rmse.append(rmse(actual, backtest))
            errors_mape.append(mape(actual, backtest))

        return np.mean(errors_rmse), np.mean(errors_mape)

    except Exception as e:
        return float("inf"), float("inf")


def tune_and_evaluate(
    model_name,
    model_cls,
    param_grid,
    mode,
    train_data,
    original_data=None,
    scaler=None,
    future_covariates=None,
    past_covariates=None,
    n_iter=5,
    use_full_grid=False,
    forecast_horizon=12,
    stride=1,
    start_split=0.7,
    logger_func=None,
):
    """Hlavní funkce volaná z notebooku."""
    start_time_all = time.time()
    combinations = generate_grid(param_grid, n_iter=n_iter, use_full_grid=use_full_grid)

    best_rmse = float("inf")
    best_mape = float("inf")
    best_params = None
    best_cfg_time = 0

    print(f"--- Tuning {model_name} ({len(combinations)} configs) | Mode: {mode} ---")

    for params in tqdm(combinations, desc=model_name):
        iter_start = time.time()

        if mode == TuningMode.LOCAL:
            curr_rmse, curr_mape = _eval_local(
                model_cls,
                params,
                train_data,
                future_covariates,
                past_covariates,
                forecast_horizon,
                stride,
                start_split,
            )
        else:  # GLOBAL
            curr_rmse, curr_mape = _eval_global(
                model_cls,
                params,
                train_data,
                original_data,
                scaler,
                future_covariates,
                past_covariates,
                forecast_horizon,
                stride,
                start_split,
            )

        # Optimalizujeme podle RMSE, ale ukládáme i MAPE
        if curr_rmse < best_rmse:
            best_rmse = curr_rmse
            best_mape = curr_mape
            best_params = params
            best_cfg_time = time.time() - iter_start

    total_time = time.time() - start_time_all

    # Logování výsledků do notebooku
    if logger_func and best_params:
        logger_func(
            model_name=model_name,
            rmse_val=best_rmse,
            mape_val=best_mape,
            tuning_time=total_time,
            best_config_time=best_cfg_time,
            params=best_params,
            n_combinations=len(combinations),
        )
    elif best_rmse == float("inf"):
        print(f"FAILED: {model_name} - všechny konfigurace selhaly.")

    return best_params
