import time
import pandas as pd
import numpy as np
import torch
import traceback
from darts import TimeSeries
from darts.metrics import rmse, mape
from darts.models import (
    ExponentialSmoothing,
    AutoARIMA,
    Prophet,
    TiDEModel,
    NBEATSModel,
    TFTModel,
)

# New Imports
Chronos2Model = None
try:
    from darts.models import Chronos2Model
except ImportError:
    pass
try:
    from darts.models import ChronosModel
except ImportError:
    ChronosModel = None

from src.wrappers.granite_ttm import GraniteTTMModel
from src.config import (
    CHRONOS_AVAILABLE,
    TIMEGPT_AVAILABLE,
    NIXTLA_API_KEY,
    NixtlaClient,
)
from src.model_config import get_foundation_grids
from src.runtime_context import get_current_dataset_config
from src.statistical_transforms import (
    build_target_transform,
    clean_model_name,
    get_target_transform_name,
    strip_internal_params,
)

def _load_foundation_model(model_name, params=None):
    """
    Factory to load Foundation Models.
    """
    params = params or {}
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # default config from model_config if available
    foundation_grids = get_foundation_grids()
    
    if model_name in ["Chronos", "Chronos2"]:
        if Chronos2Model is None:
            return None
            
        # Select base config
        cfg = foundation_grids.get("Chronos2", {})
            
        # Merge params
        final_params = {**cfg, **params}
        
        # ChronosModel specific args
        return Chronos2Model(
            input_chunk_length=final_params.get("input_chunk_length", 512),
            output_chunk_length=final_params.get("output_chunk_length", 64),
            model_name=final_params.get("model_name", "amazon/chronos-2")
        )
        
    elif model_name == "GraniteTTM":
        cfg = foundation_grids.get("GraniteTTM", {})
        final_params = {**cfg, **params}
        return GraniteTTMModel(
            model_name=final_params.get("model_name", "ibm/ttm-research-v1"),
            context_length=final_params.get("context_length", 512)
        )
        
    return None


def _rolling_last_point_validation(
    series, horizon, stride, start_ratio, predict_fn, min_context_points=1
):
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    n = len(series)
    max_pred_start = n - horizon
    if max_pred_start < 1:
        raise ValueError("Not enough history for rolling validation.")

    stride = max(1, int(stride))
    requested_start_idx = max(
        1,
        int(np.floor(n * start_ratio)),
        int(min_context_points),
    )
    if requested_start_idx > max_pred_start:
        raise ValueError(
            "Series too short for rolling validation: "
            f"need at least {requested_start_idx} context points for horizon={horizon}, "
            f"but only {max_pred_start} are available."
        )

    start_idx = requested_start_idx

    pred_times = []
    pred_values = []
    actual_values = []

    for pred_start in range(start_idx, max_pred_start + 1, stride):
        context = series[:pred_start]
        target_window = series[pred_start : pred_start + horizon]
        if len(target_window) < horizon:
            continue

        pred_window = predict_fn(context, target_window)
        pred_times.append(target_window.time_index[-1])
        pred_values.append(pred_window.values(copy=False)[-1])
        actual_values.append(target_window.values(copy=False)[-1])

    if not pred_times:
        raise ValueError("No rolling validation windows available.")

    if isinstance(pred_times[0], pd.Timestamp):
        time_index = pd.DatetimeIndex(pred_times)
        validation_freq = time_index.inferred_freq
        if validation_freq is None:
            base_freq = getattr(series.time_index, "freq", None)
            if base_freq is None and hasattr(series, "freq"):
                base_freq = series.freq
            if base_freq is not None:
                try:
                    validation_freq = base_freq * stride
                except TypeError:
                    validation_freq = base_freq

        static_covariates = (
            series.static_covariates
            if hasattr(series, "static_covariates")
            else None
        )
        hierarchy = series.hierarchy if hasattr(series, "hierarchy") else None
        metadata = series.metadata if hasattr(series, "metadata") else None

        actual_ts = TimeSeries.from_times_and_values(
            time_index,
            np.vstack(actual_values),
            freq=validation_freq,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
        )
        pred_ts = TimeSeries.from_times_and_values(
            time_index,
            np.vstack(pred_values),
            freq=validation_freq,
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
        )
    else:
        time_index = pd.Index(pred_times)
        static_covariates = (
            series.static_covariates
            if hasattr(series, "static_covariates")
            else None
        )
        hierarchy = series.hierarchy if hasattr(series, "hierarchy") else None
        metadata = series.metadata if hasattr(series, "metadata") else None
        actual_ts = TimeSeries.from_times_and_values(
            time_index,
            np.vstack(actual_values),
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
        )
        pred_ts = TimeSeries.from_times_and_values(
            time_index,
            np.vstack(pred_values),
            static_covariates=static_covariates,
            hierarchy=hierarchy,
            metadata=metadata,
        )
    return actual_ts.astype(np.float32), pred_ts.astype(np.float32)


def _log_unavailable_foundation_validation(
    tracker, model_name, params, elapsed, reason
):
    print(f"SKIP {model_name} validation: {reason}")
    tracker.log(
        model_name,
        float("inf"),
        0,
        elapsed,
        elapsed,
        params,
        1,
        validation_artifact=None,
        selection_basis="validation_unavailable",
    )

def run_foundation_models(tracker, train, test, freq):
    """
    Runs foundation models with rolling validation on the TRAIN set.
    """
    is_multiseries = isinstance(train, list)
    dataset_config = get_current_dataset_config() or {}
    stride = dataset_config.get("seasonal_period", 1)
    start_ratio = dataset_config.get("cv_start_ratio", 0.7)

    # Determine horizon length (for split)
    if is_multiseries:
        horizon = len(test[0])
    else:
        horizon = len(test)

    # Define Foundation Models to Run
    models_to_run = ["Chronos2", "GraniteTTM"] 

    # 1. Local Foundation Models (Chronos, TTM)
    if is_multiseries:
        min_len = min(len(s) for s in train)
    else:
        min_len = len(train)

    # Dynamic adjustment for validation: keep the horizon fixed, but cap context so that
    # rolling windows remain feasible even on short yearly datasets.
    safe_output_chunk = horizon
    max_possible_input = min_len - safe_output_chunk
    earliest_context = max(1, int(np.floor(min_len * start_ratio)))
    validation_max_input = max(1, earliest_context - safe_output_chunk)
    safe_input_chunk = min(512, max_possible_input, validation_max_input)
    
    if safe_input_chunk < 1:
        safe_input_chunk = 1

    override_params = {
        "input_chunk_length": safe_input_chunk,
        "output_chunk_length": safe_output_chunk
    }

    for model_name in models_to_run:
        try:
            start = time.time()
            model = _load_foundation_model(model_name, params=override_params)
            
            if model is None:
                continue

            def predict_fn(context, target_window):
                if model_name.startswith("Chronos"):
                    model.fit(context)
                pred_ts = model.predict(n=horizon, series=context)
                return TimeSeries.from_times_and_values(
                    target_window.time_index,
                    pred_ts.values(copy=False)[: len(target_window)],
                )

            try:
                if is_multiseries:
                    all_pred, all_actual = [], []
                    for train_s in train:
                        actual_ts, pred_ts = _rolling_last_point_validation(
                            train_s,
                            horizon=horizon,
                            stride=stride,
                            start_ratio=start_ratio,
                            predict_fn=predict_fn,
                            min_context_points=getattr(
                                model,
                                "min_train_series_length",
                                safe_input_chunk + safe_output_chunk,
                            ),
                        )
                        all_actual.append(actual_ts)
                        all_pred.append(pred_ts)

                    rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
                    mape_val = 0  # Ignore MAPE for multiseries
                else:
                    all_actual, all_pred = _rolling_last_point_validation(
                        train,
                        horizon=horizon,
                        stride=stride,
                        start_ratio=start_ratio,
                        predict_fn=predict_fn,
                        min_context_points=getattr(
                            model,
                            "min_train_series_length",
                            safe_input_chunk + safe_output_chunk,
                        ),
                    )
                    rmse_val, mape_val = rmse(all_actual, all_pred), mape(all_actual, all_pred)
            except ValueError as e:
                dur = time.time() - start
                _log_unavailable_foundation_validation(
                    tracker,
                    model_name,
                    {"model": model.model_name},
                    dur,
                    str(e),
                )
                continue

            dur = time.time() - start
            tracker.log(
                model_name,
                rmse_val,
                mape_val,
                dur,
                dur,
                {"model": model.model_name}, # Log specific sub-model name
                1,
                validation_artifact={
                    "actual": all_actual,
                    "prediction": all_pred,
                    "forecast_horizon": horizon,
                    "stride": stride,
                    "source": "rolling_validation_foundation",
                    "last_points_only": True,
                },
            )
        except Exception as e:
            print(f"ERROR {model_name}: {e}")

    # 2. TimeGPT
    if TIMEGPT_AVAILABLE and NIXTLA_API_KEY:
        try:
            start = time.time()
            client = NixtlaClient(api_key=NIXTLA_API_KEY)

            def predict_fn(context, target_window):
                df = pd.DataFrame({"ds": context.time_index, "y": context.values().flatten()})
                fc_df = client.forecast(df=df, h=horizon, model="timegpt-1", freq=freq)
                return TimeSeries.from_times_and_values(
                    target_window.time_index,
                    fc_df["TimeGPT"].values[: len(target_window)],
                )

            try:
                if is_multiseries:
                    all_pred, all_actual = [], []
                    for train_s in train:
                        actual_ts, pred_ts = _rolling_last_point_validation(
                            train_s,
                            horizon=horizon,
                            stride=stride,
                            start_ratio=start_ratio,
                            predict_fn=predict_fn,
                        )
                        all_actual.append(actual_ts)
                        all_pred.append(pred_ts)

                    rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
                    mape_val = 0
                else:
                    all_actual, all_pred = _rolling_last_point_validation(
                        train,
                        horizon=horizon,
                        stride=stride,
                        start_ratio=start_ratio,
                        predict_fn=predict_fn,
                    )
                    rmse_val, mape_val = rmse(all_actual, all_pred), mape(all_actual, all_pred)
            except ValueError as e:
                dur = time.time() - start
                _log_unavailable_foundation_validation(
                    tracker,
                    "TimeGPT",
                    {"model": "timegpt-1"},
                    dur,
                    str(e),
                )
                return

            dur = time.time() - start
            tracker.log(
                "TimeGPT",
                rmse_val,
                mape_val,
                dur,
                dur,
                {"model": "timegpt-1"},
                1,
                validation_artifact={
                    "actual": all_actual,
                    "prediction": all_pred,
                    "forecast_horizon": horizon,
                    "stride": stride,
                    "source": "rolling_validation_foundation",
                    "last_points_only": True,
                },
            )
        except Exception as e:
            print(f"ERROR TimeGPT: {e}")


def get_final_predictions(
    tracker,
    train,
    test,
    scaler,
    train_scaled,
    freq,
    future_covs=None,
    train_future_covs_scaled=None,
    train_past_covs_scaled=None,
    all_future_covs_scaled=None,
    all_past_covs_scaled=None,
    models_to_predict=None,
):
    """
    Retrains and predicts using specific models.
    """

    results_df = tracker.get_results_df()
    if results_df.empty:
        return {}

    is_multiseries = isinstance(train, list)
    predictions = {}

    # Define Categories
    cat_stat = ["ARIMA", "AutoARIMA", "Prophet", "Holt-Winters", "ExponentialSmoothing", "BATS", "Theta", "4Theta"]
    cat_dl = ["N-BEATS", "N-HiTS", "TFT", "TiDE", "Transformer", "RNN", "BlockRNN"]
    cat_foundation = ["Chronos", "Chronos2", "TimeGPT", "GraniteTTM"]

    def _get_category(model_name):
        clean = clean_model_name(model_name)
        if clean in cat_stat: return "statistical"
        if clean in cat_dl: return "dl"
        if clean in cat_foundation: return "foundation"
        return "other"

    def _predict(model_name, params):
        # 1. Foundation Models (New Unified Logic)
        if model_name in ["Chronos", "Chronos2", "GraniteTTM"]:
            try:
                # Map legacy "Chronos" to "Chronos2" if needed
                if model_name == "Chronos":
                    target_name = "Chronos2"
                else:
                    target_name = model_name

                # Dynamic length adjustment
                if is_multiseries:
                    min_len = min(len(s) for s in train)
                    horizon_local = len(test[0])
                else:
                    min_len = len(train)
                    horizon_local = len(test)

                # Dynamic adjustment: Match horizon, maximize context
                safe_output = horizon_local
                max_possible = min_len - safe_output
                safe_input = min(512, max_possible)
                if safe_input < 1: safe_input = 1
                
                # Merge with existing params, favoring dynamic calculation for lengths
                override = params.copy() if params else {}
                override["input_chunk_length"] = safe_input
                override["output_chunk_length"] = safe_output
                
                model = _load_foundation_model(target_name, override)
                
                if model:
                    if is_multiseries:
                        preds = []
                        for train_s, test_s in zip(train, test):
                            # Predict horizon = len(test_s)
                            if target_name.startswith("Chronos"): # Use target_name for mapped name
                                model.fit(train_s)
                            preds.append(model.predict(n=len(test_s), series=train_s))
                        return preds
                    else:
                        if target_name.startswith("Chronos"):
                            model.fit(train)
                        return model.predict(n=len(test), series=train)
            except Exception as e:
                print(f"Error predicting {model_name}: {e}")
                return None
            return None

        if model_name == "TimeGPT" and TIMEGPT_AVAILABLE:
            try:
                client = NixtlaClient(api_key=NIXTLA_API_KEY)
                if is_multiseries:
                    combined_df = []
                    for i, train_s in enumerate(train):
                        df_s = pd.DataFrame(
                            {"ds": train_s.time_index, "y": train_s.values().flatten()}
                        )
                        df_s["unique_id"] = f"series_{i}"
                        combined_df.append(df_s)
                    combined_df = pd.concat(combined_df, ignore_index=True)
                    fc_df = client.forecast(
                        df=combined_df, h=len(test[0]), model="timegpt-1", freq=freq
                    )

                    preds = []
                    for i, test_s in enumerate(test):
                        pred_vals = fc_df[fc_df["unique_id"] == f"series_{i}"][
                            "TimeGPT"
                        ].values
                        preds.append(
                            TimeSeries.from_times_and_values(
                                test_s.time_index, pred_vals
                            )
                        )
                    return preds
                else:
                    df = pd.DataFrame(
                        {"ds": train.time_index, "y": train.values().flatten()}
                    )
                    fc_df = client.forecast(
                        df=df, h=len(test), model="timegpt-1", freq=freq
                    )
                    return TimeSeries.from_times_and_values(
                        test.time_index, fc_df["TimeGPT"].values
                    )
            except Exception as e:
                print(f"Error predicting TimeGPT: {e}")
                return None

        # 2. Darts Models
        clean_name = clean_model_name(model_name)
        cls_map = {
            "Holt-Winters": ExponentialSmoothing,
            "AutoARIMA": AutoARIMA,
            "Prophet": Prophet,
            "TiDE": TiDEModel,
            "N-BEATS": NBEATSModel,
            "TFT": TFTModel,
        }

        if clean_name not in cls_map:
            return None

        try:
            transform_name = get_target_transform_name(params)
            clean_params = strip_internal_params(params)

            # --- LOCAL TRAINING ---
            if not is_multiseries or "LOCAL" in model_name:
                if is_multiseries:
                    preds = []
                    for i, (train_s, test_s) in enumerate(zip(train, test)):
                        target_transform = build_target_transform(transform_name).fit(train_s)
                        train_transformed = target_transform.transform_series(train_s)
                        model = cls_map[clean_name](**clean_params)
                        # Handle Covariates for Local
                        cov_args = {}
                        if clean_name in ["AutoARIMA", "Prophet"] and future_covs:
                            cov_args["future_covariates"] = future_covs[i]

                        model.fit(train_transformed, **cov_args)
                        pred_transformed = model.predict(len(test_s), **cov_args)
                        preds.append(target_transform.inverse_series(pred_transformed))
                    return preds
                else:
                    # Single Series
                    model = cls_map[clean_name](**clean_params)
                    is_dl = clean_name in ["TiDE", "N-BEATS", "TFT"]
                    if is_dl:
                        model.fit(train_scaled, verbose=False)
                        return scaler.inverse_transform(model.predict(len(test)))
                    else:
                        target_transform = build_target_transform(transform_name).fit(train)
                        train_transformed = target_transform.transform_series(train)
                        model.fit(train_transformed)
                        pred_transformed = model.predict(len(test))
                        return target_transform.inverse_series(pred_transformed)

            # --- GLOBAL TRAINING ---
            else:
                model = cls_map[clean_name](**params)
                uses_cov = clean_name != "N-BEATS"

                model.fit(
                    series=train_scaled,
                    future_covariates=train_future_covs_scaled if uses_cov else None,
                    past_covariates=train_past_covs_scaled if uses_cov else None,
                    verbose=False,
                )

                preds_list = []
                for i, test_s in enumerate(test):
                    p = model.predict(
                        n=len(test_s),
                        series=train_scaled[i],
                        future_covariates=(
                            all_future_covs_scaled[i]
                            if (uses_cov and all_future_covs_scaled)
                            else None
                        ),
                        past_covariates=(
                            all_past_covs_scaled[i]
                            if (uses_cov and all_past_covs_scaled)
                            else None
                        ),
                    )
                    preds_list.append(scaler.inverse_transform(p))
                return preds_list

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            return None

    # --- EXECUTION ---
    # Helper to calculate metrics
    def _calc_metrics(pred):
        if is_multiseries:
            rmse_val = np.mean([rmse(test[i], pred[i]) for i in range(len(test))])
            mape_val = 0
        else:
            rmse_val = rmse(test, pred)
            mape_val = mape(test, pred)
        return rmse_val, mape_val

    # If explicit models are requested
    if models_to_predict:
        calculated_preds = []
        for model_name in models_to_predict:
            row = results_df[results_df["Model"] == model_name]
            if not row.empty:
                params = row.iloc[0]["Params"]
                tuning_time = row.iloc[0]["Tuning Time (s)"]
                print(f"Retraining {model_name}...")
                p = _predict(model_name, params)
                if p:
                    r, m = _calc_metrics(p)
                    category = _get_category(model_name)
                    entry = {
                        "model": model_name,
                        "prediction": p,
                        "rmse": r,
                        "mape": m,
                        "tuning_time": tuning_time,
                        "category": category
                    }
                    predictions[model_name] = entry
                    calculated_preds.append(entry)
        
        # Identify Best and Fastest and Category Winners
        if calculated_preds:
            # 1. Best RMSE (Overall)
            best_entry = sorted(calculated_preds, key=lambda x: x["rmse"])[0]
            predictions["best_rmse"] = best_entry
            
            # 2. Fastest
            fast_entry = sorted(calculated_preds, key=lambda x: x["tuning_time"])[0]
            predictions["fastest"] = fast_entry

            # 3. Best Statistical
            stats = [x for x in calculated_preds if x["category"] == "statistical"]
            if stats:
                predictions["best_stat"] = sorted(stats, key=lambda x: x["rmse"])[0]

            # 4. Best DL
            dls = [x for x in calculated_preds if x["category"] == "dl"]
            if dls:
                predictions["best_dl"] = sorted(dls, key=lambda x: x["rmse"])[0]

            # 5. Best Foundation
            foundations = [x for x in calculated_preds if x["category"] == "foundation"]
            if foundations:
                predictions["best_foundation"] = sorted(foundations, key=lambda x: x["rmse"])[0]

        return predictions

    # Default Behavior: Best + Fastest (Legacy Fallback)
    best = tracker.get_best_model()
    print(f"Retraining Best: {best['Model']}")
    p_best = _predict(best["Model"], best["Params"])
    if p_best:
        r, m = _calc_metrics(p_best)
        predictions["best_rmse"] = {
            "model": best["Model"],
            "prediction": p_best,
            "rmse": r,
            "mape": m,
            "tuning_time": best["Tuning Time (s)"],
        }
        predictions[best["Model"]] = predictions["best_rmse"]

    fast = tracker.get_fastest_model()
    if fast["Model"] != best["Model"]:
        print(f"Retraining Fastest: {fast['Model']}")
        p_fast = _predict(fast["Model"], fast["Params"])
        if p_fast:
            r, m = _calc_metrics(p_fast)
            predictions["fastest"] = {
                "model": fast["Model"],
                "prediction": p_fast,
                "rmse": r,
                "mape": m,
                "tuning_time": fast["Tuning Time (s)"],
            }
            predictions[fast["Model"]] = predictions["fastest"]
    else:
        predictions["fastest"] = predictions.get("best_rmse")

    return predictions
