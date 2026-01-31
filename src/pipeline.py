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

def run_foundation_models(tracker, train, test, freq):
    """
    Runs Foundation models (ChronosBolt, GraniteTTM, TimeGPT).

    Performs hold-out validation within the TRAIN set.
    """
    is_multiseries = isinstance(train, list)

    # Determine horizon length (for split)
    if is_multiseries:
        horizon = len(test[0])
    else:
        horizon = len(test)

    # --- DATA PREPARATION FOR VALIDATION (INTERNAL TRAIN SPLIT) ---
    if is_multiseries:
        # For each series: cut off end for validation
        val_inputs = [s[:-horizon] for s in train]  # Model input (Context)
        val_targets = [s[-horizon:] for s in train]  # Ground Truth
    else:
        val_inputs = train[:-horizon]
        val_targets = train[-horizon:]

    # Define Foundation Models to Run
    # We map the generic names to our implementation
    models_to_run = ["Chronos2", "GraniteTTM"] 
    # TimeGPT is handled separately due to API nature, but could be unified if wrapper existed.
    # We keep TimeGPT separate as in original code for now, or unified?
    # Original code had TimeGPT separate. We'll keep it separate to avoid breaking Nixtla logic unless requested.

    # 1. Local Foundation Models (Chronos, TTM)
    # Determine available history for config
    if is_multiseries:
        min_len = min(len(s) for s in val_inputs)
    else:
        min_len = len(val_inputs)

    # Dynamic adjustment: Configure based on dataset properties (Horizon & Size)
    # 1. Output Chunk: Should match the forecast horizon we want to evaluate.
    safe_output_chunk = horizon

    # 2. Input Chunk: Maximize context history, bounded by model limit (512) and available data.
    # Constraint: input + output <= series_length
    max_possible_input = min_len - safe_output_chunk
    safe_input_chunk = min(512, max_possible_input)
    
    # Ensure at least minimal input (1 sample)
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

            if is_multiseries:
                all_pred, all_actual = [], []
                # Iterate over prepared SPLITS
                for inp, tgt in zip(val_inputs, val_targets):
                    # Zero-shot prediction
                    # ChronosModel/GraniteTTM wrapper expect 'series' in predict for context
                    if model_name.startswith("Chronos"):
                        model.fit(inp)

                    pred_ts = model.predict(n=horizon, series=inp)
                    all_pred.append(pred_ts)
                    all_actual.append(tgt)

                rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
                mape_val = 0  # Ignore MAPE for multiseries
            else:
                # Single series
                if model_name.startswith("Chronos"):
                    model.fit(val_inputs)
                pred_ts = model.predict(n=horizon, series=val_inputs)
                rmse_val, mape_val = rmse(val_targets, pred_ts), mape(val_targets, pred_ts)

            dur = time.time() - start
            tracker.log(
                model_name,
                rmse_val,
                mape_val,
                dur,
                dur,
                {"model": model.model_name}, # Log specific sub-model name
                1,
            )
        except Exception as e:
            print(f"ERROR {model_name}: {e}")

    # 2. TimeGPT (Kept as is, assuming it works)
    if TIMEGPT_AVAILABLE and NIXTLA_API_KEY:
        try:
            start = time.time()
            client = NixtlaClient(api_key=NIXTLA_API_KEY)

            if is_multiseries:
                combined_df = []
                for i, inp in enumerate(val_inputs):
                    df_s = pd.DataFrame(
                        {"ds": inp.time_index, "y": inp.values().flatten()}
                    )
                    df_s["unique_id"] = f"series_{i}"
                    combined_df.append(df_s)
                combined_df = pd.concat(combined_df, ignore_index=True)

                fc_df = client.forecast(
                    df=combined_df, h=horizon, model="timegpt-1", freq=freq
                )

                all_pred, all_actual = [], []
                for i, tgt in enumerate(val_targets):
                    pred_vals = fc_df[fc_df["unique_id"] == f"series_{i}"][
                        "TimeGPT"
                    ].values
                    all_pred.append(
                        TimeSeries.from_times_and_values(tgt.time_index, pred_vals)
                    )
                    all_actual.append(tgt)

                rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
                mape_val = 0
            else:
                df = pd.DataFrame(
                    {"ds": val_inputs.time_index, "y": val_inputs.values().flatten()}
                )
                fc_df = client.forecast(df=df, h=horizon, model="timegpt-1", freq=freq)
                pred_ts = TimeSeries.from_times_and_values(
                    val_targets.time_index, fc_df["TimeGPT"].values
                )
                rmse_val, mape_val = rmse(val_targets, pred_ts), mape(
                    val_targets, pred_ts
                )

            dur = time.time() - start
            tracker.log(
                "TimeGPT", rmse_val, mape_val, dur, dur, {"model": "timegpt-1"}, 1
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
        clean = model_name.replace(" (LOCAL)", "").replace(" (GLOBAL)", "")
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
        clean_name = model_name.replace(" (LOCAL)", "").replace(" (GLOBAL)", "")
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
            # --- LOCAL TRAINING ---
            if not is_multiseries or "LOCAL" in model_name:
                if is_multiseries:
                    preds = []
                    for i, (train_s, test_s) in enumerate(zip(train, test)):
                        model = cls_map[clean_name](**params)
                        # Handle Covariates for Local
                        cov_args = {}
                        if clean_name in ["AutoARIMA", "Prophet"] and future_covs:
                            cov_args["future_covariates"] = future_covs[i]

                        model.fit(train_s, **cov_args)
                        preds.append(model.predict(len(test_s), **cov_args))
                    return preds
                else:
                    # Single Series
                    model = cls_map[clean_name](**params)
                    is_dl = clean_name in ["TiDE", "N-BEATS", "TFT"]
                    if is_dl:
                        model.fit(train_scaled, verbose=False)
                        return scaler.inverse_transform(model.predict(len(test)))
                    else:
                        model.fit(train)
                        return model.predict(len(test))

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