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
from src.config import (
    CHRONOS_AVAILABLE,
    TIMEGPT_AVAILABLE,
    NIXTLA_API_KEY,
    ChronosPipeline,
    NixtlaClient,
)


def run_foundation_models(tracker, train, test, freq):
    """Spustí Foundation modely (Chronos, TimeGPT)."""
    is_multiseries = isinstance(train, list)

    # 1. Chronos
    if CHRONOS_AVAILABLE and torch:
        try:
            start = time.time()
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="mps" if torch.backends.mps.is_available() else "cpu",
                dtype=torch.float32,
            )

            if is_multiseries:
                all_pred, all_actual = [], []
                for train_s, test_s in zip(train, test):
                    context = torch.tensor(train_s.values().flatten())
                    fc = pipeline.predict(
                        context, prediction_length=len(test_s), num_samples=20
                    )
                    vals = fc.median(dim=1).values.numpy().flatten()[: len(test_s)]
                    all_pred.append(
                        TimeSeries.from_times_and_values(test_s.time_index, vals)
                    )
                    all_actual.append(test_s)
                rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
                mape_val = np.mean([mape(a, p) for a, p in zip(all_actual, all_pred)])
            else:
                context = torch.tensor(train.values().flatten())
                fc = pipeline.predict(
                    context, prediction_length=len(test), num_samples=20
                )
                vals = fc.median(dim=1).values.numpy().flatten()[: len(test)]
                pred_ts = TimeSeries.from_times_and_values(test.time_index, vals)
                rmse_val, mape_val = rmse(test, pred_ts), mape(test, pred_ts)

            dur = time.time() - start
            tracker.log(
                "Chronos",
                rmse_val,
                mape_val,
                dur,
                dur,
                {"model": "chronos-t5-small"},
                1,
            )
        except Exception as e:
            print(f"ERROR Chronos: {e}")

    # 2. TimeGPT
    if TIMEGPT_AVAILABLE and NIXTLA_API_KEY:
        try:
            start = time.time()
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

                all_pred, all_actual = [], []
                for i, test_s in enumerate(test):
                    pred_vals = fc_df[fc_df["unique_id"] == f"series_{i}"][
                        "TimeGPT"
                    ].values
                    all_pred.append(
                        TimeSeries.from_times_and_values(test_s.time_index, pred_vals)
                    )
                    all_actual.append(test_s)
                rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
                mape_val = np.mean([mape(a, p) for a, p in zip(all_actual, all_pred)])
            else:
                df = pd.DataFrame(
                    {"ds": train.time_index, "y": train.values().flatten()}
                )
                fc_df = client.forecast(
                    df=df, h=len(test), model="timegpt-1", freq=freq
                )
                pred_ts = TimeSeries.from_times_and_values(
                    test.time_index, fc_df["TimeGPT"].values
                )
                rmse_val, mape_val = rmse(test, pred_ts), mape(test, pred_ts)

            dur = time.time() - start
            tracker.log(
                "TimeGPT", rmse_val, mape_val, dur, dur, {"model": "timegpt-1"}, 1
            )
        except Exception as e:
            print(f"ERROR TimeGPT: {e}")


def get_final_predictions(tracker, train, test, scaler, train_scaled, freq, **kwargs):
    """Univerzální funkce pro predikce (Single i Multi series)."""
    results_df = tracker.get_results_df()
    if results_df.empty:
        return {}

    is_multiseries = isinstance(train, list)
    predictions = {}

    def _predict(model_name, params):
        # 1. Foundation Models
        if model_name == "Chronos":
            if CHRONOS_AVAILABLE and torch:
                try:
                    pipeline = ChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-small",
                        device_map=(
                            "mps" if torch.backends.mps.is_available() else "cpu"
                        ),
                        dtype=torch.float32,
                    )
                    if is_multiseries:
                        preds = []
                        for train_s, test_s in zip(train, test):
                            ctx = torch.tensor(train_s.values().flatten())
                            fc = pipeline.predict(
                                ctx, prediction_length=len(test_s), num_samples=20
                            )
                            preds.append(
                                TimeSeries.from_times_and_values(
                                    test_s.time_index,
                                    fc.median(dim=1)
                                    .values.numpy()
                                    .flatten()[: len(test_s)],
                                )
                            )
                        return preds
                    else:
                        ctx = torch.tensor(train.values().flatten())
                        fc = pipeline.predict(
                            ctx, prediction_length=len(test), num_samples=20
                        )
                        return TimeSeries.from_times_and_values(
                            test.time_index,
                            fc.median(dim=1).values.numpy().flatten()[: len(test)],
                        )
                except Exception:
                    traceback.print_exc()
                    return None
            return None

        if model_name == "TimeGPT" and TIMEGPT_AVAILABLE:
            try:
                client = NixtlaClient(api_key=NIXTLA_API_KEY)
                if is_multiseries:
                    # (Zjednodušeno - v reálu potřeba DataFrame s ID, viz run_foundation_models)
                    return None
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
            except Exception:
                traceback.print_exc()
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
                        model.fit(train_s)
                        preds.append(model.predict(len(test_s)))
                    return preds
                else:
                    # Single Series
                    model = cls_map[clean_name](**params)
                    is_dl = clean_name in ["TiDE", "N-BEATS", "TFT"]
                    if is_dl:
                        model.fit(train_scaled, verbose=False)
                        pred = model.predict(len(test))
                        return scaler.inverse_transform(pred)
                    else:
                        model.fit(train)
                        return model.predict(len(test))

            # --- GLOBAL TRAINING ---
            else:
                # Global logic (simplified for brevity, assumes DL)
                model = cls_map[clean_name](**params)
                model.fit(train_scaled, verbose=False)
                preds = []
                for i, test_s in enumerate(test):
                    p = model.predict(n=len(test_s), series=train_scaled[i])
                    preds.append(scaler.inverse_transform(p))
                return preds

        except Exception:
            traceback.print_exc()
            return None

    # --- EXECUTION ---
    best = tracker.get_best_model()
    p_best = _predict(best["Model"], best["Params"])
    if p_best:
        rmse_val = (
            np.mean([rmse(test[i], p_best[i]) for i in range(len(test))])
            if is_multiseries
            else rmse(test, p_best)
        )
        predictions["best_rmse"] = {
            "model": best["Model"],
            "prediction": p_best,
            "rmse": rmse_val,
            "tuning_time": best["Tuning Time (s)"],
        }

    fast = tracker.get_fastest_model()
    if fast["Model"] != best["Model"]:
        p_fast = _predict(fast["Model"], fast["Params"])
        if p_fast:
            rmse_val = (
                np.mean([rmse(test[i], p_fast[i]) for i in range(len(test))])
                if is_multiseries
                else rmse(test, p_fast)
            )
            predictions["fastest"] = {
                "model": fast["Model"],
                "prediction": p_fast,
                "rmse": rmse_val,
                "tuning_time": fast["Tuning Time (s)"],
            }
    else:
        predictions["fastest"] = predictions.get("best_rmse")

    return predictions
