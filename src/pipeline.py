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
    """
    Spustí Foundation modely (Chronos, TimeGPT).

    ZMENA: Provádí hold-out validaci uvnitř TRAIN setu.
    1. Vezme délku testovacího horizontu (h).
    2. Rozdělí TRAIN na:
       - Context (vše kromě posledních h bodů)
       - Validation (posledních h bodů)
    3. Předpovídá Validation část a počítá chybu.
    """
    is_multiseries = isinstance(train, list)

    # Zjištění délky horizontu (pro split)
    if is_multiseries:
        horizon = len(test[0])
    else:
        horizon = len(test)

    # --- PŘÍPRAVA DAT PRO VALIDACI (SPLIT UVNITŘ TRAINU) ---
    if is_multiseries:
        # Pro každou sérii: uříznout konec pro validaci
        val_inputs = [s[:-horizon] for s in train]  # To, co model vidí (Context)
        val_targets = [s[-horizon:] for s in train]  # To, co model hádá (Ground Truth)
    else:
        val_inputs = train[:-horizon]
        val_targets = train[-horizon:]

    # print(
    #     "Val Inputs Length:",
    #     [len(s) for s in val_inputs] if is_multiseries else len(val_inputs),
    # )
    # print(
    #     "Val Targets Length:",
    #     [len(s) for s in val_targets] if is_multiseries else len(val_targets),
    # )

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
                # Iterujeme přes připravené SPLITY, ne přes train/test
                for inp, tgt in zip(val_inputs, val_targets):
                    context = torch.tensor(inp.values().flatten())
                    # Předpovídáme délku 'horizon' (což je délka tgt)
                    fc = pipeline.predict(
                        context, prediction_length=horizon, num_samples=20
                    )
                    vals = fc.median(dim=1).values.numpy().flatten()[:horizon]
                    all_pred.append(
                        TimeSeries.from_times_and_values(tgt.time_index, vals)
                    )
                    all_actual.append(tgt)

                rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
                mape_val = 0  # Ignorujeme MAPE pro multiseries
            else:
                # Single series: Input je val_inputs, porovnáváme s val_targets
                context = torch.tensor(val_inputs.values().flatten())
                fc = pipeline.predict(
                    context, prediction_length=horizon, num_samples=20
                )
                vals = fc.median(dim=1).values.numpy().flatten()[:horizon]
                pred_ts = TimeSeries.from_times_and_values(val_targets.time_index, vals)

                rmse_val, mape_val = rmse(val_targets, pred_ts), mape(
                    val_targets, pred_ts
                )

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
                # Tvoříme DataFrame z val_inputs (zkrácený train)
                for i, inp in enumerate(val_inputs):
                    df_s = pd.DataFrame(
                        {"ds": inp.time_index, "y": inp.values().flatten()}
                    )
                    df_s["unique_id"] = f"series_{i}"
                    combined_df.append(df_s)
                combined_df = pd.concat(combined_df, ignore_index=True)

                # Předpovídáme horizont h
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
                # Single series
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
):

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
                            vals = (
                                fc.median(dim=1).values.numpy().flatten()[: len(test_s)]
                            )
                            preds.append(
                                TimeSeries.from_times_and_values(
                                    test_s.time_index, vals
                                )
                            )
                        return preds
                    else:
                        ctx = torch.tensor(train.values().flatten())
                        fc = pipeline.predict(
                            ctx, prediction_length=len(test), num_samples=20
                        )
                        vals = fc.median(dim=1).values.numpy().flatten()[: len(test)]
                        return TimeSeries.from_times_and_values(test.time_index, vals)
                except Exception as e:
                    print(f"Error predicting Chronos: {e}")
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
    best = tracker.get_best_model()
    print(f"Retraining Best: {best['Model']}")
    p_best = _predict(best["Model"], best["Params"])
    if p_best:
        rmse_val = (
            np.mean([rmse(test[i], p_best[i]) for i in range(len(test))])
            if is_multiseries
            else rmse(test, p_best)
        )
        mape_val = 0 if is_multiseries else mape(test, p_best)
        predictions["best_rmse"] = {
            "model": best["Model"],
            "prediction": p_best,
            "rmse": rmse_val,
            "mape": mape_val,
            "tuning_time": best["Tuning Time (s)"],
        }

    fast = tracker.get_fastest_model()
    if fast["Model"] != best["Model"]:
        print(f"Retraining Fastest: {fast['Model']}")
        p_fast = _predict(fast["Model"], fast["Params"])
        if p_fast:
            rmse_val = (
                np.mean([rmse(test[i], p_fast[i]) for i in range(len(test))])
                if is_multiseries
                else rmse(test, p_fast)
            )
            mape_val = 0 if is_multiseries else mape(test, p_fast)
            predictions["fastest"] = {
                "model": fast["Model"],
                "prediction": p_fast,
                "rmse": rmse_val,
                "mape": mape_val,
                "tuning_time": fast["Tuning Time (s)"],
            }
    else:
        predictions["fastest"] = predictions.get("best_rmse")

    return predictions
