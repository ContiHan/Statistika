from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t
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

from src.config import TIMEGPT_AVAILABLE, NIXTLA_API_KEY, NixtlaClient
from src.pipeline import _load_foundation_model, _rolling_last_point_validation
from src.runtime_context import get_current_dataset_config
from src.timegpt_utils import timegpt_forecast
from src.statistical_transforms import clean_model_name
from src.tuning import (
    _build_validation_artifact,
    evaluate_model,
    evaluate_local_model,
    evaluate_global_model,
)


def diebold_mariano_test(
    target,
    pred1,
    pred2,
    h: int = 1,
    criterion: str = "mse",
):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold correction.
    Accepts TimeSeries, list[TimeSeries], or already aligned numpy-like arrays.
    """
    y = _to_1d_array(target)
    y_hat1 = _to_1d_array(pred1)
    y_hat2 = _to_1d_array(pred2)

    n = len(y)
    if len(y_hat1) != n or len(y_hat2) != n:
        raise ValueError("All inputs must have the same length.")

    e1 = y - y_hat1
    e2 = y - y_hat2

    if criterion.lower() == "mse":
        d = e1**2 - e2**2
    elif criterion.lower() == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Criterion must be 'mse' or 'mae'.")

    d_bar = np.mean(d)

    def autocovariance(xi, k):
        n_xi = len(xi)
        if k == 0:
            return np.var(xi)
        if k >= n_xi:
            return 0.0
        xi_mean = np.mean(xi)
        return np.sum((xi[: n_xi - k] - xi_mean) * (xi[k:] - xi_mean)) / n_xi

    max_lags = min(h, n)
    gamma = np.array([autocovariance(d, i) for i in range(max_lags)])

    if len(gamma) > 1:
        v_d = (gamma[0] + 2 * np.sum(gamma[1:])) / n
    else:
        v_d = gamma[0] / n

    if v_d < 1e-12:
        v_d = np.var(d) / n if np.var(d) > 0 else 1e-12

    dm_stat = d_bar / np.sqrt(v_d)

    if n > (2 * h - 1):
        hln_correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1) / n)) / n)
        dm_stat_corrected = hln_correction * dm_stat
        applied_hln = True
    else:
        dm_stat_corrected = dm_stat
        applied_hln = False

    p_value = 2 * (1 - t.cdf(np.abs(dm_stat_corrected), df=n - 1))

    return {
        "dm_stat": dm_stat_corrected,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "better_model": "Model 1" if dm_stat_corrected < 0 else "Model 2",
        "criterion": criterion,
        "n": n,
        "h": h,
        "hln_applied": applied_hln,
    }


def run_statistical_comparison(tracker, final_predictions=None, test_series=None, h=1):
    """
    Runs Diebold-Mariano comparisons on rolling validation forecasts stored in the tracker.
    Model pairs are selected by validation RMSE, not by final test performance.
    """
    results_tracker = tracker.get_results_df()
    if results_tracker.empty:
        return pd.DataFrame()

    available_models = [
        model_name
        for model_name in results_tracker["Model"]
        if tracker.get_validation_artifact(model_name) is not None
    ]
    if not available_models:
        return pd.DataFrame()

    df_validation = results_tracker[
        results_tracker["Model"].isin(available_models)
    ].copy()

    def get_best_validation(category_substrings):
        subset = df_validation[
            df_validation["Model"].apply(
                lambda x: any(sub in clean_model_name(x) for sub in category_substrings)
            )
        ]
        if subset.empty:
            return None
        return subset.sort_values("RMSE").iloc[0]["Model"]

    dl_models = ["TiDE", "N-BEATS", "TFT"]
    stat_models = ["AutoARIMA", "Prophet", "Holt-Winters"]
    found_models = ["Chronos", "Chronos2", "TimeGPT", "GraniteTTM"]

    comparisons = []

    if len(df_validation) >= 2:
        sorted_models = df_validation.sort_values("RMSE")
        comparisons.append(
            ("Best CV vs 2nd Best CV", sorted_models.iloc[0]["Model"], sorted_models.iloc[1]["Model"])
        )

    bdl = get_best_validation(dl_models)
    bstat = get_best_validation(stat_models)
    bfound = get_best_validation(found_models)

    if bdl and bstat:
        comparisons.append(("DL vs Statistical", bdl, bstat))

    if bfound and bdl:
        comparisons.append(("Foundation vs DL", bfound, bdl))

    if bstat and bfound:
        comparisons.append(("Statistical vs Foundation", bstat, bfound))

    best_cv = df_validation.sort_values("RMSE").iloc[0]["Model"]
    fastest_cv = df_validation.sort_values("Tuning Time (s)").iloc[0]["Model"]
    comparisons.append(("Best CV vs Fastest", best_cv, fastest_cv))

    data = []
    for label, m1, m2 in comparisons:
        if m1 == m2:
            data.append(
                {
                    "Comparison": label,
                    "Model A": m1,
                    "Model B": m2,
                    "DM Stat": 0.0,
                    "P-Value": 1.0,
                    "Significant": "N/A",
                    "Winner": "Same Model",
                }
            )
            continue

        try:
            artifact1 = tracker.get_validation_artifact(m1)
            artifact2 = tracker.get_validation_artifact(m2)
            if artifact1 is None or artifact2 is None:
                continue

            target, pred1, pred2 = _align_validation_artifacts(artifact1, artifact2)
            if len(target) < 3:
                continue

            effective_h = max(
                int(artifact1.get("forecast_horizon", h) or h),
                int(artifact2.get("forecast_horizon", h) or h),
            )

            if np.allclose(pred1, pred2):
                continue

            res = diebold_mariano_test(target, pred1, pred2, h=effective_h)
            winner = m1 if res["better_model"] == "Model 1" else m2

            data.append(
                {
                    "Comparison": label,
                    "Model A": m1,
                    "Model B": m2,
                    "DM Stat": res["dm_stat"],
                    "P-Value": res["p_value"],
                    "Significant": "Yes" if res["is_significant"] else "No",
                    "Winner": winner if res["is_significant"] else "-",
                }
            )
        except Exception as e:
            print(f"Error comparing {m1} vs {m2}: {e}")

    return pd.DataFrame(data)


def _to_1d_array(data):
    if isinstance(data, TimeSeries):
        return data.values(copy=False).astype(float).flatten()
    if isinstance(data, np.ndarray):
        return data.astype(float).flatten()
    if isinstance(data, (list, tuple)):
        if not data:
            return np.array([], dtype=float)
        if isinstance(data[0], TimeSeries):
            return np.concatenate([series.values(copy=False).astype(float).flatten() for series in data])
        return np.asarray(data, dtype=float).flatten()
    if hasattr(data, "to_numpy"):
        return np.asarray(data.to_numpy(), dtype=float).flatten()
    return np.asarray(data, dtype=float).flatten()


def _align_validation_artifacts(artifact1, artifact2):
    df1 = _artifact_to_frame(artifact1).rename(
        columns={"actual": "actual_a", "prediction": "prediction_a"}
    )
    df2 = _artifact_to_frame(artifact2).rename(
        columns={"actual": "actual_b", "prediction": "prediction_b"}
    )

    merged = df1.merge(df2, on=["series_id", "time"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping validation points between the compared models.")

    if not np.allclose(merged["actual_a"].to_numpy(), merged["actual_b"].to_numpy()):
        raise ValueError("Validation targets are misaligned across compared models.")

    target = merged["actual_a"].to_numpy(dtype=float)
    pred1 = merged["prediction_a"].to_numpy(dtype=float)
    pred2 = merged["prediction_b"].to_numpy(dtype=float)
    return target, pred1, pred2


def _artifact_to_frame(artifact):
    actual = artifact["actual"]
    prediction = artifact["prediction"]

    if isinstance(actual, list):
        frames = []
        for idx, (actual_s, pred_s) in enumerate(zip(actual, prediction)):
            frames.append(_single_series_frame(actual_s, pred_s, idx))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return _single_series_frame(actual, prediction, 0)


def _single_series_frame(actual, prediction, fallback_idx):
    actual_aligned = actual.slice_intersect(prediction)
    pred_aligned = prediction.slice_intersect(actual)
    series_id = _series_identifier(actual, fallback_idx)
    return pd.DataFrame(
        {
            "series_id": series_id,
            "time": actual_aligned.time_index,
            "actual": actual_aligned.values(copy=False).flatten(),
            "prediction": pred_aligned.values(copy=False).flatten(),
        }
    )


def _series_identifier(series, fallback_idx):
    if hasattr(series, "static_covariates") and series.static_covariates is not None:
        cols = series.static_covariates.columns
        if "unique_id" in cols:
            return str(series.static_covariates["unique_id"].values[0])
        if "id" in cols:
            return str(series.static_covariates["id"].values[0])
    return f"series_{fallback_idx}"


_STAT_MODEL_MAP = {
    "Holt-Winters": ExponentialSmoothing,
    "AutoARIMA": AutoARIMA,
    "Prophet": Prophet,
}
_DL_MODEL_MAP = {
    "TiDE": TiDEModel,
    "N-BEATS": NBEATSModel,
    "TFT": TFTModel,
}
_FOUNDATION_MODELS = {"Chronos", "Chronos2", "GraniteTTM", "TimeGPT"}


def select_dm_shortlist_models(
    tracker,
    include_second_best=True,
    include_fastest=False,
    include_foundation=True,
):
    shortlist, _ = _select_dm_shortlist_with_reasons(
        tracker,
        include_second_best=include_second_best,
        include_fastest=include_fastest,
        include_foundation=include_foundation,
    )
    return shortlist


def run_pairwise_dm_analysis(
    tracker,
    train,
    freq,
    scaler=None,
    train_scaled=None,
    future_covs_list=None,
    train_future_covs_scaled=None,
    train_past_covs_scaled=None,
    models_to_compare=None,
    forecast_horizon=1,
    stride=1,
    cv_start_ratio=None,
    dm_h=None,
    include_second_best=True,
    include_fastest=False,
    include_foundation=True,
    criterion="mse",
    adjust_pvalues=True,
):
    shortlist_reasons = {}
    if models_to_compare is None:
        models_to_compare, shortlist_reasons = _select_dm_shortlist_with_reasons(
            tracker,
            include_second_best=include_second_best,
            include_fastest=include_fastest,
            include_foundation=include_foundation,
        )
    else:
        shortlist_reasons = {model_name: "Manually selected" for model_name in models_to_compare}

    if not models_to_compare:
        return {
            "shortlist": [],
            "shortlist_reasons": {},
            "backtest_summary": pd.DataFrame(),
            "pairwise_results": pd.DataFrame(),
            "artifacts": {},
        }

    backtest_summary, artifacts = _run_dm_backtests(
        tracker=tracker,
        train=train,
        freq=freq,
        scaler=scaler,
        train_scaled=train_scaled,
        future_covs_list=future_covs_list,
        train_future_covs_scaled=train_future_covs_scaled,
        train_past_covs_scaled=train_past_covs_scaled,
        models_to_compare=models_to_compare,
        forecast_horizon=forecast_horizon,
        stride=stride,
        cv_start_ratio=cv_start_ratio,
        shortlist_reasons=shortlist_reasons,
    )

    effective_dm_h = forecast_horizon if dm_h is None else dm_h

    pairwise_results = _pairwise_dm_from_artifacts(
        artifacts=artifacts,
        criterion=criterion,
        h=effective_dm_h,
        adjust_pvalues=adjust_pvalues,
    )
    return {
        "shortlist": models_to_compare,
        "shortlist_reasons": shortlist_reasons,
        "backtest_summary": backtest_summary,
        "pairwise_results": pairwise_results,
        "artifacts": artifacts,
    }


def _finite_tracker_results(tracker):
    results = tracker.get_results_df().copy()
    if results.empty:
        return results

    results["Category"] = results["Model"].apply(_get_model_category)
    results = results[np.isfinite(results["RMSE"])]
    results = results[results["Category"] != "other"]
    return results


def _get_model_category(model_name):
    clean = clean_model_name(model_name)
    if clean in _STAT_MODEL_MAP:
        return "statistical"
    if clean in _DL_MODEL_MAP:
        return "dl"
    if clean in _FOUNDATION_MODELS:
        return "foundation"
    return "other"


def _select_dm_shortlist_with_reasons(
    tracker,
    include_second_best=True,
    include_fastest=False,
    include_foundation=True,
):
    results = _finite_tracker_results(tracker)
    if results.empty:
        return [], {}

    candidates = []
    sorted_results = results.sort_values("RMSE")
    candidates.append(
        (
            sorted_results.iloc[0]["Model"],
            "Best overall validation RMSE",
        )
    )

    if include_second_best and len(sorted_results) > 1:
        candidates.append(
            (
                sorted_results.iloc[1]["Model"],
                "Second-best overall validation RMSE",
            )
        )

    category_labels = {
        "statistical": "Best statistical validation RMSE",
        "dl": "Best deep learning validation RMSE",
        "foundation": "Best foundation validation RMSE",
    }
    for category in ["statistical", "dl", "foundation"]:
        if category == "foundation" and not include_foundation:
            continue
        subset = results[results["Category"] == category]
        if not subset.empty:
            candidates.append(
                (
                    subset.sort_values("RMSE").iloc[0]["Model"],
                    category_labels[category],
                )
            )

    if include_fastest:
        candidates.append(
            (
                results.sort_values("Tuning Time (s)").iloc[0]["Model"],
                "Fastest tuning time",
            )
        )

    deduped = []
    reasons = {}
    seen = set()
    for model_name, reason in candidates:
        if model_name not in seen:
            deduped.append(model_name)
            seen.add(model_name)
            reasons[model_name] = [reason]
        else:
            reasons.setdefault(model_name, []).append(reason)

    reason_map = {
        model_name: "; ".join(reason_list)
        for model_name, reason_list in reasons.items()
    }
    return deduped, reason_map


def _run_dm_backtests(
    tracker,
    train,
    freq,
    scaler,
    train_scaled,
    future_covs_list,
    train_future_covs_scaled,
    train_past_covs_scaled,
    models_to_compare,
    forecast_horizon,
    stride,
    cv_start_ratio,
    shortlist_reasons,
):
    results_df = tracker.get_results_df().copy()
    if results_df.empty:
        return pd.DataFrame(), {}

    start_ratio = _resolve_cv_start_ratio(cv_start_ratio)
    artifacts = {}
    summary_rows = []

    for model_name in models_to_compare:
        row = results_df[results_df["Model"] == model_name]
        if row.empty:
            continue

        params = row.iloc[0]["Params"]
        category = _get_model_category(model_name)
        clean_name = clean_model_name(model_name)

        try:
            if category == "statistical":
                result = _backtest_statistical_model(
                    model_name=model_name,
                    clean_name=clean_name,
                    params=params,
                    train=train,
                    future_covs_list=future_covs_list,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    cv_start_ratio=start_ratio,
                )
            elif category == "dl":
                result = _backtest_dl_model(
                    model_name=model_name,
                    clean_name=clean_name,
                    params=params,
                    train=train,
                    train_scaled=train_scaled,
                    scaler=scaler,
                    train_future_covs_scaled=train_future_covs_scaled,
                    train_past_covs_scaled=train_past_covs_scaled,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    cv_start_ratio=start_ratio,
                )
            elif category == "foundation":
                result = _backtest_foundation_model(
                    model_name=clean_name,
                    train=train,
                    freq=freq,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    cv_start_ratio=start_ratio,
                )
            else:
                continue
        except Exception as e:
            result = {
                "rmse": float("inf"),
                "mape": 0.0,
                "artifact": None,
                "error": str(e),
            }

        artifact = result.get("artifact")
        if artifact is not None:
            artifacts[model_name] = artifact

        summary_rows.append(
            {
                "Model": model_name,
                "Category": category,
                "Included Because": shortlist_reasons.get(model_name, "-"),
                "Backtest RMSE": result.get("rmse", float("inf")),
                "Backtest MAPE": result.get("mape", 0.0),
                "Backtest Points": _artifact_point_count(artifact),
                "Forecast Horizon": forecast_horizon,
                "Stride": stride,
                "Status": "OK" if artifact is not None else "Skipped",
                "Notes": result.get("error", ""),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["Status", "Backtest RMSE"],
            ascending=[True, True],
        )
    return summary_df, artifacts


def _resolve_cv_start_ratio(cv_start_ratio):
    if cv_start_ratio is not None:
        return cv_start_ratio
    dataset_config = get_current_dataset_config() or {}
    return dataset_config.get("cv_start_ratio", 0.7)


def _backtest_statistical_model(
    model_name,
    clean_name,
    params,
    train,
    future_covs_list,
    forecast_horizon,
    stride,
    cv_start_ratio,
):
    model_cls = _STAT_MODEL_MAP[clean_name]
    is_multiseries = isinstance(train, list)

    if is_multiseries:
        supports_covariates = clean_name in {"AutoARIMA", "Prophet"}
        future_covs = future_covs_list if supports_covariates else None
        rmse_val, mape_val, _, err, artifact = evaluate_local_model(
            model_cls,
            params,
            train,
            future_covs_list=future_covs,
            supports_covariates=supports_covariates,
            test_periods=forecast_horizon,
            stride=stride,
            cv_start_ratio=cv_start_ratio,
        )
    else:
        rmse_val, mape_val, _, err, artifact = evaluate_model(
            model_cls,
            params,
            train,
            test_periods=forecast_horizon,
            is_dl=False,
            cv_start_ratio=cv_start_ratio,
            stride=stride,
        )

    return {"rmse": rmse_val, "mape": mape_val, "artifact": artifact, "error": err}


def _backtest_dl_model(
    model_name,
    clean_name,
    params,
    train,
    train_scaled,
    scaler,
    train_future_covs_scaled,
    train_past_covs_scaled,
    forecast_horizon,
    stride,
    cv_start_ratio,
):
    model_cls = _DL_MODEL_MAP[clean_name]
    is_multiseries = isinstance(train, list)

    if is_multiseries:
        uses_covariates = clean_name != "N-BEATS"
        rmse_val, mape_val, _, err, artifact = evaluate_global_model(
            model_cls,
            params,
            train_scaled,
            train,
            scaler,
            future_covs=train_future_covs_scaled if uses_covariates else None,
            past_covs=train_past_covs_scaled if uses_covariates else None,
            stride=stride,
            cv_start_ratio=cv_start_ratio,
            test_periods=forecast_horizon,
        )
    else:
        rmse_val, mape_val, _, err, artifact = evaluate_model(
            model_cls,
            params,
            train_scaled,
            test_periods=forecast_horizon,
            is_dl=True,
            scaler=scaler,
            original_train=train,
            cv_start_ratio=cv_start_ratio,
            stride=stride,
        )

    return {"rmse": rmse_val, "mape": mape_val, "artifact": artifact, "error": err}


def _backtest_foundation_model(
    model_name,
    train,
    freq,
    forecast_horizon,
    stride,
    cv_start_ratio,
):
    is_multiseries = isinstance(train, list)
    target_name = "Chronos2" if model_name == "Chronos" else model_name

    if target_name == "TimeGPT":
        return _backtest_timegpt(
            train=train,
            freq=freq,
            forecast_horizon=forecast_horizon,
            stride=stride,
            cv_start_ratio=cv_start_ratio,
        )

    min_len = min(len(s) for s in train) if is_multiseries else len(train)
    max_possible_input = max(1, min_len - forecast_horizon)
    earliest_context = max(1, int(np.floor(min_len * cv_start_ratio)))
    validation_max_input = max(1, earliest_context - forecast_horizon)
    safe_input = min(512, max_possible_input, validation_max_input)
    override_params = {
        "input_chunk_length": safe_input,
        "output_chunk_length": forecast_horizon,
    }

    model = _load_foundation_model(target_name, params=override_params)
    if model is None:
        return {
            "rmse": float("inf"),
            "mape": 0.0,
            "artifact": None,
            "error": f"{target_name} is not available in this environment.",
        }

    def predict_fn(context, target_window):
        if target_name.startswith("Chronos"):
            model.fit(context)
        pred_ts = model.predict(n=forecast_horizon, series=context)
        return TimeSeries.from_times_and_values(
            target_window.time_index,
            pred_ts.values(copy=False)[: len(target_window)],
        )

    try:
        if is_multiseries:
            all_actual, all_pred = [], []
            for train_s in train:
                actual_ts, pred_ts = _rolling_last_point_validation(
                    train_s,
                    horizon=forecast_horizon,
                    stride=stride,
                    start_ratio=cv_start_ratio,
                    predict_fn=predict_fn,
                    min_context_points=getattr(
                        model,
                        "min_train_series_length",
                        safe_input + forecast_horizon,
                    ),
                )
                all_actual.append(actual_ts)
                all_pred.append(pred_ts)

            rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
            mape_val = 0.0
        else:
            all_actual, all_pred = _rolling_last_point_validation(
                train,
                horizon=forecast_horizon,
                stride=stride,
                start_ratio=cv_start_ratio,
                predict_fn=predict_fn,
                min_context_points=getattr(
                    model,
                    "min_train_series_length",
                    safe_input + forecast_horizon,
                ),
            )
            rmse_val = rmse(all_actual, all_pred)
            mape_val = mape(all_actual, all_pred)

        artifact = _build_validation_artifact(
            actual=all_actual,
            prediction=all_pred,
            forecast_horizon=forecast_horizon,
            stride=stride,
            source="dm_backtest_foundation",
        )
        return {"rmse": rmse_val, "mape": mape_val, "artifact": artifact, "error": None}
    except Exception as e:
        return {
            "rmse": float("inf"),
            "mape": 0.0,
            "artifact": None,
            "error": str(e),
        }


def _backtest_timegpt(
    train,
    freq,
    forecast_horizon,
    stride,
    cv_start_ratio,
):
    if not (TIMEGPT_AVAILABLE and NIXTLA_API_KEY):
        return {
            "rmse": float("inf"),
            "mape": 0.0,
            "artifact": None,
            "error": "TimeGPT is not available in this environment.",
        }

    client = NixtlaClient(api_key=NIXTLA_API_KEY)
    is_multiseries = isinstance(train, list)

    def predict_fn(context, target_window):
        df = pd.DataFrame({"ds": context.time_index, "y": context.values().flatten()})
        fc_df = timegpt_forecast(
            client,
            df=df,
            h=forecast_horizon,
            model="timegpt-1",
            freq=freq,
        )
        return TimeSeries.from_times_and_values(
            target_window.time_index,
            fc_df["TimeGPT"].values[: len(target_window)],
        )

    try:
        if is_multiseries:
            all_actual, all_pred = [], []
            for train_s in train:
                actual_ts, pred_ts = _rolling_last_point_validation(
                    train_s,
                    horizon=forecast_horizon,
                    stride=stride,
                    start_ratio=cv_start_ratio,
                    predict_fn=predict_fn,
                )
                all_actual.append(actual_ts)
                all_pred.append(pred_ts)

            rmse_val = np.mean([rmse(a, p) for a, p in zip(all_actual, all_pred)])
            mape_val = 0.0
        else:
            all_actual, all_pred = _rolling_last_point_validation(
                train,
                horizon=forecast_horizon,
                stride=stride,
                start_ratio=cv_start_ratio,
                predict_fn=predict_fn,
            )
            rmse_val = rmse(all_actual, all_pred)
            mape_val = mape(all_actual, all_pred)

        artifact = _build_validation_artifact(
            actual=all_actual,
            prediction=all_pred,
            forecast_horizon=forecast_horizon,
            stride=stride,
            source="dm_backtest_foundation",
        )
        return {"rmse": rmse_val, "mape": mape_val, "artifact": artifact, "error": None}
    except Exception as e:
        return {
            "rmse": float("inf"),
            "mape": 0.0,
            "artifact": None,
            "error": str(e),
        }


def _artifact_point_count(artifact):
    if artifact is None:
        return 0

    actual = artifact.get("actual")
    if isinstance(actual, list):
        return int(sum(len(series) for series in actual))
    if actual is None:
        return 0
    return int(len(actual))


def _pairwise_dm_from_artifacts(artifacts, criterion="mse", h=1, adjust_pvalues=True):
    rows = []
    model_names = sorted(artifacts.keys())
    for model_a, model_b in combinations(model_names, 2):
        try:
            target, pred1, pred2 = _align_validation_artifacts(
                artifacts[model_a],
                artifacts[model_b],
            )
        except ValueError as e:
            print(f"SKIP DM pair {model_a} vs {model_b}: {e}")
            continue
        if len(target) < 3:
            continue

        res = diebold_mariano_test(
            target,
            pred1,
            pred2,
            h=h,
            criterion=criterion,
        )
        mean_loss_diff = _mean_loss_differential(target, pred1, pred2, criterion)
        winner = model_a if res["better_model"] == "Model 1" else model_b
        rows.append(
            {
                "Model A": model_a,
                "Model B": model_b,
                "N": res["n"],
                "H": res["h"],
                "Criterion": criterion,
                "Mean Loss Diff": mean_loss_diff,
                "DM Stat": res["dm_stat"],
                "P-Value": res["p_value"],
                "Significant": "Yes" if res["is_significant"] else "No",
                "Winner": winner if res["is_significant"] else "-",
            }
        )

    pairwise_df = pd.DataFrame(rows)
    if pairwise_df.empty:
        return pairwise_df

    if adjust_pvalues:
        pairwise_df["P-Value Holm"] = _holm_adjust(pairwise_df["P-Value"].to_numpy())
        pairwise_df["Significant Holm"] = np.where(
            pairwise_df["P-Value Holm"] < 0.05,
            "Yes",
            "No",
        )
    ordered_cols = [
        "Model A",
        "Model B",
        "N",
        "H",
        "Criterion",
        "Mean Loss Diff",
        "DM Stat",
        "P-Value Holm",
        "Significant Holm",
        "P-Value",
        "Significant",
        "Winner",
    ]
    ordered_cols = [col for col in ordered_cols if col in pairwise_df.columns]
    pairwise_df = pairwise_df[ordered_cols]
    sort_col = "P-Value Holm" if "P-Value Holm" in pairwise_df.columns else "P-Value"
    return pairwise_df.sort_values([sort_col, "DM Stat"], ascending=[True, False])


def _mean_loss_differential(target, pred1, pred2, criterion):
    y = _to_1d_array(target)
    y_hat1 = _to_1d_array(pred1)
    y_hat2 = _to_1d_array(pred2)
    e1 = y - y_hat1
    e2 = y - y_hat2
    if criterion.lower() == "mae":
        diff = np.abs(e1) - np.abs(e2)
    else:
        diff = e1**2 - e2**2
    return float(np.mean(diff))


def _holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted
