import numpy as np
import pandas as pd
from darts.utils.utils import ModelMode


RAW_TRANSFORM = "raw"
LOG_TRANSFORM = "log"
INTERNAL_PARAM_PREFIX = "__"


class IdentityTransform:
    name = RAW_TRANSFORM

    def fit(self, series):
        return self

    def transform_series(self, series):
        return series

    def inverse_series(self, series):
        return series


class LogTransform:
    name = LOG_TRANSFORM

    def fit(self, series):
        values = _series_values(series)
        if np.any(values <= 0):
            raise ValueError("Log transform requires strictly positive target values.")
        return self

    def transform_series(self, series):
        return series.with_values(np.log(series.values(copy=False)))

    def inverse_series(self, series):
        return series.with_values(np.exp(series.values(copy=False)))


def clean_model_name(model_name: str) -> str:
    clean = model_name
    for suffix in [" (LOCAL)", " (GLOBAL)", " [raw]", " [log]"]:
        clean = clean.replace(suffix, "")
    return clean


def format_model_variant_name(model_name: str, transform_name: str) -> str:
    return f"{model_name} [{transform_name}]"


def strip_internal_params(params):
    return {
        key: value
        for key, value in (params or {}).items()
        if not key.startswith(INTERNAL_PARAM_PREFIX)
    }


def get_target_transform_name(params) -> str:
    if not params:
        return RAW_TRANSFORM
    return params.get("__target_transform__", RAW_TRANSFORM)


def get_base_model_name_from_params(model_name: str, params) -> str:
    if params and "__base_model__" in params:
        return params["__base_model__"]
    return clean_model_name(model_name)


def with_internal_params(params, model_name: str, transform_name: str):
    new_params = dict(params)
    new_params["__target_transform__"] = transform_name
    new_params["__base_model__"] = clean_model_name(model_name)
    return new_params


def build_target_transform(transform_name: str):
    if transform_name == LOG_TRANSFORM:
        return LogTransform()
    return IdentityTransform()


def prepare_statistical_candidate_params(model_name: str, params, transform_name: str):
    clean_name = clean_model_name(model_name)
    clean_params = strip_internal_params(params)

    if transform_name == LOG_TRANSFORM:
        if clean_name == "Holt-Winters":
            if clean_params.get("trend") == ModelMode.MULTIPLICATIVE:
                return None
            if clean_params.get("seasonal") == ModelMode.MULTIPLICATIVE:
                return None

        if clean_name == "Prophet":
            if clean_params.get("seasonality_mode") != "additive":
                return None

    return with_internal_params(clean_params, model_name, transform_name)


def resolve_statistical_transform_candidates(model_name: str, diagnostics=None, config=None):
    diagnostics = diagnostics or {}
    config = config or {}
    clean_name = clean_model_name(model_name)

    supported_models = {"AutoARIMA", "Holt-Winters", "Prophet"}
    if clean_name not in supported_models:
        return [RAW_TRANSFORM]

    policy = config.get("statistical_target_transforms", "auto")
    supports_log = diagnostics.get("supports_log", False)
    recommend_log = diagnostics.get("recommend_log", False)

    if policy == "raw_only":
        return [RAW_TRANSFORM]

    if policy == "allow_log":
        return [RAW_TRANSFORM, LOG_TRANSFORM] if supports_log else [RAW_TRANSFORM]

    if isinstance(policy, (list, tuple, set)):
        requested = [name for name in policy if name in {RAW_TRANSFORM, LOG_TRANSFORM}]
        candidates = [RAW_TRANSFORM]
        if LOG_TRANSFORM in requested and supports_log:
            candidates.append(LOG_TRANSFORM)
        return candidates

    if isinstance(policy, dict):
        allowed_models = set(policy.get("models", list(supported_models)))
        if clean_name not in allowed_models:
            return [RAW_TRANSFORM]

        allow_log = policy.get("allow_log", "auto")
        if allow_log is True and supports_log:
            return [RAW_TRANSFORM, LOG_TRANSFORM]
        if allow_log == "recommended" and recommend_log:
            return [RAW_TRANSFORM, LOG_TRANSFORM]
        return [RAW_TRANSFORM]

    if policy == "auto" and recommend_log:
        return [RAW_TRANSFORM, LOG_TRANSFORM]

    return [RAW_TRANSFORM]


def analyze_series_for_log(series, frequency=None):
    if isinstance(series, list):
        per_series = [_single_series_diagnostics(s, frequency) for s in series]
        supports_log = all(item["supports_log"] for item in per_series)
        median_corr = _median_metric(per_series, "rolling_mean_std_corr")
        median_ratio = _median_metric(per_series, "late_early_ratio")
        median_cv = _median_metric(per_series, "rolling_cv_median")
        recommend_log = supports_log and (
            (np.isfinite(median_corr) and median_corr >= 0.6)
            or (np.isfinite(median_ratio) and median_ratio >= 3.0)
        )
        return {
            "is_multiseries": True,
            "series_count": len(per_series),
            "supports_log": supports_log,
            "recommend_log": recommend_log,
            "rolling_mean_std_corr": median_corr,
            "late_early_ratio": median_ratio,
            "rolling_cv_median": median_cv,
            "reason": _diagnostic_reason(supports_log, recommend_log, median_corr, median_ratio),
            "details": per_series,
        }

    single = _single_series_diagnostics(series, frequency)
    return {
        "is_multiseries": False,
        "series_count": 1,
        **single,
        "reason": _diagnostic_reason(
            single["supports_log"],
            single["recommend_log"],
            single["rolling_mean_std_corr"],
            single["late_early_ratio"],
        ),
    }


def _single_series_diagnostics(series, frequency=None):
    values = _series_values(series)
    n = len(values)
    positives = int(np.sum(values > 0))
    zeros = int(np.sum(values == 0))
    negatives = int(np.sum(values < 0))
    supports_log = positives == n

    segment = max(3, n // 5)
    early_mean = float(np.mean(values[:segment]))
    late_mean = float(np.mean(values[-segment:]))
    late_early_ratio = float(late_mean / early_mean) if early_mean > 0 else np.nan

    window = _window_for_frequency(frequency, n)
    rolling_mean = pd.Series(values).rolling(window).mean().to_numpy()
    rolling_std = pd.Series(values).rolling(window).std().to_numpy()
    rolling_cv = (pd.Series(values).rolling(window).std() / pd.Series(values).rolling(window).mean()).to_numpy()

    rolling_mean_std_corr = _corr_safe(rolling_mean, rolling_std)
    level_absdiff_corr = _corr_safe(values[1:], np.abs(np.diff(values)))

    finite_cv = rolling_cv[np.isfinite(rolling_cv)]
    rolling_cv_median = float(np.median(finite_cv)) if len(finite_cv) else np.nan

    recommend_log = supports_log and (
        (np.isfinite(rolling_mean_std_corr) and rolling_mean_std_corr >= 0.6)
        or (np.isfinite(late_early_ratio) and late_early_ratio >= 3.0)
    )

    return {
        "supports_log": supports_log,
        "recommend_log": recommend_log,
        "n_positive": positives,
        "n_zero": zeros,
        "n_negative": negatives,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "late_early_ratio": late_early_ratio,
        "rolling_mean_std_corr": rolling_mean_std_corr,
        "level_absdiff_corr": level_absdiff_corr,
        "rolling_cv_median": rolling_cv_median,
    }


def _series_values(series):
    return series.values(copy=False).astype(float).flatten()


def _window_for_frequency(frequency, n):
    default_map = {
        "YS": 5,
        "QS": 8,
        "MS": 12,
        "D": 28,
        "H": 168,
    }
    base = default_map.get(frequency, 12)
    if n < 6:
        return max(2, n - 1)
    return max(3, min(base, n // 2))


def _corr_safe(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _median_metric(items, key):
    values = np.array([item.get(key, np.nan) for item in items], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _diagnostic_reason(supports_log, recommend_log, corr_value, ratio_value):
    if not supports_log:
        return "Log disabled: target contains zero or negative values."
    if recommend_log:
        return (
            "Log recommended: positive target with signs of multiplicative scale "
            f"(corr={corr_value:.2f}, ratio={ratio_value:.2f})."
        )
    return (
        "Log not recommended automatically: positive target but weak multiplicative "
        f"signal (corr={corr_value:.2f}, ratio={ratio_value:.2f})."
    )
