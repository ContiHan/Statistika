from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from darts.dataprocessing.transformers import BoxCox
from darts.utils.utils import ModelMode


RAW_TRANSFORM = "raw"
LOG_TRANSFORM = "log"
INTERNAL_PARAM_PREFIX = "__"
BOXCOX_ALPHA = 0.05
BOXCOX_LOG_THRESHOLD = 0.30


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

    def __init__(self):
        # Box-Cox with lambda=0 is the standard log transform and stays invertible in Darts.
        self._transformer = BoxCox(lmbda=0.0)

    def fit(self, series):
        self._transformer.fit(series)
        return self

    def transform_series(self, series):
        return self._transformer.transform(series)

    def inverse_series(self, series):
        return self._transformer.inverse_transform(series)


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

    return [resolve_dataset_statistical_transform(diagnostics, config)]


def resolve_dataset_statistical_transform(diagnostics=None, config=None) -> str:
    diagnostics = diagnostics or {}
    config = config or {}

    supports_log = diagnostics.get("supports_log", False)
    selected_transform = diagnostics.get("selected_transform", RAW_TRANSFORM)
    policy = config.get("statistical_target_transforms", "auto")

    if policy in {"raw_only", RAW_TRANSFORM}:
        return RAW_TRANSFORM

    if policy in {"log_only", LOG_TRANSFORM}:
        return LOG_TRANSFORM if supports_log else RAW_TRANSFORM

    if isinstance(policy, (list, tuple, set)):
        requested = [name for name in policy if name in {RAW_TRANSFORM, LOG_TRANSFORM}]
        if not requested:
            return RAW_TRANSFORM
        if len(requested) == 1:
            chosen = requested[0]
            return chosen if (chosen != LOG_TRANSFORM or supports_log) else RAW_TRANSFORM
        if selected_transform in requested and (selected_transform != LOG_TRANSFORM or supports_log):
            return selected_transform
        return RAW_TRANSFORM

    if isinstance(policy, dict):
        forced = policy.get("transform")
        if forced in {RAW_TRANSFORM, LOG_TRANSFORM}:
            return forced if (forced != LOG_TRANSFORM or supports_log) else RAW_TRANSFORM
        if policy.get("allow_log") is True and supports_log and selected_transform == LOG_TRANSFORM:
            return LOG_TRANSFORM
        return RAW_TRANSFORM

    return selected_transform if (selected_transform != LOG_TRANSFORM or supports_log) else RAW_TRANSFORM


def analyze_series_for_log(series, frequency=None):
    if isinstance(series, list):
        per_series = [_single_series_diagnostics(s) for s in series]
        lambdas = _finite_values([item["boxcox_lambda"] for item in per_series])
        lambda_median = float(np.median(lambdas)) if len(lambdas) else np.nan
        log_votes = sum(item["selected_transform"] == LOG_TRANSFORM for item in per_series)
        supports_log = all(item["supports_log"] for item in per_series)
        selected_transform = (
            LOG_TRANSFORM
            if supports_log and log_votes >= max(1, len(per_series) // 2 + len(per_series) % 2)
            else RAW_TRANSFORM
        )
        return {
            "is_multiseries": True,
            "series_count": len(per_series),
            "supports_log": supports_log,
            "recommend_log": selected_transform == LOG_TRANSFORM,
            "selected_transform": selected_transform,
            "boxcox_lambda": lambda_median,
            "selection_counts": dict(
                Counter(item["selected_transform"] for item in per_series)
            ),
            "reason": _diagnostic_reason(
                supports_log=supports_log,
                selected_transform=selected_transform,
                lambda_value=lambda_median,
            ),
            "details": per_series,
        }

    single = _single_series_diagnostics(series)
    return {
        "is_multiseries": False,
        "series_count": 1,
        **single,
        "recommend_log": single["selected_transform"] == LOG_TRANSFORM,
        "reason": _diagnostic_reason(
            supports_log=single["supports_log"],
            selected_transform=single["selected_transform"],
            lambda_value=single["boxcox_lambda"],
        ),
    }


def _single_series_diagnostics(series):
    values = _series_values(series)
    n = len(values)
    positives = int(np.sum(values > 0))
    zeros = int(np.sum(values == 0))
    negatives = int(np.sum(values < 0))
    supports_log = positives == n

    lambda_value = np.nan
    ci_low = np.nan
    ci_high = np.nan
    lambda_supports_log = False
    lambda_supports_raw = False
    selected_transform = RAW_TRANSFORM
    rule = "nonpositive_values"

    if supports_log:
        _, lambda_value, ci = boxcox(values, lmbda=None, alpha=BOXCOX_ALPHA)
        ci_low, ci_high = float(ci[0]), float(ci[1])
        lambda_supports_log = ci_low <= 0.0 <= ci_high
        lambda_supports_raw = ci_low <= 1.0 <= ci_high
        selected_transform, rule = _choose_transform_from_lambda(
            lambda_value=lambda_value,
            lambda_supports_log=lambda_supports_log,
            lambda_supports_raw=lambda_supports_raw,
        )

    return {
        "supports_log": supports_log,
        "selected_transform": selected_transform,
        "boxcox_lambda": lambda_value,
        "boxcox_ci_low": ci_low,
        "boxcox_ci_high": ci_high,
        "boxcox_supports_log": lambda_supports_log,
        "boxcox_supports_raw": lambda_supports_raw,
        "selection_rule": rule,
        "n_positive": positives,
        "n_zero": zeros,
        "n_negative": negatives,
    }


def _choose_transform_from_lambda(
    lambda_value: float,
    lambda_supports_log: bool,
    lambda_supports_raw: bool,
):
    if lambda_supports_log and not lambda_supports_raw:
        return LOG_TRANSFORM, "boxcox_ci_prefers_log"
    if -BOXCOX_LOG_THRESHOLD <= lambda_value <= BOXCOX_LOG_THRESHOLD:
        return LOG_TRANSFORM, "boxcox_lambda_near_zero"
    return RAW_TRANSFORM, "boxcox_prefers_raw"


def _series_values(series):
    return series.values(copy=False).astype(float).flatten()


def _finite_values(values):
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _diagnostic_reason(supports_log, selected_transform, lambda_value):
    if not supports_log:
        return "Raw selected: target contains zero or negative values."
    if not np.isfinite(lambda_value):
        return "Raw selected: Box-Cox lambda could not be estimated."
    if selected_transform == LOG_TRANSFORM:
        return (
            "Log selected for all statistical models: Box-Cox lambda on the train split "
            f"supports a log-like transform (lambda={lambda_value:.3f})."
        )
    return (
        "Raw selected for all statistical models: Box-Cox lambda on the train split "
        f"does not support a log-like transform strongly enough (lambda={lambda_value:.3f})."
    )


def build_transform_diagnostics_df(
    dataset_name,
    diagnostics,
    train_series=None,
    include_details=False,
):
    if not diagnostics:
        return pd.DataFrame()

    rows = []
    if diagnostics.get("is_multiseries", False):
        rows.append(
            _diagnostic_row(
                dataset_name=dataset_name,
                series_label="ALL",
                diagnostics=diagnostics,
                train_series=train_series,
            )
        )

        if include_details:
            details = diagnostics.get("details", [])
            labels = _infer_series_labels(train_series, len(details))
            lengths = _infer_series_lengths(train_series, len(details))
            for idx, detail in enumerate(details):
                rows.append(
                    _diagnostic_row(
                        dataset_name=dataset_name,
                        series_label=labels[idx],
                        diagnostics=detail,
                        train_length=lengths[idx],
                        series_count=1,
                    )
                )
    else:
        rows.append(
            _diagnostic_row(
                dataset_name=dataset_name,
                series_label="-",
                diagnostics=diagnostics,
                train_series=train_series,
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ordered_cols = [
        "Dataset",
        "Series",
        "Series Count",
        "Train N",
        "Positive Only",
        "Zero Count",
        "Negative Count",
        "Box-Cox Lambda",
        "CI Low",
        "CI High",
        "Supports Log",
        "Supports Raw",
        "Selected Transform",
        "Selection Rule",
        "Reason",
    ]
    return df[[col for col in ordered_cols if col in df.columns]]


def _diagnostic_row(
    dataset_name,
    series_label,
    diagnostics,
    train_series=None,
    train_length=None,
    series_count=None,
):
    if train_length is None:
        lengths = _infer_series_lengths(train_series, 1)
        train_length = lengths[0] if lengths else np.nan

    if series_count is None:
        if isinstance(train_series, list):
            series_count = len(train_series)
        else:
            series_count = diagnostics.get("series_count", 1)

    details = diagnostics.get("details", []) if diagnostics.get("is_multiseries") else []
    positive_only = bool(diagnostics.get("supports_log", False))
    lambda_value = diagnostics.get("boxcox_lambda", np.nan)
    selected_transform = diagnostics.get("selected_transform", RAW_TRANSFORM)
    zero_count = int(diagnostics.get("n_zero", 0))
    negative_count = int(diagnostics.get("n_negative", 0))
    supports_log_boxcox = diagnostics.get("boxcox_supports_log", False)
    supports_raw_boxcox = diagnostics.get("boxcox_supports_raw", False)
    selection_rule = diagnostics.get("selection_rule", diagnostics.get("reason", "-"))

    if details:
        zero_count = int(sum(item.get("n_zero", 0) for item in details))
        negative_count = int(sum(item.get("n_negative", 0) for item in details))
        supports_log_boxcox = all(item.get("boxcox_supports_log", False) for item in details)
        supports_raw_boxcox = all(item.get("boxcox_supports_raw", False) for item in details)
        selection_rule = "aggregate_dataset_decision"

    return {
        "Dataset": dataset_name,
        "Series": series_label,
        "Series Count": series_count,
        "Train N": train_length,
        "Positive Only": positive_only,
        "Zero Count": zero_count,
        "Negative Count": negative_count,
        "Box-Cox Lambda": lambda_value,
        "CI Low": diagnostics.get("boxcox_ci_low", np.nan),
        "CI High": diagnostics.get("boxcox_ci_high", np.nan),
        "Supports Log": bool(supports_log_boxcox)
        if positive_only
        else False,
        "Supports Raw": bool(supports_raw_boxcox)
        if positive_only
        else True,
        "Selected Transform": selected_transform,
        "Selection Rule": selection_rule,
        "Reason": diagnostics.get(
            "reason",
            _diagnostic_reason(positive_only, selected_transform, lambda_value),
        ),
    }


def _infer_series_labels(train_series, expected_count):
    if not isinstance(train_series, list):
        return ["-"]

    labels = []
    for idx, series in enumerate(train_series):
        label = f"series_{idx}"
        try:
            static_covs = getattr(series, "static_covariates", None)
            if static_covs is not None:
                for col in ["unique_id", "id"]:
                    if col in static_covs.columns:
                        label = str(static_covs[col].values[0])
                        break
        except Exception:
            pass
        labels.append(label)

    if len(labels) < expected_count:
        labels.extend(f"series_{i}" for i in range(len(labels), expected_count))
    return labels[:expected_count]


def _infer_series_lengths(train_series, expected_count):
    if isinstance(train_series, list):
        lengths = [len(series) for series in train_series]
    elif train_series is None:
        lengths = []
    else:
        lengths = [len(train_series)]

    if len(lengths) < expected_count:
        lengths.extend([np.nan] * (expected_count - len(lengths)))
    return lengths[:expected_count]
