import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries

from src.experiment import build_selected_params_df
from src.statistical_transforms import (
    build_transform_diagnostics_df,
    clean_model_name,
    strip_internal_params,
)


_STAT_MODELS = {"Holt-Winters", "AutoARIMA", "Prophet"}
_DL_MODELS = {"TiDE", "N-BEATS", "TFT"}
_FOUNDATION_MODELS = {"Chronos", "Chronos2", "GraniteTTM", "TimeGPT"}

_ROLE_KEYS = {
    "best_rmse": "best_overall",
    "fastest": "fastest",
    "best_stat": "best_statistical",
    "best_dl": "best_dl",
    "best_foundation": "best_foundation",
}


def export_forecasting_data(
    dataset_name,
    dataset_config,
    tracker,
    train,
    test,
    final_predictions=None,
    dm_analysis=None,
    transform_diagnostics_df=None,
    target_diagnostics=None,
    output_root=None,
    include_optional=True,
    export_tables=True,
    export_point_level=True,
    export_metadata=True,
    export_tracker_results=None,
    export_dm_shortlist=None,
    export_params_long=None,
    export_validation_points=None,
    export_dm_points=None,
):
    """
    Export machine-readable artifacts for one forecasting notebook run.

    By default, exports both the core tables and the optional audit-friendly
    datasets so the results can later be reused for thesis tables or a small
    interactive viewer.
    """
    export_tracker_results = _resolve_optional_flag(
        export_tracker_results,
        include_optional=include_optional,
    )
    export_dm_shortlist = _resolve_optional_flag(
        export_dm_shortlist,
        include_optional=include_optional,
    )
    export_params_long = _resolve_optional_flag(
        export_params_long,
        include_optional=include_optional,
    )
    export_validation_points = _resolve_optional_flag(
        export_validation_points,
        include_optional=include_optional,
    )
    export_dm_points = _resolve_optional_flag(
        export_dm_points,
        include_optional=include_optional,
    )

    dataset_slug = _safe_slug(dataset_name, max_len=80)
    run_dir = _get_run_dir(dataset_slug, output_root=output_root)
    tables_dir = run_dir / "tables"
    series_dir = run_dir / "series"

    tables_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    results_df = tracker.get_results_df().copy()
    transform_df = _resolve_transform_df(
        dataset_name=dataset_name,
        train=train,
        transform_diagnostics_df=transform_diagnostics_df,
        target_diagnostics=target_diagnostics,
    )
    roles_map = _extract_prediction_roles(final_predictions or {})

    if final_predictions is None:
        print(
            "WARNING export_forecasting_data: final_predictions is None. "
            "Test metrics in comparison_metrics.csv will stay empty and "
            "final_forecasts.csv will contain no forecast rows."
        )

    exported_files = []

    if export_tables:
        comparison_df = _build_comparison_metrics_df(
            dataset_name=dataset_name,
            results_df=results_df,
            final_predictions=final_predictions or {},
            roles_map=roles_map,
        )
        exported_files.append(
            _write_csv(comparison_df, tables_dir / "comparison_metrics.csv")
        )

        selected_params_df = build_selected_params_df(tracker)
        exported_files.append(
            _write_csv(selected_params_df, tables_dir / "selected_params.csv")
        )

        if transform_df is not None:
            exported_files.append(
                _write_csv(transform_df, tables_dir / "transform_diagnostics.csv")
            )

        if dm_analysis:
            backtest_summary_df = dm_analysis.get("backtest_summary", pd.DataFrame())
            pairwise_df = dm_analysis.get("pairwise_results", pd.DataFrame())
            exported_files.append(
                _write_csv(backtest_summary_df, tables_dir / "dm_backtest_summary.csv")
            )
            exported_files.append(
                _write_csv(pairwise_df, tables_dir / "dm_pairwise.csv")
            )

            if export_dm_shortlist:
                shortlist_df = _build_dm_shortlist_df(dm_analysis)
                exported_files.append(
                    _write_csv(shortlist_df, tables_dir / "dm_shortlist.csv")
                )

        if export_tracker_results:
            tracker_export_df = _build_tracker_results_export_df(results_df)
            exported_files.append(
                _write_csv(tracker_export_df, tables_dir / "tracker_results.csv")
            )

        if export_params_long:
            params_long_df = _build_params_long_df(
                dataset_name=dataset_name,
                results_df=results_df,
            )
            exported_files.append(
                _write_csv(params_long_df, tables_dir / "params_long.csv")
            )

    if export_point_level:
        reference_df = _build_reference_series_df(
            dataset_name=dataset_name,
            train=train,
            test=test,
        )
        exported_files.append(
            _write_csv(reference_df, series_dir / "reference_series.csv")
        )

        final_forecasts_df = _build_final_forecasts_df(
            dataset_name=dataset_name,
            test=test,
            final_predictions=final_predictions or {},
            roles_map=roles_map,
            results_df=results_df,
        )
        exported_files.append(
            _write_csv(final_forecasts_df, series_dir / "final_forecasts.csv")
        )

        if export_validation_points:
            validation_points_df = _build_validation_points_df(
                dataset_name=dataset_name,
                tracker=tracker,
                results_df=results_df,
            )
            exported_files.append(
                _write_csv(validation_points_df, series_dir / "validation_points.csv")
            )

        if export_dm_points and dm_analysis:
            dm_points_df = _build_dm_backtest_points_df(
                dataset_name=dataset_name,
                dm_analysis=dm_analysis,
                results_df=results_df,
            )
            exported_files.append(
                _write_csv(dm_points_df, series_dir / "dm_backtest_points.csv")
            )

    if export_metadata:
        metadata = _build_run_metadata(
            dataset_name=dataset_name,
            dataset_slug=dataset_slug,
            dataset_config=dataset_config,
            results_df=results_df,
            transform_df=transform_df,
            final_predictions=final_predictions or {},
            dm_analysis=dm_analysis or {},
            export_flags={
                "export_tables": export_tables,
                "export_point_level": export_point_level,
                "export_metadata": export_metadata,
                "export_tracker_results": export_tracker_results,
                "export_dm_shortlist": export_dm_shortlist,
                "export_params_long": export_params_long,
                "export_validation_points": export_validation_points,
                "export_dm_points": export_dm_points,
            },
        )
        metadata_path = run_dir / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        exported_files.append(str(metadata_path))

    print(f"Exported forecasting data bundle to: {run_dir}")
    for path in exported_files:
        if path:
            print(f"  - {path}")

    return {
        "dataset_slug": dataset_slug,
        "output_dir": str(run_dir),
        "tables_dir": str(tables_dir),
        "series_dir": str(series_dir),
        "files": [path for path in exported_files if path],
    }


def _get_run_dir(dataset_slug, output_root=None) -> Path:
    if output_root is not None:
        root = Path(output_root)
    else:
        root = Path(__file__).resolve().parent.parent / "artifacts" / "forecasting"
    return root / dataset_slug


def _resolve_transform_df(
    dataset_name,
    train,
    transform_diagnostics_df=None,
    target_diagnostics=None,
):
    if transform_diagnostics_df is not None:
        return transform_diagnostics_df.copy()
    if target_diagnostics is None:
        return None
    return build_transform_diagnostics_df(
        dataset_name,
        target_diagnostics,
        train,
        include_details=True,
    )


def _build_comparison_metrics_df(
    dataset_name,
    results_df,
    final_predictions,
    roles_map,
):
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df["Dataset"] = dataset_name
    df["Category"] = df["Model"].apply(_get_model_category)
    df["Test RMSE"] = np.nan
    df["Test MAPE"] = np.nan
    df["Final Forecast Available"] = False
    df["Roles"] = ""

    for model_name, role_list in roles_map.items():
        if model_name in df["Model"].values:
            df.loc[df["Model"] == model_name, "Roles"] = "; ".join(role_list)

    for model_name, info in _iter_unique_model_predictions(final_predictions):
        if model_name in df["Model"].values:
            df.loc[df["Model"] == model_name, "Test RMSE"] = info.get("rmse", np.nan)
            df.loc[df["Model"] == model_name, "Test MAPE"] = info.get("mape", np.nan)
            df.loc[df["Model"] == model_name, "Final Forecast Available"] = True

    for role_key, role_name in _ROLE_KEYS.items():
        column_name = f"Is {role_name.replace('_', ' ').title()}"
        df[column_name] = df["Model"].apply(lambda model: role_name in roles_map.get(model, []))

    ordered_columns = [
        "Dataset",
        "Model",
        "Base Model",
        "Category",
        "Target Transform",
        "RMSE",
        "MAPE",
        "Test RMSE",
        "Test MAPE",
        "Tuning Time (s)",
        "Best Config Time (s)",
        "Combinations",
        "Selection Basis",
        "Final Forecast Available",
        "Roles",
        "Is Best Overall",
        "Is Fastest",
        "Is Best Statistical",
        "Is Best Dl",
        "Is Best Foundation",
    ]
    existing = [col for col in ordered_columns if col in df.columns]
    return df[existing].reset_index(drop=True)


def _build_tracker_results_export_df(results_df):
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()
    df["Category"] = df["Model"].apply(_get_model_category)
    df["Params JSON"] = df["Params"].apply(_safe_json_dumps)
    df["Params Compact"] = df["Params"].apply(_format_params_compact)
    return df.drop(columns=["Params"], errors="ignore").reset_index(drop=True)


def _build_params_long_df(dataset_name, results_df):
    rows = []
    if results_df.empty:
        return pd.DataFrame()

    for _, row in results_df.iterrows():
        params = strip_internal_params(row.get("Params"))
        if not params:
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": row["Model"],
                    "Base Model": row.get("Base Model", "-"),
                    "Target Transform": row.get("Target Transform", "-"),
                    "Param": "-",
                    "Value": "-",
                }
            )
            continue

        for key, value in params.items():
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": row["Model"],
                    "Base Model": row.get("Base Model", "-"),
                    "Target Transform": row.get("Target Transform", "-"),
                    "Param": key,
                    "Value": value,
                }
            )

    return pd.DataFrame(rows)


def _build_dm_shortlist_df(dm_analysis):
    shortlist = dm_analysis.get("shortlist", []) or []
    reasons = dm_analysis.get("shortlist_reasons", {}) or {}
    rows = []
    for model_name in shortlist:
        rows.append(
            {
                "Model": model_name,
                "Category": _get_model_category(model_name),
                "Included Because": reasons.get(model_name, "-"),
            }
        )
    return pd.DataFrame(rows)


def _build_reference_series_df(dataset_name, train, test):
    frames = []
    frames.extend(_partition_to_frames(dataset_name, train, partition="train"))
    frames.extend(_partition_to_frames(dataset_name, test, partition="test"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_final_forecasts_df(dataset_name, test, final_predictions, roles_map, results_df):
    columns = [
        "Dataset",
        "Model",
        "Base Model",
        "Category",
        "Target Transform",
        "Roles",
        "Series ID",
        "Time",
        "Actual",
        "Prediction",
        "Test RMSE",
        "Test MAPE",
    ]
    rows = []
    results_lookup = {}
    if results_df is not None and not results_df.empty:
        for _, row in results_df.iterrows():
            results_lookup[row["Model"]] = row.to_dict()

    for model_name, info in _iter_unique_model_predictions(final_predictions):
        prediction = info.get("prediction")
        if prediction is None:
            continue

        test_series_list = test if isinstance(test, list) else [test]
        pred_series_list = prediction if isinstance(prediction, list) else [prediction]
        role_string = "; ".join(roles_map.get(model_name, []))
        model_meta = results_lookup.get(model_name, {})

        for idx, (actual_s, pred_s) in enumerate(zip(test_series_list, pred_series_list)):
            actual_aligned = actual_s.slice_intersect(pred_s)
            pred_aligned = pred_s.slice_intersect(actual_s)
            series_id = _series_identifier(actual_s, idx)
            for time_value, actual_value, pred_value in zip(
                actual_aligned.time_index,
                actual_aligned.values(copy=False).flatten(),
                pred_aligned.values(copy=False).flatten(),
            ):
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Base Model": model_meta.get("Base Model", clean_model_name(model_name)),
                        "Category": info.get("category") or _get_model_category(model_name),
                        "Target Transform": model_meta.get("Target Transform", "-"),
                        "Roles": role_string,
                        "Series ID": series_id,
                        "Time": time_value,
                        "Actual": actual_value,
                        "Prediction": pred_value,
                        "Test RMSE": info.get("rmse", np.nan),
                        "Test MAPE": info.get("mape", np.nan),
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def _build_validation_points_df(dataset_name, tracker, results_df):
    rows = []
    if results_df.empty:
        return pd.DataFrame()

    for _, row in results_df.iterrows():
        model_name = row["Model"]
        artifact = tracker.get_validation_artifact(model_name)
        if artifact is None:
            continue
        frame = _artifact_to_frame(artifact)
        if frame.empty:
            continue
        frame.insert(0, "Dataset", dataset_name)
        frame.insert(1, "Model", model_name)
        frame.insert(2, "Base Model", row.get("Base Model", clean_model_name(model_name)))
        frame.insert(3, "Category", _get_model_category(model_name))
        frame.insert(4, "Target Transform", row.get("Target Transform", "-"))
        frame["Forecast Horizon"] = artifact.get("forecast_horizon")
        frame["Stride"] = artifact.get("stride")
        frame["Source"] = artifact.get("source", "rolling_validation")
        rows.append(frame)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_dm_backtest_points_df(dataset_name, dm_analysis, results_df):
    artifacts = dm_analysis.get("artifacts", {}) or {}
    results_lookup = {}
    if results_df is not None and not results_df.empty:
        for _, row in results_df.iterrows():
            results_lookup[row["Model"]] = row.to_dict()

    rows = []
    for model_name, artifact in artifacts.items():
        frame = _artifact_to_frame(artifact)
        if frame.empty:
            continue
        model_meta = results_lookup.get(model_name, {})
        frame.insert(0, "Dataset", dataset_name)
        frame.insert(1, "Model", model_name)
        frame.insert(2, "Base Model", model_meta.get("Base Model", clean_model_name(model_name)))
        frame.insert(3, "Category", _get_model_category(model_name))
        frame.insert(4, "Target Transform", model_meta.get("Target Transform", "-"))
        frame["Forecast Horizon"] = artifact.get("forecast_horizon")
        frame["Stride"] = artifact.get("stride")
        frame["Source"] = artifact.get("source", "dm_backtest")
        rows.append(frame)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_run_metadata(
    dataset_name,
    dataset_slug,
    dataset_config,
    results_df,
    transform_df,
    final_predictions,
    dm_analysis,
    export_flags,
):
    metadata = {
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_config": _json_safe(dataset_config),
        "n_tracked_models": int(len(results_df)) if results_df is not None else 0,
        "tracked_models": [] if results_df is None or results_df.empty else results_df["Model"].tolist(),
        "final_prediction_models": [model_name for model_name, _ in _iter_unique_model_predictions(final_predictions)],
        "dm_shortlist": dm_analysis.get("shortlist", []) if dm_analysis else [],
        "dm_shortlist_reasons": dm_analysis.get("shortlist_reasons", {}) if dm_analysis else {},
        "selected_transforms": (
            []
            if transform_df is None or transform_df.empty or "Selected Transform" not in transform_df.columns
            else transform_df["Selected Transform"].astype(str).tolist()
        ),
        "export_flags": export_flags,
    }
    return metadata


def _partition_to_frames(dataset_name, series_or_list, partition):
    series_list = series_or_list if isinstance(series_or_list, list) else [series_or_list]
    frames = []
    for idx, series in enumerate(series_list):
        if series is None:
            continue
        series_id = _series_identifier(series, idx)
        frames.append(
            pd.DataFrame(
                {
                    "Dataset": dataset_name,
                    "Series ID": series_id,
                    "Partition": partition,
                    "Time": series.time_index,
                    "Value": series.values(copy=False).flatten(),
                }
            )
        )
    return frames


def _artifact_to_frame(artifact):
    if not artifact:
        return pd.DataFrame()

    actual = artifact.get("actual")
    prediction = artifact.get("prediction")
    if actual is None or prediction is None:
        return pd.DataFrame()

    actual_list = actual if isinstance(actual, list) else [actual]
    pred_list = prediction if isinstance(prediction, list) else [prediction]
    frames = []
    for idx, (actual_s, pred_s) in enumerate(zip(actual_list, pred_list)):
        actual_aligned = actual_s.slice_intersect(pred_s)
        pred_aligned = pred_s.slice_intersect(actual_s)
        frames.append(
            pd.DataFrame(
                {
                    "Series ID": _series_identifier(actual_s, idx),
                    "Time": actual_aligned.time_index,
                    "Actual": actual_aligned.values(copy=False).flatten(),
                    "Prediction": pred_aligned.values(copy=False).flatten(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _extract_prediction_roles(final_predictions):
    roles = {}
    for key, role_name in _ROLE_KEYS.items():
        entry = final_predictions.get(key)
        if not entry:
            continue
        model_name = entry.get("model")
        if not model_name:
            continue
        roles.setdefault(model_name, []).append(role_name)
    return roles


def _iter_unique_model_predictions(final_predictions):
    seen = set()
    for key, value in (final_predictions or {}).items():
        if key in _ROLE_KEYS:
            continue
        if not isinstance(value, dict):
            continue
        model_name = value.get("model", key)
        if model_name in seen:
            continue
        if "prediction" not in value:
            continue
        seen.add(model_name)
        yield model_name, value


def _get_model_category(model_name):
    clean = clean_model_name(model_name)
    if clean in _STAT_MODELS:
        return "statistical"
    if clean in _DL_MODELS:
        return "dl"
    if clean in _FOUNDATION_MODELS:
        return "foundation"
    return "other"


def _series_identifier(series, fallback_idx):
    if hasattr(series, "static_covariates") and series.static_covariates is not None:
        cols = series.static_covariates.columns
        if "unique_id" in cols:
            return str(series.static_covariates["unique_id"].values[0])
        if "id" in cols:
            return str(series.static_covariates["id"].values[0])
    return f"series_{fallback_idx}"


def _format_params_compact(params):
    clean_params = strip_internal_params(params)
    if not clean_params:
        return "-"
    return "; ".join(f"{key}={value}" for key, value in clean_params.items())


def _safe_json_dumps(value):
    return json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (set, tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        enum_value = value.value if getattr(value, "value", None) is not None else value.name
        return _json_safe(enum_value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _write_csv(df, path: Path):
    if df is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _resolve_optional_flag(value, include_optional):
    if value is None:
        return bool(include_optional)
    return bool(value)


def _safe_slug(value, max_len=60):
    text = str(value or "").lower()
    allowed = []
    prev_underscore = False
    for char in text:
        if char.isalnum():
            allowed.append(char)
            prev_underscore = False
        else:
            if not prev_underscore:
                allowed.append("_")
            prev_underscore = True
    slug = "".join(allowed).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug or "dataset"
