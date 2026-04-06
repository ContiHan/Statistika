import os
import textwrap
import hashlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from darts.metrics import rmse, mape
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.statistical_transforms import BOXCOX_LOG_THRESHOLD


def _get_forecasting_output_dir():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "images", "forecasting"
        )
    )


def plot_data_split(train, test, config, series=None):
    """
    Visualize Train/Test split using Plotly.
    Handles single-series and multi-series (plots up to 5 series).
    """
    is_multi = isinstance(train, list)
    dataset_name = config["name"]
    smoke_test = config.get("smoke_test") or config.get("smoke_test_points")
    value_unit = config.get("value_unit", "Value")
    title_suffix = " [SMOKE TEST]" if smoke_test else ""

    if is_multi:
        if series is None:
            series = [t.append(te) for t, te in zip(train, test)]

        limit = 5
        to_plot = series[:limit]
        fig = go.Figure()
        for s in to_plot:
            label = "Series"
            if hasattr(s, "static_covariates") and s.static_covariates is not None:
                cols = s.static_covariates.columns
                if "unique_id" in cols:
                    label = str(s.static_covariates["unique_id"].values[0])
                elif "id" in cols:
                    label = str(s.static_covariates["id"].values[0])

            fig.add_trace(
                go.Scatter(
                    x=s.time_index,
                    y=s.values().flatten(),
                    mode="lines",
                    name=label,
                    line=dict(width=2),
                )
            )

        split_time = test[0].start_time()
        series_end = to_plot[0].end_time()
        fig.add_vrect(
            x0=split_time,
            x1=series_end,
            fillcolor="rgba(220, 53, 69, 0.12)",
            line_width=0,
        )
        fig.add_vline(
            x=split_time,
            line_color="red",
            line_dash="dash",
        )
        fig.add_annotation(
            x=split_time,
            y=1.02,
            xref="x",
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            font=dict(size=11, color="red"),
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{dataset_name}</b> - Train/Test Split{title_suffix}",
                font=dict(size=16),
            ),
            width=1700,
            height=500,
            yaxis_title=value_unit,
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=80, r=30, t=80, b=60),
        )
        return fig

    else:
        if series is None:
            series = train.append(test)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.time_index,
                y=series.values().flatten(),
                mode="lines",
                name="Full Series",
                line=dict(width=2),
            )
        )
        split_time = test.start_time()
        fig.add_vrect(
            x0=split_time,
            x1=series.end_time(),
            fillcolor="rgba(220, 53, 69, 0.12)",
            line_width=0,
        )
        fig.add_vline(
            x=split_time,
            line_color="red",
            line_dash="dash",
        )
        fig.add_annotation(
            x=split_time,
            y=1.02,
            xref="x",
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            font=dict(size=11, color="red"),
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.75)",
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{dataset_name}</b> - Train/Test Split{title_suffix}",
                font=dict(size=16),
            ),
            width=1700,
            height=500,
            yaxis_title=value_unit,
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=80, r=30, t=80, b=60),
        )
        return fig


def plot_model_comparison(
    results_dataframe,
    dataset_name,
    value_unit=None,
    plot_mape=True,
    test_predictions=None,
):
    """
    Plots model comparison (RMSE, MAPE, Time).
    If test_predictions is provided, it compares CV vs Test metrics.
    """
    if results_dataframe.empty:
        return None

    # Prepare CV Data
    df_cv = results_dataframe.sort_values("RMSE", ascending=True).drop_duplicates(
        subset=["Model"], keep="first"
    )
    models_cv = df_cv["Model"].tolist()

    # Prepare Test Data (if available)
    df_test = pd.DataFrame()
    if test_predictions:
        test_data = []
        for m_name, info in test_predictions.items():
            if m_name in ["best_rmse", "fastest"]:
                continue
            test_data.append(
                {"Model": m_name, "RMSE_Test": info["rmse"], "MAPE_Test": info["mape"]}
            )
        if test_data:
            df_test = pd.DataFrame(test_data)

    # Merge if possible to align models
    if not df_test.empty:
        df_merged = pd.merge(df_cv, df_test, on="Model", how="left")
        # Sort by Test RMSE if available, otherwise by CV RMSE
        if "RMSE_Test" in df_merged.columns:
            df_merged = df_merged.sort_values("RMSE_Test", ascending=True)
    else:
        df_merged = df_cv

    models = df_merged["Model"].tolist()

    rmse_title = (
        f"RMSE ({value_unit}, lower is better)"
        if value_unit
        else "RMSE (lower is better)"
    )
    mape_title = "MAPE % (lower is better)" if plot_mape else ""

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            rmse_title,
            mape_title,
            "Total Tuning Time (s)",
            "Best Config Training Time (s)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # --- RMSE Plot (CV vs Test) ---
    fig.add_trace(
        go.Bar(
            y=models,
            x=df_merged["RMSE"],
            name="CV (Validation)",
            orientation="h",
            marker_color="skyblue",
            hovertemplate="<b>%{y}</b><br>CV RMSE: %{x:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if not df_test.empty and "RMSE_Test" in df_merged.columns:
        fig.add_trace(
            go.Bar(
                y=models,
                x=df_merged["RMSE_Test"],
                name="Test (Future)",
                orientation="h",
                marker_color="royalblue",
                hovertemplate="<b>%{y}</b><br>Test RMSE: %{x:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # --- MAPE Plot (CV vs Test) ---
    if plot_mape:
        fig.add_trace(
            go.Bar(
                y=models,
                x=df_merged["MAPE"],
                name="CV (Validation)",
                orientation="h",
                marker_color="salmon",
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>CV MAPE: %{x:.2f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

        if not df_test.empty and "MAPE_Test" in df_merged.columns:
            fig.add_trace(
                go.Bar(
                    y=models,
                    x=df_merged["MAPE_Test"],
                    name="Test (Future)",
                    orientation="h",
                    marker_color="firebrick",
                    showlegend=False,
                    hovertemplate="<b>%{y}</b><br>Test MAPE: %{x:.2f}%<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # --- Tuning Time ---
    fig.add_trace(
        go.Bar(
            y=models,
            x=df_merged["Tuning Time (s)"],
            orientation="h",
            marker_color="lightgreen",
            text=[f"({c})" for c in df_merged["Combinations"]],
            textposition="outside",
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Time: %{x:.2f}s<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # --- Best Config Time ---
    fig.add_trace(
        go.Bar(
            y=models,
            x=df_merged["Best Config Time (s)"],
            orientation="h",
            marker_color="plum",
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Time: %{x:.2f}s<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{dataset_name}</b> - Model Comparison: Validation (CV) vs Test (Future)",
            font=dict(size=16),
        ),
        width=1700,
        height=700,  # Increased height slightly
        legend=dict(orientation="h", y=1.1, x=0),  # Legend on top
        margin=dict(l=120),
        barmode="group",  # Key for side-by-side bars
    )

    for i in range(1, 3):
        for j in range(1, 3):
            if not plot_mape and i == 1 and j == 2:
                continue
            fig.update_yaxes(autorange="reversed", row=i, col=j)

    return fig


def plot_dm_results(dm_results_df, dataset_name):
    """
    Plots Diebold-Mariano test results as a table.
    """
    if dm_results_df is None or dm_results_df.empty:
        return None
    significance_col = (
        "Significant Holm"
        if "Significant Holm" in dm_results_df.columns
        else "Significant"
    )
    note = (
        "Models are shortlisted by validation RMSE. "
        "Use P-Value Holm as the main significance decision because it corrects for multiple pairwise tests."
    )
    return _plot_styled_table(
        dm_results_df,
        title=f"<b>{dataset_name}</b> - Pairwise Diebold-Mariano Results",
        note=note,
        highlight_values={
            "Winner": "lightgreen",
            "Significant": ("lightgreen"),
            "Significant Holm": ("lightgreen"),
        },
        wrap_widths={
            "Model A": 22,
            "Model B": 22,
            "Winner": 22,
        },
        significance_columns=[significance_col],
        min_cell_height=56,
    )


def plot_dm_backtest_summary_table(summary_df, dataset_name):
    """
    Render the dedicated DM backtest summary as a styled Plotly table.
    """
    if summary_df is None or summary_df.empty:
        return None

    table_df = summary_df.copy()
    if "Included Because" in table_df.columns:
        table_df["Included Because"] = table_df["Included Because"].apply(
            _shorten_included_because
        )
    if "Notes" in table_df.columns:
        notes_series = table_df["Notes"].fillna("").astype(str).str.strip()
        if notes_series.eq("").all():
            table_df = table_df.drop(columns=["Notes"])

    note = (
        "Each shortlisted model is re-evaluated on a dedicated rolling train backtest before the pairwise DM test.<br>"
        "Included Because: Overall #1/#2 = best or second-best validation RMSE; "
        "Stat #1 / DL #1 / Foundation #1 = category winner; Fastest = shortest tuning time."
    )
    return _plot_styled_table(
        table_df,
        title=f"<b>{dataset_name}</b> - DM Backtest Summary",
        note=note,
        highlight_values={
            "Status": "lightgreen",
        },
        wrap_widths={
            "Model": 26,
            "Included Because": 24,
            "Notes": 48,
        },
        significance_columns=["Status"],
        positive_values={"OK"},
        min_cell_height=56,
    )


def plot_selected_params_table(params_df, dataset_name):
    """
    Render the selected winning configuration for each model as a styled Plotly table.
    """
    if params_df is None or params_df.empty:
        return None

    table_df = params_df[
        [
            "Base Model",
            "RMSE",
            "MAPE",
            "Tuning Time (s)",
            "Best Config Time (s)",
            "Selected Params",
        ]
    ].copy()
    table_df["Selected Params"] = (
        table_df["Selected Params"]
        .fillna("-")
        .astype(str)
        .str.replace("; ", "<br>", regex=False)
    )
    note = (
        "Each row shows the selected winning configuration stored in the experiment tracker after tuning.<br>"
        "Selected Params contains the final hyperparameter combination used for that model variant."
    )
    return _plot_styled_table(
        table_df,
        title=f"<b>{dataset_name}</b> - Selected Model Configurations",
        note=note,
        wrap_widths={
            "Base Model": 24,
        },
        min_cell_height=16,
        per_line_height=14,
    )


def plot_boxcox_diagnostics_table(diagnostics_df, dataset_name):
    """
    Render Box-Cox / transform diagnostics as a Plotly table.
    """
    if diagnostics_df is None or diagnostics_df.empty:
        return None

    table_df = diagnostics_df.copy()
    if "Dataset" in table_df.columns and table_df["Dataset"].nunique() == 1:
        table_df = table_df.drop(columns=["Dataset"])
    if "Series" in table_df.columns:
        series_values = table_df["Series"].fillna("").astype(str).str.strip()
        if series_values.isin({"", "-", "ALL"}).all():
            table_df = table_df.drop(columns=["Series"])
    if "Series Count" in table_df.columns:
        series_count = pd.to_numeric(table_df["Series Count"], errors="coerce")
        if series_count.notna().all() and (series_count == 1).all():
            table_df = table_df.drop(columns=["Series Count"])
    if "Reason" in table_df.columns:
        table_df = table_df.drop(columns=["Reason"])

    if "Selection Rule" in table_df.columns:
        table_df["Selection Rule"] = table_df["Selection Rule"].apply(
            _shorten_boxcox_rule
        )

    note = (
        "Lambda near 0 supports log; lambda near 1 supports raw.<br>"
        "CI Low and CI High show the 95% Box-Cox confidence interval on the train split.<br>"
        "Rule legend: nonpositive = zero/negative values; lambda≈0 = near zero; "
        "CI→log = CI supports log; raw = raw kept; aggregate = dataset-level decision."
    )
    return _plot_styled_table(
        table_df,
        title=f"<b>{dataset_name}</b> - Transform Diagnostics (Box-Cox)",
        note=note,
        highlight_values={
            "Selected Transform": ("lightgreen"),
        },
        wrap_widths={
            "Series": 22,
            "Selection Rule": 18,
        },
        min_cell_height=56,
    )


def plot_boxcox_lambda_chart(diagnostics_df, dataset_name):
    """
    Plot Box-Cox lambda estimates with confidence intervals.
    """
    if diagnostics_df is None or diagnostics_df.empty:
        return None

    df = diagnostics_df.copy()
    df = df[np.isfinite(df["Box-Cox Lambda"])]
    if df.empty:
        return None

    labels = [
        row["Dataset"] if row["Series"] in {"-", "ALL"} else f"{row['Series']}"
        for _, row in df.iterrows()
    ]
    error_plus = np.maximum(
        0, df["CI High"].to_numpy() - df["Box-Cox Lambda"].to_numpy()
    )
    error_minus = np.maximum(
        0, df["Box-Cox Lambda"].to_numpy() - df["CI Low"].to_numpy()
    )
    colors = [
        "seagreen" if val == "log" else "slategray" for val in df["Selected Transform"]
    ]
    lambda_values = df["Box-Cox Lambda"].to_numpy(dtype=float)
    ci_lows = df["CI Low"].to_numpy(dtype=float)
    ci_highs = df["CI High"].to_numpy(dtype=float)
    y_extent = max(
        1.1,
        np.nanmax(np.abs(np.concatenate([lambda_values, ci_lows, ci_highs]))),
    )
    y_extent = float(np.ceil((y_extent + 0.2) * 10) / 10)

    fig = go.Figure()
    fig.add_hrect(
        y0=-BOXCOX_LOG_THRESHOLD,
        y1=BOXCOX_LOG_THRESHOLD,
        fillcolor="rgba(46, 139, 87, 0.12)",
        line_width=0,
        annotation_text="log-like zone",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=df["Box-Cox Lambda"],
            mode="markers",
            marker=dict(size=12, color=colors),
            error_y=dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                thickness=1.5,
                width=6,
            ),
            customdata=np.stack(
                [
                    df["Selected Transform"],
                    df["Selection Rule"],
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Lambda: %{y:.4f}<br>"
                "Transform: %{customdata[0]}<br>"
                "Rule: %{customdata[1]}<extra></extra>"
            ),
            name="Box-Cox lambda",
        )
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color="seagreen", annotation_text="log (lambda = 0)"
    )
    fig.add_hline(
        y=BOXCOX_LOG_THRESHOLD,
        line_dash="dot",
        line_color="seagreen",
        line_width=1,
        annotation_text=f"+{BOXCOX_LOG_THRESHOLD:.1f}",
        annotation_position="right",
    )
    fig.add_hline(
        y=-BOXCOX_LOG_THRESHOLD,
        line_dash="dot",
        line_color="seagreen",
        line_width=1,
        annotation_text=f"-{BOXCOX_LOG_THRESHOLD:.1f}",
        annotation_position="right",
    )
    fig.add_hline(
        y=-1,
        line_dash="dash",
        line_color="slategray",
        annotation_text="raw (lambda = -1)",
    )
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="slategray",
        annotation_text="raw (lambda = 1)",
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{dataset_name}</b> - Box-Cox Lambda Diagnostics"
                "<br><sup>The shaded band marks the log-like zone [-0.3, 0.3]. "
                "Negative lambda alone does not imply log; lambda = 1 marks the raw scale.</sup>"
            ),
            font=dict(size=16),
        ),
        width=1700,
        height=550,
        xaxis_title="Series",
        yaxis_title="Box-Cox lambda",
        yaxis=dict(range=[-y_extent, y_extent]),
        margin=dict(l=80, r=30, t=80, b=80),
        showlegend=False,
    )
    return fig


def plot_dm_heatmap(pairwise_df, dataset_name, value_col="P-Value Holm"):
    """
    Plot pairwise DM results as a symmetric heatmap.
    """
    if pairwise_df is None or pairwise_df.empty:
        return None

    if value_col not in pairwise_df.columns:
        value_col = "P-Value"

    models = sorted(set(pairwise_df["Model A"]).union(pairwise_df["Model B"]))
    heatmap = pd.DataFrame(np.nan, index=models, columns=models)
    text = pd.DataFrame("", index=models, columns=models)

    for model in models:
        heatmap.loc[model, model] = 0.0
        text.loc[model, model] = "-"

    for _, row in pairwise_df.iterrows():
        model_a = row["Model A"]
        model_b = row["Model B"]
        p_value = row[value_col]
        winner = row.get("Winner", "-")
        signif_col = (
            "Significant Holm" if "Significant Holm" in row.index else "Significant"
        )
        significant = row.get(signif_col, "No")

        label = f"p={row[value_col]:.4f}<br>{winner if significant == 'Yes' else 'ns'}"
        heatmap.loc[model_a, model_b] = p_value
        heatmap.loc[model_b, model_a] = p_value
        text.loc[model_a, model_b] = label
        text.loc[model_b, model_a] = label

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=heatmap.to_numpy(),
                x=models,
                y=models,
                text=text.to_numpy(),
                texttemplate="%{text}",
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=max(0.05, float(np.nanmax(heatmap.to_numpy()))),
                colorbar=dict(title=value_col),
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>%{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{dataset_name}</b> - Pairwise DM Heatmap"
                "<br><sup>Each cell shows the corrected p-value and the significant winner; "
                "ns means the difference is not significant.</sup>"
            ),
            font=dict(size=16),
        ),
        width=1700,
        height=max(550, 140 * len(models)),
        margin=dict(l=120, r=30, t=80, b=120),
    )
    return fig


def _plot_styled_table(
    df,
    title,
    note=None,
    highlight_values=None,
    wrap_widths=None,
    significance_columns=None,
    positive_values=None,
    min_cell_height=48,
    per_line_height=30,
):
    display_df, line_counts = _wrap_table_dataframe(df, wrap_widths or {})
    significance_columns = significance_columns or []
    positive_values = positive_values or {"Yes", "log"}
    highlight_values = highlight_values or {}

    column_fill = []
    column_formats = []
    for col in df.columns:
        if col in highlight_values:
            fill = [
                (
                    highlight_values[col]
                    if _is_positive_cell(val, positive_values)
                    else "#f2f2f2"
                )
                for val in df[col]
            ]
            column_fill.append(fill)
        else:
            column_fill.append(["#f2f2f2"] * len(df))

        if pd.api.types.is_float_dtype(df[col]):
            column_formats.append(".4f")
        elif pd.api.types.is_integer_dtype(df[col]):
            column_formats.append(".0f")
        else:
            column_formats.append(None)

    note_lines = [line.strip() for line in (note or "").split("<br>") if line.strip()]
    header_height = 40
    cell_height = max(min_cell_height, per_line_height * max(line_counts or [1]))
    top_margin = 104
    note_font_size = 11
    note_line_spacing = 16
    bottom_margin = 22 + (len(note_lines) * note_line_spacing if note_lines else 0)
    fig_height = (
        top_margin + bottom_margin + header_height + (len(df) * cell_height) + 24
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(display_df.columns),
                    fill_color="#2a3f5f",
                    align="center",
                    font=dict(color="white", size=12),
                    height=header_height,
                ),
                cells=dict(
                    values=[display_df[k].tolist() for k in display_df.columns],
                    fill_color=column_fill,
                    align="center",
                    font=dict(color="black", size=11),
                    format=column_formats,
                    height=cell_height,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16),
            y=0.975,
            yanchor="top",
            pad=dict(t=10, b=6),
        ),
        width=1700,
        height=fig_height,
        margin=dict(l=20, r=20, t=top_margin, b=bottom_margin),
    )
    for idx, line in enumerate(note_lines):
        fig.add_annotation(
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            yshift=-(10 + idx * note_line_spacing),
            text=f"<span style='font-size:{note_font_size}px'>{line}</span>",
        )
    return fig


def _wrap_table_dataframe(df, wrap_widths):
    wrapped = df.copy()
    line_counts = []

    for col, width in wrap_widths.items():
        if col in wrapped.columns:
            wrapped[col] = wrapped[col].apply(lambda val: _wrap_cell_text(val, width))

    for row_idx in range(len(wrapped)):
        row_max = 1
        for col in wrapped.columns:
            row_max = max(row_max, _cell_line_count(wrapped.iloc[row_idx][col]))
        line_counts.append(row_max)

    return wrapped, line_counts


def _wrap_cell_text(value, width):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value
    text = str(value)
    if not text:
        return ""
    lines = textwrap.wrap(
        text, width=width, break_long_words=False, break_on_hyphens=False
    )
    if not lines:
        return text
    return "<br>".join(lines)


def _cell_line_count(value):
    if isinstance(value, str):
        return value.count("<br>") + 1 if value else 1
    return 1


def _is_positive_cell(value, positive_values):
    if pd.isna(value):
        return False
    return str(value) in {str(item) for item in positive_values}


def _shorten_boxcox_rule(value):
    mapping = {
        "nonpositive_values": "nonpositive",
        "boxcox_lambda_near_zero": "lambda≈0",
        "boxcox_ci_prefers_log": "CI→log",
        "boxcox_prefers_raw": "raw",
        "aggregate_dataset_decision": "aggregate",
    }
    return mapping.get(str(value), str(value))


def _shorten_boxcox_reason(value):
    text = str(value)
    replacements = {
        "Raw selected: target contains zero or negative values.": "Nonpositive values -> raw only.",
        "Raw selected: Box-Cox lambda could not be estimated.": "Box-Cox lambda unavailable.",
    }
    if text in replacements:
        return replacements[text]
    if "supports a log-like transform" in text:
        return "Train diagnostics favor log."
    if "does not support a log-like transform strongly enough" in text:
        return "Train diagnostics favor raw."
    return text


def _shorten_included_because(value):
    text = str(value)
    replacements = {
        "Best overall validation RMSE": "Overall #1",
        "Second-best overall validation RMSE": "Overall #2",
        "Best statistical validation RMSE": "Stat #1",
        "Best deep learning validation RMSE": "DL #1",
        "Best foundation validation RMSE": "Foundation #1",
        "Fastest tuning time": "Fastest",
        "Manually selected": "Manual",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("; ", " + ")
    return text


def plot_forecast_comparison(
    train_series,
    test_series,
    predictions_dict,
    dataset_name,
    target_column=None,
    value_unit=None,
    series_idx=0,
):
    """
    Universal interactive chart for Forecast vs Actual.
    Automatically visualizes Best RMSE, Fastest, and Category Winners (Stat, DL, Foundation).
    """
    is_multiseries = isinstance(train_series, list)

    train_s = train_series[series_idx] if is_multiseries else train_series
    test_s = test_series[series_idx] if is_multiseries else test_series

    title_suffix = ""
    if is_multiseries and hasattr(train_s, "static_covariates"):
        sid = train_s.static_covariates["unique_id"].values[0]
        title_suffix = f" ({sid})"

    # Select keys to plot in order of importance
    potential_keys = ["best_rmse", "fastest", "best_stat", "best_dl", "best_foundation"]
    keys_to_plot = []
    model_roles = {}  # Map model_name -> list of roles

    # Filter available keys and deduplicate models, aggregating roles
    role_labels = {
        "best_rmse": "Best RMSE",
        "fastest": "Fastest",
        "best_stat": "Best Statistical",
        "best_dl": "Best Deep Learning",
        "best_foundation": "Best Foundation",
    }

    for k in potential_keys:
        if k in predictions_dict:
            model_name = predictions_dict[k]["model"]

            # Add role to this model
            if model_name not in model_roles:
                model_roles[model_name] = []
                keys_to_plot.append(k)  # Only add to plot list first time seen

            model_roles[model_name].append(role_labels.get(k, k))

    # If no keys found (legacy support), pick first two
    if not keys_to_plot:
        keys_to_plot = list(predictions_dict.keys())[:2]
        if not keys_to_plot:
            return None

    n_plots = len(keys_to_plot)
    # Stack vertically: rows=n_plots, cols=1
    fig = make_subplots(
        rows=n_plots,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=[
            f"<b>{predictions_dict[k]['model']}</b> ({', '.join(model_roles[predictions_dict[k]['model']])})"
            for k in keys_to_plot
        ],
    )

    colors = {
        "train": "#1f77b4",
        "test": "#2ca02c",
        "pred": [
            "#d62728",
            "#ff7f0e",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ],  # Different colors for each plot
    }
    y_label = (
        value_unit if value_unit else (target_column if target_column else "Value")
    )

    for i, pred_key in enumerate(keys_to_plot):
        row = i + 1
        col = 1
        pred_info = predictions_dict[pred_key]
        pred = pred_info["prediction"]

        if is_multiseries:
            pred = pred[series_idx]

        # Calculate local metrics
        local_rmse = rmse(test_s, pred)
        try:
            local_mape = mape(test_s, pred)
        except:
            local_mape = 0.0

        model_name = pred_info["model"]
        roles_str = ", ".join(model_roles[model_name])
        legend_name = f"{model_name}"  # Roles are now in title

        # Plot Train (only once for legend or maybe first plot)
        show_legend = True if row == 1 else False

        # Use simpler train line for clarity (last 200 points if long)
        plot_train_s = train_s
        if len(train_s) > 200:
            plot_train_s = train_s[-200:]

        fig.add_trace(
            go.Scatter(
                x=plot_train_s.time_index,
                y=plot_train_s.values().flatten(),
                mode="lines",
                name="Train (Last part)",
                line=dict(color=colors["train"], width=1),
                showlegend=show_legend,
                hovertemplate="<b>Train</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Plot Test
        fig.add_trace(
            go.Scatter(
                x=test_s.time_index,
                y=test_s.values().flatten(),
                mode="lines",
                name="Test (Actual)",
                line=dict(color=colors["test"], width=2),
                showlegend=show_legend,
                hovertemplate="<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Plot Forecast
        c_idx = i % len(colors["pred"])
        fig.add_trace(
            go.Scatter(
                x=pred.time_index,
                y=pred.values().flatten(),
                mode="lines",
                name=legend_name,
                line=dict(color=colors["pred"][c_idx], width=2, dash="dash"),
                showlegend=True,
                hovertemplate=f"<b>{model_name}</b><br>Roles: {roles_str}<br>Date: %{{x}}<br>Pred: %{{y:.2f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Annotations (Metrics in bottom right corner of subplot)
        mape_txt = f" | MAPE: {local_mape:.2f}%" if local_mape > 0 else ""
        tuning_txt = f" | Tune: {pred_info.get('tuning_time', 0):.0f}s"

        # We use a trick to place annotation relative to axis
        xref_val = "x domain" if row == 1 else f"x{row} domain"
        yref_val = "y domain" if row == 1 else f"y{row} domain"

        fig.add_annotation(
            x=1,
            y=0,
            xref=xref_val,
            yref=yref_val,
            text=f"RMSE: {local_rmse:.2f}{mape_txt}{tuning_txt}",
            showarrow=False,
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.7)",
            xanchor="right",
            yanchor="bottom",
        )

        # Add Y-axis label to each subplot
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    # Dynamic height: 300px per plot + some padding
    total_height = max(500, n_plots * 300)

    fig.update_layout(
        title=f"<b>{dataset_name}</b>{title_suffix} - Forecast Comparison by Category",
        width=1700,
        height=total_height,
        margin=dict(b=50, t=80),
        showlegend=True,
    )

    return fig


def export_plots(fig_forecast, fig_comparison, dataset_name, suffix="", fig_dm=None):
    """Export plots including DM test table."""
    output_dir = _get_forecasting_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    slug = _safe_slug(dataset_name, max_len=80)
    safe_suffix = _safe_slug(suffix, max_len=40) if suffix else ""

    if fig_forecast is not None:
        name = (
            f"{slug}_forecast_{safe_suffix}.png"
            if safe_suffix
            else f"{slug}_forecast.png"
        )
        _write_image_safe(
            fig_forecast,
            output_dir,
            name,
            width=1700,
            height=fig_forecast.layout.height or 500,
            scale=2,
        )

    if fig_comparison is not None and not suffix:
        _write_image_safe(
            fig_comparison,
            output_dir,
            f"{slug}_comparison.png",
            width=1700,
            height=fig_comparison.layout.height or 600,
            scale=2,
        )

    if fig_dm is not None:
        name = (
            f"{slug}_dm_test_{safe_suffix}.png"
            if safe_suffix
            else f"{slug}_dm_test.png"
        )
        _write_image_safe(
            fig_dm,
            output_dir,
            name,
            width=1000,
            height=fig_dm.layout.height,
            scale=2,
        )


def export_named_plots(dataset_name, figures, suffix=""):
    """
    Export arbitrary named figures with the same slug logic as the default plots.
    """
    output_dir = _get_forecasting_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    slug = _safe_slug(dataset_name, max_len=80)
    safe_suffix = _safe_slug(suffix, max_len=40) if suffix else ""

    for plot_name, fig in (figures or {}).items():
        if fig is None:
            continue
        safe_plot_name = _safe_slug(plot_name, max_len=40) or "plot"
        file_name = (
            f"{slug}_{safe_plot_name}_{safe_suffix}.png"
            if safe_suffix
            else f"{slug}_{safe_plot_name}.png"
        )
        width = fig.layout.width or 1700
        height = fig.layout.height or 600
        _write_image_safe(
            fig,
            output_dir,
            file_name,
            width=width,
            height=height,
            scale=2,
        )


def _safe_slug(value, max_len=60):
    if value is None:
        return ""
    text = str(value).strip().lower()
    allowed = []
    prev_underscore = False
    for ch in text:
        if ch.isalnum():
            allowed.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                allowed.append("_")
                prev_underscore = True
    slug = "".join(allowed).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def _write_image_safe(fig, output_dir, file_name, width, height, scale):
    output_path = os.path.join(output_dir, file_name)
    try:
        image_bytes = fig.to_image(
            format="png", width=width, height=height, scale=scale
        )
        with open(output_path, "wb") as f:
            f.write(image_bytes)
    except OSError as e:
        if getattr(e, "errno", None) != 63:
            raise
        base, ext = os.path.splitext(file_name)
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
        safe_base = _safe_slug(base, max_len=70)
        fallback_name = f"{safe_base[:50]}_{digest}{ext or '.png'}"
        fallback_path = os.path.join(output_dir, fallback_name)
        image_bytes = fig.to_image(
            format="png", width=width, height=height, scale=scale
        )
        with open(fallback_path, "wb") as f:
            f.write(image_bytes)
