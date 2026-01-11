import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_model_comparison(results_dataframe, dataset_name, value_unit=None):
    """Vykreslí porovnání modelů."""
    if results_dataframe.empty:
        return None

    df_plot = results_dataframe.sort_values("RMSE", ascending=True).drop_duplicates(
        subset=["Model"], keep="first"
    )
    models = df_plot["Model"].tolist()

    rmse_title = (
        f"RMSE ({value_unit}, lower is better)"
        if value_unit
        else "RMSE (lower is better)"
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            rmse_title,
            "MAPE % (lower is better)",
            "Total Tuning Time (s)",
            "Best Config Training Time (s)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Bar(
            y=models,
            x=df_plot["RMSE"],
            orientation="h",
            marker_color="skyblue",
            hovertemplate="<b>%{y}</b><br>RMSE: %{x:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=models,
            x=df_plot["MAPE"],
            orientation="h",
            marker_color="salmon",
            hovertemplate="<b>%{y}</b><br>MAPE: %{x:.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            y=models,
            x=df_plot["Tuning Time (s)"],
            orientation="h",
            marker_color="lightgreen",
            text=[f"({c})" for c in df_plot["Combinations"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Time: %{x:.2f}s<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=models,
            x=df_plot["Best Config Time (s)"],
            orientation="h",
            marker_color="plum",
            hovertemplate="<b>%{y}</b><br>Time: %{x:.2f}s<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{dataset_name}</b> - All Models Comparison (Cross-Validation)",
            font=dict(size=16),
        ),
        width=1700,
        height=600,
        showlegend=False,
        margin=dict(l=120),
    )
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(autorange="reversed", row=i, col=j)
    return fig


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
    Univerzální interaktivní graf pro Single i Multi series.
    """
    # Detekce typu (List vs TimeSeries)
    is_multiseries = isinstance(train_series, list)

    # Výběr dat
    train_s = train_series[series_idx] if is_multiseries else train_series
    test_s = test_series[series_idx] if is_multiseries else test_series

    title_suffix = ""
    if is_multiseries and hasattr(train_s, "static_covariates"):
        sid = train_s.static_covariates["unique_id"].values[0]
        title_suffix = f" ({sid})"

    n_plots = len([k for k in ["best_rmse", "fastest"] if predictions_dict.get(k)])
    if n_plots == 0:
        return None

    same_model = False
    if "best_rmse" in predictions_dict and "fastest" in predictions_dict:
        if (
            predictions_dict["best_rmse"]["model"]
            == predictions_dict["fastest"]["model"]
        ):
            same_model = True
            n_plots = 1

    fig = make_subplots(rows=1, cols=n_plots, horizontal_spacing=0.08)
    colors = {
        "train": "#1f77b4",
        "test": "#2ca02c",
        "pred1": "#d62728",
        "pred2": "#ff7f0e",
    }
    y_label = (
        value_unit if value_unit else (target_column if target_column else "Value")
    )

    def _add_traces(pred_key, col, show_legend=True):
        if pred_key not in predictions_dict:
            return
        pred_info = predictions_dict[pred_key]

        # Získání predikce (List nebo TimeSeries)
        pred = pred_info["prediction"]
        if is_multiseries:
            pred = pred[series_idx]

        # Vykreslení
        fig.add_trace(
            go.Scatter(
                x=train_s.time_index,
                y=train_s.values().flatten(),
                mode="lines",
                name="Train",
                line=dict(color=colors["train"], width=1.5),
                showlegend=show_legend,
                hovertemplate="<b>Train</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
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
            row=1,
            col=col,
        )

        c = colors["pred1"] if col == 1 else colors["pred2"]
        fig.add_trace(
            go.Scatter(
                x=pred.time_index,
                y=pred.values().flatten(),
                mode="lines",
                name=f"Forecast ({pred_info['model']})",
                line=dict(color=c, width=2, dash="dash"),
                showlegend=show_legend,
                hovertemplate=f"<b>{pred_info['model']}</b><br>Date: %{{x}}<br>Pred: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    if "best_rmse" in predictions_dict:
        _add_traces("best_rmse", 1)
    if not same_model and "fastest" in predictions_dict:
        _add_traces("fastest", 2, show_legend=False)

    # === ANOTACE ===
    annotations = []

    if "best_rmse" in predictions_dict:
        info = predictions_dict["best_rmse"]
        annotations.append(
            dict(
                x=0.22 if n_plots == 2 else 0.5,
                y=1.02,
                xref="paper",
                yref="paper",
                text=f"<b>Best Model: {info['model']}</b>",
                showarrow=False,
                font=dict(size=13),
                xanchor="center",
            )
        )
        annotations.append(
            dict(
                x=0.22 if n_plots == 2 else 0.5,
                y=-0.18,
                xref="paper",
                yref="paper",
                text=f"Test RMSE: {info['rmse']:.4f} | Tuning: {info['tuning_time']:.1f}s",
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
            )
        )

    if not same_model and "fastest" in predictions_dict:
        info = predictions_dict["fastest"]
        annotations.append(
            dict(
                x=0.78,
                y=1.02,
                xref="paper",
                yref="paper",
                text=f"<b>Fastest Model: {info['model']}</b>",
                showarrow=False,
                font=dict(size=13),
                xanchor="center",
            )
        )
        annotations.append(
            dict(
                x=0.78,
                y=-0.18,
                xref="paper",
                yref="paper",
                text=f"Test RMSE: {info['rmse']:.4f} | Tuning: {info['tuning_time']:.1f}s",
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{dataset_name}</b>{title_suffix} - Out-of-Sample Forecast",
            font=dict(size=16),
        ),
        width=1700,
        height=500,
        hovermode="x unified",
        yaxis=dict(title=y_label),
        annotations=annotations,
        margin=dict(b=100),
    )

    # Zoom tlačítka
    full_start = train_s.start_time()
    full_end = test_s.end_time()
    test_start = test_s.start_time()

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        args=[{"xaxis.autorange": True, "xaxis2.autorange": True}],
                        label="Reset Zoom",
                        method="relayout",
                    ),
                    dict(
                        args=[
                            {
                                "xaxis.range": [test_start, full_end],
                                "xaxis2.range": [test_start, full_end],
                            }
                        ],
                        label="Zoom to Test Period",
                        method="relayout",
                    ),
                ],
            )
        ]
    )

    if n_plots == 2:
        fig.update_yaxes(title=y_label, row=1, col=2)

    return fig


def export_plots(fig_forecast, fig_comparison, dataset_name, suffix=""):
    """Export plots."""
    output_dir = os.path.abspath(
        os.path.join(os.getcwd(), "..", "images", "forecasting")
    )
    os.makedirs(output_dir, exist_ok=True)
    slug = dataset_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")

    if fig_forecast:
        name = f"{slug}_forecast_{suffix}.png" if suffix else f"{slug}_forecast.png"
        fig_forecast.write_image(
            os.path.join(output_dir, name), width=1700, height=500, scale=2
        )
        print(f"Forecast saved: {name}")

    if fig_comparison and not suffix:
        fig_comparison.write_image(
            os.path.join(output_dir, f"{slug}_comparison.png"),
            width=1700,
            height=600,
            scale=2,
        )
        print("Comparison saved.")
