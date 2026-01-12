import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from darts.metrics import rmse, mape


def plot_model_comparison(
    results_dataframe, dataset_name, value_unit=None, plot_mape=True
):
    """
    Vykreslí porovnání modelů.
    """
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

    if plot_mape:
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
            if not plot_mape and i == 1 and j == 2:
                continue
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
    Univerzální interaktivní graf.
    Automaticky dopočítává metriky pro konkrétní zobrazenou sérii.
    """
    is_multiseries = isinstance(train_series, list)

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

    def _add_traces(pred_key, col, role_label):
        if pred_key not in predictions_dict:
            return
        pred_info = predictions_dict[pred_key]
        pred = pred_info["prediction"]

        # 1. Vybrat správnou predikci (pokud list)
        if is_multiseries:
            pred = pred[series_idx]

        # 2. Dopočítat metriky LOCALLY pro tuto sérii
        local_rmse = rmse(test_s, pred)
        try:
            local_mape = mape(test_s, pred)
        except:
            local_mape = 0.0

        model_name = pred_info["model"]
        legend_name = f"{model_name} ({role_label})"

        # Vykreslení
        show_legend = True if col == 1 else False  # Jen jednou pro Train/Test

        # Train
        fig.add_trace(
            go.Scatter(
                x=train_s.time_index,
                y=train_s.values().flatten(),
                mode="lines",
                name="Train",
                line=dict(color=colors["train"], width=1),
                showlegend=show_legend,
                hovertemplate="<b>Train</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        # Test
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

        # Forecast
        c = colors["pred1"] if col == 1 else colors["pred2"]
        fig.add_trace(
            go.Scatter(
                x=pred.time_index,
                y=pred.values().flatten(),
                mode="lines",
                name=legend_name,
                line=dict(color=c, width=2, dash="dash"),
                showlegend=True,
                hovertemplate=f"<b>{model_name}</b><br>Date: %{{x}}<br>Pred: %{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=col,
        )

        # === OPRAVA XREF ===
        # Plotly používá "x domain" pro 1. graf a "x2 domain" pro 2. graf (nikdy ne x1)
        xref_val = "x domain" if col == 1 else f"x{col} domain"

        # Anotace (Nadpis grafu a Metriky dole)
        fig.add_annotation(
            x=0.5,
            y=1.08,
            xref=xref_val,
            yref="paper",
            text=f"<b>{model_name}</b> <span style='font-size:10px;color:gray'>({role_label})</span>",
            showarrow=False,
            font=dict(size=14),
            xanchor="center",
        )

        # Metriky dole (Locally calculated!)
        mape_txt = f" | MAPE: {local_mape:.2f}%" if local_mape > 0 else ""
        fig.add_annotation(
            x=0.5,
            y=-0.18,
            xref=xref_val,
            yref="paper",
            text=f"RMSE: {local_rmse:.2f}{mape_txt} | Tuning: {pred_info['tuning_time']:.1f}s",
            showarrow=False,
            font=dict(size=11),
            xanchor="center",
        )

    # Vykreslení
    current_col = 1
    if "best_rmse" in predictions_dict:
        _add_traces("best_rmse", current_col, "Best RMSE")
        if not same_model and "fastest" in predictions_dict:
            current_col += 1

    if not same_model and "fastest" in predictions_dict:
        _add_traces("fastest", current_col, "Fastest")

    fig.update_layout(
        title=f"<b>{dataset_name}</b>{title_suffix} - Out-of-Sample Forecast",
        height=500,
        margin=dict(b=100),
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

    if fig_comparison and not suffix:
        fig_comparison.write_image(
            os.path.join(output_dir, f"{slug}_comparison.png"),
            width=1700,
            height=600,
            scale=2,
        )
