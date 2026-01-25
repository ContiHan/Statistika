import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from darts.metrics import rmse, mape
import pandas as pd


def plot_model_comparison(
    results_dataframe, dataset_name, value_unit=None, plot_mape=True, test_predictions=None
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
            if m_name in ["best_rmse", "fastest"]: continue
            test_data.append({
                "Model": m_name,
                "RMSE_Test": info["rmse"],
                "MAPE_Test": info["mape"]
            })
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
        row=1, col=1
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
            row=1, col=1
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
            row=1, col=2
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
                row=1, col=2
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
        row=2, col=1
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
        row=2, col=2
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{dataset_name}</b> - Model Comparison: Validation (CV) vs Test (Future)",
            font=dict(size=16),
        ),
        width=1700,
        height=700, # Increased height slightly
        legend=dict(orientation="h", y=1.1, x=0), # Legend on top
        margin=dict(l=120),
        barmode='group' # Key for side-by-side bars
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

    # Color logic for Significance
    # Green for Yes, Light Gray for No
    fill_colors = []
    for sig in dm_results_df["Significant"]:
        if sig == "Yes":
            fill_colors.append("lightgreen")
        else:
            fill_colors.append("#f2f2f2") # light gray

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(dm_results_df.columns),
                    fill_color="#2a3f5f",
                    align="left",
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=[dm_results_df[k].tolist() for k in dm_results_df.columns],
                    fill_color=[
                        ["#f2f2f2"] * len(dm_results_df),  # Comparison
                        ["#f2f2f2"] * len(dm_results_df),  # Model A
                        ["#f2f2f2"] * len(dm_results_df),  # Model B
                        ["#f2f2f2"] * len(dm_results_df),  # DM Stat
                        ["#f2f2f2"] * len(dm_results_df),  # P-Value
                        fill_colors,                       # Significant (Colored!)
                        fill_colors,                       # Winner (Colored!)
                    ],
                    align="left",
                    font=dict(color="black", size=11),
                    format=[None, None, None, ".4f", ".4f", None, None]
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{dataset_name}</b> - Statistical Significance (Diebold-Mariano Test)",
            font=dict(size=16),
        ),
        width=1700,
        # Dynamically calculate height to fit content without excessive whitespace
        # Base height for header + title (~120px) + row height (~30px per row)
        height=120 + (len(dm_results_df) * 35),
        margin=dict(l=20, r=20, t=60, b=20),
    )
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
    model_roles = {} # Map model_name -> list of roles

    # Filter available keys and deduplicate models, aggregating roles
    role_labels = {
        "best_rmse": "Best Overall",
        "fastest": "Fastest",
        "best_stat": "Best Stat",
        "best_dl": "Best DL",
        "best_foundation": "Best Found."
    }

    for k in potential_keys:
        if k in predictions_dict:
            model_name = predictions_dict[k]["model"]
            
            # Add role to this model
            if model_name not in model_roles:
                model_roles[model_name] = []
                keys_to_plot.append(k) # Only add to plot list first time seen
            
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
        subplot_titles=[f"<b>{predictions_dict[k]['model']}</b> ({', '.join(model_roles[predictions_dict[k]['model']])})" for k in keys_to_plot]
    )
    
    colors = {
        "train": "#1f77b4",
        "test": "#2ca02c",
        "pred": ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"], # Different colors for each plot
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
        legend_name = f"{model_name}" # Roles are now in title

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
            row=row, col=col
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
            row=row, col=col
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
            row=row, col=col
        )

        # Annotations (Metrics in bottom right corner of subplot)
        mape_txt = f" | MAPE: {local_mape:.2f}%" if local_mape > 0 else ""
        tuning_txt = f" | Tune: {pred_info.get('tuning_time', 0):.0f}s"
        
        # We use a trick to place annotation relative to axis
        xref_val = "x domain" if row == 1 else f"x{row} domain"
        yref_val = "y domain" if row == 1 else f"y{row} domain"
        
        fig.add_annotation(
            x=1, y=0, xref=xref_val, yref=yref_val,
            text=f"RMSE: {local_rmse:.2f}{mape_txt}{tuning_txt}",
            showarrow=False, 
            font=dict(size=11, color="black"), 
            bgcolor="rgba(255,255,255,0.7)",
            xanchor="right", yanchor="bottom"
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
        showlegend=True
    )
            
    return fig


def export_plots(fig_forecast, fig_comparison, dataset_name, suffix="", fig_dm=None):
    """Export plots including DM test table."""
    output_dir = os.path.abspath(
        os.path.join(os.getcwd(), "..", "images", "forecasting")
    )
    os.makedirs(output_dir, exist_ok=True)
    slug = dataset_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")

    if fig_forecast:
        name = f"{slug}_forecast_{suffix}.png" if suffix else f"{slug}_forecast.png"
        fig_forecast.write_image(
            os.path.join(output_dir, name), width=1700, height=fig_forecast.layout.height or 500, scale=2
        )

    if fig_comparison and not suffix:
        fig_comparison.write_image(
            os.path.join(output_dir, f"{slug}_comparison.png"),
            width=1700,
            height=fig_comparison.layout.height or 600,
            scale=2,
        )
        
    if fig_dm:
        name = f"{slug}_dm_test_{suffix}.png" if suffix else f"{slug}_dm_test.png"
        fig_dm.write_image(
            os.path.join(output_dir, name), width=1000, height=fig_dm.layout.height, scale=2
        )
