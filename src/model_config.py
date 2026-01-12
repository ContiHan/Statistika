from darts.utils.utils import ModelMode

# === 1. GLOBAL TUNING CONFIGURATION ===
TUNING_CONFIG = {
    "RANDOM_STATE": 42,
    "N_ITER": 10,  # Number of random combinations (if full grid is not used)
    "USE_FULL_GRID": False,
}


# === 2. MODEL PARAMETERS COOKBOOK ===
def get_statistical_grids(seasonal_period):
    """
    Returns grids for statistical models (Holt-Winters, ARIMA, Prophet).
    Depends on dataset seasonality.
    """
    return {
        "Holt-Winters": {
            "seasonal_periods": [seasonal_period],
            "trend": [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
            "seasonal": [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
            "damped": [True, False],
        },
        "AutoARIMA": {
            "season_length": [seasonal_period],
            "seasonal": [True] if seasonal_period > 1 else [False],
            "max_p": [3],
            "max_q": [3],
            "stepwise": [True],
        },
        "Prophet": {
            "seasonality_mode": ["additive", "multiplicative"],
            "changepoint_prior_scale": [0.01, 0.1, 0.5],
        },
    }


def get_dl_grids(seasonal_period):
    """
    Returns grids for Deep Learning models.
    Optimized to prevent combination explosion.
    """
    # Base chunk size logic
    base = seasonal_period if seasonal_period > 1 else 12

    # Common Parameters (Shared for all DL models)
    common = {
        "input_chunk_length": [base, base * 2],
        "output_chunk_length": [base // 2, base],
        "n_epochs": [15, 30],
        "batch_size": [32, 64],
        "random_state": [TUNING_CONFIG["RANDOM_STATE"]],
    }

    return {
        "TiDE": {
            **common,
            "hidden_size": [64, 128],
            "dropout": [0.1],
            "num_encoder_layers": [1],
            "num_decoder_layers": [1],
        },
        "N-BEATS": {
            **common,
            "num_stacks": [10],
            "num_blocks": [1],
            "layer_widths": [128, 256],
        },
        "TFT": {
            **common,
            "hidden_size": [32, 64],
            "lstm_layers": [1],
            "num_attention_heads": [4],
            "dropout": [0.1],
            "add_relative_index": [True],
        },
    }
