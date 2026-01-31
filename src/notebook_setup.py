import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import warnings
import logging
from itertools import product
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Darts imports
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.metrics import rmse, mape
from darts.utils.utils import ModelMode
from darts.models import (
    ExponentialSmoothing,
    AutoARIMA,
    Prophet,
    TiDEModel,
    NBEATSModel,
    TFTModel,
)

# Environment setup (Chronos, TimeGPT)
try:
    from src.config import CHRONOS_AVAILABLE, TIMEGPT_AVAILABLE, NIXTLA_API_KEY
except ImportError:
    # Fallback if config does not exist
    CHRONOS_AVAILABLE = False
    TIMEGPT_AVAILABLE = False
    NIXTLA_API_KEY = None

# === LOCAL MODULE IMPORTS ===
from src.experiment import ExperimentTracker
from src.tuning import (
    run_tuning_and_eval,
    run_tuning_local_and_eval,
    run_tuning_global_and_eval,
)
from src.pipeline import run_foundation_models, get_final_predictions
from src.visualization import (
    plot_model_comparison,
    plot_forecast_comparison,
    export_plots,
    plot_dm_results,
)
from src.model_config import get_statistical_grids, get_dl_grids, TUNING_CONFIG
from src.evaluation import run_statistical_comparison

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["CMDSTAN_VERBOSE"] = "FALSE"
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
for logger in ["darts", "prophet", "pytorch_lightning", "nixtla"]:
    logging.getLogger(logger).setLevel(logging.CRITICAL)

print("Notebook setup complete.")
