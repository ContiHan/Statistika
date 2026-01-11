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

# Načtení environmentu (Chronos, TimeGPT)
# Zkusíme importovat config, pokud existuje
try:
    from src.config import CHRONOS_AVAILABLE, TIMEGPT_AVAILABLE, NIXTLA_API_KEY
except ImportError:
    # Fallback pokud config neexistuje
    CHRONOS_AVAILABLE = False
    TIMEGPT_AVAILABLE = False
    NIXTLA_API_KEY = None

# === DŮLEŽITÉ: IMPORTY Z NAŠICH MODULŮ ===
# Toto zpřístupní funkce run_tuning_*, run_foundation_models, atd. v noteboocích
from src.experiment import ExperimentTracker
from src.tuning import run_tuning_and_eval, run_tuning_local, run_tuning_global
from src.pipeline import run_foundation_models, get_final_predictions
from src.visualization import (
    plot_model_comparison,
    plot_forecast_comparison,
    export_plots,
)

# Potlačení warningů
warnings.filterwarnings("ignore")
os.environ["CMDSTAN_VERBOSE"] = "FALSE"
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
for logger in ["darts", "prophet", "pytorch_lightning", "nixtla"]:
    logging.getLogger(logger).setLevel(logging.CRITICAL)

print("Notebook setup complete.")
