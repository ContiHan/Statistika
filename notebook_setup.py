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

# Darts - Time Series Library
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

# Foundation Models (optional dependencies)
try:
    from chronos import ChronosPipeline
    import torch

    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False

try:
    from nixtla import NixtlaClient

    TIMEGPT_AVAILABLE = True
except ImportError:
    TIMEGPT_AVAILABLE = False

# Load API keys
# Attempt to find api_keys in the same directory as this module
try:
    # If running from a notebook where sys.path includes the root, this works
    from api_keys import NIXTLA_API_KEY
except ImportError:
    NIXTLA_API_KEY = None

# Suppress all warnings and verbose logging
warnings.filterwarnings("ignore")
os.environ["CMDSTAN_VERBOSE"] = "FALSE"
for logger_name in [
    "darts",
    "prophet",
    "cmdstanpy",
    "stan",
    "pystan",
    "prophet.models",
    "pytorch_lightning",
    "lightning",
    "nixtla",
    "kaleido",
    "choreographer",
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

print(f"Chronos: {'Available' if CHRONOS_AVAILABLE else 'Not installed'}")
print(f"TimeGPT: {'Available' if TIMEGPT_AVAILABLE else 'Not installed'}")
