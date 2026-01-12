import os
import sys
import warnings
import logging

# === 0. VARIABLE INITIALIZATION ===
ChronosPipeline = None
NixtlaClient = None
torch = None
CHRONOS_AVAILABLE = False
TIMEGPT_AVAILABLE = False
NIXTLA_API_KEY = None

# === 1. PATH SETUP ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === 2. SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore")
os.environ["CMDSTAN_VERBOSE"] = "FALSE"
# ... (warning logging setup) ...
loggers_to_silence = [
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
]
for logger_name in loggers_to_silence:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# === 3. LIBRARY IMPORTS ===
print("Checking for crucial dependencies...")

# Chronos
try:
    from chronos import ChronosPipeline
    import torch

    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False

# TimeGPT
try:
    from nixtla import NixtlaClient

    TIMEGPT_AVAILABLE = True
except ImportError:
    TIMEGPT_AVAILABLE = False

# === 4. LOAD API KEYS ===
try:
    from api_keys import NIXTLA_API_KEY
except ImportError:
    NIXTLA_API_KEY = None

# Debug output
print(f"  -> Chronos: {'Available' if CHRONOS_AVAILABLE else 'Not Available'}")
print(f"  -> TimeGPT: {'Available' if TIMEGPT_AVAILABLE else 'Not Available'}")
print(f"  -> API Key: {'Found' if NIXTLA_API_KEY else 'Not Found'}")
