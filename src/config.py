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

# Chronos (via Darts)
try:
    from darts.models import Chronos2Model
    import torch

    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False

# IBM Granite TTM (via Transformers)
try:
    import transformers

    GRANITE_AVAILABLE = True
except ImportError:
    GRANITE_AVAILABLE = False

# TimeGPT
try:
    from nixtla import NixtlaClient

    TIMEGPT_AVAILABLE = True
except ImportError:
    TIMEGPT_AVAILABLE = False

# === 4. LOAD API KEYS ===
try:
    from config.api_keys import NIXTLA_API_KEY
except ImportError:
    NIXTLA_API_KEY = None

# === 5. SUMMARY OUTPUT ===
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

print("  -> Chronos model (Amazon)")
color = GREEN if CHRONOS_AVAILABLE else RED
print(f"     - Model available: {color}{'Yes' if CHRONOS_AVAILABLE else 'No'}{RESET}")

print("  -> Granite TTM (IBM)")
color = GREEN if GRANITE_AVAILABLE else RED
print(f"     - Model available: {color}{'Yes' if GRANITE_AVAILABLE else 'No'}{RESET}")

print("  -> TimeGPT (Nixtla)")
color = GREEN if TIMEGPT_AVAILABLE else RED
print(f"     - Model available: {color}{'Yes' if TIMEGPT_AVAILABLE else 'No'}{RESET}")

color = GREEN if NIXTLA_API_KEY else RED
print(f"     - API Key: {color}{'Found' if NIXTLA_API_KEY else 'Not Found'}{RESET}")
