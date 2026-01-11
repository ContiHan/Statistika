import os
import sys
import warnings
import logging

# === 0. INICIALIZACE PROMĚNNÝCH ===
ChronosPipeline = None
NixtlaClient = None
torch = None  # <--- PŘIDÁNO: Inicializace torch
CHRONOS_AVAILABLE = False
TIMEGPT_AVAILABLE = False
NIXTLA_API_KEY = None

# === 1. NASTAVENÍ CEST (PATH) ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === 2. SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore")
os.environ["CMDSTAN_VERBOSE"] = "FALSE"
# ... (zbytek logování warningů beze změny) ...
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

# === 3. IMPORT KNIHOVEN ===
print("Checking dependencies in src/config.py...")

# Chronos
try:
    from chronos import ChronosPipeline
    import torch  # <--- Tady se načte do namespace modulu

    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    # torch zůstane None, pokud import selže

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

# Debug výpis
print(f"  -> Chronos: {CHRONOS_AVAILABLE}")
print(f"  -> TimeGPT: {TIMEGPT_AVAILABLE}")
print(f"  -> API Key: {'Found' if NIXTLA_API_KEY else 'Not Found'}")
