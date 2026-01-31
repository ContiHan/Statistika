# Time Series Forecasting & Analysis Framework

A comprehensive framework for benchmarking state-of-the-art forecasting models (Deep Learning, Foundation Models) against traditional statistical approaches.

## 🚀 Getting Started

### Prerequisites

*   **Python 3.10+**
*   **Virtual Environment** (Recommended)
    *   Mac/Linux: `python3 -m venv .mac-venv`
    *   Windows: `python -m venv .win-venv`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Activate Virtual Environment:**
    *   Mac/Linux: `source .mac-venv/bin/activate`
    *   Windows: `.win-venv\Scripts\activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 🔑 API Configuration

This project supports **TimeGPT** (via Nixtla), which requires an API key.

1.  Navigate to the `config/` directory.
2.  Rename the template file:
    ```bash
    mv config/api_keys.template.py config/api_keys.py
    ```
3.  Open `config/api_keys.py` and insert your API key:
    ```python
    NIXTLA_API_KEY = 'your_api_key_here'
    ```

> **Note:** The `config/api_keys.py` file is ignored by Git to keep your secrets safe.

## 🏃 Usage

The workflow is structured into numbered notebooks to be executed in order:

1.  **Preprocessing (`preprocessing/`)**: 
    *   Run notebooks `01` to `05` to clean and prepare the raw data.
2.  **Exploration (`exploration_data_analyses/`)**: 
    *   Run notebooks `01` to `05` to visualize trends, seasonality, and stationarity.
3.  **Forecasting (`forecasting/`)**: 
    *   Run notebooks `01` to `05` to train models and generate forecasts.

## 📁 Project Structure

*   `src/`: Core logic and helper functions.
*   `config/`: Configuration files and API keys.
*   `datasets/`: Input CSV data.
*   `images/`: Output visualizations.
*   `GEMINI.md`: Detailed project methodology and context.

## 🌍 Language Policy

*   **Code:** English (comments, variables, logic).
*   **Documentation:** English.
*   **Interaction:** User interaction with the AI agent is in **Czech**.
