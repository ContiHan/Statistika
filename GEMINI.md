# Project Context: Time Series Forecasting & Analysis Framework

## 1. Overview
This project establishes a comprehensive framework for time series forecasting, benchmarking state-of-the-art approaches against traditional statistical methods. It is designed to compare the performance, computational efficiency, and architectural differences of three distinct categories of models:
1.  **Statistical Models**: Classical approaches relying on mathematical properties of time series.
2.  **Deep Learning (Global) Models**: Neural networks trained across time series (or windows) to learn complex patterns.
3.  **Foundation Models**: Pre-trained Large Models (Zero-Shot) leveraging transfer learning.

**Language Policy:**
*   **Codebase:** All source code, variable names, and technical comments are written strictly in **English**.
*   **Interaction:** Communication with the AI agent (Gemini CLI) is conducted in **Czech**.

### 1.1. Dataset Portfolio
The framework is validated across diverse domains and frequencies to ensure robustness:

| Dataset | Domain | Frequency | Description |
| :--- | :--- | :--- | :--- |
| **wb_usa_real_gdp_yearly** | Macroeconomics | Yearly | World Bank USA Real GDP |
| **fred_gpdic1_investments_quarterly** | Finance/Macro | Quarterly | Gross Private Domestic Investment (FRED) |
| **ecb_eurczk_monthly** | Forex | Monthly | ECB EUR/CZK Exchange Rate |
| **m5_walmart_hobbies_daily** | Retail | Daily | M5 Walmart Sales (Hobbies Category) |
| **kaggle_btcusd_hourly** | Cryptocurrency | Hourly | BTC/USD Hourly Data (Kaggle) |

## 2. Methodological Framework

### 2.1. Experimental Design
The research follows a strict three-stage workflow for each dataset:
1.  **Preprocessing**: Data cleaning, frequency inference, and handling of missing values.
2.  **Exploratory Data Analysis (EDA)**: Statistical decomposition (Trend, Seasonality, Residuals), stationarity tests (ADF, KPSS), and autocorrelation analysis (ACF/PACF).
3.  **Forecasting & Evaluation**: The core experimental phase involving model training, tuning, and testing.

### 2.2. Data Partitioning Strategy
To ensure unbiased evaluation, the data is split into two distinct sets:
*   **Training Set (Observed)**: Used for model fitting, hyperparameter tuning, and cross-validation.
*   **Test Set (Hidden)**: A hold-out set (the last $n$ periods, defined by `test_periods`) used strictly for the final evaluation. The models **never** see this data during the tuning phase.

### 2.3. Evaluation Metrics
*   **RMSE (Root Mean Squared Error)**: The primary metric for optimization and model selection. Penalizes large errors heavily.
*   **MAPE (Mean Absolute Percentage Error)**: Used for interpretability of relative error.
*   **Tuning Time**: Measured to assess the computational cost of each model.

### 2.4. Statistical Significance Testing
To move beyond simple point estimates (RMSE/MAPE), the framework implements a standardized **Statistical Significance Protocol** using the **Diebold-Mariano (DM) Test**.

*   **Winner Selection Logic**: While the models are tuned using Cross-Validation (Phase 1), the statistical comparison identifies the "Best Accuracy" models based on their actual performance on the **hidden Test Set** (Phase 2). This ensures that the significance test evaluates the final predictive capability of the models.
*   **4-Way Comparison Protocol**: For every dataset, the framework automatically performs four targeted analyses:
    1.  **Best vs. 2nd Best**: Verifies if the overall winner is significantly better than the runner-up.
    2.  **Deep Learning (DL) vs. Statistical**: Compares the best performing neural network against the best traditional method.
    3.  **Foundation vs. Deep Learning**: Benchmarks zero-shot models (Chronos/TimeGPT) against custom-trained networks (TiDE/N-BEATS/TFT).
    4.  **Best Accuracy vs. Fastest**: Evaluates the trade-off between predictive precision and computational efficiency (Tuning Time).
*   **Robustness for Small Samples (HLN)**: 
    *   Macroeconomic data often results in short test sets ($n$). Standard DM tests are biased in these cases.
    *   The framework applies the **Harvey-Leybourne-Newbold (HLN)** correction to adjust the statistic.
    *   **Numerical Stability**: The HAC (Heteroskedasticity and Autocorrelation Consistent) variance estimator is modified to prevent zero-value statistics when the forecast horizon $h$ is close to the sample size $n$.
*   **Multi-Series Support**: For datasets with multiple time series (e.g., M5 Walmart), the framework flattens predictions and ground truth across all series into a single pooled sample before computing the DM statistic. This provides a global assessment of model superiority.
*   **Implementation**: All tests are performed as two-tailed tests at $\alpha=0.05$.

### 3.4. Visualization & Reporting (`visualization.py`)
The framework generates automated artifacts for academic reporting:
*   **Forecast Comparison Plot**: Interactive Plotly graph overlaying the Training data, Test data (Ground Truth), and the forecasts. It explicitly visualizes up to 5 key models:
    *   **Best Overall**: Lowest RMSE on Test set.
    *   **Fastest**: Lowest Tuning Time.
    *   **Best Statistical**: Best performer among statistical models.
    *   **Best DL**: Best performer among Deep Learning models.
    *   **Best Foundation**: Best performer among Foundation models.
*   **Model Comparison Chart (CV vs. Test)**: A grouped bar chart comparing all models across key metrics. It explicitly juxtaposes:
    *   **Validation Performance (CV)**: How the model performed during hyperparameter tuning (historical data).
    *   **Test Performance (Future)**: How the model performed on the unseen test set.
    *   *Insight*: This visualization highlights overfitting (good CV, poor Test) vs. robustness.
*   **Statistical Significance Table**: A stylized table summarizing the 4-way DM test results, including p-values and explicit "Winner" identification.
*   **Automated Export**: All charts and tables are exported as high-resolution PNG images (`images/forecasting/`) for direct inclusion in thesis documents.

---

## 4. Technical Stack & Implementation

### 4.1. Core Libraries
*   **`darts`**: The backbone framework providing a unified API for all models (PyTorch-based for DL).
*   **`torch`**: Underlying tensor computation for Deep Learning models.
*   **`transformers`**: Hugging Face library, used for IBM Granite TTM.
*   **`nixtla`**: Client for TimeGPT API.
*   **`optuna` / `scikit-learn`**: Utilized implicitly for optimization and preprocessing.

### 4.2. Supported Models
The framework currently supports and tunes the following models:
*   **Statistical**: Holt-Winters, AutoARIMA, Prophet.
*   **Deep Learning**: TiDE (Time-series Dense Encoder), N-BEATS, TFT (Temporal Fusion Transformer).
*   **Foundation**: Chronos 2.0 (Amazon, via Darts), IBM Granite TTM (Tiny Time Mixer), TimeGPT (Nixtla).

### 4.3. Project Structure
*   **`src/`**: Core logic modules.
    *   `pipeline.py`: Orchestrator. Handles Foundation models and final prediction generation.
    *   `tuning.py`: The engine room. Contains loops for Grid Search and backtesting logic.
    *   `experiment.py`: Logging utility and leaderboard tracking.
    *   `model_config.py`: Configuration "Cookbook". Defines hyperparameter search spaces.
    *   `visualization.py`: Plotting utilities (Forecasts, Model Comparison, DM Tables).
    *   `evaluation.py`: Statistical tests (Diebold-Mariano) with HLN correction and multi-series support.
    *   **`wrappers/`**: Custom model adapters (e.g., Granite TTM).
*   **`datasets/`**: Raw CSV data files.
*   **`preprocessing/`**: Notebooks for data cleaning and preparation (01-05).
*   **`exploration_data_analyses/`**: Notebooks for EDA (01-05).
*   **`forecasting/`**: Main notebooks for running the full forecasting pipeline (01-05).
*   **`images/forecasting/`**: Generated plots and tables.

## 5. Setup & Usage

### Prerequisites
*   Python 3.10+
*   Virtual Environment (`.win-venv` or `.mac-venv`)
*   API Keys (Optional): Required only for `TimeGPT` (Nixtla).

### Execution Flow
1.  **Environment**: Activate the virtual environment.
2.  **Configuration**: Ensure `src/config.py` is set up.
3.  **Workflow**: Run notebooks in numerical order:
    *   `preprocessing/`: Generates clean data.
    *   `exploration_data_analyses/`: Visualizes patterns.
    *   `forecasting/`: Runs the full pipeline defined above.

---
*This file serves as the definitive guide to the project's methodology and architecture.*
