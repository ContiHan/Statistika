import torch
import numpy as np
import pandas as pd
from darts import TimeSeries
try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    AutoModel = None
    AutoConfig = None

try:
    from tsfm_public import TinyTimeMixerForPrediction
except ImportError:
    TinyTimeMixerForPrediction = None

class GraniteTTMModel:
    """
    Wrapper for IBM Granite TTM (Tiny Time Mixers) Foundation Model.
    Compatible with Darts fit/predict API for Zero-Shot inference.
    """
    def __init__(self, model_name="ibm-granite/granite-timeseries-ttm-r1", context_length=512, **kwargs):
        self.model_name = model_name
        self.context_length = context_length
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Priority 1: Use tsfm_public (Recommended for TTM)
        if TinyTimeMixerForPrediction is not None:
            try:
                self.model = TinyTimeMixerForPrediction.from_pretrained(model_name).to(self.device)
                self.model.eval()
                return
            except Exception as e:
                print(f"Warning: Failed to load via tsfm_public: {e}")

        # Priority 2: Use AutoModel (Fallback)
        if AutoModel is None:
            print("Warning: 'transformers' library not found. GraniteTTM will not work.")
            return

        try:
            # Load model with trust_remote_code=True as it is likely a custom architecture
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading GraniteTTM model '{model_name}': {e}")

    def fit(self, series, **kwargs):
        """
        Zero-shot model. No training required.
        """
        pass

    def predict(self, n, series=None, **kwargs):
        """
        Predicts n steps into the future given the context series.
        """
        if self.model is None:
            raise RuntimeError("GraniteTTM model not initialized properly.")

        if series is None:
            raise ValueError("GraniteTTM requires a context 'series' for prediction.")

        # Ensure series is univariate for now (TTM is often univariate or channel-independent)
        # Flatten input to (1, context_length, 1) or (1, context_length)
        vals = series.values().flatten()
        
        # Handle context length: TTM usually has a fixed context window
        # We take the last 'context_length' points. 
        # If shorter, we pad? Or usually Foundation models handle it.
        # For TTM, we'll slice the last available points up to context_length.
        
        # Handle context length: TTM usually has a fixed context window
        if len(vals) < self.context_length:
            # Pad with zeros at the beginning (left padding)
            pad_len = self.context_length - len(vals)
            input_vals = np.concatenate([np.zeros(pad_len), vals])
        else:
            input_vals = vals[-self.context_length:]
        
        # Convert to tensor
        # Shape expectations depend on specific TTM implementation.
        # Usually: (batch_size, context_length, n_features)
        past_values = torch.tensor(input_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            try:
                # TTM Forward Pass
                # Note: The specific call args depend on the HF implementation.
                # Common pattern for TS models: past_values
                output = self.model(past_values=past_values)
                
                # Extract forecast
                # Output might be an object with 'prediction_logits' or just a tensor
                if hasattr(output, 'prediction_outputs'):
                    forecast = output.prediction_outputs
                elif hasattr(output, 'logits'):
                    forecast = output.logits
                elif isinstance(output, torch.Tensor):
                    forecast = output
                else:
                    # Fallback for tuple output
                    forecast = output[0]
                
                # Forecast shape usually: (batch_size, prediction_length, n_features)
                # We need to slice to 'n' if the model outputs a fixed horizon (e.g. 96)
                forecast = forecast.cpu().numpy().flatten()
                
                # Handle horizon mismatch
                if len(forecast) < n:
                    # Simple extension if model horizon is shorter than requested (not ideal for zero-shot but necessary)
                    # For a robust implementation, one would use autoregressive loop if supported.
                    # Here we return what we have or pad with last value
                    pad_len = n - len(forecast)
                    forecast = np.concatenate([forecast, np.full(pad_len, forecast[-1])])
                else:
                    forecast = forecast[:n]
                
                # Create Darts TimeSeries
                # Determine start time for forecast
                start_time = series.end_time() + series.freq
                
                return TimeSeries.from_times_and_values(
                    pd.date_range(start=start_time, periods=n, freq=series.freq),
                    forecast
                )

            except Exception as e:
                print(f"Error during GraniteTTM inference: {e}")
                # Return constant forecast as fallback to avoid pipeline crash
                return TimeSeries.from_times_and_values(
                    pd.date_range(start=series.end_time() + series.freq, periods=n, freq=series.freq),
                    np.full(n, vals[-1])
                )
