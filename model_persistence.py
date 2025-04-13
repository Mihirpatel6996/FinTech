import os
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model, Sequential
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class ModelPersistence:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def ensure_symbol_dir(self, symbol: str) -> Path:
        """Create and return path to symbol-specific directory"""
        symbol_dir = self.models_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir

    def save_lstm_model(self, symbol: str, model: Sequential, scaler: MinMaxScaler) -> str:
        """Save LSTM model and scaler to disk"""
        symbol_dir = self.ensure_symbol_dir(symbol)

        # Save model
        model_path = symbol_dir / "lstm_model.h5"
        model.save(str(model_path))

        # Save scaler
        scaler_path = symbol_dir / "lstm_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save metadata
        metadata = {
            "model_type": "LSTM",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0"
        }

        metadata_path = symbol_dir / "lstm_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        return str(model_path)

    def load_lstm_model(self, symbol: str):
        """Load LSTM model and scaler from disk if they exist"""
        symbol_dir = self.ensure_symbol_dir(symbol)
        model_path = symbol_dir / "lstm_model.h5"
        scaler_path = symbol_dir / "lstm_scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            return None, None

        try:
            # Load model architecture and weights, but not the optimizer state
            from tensorflow.keras.models import load_model
            model = load_model(str(model_path), compile=False)

            # Recompile the model with a fresh optimizer
            from tensorflow.keras.optimizers import Adam
            model.compile(optimizer=Adam(), loss='mean_squared_error')

            # Load scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            return model, scaler
        except Exception as e:
            print(f"Error loading LSTM model for {symbol}: {str(e)}")
            return None, None

    def save_linear_model(self, symbol: str, model: LinearRegression, scaler: StandardScaler) -> str:
        """Save Linear Regression model and scaler to disk"""
        symbol_dir = self.ensure_symbol_dir(symbol)

        # Save model
        model_path = symbol_dir / "linear_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save scaler
        scaler_path = symbol_dir / "linear_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save metadata
        metadata = {
            "model_type": "LINEAR",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0"
        }

        metadata_path = symbol_dir / "linear_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        return str(model_path)

    def load_linear_model(self, symbol: str):
        """Load Linear Regression model and scaler from disk if they exist"""
        symbol_dir = self.ensure_symbol_dir(symbol)
        model_path = symbol_dir / "linear_model.pkl"
        scaler_path = symbol_dir / "linear_scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            return None, None

        try:
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Load scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            return model, scaler
        except Exception as e:
            print(f"Error loading Linear model for {symbol}: {str(e)}")
            return None, None

    def save_arima_history(self, symbol: str, history: list) -> str:
        """Save ARIMA history to disk"""
        symbol_dir = self.ensure_symbol_dir(symbol)

        # Save history
        history_path = symbol_dir / "arima_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        # Save metadata
        metadata = {
            "model_type": "ARIMA",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0",
            "history_length": len(history)
        }

        metadata_path = symbol_dir / "arima_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        return str(history_path)

    def load_arima_history(self, symbol: str):
        """Load ARIMA history from disk if it exists"""
        symbol_dir = self.ensure_symbol_dir(symbol)
        history_path = symbol_dir / "arima_history.pkl"

        if not history_path.exists():
            return None

        try:
            # Load history
            with open(history_path, 'rb') as f:
                history = pickle.load(f)

            return history
        except Exception as e:
            print(f"Error loading ARIMA history for {symbol}: {str(e)}")
            return None

    def model_exists(self, symbol: str, model_type: str) -> bool:
        """Check if model exists for the given symbol and type"""
        symbol_dir = self.ensure_symbol_dir(symbol)

        if model_type.upper() == "LSTM":
            return (symbol_dir / "lstm_model.h5").exists() and (symbol_dir / "lstm_scaler.pkl").exists()
        elif model_type.upper() == "LINEAR":
            return (symbol_dir / "linear_model.pkl").exists() and (symbol_dir / "linear_scaler.pkl").exists()
        elif model_type.upper() == "ARIMA":
            return (symbol_dir / "arima_history.pkl").exists()
        else:
            return False

    def get_model_metadata(self, symbol: str, model_type: str) -> dict:
        """Get metadata for the given model if it exists"""
        symbol_dir = self.ensure_symbol_dir(symbol)

        if model_type.upper() == "LSTM":
            metadata_path = symbol_dir / "lstm_metadata.json"
        elif model_type.upper() == "LINEAR":
            metadata_path = symbol_dir / "linear_metadata.json"
        elif model_type.upper() == "ARIMA":
            metadata_path = symbol_dir / "arima_metadata.json"
        else:
            return {}

        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
