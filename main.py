from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import warnings
import time
import json
import sqlite3
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from database import StockDatabase
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from model_persistence import ModelPersistence
from ensemble import EnsemblePredictor

# Initialize database and model persistence
db = StockDatabase()
model_persistence = ModelPersistence()
ensemble_predictor = EnsemblePredictor(db)

# Delete existing database file if you want to start fresh (optional)
# if os.path.exists("stock_data.db"):
#     os.remove("stock_data.db")

app = FastAPI()

# Mount static files - make sure this comes BEFORE templates initialization
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

async def get_historical(quote: str) -> pd.DataFrame:
    try:
        end = datetime.now()
        start = datetime(end.year-2, end.month, end.day)

        # Add debug logging
        print(f"Fetching data from {start} to {end}")
        print(f"Training period: {(end - start).days} days")

        # First check cache
        if not db.needs_update(quote):
            stored_data = await db.get_stock_data(quote, start, end)
            if stored_data is not None:
                print(f"Using cached data for {quote}")
                return stored_data

        print(f"Fetching new data for {quote}")

        # Use synchronous approach instead of asyncio.to_thread
        def fetch_data():
            try:
                return yf.download(
                    quote,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False  # Disable multi-threading
                )
            except Exception as e:
                print(f"Download error: {str(e)}")
                return None

        # Try multiple times with increasing delays
        max_retries = 5
        delays = [1, 2, 4, 8, 16]  # Exponential backoff

        for attempt, delay in enumerate(delays[:max_retries]):
            try:
                # Run in executor to prevent blocking
                data = await asyncio.get_event_loop().run_in_executor(None, fetch_data)

                if data is not None and not data.empty:
                    # Process the data
                    df = pd.DataFrame(data)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # Ensure required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = df['Close'] if col == 'Adj Close' else 0

                    # Store and return
                    db.store_stock_data(quote, df)
                    return df

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")

            print(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

        raise ValueError(f"Failed to fetch data for {quote} after {max_retries} attempts")

    except Exception as e:
        print(f"Final error in get_historical: {str(e)}")
        raise ValueError(f"Unable to fetch data for {quote}. Please try again later.")

def download_with_retry(quote, **kwargs):
    try:
        data = yf.download(quote, **kwargs)
        return data
    except Exception as e:
        print(f"Error in download_with_retry: {str(e)}")
        raise e



def arima_model(train, test, symbol=None, model_persistence=None):
    # Check if we have a saved history for incremental learning
    history = None
    incremental = False

    if symbol and model_persistence and model_persistence.model_exists(symbol, "ARIMA"):
        # Load saved history for incremental learning
        saved_history = model_persistence.load_arima_history(symbol)
        if saved_history:
            history = saved_history
            incremental = True
            print(f"Using incremental learning for ARIMA model on {symbol}")

    # If no saved history or loading failed, use the training data
    if not history:
        history = [x[0] for x in train]  # Extract values from numpy array

    predictions = []

    for t in range(len(test)):
        try:
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t][0]  # Extract value from numpy array
            history.append(obs)
        except Exception as e:
            print(f"Error in ARIMA prediction step {t}: {str(e)}")
            # If prediction fails, use last known value
            yhat = history[-1] if history else 0
            predictions.append(yhat)
            history.append(test[t][0])

    # Save the updated history for future incremental learning
    if symbol and model_persistence:
        model_persistence.save_arima_history(symbol, history)

    return predictions, incremental

def ensure_stock_dir(symbol: str) -> str:
    """Create and return the path to stock-specific directory"""
    stock_dir = Path(f"static/{symbol}")
    stock_dir.mkdir(parents=True, exist_ok=True)
    return str(stock_dir)

def save_visualization(symbol: str, model_type: str, fig: plt.Figure) -> str:
    """Save visualization and return the relative path"""
    stock_dir = ensure_stock_dir(symbol)
    file_name = f"{model_type.lower()}.png"
    file_path = f"{symbol}/{file_name}"
    full_path = os.path.join("static", file_path)

    fig.savefig(full_path)
    plt.close(fig)
    return file_path

def ARIMA_ALGO(df, symbol: str, model_persistence=None):
    try:
        df = df.dropna()

        # Ensure index is datetime and create a copy to avoid modifying the original
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Create Date_Str column from index
        df['Date_Str'] = df.index.strftime('%Y-%m-%d')
        df['Code'] = df['Close']

        # Prepare data for ARIMA
        data = df.copy()
        data['Price'] = data['Close']

        # Create Quantity_date DataFrame
        Quantity_date = pd.DataFrame(data['Price'])
        Quantity_date = Quantity_date.fillna(method='bfill')

        # Plot trends
        fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(Quantity_date)
        plt.title('Stock Price Trends')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        trends_path = save_visualization(symbol, "Trends", fig)
        db.store_visualization_path(symbol, "Trends", trends_path)

        # Prepare training data
        quantity = Quantity_date.values
        size = int(len(quantity) * 0.80)
        train, test = quantity[0:size], quantity[size:len(quantity)]

        # Use incremental learning if model_persistence is provided
        predictions, incremental = arima_model(train, test, symbol, model_persistence)

        # Plot ARIMA predictions
        fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(test, label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.title('ARIMA Predictions' + (' (Incremental)' if incremental else ''))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc=4)
        plt.tight_layout()
        arima_path = save_visualization(symbol, "ARIMA", fig)
        db.store_visualization_path(symbol, "ARIMA", arima_path)

        arima_pred = predictions[-2]
        error_arima = math.sqrt(mean_squared_error(test, predictions))

        # Store training information
        if db and model_persistence:
            db.store_model_training(
                symbol=symbol,
                model_type="ARIMA",
                data_points=len(train),
                incremental=incremental,
                training_error=error_arima
            )

        return arima_pred, error_arima
    except Exception as e:
        print(f"ARIMA Error Details: {str(e)}")  # Add detailed error logging
        raise ValueError(f"Error in ARIMA algorithm: {str(e)}")

def LSTM_ALGO(df, symbol: str, model_persistence=None):
    try:
        dataset_train = df.iloc[0:int(0.8*len(df)),:]
        dataset_test = df.iloc[int(0.8*len(df)):,:]

        # Use 'Close' column for training
        training_set = df['Close'].values.reshape(-1, 1)

        # Check if we have a saved model for incremental learning
        incremental = False
        regressor = None
        sc = None

        if model_persistence and model_persistence.model_exists(symbol, "LSTM"):
            # Load saved model and scaler
            regressor, sc = model_persistence.load_lstm_model(symbol)
            if regressor and sc:
                incremental = True
                print(f"Using incremental learning for LSTM model on {symbol}")

        # If no saved model or loading failed, create a new one
        if not regressor or not sc:
            sc = MinMaxScaler(feature_range=(0,1))
            training_set_scaled = sc.fit_transform(training_set)

            X_train = []
            y_train = []
            for i in range(7, len(training_set_scaled)):
                X_train.append(training_set_scaled[i-7:i,0])
                y_train.append(training_set_scaled[i,0])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Reshape for LSTM [samples, time steps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Build LSTM model
            regressor = Sequential()
            regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.2))
            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.2))
            regressor.add(Dense(units=1))
            regressor.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            regressor.fit(X_train, y_train, epochs=25, batch_size=32)
        else:
            # For incremental learning, we still need to transform the data
            training_set_scaled = sc.transform(training_set)

            # Prepare incremental training data
            X_train = []
            y_train = []
            for i in range(7, len(training_set_scaled)):
                X_train.append(training_set_scaled[i-7:i,0])
                y_train.append(training_set_scaled[i,0])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Reshape for LSTM [samples, time steps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # For incremental learning with LSTM, it's better to create a new model with the same architecture
            # and initialize it with the weights from the saved model, then train it on new data
            # This avoids issues with the optimizer state

            # Get the weights from the loaded model
            old_weights = regressor.get_weights()

            # Create a new model with the same architecture
            new_regressor = Sequential()
            new_regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            new_regressor.add(Dropout(0.2))
            new_regressor.add(LSTM(units=50, return_sequences=True))
            new_regressor.add(Dropout(0.2))
            new_regressor.add(LSTM(units=50))
            new_regressor.add(Dropout(0.2))
            new_regressor.add(Dense(units=1))
            new_regressor.compile(optimizer='adam', loss='mean_squared_error')

            # Set the weights from the old model
            new_regressor.set_weights(old_weights)

            # Train the new model on the new data
            new_regressor.fit(X_train, y_train, epochs=5, batch_size=32)

            # Replace the old model with the new one
            regressor = new_regressor

        # Save the model for future incremental learning
        if model_persistence:
            model_persistence.save_lstm_model(symbol, regressor, sc)

        # Prepare test data
        real_stock_price = dataset_test['Close'].values.reshape(-1, 1)

        # Get the predicted stock price
        dataset_total = pd.concat([dataset_train['Close'], dataset_test['Close']], axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)

        X_test = []
        for i in range(7, len(inputs)):
            X_test.append(inputs[i-7:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        # Prepare forecast data
        last_seven_days = training_set_scaled[-7:]
        X_forecast = np.array([last_seven_days[:,0]])
        X_forecast = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))

        # Plot LSTM predictions
        fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')
        plt.title('LSTM Predictions' + (' (Incremental)' if incremental else ''))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc=4)
        plt.tight_layout()
        lstm_path = save_visualization(symbol, "LSTM", fig)
        db.store_visualization_path(symbol, "LSTM", lstm_path)

        error_lstm = float(math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price)))

        # Get the forecast
        forecasted_stock_price = regressor.predict(X_forecast)
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
        lstm_pred = float(forecasted_stock_price[0,0])

        # Store training information
        if db and model_persistence:
            db.store_model_training(
                symbol=symbol,
                model_type="LSTM",
                data_points=len(dataset_train),
                incremental=incremental,
                training_error=error_lstm
            )

        return lstm_pred, error_lstm

    except Exception as e:
        print(f"Error in LSTM algorithm: {str(e)}")
        raise ValueError(f"Error in LSTM prediction: {str(e)}")

def LIN_REG_ALGO(df, symbol: str, model_persistence=None):
    forecast_out = int(7)
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Close', 'Close after n days']]

    y = np.array(df_new.iloc[:-forecast_out,-1])
    y = np.reshape(y, (-1,1))
    X = np.array(df_new.iloc[:-forecast_out,0:-1])
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:,0:-1])

    X_train = X[0:int(0.8*len(df)),:]
    X_test = X[int(0.8*len(df)):,:]
    y_train = y[0:int(0.8*len(df)),:]
    y_test = y[int(0.8*len(df)):,:]

    # Check if we have a saved model for incremental learning
    incremental = False
    clf = None
    sc = None

    if model_persistence and model_persistence.model_exists(symbol, "LINEAR"):
        # Load saved model and scaler
        clf, sc = model_persistence.load_linear_model(symbol)
        if clf and sc:
            incremental = True
            print(f"Using incremental learning for Linear Regression model on {symbol}")

    # If no saved model or loading failed, create a new one
    if not clf or not sc:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_to_be_forecasted = sc.transform(X_to_be_forecasted)

        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
    else:
        # For incremental learning, we still need to transform the data with the saved scaler
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
        X_to_be_forecasted = sc.transform(X_to_be_forecasted)

        # Update the model with new data (incremental learning)
        # For linear regression, we can just fit again with the new data
        # This is a simple form of incremental learning
        clf.fit(X_train, y_train)

    # Save the model for future incremental learning
    if model_persistence:
        model_persistence.save_linear_model(symbol, clf, sc)

    y_test_pred = clf.predict(X_test)
    y_test_pred = y_test_pred*(1.04)

    # Plot Linear Regression predictions with title and labels
    fig = plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_test_pred, label='Predicted Price')
    plt.title('Linear Regression Predictions' + (' (Incremental)' if incremental else ''))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc=4)
    plt.tight_layout()
    lr_path = save_visualization(symbol, "LR", fig)
    db.store_visualization_path(symbol, "LR", lr_path)

    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set*(1.04)
    mean = forecast_set.mean()
    lr_pred = forecast_set[0]

    # Store training information
    if db and model_persistence:
        db.store_model_training(
            symbol=symbol,
            model_type="LINEAR",
            data_points=len(X_train),
            incremental=incremental,
            training_error=error_lr
        )

    return df, lr_pred, forecast_set, mean, error_lr

async def store_model_prediction(symbol: str, model_type: str, prediction: float, current_date: datetime):
    try:
        db.store_prediction(
            symbol=symbol,
            model_type=model_type,
            prediction=prediction,
            target_date=current_date + timedelta(days=1)
        )
    except Exception as e:
        print(f"Warning: Failed to store {model_type} prediction for {symbol}: {e}")
        # Continue execution despite storage failure

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/education", response_class=HTMLResponse)
async def education(request: Request):
    return templates.TemplateResponse("education.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, symbol: str = Form(...)):
    try:
        # Increase timeout and add user feedback
        timeout = 45  # Increased timeout

        try:
            df = await asyncio.wait_for(get_historical(symbol), timeout=timeout)
        except asyncio.TimeoutError:
            return templates.TemplateResponse("index.html",
                                           {"request": request,
                                            "error": "Request timed out. The server might be busy, please try again."})
        except ValueError as ve:
            return templates.TemplateResponse("index.html",
                                           {"request": request,
                                            "error": str(ve)})

        if df.empty:
            return templates.TemplateResponse("index.html",
                                           {"request": request,
                                            "error": f"No data available for {symbol}"})

        cached_predictions = db.get_cached_predictions(symbol)
        current_date = datetime.now()

        if not db.needs_prediction_update(symbol) and cached_predictions:
            print(f"Using cached predictions for {symbol}")
            predictions = json.loads(cached_predictions) if isinstance(cached_predictions, str) else cached_predictions

            # Get the latest stock data
            df = await get_historical(symbol)
            latest_data = df.iloc[-1]

            # Store predictions in history table with error handling
            await store_model_prediction(symbol, "ARIMA", predictions["arima_pred"], current_date)
            await store_model_prediction(symbol, "LSTM", predictions["lstm_pred"], current_date)
            await store_model_prediction(symbol, "LINEAR", predictions["lr_pred"], current_date)

        else:
            print(f"Generating new predictions for {symbol}")

            # Get historical data
            df = await get_historical(symbol)
            if df.empty:
                return templates.TemplateResponse("index.html",
                                               {"request": request,
                                                "error": f"No data available for {symbol}"})

            # Run predictions with symbol parameter and model persistence
            arima_pred, error_arima = ARIMA_ALGO(df, symbol, model_persistence)
            lstm_pred, error_lstm = LSTM_ALGO(df, symbol, model_persistence)
            df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df, symbol, model_persistence)

            # Generate ensemble prediction
            model_predictions = {
                "arima_pred": float(arima_pred),
                "lstm_pred": float(lstm_pred),
                "lr_pred": float(lr_pred),
                "error_arima": float(error_arima),
                "error_lstm": float(error_lstm),
                "error_lr": float(error_lr)
            }

            # Get ensemble prediction
            ensemble_pred = ensemble_predictor.ensemble_prediction(symbol, model_predictions)

            # Create ensemble visualization
            real_values = df['Close'].values[-len(forecast_set):]
            dates = df.index[-len(forecast_set):]
            ensemble_path = ensemble_predictor.create_ensemble_visualization(
                symbol, model_predictions, real_values, dates)

            # Store predictions in history table with error handling
            await store_model_prediction(symbol, "ARIMA", arima_pred, current_date)
            await store_model_prediction(symbol, "LSTM", lstm_pred, current_date)
            await store_model_prediction(symbol, "LINEAR", lr_pred, current_date)
            await store_model_prediction(symbol, "ENSEMBLE", ensemble_pred, current_date)

            # Get visualization paths
            viz_paths = db.get_visualization_paths(symbol)

            # Store predictions
            predictions = {
                "symbol": symbol,
                "arima_pred": float(arima_pred),
                "lstm_pred": float(lstm_pred),
                "lr_pred": float(lr_pred),
                "ensemble_pred": float(ensemble_pred),
                "error_arima": float(error_arima),
                "error_lstm": float(error_lstm),
                "error_lr": float(error_lr),
                "forecast_set": forecast_set.tolist(),
                "mean": float(mean),
                "visualization_paths": viz_paths
            }
            # Convert to JSON string before storing
            db.store_cached_predictions(symbol, json.dumps(predictions))

            latest_data = df.iloc[-1]

        # Add a function to update prediction accuracies for previous predictions
        await update_prediction_accuracies(symbol)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": True,
                "symbol": symbol,
                "arima_pred": predictions["arima_pred"],
                "lstm_pred": predictions["lstm_pred"],
                "lr_pred": predictions["lr_pred"],
                "ensemble_pred": predictions.get("ensemble_pred", (predictions["arima_pred"] + predictions["lstm_pred"] + predictions["lr_pred"]) / 3),
                "error_arima": predictions["error_arima"],
                "error_lstm": predictions["error_lstm"],
                "error_lr": predictions["error_lr"],
                "open_s": f"{latest_data['Open']:.2f}",
                "close_s": f"{latest_data['Close']:.2f}",
                "adj_close": f"{latest_data['Adj Close']:.2f}",
                "high_s": f"{latest_data['High']:.2f}",
                "low_s": f"{latest_data['Low']:.2f}",
                "vol": f"{latest_data['Volume']:,}",
                "viz_paths": predictions["visualization_paths"]
            }
        )
    except asyncio.TimeoutError:
        return templates.TemplateResponse("index.html",
                                        {"request": request,
                                         "error": "Request timed out. Please try again."})
    except Exception as e:
        error_msg = f"An error occurred while processing {symbol}: {str(e)}"
        print(error_msg)  # Log the error
        return templates.TemplateResponse("index.html",
                                        {"request": request,
                                         "error": error_msg})

async def update_prediction_accuracies(symbol: str):
    """Update accuracy for predictions made in the past"""
    try:
        # Get today's actual closing price
        df = await get_historical(symbol)
        if df.empty:
            return

        current_close = df.iloc[-1]['Close']
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Use the database instance directly instead of creating a new connection
        # This avoids the need for the sqlite3 import in this function
        predictions = db.get_pending_predictions(symbol, yesterday)

        # Update each prediction with actual value and error rate
        for pred_date, model_type, pred_value in predictions:
            db.update_prediction_accuracy(
                symbol=symbol,
                actual_value=current_close,
                prediction_date=datetime.strptime(pred_date, '%Y-%m-%d %H:%M:%S'),
                model_type=model_type
            )

    except Exception as e:
        print(f"Error updating prediction accuracies: {str(e)}")

async def get_alternative_data(symbol: str, start_date, end_date):
    """Fallback method using alternative data source"""
    try:
        # You can implement alternative data sources here
        # For example, using Alpha Vantage, IEX Cloud, or other providers
        pass
    except Exception as e:
        print(f"Alternative data source failed: {str(e)}")
        return None

@app.get("/health")
async def health_check():
    try:
        # Try to fetch a known stock symbol
        test_symbol = "AAPL"
        df = await asyncio.wait_for(get_historical(test_symbol), timeout=10)
        return {"status": "healthy", "message": "Service is running normally"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

