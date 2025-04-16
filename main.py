from fastapi import FastAPI, Request, Form, Cookie, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional
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
from sentiment_service import SentimentAnalysisService
from technical_indicators_ta import TechnicalIndicators
from portfolio_management import PortfolioManager
from risk_analysis import RiskAnalyzer
from enhanced_visualization import EnhancedVisualization
from user_management import UserManager

# Initialize all services
db = StockDatabase()
model_persistence = ModelPersistence()
ensemble_predictor = EnsemblePredictor(db)
sentiment_service = SentimentAnalysisService(db, use_mock=False)  # Use real NewsAPI service
portfolio_manager = PortfolioManager(db_path="stock_data.db")
risk_analyzer = RiskAnalyzer()
technical_indicator = TechnicalIndicators()
enhanced_viz = EnhancedVisualization()
user_manager = UserManager()

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
async def index(request: Request, search: str = None):
    # Load the stock symbols from CSV
    csv_path = "Yahoo-Finance-Ticker-Symbols.csv"
    try:
        stocks_df = pd.read_csv(csv_path)

        # Filter by search term if provided
        if search:
            search = search.upper()
            filtered_df = stocks_df[
                (stocks_df['Ticker'].str.contains(search, case=False, na=False)) |
                (stocks_df['Name'].str.contains(search, case=False, na=False))
            ]
        else:
            # If no search term, just show the first 100 stocks
            filtered_df = stocks_df.head(100)

        # Convert to list of dictionaries for the template
        stocks = filtered_df.to_dict('records')

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "stocks": stocks,
                "search": search,
                "total_stocks": len(stocks_df),
                "showing_stocks": len(stocks)
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Error loading stock data: {str(e)}"
            }
        )

@app.get("/stock/{symbol}", response_class=HTMLResponse)
async def stock_detail(request: Request, symbol: str):
    try:
        # Get stock information from CSV
        csv_path = "Yahoo-Finance-Ticker-Symbols.csv"
        stocks_df = pd.read_csv(csv_path)
        stock_info = stocks_df[stocks_df['Ticker'] == symbol].to_dict('records')

        if not stock_info:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error_message": f"Stock symbol {symbol} not found"}
            )

        company_info = stock_info[0]

        # Get historical data
        df = await get_historical(symbol)

        if df is None or df.empty:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error_message": f"No data found for {symbol}"}
            )

        # Calculate price change
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1]
        price_change = latest_price - prev_price
        price_change_percent = (price_change / prev_price) * 100 if prev_price > 0 else 0

        # Get 52-week high and low
        year_data = df.iloc[-252:] if len(df) > 252 else df
        year_high = year_data['High'].max()
        year_low = year_data['Low'].min()

        # Get technical indicators
        rsi = technical_indicator.calculate_rsi(df).iloc[-1]
        macd_line, signal_line, _ = technical_indicator.calculate_macd(df)
        macd_value = macd_line.iloc[-1]

        # Get moving averages
        ma_dict = technical_indicator.calculate_moving_averages(df)
        sma_50 = ma_dict['SMA_50'].iloc[-1] if 'SMA_50' in ma_dict else None
        sma_200 = ma_dict['SMA_200'].iloc[-1] if 'SMA_200' in ma_dict else None

        # Get risk metrics
        returns = risk_analyzer.calculate_returns(df['Close'])
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        var_value = risk_analyzer.calculate_value_at_risk(returns) * 100

        # Calculate beta (using S&P 500 as market)
        try:
            market_data = await get_historical("SPY")
            market_returns = risk_analyzer.calculate_returns(market_data['Close'])
            # Align the dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 0:
                stock_returns_aligned = returns.loc[common_dates]
                market_returns_aligned = market_returns.loc[common_dates]
                covariance = np.cov(stock_returns_aligned, market_returns_aligned)[0, 1]
                market_variance = np.var(market_returns_aligned)
                beta = covariance / market_variance if market_variance > 0 else 1.0
            else:
                beta = 1.0
        except:
            beta = 1.0

        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        annual_return = returns.mean() * 252
        sharpe_ratio = (annual_return - risk_free_rate) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Get sentiment analysis
        sentiment_data = sentiment_service.get_sentiment_for_symbol(symbol)
        sentiment_score = sentiment_service.get_average_sentiment(sentiment_data) if sentiment_data else 0

        # Determine sentiment description and color
        if sentiment_score > 0.2:
            sentiment_description = "Bullish"
            sentiment_color = "success"
        elif sentiment_score < -0.2:
            sentiment_description = "Bearish"
            sentiment_color = "danger"
        else:
            sentiment_description = "Neutral"
            sentiment_color = "secondary"

        # Get prediction data
        cached_predictions = db.get_cached_predictions(symbol)
        if cached_predictions:
            predictions = json.loads(cached_predictions) if isinstance(cached_predictions, str) else cached_predictions
            arima_pred = predictions.get("arima_pred", 0)
            lstm_pred = predictions.get("lstm_pred", 0)
            lr_pred = predictions.get("lr_pred", 0)
            ensemble_pred = predictions.get("ensemble_pred", (arima_pred + lstm_pred + lr_pred) / 3)
            prediction_accuracy = 100 - (predictions.get("error_arima", 0) + predictions.get("error_lstm", 0) + predictions.get("error_lr", 0)) / 3
        else:
            # Default values if no predictions available
            arima_pred = lstm_pred = lr_pred = ensemble_pred = latest_price
            prediction_accuracy = 0

        # Get portfolios containing this stock
        portfolios = portfolio_manager.get_portfolios_containing_stock(symbol)

        # Get all portfolios for the modal
        all_portfolios = portfolio_manager.get_portfolios()
        all_portfolios = all_portfolios.to_dict('records') if not all_portfolios.empty else []

        # Create price chart
        fig = plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Close'])
        plt.title(f"{symbol} Price History")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True)
        price_chart_path = save_visualization(symbol, "price_history", fig)

        return templates.TemplateResponse(
            "stock_detail.html",
            {
                "request": request,
                "symbol": symbol,
                "company_name": company_info.get('Name', symbol),
                "exchange": company_info.get('Exchange', ''),
                "category": company_info.get('Category Name', ''),
                "latest_price": round(latest_price, 2),
                "price_change": round(price_change, 2),
                "price_change_amount": abs(round(price_change, 2)),
                "price_change_percent": round(price_change_percent, 2),
                "latest_date": df.index[-1].strftime('%Y-%m-%d'),
                "open_price": round(df['Open'].iloc[-1], 2),
                "high_price": round(df['High'].iloc[-1], 2),
                "low_price": round(df['Low'].iloc[-1], 2),
                "volume": f"{int(df['Volume'].iloc[-1]):,}",
                "year_high": round(year_high, 2),
                "year_low": round(year_low, 2),
                "rsi_value": round(rsi, 2),
                "macd_value": round(macd_value, 2),
                "sma_50": round(sma_50, 2) if sma_50 is not None else "N/A",
                "sma_200": round(sma_200, 2) if sma_200 is not None else "N/A",
                "volatility": round(volatility, 2),
                "var_value": round(var_value, 2),
                "beta": round(beta, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "sentiment_score": sentiment_score,
                "sentiment_description": sentiment_description,
                "sentiment_color": sentiment_color,
                "sentiment_data": sentiment_data,
                "sentiment_articles_count": len(sentiment_data) if sentiment_data else 0,
                "arima_pred": round(arima_pred, 2),
                "lstm_pred": round(lstm_pred, 2),
                "lr_pred": round(lr_pred, 2),
                "ensemble_pred": round(ensemble_pred, 2),
                "prediction_accuracy": round(prediction_accuracy, 2),
                "portfolios": portfolios,
                "all_portfolios": all_portfolios,
                "price_chart_path": price_chart_path
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error loading stock details: {str(e)}"}
        )

@app.get("/education", response_class=HTMLResponse)
async def education(request: Request):
    return templates.TemplateResponse("education.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, message: str = None, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "message": message, "error": error})

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...), remember: bool = Form(default=False)):
    success, user_data, message = user_manager.authenticate_user(email, password)

    if success:
        # Create session
        session_id = user_manager.create_session(user_data["user_id"], expires_days=30 if remember else 1)

        # Create response with redirect
        response = RedirectResponse(url="/", status_code=303)

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=2592000 if remember else 86400)

        return response
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": message})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = None):
    return templates.TemplateResponse("register.html", {"request": request, "error": error})

@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, email: str = Form(...), password: str = Form(...), confirm_password: str = Form(...), first_name: str = Form(...), last_name: str = Form(...)):
    # Validate password match
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Passwords do not match"})

    # Register user
    success, message = user_manager.register_user(email, password, first_name, last_name)

    if success:
        # Redirect to login page with success message
        return RedirectResponse(url=f"/login?message={message}", status_code=303)
    else:
        return templates.TemplateResponse("register.html", {"request": request, "error": message})

@app.get("/logout")
async def logout(request: Request, session_id: str = Cookie(None)):
    if session_id:
        user_manager.end_session(session_id)

    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="session_id")

    return response

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

            # Get sentiment analysis
            sentiment_data = sentiment_service.analyze_sentiment(symbol)
            sentiment_score = sentiment_data.get("sentiment_score", 0.0)
            sentiment_description = sentiment_service.get_sentiment_description(sentiment_score)
            sentiment_color = sentiment_service.get_sentiment_color(sentiment_score)
            is_mock = sentiment_data.get("is_mock", False)

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
                "visualization_paths": viz_paths,
                "sentiment_score": sentiment_score,
                "sentiment_description": sentiment_description,
                "sentiment_color": sentiment_color,
                "sentiment_data": sentiment_data,
                "is_mock": is_mock
            }
            # Convert to JSON string before storing
            db.store_cached_predictions(symbol, json.dumps(predictions))

            latest_data = df.iloc[-1]

        # Add a function to update prediction accuracies for previous predictions
        await update_prediction_accuracies(symbol)

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "quote": symbol,
                "arima_pred": f"{predictions['arima_pred']:.2f}",
                "lstm_pred": f"{predictions['lstm_pred']:.2f}",
                "lr_pred": f"{predictions['lr_pred']:.2f}",
                "error_arima": f"{predictions['error_arima']:.2f}%",
                "error_lstm": f"{predictions['error_lstm']:.2f}%",
                "error_lr": f"{predictions['error_lr']:.2f}%",
                "open_s": f"{latest_data['Open']:.2f}",
                "close_s": f"{latest_data['Close']:.2f}",
                "adj_close": f"{latest_data['Adj Close']:.2f}",
                "high_s": f"{latest_data['High']:.2f}",
                "low_s": f"{latest_data['Low']:.2f}",
                "vol": f"{latest_data['Volume']:,}"
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


@app.post("/technical-indicators")
async def get_technical_indicators(request: Request, symbol: str = Form(...), indicator_type: str = Form("all")):
    """Get technical indicators for a stock"""
    try:
        # Get historical data
        df = await get_historical(symbol)

        if df is None or df.empty:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error_message": f"No data found for {symbol}"}
            )

        # Calculate indicators based on the requested type
        if indicator_type == "all":
            try:
                # Ensure the DataFrame has the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in df.columns:
                        return templates.TemplateResponse(
                            "error.html",
                            {"request": request, "error_message": f"Missing required column: {col}"}
                        )

                # Calculate all technical indicators
                indicators = technical_indicator.calculate_all_indicators(df)

                # Save visualizations
                viz_paths = {}

                # Generate interactive charts
                try:
                    # Create interactive candlestick chart
                    candlestick_html = enhanced_viz.create_interactive_chart_html(df, indicators, f"{symbol} Price Chart")

                    # Create interactive RSI chart
                    rsi = indicators.get('RSI')
                    if rsi is not None and not rsi.empty and not rsi.isna().all():
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots

                        # Create RSI figure
                        rsi_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                              vertical_spacing=0.03, subplot_titles=(f"{symbol} Price", "RSI"),
                                              row_heights=[0.7, 0.3])

                        # Add price chart
                        rsi_fig.add_trace(
                            go.Scatter(x=df.index, y=df['Close'], name='Close Price'),
                            row=1, col=1
                        )

                        # Add RSI
                        rsi_fig.add_trace(
                            go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')),
                            row=2, col=1
                        )

                        # Add RSI reference lines
                        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                        # Update layout
                        rsi_fig.update_layout(height=600, showlegend=True)
                        rsi_fig.update_yaxes(title_text="Price", row=1, col=1)
                        rsi_fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

                        # Convert to HTML
                        rsi_html = rsi_fig.to_html(include_plotlyjs=False, full_html=False)
                        viz_paths['RSI_html'] = rsi_html
                except Exception as e:
                    print(f"Error creating RSI chart: {str(e)}")

                try:
                    # Create interactive MACD chart
                    macd_line = indicators.get('MACD_line')
                    signal_line = indicators.get('MACD_signal')
                    histogram = indicators.get('MACD_histogram')

                    if all([macd_line is not None, signal_line is not None, histogram is not None]) and \
                       not macd_line.empty and not signal_line.empty and not histogram.empty:
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots

                        # Create MACD figure
                        macd_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                               vertical_spacing=0.03, subplot_titles=(f"{symbol} Price", "MACD"),
                                               row_heights=[0.7, 0.3])

                        # Add price chart
                        macd_fig.add_trace(
                            go.Scatter(x=df.index, y=df['Close'], name='Close Price'),
                            row=1, col=1
                        )

                        # Add MACD components
                        macd_fig.add_trace(
                            go.Scatter(x=df.index, y=macd_line, name='MACD Line', line=dict(color='blue')),
                            row=2, col=1
                        )
                        macd_fig.add_trace(
                            go.Scatter(x=df.index, y=signal_line, name='Signal Line', line=dict(color='red')),
                            row=2, col=1
                        )

                        # Add histogram as bar chart
                        colors = ['green' if val > 0 else 'red' for val in histogram]
                        macd_fig.add_trace(
                            go.Bar(x=df.index, y=histogram, name='Histogram', marker_color=colors),
                            row=2, col=1
                        )

                        # Update layout
                        macd_fig.update_layout(height=600, showlegend=True)
                        macd_fig.update_yaxes(title_text="Price", row=1, col=1)
                        macd_fig.update_yaxes(title_text="MACD", row=2, col=1)

                        # Convert to HTML
                        macd_html = macd_fig.to_html(include_plotlyjs=False, full_html=False)
                        viz_paths['MACD_html'] = macd_html
                except Exception as e:
                    print(f"Error creating MACD chart: {str(e)}")

                try:
                    # Create interactive Bollinger Bands chart
                    middle_band = indicators.get('BB_middle')
                    upper_band = indicators.get('BB_upper')
                    lower_band = indicators.get('BB_lower')

                    if all([middle_band is not None, upper_band is not None, lower_band is not None]) and \
                       not middle_band.empty and not upper_band.empty and not lower_band.empty:
                        import plotly.graph_objects as go

                        # Create Bollinger Bands figure
                        bb_fig = go.Figure()

                        # Add candlestick chart
                        bb_fig.add_trace(
                            go.Candlestick(
                                x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OHLC'
                            )
                        )

                        # Add Bollinger Bands
                        bb_fig.add_trace(go.Scatter(x=df.index, y=upper_band, name='Upper Band', line=dict(color='red', dash='dash')))
                        bb_fig.add_trace(go.Scatter(x=df.index, y=middle_band, name='Middle Band', line=dict(color='blue', dash='dash')))
                        bb_fig.add_trace(go.Scatter(x=df.index, y=lower_band, name='Lower Band', line=dict(color='green', dash='dash')))

                        # Fill between upper and lower bands
                        bb_fig.add_trace(go.Scatter(
                            x=df.index.tolist() + df.index.tolist()[::-1],
                            y=upper_band.tolist() + lower_band.tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=False
                        ))

                        # Update layout
                        bb_fig.update_layout(
                            title=f"{symbol} Bollinger Bands",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=600,
                            xaxis_rangeslider_visible=False
                        )

                        # Convert to HTML
                        bb_html = bb_fig.to_html(include_plotlyjs=False, full_html=False)
                        viz_paths['BB_html'] = bb_html
                except Exception as e:
                    print(f"Error creating Bollinger Bands chart: {str(e)}")

                try:
                    # Create interactive Moving Averages chart
                    ma_dict = {k: v for k, v in indicators.items() if k.startswith('SMA_') or k.startswith('EMA_')}

                    if ma_dict and any(not v.empty for v in ma_dict.values()):
                        import plotly.graph_objects as go

                        # Create Moving Averages figure
                        ma_fig = go.Figure()

                        # Add price chart
                        ma_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='black')))

                        # Add Moving Averages
                        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
                        for i, (name, values) in enumerate(ma_dict.items()):
                            if not values.empty and not values.isna().all():
                                ma_fig.add_trace(go.Scatter(
                                    x=df.index,
                                    y=values,
                                    name=name,
                                    line=dict(color=colors[i % len(colors)])
                                ))

                        # Update layout
                        ma_fig.update_layout(
                            title=f"{symbol} Moving Averages",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=600,
                            xaxis_rangeslider_visible=False
                        )

                        # Convert to HTML
                        ma_html = ma_fig.to_html(include_plotlyjs=False, full_html=False)
                        viz_paths['MA_html'] = ma_html
                except Exception as e:
                    print(f"Error creating Moving Averages chart: {str(e)}")

                # Prepare indicator values for display
                indicator_values = {}
                for name, values in indicators.items():
                    if isinstance(values, pd.Series) and not values.empty and not values.isna().all():
                        try:
                            indicator_values[name] = round(values.iloc[-1], 2)
                        except Exception as e:
                            print(f"Error processing indicator {name}: {str(e)}")

                # Generate technical analysis report
                try:
                    analysis_report = enhanced_viz.generate_technical_analysis_report(df, symbol)
                except Exception as e:
                    print(f"Error generating analysis report: {str(e)}")
                    analysis_report = {"overall_signal": "Neutral", "signals": {}}

                # Add candlestick chart to viz_paths
                viz_paths['candlestick_html'] = candlestick_html

                return templates.TemplateResponse(
                    "technical_indicators.html",
                    {
                        "request": request,
                        "symbol": symbol,
                        "indicators": indicator_values,
                        "viz_paths": viz_paths,
                        "analysis_report": analysis_report,
                        "latest_price": df['Close'].iloc[-1],
                        "latest_date": df.index[-1].strftime('%Y-%m-%d'),
                        "plotly_loaded": False  # Flag to indicate if Plotly JS is already loaded
                    }
                )
            except Exception as e:
                print(f"Error calculating technical indicators: {str(e)}")
                return templates.TemplateResponse(
                    "error.html",
                    {"request": request, "error_message": f"Error calculating technical indicators: {str(e)}"}
                )
        else:
            # Handle specific indicator types
            if indicator_type == "rsi":
                rsi = technical_indicator.calculate_rsi(df)
                fig = technical_indicator.plot_rsi(df, rsi)
                viz_path = save_visualization(symbol, "rsi", fig)
                return templates.TemplateResponse(
                    "single_indicator.html",
                    {
                        "request": request,
                        "symbol": symbol,
                        "indicator_type": "RSI",
                        "indicator_value": round(rsi.iloc[-1], 2),
                        "viz_path": viz_path,
                        "latest_price": df['Close'].iloc[-1],
                        "latest_date": df.index[-1].strftime('%Y-%m-%d')
                    }
                )
            elif indicator_type == "macd":
                macd_line, signal_line, histogram = technical_indicator.calculate_macd(df)
                fig = technical_indicator.plot_macd(df, macd_line, signal_line, histogram)
                viz_path = save_visualization(symbol, "macd", fig)
                return templates.TemplateResponse(
                    "single_indicator.html",
                    {
                        "request": request,
                        "symbol": symbol,
                        "indicator_type": "MACD",
                        "indicator_value": round(macd_line.iloc[-1], 2),
                        "signal_value": round(signal_line.iloc[-1], 2),
                        "histogram_value": round(histogram.iloc[-1], 2),
                        "viz_path": viz_path,
                        "latest_price": df['Close'].iloc[-1],
                        "latest_date": df.index[-1].strftime('%Y-%m-%d')
                    }
                )
            elif indicator_type == "bollinger":
                middle_band, upper_band, lower_band = technical_indicator.calculate_bollinger_bands(df)
                fig = technical_indicator.plot_bollinger_bands(df, middle_band, upper_band, lower_band)
                viz_path = save_visualization(symbol, "bollinger", fig)
                return templates.TemplateResponse(
                    "single_indicator.html",
                    {
                        "request": request,
                        "symbol": symbol,
                        "indicator_type": "Bollinger Bands",
                        "middle_band": round(middle_band.iloc[-1], 2),
                        "upper_band": round(upper_band.iloc[-1], 2),
                        "lower_band": round(lower_band.iloc[-1], 2),
                        "viz_path": viz_path,
                        "latest_price": df['Close'].iloc[-1],
                        "latest_date": df.index[-1].strftime('%Y-%m-%d')
                    }
                )
            elif indicator_type == "moving_averages":
                ma_dict = technical_indicator.calculate_moving_averages(df)
                fig = technical_indicator.plot_moving_averages(df, ma_dict)
                viz_path = save_visualization(symbol, "moving_averages", fig)
                ma_values = {name: round(values.iloc[-1], 2) for name, values in ma_dict.items()}
                return templates.TemplateResponse(
                    "single_indicator.html",
                    {
                        "request": request,
                        "symbol": symbol,
                        "indicator_type": "Moving Averages",
                        "ma_values": ma_values,
                        "viz_path": viz_path,
                        "latest_price": df['Close'].iloc[-1],
                        "latest_date": df.index[-1].strftime('%Y-%m-%d')
                    }
                )
            else:
                return templates.TemplateResponse(
                    "error.html",
                    {"request": request, "error_message": f"Unknown indicator type: {indicator_type}"}
                )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error calculating technical indicators: {str(e)}"}
        )


@app.get("/portfolio")
async def get_portfolio_page(request: Request):
    """Get portfolio management page"""
    try:
        # Get all portfolios
        portfolios = portfolio_manager.get_portfolios()

        return templates.TemplateResponse(
            "portfolio.html",
            {
                "request": request,
                "portfolios": portfolios.to_dict(orient="records") if not portfolios.empty else []
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error loading portfolios: {str(e)}"}
        )


@app.post("/portfolio/create")
async def create_portfolio(request: Request, name: str = Form(...), description: str = Form("")):
    """Create a new portfolio"""
    try:
        portfolio_id = portfolio_manager.create_portfolio(name, description)
        return RedirectResponse(url=f"/portfolio/{portfolio_id}", status_code=303)
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error creating portfolio: {str(e)}"}
        )


@app.get("/portfolio/{portfolio_id}")
async def get_portfolio_details(request: Request, portfolio_id: int):
    """Get portfolio details"""
    try:
        # Get portfolio holdings
        holdings = portfolio_manager.get_portfolio_holdings(portfolio_id)

        # Get current prices for all symbols in the portfolio
        current_prices = {}
        for _, row in holdings.iterrows():
            symbol = row['symbol']
            try:
                # Get latest price
                df = await get_historical(symbol)
                if df is not None and not df.empty:
                    current_prices[symbol] = df['Close'].iloc[-1]
            except Exception as e:
                print(f"Error getting price for {symbol}: {str(e)}")
                current_prices[symbol] = 0.0

        # Calculate portfolio value and allocation
        portfolio_value = portfolio_manager.get_portfolio_value(portfolio_id, current_prices)
        allocation = portfolio_manager.calculate_portfolio_allocation(portfolio_id, current_prices)

        # Create allocation chart
        allocation_chart = portfolio_manager.plot_portfolio_allocation(portfolio_id, current_prices)
        allocation_chart_path = save_visualization(f"portfolio_{portfolio_id}", "allocation", allocation_chart)

        # Get portfolio transactions
        transactions = portfolio_manager.get_portfolio_transactions(portfolio_id)

        return templates.TemplateResponse(
            "portfolio_details.html",
            {
                "request": request,
                "portfolio_id": portfolio_id,
                "holdings": portfolio_value["holdings"],
                "total_value": portfolio_value["total_value"],
                "total_cost": portfolio_value["total_cost"],
                "total_gain_loss": portfolio_value["total_gain_loss"],
                "total_gain_loss_percent": portfolio_value["total_gain_loss_percent"],
                "allocation": allocation["allocations"],
                "allocation_chart_path": allocation_chart_path,
                "transactions": transactions.to_dict(orient="records") if not transactions.empty else []
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error loading portfolio details: {str(e)}"}
        )


@app.post("/portfolio/{portfolio_id}/add-holding")
async def add_portfolio_holding(request: Request, portfolio_id: int, symbol: str = Form(...), quantity: float = Form(...), purchase_price: float = Form(...), purchase_date: str = Form(None)):
    """Add a holding to a portfolio"""
    try:
        portfolio_manager.add_holding(portfolio_id, symbol, quantity, purchase_price, purchase_date)
        return RedirectResponse(url=f"/portfolio/{portfolio_id}", status_code=303)
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error adding holding: {str(e)}"}
        )


@app.post("/portfolio/{portfolio_id}/remove-holding")
async def remove_portfolio_holding(request: Request, portfolio_id: int, symbol: str = Form(...), quantity: float = Form(...), sell_price: float = Form(...), sell_date: str = Form(None)):
    """Remove a holding from a portfolio"""
    try:
        success = portfolio_manager.remove_holding(portfolio_id, symbol, quantity, sell_price, sell_date)
        if not success:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error_message": f"Could not remove holding. Check if you have enough shares."}
            )
        return RedirectResponse(url=f"/portfolio/{portfolio_id}", status_code=303)
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error removing holding: {str(e)}"}
        )


@app.post("/risk-analysis")
async def get_risk_analysis(request: Request, symbol: str = Form(...)):
    """Get risk analysis for a stock"""
    try:
        # Get historical data
        df = await get_historical(symbol)

        if df is None or df.empty:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error_message": f"No data found for {symbol}"}
            )

        # Calculate returns
        returns = risk_analyzer.calculate_returns(df['Close'])

        # Calculate risk metrics
        risk_metrics = risk_analyzer.calculate_risk_metrics(returns)

        # Create VaR plot
        var_fig = risk_analyzer.plot_var(returns)
        var_path = save_visualization(symbol, "var", var_fig)

        # Create drawdown plot
        drawdown_fig = risk_analyzer.plot_drawdown(returns)
        drawdown_path = save_visualization(symbol, "drawdown", drawdown_fig)

        return templates.TemplateResponse(
            "risk_analysis.html",
            {
                "request": request,
                "symbol": symbol,
                "risk_metrics": risk_metrics,
                "var_path": var_path,
                "drawdown_path": drawdown_path,
                "latest_price": df['Close'].iloc[-1],
                "latest_date": df.index[-1].strftime('%Y-%m-%d'),
                "annual_return": returns.mean() * 252 * 100,  # Annualized return in percentage
                "total_return": ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100  # Total return in percentage
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": f"Error calculating risk analysis: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

