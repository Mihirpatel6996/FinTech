# Stock Market Prediction Web App

A FastAPI-based web application that predicts stock market prices using machine learning algorithms (ARIMA, LSTM, and Linear Regression).

## Features
- Real-time stock data fetching
- Price predictions using multiple algorithms
- 7-day price forecasting
- Simple and intuitive interface

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-prediction-app.git
cd stock-prediction-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

4. Open http://localhost:8000 in your browser

## Usage
1. Enter a valid stock symbol (e.g., AAPL, GOOGL)
2. Click "Predict" to see the forecast
3. View current stock data and predictions

## Built With
- FastAPI
- Python
- scikit-learn
- TensorFlow/Keras
- yfinance
- Bootstrap

## License
MIT License - see LICENSE file for details
