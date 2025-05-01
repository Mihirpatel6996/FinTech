# Stock Market Prediction Web App

A FastAPI-based web application that predicts stock market prices using machine learning algorithms (ARIMA, LSTM, and Linear Regression).

## Features
- Real-time stock data fetching
- Price predictions using multiple algorithms
- 7-day price forecasting
- Simple and intuitive interface
- Sentiment analysis of financial news

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Mihirpatel6996/FinTech.git
cd FinTech
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (if needed):
```bash
# For sentiment analysis
export NEWS_API_KEY="your_api_key_here"
# On Windows: set NEWS_API_KEY=your_api_key_here
```

5. Run the application:
```bash
uvicorn main:app --reload
```

6. Open http://localhost:8000 in your browser

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

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit the `.env` file and add your API keys:
```
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key

# NewsAPI Key
NEWS_API_KEY=your_news_api_key
```

These API keys are required for:
- Google Gemini API: Powering the AI chatbot
- NewsAPI: Fetching financial news for sentiment analysis


