# Financial News Sentiment Analysis

This module provides functionality to analyze sentiment in financial news related to specific stock symbols. It can be integrated into your "Fintech" project to add sentiment analysis capabilities.

## Features

- Fetch financial news articles related to stock symbols
- Analyze sentiment in text using both VADER and TextBlob
- Calculate overall sentiment for a stock symbol
- Clean and process text for better sentiment analysis

## Installation

1. Copy the `sentiment_analysis` directory to your "Fintech" project
2. Install the required dependencies:

```bash
pip install -r sentiment_analysis/requirements.txt
```

3. Set your NewsAPI key as an environment variable:

```bash
export NEWS_API_KEY="your_api_key_here"
```

Or you can pass it directly when initializing the `SentimentService` class.

## Usage

### Basic Usage

```python
from sentiment_analysis import SentimentService

# Initialize sentiment service
service = SentimentService()

# Analyze sentiment for a stock symbol
symbol = "TSLA"
sentiments = service.analyze_stock(symbol)

# Get overall sentiment
overall_sentiment = service.get_overall_sentiment(symbol)
print(f"Overall sentiment for {symbol}: {overall_sentiment:.4f}")
```

### Analyzing Custom Text

```python
from sentiment_analysis import SentimentService

# Initialize sentiment service
service = SentimentService()

# Analyze sentiment in custom text
text = "This company is performing exceptionally well with strong growth prospects."
sentiment = service.analyze_text(text)
print(f"Sentiment: {sentiment:.4f}")
```

## Integration with Fintech Project

To integrate this module with your "Fintech" project:

1. Import the `SentimentService` class in your prediction models:

```python
from sentiment_analysis import SentimentService
```

2. Initialize the service and get sentiment data:

```python
def predict_stock_price(symbol, days=30):
    # Get sentiment data
    sentiment_service = SentimentService()
    sentiment_score = sentiment_service.get_overall_sentiment(symbol)
    
    # Add sentiment score as a feature to your prediction models
    # ...
```

3. Use the sentiment score as an additional feature in your ARIMA, LR, LSTM, and ensemble models.

## Example

See `example.py` for a complete example of how to use the sentiment analysis module.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
