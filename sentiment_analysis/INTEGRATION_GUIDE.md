# Integration Guide for Fintech Project

This guide provides instructions on how to integrate the sentiment analysis module with your Fintech project.

## Overview

The sentiment analysis module provides functionality to analyze sentiment in financial news related to specific stock symbols. This can be used as an additional feature in your prediction models (ARIMA, LR, LSTM, and ensemble).

## Integration Steps

### 1. Copy the Module

Copy the `sentiment_analysis` directory to your Fintech project.

### 2. Install Dependencies

Install the required dependencies:

```bash
pip install -r sentiment_analysis/requirements.txt
```

### 3. Set NewsAPI Key

Set your NewsAPI key as an environment variable:

```bash
export NEWS_API_KEY="2939063f6d8a4948aae8dfe4f9378d90"  
```

Or you can pass it directly when initializing the `SentimentService` class.

### 4. Import the Module

Import the `SentimentService` class in your prediction models:

```python
from sentiment_analysis import SentimentService
```

### 5. Add Sentiment Analysis to Your Models

#### Example for ARIMA Model

```python
def predict_with_arima(symbol, days=30):
    # Get sentiment data
    sentiment_service = SentimentService()
    sentiment_score = sentiment_service.get_overall_sentiment(symbol)
    
    # Get your existing ARIMA prediction
    arima_prediction = your_existing_arima_function(symbol, days)
    
    # Adjust prediction based on sentiment
    # This is a simple example - you might want to use a more sophisticated approach
    sentiment_factor = 1 + (sentiment_score * 0.1)  # Adjust by up to 10% based on sentiment
    adjusted_prediction = arima_prediction * sentiment_factor
    
    return adjusted_prediction
```

#### Example for LSTM Model

```python
def prepare_features_for_lstm(symbol, historical_data):
    # Get sentiment data
    sentiment_service = SentimentService()
    sentiment_score = sentiment_service.get_overall_sentiment(symbol)
    
    # Add sentiment as a feature
    features = your_existing_feature_preparation(historical_data)
    features['sentiment'] = sentiment_score
    
    return features
```

#### Example for Ensemble Model

```python
def ensemble_prediction(symbol, days=30):
    # Get predictions from individual models
    arima_pred = predict_with_arima(symbol, days)
    lr_pred = predict_with_lr(symbol, days)
    lstm_pred = predict_with_lstm(symbol, days)
    
    # Get sentiment data
    sentiment_service = SentimentService()
    sentiment_score = sentiment_service.get_overall_sentiment(symbol)
    
    # Adjust weights based on sentiment
    # This is a simple example - you might want to use a more sophisticated approach
    if sentiment_score > 0.5:  # Very positive sentiment
        weights = [0.2, 0.3, 0.5]  # Give more weight to LSTM
    elif sentiment_score < -0.5:  # Very negative sentiment
        weights = [0.4, 0.4, 0.2]  # Give more weight to ARIMA and LR
    else:  # Neutral sentiment
        weights = [0.33, 0.33, 0.34]  # Equal weights
    
    # Calculate weighted average
    ensemble_pred = (
        arima_pred * weights[0] +
        lr_pred * weights[1] +
        lstm_pred * weights[2]
    )
    
    return ensemble_pred
```

### 6. Test the Integration

Create a test script to verify that the integration works correctly:

```python
def test_integration():
    symbol = "TSLA"
    days = 30
    
    # Test individual models
    arima_pred = predict_with_arima(symbol, days)
    lr_pred = predict_with_lr(symbol, days)
    lstm_pred = predict_with_lstm(symbol, days)
    
    # Test ensemble model
    ensemble_pred = ensemble_prediction(symbol, days)
    
    print(f"ARIMA prediction: {arima_pred}")
    print(f"LR prediction: {lr_pred}")
    print(f"LSTM prediction: {lstm_pred}")
    print(f"Ensemble prediction: {ensemble_pred}")

if __name__ == "__main__":
    test_integration()
```

## Advanced Integration

For more advanced integration, you might want to:

1. Store sentiment data in your database for historical analysis
2. Create a daily job to fetch and analyze sentiment for your tracked symbols
3. Develop a more sophisticated model to incorporate sentiment data
4. Add sentiment visualization to your UI

## Troubleshooting

If you encounter issues with the integration:

1. Check that your NewsAPI key is valid and has sufficient quota
2. Verify that the sentiment analysis module is correctly installed
3. Check the logs for any error messages
4. Ensure that your models are correctly handling the sentiment data

For more help, refer to the README.md file in the sentiment_analysis directory.
