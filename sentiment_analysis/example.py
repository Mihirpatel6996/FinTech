"""
Example Usage of Sentiment Analysis Module

This script demonstrates how to use the sentiment analysis module.
"""

import os
import logging
from sentiment_analysis.sentiment_service import SentimentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set your NewsAPI key (or set it as an environment variable)
    news_api_key = os.environ.get("NEWS_API_KEY", "your_api_key_here")

    # Initialize sentiment service
    service = SentimentService(news_api_key=news_api_key)

    # Analyze sentiment for a stock symbol
    symbol = "TSLA"
    days = 7

    logger.info(f"Analyzing sentiment for {symbol} over the past {days} days...")

    # Get overall sentiment
    overall_sentiment = service.get_overall_sentiment(symbol, days=days)
    logger.info(f"Overall sentiment for {symbol}: {overall_sentiment:.4f}")

    # Get detailed sentiment analysis
    sentiments = service.analyze_stock(symbol, days=days)
    logger.info(f"Found {len(sentiments)} news articles for {symbol}")

    # Print top 5 most positive articles
    logger.info("Top 5 most positive articles:")
    for article in sorted(sentiments, key=lambda x: x['sentiment'], reverse=True)[:5]:
        logger.info(f"Title: {article['title']}")
        logger.info(f"Sentiment: {article['sentiment']:.4f}")
        logger.info(f"Source: {article['source']}")
        logger.info(f"URL: {article['url']}")
        logger.info("---")

    # Print top 5 most negative articles
    logger.info("Top 5 most negative articles:")
    for article in sorted(sentiments, key=lambda x: x['sentiment'])[:5]:
        logger.info(f"Title: {article['title']}")
        logger.info(f"Sentiment: {article['sentiment']:.4f}")
        logger.info(f"Source: {article['source']}")
        logger.info(f"URL: {article['url']}")
        logger.info("---")

if __name__ == "__main__":
    main()
