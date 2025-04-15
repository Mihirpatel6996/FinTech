"""
Sentiment Analysis Service

This module provides functionality to analyze sentiment in financial news
and text related to specific stock symbols.
"""

import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .news_service import NewsService

# Configure logging
logger = logging.getLogger(__name__)

class SentimentService:
    """
    Service for analyzing sentiment in financial news and text.
    
    This class provides methods to analyze sentiment in text using both
    VADER and TextBlob sentiment analyzers, and to analyze sentiment in
    news articles related to a specific stock symbol.
    """
    
    def __init__(self, news_api_key=None):
        """
        Initialize the sentiment service.
        
        Args:
            news_api_key (str, optional): API key for NewsAPI. If not provided,
                                         it will be read from config.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.news_service = NewsService(api_key=news_api_key)
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment in text using both VADER and TextBlob.
        
        Args:
            text (str): Text to analyze.
            
        Returns:
            float: Sentiment score between -1 (negative) and 1 (positive).
        """
        try:
            # Get VADER sentiment
            vader_sentiment = self.analyzer.polarity_scores(text)
            
            # Get TextBlob sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity

            # Combine both scores
            combined_score = (vader_sentiment['compound'] + textblob_sentiment) / 2
            return combined_score
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0
    
    def analyze_stock(self, symbol: str, days: int = 7) -> list:
        """
        Analyze sentiment in news articles related to a stock symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'TSLA').
            days (int, optional): Number of days of news to analyze. Defaults to 7.
            
        Returns:
            list: List of dictionaries containing news articles and their sentiment scores.
        """
        news_articles = self.news_service.get_news(symbol, days=days)
        sentiments = []
        
        for article in news_articles:
            # Analyze both title and content
            title_sentiment = self.analyze_text(article['title'])
            content_sentiment = self.analyze_text(article.get('content', ''))
            
            # Average the sentiments
            sentiment_score = (title_sentiment + content_sentiment) / 2
            
            sentiments.append({
                'title': article['title'],
                'text': article.get('content', ''),
                'sentiment': sentiment_score,
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'published_at': article.get('published_at', '')
            })
        
        return sentiments
    
    def get_overall_sentiment(self, symbol: str, days: int = 7) -> float:
        """
        Get overall sentiment score for a stock symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'TSLA').
            days (int, optional): Number of days of news to analyze. Defaults to 7.
            
        Returns:
            float: Overall sentiment score between -1 (negative) and 1 (positive).
        """
        sentiments = self.analyze_stock(symbol, days=days)
        
        if not sentiments:
            return 0.0
        
        # Calculate average sentiment
        total_sentiment = sum(item['sentiment'] for item in sentiments)
        return total_sentiment / len(sentiments)
