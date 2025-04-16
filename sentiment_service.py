"""
Sentiment Analysis Service Wrapper

This module provides a wrapper for the sentiment analysis module.
"""

import json
import logging
from datetime import datetime, timedelta
from sentiment_analysis import SentimentService
from sentiment_analysis.mock_news_service import MockNewsService

# Configure logging
logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    """
    Wrapper for the sentiment analysis module.

    This class provides methods to analyze sentiment in news articles
    related to a specific stock symbol and store the results in the database.
    """

    def __init__(self, db, use_mock=True):
        """
        Initialize the sentiment analysis service.

        Args:
            db: Database instance.
            use_mock (bool, optional): Whether to use mock news service. Defaults to True.
        """
        self.db = db
        self.use_mock = use_mock

        # Initialize sentiment service
        if use_mock:
            # Use mock news service for testing
            from sentiment_analysis.sentiment_service import SentimentService as RealSentimentService
            self.sentiment_service = RealSentimentService()
            # Replace the news service with mock news service
            self.sentiment_service.news_service = MockNewsService()
        else:
            # Use real news service with the API key from .env file
            try:
                # Try to get API key from environment variable
                import os
                from dotenv import load_dotenv

                # Load .env file from the sentiment_analysis directory
                dotenv_path = os.path.join('sentiment_analysis', '.env')
                load_dotenv(dotenv_path)

                # Get API key
                api_key = os.getenv('NEWS_API_KEY')
                if not api_key:
                    print("Warning: NEWS_API_KEY not found in environment variables. Using default key from config.")

                # Initialize sentiment service with API key
                self.sentiment_service = SentimentService(news_api_key=api_key)
                print(f"Using real NewsAPI service with key: {api_key[:5]}...")
            except Exception as e:
                print(f"Error initializing real news service: {str(e)}. Falling back to mock service.")
                # Fall back to mock service if there's an error
                from sentiment_analysis.sentiment_service import SentimentService as RealSentimentService
                self.sentiment_service = RealSentimentService()
                self.sentiment_service.news_service = MockNewsService()
                self.use_mock = True

    def analyze_sentiment(self, symbol: str, days: int = 7) -> dict:
        """
        Analyze sentiment for a stock symbol and store the results in the database.

        Args:
            symbol (str): Stock symbol (e.g., 'TSLA').
            days (int, optional): Number of days of news to analyze. Defaults to 7.

        Returns:
            dict: Sentiment analysis results.
        """
        try:
            # Check if we have recent sentiment analysis in the database
            existing_sentiment = self.db.get_sentiment_analysis(symbol)
            if existing_sentiment:
                analysis_date = datetime.strptime(existing_sentiment["analysis_date"], '%Y-%m-%d')
                if (datetime.now() - analysis_date).days < 1:
                    # Use existing sentiment analysis if it's less than 1 day old
                    print(f"Using cached sentiment analysis for {symbol} from {existing_sentiment['analysis_date']}")
                    return existing_sentiment

            print(f"Fetching sentiment analysis for {symbol}...")

            # Get sentiment analysis
            try:
                sentiments = self.sentiment_service.analyze_stock(symbol, days=days)
                overall_sentiment = self.sentiment_service.get_overall_sentiment(symbol, days=days)

                # Check if we got any results
                if not sentiments:
                    print(f"No news articles found for {symbol}. Using neutral sentiment.")
                    overall_sentiment = 0.0
                    sentiments = []
                else:
                    print(f"Found {len(sentiments)} news articles for {symbol}")

                # Store sentiment analysis in database
                self.db.store_sentiment_analysis(
                    symbol=symbol,
                    sentiment_score=overall_sentiment,
                    news_count=len(sentiments),
                    news_data=json.dumps(sentiments)
                )

                # Return sentiment analysis results
                return {
                    "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                    "sentiment_score": overall_sentiment,
                    "news_count": len(sentiments),
                    "news_data": json.dumps(sentiments)
                }
            except Exception as api_error:
                # If there's an error with the real API, fall back to mock data if we're using the real API
                if not self.use_mock:
                    print(f"Error with NewsAPI: {str(api_error)}. Falling back to mock data for {symbol}.")
                    # Temporarily use mock service
                    original_service = self.sentiment_service
                    from sentiment_analysis.sentiment_service import SentimentService as RealSentimentService
                    temp_service = RealSentimentService()
                    temp_service.news_service = MockNewsService()

                    # Get mock data
                    sentiments = temp_service.analyze_stock(symbol, days=days)
                    overall_sentiment = temp_service.get_overall_sentiment(symbol, days=days)

                    # Store sentiment analysis in database
                    self.db.store_sentiment_analysis(
                        symbol=symbol,
                        sentiment_score=overall_sentiment,
                        news_count=len(sentiments),
                        news_data=json.dumps(sentiments)
                    )

                    # Return sentiment analysis results
                    return {
                        "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                        "sentiment_score": overall_sentiment,
                        "news_count": len(sentiments),
                        "news_data": json.dumps(sentiments),
                        "is_mock": True
                    }
                else:
                    # Re-raise the error if we're already using mock data
                    raise

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                "sentiment_score": 0.0,
                "news_count": 0,
                "news_data": "[]",
                "error": str(e)
            }

    def get_sentiment_description(self, sentiment_score: float) -> str:
        """
        Get a description of the sentiment score.

        Args:
            sentiment_score (float): Sentiment score between -1 and 1.

        Returns:
            str: Description of the sentiment score.
        """
        if sentiment_score >= 0.5:
            return "Very Positive"
        elif sentiment_score >= 0.2:
            return "Positive"
        elif sentiment_score > -0.2:
            return "Neutral"
        elif sentiment_score > -0.5:
            return "Negative"
        else:
            return "Very Negative"

    def get_sentiment_color(self, sentiment_score: float) -> str:
        """
        Get a color for the sentiment score.

        Args:
            sentiment_score (float): Sentiment score between -1 and 1.

        Returns:
            str: Color for the sentiment score.
        """
        if sentiment_score >= 0.5:
            return "success"  # Green
        elif sentiment_score >= 0.2:
            return "info"  # Blue
        elif sentiment_score > -0.2:
            return "secondary"  # Gray
        elif sentiment_score > -0.5:
            return "warning"  # Yellow
        else:
            return "danger"  # Red

    def get_sentiment_for_symbol(self, symbol: str) -> list:
        """
        Get sentiment data for a stock symbol.

        Args:
            symbol (str): Stock symbol (e.g., 'TSLA').

        Returns:
            list: List of dictionaries containing news articles and their sentiment scores.
        """
        try:
            # First check if we have cached sentiment data in the database
            cached_sentiment = self.db.get_sentiment_analysis(symbol)
            if cached_sentiment and cached_sentiment.get("news_data"):
                try:
                    # Parse the JSON string into a list of dictionaries
                    import json
                    news_data = json.loads(cached_sentiment["news_data"])
                    if news_data:
                        return news_data
                except Exception as e:
                    print(f"Error parsing cached sentiment data: {str(e)}")

            # If no cached data or error parsing, get fresh data
            result = self.analyze_sentiment(symbol)
            if result and result.get("news_data"):
                try:
                    import json
                    return json.loads(result["news_data"])
                except Exception as e:
                    print(f"Error parsing fresh sentiment data: {str(e)}")

            return []
        except Exception as e:
            print(f"Error getting sentiment for symbol {symbol}: {str(e)}")
            return []

    def get_average_sentiment(self, sentiment_data: list) -> float:
        """
        Calculate the average sentiment score from sentiment data.

        Args:
            sentiment_data (list): List of dictionaries containing news articles and their sentiment scores.

        Returns:
            float: Average sentiment score between -1 (negative) and 1 (positive).
        """
        if not sentiment_data:
            return 0.0

        total_sentiment = sum(item.get('sentiment', 0) for item in sentiment_data)
        return total_sentiment / len(sentiment_data)
