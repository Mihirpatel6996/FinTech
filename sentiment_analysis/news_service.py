"""
News Service

This module provides functionality to fetch news articles related to
specific stock symbols using the NewsAPI.
"""

import logging
from datetime import datetime, timedelta
from newsapi.newsapi_client import NewsApiClient
from .utils import clean_text
from .config import settings

# Configure logging
logger = logging.getLogger(__name__)

class NewsService:
    """
    Service for fetching news articles related to stock symbols.
    
    This class provides methods to fetch news articles from NewsAPI
    related to specific stock symbols.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the news service.
        
        Args:
            api_key (str, optional): API key for NewsAPI. If not provided,
                                    it will be read from config.
        """
        self.api_key = api_key or settings.NEWS_API_KEY
        self.newsapi = NewsApiClient(api_key=self.api_key)
    
    def get_news(self, symbol: str, days: int = 7) -> list:
        """
        Get news articles for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'TSLA').
            days (int, optional): Number of days of news to fetch. Defaults to 7.
            
        Returns:
            list: List of dictionaries containing news articles.
        """
        try:
            # Get company name from symbol
            company_name = self._get_company_name(symbol)
            
            # Get news from last X days
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            articles = self.newsapi.get_everything(
                q=f'({symbol} OR {company_name}) AND (stock OR market OR trading)',
                language='en',
                from_param=from_date,
                sort_by='relevancy'
            )

            return [{
                'title': article['title'],
                'description': article.get('description', ''),
                'content': clean_text(article.get('content', '')),
                'source': article['source']['name'],
                'url': article['url'],
                'published_at': article['publishedAt']
            } for article in articles['articles'] if article.get('description')]

        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """
        Map stock symbol to company name.
        
        Args:
            symbol (str): Stock symbol (e.g., 'TSLA').
            
        Returns:
            str: Company name.
        """
        # Add more mappings as needed
        company_map = {
            'TSLA': 'Tesla',
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'META': 'Facebook',
            'NFLX': 'Netflix',
            'IBM': 'IBM',
            'NVDA': 'Nvidia',
            'AMD': 'AMD'
        }
        return company_map.get(symbol.upper(), symbol)
