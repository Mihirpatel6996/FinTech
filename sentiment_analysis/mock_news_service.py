"""
Mock News Service for Testing

This module provides a mock news service for testing the sentiment analysis module
without requiring a real NewsAPI key.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MockNewsService:
    """
    Mock news service for testing.
    
    This class provides mock news articles for testing the sentiment analysis module
    without requiring a real NewsAPI key.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the mock news service."""
        pass
    
    def get_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get mock news articles for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'TSLA').
            days (int, optional): Number of days of news to fetch. Defaults to 7.
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing mock news articles.
        """
        # Get mock articles based on the symbol
        if symbol.upper() == 'TSLA':
            return self._get_tesla_articles()
        elif symbol.upper() == 'AAPL':
            return self._get_apple_articles()
        else:
            # Generate generic articles for any other symbol
            return self._get_generic_articles(symbol)
    
    def _get_tesla_articles(self) -> List[Dict[str, Any]]:
        """Get mock Tesla articles."""
        return [
            {
                'title': 'Tesla reports record quarterly profits as EV demand surges',
                'content': 'Tesla Inc. reported record quarterly profits on Wednesday, beating Wall Street expectations as demand for electric vehicles continues to surge globally. The company delivered over 300,000 vehicles in the quarter, up 40% from the previous year.',
                'source': 'Mock Financial News',
                'url': 'https://example.com/tesla-profits',
                'published_at': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            {
                'title': 'Tesla faces production challenges amid supply chain disruptions',
                'content': 'Tesla is facing production challenges at its factories due to ongoing supply chain disruptions. The company may struggle to meet its delivery targets for the current quarter, according to analysts.',
                'source': 'Mock Business News',
                'url': 'https://example.com/tesla-challenges',
                'published_at': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            {
                'title': 'Tesla announces new gigafactory in Asia to meet growing demand',
                'content': 'Tesla has announced plans to build a new gigafactory in Asia to meet growing demand for its electric vehicles in the region. The factory is expected to begin production in 2024.',
                'source': 'Mock Tech News',
                'url': 'https://example.com/tesla-gigafactory',
                'published_at': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
            }
        ]
    
    def _get_apple_articles(self) -> List[Dict[str, Any]]:
        """Get mock Apple articles."""
        return [
            {
                'title': 'Apple unveils new iPhone with revolutionary AI features',
                'content': 'Apple has unveiled its latest iPhone with revolutionary AI features that are expected to transform how users interact with their devices. The new iPhone will be available for pre-order next week.',
                'source': 'Mock Tech News',
                'url': 'https://example.com/apple-iphone',
                'published_at': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            {
                'title': 'Apple stock drops following disappointing earnings report',
                'content': 'Apple stock dropped 5% in after-hours trading following a disappointing earnings report. The company missed revenue expectations for the first time in four quarters.',
                'source': 'Mock Financial News',
                'url': 'https://example.com/apple-stock',
                'published_at': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            {
                'title': 'Apple announces expansion of services business with new subscription offerings',
                'content': 'Apple has announced an expansion of its services business with new subscription offerings in fitness, news, and entertainment. The company aims to double its services revenue by 2025.',
                'source': 'Mock Business News',
                'url': 'https://example.com/apple-services',
                'published_at': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
            }
        ]
    
    def _get_generic_articles(self, symbol: str) -> List[Dict[str, Any]]:
        """Get mock generic articles for any symbol."""
        return [
            {
                'title': f'{symbol} reports quarterly earnings above expectations',
                'content': f'{symbol} reported quarterly earnings above Wall Street expectations on Tuesday. The company saw strong growth in its core business segments.',
                'source': 'Mock Financial News',
                'url': f'https://example.com/{symbol.lower()}-earnings',
                'published_at': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            {
                'title': f'{symbol} stock falls amid market volatility',
                'content': f'{symbol} stock fell 3% on Wednesday amid broader market volatility. Analysts remain divided on the company\'s near-term prospects.',
                'source': 'Mock Business News',
                'url': f'https://example.com/{symbol.lower()}-stock',
                'published_at': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            {
                'title': f'{symbol} announces new strategic partnership',
                'content': f'{symbol} has announced a new strategic partnership that is expected to drive growth in key markets. The partnership will focus on expanding the company\'s product offerings.',
                'source': 'Mock Industry News',
                'url': f'https://example.com/{symbol.lower()}-partnership',
                'published_at': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
            }
        ]
