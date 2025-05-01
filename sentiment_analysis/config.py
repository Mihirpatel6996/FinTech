"""
Configuration Settings

This module provides configuration settings for the sentiment analysis module.
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """
    Configuration settings for the sentiment analysis module.
    """

    # NewsAPI settings
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY")

    # NLTK settings
    NLTK_TOKENS_REQUIRED: Tuple[str, ...] = ()
    NLTK_MIN_TOKENS: int = 1
    NLTK_TOKENS_IGNORED: Tuple[str, ...] = ("win", "giveaway")

settings = Settings()

