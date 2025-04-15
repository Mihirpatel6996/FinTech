"""
Configuration Settings

This module provides configuration settings for the sentiment analysis module.
"""

import os
from typing import List, Tuple

class Settings:
    """
    Configuration settings for the sentiment analysis module.
    """

    # NewsAPI settings
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "2939063f6d8a4948aae8dfe4f9378d90")

    # NLTK settings
    NLTK_TOKENS_REQUIRED: Tuple[str, ...] = ()
    NLTK_MIN_TOKENS: int = 1
    NLTK_TOKENS_IGNORED: Tuple[str, ...] = ("win", "giveaway")

settings = Settings()
