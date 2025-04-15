"""
Utility Functions

This module provides utility functions for text processing and other
helper functions used in the sentiment analysis module.
"""

import re
from typing import List

def clean_text(text: str) -> str:
    """
    Clean text by removing URLs, mentions, hashtags, and special characters.
    
    Args:
        text (str): Text to clean.
        
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
        
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

def filter_tokens(text: str, required: List[str], ignored: List[str]) -> bool:
    """
    Check if text contains required tokens and doesn't contain ignored tokens.
    
    Args:
        text (str): Text to check.
        required (List[str]): List of required tokens.
        ignored (List[str]): List of ignored tokens.
        
    Returns:
        bool: True if text contains required tokens and doesn't contain ignored tokens.
    """
    text_lower = text.lower()
    return (any(token.lower() in text_lower for token in required) and
            not any(token.lower() in text_lower for token in ignored))

def calculate_moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    Calculate moving average for a list of values.
    
    Args:
        values (List[float]): List of values.
        window (int, optional): Window size. Defaults to 5.
        
    Returns:
        List[float]: List of moving averages.
    """
    if not values or window <= 0:
        return []
    ret = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ret.append(sum(values[start:i+1]) / (i - start + 1))
    return ret
