"""
Chatbot Service Module

This module provides functionality for an AI-powered chatbot that can provide
insights and answer questions about stocks using Google's Gemini API.
"""

import google.generativeai as genai
import json
from typing import Dict, List, Any, Optional

class ChatbotService:
    """
    Class for managing AI chatbot interactions.
    """

    def __init__(self, api_key: str = "AIzaSyAR-PgnMjfG3TDRt9JXLhsPAtvt8_FoHvw"):
        """
        Initialize the chatbot service.

        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Gemini model."""
        genai.configure(api_key=self.api_key)

        # Configure safety settings - set to low to allow financial advice
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        # Use the Gemini 1.5 Flash model
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            safety_settings=safety_settings
        )

        # For Gemini 1.5, we'll handle chat history in our own implementation
        # rather than using the built-in chat functionality
        self.history = []

    def generate_initial_insights(self, stock_data: Dict[str, Any]) -> str:
        """
        Generate initial insights about a stock based on available data.

        Args:
            stock_data: Dictionary containing stock information

        Returns:
            Initial insights message
        """
        # Create a prompt with all the stock data
        prompt = self._create_stock_data_prompt(stock_data)

        # Add instructions for the initial insights
        prompt += """
        Based on the data above, give me a simple, easy-to-understand summary of this stock. Include:
        1. A brief overview of how the stock is doing right now
        2. What the technical indicators suggest in simple terms
        3. What the sentiment analysis means for an investor
        4. How risky this stock seems to be
        5. Whether you'd recommend buying, selling, or holding, and why

        Keep your response under 250 words. Use a simple, conversational tone as if explaining to a friend who knows about tech but isn't a finance expert.

        IMPORTANT FORMATTING INSTRUCTIONS:
        - Use bullet points with the • symbol (not asterisks or dashes)
        - Format each main section with a bold header (like "Overview:" or "Technical Indicators:")
        - Keep paragraphs short (2-3 sentences max)
        - Avoid complex financial jargon
        - Don't use markdown formatting like *** or --- for emphasis
        """

        try:
            # Configure generation parameters
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more factual responses
                "max_output_tokens": 800,  # Limit response length
                "top_p": 0.95,
                "top_k": 40
            }

            # Generate the response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Store in history
            self.history.append({"role": "user", "parts": [prompt]})
            self.history.append({"role": "model", "parts": [response.text]})

            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating initial insights: {error_msg}")

            if "quota" in error_msg.lower():
                return "I've reached my API quota limit. Please try again later."
            elif "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                return "There's an issue with the AI model configuration. Please check the model name and try again."
            else:
                return "I'm having trouble analyzing this stock at the moment. Please try asking a specific question."

    def get_response(self, user_message: str, stock_data: Dict[str, Any]) -> str:
        """
        Get a response from the chatbot for a user message.

        Args:
            user_message: User's message
            stock_data: Dictionary containing stock information

        Returns:
            Chatbot response
        """
        try:
            # Create context with stock data for each new question
            context = self._create_stock_data_prompt(stock_data)

            # Add the user's question with context
            full_prompt = f"{context}\n\nUser question: {user_message}\n\nProvide a helpful, accurate response based on the stock data above. Use a simple, conversational tone as if explaining to a tech-savvy friend who isn't a finance expert. If you don't know something, admit it rather than making up information.\n\nIMPORTANT FORMATTING INSTRUCTIONS:\n- Use bullet points with the • symbol (not asterisks or dashes)\n- Format each main section with a bold header if needed\n- Keep paragraphs short (2-3 sentences max)\n- Avoid complex financial jargon\n- Don't use markdown formatting like *** or --- for emphasis"

            # Configure generation parameters
            generation_config = {
                "temperature": 0.3,  # Slightly higher temperature for conversational responses
                "max_output_tokens": 800,  # Limit response length
                "top_p": 0.95,
                "top_k": 40
            }

            # Generate the response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Store in history
            self.history.append({"role": "user", "parts": [user_message]})
            self.history.append({"role": "model", "parts": [response.text]})

            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"Error getting chatbot response: {error_msg}")

            if "quota" in error_msg.lower():
                return "I've reached my API quota limit. Please try again later."
            elif "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                return "There's an issue with the AI model configuration. Please check the model name and try again."
            else:
                return "I'm sorry, I encountered an error processing your request. Please try again with a different question."

    def _create_stock_data_prompt(self, stock_data: Dict[str, Any]) -> str:
        """
        Create a prompt with all the stock data for context.

        Args:
            stock_data: Dictionary containing stock information

        Returns:
            Formatted prompt with stock data
        """
        # Extract data from the stock_data dictionary
        symbol = stock_data.get('symbol', 'Unknown')
        company_name = stock_data.get('company_name', 'Unknown')

        prompt = f"""
        You are a friendly AI assistant helping with stock analysis for {symbol} ({company_name}).
        Here is the current data about this stock:

        MARKET DATA:
        - Current Price: ${stock_data.get('latest_price', 'N/A')}
        - Price Change: ${stock_data.get('price_change_amount', 'N/A')} ({stock_data.get('price_change_percent', 'N/A')}%)
        - Open: ${stock_data.get('open_price', 'N/A')}
        - High: ${stock_data.get('high_price', 'N/A')}
        - Low: ${stock_data.get('low_price', 'N/A')}
        - Volume: {stock_data.get('volume', 'N/A')}
        - 52-Week High: ${stock_data.get('year_high', 'N/A')}
        - 52-Week Low: ${stock_data.get('year_low', 'N/A')}

        PREDICTIONS:
        - ARIMA Prediction: ${stock_data.get('arima_pred', 'N/A')}
        - LSTM Prediction: ${stock_data.get('lstm_pred', 'N/A')}
        - Linear Regression Prediction: ${stock_data.get('lr_pred', 'N/A')}
        - Ensemble Prediction: ${stock_data.get('ensemble_pred', 'N/A')}
        - Prediction Accuracy: {stock_data.get('prediction_accuracy', 'N/A')}%

        TECHNICAL INDICATORS:
        - RSI (14): {stock_data.get('rsi_value', 'N/A')}
        - MACD: {stock_data.get('macd_value', 'N/A')}
        - SMA (50): {stock_data.get('sma_50', 'N/A')}
        - SMA (200): {stock_data.get('sma_200', 'N/A')}

        RISK METRICS:
        - Volatility: {stock_data.get('volatility', 'N/A')}%
        - Value at Risk (95%): {stock_data.get('var_value', 'N/A')}%
        - Beta: {stock_data.get('beta', 'N/A')}
        - Sharpe Ratio: {stock_data.get('sharpe_ratio', 'N/A')}

        SENTIMENT:
        - Market Sentiment: {stock_data.get('sentiment_description', 'N/A')}
        - Sentiment Score: {stock_data.get('sentiment_score', 'N/A')}
        - Based on {stock_data.get('sentiment_articles_count', 'N/A')} recent news articles
        """

        return prompt
