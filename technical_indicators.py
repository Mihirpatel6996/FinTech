"""
Technical Indicators Module

This module provides functions to calculate various technical indicators
for stock price data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.dates as mdates
from matplotlib.figure import Figure


class TechnicalIndicators:
    """
    Class for calculating and visualizing technical indicators.
    """

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: DataFrame with 'Close' prices
            window: RSI calculation window (default: 14)

        Returns:
            Series containing RSI values
        """
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: DataFrame with 'Close' prices
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)

        Returns:
            Tuple containing (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: DataFrame with 'Close' prices
            window: Moving average window (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            Tuple containing (Middle Band, Upper Band, Lower Band)
        """
        # Calculate middle band (SMA)
        middle_band = data['Close'].rolling(window=window).mean()

        # Calculate standard deviation
        std = data['Close'].rolling(window=window).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return middle_band, upper_band, lower_band

    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        Args:
            data: DataFrame with 'Close', 'High', 'Low', and 'Volume' columns

        Returns:
            Series containing VWAP values
        """
        # Calculate typical price
        typical_price = (data['Close'] + data['High'] + data['Low']) / 3

        # Calculate VWAP
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()

        return vwap

    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame, windows: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Averages (SMA) for multiple windows.

        Args:
            data: DataFrame with 'Close' prices
            windows: List of window periods (default: [20, 50, 200])

        Returns:
            Dictionary containing SMA values for each window
        """
        sma_dict = {}

        for window in windows:
            sma_dict[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()

        return sma_dict

    @staticmethod
    def calculate_ema(data: pd.DataFrame, windows: List[int] = [12, 26, 50]) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Averages (EMA) for multiple windows.

        Args:
            data: DataFrame with 'Close' prices
            windows: List of window periods (default: [12, 26, 50])

        Returns:
            Dictionary containing EMA values for each window
        """
        ema_dict = {}

        for window in windows:
            ema_dict[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()

        return ema_dict

    @staticmethod
    def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            data: DataFrame with 'High', 'Low', and 'Close' columns
            window: ATR calculation window (default: 14)

        Returns:
            Series containing ATR values
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        atr = true_range.rolling(window=window).mean()

        return atr

    @staticmethod
    def calculate_stochastic_oscillator(data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            data: DataFrame with 'High', 'Low', and 'Close' columns
            k_window: %K window (default: 14)
            d_window: %D window (default: 3)

        Returns:
            Tuple containing (%K, %D)
        """
        # Calculate %K
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()

        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))

        # Calculate %D
        d = k.rolling(window=d_window).mean()

        return k, d

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all technical indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary containing all calculated indicators
        """
        indicators = {}

        # RSI
        indicators['RSI'] = TechnicalIndicators.calculate_rsi(data)

        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(data)
        indicators['MACD_line'] = macd_line
        indicators['MACD_signal'] = signal_line
        indicators['MACD_histogram'] = histogram

        # Bollinger Bands
        middle_band, upper_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(data)
        indicators['BB_middle'] = middle_band
        indicators['BB_upper'] = upper_band
        indicators['BB_lower'] = lower_band

        # VWAP
        if all(col in data.columns for col in ['High', 'Low', 'Volume']):
            indicators['VWAP'] = TechnicalIndicators.calculate_vwap(data)

        # Moving Averages
        sma_dict = TechnicalIndicators.calculate_moving_averages(data)
        indicators.update(sma_dict)

        # EMA
        ema_dict = TechnicalIndicators.calculate_ema(data)
        indicators.update(ema_dict)

        # ATR
        if all(col in data.columns for col in ['High', 'Low']):
            indicators['ATR'] = TechnicalIndicators.calculate_atr(data)

        # Stochastic Oscillator
        if all(col in data.columns for col in ['High', 'Low']):
            k, d = TechnicalIndicators.calculate_stochastic_oscillator(data)
            indicators['Stoch_%K'] = k
            indicators['Stoch_%D'] = d

        return indicators

    @staticmethod
    def plot_rsi(data: pd.DataFrame, rsi: pd.Series, window: int = 14) -> Figure:
        """
        Plot RSI indicator.

        Args:
            data: DataFrame with 'Close' prices
            rsi: Series containing RSI values
            window: RSI calculation window

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot price
        ax1.plot(data.index, data['Close'])
        ax1.set_title('Price and RSI')
        ax1.set_ylabel('Price')
        ax1.grid(True)

        # Plot RSI
        ax2.plot(data.index, rsi, color='purple')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.fill_between(data.index, rsi, 70, where=(rsi >= 70), color='red', alpha=0.3)
        ax2.fill_between(data.index, rsi, 30, where=(rsi <= 30), color='green', alpha=0.3)
        ax2.set_ylabel(f'RSI ({window})')
        ax2.set_ylim(0, 100)
        ax2.grid(True)

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_macd(data: pd.DataFrame, macd_line: pd.Series, signal_line: pd.Series, histogram: pd.Series) -> Figure:
        """
        Plot MACD indicator.

        Args:
            data: DataFrame with 'Close' prices
            macd_line: Series containing MACD line values
            signal_line: Series containing Signal line values
            histogram: Series containing Histogram values

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot price
        ax1.plot(data.index, data['Close'])
        ax1.set_title('Price and MACD')
        ax1.set_ylabel('Price')
        ax1.grid(True)

        # Plot MACD
        ax2.plot(data.index, macd_line, color='blue', label='MACD')
        ax2.plot(data.index, signal_line, color='red', label='Signal')
        ax2.bar(data.index, histogram, color=np.where(histogram > 0, 'green', 'red'), label='Histogram')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('MACD')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_bollinger_bands(data: pd.DataFrame, middle_band: pd.Series, upper_band: pd.Series, lower_band: pd.Series) -> Figure:
        """
        Plot Bollinger Bands.

        Args:
            data: DataFrame with 'Close' prices
            middle_band: Series containing Middle Band values
            upper_band: Series containing Upper Band values
            lower_band: Series containing Lower Band values

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price and bands
        ax.plot(data.index, data['Close'], label='Close', color='blue')
        ax.plot(data.index, middle_band, label='Middle Band', color='orange')
        ax.plot(data.index, upper_band, label='Upper Band', color='red')
        ax.plot(data.index, lower_band, label='Lower Band', color='green')
        ax.fill_between(data.index, upper_band, lower_band, alpha=0.1, color='gray')

        ax.set_title('Bollinger Bands')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_moving_averages(data: pd.DataFrame, ma_dict: Dict[str, pd.Series]) -> Figure:
        """
        Plot Moving Averages.

        Args:
            data: DataFrame with 'Close' prices
            ma_dict: Dictionary containing moving average values

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot price
        ax.plot(data.index, data['Close'], label='Close', color='black')

        # Plot moving averages
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        for i, (name, ma) in enumerate(ma_dict.items()):
            ax.plot(data.index, ma, label=name, color=colors[i % len(colors)])

        ax.set_title('Moving Averages')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_candlestick(data: pd.DataFrame, indicators: Optional[Dict[str, pd.Series]] = None) -> Figure:
        """
        Plot candlestick chart with optional indicators.

        Args:
            data: DataFrame with OHLC data
            indicators: Optional dictionary of indicators to overlay

        Returns:
            Matplotlib figure
        """
        try:
            import mplfinance as mpf

            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Create a copy of the data to avoid modifying the original
            plot_data = data.copy()

            # Create a custom style
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)

            # Create figure and axes
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))

            # Add indicators as overlay
            apds = []
            if indicators:
                # Add moving averages
                for name, indicator in indicators.items():
                    if name.startswith('SMA_') or name.startswith('EMA_'):
                        apds.append(mpf.make_addplot(indicator, panel=0, color='blue', width=0.7))

                # Add Bollinger Bands
                if all(band in indicators for band in ['BB_upper', 'BB_lower', 'BB_middle']):
                    apds.append(mpf.make_addplot(indicators['BB_upper'], panel=0, color='red', linestyle='--'))
                    apds.append(mpf.make_addplot(indicators['BB_middle'], panel=0, color='blue', linestyle='--'))
                    apds.append(mpf.make_addplot(indicators['BB_lower'], panel=0, color='green', linestyle='--'))

            # Plot the candlestick chart with indicators
            if apds:
                mpf.plot(plot_data, type='candle', style=s, title='Candlestick Chart',
                         ax=axes, addplot=apds, returnfig=False)
            else:
                mpf.plot(plot_data, type='candle', style=s, title='Candlestick Chart',
                         ax=axes, returnfig=False)

            plt.tight_layout()
            return fig

        except ImportError:
            # Fallback to using matplotlib directly
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label='Close Price')

            if indicators:
                for name, indicator in indicators.items():
                    if pd.notna(indicator).any():  # Only plot if there are non-NaN values
                        ax.plot(indicator.index, indicator, label=name)

            ax.set_title('Candlestick Chart')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            plt.tight_layout()
            return fig
