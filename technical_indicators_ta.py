"""
Technical Indicators Module using TA Library

This module provides functions to calculate various technical indicators
for stock price data using the TA (Technical Analysis) library.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from matplotlib.figure import Figure
import io
import base64

# Import TA library
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, TSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator, MFIIndicator


class TechnicalIndicators:
    """
    Class for calculating and visualizing technical indicators using the TA library.
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
        rsi_indicator = RSIIndicator(close=data['Close'], window=window)
        return rsi_indicator.rsi()

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
        macd_indicator = MACD(
            close=data['Close'],
            window_slow=slow_period,
            window_fast=fast_period,
            window_sign=signal_period
        )

        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, window_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: DataFrame with 'Close' prices
            window: Moving average window (default: 20)
            window_dev: Number of standard deviations (default: 2.0)

        Returns:
            Tuple containing (Middle Band, Upper Band, Lower Band)
        """
        bollinger = BollingerBands(
            close=data['Close'],
            window=window,
            window_dev=window_dev
        )

        middle_band = bollinger.bollinger_mavg()
        upper_band = bollinger.bollinger_hband()
        lower_band = bollinger.bollinger_lband()

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
        vwap = VolumeWeightedAveragePrice(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            volume=data['Volume']
        )

        return vwap.volume_weighted_average_price()

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
            sma_indicator = SMAIndicator(close=data['Close'], window=window)
            sma_dict[f'SMA_{window}'] = sma_indicator.sma_indicator()

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
            ema_indicator = EMAIndicator(close=data['Close'], window=window)
            ema_dict[f'EMA_{window}'] = ema_indicator.ema_indicator()

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
        atr_indicator = AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=window
        )

        return atr_indicator.average_true_range()

    @staticmethod
    def calculate_stochastic_oscillator(data: pd.DataFrame, window: int = 14, smooth_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            data: DataFrame with 'High', 'Low', and 'Close' columns
            window: %K window (default: 14)
            smooth_window: %D window (default: 3)

        Returns:
            Tuple containing (%K, %D)
        """
        stoch = StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            window=window,
            smooth_window=smooth_window
        )

        k = stoch.stoch()
        d = stoch.stoch_signal()

        return k, d

    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        Args:
            data: DataFrame with 'Close' and 'Volume' columns

        Returns:
            Series containing OBV values
        """
        obv = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
        return obv.on_balance_volume()

    @staticmethod
    def calculate_ichimoku(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components.

        Args:
            data: DataFrame with 'High', 'Low', and 'Close' columns

        Returns:
            Dictionary containing Ichimoku components
        """
        ichimoku = IchimokuIndicator(
            high=data['High'],
            low=data['Low']
        )

        return {
            'ichimoku_a': ichimoku.ichimoku_a(),
            'ichimoku_b': ichimoku.ichimoku_b(),
            'ichimoku_base_line': ichimoku.ichimoku_base_line(),
            'ichimoku_conversion_line': ichimoku.ichimoku_conversion_line()
        }

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all technical indicators using TA library.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary containing all calculated indicators
        """
        # Make sure the DataFrame has the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain all of these columns: {required_columns}")

        indicators = {}

        try:
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
            indicators['VWAP'] = TechnicalIndicators.calculate_vwap(data)

            # Moving Averages
            sma_dict = TechnicalIndicators.calculate_moving_averages(data)
            indicators.update(sma_dict)

            # EMA
            ema_dict = TechnicalIndicators.calculate_ema(data)
            indicators.update(ema_dict)

            # ATR
            indicators['ATR'] = TechnicalIndicators.calculate_atr(data)

            # Stochastic Oscillator
            k, d = TechnicalIndicators.calculate_stochastic_oscillator(data)
            indicators['Stoch_%K'] = k
            indicators['Stoch_%D'] = d

            # On-Balance Volume
            indicators['OBV'] = TechnicalIndicators.calculate_obv(data)

            # Money Flow Index
            mfi = MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14)
            indicators['MFI'] = mfi.money_flow_index()

            # Ichimoku Cloud
            try:
                ichimoku_dict = TechnicalIndicators.calculate_ichimoku(data)
                indicators.update(ichimoku_dict)
            except Exception as e:
                print(f"Error calculating Ichimoku Cloud: {str(e)}")
                # Continue without Ichimoku if it fails

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            # Return whatever indicators were successfully calculated

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

        # Plot histogram as bar chart
        for i in range(len(histogram)):
            if not np.isnan(histogram.iloc[i]):
                color = 'green' if histogram.iloc[i] > 0 else 'red'
                ax2.bar(histogram.index[i], histogram.iloc[i], color=color, width=1)

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
        # Simple line chart fallback
        def create_line_chart():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label='Close Price')

            if indicators:
                for name, indicator in indicators.items():
                    if pd.notna(indicator).any() and name.startswith(('SMA_', 'EMA_')):
                        ax.plot(indicator.index, indicator, label=name)

            ax.set_title('Price Chart')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            plt.tight_layout()
            return fig

        try:
            # Try to import mplfinance
            import mplfinance as mpf

            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Create a copy of the data to avoid modifying the original
            plot_data = data.copy()

            # Check if data is valid for candlestick chart
            if len(plot_data) < 2:
                print("Not enough data points for candlestick chart, using line chart instead")
                return create_line_chart()

            # Create a custom style
            mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)

            # Create figure and axes for the plot
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

            # Add indicators as overlay
            apds = []
            if indicators:
                # Add moving averages
                for name, indicator in indicators.items():
                    if name.startswith('SMA_') or name.startswith('EMA_'):
                        if pd.notna(indicator).any():  # Only add if there are non-NaN values
                            apds.append(mpf.make_addplot(indicator, panel=0, color='blue', width=0.7))

                # Add Bollinger Bands
                if all(band in indicators for band in ['BB_upper', 'BB_lower', 'BB_middle']):
                    if pd.notna(indicators['BB_upper']).any() and pd.notna(indicators['BB_lower']).any() and pd.notna(indicators['BB_middle']).any():
                        apds.append(mpf.make_addplot(indicators['BB_upper'], panel=0, color='red', linestyle='--'))
                        apds.append(mpf.make_addplot(indicators['BB_middle'], panel=0, color='blue', linestyle='--'))
                        apds.append(mpf.make_addplot(indicators['BB_lower'], panel=0, color='green', linestyle='--'))

            # Plot the candlestick chart
            try:
                # Try to plot with volume
                if apds:
                    mpf.plot(plot_data, type='candle', style=s,
                            ax=axes[0], volume=axes[1],
                            addplot=apds)
                else:
                    mpf.plot(plot_data, type='candle', style=s,
                            ax=axes[0], volume=axes[1])

                axes[0].set_title('Candlestick Chart')
                plt.tight_layout()
                return fig
            except Exception as e:
                print(f"Error plotting candlestick with volume: {str(e)}")
                # Try without volume as fallback
                plt.close(fig)  # Close the failed figure
                fig, ax = plt.subplots(figsize=(12, 6))

                try:
                    if apds:
                        mpf.plot(plot_data, type='candle', style=s, ax=ax, addplot=apds)
                    else:
                        mpf.plot(plot_data, type='candle', style=s, ax=ax)
                    ax.set_title('Candlestick Chart (No Volume)')
                    plt.tight_layout()
                    return fig
                except Exception as e2:
                    print(f"Error plotting candlestick without volume: {str(e2)}")
                    plt.close(fig)  # Close the failed figure
                    return create_line_chart()

        except ImportError as e:
            print(f"mplfinance not available: {str(e)}")
            return create_line_chart()

        except Exception as e:
            print(f"Error in plot_candlestick: {str(e)}")
            return create_line_chart()
