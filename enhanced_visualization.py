"""
Enhanced Visualization Module

This module provides advanced visualization capabilities for stock data,
including candlestick charts, interactive charts, technical indicator overlays,
heat maps, and correlation matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Any, Optional
import io
import base64
from technical_indicators_ta import TechnicalIndicators


class EnhancedVisualization:
    """
    Class for creating enhanced visualizations of stock data.
    """

    @staticmethod
    def create_candlestick_chart(data: pd.DataFrame, title: str = "Candlestick Chart") -> Figure:
        """
        Create a candlestick chart.

        Args:
            data: DataFrame with OHLC data
            title: Chart title

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

            # Create figure with separate axes for price and volume
            fig = plt.figure(figsize=(12, 8))

            # Plot candlestick chart with volume
            mpf.plot(plot_data, type='candle', style=s, title=title,
                    volume=True, figsize=(12, 8), returnfig=False,
                    panel_ratios=(3, 1), figratio=(12, 8), figscale=1.5)

            plt.tight_layout()
            return fig

        except ImportError:
            # Fallback to using matplotlib directly
            from matplotlib.dates import date2num
            try:
                from mplfinance.original_flavor import candlestick_ohlc
            except ImportError:
                # Create a simple line chart instead
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(data.index, data['Close'], label='Close')
                ax.set_title(title)
                ax.set_ylabel('Price')
                ax.grid(True)
                plt.tight_layout()
                return fig

            # Convert data to OHLC format
            ohlc = data[['Open', 'High', 'Low', 'Close']].copy()
            ohlc.reset_index(inplace=True)
            ohlc['Date'] = ohlc['Date'].map(date2num)

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot candlestick chart
            candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red')

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            plt.xticks(rotation=45)

            ax.set_title(title)
            ax.set_ylabel('Price')
            ax.grid(True)

            plt.tight_layout()

            return fig

    @staticmethod
    def create_candlestick_with_indicators(data: pd.DataFrame, indicators: Dict[str, Any] = None,
                                          title: str = "Candlestick Chart with Indicators") -> Figure:
        """
        Create a candlestick chart with technical indicators.

        Args:
            data: DataFrame with OHLC data
            indicators: Dictionary of technical indicators to overlay
            title: Chart title

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

            # Create figure
            fig = plt.figure(figsize=(12, 10))

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

                # Add RSI if available
                if 'RSI' in indicators:
                    apds.append(mpf.make_addplot(indicators['RSI'], panel=1, color='purple'))

                # Add MACD if available
                if all(m in indicators for m in ['MACD_line', 'MACD_signal', 'MACD_histogram']):
                    apds.append(mpf.make_addplot(indicators['MACD_line'], panel=2, color='blue'))
                    apds.append(mpf.make_addplot(indicators['MACD_signal'], panel=2, color='red'))

            # Plot the candlestick chart with indicators
            if apds:
                mpf.plot(plot_data, type='candle', style=s, title=title,
                        volume=True, figsize=(12, 10), returnfig=False,
                        panel_ratios=(4, 1, 1, 1), figratio=(12, 10), figscale=1.5,
                        addplot=apds)
            else:
                mpf.plot(plot_data, type='candle', style=s, title=title,
                        volume=True, figsize=(12, 10), returnfig=False,
                        panel_ratios=(4, 1), figratio=(12, 10), figscale=1.5)

            plt.tight_layout()

            return fig

        except ImportError:
            # Fallback to using matplotlib directly
            from matplotlib.dates import date2num

            # Convert data to OHLC format
            ohlc = data[['Open', 'High', 'Low', 'Close']].copy()
            ohlc.reset_index(inplace=True)
            ohlc['Date'] = ohlc['Date'].map(date2num)

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

            # Plot candlestick chart
            from mplfinance.original_flavor import candlestick_ohlc
            candlestick_ohlc(ax1, ohlc.values, width=0.6, colorup='green', colordown='red')

            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

            # Add indicators
            if indicators:
                for name, indicator in indicators.items():
                    if name.startswith('SMA_') or name.startswith('EMA_'):
                        ax1.plot(data.index, indicator, label=name)
                    elif name == 'RSI':
                        # Create a new axis for RSI
                        ax_rsi = ax1.twinx()
                        ax_rsi.plot(data.index, indicator, color='purple', label='RSI')
                        ax_rsi.set_ylabel('RSI')
                        ax_rsi.set_ylim(0, 100)
                        ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.5)
                        ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.5)
                    elif name.startswith('BB_'):
                        if name == 'BB_middle':
                            ax1.plot(data.index, indicator, color='blue', linestyle='--', label='BB Middle')
                        elif name == 'BB_upper':
                            ax1.plot(data.index, indicator, color='red', linestyle='--', label='BB Upper')
                        elif name == 'BB_lower':
                            ax1.plot(data.index, indicator, color='green', linestyle='--', label='BB Lower')

            # Plot volume
            ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')

            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()

            plt.tight_layout()

            return fig

    @staticmethod
    def create_correlation_matrix(returns: pd.DataFrame, title: str = "Correlation Matrix") -> Figure:
        """
        Create a correlation matrix heatmap.

        Args:
            returns: DataFrame with returns for multiple stocks
            title: Chart title

        Returns:
            Matplotlib figure
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns)
        ax.set_yticklabels(corr_matrix.columns)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add correlation values to cells
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")

        ax.set_title(title)
        plt.tight_layout()

        return fig

    @staticmethod
    def create_sector_heatmap(data: Dict[str, float], sector_mapping: Dict[str, str],
                             title: str = "Sector Performance Heatmap") -> Figure:
        """
        Create a sector performance heatmap.

        Args:
            data: Dictionary mapping symbols to performance values (e.g., returns)
            sector_mapping: Dictionary mapping symbols to sectors
            title: Chart title

        Returns:
            Matplotlib figure
        """
        # Group data by sector
        sector_data = {}

        for symbol, value in data.items():
            sector = sector_mapping.get(symbol, "Unknown")
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append((symbol, value))

        # Calculate sector averages
        sector_averages = {sector: np.mean([v for _, v in values]) for sector, values in sector_data.items()}

        # Sort sectors by average value
        sorted_sectors = sorted(sector_averages.items(), key=lambda x: x[1], reverse=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap data
        sectors = [s[0] for s in sorted_sectors]
        sector_values = [s[1] for s in sorted_sectors]

        # Create heatmap
        im = ax.imshow([sector_values], cmap='RdYlGn', aspect='auto')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Performance", rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_yticks([])
        ax.set_xticks(np.arange(len(sectors)))
        ax.set_xticklabels(sectors)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values to cells
        for i, value in enumerate(sector_values):
            text = ax.text(i, 0, f"{value:.2%}",
                           ha="center", va="center", color="black" if abs(value) < 0.1 else "white")

        ax.set_title(title)
        plt.tight_layout()

        return fig

    @staticmethod
    def create_interactive_chart_html(data: pd.DataFrame, indicators: Dict[str, Any] = None,
                                     title: str = "Interactive Stock Chart") -> str:
        """
        Create an interactive stock chart using Plotly.

        Args:
            data: DataFrame with OHLC data
            indicators: Dictionary of technical indicators to overlay
            title: Chart title

        Returns:
            HTML string containing the interactive chart
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create figure with secondary y-axis
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.03, subplot_titles=(title, 'Volume'),
                               row_heights=[0.7, 0.3])

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )

            # Add volume bar chart
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(128, 128, 128, 0.5)'
                ),
                row=2, col=1
            )

            # Add indicators
            if indicators:
                for name, indicator in indicators.items():
                    if name.startswith('SMA_') or name.startswith('EMA_'):
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=indicator,
                                name=name,
                                line=dict(width=1)
                            ),
                            row=1, col=1
                        )
                    elif name == 'RSI':
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=indicator,
                                name='RSI',
                                line=dict(color='purple', width=1)
                            ),
                            row=1, col=1
                        )
                    elif name.startswith('BB_'):
                        if name == 'BB_middle':
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=indicator,
                                    name='BB Middle',
                                    line=dict(color='blue', width=1, dash='dash')
                                ),
                                row=1, col=1
                            )
                        elif name == 'BB_upper':
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=indicator,
                                    name='BB Upper',
                                    line=dict(color='red', width=1, dash='dash')
                                ),
                                row=1, col=1
                            )
                        elif name == 'BB_lower':
                            fig.add_trace(
                                go.Scatter(
                                    x=data.index,
                                    y=indicator,
                                    name='BB Lower',
                                    line=dict(color='green', width=1, dash='dash')
                                ),
                                row=1, col=1
                            )

            # Update layout
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Update y-axes
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            # Convert to HTML
            html = fig.to_html(include_plotlyjs=True, full_html=False)

            return html

        except ImportError:
            # Fallback to static image if Plotly is not available
            fig = EnhancedVisualization.create_candlestick_with_indicators(data, indicators, title)

            # Convert figure to base64 image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')

            # Create HTML with embedded image
            html = f'<img src="data:image/png;base64,{img_str}" alt="{title}" style="width:100%">'

            return html

    @staticmethod
    def generate_technical_analysis_report(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate a comprehensive technical analysis report.

        Args:
            data: DataFrame with OHLC data
            symbol: Stock symbol

        Returns:
            Dictionary containing analysis results and visualizations
        """
        # Calculate technical indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)

        # Create visualizations
        candlestick_chart = EnhancedVisualization.create_candlestick_chart(data, f"{symbol} Candlestick Chart")

        # RSI chart
        rsi = indicators.get('RSI')
        rsi_chart = TechnicalIndicators.plot_rsi(data, rsi) if rsi is not None else None

        # MACD chart
        macd_line = indicators.get('MACD_line')
        signal_line = indicators.get('MACD_signal')
        histogram = indicators.get('MACD_histogram')
        macd_chart = TechnicalIndicators.plot_macd(data, macd_line, signal_line, histogram) if all([macd_line is not None, signal_line is not None, histogram is not None]) else None

        # Bollinger Bands chart
        middle_band = indicators.get('BB_middle')
        upper_band = indicators.get('BB_upper')
        lower_band = indicators.get('BB_lower')
        bb_chart = TechnicalIndicators.plot_bollinger_bands(data, middle_band, upper_band, lower_band) if all([middle_band is not None, upper_band is not None, lower_band is not None]) else None

        # Moving Averages chart
        ma_dict = {k: v for k, v in indicators.items() if k.startswith('SMA_') or k.startswith('EMA_')}
        ma_chart = TechnicalIndicators.plot_moving_averages(data, ma_dict) if ma_dict else None

        # Interactive chart HTML
        interactive_chart_html = EnhancedVisualization.create_interactive_chart_html(data, indicators, f"{symbol} Interactive Chart")

        # Prepare analysis results
        latest_close = data['Close'].iloc[-1]

        # RSI analysis
        rsi_value = rsi.iloc[-1] if rsi is not None else None
        rsi_signal = "Oversold" if rsi_value is not None and rsi_value < 30 else "Overbought" if rsi_value is not None and rsi_value > 70 else "Neutral"

        # MACD analysis
        macd_value = macd_line.iloc[-1] if macd_line is not None else None
        signal_value = signal_line.iloc[-1] if signal_line is not None else None
        macd_signal = "Bullish" if macd_value is not None and signal_value is not None and macd_value > signal_value else "Bearish" if macd_value is not None and signal_value is not None and macd_value < signal_value else "Neutral"

        # Bollinger Bands analysis
        bb_signal = "Neutral"
        if all([latest_close, upper_band is not None, lower_band is not None, middle_band is not None]):
            upper_value = upper_band.iloc[-1]
            lower_value = lower_band.iloc[-1]
            middle_value = middle_band.iloc[-1]

            if latest_close > upper_value:
                bb_signal = "Overbought"
            elif latest_close < lower_value:
                bb_signal = "Oversold"
            elif latest_close > middle_value:
                bb_signal = "Bullish"
            elif latest_close < middle_value:
                bb_signal = "Bearish"

        # Moving Average analysis
        ma_signals = {}
        for ma_name, ma_values in ma_dict.items():
            ma_value = ma_values.iloc[-1]
            ma_signals[ma_name] = "Bullish" if latest_close > ma_value else "Bearish"

        # Overall signal
        signals = [rsi_signal, macd_signal, bb_signal] + list(ma_signals.values())
        bullish_count = signals.count("Bullish")
        bearish_count = signals.count("Bearish")
        oversold_count = signals.count("Oversold")
        overbought_count = signals.count("Overbought")

        if bullish_count > bearish_count + oversold_count + overbought_count:
            overall_signal = "Bullish"
        elif bearish_count > bullish_count + oversold_count + overbought_count:
            overall_signal = "Bearish"
        elif oversold_count > 0:
            overall_signal = "Oversold"
        elif overbought_count > 0:
            overall_signal = "Overbought"
        else:
            overall_signal = "Neutral"

        # Compile results
        results = {
            "symbol": symbol,
            "latest_close": latest_close,
            "indicators": {
                "RSI": {
                    "value": rsi_value,
                    "signal": rsi_signal
                },
                "MACD": {
                    "value": macd_value,
                    "signal_value": signal_value,
                    "signal": macd_signal
                },
                "Bollinger_Bands": {
                    "upper": upper_band.iloc[-1] if upper_band is not None else None,
                    "middle": middle_band.iloc[-1] if middle_band is not None else None,
                    "lower": lower_band.iloc[-1] if lower_band is not None else None,
                    "signal": bb_signal
                },
                "Moving_Averages": {
                    ma_name: {
                        "value": ma_values.iloc[-1],
                        "signal": ma_signals[ma_name]
                    } for ma_name, ma_values in ma_dict.items()
                }
            },
            "overall_signal": overall_signal,
            "charts": {
                "candlestick_chart": candlestick_chart,
                "rsi_chart": rsi_chart,
                "macd_chart": macd_chart,
                "bb_chart": bb_chart,
                "ma_chart": ma_chart
            },
            "interactive_chart_html": interactive_chart_html
        }

        return results
