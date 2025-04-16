"""
Risk Analysis Module

This module provides functionality for analyzing risk in stock portfolios,
including Value at Risk (VaR), volatility analysis, beta calculation,
Sharpe ratio, and risk-adjusted returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from matplotlib.figure import Figure
from scipy import stats


class RiskAnalyzer:
    """
    Class for analyzing risk in stock portfolios.
    """

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            prices: DataFrame with 'Close' prices
            method: Method for calculating returns ('log' or 'simple')

        Returns:
            DataFrame containing returns
        """
        if method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:  # simple
            return (prices / prices.shift(1) - 1).dropna()

    @staticmethod
    def calculate_portfolio_returns(holdings: Dict[str, float], returns: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate portfolio returns.

        Args:
            holdings: Dictionary mapping symbols to quantities
            returns: Dictionary mapping symbols to DataFrames with returns

        Returns:
            Series containing portfolio returns
        """
        # Calculate portfolio weights
        total_value = sum(holdings.values())
        weights = {symbol: quantity / total_value for symbol, quantity in holdings.items()}

        # Calculate weighted returns
        weighted_returns = []

        for symbol, weight in weights.items():
            if symbol in returns:
                weighted_returns.append(returns[symbol] * weight)

        if not weighted_returns:
            return pd.Series()

        # Sum weighted returns
        portfolio_returns = pd.concat(weighted_returns, axis=1).sum(axis=1)

        return portfolio_returns

    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate volatility (standard deviation of returns).

        Args:
            returns: Series containing returns
            annualize: Whether to annualize the volatility
            trading_days: Number of trading days in a year

        Returns:
            Volatility
        """
        if returns.empty:
            return 0.0

        volatility = returns.std()

        if annualize:
            volatility *= np.sqrt(trading_days)

        return volatility

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95, time_horizon: int = 1) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series containing returns
            confidence_level: Confidence level (default: 0.95)
            time_horizon: Time horizon in days (default: 1)

        Returns:
            Value at Risk
        """
        if returns.empty:
            return 0.0

        # Calculate VaR using the historical method
        var = -np.percentile(returns, 100 * (1 - confidence_level))

        # Scale VaR for the time horizon
        var *= np.sqrt(time_horizon)

        return var

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

        Args:
            returns: Series containing returns
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Conditional Value at Risk
        """
        if returns.empty:
            return 0.0

        # Calculate VaR
        var = RiskAnalyzer.calculate_var(returns, confidence_level)

        # Calculate CVaR as the average of returns beyond VaR
        cvar = -returns[returns <= -var].mean()

        return cvar

    @staticmethod
    def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta (systematic risk).

        Args:
            returns: Series containing asset returns
            market_returns: Series containing market returns

        Returns:
            Beta
        """
        if returns.empty or market_returns.empty:
            return 0.0

        # Align the returns
        aligned_returns = pd.concat([returns, market_returns], axis=1).dropna()

        if aligned_returns.empty:
            return 0.0

        # Calculate covariance and variance
        covariance = aligned_returns.iloc[:, 0].cov(aligned_returns.iloc[:, 1])
        variance = aligned_returns.iloc[:, 1].var()

        if variance == 0:
            return 0.0

        beta = covariance / variance

        return beta

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series containing returns
            risk_free_rate: Risk-free rate (default: 0.0)
            annualize: Whether to annualize the ratio
            trading_days: Number of trading days in a year

        Returns:
            Sharpe ratio
        """
        if returns.empty:
            return 0.0

        # Calculate excess returns
        excess_returns = returns - risk_free_rate / trading_days

        # Calculate Sharpe ratio
        sharpe = excess_returns.mean() / excess_returns.std()

        if annualize:
            sharpe *= np.sqrt(trading_days)

        return sharpe

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Series containing returns
            risk_free_rate: Risk-free rate (default: 0.0)
            annualize: Whether to annualize the ratio
            trading_days: Number of trading days in a year

        Returns:
            Sortino ratio
        """
        if returns.empty:
            return 0.0

        # Calculate excess returns
        excess_returns = returns - risk_free_rate / trading_days

        # Calculate downside deviation (standard deviation of negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std()

        if downside_deviation == 0:
            return 0.0

        # Calculate Sortino ratio
        sortino = excess_returns.mean() / downside_deviation

        if annualize:
            sortino *= np.sqrt(trading_days)

        return sortino

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Series containing returns

        Returns:
            Maximum drawdown
        """
        if returns.empty:
            return 0.0

        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cum_returns.cummax()

        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1

        # Calculate maximum drawdown
        max_drawdown = drawdown.min()

        return max_drawdown

    @staticmethod
    def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95, time_horizon: int = 1) -> float:
        """
        Alias for calculate_var for backward compatibility.

        Args:
            returns: Series containing returns
            confidence_level: Confidence level (default: 0.95)
            time_horizon: Time horizon in days (default: 1)

        Returns:
            Value at Risk
        """
        return RiskAnalyzer.calculate_var(returns, confidence_level, time_horizon)

    @staticmethod
    def calculate_risk_metrics(returns: pd.Series, market_returns: pd.Series = None, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate various risk metrics.

        Args:
            returns: Series containing returns
            market_returns: Series containing market returns (optional)
            risk_free_rate: Risk-free rate (default: 0.0)

        Returns:
            Dictionary containing risk metrics
        """
        metrics = {}

        # Volatility
        metrics['volatility'] = RiskAnalyzer.calculate_volatility(returns)

        # Value at Risk
        metrics['var_95'] = RiskAnalyzer.calculate_var(returns, confidence_level=0.95)
        metrics['var_99'] = RiskAnalyzer.calculate_var(returns, confidence_level=0.99)

        # Conditional Value at Risk
        metrics['cvar_95'] = RiskAnalyzer.calculate_cvar(returns, confidence_level=0.95)
        metrics['cvar_99'] = RiskAnalyzer.calculate_cvar(returns, confidence_level=0.99)

        # Beta
        if market_returns is not None:
            metrics['beta'] = RiskAnalyzer.calculate_beta(returns, market_returns)

        # Sharpe Ratio
        metrics['sharpe_ratio'] = RiskAnalyzer.calculate_sharpe_ratio(returns, risk_free_rate)

        # Sortino Ratio
        metrics['sortino_ratio'] = RiskAnalyzer.calculate_sortino_ratio(returns, risk_free_rate)

        # Maximum Drawdown
        metrics['max_drawdown'] = RiskAnalyzer.calculate_max_drawdown(returns)

        return metrics

    @staticmethod
    def plot_var(returns: pd.Series, confidence_level: float = 0.95) -> Figure:
        """
        Plot Value at Risk (VaR).

        Args:
            returns: Series containing returns
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Matplotlib figure
        """
        if returns.empty:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No returns data available", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Calculate VaR
        var = RiskAnalyzer.calculate_var(returns, confidence_level)

        # Create histogram of returns
        fig, ax = plt.subplots(figsize=(10, 6))

        n, bins, patches = ax.hist(returns, bins=50, alpha=0.75, density=True)

        # Highlight VaR
        for i, patch in enumerate(patches):
            if bins[i] <= -var:
                patch.set_facecolor('red')

        # Add VaR line
        ax.axvline(-var, color='red', linestyle='dashed', linewidth=2,
                   label=f'VaR ({confidence_level*100:.0f}%): {var:.2%}')

        # Add normal distribution curve
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k', linewidth=2, label='Normal Distribution')

        ax.set_title('Returns Distribution and Value at Risk (VaR)')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_drawdown(returns: pd.Series) -> Figure:
        """
        Plot drawdown.

        Args:
            returns: Series containing returns

        Returns:
            Matplotlib figure
        """
        if returns.empty:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No returns data available", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cum_returns.cummax()

        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1

        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot cumulative returns
        ax1.plot(cum_returns.index, cum_returns, label='Cumulative Returns')
        ax1.plot(running_max.index, running_max, label='Running Maximum', linestyle='--', color='green')

        # Highlight maximum drawdown
        peak_date = running_max[running_max.index <= max_drawdown_date].idxmax()
        ax1.plot([peak_date, max_drawdown_date], [cum_returns[peak_date], cum_returns[max_drawdown_date]],
                 'ro-', linewidth=2, label=f'Max Drawdown: {max_drawdown:.2%}')

        ax1.set_title('Cumulative Returns and Maximum Drawdown')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True)

        # Plot drawdown
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown, color='red', label='Drawdown')

        ax2.set_ylabel('Drawdown')
        ax2.set_ylim(min(drawdown.min() * 1.1, -0.01), 0.01)  # Set y-axis limits
        ax2.grid(True)

        plt.tight_layout()

        return fig

    @staticmethod
    def plot_efficient_frontier(returns: Dict[str, pd.Series], num_portfolios: int = 10000) -> Figure:
        """
        Plot the efficient frontier.

        Args:
            returns: Dictionary mapping symbols to Series with returns
            num_portfolios: Number of random portfolios to generate

        Returns:
            Matplotlib figure
        """
        if not returns:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No returns data available", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Combine returns into a single DataFrame
        returns_df = pd.DataFrame(returns)

        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        # Generate random portfolios
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(returns))
            weights /= np.sum(weights)
            weights_record.append(weights)

            # Calculate portfolio return and volatility
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return

            # Calculate Sharpe Ratio (assuming risk-free rate of 0)
            results[2, i] = results[1, i] / results[0, i]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot random portfolios
        scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis',
                             marker='o', s=10, alpha=0.3)

        # Find portfolio with highest Sharpe Ratio
        max_sharpe_idx = np.argmax(results[2])
        ax.scatter(results[0, max_sharpe_idx], results[1, max_sharpe_idx], c='red', marker='*', s=100,
                   label='Maximum Sharpe Ratio')

        # Find portfolio with minimum volatility
        min_vol_idx = np.argmin(results[0])
        ax.scatter(results[0, min_vol_idx], results[1, min_vol_idx], c='green', marker='*', s=100,
                   label='Minimum Volatility')

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')

        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        return fig
