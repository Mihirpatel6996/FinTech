"""
Portfolio Management Module

This module provides functionality for portfolio tracking, diversification recommendations,
risk assessment, and portfolio rebalancing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from datetime import datetime, timedelta
import json
from matplotlib.figure import Figure


class PortfolioManager:
    """
    Class for managing stock portfolios.
    """

    def __init__(self, db_path: str = "stock_data.db"):
        """
        Initialize the portfolio manager.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database with portfolio tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Create portfolios table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_date TEXT,
                    last_updated TEXT
                )
            """)

            # Create portfolio_holdings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    purchase_price REAL NOT NULL,
                    purchase_date TEXT,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
                )
            """)

            # Create portfolio_transactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,  -- 'BUY' or 'SELL'
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    transaction_date TEXT,
                    notes TEXT,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
                )
            """)

    def create_portfolio(self, name: str, description: str = "") -> int:
        """
        Create a new portfolio.

        Args:
            name: Portfolio name
            description: Portfolio description

        Returns:
            Portfolio ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute("""
                INSERT INTO portfolios (name, description, created_date, last_updated)
                VALUES (?, ?, ?, ?)
            """, (name, description, current_date, current_date))

            return cursor.lastrowid

    def get_portfolios(self) -> pd.DataFrame:
        """
        Get all portfolios.

        Returns:
            DataFrame containing all portfolios
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM portfolios", conn)

    def add_holding(self, portfolio_id: int, symbol: str, quantity: float, purchase_price: float, purchase_date: str = None) -> int:
        """
        Add a holding to a portfolio.

        Args:
            portfolio_id: Portfolio ID
            symbol: Stock symbol
            quantity: Number of shares
            purchase_price: Price per share
            purchase_date: Date of purchase (default: current date)

        Returns:
            Holding ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if purchase_date is None:
                purchase_date = datetime.now().strftime('%Y-%m-%d')

            # Add holding
            cursor.execute("""
                INSERT INTO portfolio_holdings (portfolio_id, symbol, quantity, purchase_price, purchase_date)
                VALUES (?, ?, ?, ?, ?)
            """, (portfolio_id, symbol, quantity, purchase_price, purchase_date))

            # Record transaction
            cursor.execute("""
                INSERT INTO portfolio_transactions (portfolio_id, symbol, transaction_type, quantity, price, transaction_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (portfolio_id, symbol, 'BUY', quantity, purchase_price, purchase_date))

            # Update portfolio last_updated
            cursor.execute("""
                UPDATE portfolios
                SET last_updated = ?
                WHERE portfolio_id = ?
            """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), portfolio_id))

            return cursor.lastrowid

    def remove_holding(self, portfolio_id: int, symbol: str, quantity: float, sell_price: float, sell_date: str = None) -> bool:
        """
        Remove a holding from a portfolio (sell shares).

        Args:
            portfolio_id: Portfolio ID
            symbol: Stock symbol
            quantity: Number of shares to sell
            sell_price: Price per share
            sell_date: Date of sale (default: current date)

        Returns:
            True if successful, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if sell_date is None:
                sell_date = datetime.now().strftime('%Y-%m-%d')

            # Get current holdings
            cursor.execute("""
                SELECT holding_id, quantity
                FROM portfolio_holdings
                WHERE portfolio_id = ? AND symbol = ?
            """, (portfolio_id, symbol))

            holdings = cursor.fetchall()

            if not holdings:
                return False

            total_quantity = sum(holding[1] for holding in holdings)

            if total_quantity < quantity:
                return False

            # Record transaction
            cursor.execute("""
                INSERT INTO portfolio_transactions (portfolio_id, symbol, transaction_type, quantity, price, transaction_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (portfolio_id, symbol, 'SELL', quantity, sell_price, sell_date))

            # Update holdings
            remaining_to_sell = quantity

            for holding_id, holding_quantity in holdings:
                if remaining_to_sell <= 0:
                    break

                if holding_quantity <= remaining_to_sell:
                    # Remove entire holding
                    cursor.execute("""
                        DELETE FROM portfolio_holdings
                        WHERE holding_id = ?
                    """, (holding_id,))

                    remaining_to_sell -= holding_quantity
                else:
                    # Reduce holding quantity
                    cursor.execute("""
                        UPDATE portfolio_holdings
                        SET quantity = ?
                        WHERE holding_id = ?
                    """, (holding_quantity - remaining_to_sell, holding_id))

                    remaining_to_sell = 0

            # Update portfolio last_updated
            cursor.execute("""
                UPDATE portfolios
                SET last_updated = ?
                WHERE portfolio_id = ?
            """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), portfolio_id))

            return True

    def get_portfolio_holdings(self, portfolio_id: int) -> pd.DataFrame:
        """
        Get holdings for a portfolio.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            DataFrame containing portfolio holdings
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT symbol, SUM(quantity) as total_quantity,
                       AVG(purchase_price) as avg_purchase_price,
                       SUM(quantity * purchase_price) as total_cost
                FROM portfolio_holdings
                WHERE portfolio_id = ?
                GROUP BY symbol
            """, conn, params=(portfolio_id,))

    def get_portfolio_transactions(self, portfolio_id: int) -> pd.DataFrame:
        """
        Get transactions for a portfolio.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            DataFrame containing portfolio transactions
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT * FROM portfolio_transactions
                WHERE portfolio_id = ?
                ORDER BY transaction_date DESC
            """, conn, params=(portfolio_id,))

    def get_portfolio_value(self, portfolio_id: int, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate the current value of a portfolio.

        Args:
            portfolio_id: Portfolio ID
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            Dictionary containing portfolio value information
        """
        holdings = self.get_portfolio_holdings(portfolio_id)

        if holdings.empty:
            return {
                "total_value": 0.0,
                "total_cost": 0.0,
                "total_gain_loss": 0.0,
                "total_gain_loss_percent": 0.0,
                "holdings": []
            }

        holdings_list = []
        total_value = 0.0
        total_cost = 0.0

        for _, row in holdings.iterrows():
            symbol = row['symbol']
            quantity = row['total_quantity']
            avg_price = row['avg_purchase_price']
            cost = row['total_cost']

            current_price = current_prices.get(symbol, 0.0)
            current_value = quantity * current_price
            gain_loss = current_value - cost
            gain_loss_percent = (gain_loss / cost) * 100 if cost > 0 else 0.0

            holdings_list.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_purchase_price": avg_price,
                "current_price": current_price,
                "cost": cost,
                "current_value": current_value,
                "gain_loss": gain_loss,
                "gain_loss_percent": gain_loss_percent
            })

            total_value += current_value
            total_cost += cost

        total_gain_loss = total_value - total_cost
        total_gain_loss_percent = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0.0

        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_gain_loss": total_gain_loss,
            "total_gain_loss_percent": total_gain_loss_percent,
            "holdings": holdings_list
        }

    def calculate_portfolio_allocation(self, portfolio_id: int, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate the current allocation of a portfolio.

        Args:
            portfolio_id: Portfolio ID
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            Dictionary containing portfolio allocation information
        """
        portfolio_value = self.get_portfolio_value(portfolio_id, current_prices)
        total_value = portfolio_value["total_value"]

        if total_value == 0:
            return {
                "allocations": [],
                "allocation_by_sector": {}
            }

        # Calculate allocation percentages
        allocations = []
        for holding in portfolio_value["holdings"]:
            allocation_percent = (holding["current_value"] / total_value) * 100
            allocations.append({
                "symbol": holding["symbol"],
                "value": holding["current_value"],
                "allocation_percent": allocation_percent
            })

        # TODO: Add sector allocation (would require sector data for each symbol)

        return {
            "allocations": allocations,
            "allocation_by_sector": {}  # Placeholder for sector allocation
        }

    def plot_portfolio_allocation(self, portfolio_id: int, current_prices: Dict[str, float]) -> Figure:
        """
        Create a pie chart of portfolio allocation.

        Args:
            portfolio_id: Portfolio ID
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            Matplotlib figure
        """
        allocation = self.calculate_portfolio_allocation(portfolio_id, current_prices)

        if not allocation["allocations"]:
            # Create empty pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No holdings in portfolio", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Extract data for pie chart
        labels = [item["symbol"] for item in allocation["allocations"]]
        sizes = [item["allocation_percent"] for item in allocation["allocations"]]

        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        plt.title('Portfolio Allocation')
        plt.tight_layout()

        return fig

    def plot_portfolio_performance(self, portfolio_id: int, historical_prices: Dict[str, pd.DataFrame]) -> Figure:
        """
        Create a line chart of portfolio performance over time.

        Args:
            portfolio_id: Portfolio ID
            historical_prices: Dictionary mapping symbols to DataFrames with historical prices

        Returns:
            Matplotlib figure
        """
        holdings = self.get_portfolio_holdings(portfolio_id)

        if holdings.empty:
            # Create empty chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No holdings in portfolio", ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig

        # Get transactions to determine when holdings were added/removed
        transactions = self.get_portfolio_transactions(portfolio_id)

        # Find the earliest transaction date
        if transactions.empty:
            return plt.figure()  # Return empty figure if no transactions

        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        start_date = transactions['transaction_date'].min()
        end_date = datetime.now()

        # Create a date range for the portfolio value calculation
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        portfolio_values = pd.Series(index=date_range, dtype=float)

        # Calculate portfolio value for each date
        for date in date_range:
            # Get transactions up to this date
            past_transactions = transactions[transactions['transaction_date'] <= date]

            # Calculate holdings as of this date
            holdings_as_of_date = {}
            for _, txn in past_transactions.iterrows():
                symbol = txn['symbol']
                if txn['transaction_type'] == 'BUY':
                    holdings_as_of_date[symbol] = holdings_as_of_date.get(symbol, 0) + txn['quantity']
                else:  # SELL
                    holdings_as_of_date[symbol] = holdings_as_of_date.get(symbol, 0) - txn['quantity']

            # Remove symbols with zero quantity
            holdings_as_of_date = {k: v for k, v in holdings_as_of_date.items() if v > 0}

            # Calculate portfolio value as of this date
            portfolio_value = 0.0
            for symbol, quantity in holdings_as_of_date.items():
                if symbol in historical_prices:
                    # Find the closest date in historical prices
                    price_data = historical_prices[symbol]
                    price_data.index = pd.to_datetime(price_data.index)
                    closest_date = price_data.index[price_data.index <= date]

                    if not closest_date.empty:
                        closest_date = closest_date[-1]
                        price = price_data.loc[closest_date, 'Close']
                        portfolio_value += quantity * price

            portfolio_values[date] = portfolio_value

        # Plot portfolio value over time
        fig, ax = plt.subplots(figsize=(12, 6))
        portfolio_values.plot(ax=ax)

        ax.set_title('Portfolio Performance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True)

        plt.tight_layout()

        return fig

    def calculate_diversification_score(self, portfolio_id: int, current_prices: Dict[str, float]) -> float:
        """
        Calculate a diversification score for the portfolio.

        Args:
            portfolio_id: Portfolio ID
            current_prices: Dictionary mapping symbols to current prices

        Returns:
            Diversification score (0-100)
        """
        allocation = self.calculate_portfolio_allocation(portfolio_id, current_prices)

        if not allocation["allocations"]:
            return 0.0

        # Calculate Herfindahl-Hirschman Index (HHI)
        # HHI is the sum of squared market shares (in this case, allocation percentages)
        hhi = sum((item["allocation_percent"] / 100) ** 2 for item in allocation["allocations"])

        # Normalize HHI to a 0-100 scale where 100 is perfectly diversified
        # For a portfolio with n equal-weighted assets, HHI = 1/n
        # So a score of 100 would correspond to a very large number of equal-weighted assets
        n = len(allocation["allocations"])
        min_hhi = 1 / n  # Best possible HHI for n assets
        max_hhi = 1.0    # Worst possible HHI (one asset has 100%)

        # Normalize to 0-100 scale
        if max_hhi == min_hhi:  # Only one asset
            return 0.0

        normalized_score = 100 * (1 - (hhi - min_hhi) / (max_hhi - min_hhi))

        return normalized_score

    def get_rebalancing_recommendations(self, portfolio_id: int, current_prices: Dict[str, float], target_allocation: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Get recommendations for rebalancing a portfolio.

        Args:
            portfolio_id: Portfolio ID
            current_prices: Dictionary mapping symbols to current prices
            target_allocation: Dictionary mapping symbols to target allocation percentages

        Returns:
            List of rebalancing recommendations
        """
        portfolio_value = self.get_portfolio_value(portfolio_id, current_prices)
        total_value = portfolio_value["total_value"]

        if total_value == 0:
            return []

        current_allocation = self.calculate_portfolio_allocation(portfolio_id, current_prices)
        current_by_symbol = {item["symbol"]: item["allocation_percent"] for item in current_allocation["allocations"]}

        recommendations = []

        # Check for symbols in the portfolio that are not in the target allocation
        for holding in portfolio_value["holdings"]:
            symbol = holding["symbol"]
            if symbol not in target_allocation:
                recommendations.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "current_allocation": current_by_symbol.get(symbol, 0.0),
                    "target_allocation": 0.0,
                    "difference": -current_by_symbol.get(symbol, 0.0),
                    "current_value": holding["current_value"],
                    "target_value": 0.0,
                    "value_difference": -holding["current_value"]
                })

        # Check for symbols in the target allocation
        for symbol, target_percent in target_allocation.items():
            current_percent = current_by_symbol.get(symbol, 0.0)
            difference = target_percent - current_percent

            current_value = 0.0
            for holding in portfolio_value["holdings"]:
                if holding["symbol"] == symbol:
                    current_value = holding["current_value"]
                    break

            target_value = total_value * (target_percent / 100)
            value_difference = target_value - current_value

            action = "HOLD"
            if abs(difference) >= 1.0:  # Only recommend changes for differences >= 1%
                action = "BUY" if difference > 0 else "SELL"

            recommendations.append({
                "symbol": symbol,
                "action": action,
                "current_allocation": current_percent,
                "target_allocation": target_percent,
                "difference": difference,
                "current_value": current_value,
                "target_value": target_value,
                "value_difference": value_difference
            })

        # Sort recommendations by absolute difference (largest first)
        recommendations.sort(key=lambda x: abs(x["difference"]), reverse=True)

        return recommendations

    def get_portfolios_containing_stock(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all portfolios that contain a specific stock.

        Args:
            symbol: The stock symbol to search for

        Returns:
            A list of dictionaries with portfolio information and share count
        """
        portfolios = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Query to find portfolios containing the stock
                query = """
                    SELECT p.portfolio_id, p.name, SUM(h.quantity) as shares
                    FROM portfolios p
                    JOIN portfolio_holdings h ON p.portfolio_id = h.portfolio_id
                    WHERE h.symbol = ? AND h.quantity > 0
                    GROUP BY p.portfolio_id, p.name
                """

                cursor = conn.execute(query, (symbol,))
                rows = cursor.fetchall()

                for row in rows:
                    portfolios.append({
                        "id": row["portfolio_id"],
                        "name": row["name"],
                        "shares": row["shares"]
                    })

        except Exception as e:
            print(f"Error getting portfolios containing {symbol}: {str(e)}")

        return portfolios
