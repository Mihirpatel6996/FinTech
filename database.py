import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
import json

class StockDatabase:
    def __init__(self):
        self.db_path = "stock_data.db"
        with sqlite3.connect(self.db_path) as conn:
            # Create tables if they don't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    date TEXT,
                    symbol TEXT,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    "Adj Close" REAL,  -- Note the quotes around column name with space
                    Volume INTEGER,
                    PRIMARY KEY (date, symbol)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbol_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_updated TEXT,
                    data_quality REAL,
                    first_date TEXT,
                    last_date TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    symbol TEXT,
                    prediction_date TEXT,
                    target_date TEXT,
                    model_type TEXT,
                    predicted_value REAL,
                    actual_value REAL,
                    error_rate REAL,
                    PRIMARY KEY (symbol, prediction_date, model_type)
                )
            """)

            # Add new table for visualization paths
            conn.execute("""
                CREATE TABLE IF NOT EXISTS visualization_paths (
                    symbol TEXT,
                    model_type TEXT,
                    file_path TEXT,
                    created_date TEXT,
                    PRIMARY KEY (symbol, model_type)
                )
            """)

            # Add new table for prediction cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    symbol TEXT PRIMARY KEY,
                    predictions TEXT,  -- JSON string of predictions
                    timestamp TEXT,
                    last_updated DATE
                )
            """)

            # Add new table for model training history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_training_history (
                    symbol TEXT,
                    model_type TEXT,
                    training_date TEXT,
                    data_points INTEGER,
                    incremental BOOLEAN,
                    training_error REAL,
                    PRIMARY KEY (symbol, model_type, training_date)
                )
            """)

            # Add new table for ensemble weights
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_weights (
                    symbol TEXT PRIMARY KEY,
                    weights TEXT,  -- JSON string of weights
                    timestamp TEXT
                )
            """)

            # Add new table for sentiment analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    symbol TEXT,
                    analysis_date TEXT,
                    sentiment_score REAL,
                    news_count INTEGER,
                    news_data TEXT,  -- JSON string of news articles and their sentiment
                    PRIMARY KEY (symbol, analysis_date)
                )
            """)

            # Add tables for portfolio management
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_date TEXT,
                    last_updated TEXT
                )
            """)

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

            # Add table for technical analysis results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS technical_analysis (
                    symbol TEXT,
                    analysis_date TEXT,
                    indicators TEXT,  -- JSON string of technical indicators
                    overall_signal TEXT,
                    PRIMARY KEY (symbol, analysis_date)
                )
            """)

    async def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM stock_data
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            # Convert dates to strings in ISO format
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            df = pd.read_sql_query(query, conn, params=(symbol, start_str, end_str))

            if not df.empty:
                # Convert date string back to datetime
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df if not df.empty else None

    def store_stock_data(self, symbol: str, df: pd.DataFrame):
        with sqlite3.connect(self.db_path) as conn:
            # Prepare the dataframe for storage
            df_to_store = df.copy()
            df_to_store['symbol'] = symbol

            # Ensure column names match the database schema
            df_to_store = df_to_store.rename(columns={
                'Adj Close': 'Adj Close',  # Keep the space in column name
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })

            # Delete existing data for this symbol
            conn.execute("DELETE FROM stock_data WHERE symbol = ?", (symbol,))

            # Insert new data
            df_to_store.to_sql('stock_data', conn, if_exists='append', index=True, index_label='date')

            # Update metadata
            conn.execute("""
                INSERT OR REPLACE INTO symbol_metadata
                (symbol, last_updated, data_quality, first_date, last_date)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                1.0,  # Default quality score
                df.index.min().strftime('%Y-%m-%d'),
                df.index.max().strftime('%Y-%m-%d')
            ))

    def store_prediction(self, symbol: str, model_type: str, prediction: float, target_date: datetime):
        with sqlite3.connect(self.db_path) as conn:
            # Use INSERT OR REPLACE to handle duplicate entries
            conn.execute("""
                INSERT OR REPLACE INTO prediction_history
                (symbol, prediction_date, target_date, model_type, predicted_value)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                target_date.strftime('%Y-%m-%d'),
                model_type,
                prediction
            ))

    def update_prediction_accuracy(self, symbol: str, actual_value: float, prediction_date: datetime, model_type: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE prediction_history
                SET actual_value = ?, error_rate = ABS(predicted_value - ?) / ?
                WHERE symbol = ? AND prediction_date = ? AND model_type = ?
            """, (
                actual_value,
                actual_value,
                actual_value,
                symbol,
                prediction_date.strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
                model_type
            ))

    def needs_update(self, symbol: str, max_age_hours: int = 24) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            threshold_time = (datetime.now() - timedelta(hours=max_age_hours)).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("""
                SELECT last_updated FROM symbol_metadata
                WHERE symbol = ? AND last_updated > ?
            """, (symbol, threshold_time))
            return cursor.fetchone() is None

    def store_visualization_path(self, symbol: str, model_type: str, file_path: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO visualization_paths
                (symbol, model_type, file_path, created_date)
                VALUES (?, ?, ?, ?)
            """, (
                symbol,
                model_type,
                file_path,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))

    def get_visualization_paths(self, symbol: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_type, file_path, created_date
                FROM visualization_paths
                WHERE symbol = ?
            """, (symbol,))

            results = cursor.fetchall()
            return {row[0]: {'path': row[1], 'created_date': row[2]} for row in results}

    def needs_prediction_update(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT last_updated FROM prediction_cache
                WHERE symbol = ?
            """, (symbol,))
            result = cursor.fetchone()

            if not result:
                return True

            last_update = datetime.strptime(result[0], '%Y-%m-%d').date()
            current_date = datetime.now().date()

            return current_date > last_update

    def get_cached_predictions(self, symbol: str) -> dict:
        """Get cached predictions from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT predictions FROM prediction_cache
                WHERE symbol = ?
            """, (symbol,))
            result = cursor.fetchone()

            if result:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    return None
            return None

    def store_cached_predictions(self, symbol: str, predictions: str):
        """Store predictions in database. Predictions should be a JSON string."""
        with sqlite3.connect(self.db_path) as conn:
            current_date = datetime.now().date().strftime('%Y-%m-%d')
            conn.execute("""
                INSERT OR REPLACE INTO prediction_cache
                (symbol, predictions, timestamp, last_updated)
                VALUES (?, ?, ?, ?)
            """, (
                symbol,
                predictions,  # Already a JSON string
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                current_date
            ))

    def store_model_training(self, symbol: str, model_type: str, data_points: int, incremental: bool, training_error: float):
        """Store model training information in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_training_history
                (symbol, model_type, training_date, data_points, incremental, training_error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                model_type,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data_points,
                incremental,
                training_error
            ))

    def get_ensemble_weights(self, symbol: str) -> Dict[str, float]:
        """Get ensemble weights from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT weights FROM ensemble_weights
                WHERE symbol = ?
            """, (symbol,))

            result = cursor.fetchone()
            if result:
                try:
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    pass

            # Default weights if none found
            return {
                "ARIMA": 0.33,
                "LSTM": 0.33,
                "LINEAR": 0.34
            }

    def get_model_performance(self, symbol: str, model_type: str, days: int = 30) -> float:
        """Get average error rate for a model over the specified number of days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            threshold_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            cursor.execute("""
                SELECT AVG(error_rate) FROM prediction_history
                WHERE symbol = ? AND model_type = ?
                AND prediction_date >= ? AND actual_value IS NOT NULL
            """, (symbol, model_type, threshold_date))

            result = cursor.fetchone()
            if result and result[0] is not None:
                return float(result[0])
            return 1.0  # Default high error if no data

    def get_pending_predictions(self, symbol: str, target_date: str) -> List[Tuple[str, str, float]]:
        """Get predictions that need to be updated with actual values"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT prediction_date, model_type, predicted_value
                FROM prediction_history
                WHERE symbol = ?
                AND date(target_date) = date(?)
                AND actual_value IS NULL
            """, (symbol, target_date))

            return cursor.fetchall()

    def store_sentiment_analysis(self, symbol: str, sentiment_score: float, news_count: int, news_data: str):
        """Store sentiment analysis results in database"""
        with sqlite3.connect(self.db_path) as conn:
            current_date = datetime.now().strftime('%Y-%m-%d')
            conn.execute("""
                INSERT OR REPLACE INTO sentiment_analysis
                (symbol, analysis_date, sentiment_score, news_count, news_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol,
                current_date,
                sentiment_score,
                news_count,
                news_data  # JSON string of news articles
            ))

    def get_sentiment_analysis(self, symbol: str) -> Dict:
        """Get sentiment analysis results for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT analysis_date, sentiment_score, news_count, news_data
                FROM sentiment_analysis
                WHERE symbol = ?
                ORDER BY analysis_date DESC
                LIMIT 1
            """, (symbol,))

            result = cursor.fetchone()
            if result:
                return {
                    "analysis_date": result[0],
                    "sentiment_score": result[1],
                    "news_count": result[2],
                    "news_data": result[3]  # JSON string of news articles
                }
            return {}

