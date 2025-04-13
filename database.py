import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Tuple
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


