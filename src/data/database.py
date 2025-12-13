"""
SQLite Database Manager for Sales Forecasting.

Provides persistent storage for:
- Input data (raw sales data)
- Training results (metrics, parameters)
- Model comparisons
- Feature importance
- Forecasts
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np


class DatabaseManager:
    """
    SQLite database manager for the sales forecasting system.
    
    Databases:
    - sales_data.db: Input data storage
    - results.db: Training results, comparisons, forecasts
    """
    
    def __init__(self, db_dir: str = "database"):
        """Initialize database manager."""
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_db_path = self.db_dir / "sales_data.db"
        self.results_db_path = self.db_dir / "results.db"
        
        self._init_databases()
    
    def _init_databases(self):
        """Initialize database schemas."""
        # Input data database
        with sqlite3.connect(self.data_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    daily_sales REAL NOT NULL,
                    marketing_spend REAL,
                    is_holiday INTEGER,
                    product_category TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_sales_date ON raw_sales(date)
            """)
        
        # Results database
        with sqlite3.connect(self.results_db_path) as conn:
            # Training runs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    train_start_date TEXT,
                    train_end_date TEXT,
                    test_start_date TEXT,
                    test_end_date TEXT,
                    train_size INTEGER,
                    test_size INTEGER,
                    train_mape REAL,
                    test_mape REAL,
                    train_mae REAL,
                    test_mae REAL,
                    train_rmse REAL,
                    test_rmse REAL,
                    best_params TEXT,
                    training_time REAL,
                    model_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model comparisons
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    rank INTEGER,
                    test_mape REAL,
                    test_mae REAL,
                    test_rmse REAL,
                    training_time REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Feature importance
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    importance_rank INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Forecasts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    forecast_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    forecast_date TEXT NOT NULL,
                    predicted_sales REAL NOT NULL,
                    day_number INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # EDA insights
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eda_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    insight_key TEXT NOT NULL,
                    insight_value TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Learning curves for ML models
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_curves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    train_loss REAL,
                    val_loss REAL,
                    metric_type TEXT DEFAULT 'rmse',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    # ==================== INPUT DATA ====================
    
    def import_sales_data(self, csv_path: str) -> int:
        """Import sales data from CSV into database."""
        df = pd.read_csv(csv_path, parse_dates=['date'])
        
        with sqlite3.connect(self.data_db_path) as conn:
            # Clear existing data
            conn.execute("DELETE FROM raw_sales")
            
            # Insert new data
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT INTO raw_sales (date, daily_sales, marketing_spend, is_holiday, product_category)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    row['date'].strftime('%Y-%m-%d'),
                    row['daily_sales'],
                    row.get('marketing_spend'),
                    row.get('is_holiday'),
                    row.get('product_category')
                ))
            
            conn.commit()
        
        return len(df)
    
    def load_sales_data(self) -> pd.DataFrame:
        """Load sales data from database."""
        with sqlite3.connect(self.data_db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, daily_sales, marketing_spend, is_holiday, product_category FROM raw_sales ORDER BY date",
                conn,
                parse_dates=['date']
            )
        return df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of stored data."""
        with sqlite3.connect(self.data_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM raw_sales")
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(date), MAX(date) FROM raw_sales")
            date_range = cursor.fetchone()
            
            cursor.execute("SELECT AVG(daily_sales), MIN(daily_sales), MAX(daily_sales) FROM raw_sales")
            stats = cursor.fetchone()
        
        return {
            'total_records': count,
            'date_start': date_range[0],
            'date_end': date_range[1],
            'avg_sales': stats[0],
            'min_sales': stats[1],
            'max_sales': stats[2]
        }
    
    # ==================== TRAINING RESULTS ====================
    
    def save_training_result(
        self,
        run_id: str,
        model_name: str,
        train_dates: tuple,
        test_dates: tuple,
        metrics: Dict[str, float],
        best_params: Dict[str, Any],
        training_time: float,
        model_path: str
    ):
        """Save training result to database."""
        with sqlite3.connect(self.results_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO training_runs 
                (run_id, model_name, train_start_date, train_end_date, test_start_date, test_end_date,
                 train_size, test_size, train_mape, test_mape, train_mae, test_mae, train_rmse, test_rmse,
                 best_params, training_time, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                model_name,
                train_dates[0],
                train_dates[1],
                test_dates[0],
                test_dates[1],
                metrics.get('train_size', 0),
                metrics.get('test_size', 0),
                metrics.get('train_mape'),
                metrics.get('test_mape'),
                metrics.get('train_mae'),
                metrics.get('test_mae'),
                metrics.get('train_rmse'),
                metrics.get('test_rmse'),
                json.dumps(best_params),
                training_time,
                model_path
            ))
            conn.commit()
    
    def get_training_results(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get training results from database."""
        query = "SELECT * FROM training_runs"
        if model_name:
            query += f" WHERE model_name = '{model_name}'"
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.results_db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get the best performing model."""
        with sqlite3.connect(self.results_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_name, test_mape, test_mae, test_rmse, best_params, model_path
                FROM training_runs
                ORDER BY test_mape ASC
                LIMIT 1
            """)
            row = cursor.fetchone()
        
        if row:
            return {
                'model_name': row[0],
                'test_mape': row[1],
                'test_mae': row[2],
                'test_rmse': row[3],
                'best_params': json.loads(row[4]) if row[4] else {},
                'model_path': row[5]
            }
        return {}
    
    # ==================== MODEL COMPARISON ====================
    
    def save_model_comparison(self, comparison_id: str, results: List[Dict[str, Any]]):
        """Save model comparison results."""
        with sqlite3.connect(self.results_db_path) as conn:
            for rank, result in enumerate(results, 1):
                conn.execute("""
                    INSERT INTO model_comparisons 
                    (comparison_id, model_name, rank, test_mape, test_mae, test_rmse, training_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    comparison_id,
                    result['model_name'],
                    rank,
                    result.get('test_mape'),
                    result.get('test_mae'),
                    result.get('test_rmse'),
                    result.get('training_time')
                ))
            conn.commit()
    
    def get_model_comparison(self, comparison_id: Optional[str] = None) -> pd.DataFrame:
        """Get model comparison results."""
        query = "SELECT * FROM model_comparisons"
        if comparison_id:
            query += f" WHERE comparison_id = '{comparison_id}'"
        query += " ORDER BY comparison_id DESC, rank ASC"
        
        with sqlite3.connect(self.results_db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    
    # ==================== FEATURE IMPORTANCE ====================
    
    def save_feature_importance(
        self,
        run_id: str,
        model_name: str,
        importance: Dict[str, float]
    ):
        """Save feature importance to database."""
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        with sqlite3.connect(self.results_db_path) as conn:
            for rank, (feature, score) in enumerate(sorted_importance, 1):
                # Convert numpy float to Python float
                score_value = float(score) if hasattr(score, 'item') else float(score)
                conn.execute("""
                    INSERT INTO feature_importance 
                    (run_id, model_name, feature_name, importance_score, importance_rank)
                    VALUES (?, ?, ?, ?, ?)
                """, (run_id, model_name, feature, score_value, rank))
            conn.commit()
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get feature importance from database."""
        query = "SELECT * FROM feature_importance"
        if model_name:
            query += f" WHERE model_name = '{model_name}'"
        query += " ORDER BY run_id DESC, importance_rank ASC"
        
        with sqlite3.connect(self.results_db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    
    # ==================== FORECASTS ====================
    
    def save_forecast(
        self,
        forecast_id: str,
        model_name: str,
        predictions: pd.DataFrame
    ):
        """Save forecast to database."""
        with sqlite3.connect(self.results_db_path) as conn:
            for i, row in predictions.iterrows():
                conn.execute("""
                    INSERT INTO forecasts 
                    (forecast_id, model_name, forecast_date, predicted_sales, day_number)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    forecast_id,
                    model_name,
                    row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    row['predicted_sales'],
                    i + 1
                ))
            conn.commit()
    
    def get_forecasts(self, model_name: Optional[str] = None, forecast_id: Optional[str] = None) -> pd.DataFrame:
        """Get forecasts from database."""
        query = "SELECT * FROM forecasts WHERE 1=1"
        if model_name:
            query += f" AND model_name = '{model_name}'"
        if forecast_id:
            query += f" AND forecast_id = '{forecast_id}'"
        query += " ORDER BY forecast_id DESC, day_number ASC"
        
        with sqlite3.connect(self.results_db_path) as conn:
            df = pd.read_sql_query(query, conn, parse_dates=['forecast_date'])
        return df
    
    def get_latest_forecast_id(self) -> Optional[str]:
        """Get the most recent forecast ID."""
        with sqlite3.connect(self.results_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT forecast_id FROM forecasts ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
        return row[0] if row else None
    
    def get_forecast_comparison(self, forecast_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get forecast comparison view (pivot table with all models side by side).
        
        Returns DataFrame with columns: date, linear_trend, xgboost, random_forest, prophet, sarima
        """
        if forecast_id is None:
            forecast_id = self.get_latest_forecast_id()
        
        if forecast_id is None:
            return pd.DataFrame()
        
        df = self.get_forecasts(forecast_id=forecast_id)
        
        if df.empty:
            return pd.DataFrame()
        
        # Pivot to get comparison view
        comparison = df.pivot_table(
            index='forecast_date',
            columns='model_name',
            values='predicted_sales',
            aggfunc='first'
        ).reset_index()
        
        comparison.columns.name = None
        comparison = comparison.rename(columns={'forecast_date': 'date'})
        
        return comparison
    
    def get_forecast_summary(self, forecast_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a forecast.
        
        Returns dict with model-wise statistics.
        """
        if forecast_id is None:
            forecast_id = self.get_latest_forecast_id()
        
        if forecast_id is None:
            return {}
        
        df = self.get_forecasts(forecast_id=forecast_id)
        
        if df.empty:
            return {}
        
        summary = {
            'forecast_id': forecast_id,
            'n_days': int(df['day_number'].max()),
            'date_range': {
                'start': df['forecast_date'].min().strftime('%Y-%m-%d'),
                'end': df['forecast_date'].max().strftime('%Y-%m-%d')
            },
            'models': {}
        }
        
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            preds = model_df['predicted_sales'].values
            summary['models'][model_name] = {
                'total': float(preds.sum()),
                'mean': float(preds.mean()),
                'min': float(preds.min()),
                'max': float(preds.max()),
                'std': float(preds.std())
            }
        
        return summary
    
    def list_forecast_ids(self) -> List[str]:
        """Get list of all forecast IDs."""
        with sqlite3.connect(self.results_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT forecast_id FROM forecasts ORDER BY created_at DESC")
            rows = cursor.fetchall()
        return [row[0] for row in rows]
    
    # ==================== EDA INSIGHTS ====================
    
    def save_eda_insights(self, analysis_id: str, insights: Dict[str, Any]):
        """Save EDA insights to database."""
        with sqlite3.connect(self.results_db_path) as conn:
            for insight_type, data in insights.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        conn.execute("""
                            INSERT INTO eda_insights (analysis_id, insight_type, insight_key, insight_value)
                            VALUES (?, ?, ?, ?)
                        """, (analysis_id, insight_type, str(key), json.dumps(value, default=str)))
                else:
                    conn.execute("""
                        INSERT INTO eda_insights (analysis_id, insight_type, insight_key, insight_value)
                        VALUES (?, ?, ?, ?)
                    """, (analysis_id, insight_type, 'value', json.dumps(data, default=str)))
            conn.commit()
    
    def get_eda_insights(self, analysis_id: Optional[str] = None) -> pd.DataFrame:
        """Get EDA insights from database."""
        query = "SELECT * FROM eda_insights"
        if analysis_id:
            query += f" WHERE analysis_id = '{analysis_id}'"
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.results_db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    
    # ==================== LEARNING CURVES ====================
    
    def save_learning_curve(
        self,
        run_id: str,
        model_name: str,
        train_losses: List[float],
        val_losses: List[float] = None,
        metric_type: str = 'rmse'
    ):
        """Save learning curve data to database."""
        with sqlite3.connect(self.results_db_path) as conn:
            for i, train_loss in enumerate(train_losses):
                val_loss = val_losses[i] if val_losses and i < len(val_losses) else None
                conn.execute("""
                    INSERT INTO learning_curves 
                    (run_id, model_name, iteration, train_loss, val_loss, metric_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (run_id, model_name, i + 1, train_loss, val_loss, metric_type))
            conn.commit()
    
    def get_learning_curve(self, run_id: str = None, model_name: str = None) -> pd.DataFrame:
        """Get learning curve data from database."""
        query = "SELECT * FROM learning_curves WHERE 1=1"
        if run_id:
            query += f" AND run_id = '{run_id}'"
        if model_name:
            query += f" AND model_name = '{model_name}'"
        query += " ORDER BY run_id DESC, iteration ASC"
        
        with sqlite3.connect(self.results_db_path) as conn:
            df = pd.read_sql_query(query, conn)
        return df


# Singleton instance
_db_manager = None

def get_database() -> DatabaseManager:
    """Get database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

