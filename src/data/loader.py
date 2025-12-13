"""
Data loading module for sales forecasting system.

Supports loading from:
- CSV files
- SQLite database (primary source)
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import logging

from ..utils.config import Config
from ..utils.logger import setup_logger


class DataLoader:
    """
    Handles loading of sales data from various sources.
    
    Supports:
    - CSV files with automatic date parsing
    - SQLite database (preferred for production)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.logger = setup_logger(f"{__name__}.DataLoader")
    
    def load_from_database(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from SQLite database (primary method).
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with sales data
        """
        from .database import get_input_db
        
        input_db = get_input_db()
        
        # Check if database has data
        if input_db.get_row_count('raw_sales_data') == 0:
            raise ValueError("No data in database. Please import data first using init_databases()")
        
        df = input_db.get_sales_data(start_date, end_date)
        
        self.logger.info(f"Loaded {len(df)} rows from database")
        return df
    
    def load_csv(
        self,
        file_path: Union[str, Path],
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            parse_dates: Whether to parse date columns
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or unreadable
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from CSV: {file_path}")
        
        try:
            # Load with date parsing
            date_cols = [self.config.date_column] if parse_dates else None
            df = pd.read_csv(file_path, parse_dates=date_cols)
            
            if df.empty:
                raise ValueError(f"Loaded file is empty: {file_path}")
            
            self.logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"File is empty or corrupted: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file: {e}")
    
    def load(
        self,
        source: str = 'database',
        file_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Universal load method - loads from database or CSV.
        
        Args:
            source: 'database' or 'csv'
            file_path: Path to CSV file (required if source='csv')
            **kwargs: Additional arguments (start_date, end_date for database)
            
        Returns:
            DataFrame with data
        """
        if source == 'database':
            return self.load_from_database(**kwargs)
        elif source == 'csv':
            if file_path is None:
                raise ValueError("file_path required when source='csv'")
            return self.load_csv(file_path)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'database' or 'csv'")
    
    def load_from_raw(self, filename: str) -> pd.DataFrame:
        """
        Load data from the raw data directory.
        
        Args:
            filename: Name of the file in raw data directory
            
        Returns:
            DataFrame with loaded data
        """
        file_path = self.config.raw_data_path / filename
        return self.load_csv(file_path)
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get summary information about loaded data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data summary
        """
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": df.isnull().sum().to_dict(),
            "total_missing": int(df.isnull().sum().sum())
        }
        
        # Add date range if date column exists
        if self.config.date_column in df.columns:
            date_col = df[self.config.date_column]
            if pd.api.types.is_datetime64_any_dtype(date_col):
                info["date_range"] = {
                    "start": date_col.min().strftime(self.config.date_format),
                    "end": date_col.max().strftime(self.config.date_format),
                    "days": (date_col.max() - date_col.min()).days + 1
                }
        
        return info

