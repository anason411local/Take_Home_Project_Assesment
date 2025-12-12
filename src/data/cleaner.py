"""
Data cleaning module for sales forecasting system.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging

from ..utils.config import Config
from ..utils.logger import setup_logger


class DataCleaner:
    """
    Cleans and preprocesses sales data.
    
    Handles:
    - Missing value imputation
    - Date parsing and standardization
    - Data type conversions
    - Outlier handling
    - Column renaming/standardization
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataCleaner.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.logger = setup_logger(f"{__name__}.DataCleaner")
        self._cleaning_report = []
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning operations to the data.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning pipeline...")
        self._cleaning_report = []
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Apply cleaning steps in order
        df_clean = self._standardize_column_names(df_clean)
        df_clean = self._parse_dates(df_clean)
        df_clean = self._convert_data_types(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._sort_by_date(df_clean)
        df_clean = self._reset_index(df_clean)
        
        self.logger.info(f"Cleaning complete. Applied {len(self._cleaning_report)} operations.")
        return df_clean
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        original_cols = df.columns.tolist()
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        renamed = {old: new for old, new in zip(original_cols, df.columns) if old != new}
        if renamed:
            self._cleaning_report.append(f"Renamed columns: {renamed}")
            self.logger.debug(f"Renamed columns: {renamed}")
        
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column to datetime type."""
        date_col = self.config.date_column
        
        if date_col not in df.columns:
            self.logger.warning(f"Date column '{date_col}' not found")
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=self.config.date_format)
                self._cleaning_report.append(f"Parsed '{date_col}' to datetime")
                self.logger.debug(f"Parsed '{date_col}' to datetime")
            except Exception as e:
                # Try automatic parsing
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                self._cleaning_report.append(f"Parsed '{date_col}' to datetime (inferred format)")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        # Numeric columns
        numeric_cols = [self.config.target_column, 'marketing_spend']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                self._cleaning_report.append(f"Converted '{col}' to numeric")
        
        # Boolean/binary columns
        if 'is_holiday' in df.columns:
            df['is_holiday'] = df['is_holiday'].astype(int)
        
        # Categorical columns
        categorical_cols = ['product_category', 'day_of_week']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                self._cleaning_report.append(f"Converted '{col}' to category")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        initial_nulls = df.isnull().sum().sum()
        
        if initial_nulls == 0:
            self.logger.debug("No missing values found")
            return df
        
        # For numeric columns, use forward fill then backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].ffill().bfill()
        
        # For categorical columns, use mode
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        
        final_nulls = df.isnull().sum().sum()
        filled = initial_nulls - final_nulls
        
        if filled > 0:
            self._cleaning_report.append(f"Filled {filled} missing values")
            self.logger.debug(f"Filled {filled} missing values")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows, keeping first occurrence."""
        initial_rows = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicate dates (keep first)
        if self.config.date_column in df.columns:
            df = df.drop_duplicates(subset=[self.config.date_column], keep='first')
        
        removed = initial_rows - len(df)
        if removed > 0:
            self._cleaning_report.append(f"Removed {removed} duplicate rows")
            self.logger.debug(f"Removed {removed} duplicate rows")
        
        return df
    
    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by date column."""
        if self.config.date_column in df.columns:
            df = df.sort_values(self.config.date_column)
            self._cleaning_report.append("Sorted by date")
        return df
    
    def _reset_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reset index after cleaning operations."""
        return df.reset_index(drop=True)
    
    def get_cleaning_report(self) -> List[str]:
        """Get list of cleaning operations performed."""
        return self._cleaning_report.copy()
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'clip',
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Args:
            df: DataFrame to process
            columns: Columns to check (default: target and marketing_spend)
            method: 'clip' to cap values, 'remove' to drop rows
            n_std: Number of standard deviations for outlier threshold
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = [self.config.target_column, 'marketing_spend']
        
        columns = [c for c in columns if c in df.columns]
        
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower = mean - n_std * std
            upper = mean + n_std * std
            
            outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                if method == 'clip':
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                    self.logger.info(f"Clipped {outlier_count} outliers in '{col}'")
                elif method == 'remove':
                    df_clean = df_clean[~outlier_mask]
                    self.logger.info(f"Removed {outlier_count} outlier rows for '{col}'")
        
        return df_clean

