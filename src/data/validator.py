"""
Data validation module for sales forecasting system.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import logging

from ..utils.config import Config
from ..utils.logger import setup_logger


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def summary(self) -> str:
        """Get a summary of validation results."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Validation Status: {status}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}"
        ]
        
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Validates sales data for quality and consistency.
    
    Performs comprehensive checks including:
    - Schema validation (required columns, data types)
    - Value range validation
    - Date continuity checks
    - Duplicate detection
    - Outlier detection
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataValidator.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.logger = setup_logger(f"{__name__}.DataValidator")
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks on the data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with all findings
        """
        result = ValidationResult(is_valid=True)
        
        self.logger.info("Starting data validation...")
        
        # Run all validation checks
        self._validate_schema(df, result)
        self._validate_missing_values(df, result)
        self._validate_data_types(df, result)
        self._validate_value_ranges(df, result)
        self._validate_date_continuity(df, result)
        self._validate_duplicates(df, result)
        self._detect_outliers(df, result)
        
        # Calculate summary statistics
        result.statistics = self._calculate_statistics(df)
        
        status = "passed" if result.is_valid else "failed"
        self.logger.info(f"Validation {status} with {len(result.errors)} errors and {len(result.warnings)} warnings")
        
        return result
    
    def _validate_schema(self, df: pd.DataFrame, result: ValidationResult):
        """Check if all required columns are present."""
        missing_cols = set(self.config.expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(self.config.expected_columns)
        
        if missing_cols:
            result.add_error(f"Missing required columns: {missing_cols}")
        
        if extra_cols:
            result.add_warning(f"Unexpected columns found: {extra_cols}")
    
    def _validate_missing_values(self, df: pd.DataFrame, result: ValidationResult):
        """Check for missing values in critical columns."""
        critical_cols = [self.config.date_column, self.config.target_column]
        
        for col in critical_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    result.add_error(f"Column '{col}' has {missing_count} missing values")
        
        # Check other columns for missing values (warning only)
        for col in df.columns:
            if col not in critical_cols:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    result.add_warning(f"Column '{col}' has {missing_count} missing values ({pct:.1f}%)")
    
    def _validate_data_types(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data types of columns."""
        # Date column should be datetime
        if self.config.date_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[self.config.date_column]):
                result.add_warning(f"Column '{self.config.date_column}' is not datetime type")
        
        # Sales and marketing spend should be numeric
        numeric_cols = [self.config.target_column, 'marketing_spend']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    result.add_error(f"Column '{col}' should be numeric but is {df[col].dtype}")
    
    def _validate_value_ranges(self, df: pd.DataFrame, result: ValidationResult):
        """Check if values are within expected ranges."""
        # Daily sales
        if self.config.target_column in df.columns:
            col = df[self.config.target_column]
            
            if col.min() < self.config.min_sales:
                result.add_error(f"Daily sales has values below minimum ({col.min()} < {self.config.min_sales})")
            
            if col.max() > self.config.max_sales:
                result.add_warning(f"Daily sales has unusually high values ({col.max()} > {self.config.max_sales})")
        
        # Marketing spend
        if 'marketing_spend' in df.columns:
            col = df['marketing_spend']
            
            if col.min() < self.config.min_marketing_spend:
                result.add_error(f"Marketing spend has negative values ({col.min()})")
            
            if col.max() > self.config.max_marketing_spend:
                result.add_warning(f"Marketing spend has unusually high values ({col.max()})")
        
        # Holiday flag should be binary
        if 'is_holiday' in df.columns:
            unique_vals = df['is_holiday'].unique()
            if not set(unique_vals).issubset({0, 1}):
                result.add_error(f"Holiday flag should be binary (0/1), found: {unique_vals}")
    
    def _validate_date_continuity(self, df: pd.DataFrame, result: ValidationResult):
        """Check for gaps in date sequence."""
        if self.config.date_column not in df.columns:
            return
        
        date_col = df[self.config.date_column]
        if not pd.api.types.is_datetime64_any_dtype(date_col):
            return
        
        # Sort by date and check gaps
        sorted_dates = date_col.sort_values()
        date_diffs = sorted_dates.diff().dropna()
        
        # Check for gaps (more than 1 day)
        gaps = date_diffs[date_diffs > pd.Timedelta(days=1)]
        if len(gaps) > 0:
            gap_dates = sorted_dates[gaps.index - 1].tolist()
            result.add_warning(f"Found {len(gaps)} date gaps in the data")
            result.statistics['date_gaps'] = len(gaps)
        
        # Check for duplicate dates
        duplicate_dates = sorted_dates[sorted_dates.duplicated()]
        if len(duplicate_dates) > 0:
            result.add_error(f"Found {len(duplicate_dates)} duplicate dates")
    
    def _validate_duplicates(self, df: pd.DataFrame, result: ValidationResult):
        """Check for duplicate rows."""
        duplicates = df.duplicated()
        if duplicates.sum() > 0:
            result.add_warning(f"Found {duplicates.sum()} duplicate rows")
    
    def _detect_outliers(self, df: pd.DataFrame, result: ValidationResult):
        """Detect statistical outliers using IQR method."""
        numeric_cols = [self.config.target_column, 'marketing_spend']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                pct = (len(outliers) / len(df)) * 100
                result.add_warning(f"Column '{col}' has {len(outliers)} outliers ({pct:.1f}%)")
                result.statistics[f'{col}_outliers'] = len(outliers)
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the data."""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Numeric column statistics
        if self.config.target_column in df.columns:
            col = df[self.config.target_column]
            stats['sales'] = {
                'mean': float(col.mean()),
                'std': float(col.std()),
                'min': float(col.min()),
                'max': float(col.max()),
                'median': float(col.median())
            }
        
        # Date range
        if self.config.date_column in df.columns:
            date_col = df[self.config.date_column]
            if pd.api.types.is_datetime64_any_dtype(date_col):
                stats['date_range'] = {
                    'start': str(date_col.min().date()),
                    'end': str(date_col.max().date()),
                    'days': int((date_col.max() - date_col.min()).days) + 1
                }
        
        return stats

