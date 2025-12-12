"""
Feature engineering module for sales forecasting system.

Creates features for time series forecasting including:
- Lag features (previous day values)
- Rolling statistics (moving averages, std)
- Date/calendar features
- Trend features
- Cyclical encodings
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

from ..utils.config import Config
from ..utils.logger import setup_logger


class FeatureEngineer:
    """
    Engineers features from sales time series data.
    
    Supports two modes:
    - 'minimal': Essential features only (~15 features) - recommended for small datasets
    - 'full': Comprehensive feature set (~68 features) - for larger datasets
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.logger = setup_logger(f"{__name__}.FeatureEngineer")
        self._feature_names = []
    
    def create_features(
        self,
        df: pd.DataFrame,
        mode: str = 'minimal',
        include_target_lags: bool = True
    ) -> pd.DataFrame:
        """
        Create features from the input data.
        
        Args:
            df: DataFrame with sales data
            mode: 'minimal' (~15 features) or 'full' (~68 features)
            include_target_lags: Whether to create lag features of target
            
        Returns:
            DataFrame with all engineered features
        """
        self.logger.info(f"Starting feature engineering (mode={mode})...")
        self._feature_names = []
        
        # Ensure data is sorted by date
        df = df.sort_values(self.config.date_column).copy()
        
        if mode == 'minimal':
            df = self._create_minimal_features(df, include_target_lags)
        else:
            df = self._create_full_features(df, include_target_lags)
        
        self.logger.info(f"Created {len(self._feature_names)} new features")
        return df
    
    def _create_minimal_features(self, df: pd.DataFrame, include_target_lags: bool) -> pd.DataFrame:
        """
        Create minimal, essential feature set for small datasets.
        
        Features (~15):
        - 5 date features: day_of_week_num, month, is_weekend, day_of_year, quarter
        - 3 lag features: lag_1, lag_7 (for target)
        - 3 rolling features: rolling_mean_7, rolling_std_7
        - 2 trend features: days_since_start, time_normalized
        - 2 cyclical: day_of_week_sin, day_of_week_cos
        """
        date_col = df[self.config.date_column]
        target_col = self.config.target_column
        
        # === Date Features (5) ===
        df['day_of_week_num'] = date_col.dt.dayofweek
        df['month'] = date_col.dt.month
        df['quarter'] = date_col.dt.quarter
        df['day_of_year'] = date_col.dt.dayofyear
        df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)
        
        date_features = ['day_of_week_num', 'month', 'quarter', 'day_of_year', 'is_weekend']
        self._feature_names.extend(date_features)
        
        # === Cyclical Features (2) ===
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
        
        cyclical_features = ['day_of_week_sin', 'day_of_week_cos']
        self._feature_names.extend(cyclical_features)
        
        # === Lag Features (3) ===
        if include_target_lags:
            df[f'{target_col}_lag_1'] = df[target_col].shift(1)
            df[f'{target_col}_lag_7'] = df[target_col].shift(7)
            df[f'{target_col}_diff_1'] = df[target_col].diff(1)
            
            lag_features = [f'{target_col}_lag_1', f'{target_col}_lag_7', f'{target_col}_diff_1']
            self._feature_names.extend(lag_features)
        
        # === Rolling Features (3) ===
        df[f'{target_col}_rolling_mean_7'] = df[target_col].shift(1).rolling(window=7, min_periods=1).mean()
        df[f'{target_col}_rolling_std_7'] = df[target_col].shift(1).rolling(window=7, min_periods=2).std()
        
        if 'marketing_spend' in df.columns:
            df['marketing_spend_rolling_mean_7'] = df['marketing_spend'].shift(1).rolling(window=7, min_periods=1).mean()
            rolling_features = [f'{target_col}_rolling_mean_7', f'{target_col}_rolling_std_7', 'marketing_spend_rolling_mean_7']
        else:
            rolling_features = [f'{target_col}_rolling_mean_7', f'{target_col}_rolling_std_7']
        
        self._feature_names.extend(rolling_features)
        
        # === Trend Features (2) ===
        df['days_since_start'] = (date_col - date_col.min()).dt.days
        total_days = df['days_since_start'].max()
        df['time_normalized'] = df['days_since_start'] / total_days if total_days > 0 else 0
        
        trend_features = ['days_since_start', 'time_normalized']
        self._feature_names.extend(trend_features)
        
        self.logger.debug(f"Created {len(self._feature_names)} minimal features")
        return df
    
    def _create_full_features(self, df: pd.DataFrame, include_target_lags: bool) -> pd.DataFrame:
        """
        Create comprehensive feature set (original full implementation).
        """
        # Create feature groups
        df = self._create_date_features(df)
        df = self._create_cyclical_features(df)
        
        if include_target_lags:
            df = self._create_lag_features(df)
        
        df = self._create_rolling_features(df)
        df = self._create_trend_features(df)
        df = self._create_interaction_features(df)
        
        return df
    
    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create calendar/date-based features.
        """
        date_col = df[self.config.date_column]
        
        # Basic date components
        df['year'] = date_col.dt.year
        df['month'] = date_col.dt.month
        df['day'] = date_col.dt.day
        df['quarter'] = date_col.dt.quarter
        
        # Day-based features
        df['day_of_week_num'] = date_col.dt.dayofweek
        df['day_of_year'] = date_col.dt.dayofyear
        df['week_of_year'] = date_col.dt.isocalendar().week.astype(int)
        
        # Binary calendar features
        df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)
        df['is_month_start'] = date_col.dt.is_month_start.astype(int)
        df['is_month_end'] = date_col.dt.is_month_end.astype(int)
        df['is_quarter_start'] = date_col.dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = date_col.dt.is_quarter_end.astype(int)
        df['is_year_start'] = date_col.dt.is_year_start.astype(int)
        df['is_year_end'] = date_col.dt.is_year_end.astype(int)
        
        # Days in month
        df['days_in_month'] = date_col.dt.daysinmonth
        
        date_features = [
            'year', 'month', 'day', 'quarter', 'day_of_week_num',
            'day_of_year', 'week_of_year', 'is_weekend', 'is_month_start',
            'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'is_year_start', 'is_year_end', 'days_in_month'
        ]
        self._feature_names.extend(date_features)
        
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical encodings for periodic features.
        """
        # Day of week cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
        
        # Month cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of year cyclical encoding
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Week of year cyclical encoding
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        cyclical_features = [
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'week_of_year_sin', 'week_of_year_cos'
        ]
        self._feature_names.extend(cyclical_features)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for target and other numeric columns.
        """
        target_col = self.config.target_column
        lag_features = []
        
        for lag in self.config.lag_days:
            # Target variable lags
            feature_name = f'{target_col}_lag_{lag}'
            df[feature_name] = df[target_col].shift(lag)
            lag_features.append(feature_name)
            
            # Marketing spend lags
            if 'marketing_spend' in df.columns:
                feature_name = f'marketing_spend_lag_{lag}'
                df[feature_name] = df['marketing_spend'].shift(lag)
                lag_features.append(feature_name)
        
        # Difference features
        df[f'{target_col}_diff_1'] = df[target_col].diff(1)
        df[f'{target_col}_diff_7'] = df[target_col].diff(7)
        lag_features.extend([f'{target_col}_diff_1', f'{target_col}_diff_7'])
        
        # Percent change features
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_7'] = df[target_col].pct_change(7)
        lag_features.extend([f'{target_col}_pct_change_1', f'{target_col}_pct_change_7'])
        
        self._feature_names.extend(lag_features)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window statistics.
        """
        target_col = self.config.target_column
        rolling_features = []
        
        for window in self.config.rolling_windows:
            # Rolling mean
            feature_name = f'{target_col}_rolling_mean_{window}'
            df[feature_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
            rolling_features.append(feature_name)
            
            # Rolling std
            feature_name = f'{target_col}_rolling_std_{window}'
            df[feature_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
            rolling_features.append(feature_name)
            
            # Rolling min
            feature_name = f'{target_col}_rolling_min_{window}'
            df[feature_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).min()
            rolling_features.append(feature_name)
            
            # Rolling max
            feature_name = f'{target_col}_rolling_max_{window}'
            df[feature_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).max()
            rolling_features.append(feature_name)
            
            # Rolling median
            feature_name = f'{target_col}_rolling_median_{window}'
            df[feature_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).median()
            rolling_features.append(feature_name)
            
            # Rolling sum
            feature_name = f'{target_col}_rolling_sum_{window}'
            df[feature_name] = df[target_col].shift(1).rolling(window=window, min_periods=1).sum()
            rolling_features.append(feature_name)
            
            # Marketing spend rolling mean
            if 'marketing_spend' in df.columns:
                feature_name = f'marketing_spend_rolling_mean_{window}'
                df[feature_name] = df['marketing_spend'].shift(1).rolling(window=window, min_periods=1).mean()
                rolling_features.append(feature_name)
        
        # Exponential weighted moving average
        for span in [7, 14, 28]:
            feature_name = f'{target_col}_ewm_{span}'
            df[feature_name] = df[target_col].shift(1).ewm(span=span, min_periods=1).mean()
            rolling_features.append(feature_name)
        
        self._feature_names.extend(rolling_features)
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-related features.
        """
        date_col = df[self.config.date_column]
        target_col = self.config.target_column
        trend_features = []
        
        # Days since start
        df['days_since_start'] = (date_col - date_col.min()).dt.days
        trend_features.append('days_since_start')
        
        # Normalized time (0 to 1)
        total_days = df['days_since_start'].max()
        if total_days > 0:
            df['time_normalized'] = df['days_since_start'] / total_days
        else:
            df['time_normalized'] = 0
        trend_features.append('time_normalized')
        
        # Same day last week ratio
        df['sales_vs_last_week'] = df[target_col] / df[target_col].shift(7)
        df['sales_vs_last_week'] = df['sales_vs_last_week'].replace([np.inf, -np.inf], np.nan)
        trend_features.append('sales_vs_last_week')
        
        # Same day last month ratio (approximately)
        df['sales_vs_last_month'] = df[target_col] / df[target_col].shift(28)
        df['sales_vs_last_month'] = df['sales_vs_last_month'].replace([np.inf, -np.inf], np.nan)
        trend_features.append('sales_vs_last_month')
        
        self._feature_names.extend(trend_features)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        """
        target_col = self.config.target_column
        interaction_features = []
        
        # Holiday x Weekend interaction
        if 'is_holiday' in df.columns and 'is_weekend' in df.columns:
            df['holiday_weekend'] = df['is_holiday'] * df['is_weekend']
            interaction_features.append('holiday_weekend')
        
        # Marketing efficiency (sales per marketing dollar)
        if 'marketing_spend' in df.columns:
            df['marketing_efficiency'] = df[target_col] / df['marketing_spend'].replace(0, np.nan)
            df['marketing_efficiency'] = df['marketing_efficiency'].replace([np.inf, -np.inf], np.nan)
            interaction_features.append('marketing_efficiency')
            
            # Lagged marketing efficiency
            df['marketing_efficiency_lag_7'] = df['marketing_efficiency'].shift(7)
            interaction_features.append('marketing_efficiency_lag_7')
        
        # Sales deviation from rolling mean
        if f'{target_col}_rolling_mean_7' in df.columns:
            df['sales_deviation_7'] = df[target_col] - df[f'{target_col}_rolling_mean_7']
            interaction_features.append('sales_deviation_7')
        
        if f'{target_col}_rolling_mean_28' in df.columns:
            df['sales_deviation_28'] = df[target_col] - df[f'{target_col}_rolling_mean_28']
            interaction_features.append('sales_deviation_28')
        
        self._feature_names.extend(interaction_features)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all created feature names."""
        return self._feature_names.copy()
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of engineered features.
        """
        summary = {
            'total_features': len(self._feature_names),
            'feature_names': self._feature_names,
            'missing_values': {f: int(df[f].isnull().sum()) for f in self._feature_names if f in df.columns}
        }
        return summary
    
    def drop_na_rows(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
        """
        Drop rows with NaN values created by lag/rolling features.
        
        Args:
            df: DataFrame with features
            subset: Columns to check for NaN (default: all numeric columns)
            
        Returns:
            Tuple of (cleaned DataFrame, number of rows dropped)
        """
        initial_rows = len(df)
        
        if subset is None:
            # Use all numeric columns
            subset = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_clean = df.dropna(subset=subset)
        dropped = initial_rows - len(df_clean)
        
        if dropped > 0:
            self.logger.info(f"Dropped {dropped} rows with NaN values")
        
        return df_clean, dropped
