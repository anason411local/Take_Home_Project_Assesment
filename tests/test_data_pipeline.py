"""
Tests for the data processing pipeline.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.loader import DataLoader
from src.data.validator import DataValidator, ValidationResult
from src.data.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer
from src.utils.config import Config


@pytest.fixture
def sample_data():
    """Create sample sales data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'daily_sales': np.random.randint(10000, 50000, 100),
        'product_category': 'Electronics',
        'marketing_spend': np.random.randint(2000, 10000, 100),
        'day_of_week': dates.day_name(),
        'is_holiday': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    })
    return df


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        lag_days=[1, 7],
        rolling_windows=[7]
    )


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_validate_valid_data(self, sample_data, config):
        """Test validation passes for valid data."""
        validator = DataValidator(config)
        result = validator.validate(sample_data)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_missing_columns(self, sample_data, config):
        """Test validation fails for missing required columns."""
        df = sample_data.drop(columns=['daily_sales'])
        validator = DataValidator(config)
        result = validator.validate(df)
        
        assert not result.is_valid
        assert any('Missing required columns' in err for err in result.errors)
    
    def test_validate_missing_values(self, sample_data, config):
        """Test validation detects missing values."""
        df = sample_data.copy()
        df.loc[0, 'daily_sales'] = None
        
        validator = DataValidator(config)
        result = validator.validate(df)
        
        assert not result.is_valid
    
    def test_validate_negative_sales(self, sample_data, config):
        """Test validation detects negative sales values."""
        df = sample_data.copy()
        df.loc[0, 'daily_sales'] = -100
        
        validator = DataValidator(config)
        result = validator.validate(df)
        
        assert not result.is_valid


class TestDataCleaner:
    """Tests for DataCleaner class."""
    
    def test_clean_data(self, sample_data, config):
        """Test basic cleaning operations."""
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean(sample_data)
        
        assert len(df_clean) == len(sample_data)
        assert df_clean['date'].dtype == 'datetime64[ns]'
    
    def test_remove_duplicates(self, sample_data, config):
        """Test duplicate removal."""
        df = pd.concat([sample_data, sample_data.iloc[[0]]])
        
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean(df)
        
        assert len(df_clean) == len(sample_data)
    
    def test_standardize_column_names(self, config):
        """Test column name standardization."""
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Daily Sales': range(10),
            'Product Category': 'Test',
            'Marketing Spend': range(10),
            'Day Of Week': 'Monday',
            'Is Holiday': 0
        })
        
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean(df)
        
        assert 'date' in df_clean.columns
        assert 'daily_sales' in df_clean.columns


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_create_features(self, sample_data, config):
        """Test feature creation."""
        # Clean data first
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean(sample_data)
        
        engineer = FeatureEngineer(config)
        df_features = engineer.create_features(df_clean)
        
        # Check that features were created
        assert len(df_features.columns) > len(sample_data.columns)
        
        # Check for specific features
        assert 'day_of_week_num' in df_features.columns
        assert 'is_weekend' in df_features.columns
        assert 'daily_sales_lag_1' in df_features.columns
        assert 'daily_sales_rolling_mean_7' in df_features.columns
    
    def test_date_features(self, sample_data, config):
        """Test date feature creation."""
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean(sample_data)
        
        engineer = FeatureEngineer(config)
        df_features = engineer.create_features(df_clean, mode='minimal')
        
        # Check minimal date features
        assert 'month' in df_features.columns
        assert 'day_of_week_num' in df_features.columns
        assert 'quarter' in df_features.columns
        assert 'is_weekend' in df_features.columns
    
    def test_cyclical_features(self, sample_data, config):
        """Test cyclical feature creation."""
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean(sample_data)
        
        engineer = FeatureEngineer(config)
        df_features = engineer.create_features(df_clean)
        
        # Check cyclical features
        assert 'day_of_week_sin' in df_features.columns
        assert 'day_of_week_cos' in df_features.columns
        
        # Verify sin/cos values are in valid range
        assert df_features['day_of_week_sin'].between(-1, 1).all()
        assert df_features['day_of_week_cos'].between(-1, 1).all()
    
    def test_lag_features(self, sample_data, config):
        """Test lag feature creation."""
        cleaner = DataCleaner(config)
        df_clean = cleaner.clean(sample_data)
        
        engineer = FeatureEngineer(config)
        df_features = engineer.create_features(df_clean)
        
        # Check lag features
        assert 'daily_sales_lag_1' in df_features.columns
        assert 'daily_sales_lag_7' in df_features.columns
        
        # Verify lag values
        assert df_features['daily_sales_lag_1'].iloc[1] == df_features['daily_sales'].iloc[0]


class TestConfig:
    """Tests for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.date_column == 'date'
        assert config.target_column == 'daily_sales'
        assert len(config.lag_days) > 0
        assert len(config.rolling_windows) > 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = Config(
            lag_days=[1, 3, 5],
            rolling_windows=[3, 5]
        )
        
        assert config.lag_days == [1, 3, 5]
        assert config.rolling_windows == [3, 5]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

