"""
Configuration settings for the sales forecasting system.
"""
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class Config:
    """Configuration class for data processing pipeline."""
    
    # Data paths
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    
    # Column names
    date_column: str = "date"
    target_column: str = "daily_sales"
    
    # Expected columns in raw data
    expected_columns: List[str] = field(default_factory=lambda: [
        "date", "daily_sales", "product_category", 
        "marketing_spend", "day_of_week", "is_holiday"
    ])
    
    # Feature engineering parameters
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28])
    
    # Data validation thresholds
    min_sales: float = 0.0
    max_sales: float = 500000.0  # Reasonable upper bound
    min_marketing_spend: float = 0.0
    max_marketing_spend: float = 100000.0
    
    # Date format
    date_format: str = "%Y-%m-%d"
    
    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        if isinstance(self.raw_data_path, str):
            self.raw_data_path = Path(self.raw_data_path)
        if isinstance(self.processed_data_path, str):
            self.processed_data_path = Path(self.processed_data_path)

