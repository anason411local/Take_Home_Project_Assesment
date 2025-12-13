"""
Training module for Sales Forecasting.

Contains:
- ModelTrainer: Main trainer with Optuna + MLflow integration
- TimeSeriesSplitter: Data splitting utilities
- RegressionMetrics: Evaluation metrics
"""
from .trainer import ModelTrainer
from .data_splitter import TimeSeriesSplitter
from .metrics import RegressionMetrics

__all__ = [
    'ModelTrainer',
    'TimeSeriesSplitter',
    'RegressionMetrics'
]
