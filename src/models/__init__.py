"""
Models module for Sales Forecasting.

Contains forecasting models:
- LinearTrendModel: Simple linear trend with optional day-of-week effects
- XGBoostModel: Gradient boosting for time series
- ProphetModel: Facebook Prophet for business time series
- SARIMAModel: Statistical SARIMA model
"""
from .models import (
    BaseSimpleModel,
    LinearTrendModel,
    XGBoostModel,
    ProphetModel,
    SARIMAModel,
    MODEL_REGISTRY
)

__all__ = [
    'BaseSimpleModel',
    'LinearTrendModel',
    'XGBoostModel',
    'ProphetModel',
    'SARIMAModel',
    'MODEL_REGISTRY'
]
