"""
Pydantic Schemas for API Request/Response Validation.

All API inputs and outputs are validated using these schemas.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum


# ==================== ENUMS ====================

class ModelName(str, Enum):
    """Available model names."""
    LINEAR_TREND = "linear_trend"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    PROPHET = "prophet"
    SARIMA = "sarima"
    ENSEMBLE = "ensemble"
    BEST = "best"


# ==================== FORECAST ====================

class ForecastRequest(BaseModel):
    """Request schema for forecast endpoint."""
    horizon: int = Field(
        ...,
        ge=1,
        le=365,
        description="Number of days to forecast (1-365)"
    )
    model: Optional[ModelName] = Field(
        default=ModelName.BEST,
        description="Model to use for forecasting"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "horizon": 30,
                "model": "best"
            }
        }


class PredictionItem(BaseModel):
    """Single prediction item."""
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)")
    value: float = Field(..., description="Predicted sales value")
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-07-01",
                "value": 89500.50
            }
        }


class ForecastResponse(BaseModel):
    """Response schema for forecast endpoint."""
    success: bool = Field(default=True)
    model_used: str = Field(..., description="Model used for forecasting")
    horizon: int = Field(..., description="Number of days forecasted")
    predictions: List[PredictionItem] = Field(..., description="List of predictions")
    summary: Dict[str, float] = Field(..., description="Summary statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_used": "sarima",
                "horizon": 7,
                "predictions": [
                    {"date": "2024-07-01", "value": 89500.50},
                    {"date": "2024-07-02", "value": 90200.75}
                ],
                "summary": {
                    "total": 628000.0,
                    "mean": 89714.29,
                    "min": 85000.0,
                    "max": 94000.0
                }
            }
        }


# ==================== HISTORICAL DATA ====================

class HistoricalDataItem(BaseModel):
    """Single historical data item."""
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    daily_sales: float = Field(..., description="Daily sales value")
    marketing_spend: Optional[float] = Field(None, description="Marketing spend")
    is_holiday: Optional[int] = Field(None, description="Holiday flag (0 or 1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-01-15",
                "daily_sales": 75000.50,
                "marketing_spend": 5000.0,
                "is_holiday": 0
            }
        }


class HistoricalDataResponse(BaseModel):
    """Response schema for historical data endpoint."""
    success: bool = Field(default=True)
    total_records: int = Field(..., description="Total number of records")
    date_range: Dict[str, str] = Field(..., description="Date range of data")
    data: List[HistoricalDataItem] = Field(..., description="Historical data records")
    summary: Dict[str, float] = Field(..., description="Summary statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_records": 669,
                "date_range": {
                    "start": "2023-01-01",
                    "end": "2024-10-30"
                },
                "data": [
                    {"date": "2023-01-01", "daily_sales": 45000.0, "marketing_spend": 3000.0, "is_holiday": 0}
                ],
                "summary": {
                    "mean": 62424.0,
                    "min": 15234.0,
                    "max": 112789.0,
                    "std": 25727.0
                }
            }
        }


# ==================== METRICS ====================

class ModelMetrics(BaseModel):
    """Metrics for a single model."""
    model_name: str = Field(..., description="Model name")
    test_mape: float = Field(..., description="Test MAPE (%)")
    test_mae: float = Field(..., description="Test MAE ($)")
    test_rmse: float = Field(..., description="Test RMSE ($)")
    train_mape: Optional[float] = Field(None, description="Train MAPE (%)")
    train_mae: Optional[float] = Field(None, description="Train MAE ($)")
    train_rmse: Optional[float] = Field(None, description="Train RMSE ($)")
    training_time: Optional[float] = Field(None, description="Training time (seconds)")
    best_params: Optional[Dict[str, Any]] = Field(None, description="Best hyperparameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "sarima",
                "test_mape": 1.90,
                "test_mae": 1910.0,
                "test_rmse": 2610.0,
                "train_mape": 158.26,
                "training_time": 2.1
            }
        }


class MetricsResponse(BaseModel):
    """Response schema for metrics endpoint."""
    success: bool = Field(default=True)
    best_model: str = Field(..., description="Best performing model")
    target_mape: float = Field(default=20.0, description="Target MAPE threshold")
    target_met: bool = Field(..., description="Whether target MAPE is met")
    models: List[ModelMetrics] = Field(..., description="Metrics for all models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "best_model": "sarima",
                "target_mape": 20.0,
                "target_met": True,
                "models": [
                    {"model_name": "sarima", "test_mape": 1.90, "test_mae": 1910.0, "test_rmse": 2610.0}
                ]
            }
        }


# ==================== FEATURE IMPORTANCE ====================

class FeatureItem(BaseModel):
    """Single feature importance item."""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")
    rank: int = Field(..., description="Importance rank")
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature": "lag_1",
                "importance": 0.45,
                "rank": 1
            }
        }


class FeatureImportanceResponse(BaseModel):
    """Response schema for feature importance endpoint."""
    success: bool = Field(default=True)
    model_name: str = Field(..., description="Model name")
    features: List[FeatureItem] = Field(..., description="Feature importance list")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_name": "xgboost",
                "features": [
                    {"feature": "lag_1", "importance": 0.45, "rank": 1},
                    {"feature": "rolling_mean_7", "importance": 0.25, "rank": 2}
                ]
            }
        }


# ==================== UPLOAD ====================

class UploadResponse(BaseModel):
    """Response schema for upload endpoint."""
    success: bool = Field(default=True)
    message: str = Field(..., description="Status message")
    records_imported: int = Field(..., description="Number of records imported")
    date_range: Dict[str, str] = Field(..., description="Date range of imported data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Data uploaded successfully",
                "records_imported": 669,
                "date_range": {
                    "start": "2023-01-01",
                    "end": "2024-10-30"
                }
            }
        }


# ==================== ERROR ====================

class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Invalid horizon value",
                "detail": "Horizon must be between 1 and 365"
            }
        }


# ==================== SESSION ====================

class SessionCreateResponse(BaseModel):
    """Response schema for session creation."""
    success: bool = Field(default=True)
    session_id: str = Field(..., description="Unique session ID")
    message: str = Field(..., description="Status message")
    websocket_url: str = Field(..., description="WebSocket URL for this session")
    expires_in: int = Field(..., description="Session expiration time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "message": "Session created successfully",
                "websocket_url": "/ws/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "expires_in": 3600
            }
        }


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Creation timestamp")
    last_accessed: str = Field(..., description="Last access timestamp")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    preferred_model: str = Field(..., description="Preferred model")
    default_horizon: int = Field(..., description="Default forecast horizon")
    is_active: bool = Field(..., description="Session active status")
    forecast_count: int = Field(..., description="Number of forecasts made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "created_at": "2024-12-13T10:30:00",
                "last_accessed": "2024-12-13T10:45:00",
                "user_agent": "Mozilla/5.0",
                "ip_address": "127.0.0.1",
                "preferred_model": "best",
                "default_horizon": 30,
                "is_active": True,
                "forecast_count": 5
            }
        }


class SessionResponse(BaseModel):
    """Response schema for session info endpoint."""
    success: bool = Field(default=True)
    session: SessionInfo = Field(..., description="Session information")
    websocket_connected: bool = Field(..., description="WebSocket connection status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "session": {
                    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "created_at": "2024-12-13T10:30:00",
                    "preferred_model": "best",
                    "default_horizon": 30,
                    "is_active": True,
                    "forecast_count": 5
                },
                "websocket_connected": True
            }
        }


class SessionUpdateRequest(BaseModel):
    """Request schema for session update."""
    preferred_model: Optional[str] = Field(None, description="Preferred model")
    default_horizon: Optional[int] = Field(
        None, ge=1, le=365, description="Default forecast horizon (1-365)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "preferred_model": "sarima",
                "default_horizon": 30
            }
        }


# ==================== WEBSOCKET ====================

class WebSocketForecastRequest(BaseModel):
    """WebSocket forecast request schema."""
    action: str = Field(default="forecast", description="Action type")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Request payload")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "forecast",
                "payload": {
                    "horizon": 30,
                    "model": "best"
                }
            }
        }


class WebSocketProgressMessage(BaseModel):
    """WebSocket progress message schema."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: str = Field(..., description="Timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "forecast_progress",
                "data": {
                    "progress": 50,
                    "current_step": "Generating predictions...",
                    "total_steps": 100,
                    "details": {"horizon": 30}
                },
                "timestamp": "2024-12-13T10:45:00",
                "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            }
        }

