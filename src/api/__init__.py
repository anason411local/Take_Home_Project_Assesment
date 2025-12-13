"""
FastAPI Backend for Sales Forecasting System.

This module provides:
- REST API endpoints for forecasting, historical data, metrics, etc.
- WebSocket endpoints for real-time forecast streaming
- Session management for unique user sessions

REST Endpoints:
- POST /api/forecast - Generate sales forecasts
- GET /api/historical - Retrieve historical data
- GET /api/metrics - Get model metrics
- GET /api/feature-importance - Feature importance
- POST /api/upload - Upload new data

Session Endpoints:
- POST /api/session - Create new session
- GET /api/session/{id} - Get session info
- PUT /api/session/{id} - Update session preferences
- DELETE /api/session/{id} - Delete session

WebSocket Endpoints:
- WS /ws/{session_id} - Real-time forecast streaming
"""

from .main import app
from .schemas import (
    ForecastRequest,
    ForecastResponse,
    PredictionItem,
    HistoricalDataResponse,
    HistoricalDataItem,
    MetricsResponse,
    ModelMetrics,
    FeatureImportanceResponse,
    FeatureItem,
    UploadResponse,
    ErrorResponse,
    SessionCreateResponse,
    SessionInfo,
    SessionResponse,
    SessionUpdateRequest,
    WebSocketForecastRequest,
    WebSocketProgressMessage
)
from .session import SessionManager, SessionData, get_session_manager
from .websocket import (
    ConnectionManager,
    WebSocketMessage,
    MessageType,
    get_connection_manager
)

__all__ = [
    # App
    'app',
    
    # REST Schemas
    'ForecastRequest',
    'ForecastResponse',
    'PredictionItem',
    'HistoricalDataResponse',
    'HistoricalDataItem',
    'MetricsResponse',
    'ModelMetrics',
    'FeatureImportanceResponse',
    'FeatureItem',
    'UploadResponse',
    'ErrorResponse',
    
    # Session Schemas
    'SessionCreateResponse',
    'SessionInfo',
    'SessionResponse',
    'SessionUpdateRequest',
    
    # WebSocket Schemas
    'WebSocketForecastRequest',
    'WebSocketProgressMessage',
    
    # Session Management
    'SessionManager',
    'SessionData',
    'get_session_manager',
    
    # WebSocket Management
    'ConnectionManager',
    'WebSocketMessage',
    'MessageType',
    'get_connection_manager'
]
