"""
FastAPI Main Application for Sales Forecasting System.

Provides REST API endpoints for:
- POST /api/forecast - Generate sales forecasts
- GET /api/historical - Get historical sales data
- GET /api/metrics - Get model performance metrics
- GET /api/feature-importance - Get feature importance (tree-based models)
- POST /api/upload - Upload new CSV data

WebSocket endpoints for:
- WS /ws/{session_id} - Real-time forecast streaming

Session management for:
- POST /api/session - Create new session
- GET /api/session/{session_id} - Get session info
- DELETE /api/session/{session_id} - Delete session

Run with: uvicorn src.api.main:app --reload --port 8000
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import tempfile
import shutil
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status, WebSocket, WebSocketDisconnect, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

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
    ModelName
)
from .session import SessionManager, SessionData, get_session_manager
from .websocket import (
    ConnectionManager,
    WebSocketMessage,
    MessageType,
    get_connection_manager
)
from ..data.database import get_database
from ..forecasting.forecaster import Forecaster

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== APP CONFIGURATION ====================

app = FastAPI(
    title="Sales Forecasting API",
    description="""
    REST API for Sales Forecasting System with WebSocket support.
    
    ## Features
    - **Forecast**: Generate multi-day sales predictions using trained ML models
    - **Historical Data**: Access historical sales data
    - **Metrics**: View model performance metrics (MAPE, MAE, RMSE)
    - **Feature Importance**: Get feature importance for tree-based models
    - **Upload**: Upload new sales data for training
    - **Sessions**: Unique session management for each user
    - **WebSocket**: Real-time forecast streaming
    
    ## Models Available
    - Linear Trend
    - XGBoost (with learning curves)
    - Random Forest (with learning curves)
    - Prophet
    - SARIMA
    - Ensemble (weighted average)
    
    ## WebSocket
    Connect to `/ws/{session_id}` for real-time updates.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files - Mount frontend directory
frontend_dir = project_root / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ==================== GLOBAL INSTANCES ====================

# Initialize forecaster (lazy loading)
_forecaster: Optional[Forecaster] = None

def get_forecaster() -> Forecaster:
    """Get or initialize the forecaster with loaded models."""
    global _forecaster
    if _forecaster is None:
        _forecaster = Forecaster()
        models_dir = project_root / "models" / "saved"
        if models_dir.exists():
            _forecaster.load_all_models(str(models_dir))
    
    # Always update the last_data_date from current database
    # This ensures forecasts start from the actual last data point
    try:
        db = get_database()
        df = db.load_sales_data()
        if not df.empty:
            last_date = pd.to_datetime(df['date']).max()
            _forecaster.set_last_data_date(last_date)
    except Exception as e:
        logger.warning(f"Could not update last_data_date: {e}")
    
    return _forecaster


# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": "HTTPException",
            "message": exc.detail,
            "detail": None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "detail": None
        }
    )


# ==================== SESSION ENDPOINTS ====================

@app.post(
    "/api/session",
    tags=["Session"],
    summary="Create new session"
)
async def create_session(
    request: Request,
    user_agent: Optional[str] = Header(None)
):
    """
    Create a new session with unique ID.
    
    Returns session ID that should be used for WebSocket connections
    and to track user preferences.
    """
    try:
        session_manager = get_session_manager()
        
        # Get client IP
        client_ip = request.client.host if request.client else None
        
        # Create session
        session = session_manager.create_session(
            user_agent=user_agent,
            ip_address=client_ip
        )
        
        return {
            "success": True,
            "session_id": session.session_id,
            "message": "Session created successfully",
            "websocket_url": f"/ws/{session.session_id}",
            "expires_in": 3600  # 1 hour
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@app.get(
    "/api/session/{session_id}",
    tags=["Session"],
    summary="Get session info"
)
async def get_session(session_id: str):
    """
    Get information about a session.
    
    - **session_id**: Session ID to retrieve
    """
    try:
        session_manager = get_session_manager()
        session = session_manager.get_session(session_id)
        
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found or expired: {session_id}"
            )
        
        connection_manager = get_connection_manager()
        
        return {
            "success": True,
            "session": session.to_dict(),
            "websocket_connected": connection_manager.is_connected(session_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}"
        )


@app.put(
    "/api/session/{session_id}",
    tags=["Session"],
    summary="Update session preferences"
)
async def update_session(
    session_id: str,
    preferred_model: Optional[str] = Query(None, description="Preferred model"),
    default_horizon: Optional[int] = Query(None, ge=1, le=365, description="Default forecast horizon")
):
    """
    Update session preferences.
    
    - **session_id**: Session ID
    - **preferred_model**: Preferred forecasting model
    - **default_horizon**: Default forecast horizon (1-365)
    """
    try:
        session_manager = get_session_manager()
        session = session_manager.update_session(
            session_id=session_id,
            preferred_model=preferred_model,
            default_horizon=default_horizon
        )
        
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found or expired: {session_id}"
            )
        
        return {
            "success": True,
            "message": "Session updated successfully",
            "session": session.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session: {str(e)}"
        )


@app.delete(
    "/api/session/{session_id}",
    tags=["Session"],
    summary="Delete session"
)
async def delete_session(session_id: str):
    """
    Delete a session and disconnect any WebSocket connections.
    
    - **session_id**: Session ID to delete
    """
    try:
        session_manager = get_session_manager()
        connection_manager = get_connection_manager()
        
        # Disconnect WebSocket if connected
        if connection_manager.is_connected(session_id):
            await connection_manager.disconnect(session_id)
        
        # Delete session
        deleted = session_manager.delete_session(session_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        
        return {
            "success": True,
            "message": "Session deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )


@app.get(
    "/api/sessions",
    tags=["Session"],
    summary="List all active sessions (admin)"
)
async def list_sessions():
    """List all active sessions (for admin/monitoring purposes)."""
    try:
        session_manager = get_session_manager()
        connection_manager = get_connection_manager()
        
        sessions = session_manager.get_all_sessions()
        
        return {
            "success": True,
            "active_sessions": len(sessions),
            "websocket_connections": connection_manager.get_connection_count(),
            "sessions": sessions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


# ==================== WEBSOCKET ENDPOINTS ====================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time communication.
    
    Connect with a valid session_id to receive:
    - Real-time forecast progress updates
    - Forecast completion notifications
    - Error notifications
    
    Send messages to:
    - Request forecasts with progress streaming
    - Send heartbeat to keep connection alive
    """
    session_manager = get_session_manager()
    connection_manager = get_connection_manager()
    
    # Validate session
    session = session_manager.get_session(session_id)
    if session is None:
        await websocket.close(code=4001, reason="Invalid or expired session")
        return
    
    # Connect
    connected = await connection_manager.connect(websocket, session_id)
    if not connected:
        return
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                action = message.get("action", "")
                payload = message.get("payload", {})
                
                if action == "heartbeat":
                    # Respond to heartbeat
                    await connection_manager.heartbeat(session_id)
                
                elif action == "forecast":
                    # Handle forecast request with streaming
                    await handle_websocket_forecast(
                        session_id=session_id,
                        horizon=payload.get("horizon", session.default_horizon),
                        model=payload.get("model", session.preferred_model),
                        connection_manager=connection_manager,
                        session_manager=session_manager
                    )
                
                elif action == "ping":
                    # Simple ping/pong
                    await connection_manager.send_to_session(
                        session_id,
                        WebSocketMessage(
                            type=MessageType.NOTIFICATION,
                            data={"message": "pong"},
                            session_id=session_id
                        )
                    )
                
                else:
                    # Unknown action
                    await connection_manager.send_error(
                        session_id,
                        "UnknownAction",
                        f"Unknown action: {action}"
                    )
                    
            except json.JSONDecodeError:
                await connection_manager.send_error(
                    session_id,
                    "InvalidJSON",
                    "Message must be valid JSON"
                )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: session={session_id}, error={e}")
    finally:
        await connection_manager.disconnect(session_id)


async def handle_websocket_forecast(
    session_id: str,
    horizon: int,
    model: str,
    connection_manager: ConnectionManager,
    session_manager: SessionManager
):
    """
    Handle forecast request via WebSocket with real-time progress streaming.
    
    Args:
        session_id: Session ID
        horizon: Forecast horizon
        model: Model to use
        connection_manager: Connection manager
        session_manager: Session manager
    """
    try:
        # Send start notification
        await connection_manager.send_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.FORECAST_START,
                data={
                    "horizon": horizon,
                    "model": model,
                    "message": "Starting forecast generation..."
                },
                session_id=session_id
            )
        )
        
        # Progress: Loading model
        await connection_manager.send_forecast_progress(
            session_id, 10, "Loading model...", details={"model": model}
        )
        await asyncio.sleep(0.1)  # Small delay for UI update
        
        forecaster = get_forecaster()
        db = get_database()
        
        # Determine model
        model_name = model
        if model_name == "best":
            best_model = db.get_best_model()
            if best_model:
                model_name = best_model['model_name']
            else:
                model_name = "sarima"  # Default
        
        # Progress: Model loaded
        await connection_manager.send_forecast_progress(
            session_id, 30, "Model loaded", details={"model_name": model_name}
        )
        await asyncio.sleep(0.1)
        
        # Check if model is available
        if model_name == "ensemble":
            if not forecaster.models:
                raise ValueError("No models loaded for ensemble")
        elif model_name not in forecaster.models:
            model_path = project_root / "models" / "saved" / f"{model_name}.pkl"
            if model_path.exists():
                forecaster.load_model(model_name, str(model_path))
            else:
                raise ValueError(f"Model not found: {model_name}")
        
        # Progress: Generating forecast
        await connection_manager.send_forecast_progress(
            session_id, 50, "Generating predictions...", details={"horizon": horizon}
        )
        await asyncio.sleep(0.1)
        
        # Generate forecast
        if model_name == "ensemble":
            forecast_df = forecaster.get_ensemble_forecast(horizon)
        else:
            forecast_df = forecaster.forecast(model_name, horizon)
        
        # Progress: Processing results
        await connection_manager.send_forecast_progress(
            session_id, 80, "Processing results..."
        )
        await asyncio.sleep(0.1)
        
        # Format predictions
        predictions = [
            {
                "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                "value": round(row['predicted_sales'], 2)
            }
            for _, row in forecast_df.iterrows()
        ]
        
        # Calculate summary
        values = forecast_df['predicted_sales'].values
        summary = {
            "total": round(float(values.sum()), 2),
            "mean": round(float(values.mean()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2)
        }
        
        # Progress: Complete
        await connection_manager.send_forecast_progress(
            session_id, 100, "Complete!"
        )
        
        # Add to session history
        session_manager.add_forecast_to_history(
            session_id,
            {
                "horizon": horizon,
                "model": model_name,
                "summary": summary
            }
        )
        
        # Send completion
        await connection_manager.send_forecast_complete(
            session_id,
            predictions=predictions,
            model_used=model_name,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Forecast error: session={session_id}, error={e}")
        await connection_manager.send_to_session(
            session_id,
            WebSocketMessage(
                type=MessageType.FORECAST_ERROR,
                data={
                    "error": type(e).__name__,
                    "message": str(e)
                },
                session_id=session_id
            )
        )


# ==================== REST ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "healthy",
        "service": "Sales Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "forecast": "POST /api/forecast",
            "historical": "GET /api/historical",
            "metrics": "GET /api/metrics",
            "feature_importance": "GET /api/feature-importance",
            "upload": "POST /api/upload"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    db = get_database()
    summary = db.get_data_summary()
    
    return {
        "status": "healthy",
        "database": "connected",
        "records": summary.get('total_records', 0),
        "timestamp": datetime.now().isoformat()
    }


# ==================== FORECAST ENDPOINT ====================

@app.post(
    "/api/forecast",
    response_model=ForecastResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Forecasting"],
    summary="Generate sales forecast"
)
async def forecast(request: ForecastRequest):
    """
    Generate sales forecast for the specified number of days.
    
    - **horizon**: Number of days to forecast (1-365)
    - **model**: Model to use (default: best performing model)
    
    Returns predictions with date and value for each day.
    """
    try:
        forecaster = get_forecaster()
        db = get_database()
        
        # Determine which model to use
        model_name = request.model.value
        
        if model_name == "best":
            best_model = db.get_best_model()
            if not best_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No trained models found. Please train models first."
                )
            model_name = best_model['model_name']
        
        # Check if model is loaded
        if model_name == "ensemble":
            if not forecaster.models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No models loaded for ensemble forecasting."
                )
            # Generate ensemble forecast
            forecast_df = forecaster.get_ensemble_forecast(request.horizon)
        else:
            if model_name not in forecaster.models:
                # Try to load the model
                model_path = project_root / "models" / "saved" / f"{model_name}.pkl"
                if model_path.exists():
                    forecaster.load_model(model_name, str(model_path))
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model '{model_name}' not found. Available models: {list(forecaster.models.keys())}"
                    )
            
            # Generate forecast
            forecast_df = forecaster.forecast(model_name, request.horizon)
        
        # Format predictions
        predictions = [
            PredictionItem(
                date=row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                value=round(row['predicted_sales'], 2)
            )
            for _, row in forecast_df.iterrows()
        ]
        
        # Calculate summary
        values = forecast_df['predicted_sales'].values
        summary = {
            "total": round(float(values.sum()), 2),
            "mean": round(float(values.mean()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2)
        }
        
        return ForecastResponse(
            success=True,
            model_used=model_name,
            horizon=request.horizon,
            predictions=predictions,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast generation failed: {str(e)}"
        )


# ==================== HISTORICAL DATA ENDPOINT ====================

@app.get(
    "/api/historical",
    response_model=HistoricalDataResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No data found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Data"],
    summary="Get historical sales data"
)
async def get_historical_data(
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Limit number of records"),
    offset: Optional[int] = Query(None, ge=0, description="Offset for pagination"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)")
):
    """
    Get historical sales data from the database.
    
    - **limit**: Maximum number of records to return
    - **offset**: Number of records to skip (for pagination)
    - **start_date**: Filter data from this date
    - **end_date**: Filter data until this date
    """
    try:
        db = get_database()
        df = db.load_sales_data()
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No historical data found. Please upload data first."
            )
        
        # Ensure date column is datetime and sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Apply date filters
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        # Get total before pagination
        total_records = len(df)
        
        # Apply pagination
        if offset:
            df = df.iloc[offset:]
        if limit:
            df = df.head(limit)
        
        # Format data
        data = [
            HistoricalDataItem(
                date=row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                daily_sales=round(row['daily_sales'], 2),
                marketing_spend=round(row['marketing_spend'], 2) if pd.notna(row.get('marketing_spend')) else None,
                is_holiday=int(row['is_holiday']) if pd.notna(row.get('is_holiday')) else None
            )
            for _, row in df.iterrows()
        ]
        
        # Get date range
        summary_df = db.load_sales_data()
        date_range = {
            "start": summary_df['date'].min().strftime('%Y-%m-%d'),
            "end": summary_df['date'].max().strftime('%Y-%m-%d')
        }
        
        # Calculate summary
        summary = {
            "mean": round(float(summary_df['daily_sales'].mean()), 2),
            "min": round(float(summary_df['daily_sales'].min()), 2),
            "max": round(float(summary_df['daily_sales'].max()), 2),
            "std": round(float(summary_df['daily_sales'].std()), 2)
        }
        
        return HistoricalDataResponse(
            success=True,
            total_records=total_records,
            date_range=date_range,
            data=data,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve historical data: {str(e)}"
        )


# ==================== METRICS ENDPOINT ====================

@app.get(
    "/api/metrics",
    response_model=MetricsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No metrics found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Metrics"],
    summary="Get model performance metrics"
)
async def get_metrics(
    model: Optional[str] = Query(None, description="Filter by model name")
):
    """
    Get model performance metrics (MAPE, MAE, RMSE).
    
    - **model**: Optional filter for specific model
    
    Returns metrics for all trained models, sorted by test MAPE.
    """
    try:
        db = get_database()
        df = db.get_training_results(model_name=model)
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No training results found. Please train models first."
            )
        
        # Get best model
        best_model_info = db.get_best_model()
        best_model = best_model_info.get('model_name', 'unknown') if best_model_info else 'unknown'
        
        # Sort by test MAPE
        df = df.sort_values('test_mape', ascending=True)
        
        # Format metrics
        models = []
        for _, row in df.iterrows():
            import json
            best_params = None
            if row.get('best_params'):
                try:
                    best_params = json.loads(row['best_params'])
                except:
                    pass
            
            models.append(ModelMetrics(
                model_name=row['model_name'],
                test_mape=round(row['test_mape'], 2) if pd.notna(row.get('test_mape')) else 0,
                test_mae=round(row['test_mae'], 2) if pd.notna(row.get('test_mae')) else 0,
                test_rmse=round(row['test_rmse'], 2) if pd.notna(row.get('test_rmse')) else 0,
                train_mape=round(row['train_mape'], 2) if pd.notna(row.get('train_mape')) else None,
                train_mae=round(row['train_mae'], 2) if pd.notna(row.get('train_mae')) else None,
                train_rmse=round(row['train_rmse'], 2) if pd.notna(row.get('train_rmse')) else None,
                training_time=round(row['training_time'], 2) if pd.notna(row.get('training_time')) else None,
                best_params=best_params
            ))
        
        # Check if target met
        target_mape = 20.0
        best_mape = models[0].test_mape if models else 100.0
        target_met = best_mape <= target_mape
        
        return MetricsResponse(
            success=True,
            best_model=best_model,
            target_mape=target_mape,
            target_met=target_met,
            models=models
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


# ==================== FEATURE IMPORTANCE ENDPOINT ====================

@app.get(
    "/api/feature-importance",
    response_model=FeatureImportanceResponse,
    responses={
        404: {"model": ErrorResponse, "description": "No feature importance found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Metrics"],
    summary="Get feature importance"
)
async def get_feature_importance(
    model: Optional[str] = Query(
        "xgboost",
        description="Model name (xgboost or random_forest)"
    )
):
    """
    Get feature importance for tree-based models.
    
    - **model**: Model name (xgboost or random_forest)
    
    Returns feature importance scores sorted by importance.
    """
    try:
        db = get_database()
        df = db.get_feature_importance(model_name=model)
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No feature importance found for model '{model}'. Only tree-based models (xgboost, random_forest) have feature importance."
            )
        
        # Get latest run
        latest_run = df['run_id'].iloc[0]
        df = df[df['run_id'] == latest_run]
        
        # Format features
        features = [
            FeatureItem(
                feature=row['feature_name'],
                importance=round(row['importance_score'], 4),
                rank=int(row['importance_rank'])
            )
            for _, row in df.iterrows()
        ]
        
        return FeatureImportanceResponse(
            success=True,
            model_name=model,
            features=features
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve feature importance: {str(e)}"
        )


# ==================== UPLOAD ENDPOINT ====================

@app.post(
    "/api/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Data"],
    summary="Upload new CSV data"
)
async def upload_data(
    file: UploadFile = File(..., description="CSV file with sales data")
):
    """
    Upload new sales data from CSV file.
    
    The CSV file must contain at least:
    - **date**: Date column (YYYY-MM-DD format)
    - **daily_sales**: Daily sales values
    
    Optional columns:
    - **marketing_spend**: Marketing spend
    - **is_holiday**: Holiday flag (0 or 1)
    - **product_category**: Product category
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are accepted. Please upload a .csv file."
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            # Read and validate CSV
            df = pd.read_csv(tmp_path)
            
            # Check required columns
            required_columns = ['date', 'daily_sales']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required columns: {missing_columns}. Required: date, daily_sales"
                )
            
            # Parse dates
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid date format. Please use YYYY-MM-DD format. Error: {str(e)}"
                )
            
            # Validate daily_sales
            if not pd.api.types.is_numeric_dtype(df['daily_sales']):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="daily_sales column must contain numeric values."
                )
            
            # Import to database
            db = get_database()
            records_imported = db.import_sales_data(tmp_path)
            
            # Get date range
            date_range = {
                "start": df['date'].min().strftime('%Y-%m-%d'),
                "end": df['date'].max().strftime('%Y-%m-%d')
            }
            
            return UploadResponse(
                success=True,
                message="Data uploaded successfully",
                records_imported=records_imported,
                date_range=date_range
            )
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload data: {str(e)}"
        )


# ==================== ADDITIONAL ENDPOINTS ====================

@app.get(
    "/api/models",
    tags=["Models"],
    summary="List available models"
)
async def list_models():
    """List all available trained models."""
    try:
        forecaster = get_forecaster()
        db = get_database()
        
        # Get model info from database
        df = db.get_training_results()
        
        models = []
        for model_name in ['linear_trend', 'xgboost', 'random_forest', 'prophet', 'sarima']:
            model_df = df[df['model_name'] == model_name]
            model_info = {
                "name": model_name,
                "loaded": model_name in forecaster.models,
                "trained": not model_df.empty,
                "test_mape": round(model_df['test_mape'].iloc[0], 2) if not model_df.empty else None
            }
            models.append(model_info)
        
        return {
            "success": True,
            "models": models,
            "loaded_count": len(forecaster.models),
            "trained_count": len(df['model_name'].unique())
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.get(
    "/api/forecasts",
    tags=["Forecasting"],
    summary="Get stored forecasts"
)
async def get_stored_forecasts(
    forecast_id: Optional[str] = Query(None, description="Specific forecast ID"),
    model: Optional[str] = Query(None, description="Filter by model name")
):
    """Get stored forecasts from database."""
    try:
        db = get_database()
        
        if forecast_id:
            df = db.get_forecasts(forecast_id=forecast_id, model_name=model)
        else:
            # Get latest forecast
            latest_id = db.get_latest_forecast_id()
            if not latest_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No forecasts found."
                )
            df = db.get_forecasts(forecast_id=latest_id, model_name=model)
        
        # Format response
        forecasts = []
        for _, row in df.iterrows():
            forecasts.append({
                "date": row['forecast_date'].strftime('%Y-%m-%d') if hasattr(row['forecast_date'], 'strftime') else str(row['forecast_date']),
                "predicted_sales": round(row['predicted_sales'], 2),
                "model": row['model_name'],
                "day_number": row['day_number']
            })
        
        return {
            "success": True,
            "forecast_id": df['forecast_id'].iloc[0] if not df.empty else None,
            "total_predictions": len(forecasts),
            "forecasts": forecasts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve forecasts: {str(e)}"
        )


# ==================== FRONTEND ROUTES ====================

@app.get("/app", tags=["Frontend"], include_in_schema=False)
@app.get("/app/", tags=["Frontend"], include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend application."""
    frontend_path = project_root / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/app/{page}.html", tags=["Frontend"], include_in_schema=False)
async def serve_frontend_page(page: str):
    """Serve frontend HTML pages."""
    frontend_path = project_root / "frontend" / f"{page}.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    raise HTTPException(status_code=404, detail=f"Page '{page}' not found")


@app.get("/app/css/{filename}", tags=["Frontend"], include_in_schema=False)
async def serve_css(filename: str):
    """Serve CSS files."""
    css_path = project_root / "frontend" / "css" / filename
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    raise HTTPException(status_code=404, detail=f"CSS file '{filename}' not found")


@app.get("/app/js/{filename}", tags=["Frontend"], include_in_schema=False)
async def serve_js(filename: str):
    """Serve JavaScript files."""
    js_path = project_root / "frontend" / "js" / filename
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail=f"JS file '{filename}' not found")


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

