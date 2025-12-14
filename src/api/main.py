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
from .pipeline_manager import PipelineManager, PipelineConfig, PipelineStep, get_pipeline_manager
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


# ==================== LIFECYCLE EVENTS ====================

# Environment variable to control auto-start of dashboards (default: True in Docker)
AUTO_START_DASHBOARDS = os.environ.get("AUTO_START_DASHBOARDS", "true").lower() in ("true", "1", "yes")


@app.on_event("startup")
async def startup_event():
    """Initialize application and auto-start dashboards if configured."""
    logger.info("Application starting up...")
    
    if AUTO_START_DASHBOARDS:
        logger.info("Auto-starting MLflow and Optuna dashboards in background...")
        # Start dashboards in a background task to not block app startup
        import asyncio
        asyncio.create_task(_start_dashboards_background())
    else:
        logger.info("Dashboard auto-start disabled (set AUTO_START_DASHBOARDS=true to enable)")


async def _start_dashboards_background():
    """Background task to start dashboards after app is fully started."""
    import asyncio
    
    # Wait for app to be fully ready
    await asyncio.sleep(5)
    
    try:
        manager = get_dashboard_manager()
        
        # Start MLflow
        logger.info("Starting MLflow UI...")
        mlflow_result = manager.start_mlflow("localhost")
        if mlflow_result["status"] in ["started", "running"]:
            logger.info(f"✓ MLflow UI: {mlflow_result.get('message', 'Started')} - http://localhost:{manager.mlflow_port}")
        else:
            logger.warning(f"✗ MLflow UI: {mlflow_result.get('message', 'Failed to start')}")
        
        # Start Optuna
        logger.info("Starting Optuna Dashboard...")
        optuna_result = manager.start_optuna("localhost")
        if optuna_result["status"] in ["started", "running"]:
            logger.info(f"✓ Optuna Dashboard: {optuna_result.get('message', 'Started')} - http://localhost:{manager.optuna_port}")
        else:
            logger.warning(f"✗ Optuna Dashboard: {optuna_result.get('message', 'Failed to start')}")
            
    except Exception as e:
        logger.error(f"Error auto-starting dashboards: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up dashboard processes on application shutdown."""
    logger.info("Application shutting down - cleaning up dashboard processes...")
    try:
        manager = get_dashboard_manager()
        manager.stop_all()
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")


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


# ==================== DASHBOARD PROCESS MANAGER ====================

import subprocess
import socket
import signal
import atexit
import time

# Environment variables for Docker/deployment configuration
MLFLOW_PORT = int(os.environ.get("MLFLOW_PORT", "5000"))
OPTUNA_PORT = int(os.environ.get("OPTUNA_PORT", "8080"))
DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST", "")  # Empty means use request host
EXTERNAL_HOST = os.environ.get("EXTERNAL_HOST", "")  # For Docker: the external hostname/IP


class DashboardManager:
    """
    Manages MLflow and Optuna dashboard processes.
    
    Docker-ready with environment variable configuration:
    - MLFLOW_PORT: Port for MLflow UI (default: 5000)
    - OPTUNA_PORT: Port for Optuna Dashboard (default: 8080)
    - DASHBOARD_HOST: Hostname to use in URLs (default: from request)
    - EXTERNAL_HOST: External hostname for Docker (default: same as request)
    
    Example Docker usage:
        docker run -e MLFLOW_PORT=5000 -e OPTUNA_PORT=8080 -e EXTERNAL_HOST=myserver.com \\
                   -p 8000:8000 -p 5000:5000 -p 8080:8080 myapp
    """
    
    def __init__(self):
        self.mlflow_process: Optional[subprocess.Popen] = None
        self.optuna_process: Optional[subprocess.Popen] = None
        self.mlflow_port = MLFLOW_PORT
        self.optuna_port = OPTUNA_PORT
        
        # Get Python executable path from current environment
        self.python_exe = sys.executable
        self.python_dir = Path(self.python_exe).parent
        
        # Scripts directory (where mlflow and optuna-dashboard are installed)
        if os.name == 'nt':  # Windows
            self.scripts_dir = self.python_dir / "Scripts"
        else:  # Linux/Mac
            self.scripts_dir = self.python_dir
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """Clean up all subprocess on exit."""
        logger.info("Cleaning up dashboard processes...")
        self._terminate_process(self.mlflow_process, "MLflow")
        self._terminate_process(self.optuna_process, "Optuna")
    
    def _terminate_process(self, process: Optional[subprocess.Popen], name: str):
        """Safely terminate a subprocess."""
        if process is not None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                logger.info(f"{name} process terminated")
            except Exception as e:
                logger.warning(f"Error terminating {name}: {e}")
    
    def _get_host_url(self, port: int, request_host: Optional[str] = None) -> str:
        """
        Get the appropriate URL for the dashboard.
        
        Priority:
        1. EXTERNAL_HOST env var (for Docker/remote access)
        2. DASHBOARD_HOST env var
        3. Request host header
        4. Fallback to localhost
        """
        if EXTERNAL_HOST:
            host = EXTERNAL_HOST
        elif DASHBOARD_HOST:
            host = DASHBOARD_HOST
        elif request_host:
            # Extract hostname without port
            host = request_host.split(':')[0]
        else:
            host = "localhost"
        
        return f"http://{host}:{port}"
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Check on all interfaces (0.0.0.0) for Docker compatibility
            return s.connect_ex(('127.0.0.1', port)) == 0
    
    def _get_mlflow_cmd(self) -> list:
        """Get the MLflow command with proper path."""
        if os.name == 'nt':
            mlflow_exe = self.scripts_dir / "mlflow.exe"
            if mlflow_exe.exists():
                return [str(mlflow_exe)]
        else:
            # On Linux/Mac, use the executable directly from scripts_dir
            mlflow_exe = self.scripts_dir / "mlflow"
            if mlflow_exe.exists():
                return [str(mlflow_exe)]
        # Fallback to using python -m mlflow
        return [self.python_exe, "-m", "mlflow"]
    
    def _get_optuna_cmd(self) -> list:
        """Get the Optuna dashboard command with proper path."""
        if os.name == 'nt':
            optuna_exe = self.scripts_dir / "optuna-dashboard.exe"
            if optuna_exe.exists():
                return [str(optuna_exe)]
        else:
            # On Linux/Mac, use the executable directly from scripts_dir
            # Note: optuna_dashboard cannot be run with python -m
            optuna_exe = self.scripts_dir / "optuna-dashboard"
            if optuna_exe.exists():
                return [str(optuna_exe)]
        # Fallback - this may not work for optuna-dashboard
        return [self.python_exe, "-m", "optuna_dashboard"]
    
    def start_mlflow(self, request_host: Optional[str] = None) -> Dict[str, Any]:
        """Start MLflow UI server."""
        url = self._get_host_url(self.mlflow_port, request_host)
        
        if self.is_port_in_use(self.mlflow_port):
            return {
                "status": "running",
                "port": self.mlflow_port,
                "url": url,
                "message": "MLflow UI is already running"
            }
        
        try:
            mlruns_dir = project_root / "mlruns"
            if not mlruns_dir.exists():
                return {
                    "status": "error",
                    "message": "MLflow runs directory not found. Train models first."
                }
            
            # Build command - bind to 0.0.0.0 for Docker accessibility
            cmd = self._get_mlflow_cmd() + ["ui", "--port", str(self.mlflow_port), "--host", "0.0.0.0"]
            logger.info(f"Starting MLflow with command: {cmd}")
            
            # Platform-specific process creation
            kwargs = {
                "cwd": str(project_root),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            
            if os.name == 'nt':
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                # On Unix, start new process group for proper cleanup
                kwargs["start_new_session"] = True
            
            # Start MLflow UI
            self.mlflow_process = subprocess.Popen(cmd, **kwargs)
            
            # Wait for startup with retries
            max_retries = 10
            for i in range(max_retries):
                time.sleep(1)
                if self.is_port_in_use(self.mlflow_port):
                    return {
                        "status": "started",
                        "port": self.mlflow_port,
                        "url": url,
                        "message": "MLflow UI started successfully"
                    }
                # Check if process died
                if self.mlflow_process.poll() is not None:
                    break
            
            # If we get here, startup failed
            stderr = ""
            if self.mlflow_process.poll() is not None:
                try:
                    _, stderr_bytes = self.mlflow_process.communicate(timeout=1)
                    stderr = stderr_bytes.decode('utf-8', errors='ignore') if stderr_bytes else ""
                except:
                    pass
            return {
                "status": "error",
                "message": f"Failed to start MLflow UI. {stderr[:200] if stderr else 'Process may still be starting...'}"
            }
                
        except Exception as e:
            logger.error(f"MLflow start error: {e}")
            return {
                "status": "error",
                "message": f"Failed to start MLflow UI: {str(e)}"
            }
    
    def start_optuna(self, request_host: Optional[str] = None) -> Dict[str, Any]:
        """Start Optuna Dashboard server."""
        url = self._get_host_url(self.optuna_port, request_host)
        
        if self.is_port_in_use(self.optuna_port):
            return {
                "status": "running",
                "port": self.optuna_port,
                "url": url,
                "message": "Optuna Dashboard is already running"
            }
        
        try:
            optuna_db = project_root / "models" / "optuna" / "optuna_studies.db"
            if not optuna_db.exists():
                return {
                    "status": "error",
                    "message": "Optuna studies database not found. Train models first."
                }
            
            # Build command - bind to 0.0.0.0 for Docker accessibility
            cmd = self._get_optuna_cmd() + [f"sqlite:///{str(optuna_db)}", "--port", str(self.optuna_port), "--host", "0.0.0.0"]
            logger.info(f"Starting Optuna with command: {cmd}")
            
            # Platform-specific process creation
            kwargs = {
                "cwd": str(project_root),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            
            if os.name == 'nt':
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                # On Unix, start new process group for proper cleanup
                kwargs["start_new_session"] = True
            
            # Start Optuna Dashboard
            self.optuna_process = subprocess.Popen(cmd, **kwargs)
            
            # Wait for startup with retries
            max_retries = 10
            for i in range(max_retries):
                time.sleep(1)
                if self.is_port_in_use(self.optuna_port):
                    return {
                        "status": "started",
                        "port": self.optuna_port,
                        "url": url,
                        "message": "Optuna Dashboard started successfully"
                    }
                # Check if process died
                if self.optuna_process.poll() is not None:
                    break
            
            # If we get here, startup failed
            stderr = ""
            if self.optuna_process.poll() is not None:
                try:
                    _, stderr_bytes = self.optuna_process.communicate(timeout=1)
                    stderr = stderr_bytes.decode('utf-8', errors='ignore') if stderr_bytes else ""
                except:
                    pass
            return {
                "status": "error",
                "message": f"Failed to start Optuna Dashboard. {stderr[:200] if stderr else 'Process may still be starting...'}"
            }
                
        except Exception as e:
            logger.error(f"Optuna start error: {e}")
            return {
                "status": "error",
                "message": f"Failed to start Optuna Dashboard: {str(e)}"
            }
    
    def get_status(self, request_host: Optional[str] = None) -> Dict[str, Any]:
        """Get status of both dashboards."""
        mlflow_running = self.is_port_in_use(self.mlflow_port)
        optuna_running = self.is_port_in_use(self.optuna_port)
        
        return {
            "mlflow": {
                "running": mlflow_running,
                "port": self.mlflow_port,
                "url": self._get_host_url(self.mlflow_port, request_host) if mlflow_running else None
            },
            "optuna": {
                "running": optuna_running,
                "port": self.optuna_port,
                "url": self._get_host_url(self.optuna_port, request_host) if optuna_running else None
            }
        }
    
    def stop_mlflow(self) -> Dict[str, Any]:
        """Stop MLflow UI server."""
        if self.mlflow_process:
            self._terminate_process(self.mlflow_process, "MLflow")
            self.mlflow_process = None
            return {"status": "stopped", "message": "MLflow UI stopped"}
        return {"status": "not_running", "message": "MLflow UI was not running"}
    
    def stop_optuna(self) -> Dict[str, Any]:
        """Stop Optuna Dashboard server."""
        if self.optuna_process:
            self._terminate_process(self.optuna_process, "Optuna")
            self.optuna_process = None
            return {"status": "stopped", "message": "Optuna Dashboard stopped"}
        return {"status": "not_running", "message": "Optuna Dashboard was not running"}
    
    def stop_all(self) -> Dict[str, Any]:
        """Stop all dashboard processes."""
        mlflow_result = self.stop_mlflow()
        optuna_result = self.stop_optuna()
        return {
            "mlflow": mlflow_result,
            "optuna": optuna_result
        }


# Global dashboard manager
_dashboard_manager: Optional[DashboardManager] = None

def get_dashboard_manager() -> DashboardManager:
    """Get or initialize the dashboard manager."""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager()
    return _dashboard_manager


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
        
        # Generate forecast with confidence intervals
        if model_name == "ensemble":
            forecast_df = forecaster.get_ensemble_forecast(horizon, include_ci=True)
        else:
            forecast_df = forecaster.forecast(model_name, horizon, include_ci=True)
        
        # Progress: Processing results
        await connection_manager.send_forecast_progress(
            session_id, 80, "Processing results..."
        )
        await asyncio.sleep(0.1)
        
        # Check if CI columns exist
        has_ci = 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns
        
        # Format predictions with CI
        predictions = []
        for _, row in forecast_df.iterrows():
            pred = {
                "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                "value": round(row['predicted_sales'], 2)
            }
            if has_ci:
                pred["lower_bound"] = round(row['lower_bound'], 2)
                pred["upper_bound"] = round(row['upper_bound'], 2)
            predictions.append(pred)
        
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
    summary="Generate sales forecast with confidence intervals"
)
async def forecast(request: ForecastRequest):
    """
    Generate sales forecast for the specified number of days with confidence intervals.
    
    - **horizon**: Number of days to forecast (1-365)
    - **model**: Model to use (default: best performing model)
    - **include_ci**: Whether to include 95% confidence intervals (default: True)
    
    Confidence Interval Methods:
    - **Prophet/SARIMA**: Native SD-based intervals from model
    - **XGBoost/RandomForest/LinearTrend**: MAD-based intervals (robust to outliers)
    
    Returns predictions with date, value, and optional lower/upper bounds.
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
        
        # Determine CI method based on model type
        native_ci_models = ['prophet', 'sarima']
        ci_method = "native" if model_name in native_ci_models else "mad"
        
        # Check if model is loaded
        if model_name == "ensemble":
            if not forecaster.models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No models loaded for ensemble forecasting."
                )
            # Generate ensemble forecast with CI
            forecast_df = forecaster.get_ensemble_forecast(
                request.horizon, 
                include_ci=request.include_ci
            )
            ci_method = "ensemble"
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
            
            # Generate forecast with CI
            forecast_df = forecaster.forecast(
                model_name, 
                request.horizon, 
                include_ci=request.include_ci
            )
        
        # Check if CI columns exist
        has_ci = 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns
        
        # Format predictions with CI
        predictions = []
        for _, row in forecast_df.iterrows():
            pred_item = PredictionItem(
                date=row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                value=round(row['predicted_sales'], 2),
                lower_bound=round(row['lower_bound'], 2) if has_ci and request.include_ci else None,
                upper_bound=round(row['upper_bound'], 2) if has_ci and request.include_ci else None
            )
            predictions.append(pred_item)
        
        # Calculate summary
        values = forecast_df['predicted_sales'].values
        summary = {
            "total": round(float(values.sum()), 2),
            "mean": round(float(values.mean()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2)
        }
        
        # Add CI summary if available
        if has_ci and request.include_ci:
            summary["ci_mean_width"] = round(float(
                (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
            ), 2)
        
        return ForecastResponse(
            success=True,
            model_used=model_name,
            horizon=request.horizon,
            predictions=predictions,
            summary=summary,
            confidence_level=0.95,
            ci_method=ci_method if request.include_ci else "none"
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


# ==================== MLOPS DASHBOARD ENDPOINTS ====================

@app.get(
    "/api/dashboards/status",
    tags=["Dashboards"],
    summary="Get dashboard status"
)
async def get_dashboard_status(request: Request):
    """
    Get status of MLflow and Optuna dashboards.
    
    Returns whether each dashboard is running and their URLs.
    URLs are dynamically generated based on the request host for Docker compatibility.
    """
    try:
        manager = get_dashboard_manager()
        request_host = request.headers.get("host", "localhost")
        dashboard_status = manager.get_status(request_host)
        return {
            "success": True,
            **dashboard_status
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard status: {str(e)}"
        )


@app.post(
    "/api/dashboards/mlflow/start",
    tags=["Dashboards"],
    summary="Start MLflow UI"
)
async def start_mlflow_dashboard(request: Request):
    """
    Start the MLflow UI server.
    
    Port can be configured via MLFLOW_PORT environment variable (default: 5000).
    
    MLflow UI provides:
    - Experiment tracking
    - Model metrics visualization
    - Artifact management
    - Run comparison
    """
    try:
        manager = get_dashboard_manager()
        request_host = request.headers.get("host", "localhost")
        result = manager.start_mlflow(request_host)
        return {
            "success": result["status"] in ["started", "running"],
            **result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start MLflow UI: {str(e)}"
        )


@app.post(
    "/api/dashboards/optuna/start",
    tags=["Dashboards"],
    summary="Start Optuna Dashboard"
)
async def start_optuna_dashboard(request: Request):
    """
    Start the Optuna Dashboard server.
    
    Port can be configured via OPTUNA_PORT environment variable (default: 8080).
    
    Optuna Dashboard provides:
    - Hyperparameter optimization visualization
    - Study history and progress
    - Parameter importance analysis
    - Optimization history plots
    """
    try:
        manager = get_dashboard_manager()
        request_host = request.headers.get("host", "localhost")
        result = manager.start_optuna(request_host)
        return {
            "success": result["status"] in ["started", "running"],
            **result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start Optuna Dashboard: {str(e)}"
        )


@app.post(
    "/api/dashboards/mlflow/stop",
    tags=["Dashboards"],
    summary="Stop MLflow UI"
)
async def stop_mlflow_dashboard():
    """Stop the MLflow UI server."""
    try:
        manager = get_dashboard_manager()
        result = manager.stop_mlflow()
        return {
            "success": True,
            **result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop MLflow UI: {str(e)}"
        )


@app.post(
    "/api/dashboards/optuna/stop",
    tags=["Dashboards"],
    summary="Stop Optuna Dashboard"
)
async def stop_optuna_dashboard():
    """Stop the Optuna Dashboard server."""
    try:
        manager = get_dashboard_manager()
        result = manager.stop_optuna()
        return {
            "success": True,
            **result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop Optuna Dashboard: {str(e)}"
        )


# ==================== EDA ENDPOINTS ====================

@app.get(
    "/api/eda/report",
    tags=["EDA"],
    summary="Get EDA report"
)
async def get_eda_report():
    """
    Get the exploratory data analysis report.
    
    Returns the EDA report text and key insights.
    """
    try:
        eda_dir = project_root / "reports" / "eda"
        
        if not eda_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="EDA reports not found. Run step_2_eda_analysis.py first."
            )
        
        # Read EDA report
        report_path = eda_dir / "eda_report.txt"
        report_content = ""
        if report_path.exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            except UnicodeDecodeError:
                with open(report_path, 'r', encoding='latin-1') as f:
                    report_content = f.read()
        
        # Read key insights
        insights_path = eda_dir / "key_insights.txt"
        insights_content = ""
        if insights_path.exists():
            try:
                with open(insights_path, 'r', encoding='utf-8') as f:
                    insights_content = f.read()
            except UnicodeDecodeError:
                with open(insights_path, 'r', encoding='latin-1') as f:
                    insights_content = f.read()
        
        # Parse report into sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in report_content.split('\n'):
            if line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. ') or \
               line.startswith('4. ') or line.startswith('5. ') or line.startswith('6. '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip()
                current_content = []
            elif current_section and line.strip() and not line.startswith('===') and not line.startswith('---'):
                current_content.append(line.strip())
        
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        # Parse key insights into list
        insights_list = []
        for line in insights_content.split('\n'):
            if line.strip() and line[0].isdigit():
                # Remove the number prefix
                insight = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                insights_list.append(insight)
        
        return {
            "success": True,
            "report": report_content,
            "insights": insights_list,
            "sections": sections
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get EDA report: {str(e)}"
        )


@app.get(
    "/api/eda/images",
    tags=["EDA"],
    summary="List EDA images"
)
async def list_eda_images():
    """
    List all available EDA visualization images.
    
    Returns a list of image names and their descriptions.
    """
    try:
        eda_dir = project_root / "reports" / "eda"
        
        if not eda_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="EDA reports not found. Run step_2_eda_analysis.py first."
            )
        
        # Define image descriptions
        image_info = {
            "summary_dashboard.png": {
                "title": "Summary Dashboard",
                "description": "Complete overview of all EDA visualizations",
                "category": "overview"
            },
            "time_series.png": {
                "title": "Time Series Analysis",
                "description": "Daily sales over time with moving averages",
                "category": "trend"
            },
            "trend_analysis.png": {
                "title": "Trend Analysis",
                "description": "Linear trend and growth rate analysis",
                "category": "trend"
            },
            "seasonality.png": {
                "title": "Seasonality Patterns",
                "description": "Day-of-week and monthly patterns",
                "category": "seasonality"
            },
            "distribution.png": {
                "title": "Sales Distribution",
                "description": "Histogram and statistical distribution of sales",
                "category": "distribution"
            },
            "correlation_heatmap.png": {
                "title": "Correlation Heatmap",
                "description": "Feature correlations with daily sales",
                "category": "correlation"
            },
            "boxplots.png": {
                "title": "Box Plots",
                "description": "Sales distribution by day of week and outlier analysis",
                "category": "distribution"
            }
        }
        
        # Find available images
        images = []
        for img_file in eda_dir.glob("*.png"):
            img_name = img_file.name
            if img_name in image_info:
                images.append({
                    "filename": img_name,
                    "url": f"/api/eda/image/{img_name}",
                    **image_info[img_name]
                })
            else:
                images.append({
                    "filename": img_name,
                    "url": f"/api/eda/image/{img_name}",
                    "title": img_name.replace('_', ' ').replace('.png', '').title(),
                    "description": "EDA visualization",
                    "category": "other"
                })
        
        # Sort by category order
        category_order = ["overview", "trend", "seasonality", "distribution", "correlation", "other"]
        images.sort(key=lambda x: (category_order.index(x["category"]) if x["category"] in category_order else 99, x["title"]))
        
        return {
            "success": True,
            "total": len(images),
            "images": images
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list EDA images: {str(e)}"
        )


@app.get(
    "/api/eda/image/{filename}",
    tags=["EDA"],
    summary="Get EDA image"
)
async def get_eda_image(filename: str):
    """
    Get a specific EDA visualization image.
    
    - **filename**: Name of the image file (e.g., time_series.png)
    """
    try:
        eda_dir = project_root / "reports" / "eda"
        image_path = eda_dir / filename
        
        if not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {filename}"
            )
        
        return FileResponse(
            str(image_path),
            media_type="image/png",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get EDA image: {str(e)}"
        )


# ==================== TRAINING PIPELINE ENDPOINTS ====================

# Store active training WebSocket connections
_training_connections: Dict[str, WebSocket] = {}


@app.post(
    "/api/training/upload",
    tags=["Training"],
    summary="Upload CSV for training"
)
async def upload_training_csv(
    file: UploadFile = File(..., description="CSV file with sales data")
):
    """
    Upload a CSV file to start the training pipeline.
    
    The CSV file must contain:
    - **date**: Date column (YYYY-MM-DD format)
    - **daily_sales**: Daily sales values
    
    After upload, call `/api/training/start` to begin the pipeline.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are accepted"
            )
        
        # Read file content
        content = await file.read()
        
        # Get pipeline manager and upload
        manager = get_pipeline_manager(project_root)
        result = await manager.upload_csv(content, file.filename)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Upload failed")
            )
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.post(
    "/api/training/start",
    tags=["Training"],
    summary="Start training pipeline"
)
async def start_training_pipeline(
    optuna_trials: int = Query(10, ge=1, le=100, description="Number of Optuna trials"),
    models: str = Query(
        "linear_trend,xgboost,random_forest,prophet,sarima",
        description="Comma-separated list of models to train"
    ),
    test_size: float = Query(0.2, ge=0.1, le=0.4, description="Test set fraction"),
    use_holdout: bool = Query(False, description="Use holdout split (Option 1)"),
    skip_eda: bool = Query(False, description="Skip EDA step"),
    skip_training: bool = Query(False, description="Skip training (only preprocessing)")
):
    """
    Start the complete training pipeline.
    
    Pipeline steps:
    1. **Preprocessing**: Data validation, cleaning, feature engineering
    2. **EDA**: Exploratory data analysis (optional)
    3. **Training**: Model training with Optuna optimization
    
    Connect to WebSocket `/ws/training/{session_id}` to receive real-time logs.
    """
    try:
        manager = get_pipeline_manager(project_root)
        
        # Check if already running
        status_info = manager.get_status()
        if status_info["is_running"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Pipeline is already running"
            )
        
        # Parse models
        model_list = [m.strip() for m in models.split(",")]
        
        # Create config
        config = PipelineConfig(
            optuna_trials=optuna_trials,
            models=model_list,
            test_size=test_size,
            use_holdout=use_holdout,
            skip_eda=skip_eda,
            skip_training=skip_training
        )
        
        # Start pipeline in background
        asyncio.create_task(_run_pipeline_async(manager, config))
        
        return {
            "success": True,
            "message": "Pipeline started",
            "config": config.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start pipeline: {str(e)}"
        )


async def _run_pipeline_async(manager: PipelineManager, config: PipelineConfig):
    """Run pipeline in background and broadcast logs."""
    
    # Set up log streaming to WebSocket clients
    async def log_callback(message: str, log_type: str):
        """Broadcast log to all connected training WebSockets."""
        for session_id, ws in list(_training_connections.items()):
            try:
                await ws.send_json({
                    "type": "log",
                    "message": message,
                    "log_type": log_type,
                    "timestamp": datetime.now().isoformat()
                })
            except:
                # Connection closed
                _training_connections.pop(session_id, None)
    
    async def status_callback(status_obj):
        """Broadcast status to all connected training WebSockets."""
        for session_id, ws in list(_training_connections.items()):
            try:
                await ws.send_json({
                    "type": "status",
                    "status": status_obj.to_dict()
                })
            except:
                _training_connections.pop(session_id, None)
    
    manager.set_log_callback(lambda msg, log_type: asyncio.create_task(log_callback(msg, log_type)))
    manager.set_status_callback(lambda s: asyncio.create_task(status_callback(s)))
    
    # Run pipeline
    result = await manager.run_pipeline(config)
    
    # Notify completion
    for session_id, ws in list(_training_connections.items()):
        try:
            await ws.send_json({
                "type": "complete",
                "success": result["success"],
                "results": result.get("results"),
                "error": result.get("error")
            })
        except:
            pass
    
    # Reset forecaster to load new models
    global _forecaster
    _forecaster = None


@app.get(
    "/api/training/status",
    tags=["Training"],
    summary="Get training status"
)
async def get_training_status():
    """Get current training pipeline status."""
    try:
        manager = get_pipeline_manager(project_root)
        status_info = manager.get_status()
        logs = manager.get_logs(limit=50)
        
        return {
            "success": True,
            **status_info,
            "recent_logs": logs
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@app.post(
    "/api/training/cancel",
    tags=["Training"],
    summary="Cancel training pipeline"
)
async def cancel_training_pipeline():
    """Cancel the currently running training pipeline."""
    try:
        manager = get_pipeline_manager(project_root)
        
        if not manager.get_status()["is_running"]:
            return {
                "success": False,
                "message": "No pipeline is currently running"
            }
        
        manager.cancel()
        
        return {
            "success": True,
            "message": "Pipeline cancellation requested"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel: {str(e)}"
        )


@app.post(
    "/api/training/reset",
    tags=["Training"],
    summary="Reset training state"
)
async def reset_training_state():
    """Reset the training pipeline state."""
    try:
        manager = get_pipeline_manager(project_root)
        manager.reset()
        
        return {
            "success": True,
            "message": "Training state reset"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset: {str(e)}"
        )


@app.websocket("/ws/training/{session_id}")
async def training_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time training log streaming.
    
    Connect to receive:
    - **log**: Real-time log messages from training scripts
    - **status**: Pipeline status updates
    - **complete**: Pipeline completion notification
    """
    await websocket.accept()
    _training_connections[session_id] = websocket
    
    logger.info(f"Training WebSocket connected: session={session_id}")
    
    try:
        # Send current status
        manager = get_pipeline_manager(project_root)
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "status": manager.get_status()
        })
        
        # Send recent logs
        for log in manager.get_logs(limit=100):
            await websocket.send_json({
                "type": "log",
                "message": log,
                "log_type": "info"
            })
        
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"Training WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")
    finally:
        _training_connections.pop(session_id, None)


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

