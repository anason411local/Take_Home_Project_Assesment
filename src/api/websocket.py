"""
WebSocket Connection Manager for Sales Forecasting API.

Provides:
- Connection management for multiple concurrent users
- Real-time forecast progress streaming
- Broadcast capabilities
- Session-based WebSocket connections
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    # Connection
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    
    # Forecast
    FORECAST_START = "forecast_start"
    FORECAST_PROGRESS = "forecast_progress"
    FORECAST_COMPLETE = "forecast_complete"
    FORECAST_ERROR = "forecast_error"
    
    # Data
    DATA_UPDATE = "data_update"
    METRICS_UPDATE = "metrics_update"
    
    # System
    HEARTBEAT = "heartbeat"
    NOTIFICATION = "notification"


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format."""
    type: MessageType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "session_id": self.session_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data.get("type", "notification")),
            data=data.get("data", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            session_id=data.get("session_id")
        )


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    session_id: str
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    is_active: bool = True


class ConnectionManager:
    """
    Manages WebSocket connections for multiple concurrent users.
    
    Features:
    - Session-based connection tracking
    - Broadcast to all connections
    - Send to specific session
    - Connection health monitoring
    - Automatic cleanup of dead connections
    """
    
    def __init__(self):
        """Initialize connection manager."""
        # Map session_id -> ConnectionInfo
        self._connections: Dict[str, ConnectionInfo] = {}
        # Set of all active websockets for broadcast
        self._active_websockets: Set[WebSocket] = set()
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            session_id: Session ID for this connection
            
        Returns:
            True if connected successfully
        """
        try:
            await websocket.accept()
            
            async with self._lock:
                # Close existing connection for this session if any
                if session_id in self._connections:
                    old_conn = self._connections[session_id]
                    if old_conn.websocket in self._active_websockets:
                        self._active_websockets.discard(old_conn.websocket)
                        try:
                            await old_conn.websocket.close()
                        except:
                            pass
                
                # Store new connection
                self._connections[session_id] = ConnectionInfo(
                    websocket=websocket,
                    session_id=session_id
                )
                self._active_websockets.add(websocket)
            
            # Send connection confirmation
            await self.send_to_session(
                session_id,
                WebSocketMessage(
                    type=MessageType.CONNECTED,
                    data={
                        "message": "Connected to Sales Forecasting API",
                        "session_id": session_id
                    },
                    session_id=session_id
                )
            )
            
            logger.info(f"WebSocket connected: session={session_id}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self, session_id: str):
        """
        Disconnect a WebSocket connection.
        
        Args:
            session_id: Session ID to disconnect
        """
        async with self._lock:
            if session_id in self._connections:
                conn = self._connections[session_id]
                self._active_websockets.discard(conn.websocket)
                del self._connections[session_id]
                
                try:
                    if conn.websocket.client_state == WebSocketState.CONNECTED:
                        await conn.websocket.close()
                except:
                    pass
                
                logger.info(f"WebSocket disconnected: session={session_id}")
    
    async def send_to_session(
        self,
        session_id: str,
        message: WebSocketMessage
    ) -> bool:
        """
        Send message to a specific session.
        
        Args:
            session_id: Target session ID
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        async with self._lock:
            conn = self._connections.get(session_id)
            if conn is None:
                return False
        
        try:
            if conn.websocket.client_state == WebSocketState.CONNECTED:
                await conn.websocket.send_text(message.to_json())
                return True
        except Exception as e:
            logger.error(f"Failed to send to session {session_id}: {e}")
            await self.disconnect(session_id)
        
        return False
    
    async def broadcast(self, message: WebSocketMessage):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        disconnected = []
        
        async with self._lock:
            connections = list(self._connections.items())
        
        for session_id, conn in connections:
            try:
                if conn.websocket.client_state == WebSocketState.CONNECTED:
                    await conn.websocket.send_text(message.to_json())
                else:
                    disconnected.append(session_id)
            except Exception as e:
                logger.error(f"Broadcast failed for session {session_id}: {e}")
                disconnected.append(session_id)
        
        # Clean up disconnected
        for session_id in disconnected:
            await self.disconnect(session_id)
    
    async def send_forecast_progress(
        self,
        session_id: str,
        progress: int,
        current_step: str,
        total_steps: int = 100,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Send forecast progress update.
        
        Args:
            session_id: Target session ID
            progress: Progress percentage (0-100)
            current_step: Description of current step
            total_steps: Total number of steps
            details: Additional details
        """
        message = WebSocketMessage(
            type=MessageType.FORECAST_PROGRESS,
            data={
                "progress": progress,
                "current_step": current_step,
                "total_steps": total_steps,
                "details": details or {}
            },
            session_id=session_id
        )
        await self.send_to_session(session_id, message)
    
    async def send_forecast_complete(
        self,
        session_id: str,
        predictions: List[Dict[str, Any]],
        model_used: str,
        summary: Dict[str, Any]
    ):
        """
        Send forecast completion message.
        
        Args:
            session_id: Target session ID
            predictions: List of predictions
            model_used: Model that was used
            summary: Summary statistics
        """
        message = WebSocketMessage(
            type=MessageType.FORECAST_COMPLETE,
            data={
                "success": True,
                "model_used": model_used,
                "predictions": predictions,
                "summary": summary
            },
            session_id=session_id
        )
        await self.send_to_session(session_id, message)
    
    async def send_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str
    ):
        """
        Send error message.
        
        Args:
            session_id: Target session ID
            error_type: Type of error
            error_message: Error message
        """
        message = WebSocketMessage(
            type=MessageType.ERROR,
            data={
                "error_type": error_type,
                "message": error_message
            },
            session_id=session_id
        )
        await self.send_to_session(session_id, message)
    
    async def heartbeat(self, session_id: str) -> bool:
        """
        Send heartbeat to check connection health.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            True if connection is healthy
        """
        async with self._lock:
            conn = self._connections.get(session_id)
            if conn is None:
                return False
            conn.last_heartbeat = datetime.now()
        
        message = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={"status": "alive"},
            session_id=session_id
        )
        return await self.send_to_session(session_id, message)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self._connections)
    
    def get_connected_sessions(self) -> List[str]:
        """Get list of connected session IDs."""
        return list(self._connections.keys())
    
    def is_connected(self, session_id: str) -> bool:
        """Check if session is connected."""
        return session_id in self._connections
    
    async def cleanup_stale_connections(self, timeout_seconds: int = 300):
        """
        Clean up connections that haven't sent heartbeat.
        
        Args:
            timeout_seconds: Timeout in seconds (default: 5 minutes)
        """
        now = datetime.now()
        stale = []
        
        async with self._lock:
            for session_id, conn in self._connections.items():
                delta = (now - conn.last_heartbeat).total_seconds()
                if delta > timeout_seconds:
                    stale.append(session_id)
        
        for session_id in stale:
            await self.disconnect(session_id)
            logger.info(f"Cleaned up stale connection: session={session_id}")


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager

