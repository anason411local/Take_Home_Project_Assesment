"""
Session Management for Sales Forecasting API.

Provides:
- Unique session ID generation
- Session storage and retrieval
- Session expiration handling
- User preferences per session
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from threading import Lock
import json


@dataclass
class SessionData:
    """Data stored for each session."""
    session_id: str
    created_at: float
    last_accessed: float
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # User preferences
    preferred_model: str = "best"
    default_horizon: int = 30
    
    # Session state
    is_active: bool = True
    forecast_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_accessed": datetime.fromtimestamp(self.last_accessed).isoformat(),
            "user_agent": self.user_agent,
            "ip_address": self.ip_address,
            "preferred_model": self.preferred_model,
            "default_horizon": self.default_horizon,
            "is_active": self.is_active,
            "forecast_count": len(self.forecast_history)
        }


class SessionManager:
    """
    Manages user sessions with thread-safe operations.
    
    Features:
    - Unique session ID generation (UUID4)
    - Session storage with expiration
    - Thread-safe concurrent access
    - Session preferences storage
    """
    
    def __init__(self, session_timeout: int = 3600):
        """
        Initialize session manager.
        
        Args:
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self._sessions: Dict[str, SessionData] = {}
        self._lock = Lock()
        self._session_timeout = session_timeout
    
    def create_session(
        self,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> SessionData:
        """
        Create a new session with unique ID.
        
        Args:
            user_agent: Client user agent string
            ip_address: Client IP address
            
        Returns:
            SessionData object
        """
        session_id = str(uuid.uuid4())
        current_time = time.time()
        
        session = SessionData(
            session_id=session_id,
            created_at=current_time,
            last_accessed=current_time,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        with self._lock:
            self._sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session by ID and update last accessed time.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            SessionData or None if not found/expired
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session is None:
                return None
            
            # Check if expired
            if self._is_expired(session):
                del self._sessions[session_id]
                return None
            
            # Update last accessed
            session.last_accessed = time.time()
            return session
    
    def update_session(
        self,
        session_id: str,
        preferred_model: Optional[str] = None,
        default_horizon: Optional[int] = None
    ) -> Optional[SessionData]:
        """
        Update session preferences.
        
        Args:
            session_id: Session ID
            preferred_model: Preferred forecasting model
            default_horizon: Default forecast horizon
            
        Returns:
            Updated SessionData or None if not found
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        
        with self._lock:
            if preferred_model:
                session.preferred_model = preferred_model
            if default_horizon:
                session.default_horizon = default_horizon
            session.last_accessed = time.time()
        
        return session
    
    def add_forecast_to_history(
        self,
        session_id: str,
        forecast_data: Dict[str, Any]
    ) -> bool:
        """
        Add a forecast to session history.
        
        Args:
            session_id: Session ID
            forecast_data: Forecast data to store
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        
        with self._lock:
            # Keep last 10 forecasts
            session.forecast_history.append({
                "timestamp": datetime.now().isoformat(),
                **forecast_data
            })
            if len(session.forecast_history) > 10:
                session.forecast_history = session.forecast_history[-10:]
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        removed = 0
        with self._lock:
            expired_ids = [
                sid for sid, session in self._sessions.items()
                if self._is_expired(session)
            ]
            for sid in expired_ids:
                del self._sessions[sid]
                removed += 1
        return removed
    
    def _is_expired(self, session: SessionData) -> bool:
        """Check if session is expired."""
        return time.time() - session.last_accessed > self._session_timeout
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        with self._lock:
            return sum(
                1 for session in self._sessions.values()
                if not self._is_expired(session)
            )
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions (for admin purposes)."""
        with self._lock:
            return [
                session.to_dict()
                for session in self._sessions.values()
                if not self._is_expired(session)
            ]


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(session_timeout=3600)  # 1 hour timeout
    return _session_manager

