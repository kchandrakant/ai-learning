"""
Session Management for Conversation History

Manages conversation state across chat interactions.
"""

import time
import logging
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single conversation message."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """A conversation session."""
    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class SessionManager:
    """
    Manages conversation sessions and history.
    
    Features:
    - In-memory session storage (swap for Redis in production)
    - Automatic session expiration
    - Conversation history truncation
    """
    
    def __init__(self):
        """Initialize session manager."""
        self._sessions: dict[str, Session] = {}
        self._timeout_seconds = settings.session_timeout_minutes * 60
        self._max_history = settings.max_conversation_history
    
    def get_or_create_session(self, session_id: str) -> Session:
        """Get existing session or create a new one."""
        self._cleanup_expired_sessions()
        
        if session_id not in self._sessions:
            logger.info(f"Creating new session: {session_id}")
            self._sessions[session_id] = Session(session_id=session_id)
        
        session = self._sessions[session_id]
        session.last_active = time.time()
        return session
    
    def get_history(self, session_id: str) -> list[dict]:
        """
        Get conversation history for a session.
        
        Returns:
            List of message dicts with 'role' and 'content'
        """
        session = self.get_or_create_session(session_id)
        
        # Return last N messages
        messages = session.messages[-self._max_history:]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str
    ) -> None:
        """
        Add a message to session history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
        """
        session = self.get_or_create_session(session_id)
        
        message = Message(role=role, content=content)
        session.messages.append(message)
        session.last_active = time.time()
        
        # Trim if exceeds max
        if len(session.messages) > self._max_history * 2:
            session.messages = session.messages[-self._max_history:]
            logger.debug(f"Trimmed session {session_id} to {len(session.messages)} messages")
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages in a session.
        
        Returns:
            True if session existed, False otherwise
        """
        if session_id in self._sessions:
            self._sessions[session_id].messages = []
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.
        
        Returns:
            True if session existed, False otherwise
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def get_session_metadata(
        self, 
        session_id: str
    ) -> Optional[dict]:
        """Get session metadata including statistics."""
        if session_id not in self._sessions:
            return None
        
        session = self._sessions[session_id]
        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "created_at": session.created_at,
            "last_active": session.last_active,
            "metadata": session.metadata
        }
    
    def set_session_metadata(
        self, 
        session_id: str, 
        key: str, 
        value: any
    ) -> None:
        """Set custom metadata on a session."""
        session = self.get_or_create_session(session_id)
        session.metadata[key] = value
    
    def _cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_active > self._timeout_seconds
        ]
        
        for sid in expired:
            del self._sessions[sid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        self._cleanup_expired_sessions()
        return len(self._sessions)
    
    def export_session(self, session_id: str) -> Optional[dict]:
        """
        Export session data for persistence or debugging.
        
        Returns:
            Full session data as dict, or None if not found
        """
        if session_id not in self._sessions:
            return None
        
        session = self._sessions[session_id]
        return {
            "session_id": session.session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in session.messages
            ],
            "created_at": session.created_at,
            "last_active": session.last_active,
            "metadata": session.metadata
        }


# =============================================================================
# Alternative: Redis-backed Session Manager (Production)
# =============================================================================

class RedisSessionManager:
    """
    Redis-backed session manager for production use.
    
    This is a stub showing the interface. Implementation would use
    redis-py or aioredis for async support.
    
    Benefits over in-memory:
    - Persistence across restarts
    - Shared across multiple instances
    - Built-in TTL expiration
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis connection."""
        # import redis
        # self._client = redis.from_url(redis_url)
        raise NotImplementedError(
            "Redis session manager not implemented. "
            "Install: pip install redis and implement methods."
        )
    
    # Same interface as SessionManager
    # Keys: session:{session_id}
    # Values: JSON-serialized session data
    # TTL: settings.session_timeout_minutes * 60
