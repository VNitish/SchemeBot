import time
import logging
from typing import Dict, List, Any, Optional
from enum import Enum

from models.user_info import UserInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Enum for conversation states."""
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    RECOMMENDING = "recommending"
    COMPLETED = "completed"


class StateManager:
    """
    Centralized state management for the application.
    Implements singleton pattern to ensure consistency.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize state manager with default values."""
        # Conversation state
        self.conversation_state = ConversationState.GREETING
        self.conversation_history = []
        self.start_time = time.time()
        self.last_activity = time.time()
        
        # User information
        self.user_info = UserInfo()
        
        # Recommendations
        self.recommendations = []
        self.schemes_expanded = {}
        
        # UI state
        self.page_num = 0
        
        # Language
        self.language = "en"
        self.auto_detect_language = True
        
        # Retry tracking
        self.retry_count = {}
    
    def reset_conversation(self):
        """Reset conversation state to initial values."""
        self.conversation_state = ConversationState.GREETING
        self.conversation_history = []
        self.start_time = time.time()
        self.last_activity = time.time()
        self.user_info = UserInfo()
        self.retry_count = {}
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        self.last_activity = time.time()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, str]]:
        """
        Get the most recent messages from conversation history.
        
        Args:
            count: Number of recent messages to retrieve
            
        Returns:
            List of recent messages
        """
        return self.conversation_history[-count:] if len(self.conversation_history) > count else self.conversation_history
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last message from the user."""
        for message in reversed(self.conversation_history):
            if message["role"] == "user":
                return message["content"]
        return None
    
    def set_state(self, state: ConversationState) -> None:
        """Set conversation state."""
        self.conversation_state = state
    
    def get_state(self) -> ConversationState:
        """Get current conversation state."""
        return self.conversation_state
    
    def is_expired(self, timeout: int = 600) -> bool:
        """
        Check if conversation has been inactive for too long.
        
        Args:
            timeout: Inactivity timeout in seconds
            
        Returns:
            Whether the conversation has expired
        """
        return (time.time() - self.last_activity) > timeout
    
    def update_user_info(self, field: str, value: Any) -> None:
        """
        Update user information.
        
        Args:
            field: Field to update
            value: Value to set
        """
        self.user_info.update(field, value)
    
    def get_user_info(self) -> UserInfo:
        """Get user information."""
        return self.user_info
    
    def set_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """Set scheme recommendations."""
        self.recommendations = recommendations
        self.page_num = 0  # Reset to first page
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get scheme recommendations."""
        return self.recommendations
    
    def set_language(self, language: str) -> None:
        """Set language preference."""
        self.language = language
    
    def get_language(self) -> str:
        """Get current language preference."""
        return self.language
    
    def set_auto_detect_language(self, auto_detect: bool) -> None:
        """Set whether to auto-detect language."""
        self.auto_detect_language = auto_detect
    
    def is_auto_detect_language(self) -> bool:
        """Check if auto-detect language is enabled."""
        return self.auto_detect_language
    
    def increment_retry_count(self, field: str) -> int:
        """
        Increment retry count for a field.
        
        Args:
            field: Field that's being retried
            
        Returns:
            Updated retry count
        """
        self.retry_count[field] = self.retry_count.get(field, 0) + 1
        return self.retry_count[field]
    
    def get_retry_count(self, field: str) -> int:
        """Get retry count for a field."""
        return self.retry_count.get(field, 0)
    
    def reset_retry_count(self, field: str = None) -> None:
        """
        Reset retry count for a field or all fields.
        
        Args:
            field: Specific field to reset, or None for all fields
        """
        if field is None:
            self.retry_count = {}
        else:
            self.retry_count[field] = 0


# Initialize singleton instance
state_manager = StateManager() 