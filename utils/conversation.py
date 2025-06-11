from typing import Dict, List, Optional, Any
import time
from enum import Enum

class ConversationState(Enum):
    """Enum for conversation states."""
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    RECOMMENDING = "recommending"
    COMPLETED = "completed"


class Conversation:
    """Class to manage conversation state and history."""
    
    def __init__(self):
        """Initialize a new conversation."""
        self.history: List[Dict[str, str]] = []
        self.state = ConversationState.GREETING
        self.start_time = time.time()
        self.last_activity = time.time()
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.history.append({
            "role": role,
            "content": content
        })
        self.last_activity = time.time()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, str]]:
        """
        Get the most recent messages from conversation history.
        
        Args:
            count: Number of recent messages to retrieve
            
        Returns:
            List of recent messages
        """
        return self.history[-count:] if len(self.history) > count else self.history
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last message from the user."""
        for message in reversed(self.history):
            if message["role"] == "user":
                return message["content"]
        return None
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        self.state = ConversationState.GREETING
        self.start_time = time.time()
        self.last_activity = time.time()
    
    def set_state(self, state: ConversationState) -> None:
        """Set conversation state."""
        self.state = state
    
    def get_state(self) -> ConversationState:
        """Get current conversation state."""
        return self.state
    
    def is_expired(self, timeout: int = 600) -> bool:
        """
        Check if conversation has been inactive for too long.
        
        Args:
            timeout: Inactivity timeout in seconds
            
        Returns:
            Whether the conversation has expired
        """
        return (time.time() - self.last_activity) > timeout 