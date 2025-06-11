from typing import Dict, Any, Optional, List

class UserInfo:
    """Model to store and manage user information."""
    
    def __init__(self):
        """Initialize user information with None values."""
        self.name: Optional[str] = None
        self.gender: Optional[str] = None
        self.age: Optional[int] = None
        self.state: Optional[str] = None
        self.required_fields = ["name", "gender", "age", "state"]
    
    def update(self, field: str, value: Any) -> None:
        """
        Update a specific field with a value.
        
        Args:
            field: Field to update
            value: Value to set
        """
        if field == "name":
            self.name = value
        elif field == "gender":
            self.gender = value
        elif field == "age":
            self.age = value
        elif field == "state":
            self.state = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user information to a dictionary.
        
        Returns:
            Dictionary with user information
        """
        return {
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "state": self.state
        }
    
    def is_complete(self) -> bool:
        """
        Check if all required information is provided.
        
        Returns:
            Whether the user information is complete
        """
        return all([
            self.name is not None,
            self.gender is not None,
            self.age is not None,
            self.state is not None
        ])
    
    def next_required_field(self) -> Optional[str]:
        """
        Get the next field that needs to be collected.
        
        Returns:
            Field name or None if all fields are collected
        """
        if self.name is None:
            return "name"
        elif self.gender is None:
            return "gender"
        elif self.age is None:
            return "age"
        elif self.state is None:
            return "state"
        return None
    
    def get_field(self, field: str) -> Any:
        """
        Get the value of a specific field.
        
        Args:
            field: Field name
            
        Returns:
            Field value
        """
        if field == "name":
            return self.name
        elif field == "gender":
            return self.gender
        elif field == "age":
            return self.age
        elif field == "state":
            return self.state
        return None
    
    def __str__(self) -> str:
        """String representation of user information."""
        return f"Name: {self.name}, Gender: {self.gender}, Age: {self.age}, State: {self.state}" 