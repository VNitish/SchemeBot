from typing import Any, Dict, Optional, Tuple, Union
import re
from config.config import INDIAN_STATES, GENDER_OPTIONS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationService:
    """
    Centralized validation service for user input.
    Implements singleton pattern to ensure validation consistency.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def validate_name(self, name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate and normalize a name.
        
        Args:
            name: Name to validate
            
        Returns:
            Tuple of (is_valid, normalized_name)
        """
        if not name or not isinstance(name, str):
            return False, None
        
        # Remove leading/trailing whitespace
        name = name.strip()
        
        # Check if name is too short or empty
        if len(name) < 2:
            return False, None
        
        # Check if name contains non-alphabetic characters (except spaces, hyphens, and apostrophes)
        if re.search(r'[^a-zA-Z\s\'\-]', name):
            # Allow Hindi characters
            if not re.search(r'[\u0900-\u097F]', name):
                return False, None
        
        # Normalize capitalization (first letter of each word capitalized)
        normalized_name = ' '.join(word.capitalize() for word in name.split())
        
        return True, normalized_name
    
    def validate_gender(self, gender: str) -> Tuple[bool, Optional[str]]:
        """
        Validate and normalize gender.
        
        Args:
            gender: Gender to validate
            
        Returns:
            Tuple of (is_valid, normalized_gender)
        """
        if not gender or not isinstance(gender, str):
            return False, None
        
        # Remove leading/trailing whitespace and convert to lowercase
        gender = gender.strip().lower()
        
        # Map common variations to standard options
        gender_mappings = {
            # Male variations
            'male': 'Male', 'm': 'Male', 'man': 'Male', 'boy': 'Male', 'guy': 'Male', 'लड़का': 'Male', 
            'पुरुष': 'Male', 'आदमी': 'Male', 'लडका': 'Male', 'पुरूष': 'Male', 'ladka': 'Male', 'purush': 'Male',
            # Female variations
            'female': 'Female', 'f': 'Female', 'woman': 'Female', 'girl': 'Female', 'lady': 'Female', 
            'महिला': 'Female', 'लड़की': 'Female', 'औरत': 'Female', 'स्त्री': 'Female', 'लडकी': 'Female',
            'ladki': 'Female', 'mahila': 'Female', 'aurat': 'Female',
            # Other variations
            'other': 'Other', 'o': 'Other', 'non-binary': 'Other', 'nonbinary': 'Other', 'trans': 'Other', 
            'transgender': 'Other', 'prefer not to say': 'Other', 'अन्य': 'Other', 'थर्ड जेंडर': 'Other',
            'third gender': 'Other', 'anya': 'Other'
        }
        
        normalized_gender = gender_mappings.get(gender)
        if normalized_gender:
            return True, normalized_gender
        
        # Check if any substring matches
        for key, value in gender_mappings.items():
            if key in gender:
                return True, value
        
        return False, None
    
    def validate_age(self, age: Any) -> Tuple[bool, Optional[int]]:
        """
        Validate and normalize age.
        
        Args:
            age: Age to validate (can be string or number)
            
        Returns:
            Tuple of (is_valid, normalized_age)
        """
        # Convert string to number if necessary
        if isinstance(age, str):
            # Extract numbers from the string
            age_str = age.strip()
            number_match = re.search(r'\d+', age_str)
            if number_match:
                try:
                    age = int(number_match.group())
                except ValueError:
                    return False, None
            else:
                return False, None
        
        # Validate age range
        try:
            age_int = int(age)
            if 0 <= age_int <= 120:
                return True, age_int
        except (ValueError, TypeError):
            pass
        
        return False, None
    
    def validate_state(self, state: str) -> Tuple[bool, Optional[str]]:
        """
        Validate and normalize state.
        
        Args:
            state: State to validate
            
        Returns:
            Tuple of (is_valid, normalized_state)
        """
        if not state or not isinstance(state, str):
            return False, None
        
        # Remove leading/trailing whitespace
        state = state.strip()
        
        # Direct match with list of Indian states and UTs
        for valid_state in INDIAN_STATES:
            if state.lower() == valid_state.lower():
                return True, valid_state
        
        # Common misspellings and abbreviations
        state_mappings = {
            'delhi': 'Delhi',
            'dilli': 'Delhi',
            'new delhi': 'Delhi',
            'ncr': 'Delhi',
            'mumbai': 'Maharashtra',
            'bombay': 'Maharashtra',
            'bangalore': 'Karnataka',
            'bengaluru': 'Karnataka',
            'calcutta': 'West Bengal',
            'kolkata': 'West Bengal',
            'madras': 'Tamil Nadu',
            'chennai': 'Tamil Nadu',
            'hyderabad': 'Telangana',
            'ap': 'Andhra Pradesh',
            'up': 'Uttar Pradesh',
            'mp': 'Madhya Pradesh',
            'hp': 'Himachal Pradesh',
            'uk': 'Uttarakhand',
            'uttaranchal': 'Uttarakhand',
            'tn': 'Tamil Nadu',
            'wb': 'West Bengal',
            'jk': 'Jammu and Kashmir',
            'j&k': 'Jammu and Kashmir',
            'andra': 'Andhra Pradesh',
            'andhrapradesh': 'Andhra Pradesh',
            'arunachalpradesh': 'Arunachal Pradesh',
            'tamilnadu': 'Tamil Nadu',
            'westbengal': 'West Bengal',
            'uttarpradesh': 'Uttar Pradesh',
            'madhyapradesh': 'Madhya Pradesh',
            'himachalpradesh': 'Himachal Pradesh'
        }
        
        normalized_state = state_mappings.get(state.lower())
        if normalized_state:
            return True, normalized_state
        
        # Check if any substring matches
        for key, value in state_mappings.items():
            if key in state.lower():
                return True, value
        
        # Fuzzy matching for close matches
        for valid_state in INDIAN_STATES:
            # Check if 70% of characters match
            if len(set(state.lower()) & set(valid_state.lower())) / len(set(valid_state.lower())) > 0.7:
                return True, valid_state
        
        return False, None


# Initialize singleton instance
validation_service = ValidationService()

def validate_field(field: str, value: Any) -> Dict[str, Any]:
    """
    Validate user input for a specific field.
    
    Args:
        field: Field to validate (name, gender, age, state)
        value: Value to validate
        
    Returns:
        Dictionary with validated value and whether it's valid
    """
    is_valid = False
    normalized_value = None
    
    if field == "name":
        is_valid, normalized_value = validation_service.validate_name(value)
    elif field == "gender":
        is_valid, normalized_value = validation_service.validate_gender(value)
    elif field == "age":
        is_valid, normalized_value = validation_service.validate_age(value)
    elif field == "state":
        is_valid, normalized_value = validation_service.validate_state(value)
    
    return {
        "valid": is_valid,
        "normalized_value": normalized_value
    } 