from typing import Dict, List, Any, Optional
import logging

from utils.openai_client import OpenAIClient
from utils.validator import validate_field
from config.config import EXTRACTION_MODEL, VALIDATION_MODEL, MIN_CONFIDENCE_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionService:
    """Service to extract and validate user information from conversation."""
    
    def __init__(self):
        """Initialize extraction service."""
        self.openai_client = OpenAIClient()
    
    def extract_field(
        self, 
        conversation_history: List[Dict[str, str]], 
        field: str
    ) -> Dict[str, Any]:
        """
        Extract a specific field from conversation history.
        
        Args:
            conversation_history: List of conversation messages
            field: Field to extract (name, gender, age, state)
            
        Returns:
            Dictionary with extraction results
        """
        # Extract information using LLM
        extraction_result = self.openai_client.extract_information(
            conversation_history=conversation_history,
            field_to_extract=field,
            model=EXTRACTION_MODEL
        )
        
        value = extraction_result.get("value", "")
        confidence = extraction_result.get("confidence", 0)
        
        logger.info(f"Extracted {field}: {value} (confidence: {confidence})")
        
        # Check if confidence is too low
        if not value or confidence < MIN_CONFIDENCE_THRESHOLD:
            return {
                "value": value,
                "confidence": confidence,
                "is_valid": False,
                "validated_value": None
            }
        
        # Validate the extracted value
        validation_result = self.validate_field(field, value)
        
        return {
            "value": value,
            "confidence": confidence,
            "is_valid": validation_result["is_valid"],
            "validated_value": validation_result["value"]
        }
    
    def validate_field(self, field: str, value: Any) -> Dict[str, Any]:
        """
        Validate a field value.
        
        Args:
            field: Field name (name, gender, age, state)
            value: Value to validate
            
        Returns:
            Dictionary with validation results
        """
        # First try local validation
        local_validation = validate_field(field, value)
        
        # If local validation passes, return the result
        if local_validation["is_valid"]:
            return local_validation
        
        # If local validation fails, try LLM validation as fallback
        llm_validation = self.openai_client.validate_information(
            field=field,
            value=value,
            model=VALIDATION_MODEL
        )
        
        return llm_validation 