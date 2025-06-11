import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
import functools

import openai

from config.config import OPENAI_API_KEY
from utils.language_utils import get_bilingual_system_prompt, get_next_question_prompt, detect_language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_openai_errors(default_return=None):
    """Decorator to handle OpenAI API errors consistently."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        try:
            # Set the API key for the older version of the OpenAI package
            openai.api_key = OPENAI_API_KEY
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _base_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        function_schema: Optional[Dict] = None,
        function_name: Optional[str] = None,
        parse_json: bool = False
    ) -> Any:
        """
        Base method for OpenAI chat completion calls.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model to use for completion
            temperature: Sampling temperature (0 to 2)
            max_tokens: Maximum tokens in the response
            function_schema: Function schema for function calling
            function_name: Name of the function to call
            parse_json: Whether to parse JSON from the response
            
        Returns:
            The response from the API (text, JSON, or raw response)
        """
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add function calling if schema is provided
        if function_schema:
            params["functions"] = [function_schema]
            
            if function_name:
                params["function_call"] = {"name": function_name}
        
        try:
            response = openai.ChatCompletion.create(**params)
            
            # Check if function calling was used
            if function_schema and function_name and response.choices[0].message.get("function_call"):
                function_args = response.choices[0].message.function_call.arguments
                
                # Parse JSON if requested
                if parse_json:
                    try:
                        return json.loads(function_args)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from function call")
                        return {}
                
                return function_args
            else:
                # Regular content response
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error in base chat completion: {e}")
            return "" if not parse_json else {}
    
    @handle_openai_errors("")
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Send a request to the OpenAI chat completion API.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model to use for completion
            temperature: Sampling temperature (0 to 2)
            max_tokens: Maximum tokens in the response
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            The text response from the API
        """
        # Use function calling if JSON response format is requested
        if response_format and "type" in response_format and response_format["type"] == "json_object":
            function_schema = {
                "name": "get_response",
                "description": "Get the response in JSON format",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
            return self._base_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                function_schema=function_schema,
                function_name="get_response"
            )
        else:
            return self._base_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    @handle_openai_errors({"value": "", "confidence": 0})
    def extract_information(
        self, 
        conversation_history: List[Dict[str, str]],
        field_to_extract: str,
        model: str = "gpt-3.5-turbo",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Extract specific information from conversation history.
        
        Args:
            conversation_history: Previous messages in the conversation
            field_to_extract: Field to extract (name, gender, age, state)
            model: Model to use
            language: Language code ('en' for English, 'hi' for Hindi)
            
        Returns:
            Dictionary with extracted value and confidence score
        """
        # Use bilingual system prompt if language is Hindi
        system_prompt = get_bilingual_system_prompt(field_to_extract, language)
        
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history
        ]
        
        # Define the JSON structure we expect as output
        function_schema = {
            "name": "extract_information",
            "description": f"Extract {field_to_extract} information",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": f"The extracted {field_to_extract}"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    }
                },
                "required": ["value", "confidence"]
            }
        }
        
        return self._base_chat_completion(
            messages=messages,
            model=model,
            temperature=0.1,
            function_schema=function_schema,
            function_name="extract_information",
            parse_json=True
        )
    
    @handle_openai_errors({"valid": False, "normalized_value": None})
    def validate_information(
        self, 
        field: str, 
        value: Any,
        model: str = "gpt-3.5-turbo",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Validate and normalize extracted information.
        
        Args:
            field: Type of information (name, gender, age, state)
            value: Value to validate
            model: Model to use
            language: Language code ('en' for English, 'hi' for Hindi)
            
        Returns:
            Dictionary with validated value and whether it's valid
        """
        field_specific_instructions = {
            "name": "Return the proper name with correct capitalization. Invalid cases include empty strings or non-name text.",
            "gender": "Normalize to 'Male', 'Female', or 'Other'. Map words like 'boy', 'man', 'M' to 'Male'; 'girl', 'woman', 'F' to 'Female'; etc.",
            "age": "Convert to an integer between 0 and 120. Return -1 if invalid.",
            "state": "Normalize to a proper Indian state/UT name with correct spelling and capitalization. Correct common misspellings like 'delha' to 'Delhi'."
        }
        
        # Add bilingual instructions if language is Hindi
        if language == "hi":
            hinglish_examples = {
                "name": "Examples: 'Mera naam Rahul hai' → 'Rahul', 'मेरा नाम अनिल है' → 'अनिल'",
                "gender": "Examples: 'Main ladka hoon' → 'Male', 'मैं महिला हूँ' → 'Female'",
                "age": "Examples: 'Meri umar 25 saal hai' → 25, 'मैं 30 वर्ष का हूँ' → 30",
                "state": "Examples: 'Main dilli mein rehta hoon' → 'Delhi', 'मैं महाराष्ट्र से हूँ' → 'Maharashtra'"
            }
            
            for f, instruction in field_specific_instructions.items():
                if f == field and f in hinglish_examples:
                    field_specific_instructions[f] = f"{instruction}\n{hinglish_examples[f]}"
        
        # Create system prompt
        system_prompt = f"""
        You are a validation system for user information.
        You will be given a value for {field} and need to determine if it's valid.
        
        {field_specific_instructions[field]}
        
        Return a JSON object with:
        1. "valid": boolean indicating if the value is valid
        2. "normalized_value": the normalized value if valid, null if invalid
        """
        
        # Create user prompt with the value to validate
        user_prompt = f"Validate this {field} value: {value}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Define the JSON structure we expect as output
        function_schema = {
            "name": "validate_information",
            "description": f"Validate {field} information",
            "parameters": {
                "type": "object",
                "properties": {
                    "valid": {
                        "type": "boolean",
                        "description": "Whether the value is valid"
                    },
                    "normalized_value": {
                        "type": ["string", "number", "null"],
                        "description": "Normalized value if valid, null if invalid"
                    }
                },
                "required": ["valid", "normalized_value"]
            }
        }
        
        return self._base_chat_completion(
            messages=messages,
            model=model,
            temperature=0.1,
            function_schema=function_schema,
            function_name="validate_information",
            parse_json=True
        )
            
    @handle_openai_errors("Could you please tell me more about yourself?")
    def get_next_question(
        self,
        user_info: Dict[str, Any],
        next_field: str,
        previous_attempt: bool = False,
        model: str = "gpt-3.5-turbo",
        language: str = "en"
    ) -> str:
        """
        Generate the next question to ask the user.
        
        Args:
            user_info: Current user information
            next_field: Next field to collect
            previous_attempt: Whether this is a retry after a failed attempt
            model: Model to use
            language: Language code ('en' for English, 'hi' for Hindi)
            
        Returns:
            Question to ask
        """
        # Get language-specific prompt
        system_prompt = get_next_question_prompt(language)
        
        # Create context for the model
        context = {
            "collected_info": user_info,
            "next_field": next_field,
            "is_retry": previous_attempt,
            "language": language
        }
        
        user_prompt = json.dumps(context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._base_chat_completion(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=200
        )
    
    def recommend_schemes(
        self,
        user_info: Dict[str, Any],
        schemes: List[Dict[str, Any]],
        model: str = "gpt-3.5-turbo"
    ) -> List[Dict[str, Any]]:
        """
        Recommend government schemes based on user information.
        
        Args:
            user_info: User's complete information
            schemes: List of all available schemes
            model: Model to use
            
        Returns:
            List of recommended schemes with relevance scores
        """
        system_prompt = """
        You are SchemeBot, an intelligent chatbot designed to help Indian citizens 
        find government schemes they may be eligible for. Your task is to recommend the most
        relevant government schemes for a user based on their demographic information.
        
        Analyze the eligibility criteria of each scheme and score them based on relevance to the user.
        Return your response as a JSON array of objects, each containing:
        1. "scheme_id": The ID of the scheme
        2. "relevance_score": A score from 0 to 1 indicating how relevant this scheme is for the user
        3. "reason": A brief explanation of why this scheme is relevant or not
        
        Only include schemes with relevance score > 0.2.
        """
        
        user_prompt = f"""
        User information:
        {json.dumps(user_info, indent=2)}
        
        Available schemes:
        {json.dumps(schemes, indent=2)}
        
        Return the recommended schemes for this user, sorted by relevance (highest first).
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Define the JSON structure we expect as output
            function_schema = {
                "name": "recommend_schemes",
                "description": "Recommend schemes based on user information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommended_schemes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scheme_id": {
                                        "type": "string",
                                        "description": "ID of the scheme"
                                    },
                                    "relevance_score": {
                                        "type": "number",
                                        "description": "Score from 0 to 1 indicating relevance"
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Explanation of relevance"
                                    }
                                },
                                "required": ["scheme_id", "relevance_score"]
                            }
                        }
                    },
                    "required": ["recommended_schemes"]
                }
            }
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                functions=[function_schema],
                function_call={"name": "recommend_schemes"}
            )
            
            function_args = response.choices[0].message.get("function_call", {}).get("arguments", "{}")
            
            try:
                result = json.loads(function_args)
                results = result.get("recommended_schemes", [])
                
                # Match with full scheme details
                recommended_schemes = []
                for result in results:
                    scheme_id = result.get("scheme_id")
                    for scheme in schemes:
                        if scheme.get("id") == scheme_id:
                            recommended_schemes.append({
                                **scheme,
                                "relevance_score": result.get("relevance_score", 0),
                                "reason": result.get("reason", "")
                            })
                            break
                
                return sorted(recommended_schemes, key=lambda x: x.get("relevance_score", 0), reverse=True)
            except json.JSONDecodeError:
                return []
        except Exception as e:
            logger.error(f"Error recommending schemes: {e}")
            return [] 