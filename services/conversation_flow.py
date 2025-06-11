from typing import Dict, List, Any, Optional, Tuple
import logging

from models.user_info import UserInfo
from utils.conversation import Conversation, ConversationState
from utils.openai_client import OpenAIClient
from utils.validator import validate_field
from utils.language_utils import language_service, detect_language, get_language_specific_greeting, translate_display_fields
from services.recommendation import RecommendationService
from config.config import MIN_CONFIDENCE_THRESHOLD, MAX_RETRIES, EXTRACTION_MODEL, VALIDATION_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationFlow:
    """Service to manage conversation flow and state transitions."""
    
    def __init__(self):
        """Initialize conversation flow service."""
        self.conversation = Conversation()
        self.user_info = UserInfo()
        self.openai_client = OpenAIClient()
        self.recommendation_service = RecommendationService()
        self.retry_count = {}  # Track retry attempts by field
    
    def process_user_message(self, message: str, language: str = "en") -> str:
        """
        Process user message and generate response using intent detection.
        
        Args:
            message: User's message
            language: Language code ('en' for English, 'hi' for Hindi)
            
        Returns:
            Bot's response
        """
        # Set the language in the language service
        language_service.set_language(language)
        
        # Add user message to conversation history
        self.conversation.add_message("user", message)
        
        # Check if this is the first message or a greeting
        if len(self.conversation.get_history()) <= 1 or message.lower() in ["hello", "hi", "hey", "नमस्ते", "हैलो", "हाय"]:
            # Handle as a greeting
            return self._handle_greeting()
            
        # Detect user intent using LLM
        intent = self._detect_intent(message)
        logger.info(f"Detected intent: {intent}")
        
        # Handle message based on detected intent
        if intent == "greeting":
            return self._handle_greeting()
        elif intent == "provide_info":
            return self._handle_information_collection(message)
        elif intent == "request_recommendations":
            # Set state to recommending if user explicitly asks for recommendations
            self.conversation.set_state(ConversationState.RECOMMENDING)
            return self._generate_recommendations()
        elif intent == "restart":
            # Reset conversation
            self.conversation.clear_history()
            return self._handle_greeting()
        elif intent == "ask_specific_scheme":
            # User is asking about a specific scheme after recommendations
            return self._handle_completed_state(message)
        else:
            # Default to state-based handling if intent is unclear
            current_state = self.conversation.get_state()
            
            if current_state == ConversationState.GREETING:
                return self._handle_greeting()
            elif current_state == ConversationState.COLLECTING_INFO:
                return self._handle_information_collection(message)
            elif current_state == ConversationState.RECOMMENDING:
                return self._generate_recommendations()
            elif current_state == ConversationState.COMPLETED:
                return self._handle_completed_state(message)
            else:
                # Default response
                return language_service.get_message("error_message")
    
    def _detect_intent(self, message: str) -> str:
        """
        Detect user intent using LLM.
        
        Args:
            message: User's message
            
        Returns:
            Detected intent
        """
        # Get conversation context
        recent_messages = self.conversation.get_recent_messages(5)
        current_state = self.conversation.get_state()
        user_info_status = {
            "name": self.user_info.name is not None,
            "gender": self.user_info.gender is not None,
            "age": self.user_info.age is not None,
            "state": self.user_info.state is not None
        }
        
        # Prepare prompt for intent detection
        language_code = language_service.get_current_language()
        if language_code == "hi":
            system_prompt = """
            आप एक स्कीमबॉट के लिए इरादा पहचान प्रणाली हैं जो उपयोगकर्ताओं को सरकारी योजनाएँ खोजने में मदद करता है।
            उपयोगकर्ता के संदेश का विश्लेषण करें और निम्न विकल्पों में से उनके इरादे का निर्धारण करें:
            
            - greeting: उपयोगकर्ता अभिवादन कर रहा है या बातचीत शुरू कर रहा है
            - provide_info: उपयोगकर्ता अपने बारे में जानकारी प्रदान कर रहा है
            - request_recommendations: उपयोगकर्ता स्पष्ट रूप से योजना सिफारिशें मांग रहा है
            - restart: उपयोगकर्ता बातचीत को फिर से शुरू करना चाहता है
            - ask_specific_scheme: उपयोगकर्ता किसी विशिष्ट योजना के बारे में पूछ रहा है
            - other: उपरोक्त में से कोई नहीं
            
            केवल इरादे का नाम वापस करें, और कुछ नहीं।
            
            ध्यान दें: उपयोगकर्ता हिंदी या हिंग्लिश में बातचीत कर सकता है (हिंदी शब्द अंग्रेजी लिपि में लिखे गए)। आपको दोनों प्रकार के इनपुट को समझना होगा।
            """
        else:
            system_prompt = """
            You are an intent detection system for a conversational bot that helps users find government schemes.
            Analyze the user's message and determine their intent from the following options:
            
            - greeting: User is greeting or starting a conversation
            - provide_info: User is providing information about themselves
            - request_recommendations: User is explicitly asking for scheme recommendations
            - restart: User wants to restart the conversation
            - ask_specific_scheme: User is asking about a specific scheme
            - other: None of the above
            
            Return only the intent name, nothing else.
            """
        
        yes_text = language_code == "hi" and "हां" or "Yes"
        no_text = language_code == "hi" and "नहीं" or "No"
        
        user_prompt = f"""
        Current bot state: {current_state.name}
        
        User information collected:
        - Name: {yes_text if user_info_status["name"] else no_text}
        - Gender: {yes_text if user_info_status["gender"] else no_text}
        - Age: {yes_text if user_info_status["age"] else no_text}
        - State: {yes_text if user_info_status["state"] else no_text}
        
        Recent conversation:
        {self._format_conversation_history(recent_messages)}
        
        User message: "{message}"
        
        What is the user's intent?
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        intent = self.openai_client.chat_completion(messages=messages).strip().lower()
        
        # Validate that the response is one of the expected intents
        valid_intents = ["greeting", "provide_info", "request_recommendations", "restart", "ask_specific_scheme", "other"]
        if intent not in valid_intents:
            logger.warning(f"Unexpected intent detected: {intent}. Defaulting to 'other'.")
            intent = "other"
        
        return intent
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompts."""
        formatted = ""
        for msg in history:
            role = "Bot" if msg["role"] == "assistant" else "User"
            formatted += f"{role}: {msg['content']}\n"
        return formatted
    
    def _handle_greeting(self) -> str:
        """
        Handle greeting state.
        
        Returns:
            Bot's response
        """
        # Reset user info
        self.user_info = UserInfo()
        self.retry_count = {}
        
        # Generate greeting message based on language
        greeting_message = get_language_specific_greeting(language_service.get_current_language())
        
        # Set next state
        self.conversation.set_state(ConversationState.COLLECTING_INFO)
        
        # Get first question (for name)
        next_question = language_service.get_message("greeting_question")
        
        # Combine greeting and first question
        response = f"{greeting_message}\n\n{next_question}"
        
        # Add bot message to conversation history
        self.conversation.add_message("assistant", response)
        
        return response
    
    def _handle_information_collection(self, message: str) -> str:
        """
        Handle information collection state.
        
        Args:
            message: User's message
            
        Returns:
            Bot's response
        """
        # Get current field to collect
        current_field = self.user_info.next_required_field()
        
        response = ""
        
        if current_field:
            # Extract and validate information
            extracted_info, is_valid = self._extract_and_validate(message, current_field)
            
            # If extraction failed or validation failed
            if not is_valid:
                # Increment retry count
                current_retry = self.retry_count.get(current_field, 0) + 1
                self.retry_count[current_field] = current_retry
                
                # If retry limit reached, skip this field and move to next
                if current_retry >= MAX_RETRIES:
                    # Reset retry count for this field
                    self.retry_count[current_field] = 0
                    
                    # Update user info with best guess
                    if extracted_info:
                        self.user_info.update(current_field, extracted_info)
                    
                    # Get next field to collect
                    next_field = self.user_info.next_required_field()
                    
                    # If no more fields to collect, move to recommendations
                    if not next_field:
                        self.conversation.set_state(ConversationState.RECOMMENDING)
                        return self._generate_recommendations()
                      
                    # Generate message about skipping
                    skip_message = language_service.get_message("skip_message")
                    
                    # Get next question - use hardcoded questions for clarity
                    if next_field == "gender":
                        next_question = language_service.get_message("gender_question")
                    elif next_field == "age":
                        next_question = language_service.get_message("age_question")
                    elif next_field == "state":
                        next_question = language_service.get_message("state_question")
                    else:
                        next_question = language_service.get_message("other_field_question")
                    
                    response = f"{skip_message}\n\n{next_question}"
                else:
                    # Retry the same field with hardcoded retry messages
                    if current_field == "name":
                        response = language_service.get_message("name_retry_message")
                    elif current_field == "gender":
                        response = language_service.get_message("gender_retry_message")
                    elif current_field == "age":
                        response = language_service.get_message("age_retry_message")
                    elif current_field == "state":
                        response = language_service.get_message("state_retry_message")
            else:
                # Update user info with validated value
                self.user_info.update(current_field, extracted_info)
                
                # Reset retry count for this field
                self.retry_count[current_field] = 0
                
                # Get next field to collect
                next_field = self.user_info.next_required_field()
                
                # If no more fields to collect, move to recommendations
                if not next_field:
                    self.conversation.set_state(ConversationState.RECOMMENDING)
                    
                    # Thank you message based on language
                    thank_you_message = language_service.get_message("thank_you_message")
                    
                    # Add thank you message to conversation history
                    self.conversation.add_message("assistant", thank_you_message)
                    
                    # Generate recommendations
                    recommendation_response = self._generate_recommendations()
                    
                    # Return the thank you message - recommendations will be shown in dedicated section
                    return thank_you_message
                else:
                    # Get next question using hardcoded questions for clarity
                    if next_field == "gender":
                        response = language_service.get_message("gender_question")
                    elif next_field == "age":
                        response = language_service.get_message("age_question")
                    elif next_field == "state":
                        response = language_service.get_message("state_question")
                    else:
                        response = language_service.get_message("other_field_question")
        else:
            # All information collected, move to recommendations
            self.conversation.set_state(ConversationState.RECOMMENDING)
            response = language_service.get_message("thank_you_message")
        
        # Add bot response to conversation history
        self.conversation.add_message("assistant", response)
        
        return response
    
    def _extract_and_validate(self, message: str, field: str) -> Tuple[Any, bool]:
        """
        Extract and validate information from user message.
        
        Args:
            message: User's message
            field: Field to extract
            
        Returns:
            Tuple of (extracted_value, is_valid)
        """
        # Extract information using OpenAI
        extraction_result = self.openai_client.extract_information(
            conversation_history=self.conversation.get_recent_messages(5),
            field_to_extract=field,
            model=EXTRACTION_MODEL,
            language=language_service.get_current_language()
        )
        
        extracted_value = extraction_result["value"]
        confidence = extraction_result["confidence"]
        
        logger.info(f"Extracted {field}: {extracted_value} (confidence: {confidence})")
        
        # If confidence is high enough, validate the extracted information
        if confidence >= MIN_CONFIDENCE_THRESHOLD:
            # Validate using validation service
            validation_result = validate_field(
                field=field,
                value=extracted_value
            )
            
            # Check if the field is valid
            if validation_result["valid"]:
                # Return normalized value
                return validation_result["normalized_value"], True
        
        # If extraction failed or validation failed
        return extracted_value, False
    
    def _generate_recommendations(self) -> str:
        """
        Generate scheme recommendations based on user information.
        
        Returns:
            Bot's response with information about recommendations
        """
        # Get scheme recommendations using the recommendation service
        recommendations = self.recommendation_service.get_recommendations(self.user_info, language_service.get_current_language())
        
        # Format recommendations for display
        if recommendations:
            formatted_recommendations = self.recommendation_service.format_recommendations(recommendations, language_service.get_current_language())
            response = formatted_recommendations
        else:
            # No recommendations found
            response = language_service.get_message("no_recommendations_message")
        
        # Set state to completed
        self.conversation.set_state(ConversationState.COMPLETED)
        
        # Add bot response to conversation history
        self.conversation.add_message("assistant", response)
        
        return response
    
    def _handle_completed_state(self, message: str) -> str:
        """
        Handle completed state.
        
        Args:
            message: User's message
            
        Returns:
            Bot's response
        """
        # Check if user wants to restart
        restart_keywords = {
            "en": ["restart", "start over", "reset", "new", "another"],
            "hi": ["शुरू", "फिर से", "रीसेट", "नया", "दोबारा", "पुनः", "restart"]
        }
        
        selected_keywords = restart_keywords.get(language_service.get_current_language(), restart_keywords["en"])
        
        if any(keyword in message.lower() for keyword in selected_keywords):
            # Reset conversation
            self.conversation.clear_history()
            return self._handle_greeting()
        
        # Generate response using LLM
        system_prompt = language_service.get_message("completed_system_prompt")
        
        user_prompt = f"""
        User's message: {message}
        
        User's information: {self.user_info.to_dict()}
        
        Provide a helpful response to the user's query. If they're asking about
        a specific scheme, give details about eligibility, benefits, and how to apply.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.openai_client.chat_completion(messages=messages)
        
        # Add bot response to conversation history
        self.conversation.add_message("assistant", response)
        
        return response
    
    def reset_conversation(self) -> str:
        """
        Reset conversation to initial state.
        
        Returns:
            Bot's greeting message
        """
        # Clear conversation history
        self.conversation.clear_history()
        
        # Return greeting message
        return self._handle_greeting()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation.get_history()
        
    def get_user_info(self) -> UserInfo:
        """Get current user information."""
        return self.user_info 