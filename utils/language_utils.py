import re
import logging
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hindi unicode range (Devanagari)
HINDI_PATTERN = re.compile(r'[\u0900-\u097F]')

# Common Hinglish patterns
HINGLISH_PATTERNS = [
    r'\b(mera|meri|mujhe|hum|humko|hamara)\b',  # Personal pronouns
    r'\b(naam|umr|saal|sal|varsh|umar)\b',      # Age and name related
    r'\b(hai|hain|ho|hoon|hu|tha|thi|the)\b',   # Forms of "to be"
    r'\b(kya|kaun|kahan|kaise|kyun|kab)\b',     # Question words
    r'\b(aur|ya|lekin|magar|par|phir)\b',       # Conjunctions
    r'\b(accha|theek|bahut|jyada|kam)\b',       # Adjectives
    r'\b(nahi|na|no)\b',                        # Negation
]

# Compiled Hinglish pattern
HINGLISH_PATTERN = re.compile('|'.join(HINGLISH_PATTERNS), re.IGNORECASE)

class LanguageService:
    """
    Centralized service for handling multilingual functionality.
    Implements singleton pattern to ensure consistent language handling.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize language service with default values."""
        self.current_language = "en"  # Default language
        self.supported_languages = ["en", "hi"]
        
        # Initialize language-specific message templates
        self.message_templates = {
            "en": {
                "greeting": "Hello! I'm SchemeBot, your assistant for finding Indian government schemes you may be eligible for. To provide personalized recommendations, I need to ask you a few questions.",
                "no_schemes_found": "I couldn't find any schemes that match your profile. You might want to check the official government websites for more information.",
                "recommendation_intro": "Based on your information, I've found {num_schemes} government schemes you might be eligible for.",
                "error_message": "I'm not sure how to respond to that. Can you please try rephrasing your question?",
                "ask_name": "Please tell me your name.",
                "ask_gender": "Are you male, female, or other?",
                "ask_age": "What is your age?", 
                "ask_state": "Which state in India do you live in?",
                "retry_name": "I'm having trouble understanding your name. Could you please tell me your name again?",
                "retry_gender": "I'm having trouble understanding your gender. Please specify if you are male, female, or other.",
                "retry_age": "I'm having trouble understanding your age. Please provide your age in years.",
                "retry_state": "I'm having trouble understanding your state. Please specify which state or union territory in India you live in.",
                "thank_you_message": "Thank you for providing all the information! Let me find some schemes that might be relevant for you.",
                "skip_message": "I'm having trouble understanding your response. Let's move on to the next question.",
                "gender_question": "Are you male, female, or other?",
                "age_question": "What is your age?",
                "state_question": "Which state in India do you live in?",
                "other_field_question": "Please provide the requested information.",
                "name_retry_message": "I'm having trouble understanding your name. Could you please tell me your name again?",
                "gender_retry_message": "I'm having trouble understanding your gender. Please specify if you are male, female, or other.",
                "age_retry_message": "I'm having trouble understanding your age. Please provide your age in years.",
                "state_retry_message": "I'm having trouble understanding your state. Please specify which state or union territory in India you live in.",
                "greeting_question": "Please tell me your name.",
                "no_recommendations_message": "Based on your information, I couldn't find any government schemes that match your profile. You might want to check official government websites for more information.",
                "completed_system_prompt": "You are SchemeBot, a helpful assistant for Indian government schemes. The user has already received scheme recommendations, so provide information about specific schemes or answer questions they might have. Be concise and friendly in your response."
            },
            "hi": {
                "greeting": "नमस्ते! मैं स्कीमबॉट हूँ, आपका सहायक जो आपको पात्र हो सकने वाली भारत सरकार की योजनाओं को खोजने में मदद करता है। व्यक्तिगत सिफारिशें प्रदान करने के लिए, मुझे आपसे कुछ प्रश्न पूछने होंगे।",
                "no_schemes_found": "मुझे आपके प्रोफ़ाइल से मेल खाने वाली कोई योजना नहीं मिली। आप अधिक जानकारी के लिए सरकारी वेबसाइटों को देख सकते हैं।",
                "recommendation_intro": "आपकी जानकारी के आधार पर, मुझे {num_schemes} सरकारी योजनाएँ मिली हैं जिनके लिए आप पात्र हो सकते हैं।",
                "error_message": "मुझे समझ नहीं आया। क्या आप अपना प्रश्न दोबारा बता सकते हैं?",
                "ask_name": "कृपया मुझे अपना नाम बताएं।",
                "ask_gender": "क्या आप पुरुष हैं, महिला हैं, या अन्य हैं?",
                "ask_age": "आपकी उम्र क्या है?",
                "ask_state": "आप भारत के किस राज्य में रहते हैं?",
                "retry_name": "मुझे आपका नाम समझने में कठिनाई हो रही है। कृपया अपना नाम फिर से बताएं।",
                "retry_gender": "मुझे आपका लिंग समझने में कठिनाई हो रही है। कृपया स्पष्ट करें कि आप पुरुष हैं, महिला हैं, या अन्य हैं।",
                "retry_age": "मुझे आपकी उम्र समझने में कठिनाई हो रही है। कृपया अपनी उम्र वर्षों में बताएं।",
                "retry_state": "मुझे आपका राज्य समझने में कठिनाई हो रही है। कृपया भारत का राज्य या केंद्र शासित प्रदेश बताएं जहां आप रहते हैं।",
                "thank_you_message": "सभी जानकारी प्रदान करने के लिए धन्यवाद! मुझे आपके लिए प्रासंगिक योजनाएँ खोजने दें।",
                "skip_message": "मुझे आपके जवाब को समझने में परेशानी हो रही है। आइए अगले सवाल पर चलते हैं।",
                "gender_question": "क्या आप पुरुष हैं, महिला हैं, या अन्य हैं?",
                "age_question": "आपकी उम्र क्या है?",
                "state_question": "आप भारत के किस राज्य में रहते हैं?",
                "other_field_question": "कृपया अनुरोधित जानकारी प्रदान करें।",
                "name_retry_message": "मुझे आपका नाम समझने में कठिनाई हो रही है। कृपया अपना नाम फिर से बताएं।",
                "gender_retry_message": "मुझे आपका लिंग समझने में कठिनाई हो रही है। कृपया स्पष्ट करें कि आप पुरुष हैं, महिला हैं, या अन्य हैं।",
                "age_retry_message": "मुझे आपकी उम्र समझने में कठिनाई हो रही है। कृपया अपनी उम्र वर्षों में बताएं।",
                "state_retry_message": "मुझे आपका राज्य समझने में कठिनाई हो रही है। कृपया भारत का राज्य या केंद्र शासित प्रदेश बताएं जहां आप रहते हैं।",
                "greeting_question": "कृपया मुझे अपना नाम बताएं।",
                "no_recommendations_message": "आपकी जानकारी के आधार पर, मुझे कोई ऐसी सरकारी योजना नहीं मिली जो आपके प्रोफ़ाइल से मेल खाती हो। अधिक जानकारी के लिए आप सरकारी वेबसाइटों की जांच कर सकते हैं।",
                "completed_system_prompt": "आप स्कीमबॉट हैं, भारतीय सरकारी योजनाओं के लिए एक सहायक सहायक। उपयोगकर्ता को पहले से ही योजना सिफारिशें मिल चुकी हैं, इसलिए विशिष्ट योजनाओं के बारे में जानकारी प्रदान करें या उनके पास हो सकने वाले प्रश्नों के उत्तर दें। अपने उत्तर में संक्षिप्त और मित्रवत रहें। हमेशा हिंदी में उत्तर दें क्योंकि उपयोगकर्ता हिंदी में बातचीत कर रहा है।"
            }
        }
    
    def set_language(self, language_code: str) -> None:
        """
        Set the current language.
        
        Args:
            language_code: Language code ('en' for English, 'hi' for Hindi)
        """
        if language_code in self.supported_languages:
            self.current_language = language_code
            logger.info(f"Language set to {language_code}")
        else:
            logger.warning(f"Unsupported language: {language_code}")
    
    def get_current_language(self) -> str:
        """Get the current language code."""
        return self.current_language
    
    def get_message(self, message_key: str, **kwargs) -> str:
        """
        Get a message in the current language.
        
        Args:
            message_key: Key for the message template
            **kwargs: Format arguments for the message
            
        Returns:
            Formatted message in the current language
        """
        language = self.current_language
        templates = self.message_templates.get(language, self.message_templates["en"])
        message_template = templates.get(message_key, f"[{message_key}]")
        
        try:
            return message_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing format argument for message {message_key}: {e}")
            return message_template
    
    def get_field_question(self, field: str, is_retry: bool = False) -> str:
        """
        Get a question for a specific field in the current language.
        
        Args:
            field: Field to ask about (name, gender, age, state)
            is_retry: Whether this is a retry question
            
        Returns:
            Question string in the current language
        """
        prefix = "retry_" if is_retry else "ask_"
        return self.get_message(f"{prefix}{field}")


# Initialize singleton instance
language_service = LanguageService()

def detect_language(text: str) -> str:
    """
    Detect language from text.
    
    Args:
        text: Text to detect language from
        
    Returns:
        Language code ('en' for English, 'hi' for Hindi)
    """
    # Simple detection based on Devanagari script or Hinglish patterns
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')  # Devanagari Unicode range
    
    if devanagari_pattern.search(text):
        return "hi"  # Hindi detected
    
    # Check for common Hindi/Hinglish patterns
    hinglish_patterns = [
        r'\b(namaste|namaskar|dhanyavaad|shukriya)\b',
        r'\b(kya|kaise|kaun|kahan|kyun|aap|tum|mein|hai|hain|tha|the|gaya|gaye)\b',
        r'\b(nahi|haan|accha|theek)\b'
    ]
    
    for pattern in hinglish_patterns:
        if re.search(pattern, text.lower()):
            return "hi"  # Hinglish/Hindi detected
    
    return "en"  # Default to English

def get_language_specific_greeting(language: str = "en") -> str:
    """
    Get greeting message in specified language.
    
    Args:
        language: Language code ('en' for English, 'hi' for Hindi)
        
    Returns:
        Greeting message
    """
    return language_service.get_message("greeting")

def get_next_question_prompt(language: str, field: str, previous_attempt: bool = False) -> Tuple[str, str]:
    """
    Get language-specific prompt for the next question.
    
    Args:
        language: Language code ('en' or 'hi')
        field: Field to ask about (name, gender, age, state)
        previous_attempt: Whether this is a retry
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = ""
    user_prompt = ""
    
    if language == 'hi':
        system_prompt = """
        आप स्कीमबॉट हैं, एक बुद्धिमान चैटबॉट जो भारतीय नागरिकों को सरकारी योजनाओं को खोजने में मदद करने के लिए डिज़ाइन किया गया है। अपने उत्तरों में मित्रवत, संक्षिप्त और प्राकृतिक रहें।
        ध्यान दें: उपयोगकर्ता हिंदी या हिंग्लिश में बातचीत कर रहा है, इसलिए आपको हमेशा हिंदी में जवाब देना चाहिए।
        """
        
        retry_text = "क्योंकि मैं आपका पिछला उत्तर नहीं समझ पाया, " if previous_attempt else ""
        
        if field == "name":
            user_prompt = f"{retry_text}कृपया मुझे अपना नाम बताएं।"
        elif field == "gender":
            user_prompt = f"{retry_text}आप पुरुष हैं, महिला हैं, या अन्य हैं?"
        elif field == "age":
            user_prompt = f"{retry_text}आपकी उम्र क्या है?"
        elif field == "state":
            user_prompt = f"{retry_text}आप भारत के किस राज्य में रहते हैं?"
        else:
            user_prompt = f"{retry_text}कृपया अपना {field} बताएं।"
    else:
        system_prompt = """
        You are SchemeBot, an intelligent chatbot designed to help Indian citizens 
        find government schemes they may be eligible for. Be friendly, concise, and natural in your responses.
        """
        
        retry_text = "I couldn't understand your previous response, so " if previous_attempt else ""
        
        if field == "name":
            user_prompt = f"{retry_text}please tell me your name."
        elif field == "gender":
            user_prompt = f"{retry_text}are you male, female, or other?"
        elif field == "age":
            user_prompt = f"{retry_text}what is your age?"
        elif field == "state":
            user_prompt = f"{retry_text}which state in India do you live in?"
        else:
            user_prompt = f"{retry_text}please tell me your {field}."
    
    return system_prompt, user_prompt

def get_bilingual_system_prompt(field: str, language: str) -> str:
    """
    Get a bilingual system prompt for extracting information.
    
    Args:
        field: Field to extract (name, gender, age, state)
        language: Language code ('en' or 'hi')
        
    Returns:
        System prompt for OpenAI
    """
    base_prompt = f"""
    You are an information extraction assistant specialized in bilingual (Hindi-English) processing.
    Your task is to extract the user's {field} from the conversation history.
    
    Return your response as a JSON object with these fields:
    1. "value": The extracted {field} (string)
    2. "confidence": Your confidence in the extraction (float between 0 and 1)
    
    If you cannot find the information, return an empty string for value and 0 for confidence.
    """
    
    if language == 'hi':
        base_prompt += """
        IMPORTANT: The user may be communicating in Hindi or Hinglish (Hindi words written in English script). 
        You must be able to understand both and extract the information correctly.
        
        Examples for Hinglish understanding:
        - "Mera naam Rahul hai" -> Name: "Rahul"
        - "Main 25 saal ka hoon" -> Age: "25"
        - "Main Dilli mein rehta hoon" -> State: "Delhi"
        - "Main ladka hoon" -> Gender: "Male"
        
        Examples for Hindi understanding:
        - "मेरा नाम राहुल है" -> Name: "राहुल"
        - "मेरी उम्र 25 साल है" -> Age: "25"
        - "मैं दिल्ली में रहता हूँ" -> State: "Delhi"
        - "मैं पुरुष हूँ" -> Gender: "Male"
        
        Always normalize state names to standard Indian state names in English.
        Always normalize gender to "Male", "Female", or "Other" in English.
        Always normalize age to a number.
        Names can be kept in the original language (Hindi script if provided in Hindi).
        """
    
    return base_prompt

def translate_display_fields(recommendations: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    """
    Translate display fields in recommendations based on language.
    
    Args:
        recommendations: List of scheme recommendations
        language: Language code ('en' or 'hi')
        
    Returns:
        Updated recommendations with appropriate language fields
    """
    if language != 'hi':
        return recommendations
    
    field_mapping = {
        'name': 'name_hi',
        'description': 'description_hi',
        'eligibility': 'eligibility_hi',
        'benefits': 'benefits_hi',
        'documents_required': 'documents_required_hi',
        'how_to_apply': 'how_to_apply_hi',
        'category': 'category_hi',
        'implementing_agency': 'implementing_agency_hi'
    }
    
    translated_recs = []
    for rec in recommendations:
        # Create a copy of the recommendation
        translated_rec = rec.copy()
        
        # Replace fields with Hindi versions if available
        for eng_field, hi_field in field_mapping.items():
            if hi_field in rec and rec[hi_field]:
                # If Hindi field exists, use it in place of English field
                translated_rec[eng_field] = rec[hi_field]
        
        # Handle 'reason' field specially - needs to be translated on the fly
        if 'reason' in translated_rec:
            # Note: In a production system, you might want to use a translation service here
            # For demo purposes, we're keeping it as is
            pass
            
        translated_recs.append(translated_rec)
    
    return translated_recs 