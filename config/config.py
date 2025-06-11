import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXTRACTION_MODEL = "gpt-3.5-turbo"
VALIDATION_MODEL = "gpt-3.5-turbo"
RECOMMENDATION_MODEL = "gpt-3.5-turbo"

# Conversation flow settings
MAX_RETRIES = 3  # Maximum attempts to extract information
CONVERSATION_TIMEOUT = 600  # Seconds before conversation is reset

# Extraction confidence threshold
MIN_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score for extracted information

# States in India
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", 
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", 
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", 
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", 
    "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", 
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", 
    "Ladakh", "Lakshadweep", "Puducherry"
]

# Gender options
GENDER_OPTIONS = ["Male", "Female", "Other"] 