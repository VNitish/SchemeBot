import json
import logging
from typing import Dict, List, Any

from utils.scheme_matcher import match_schemes
from models.user_info import UserInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationService:
    """Service to recommend government schemes based on user information."""
    
    def __init__(self, schemes_file: str = "data/schemes.json"):
        """
        Initialize recommendation service.
        
        Args:
            schemes_file: Path to schemes data file with multilingual support
        """
        self.schemes_file = schemes_file
        self.schemes = self._load_schemes(self.schemes_file)
    
    def _load_schemes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load schemes data from file.
        
        Args:
            file_path: Path to schemes data file
            
        Returns:
            List of scheme dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load schemes data from {file_path}: {e}")
            return []
    
    def get_recommendations(self, user_info: UserInfo, language: str = "en") -> List[Dict[str, Any]]:
        """
        Get scheme recommendations for a user.
        
        Args:
            user_info: User information
            language: Language preference ('en' for English, 'hi' for Hindi)
            
        Returns:
            List of recommended schemes with relevance scores
        """
        if not self.schemes:
            logger.warning("No schemes data available for recommendations")
            return []
        
        # Use local matching algorithm
        recommendations = match_schemes(
            user_info=user_info.to_dict(),
            schemes=self.schemes
        )
        
        # If language is Hindi, replace English fields with Hindi fields
        if language == "hi":
            for scheme in recommendations:
                # Map of fields to replace with Hindi versions
                hindi_fields = {
                    "name": "name_hi",
                    "description": "description_hi",
                    "eligibility": "eligibility_hi",
                    "benefits": "benefits_hi",
                    "documents_required": "documents_required_hi",
                    "how_to_apply": "how_to_apply_hi",
                    "category": "category_hi",
                    "implementing_agency": "implementing_agency_hi"
                }
                
                # Replace each field with its Hindi version if available
                for eng_field, hi_field in hindi_fields.items():
                    if hi_field in scheme and scheme[hi_field]:
                        scheme[eng_field] = scheme[hi_field]
        
        logger.info(f"Found {len(recommendations)} matching schemes for user")
        return recommendations
    
    def format_recommendations(self, recommendations: List[Dict[str, Any]], language: str = "en") -> str:
        """
        Format recommendations for display.
        
        Args:
            recommendations: List of recommended schemes
            language: Language preference ('en' for English, 'hi' for Hindi)
            
        Returns:
            Formatted string for display
        """
        if not recommendations:
            if language == "hi":
                return "मुझे आपके प्रोफ़ाइल से मेल खाने वाली कोई योजना नहीं मिली। आप अधिक जानकारी के लिए सरकारी वेबसाइटों को देख सकते हैं।"
            else:
                return "I couldn't find any schemes that match your profile. You might want to check the official government websites for more information."
        
        num_schemes = len(recommendations)
        
        if language == "hi":
            result = f"आपकी जानकारी के आधार पर, मुझे {num_schemes} सरकारी योजनाएँ मिली हैं जिनके लिए आप पात्र हो सकते हैं। "
            
            if num_schemes > 5:
                result += f"मैं नीचे शीर्ष 5 मैचों को दिखा रहा हूँ, बाकी को देखने के विकल्प के साथ।"
            else:
                result += f"आप {'सभी को' if num_schemes > 1 else 'इसे'} नीचे देख सकते हैं।"
            
            result += "\n\nप्रत्येक कार्यक्रम के विवरण के लिए 'अनुशंसित योजनाएँ' अनुभाग देखें।"
        else:
            result = f"Based on your information, I've found {num_schemes} government schemes you might be eligible for. "
            
            if num_schemes > 5:
                result += f"I'm showing the top 5 matches below, with options to see the rest."
            else:
                result += f"You can see {'all of them' if num_schemes > 1 else 'it'} below."
            
            result += "\n\nCheck the 'Recommended Schemes' section for details on each program."
        
        return result
    
    def get_scheme_details(self, scheme_id: str, language: str = "en") -> Dict[str, Any]:
        """
        Get detailed information about a specific scheme.
        
        Args:
            scheme_id: ID of the scheme
            language: Language preference ('en' for English, 'hi' for Hindi)
            
        Returns:
            Scheme details
        """
        for scheme in self.schemes:
            if scheme.get('id') == scheme_id:
                return scheme
        return {} 