import re
import logging
from typing import Dict, List, Any, Optional, Union
from .constants import INDIAN_STATES_AND_UTS, GENDER_VALUES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_age_range(age_str: str) -> Dict[str, int]:
    """
    Extract minimum and maximum age from age string.
    
    Args:
        age_str: String representation of age range (e.g., "18-40 years", "Above 18 years")
        
    Returns:
        Dictionary with min and max age values
    """
    # Default values
    age_range = {"min": 0, "max": 120}
    
    if "all" in age_str.lower():
        return age_range
    
    # Extract age ranges like "18-40 years"
    range_match = re.search(r'(\d+)\s*[-–—to]\s*(\d+)', age_str)
    if range_match:
        age_range["min"] = int(range_match.group(1))
        age_range["max"] = int(range_match.group(2))
        return age_range
    
    # Extract "Above X years" or "X years and above"
    above_match = re.search(r'above\s+(\d+)|(\d+)\s+.*above', age_str.lower())
    if above_match:
        min_age = above_match.group(1) or above_match.group(2)
        age_range["min"] = int(min_age)
        return age_range
    
    # Extract "Below X years" or "X years and below"
    below_match = re.search(r'below\s+(\d+)|(\d+)\s+.*below', age_str.lower())
    if below_match:
        max_age = below_match.group(1) or below_match.group(2)
        age_range["max"] = int(max_age)
        return age_range
    
    # Extract specific age like "18 years"
    exact_match = re.search(r'(\d+)\s+years?', age_str)
    if exact_match:
        age = int(exact_match.group(1))
        age_range["min"] = age
        age_range["max"] = age
        return age_range
    
    # Catch-all for other formats like "Adult" or "Adult women"
    if "adult" in age_str.lower():
        age_range["min"] = 18
        return age_range
    
    return age_range

def parse_location_string(location_str: str) -> List[str]:
    """
    Parse location string to extract list of states/UTs.
    
    Args:
        location_str: String containing location information
        
    Returns:
        List of states/UTs
    """
    if not location_str or not isinstance(location_str, str):
        return ["All States"]
        
    location_str = location_str.strip()
    
    # Handle special cases
    if location_str == "All":
        return ["All States"]
    elif location_str == "Rural":
        return ["All States"]  # Will be handled by rural_only flag
    
    # Handle "except" cases
    if "except" in location_str.lower():
        excluded_states = []
        # Extract states after "except"
        parts = location_str.lower().split("except")
        if len(parts) > 1:
            excluded = parts[1].strip()
            # Split by commas and clean up
            excluded_states = [state.strip() for state in excluded.split(",")]
            # Match with our list of states
            excluded_states = [state for state in excluded_states if state in [s.lower() for s in INDIAN_STATES_AND_UTS]]
            # Return all states except the excluded ones
            return [state for state in INDIAN_STATES_AND_UTS if state.lower() not in excluded_states]
    
    # Handle direct state names
    states = []
    for state in INDIAN_STATES_AND_UTS:
        if state.lower() in location_str.lower():
            states.append(state)
    
    return states if states else ["All States"]

def preprocess_schemes(schemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess schemes to extract structured eligibility criteria.
    
    Args:
        schemes: Raw schemes data from JSON file
        
    Returns:
        Processed schemes with structured eligibility criteria
    """
    processed_schemes = []
    
    for scheme in schemes:
        # Create structured eligibility criteria
        eligibility_criteria = {}
        
        # Extract age criteria from target_demographics
        if "target_demographics" in scheme:
            # Use min_age and max_age directly if available
            if "min_age" in scheme["target_demographics"] and "max_age" in scheme["target_demographics"]:
                eligibility_criteria["age"] = {
                    "min": int(scheme["target_demographics"]["min_age"]),
                    "max": int(scheme["target_demographics"]["max_age"])
                }
            else:
                eligibility_criteria["age"] = {"min": 0, "max": 120}  # Default: all ages
        else:
            eligibility_criteria["age"] = {"min": 0, "max": 120}  # Default: all ages
        
        # Extract gender criteria from target_demographics
        if "target_demographics" in scheme and "gender" in scheme["target_demographics"]:
            gender_value = scheme["target_demographics"]["gender"]
            
            if isinstance(gender_value, list):
                if "All" in gender_value:
                    eligibility_criteria["gender"] = GENDER_VALUES
                else:
                    # Validate each gender value
                    eligibility_criteria["gender"] = [g for g in gender_value if g in GENDER_VALUES]
            else:
                # Default to all genders if format is incorrect
                eligibility_criteria["gender"] = GENDER_VALUES
        else:
            eligibility_criteria["gender"] = GENDER_VALUES  # Default: all genders
        
        # Extract location criteria from target_demographics
        if "target_demographics" in scheme and "location" in scheme["target_demographics"]:
            location_value = scheme["target_demographics"]["location"]
            
            if isinstance(location_value, list):
                if "All" in location_value:
                    eligibility_criteria["states"] = INDIAN_STATES_AND_UTS
                else:
                    # Validate each state value
                    eligibility_criteria["states"] = [loc for loc in location_value if loc in INDIAN_STATES_AND_UTS]
            else:
                # Default to all states if format is incorrect
                eligibility_criteria["states"] = INDIAN_STATES_AND_UTS
        else:
            eligibility_criteria["states"] = INDIAN_STATES_AND_UTS  # Default: all states
        
        # Extract income criteria from target_demographics
        if "target_demographics" in scheme and "income" in scheme["target_demographics"]:
            income_value = scheme["target_demographics"]["income"]
            
            if isinstance(income_value, list):
                eligibility_criteria["income"] = income_value
            else:
                eligibility_criteria["income"] = [income_value]
        else:
            eligibility_criteria["income"] = ["All"]  # Default: all income levels
        
        # Add structured criteria to scheme
        processed_scheme = {**scheme, "eligibility_criteria": eligibility_criteria}
        processed_schemes.append(processed_scheme)
    
    return processed_schemes

def match_schemes(user_info: Dict[str, Any], schemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Match user information with eligible schemes locally.
    
    Args:
        user_info: User information (name, gender, age, state)
        schemes: List of schemes with eligibility criteria
        
    Returns:
        List of recommended schemes with relevance scores
    """
    # Preprocess the schemes to ensure they have structured eligibility criteria
    processed_schemes = preprocess_schemes(schemes)
    matched_schemes = []
    
    try:
        # Extract user information
        user_age = user_info.get("age", 0)
        user_gender = user_info.get("gender", "")
        user_state = user_info.get("state", "")
        
        # Ensure values are in correct format
        if not isinstance(user_age, int):
            try:
                user_age = int(user_age)
            except (ValueError, TypeError):
                user_age = 0
        
        if not isinstance(user_gender, str):
            user_gender = str(user_gender)
        
        if not isinstance(user_state, str):
            user_state = str(user_state)
        
        # Match each scheme
        for scheme in processed_schemes:
            score = 0.0
            reasons = []
            
            # Get eligibility criteria
            criteria = scheme.get("eligibility_criteria", {})
            
            # Age matching
            age_criteria = criteria.get("age", {"min": 0, "max": 120})
            if age_criteria["min"] <= user_age <= age_criteria["max"]:
                # Age is within range
                age_score = 0.4
                
                # Adjust score based on how specific the targeting is
                age_range = age_criteria["max"] - age_criteria["min"]
                if age_range < 30:  # More specific targeting gets higher score
                    age_score += 0.1
                
                score += age_score
                
                # Add reason
                if age_criteria["min"] > 0 and age_criteria["max"] < 120:
                    reasons.append(f"Age {user_age} is within eligible range ({age_criteria['min']}-{age_criteria['max']})")
                elif age_criteria["min"] > 0:
                    reasons.append(f"Age {user_age} meets minimum age requirement of {age_criteria['min']}")
                elif age_criteria["max"] < 120:
                    reasons.append(f"Age {user_age} is below maximum age limit of {age_criteria['max']}")
            else:
                # Age is outside range, skip this scheme
                continue
            
            # Gender matching
            gender_criteria = criteria.get("gender", GENDER_VALUES)
            
            # Check if "All" is in gender criteria or if user's gender matches
            if "All" in gender_criteria or user_gender in gender_criteria:
                # Gender matches
                gender_score = 0.3
                
                # Adjust score if the scheme is specifically for this gender
                if len(gender_criteria) == 1 and user_gender in gender_criteria:
                    gender_score += 0.1
                
                score += gender_score
                
                # Add reason if scheme is gender-specific
                if len(gender_criteria) < 3 and user_gender in gender_criteria:
                    reasons.append(f"Scheme is designed for {user_gender.lower()} beneficiaries")
            else:
                # Gender doesn't match, skip this scheme
                continue
            
            # State matching
            state_criteria = criteria.get("states", INDIAN_STATES_AND_UTS)
            
            # Check if "All" is in state criteria or if user's state matches
            if "All" in state_criteria or "All States" in state_criteria or user_state in state_criteria:
                # State matches
                state_score = 0.3
                
                # Adjust score if scheme is state-specific
                if len(state_criteria) < len(INDIAN_STATES_AND_UTS) and user_state in state_criteria:
                    state_score += 0.1
                
                score += state_score
                
                # Add reason based on state match type
                if "All" in state_criteria or "All States" in state_criteria:
                    reasons.append("Scheme is available across all states in India")
                elif len(state_criteria) < len(INDIAN_STATES_AND_UTS) and user_state in state_criteria:
                    reasons.append(f"Scheme is available in {user_state}")
            else:
                # State doesn't match, skip this scheme
                continue
            
            # If scheme passed all filters, add it to matches
            if score > 0:
                reason_text = "; ".join(reasons) if reasons else "Your profile matches basic eligibility criteria"
                
                matched_schemes.append({
                    **scheme,
                    "relevance_score": round(score, 2),
                    "reason": reason_text
                })
    
    except Exception as e:
        logger.error(f"Error in scheme matching: {e}")
    
    # Sort by relevance score (highest first)
    return sorted(matched_schemes, key=lambda x: x.get("relevance_score", 0), reverse=True) 