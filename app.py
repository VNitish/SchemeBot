import os
import time
import streamlit as st
import json
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import whisper
import torch

from models.user_info import UserInfo
from services.conversation_flow import ConversationFlow
from services.recommendation import RecommendationService
from utils.openai_client import OpenAIClient
from utils.conversation import ConversationState
from utils.language_utils import detect_language, language_service
from utils.state_manager import state_manager
from config.config import OPENAI_API_KEY

# Configure page
st.set_page_config(
    page_title="SchemeBot - Find Indian Government Schemes",
    page_icon="🇮🇳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Check if OpenAI API key is present
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF9933;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #138808;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-bubble {
        background-color: #F0F2F6;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        display: inline-block;
        max-width: 80%;
        color: #333333;
    }
    .bot-bubble {
        background-color: #005C97;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        display: inline-block;
        max-width: 80%;
        color: white;
        line-height: 1.5;
        white-space: normal;
        word-wrap: break-word;
        text-align: left;
    }
    .scheme-card {
        background-color: #FFFFFF;
        border: 1px solid #DDDDDD;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .scheme-title {
        font-weight: bold;
        color: #000080;
        margin-bottom: 5px;
    }
    .scheme-description {
        font-size: 0.9rem;
        color: #333333;
        margin-bottom: 5px;
    }
    .scheme-eligibility {
        font-size: 0.85rem;
        color: #666666;
        margin-bottom: 5px;
    }
    .scheme-reason {
        font-size: 0.85rem;
        color: #006400;
        margin-bottom: 5px;
    }
    .scheme-link {
        font-size: 0.8rem;
        color: #0000EE;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        background-color: #1E1E1E;
        border-radius: 10px;
        border: 1px solid #333333;
        margin-bottom: 10px;
    }
    .language-toggle {
        text-align: right;
        margin-bottom: 10px;
    }
    /* Style for the recording toggle button */
    [data-testid="baseButton-primary"] {
        border-radius: 50% !important;
        width: 48px !important;
        height: 48px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
    }
    /* Record button (mic icon) */
    [data-testid="baseButton-primary"]:has(div:contains("🎙️")) {
        background-color: #FF9933 !important;
        color: white !important;
    }
    [data-testid="baseButton-primary"]:has(div:contains("🎙️")):hover {
        background-color: #FF8C00 !important;
    }
    /* Stop button (stop icon) */
    [data-testid="baseButton-primary"]:has(div:contains("⏹️")) {
        background-color: #FF0000 !important;
        color: white !important;
    }
    [data-testid="baseButton-primary"]:has(div:contains("⏹️")):hover {
        background-color: #CC0000 !important;
    }
    .recording-in-progress {
        animation: pulse 1.5s infinite;
        color: #FF0000;
        font-weight: bold;
        margin-top: 5px;
    }
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            opacity: 1;
        }
    }
    .stButton > button {
        border: none !important;
        box-shadow: none !important;
    }
    /* Error message styling */
    .stAlert {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_DURATION = 30  # seconds

# Load Whisper model (cached)
@st.cache_resource
def load_whisper_model():
    with st.spinner("Loading Whisper model..."):
        # Force CPU usage to avoid sparse tensor errors on MPS/Metal
        device = "cpu"
        model = whisper.load_model("large", device=device)
    return model

def initialize_session_state():
    """Initialize session state variables."""
    if "conversation_flow" not in st.session_state:
        st.session_state.conversation_flow = ConversationFlow()
        # Generate initial greeting and question
        language = "en"  # Default language
        initial_greeting = st.session_state.conversation_flow.process_user_message("hello", language)
        st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]
    
    if "recommendation_service" not in st.session_state:
        st.session_state.recommendation_service = RecommendationService()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    
    if "schemes_expanded" not in st.session_state:
        st.session_state.schemes_expanded = {}
        
    if "language" not in st.session_state:
        st.session_state.language = "en"
        
    if "auto_detect_language" not in st.session_state:
        st.session_state.auto_detect_language = False
    
    # Audio recording session state variables
    if "whisper_model" not in st.session_state:
        st.session_state.whisper_model = None
    
    if "recording" not in st.session_state:
        st.session_state.recording = False
    
    if "recorded_frames" not in st.session_state:
        st.session_state.recorded_frames = None
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    
    if "audio_file" not in st.session_state:
        st.session_state.audio_file = None
        
    if "transcription" not in st.session_state:
        st.session_state.transcription = ""
        
    if "audio_device_info" not in st.session_state:
        st.session_state.audio_device_info = check_audio_devices()

def check_audio_devices():
    """Check available audio devices and return information about them."""
    try:
        # Get list of available devices
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        
        # Return device info
        return {
            "devices_available": True,
            "default_input": default_input.get('name', 'Unknown'),
            "input_devices": [d['name'] for d in devices if d.get('max_input_channels', 0) > 0],
            "error": None
        }
    except Exception as e:
        return {
            "devices_available": False,
            "default_input": None,
            "input_devices": [],
            "error": str(e)
        }

def start_recording():
    """Start audio recording with proper error handling."""
    try:
        # Get default input device
        device_info = sd.query_devices(kind='input')
        device_idx = device_info['index'] if 'index' in device_info else None
        
        # Start recording with explicit device selection
        total_frames = int(SAMPLE_RATE * MAX_DURATION)
        frames = sd.rec(
            total_frames, 
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype='float32',
            device=device_idx  # Explicitly use default input device
        )
        return frames, None
    except Exception as e:
        return None, str(e)

def reset_conversation():
    """Reset the conversation state."""
    st.session_state.conversation_flow = ConversationFlow()
    
    # Generate initial greeting and question
    language = st.session_state.language  # Preserve current language
    initial_greeting = st.session_state.conversation_flow.process_user_message("hello", language)
    
    # Reset other states
    st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]
    st.session_state.recommendations = []
    st.session_state.schemes_expanded = {}
    
    # Reset audio recording states
    st.session_state.recording = False
    st.session_state.recorded_frames = None
    st.session_state.start_time = None
    st.session_state.audio_file = None
    st.session_state.transcription = ""

def handle_user_input():
    """Process user input and update conversation."""
    if st.session_state.user_input and st.session_state.user_input.strip():
        user_message = st.session_state.user_input.strip()
        
        # Auto-detect language if enabled
        if st.session_state.auto_detect_language:
            detected_language = detect_language(user_message)
            if detected_language != st.session_state.language:
                st.session_state.language = detected_language
        
        # Add user message to UI messages
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Process user message with language preference
        with st.spinner("Thinking..."):
            bot_response = st.session_state.conversation_flow.process_user_message(
                user_message, 
                language=st.session_state.language
            )
        
        # Add bot response to UI messages
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Get recommendations if the conversation is in the completed state
        # or if the bot response contains recommendation indicator text
        if (st.session_state.conversation_flow.conversation.get_state() == ConversationState.COMPLETED or 
            "schemes you might be eligible for" in bot_response or 
            "सरकारी योजनाएँ" in bot_response) and not st.session_state.recommendations:
            # Get scheme recommendations for display
            user_info = st.session_state.conversation_flow.get_user_info()
            if user_info.is_complete():
                st.session_state.recommendations = st.session_state.recommendation_service.get_recommendations(
                    user_info,
                    language=st.session_state.language
                )
                # Reset pagination to first page
                st.session_state.page_num = 0
        
        # Clear input
        st.session_state.user_input = ""

def display_chat():
    """Display chat messages."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div style="text-align: right;"><div class="user-bubble">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            # Skip displaying recommendation listings in chat bubbles
            # If this contains a recommendation listing, summarize it
            content = message["content"]
            if (("Based on your information, I've found" in content and "schemes you might be eligible for" in content) or
                ("आपकी जानकारी के आधार पर" in content and "सरकारी योजनाएँ" in content)):
                
                if st.session_state.language == "hi":
                    content = "मैंने कुछ योजनाएँ खोजी हैं जिनके लिए आप पात्र हो सकते हैं। कृपया नीचे 'अनुशंसित योजनाएँ' अनुभाग देखें।"
                else:
                    content = "I've found some schemes you might be eligible for. Please check the 'Recommended Schemes' section below."
            
            # Replace newlines with HTML breaks to preserve formatting
            formatted_content = content.replace('\n', '<br>')
            st.markdown(f'<div style="text-align: left;"><div class="bot-bubble">{formatted_content}</div></div>', unsafe_allow_html=True)
            
            # Check if this is a recommendation message and we don't have recommendations loaded yet
            if (("schemes you might be eligible for" in message["content"]) or 
                ("सरकारी योजनाएँ" in message["content"])) and not st.session_state.recommendations:
                user_info = st.session_state.conversation_flow.get_user_info()
                if user_info.is_complete():
                    st.session_state.recommendations = st.session_state.recommendation_service.get_recommendations(
                        user_info,
                        language=st.session_state.language
                    )
                    # Initialize page number to 0 (first page)
                    if "page_num" not in st.session_state:
                        st.session_state.page_num = 0

def display_recommendations():
    """Display scheme recommendations with pagination."""
    if st.session_state.recommendations:
        if st.session_state.language == "hi":
            st.markdown("### अनुशंसित योजनाएँ")
        else:
            st.markdown("### Recommended Schemes")
        
        # Initialize pagination state if not exists
        if "page_num" not in st.session_state:
            st.session_state.page_num = 0
            
        # Calculate total number of pages
        total_schemes = len(st.session_state.recommendations)
        schemes_per_page = 5
        total_pages = (total_schemes + schemes_per_page - 1) // schemes_per_page
        
        # Get current page schemes
        start_idx = st.session_state.page_num * schemes_per_page
        end_idx = min(start_idx + schemes_per_page, total_schemes)
        current_schemes = st.session_state.recommendations[start_idx:end_idx]
        
        # Display current page schemes
        for scheme in current_schemes:
            scheme_name = scheme["name"]
            with st.expander(scheme_name, expanded=st.session_state.schemes_expanded.get(scheme["id"], False)):
                # Display information in the appropriate language
                st.markdown(f'**{st.session_state.language == "hi" and "विवरण:" or "Description:"}** {scheme["description"]}')
                st.markdown(f'**{st.session_state.language == "hi" and "पात्रता:" or "Eligibility:"}** {scheme["eligibility"]}')
                
                benefits_label = st.session_state.language == "hi" and "लाभ:" or "Benefits:"
                st.markdown(f'**{benefits_label}**')
                for benefit in scheme["benefits"]:
                    st.markdown(f"- {benefit}")
                
                if "reason" in scheme:
                    reason_label = st.session_state.language == "hi" and "यह आपकी प्रोफ़ाइल से क्यों मेल खाता है:" or "Why it matches your profile:"
                    st.markdown(f'**{reason_label}** {scheme["reason"]}')
                
                apply_label = st.session_state.language == "hi" and "आवेदन कैसे करें:" or "How to apply:"
                st.markdown(f'**{apply_label}** {scheme["how_to_apply"]}')
                
                docs_label = st.session_state.language == "hi" and "आवश्यक दस्तावेज़:" or "Required documents:"
                st.markdown(f'**{docs_label}**')
                for doc in scheme["documents_required"]:
                    st.markdown(f"- {doc}")
                
                info_label = st.session_state.language == "hi" and "अधिक जानकारी:" or "More information:"
                st.markdown(f'**{info_label}** [Official Website]({scheme["link"]})')
                
                # Add scheme category and implementing agency
                category_label = st.session_state.language == "hi" and "श्रेणी:" or "Category:"
                implementing_agency_label = st.session_state.language == "hi" and "कार्यान्वयन एजेंसी:" or "Implementing Agency:"
                
                st.markdown(f'**{category_label}** {scheme["category"]}')
                st.markdown(f'**{implementing_agency_label}** {scheme["implementing_agency"]}')
        
        # Add pagination controls
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.session_state.page_num > 0:
                prev_label = st.session_state.language == "hi" and "← पिछला" or "← Previous"
                if st.button(prev_label):
                    st.session_state.page_num -= 1
                    st.experimental_rerun()
        
        with col2:
            page_text = st.session_state.language == "hi" and f"पृष्ठ {st.session_state.page_num + 1} / {total_pages}" or f"Page {st.session_state.page_num + 1} of {total_pages}"
            st.write(page_text)
        
        with col3:
            if st.session_state.page_num < total_pages - 1:
                next_label = st.session_state.language == "hi" and "अगला →" or "Next →"
                if st.button(next_label):
                    st.session_state.page_num += 1
                    st.experimental_rerun()

def language_selector():
    """Display language selection UI."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Language selection
            selected_language = st.radio(
                "भाषा / Language",
                options=["English", "हिंदी"],
                horizontal=True,
                index=0 if st.session_state.language == "en" else 1
            )
            
            # Auto-detect toggle with language-specific label
            auto_detect_label = "Auto-detect language" if st.session_state.language == "en" else "भाषा स्वतः पहचानें"
            st.checkbox(
                auto_detect_label,
                value=st.session_state.auto_detect_language,
                key="auto_detect_language_toggle",
                help="Automatically detect language from your input"
            )
            
            # Update session state
            if selected_language == "English" and st.session_state.language != "en":
                st.session_state.language = "en"
                # Regenerate initial greeting if it's the only message
                if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
                    initial_greeting = st.session_state.conversation_flow.process_user_message("hello", "en")
                    st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]
                st.experimental_rerun()
            elif selected_language == "हिंदी" and st.session_state.language != "hi":
                st.session_state.language = "hi"
                # Regenerate initial greeting if it's the only message
                if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
                    initial_greeting = st.session_state.conversation_flow.process_user_message("hello", "hi")
                    st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]
                st.experimental_rerun()
            
            # Update auto-detect setting
            st.session_state.auto_detect_language = st.session_state.auto_detect_language_toggle

def save_audio_to_file(frames):
    """Save recorded audio frames to a temporary WAV file."""
    if frames is not None and len(frames) > 0:
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, frames, SAMPLE_RATE)
        return temp_file.name
    return None

def transcribe_audio(audio_file, model, language=None):
    """Transcribe audio using Whisper model."""
    with st.spinner("Transcribing... This may take a moment."):
        # Create transcription options
        options = {
            "task": "transcribe",
            "fp16": False  # Use full precision to avoid MPS issues
        }
        
        # Add language parameter if specified
        if language:
            options["language"] = language
            
            # For Hindi specifically, add some guidance
            if language == "hi":
                options["initial_prompt"] = "यह हिंदी में है"  # "This is in Hindi"
        
        # Call transcribe with all options unpacked
        result = model.transcribe(audio_file, **options)
        
    return result["text"]

def handle_voice_input():
    """Process voice input and update conversation."""
    if st.session_state.transcription:
        user_message = st.session_state.transcription.strip()
        
        # Auto-detect language if enabled
        if st.session_state.auto_detect_language:
            detected_language = detect_language(user_message)
            if detected_language != st.session_state.language:
                st.session_state.language = detected_language
        
        # Add user message to UI messages
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # Process user message with language preference
        with st.spinner("Thinking..."):
            bot_response = st.session_state.conversation_flow.process_user_message(
                user_message, 
                language=st.session_state.language
            )
        
        # Add bot response to UI messages
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Get recommendations if the conversation is in the completed state
        # or if the bot response contains recommendation indicator text
        if (st.session_state.conversation_flow.conversation.get_state() == ConversationState.COMPLETED or 
            "schemes you might be eligible for" in bot_response or 
            "सरकारी योजनाएँ" in bot_response) and not st.session_state.recommendations:
            # Get scheme recommendations for display
            user_info = st.session_state.conversation_flow.get_user_info()
            if user_info.is_complete():
                st.session_state.recommendations = st.session_state.recommendation_service.get_recommendations(
                    user_info,
                    language=st.session_state.language
                )
                # Reset pagination to first page
                st.session_state.page_num = 0
        
        # Clear input
        st.session_state.transcription = ""
        st.session_state.audio_file = None

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Load Whisper model if not already loaded
    if st.session_state.whisper_model is None:
        st.session_state.whisper_model = load_whisper_model()
    
    # Page header
    st.markdown('<h1 class="main-header">SchemeBot 🇮🇳</h1>', unsafe_allow_html=True)
    
    if st.session_state.language == "hi":
        sub_header = '<p class="sub-header">आप जिन सरकारी योजनाओं के पात्र हो सकते हैं, उन्हें खोजें</p>'
    else:
        sub_header = '<p class="sub-header">Find government schemes you may be eligible for</p>'
    
    st.markdown(sub_header, unsafe_allow_html=True)
    
    # Language selector
    language_selector()
    
    # Sidebar
    with st.sidebar:
        if st.session_state.language == "hi":
            st.markdown("### स्कीमबॉट के बारे में")
            st.markdown("""
            स्कीमबॉट भारतीय नागरिकों को उन सरकारी योजनाओं को खोजने में मदद करता है जिनके लिए वे पात्र हो सकते हैं।
            
            बस बॉट के साथ बातचीत करें, कुछ सवालों के जवाब दें, और व्यक्तिगत योजना सिफारिशें प्राप्त करें।
            """)
            
            st.markdown("### यह कैसे काम करता है")
            st.markdown("""
            1. स्कीमबॉट को अपने बारे में बताएं
            2. बॉट आपका नाम, लिंग, उम्र और राज्य पूछेगा
            3. आपकी जानकारी के आधार पर, यह प्रासंगिक सरकारी योजनाओं की सिफारिश करेगा
            4. 🎙️ आप अब बोलकर भी जवाब दे सकते हैं
            """)
            
            if st.button("बातचीत रीसेट करें"):
                reset_conversation()
                st.experimental_rerun()
                
            # Audio device info
            st.markdown("### ऑडियो डिवाइस की जानकारी")
            if st.session_state.audio_device_info["devices_available"]:
                st.success(f"डिफ़ॉल्ट इनपुट: {st.session_state.audio_device_info['default_input']}")
                with st.expander("उपलब्ध इनपुट डिवाइस"):
                    for device in st.session_state.audio_device_info["input_devices"]:
                        st.write(f"- {device}")
            else:
                st.error(f"ऑडियो डिवाइस एरर: {st.session_state.audio_device_info['error']}")
                st.warning("कृपया अपने माइक्रोफोन की अनुमतियों की जांच करें और सुनिश्चित करें कि यह ठीक से कनेक्ट किया गया है।")
        else:
            st.markdown("### About SchemeBot")
            st.markdown("""
            SchemeBot helps Indian citizens find government schemes they may be eligible for.
            
            Just have a conversation with the bot, answer a few questions, and get personalized scheme recommendations.
            """)
            
            st.markdown("### How it works")
            st.markdown("""
            1. Tell SchemeBot about yourself
            2. The bot will ask for your name, gender, age, and state
            3. Based on your information, it will recommend relevant government schemes
            4. 🎙️ You can now also speak your answers
            """)
            
            if st.button("Reset Conversation"):
                reset_conversation()
                st.experimental_rerun()
                
            # Audio device info
            st.markdown("### Audio Device Information")
            if st.session_state.audio_device_info["devices_available"]:
                st.success(f"Default input: {st.session_state.audio_device_info['default_input']}")
                with st.expander("Available input devices"):
                    for device in st.session_state.audio_device_info["input_devices"]:
                        st.write(f"- {device}")
            else:
                st.error(f"Audio device error: {st.session_state.audio_device_info['error']}")
                st.warning("Please check your microphone permissions and make sure it's properly connected.")
    
    # Chat interface
    if st.session_state.language == "hi":
        st.markdown("### स्कीमबॉट से चैट करें")
    else:
        st.markdown("### Chat with SchemeBot")
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        display_chat()
    
    # User input section with text and voice options
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Text input
        if st.session_state.language == "hi":
            placeholder_text = "अपना संदेश यहां टाइप करें"
        else:
            placeholder_text = "Type your message"
            
        st.text_input(placeholder_text, key="user_input", on_change=handle_user_input)
    
    with col2:
        # Voice input controls - single toggle button
        button_text = "⏹️" if st.session_state.recording else "🎙️"
        button_help = "Stop recording" if st.session_state.recording else "Start voice recording"
        
        if st.button(button_text, key="toggle_recording", help=button_help, 
                   use_container_width=True, 
                   type="primary"):
            
            if st.session_state.recording:
                # Stop recording logic
                try:
                    sd.stop()
                    
                    # Calculate how many frames to keep based on elapsed time
                    elapsed_time = time.time() - st.session_state.start_time
                    frames_to_keep = min(
                        int(SAMPLE_RATE * elapsed_time),
                        len(st.session_state.recorded_frames)
                    )
                    
                    # Trim the array to keep only the frames we recorded
                    audio_data = st.session_state.recorded_frames[:frames_to_keep]
                    
                    # Save the audio file
                    audio_file_path = save_audio_to_file(audio_data)
                    st.session_state.audio_file = audio_file_path
                    
                    # Reset recording state
                    st.session_state.recording = False
                    
                    # Process the audio file with Whisper
                    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                        # Get current language from session state
                        current_language = st.session_state.language
                        
                        # Perform transcription with the selected language
                        with st.spinner("Transcribing your speech..."):
                            st.session_state.transcription = transcribe_audio(
                                st.session_state.audio_file, 
                                st.session_state.whisper_model,
                                language=current_language
                            )
                        
                        # Process the transcribed text
                        handle_voice_input()
                except Exception as e:
                    st.error(f"Error stopping recording: {str(e)}")
                    st.session_state.recording = False
            else:
                # Start recording logic
                try:
                    # Start a new recording session
                    st.session_state.recording = True
                    st.session_state.start_time = time.time()
                    
                    # Start recording using our helper function
                    frames, error = start_recording()
                    
                    if error:
                        st.error(f"Could not access microphone: {error}")
                        st.error("Please check your microphone permissions and make sure it's properly connected.")
                        st.session_state.recording = False
                    else:
                        st.session_state.recorded_frames = frames
                except Exception as e:
                    st.error(f"Could not access microphone: {str(e)}")
                    st.error("Please check your microphone permissions and make sure it's properly connected.")
                    st.session_state.recording = False
            
            st.experimental_rerun()
    
    # Show recording status
    if st.session_state.recording:
        elapsed_time = time.time() - st.session_state.start_time
        time_left = MAX_DURATION - elapsed_time
        
        # Progress bar
        progress = min(elapsed_time / MAX_DURATION, 1.0)
        st.progress(progress)
        
        if st.session_state.language == "hi":
            st.markdown("<div class='recording-in-progress'>🔴 रिकॉर्डिंग... ({}/{}s)</div>".format(int(elapsed_time), MAX_DURATION), unsafe_allow_html=True)
        else:
            st.markdown("<div class='recording-in-progress'>🔴 Recording... ({}/{}s)</div>".format(int(elapsed_time), MAX_DURATION), unsafe_allow_html=True)
    
    # Check for recommendations in message content
    if not st.session_state.recommendations and st.session_state.messages:
        for message in st.session_state.messages:
            recommendation_phrases = [
                "schemes you might be eligible for", 
                "सरकारी योजनाएँ"
            ]
            if message["role"] == "assistant" and any(phrase in message["content"] for phrase in recommendation_phrases):
                user_info = st.session_state.conversation_flow.get_user_info()
                if user_info.is_complete():
                    st.session_state.recommendations = st.session_state.recommendation_service.get_recommendations(
                        user_info,
                        language=st.session_state.language
                    )
                break
    
    # Recommendations (if available)
    if st.session_state.recommendations:
        display_recommendations()
    
    # Send greeting message if conversation is empty
    if not st.session_state.messages:
        with st.spinner("Starting conversation..."):
            greeting = st.session_state.conversation_flow._handle_greeting(language=st.session_state.language)
            st.session_state.messages.append({"role": "assistant", "content": greeting})
            st.experimental_rerun()

if __name__ == "__main__":
    main() 