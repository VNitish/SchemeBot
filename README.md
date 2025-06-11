# SchemeBot 🇮🇳

SchemeBot is an intelligent chatbot designed to help Indian citizens find government schemes they may be eligible for. Using natural language processing and OpenAI's language models, it engages in natural conversations to gather user information and recommend relevant government schemes tailored to their profile.

## Features

- **Conversational Interface**: Natural dialogue to collect user information in a friendly manner
- **Information Extraction**: Automatically extracts name, gender, age, and state from user responses
- **Smart Validation**: Validates and normalizes extracted information (e.g., converts "boy" to "Male", corrects misspelled state names)
- **Efficient Local Matching**: Uses a local matching algorithm to find relevant schemes without API calls
- **Personalized Recommendations**: Recommends government schemes based on user's demographic information
- **User-Friendly UI**: Clean Streamlit interface with chat bubbles and expandable scheme details
- **Speech-to-Text**: Voice input capability using OpenAI's Whisper model running locally, supporting both English and Hindi

## Project Structure

```
llmSchemeBot/
├── app.py                 # Main Streamlit application
├── config/
│   └── config.py          # Configuration variables (API keys, model settings)
├── utils/
│   ├── openai_client.py   # OpenAI API integration
│   ├── conversation.py    # Conversation state management
│   ├── validator.py       # Information validation utilities
│   └── scheme_matcher.py  # Local scheme matching algorithm
├── data/
│   └── schemes.json       # Government schemes database
├── models/
│   └── user_info.py       # User information model
├── services/
│   ├── extraction.py      # Information extraction from conversations
│   ├── recommendation.py  # Scheme recommendation engine
│   └── conversation_flow.py # Manages conversation flow
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llmSchemeBot.git
   cd llmSchemeBot
   ```

2. Create a virtual environment:
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL shown in the console (typically http://localhost:8501)

3. Start chatting with SchemeBot to discover government schemes you might be eligible for!

4. Use the voice input feature by clicking the microphone button (🎙️) to speak your responses.

## How It Works

SchemeBot follows these steps to help users find relevant government schemes:

1. **Information Collection**: The bot engages in a natural conversation to collect the user's:
   - Name
   - Gender
   - Age
   - State

2. **Information Validation**: Each piece of information is:
   - Extracted using OpenAI's LLM
   - Validated and normalized using rule-based checks
   - Corrected if needed (e.g., fixing misspelled state names)

3. **Scheme Matching**: The local matching algorithm:
   - Preprocesses scheme eligibility criteria into structured format
   - Matches user demographics against eligibility requirements
   - Scores schemes based on relevance to the user
   - Provides explanations for why each scheme matches

4. **Recommendation Display**: Matching schemes are presented to the user with:
   - Relevance score
   - Explanation of why the scheme is recommended
   - Details about benefits and how to apply

### Speech-to-Text Feature

The voice input capability uses OpenAI's Whisper large model running locally:

1. **Recording**: Click the microphone button (🎙️) to start recording your voice
2. **Transcription**: After stopping the recording, Whisper processes your speech
3. **Language Support**: Automatically works with both English and Hindi
4. **Integration**: Transcribed text is seamlessly processed just like typed input

The speech recognition system:
- Runs completely locally on your machine
- Uses Whisper large model for high-quality transcription
- Optimizes for Mac M-series chips with MPS acceleration when available
- Supports recordings up to 30 seconds in length

### Matching Algorithm

The matching algorithm works by:

1. Converting textual eligibility criteria from schemes.json into structured format:
   - Age ranges (min-max)
   - Gender requirements
   - Geographic restrictions

2. Filtering out schemes where the user doesn't meet basic requirements

3. Scoring remaining schemes based on:
   - How well the user's age matches (40% of score)
   - Gender match (30% of score)
   - State match (30% of score)
   - Additional bonuses for schemes specifically targeting the user's demographic

This approach allows for efficient matching without requiring API calls for recommendations, making the system more scalable and responsive.

## Conversation Flow

SchemeBot follows a systematic approach to gather user information:

1. **Name**: First asks for the user's name
2. **Gender**: Then asks for the user's gender
3. **Age**: Next asks for the user's age
4. **State**: Finally asks which state in India the user lives in

After collecting all necessary information, SchemeBot recommends relevant government schemes based on the user's profile.

## Example Conversation

```
Bot: Hello! I'm SchemeBot. How can I help you today?

User: Hi

Bot: Can you tell me your name?

User: Ravi Jangir

Bot: Nice to meet you, Ravi! Can you tell me your gender?

User: Male

Bot: What is your age?

User: 25

Bot: Which state in India do you live in?

User: Karnataka

Bot: Based on your information, here are schemes you might be eligible for...
```

## Technology Stack

- **Streamlit**: For the web interface
- **OpenAI API**: For natural language understanding and generation
- **Python**: Core programming language
- **JSON**: For storing government scheme data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

SchemeBot's recommendations are suggestive and should not be considered official guidance. Always verify eligibility criteria through official government sources. 