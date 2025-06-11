#!/bin/bash

echo "Setting up SchemeBot..."

# Activate virtual environment if it exists or create a new one
if [ -d "myenv" ]; then
    echo "Activating existing virtual environment..."
    source myenv/bin/activate
else
    echo "Creating new virtual environment..."
    python -m venv myenv
    source myenv/bin/activate
fi

# Install dependencies
echo "Installing required packages..."
pip install -r requirements.txt

# Display information about Whisper model
echo "Note: The first time you run the app, it will download the Whisper large model (~3GB)."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "# Add your OpenAI API key below" > .env
    echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env
    echo "Created .env file. Please edit it to add your OpenAI API key."
fi

echo "Setup complete!"
echo "To run SchemeBot: streamlit run app.py" 