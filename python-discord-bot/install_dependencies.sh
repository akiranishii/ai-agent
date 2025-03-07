#!/bin/bash

echo "Installing TheraBot dependencies..."

# Create and activate virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    
    # Activate virtual environment based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    echo "Virtual environment activated"
fi

# Install dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Check if dotenv exists
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env file with your API keys"
else
    echo ".env file already exists"
fi

echo "Installation complete!"
echo ""
echo "To run tests:"
echo "  python test_llm_agents_mock.py    # Mock tests (no API keys required)"
echo "  python test_llm_agents.py         # Real tests (API keys required)"
echo ""
echo "To run the bot:"
echo "  python main.py" 