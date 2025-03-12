# Thera-VL Discord Bot

This Discord bot integrates with the Thera-VL web application to allow users to create and manage virtual lab sessions through Discord.

## Features

- Create and manage virtual lab sessions
- Interact with AI scientists and collaborators
- Access session history
- Receive real-time transcripts
- Facilitate multi-agent scientific discussions

## Installation

1. Clone the repository
2. Run the installation script:

```bash
./install_dependencies.sh  # Use --venv flag to create a virtual environment
```

3. Edit the `.env` file created by the installer with your API keys

Alternatively, install manually:

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Environment Setup

The bot requires several API keys to function:

1. **Discord Configuration**:
   - `DISCORD_BOT_TOKEN` - Your Discord bot token
   - `DISCORD_GUILD_ID` - Your Discord server ID
  
2. **LLM Provider Keys** (at least one is required):
   - `OPENAI_API_KEY` - OpenAI API key
   - `ANTHROPIC_API_KEY` - Anthropic API key
   - `MISTRAL_API_KEY` - Mistral API key

3. **Backend Integration**:
   - `API_BASE_URL` - URL of the Thera-VL backend (default: http://localhost:3000/api)

## Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to the "Bot" tab and click "Add Bot"
4. Under the "Privileged Gateway Intents" section, enable:
   - Message Content Intent
   - Server Members Intent
5. Copy the bot token and add it to your `.env` file
6. Go to the "OAuth2" tab, then "URL Generator"
7. Select the following scopes:
   - bot
   - applications.commands
8. Select the following bot permissions:
   - Send Messages
   - Embed Links
   - Attach Files
   - Read Message History
   - Use Slash Commands
9. Copy the generated URL and use it to invite the bot to your server

## Running the Bot

```bash
# Run with the helper script (recommended)
./run.py

# Run directly
python main.py
```

You can also use these commands with the helper script:

```bash
# Check environment setup without running the bot
./run.py --check-env

# Check if the API connection is working properly
./run.py --check-api

# Run mock tests
./run.py --test

# Run real tests with API keys
./run.py --real-test
```

## Testing

The bot comes with comprehensive test scripts:

```bash
# Run mock tests (no API keys needed)
python test_llm_agents_mock.py

# Run real tests (requires API keys)
python test_llm_agents.py
```

See `README_TESTS.md` for more detailed information about tests.

## Available Commands

### Session Management

- `/start [title] [description] [public]` - Start a new virtual lab session
  - `title` - Title of your session
  - `description` (optional) - Brief description of your research topic or question
  - `public` (optional) - Whether the session should be public (default: false)
  
- `/end` - End your current active session

### Lab Meeting Commands

- `/lab_meeting [topic] [summary] [details]` - Start a new lab meeting with AI agents
  - `topic` - The main topic for discussion
  - `summary` (optional) - A brief summary of the topic
  - `details` (optional) - Additional details or context for the agents

Additional commands will be implemented in future updates.

## Development

The bot is structured as follows:

- `main.py` - The entry point for the bot
- `config.py` - Configuration management
- `db_client.py` - Client for interacting with the web app's API
- `commands/` - Command modules organized by functionality
  - `session_commands.py` - Commands for managing sessions
  - `lab_meeting_commands.py` - Commands for running lab meetings
  - (Additional command modules to be added)
- `models.py` - Data models and enums for LLM interactions
- `llm_client.py` - Client for interacting with LLM providers
- `orchestrator.py` - Agent orchestration for multi-agent discussions

To add new commands:

1. Create a new file in the `commands` directory or add to an existing one
2. Create a new class that inherits from `commands.Cog`
3. Add your command methods using the `@app_commands.command()` decorator
4. Register your command in the `INITIAL_EXTENSIONS` list in `main.py`

## Troubleshooting

Check the `bot.log` file for detailed error messages.

Common issues:

1. **API Key Issues**: Ensure your LLM provider API keys are correctly set in the `.env` file
2. **Discord Connection Issues**: Check your bot token and ensure the bot has the necessary permissions
3. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
4. **API Connection Issues**: 
   - Run `./run.py --check-api` to verify the API connection
   - Make sure the `API_BASE_URL` in your `.env` file is correct
   - Check for duplicated `/api` in the URL (e.g., `http://localhost:3000/api/api/`). This can cause 404 errors
   - Ensure the backend API is running and accessible

If you see errors like:
```
The API service is currently unavailable. Please try again later.
GET /api/api/discord/sessions/active?userId=... 404
```
This typically indicates a duplicated `/api` in the URL. Edit your `.env` file to fix the `API_BASE_URL` value.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Agent Variables

The bot now supports dynamic agent variables for customizing agent prompts. This feature allows for more flexible and context-specific agent responses.

### How It Works

- Agent prompts now use template variables for expertise, goal, and role
- The Principal Investigator prompt uses: `{expertise}`, `{goal}`, and `{role}`
- The Scientist prompt uses: `{agent_name}`, `{expertise}`, `{goal}`, and `{role}`
- The Scientific Critic prompt remains unchanged

### Usage

There are two ways to use agent variables:

1. **Auto-generation with GPT-4o**:
   - When using `/quickstart`, agent variables are automatically generated based on the topic
   - When using `/lab team_meeting` with `auto_generate:true`, agent variables are generated based on the agenda

2. **Manual specification**:
   - When using `/lab agent create`, you can specify expertise, goal, and role manually
   - These values will be used in the agent's prompt

### Testing

You can test the agent variables feature using the provided test script:

```bash
python test_agent_variables.py
```

This script tests:
- Generating variables for Principal Investigator and Scientist roles
- Calling agents with custom variables 

## Multi-Agent Conversation System

The bot now features an improved multi-agent conversation system that provides more dynamic and interactive discussions.

### How It Works

1. **Dynamic Orchestrator**:
   - An AI orchestrator decides which agent should speak next based on the conversation context
   - The orchestrator analyzes the conversation history and selects the most appropriate agent to contribute next
   - This creates a more natural flow than a fixed round-robin approach

2. **Round-Based Structure**:
   - Conversations are organized into rounds (default: 3)
   - Each round includes multiple agent contributions
   - At the end of each round, the Principal Investigator synthesizes the key points and asks follow-up questions
   - After all rounds, a summary agent provides a final summary

3. **Discord Integration**:
   - All agent responses are sent to the Discord channel in real-time (when live_mode is enabled)
   - Messages are formatted with agent names and round indicators
   - The conversation is updated in the same message, creating a flowing discussion thread

4. **Transcript Creation**:
   - Each agent response is automatically saved as a transcript in the database
   - Transcripts can be viewed later using `/lab transcript view`

### Commands

Use these commands to start and manage multi-agent conversations:

- `/quickstart [topic]` - Create a session with agents and start a discussion on the specified topic
- `/lab team_meeting [agenda]` - Start a team meeting with existing agents in the current session
- `/lab end_team_meeting` - End an ongoing team meeting 