# TheraLab Discord Bot Testing

This directory contains test scripts for the TheraLab Discord bot, focusing on testing LLM providers and agent discussions.

## Setting Up Environment Variables

The bot and test scripts now use a `.env` file to load API keys and other configuration options:

1. Copy the `.env.example` file to create a new `.env` file:
   ```
   cp .env.example .env
   ```

2. Edit the `.env` file and replace the placeholder values with your actual API keys:
   ```
   # LLM Provider API Keys
   OPENAI_API_KEY=your_actual_openai_key
   ANTHROPIC_API_KEY=your_actual_anthropic_key
   MISTRAL_API_KEY=your_actual_mistral_key
   ```

3. Ensure you have the `python-dotenv` package installed:
   ```
   pip install python-dotenv
   ```

## Test Files

- `test_llm_agents.py` - Tests real LLM API calls and agent discussions when API keys are available, with graceful fallback to mocks when needed
- `test_llm_agents_mock.py` - Pure mock version that simulates responses without ever making actual API calls

## Running the Tests

### Mock Tests (No API Keys Required)

To run the pure mock tests, which don't require any API keys:

```bash
python test_llm_agents_mock.py
```

This will simulate:
1. Testing all LLM providers (OpenAI, Anthropic, Mistral) with mock responses
2. Testing the agent discussion functionality with mock agents
3. Testing the combined summary generation for parallel meetings

### Hybrid Tests (Works With or Without API Keys)

The main test script can work in two modes:

```bash
# Run with fallback to mocks when needed
python test_llm_agents.py
```

This script will:
1. Try to use the real LLMClient, but fall back to a mock client if API keys are missing
2. Use the real AgentOrchestrator if available
3. Skip tests for providers that aren't available

### Real Tests (API Keys Required)

For full testing with real API calls, make sure your API keys are correctly configured in the `.env` file:

```bash
# Run the tests with API keys from .env file
python test_llm_agents.py
```

You can also override the environment variables from the command line if needed:

```bash
# Override the API keys from .env
OPENAI_API_KEY=your_openai_key python test_llm_agents.py
```

## Test Output

The tests will provide detailed logging output, showing:

1. For each LLM provider:
   - The response content
   - The model used
   - Token usage statistics

2. For agent discussions:
   - Agent initialization
   - The conversation flow
   - Individual agent responses

3. For combined summaries:
   - How multiple meeting summaries are combined into one cohesive summary

## Troubleshooting

If you encounter errors with the real tests:

1. Check that your `.env` file exists and contains valid API keys
2. Verify the file permissions on your `.env` file
3. Check your internet connection
4. Verify that you have the required Python packages installed:
   ```
   pip install python-dotenv openai anthropic mistralai
   ```
5. If specific providers fail, you can comment out their tests in the `test_providers` method

## Adding New Tests

To add new tests:

1. Add a new method to the `LLMTester` class
2. Call your new method from the `main` function
3. Update the success criteria in the results section

## Implementation Details

### Mock Implementation

The mock implementation provides:

1. A `MockLLMClient` class that mimics the behavior of the real LLMClient
2. Realistic-looking but fixed responses for each provider
3. Simulated delays to mimic real API calls

### Hybrid Approach

The `test_llm_agents.py` script uses a hybrid approach:

1. It attempts to import and use the real LLMClient
2. If that fails (e.g., due to missing API keys), it falls back to the MockLLMClient
3. This allows the same test script to work in environments with or without API keys

### Real Testing

When running with proper API keys, the tests:

1. Make actual API calls to the LLM providers
2. Use the real AgentOrchestrator to simulate agent discussions
3. Provide genuine responses based on the provided prompts

## Using in Development

These tests are useful for:

1. Verifying that the LLM clients are functioning properly
2. Testing agent orchestration without involving the Discord API
3. Debugging issues with multi-agent conversations
4. Testing new features before integrating them with Discord 