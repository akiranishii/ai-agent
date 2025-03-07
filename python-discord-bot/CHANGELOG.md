# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dynamic agent variables for customizing agent prompts
- Auto-generation of agent variables using GPT-4o based on topic/agenda
- Support for specifying expertise, goal, and role for Principal Investigator and Scientist agents
- Test script for verifying agent variables functionality
- Added README_TESTS.md with testing documentation
- Added API troubleshooting section to README.md
- Dynamic orchestrator that decides which agent speaks next in conversations
- Round-based conversation structure with PI synthesis at the end of each round
- Real-time Discord channel updates for multi-agent conversations
- Automatic transcript creation for all agent responses
- Enhanced conversation flow with interactive agent selection

### Changed
- Updated `llm_client.py` to properly initialize providers dictionary and support agent variables
- Improved error handling in LLM client
- Enhanced configuration loading in `config.py`
- Better logging throughout the application
- Enhanced `DatabaseClient` to prevent API URL path duplication
- Refactored `orchestrator.py` to use the new conversation system
- Updated README.md with documentation for the new multi-agent system

### Fixed
- Fixed 'LLMClient' object has no attribute 'providers' error
- Improved error handling when API keys are missing
- Added graceful fallback to mock implementations when needed
- Fixed API URL duplication issue causing 404 errors
- Added detection and prevention of duplicate `/api` segments in URLs

## [Unreleased] - 2025-03-03

### Added
- Environment variable loading from `.env` file using python-dotenv
- Support for multiple LLM providers (OpenAI, Anthropic, Mistral)
- Comprehensive test scripts for LLM functionality
  - `test_llm_agents.py` - Real API tests with fallback to mocks
  - `test_llm_agents_mock.py` - Pure mock tests
- Installation script `install_dependencies.sh` for easy setup
- Runner script `run.py` with environment validation and test options
- API connection validation with `--check-api` option in run.py
- Detailed documentation in README files
  - Updated main README.md with new setup instructions
  - Added README_TESTS.md with testing documentation
  - Added API troubleshooting section to README.md

### Changed
- Updated `llm_client.py`