import os
import logging
from dotenv import load_dotenv
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
def load_environment():
    """Load environment variables from .env files and YAML"""
    logger.info("Loading environment variables")
    
    # Current directory for the bot
    current_dir = Path(__file__).parent
    
    # Try loading from .env file in current directory first (preferred method)
    env_path = current_dir / ".env"
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    
    # Project root .env.local for backward compatibility
    project_root = current_dir.parent
    env_local_path = project_root / ".env.local"
    if env_local_path.exists():
        logger.info(f"Loading environment variables from {env_local_path}")
        load_dotenv(dotenv_path=env_local_path)
    
    # YAML config for backward compatibility
    local_env_path = current_dir / "local_env.yml"
    if local_env_path.exists():
        logger.info(f"Loading environment variables from {local_env_path}")
        with open(local_env_path, "r") as file:
            try:
                config = yaml.safe_load(file)
                if config and isinstance(config, dict):
                    for key, value in config.items():
                        if key not in os.environ:
                            os.environ[key] = str(value)
                            logger.debug(f"Loaded {key} from local_env.yml")
            except Exception as e:
                logger.error(f"Error loading local_env.yml: {e}")

# Load environment variables
load_environment()

# Set log level if specified
log_level = os.getenv("LOG_LEVEL", "INFO")
numeric_level = getattr(logging, log_level.upper(), None)
if isinstance(numeric_level, int):
    logging.getLogger().setLevel(numeric_level)
    logger.info(f"Set log level to {log_level}")

# Discord configuration
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN") or os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    logger.error("Discord bot token not found in environment variables")
    logger.error("Please set DISCORD_BOT_TOKEN in your .env file")

COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!")
APPLICATION_ID = os.getenv("APPLICATION_ID")

# Validate GUILD_ID - it should be None if not set or if it contains a placeholder
guild_id_value = os.getenv("DISCORD_GUILD_ID")
if guild_id_value and guild_id_value != "your_discord_server_id_here" and not "your_" in guild_id_value:
    try:
        # Check if it can be converted to integer
        int(guild_id_value)
        GUILD_ID = guild_id_value
    except ValueError:
        logger.warning(f"DISCORD_GUILD_ID must be a valid integer. Got: {guild_id_value}")
        logger.warning("Defaulting to global command registration.")
        GUILD_ID = None
else:
    GUILD_ID = None

# API URLs
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000/api")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# LLM configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")

# Check if any LLM providers are configured
any_llm_configured = any([OPENAI_API_KEY, MISTRAL_API_KEY, ANTHROPIC_API_KEY])
if not any_llm_configured:
    logger.warning("No LLM provider API keys found. Some functionality will be limited.")
    logger.warning("Please set at least one of OPENAI_API_KEY, ANTHROPIC_API_KEY, or MISTRAL_API_KEY in your .env file")

# Other configurations
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true" 