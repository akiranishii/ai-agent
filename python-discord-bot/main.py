import os
import sys
import logging
import asyncio
import discord
from discord.ext import commands
import traceback
from typing import List, Optional

# Import configuration
from config import DISCORD_TOKEN, COMMAND_PREFIX, APPLICATION_ID, GUILD_ID, DEBUG_MODE

# Set up logging
logger = logging.getLogger(__name__)

# Initialize intents
intents = discord.Intents.default()
intents.message_content = True  # Need message content for processing messages
intents.members = True  # Need members for user lookups

# Create bot instance
bot = commands.Bot(
    command_prefix=COMMAND_PREFIX,
    intents=intents,
    application_id=APPLICATION_ID,
    help_command=commands.DefaultHelpCommand(
        no_category="Commands"
    )
)

# Extension/cog settings
INITIAL_EXTENSIONS = [
    "commands.lab_session_commands",  # Use this for all session-related commands
    "commands.lab_agent_commands",
    "commands.lab_meeting_commands",
    "commands.lab_transcript_commands",
    # "commands.session_commands",  # DEPRECATED: Do not use, replaced by lab_session_commands.py
    "commands.quickstart_command",
    "commands.help_command"
]

# Event Handlers
@bot.event
async def on_ready():
    """Called when the bot is connected and ready."""
    logger.info(f"{bot.user.name} is connected to Discord!")
    logger.info(f"Bot ID: {bot.user.id}")
    logger.info(f"Connected to {len(bot.guilds)} guilds")
    
    # Get available LLM providers
    try:
        from llm_client import LLMClient
        llm_client = LLMClient()
        
        # Log available providers
        available_providers = [
            provider for provider, details in llm_client.providers.items() 
            if details.get("is_available", False)
        ]
        
        if available_providers:
            logger.info(f"Available LLM providers: {', '.join(available_providers)}")
            logger.info(f"Default models per provider:")
            for provider, details in llm_client.providers.items():
                if details.get("is_available", False):
                    logger.info(f"  - {provider}: {details.get('default_model', 'unknown')}")
        else:
            logger.warning("No LLM providers available. Lab meeting features will be limited.")
            logger.warning("Please check your API keys in the .env file.")
    except Exception as e:
        logger.error(f"Error loading LLM providers: {str(e)}")
        logger.debug(traceback.format_exc())

    # Display server registration
    if GUILD_ID:
        logger.info(f"Using guild ID: {GUILD_ID}")
    else:
        logger.info("No guild ID specified. Commands will be registered globally.")

    # After connection is established, sync slash commands
    try:
        logger.info("Syncing commands with Discord...")
        if GUILD_ID:
            await bot.tree.sync(guild=discord.Object(id=GUILD_ID))
            logger.info(f"Slash commands synced to guild: {GUILD_ID}")
        else:
            await bot.tree.sync()
            logger.info("Slash commands synced globally")
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")
        logger.debug(traceback.format_exc())
        
    # Set bot status
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching, 
            name="for /help commands"
        )
    )
    
    logger.info("Bot is ready!")

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors."""
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore command not found errors
    
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: {error.param}")
        return
    
    if isinstance(error, commands.BadArgument):
        await ctx.send(f"Bad argument: {error}")
        return
    
    # Log the error
    logger.error(f"Command error in {ctx.command}: {error}")
    logger.error(traceback.format_exc())
    
    # If in debug mode, send the error to the channel
    if DEBUG_MODE:
        await ctx.send(f"An error occurred: {error}")
    else:
        await ctx.send("An error occurred while processing your command. Please try again later.")

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
    """Handle errors that occur during app command execution."""
    logger.error(f"App command error: {error}")
    
    # Extract the original error if it's wrapped in a CommandInvokeError
    if isinstance(error, discord.app_commands.CommandInvokeError):
        error = error.original
        logger.error(f"Original error: {error}")
    
    # Check for common interaction errors - use string matching as the error can be wrapped
    error_str = str(error)
    
    # Handle 10062 (Unknown interaction) error - this means the interaction token expired
    if "error code: 10062" in error_str or "Unknown interaction" in error_str:
        logger.warning(f"Interaction timed out (already expired): {error_str}")
        return  # Can't respond to an expired interaction
    
    # Handle 40060 (Interaction already acknowledged) error
    if "error code: 40060" in error_str or "interaction has already been acknowledged" in error_str:
        logger.warning(f"Interaction already acknowledged: {error_str}")
        try:
            await interaction.followup.send(
                "An error occurred. Please try again later.",
                ephemeral=True
            )
        except Exception as follow_up_error:
            logger.error(f"Failed to send followup message: {follow_up_error}")
        return
    
    # Try to respond with an error message
    try:
        if interaction.response.is_done():
            await interaction.followup.send(
                "An error occurred while processing your command. Please try again later.",
                ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "An error occurred while processing your command. Please try again later.",
                ephemeral=True
            )
    except Exception as response_error:
        logger.error(f"Failed to send error message: {response_error}")

async def load_extensions():
    """Load all extensions/cogs."""
    logger.info("Starting extension loading process")
    logger.info("Beginning extension loading sequence")
    
    loaded_count = 0
    total_count = len(INITIAL_EXTENSIONS)
    
    for extension in INITIAL_EXTENSIONS:
        try:
            logger.info(f"Attempting to load extension: {extension}")
            await bot.load_extension(extension)
            logger.info(f"Successfully loaded extension: {extension}")
            loaded_count += 1
        except Exception as e:
            logger.error(f"Failed to load extension {extension}: {e}")
            logger.error(traceback.format_exc())
    
    logger.info(f"Completed extension loading. Loaded {loaded_count} of {total_count} extensions")
    
    # Give a small delay to ensure all commands are registered
    if loaded_count > 0:
        logger.info("Extension loading completed successfully")
        logger.info("Waiting for commands to be fully registered...")
        await asyncio.sleep(1)  # Short delay to ensure all commands are registered
        logger.info("Starting command sync process")
    else:
        logger.error("No extensions were loaded successfully")
        
    return loaded_count > 0

async def main():
    """Main entry point for the bot."""
    try:
        # Load extensions before connecting to Discord
        extensions_loaded = await load_extensions()
        
        if not extensions_loaded:
            logger.error("Failed to load any extensions. Exiting.")
            return 1
        
        # Start the bot
        logger.info("Starting bot...")
        async with bot:
            await bot.start(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        logger.error(traceback.format_exc())
        return 1
    return 0

if __name__ == "__main__":
    # Run the bot
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 