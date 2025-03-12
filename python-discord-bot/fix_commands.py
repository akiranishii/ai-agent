#!/usr/bin/env python
"""
Command Fix Utility

This script manually rebuilds and syncs slash commands with Discord's API.
Use this when commands are broken or missing.
"""

import os
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv
import logging
import importlib
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fix_commands")

# Load environment variables from .env file
load_dotenv()

# Get the Discord token from environment variables
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
APPLICATION_ID = os.getenv("APPLICATION_ID")
GUILD_ID = os.getenv("DISCORD_GUILD_ID")

if not DISCORD_TOKEN:
    logger.error("DISCORD_BOT_TOKEN not found in environment variables")
    exit(1)

async def rebuild_commands(global_sync=True, guild_id=None):
    """Rebuild and sync commands with Discord's API."""
    # Set up intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    
    # Create bot instance
    bot = commands.Bot(
        command_prefix="!",
        intents=intents,
        application_id=APPLICATION_ID
    )
    
    # Define extensions to load
    extensions = [
        "commands.help_command",
        "commands.lab_session_commands",
        "commands.lab_agent_commands",
        "commands.lab_meeting_commands",
        "commands.lab_transcript_commands",
        "commands.quickstart_command"
    ]
    
    # Create event handlers
    @bot.event
    async def on_ready():
        logger.info(f"Bot is ready! Logged in as {bot.user} (ID: {bot.user.id})")
        
        try:
            # Load all extensions
            logger.info("Loading extensions...")
            for extension in extensions:
                try:
                    # Force reload the module first to ensure we have the latest code
                    if extension in sys.modules:
                        logger.info(f"Force reloading module: {extension}")
                        importlib.reload(sys.modules[extension])
                    
                    # Then load the extension
                    await bot.load_extension(extension)
                    logger.info(f"Loaded extension: {extension}")
                except Exception as e:
                    logger.error(f"Failed to load extension {extension}: {e}")
            
            # Wait a brief moment for commands to register
            await asyncio.sleep(2)
            logger.info("Allowing time for commands to register...")
            
            if global_sync:
                # Clear and sync global commands
                logger.info("Clearing global commands...")
                bot.tree.clear_commands(guild=None)
                
                logger.info("Syncing global commands...")
                synced = await bot.tree.sync()
                logger.info(f"Successfully synced {len(synced)} commands globally")
                
                # Print synced commands for debugging
                for cmd in synced:
                    logger.info(f"Synced command: {cmd.name}")
            else:
                # Sync commands to a specific guild
                if not guild_id and GUILD_ID:
                    guild_id = GUILD_ID
                    
                if not guild_id:
                    logger.error("No guild ID provided for guild sync")
                    return
                
                # Clear and sync guild-specific commands
                logger.info(f"Clearing commands for guild {guild_id}...")
                bot.tree.clear_commands(guild=discord.Object(id=guild_id))
                
                logger.info(f"Syncing commands to guild {guild_id}...")
                synced = await bot.tree.sync(guild=discord.Object(id=guild_id))
                logger.info(f"Successfully synced {len(synced)} commands to guild {guild_id}")
                
                # Print synced commands for debugging
                for cmd in synced:
                    logger.info(f"Synced command: {cmd.name}")
            
            # Stop the bot after syncing
            logger.info("Command sync complete, stopping bot...")
            await bot.close()
        except Exception as e:
            logger.error(f"Error syncing commands: {e}")
            await bot.close()
    
    try:
        # Connect to Discord
        logger.info("Connecting to Discord...")
        await bot.start(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Error connecting to Discord: {e}")
    finally:
        logger.info("Bot connection closed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix Discord slash commands")
    parser.add_argument(
        "--guild", "-g", 
        help="Sync to a specific guild instead of globally",
        action="store_true"
    )
    parser.add_argument(
        "--guild-id", 
        help="Specific guild ID to sync to (overrides DISCORD_GUILD_ID env var)"
    )
    
    args = parser.parse_args()
    
    # Run the sync function
    asyncio.run(rebuild_commands(
        global_sync=not args.guild,
        guild_id=args.guild_id
    )) 