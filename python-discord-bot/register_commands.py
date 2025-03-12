#!/usr/bin/env python
"""
Command Registration Utility

This script runs the bot temporarily to register commands with Discord.
It waits for the on_ready event, registers all commands, and then shuts down.
"""

import os
import sys
import asyncio
import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("register_commands")

# Load environment variables from .env file
load_dotenv()

# Get the Discord token from environment variables
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
APPLICATION_ID = os.getenv("APPLICATION_ID")
GUILD_ID = os.getenv("DISCORD_GUILD_ID")

if not DISCORD_TOKEN:
    logger.error("DISCORD_BOT_TOKEN not found in environment variables")
    sys.exit(1)

async def register_commands(global_sync=True, guild_id=None):
    """Run the bot temporarily to register commands."""
    # Set up intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    
    # Create bot instance
    logger.info("Creating bot instance...")
    bot = commands.Bot(
        command_prefix="!",
        intents=intents,
        application_id=APPLICATION_ID
    )
    
    # List of extensions to load
    extensions = [
        "commands.help_command",
        "commands.lab_session_commands",
        "commands.lab_agent_commands",
        "commands.lab_meeting_commands",
        "commands.lab_transcript_commands",
        "commands.quickstart_command"
    ]
    
    # Command registration flag
    registered = False
    
    @bot.event
    async def on_ready():
        nonlocal registered
        if registered:
            return
        
        registered = True
        logger.info(f"Bot is ready! Logged in as {bot.user} (ID: {bot.user.id})")
        
        try:
            # Load all extensions
            logger.info("Loading extensions...")
            for extension in extensions:
                try:
                    await bot.load_extension(extension)
                    logger.info(f"Loaded extension: {extension}")
                except Exception as e:
                    logger.error(f"Failed to load extension {extension}: {e}")
            
            # Wait a moment for commands to register
            logger.info("Waiting for commands to register...")
            await asyncio.sleep(2)
            
            # Sync commands
            if global_sync:
                logger.info("Syncing commands globally...")
                synced = await bot.tree.sync()
                logger.info(f"Successfully synced {len(synced)} commands globally")
                
                for cmd in synced:
                    logger.info(f"Synced command: {cmd.name}")
            else:
                if not guild_id:
                    logger.error("No guild ID provided for guild sync")
                    await bot.close()
                    return
                
                guild = discord.Object(id=guild_id)
                logger.info(f"Syncing commands to guild {guild_id}...")
                synced = await bot.tree.sync(guild=guild)
                logger.info(f"Successfully synced {len(synced)} commands to guild {guild_id}")
                
                for cmd in synced:
                    logger.info(f"Synced command: {cmd.name}")
        except Exception as e:
            logger.error(f"Error registering commands: {e}")
        finally:
            # Close the bot after syncing
            logger.info("Command registration complete, shutting down bot...")
            await asyncio.sleep(1)
            await bot.close()
    
    # Run the bot
    try:
        logger.info("Starting bot...")
        await bot.start(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        logger.info("Bot has been shut down")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register Discord slash commands")
    parser.add_argument(
        "--guild", "-g", 
        help="Register commands to a specific guild instead of globally",
        action="store_true"
    )
    parser.add_argument(
        "--guild-id", 
        help="Specific guild ID to register to (overrides DISCORD_GUILD_ID env var)"
    )
    
    args = parser.parse_args()
    
    # Determine guild ID if needed
    guild_id = None
    if args.guild:
        guild_id = args.guild_id or GUILD_ID
        if not guild_id:
            logger.error("No guild ID provided. Either set DISCORD_GUILD_ID in .env or use --guild-id")
            sys.exit(1)
    
    # Run the command registration
    asyncio.run(register_commands(
        global_sync=not args.guild,
        guild_id=guild_id
    )) 