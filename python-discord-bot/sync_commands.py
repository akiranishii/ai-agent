#!/usr/bin/env python
"""
Command Sync Utility

This script manually syncs slash commands with Discord's API.
"""

import os
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log")
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the Discord token from environment variables
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
APPLICATION_ID = os.getenv("APPLICATION_ID")
GUILD_ID = os.getenv("DISCORD_GUILD_ID")

if not DISCORD_TOKEN:
    logger.error("DISCORD_BOT_TOKEN not found in environment variables")
    exit(1)

async def sync_commands(global_sync=True, guild_id=None):
    """Sync commands with Discord's API."""
    # Set up intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    
    # Create bot instance
    bot = commands.Bot(
        command_prefix="!",  # This doesn't matter for syncing
        intents=intents,
        application_id=APPLICATION_ID
    )
    
    @bot.event
    async def on_ready():
        logger.info(f'Bot connected as {bot.user}')
        
        try:
            # Load all command modules
            from commands import lab_meeting_commands, management_commands
            await lab_meeting_commands.setup(bot)
            await management_commands.setup(bot)
            
            # Sync commands with Discord
            if GUILD_ID:
                guild = discord.Object(id=GUILD_ID)
                await bot.tree.sync(guild=guild)
                logger.info(f"Successfully synced commands to guild {GUILD_ID}")
            else:
                await bot.tree.sync()
                logger.info("Successfully synced commands globally")
                
            # Log registered commands
            commands = bot.tree.get_commands()
            command_groups = [cmd for cmd in commands if isinstance(cmd, discord.app_commands.Group)]
            
            for group in command_groups:
                logger.info(f"Command group: {group.name} - {len(group.commands)} commands")
                for cmd in group.commands:
                    logger.info(f"  - {cmd.name}: {cmd.description}")
                    
            # Exit after syncing
            logger.info("Command sync complete. Exiting...")
            await bot.close()
            
        except Exception as e:
            logger.error(f"Error syncing commands: {e}")
            await bot.close()
    
    try:
        # Log in to Discord
        await bot.login(DISCORD_TOKEN)
        logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
        
        # Load the help_command extension which contains our admin_sync command
        logger.info("Loading extensions...")
        extensions = [
            "commands.help_command",
            "commands.lab_session_commands",
            "commands.lab_agent_commands",
            "commands.lab_meeting_commands",
            "commands.lab_transcript_commands",
            "commands.quickstart_command"
        ]
        
        for extension in extensions:
            try:
                await bot.load_extension(extension)
                logger.info(f"Loaded extension: {extension}")
            except Exception as e:
                logger.error(f"Failed to load extension {extension}: {e}")
        
        if global_sync:
            # Sync commands globally
            logger.info("Syncing commands globally...")
            synced = await bot.tree.sync()
            logger.info(f"Successfully synced {len(synced)} commands globally")
        else:
            # Sync commands to a specific guild
            if not guild_id and GUILD_ID:
                guild_id = GUILD_ID
                
            if not guild_id:
                logger.error("No guild ID provided for guild sync")
                return
                
            logger.info(f"Syncing commands to guild {guild_id}...")
            synced = await bot.tree.sync(guild=discord.Object(id=guild_id))
            logger.info(f"Successfully synced {len(synced)} commands to guild {guild_id}")
            
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")
    finally:
        # Close the connection
        await bot.close()
        logger.info("Bot connection closed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync Discord slash commands")
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
    asyncio.run(sync_commands(
        global_sync=not args.guild,
        guild_id=args.guild_id
    )) 