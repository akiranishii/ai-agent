#!/usr/bin/env python
"""
Command Reset Utility

This script provides a comprehensive solution for resetting Discord slash commands.
It:
1. Uses direct API calls to clear all commands
2. Runs the bot to register and sync commands
"""

import os
import sys
import asyncio
import requests
import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("reset_commands")

# Load environment variables from .env file
load_dotenv()

# Get required tokens from environment variables
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
APPLICATION_ID = os.getenv("APPLICATION_ID")
GUILD_ID = os.getenv("DISCORD_GUILD_ID")

if not TOKEN:
    logger.error("DISCORD_BOT_TOKEN not found in environment variables")
    sys.exit(1)

if not APPLICATION_ID:
    logger.error("APPLICATION_ID not found in environment variables")
    sys.exit(1)

def clear_global_commands():
    """Directly clear all global commands using Discord's API."""
    logger.info(f"Clearing all global commands for application {APPLICATION_ID}")
    
    url = f"https://discord.com/api/v10/applications/{APPLICATION_ID}/commands"
    
    headers = {
        "Authorization": f"Bot {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Send PUT request with empty array to replace all commands with nothing
    try:
        response = requests.put(url, headers=headers, json=[])
        
        if response.status_code == 200:
            logger.info("Successfully cleared all global commands")
            return True
        else:
            logger.error(f"Failed to clear global commands: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error clearing global commands: {e}")
        return False

def clear_guild_commands(guild_id):
    """Directly clear all guild commands using Discord's API."""
    logger.info(f"Clearing all commands for guild {guild_id}")
    
    url = f"https://discord.com/api/v10/applications/{APPLICATION_ID}/guilds/{guild_id}/commands"
    
    headers = {
        "Authorization": f"Bot {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Send PUT request with empty array to replace all commands with nothing
    try:
        response = requests.put(url, headers=headers, json=[])
        
        if response.status_code == 200:
            logger.info(f"Successfully cleared all commands for guild {guild_id}")
            return True
        else:
            logger.error(f"Failed to clear guild commands: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error clearing guild commands: {e}")
        return False

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
        await bot.start(TOKEN)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        logger.info("Bot has been shut down")

async def reset_commands(global_sync=True, guild_id=None):
    """Reset (clear and re-register) all Discord slash commands."""
    try:
        # Step 1: Clear commands
        if global_sync:
            clear_global_commands()
        else:
            if not guild_id:
                logger.error("No guild ID provided for guild operations")
                return
            clear_guild_commands(guild_id)
            
        # Step 2: Register commands
        await register_commands(global_sync, guild_id)
        
        logger.info("Command reset process completed successfully")
    except Exception as e:
        logger.error(f"Error during command reset process: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reset Discord slash commands")
    parser.add_argument(
        "--guild", "-g", 
        help="Reset guild commands instead of global commands",
        action="store_true"
    )
    parser.add_argument(
        "--guild-id", 
        help="Specific guild ID to reset (overrides DISCORD_GUILD_ID env var)"
    )
    parser.add_argument(
        "--skip-confirm", "-y",
        help="Skip confirmation prompt",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Determine guild ID if needed
    guild_id = None
    if args.guild:
        guild_id = args.guild_id or GUILD_ID
        if not guild_id:
            logger.error("No guild ID provided. Either set DISCORD_GUILD_ID in .env or use --guild-id")
            sys.exit(1)
    
    # Confirm with user unless --skip-confirm is used
    if not args.skip_confirm:
        if args.guild:
            confirm = input(f"Are you sure you want to reset ALL commands for guild {guild_id}? (y/N): ")
        else:
            confirm = input(f"Are you sure you want to reset ALL global commands? This will affect all servers! (y/N): ")
            
        if confirm.lower() != "y":
            logger.info("Operation cancelled")
            sys.exit(0)
    
    # Run the reset process
    asyncio.run(reset_commands(
        global_sync=not args.guild,
        guild_id=guild_id
    )) 