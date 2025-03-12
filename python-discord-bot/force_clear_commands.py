#!/usr/bin/env python
"""
Direct API Command Clearing Utility

This script uses direct API calls to Discord to clear all global slash commands.
Use this as a last resort when commands are broken or stuck.
"""

import os
import requests
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("force_clear_commands")

# Load environment variables from .env file
load_dotenv()

# Get required tokens from environment variables
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
APPLICATION_ID = os.getenv("APPLICATION_ID")
GUILD_ID = os.getenv("DISCORD_GUILD_ID")

if not TOKEN:
    logger.error("DISCORD_BOT_TOKEN not found in environment variables")
    exit(1)

if not APPLICATION_ID:
    logger.error("APPLICATION_ID not found in environment variables")
    exit(1)

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

def list_global_commands():
    """List all global commands using Discord's API."""
    logger.info(f"Listing all global commands for application {APPLICATION_ID}")
    
    url = f"https://discord.com/api/v10/applications/{APPLICATION_ID}/commands"
    
    headers = {
        "Authorization": f"Bot {TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            commands = response.json()
            logger.info(f"Found {len(commands)} global commands")
            
            for cmd in commands:
                logger.info(f"Command: {cmd['name']} (ID: {cmd['id']})")
            
            return commands
        else:
            logger.error(f"Failed to list global commands: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing global commands: {e}")
        return []

def list_guild_commands(guild_id):
    """List all guild commands using Discord's API."""
    logger.info(f"Listing all commands for guild {guild_id}")
    
    url = f"https://discord.com/api/v10/applications/{APPLICATION_ID}/guilds/{guild_id}/commands"
    
    headers = {
        "Authorization": f"Bot {TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            commands = response.json()
            logger.info(f"Found {len(commands)} commands for guild {guild_id}")
            
            for cmd in commands:
                logger.info(f"Command: {cmd['name']} (ID: {cmd['id']})")
            
            return commands
        else:
            logger.error(f"Failed to list guild commands: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing guild commands: {e}")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Force clear Discord slash commands")
    parser.add_argument(
        "--list", "-l", 
        help="Only list commands without clearing them",
        action="store_true"
    )
    parser.add_argument(
        "--guild", "-g", 
        help="Clear guild commands instead of global commands",
        action="store_true"
    )
    parser.add_argument(
        "--guild-id", 
        help="Specific guild ID to clear (overrides DISCORD_GUILD_ID env var)"
    )
    
    args = parser.parse_args()
    
    # Determine guild ID if needed
    guild_id = None
    if args.guild:
        guild_id = args.guild_id or GUILD_ID
        if not guild_id:
            logger.error("No guild ID provided. Either set DISCORD_GUILD_ID in .env or use --guild-id")
            exit(1)
    
    if args.list:
        # Only list commands
        if args.guild:
            list_guild_commands(guild_id)
        else:
            list_global_commands()
    else:
        # Clear commands
        if args.guild:
            # First list existing commands
            list_guild_commands(guild_id)
            
            # Confirm with user
            confirm = input(f"Are you sure you want to clear ALL commands for guild {guild_id}? (y/N): ")
            if confirm.lower() == "y":
                clear_guild_commands(guild_id)
            else:
                logger.info("Operation cancelled")
        else:
            # First list existing commands
            list_global_commands()
            
            # Confirm with user
            confirm = input(f"Are you sure you want to clear ALL global commands? This will affect all servers! (y/N): ")
            if confirm.lower() == "y":
                clear_global_commands()
            else:
                logger.info("Operation cancelled") 