#!/usr/bin/env python3
"""
Simple Discord Command Deletion Script

This script:
1. Fetches all Discord application commands (global or for a specific guild)
2. Identifies commands from a specified list (like 'brainstorm', 'cancel_brainstorm')
3. Deletes those commands using the Discord API

IMPORTANT: Before running, set these environment variables:
- DISCORD_TOKEN - Your bot token
- APPLICATION_ID - Your Discord application ID
- GUILD_ID (optional) - Specific guild ID, leave unset for global commands

Usage:
    python simple_delete_commands.py
"""

import os
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("delete-commands")

# Get configuration from environment variables
TOKEN = os.getenv("DISCORD_TOKEN")
APP_ID = os.getenv("APPLICATION_ID")
GUILD_ID = os.getenv("GUILD_ID")  # Optional: for guild-specific commands

# Commands to find and delete - add any others you need
COMMAND_NAMES_TO_DELETE = [
    "brainstorm",
    "cancel_brainstorm",
    # Add any other command names here
]

async def fetch_commands(session: aiohttp.ClientSession, headers: Dict[str, str]) -> List[Dict]:
    """Fetch all application commands."""
    # Build the URL based on whether we're targeting global or guild commands
    if GUILD_ID:
        url = f"https://discord.com/api/v10/applications/{APP_ID}/guilds/{GUILD_ID}/commands"
    else:
        url = f"https://discord.com/api/v10/applications/{APP_ID}/commands"
    
    # Make the API request
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            commands = await response.json()
            return commands
        else:
            error = await response.text()
            logger.error(f"Failed to fetch commands: {response.status} - {error}")
            return []

async def delete_command(session: aiohttp.ClientSession, headers: Dict[str, str], command_id: str) -> bool:
    """Delete a specific command by ID."""
    # Build the URL based on whether we're targeting global or guild commands
    if GUILD_ID:
        url = f"https://discord.com/api/v10/applications/{APP_ID}/guilds/{GUILD_ID}/commands/{command_id}"
    else:
        url = f"https://discord.com/api/v10/applications/{APP_ID}/commands/{command_id}"
    
    # Make the API request
    async with session.delete(url, headers=headers) as response:
        if response.status == 204:
            return True
        else:
            error = await response.text()
            logger.error(f"Failed to delete command {command_id}: {response.status} - {error}")
            return False

async def main():
    """Main function to find and delete commands."""
    # Check for required environment variables
    if not TOKEN or not APP_ID:
        logger.error("Missing required environment variables: DISCORD_TOKEN and APPLICATION_ID")
        return
    
    # Setup headers for API requests
    headers = {
        "Authorization": f"Bot {TOKEN}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Fetch all commands
        commands = await fetch_commands(session, headers)
        
        if not commands:
            logger.info("No commands found.")
            return
        
        # Step 2: Display all found commands
        logger.info(f"Found {len(commands)} commands:")
        for cmd in commands:
            logger.info(f"- {cmd['name']} (ID: {cmd['id']})")
        
        # Step 3: Find commands by name
        commands_to_delete = []
        for cmd in commands:
            if cmd["name"] in COMMAND_NAMES_TO_DELETE:
                commands_to_delete.append(cmd)
        
        if not commands_to_delete:
            logger.info(f"None of the specified commands ({', '.join(COMMAND_NAMES_TO_DELETE)}) were found.")
            return
        
        # Step 4: Delete the commands
        logger.info(f"Found {len(commands_to_delete)} commands to delete:")
        for cmd in commands_to_delete:
            logger.info(f"Deleting command: {cmd['name']} (ID: {cmd['id']})")
            success = await delete_command(session, headers, cmd["id"])
            if success:
                logger.info(f"Successfully deleted command: {cmd['name']}")
            else:
                logger.error(f"Failed to delete command: {cmd['name']}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 