#!/usr/bin/env python3
"""
Command Deletion Script for Discord Bot

This script allows you to:
1. List all registered slash commands (global or guild-specific)
2. Delete specific slash commands by name
3. Delete all slash commands

Usage:
    python delete_commands.py
"""

import os
import asyncio
import discord
from discord.ext import commands
import logging
import aiohttp
import json
from typing import List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('command-deletion')

# Import configuration values
# You need to define these in your .env file or directly in this script
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')  # Your Discord bot token
APPLICATION_ID = os.getenv('APPLICATION_ID')  # Your application/bot ID
GUILD_ID = os.getenv('GUILD_ID')  # Optional: Specific guild ID to target (or None for global commands)

# Which commands to delete - add any command names you want to remove
COMMANDS_TO_DELETE = [
    'brainstorm',
    'cancel_brainstorm',
    # Add other command names here
]

DELETE_ALL_COMMANDS = False  # Set to True if you want to delete all commands

class CommandManager:
    """Manage Discord application commands."""
    
    def __init__(self, token: str, application_id: str, guild_id: Optional[str] = None):
        """Initialize the command manager.
        
        Args:
            token: Discord bot token
            application_id: Discord application ID
            guild_id: Optional guild ID for guild-specific commands
        """
        self.token = token
        self.application_id = application_id
        self.guild_id = guild_id
        self.base_url = f"https://discord.com/api/v10/applications/{application_id}"
        self.headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json"
        }
    
    async def fetch_commands(self) -> List[dict]:
        """Fetch all application commands.
        
        Returns:
            List of command objects
        """
        url = f"{self.base_url}/commands"
        if self.guild_id:
            url = f"{self.base_url}/guilds/{self.guild_id}/commands"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    commands = await response.json()
                    return commands
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to fetch commands: {response.status} - {error_text}")
                    return []
    
    async def delete_command(self, command_id: str) -> bool:
        """Delete a command by ID.
        
        Args:
            command_id: ID of the command to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        url = f"{self.base_url}/commands/{command_id}"
        if self.guild_id:
            url = f"{self.base_url}/guilds/{self.guild_id}/commands/{command_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=self.headers) as response:
                if response.status == 204:
                    logger.info(f"Successfully deleted command: {command_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to delete command: {response.status} - {error_text}")
                    return False

async def main():
    """Main entry point for the script."""
    if not DISCORD_TOKEN or not APPLICATION_ID:
        logger.error("Missing required environment variables. Make sure DISCORD_TOKEN and APPLICATION_ID are set.")
        return 1
    
    # Initialize command manager
    command_manager = CommandManager(DISCORD_TOKEN, APPLICATION_ID, GUILD_ID)
    
    # Fetch all commands
    commands = await command_manager.fetch_commands()
    
    if not commands:
        logger.info("No commands found.")
        return 0
    
    # Display all commands
    logger.info(f"Found {len(commands)} commands:")
    command_table = []
    for command in commands:
        command_table.append({
            "id": command.get("id"),
            "name": command.get("name"),
            "description": command.get("description")
        })
    
    print(json.dumps(command_table, indent=2))
    
    # Delete specified commands
    if DELETE_ALL_COMMANDS:
        logger.warning("Deleting ALL commands!")
        for command in commands:
            await command_manager.delete_command(command["id"])
        logger.info("All commands deleted.")
    else:
        logger.info(f"Deleting commands in list: {COMMANDS_TO_DELETE}")
        for command in commands:
            if command["name"] in COMMANDS_TO_DELETE:
                logger.info(f"Deleting command: {command['name']} ({command['id']})")
                await command_manager.delete_command(command["id"])

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 