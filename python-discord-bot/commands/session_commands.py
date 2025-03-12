# DEPRECATED: This file is no longer used and has been replaced by lab_session_commands.py
# DO NOT USE OR UNCOMMENT THIS FILE

# import discord
# from discord import app_commands
# from discord.ext import commands
# import logging
# from typing import Optional

# from db_client import db_client

# logger = logging.getLogger(__name__)

# class SessionCommands(commands.Cog):
#     """Commands for managing virtual lab sessions"""
#     
#     def __init__(self, bot):
#         self.bot = bot
#     
#     @app_commands.command(
#         name="start",
#         description="Start a new virtual lab session"
#     )
#     @app_commands.describe(
#         title="The title of your virtual lab session",
#         description="A brief description of your research topic or question",
#         public="Whether the session should be public (default: false)"
#     )
#     async def start_session(
#         self, 
#         interaction: discord.Interaction,
#         title: str,
#         description: str = None,
#         public: bool = False
#     ):
#         """Start a new virtual lab session.
#         
#         Args:
#             interaction: The Discord interaction
#             title: Title of the session
#             description: Optional description of the session
#             public: Whether the session should be public
#         """
#         try:
#             # Get the user's Discord ID
#             user_id = str(interaction.user.id)
#             
#             # Defer reply to prevent timeout during API calls
#             await interaction.response.defer(ephemeral=True)
#             
#             # Check if user already has an active session
#             active_session = await db_client.get_active_session(user_id)
#             
#             if active_session and active_session.get("isSuccess") and active_session.get("data"):
#                 # User already has an active session
#                 session_id = active_session.get("data").get("id")
#                 session_title = active_session.get("data").get("title")
#                 
#                 embed = discord.Embed(
#                     title="You already have an active session",
#                     description=f"You already have an active session titled '{session_title}'.",
#                     color=discord.Color.yellow()
#                 )
#                 embed.add_field(
#                     name="Options",
#                     value="You can end your current session with `/end` before starting a new one.",
#                     inline=False
#                 )
#                 
#                 await interaction.followup.send(embed=embed, ephemeral=True)
#                 return
#             
#             # Create a new session
#             result = await db_client.create_session(
#                 user_id=user_id,
#                 title=title,
#                 description=description or "",
#                 is_public=public
#             )
#             
#             if result.get("isSuccess"):
#                 session_id = result.get("data", {}).get("id")
#                 
#                 # Create success embed
#                 embed = discord.Embed(
#                     title="Session Started",
#                     description=f"Your virtual lab session '{title}' has been started.",
#                     color=discord.Color.green()
#                 )
#                 
#                 # Include the session ID
#                 embed.add_field(
#                     name="Session ID",
#                     value=session_id,
#                     inline=False
#                 )
#                 
#                 # Add instructions for next steps
#                 embed.add_field(
#                     name="Next Steps",
#                     value="Use /lab commands to work with AI scientists in your virtual lab.",
#                     inline=False
#                 )
#                 
#                 await interaction.followup.send(embed=embed, ephemeral=True)
#                 
#                 # Log session creation
#                 logger.info(f"User {user_id} created session {session_id}")
#             else:
#                 # Handle API error
#                 error_message = result.get("message", "Unknown error")
#                 embed = discord.Embed(
#                     title="Error",
#                     description=f"Failed to create session: {error_message}",
#                     color=discord.Color.red()
#                 )
#                 await interaction.followup.send(embed=embed, ephemeral=True)
#                 
#         except Exception as e:
#             logger.error(f"Error in start_session: {e}")
#             await interaction.followup.send("An error occurred while creating your session.", ephemeral=True)
#     
#     @app_commands.command(
#         name="end",
#         description="End the current virtual lab session"
#     )
#     async def end_session(self, interaction: discord.Interaction):
#         """End the user's active virtual lab session."""

# async def setup(bot: commands.Bot):
#     """Add the cog to the bot."""
#     await bot.add_cog(SessionCommands(bot)) 