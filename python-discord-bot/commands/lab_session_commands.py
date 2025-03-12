import discord
from discord import app_commands
from discord.ext import commands
import logging
from typing import Optional

from db_client import db_client

logger = logging.getLogger(__name__)

class LabSessionCommands(commands.Cog):
    """Commands for managing lab sessions"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Find the existing lab group or create a new one if it doesn't exist
        self.lab_group = None
        for command in bot.tree.get_commands():
            if command.name == "lab" and isinstance(command, app_commands.Group):
                self.lab_group = command
                break
                
        if not self.lab_group:
            # Create a command group for lab commands if it doesn't exist
            self.lab_group = app_commands.Group(
                name="lab",
                description="Lab management commands",
                guild_ids=None  # None means global commands
            )
            bot.tree.add_command(self.lab_group)
            logger.info("Created new lab command group")
        else:
            logger.info("Using existing lab command group")
        
        # Add session commands to the lab group
        self.lab_group.add_command(app_commands.Command(
            name="start",
            description="Start a new lab session",
            callback=self.start_session_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="end",
            description="End the current lab session",
            callback=self.end_session_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="list",
            description="List all your lab sessions",
            callback=self.list_sessions_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="reopen",
            description="Reopen a previously closed lab session",
            callback=self.reopen_session_callback,
            extras={"cog": self}
        ))
        
        logger.info("Registered lab session commands")

    @app_commands.command(
        name="start",
        description="Start a new lab session"
    )
    @app_commands.describe(
        title="The title of your lab session",
        description="Optional description of the session's purpose",
        is_public="Whether the session should be publicly viewable"
    )
    async def start_session(
        self, 
        interaction: discord.Interaction, 
        title: str, 
        description: Optional[str] = None,
        is_public: Optional[bool] = False
    ):
        """Start a new lab session.
        
        This command creates a new session and makes it the active session for the user.
        Any previous active session will be ended.
        """
        # Acknowledge the command to avoid timeout
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        user_id = str(interaction.user.id)
        
        try:
            # Check for existing active session
            active_session = await db_client.get_active_session(user_id=user_id)
            if active_session.get("isSuccess") and active_session.get("data"):
                # End the current session
                await db_client.end_session(
                    session_id=active_session["data"]["id"]
                )
                logger.info(f"Ended previous active session for user {user_id}")
            
            # Create the new session
            session_result = await db_client.create_session(
                user_id=user_id,
                title=title,
                description=description,
                is_public=is_public
            )
            
            if not session_result.get("isSuccess", False):
                await interaction.followup.send(
                    f"Failed to create session: {session_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            
            # Send a welcome message to the user
            embed = discord.Embed(
                title="🧪 Lab Session Started",
                description=f"Your new lab session '{title}' has been created.",
                color=discord.Color.green()
            )
            embed.add_field(name="Session ID", value=f"`{session_data.get('id')}`", inline=False)
            embed.add_field(name="Type", value="Public" if is_public else "Private", inline=True)
            embed.add_field(name="Status", value="Active", inline=True)
            embed.set_footer(text=f"Created by {interaction.user.display_name}")
            
            await interaction.followup.send(
                "✅ Lab session started successfully!",
                embed=embed,
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}", exc_info=True)
            await interaction.followup.send(
                "An error occurred while creating the session. Please try again later.",
                ephemeral=True
            )

    @app_commands.command(
        name="end",
        description="End your current lab session"
    )
    @app_commands.describe(
        confirm="Confirm that you want to end the session",
        is_public="Make the session public after ending it"
    )
    async def end_session(
        self, 
        interaction: discord.Interaction,
        confirm: Optional[bool] = False,
        is_public: Optional[bool] = False
    ):
        """End the user's active lab session."""
        # Acknowledge the command to avoid timeout
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        if not confirm:
            await interaction.followup.send(
                "Please confirm that you want to end your session by using `confirm:true`",
                ephemeral=True
            )
            return
        
        user_id = str(interaction.user.id)
        
        try:
            # Get the active session
            session_result = await db_client.get_active_session(user_id=user_id)
            
            if not session_result.get("isSuccess", False) or not session_result.get("data"):
                await interaction.followup.send(
                    "You don't have an active session to end.",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            session_id = session_data.get("id")
            
            # Update session visibility if requested
            if is_public:
                await db_client.update_session(
                    session_id=session_id,
                    updates={"isPublic": True}  # Use camelCase to match Next.js API expectations
                )
            
            # End the session
            end_result = await db_client.end_session(session_id=session_id)
            
            if not end_result.get("isSuccess", False):
                await interaction.followup.send(
                    f"Failed to end session: {end_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            # Create response embed
            embed = discord.Embed(
                title="Lab Session Ended",
                description=f"Session '{session_data.get('title')}' has been ended.",
                color=discord.Color.blue()
            )
            embed.add_field(
                name="Session ID",
                value=session_id,
                inline=True
            )
            embed.add_field(
                name="Visibility",
                value="Public" if is_public else "Private",
                inline=True
            )
            embed.add_field(
                name="View Transcripts",
                value="Use `/lab transcript list` to view session transcripts.",
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error in end_session command: {e}")
            await interaction.followup.send(
                "An error occurred while ending your session. Please try again later.",
                ephemeral=True
            )

    @app_commands.command(
        name="list",
        description="List your lab sessions"
    )
    async def list_sessions(self, interaction: discord.Interaction):
        """List active or recently ended sessions."""
        # Acknowledge the command to avoid timeout
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        user_id = str(interaction.user.id)
        
        try:
            # Get user's sessions
            sessions_result = await db_client.get_user_sessions(user_id=user_id)
            
            if not sessions_result.get("isSuccess", False):
                await interaction.followup.send(
                    f"Failed to fetch sessions: {sessions_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            sessions = sessions_result.get("data", [])
            
            if not sessions:
                await interaction.followup.send(
                    "You don't have any lab sessions yet. Use `/lab start` to create one.",
                    ephemeral=True
                )
                return
            
            # Filter sessions based on include_closed parameter
            if not include_closed:
                # Only include active and recently closed sessions
                sessions = [s for s in sessions if s.get("status") == "active" or s.get("status") == "ended"]
            
            # Apply limit
            sessions = sessions[:limit]
            
            # Create response embed
            embed = discord.Embed(
                title="Your Lab Sessions",
                description=f"Showing up to {limit} sessions" + (" (including closed)" if include_closed else ""),
                color=discord.Color.blue()
            )
            
            # Add active session if exists
            active_session = next((s for s in sessions if s.get("status") == "active"), None)
            if active_session:
                embed.add_field(
                    name="🟢 Active Session",
                    value=(
                        f"**{active_session.get('title')}**\n"
                        f"ID: {active_session.get('id')}\n"
                        f"Status: Active\n"
                        f"Privacy: {'Public' if active_session.get('is_public') else 'Private'}"
                    ),
                    inline=False
                )
            
            # Add other sessions
            other_sessions = [s for s in sessions if s.get("status") != "active"]
            if other_sessions:
                # Split sessions into chunks to avoid Discord's 1024 character limit per field
                MAX_SESSIONS_PER_FIELD = 3  # Adjust this based on your average session description length
                session_chunks = [other_sessions[i:i + MAX_SESSIONS_PER_FIELD] 
                                 for i in range(0, len(other_sessions), MAX_SESSIONS_PER_FIELD)]
                
                for i, chunk in enumerate(session_chunks):
                    other_list = []
                    for session in chunk:
                        status_emoji = "🔴" if session.get("status") == "ended" else "⚪"
                        other_list.append(
                            f"{status_emoji} **{session.get('title')}**\n"
                            f"ID: {session.get('id')}\n"
                            f"Status: {session.get('status', 'Unknown').title()}\n"
                            f"Privacy: {'Public' if session.get('is_public') else 'Private'}"
                        )
                    
                    field_name = "Your Other Sessions" if i == 0 else f"Your Other Sessions (continued {i+1})"
                    embed.add_field(
                        name=field_name,
                        value="\n\n".join(other_list),
                        inline=False
                    )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error in list_sessions command: {e}")
            await interaction.followup.send(
                "An error occurred while fetching your sessions. Please try again later.",
                ephemeral=True
            )

    @app_commands.command(
        name="reopen",
        description="Reopen a previously ended session"
    )
    @app_commands.describe(
        session_id="The ID of the ended session you wish to reopen",
        confirm="Confirm that you want to reopen the session"
    )
    async def reopen_session(
        self,
        interaction: discord.Interaction,
        session_id: str,
        confirm: Optional[bool] = False
    ):
        """Reopen a previously ended session."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        if not confirm:
            await interaction.followup.send(
                "Please confirm that you want to reopen this session by using `confirm:true`",
                ephemeral=True
            )
            return
        
        user_id = str(interaction.user.id)
        
        try:
            # Check if user has an active session
            active_session_result = await db_client.get_active_session(user_id=user_id)
            logger.info(f"Active session check result: {active_session_result}")
            
            if active_session_result.get("isSuccess") and active_session_result.get("data"):
                # End the current session
                await db_client.end_session(
                    session_id=active_session_result["data"]["id"]
                )
                logger.info(f"Ended previous active session for user {user_id}")
            else:
                logger.info(f"No active session to end for user {user_id}")
            
            # Get the session to reopen
            session_result = await db_client.get_session(session_id=session_id)
            
            if not session_result.get("isSuccess") or not session_result.get("data"):
                logger.error(f"Session not found: {session_id}")
                await interaction.followup.send(
                    "Session not found. Please check the session ID and try again.",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            
            # Extract user ID from session data, handling both snake_case and camelCase
            session_user_id = session_data.get("user_id") or session_data.get("userId")
            
            # Verify ownership
            if session_user_id != user_id:
                logger.warning(f"User {user_id} attempted to reopen session {session_id} owned by {session_user_id}")
                await interaction.followup.send(
                    "You can only reopen your own sessions.",
                    ephemeral=True
                )
                return
            
            # Check if session is already active
            session_status = session_data.get("status")
            if session_status == "active":
                logger.info(f"Session {session_id} is already active")
                await interaction.followup.send(
                    "This session is already active.",
                    ephemeral=True
                )
                return
            
            # Reopen the session
            logger.info(f"Reopening session {session_id}")
            reopen_result = await db_client.reopen_session(session_id=session_id)
            
            if not reopen_result.get("isSuccess"):
                logger.error(f"Failed to reopen session: {reopen_result}")
                await interaction.followup.send(
                    f"Failed to reopen session: {reopen_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            reopened_data = reopen_result.get("data", {})
            logger.info(f"Successfully reopened session {session_id}")
            
            # Create response embed
            embed = discord.Embed(
                title="Session Reopened",
                description=f"Session '{session_data.get('title')}' has been reopened.",
                color=discord.Color.green()
            )
            embed.add_field(
                name="Session ID",
                value=session_id,
                inline=True
            )
            embed.add_field(
                name="Privacy",
                value="Public" if session_data.get("is_public") or session_data.get("isPublic") else "Private",
                inline=True
            )
            embed.add_field(
                name="Next Steps",
                value="Use `/lab agent create` to add agents or `/lab team_meeting` to begin a discussion.",
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error reopening session: {str(e)}", exc_info=True)
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred while reopening the session. Please try again later.",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "An error occurred while reopening the session. Please try again later.",
                        ephemeral=True
                    )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error message: {follow_up_error}")

    async def start_session_callback(self, interaction: discord.Interaction, title: str, description: Optional[str] = None, is_public: Optional[bool] = False):
        """Callback for the start_session command."""
        try:
            # Immediately acknowledge the interaction to prevent timeout
            await interaction.response.defer(ephemeral=True)
            
            user_id = str(interaction.user.id)
            
            # Check for existing active session
            active_session = await db_client.get_active_session(user_id=user_id)
            if active_session.get("isSuccess") and active_session.get("data"):
                # End the current session
                await db_client.end_session(
                    session_id=active_session["data"]["id"]
                )
                logger.info(f"Ended previous active session for user {user_id}")
            
            # Create the new session
            session_result = await db_client.create_session(
                user_id=user_id,
                title=title,
                description=description,
                is_public=is_public
            )
            
            if not session_result.get("isSuccess", False):
                await interaction.followup.send(
                    f"Failed to create session: {session_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            
            # Send a welcome message to the user
            embed = discord.Embed(
                title="🧪 Lab Session Started",
                description=f"Your new lab session '{title}' has been created.",
                color=discord.Color.green()
            )
            embed.add_field(name="Session ID", value=f"`{session_data.get('id')}`", inline=False)
            embed.add_field(name="Type", value="Public" if is_public else "Private", inline=True)
            embed.add_field(name="Status", value="Active", inline=True)
            embed.set_footer(text=f"Created by {interaction.user.display_name}")
            
            await interaction.followup.send(
                "✅ Lab session started successfully!",
                embed=embed,
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}", exc_info=True)
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred while creating the session. Please try again later.",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "An error occurred while creating the session. Please try again later.",
                        ephemeral=True
                    )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error message: {follow_up_error}")
        
    async def end_session_callback(self, interaction: discord.Interaction, session_id: Optional[str] = None):
        """Callback for the end_session command."""
        try:
            # Immediately acknowledge the interaction to prevent timeout
            await interaction.response.defer(ephemeral=True)
            
            user_id = str(interaction.user.id)
            
            if not session_id:
                # Get the user's active session
                active_session = await db_client.get_active_session(user_id=user_id)
                
                if not active_session.get("isSuccess") or not active_session.get("data"):
                    await interaction.followup.send(
                        "You don't have an active session to end.",
                        ephemeral=True
                    )
                    return
                
                session_id = active_session["data"]["id"]
            
            # End the session
            result = await db_client.end_session(session_id=session_id)
            
            if not result.get("isSuccess", False):
                await interaction.followup.send(
                    f"Failed to end session: {result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            # Confirm to the user
            await interaction.followup.send(
                "✅ Lab session ended successfully.",
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}", exc_info=True)
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred while ending the session. Please try again later.",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "An error occurred while ending the session. Please try again later.",
                        ephemeral=True
                    )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error message: {follow_up_error}")
        
    async def list_sessions_callback(self, interaction: discord.Interaction, include_closed: Optional[bool] = False, limit: Optional[int] = 10):
        """Callback for the list_sessions command."""
        try:
            # Immediately acknowledge the interaction to prevent timeout
            await interaction.response.defer(ephemeral=True)
            
            user_id = str(interaction.user.id)
            
            # Get user's sessions
            sessions_result = await db_client.get_user_sessions(user_id=user_id)
            
            if not sessions_result.get("isSuccess", False):
                await interaction.followup.send(
                    f"Failed to fetch sessions: {sessions_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            sessions = sessions_result.get("data", [])
            
            if not sessions:
                await interaction.followup.send(
                    "You don't have any lab sessions yet. Use `/lab start` to create one.",
                    ephemeral=True
                )
                return
            
            # Filter sessions based on include_closed parameter
            if not include_closed:
                # Only include active and recently closed sessions
                sessions = [s for s in sessions if s.get("status") == "active" or s.get("status") == "ended"]
            
            # Apply limit
            sessions = sessions[:limit]
            
            # Create response embed
            embed = discord.Embed(
                title="Your Lab Sessions",
                description=f"Showing up to {limit} sessions" + (" (including closed)" if include_closed else ""),
                color=discord.Color.blue()
            )
            
            # Add active session if exists
            active_session = next((s for s in sessions if s.get("status") == "active"), None)
            if active_session:
                embed.add_field(
                    name="🟢 Active Session",
                    value=(
                        f"**{active_session.get('title')}**\n"
                        f"ID: {active_session.get('id')}\n"
                        f"Status: Active\n"
                        f"Privacy: {'Public' if active_session.get('is_public') else 'Private'}"
                    ),
                    inline=False
                )
            
            # Add other sessions
            other_sessions = [s for s in sessions if s.get("status") != "active"]
            if other_sessions:
                # Split sessions into chunks to avoid Discord's 1024 character limit per field
                MAX_SESSIONS_PER_FIELD = 3  # Adjust this based on your average session description length
                session_chunks = [other_sessions[i:i + MAX_SESSIONS_PER_FIELD] 
                                 for i in range(0, len(other_sessions), MAX_SESSIONS_PER_FIELD)]
                
                for i, chunk in enumerate(session_chunks):
                    other_list = []
                    for session in chunk:
                        status_emoji = "🔴" if session.get("status") == "ended" else "⚪"
                        other_list.append(
                            f"{status_emoji} **{session.get('title')}**\n"
                            f"ID: {session.get('id')}\n"
                            f"Status: {session.get('status', 'Unknown').title()}\n"
                            f"Privacy: {'Public' if session.get('is_public') else 'Private'}"
                        )
                    
                    field_name = "Your Other Sessions" if i == 0 else f"Your Other Sessions (continued {i+1})"
                    embed.add_field(
                        name=field_name,
                        value="\n\n".join(other_list),
                        inline=False
                    )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred while listing sessions.",
                        ephemeral=True
                    )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error message: {follow_up_error}")
        
    async def reopen_session_callback(
        self,
        interaction: discord.Interaction,
        session_id: str,
        confirm: Optional[bool] = False
    ):
        """Callback for the reopen_session command."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        if not confirm:
            await interaction.followup.send(
                "Please confirm that you want to reopen this session by using `confirm:true`",
                ephemeral=True
            )
            return
        
        user_id = str(interaction.user.id)
        
        try:
            # Check if user has an active session
            active_session_result = await db_client.get_active_session(user_id=user_id)
            logger.info(f"Active session check result: {active_session_result}")
            
            if active_session_result.get("isSuccess") and active_session_result.get("data"):
                # End the current session
                await db_client.end_session(
                    session_id=active_session_result["data"]["id"]
                )
                logger.info(f"Ended previous active session for user {user_id}")
            else:
                logger.info(f"No active session to end for user {user_id}")
            
            # Get the session to reopen
            session_result = await db_client.get_session(session_id=session_id)
            
            if not session_result.get("isSuccess") or not session_result.get("data"):
                logger.error(f"Session not found: {session_id}")
                await interaction.followup.send(
                    "Session not found. Please check the session ID and try again.",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            
            # Extract user ID from session data, handling both snake_case and camelCase
            session_user_id = session_data.get("user_id") or session_data.get("userId")
            
            # Verify ownership
            if session_user_id != user_id:
                logger.warning(f"User {user_id} attempted to reopen session {session_id} owned by {session_user_id}")
                await interaction.followup.send(
                    "You can only reopen your own sessions.",
                    ephemeral=True
                )
                return
            
            # Check if session is already active
            session_status = session_data.get("status")
            if session_status == "active":
                logger.info(f"Session {session_id} is already active")
                await interaction.followup.send(
                    "This session is already active.",
                    ephemeral=True
                )
                return
            
            # Reopen the session
            logger.info(f"Reopening session {session_id}")
            reopen_result = await db_client.reopen_session(session_id=session_id)
            
            if not reopen_result.get("isSuccess"):
                logger.error(f"Failed to reopen session: {reopen_result}")
                await interaction.followup.send(
                    f"Failed to reopen session: {reopen_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            reopened_data = reopen_result.get("data", {})
            logger.info(f"Successfully reopened session {session_id}")
            
            # Create response embed
            embed = discord.Embed(
                title="Session Reopened",
                description=f"Session '{session_data.get('title')}' has been reopened.",
                color=discord.Color.green()
            )
            embed.add_field(
                name="Session ID",
                value=session_id,
                inline=True
            )
            embed.add_field(
                name="Privacy",
                value="Public" if session_data.get("is_public") or session_data.get("isPublic") else "Private",
                inline=True
            )
            embed.add_field(
                name="Next Steps",
                value="Use `/lab agent create` to add agents or `/lab team_meeting` to begin a discussion.",
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error reopening session: {str(e)}", exc_info=True)
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred while reopening the session. Please try again later.",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "An error occurred while reopening the session. Please try again later.",
                        ephemeral=True
                    )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error message: {follow_up_error}")

async def setup(bot: commands.Bot):
    """Add the cog to the bot."""
    await bot.add_cog(LabSessionCommands(bot))
