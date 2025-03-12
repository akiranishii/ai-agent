import discord
from discord import app_commands
from discord.ext import commands
import logging
from typing import Optional, List, Dict
from datetime import datetime

from db_client import db_client

logger = logging.getLogger(__name__)

class LabTranscriptCommands(commands.Cog):
    """Commands for managing lab transcripts."""
    
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
        
        # Add transcript commands to the lab group
        self.lab_group.add_command(app_commands.Command(
            name="transcript_list",
            description="List all meeting transcripts for the current session",
            callback=self.transcript_list_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="transcript_view",
            description="View transcript for a specific meeting",
            callback=self.transcript_view_callback,
            extras={"cog": self}
        ))
        
        logger.info("Registered lab transcript commands")

    async def transcript_list_callback(self, interaction: discord.Interaction):
        """Callback for the transcript_list command."""
        try:
            # Immediately acknowledge the interaction to prevent timeout
            await interaction.response.defer(ephemeral=True)
            
            user_id = str(interaction.user.id)
            
            # Get active session
            session_result = await db_client.get_active_session(user_id=user_id)
            
            if not session_result.get("isSuccess") or not session_result.get("data"):
                await interaction.followup.send(
                    "You don't have an active session. Use `/lab start` to create one.",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            session_id = session_data.get("id")
            
            # List all meetings with transcripts for the session
            await self.list_transcripts(interaction, session_id)
        except Exception as e:
            logger.error(f"Error in transcript_list command: {e}")
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred. Please try again later.",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "An error occurred. Please try again later.",
                        ephemeral=True
                    )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error message: {follow_up_error}")

    async def transcript_view_callback(
        self, 
        interaction: discord.Interaction, 
        meeting_id: str,
        round_number: Optional[int] = None,
        agent_name: Optional[str] = None,
        format: Optional[str] = "embed"
    ):
        """Callback for the transcript_view command."""
        try:
            # Immediately acknowledge the interaction to prevent timeout
            await interaction.response.defer(ephemeral=True)
            
            # View transcript for the specified meeting
            await self.view_transcript(interaction, meeting_id, round_number, agent_name, format)
        except Exception as e:
            logger.error(f"Error in transcript_view command: {e}")
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(
                        "An error occurred. Please try again later.",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "An error occurred. Please try again later.",
                        ephemeral=True
                    )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error message: {follow_up_error}")

    # Helper methods
    async def list_transcripts(self, interaction: discord.Interaction, session_id: str):
        """List all meetings with transcripts for the session."""
        # List all meetings with transcripts for the session
        meetings_result = await db_client.get_session_meetings(session_id=session_id)
        
        if not meetings_result.get("isSuccess"):
            await interaction.followup.send(
                f"Failed to fetch meetings: {meetings_result.get('message', 'Unknown error')}",
                ephemeral=True
            )
            return
        
        meetings = meetings_result.get("data", [])
        
        if not meetings:
            await interaction.followup.send(
                "No meetings found in this session.",
                ephemeral=True
            )
            return
        
        # Create embed for meetings list
        embed = discord.Embed(
            title="Meeting Transcripts",
            description="Here are your meeting transcripts:",
            color=discord.Color.blue()
        )
        
        for meeting in meetings:
            parallel_suffix = f" (Run {meeting.get('parallel_index', 0) + 1})" if meeting.get('parallel_index', 0) > 0 else ""
            
            embed.add_field(
                name=f"Meeting {meeting.get('id')}{parallel_suffix}",
                value=(
                    f"**Agenda**: {meeting.get('agenda')}\n"
                    f"**Rounds**: {meeting.get('round_count')}\n"
                    f"**Status**: {'Completed' if meeting.get('is_completed') else 'In Progress'}\n"
                    f"{self._format_created_time(meeting.get('created_at'))}\n"
                    f"Use `/lab transcript view meeting_id:{meeting.get('id')}` to view transcript"
                ),
                inline=False
            )
        
        await interaction.followup.send(embed=embed, ephemeral=True)

    def _format_created_time(self, created_at):
        """Safely format a created_at timestamp for display."""
        try:
            if created_at and isinstance(created_at, str):
                timestamp = int(datetime.fromisoformat(created_at).timestamp())
                return f"**Created**: <t:{timestamp}:R>"
            else:
                return "**Created**: Unknown"
        except Exception as e:
            logger.warning(f"Error formatting timestamp: {e}")
            return "**Created**: Unknown"

    async def view_transcript(self, interaction: discord.Interaction, meeting_id: str, round_number: Optional[int] = None,
                          agent_name: Optional[str] = None, format: Optional[str] = "embed"):
        """View a specific meeting transcript."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        try:
            # Get meeting details
            meeting_result = await db_client.get_meeting(meeting_id=meeting_id)
            
            if not meeting_result.get("isSuccess") or not meeting_result.get("data"):
                await interaction.followup.send(
                    f"Failed to fetch meeting: {meeting_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            meeting = meeting_result.get("data", {})
            
            # Get transcripts for the meeting
            transcripts_result = await db_client.get_meeting_transcripts(
                meeting_id=meeting_id
            )
            
            if not transcripts_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Failed to fetch transcripts: {transcripts_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            # If we need to filter by round_number or agent_name, do it client-side
            transcripts = transcripts_result.get("data", [])
            if round_number is not None and transcripts:
                transcripts = [t for t in transcripts if t.get("roundNumber") == round_number]
            if agent_name and transcripts:
                transcripts = [t for t in transcripts if t.get("agentName") == agent_name]
            
            if not transcripts:
                await interaction.followup.send(
                    "No transcripts found for this meeting.",
                    ephemeral=True
                )
                return
            
            if format.lower() == "text":
                # Format as plain text
                lines = [
                    f"Meeting Transcript: {meeting.get('agenda')}\n",
                    f"Meeting ID: {meeting_id}\n",
                    "=" * 40 + "\n"
                ]
                
                current_round = None
                for transcript in transcripts:
                    if transcript.get("round_number") != current_round:
                        current_round = transcript.get("round_number")
                        lines.append(f"\nRound {current_round}:\n")
                        lines.append("-" * 20 + "\n")
                    
                    lines.append(
                        f"{transcript.get('agent_name')}: {transcript.get('content')}\n"
                    )
                
                # Split into chunks if needed (Discord has a 2000 char limit)
                content = "".join(lines)
                chunks = [content[i:i+1990] for i in range(0, len(content), 1990)]
                
                for i, chunk in enumerate(chunks):
                    is_first = i == 0
                    is_last = i == len(chunks) - 1
                    
                    if is_first:
                        await interaction.followup.send(chunk, ephemeral=True)
                    else:
                        await interaction.followup.send(chunk, ephemeral=True)
                    
            else:  # format == "embed"
                # Create embed for transcript
                embed = discord.Embed(
                    title=f"Meeting Transcript",
                    description=meeting.get('agenda'),
                    color=discord.Color.blue()
                )
                
                # Basic meeting info
                embed.add_field(
                    name="Meeting Information",
                    value=(
                        f"**Agenda**: {meeting.get('agenda')}\n"
                        f"**Rounds**: {meeting.get('round_count')}\n"
                        f"**Parallel Run**: {meeting.get('parallel_index', 0) + 1}\n"
                        f"**Status**: {'Completed' if meeting.get('is_completed') else 'In Progress'}\n"
                        f"{self._format_created_time(meeting.get('created_at'))}"
                    ),
                    inline=False
                )
                
                # Group messages by round
                rounds: Dict[int, List[Dict]] = {}
                for transcript in transcripts:
                    round_num = transcript.get("round_number", 0)
                    if round_num not in rounds:
                        rounds[round_num] = []
                    rounds[round_num].append(transcript)
                
                # Add each round to the embed
                for round_num in sorted(rounds.keys()):
                    round_messages = []
                    for transcript in rounds[round_num]:
                        round_messages.append(
                            f"**{transcript.get('agent_name')}**: {transcript.get('content')}"
                        )
                    
                    # Split round content if too long
                    round_content = "\n\n".join(round_messages)
                    if len(round_content) > 1024:
                        round_content = round_content[:1021] + "..."
                    
                    embed.add_field(
                        name=f"Round {round_num}",
                        value=round_content,
                        inline=False
                    )
                
                await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error in view_transcript: {e}")
            await interaction.followup.send(
                "An error occurred while processing your request. Please try again later.",
                ephemeral=True
            )

async def setup(bot: commands.Bot):
    """Add the cog to the bot."""
    await bot.add_cog(LabTranscriptCommands(bot)) 