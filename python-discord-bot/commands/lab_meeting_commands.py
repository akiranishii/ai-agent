import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Set

import discord
from discord import app_commands
from discord.ext import commands
from discord.interactions import Interaction

from db_client import db_client
from orchestrator import AgentOrchestrator
from llm_client import LLMClient, llm_client
from models import ModelConfig, LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

class LabMeetingCommands(commands.Cog):
    """Commands for managing lab meetings"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.orchestrator = AgentOrchestrator(llm_client)
        self.conversation_tasks = {}  # Dictionary to track conversation tasks
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
        
        # Add meeting commands directly to avoid intermediate callbacks
        # This helps prevent discord interaction timeouts
        self.lab_group.add_command(app_commands.Command(
            name="team_meeting",
            description="Start a team meeting in the current lab session",
            callback=self.team_meeting,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="end_team_meeting",
            description="End the current team meeting",
            callback=self.end_team_meeting,
            extras={"cog": self}
        ))
        
        logger.info("Registered lab meeting commands")

    @app_commands.describe(
        agenda="The main topic or question for discussion",
        rounds="Number of conversation rounds (default: 3)",
        parallel_meetings="Number of parallel runs (default: 1)",
        agent_list="Optional comma-separated list of agent names to include",
        auto_generate="Automatically generate PI, critic, and scientist agents",
        auto_scientist_count="Number of scientists to generate if auto_generate is true (default: 3)",
        auto_include_critic="Include a critic agent if auto_generate is true (default: true)",
        temperature_variation="Increase temperature variation for parallel runs (default: true)",
        live_mode="Show agent responses in real-time (default: true)",
        speakers_per_round="Number of agent speakers selected per round (default: all agents excluding PI)"
    )
    async def team_meeting(
        self,
        interaction: discord.Interaction,
        agenda: str,
        rounds: Optional[int] = 3,
        parallel_meetings: Optional[int] = 1,
        agent_list: Optional[str] = None,
        auto_generate: Optional[bool] = False,
        auto_scientist_count: Optional[int] = 3,
        auto_include_critic: Optional[bool] = True,
        temperature_variation: Optional[bool] = True,
        live_mode: Optional[bool] = True,
        speakers_per_round: Optional[int] = None
    ):
        """Start a multi-agent team meeting."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        user_id = str(interaction.user.id)
        
        try:
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
            
            # If auto_generate is true, create the agents
            if auto_generate:
                # Generate variables for the PI based on the agenda
                pi_variables = await llm_client.generate_agent_variables(
                    topic=agenda,
                    agent_type="principal_investigator"
                )
                
                # Create PI
                await db_client.create_agent(
                    session_id=session_id,
                    user_id=user_id,
                    name=ModelConfig.PRINCIPAL_INVESTIGATOR_ROLE,
                    role="Lead",
                    goal=pi_variables.get("goal"),
                    expertise=pi_variables.get("expertise"),
                    model="openai"
                )
                
                # Create Scientists with varied expertise for parallel runs
                # Keep track of created agent details for better diversity
                created_agents_info = []
                
                for i in range(auto_scientist_count):
                    # Generate variables for each scientist based on the agenda
                    # Add additional context to ensure each scientist is unique
                    diversity_context = [
                        "Create a scientist with expertise COMPLETELY DIFFERENT from previously generated scientists.",
                        f"This is scientist #{i+1} of {auto_scientist_count}."
                    ]
                    
                    # For the second and third scientists, specify that they should differ by including details of previous agents
                    if created_agents_info:
                        previous_agents_text = "\n\nPreviously created scientists:\n"
                        for j, agent_info in enumerate(created_agents_info):
                            previous_agents_text += f"{j+1}. {agent_info['name']} - Expertise: {agent_info['expertise']}" 
                            if agent_info.get('goal'):
                                previous_agents_text += f", Goal: {agent_info['goal']}"
                            previous_agents_text += "\n"
                        
                        diversity_context.append(f"Current team composition: {previous_agents_text}")
                        diversity_context.append("Your role must be complementary to the existing team and fill a knowledge gap.")
                        
                    # Add position-specific guidance for more diversity
                    if i == 0:
                        diversity_context.append("Create a scientist from a PHYSICAL SCIENCES domain (physics, chemistry, materials science, etc.) rather than biology or computer science.")
                    elif i == 1:
                        diversity_context.append("Create a scientist from an APPLIED SCIENCE field (engineering, robotics, energy systems, etc.) rather than theoretical domains.")
                    else:
                        diversity_context.append("Create a scientist from a completely different discipline like geology, astronomy, mathematics, or social sciences that can bring a unique perspective.")
                        
                    # Generate variables with the diversity context
                    scientist_variables = await llm_client.generate_agent_variables(
                        topic=agenda,
                        agent_type="scientist",
                        additional_context="\n".join(diversity_context)
                    )
                    
                    # Get the agent name, ensuring it's unique
                    agent_name = scientist_variables.get("agent_name", f"Scientist {i+1}")
                    agent_expertise = scientist_variables.get("expertise", "")
                    agent_goal = scientist_variables.get("goal", "")
                    
                    # Track this agent's details for future diversity
                    created_agents_info.append({
                        "name": agent_name,
                        "expertise": agent_expertise,
                        "goal": agent_goal
                    })
                    
                    # Create the agent
                    await db_client.create_agent(
                        session_id=session_id,
                        user_id=user_id,
                        name=agent_name,
                        role=ModelConfig.SCIENTIST_ROLE,
                        expertise=agent_expertise,
                        goal=agent_goal,
                        model="openai"
                    )
                
                # Create Critic if requested
                if auto_include_critic:
                    await db_client.create_agent(
                        session_id=session_id,
                        user_id=user_id,
                        name=ModelConfig.CRITIC_ROLE,
                        role="Critical Reviewer",
                        expertise="Critical analysis of scientific research, identification of methodological flaws, and evaluation of research validity",
                        goal="Ensure scientific rigor and identify potential weaknesses in proposed research approaches",
                        model="openai"
                    )
            
            # Get agents for the meeting
            if agent_list:
                agent_names = [name.strip() for name in agent_list.split(",")]
                agents_result = await db_client.get_agents_by_names(
                    session_id=session_id,
                    agent_names=agent_names
                )
            else:
                agents_result = await db_client.get_session_agents(
                    session_id=session_id,
                    user_id=str(interaction.user.id)
                )
            
            if not agents_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Failed to fetch agents: {agents_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            agents = agents_result.get("data", [])
            
            if not agents:
                await interaction.followup.send(
                    "No agents found for the meeting. Use `/lab agent create` to add agents or set auto_generate:true.",
                    ephemeral=True
                )
                return
            
            # Check for PI
            has_pi = any(
                agent.get("name") == ModelConfig.PRINCIPAL_INVESTIGATOR_ROLE or
                agent.get("role") == ModelConfig.PRINCIPAL_INVESTIGATOR_ROLE or
                agent.get("role") == "Lead" or
                "PI" in agent.get("name", "") or
                "Principal" in agent.get("name", "")
                for agent in agents
            )
            
            if not has_pi:
                await interaction.followup.send(
                    "A Principal Investigator is required for team meetings. Create one with `/lab agent create` or use auto_generate:true.",
                    ephemeral=True
                )
                return
            
            # Create meeting record(s)
            meetings = []
            for i in range(parallel_meetings):
                meeting_result = await db_client.create_meeting(
                    session_id=session_id,
                    title=f"Meeting {i+1} on: {agenda}",
                    agenda=agenda,
                    max_rounds=rounds,
                    parallel_index=i
                )
                
                if meeting_result.get("isSuccess"):
                    meetings.append(meeting_result.get("data"))
            
            if not meetings:
                await interaction.followup.send(
                    "Failed to create meeting records.",
                    ephemeral=True
                )
                return
            
            # Start the meeting(s)
            for i, meeting in enumerate(meetings):
                meeting_id = meeting.get("id")
                
                # Initialize the orchestrator with the meeting
                await self.orchestrator.initialize_meeting(
                    meeting_id=meeting_id,
                    session_id=session_id,
                    agents=agents,
                    agenda=agenda,
                    round_count=rounds,
                    parallel_index=i,
                    total_parallel_meetings=parallel_meetings
                )
                
                # Start the conversation in a background task
                task = asyncio.create_task(
                    self.orchestrator.start_conversation(
                        meeting_id=meeting_id,
                        interaction=interaction,
                        live_mode=live_mode,
                        conversation_length=speakers_per_round
                    )
                )
                
                # Store the task with the meeting ID as the key
                self.conversation_tasks[meeting_id] = task
                
                # Add a done callback to clean up the task when it completes
                task.add_done_callback(lambda t, mid=meeting_id: self._cleanup_conversation_task(mid, t))
            
            # Create response embed
            embed = discord.Embed(
                title="Team Meeting Started",
                description=f"Started {parallel_meetings} parallel meeting(s) with {len(agents)} agents.",
                color=discord.Color.green()
            )
            
            embed.add_field(
                name="Agenda",
                value=agenda,
                inline=False
            )
            
            embed.add_field(
                name="Rounds per Run",
                value=str(rounds),
                inline=True
            )
            
            embed.add_field(
                name="Parallel Runs",
                value=str(parallel_meetings),
                inline=True
            )
            
            # Add live_mode field
            embed.add_field(
                name="Live Mode",
                value="On" if live_mode else "Off",
                inline=True
            )
            
            agent_list = "\n".join([
                f"- {agent.get('name')} ({agent.get('expertise', 'General expertise')})"
                for agent in agents
            ])
            embed.add_field(
                name="Participating Agents",
                value=agent_list,
                inline=False
            )
            
            if parallel_meetings > 1:
                embed.add_field(
                    name="Note",
                    value="Each parallel run will explore different perspectives. A combined summary will be generated when all runs complete.",
                    inline=False
                )
            
            embed.add_field(
                name="Meeting Details",
                value=(
                    f"**Rounds**: {rounds}\n"
                    f"**Parallel Runs**: {parallel_meetings}\n"
                    f"**Speakers Per Round**: {speakers_per_round if speakers_per_round is not None else 'All'}\n"
                    f"**Live Mode**: {'On' if live_mode else 'Off'}"
                ),
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error in team_meeting command: {e}")
            await interaction.followup.send(
                "An error occurred while starting the team meeting. Please try again later.",
                ephemeral=True
            )

    @app_commands.describe(
        meeting_id="Optional meeting ID if multiple meetings are active",
        end_all_parallel="End all parallel runs of the meeting (default: true)",
        force_combined_summary="Force generation of a combined summary even for single meetings (for testing)",
        is_public="Make the session public after ending the meeting (default: false)"
    )
    async def end_team_meeting(
        self,
        interaction: discord.Interaction,
        meeting_id: Optional[str] = None,
        end_all_parallel: Optional[bool] = True,
        force_combined_summary: Optional[bool] = False,
        is_public: Optional[bool] = False
    ):
        """End an ongoing team meeting."""
        # Immediately defer the response to prevent timeout
        try:
            # Immediately acknowledge that we're processing the command
            await interaction.response.defer(ephemeral=True)
            
            # Send a quick follow-up to let the user know we're working
            initial_response = await interaction.followup.send(
                "Processing your request to end the meeting(s)... This might take a moment.",
                ephemeral=True
            )
        except Exception as e:
            logger.warning(f"Could not defer response or send immediate followup: {e}")
            # Continue anyway - we'll try to handle errors below
        
        user_id = str(interaction.user.id)
        logger.info(f"User {user_id} initiated end_team_meeting command")
        
        try:
            # Get active session
            session_result = await db_client.get_active_session(user_id=user_id)
            
            if not session_result.get("isSuccess") or not session_result.get("data"):
                try:
                    await interaction.followup.send(
                        "You don't have an active session.",
                        ephemeral=True
                    )
                except:
                    if interaction.channel:
                        await interaction.channel.send(
                            f"<@{user_id}> You don't have an active session."
                        )
                return
            
            session_data = session_result.get("data", {})
            session_id = session_data.get("id")
            logger.info(f"Found active session {session_id} for user {user_id}")
            
            # Update session visibility if requested
            if is_public:
                logger.info(f"Marking session {session_id} as public")
                await db_client.update_session(
                    session_id=session_id,
                    updates={"isPublic": True}  # Use camelCase to match Next.js API expectations
                )
            
            # Get active meetings from the database
            db_meetings = []
            if meeting_id:
                # Get specific meeting and its parallel runs if requested
                meeting_result = await db_client.get_meeting(meeting_id=meeting_id)
                if end_all_parallel and meeting_result.get("isSuccess"):
                    parallel_result = await db_client.get_parallel_meetings(
                        session_id=session_id,
                        base_meeting_id=meeting_result.get("data", {}).get("id")
                    )
                    db_meetings = parallel_result.get("data", []) if parallel_result.get("isSuccess") else []
                else:
                    db_meetings = [meeting_result.get("data")] if meeting_result.get("isSuccess") else []
            else:
                # Get all active meetings for the session
                meetings_result = await db_client.get_active_meetings(session_id=session_id)
                db_meetings = meetings_result.get("data", []) if meetings_result.get("isSuccess") else []
            
            # Also check the orchestrator's active meetings
            orchestrator_meetings = []
            db_meeting_ids = [m.get("id") for m in db_meetings if m and m.get("id")]
            
            # Get meetings from orchestrator that match this session
            for orch_meeting_id, orch_meeting_data in self.orchestrator.active_meetings.items():
                if orch_meeting_data.get("session_id") == session_id:
                    # Check if this meeting ID is already in our DB results
                    if orch_meeting_id not in db_meeting_ids:
                        logger.info(f"Found active meeting {orch_meeting_id} in orchestrator that's not in DB results")
                        # Create a meeting dict similar to what the DB would return
                        orchestrator_meetings.append({
                            "id": orch_meeting_id,
                            "sessionId": session_id,
                            "agenda": orch_meeting_data.get("agenda", "No agenda"),
                            "roundCount": orch_meeting_data.get("current_round", 0),
                            "parallelIndex": orch_meeting_data.get("parallel_index", 0)
                        })
            
            # Combine database and orchestrator meetings
            meetings = db_meetings + orchestrator_meetings
            
            if not meetings:
                try:
                    await interaction.followup.send(
                        "No active meetings found to end.",
                        ephemeral=True
                    )
                except:
                    if interaction.channel:
                        await interaction.channel.send(
                            f"<@{user_id}> No active meetings found to end."
                        )
                return
            
            logger.info(f"Found {len(meetings)} active meetings to end")
            
            # End each meeting
            ended_meetings = []
            for meeting in meetings:
                meeting_id = meeting.get("id")
                
                try:
                    # Cancel any running conversation tasks in LabMeetingCommands
                    if meeting_id in self.conversation_tasks:
                        logger.info(f"Cancelling conversation task for meeting {meeting_id} in LabMeetingCommands")
                        task = self.conversation_tasks[meeting_id]
                        if not task.done():
                            task.cancel()
                        self.conversation_tasks.pop(meeting_id, None)
                    
                    # Also check for tasks in QuickstartCommand
                    quickstart_command = None
                    for cog in self.bot.cogs.values():
                        if cog.__class__.__name__ == "QuickstartCommand":
                            quickstart_command = cog
                            break
                    
                    if quickstart_command and hasattr(quickstart_command, 'conversation_tasks'):
                        if meeting_id in quickstart_command.conversation_tasks:
                            logger.info(f"Cancelling conversation task for meeting {meeting_id} in QuickstartCommand")
                            task = quickstart_command.conversation_tasks[meeting_id]
                            if not task.done():
                                task.cancel()
                            quickstart_command.conversation_tasks.pop(meeting_id, None)
                
                    # Stop the orchestrator
                    logger.info(f"Ending meeting {meeting_id} in orchestrator")
                    await self.orchestrator.end_conversation(meeting_id=meeting_id)
                    
                    # Update meeting status
                    end_result = await db_client.end_meeting(meeting_id=meeting_id)
                    
                    if end_result.get("isSuccess"):
                        ended_meetings.append(meeting)
                        logger.info(f"Successfully ended meeting {meeting_id}")
                    else:
                        logger.error(f"Failed to end meeting {meeting_id}: {end_result.get('message')}")
                except Exception as e:
                    logger.error(f"Error ending meeting {meeting_id}: {e}")
            
            # Create response embed
            embed = discord.Embed(
                title="Team Meeting(s) Ended",
                description=f"Ended {len(ended_meetings)} meeting(s).",
                color=discord.Color.blue()
            )
            
            # Group meetings by parallel index
            parallel_groups = {}
            for meeting in ended_meetings:
                parallel_index = meeting.get("parallelIndex", 0)  # Note: API uses camelCase
                if parallel_index not in parallel_groups:
                    parallel_groups[parallel_index] = []
                parallel_groups[parallel_index].append(meeting)
            
            for parallel_index, group in parallel_groups.items():
                embed.add_field(
                    name=f"Parallel Run {parallel_index + 1}",
                    value="\n".join([
                        f"Meeting {m.get('id')}: {m.get('agenda')}\n"
                        f"Rounds completed: {m.get('roundCount', 0)}"  # Note: API uses camelCase
                        for m in group
                    ]),
                    inline=False
                )
            
            embed.add_field(
                name="View Transcripts",
                value="Use `/lab transcript list` to view the meeting transcripts.",
                inline=False
            )
            
            # Add session visibility information
            embed.add_field(
                name="Session Status",
                value=f"Privacy: {'Public' if is_public else 'Private'}",
                inline=False
            )
            
            if len(parallel_groups) > 1 or force_combined_summary:
                embed.add_field(
                    name="Combined Summary",
                    value="A combined summary of all runs will be generated now...",
                    inline=False
                )
            
            # Start combined summary generation if needed
            # We need to do this before sending final response to avoid early return
            combined_summary_task = None
            if (len(parallel_groups) > 1 or force_combined_summary) and interaction.channel:
                logger.info(f"Creating combined summary task for {len(ended_meetings)} meetings (forced: {force_combined_summary})")
                combined_summary_task = asyncio.create_task(
                    self._generate_combined_summary_for_ended_meetings(
                        interaction=interaction,
                        ended_meetings=ended_meetings,
                        channel=interaction.channel
                    )
                )
            elif len(parallel_groups) > 1 or force_combined_summary:
                logger.error("Cannot generate combined summary - no channel available")
            
            # Send final response (private to user)
            try:
                # Try to edit the initial response instead of sending a new one
                if 'initial_response' in locals():
                    try:
                        await initial_response.edit(content=None, embed=embed)
                        logger.info("Successfully edited initial response with final results")
                        # Don't return here, as we may need to wait for the combined summary task
                    except Exception as edit_error:
                        logger.warning(f"Could not edit initial response: {edit_error}")
                        # Fall through to other methods
                
                # Try to send a new followup if editing failed or no initial response
                if 'initial_response' not in locals() or 'edit_error' in locals():
                    await interaction.followup.send(
                        embed=embed,
                        ephemeral=True
                    )
                    logger.info("Successfully sent final response as a new followup")
            except discord.NotFound as webhook_error:
                logger.warning(f"Webhook expired: {webhook_error}")
                # Interaction webhook expired, send to channel instead
                if interaction.channel:
                    await interaction.channel.send(
                        f"<@{user_id}> Meetings ended successfully.",
                        embed=embed
                    )
                    logger.info("Sent fallback message to channel after webhook expired")
            except Exception as e:
                logger.warning(f"Could not send final followup: {e}")
                # If interaction is expired, try to send to channel
                if interaction.channel:
                    await interaction.channel.send(
                        f"<@{user_id}> Meetings ended successfully.",
                        embed=embed
                    )
                    logger.info("Sent fallback message to channel after error")
            
            # Note: We're no longer generating the summary here since we do it before sending the final response
            # This ensures the task is created even if we have early returns
            
        except Exception as e:
            logger.error(f"Error in end_team_meeting command: {e}")
            # Try to send an error message any way we can
            try:
                await interaction.followup.send(
                    f"An error occurred while ending the meeting: {e}",
                    ephemeral=True
                )
            except:
                if interaction.channel:
                    await interaction.channel.send(
                        f"<@{user_id}> An error occurred while ending the meeting: {e}"
                    )
                    logger.info("Sent error message to channel after followup failed")

    async def _generate_combined_summary_for_ended_meetings(self, interaction, ended_meetings, channel):
        """Helper method to generate combined summary after meetings end.
        
        This runs as a separate task to prevent Discord interaction timeouts.
        """
        try:
            # IMPORTANT: Send this message to everyone in the channel
            await channel.send(
                "🔄 **Generating combined summary of all runs. This might take a moment...**"
            )
            
            # Log all meetings for debugging
            logger.info(f"Found {len(ended_meetings)} ended meetings")
            for meeting in ended_meetings:
                logger.info(f"Meeting ID: {meeting.get('id')}, Parallel Index: {meeting.get('parallelIndex', 0)}, Agenda: {meeting.get('agenda', 'Unknown')}")
            
            # Collect meeting summaries directly - we'll try multiple methods
            meeting_summaries = []
            
            # First method: Try to get summaries from orchestrator's active_meetings
            for meeting in ended_meetings:
                meeting_id = meeting.get("id")
                parallel_idx = meeting.get("parallelIndex", 0)
                
                # Get meeting data from orchestrator
                meeting_data = self.orchestrator.active_meetings.get(meeting_id)
                if meeting_data and meeting_data.get("summary"):
                    logger.info(f"Found summary in orchestrator for meeting {meeting_id}")
                    meeting_summaries.append({
                        "index": parallel_idx + 1,  # 1-based for display
                        "summary": meeting_data.get("summary")
                    })
            
            # Second method: If we couldn't get all summaries from orchestrator, try to get them from transcripts
            if len(meeting_summaries) < len(ended_meetings):
                logger.info(f"Missing some summaries from orchestrator. Found {len(meeting_summaries)}/{len(ended_meetings)}")
                for meeting in ended_meetings:
                    meeting_id = meeting.get("id")
                    parallel_idx = meeting.get("parallelIndex", 0)
                    
                    # Skip if we already have summary for this meeting
                    if any(s["index"] == parallel_idx + 1 for s in meeting_summaries):
                        continue
                        
                    # Try to get from transcripts
                    try:
                        transcripts_result = await db_client.get_meeting_transcripts(meeting_id=meeting_id)
                        if transcripts_result.get("isSuccess") and transcripts_result.get("data"):
                            # Find the summary transcript (usually has round number 9999 or -1)
                            summary_transcript = next(
                                (t for t in transcripts_result.get("data", []) 
                                 if t.get("agentName") == "Summary Agent" or 
                                    t.get("roundNumber") in [9999, -1, 999]),
                                None
                            )
                            
                            if summary_transcript:
                                logger.info(f"Found summary in transcript for meeting {meeting_id}")
                                meeting_summaries.append({
                                    "index": parallel_idx + 1,
                                    "summary": summary_transcript.get("content")
                                })
                    except Exception as e:
                        logger.error(f"Error getting transcripts for meeting {meeting_id}: {e}")
            
            # Third method (fallback): Check if we can extract summaries from final messages in the orchestrator
            if len(meeting_summaries) < len(ended_meetings):
                logger.info(f"Still missing summaries. Found {len(meeting_summaries)}/{len(ended_meetings)}")
                for meeting in ended_meetings:
                    meeting_id = meeting.get("id")
                    parallel_idx = meeting.get("parallelIndex", 0)
                    
                    # Skip if we already have summary for this meeting
                    if any(s["index"] == parallel_idx + 1 for s in meeting_summaries):
                        continue
                    
                    # Try to extract from conversation history
                    meeting_data = self.orchestrator.active_meetings.get(meeting_id)
                    if meeting_data and meeting_data.get("conversation_history"):
                        logger.info(f"Attempting to extract summary from conversation history for meeting {meeting_id}")
                        history = meeting_data.get("conversation_history", "")
                        
                        # Look for final summary marker
                        final_summary_match = re.search(r"=== FINAL SUMMARY ===\s*(.*?)($|\n\n)", history, re.DOTALL)
                        if final_summary_match:
                            logger.info(f"Extracted summary from conversation history for meeting {meeting_id}")
                            summary_text = final_summary_match.group(1).strip()
                            meeting_summaries.append({
                                "index": parallel_idx + 1,
                                "summary": summary_text
                            })
            
            logger.info(f"Found {len(meeting_summaries)} meeting summaries for combined summary")
            
            # Generate combined summary if we have at least 1 meeting summary
            # Changed from 2 to 1 to handle forced combined summary for single meetings
            if len(meeting_summaries) >= 1:
                try:
                    # Format summaries
                    formatted_summaries = []
                    for summary_data in sorted(meeting_summaries, key=lambda x: x["index"]):
                        if len(meeting_summaries) > 1:
                            # Multiple meetings - use run labels
                            formatted_summaries.append(f"RUN #{summary_data['index']} SUMMARY:\n{summary_data['summary']}")
                        else:
                            # Single meeting - omit the run label
                            formatted_summaries.append(summary_data['summary'])
                    
                    # Create prompt
                    messages = [
                        LLMMessage(
                            role="system",
                            content="You are an expert research synthesizer. Your task is to combine multiple parallel brainstorming sessions into a cohesive summary."
                        ),
                        LLMMessage(
                            role="user",
                            content=(
                                f"Topic: {ended_meetings[0].get('agenda', 'Unknown Topic')}\n\n"
                                f"Individual Run Summaries:\n{'-'*50}\n" + 
                                f"\n{'-'*50}\n".join(formatted_summaries) + 
                                f"\n{'-'*50}\n\n"
                                "Provide a synthesis that includes:\n"
                                "1. Common themes and consensus across runs\n"
                                "2. Unique insights and novel approaches from each run\n"
                                "3. Contrasting perspectives and alternative solutions\n"
                                "4. Integrated conclusions and recommendations\n"
                                "5. Key areas for further investigation"
                            )
                        )
                    ]
                    
                    # Generate combined summary
                    logger.info("Calling LLM directly to generate combined summary")
                    response = await llm_client.generate_response(
                        provider=LLMProvider.OPENAI,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    combined_summary = response.content
                    logger.info(f"Combined summary generated: {len(combined_summary)} chars")
                    
                    # Send the combined summary
                    # First message with header and separator
                    await channel.send("───────────────────────────────────────────────")
                    await channel.send("## 🌟 **FINAL SYNERGIZED SUMMARY** 🌟")
                    await channel.send("───────────────────────────────────────────────")
                    
                    header_embed = discord.Embed(
                        title="Combined Insights from All Meetings",
                        description=f"Topic: {ended_meetings[0].get('agenda', 'Unknown')}",
                        color=discord.Color.brand_green()  # Use a distinctive color
                    )
                    
                    # Add debug info about which meetings were included
                    if len(meeting_summaries) > 1:
                        run_info = "\n".join([f"Run #{s['index']}" for s in meeting_summaries])
                        header_embed.add_field(
                            name="Synthesized from",
                            value=run_info,
                            inline=False
                        )
                    
                    # Send the header to the channel
                    await channel.send(embed=header_embed)
                    
                    # Split long summaries into chunks if needed
                    MAX_DISCORD_MESSAGE_LENGTH = 1900  # Discord limit is 2000, leaving some room
                    summary_chunks = [combined_summary[i:i+MAX_DISCORD_MESSAGE_LENGTH] 
                                    for i in range(0, len(combined_summary), MAX_DISCORD_MESSAGE_LENGTH)]
                    
                    # Send each chunk in a separate message
                    for i, chunk in enumerate(summary_chunks):
                        # Add part numbers if multiple chunks
                        prefix = f"**Part {i+1}/{len(summary_chunks)}:**\n" if len(summary_chunks) > 1 else ""
                        
                        # Create an embed for the content to make it stand out more
                        content_embed = discord.Embed(
                            description=prefix + chunk,
                            color=discord.Color.brand_green()  # Match the header color
                        )
                        
                        # Send to channel as an embed for better formatting
                        await channel.send(embed=content_embed)
                        
                    # Add a final separator
                    await channel.send("───────────────────────────────────────────────")
                
                except Exception as e:
                    logger.error(f"Error generating combined summary: {e}")
                    await channel.send(
                        "❌ An error occurred while generating the combined summary. Please check the individual meeting summaries in each thread."
                    )
            else:
                logger.warning(f"Not enough meeting summaries found ({len(meeting_summaries)}/{len(ended_meetings)})")
                await channel.send(
                    "⚠️ Unable to generate combined summary - couldn't find enough meeting summaries. Please check the individual summaries in each meeting thread."
                )
        
        except Exception as e:
            logger.error(f"Error in _generate_combined_summary_for_ended_meetings: {e}")
            try:
                await channel.send(
                    "❌ An error occurred while generating the combined summary. You can try manually ending the meetings with `/lab end_team_meeting force_combined_summary:true`."
                )
            except:
                logger.error("Could not send error message to channel")

    async def generate_and_post_combined_summary(self, interaction, ended_meetings, orchestrator):
        """Generate and post a combined summary of all parallel meetings.
        
        This method is called by the orchestrator when all parallel meetings in a group have completed.
        It extracts summaries from the meetings and generates a combined summary that is posted
        to the Discord channel.
        
        Args:
            interaction: The Discord interaction object
            ended_meetings: List of meeting data objects that have ended
            orchestrator: Reference to the orchestrator that called this method
        """
        try:
            logger.info(f"Generating combined summary for {len(ended_meetings)} meetings")
            
            # Get the channel to send messages to
            channel = interaction.channel
            
            # Log all meetings for debugging
            for meeting in ended_meetings:
                logger.info(f"Meeting ID: {meeting.get('id')}, Parallel Index: {meeting.get('parallelIndex', 0)}, Agenda: {meeting.get('agenda', 'Unknown')}")
            
            # Collect meeting summaries directly - we'll try multiple methods
            meeting_summaries = []
            
            # First method: Try to get summaries from orchestrator's active_meetings
            for meeting in ended_meetings:
                meeting_id = meeting.get("id")
                parallel_idx = meeting.get("parallelIndex", 0)
                
                # Get meeting data from orchestrator
                meeting_data = orchestrator.active_meetings.get(meeting_id)
                if meeting_data and meeting_data.get("summary"):
                    logger.info(f"Found summary in orchestrator for meeting {meeting_id}")
                    meeting_summaries.append({
                        "index": parallel_idx + 1,  # 1-based for display
                        "summary": meeting_data.get("summary")
                    })
            
            # Second method: If we couldn't get all summaries from orchestrator, try to get them from transcripts
            if len(meeting_summaries) < len(ended_meetings):
                logger.info(f"Missing some summaries from orchestrator. Found {len(meeting_summaries)}/{len(ended_meetings)}")
                for meeting in ended_meetings:
                    meeting_id = meeting.get("id")
                    parallel_idx = meeting.get("parallelIndex", 0)
                    
                    # Skip if we already have summary for this meeting
                    if any(s["index"] == parallel_idx + 1 for s in meeting_summaries):
                        continue
                        
                    # Try to get from transcripts
                    try:
                        transcripts_result = await db_client.get_meeting_transcripts(meeting_id=meeting_id)
                        if transcripts_result.get("isSuccess") and transcripts_result.get("data"):
                            # Find the summary transcript (usually has round number 9999 or -1)
                            summary_transcript = next(
                                (t for t in transcripts_result.get("data", []) 
                                    if t.get("agentName") == "Summary Agent" or 
                                    t.get("roundNumber") in [9999, -1, 999]),
                                None
                            )
                            
                            if summary_transcript:
                                logger.info(f"Found summary in transcript for meeting {meeting_id}")
                                meeting_summaries.append({
                                    "index": parallel_idx + 1,
                                    "summary": summary_transcript.get("content")
                                })
                    except Exception as e:
                        logger.error(f"Error getting transcripts for meeting {meeting_id}: {e}")
            
            # Third method (fallback): Check if we can extract summaries from final messages in the orchestrator
            if len(meeting_summaries) < len(ended_meetings):
                logger.info(f"Still missing summaries. Found {len(meeting_summaries)}/{len(ended_meetings)}")
                for meeting in ended_meetings:
                    meeting_id = meeting.get("id")
                    parallel_idx = meeting.get("parallelIndex", 0)
                    
                    # Skip if we already have summary for this meeting
                    if any(s["index"] == parallel_idx + 1 for s in meeting_summaries):
                        continue
                    
                    # Try to extract from conversation history
                    meeting_data = orchestrator.active_meetings.get(meeting_id)
                    if meeting_data and meeting_data.get("conversation_history"):
                        logger.info(f"Attempting to extract summary from conversation history for meeting {meeting_id}")
                        history = meeting_data.get("conversation_history", "")
                        
                        # Look for final summary marker
                        final_summary_match = re.search(r"=== FINAL SUMMARY ===\s*(.*?)($|\n\n)", history, re.DOTALL)
                        if final_summary_match:
                            logger.info(f"Extracted summary from conversation history for meeting {meeting_id}")
                            summary_text = final_summary_match.group(1).strip()
                            meeting_summaries.append({
                                "index": parallel_idx + 1,
                                "summary": summary_text
                            })
            
            logger.info(f"Found {len(meeting_summaries)} meeting summaries for combined summary")
            
            # Generate combined summary if we have at least 2 meeting summaries
            if len(meeting_summaries) >= 2:
                try:
                    # Format summaries
                    formatted_summaries = []
                    for summary_data in sorted(meeting_summaries, key=lambda x: x["index"]):
                        formatted_summaries.append(f"RUN #{summary_data['index']} SUMMARY:\n{summary_data['summary']}")
                    
                    # Create prompt
                    messages = [
                        LLMMessage(
                            role="system",
                            content="You are an expert research synthesizer. Your task is to combine multiple parallel brainstorming sessions into a cohesive summary."
                        ),
                        LLMMessage(
                            role="user",
                            content=(
                                f"Topic: {ended_meetings[0].get('agenda', 'Unknown Topic')}\n\n"
                                f"Individual Run Summaries:\n{'-'*50}\n" + 
                                f"\n{'-'*50}\n".join(formatted_summaries) + 
                                f"\n{'-'*50}\n\n"
                                "Provide a synthesis that includes:\n"
                                "1. Common themes and consensus across runs\n"
                                "2. Unique insights and novel approaches from each run\n"
                                "3. Contrasting perspectives and alternative solutions\n"
                                "4. Integrated conclusions and recommendations\n"
                                "5. Key areas for further investigation"
                            )
                        )
                    ]
                    
                    # Generate combined summary
                    logger.info("Calling LLM directly to generate combined summary")
                    response = await llm_client.generate_response(
                        provider=LLMProvider.OPENAI,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    combined_summary = response.content
                    logger.info(f"Combined summary generated: {len(combined_summary)} chars")
                    
                    # Send the combined summary
                    # First message with header and separator
                    await channel.send("───────────────────────────────────────────────")
                    await channel.send("## 🌟 **FINAL SYNERGIZED SUMMARY** 🌟")
                    await channel.send("───────────────────────────────────────────────")
                    
                    header_embed = discord.Embed(
                        title="Combined Insights from All Meetings",
                        description=f"Topic: {ended_meetings[0].get('agenda', 'Unknown')}",
                        color=discord.Color.brand_green()  # Use a distinctive color
                    )
                    
                    # Add info about which meetings were included
                    run_info = "\n".join([f"Run #{s['index']}" for s in meeting_summaries])
                    header_embed.add_field(
                        name="Synthesized from",
                        value=run_info,
                        inline=False
                    )
                    
                    # Send the header to the channel
                    await channel.send(embed=header_embed)
                    
                    # Split long summaries into chunks if needed
                    MAX_DISCORD_MESSAGE_LENGTH = 1900  # Discord limit is 2000, leaving some room
                    summary_chunks = [combined_summary[i:i+MAX_DISCORD_MESSAGE_LENGTH] 
                                    for i in range(0, len(combined_summary), MAX_DISCORD_MESSAGE_LENGTH)]
                    
                    # Send each chunk in a separate message
                    for i, chunk in enumerate(summary_chunks):
                        # Add part numbers if multiple chunks
                        prefix = f"**Part {i+1}/{len(summary_chunks)}:**\n" if len(summary_chunks) > 1 else ""
                        
                        # Create an embed for the content to make it stand out more
                        content_embed = discord.Embed(
                            description=prefix + chunk,
                            color=discord.Color.brand_green()  # Match the header color
                        )
                        
                        # Send to channel as an embed for better formatting
                        await channel.send(embed=content_embed)
                        
                    # Add a final separator
                    await channel.send("───────────────────────────────────────────────")
                
                except Exception as e:
                    logger.error(f"Error generating combined summary: {e}")
                    await channel.send(
                        "❌ An error occurred while generating the combined summary. Please check the individual meeting summaries in each thread."
                    )
            else:
                logger.warning(f"Not enough meeting summaries found ({len(meeting_summaries)}/{len(ended_meetings)})")
                await channel.send(
                    "⚠️ Unable to generate combined summary - couldn't find enough meeting summaries. Please check the individual summaries in each meeting thread."
                )
        
        except Exception as e:
            logger.error(f"Error in generate_and_post_combined_summary: {e}")
            try:
                # Try to send a message to the channel if available
                if hasattr(interaction, "channel") and interaction.channel:
                    await interaction.channel.send(
                        "❌ An error occurred while generating the combined summary. You can try manually ending the meetings with `/lab end_team_meeting force_combined_summary:true`."
                    )
            except Exception:
                # If we can't send a message, just log it
                logger.error("Could not send error message to channel")

    async def _cleanup_conversation_task(self, meeting_id, task):
        """Callback method to clean up conversation tasks."""
        try:
            # Remove the task from the conversation_tasks dictionary
            self.conversation_tasks.pop(meeting_id, None)
            
            # Check if the task completed successfully
            if task.exception():
                logger.error(f"Conversation task for meeting {meeting_id} failed with exception: {task.exception()}")
            else:
                logger.info(f"Conversation task for meeting {meeting_id} completed successfully")
        except Exception as e:
            logger.error(f"Error cleaning up conversation task for meeting {meeting_id}: {e}")

async def setup(bot: commands.Bot):
    """Add the cog to the bot."""
    await bot.add_cog(LabMeetingCommands(bot)) 