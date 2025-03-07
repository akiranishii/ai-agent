import discord
from discord import app_commands
from discord.ext import commands
import logging
from typing import Optional, List

from db_client import db_client
from llm_client import LLMClient, llm_client
from models import ModelConfig, LLMMessage, LLMProvider
from orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

class LabMeetingCommands(commands.Cog):
    """Commands for managing lab meetings"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.orchestrator = AgentOrchestrator(llm_client)
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
        
        # Add meeting commands to the lab group
        self.lab_group.add_command(app_commands.Command(
            name="team_meeting",
            description="Start a team meeting in the current lab session",
            callback=self.start_team_meeting_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="end_team_meeting",
            description="End the current team meeting",
            callback=self.end_team_meeting_callback,
            extras={"cog": self}
        ))
        
        logger.info("Registered lab meeting commands")

    @app_commands.command(
        name="team_meeting",
        description="Start a multi-agent conversation in the current lab session"
    )
    @app_commands.describe(
        agenda="The main topic or question for discussion",
        rounds="Number of conversation rounds (default: 3)",
        parallel_meetings="Number of parallel runs (default: 1)",
        agent_list="Optional comma-separated list of agent names to include",
        auto_generate="Automatically generate PI, critic, and scientist agents",
        auto_scientist_count="Number of scientists to generate if auto_generate is true (default: 3)",
        auto_include_critic="Include a critic agent if auto_generate is true (default: true)",
        temperature_variation="Increase temperature variation for parallel runs (default: true)",
        live_mode="Show agent responses in real-time (default: true)"
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
        live_mode: Optional[bool] = True
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
                # Keep track of created agent names to avoid duplicates
                created_agent_names = set()
                
                for i in range(auto_scientist_count):
                    # Generate variables for each scientist based on the agenda
                    # Add additional context to ensure each scientist is unique
                    diversity_context = [
                        "Create a different type of scientist than previously generated.",
                        "Ensure each scientist has a distinct specialty.",
                        f"This is scientist #{i+1} of {auto_scientist_count}."
                    ]
                    
                    # For the second and third scientists, specify that they should differ
                    if i > 0:
                        diversity_context.append(f"Previously created: {', '.join(created_agent_names)}")
                        
                    # Add position-specific guidance
                    if i == 0:
                        topic_focus = f"Focus on core aspects of {agenda}"
                    elif i == 1:
                        topic_focus = f"Focus on technical or methodological aspects of {agenda}"
                    else:
                        topic_focus = f"Focus on interdisciplinary or novel aspects of {agenda}"
                        
                    diversity_context.append(topic_focus)
                    
                    # Generate variables with the diversity context
                    scientist_variables = await llm_client.generate_agent_variables(
                        topic=agenda,
                        agent_type="scientist",
                        additional_context="\n".join(diversity_context)
                    )
                    
                    # Get the agent name, ensuring it's unique
                    agent_name = scientist_variables.get("agent_name", f"Scientist {i+1}")
                    
                    # If we somehow get a duplicate, modify the name to make it unique
                    counter = 1
                    original_name = agent_name
                    while agent_name in created_agent_names:
                        agent_name = f"{original_name} {counter}"
                        counter += 1
                    
                    # Add to our tracking set
                    created_agent_names.add(agent_name)
                    
                    # Create the agent
                    await db_client.create_agent(
                        session_id=session_id,
                        user_id=user_id,
                        name=agent_name,
                        role=ModelConfig.SCIENTIST_ROLE,
                        expertise=scientist_variables.get("expertise"),
                        goal=scientist_variables.get("goal"),
                        model="openai"
                    )
                
                # Create Critic if requested
                if auto_include_critic:
                    await db_client.create_agent(
                        session_id=session_id,
                        user_id=user_id,
                        name=ModelConfig.CRITIC_ROLE,
                        role="Critical Reviewer",
                        goal="Challenge assumptions and ensure scientific rigor",
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
                agent.get("name") == ModelConfig.PRINCIPAL_INVESTIGATOR_ROLE 
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
                    parallel_index=i
                )
                
                # Start the conversation
                await self.orchestrator.start_conversation(
                    meeting_id=meeting_id,
                    interaction=interaction,
                    live_mode=live_mode
                )
            
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

    @app_commands.command(
        name="end_team_meeting",
        description="End an ongoing team meeting"
    )
    @app_commands.describe(
        meeting_id="Optional meeting ID if multiple meetings are active",
        end_all_parallel="End all parallel runs of the meeting (default: true)"
    )
    async def end_team_meeting(
        self,
        interaction: discord.Interaction,
        meeting_id: Optional[str] = None,
        end_all_parallel: Optional[bool] = True
    ):
        """End an ongoing team meeting."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        user_id = str(interaction.user.id)
        
        try:
            # Get active session
            session_result = await db_client.get_active_session(user_id=user_id)
            
            if not session_result.get("isSuccess") or not session_result.get("data"):
                await interaction.followup.send(
                    "You don't have an active session.",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            session_id = session_data.get("id")
            
            # Get active meetings
            if meeting_id:
                # Get specific meeting and its parallel runs if requested
                meeting_result = await db_client.get_meeting(meeting_id=meeting_id)
                if end_all_parallel and meeting_result.get("isSuccess"):
                    parallel_result = await db_client.get_parallel_meetings(
                        session_id=session_id,
                        base_meeting_id=meeting_result.get("data", {}).get("id")
                    )
                    meetings = parallel_result.get("data", []) if parallel_result.get("isSuccess") else []
                else:
                    meetings = [meeting_result.get("data")] if meeting_result.get("isSuccess") else []
            else:
                # Get all active meetings for the session
                meetings_result = await db_client.get_active_meetings(session_id=session_id)
                meetings = meetings_result.get("data", []) if meetings_result.get("isSuccess") else []
            
            if not meetings:
                await interaction.followup.send(
                    "No active meetings found to end.",
                    ephemeral=True
                )
                return
            
            # End each meeting
            ended_meetings = []
            for meeting in meetings:
                meeting_id = meeting.get("id")
                
                # Stop the orchestrator
                await self.orchestrator.end_conversation(meeting_id=meeting_id)
                
                # Update meeting status
                end_result = await db_client.end_meeting(meeting_id=meeting_id)
                
                if end_result.get("isSuccess"):
                    ended_meetings.append(meeting)
            
            # Create response embed
            embed = discord.Embed(
                title="Team Meeting(s) Ended",
                description=f"Ended {len(ended_meetings)} meeting(s).",
                color=discord.Color.blue()
            )
            
            # Group meetings by parallel index
            parallel_groups = {}
            for meeting in ended_meetings:
                parallel_index = meeting.get("parallel_index", 0)
                if parallel_index not in parallel_groups:
                    parallel_groups[parallel_index] = []
                parallel_groups[parallel_index].append(meeting)
            
            for parallel_index, group in parallel_groups.items():
                embed.add_field(
                    name=f"Parallel Run {parallel_index + 1}",
                    value="\n".join([
                        f"Meeting {m.get('id')}: {m.get('agenda')}\n"
                        f"Rounds completed: {m.get('round_count')}"
                        for m in group
                    ]),
                    inline=False
                )
            
            embed.add_field(
                name="View Transcripts",
                value="Use `/lab transcript list` to view the meeting transcripts.",
                inline=False
            )
            
            if len(parallel_groups) > 1:
                embed.add_field(
                    name="Combined Summary",
                    value="A combined summary of all parallel runs will be generated and posted in a new thread.",
                    inline=False
                )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error in end_team_meeting command: {e}")
            await interaction.followup.send(
                "An error occurred while ending the team meeting. Please try again later.",
                ephemeral=True
            )

    async def start_team_meeting_callback(self, interaction: discord.Interaction, agenda: str, rounds: Optional[int] = 3, parallel_meetings: Optional[int] = 1, 
                                 agent_list: Optional[str] = None, auto_generate: Optional[bool] = False, auto_scientist_count: Optional[int] = 3, 
                                 auto_include_critic: Optional[bool] = True, temperature_variation: Optional[bool] = True, live_mode: Optional[bool] = True):
        """Callback for the team_meeting command."""
        await self.team_meeting(interaction, agenda, rounds, parallel_meetings, agent_list, 
                               auto_generate, auto_scientist_count, auto_include_critic, temperature_variation, live_mode)
        
    async def end_team_meeting_callback(self, interaction: discord.Interaction, meeting_id: Optional[str] = None, end_all_parallel: Optional[bool] = True):
        """Callback for the end_team_meeting command."""
        await self.end_team_meeting(interaction, meeting_id, end_all_parallel)

async def setup(bot: commands.Bot):
    """Add the cog to the bot."""
    await bot.add_cog(LabMeetingCommands(bot)) 