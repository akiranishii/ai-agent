import discord
from discord import app_commands
from discord.ext import commands
import logging
from typing import Optional
import asyncio

from db_client import db_client
from models import ModelConfig
from orchestrator import AgentOrchestrator
from llm_client import llm_client

logger = logging.getLogger(__name__)

class QuickstartCommand(commands.Cog):
    """Command for quickly starting a lab session with agents and a meeting."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.orchestrator = AgentOrchestrator(llm_client)
        self.conversation_tasks = {}  # Dictionary to track conversation tasks
        logger.info("Initialized quickstart command")

    @app_commands.command(
        name="quickstart",
        description="Quickly create a lab session with agents and start a brainstorming session"
    )
    @app_commands.describe(
        topic="The topic or question to discuss",
        agent_count="Number of Scientist agents to create (default: 3)",
        include_critic="Whether to include a Critic agent (default: true)",
        public="Whether the session should be publicly viewable (default: false)",
        live_mode="Show agent responses in real-time (default: true)",
        rounds="Number of conversation rounds (default: 3)",
        speakers_per_round="Number of agent speakers selected per round (default: all agents excluding PI)"
    )
    async def quickstart(
        self,
        interaction: discord.Interaction,
        topic: str,
        agent_count: Optional[int] = 3,
        include_critic: Optional[bool] = True,
        public: Optional[bool] = False,
        live_mode: Optional[bool] = True,
        rounds: Optional[int] = 3,
        speakers_per_round: Optional[int] = None
    ):
        """Quickly create a lab session with agents and start a meeting."""
        # IMMEDIATELY acknowledge the interaction to prevent timeout
        # This must be the very first thing we do
        await interaction.response.defer(ephemeral=True)
        
        try:
            user_id = str(interaction.user.id)
            logger.info(f"Starting quickstart for user {user_id} on topic: {topic}")
            
            # Check API connectivity
            health_check = await db_client.health_check()
            if not health_check.get("isSuccess"):
                logger.error(f"API is not available: {health_check.get('message')}")
                await interaction.followup.send(
                    f"Error: The API service is currently unavailable. Please try again later.",
                    ephemeral=True
                )
                return
                
            # Check for existing active session
            active_session = await db_client.get_active_session(user_id=user_id)
            
            if active_session.get("isSuccess") and active_session.get("data"):
                # End the current session
                session_id = active_session["data"]["id"]
                logger.info(f"Found active session {session_id} for user {user_id}")
                end_result = await db_client.end_session(session_id=session_id)
                if end_result.get("isSuccess"):
                    logger.info(f"Ended previous active session {session_id} for user {user_id}")
                else:
                    logger.warning(f"Failed to end session {session_id}: {end_result.get('message')}")
            
            # Create the new session
            logger.info(f"Creating new session for user {user_id}")
            session_result = await db_client.create_session(
                user_id=user_id,
                title=f"Research on: {topic}",
                description=f"Quickstart session on: {topic}",
                is_public=public
            )
            
            if not session_result.get("isSuccess"):
                error_message = session_result.get('message', 'Unknown error')
                logger.error(f"Failed to create session: {error_message}")
                await interaction.followup.send(
                    f"Failed to create session: {error_message}",
                    ephemeral=True
                )
                return
            
            session_data = session_result.get("data", {})
            session_id = session_data.get("id")
            
            # Send initial progress message
            progress_message = await interaction.followup.send(
                "Creating your research team for brainstorming session...",
                ephemeral=True
            )
            
            # Create Principal Investigator
            # Generate variables for the PI based on the topic
            pi_variables = await llm_client.generate_agent_variables(
                topic=topic,
                agent_type="principal_investigator"
            )
            
            # Store PI details for reference in scientist generation
            pi_name = ModelConfig.PRINCIPAL_INVESTIGATOR_ROLE
            pi_expertise = pi_variables.get("expertise", "")
            pi_goal = pi_variables.get("goal", "")
            
            # Update progress with PI info
            await progress_message.edit(content=(
                "Creating your research team for brainstorming session...\n\n"
                f"🔬 **{pi_name}**\n"
                f"• Expertise: {pi_expertise}\n"
                f"• Goal: {pi_goal}"
            ))
            
            await db_client.create_agent(
                session_id=session_id,
                user_id=user_id,
                name=pi_name,
                role="Lead",
                goal=pi_goal,
                expertise=pi_expertise,
                model="openai"
            )
            
            # Keep track of all created agents for diversity
            created_agents_info = [{
                "name": pi_name,
                "role": "Lead",
                "expertise": pi_expertise,
                "goal": pi_goal
            }]
            
            for i in range(agent_count):
                # Create diversity context with information about existing team
                diversity_context = [
                    "Create a scientist with expertise COMPLETELY DIFFERENT from previously generated team members.",
                    f"This is scientist #{i+1} of {agent_count}."
                ]
                
                # Include details of existing team for better complementary expertise
                if created_agents_info:
                    previous_agents_text = "\n\nCurrent research team:\n"
                    for j, agent_info in enumerate(created_agents_info):
                        previous_agents_text += f"{j+1}. {agent_info['name']} ({agent_info['role']}) - Expertise: {agent_info['expertise']}"
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
                    topic=topic,
                    agent_type="scientist",
                    additional_context="\n".join(diversity_context)
                )
                
                # Get the agent details
                agent_name = scientist_variables.get("agent_name", f"Scientist {i+1}")
                agent_expertise = scientist_variables.get("expertise", "")
                agent_goal = scientist_variables.get("goal", "")
                
                # Track this agent's details for future diversity
                created_agents_info.append({
                    "name": agent_name,
                    "role": ModelConfig.SCIENTIST_ROLE,
                    "expertise": agent_expertise,
                    "goal": agent_goal
                })
                
                # Update progress with this scientist's info
                current_content = progress_message.content
                await progress_message.edit(content=(
                    current_content + "\n\n"
                    f"🔬 **{agent_name}**\n"
                    f"• Expertise: {agent_expertise}\n"
                    f"• Goal: {agent_goal}"
                ))
                
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
            if include_critic:
                critic_expertise = "Critical analysis of scientific research, identification of methodological flaws, and evaluation of research validity"
                critic_goal = "Ensure scientific rigor and identify potential weaknesses in proposed research approaches"
                
                # Update progress with critic info
                current_content = progress_message.content
                await progress_message.edit(content=(
                    current_content + "\n\n"
                    f"🔬 **Critic**\n"
                    f"• Expertise: {critic_expertise}\n"
                    f"• Goal: {critic_goal}"
                ))
                
                await db_client.create_agent(
                    session_id=session_id,
                    user_id=user_id,
                    name="Critic",
                    role=ModelConfig.CRITIC_ROLE,
                    expertise=critic_expertise,
                    goal=critic_goal,
                    model="openai"
                )
            
            # Get all created agents
            agents_result = await db_client.get_session_agents(
                session_id=session_id,
                user_id=user_id
            )
            if not agents_result.get("isSuccess"):
                logger.error(f"Failed to retrieve agents: {agents_result.get('message')}")
                await interaction.followup.send(
                    f"Failed to retrieve agents: {agents_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            agents = agents_result.get("data", [])
            
            # Create and start the meeting
            meeting_result = await db_client.create_meeting(
                session_id=session_id,
                title=f"Discussion on: {topic}",
                agenda=topic,
                max_rounds=3
            )
            
            if not meeting_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Failed to create meeting: {meeting_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            meeting = meeting_result.get("data", {})
            meeting_id = meeting.get("id")
            
            # Initialize and start the meeting
            await self.orchestrator.initialize_meeting(
                meeting_id=meeting_id,
                session_id=session_id,
                agents=agents,
                agenda=topic,
                round_count=rounds
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
            
            # Calculate the total number of agents
            agent_total = agent_count + 1  # Scientists + Principal Investigator
            if include_critic:
                agent_total += 1  # Add Critic if included
            
            # Update progress message one last time to indicate completion
            await progress_message.edit(content=(
                progress_message.content + "\n\n"
                "✅ All agents have been created. Starting the brainstorming session now..."
            ))
                
            # Create response embed
            embed = discord.Embed(
                title="Quickstart Complete",
                description=f"Created a new session and started a brainstorming discussion on: {topic}",
                color=discord.Color.green()
            )
            
            embed.add_field(
                name="Session Details",
                value=(
                    f"**ID**: {session_id}\n"
                    f"**Privacy**: {'Public' if public else 'Private'}\n"
                    f"**Agents**: {agent_total} total\n"
                    f"- 1 Principal Investigator\n"
                    f"- {agent_count} Scientists\n"
                    f"{f'- 1 Critic' if include_critic else ''}"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Meeting",
                value=(
                    f"**ID**: {meeting_id}\n"
                    f"**Rounds**: {rounds}\n"
                    f"**Status**: In Progress\n"
                    f"**Live Mode**: {'On' if live_mode else 'Off'}"
                ),
                inline=False
            )
            
            embed.add_field(
                name="View Progress",
                value=f"Use `/lab transcript_view {meeting_id}` to view the discussion.",
                inline=False
            )
            
            await interaction.followup.send(
                embed=embed,
                ephemeral=True
            )
            
        except Exception as e:
            logger.error(f"Error in quickstart command: {e}")
            await interaction.followup.send(
                "An error occurred while setting up your session. Please try again later.",
                ephemeral=True
            )

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
    await bot.add_cog(QuickstartCommand(bot)) 