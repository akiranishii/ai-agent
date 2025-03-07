import discord
from discord import app_commands
from discord.ext import commands
import logging
from typing import Optional

from db_client import db_client

logger = logging.getLogger(__name__)

class LabAgentCommands(commands.Cog):
    """Commands for managing lab agents"""
    
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
        
        # Add agent commands to the lab group
        self.lab_group.add_command(app_commands.Command(
            name="agent_create",
            description="Create a new AI agent in the current lab session",
            callback=self.create_agent_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="agent_update",
            description="Update an existing agent in the current lab session",
            callback=self.update_agent_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="agent_delete",
            description="Delete an agent from the current lab session",
            callback=self.delete_agent_callback,
            extras={"cog": self}
        ))
        
        self.lab_group.add_command(app_commands.Command(
            name="agent_list",
            description="List all agents in the current lab session",
            callback=self.list_agents_callback,
            extras={"cog": self}
        ))
        
        logger.info("Registered lab agent commands")

    async def create_agent_callback(self, interaction: discord.Interaction, agent_name: str, expertise: Optional[str] = None, goal: Optional[str] = None, role: Optional[str] = None, model: Optional[str] = "openai"):
        """Callback for the create_agent command."""
        await self.create_agent(interaction, agent_name, expertise, goal, role, model)
        
    async def update_agent_callback(self, interaction: discord.Interaction, agent_name: str, expertise: Optional[str] = None, goal: Optional[str] = None, role: Optional[str] = None, model: Optional[str] = None):
        """Callback for the update_agent command."""
        await self.update_agent(interaction, agent_name, expertise, goal, role, model)
        
    async def delete_agent_callback(self, interaction: discord.Interaction, agent_name: str):
        """Callback for the delete_agent command."""
        await self.delete_agent(interaction, agent_name)
        
    async def list_agents_callback(self, interaction: discord.Interaction):
        """Callback for the list_agents command."""
        await self.list_agents(interaction)

    @app_commands.command(
        name="agent_create",
        description="Create a new AI agent in the current lab session"
    )
    @app_commands.describe(
        agent_name="Name of the agent (e.g., 'Principal Investigator', 'Biologist')",
        expertise="Agent's area of expertise (e.g., 'Structural biology')",
        goal="Agent's main objective (e.g., 'Propose novel protein scaffolds')",
        role="Agent's functional role (e.g., 'Provide domain insights')",
        model="LLM model to use (e.g., 'openai', 'anthropic', 'mistral')"
    )
    async def create_agent(
        self,
        interaction: discord.Interaction,
        agent_name: str,
        expertise: Optional[str] = None,
        goal: Optional[str] = None,
        role: Optional[str] = None,
        model: Optional[str] = "openai"
    ):
        """Create a new AI agent in the current lab session."""
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
            
            # Create the agent
            agent_result = await db_client.create_agent(
                session_id=session_id,
                user_id=user_id,
                name=agent_name,
                expertise=expertise,
                goal=goal,
                role=role,
                model=model
            )
            
            if not agent_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Failed to create agent: {agent_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            agent_data = agent_result.get("data", {})
            
            # Create response embed
            embed = discord.Embed(
                title="Agent Created",
                description=f"Agent '{agent_name}' has been created.",
                color=discord.Color.green()
            )
            embed.add_field(name="Name", value=agent_name, inline=True)
            if expertise:
                embed.add_field(name="Expertise", value=expertise, inline=True)
            if goal:
                embed.add_field(name="Goal", value=goal, inline=True)
            if role:
                embed.add_field(name="Role", value=role, inline=True)
            embed.add_field(name="Model", value=model, inline=True)
            
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error in create_agent command: {e}")
            await interaction.followup.send(
                "An error occurred while creating the agent. Please try again later.",
                ephemeral=True
            )

    @app_commands.command(
        name="agent_update",
        description="Update an existing agent in the current lab session"
    )
    @app_commands.describe(
        agent_name="Name of the agent to update",
        expertise="New expertise (optional)",
        goal="New goal (optional)",
        role="New role (optional)",
        model="New LLM model (optional)"
    )
    async def update_agent(
        self,
        interaction: discord.Interaction,
        agent_name: str,
        expertise: Optional[str] = None,
        goal: Optional[str] = None,
        role: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Update an existing agent in the current lab session."""
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
            
            # Build updates object
            updates = {}
            if expertise is not None:
                updates["expertise"] = expertise
            if goal is not None:
                updates["description"] = goal
            if role is not None:
                updates["role"] = role
            if model is not None:
                updates["model"] = model
            
            # First, get the agent's ID by name
            lookup_result = await db_client.get_agent_by_name(
                session_id=session_id,
                agent_name=agent_name
            )
            
            if not lookup_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Could not find agent '{agent_name}': {lookup_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            agent_id = lookup_result.get("data", {}).get("id")
            if not agent_id:
                await interaction.followup.send(
                    f"Invalid agent data retrieved for '{agent_name}'",
                    ephemeral=True
                )
                return
                
            # Now update the agent using its ID
            agent_result = await db_client.update_agent(
                agent_id=agent_id,
                name=agent_name if "name" in updates else None,
                role=updates.get("role"),
                description=updates.get("description"),
                expertise=updates.get("expertise"),
                model=updates.get("model")
            )
            
            if not agent_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Failed to update agent: {agent_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            agent_data = agent_result.get("data", {})
            
            # Create response embed
            embed = discord.Embed(
                title="Agent Updated",
                description=f"Agent '{agent_name}' has been updated.",
                color=discord.Color.blue()
            )
            embed.add_field(name="Name", value=agent_name, inline=True)
            if expertise:
                embed.add_field(name="New Expertise", value=expertise, inline=True)
            if goal:
                embed.add_field(name="New Goal", value=goal, inline=True)
            if role:
                embed.add_field(name="New Role", value=role, inline=True)
            if model:
                embed.add_field(name="New Model", value=model, inline=True)
            
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error in update_agent command: {e}")
            await interaction.followup.send(
                "An error occurred while updating the agent. Please try again later.",
                ephemeral=True
            )

    @app_commands.command(
        name="agent_delete",
        description="Delete an agent from the current lab session"
    )
    @app_commands.describe(
        agent_name="Name of the agent to delete"
    )
    async def delete_agent(
        self,
        interaction: discord.Interaction,
        agent_name: str
    ):
        """Delete an agent from the current lab session."""
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
            
            # First, get the agent's ID by name
            lookup_result = await db_client.get_agent_by_name(
                session_id=session_id,
                agent_name=agent_name
            )
            
            if not lookup_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Could not find agent '{agent_name}': {lookup_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            agent_id = lookup_result.get("data", {}).get("id")
            if not agent_id:
                await interaction.followup.send(
                    f"Invalid agent data retrieved for '{agent_name}'",
                    ephemeral=True
                )
                return
                
            # Delete the agent using its ID
            delete_result = await db_client.delete_agent(
                agent_id=agent_id
            )
            
            if not delete_result.get("isSuccess"):
                await interaction.followup.send(
                    f"Failed to delete agent: {delete_result.get('message', 'Unknown error')}",
                    ephemeral=True
                )
                return
            
            # Create response embed
            embed = discord.Embed(
                title="Agent Deleted",
                description=f"Agent '{agent_name}' has been deleted.",
                color=discord.Color.red()
            )
            
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error in delete_agent command: {e}")
            await interaction.followup.send(
                "An error occurred while deleting the agent. Please try again later.",
                ephemeral=True
            )

    @app_commands.command(
        name="agent_list",
        description="List all agents in the current lab session"
    )
    async def list_agents(
        self,
        interaction: discord.Interaction
    ):
        """List all agents in the current lab session."""
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
            
            # Get all created agents
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
                    "No agents found in the current session. Use `/lab agent create` to add one.",
                    ephemeral=True
                )
                return
            
            # Create response embed
            embed = discord.Embed(
                title="Lab Session Agents",
                description=f"Agents in session '{session_data.get('title')}':",
                color=discord.Color.blue()
            )
            
            for agent in agents:
                field_value = []
                if agent.get("expertise"):
                    field_value.append(f"Expertise: {agent['expertise']}")
                if agent.get("goal"):
                    field_value.append(f"Goal: {agent['goal']}")
                if agent.get("role"):
                    field_value.append(f"Role: {agent['role']}")
                if agent.get("model"):
                    field_value.append(f"Model: {agent['model']}")
                
                embed.add_field(
                    name=agent["name"],
                    value="\n".join(field_value) if field_value else "No additional details",
                    inline=False
                )
            
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Error in list_agents command: {e}")
            await interaction.followup.send(
                "An error occurred while fetching the agents. Please try again later.",
                ephemeral=True
            )

async def setup(bot: commands.Bot):
    """Add the cog to the bot."""
    await bot.add_cog(LabAgentCommands(bot)) 