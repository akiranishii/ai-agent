import discord
from discord import app_commands
from discord.ext import commands
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class HelpCommand(commands.Cog):
    """Command for displaying help information about all available commands."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        logger.info("Initialized help command")

    @app_commands.command(
        name="help",
        description="Get help information about available commands"
    )
    @app_commands.describe(
        command="The specific command to get help for (optional)"
    )
    async def help(
        self,
        interaction: discord.Interaction,
        command: Optional[str] = None
    ):
        """Display help information about commands."""
        try:
            # Try to defer the response, but handle if it's already been deferred
            try:
                # Use ephemeral=False to make help messages visible to all users,
                # reducing the chance of timeouts or 404 errors from ephemeral interactions
                await interaction.response.defer(ephemeral=False)
            except (discord.errors.NotFound, discord.errors.InteractionResponded) as e:
                logger.warning(f"Could not defer help command interaction: {e}")
                # Continue anyway, we'll try to respond with followup
            
            if command:
                # Show detailed help for specific command
                await self._show_command_help(interaction, command)
            else:
                # Show general help overview
                await self._show_general_help(interaction)
                
        except Exception as e:
            logger.error(f"Error in help command: {e}")
            try:
                await interaction.followup.send(
                    "An error occurred while fetching help information. Please try again later.",
                    ephemeral=False
                )
            except Exception as follow_up_error:
                logger.error(f"Failed to send error followup: {follow_up_error}")

    async def _show_general_help(self, interaction: discord.Interaction):
        """Show overview of all available commands."""
        try:
            embed = discord.Embed(
                title="TheraLab Help",
                description="Here are all available commands. Use `/help command:\"command_name\"` for detailed information about a specific command.",
                color=discord.Color.blue()
            )

            # Quickstart Command
            embed.add_field(
                name="🚀 Quickstart",
                value=(
                    "**`/quickstart`**\n"
                    "Quick way to start a session with AI agents.\n"
                    "• `topic` - Topic to discuss\n"
                    "• `agent_count` - Number of scientists (default: 3)\n"
                    "• `include_critic` - Add a critic (default: true)\n"
                    "• `public` - Make session public (default: false)"
                ),
                inline=False
            )

            # Lab Session Commands
            embed.add_field(
                name="📋 Lab Session Management",
                value=(
                    "**`/lab start`** - Start a new lab session\n"
                    "**`/lab end`** - End current session\n"
                    "**`/lab list`** - List your sessions\n"
                    "**`/lab reopen`** - Reopen an ended session"
                ),
                inline=False
            )

            # Lab Agent Commands
            embed.add_field(
                name="🤖 Agent Management",
                value=(
                    "**`/lab agent_create`** - Create a new agent\n"
                    "**`/lab agent_update`** - Update an agent\n"
                    "**`/lab agent_delete`** - Delete an agent\n"
                    "**`/lab agent_list`** - List all agents"
                ),
                inline=False
            )

            # Team Meeting Commands
            embed.add_field(
                name="🗣️ Team Meetings",
                value=(
                    "**`/lab team_meeting`** - Start a team discussion\n"
                    "**`/lab end_team_meeting`** - End ongoing meeting"
                ),
                inline=False
            )

            # Transcript Commands
            embed.add_field(
                name="📝 Transcripts",
                value=(
                    "**`/lab transcript_list`** - List available transcripts\n"
                    "**`/lab transcript_view`** - View a specific transcript"
                ),
                inline=False
            )

            # Help Command
            embed.add_field(
                name="❓ Help",
                value="**`/help`** - Show this help message\n",
                inline=False
            )

            # Admin Commands - show to everyone
            embed.add_field(
                name="⚙️ Admin Commands",
                value="**`/admin_sync`** - Clear and sync all slash commands\n• `global_commands` - Sync globally or just for this server (default: true)\n• `password` - Required password for authorization",
                inline=False
            )

            await interaction.followup.send(embed=embed, ephemeral=False)
        except Exception as e:
            logger.error(f"Error showing general help: {e}")
            # We don't need to try sending a follow-up here, as the main help method already handles errors

    async def _show_command_help(self, interaction: discord.Interaction, command_name: str):
        """Show detailed help for a specific command."""
        try:
            # Remove leading slash if present
            command_name = command_name.lstrip('/')
            
            # Command details dictionary
            command_details = {
                "quickstart": {
                    "title": "Quickstart Command",
                    "description": "Quickly create a lab session with agents and start a brainstorming session.",
                    "usage": "/quickstart topic:\"Your topic\" [agent_count:3] [include_critic:true] [public:false]",
                    "parameters": {
                        "topic": "The topic or question to discuss (required)",
                        "agent_count": "Number of Scientist agents to create (default: 3)",
                        "include_critic": "Whether to include a Critic agent (default: true)",
                        "public": "Whether the session should be publicly viewable (default: false)"
                    },
                    "example": "/quickstart topic:\"How can we improve renewable energy storage?\" agent_count:4 include_critic:true",
                    "color": discord.Color.green()
                },
                "lab start": {
                    "title": "Lab Start Command",
                    "description": "Start a new lab session for advanced workflows.",
                    "usage": "/lab start [title:\"Session Title\"] [description:\"Description\"] [is_public:false]",
                    "parameters": {
                        "title": "Title for the session (optional)",
                        "description": "Purpose or context for the session (optional)",
                        "is_public": "Whether the session should be public (default: false)"
                    },
                    "example": "/lab start title:\"Protein Modeling\" description:\"Focus on novel folding approaches\"",
                    "color": discord.Color.blue()
                },
                "lab end": {
                    "title": "Lab End Command",
                    "description": "End or archive the current lab session.",
                    "usage": "/lab end [confirm:false] [public:false]",
                    "parameters": {
                        "confirm": "Confirm ending the session (default: false)",
                        "public": "Make the session public after ending (default: false)"
                    },
                    "example": "/lab end confirm:true public:true",
                    "color": discord.Color.red()
                },
                "lab reopen": {
                    "title": "Lab Reopen Command",
                    "description": "Reopen a previously ended session.",
                    "usage": "/lab reopen session_id:\"id\" [confirm:false]",
                    "parameters": {
                        "session_id": "ID of the session to reopen (required)",
                        "confirm": "Confirm reopening the session (default: false)"
                    },
                    "example": "/lab reopen session_id:1234 confirm:true",
                    "color": discord.Color.green()
                },
                "lab team_meeting": {
                    "title": "Team Meeting Command",
                    "description": "Start a multi-agent conversation in the active session.",
                    "usage": "/lab team_meeting agenda:\"topic\" [rounds:3] [parallel_meetings:1] [agent_list:\"Agent1,Agent2\"] [auto_generate:false]",
                    "parameters": {
                        "agenda": "The main topic or question (required)",
                        "rounds": "Number of conversation rounds (default: 3)",
                        "parallel_meetings": "Number of parallel runs (default: 1)",
                        "agent_list": "Names of agents to include (optional)",
                        "auto_generate": "Auto-generate agents (default: false)",
                        "auto_scientist_count": "Number of scientists if auto-generating (default: 3)",
                        "auto_include_critic": "Include critic if auto-generating (default: true)"
                    },
                    "example": "/lab team_meeting agenda:\"Novel immunotherapy approaches\" rounds:4 agent_list:\"PI,Scientist1,Critic\"",
                    "color": discord.Color.gold()
                },
                "lab transcript": {
                    "title": "Transcript Commands",
                    "description": "View and manage meeting transcripts.",
                    "usage": [
                        "/lab transcript_list",
                        "/lab transcript_view transcript_id:\"id\""
                    ],
                    "parameters": {
                        "transcript_id": "ID of the transcript to view (required for view)"
                    },
                    "example": "/lab transcript view transcript_id:12345",
                    "color": discord.Color.purple()
                },
                "admin_sync": {
                    "title": "Admin Sync Command",
                    "description": "Command to clear and re-sync all slash commands. Requires authorization.",
                    "usage": "/admin_sync [global_commands:true] password:\"password\"",
                    "parameters": {
                        "global_commands": "Whether to sync commands globally across all servers (true) or just this server (false). Default is true.",
                        "password": "Required authorization password to run this command."
                    },
                    "example": "/admin_sync global_commands:true password:\"admin123\"",
                    "color": discord.Color.dark_red()
                }
            }

            # Get command details
            details = command_details.get(command_name.lower())
            if not details:
                await interaction.followup.send(
                    f"No detailed help available for command: {command_name}\nUse `/help` to see all available commands.",
                    ephemeral=False
                )
                return

            # Create embed
            embed = discord.Embed(
                title=details["title"],
                description=details["description"],
                color=details["color"]
            )

            # Add usage
            if isinstance(details["usage"], list):
                embed.add_field(
                    name="📝 Usage",
                    value="\n".join(details["usage"]),
                    inline=False
                )
            else:
                embed.add_field(
                    name="📝 Usage",
                    value=details["usage"],
                    inline=False
                )

            # Add parameters
            params = "\n".join([f"• `{param}` - {desc}" for param, desc in details["parameters"].items()])
            embed.add_field(
                name="⚙️ Parameters",
                value=params,
                inline=False
            )

            # Add example
            embed.add_field(
                name="💡 Example",
                value=details["example"],
                inline=False
            )

            await interaction.followup.send(embed=embed, ephemeral=False)
        except Exception as e:
            logger.error(f"Error showing command help: {e}")
            # We don't need to try sending a follow-up here, as the main help method already handles errors

    @app_commands.command(
        name="admin_sync",
        description="Admin command to clear and sync slash commands"
    )
    @app_commands.describe(
        global_commands="Whether to sync commands globally (across all servers) or just this server",
        password="Admin password to authorize the command"
    )
    async def admin_sync(
        self, 
        interaction: discord.Interaction,
        global_commands: bool = True,
        password: str = ""
    ):
        """Clear all commands and re-sync them (requires password)."""
        # Check password instead of administrator permissions
        if password != "admin123":
            await interaction.response.send_message(
                "Invalid password. This command requires authorization.",
                ephemeral=True
            )
            return
            
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        try:
            # Log the action
            logger.info(f"User {interaction.user.name} ({interaction.user.id}) initiated command sync")
            
            # Extensions to reload after clearing
            extensions = [
                "commands.help_command",
                "commands.lab_session_commands",
                "commands.lab_agent_commands",
                "commands.lab_meeting_commands",
                "commands.lab_transcript_commands",
                "commands.quickstart_command"
            ]
            
            if global_commands:
                # Step 1: Store all commands that will be cleared
                logger.info("Capturing current command list before clearing...")
                stored_commands = [cmd for cmd in self.bot.tree.get_commands()]
                
                # Step 2: Clear global commands
                logger.info("Clearing global commands")
                self.bot.tree.clear_commands(guild=None)
                
                # Step 3: Manually re-add all commands that were cleared
                logger.info("Re-adding commands that were cleared...")
                for cmd in stored_commands:
                    self.bot.tree.add_command(cmd)
                
                # Step 4: Sync global commands
                logger.info("Syncing global commands")
                synced = await self.bot.tree.sync()
                
                # Report success
                logger.info(f"Successfully synced {len(synced)} global commands")
                await interaction.followup.send(
                    f"✅ Cleared and re-synced {len(synced)} global commands successfully.\n"
                    f"These changes will propagate to all servers using the bot.",
                    ephemeral=True
                )
            else:
                # Step 1: Store all commands that will be cleared
                logger.info(f"Capturing current command list before clearing for guild {interaction.guild.id}...")
                stored_commands = [cmd for cmd in self.bot.tree.get_commands(guild=interaction.guild)]
                
                # Step 2: Clear guild-specific commands
                logger.info(f"Clearing commands for guild {interaction.guild.id}")
                self.bot.tree.clear_commands(guild=interaction.guild)
                
                # Step 3: Manually re-add all commands that were cleared
                logger.info("Re-adding commands that were cleared...")
                for cmd in stored_commands:
                    self.bot.tree.add_command(cmd, guild=interaction.guild)
                
                # Step 4: Sync guild-specific commands
                logger.info(f"Syncing commands for guild {interaction.guild.id}")
                synced = await self.bot.tree.sync(guild=interaction.guild)
                
                # Report success
                logger.info(f"Successfully synced {len(synced)} guild commands")
                await interaction.followup.send(
                    f"✅ Cleared and re-synced {len(synced)} commands for this server successfully.\n"
                    f"These changes only affect this server.",
                    ephemeral=True
                )
        except Exception as e:
            logger.error(f"Error syncing commands: {e}")
            await interaction.followup.send(
                f"❌ Error syncing commands: {str(e)}",
                ephemeral=True
            )

async def setup(bot: commands.Bot):
    """Add the cog to the bot."""
    await bot.add_cog(HelpCommand(bot)) 