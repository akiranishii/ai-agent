import logging
import asyncio
import json
from typing import List, Dict, Optional, Any, Set
from models import LLMProvider, LLMMessage
from pydantic import BaseModel, ValidationError
import discord

logger = logging.getLogger(__name__)

class OrchestratorResponse(BaseModel):
    """Pydantic model for the orchestrator's agent selection response."""
    agent: str
    rationale: str

class AgentOrchestrator:
    """Orchestrates agent interactions in meetings."""
    
    def __init__(self, llm_client):
        """Initialize the orchestrator with an LLM client."""
        self.llm_client = llm_client
        self.active_meetings = {}
        self.parallel_groups = {}
        logger.info("Initialized AgentOrchestrator")
    
    async def initialize_meeting(self, meeting_id, session_id, agents, agenda, round_count, parallel_index=0):
        """Initialize a new meeting."""
        logger.info("Initializing meeting " + str(meeting_id))
        self.active_meetings[meeting_id] = {
            "id": meeting_id,
            "session_id": session_id,
            "agents": agents,
            "agenda": agenda,
            "round_count": round_count,
            "current_round": 0,
            "is_completed": False,
            "thread": None,
            "parallel_index": parallel_index,
            "messages": [],
            "is_active": True,
            "summary": None,
            "conversation_history": f"The user wants to discuss: {agenda}\n\n"
        }
        
        # Track parallel meetings
        if session_id not in self.parallel_groups:
            self.parallel_groups[session_id] = set()
        self.parallel_groups[session_id].add(meeting_id)
        
        logger.info("Initialized meeting " + str(meeting_id) + " (parallel index " + str(parallel_index) + ") for session " + str(session_id))
        return True
    
    async def start_conversation(self, meeting_id: str, interaction: discord.Interaction, live_mode: bool = True, conversation_length: int = 2):
        """
        Run a conversation with the agents.
        
        Args:
            meeting_id: The ID of the meeting
            interaction: The Discord interaction
            live_mode: Whether to send updates to Discord
            conversation_length: Number of agent exchanges per round
            
        Returns:
            True if successful, False otherwise
        """
        # Get meeting data
        meeting_data = self.active_meetings.get(meeting_id)
        if not meeting_data:
            logger.error(f"Meeting {meeting_id} not found")
            return False
            
        # Get session ID and round count from meeting data
        session_id = meeting_data.get("session_id")
        round_count = meeting_data.get("round_count", 3)
        
        # Initialize conversation history if it doesn't exist
        conversation_history = meeting_data.get("conversation_history", "")
        if not conversation_history:
            # Add basic info to conversation history
            agenda = meeting_data.get("agenda", "No agenda specified")
            parallel_index = meeting_data.get("parallel_index", 0)
            
            conversation_history = f"Lab Meeting #{parallel_index + 1}\nAgenda: {agenda}\n\nParticipants:\n"
            meeting_data["conversation_history"] = conversation_history
        
        # Get agents
        agents = meeting_data.get("agents", [])
        if not agents:
            logger.error(f"No agents found for meeting {meeting_id}")
            return False
            
        # Find the Principal Investigator (PI)
        pi_agent = next((a for a in agents if a["role"] == "Lead"), None)
        if not pi_agent:
            logger.error(f"No PI found for meeting {meeting_id}")
            return False
            
        # Initial message to set up the conversation if in live mode
        if live_mode:
            # Send initial message
            initial_message = f"**Lab Meeting Started**\n\nAgenda: {meeting_data.get('agenda', 'No agenda specified')}\n\n**Participants:**\n"
            
            # Helper function to convert system prompt to first person
            def to_first_person(prompt):
                if not prompt:
                    return ""
                # Replace "You are" with "I am"
                prompt = prompt.replace("You are", "I am").replace("you are", "I am")
                # Replace "Your expertise" with "My expertise"
                prompt = prompt.replace("Your expertise", "My expertise").replace("your expertise", "my expertise")
                # Replace "Your goal" with "My goal"
                prompt = prompt.replace("Your goal", "My goal").replace("your goal", "my goal")
                # Replace "Your role" with "My role"
                prompt = prompt.replace("Your role", "My role").replace("your role", "my role")
                return prompt.strip()
            
            # Add each agent with their formatted description
            for agent in agents:
                name = agent.get('name', 'Unknown Agent')
                role = agent.get('role', 'Unknown Role')
                
                # Create a formatted prompt showing this agent's capabilities
                prompt_parts = []
                
                # Start with the basic name and role
                initial_message += f"**{name}** ({role})\n"
                
                # Add expertise if available
                if agent.get('expertise'):
                    prompt_parts.append(f"My expertise is in {agent.get('expertise')}")
                
                # Add goal if available
                if agent.get('goal'):
                    prompt_parts.append(f"My goal is to {agent.get('goal')}")
                
                # Add any specific role details
                if agent.get('role') and "scientist" in agent.get('role', '').lower():
                    prompt_parts.append(f"My role is to {agent.get('role')}")
                
                # Add the formatted prompt parts if available
                if prompt_parts:
                    formatted_prompt = "\n".join(prompt_parts)
                    initial_message += f"```\n{formatted_prompt}\n```\n\n"
                else:
                    initial_message += "\n"
            
            await interaction.followup.send(initial_message, ephemeral=False)
            
        # Set meeting to active
        meeting_data["is_active"] = True
        
        # Loop through rounds
        for round_index in range(1, round_count + 1):
            if not meeting_data["is_active"]:
                logger.info(f"Meeting {meeting_id} was deactivated, stopping conversation")
                return False
                
            # Add round indicator to conversation history
            conversation_history += f"\n\n=== ROUND {round_index} of {round_count} ==="
            meeting_data["conversation_history"] = conversation_history
            
            # Send a new message for the round indicator
            if live_mode:
                await interaction.followup.send(f"**=== ROUND {round_index} of {round_count} ===**", ephemeral=False)
                
            # Inner loop: up to 'conversation_length' calls (the orchestrator chooses who speaks)
            calls_this_round = 0
            while calls_this_round < conversation_length:
                if not meeting_data["is_active"]:
                    logger.info(f"Meeting {meeting_id} was deactivated, stopping conversation")
                    return False
                    
                # Ask the orchestrator who should speak next
                try:
                    # Get available agent keys from the agents in the meeting (excluding PI and summary)
                    agent_keys = [a["name"] for a in meeting_data["agents"] if a["name"] != "Principal Investigator"]
                    agent_keys_str = ", ".join(agent_keys)
                    
                    # Build orchestrator prompt
                    orchestrator_prompt = f"""You are the Orchestrator. 
Read the conversation so far, then decide which agent should speak next. 
Your possible agents are: {agent_keys_str}.

Output your choice in a JSON format, for example:
{{
  "agent": "Scientist 1",
  "rationale": "I want this specialist's insight next."
}}
Only output valid JSON and nothing else.
"""
                    
                    # Call orchestrator
                    orch_resp = await self.llm_client.generate_response(
                        provider=LLMProvider.OPENAI,
                        messages=[
                            LLMMessage(role="system", content=orchestrator_prompt),
                            LLMMessage(role="user", content=conversation_history)
                        ],
                        temperature=1,
                        max_tokens=300
                    )
                    
                    orch_json_str = orch_resp.content
                    
                    # Parse orchestrator response
                    try:
                        orch_data = json.loads(orch_json_str)
                        chosen_agent = orch_data.get("agent")
                        rationale = orch_data.get("rationale", "No rationale provided")
                        
                        if not chosen_agent:
                            raise ValueError("No agent specified in orchestrator response")
                            
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.error(f"Error parsing orchestrator response: {e}")
                        # Default to a random agent if orchestrator fails
                        import random
                        chosen_agent = random.choice(agent_keys)
                        rationale = "Orchestrator failed to select an agent"
                        
                    # Find the agent in our list
                    agent = next((a for a in meeting_data["agents"] if a["name"] == chosen_agent), None)
                    if not agent:
                        logger.warning(f"Agent {chosen_agent} not found, using first available agent")
                        # Use the first non-PI agent as fallback
                        agent = next((a for a in meeting_data["agents"] if a["name"] != "Principal Investigator"), None)
                        chosen_agent = agent["name"] if agent else "Unknown Agent"
                        
                    # Determine the appropriate agent_key based on the agent's role
                    agent_role = agent["role"] if agent else "Unknown"
                    
                    # Map agent roles to valid agent_keys that exist in the AGENTS dictionary
                    if "Scientist" in agent_role or "specialist" in agent_role.lower() or "expert" in agent_role.lower():
                        # Use the generic scientist template for all scientist types
                        agent_key = "scientist"
                    elif "Critic" in agent_role or "reviewer" in agent_role.lower():
                        agent_key = "scientific_critic"
                    elif "Lead" in agent_role or "PI" in agent_role or "Principal" in agent_role:
                        agent_key = "principal_investigator"
                    else:
                        # Default to scientist for unknown roles
                        agent_key = "scientist"
                    
                    # Log the agent mapping decision
                    logger.info(f"Mapping agent '{chosen_agent}' with role '{agent_role}' to agent_key '{agent_key}'")
                    
                    try:
                        # Call the chosen agent
                        agent_reply = await self.llm_client.call_agent(
                            agent_key=agent_key,
                            conversation_history=conversation_history,
                            expertise=agent.get("expertise") if agent else None,
                            goal=agent.get("goal") if agent else None,
                            agent_role=agent.get("role") if agent else None,
                            agent_name=chosen_agent
                        )
                    except Exception as e:
                        logger.error(f"Error calling agent {chosen_agent} with agent_key {agent_key}: {e}")
                        # Provide a fallback response rather than failing completely
                        agent_reply = f"[System: Unable to get a response from {chosen_agent} due to an error. The conversation will continue with other agents.]"
                    
                    # Update conversation history
                    conversation_history += f"\n[{chosen_agent}]: {agent_reply}"
                    meeting_data["conversation_history"] = conversation_history
                    
                    # Create a transcript entry
                    await self.create_transcript(
                        meeting_id=meeting_id,
                        agent_name=chosen_agent,
                        round_number=round_index,
                        content=agent_reply
                    )
                    
                    # Send a new message for the agent's response instead of editing
                    if live_mode:
                        await interaction.followup.send(f"**[{chosen_agent}]**: {agent_reply}", ephemeral=False)
                        
                    calls_this_round += 1
                    
                except Exception as e:
                    logger.error(f"Error in conversation round {round_index}, call {calls_this_round}: {e}")
                    calls_this_round += 1  # Still increment to prevent infinite loops
                    
            # End of round: PI synthesizes what's been said
            try:
                pi_synthesis_prompt = conversation_history + "\nNow please synthesize this round's points concisely, and ask a couple focused follow-up questions for the next round."
                
                pi_synthesis = await self.llm_client.call_agent(
                    agent_key="principal_investigator",
                    conversation_history=pi_synthesis_prompt,
                    expertise=pi_agent.get("expertise"),
                    goal=pi_agent.get("goal"),
                    agent_role=pi_agent.get("role")
                )
                
                # Update conversation history
                conversation_history += f"\n\n[Principal Investigator (round synthesis)]: {pi_synthesis}"
                meeting_data["conversation_history"] = conversation_history
                
                # Create a transcript entry
                await self.create_transcript(
                    meeting_id=meeting_id,
                    agent_name="Principal Investigator (synthesis)",
                    round_number=round_index,
                    content=pi_synthesis
                )
                
                # Send a new message for the PI's synthesis
                if live_mode:
                    await interaction.followup.send(f"**[Principal Investigator (synthesis)]**: {pi_synthesis}", ephemeral=False)
                    
            except Exception as e:
                logger.error(f"Error in PI synthesis for round {round_index}: {e}")
                
        # Final summary after all rounds
        try:
            final_summary = await self.llm_client.call_agent(
                agent_key="summary_agent",
                conversation_history=conversation_history
            )
            
            # Update conversation history
            conversation_history += f"\n\n=== FINAL SUMMARY ===\n{final_summary}"
            meeting_data["conversation_history"] = conversation_history
            meeting_data["summary"] = final_summary
            
            # Create a transcript entry
            await self.create_transcript(
                meeting_id=meeting_id,
                agent_name="Summary Agent",
                round_number=round_count + 1,
                content=final_summary
            )
            
            # Send a new message for the final summary
            if live_mode:
                await interaction.followup.send(f"**=== FINAL SUMMARY ===**\n{final_summary}", ephemeral=False)
                
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            
        # Mark meeting as completed
        meeting_data["is_active"] = False
        
        return True
        
    async def end_conversation(self, meeting_id):
        """End a conversation for a meeting.
        
        Args:
            meeting_id: ID of the meeting
        """
        logger.info("Ending conversation for meeting " + str(meeting_id))
        
        meeting_data = self.active_meetings.get(meeting_id)
        if meeting_data:
            meeting_data["is_completed"] = True
            meeting_data["is_active"] = False
            
        return True
        
    async def run_conversation(self, meeting_id):
        """
        DEPRECATED: This method is deprecated. Use start_conversation() instead.
        
        Run a multi-agent conversation with rounds.
        - Each round has multiple agent calls (orchestrator decides who speaks)
        - At the end of each round, the PI synthesizes the discussion
        """
        # Log deprecation warning
        logger.warning("run_conversation() is deprecated. Use start_conversation() instead.")
        
        # Get meeting data
        meeting_data = self.active_meetings.get(meeting_id)
        if not meeting_data:
            logger.error(f"Meeting {meeting_id} not found")
            return False
            
        # Redirect to the new method if possible
        interaction = meeting_data.get("interaction")
        live_mode = meeting_data.get("live_mode", True)
        
        if interaction:
            return await self.start_conversation(
                meeting_id=meeting_id,
                interaction=interaction,
                live_mode=live_mode
            )
        else:
            logger.error("Cannot redirect to start_conversation: missing interaction")
            return False
        
    async def create_transcript(self, meeting_id, agent_name, round_number, content):
        """Create a transcript entry in the database."""
        from db_client import db_client
        
        # Check if meeting exists
        meeting_data = self.active_meetings.get(meeting_id)
        if not meeting_data:
            logger.error(f"Meeting {meeting_id} not found for transcript creation")
            return False
            
        try:
            # Determine a suitable role based on agent name if not found
            agent_role = None
            if "Principal Investigator" in agent_name or "PI" in agent_name:
                agent_role = "Lead"
            elif "Critic" in agent_name:
                agent_role = "Critical Reviewer"
            elif "Summary" in agent_name:
                agent_role = "Summarizer" 
            else:
                agent_role = "Scientist"  # Default role for most agents
            
            # Validate required fields
            if not meeting_id or not agent_name or not content:
                logger.error(f"Missing required transcript fields: meeting_id={meeting_id}, agent_name={agent_name}")
                return False
                
            # Create transcript entry with the required fields
            logger.info(f"Creating transcript: meeting={meeting_id}, agent={agent_name}, role={agent_role}, round={round_number}")
            result = await db_client.create_transcript(
                meeting_id=meeting_id,
                agent_name=agent_name,
                round_number=round_number,
                content=content,
                agent_role=agent_role
            )
            
            # Check for API errors
            if isinstance(result, dict) and result.get("isSuccess") is False:
                error_msg = result.get('message', 'Unknown API error')
                logger.error(f"API error creating transcript: {error_msg}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error creating transcript: {e}")
            return False

    async def generate_combined_summary(self, meetings):
        """Generate a combined summary of parallel meetings."""
        logger.info("Generating combined summary for " + str(len(meetings)) + " meetings")
        
        # Extract summaries from each meeting
        parallel_summaries = []
        for meeting in meetings:
            if meeting.get("summary"):
                parallel_summaries.append(meeting.get("summary"))
        
        # If only one meeting or no summaries, return the single summary
        if len(parallel_summaries) <= 1:
            return parallel_summaries[0] if parallel_summaries else "No summary available."
        
        # Create a prompt for combining the summaries
        messages = [
            LLMMessage(
                role="system",
                content="You are an expert research synthesizer. Your task is to combine multiple parallel brainstorming sessions into a cohesive summary."
            ),
            LLMMessage(
                role="user",
                content="Topic: " + meetings[0]['agenda'] + "\n\nParticipants: " + ", ".join(agent['name'] for agent in meetings[0]['agents']) + "\n\nIndividual Run Summaries:\n" + "\n\n".join(parallel_summaries) + "\n\nProvide a synthesis that includes:\n1. Common themes and consensus across runs\n2. Unique insights and novel approaches from each run\n3. Contrasting perspectives and alternative solutions\n4. Integrated conclusions and recommendations\n5. Key areas for further investigation"
            )
        ]
        
        try:
            response = await self.llm_client.generate_response(
                provider=LLMProvider.OPENAI,
                messages=messages,
                temperature=0.7
            )
            
            return response.content
            
        except Exception as e:
            logger.error("Error generating combined summary: " + str(e))
            return "Error generating combined summary." 