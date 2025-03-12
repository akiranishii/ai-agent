import logging
import asyncio
import json
from typing import List, Dict, Optional, Any, Set
from models import LLMProvider, LLMMessage
from pydantic import BaseModel, ValidationError
import discord
from datetime import datetime

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
    
    async def initialize_meeting(self, meeting_id, session_id, agents, agenda, round_count, parallel_index=0, total_parallel_meetings=1):
        """Initialize a meeting.
        
        Args:
            meeting_id: ID of the meeting
            session_id: ID of the session
            agents: List of agents for the meeting
            agenda: Agenda/topic for the meeting
            round_count: Number of rounds for the meeting
            parallel_index: Index of this meeting in parallel meetings (0-based)
            total_parallel_meetings: Total number of parallel meetings
        """
        logger.info(f"Initializing meeting {meeting_id} with {len(agents)} agents, parallel_index={parallel_index}, total_parallel_meetings={total_parallel_meetings}")
        
        # Add meeting to active meetings dict
        self.active_meetings[meeting_id] = {
            "id": meeting_id,
            "session_id": session_id,
            "agents": agents,
            "agenda": agenda,
            "round_count": round_count,
            "current_round": 0,
            "is_active": True,
            "is_completed": False,
            "thread": None,
            "parallel_index": parallel_index,
            "total_parallel_meetings": total_parallel_meetings,
            "base_meeting_id": meeting_id,  # Store the original meeting ID for parallel meetings
            "start_time": datetime.now().isoformat(),
            "messages": [],
            "summary": None,
            "conversation_history": f"The user wants to discuss: {agenda}\n\n"
        }
        
        # Create a unique opening for the meeting based on the agenda
        for agent in agents:
            if agent.get("role", "").lower() == "lead" or "principal investigator" in agent.get("name", "").lower():
                # Set up the PI's opening message
                try:
                    opening_message = await self.llm_client.generate_opening(
                        agent_key="pi",
                        agenda=agenda,
                        agents=[agent.get("name") for agent in agents]
                    )
                    
                    # Store the opening message
                    self.active_meetings[meeting_id]["opening_message"] = opening_message
                except Exception as e:
                    logger.error(f"Error generating opening message: {e}")
                    self.active_meetings[meeting_id]["opening_message"] = f"Welcome to our discussion on {agenda}. Let's begin our collaborative exploration of this topic."
                
                break
        
        # Track parallel meetings
        if session_id not in self.parallel_groups:
            self.parallel_groups[session_id] = set()
        self.parallel_groups[session_id].add(meeting_id)
        
        logger.info("Initialized meeting " + str(meeting_id) + " (parallel index " + str(parallel_index) + ") for session " + str(session_id))
        return True
    
    async def start_conversation(self, meeting_id: str, interaction: discord.Interaction, live_mode: bool = True, conversation_length: int = None):
        """
        Start a multi-agent conversation with rounds.
        
        Args:
            meeting_id: ID of the meeting
            interaction: Discord interaction object
            live_mode: Whether to show agent responses in real-time (default: True)
            conversation_length: Number of speakers per round (default: all agents excluding PI)
        """
        # Get meeting data
        meeting_data = self.active_meetings.get(meeting_id)
        if not meeting_data:
            logger.error(f"Meeting {meeting_id} not found")
            return False
            
        # Store live_mode in meeting data to ensure it's consistent
        meeting_data["live_mode"] = live_mode
        
        # Log key information for debugging
        logger.info(f"Starting conversation for meeting {meeting_id} with live_mode={live_mode}")
        logger.info(f"Number of agents: {len(meeting_data.get('agents', []))}")
        for agent in meeting_data.get('agents', []):
            logger.info(f"Agent: {agent.get('name')}, Role: {agent.get('role')}")
            
        # Get session ID and round count from meeting data
        session_id = meeting_data.get("session_id")
        round_count = meeting_data.get("round_count", 3)
        parallel_index = meeting_data.get("parallel_index", 0)
        total_parallel_meetings = meeting_data.get("total_parallel_meetings", 1)
        
        # Determine if we should use simple mode (no threads, no meeting numbers)
        # Use simple mode when there's only one meeting or this is quickstart
        use_simple_mode = total_parallel_meetings <= 1
        logger.info(f"Using simple mode: {use_simple_mode} (total meetings: {total_parallel_meetings})")
        
        # Initialize conversation history if it doesn't exist
        conversation_history = meeting_data.get("conversation_history", "")
        if not conversation_history:
            # Add basic info to conversation history
            agenda = meeting_data.get("agenda", "No agenda specified")
            
            if use_simple_mode:
                conversation_history = f"Lab Meeting\nAgenda: {agenda}\n\nParticipants:\n"
            else:
                conversation_history = f"Lab Meeting #{parallel_index + 1}\nAgenda: {agenda}\n\nParticipants:\n"
                
            meeting_data["conversation_history"] = conversation_history
        
        # Get agents
        agents = meeting_data.get("agents", [])
        if not agents:
            logger.error(f"No agents found for meeting {meeting_id}")
            return False
            
        # Find the Principal Investigator (PI)
        pi_agent = next((a for a in agents if a["role"] == "Lead" or a["role"] == "Principal Investigator"), None)
        if not pi_agent:
            logger.error(f"No PI found for meeting {meeting_id}")
            return False
            
        # If conversation_length is None, default to all non-PI agents
        if conversation_length is None:
            # Count all agents except the PI
            non_pi_count = sum(1 for a in agents if a["name"] != "Principal Investigator")
            conversation_length = non_pi_count
            logger.info(f"Using default conversation_length of {conversation_length} (all non-PI agents)")
        else:
            logger.info(f"Using specified conversation_length of {conversation_length}")
            
        # Store conversation_length in meeting data
        meeting_data["conversation_length"] = conversation_length
        
        # Initial message to set up the conversation if in live mode
        if live_mode:
            try:
                if use_simple_mode:
                    # SIMPLE MODE: Just send messages directly to the channel
                    initial_message = f"**Starting Lab Meeting**\n\nAgenda: {meeting_data.get('agenda', 'No agenda specified')}"
                    await interaction.followup.send(initial_message, ephemeral=False)
                    
                    # Show that we're generating participants
                    await interaction.followup.send("**Generating participants...**", ephemeral=False)
                    
                    # Add each agent in separate messages with clear formatting
                    for agent in agents:
                        name = agent.get('name', 'Unknown Agent')
                        role = agent.get('role', 'Unknown Role')
                        
                        # Create a formatted card showing this agent's capabilities
                        agent_card = f"**Generated {name}** ({role})\n"
                        
                        # Format fields with clear labels
                        field_parts = []
                        
                        # Add expertise if available
                        if agent.get('expertise'):
                            field_parts.append(f"**Expertise:** {agent.get('expertise')}")
                        
                        # Add goal if available
                        if agent.get('goal'):
                            field_parts.append(f"**Goal:** {agent.get('goal')}")
                        
                        # Add the formatted parts
                        if field_parts:
                            formatted_fields = "\n".join(field_parts)
                            agent_card += f"```\n{formatted_fields}\n```"
                        
                        # Send each agent as a separate message
                        await interaction.followup.send(agent_card, ephemeral=False)
                    
                    # Send the final participants list message
                    participants_message = "**All participants ready! Starting lab meeting...**"
                    await interaction.followup.send(participants_message, ephemeral=False)
                    
                else:
                    # THREAD MODE: Create a dedicated thread for parallel meetings
                    # Create a dedicated thread for this parallel meeting with a unique name
                    thread_name = f"Lab Meeting #{parallel_index + 1} - {meeting_data.get('agenda', 'Discussion')[:50]}"
                    logger.info(f"Creating thread with name: {thread_name}")
                    
                    # Send initial message to the main channel
                    initial_message = f"**Starting Lab Meeting #{parallel_index + 1}**\n\nAgenda: {meeting_data.get('agenda', 'No agenda specified')}"
                    
                    # Send the message to create a thread from
                    initial_msg = await interaction.channel.send(initial_message)
                    
                    # Create a thread from this message
                    thread = await initial_msg.create_thread(
                        name=thread_name,
                        auto_archive_duration=1440  # 24 hours
                    )
                    
                    # Store the thread in meeting data
                    meeting_data["thread"] = thread
                    logger.info(f"Created thread '{thread_name}' for meeting {meeting_id}")
                    
                    # Send welcome message in the thread
                    welcome_message = f"**Welcome to Lab Meeting #{parallel_index + 1}!**\n\nThis thread will contain all the conversation for parallel run #{parallel_index + 1}."
                    await thread.send(welcome_message)
                    
                    # Show that we're generating participants
                    await thread.send("**Generating participants...**")
                    
                    # Add each agent in separate messages with clear formatting
                    for agent in agents:
                        name = agent.get('name', 'Unknown Agent')
                        role = agent.get('role', 'Unknown Role')
                        
                        # Create a formatted card showing this agent's capabilities
                        agent_card = f"**Generated {name}** ({role})\n"
                        
                        # Format fields with clear labels
                        field_parts = []
                        
                        # Add expertise if available
                        if agent.get('expertise'):
                            field_parts.append(f"**Expertise:** {agent.get('expertise')}")
                        
                        # Add goal if available
                        if agent.get('goal'):
                            field_parts.append(f"**Goal:** {agent.get('goal')}")
                        
                        # Add the formatted parts
                        if field_parts:
                            formatted_fields = "\n".join(field_parts)
                            agent_card += f"```\n{formatted_fields}\n```"
                        
                        # Send each agent as a separate message in the thread
                        await thread.send(agent_card)
                    
                    # Send the final participants list message
                    participants_message = f"**All participants ready! Starting lab meeting #{parallel_index + 1}...**"
                    await thread.send(participants_message)
                
            except Exception as e:
                # Log the error but continue with the conversation
                logger.error(f"Error setting up conversation: {str(e)}")
                # Fallback to regular messages
                if use_simple_mode:
                    await interaction.followup.send(
                        f"**Lab Meeting**\n\nAgenda: {meeting_data.get('agenda', 'No agenda specified')}\n\n**Generating participants...**", 
                        ephemeral=False
                    )
                else:
                    await interaction.followup.send(
                        f"**Lab Meeting #{parallel_index + 1}**\n\nAgenda: {meeting_data.get('agenda', 'No agenda specified')}\n\n**Generating participants...**", 
                        ephemeral=False
                    )
                
                # Add each agent as fallback
                for agent in agents:
                    await interaction.followup.send(
                        f"**Generated {agent.get('name', 'Unknown Agent')}** ({agent.get('role', 'Unknown Role')})",
                        ephemeral=False
                    )
                
                # Final message
                if use_simple_mode:
                    await interaction.followup.send(
                        "**All participants ready! Starting lab meeting...**",
                        ephemeral=False
                    )
                else:
                    await interaction.followup.send(
                        f"**All participants ready! Starting lab meeting #{parallel_index + 1}...**",
                        ephemeral=False
                    )
            
        # Set meeting to active
        meeting_data["is_active"] = True
        
        # Have the PI start the discussion with initial thoughts and guiding questions
        try:
            # Get the agenda and any existing context
            agenda = meeting_data.get('agenda', 'No agenda specified')
            
            # Include the agenda and supplementary information in the conversation history
            pi_context = f"Lab Meeting Topic: {agenda}\n\n"
            
            # Add any existing conversation history if available
            if conversation_history:
                pi_context += f"Prior discussion: {conversation_history}\n\n"
                
            # Construct a special prompt for the PI to start the meeting
            pi_instructions = f"""You are the Principal Investigator leading this lab meeting.
            
            Please provide:
            1. A brief introduction to the topic
            2. Your initial thoughts on the approach
            3. 1-3 specific guiding questions for your team to address (maximum 3 questions)
            
            Keep your response focused and concise."""
            
            # Combine context and instructions
            pi_full_prompt = pi_context + pi_instructions
            
            # Call the PI with this special opening prompt
            pi_opening = await self.llm_client.call_agent(
                agent_key="principal_investigator",
                conversation_history=pi_full_prompt,
                expertise=pi_agent.get("expertise"),
                goal=pi_agent.get("goal"),
                agent_role=None  # Let call_agent use the default role
            )
            
            # Update conversation history with the PI's opening
            conversation_history += f"\n\n[Principal Investigator (Opening)]: {pi_opening}"
            meeting_data["conversation_history"] = conversation_history
            
            # Create a transcript entry for the PI's opening
            await self.create_transcript(
                meeting_id=meeting_id,
                agent_name="Principal Investigator (Opening)",
                round_number=0,  # Round 0 indicates pre-rounds
                content=pi_opening
            )
            
            # Send the PI's opening in Discord
            if live_mode:
                use_simple_mode = meeting_data.get("total_parallel_meetings", 1) <= 1
                
                # Check if we have a thread to use (for parallel meetings only)
                thread = meeting_data.get("thread") if not use_simple_mode else None
                if thread and hasattr(thread, "send"):
                    try:
                        await thread.send(f"**[Principal Investigator (Opening)]**: {pi_opening}")
                    except Exception as e:
                        logger.error(f"Error sending to thread: {e}")
                        # Fallback to regular channel
                        await interaction.followup.send(f"**[Principal Investigator (Opening)]**: {pi_opening}", ephemeral=False)
                else:
                    # Use regular channel
                    await interaction.followup.send(f"**[Principal Investigator (Opening)]**: {pi_opening}", ephemeral=False)
            
        except Exception as e:
            logger.error(f"Error getting PI opening statement: {e}")
        
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
                use_simple_mode = meeting_data.get("total_parallel_meetings", 1) <= 1
                
                if use_simple_mode:
                    round_message = f"**=== ROUND {round_index} of {round_count} ===**"
                else:
                    round_message = f"**=== ROUND {round_index} of {round_count} ===**"
                
                # Check if we have a thread to use (for parallel meetings only)
                thread = meeting_data.get("thread") if not use_simple_mode else None
                if thread and hasattr(thread, "send"):
                    try:
                        await thread.send(round_message)
                    except Exception as e:
                        logger.error(f"Error sending to thread: {e}")
                        # Fallback to regular channel
                        await interaction.followup.send(round_message, ephemeral=False)
                else:
                    # Use regular channel
                    await interaction.followup.send(round_message, ephemeral=False)
                
            # Inner loop: up to 'conversation_length' calls (the orchestrator chooses who speaks)
            calls_this_round = 0
            max_calls = meeting_data.get("conversation_length", 2)  # Get from meeting data with fallback to 2
            while calls_this_round < max_calls:
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
                            agent_role=None,  # Let call_agent use the appropriate default role
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
                        try:
                            logger.info(f"Sending message from {chosen_agent} to Discord (live_mode={live_mode})")
                            use_simple_mode = meeting_data.get("total_parallel_meetings", 1) <= 1
                            
                            # Check if we have a thread to use (for parallel meetings only)
                            thread = meeting_data.get("thread") if not use_simple_mode else None
                            if thread and hasattr(thread, "send"):
                                try:
                                    await thread.send(f"**[{chosen_agent}]**: {agent_reply}")
                                except Exception as e:
                                    logger.error(f"Error sending to thread: {e}")
                                    # Fallback to regular channel
                                    await interaction.followup.send(f"**[{chosen_agent}]**: {agent_reply}", ephemeral=False)
                            else:
                                # Use regular channel
                                await interaction.followup.send(f"**[{chosen_agent}]**: {agent_reply}", ephemeral=False)
                        except Exception as discord_error:
                            logger.error(f"Error sending message to Discord: {discord_error}")
                        
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
                    agent_role=None  # Let call_agent use the default role
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
                    try:
                        logger.info("Sending PI synthesis to Discord")
                        use_simple_mode = meeting_data.get("total_parallel_meetings", 1) <= 1
                        
                        # Check if we have a thread to use (for parallel meetings only)
                        thread = meeting_data.get("thread") if not use_simple_mode else None
                        if thread and hasattr(thread, "send"):
                            try:
                                await thread.send(f"**[Principal Investigator (synthesis)]**: {pi_synthesis}")
                            except Exception as e:
                                logger.error(f"Error sending to thread: {e}")
                                # Fallback to regular channel
                                await interaction.followup.send(f"**[Principal Investigator (synthesis)]**: {pi_synthesis}", ephemeral=False)
                        else:
                            # Use regular channel
                            await interaction.followup.send(f"**[Principal Investigator (synthesis)]**: {pi_synthesis}", ephemeral=False)
                    except Exception as discord_error:
                        logger.error(f"Error sending PI synthesis to Discord: {discord_error}")
                    
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
                round_number=-1,  # -1 indicates summary
                content=final_summary
            )
            
            # Send a new message for the final summary
            if live_mode:
                try:
                    logger.info("Sending final summary to Discord")
                    use_simple_mode = meeting_data.get("total_parallel_meetings", 1) <= 1
                    
                    # Check if we have a thread to use (for parallel meetings only)
                    thread = meeting_data.get("thread") if not use_simple_mode else None
                    if thread and hasattr(thread, "send"):
                        try:
                            await thread.send(f"**=== FINAL SUMMARY ===**\n{final_summary}")
                        except Exception as e:
                            logger.error(f"Error sending to thread: {e}")
                            # Fallback to regular channel
                            await interaction.followup.send(f"**=== FINAL SUMMARY ===**\n{final_summary}", ephemeral=False)
                    else:
                        # Use regular channel
                        await interaction.followup.send(f"**=== FINAL SUMMARY ===**\n{final_summary}", ephemeral=False)
                except Exception as discord_error:
                    logger.error(f"Error sending final summary to Discord: {discord_error}")
                
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            
        # Mark meeting as completed
        meeting_data["is_active"] = False
        
        return True
        
    async def end_conversation(self, meeting_id):
        """End a conversation."""
        from db_client import db_client
        
        # Check if meeting exists
        meeting_data = self.active_meetings.get(meeting_id)
        if not meeting_data:
            logger.error(f"Meeting {meeting_id} not found for ending")
            return False
            
        # Mark the meeting as inactive 
        meeting_data["is_active"] = False
        logger.info(f"Marked meeting {meeting_id} as inactive")
            
        # If meeting is part of a parallel group, check if it's the last one to finish
        # and generate a combined summary if it is
        session_id = meeting_data.get("session_id")
        parallel_index = meeting_data.get("parallel_index", 0)
        total_parallel_meetings = meeting_data.get("total_parallel_meetings", 1)
        
        # Only check for combined summary if we have multiple parallel meetings
        generate_combined_summary = total_parallel_meetings > 1
        
        # Get all meetings for this session and check their state
        if generate_combined_summary and session_id:
            # Build a list of all meetings in the same parallel group (same session + same base meeting)
            parallel_meetings = []
            parallel_meeting_ids = []
            
            for m_id, m_data in self.active_meetings.items():
                if m_data.get("session_id") == session_id and m_data.get("base_meeting_id") == meeting_data.get("base_meeting_id"):
                    parallel_meetings.append(m_data)
                    parallel_meeting_ids.append(m_id)
                    
            logger.info(f"Found {len(parallel_meetings)} parallel meetings in session {session_id}")
            
            # Prepare to check if this is the last meeting to end
            # We need to verify with the database which meetings are still active
            try:
                # Save current meeting's state before checking others
                await db_client.end_meeting(meeting_id=meeting_id)
                
                # Check how many meetings are still active in the database
                meetings_result = await db_client.get_active_meetings(session_id=session_id)
                active_meetings = []
                
                if meetings_result.get("isSuccess") and meetings_result.get("data"):
                    # Filter to only include meetings in our parallel group
                    active_meetings = [
                        m for m in meetings_result.get("data", [])
                        if m.get("id") in parallel_meeting_ids
                    ]
                
                # If this was the last meeting to end, generate combined summary
                if len(active_meetings) == 0 and len(parallel_meetings) > 1 and parallel_meetings[0].get("interaction"):
                    logger.info(f"All {len(parallel_meetings)} parallel meetings have ended. Generating combined summary.")
                    interaction = parallel_meetings[0].get("interaction")
                    
                    # Use the lab_meeting_commands module to generate and post the combined summary
                    # We need to import here to avoid circular imports
                    import importlib
                    try:
                        # Dynamically import the lab_meeting_commands module
                        lab_meeting_commands = importlib.import_module("commands.lab_meeting_commands")
                        
                        # Try to get the command cog
                        for cog in interaction.client.cogs.values():
                            if isinstance(cog, lab_meeting_commands.LabMeetingCommands):
                                logger.info("Found LabMeetingCommands cog. Generating combined summary.")
                                
                                # Get meeting data from API for all parallel meetings
                                ended_meetings = []
                                for m_id in parallel_meeting_ids:
                                    meeting_result = await db_client.get_meeting(meeting_id=m_id)
                                    if meeting_result.get("isSuccess") and meeting_result.get("data"):
                                        ended_meetings.append(meeting_result.get("data"))
                                
                                if ended_meetings:
                                    # Send notification to channel that combined summary is being generated
                                    await interaction.channel.send(
                                        "🔄 **All parallel meetings have completed. Generating combined summary of all runs...**"
                                    )
                                    
                                    # Call the method to generate and post the combined summary
                                    await cog.generate_and_post_combined_summary(
                                        interaction=interaction, 
                                        ended_meetings=ended_meetings,
                                        orchestrator=self
                                    )
                                    
                                break
                    except Exception as e:
                        logger.error(f"Error generating automatic combined summary: {e}")
            except Exception as e:
                logger.error(f"Error checking for combined summary generation: {e}")
        
        # Remove from active meetings dict
        if meeting_id in self.active_meetings:
            del self.active_meetings[meeting_id]
            
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
            # For database enum compatibility, only use standard role values
            # Based on errors, the database likely only accepts "user" or "assistant"
            agent_role = "assistant"  # Default to assistant for all roles
            
            # Convert negative round numbers to special positive ones for database compatibility
            # Use 9999 for summary (-1) to ensure it's at the end but still a positive number
            api_round_number = round_number
            if round_number < 0:
                if round_number == -1:  # Summary
                    api_round_number = 9999
                else:
                    api_round_number = 9000 + abs(round_number)  # Other special rounds if needed
                    
            logger.info(f"Creating transcript with round number {api_round_number} (original: {round_number})")
            
            # Validate required fields
            if not meeting_id or not agent_name or not content:
                logger.error(f"Missing required transcript fields: meeting_id={meeting_id}, agent_name={agent_name}")
                return False
                
            # Create transcript entry with the required fields
            logger.info(f"Creating transcript: meeting={meeting_id}, agent={agent_name}, role={agent_role}, round={api_round_number}")
            
            # Truncate the content if it's too long (most databases have limits)
            MAX_CONTENT_LENGTH = 8000  # Reduced from 10000 to be safer
            if len(content) > MAX_CONTENT_LENGTH:
                original_length = len(content)
                content = content[:MAX_CONTENT_LENGTH - 100] + f"\n\n[Note: Content truncated from {original_length} characters]"
                logger.warning(f"Truncated transcript content from {original_length} to {len(content)} characters")
            
            # Make sure agent_name is within reasonable bounds
            if len(agent_name) > 100:  # Most databases have varchar limits
                agent_name = agent_name[:97] + "..."
                
            # IMPORTANT: Remember the summary in the meeting data even if saving to DB fails
            if round_number == -1 and meeting_data:
                meeting_data["summary"] = content
                logger.info(f"Saved summary to meeting_data for meeting {meeting_id}")
                
            try:
                result = await db_client.create_transcript(
                    meeting_id=meeting_id,
                    agent_name=agent_name,
                    round_number=api_round_number,
                    content=content,
                    agent_role=agent_role
                )
                
                # Check for API errors
                if isinstance(result, dict) and result.get("isSuccess") is False:
                    error_msg = result.get('message', 'Unknown API error')
                    logger.error(f"API error creating transcript: {error_msg}")
                    
                    # Try one more time with an even safer approach for special cases
                    logger.info("Retrying transcript creation with safer parameters")
                    retry_result = await db_client.create_transcript(
                        meeting_id=meeting_id,
                        agent_name="Meeting Summary",  # Simpler name
                        round_number=meeting_data.get("round_count", 999),  # Use final round from meeting data
                        content=content[:5000] if len(content) > 5000 else content,  # Ensure shorter content
                        agent_role="assistant"  # Use a standard role that should be in the enum
                    )
                    
                    if isinstance(retry_result, dict) and retry_result.get("isSuccess") is True:
                        logger.info("Retry transcript creation succeeded")
                        return True
                        
                return False
            except Exception as api_error:
                logger.error(f"Exception during API call to create transcript: {api_error}")
                # Try one final time with minimal data
                try:
                    logger.info("Final retry with minimal data")
                    final_retry = await db_client.create_transcript(
                        meeting_id=meeting_id,
                        agent_name="System",
                        round_number=888,  # A safe, positive number
                        content=content[:2000] if len(content) > 2000 else content,  # Very short content
                        agent_role="assistant"  # Use a standard role that should be in the enum
                    )
                    return isinstance(final_retry, dict) and final_retry.get("isSuccess") is True
                except Exception as final_error:
                    logger.error(f"Final retry also failed: {final_error}")
                    return False
                
            return True
        except Exception as e:
            logger.error(f"Error creating transcript: {e}")
            return False

    async def generate_combined_summary(self, meetings):
        """Generate a combined summary of parallel meetings.
        
        Args:
            meetings: List of meeting data objects
            
        Returns:
            Combined summary text
        """
        logger.info(f"Generating combined summary for {len(meetings)} meetings")
        
        if not meetings:
            logger.error("No meetings provided for combined summary")
            return "No meeting data available for generating a combined summary."
        
        # Extract summaries from each meeting
        parallel_summaries = []
        meeting_agenda = None
        participants = set()
        
        for meeting in meetings:
            if meeting.get("summary"):
                parallel_idx = meeting.get("parallel_index", 0)
                parallel_summaries.append({
                    "index": parallel_idx + 1,  # 1-based for user-facing content
                    "summary": meeting.get("summary")
                })
                
            # Capture agenda (should be the same for all meetings)
            if not meeting_agenda and meeting.get("agenda"):
                meeting_agenda = meeting.get("agenda")
                
            # Capture unique participant names
            for agent in meeting.get("agents", []):
                if agent.get("name"):
                    participants.add(agent.get("name"))
        
        # If no summaries found
        if not parallel_summaries:
            logger.warning("No summaries found in meeting data")
            return "No summaries available for the parallel runs."
        
        # If only one meeting or summary, return it directly
        if len(parallel_summaries) == 1:
            logger.info("Only one summary found, returning it directly")
            return parallel_summaries[0]["summary"]
        
        # Sort summaries by their parallel index
        parallel_summaries.sort(key=lambda x: x["index"])
        
        # Create a formatted input for the combined summary
        formatted_summaries = []
        for summary_data in parallel_summaries:
            formatted_summaries.append(f"RUN #{summary_data['index']} SUMMARY:\n{summary_data['summary']}")
        
        # Create a prompt for combining the summaries
        messages = [
            LLMMessage(
                role="system",
                content="You are an expert research synthesizer. Your task is to combine multiple parallel brainstorming sessions into a cohesive summary."
            ),
            LLMMessage(
                role="user",
                content=(
                    f"Topic: {meeting_agenda or 'Unknown Topic'}\n\n"
                    f"Participants: {', '.join(sorted(participants))}\n\n"
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
        
        try:
            logger.info("Calling LLM to generate combined summary")
            response = await self.llm_client.generate_response(
                provider=LLMProvider.OPENAI,
                messages=messages,
                temperature=0.7,
                max_tokens=2000  # Ensure we get a substantial response
            )
            
            logger.info(f"Generated combined summary of length {len(response.content)}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating combined summary: {str(e)}")
            return f"Error generating combined summary: {str(e)}" 