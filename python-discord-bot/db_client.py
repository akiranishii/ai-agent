import os
import logging
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
import asyncio

from config import API_BASE_URL

logger = logging.getLogger(__name__)

class DatabaseClient:
    """Client for interacting with the application's database via API calls."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        """Initialize the database client with the base API URL.
        
        Args:
            base_url: Base URL for the API endpoints
        """
        # Ensure the base_url doesn't end with a slash
        self.base_url = base_url.rstrip('/')
        logger.info(f"DatabaseClient initialized with base URL: {self.base_url}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the API is reachable.
        
        Returns:
            Status information
        """
        health_endpoint = "/health"
        # Remove duplicate /api in the endpoint if base_url already ends with /api
        if self.base_url.endswith("/api") and health_endpoint.startswith("/api/"):
            health_endpoint = health_endpoint.replace("/api/", "/", 1)
        
        full_url = f"{self.base_url}{health_endpoint}"
        logger.info(f"Performing health check to: {full_url}")
        
        try:
            # Try to reach the base API to check connectivity
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, timeout=5) as response:
                    if response.status == 200:
                        return {"isSuccess": True, "message": "API is reachable", "data": None}
                    elif response.status == 404:
                        logger.error(f"Health check failed - endpoint not found (404): {full_url}")
                        return {"isSuccess": False, "message": f"API endpoint not found (404). Check if API_BASE_URL={self.base_url} is correct.", "data": None}
                    else:
                        logger.error(f"Health check failed - API returned status {response.status}: {full_url}")
                        return {"isSuccess": False, "message": f"API returned status {response.status}", "data": None}
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error during health check: {str(e)} - URL: {full_url}")
            return {"isSuccess": False, "message": f"Cannot connect to API: {str(e)}", "data": None}
        except Exception as e:
            logger.error(f"Unexpected error during health check: {str(e)} - URL: {full_url}")
            return {"isSuccess": False, "message": f"Unexpected error during health check: {str(e)}", "data": None}
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a request to the API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            
        Returns:
            Response data or error information
        """
        # Ensure endpoint starts with a slash
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        
        # Remove duplicate /api in the endpoint if base_url already ends with /api
        if self.base_url.endswith("/api") and endpoint.startswith("/api/"):
            endpoint = endpoint.replace("/api/", "/", 1)
            
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        logger.debug(f"Making {method} request to {url}")
        if params:
            logger.debug(f"Request params: {params}")
        if data:
            logger.debug(f"Request data: {data}")
        
        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    try:
                        async with session.get(url, params=params, headers=headers) as response:
                            if response.status >= 400:
                                error_text = await response.text()
                                logger.error(f"API error ({response.status}): {error_text}")
                                return {"isSuccess": False, "message": f"API error ({response.status}): {error_text}", "data": None}
                            
                            result = await response.json()
                            logger.debug(f"Response from {url}: {result}")
                            return result
                    except aiohttp.ClientConnectorError as e:
                        logger.error(f"HTTP error: {str(e)}")
                        return {"isSuccess": False, "message": f"Cannot connect to API: {str(e)}", "data": None}
                
                elif method == "POST":
                    try:
                        async with session.post(url, json=data, headers=headers) as response:
                            if response.status >= 400:
                                error_text = await response.text()
                                logger.error(f"API error ({response.status}): {error_text}")
                                return {"isSuccess": False, "message": f"API error ({response.status}): {error_text}", "data": None}
                            
                            result = await response.json()
                            logger.debug(f"Response from {url}: {result}")
                            return result
                    except aiohttp.ClientConnectorError as e:
                        logger.error(f"HTTP error: {str(e)}")
                        return {"isSuccess": False, "message": f"Cannot connect to API: {str(e)}", "data": None}
                
                elif method == "PUT":
                    async with session.put(url, json=data, headers=headers) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"API error ({response.status}): {error_text}")
                            return {"isSuccess": False, "message": f"API error: {response.status}", "data": None}
                        
                        result = await response.json()
                        return result
                
                elif method == "DELETE":
                    async with session.delete(url, json=data, headers=headers) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"API error ({response.status}): {error_text}")
                            return {"isSuccess": False, "message": f"API error: {response.status}", "data": None}
                        
                        result = await response.json()
                        return result
                
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {str(e)}")
            return {"isSuccess": False, "message": f"HTTP error: {str(e)}", "data": None}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"isSuccess": False, "message": f"Unexpected error: {str(e)}", "data": None}
    
    # Session-related methods
    async def get_active_session(self, user_id: str) -> Dict[str, Any]:
        """Get the active session for a user.
        
        Sessions use a boolean 'isActive' field to track their status.
        A session is considered active if isActive = true.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Session data or error information
        """
        return await self._make_request("GET", f"/discord/sessions/active", params={"userId": user_id})
    
    async def create_session(
        self, 
        user_id: str, 
        title: str, 
        description: Optional[str] = None,
        is_public: bool = False
    ) -> Dict[str, Any]:
        """Create a new session.
        
        Args:
            user_id: ID of the user
            title: Title of the session
            description: Optional description of the session
            is_public: Whether the session is public
            
        Returns:
            Session data or error information
        """
        data = {
            "userId": user_id,  # The API expects "userId"
            "title": title,
            "isPublic": is_public
        }
        
        if description:
            data["description"] = description
            
        # Log the data being sent
        logger.debug(f"Sending session creation data: {data}")
            
        return await self._make_request("POST", "/discord/sessions", data)
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session data or error information
        """
        return await self._make_request("PUT", f"/discord/sessions/{session_id}/end")
    
    async def get_session_agents(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Get agents for a session.
        
        Args:
            session_id: ID of the session
            user_id: ID of the user
            
        Returns:
            Agents data or error information
        """
        return await self._make_request(
            "GET", 
            f"/discord/sessions/{session_id}/agents", 
            params={"userId": user_id}
        )
    
    # Agent-related methods
    async def create_agent(
        self,
        session_id: str,
        name: str,
        role: str,
        user_id: str,
        goal: Optional[str] = None,
        expertise: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new agent.
        
        Args:
            session_id: ID of the session
            name: Name of the agent
            role: Role of the agent
            user_id: Discord user ID of the creator
            goal: Optional goal or description of the agent
            expertise: Optional area of expertise
            model: Optional model to use for the agent
            
        Returns:
            Agent data or error information
        """
        data = {
            "sessionId": session_id,
            "name": name,
            "role": role,
            "userId": user_id,
            "goal": goal,
            "expertise": expertise,
            "model": model
        }
        
        return await self._make_request("POST", "/discord/agents", data=data)
    
    # Meeting-related methods
    async def create_meeting(
        self, 
        session_id: str, 
        title: str, 
        agenda: Optional[str] = None,
        task_description: Optional[str] = None,
        max_rounds: Optional[int] = None,
        parallel_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new meeting in a session.
        
        Args:
            session_id: ID of the session
            title: Title of the meeting
            agenda: Optional agenda for the meeting
            task_description: Optional task description for the meeting
            max_rounds: Optional maximum number of rounds for the meeting
            parallel_index: Optional index for parallel meetings
            
        Returns:
            Meeting data or error information
        """
        data = {
            "sessionId": session_id,
            "title": title
        }
        
        if agenda:
            data["agenda"] = agenda
        if task_description:
            data["taskDescription"] = task_description
        if max_rounds:
            data["maxRounds"] = max_rounds
        if parallel_index is not None:  # Use is not None to allow 0 as a valid value
            data["parallelIndex"] = parallel_index
            
        return await self._make_request("POST", "/discord/meetings", data)
    
    async def end_meeting(self, meeting_id: str) -> Dict[str, Any]:
        """End a meeting.
        
        Args:
            meeting_id: ID of the meeting
            
        Returns:
            Meeting data or error information
        """
        return await self._make_request("PUT", f"/discord/meetings/{meeting_id}/end")
    
    # Transcript-related methods
    async def add_message(
        self, 
        meeting_id: str, 
        content: str, 
        role: str, 
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        round_number: Optional[int] = None,
        sequence_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Add a message to a meeting transcript.
        
        Args:
            meeting_id: ID of the meeting
            content: Content of the message
            role: Role of the message sender (user, assistant, system)
            agent_id: Optional ID of the agent
            agent_name: Optional name of the agent
            round_number: Optional round number for the message
            sequence_number: Optional sequence number within the round
            
        Returns:
            Message data or error information
        """
        data = {
            "meetingId": meeting_id,
            "content": content,
            "role": role
        }
        
        if agent_id:
            data["agentId"] = agent_id
        if agent_name:
            data["agentName"] = agent_name
        if round_number is not None:  # Allow 0 as a valid value
            data["roundNumber"] = round_number
        if sequence_number is not None:  # Allow 0 as a valid value
            data["sequenceNumber"] = sequence_number
            
        return await self._make_request("POST", "/discord/transcripts", data)

    # Additional Session-related methods
    async def get_user_sessions(self, user_id: str) -> Dict[str, Any]:
        """Get all sessions for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of sessions or error information
        """
        return await self._make_request("GET", "/discord/sessions", params={"userId": user_id})
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get a specific session by ID.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session data or error information
        """
        return await self._make_request("GET", f"/discord/sessions/{session_id}")
    
    async def reopen_session(self, session_id: str) -> Dict[str, Any]:
        """Reopen a previously ended session.
        
        Args:
            session_id: ID of the session to reopen
            
        Returns:
            Session data or error information
        """
        return await self._make_request("PUT", f"/discord/sessions/{session_id}/reopen")
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a session with new data.
        
        Args:
            session_id: ID of the session to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated session data or error information
        """
        return await self._make_request("PUT", f"/discord/sessions/{session_id}", updates)
    
    # Additional Meeting-related methods
    async def get_session_meetings(self, session_id: str) -> Dict[str, Any]:
        """Get all meetings for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of meetings or error information
        """
        return await self._make_request("GET", "/discord/meetings", params={"sessionId": session_id})
    
    async def get_meeting(self, meeting_id: str) -> Dict[str, Any]:
        """Get a specific meeting by ID.
        
        Args:
            meeting_id: ID of the meeting
            
        Returns:
            Meeting data or error information
        """
        return await self._make_request("GET", f"/discord/meetings/{meeting_id}")
    
    async def get_active_meetings(self, session_id: str) -> Dict[str, Any]:
        """Get active meetings for a session.
        
        Meetings use a status enum field: 'pending', 'in_progress', 'completed', 'failed'.
        A meeting is considered active if status = 'pending' or 'in_progress'.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of active meetings or error information
        """
        return await self._make_request("GET", "/discord/meetings/active", params={"sessionId": session_id})
    
    async def get_parallel_meetings(self, session_id: str, base_meeting_id: str) -> Dict[str, Any]:
        """Get parallel meetings related to a base meeting.
        
        Args:
            session_id: ID of the session
            base_meeting_id: ID of the base meeting
            
        Returns:
            List of parallel meetings or error information
        """
        return await self._make_request(
            "GET", 
            "/discord/meetings/parallel", 
            params={"sessionId": session_id, "baseMeetingId": base_meeting_id}
        )
    
    # Additional Agent-related methods
    async def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        role: Optional[str] = None,
        description: Optional[str] = None,
        expertise: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an existing agent.
        
        Args:
            agent_id: ID of the agent to update
            name: New name for the agent
            role: New role for the agent
            description: New description/goal for the agent
            expertise: New expertise for the agent
            model: New model for the agent
            
        Returns:
            Updated agent data or error information
        """
        data = {}
        
        if name:
            data["name"] = name
        if role:
            data["role"] = role
        if description:
            data["description"] = description
        if expertise:
            data["expertise"] = expertise
        if model:
            data["model"] = model
            
        return await self._make_request("PUT", f"/discord/agents/{agent_id}", data)
    
    async def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """Delete an agent.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            Result of deletion operation or error information
        """
        return await self._make_request("DELETE", f"/discord/agents/{agent_id}")
    
    async def get_agents_by_names(self, session_id: str, agent_names: List[str]) -> Dict[str, Any]:
        """Get agents by their names within a session.
        
        Args:
            session_id: ID of the session
            agent_names: List of agent names to find
            
        Returns:
            List of matching agents or error information
        """
        names_param = ",".join(agent_names)
        return await self._make_request(
            "GET", 
            f"/discord/sessions/{session_id}/agents", 
            params={"names": names_param}
        )
    
    # Transcript-related methods
    async def get_meeting_transcripts(self, meeting_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get transcripts for a meeting.
        
        Args:
            meeting_id: ID of the meeting
            limit: Optional limit on the number of transcripts to return
            
        Returns:
            List of transcripts or error information
        """
        params = {"meetingId": meeting_id}
        if limit:
            params["limit"] = limit
            
        return await self._make_request("GET", "/discord/transcripts", params=params)
        
    async def create_transcript(self, meeting_id: str, agent_name: str, round_number: int, content: str, agent_role: str = None) -> Dict[str, Any]:
        """Create a transcript entry for a meeting.
        
        Args:
            meeting_id: ID of the meeting
            agent_name: Name of the agent who generated the transcript
            round_number: Round number of the conversation
            content: Content of the agent's message
            agent_role: Role of the agent (optional, but may be required by API)
            
        Returns:
            Created transcript data or error information
        """
        data = {
            "meetingId": meeting_id,
            "agentName": agent_name,
            "roundNumber": round_number,
            "content": content
        }
        
        # Add agent role if provided
        if agent_role:
            data["role"] = agent_role
        
        return await self._make_request("POST", "/discord/transcripts", data=data)

    async def get_agent_by_name(self, session_id: str, agent_name: str) -> Dict[str, Any]:
        """Get an agent by name within a session.
        
        Args:
            session_id: ID of the session
            agent_name: Name of the agent to find
            
        Returns:
            Agent data or error information
        """
        # Use the existing get_agents_by_names method with a single name
        result = await self.get_agents_by_names(session_id, [agent_name])
        
        if not result.get("isSuccess", False):
            return result
        
        agents = result.get("data", [])
        
        if not agents:
            return {
                "isSuccess": False,
                "message": f"Agent with name '{agent_name}' not found in session {session_id}",
                "data": None
            }
        
        # Return the first (and hopefully only) agent with this name
        return {
            "isSuccess": True,
            "message": "Agent retrieved successfully",
            "data": agents[0]
        }

# Create a singleton instance
db_client = DatabaseClient() 