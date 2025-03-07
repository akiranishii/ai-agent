#!/usr/bin/env python3
"""
Mock test script for agent variables generation and usage.
This script simulates the behavior of the agent variables feature without calling the actual LLM client.
"""

import logging
import json
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self):
        """Initialize the mock LLM client."""
        self.providers = {
            "openai": True,
            "anthropic": True,
            "mistral": True
        }
    
    def get_available_providers(self):
        """Get a dictionary of available providers."""
        return self.providers
    
    async def generate_agent_variables(self, topic: str, agent_type: str) -> Dict[str, str]:
        """
        Generate variables for an agent based on the topic.
        
        Args:
            topic: The research topic or question
            agent_type: The type of agent ("principal_investigator" or "scientist")
            
        Returns:
            Dictionary with keys: expertise, goal, role, and for scientists: agent_name
        """
        logger.info(f"Generating variables for {agent_type} on topic: {topic}")
        
        if agent_type == "principal_investigator":
            return {
                "expertise": f"computational approaches to {topic}",
                "goal": f"develop innovative solutions for {topic} that advance the field",
                "role": "lead a multidisciplinary team and coordinate research efforts"
            }
        elif agent_type == "scientist":
            return {
                "agent_name": "Computational Biologist",
                "expertise": f"applying machine learning to {topic}",
                "goal": f"discover novel patterns and insights in {topic} data",
                "role": "analyze complex datasets and develop predictive models"
            }
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    async def call_agent(
        self, 
        agent_key: str, 
        conversation_history: str,
        expertise: Optional[str] = None,
        goal: Optional[str] = None,
        agent_role: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> str:
        """
        Mock call to an agent with the conversation so far.
        
        Args:
            agent_key: Key of the agent
            conversation_history: The conversation to respond to
            expertise: The agent's area of expertise (optional)
            goal: The agent's goal (optional)
            agent_role: The specific role description (optional)
            agent_name: The specific name for scientist agents (optional)
            
        Returns:
            The agent's response text
        """
        logger.info(f"Calling agent {agent_key} with variables:")
        logger.info(f"  Expertise: {expertise}")
        logger.info(f"  Goal: {goal}")
        logger.info(f"  Role: {agent_role}")
        logger.info(f"  Agent Name: {agent_name}")
        
        if agent_key == "principal_investigator":
            return f"As a Principal Investigator with expertise in {expertise or 'default expertise'}, my goal is to {goal or 'default goal'}. My role is to {agent_role or 'default role'}. I'm responding to: {conversation_history}"
        elif agent_key in ["biologist", "computer_scientist", "computational_biologist"]:
            return f"As a {agent_name or agent_key} with expertise in {expertise or 'default expertise'}, my goal is to {goal or 'default goal'}. My role is to {agent_role or 'default role'}. I'm responding to: {conversation_history}"
        else:
            return f"Unknown agent {agent_key} responding to: {conversation_history}"

# Create a mock instance
mock_llm_client = MockLLMClient()

async def test_generate_agent_variables():
    """Test the generate_agent_variables method."""
    try:
        # Test generating variables for Principal Investigator
        logger.info("Generating variables for Principal Investigator...")
        pi_variables = await mock_llm_client.generate_agent_variables(
            topic="protein folding prediction using deep learning",
            agent_type="principal_investigator"
        )
        logger.info("Principal Investigator variables:")
        logger.info(f"  Expertise: {pi_variables.get('expertise')}")
        logger.info(f"  Goal: {pi_variables.get('goal')}")
        logger.info(f"  Role: {pi_variables.get('role')}")
        
        # Test generating variables for Scientist
        logger.info("\nGenerating variables for Scientist...")
        scientist_variables = await mock_llm_client.generate_agent_variables(
            topic="protein folding prediction using deep learning",
            agent_type="scientist"
        )
        logger.info("Scientist variables:")
        logger.info(f"  Agent Name: {scientist_variables.get('agent_name')}")
        logger.info(f"  Expertise: {scientist_variables.get('expertise')}")
        logger.info(f"  Goal: {scientist_variables.get('goal')}")
        logger.info(f"  Role: {scientist_variables.get('role')}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing generate_agent_variables: {e}")
        return False

async def test_call_agent_with_variables():
    """Test the call_agent method with variables."""
    try:
        # Test calling Principal Investigator with variables
        logger.info("Calling Principal Investigator with variables...")
        pi_response = await mock_llm_client.call_agent(
            agent_key="principal_investigator",
            conversation_history="What is your expertise and goal?",
            expertise="computational biology and protein structure prediction",
            goal="develop novel methods for predicting protein folding",
            agent_role="lead a team of scientists to solve complex protein folding problems"
        )
        logger.info("Principal Investigator response:")
        logger.info(pi_response)
        
        # Test calling Scientist with variables
        logger.info("\nCalling Scientist with variables...")
        scientist_response = await mock_llm_client.call_agent(
            agent_key="biologist",
            conversation_history="What is your expertise and goal?",
            expertise="molecular dynamics simulations for protein folding",
            goal="improve the accuracy of protein structure predictions",
            agent_role="provide specialized insights on protein dynamics",
            agent_name="Computational Biophysicist"
        )
        logger.info("Scientist response:")
        logger.info(scientist_response)
        
        return True
    except Exception as e:
        logger.error(f"Error testing call_agent with variables: {e}")
        return False

async def main():
    """Main function to run the tests."""
    logger.info("Starting agent variables mock tests...")
    
    # Get available providers
    providers = mock_llm_client.get_available_providers()
    logger.info(f"Available providers: {providers}")
    
    # Test generate_agent_variables
    logger.info("\nTesting generate_agent_variables...")
    if await test_generate_agent_variables():
        logger.info("✅ generate_agent_variables test passed")
    else:
        logger.error("❌ generate_agent_variables test failed")
    
    # Test call_agent with variables
    logger.info("\nTesting call_agent with variables...")
    if await test_call_agent_with_variables():
        logger.info("✅ call_agent with variables test passed")
    else:
        logger.error("❌ call_agent with variables test failed")
    
    logger.info("\nAll tests completed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 