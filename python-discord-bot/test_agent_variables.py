#!/usr/bin/env python3
"""
Test script for agent variables generation and usage.
This script tests the generate_agent_variables method and the call_agent method with the new variables.
"""

import asyncio
import logging
from llm_client import llm_client
from models import LLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_generate_agent_variables():
    """Test the generate_agent_variables method."""
    try:
        # Test generating variables for Principal Investigator
        logger.info("Generating variables for Principal Investigator...")
        pi_variables = await llm_client.generate_agent_variables(
            topic="protein folding prediction using deep learning",
            agent_type="principal_investigator"
        )
        logger.info("Principal Investigator variables:")
        logger.info(f"  Expertise: {pi_variables.get('expertise')}")
        logger.info(f"  Goal: {pi_variables.get('goal')}")
        logger.info(f"  Role: {pi_variables.get('role')}")
        
        # Test generating variables for Scientist
        logger.info("\nGenerating variables for Scientist...")
        scientist_variables = await llm_client.generate_agent_variables(
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
        pi_response = await llm_client.call_agent(
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
        scientist_response = await llm_client.call_agent(
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
    logger.info("Starting agent variables tests...")
    
    # Get available providers
    providers = llm_client.get_available_providers()
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
    asyncio.run(main()) 