#!/usr/bin/env python3
"""
Test script for the updated LLM client using litellm.
"""

import asyncio
import logging
from llm_client import llm_client, AGENTS
from models import LLMMessage, LLMProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_generate_response():
    """Test the generate_response method."""
    messages = [
        LLMMessage(role="system", content="You are a helpful AI assistant."),
        LLMMessage(role="user", content="What is the capital of France?")
    ]
    
    try:
        response = await llm_client.generate_response(
            provider=LLMProvider.OPENAI,
            messages=messages,
            temperature=0.7
        )
        logger.info(f"Response content: {response.content}")
        logger.info(f"Token usage: {response.usage}")
        return True
    except Exception as e:
        logger.error(f"Error testing generate_response: {e}")
        return False

async def test_call_agent():
    """Test the call_agent method."""
    try:
        # Test basic agent call
        agent_key = "principal_investigator"
        logger.info(f"Testing basic agent call: {agent_key}")
        response = await llm_client.call_agent(
            agent_key=agent_key,
            conversation_history="What are the latest advancements in protein folding prediction?"
        )
        logger.info(f"Basic agent response: {response[:100]}...")
        
        # Test agent with custom variables
        logger.info(f"Testing agent with custom variables: {agent_key}")
        response = await llm_client.call_agent(
            agent_key=agent_key,
            conversation_history="What are the latest advancements in protein folding prediction?",
            expertise="computational biology and protein structure prediction",
            goal="advance our understanding of protein folding mechanisms",
            agent_role="guide research into novel protein structure prediction methods"
        )
        logger.info(f"Custom agent response: {response[:100]}...")
        
        # Test scientist agent with custom name
        agent_key = "biologist"
        logger.info(f"Testing scientist agent with custom name: {agent_key}")
        response = await llm_client.call_agent(
            agent_key=agent_key,
            conversation_history="What are the latest advancements in CRISPR technology?",
            expertise="genetic engineering and CRISPR-Cas9 systems",
            goal="develop more precise gene editing techniques",
            agent_role="evaluate cutting-edge CRISPR applications",
            agent_name="Molecular Geneticist"
        )
        logger.info(f"Scientist agent response: {response[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error testing call_agent: {e}")
        return False

async def test_generate_agent_variables():
    """Test the generate_agent_variables method."""
    try:
        # Test PI variable generation
        logger.info("Testing PI variable generation")
        topic = "developing AI models for drug discovery"
        pi_vars = await llm_client.generate_agent_variables(topic, "principal_investigator")
        logger.info(f"PI variables for '{topic}':")
        for key, value in pi_vars.items():
            logger.info(f"  {key}: {value}")
        
        # Test scientist variable generation
        logger.info("Testing scientist variable generation")
        topic = "quantum computing applications in cryptography"
        sci_vars = await llm_client.generate_agent_variables(topic, "scientist")
        logger.info(f"Scientist variables for '{topic}':")
        for key, value in sci_vars.items():
            logger.info(f"  {key}: {value}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing generate_agent_variables: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("Starting LLM client tests...")
    
    # Get available providers
    providers = llm_client.get_available_providers()
    logger.info(f"Available providers: {providers}")
    
    # Test generate_response
    logger.info("Testing generate_response...")
    if await test_generate_response():
        logger.info("✅ generate_response test passed")
    else:
        logger.error("❌ generate_response test failed")
    
    # Test call_agent
    logger.info("Testing call_agent...")
    if await test_call_agent():
        logger.info("✅ call_agent test passed")
    else:
        logger.error("❌ call_agent test failed")
    
    # Test generate_agent_variables
    logger.info("Testing generate_agent_variables...")
    if await test_generate_agent_variables():
        logger.info("✅ generate_agent_variables test passed")
    else:
        logger.error("❌ generate_agent_variables test failed")
    
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 