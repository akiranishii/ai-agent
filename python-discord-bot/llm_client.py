import os
import logging
from typing import List, Dict, Optional, Union
from models import LLMProvider, LLMMessage, LLMResponse
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion

# Load environment variables from .env file
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

# Dictionary of agent configurations
AGENTS = {
    "principal_investigator": {
        "name": "Principal Investigator",
        "model": "openai/gpt-4o",
        "system_prompt": """You are a Principal Investigator. 
        Your expertise is in {expertise}. 
        Your goal is to {goal}. 
        Your role is to {role}.
        Be focused and provide concise answers. Reply in a conversational tone and in paragraph form.""",
    },
    "scientific_critic": {
        "name": "Scientific Critic",
        "model": "openai/gpt-4o",
        "system_prompt": """You are a Scientific Critic. 
        Your expertise is in providing critical feedback for scientific research. 
        Your goal is to ensure that proposed research projects and implementations are rigorous, 
        detailed, feasible, and scientifically sound. 
        Your role is to provide critical feedback to identify and correct all errors and demand 
        that scientific answers that are maximally complete and detailed but 
        simple and not overly complex. Be focused and provide concise answers. Reply in a conversational tone and in paragraph form.""",
    },
    # "biologist": {
    #     "name": "Biologist",
    #     "model": "openai/gpt-4o",
    #     "system_prompt": """You are a {agent_name}. 
    #     Your expertise is in {expertise}. 
    #     Your goal is to {goal}. 
    #     Your role is to {role}.
    #     Be focused and provide concise answers. Reply in a conversational tone and in paragraph form."""
    # },
    # "computer_scientist": {
    #     "name": "Computer Scientist",
    #     "model": "openai/gpt-4o",
    #     "system_prompt": """You are a {agent_name}. 
    #     Your expertise is in {expertise}. 
    #     Your goal is to {goal}. 
    #     Your role is to {role}.
    #     Be focused and provide concise answers. Reply in a conversational tone and in paragraph form."""
    # },
    # "computational_biologist": {
    #     "name": "Computational Biologist",
    #     "model": "openai/gpt-4o",
    #     "system_prompt": """You are a {agent_name}. 
    #     Your expertise is in {expertise}. 
    #     Your goal is to {goal}. 
    #     Your role is to {role}.
    #     Be focused and provide concise answers. Reply in a conversational tone and in paragraph form."""
    # },
    "summary_agent": {
        "name": "Summary Agent",
        "model": "openai/gpt-4o",
        "system_prompt": """You are the Summary Agent; 
        your job is to read the entire conversation and produce a final summary.
        Also provide an answer to the user's question.

        Format as follows:
        Summary: <summary>
        Answer: <answer>
        """
    },
    # Generic scientist template for dynamically created scientists
    "scientist": {
        "name": "Scientist",
        "model": "openai/gpt-4o",
        "system_prompt": """You are a {agent_name}. 
        Your expertise is in {expertise}. 
        Your goal is to {goal}. 
        Your role is to {role}.
        Be focused and provide concise answers. Reply in a conversational tone and in paragraph form."""
    }
}

# Model mapping between LLMProvider and litellm model strings
MODEL_MAPPING = {
    LLMProvider.OPENAI: {
        "default": "openai/gpt-4o",  # Default is gpt-4o
        "gpt-4o": "openai/gpt-4o",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    },
    LLMProvider.ANTHROPIC: {
        "default": "anthropic/claude-3-opus-20240229",
        "claude-3-opus-20240229": "anthropic/claude-3-opus-20240229",
        "claude-3-5-sonnet-20240620": "anthropic/claude-3-5-sonnet-20240620",
        # Cross-provider mapping to allow using gpt-4o with Anthropic provider
        "gpt-4o": "openai/gpt-4o"
    },
    LLMProvider.MISTRAL: {
        "default": "mistral/mistral-small-latest",
        "mistral-tiny": "mistral/mistral-tiny",
        "mistral-small-latest": "mistral/mistral-small-latest",
        # Cross-provider mapping to allow using gpt-4o with Mistral provider
        "gpt-4o": "openai/gpt-4o"
    }
}

class LLMClient:
    """Simplified LLM client that uses litellm.completion."""
    
    def __init__(self):
        """Initialize the LLM client."""
        # Log API key availability
        logger.info(f"Loaded environment variables from {dotenv_path}")
        logger.info(f"OpenAI API Key available: {bool(os.getenv('OPENAI_API_KEY'))}")
        logger.info(f"Anthropic API Key available: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
        logger.info(f"Mistral API Key available: {bool(os.getenv('MISTRAL_API_KEY'))}")
        
        # Track which providers are available
        self.providers = {
            LLMProvider.OPENAI: bool(os.getenv("OPENAI_API_KEY")),
            LLMProvider.ANTHROPIC: bool(os.getenv("ANTHROPIC_API_KEY")),
            LLMProvider.MISTRAL: bool(os.getenv("MISTRAL_API_KEY")),
        }
    
    async def generate_response(
        self,
        provider: LLMProvider,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate a response from the specified LLM provider given a list of messages.
        
        Args:
            provider: The LLM provider to use
            messages: List of messages in the conversation
            model: Model to use (defaults to gpt-4o if not specified)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in the response
            
        Returns:
            LLMResponse object with content and usage information
        """
        # Check if provider is available
        if not self.providers.get(provider):
            available_providers = [p for p, available in self.providers.items() if available]
            if not available_providers:
                raise ValueError("No LLM providers are available. Please set at least one API key.")
            
            logger.warning(f"{provider} is not available. Falling back to {available_providers[0]}")
            provider = available_providers[0]
        
        # Get the full model name with provider prefix
        provider_models = MODEL_MAPPING.get(provider, {})
        # Default to gpt-4o if no specific model is provided
        model_key = model or "gpt-4o"
        full_model = provider_models.get(model_key, provider_models.get("default"))
        
        if not full_model:
            # If provider doesn't have gpt-4o or the specified model, use whatever default is available
            full_model = provider_models.get("default")
            if not full_model:
                raise ValueError(f"Unknown model for provider {provider}")
        
        # Convert our message format to litellm format
        litellm_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            # Call litellm completion
            response = completion(
                model=full_model,
                messages=litellm_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract completion content
            content = response.choices[0].message.content.strip()
            
            # Construct usage dict (simplified)
            usage = {
                "prompt_tokens": getattr(response, "usage", {}).get("prompt_tokens", 0),
                "completion_tokens": getattr(response, "usage", {}).get("completion_tokens", 0),
                "total_tokens": getattr(response, "usage", {}).get("total_tokens", 0),
            }
            
            return LLMResponse(
                content=content,
                provider=provider,
                model=model or "default",
                usage=usage
            )
            
        except Exception as e:
            logger.error(f"Error generating response with {provider} ({full_model}): {str(e)}")
            raise
    
    def get_available_providers(self):
        """Get a dictionary of available providers."""
        return {provider: available for provider, available in self.providers.items() if available}

    async def call_agent(
        self, 
        agent_key: str, 
        conversation_history: str,
        expertise: Optional[str] = None,
        goal: Optional[str] = None,
        agent_role: Optional[str] = None,
        agent_name: Optional[str] = None,
        model: Optional[str] = None  # Add model parameter
    ) -> str:
        """
        Calls the specified agent with the conversation so far.
        The agent's system prompt is used, plus the `conversation_history` is
        appended as the user input. Returns the agent's text response.
        
        Args:
            agent_key: Key of the agent in the AGENTS dictionary
            conversation_history: The conversation to respond to
            expertise: The agent's area of expertise (optional)
            goal: The agent's goal (optional)
            agent_role: The specific role description (optional)
            agent_name: The specific name for scientist agents (optional)
            model: Specific model to use (defaults to gpt-4o if not specified)
            
        Returns:
            The agent's response text
        """
        if agent_key not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_key}")
            
        agent_config = AGENTS[agent_key]
        agent_system_prompt_template = agent_config["system_prompt"]
        
        # Use specified model or agent's default, with gpt-4o as fallback
        agent_model = model or agent_config.get("model", "openai/gpt-4o")
        
        # Fill in the template variables if needed
        if "{" in agent_system_prompt_template:
            # Defaults based on agent type
            defaults = {
                "expertise": "applying artificial intelligence to biomedical research",
                "goal": "perform research that maximizes scientific impact",
                "role": "lead a team of experts to solve important problems",
                "agent_name": agent_config.get("name", "Scientist")
            }
            
            # Use provided values if available
            format_vars = {
                "expertise": expertise or defaults["expertise"],
                "goal": goal or defaults["goal"],
                "role": agent_role or defaults["role"],
                "agent_name": agent_name or defaults["agent_name"]
            }
            
            agent_system_prompt = agent_system_prompt_template.format(**format_vars)
        else:
            # No formatting needed
            agent_system_prompt = agent_system_prompt_template
        
        # Create messages
        messages = [
            LLMMessage(role="system", content=agent_system_prompt),
            LLMMessage(role="user", content=conversation_history),
        ]
        
        # Determine provider from model string
        provider = LLMProvider.OPENAI  # Default
        if "anthropic" in agent_model.lower():
            provider = LLMProvider.ANTHROPIC
        elif "mistral" in agent_model.lower():
            provider = LLMProvider.MISTRAL
            
        model_name = agent_model.split("/")[-1] if "/" in agent_model else agent_model
        
        # Use gpt-4o as default model if no specific model is provided
        if model_name == "default":
            model_name = "gpt-4o"
        
        # Generate response
        response = await self.generate_response(
            provider=provider,
            messages=messages,
            model=model_name,
            temperature=1,
            max_tokens=500,
        )
        
        return response.content

    async def generate_agent_variables(self, topic: str, agent_type: str, additional_context: str = "") -> Dict[str, str]:
        """
        Generate agent variables (expertise, goal, role) based on a topic.
        
        Args:
            topic: The research topic or question
            agent_type: The type of agent ("principal_investigator" or "scientist")
            additional_context: Optional additional context to guide the generation (e.g., to ensure diversity)
            
        Returns:
            Dictionary with keys: expertise, goal, role, and for scientists: agent_name
        """
        if agent_type not in ["principal_investigator", "scientist"]:
            raise ValueError(f"Unsupported agent type: {agent_type}")
            
        # Determine what to generate based on agent type
        if agent_type == "principal_investigator":
            system_prompt = """You are an AI assistant helping to create research agent prompts.
            Your task is to generate detailed variables for a Principal Investigator AI agent.
            
            IMPORTANT: You must format your response as a valid JSON object with ONLY these keys:
            {
              "expertise": "...",
              "goal": "...",
              "role": "..."
            }
            
            Make each field 1-2 sentences, specific to the topic, and suitable for scientific research.
            DO NOT include any explanation or text outside the JSON object.
            """
            
            user_prompt = f"""Generate Principal Investigator variables for the topic: "{topic}"
            
            ONLY return a valid JSON object - no markdown, no explanation, no extra text."""
            
        else:  # scientist
            system_prompt = """You are an AI assistant helping to create research agent prompts.
            Your task is to generate detailed variables for a Scientist AI agent.
            
            IMPORTANT: You must format your response as a valid JSON object with ONLY these keys:
            {
              "agent_name": "...",
              "expertise": "...",
              "goal": "...",
              "role": "..."
            }
            
            For agent_name, use a specific scientific discipline (e.g., "Molecular Biologist", "Computer Vision Specialist").
            Make each field 1-2 sentences, specific to the topic, and suitable for scientific research.
            DO NOT include any explanation or text outside the JSON object.
            """
            
            user_prompt = f"""Generate Scientist variables for the topic: "{topic}"
            
            ONLY return a valid JSON object - no markdown, no explanation, no extra text."""
        
        # Add additional context if provided
        if additional_context:
            user_prompt += f"\n\nAdditional context:\n{additional_context}"
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        # Use GPT-4o to generate the variables
        try:
            response = await self.generate_response(
                provider=LLMProvider.OPENAI,
                messages=messages,
                model="gpt-4o",
                temperature=0.7
            )
            
            # Parse the JSON response with improved error handling
            import json
            import re
            
            content = response.content.strip()
            
            # Find JSON-like content if wrapped in other text
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            # Try to parse the JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to clean up common issues
                # Remove markdown code block syntax
                content = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', content, flags=re.DOTALL)
                # Try again
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response after cleanup: {e}")
                    logger.error(f"Response content: {response.content}")
                    raise
            
            # Validate the expected keys
            expected_keys = ["expertise", "goal", "role"]
            if agent_type == "scientist":
                expected_keys.append("agent_name")
                
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                logger.error(f"Missing keys in generated agent variables: {missing_keys}")
                if agent_type == "principal_investigator":
                    result.update({
                        "expertise": "applying artificial intelligence to biomedical research" if "expertise" not in result else result["expertise"],
                        "goal": "perform research that maximizes scientific impact" if "goal" not in result else result["goal"],
                        "role": "lead a team of experts to solve important problems" if "role" not in result else result["role"]
                    })
                else:  # scientist
                    result.update({
                        "agent_name": "Domain Scientist" if "agent_name" not in result else result["agent_name"],
                        "expertise": f"scientific research related to {topic}" if "expertise" not in result else result["expertise"],
                        "goal": "contribute domain expertise to the research project" if "goal" not in result else result["goal"],
                        "role": "provide specialized insights and collaborate with the team" if "role" not in result else result["role"]
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating agent variables: {e}")
            # Return default values if generation fails
            if agent_type == "principal_investigator":
                return {
                    "expertise": "applying artificial intelligence to biomedical research",
                    "goal": "perform research that maximizes scientific impact",
                    "role": "lead a team of experts to solve important problems"
                }
            else:  # scientist
                return {
                    "agent_name": "Domain Scientist",
                    "expertise": f"scientific research related to {topic}",
                    "goal": "contribute domain expertise to the research project",
                    "role": "provide specialized insights and collaborate with the team"
                }

# Create a singleton instance
llm_client = LLMClient() 