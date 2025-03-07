from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"

@dataclass
class LLMMessage:
    """A message in a conversation with an LLM."""
    role: str  # "system", "user", or "assistant"
    content: str
    name: Optional[str] = None  # For multi-agent conversations

@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int]  # Token usage statistics

class ModelConfig:
    """Configuration for LLM models."""
    
    # OpenAI Models
    OPENAI_CHAT_MODEL = "gpt-4o"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Anthropic Models
    ANTHROPIC_CHAT_MODEL = "claude-3-opus-20240229"
    
    # Mistral Models
    MISTRAL_CHAT_MODEL = "mistral-large"
    
    # Default parameters
    DEFAULT_TEMPERATURE = 1
    DEFAULT_MAX_TOKENS = 2000
    
    # Role definitions
    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"
    
    # Agent roles
    PRINCIPAL_INVESTIGATOR_ROLE = "Principal Investigator"
    SCIENTIST_ROLE = "Scientist"
    CRITIC_ROLE = "Critic"
    
    # Default values for agent variables
    DEFAULT_PI_EXPERTISE = "applying artificial intelligence to biomedical research"
    DEFAULT_PI_GOAL = "perform research in your area of expertise that maximizes the scientific impact of the work"
    DEFAULT_PI_ROLE = "lead a team of experts to solve an important problem in artificial intelligence for biomedicine, make key decisions about the project direction based on team member input, and manage the project timeline and resources"
    
    DEFAULT_SCIENTIST_GOAL = "contribute your domain expertise to the research project"
    DEFAULT_SCIENTIST_ROLE = "provide specialized insights, suggest experiments, and collaborate with the team"
    
    # System prompts for different agent roles
    SYSTEM_PROMPTS = {
        PRINCIPAL_INVESTIGATOR_ROLE: """You are a Principal Investigator. Your expertise is in {expertise}. Your goal is to {goal}. Your role is to {role}. Be focused and provide concise answers. Reply in a conversational tone and in paragraph form.""",
        
        SCIENTIST_ROLE: """You are a {agent_name}. Your expertise is in {expertise}. Your goal is to {goal}. Your role is to {role}. Be focused and provide concise answers. Reply in a conversational tone and in paragraph form.""",
        
        CRITIC_ROLE: """You are a Scientific Critic. Your expertise is in providing critical feedback for scientific research. Your goal is to ensure that proposed research projects and implementations are rigorous, detailed, feasible, and scientifically sound. Your role is to provide critical feedback to identify and correct all errors and demand that scientific answers that are maximally complete and detailed but simple and not overly complex. Be focused and provide concise answers. Reply in a conversational tone and in paragraph form."""
    }
    
    @classmethod
    def get_system_prompt(cls, role: str, expertise: Optional[str] = None, goal: Optional[str] = None, 
                        agent_role: Optional[str] = None, agent_name: Optional[str] = None) -> str:
        """Get the system prompt for a given role.
        
        Args:
            role: The agent role (PI, Scientist, Critic)
            expertise: The agent's area of expertise
            goal: The agent's goal
            agent_role: The specific role description for the agent
            agent_name: The specific name for the agent (for Scientist)
            
        Returns:
            Formatted system prompt with variables filled in
        """
        # Get the base prompt template
        prompt_template = cls.SYSTEM_PROMPTS.get(role)
        if not prompt_template:
            return "You are a helpful AI assistant."
            
        # Fill in the template based on the role
        if role == cls.PRINCIPAL_INVESTIGATOR_ROLE:
            return prompt_template.format(
                expertise=expertise or cls.DEFAULT_PI_EXPERTISE,
                goal=goal or cls.DEFAULT_PI_GOAL,
                role=agent_role or cls.DEFAULT_PI_ROLE
            )
        elif role == cls.SCIENTIST_ROLE:
            return prompt_template.format(
                agent_name=agent_name or "Scientist",
                expertise=expertise or f"scientific research in {expertise or 'your field'}",
                goal=goal or cls.DEFAULT_SCIENTIST_GOAL,
                role=agent_role or cls.DEFAULT_SCIENTIST_ROLE
            )
        elif role == cls.CRITIC_ROLE:
            # Critic doesn't have variables in the current implementation
            return prompt_template
        else:
            return prompt_template 