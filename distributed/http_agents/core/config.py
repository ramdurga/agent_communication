"""
Agent configuration management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_type: str
    system_prompt: str
    next_agent: Optional[str] = None
    output_field: str = "result"
    tools: List[str] = field(default_factory=list)

    @classmethod
    def get_config(cls, agent_type: str) -> "AgentConfig":
        """Get configuration for a specific agent type."""
        configs = {
            "researcher": cls(
                agent_type="researcher",
                system_prompt="""You are a Research Agent. Your role is to:
1. Understand the topic given to you
2. Gather comprehensive information and facts
3. Provide detailed research notes

Be thorough and factual. Focus on key facts and important details.""",
                next_agent="analyst",
                output_field="research_notes",
                tools=["search_knowledge_base", "extract_keywords"]
            ),
            "analyst": cls(
                agent_type="analyst",
                system_prompt="""You are an Analyst Agent. Your role is to:
1. Review the research notes provided
2. Identify key insights and patterns
3. Provide structured analysis

Focus on extracting actionable insights and identifying trends.""",
                next_agent="writer",
                output_field="analysis",
                tools=["analyze_sentiment", "summarize_points", "extract_keywords"]
            ),
            "writer": cls(
                agent_type="writer",
                system_prompt="""You are a Writer Agent. Your role is to:
1. Review all research and analysis
2. Compose a clear, well-structured final response
3. Ensure the response addresses the original task

Be concise but comprehensive. Use clear language.""",
                next_agent=None,
                output_field="final_response",
                tools=["format_output", "summarize_points"]
            )
        }
        return configs.get(agent_type, cls(agent_type=agent_type, system_prompt="Process the given task."))


@dataclass
class LLMConfig:
    """LLM configuration."""
    host: str = "localhost"
    port: int = 11434
    model: str = "llama3.2"
    temperature: float = 0.7
    num_ctx: int = 4096
    num_predict: int = 1024
    num_gpu: int = 99

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("OLLAMA_HOST", "localhost"),
            port=int(os.getenv("OLLAMA_PORT", "11434")),
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            num_ctx=int(os.getenv("LLM_NUM_CTX", "4096")),
            num_predict=int(os.getenv("LLM_NUM_PREDICT", "1024")),
            num_gpu=int(os.getenv("LLM_NUM_GPU", "99")),
        )
