"""
Shared Pydantic models for distributed agents.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict


class TaskRequest(BaseModel):
    """Request to process a task."""
    task_id: str
    task: str
    research_notes: Optional[str] = ""
    analysis: Optional[str] = ""
    context: Optional[Dict] = {}


class ToolUsage(BaseModel):
    """Record of a tool being used."""
    tool_name: str
    input_preview: str
    output_preview: str
    execution_time_ms: float


class ThinkingOutput(BaseModel):
    """Parsed output from a thinking model like DeepSeek-R1."""
    thinking: str = ""
    answer: str = ""


class TaskResponse(BaseModel):
    """Response from processing a task."""
    task_id: str
    agent_type: str
    agent_id: str
    instance_id: str = ""  # For multiple instances of same agent type
    status: str
    result: str
    thinking: str = ""
    next_agent: Optional[str] = None
    output_field: str
    processing_time_ms: float = 0
    llm_tokens_used: Optional[int] = None
    hostname: str = ""
    timestamp: str = ""
    tools_used: List[ToolUsage] = []
    steps_performed: List[str] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agent_type: str
    agent_id: str
    instance_id: str = ""
    hostname: str = ""
    uptime_seconds: float = 0
    tasks_processed: int = 0
    llm_model: str = ""
    available: bool = True


class AgentRegistration(BaseModel):
    """Agent registration for discovery."""
    agent_type: str
    agent_id: str
    instance_id: str
    url: str
    hostname: str
    port: int
    model: str
    status: str = "available"
    tasks_processed: int = 0
    registered_at: str = ""
