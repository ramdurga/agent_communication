"""
Base Agent class - common functionality for all agent types.
"""
import asyncio
import logging
import re
import socket
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import AgentConfig, LLMConfig
from .models import (
    HealthResponse,
    TaskRequest,
    TaskResponse,
    ThinkingOutput,
    ToolUsage,
)
from .tools import AGENT_TOOLS


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors."""

    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return f"{log_color}[{timestamp}] [{record.levelname}]{reset} {record.getMessage()}"


def setup_logger(name: str) -> logging.Logger:
    """Setup a colored logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    return logger


class BaseAgent(ABC):
    """
    Base class for all distributed agents.

    Provides common functionality:
    - HTTP server setup
    - LLM integration
    - Tool execution
    - Logging
    - Health checks
    """

    def __init__(
        self,
        agent_type: str,
        port: int = 8000,
        instance_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        registry_url: Optional[str] = None,
    ):
        self.agent_type = agent_type
        self.port = port
        self.instance_id = instance_id or str(uuid.uuid4())[:8]
        self.agent_id = f"{agent_type}-{port}-{self.instance_id}"
        self.hostname = socket.gethostname()
        self.start_time = time.time()
        self.tasks_processed = 0
        self.registry_url = registry_url

        # Load configurations
        self.config = AgentConfig.get_config(agent_type)
        self.llm_config = llm_config or LLMConfig.from_env()

        # Setup logger
        self.logger = setup_logger(f"agent.{agent_type}.{self.instance_id}")

        # Create FastAPI app
        self.app = FastAPI(
            title=f"{agent_type.capitalize()} Agent - {self.instance_id}",
            description=f"Distributed {agent_type} agent instance"
        )

        # Create LLM
        self.llm = ChatOllama(
            model=self.llm_config.model,
            base_url=f"http://{self.llm_config.host}:{self.llm_config.port}",
            temperature=self.llm_config.temperature,
            num_ctx=self.llm_config.num_ctx,
            num_predict=self.llm_config.num_predict,
            num_gpu=self.llm_config.num_gpu,
        )

        # Get tools for this agent type
        self.tools = AGENT_TOOLS.get(agent_type, [])

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API endpoints."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            uptime = time.time() - self.start_time
            return HealthResponse(
                status="healthy",
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                instance_id=self.instance_id,
                hostname=self.hostname,
                uptime_seconds=uptime,
                tasks_processed=self.tasks_processed,
                llm_model=self.llm_config.model,
                available=True
            )

        @self.app.post("/process", response_model=TaskResponse)
        async def process_task(request: TaskRequest):
            self.logger.info("-" * 50)
            self.logger.info(f"📥 RECEIVED TASK: {request.task_id}")
            self.logger.info(f"   Instance: {self.instance_id}")
            self.logger.info(f"   Task: {request.task[:80]}...")

            try:
                result = await self._process(request)
                self.tasks_processed += 1
                self.logger.info(f"✅ TASK COMPLETED: {request.task_id}")
                self.logger.info(f"   Processing time: {result.processing_time_ms:.0f}ms")
                return result
            except Exception as e:
                self.logger.error(f"❌ TASK FAILED: {request.task_id}")
                self.logger.error(f"   Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/info")
        async def info():
            return {
                "agent_type": self.agent_type,
                "agent_id": self.agent_id,
                "instance_id": self.instance_id,
                "hostname": self.hostname,
                "port": self.port,
                "uptime_seconds": time.time() - self.start_time,
                "tasks_processed": self.tasks_processed,
                "llm_model": self.llm_config.model,
                "tools": [t.name for t in self.tools],
                "config": {
                    "next_agent": self.config.next_agent,
                    "output_field": self.config.output_field
                }
            }

    @abstractmethod
    async def _process(self, request: TaskRequest) -> TaskResponse:
        """Process a task. Must be implemented by subclasses."""
        pass

    def _execute_tools(self, request: TaskRequest) -> tuple[dict, List[ToolUsage], List[str]]:
        """Execute agent-specific tools and return results."""
        tool_results = {}
        tools_used = []
        steps_performed = []

        steps_performed.append(f"Started processing as {self.agent_type}")
        self.logger.info(f"🛠️  Available tools: {[t.name for t in self.tools]}")
        steps_performed.append(f"Loaded {len(self.tools)} tools")

        # Execute each tool based on agent type
        for tool in self.tools:
            tool_name = tool.name
            self.logger.info(f"")
            self.logger.info(f"   ┌─ Executing: {tool_name}")

            tool_start = time.time()

            try:
                # Determine input based on tool
                if tool_name == "search_knowledge_base":
                    result = tool.invoke({"query": request.task})
                elif tool_name == "extract_keywords":
                    text = request.research_notes or request.task
                    result = tool.invoke({"text": text})
                elif tool_name == "analyze_sentiment":
                    result = tool.invoke({"text": request.research_notes or request.task})
                elif tool_name == "summarize_points":
                    combined = f"{request.research_notes or ''}\n{request.analysis or ''}".strip()
                    result = tool.invoke({"text": combined or request.task})
                elif tool_name == "format_output":
                    result = tool.invoke({"content": request.task, "format_type": "paragraph"})
                else:
                    result = str(tool.invoke({"text": request.task}))

                tool_time = (time.time() - tool_start) * 1000

                self.logger.info(f"   │  📤 Output: \"{str(result)[:60]}...\"")
                self.logger.info(f"   │  ⏱️  Time: {tool_time:.0f}ms")
                self.logger.info(f"   └─ ✓ Complete")

                tool_results[tool_name] = result
                tools_used.append(ToolUsage(
                    tool_name=tool_name,
                    input_preview=request.task[:100],
                    output_preview=str(result)[:200],
                    execution_time_ms=tool_time
                ))
                steps_performed.append(f"Executed {tool_name}")

            except Exception as e:
                self.logger.error(f"   └─ ✗ Failed: {e}")

        return tool_results, tools_used, steps_performed

    def _parse_thinking_response(self, content: str) -> ThinkingOutput:
        """Parse response from thinking models like DeepSeek-R1."""
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, content, re.DOTALL)

        if think_match:
            thinking = think_match.group(1).strip()
            answer = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
            return ThinkingOutput(thinking=thinking, answer=answer)
        return ThinkingOutput(thinking="", answer=content)

    def _call_llm(self, prompt: str) -> ThinkingOutput:
        """Call the LLM synchronously."""
        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages)
        return self._parse_thinking_response(response.content)

    async def _call_llm_async(self, prompt: str) -> ThinkingOutput:
        """Call the LLM asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._call_llm(prompt))

    def _log_startup(self):
        """Log startup information."""
        self.logger.info("=" * 60)
        self.logger.info(f"🤖 {self.agent_type.upper()} AGENT STARTING")
        self.logger.info("=" * 60)
        self.logger.info(f"   Agent ID:    {self.agent_id}")
        self.logger.info(f"   Instance:    {self.instance_id}")
        self.logger.info(f"   Hostname:    {self.hostname}")
        self.logger.info(f"   Port:        {self.port}")
        self.logger.info(f"   LLM:         {self.llm_config.host}:{self.llm_config.port}/{self.llm_config.model}")
        self.logger.info(f"   Tools:       {[t.name for t in self.tools]}")
        self.logger.info(f"   Next Agent:  {self.config.next_agent or 'None (final)'}")
        self.logger.info("=" * 60)

    def run(self, host: str = "0.0.0.0"):
        """Run the agent server."""
        self._log_startup()
        uvicorn.run(self.app, host=host, port=self.port)
