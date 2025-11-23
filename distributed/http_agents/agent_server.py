"""
================================================================================
HTTP-Based Distributed Agent Server
================================================================================

Each agent runs as an independent HTTP service that can be deployed anywhere.
Agents communicate via REST APIs.

Run on different machines:
    VM1: python agent_server.py --type researcher --port 8001
    VM2: python agent_server.py --type analyst --port 8002
    VM3: python agent_server.py --type writer --port 8003

Then use the orchestrator to coordinate them.
"""

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import asyncio
import logging
from datetime import datetime
import time
import socket

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import re


# =============================================================================
# Agent Tools
# =============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information on a topic."""
    # Simulated knowledge base
    knowledge = {
        "renewable energy": "Renewable energy comes from natural sources like solar, wind, hydro, and geothermal.",
        "solar": "Solar energy harnesses sunlight using photovoltaic cells or solar thermal systems.",
        "wind": "Wind energy uses turbines to convert kinetic energy from wind into electricity.",
        "ai": "Artificial Intelligence enables machines to learn, reason, and make decisions.",
        "machine learning": "Machine learning is a subset of AI that learns patterns from data.",
        "climate": "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "healthcare": "Healthcare AI applications include diagnosis, drug discovery, and patient monitoring.",
        "quantum": "Quantum computing uses quantum mechanics principles for computation.",
    }
    results = []
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            results.append(f"[{key.upper()}]: {value}")
    return "\n".join(results) if results else f"No specific knowledge found for: {query}"


@tool
def extract_keywords(text: str) -> str:
    """Extract key topics and keywords from text."""
    # Simple keyword extraction (simulated)
    common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                   'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                   'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
                   'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
                   'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                   'than', 'too', 'very', 'just', 'to', 'of', 'in', 'for', 'on', 'with'}

    words = text.lower().split()
    keywords = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in common_words and len(w) > 3]
    unique_keywords = list(dict.fromkeys(keywords))[:10]
    return f"Keywords: {', '.join(unique_keywords)}"


@tool
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment and tone of the provided text."""
    positive_words = ['good', 'great', 'excellent', 'positive', 'benefit', 'advantage',
                     'improve', 'success', 'effective', 'efficient', 'innovative']
    negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'risk',
                     'challenge', 'difficult', 'concern', 'failure', 'limitation']

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        sentiment = "POSITIVE"
        confidence = min(95, 60 + (pos_count - neg_count) * 5)
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
        confidence = min(95, 60 + (neg_count - pos_count) * 5)
    else:
        sentiment = "NEUTRAL"
        confidence = 70

    return f"Sentiment: {sentiment} (confidence: {confidence}%)"


@tool
def summarize_points(text: str) -> str:
    """Extract and summarize key points from text."""
    sentences = text.replace('\n', ' ').split('.')
    key_points = [s.strip() for s in sentences if len(s.strip()) > 30][:5]
    if key_points:
        return "Key Points:\n" + "\n".join(f"• {p}" for p in key_points)
    return "No key points extracted."


@tool
def format_output(content: str, format_type: str = "paragraph") -> str:
    """Format content into a specific structure (paragraph, bullets, numbered)."""
    if format_type == "bullets":
        lines = content.split('\n')
        return "\n".join(f"• {line.strip()}" for line in lines if line.strip())
    elif format_type == "numbered":
        lines = content.split('\n')
        return "\n".join(f"{i+1}. {line.strip()}" for i, line in enumerate(lines) if line.strip())
    return content


# Tool registry for each agent type
AGENT_TOOLS = {
    "researcher": [search_knowledge_base, extract_keywords],
    "analyst": [analyze_sentiment, summarize_points, extract_keywords],
    "writer": [format_output, summarize_points]
}


# =============================================================================
# Logging Configuration
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'
    }

    AGENT_COLORS = {
        'researcher': '\033[96m',  # Cyan
        'analyst': '\033[93m',     # Yellow
        'writer': '\033[95m',      # Magenta
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Format the message
        formatted = f"{log_color}[{timestamp}] [{record.levelname}]{reset} {record.getMessage()}"
        return formatted


def setup_logger(agent_type: str) -> logging.Logger:
    """Setup a colored logger for the agent."""
    logger = logging.getLogger(f"agent.{agent_type}")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# Request/Response Models
# =============================================================================

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
    thinking: str = ""  # The reasoning/thinking process
    answer: str = ""    # The final answer


class TaskResponse(BaseModel):
    """Response from processing a task."""
    task_id: str
    agent_type: str
    agent_id: str
    status: str
    result: str
    thinking: str = ""  # Reasoning process from thinking models
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
    hostname: str = ""
    uptime_seconds: float = 0
    tasks_processed: int = 0
    llm_model: str = ""


# =============================================================================
# Agent Server
# =============================================================================

class AgentServer:
    """HTTP server for a single agent."""

    def __init__(
        self,
        agent_type: str,
        port: int = 8000,
        llm_host: str = "localhost",
        llm_port: int = 11434,
        model_name: str = "llama3.2"
    ):
        self.agent_type = agent_type
        self.port = port
        self.agent_id = f"{agent_type}-{port}"
        self.model_name = model_name
        self.hostname = socket.gethostname()
        self.start_time = time.time()
        self.tasks_processed = 0

        # Setup logger
        self.logger = setup_logger(agent_type)

        # Create FastAPI app
        self.app = FastAPI(
            title=f"{agent_type.capitalize()} Agent",
            description=f"Distributed {agent_type} agent service"
        )

        # Create LLM with optimized settings for faster inference
        self.llm = ChatOllama(
            model=model_name,
            base_url=f"http://{llm_host}:{llm_port}",
            temperature=0.7,
            num_ctx=4096,      # Limit context window for faster processing
            num_predict=1024,  # Limit max output tokens
            num_gpu=99,        # Ensure all layers on GPU
        )

        # Agent configuration
        self.config = self._get_agent_config()

        # Register routes
        self._register_routes()

        # Log startup
        self._log_startup(llm_host, llm_port)

    def _log_startup(self, llm_host: str, llm_port: int):
        """Log startup information."""
        self.logger.info("=" * 60)
        self.logger.info(f"🤖 {self.agent_type.upper()} AGENT STARTING")
        self.logger.info("=" * 60)
        self.logger.info(f"   Agent ID:    {self.agent_id}")
        self.logger.info(f"   Hostname:    {self.hostname}")
        self.logger.info(f"   Port:        {self.port}")
        self.logger.info(f"   LLM:         {llm_host}:{llm_port}/{self.model_name}")
        self.logger.info(f"   Next Agent:  {self.config.get('next_agent', 'None (final)')}")
        self.logger.info("=" * 60)

    def _get_agent_config(self) -> dict:
        """Get configuration based on agent type."""
        configs = {
            "researcher": {
                "system_prompt": """You are a Research Agent. Your role is to:
1. Understand the topic given to you
2. Gather comprehensive information and facts
3. Provide detailed research notes

Be thorough and factual.""",
                "next_agent": "analyst",
                "output_field": "research_notes"
            },
            "analyst": {
                "system_prompt": """You are an Analyst Agent. Your role is to:
1. Review the research notes provided
2. Identify key insights and patterns
3. Provide structured analysis

Focus on extracting actionable insights.""",
                "next_agent": "writer",
                "output_field": "analysis"
            },
            "writer": {
                "system_prompt": """You are a Writer Agent. Your role is to:
1. Review all research and analysis
2. Compose a clear, well-structured final response
3. Ensure the response addresses the original task

Be concise but comprehensive.""",
                "next_agent": None,
                "output_field": "final_response"
            }
        }
        return configs.get(self.agent_type, {})

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            uptime = time.time() - self.start_time
            self.logger.debug(f"Health check - uptime: {uptime:.1f}s, tasks: {self.tasks_processed}")
            return HealthResponse(
                status="healthy",
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                hostname=self.hostname,
                uptime_seconds=uptime,
                tasks_processed=self.tasks_processed,
                llm_model=self.model_name
            )

        @self.app.post("/process", response_model=TaskResponse)
        async def process_task(request: TaskRequest):
            """Process a task and return result."""
            self.logger.info("-" * 50)
            self.logger.info(f"📥 RECEIVED TASK: {request.task_id}")
            self.logger.info(f"   Task: {request.task[:80]}...")

            try:
                result = await self._process(request)
                self.tasks_processed += 1
                self.logger.info(f"✅ TASK COMPLETED: {request.task_id}")
                self.logger.info(f"   Processing time: {result.processing_time_ms:.0f}ms")
                self.logger.info(f"   Output length: {len(result.result)} chars")
                self.logger.info(f"   Next agent: {result.next_agent or 'None (final)'}")
                return result
            except Exception as e:
                self.logger.error(f"❌ TASK FAILED: {request.task_id}")
                self.logger.error(f"   Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/info")
        async def info():
            """Get agent information."""
            return {
                "agent_type": self.agent_type,
                "agent_id": self.agent_id,
                "hostname": self.hostname,
                "uptime_seconds": time.time() - self.start_time,
                "tasks_processed": self.tasks_processed,
                "config": {
                    "next_agent": self.config.get("next_agent"),
                    "output_field": self.config.get("output_field")
                }
            }

        @self.app.get("/stats")
        async def stats():
            """Get agent statistics."""
            uptime = time.time() - self.start_time
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "hostname": self.hostname,
                "uptime_seconds": uptime,
                "uptime_formatted": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
                "tasks_processed": self.tasks_processed,
                "tasks_per_minute": (self.tasks_processed / uptime * 60) if uptime > 0 else 0,
                "llm_model": self.model_name
            }

    async def _process(self, request: TaskRequest) -> TaskResponse:
        """Process a task request with tool usage and detailed logging."""
        config = self.config
        start_time = time.time()
        tools_used = []
        steps_performed = []

        # Log input details
        self.logger.info(f"📋 Processing as {self.agent_type.upper()}...")
        steps_performed.append(f"Started processing as {self.agent_type}")

        if self.agent_type == "analyst" and request.research_notes:
            self.logger.info(f"   📚 Received research notes: {len(request.research_notes)} chars")
        if self.agent_type == "writer":
            if request.research_notes:
                self.logger.info(f"   📚 Received research notes: {len(request.research_notes)} chars")
            if request.analysis:
                self.logger.info(f"   🔍 Received analysis: {len(request.analysis)} chars")

        # Get agent's tools
        agent_tools = AGENT_TOOLS.get(self.agent_type, [])
        self.logger.info(f"🛠️  Available tools: {[t.name for t in agent_tools]}")
        steps_performed.append(f"Loaded {len(agent_tools)} tools: {[t.name for t in agent_tools]}")

        # Execute tools based on agent type
        tool_results = {}

        if self.agent_type == "researcher":
            # Step 1: Search knowledge base
            self.logger.info(f"")
            self.logger.info(f"   ┌─ STEP 1: Searching Knowledge Base")
            self.logger.info(f"   │  🔧 Tool: search_knowledge_base")
            self.logger.info(f"   │  📥 Input: \"{request.task[:50]}...\"")

            tool_start = time.time()
            kb_result = search_knowledge_base.invoke({"query": request.task})
            tool_time = (time.time() - tool_start) * 1000

            self.logger.info(f"   │  📤 Output: \"{kb_result[:80]}...\"")
            self.logger.info(f"   │  ⏱️  Time: {tool_time:.0f}ms")
            self.logger.info(f"   └─ ✓ Complete")

            tools_used.append(ToolUsage(
                tool_name="search_knowledge_base",
                input_preview=request.task[:100],
                output_preview=kb_result[:200],
                execution_time_ms=tool_time
            ))
            steps_performed.append(f"Searched knowledge base for: {request.task[:50]}")
            tool_results["knowledge"] = kb_result

            # Step 2: Extract keywords
            self.logger.info(f"")
            self.logger.info(f"   ┌─ STEP 2: Extracting Keywords")
            self.logger.info(f"   │  🔧 Tool: extract_keywords")

            tool_start = time.time()
            keywords_result = extract_keywords.invoke({"text": request.task})
            tool_time = (time.time() - tool_start) * 1000

            self.logger.info(f"   │  📤 Output: \"{keywords_result}\"")
            self.logger.info(f"   │  ⏱️  Time: {tool_time:.0f}ms")
            self.logger.info(f"   └─ ✓ Complete")

            tools_used.append(ToolUsage(
                tool_name="extract_keywords",
                input_preview=request.task[:100],
                output_preview=keywords_result,
                execution_time_ms=tool_time
            ))
            steps_performed.append(f"Extracted keywords: {keywords_result}")
            tool_results["keywords"] = keywords_result

        elif self.agent_type == "analyst":
            # Step 1: Analyze sentiment
            self.logger.info(f"")
            self.logger.info(f"   ┌─ STEP 1: Analyzing Sentiment")
            self.logger.info(f"   │  🔧 Tool: analyze_sentiment")
            self.logger.info(f"   │  📥 Input: Research notes ({len(request.research_notes)} chars)")

            tool_start = time.time()
            sentiment_result = analyze_sentiment.invoke({"text": request.research_notes})
            tool_time = (time.time() - tool_start) * 1000

            self.logger.info(f"   │  📤 Output: \"{sentiment_result}\"")
            self.logger.info(f"   │  ⏱️  Time: {tool_time:.0f}ms")
            self.logger.info(f"   └─ ✓ Complete")

            tools_used.append(ToolUsage(
                tool_name="analyze_sentiment",
                input_preview=f"Research notes ({len(request.research_notes)} chars)",
                output_preview=sentiment_result,
                execution_time_ms=tool_time
            ))
            steps_performed.append(f"Analyzed sentiment: {sentiment_result}")
            tool_results["sentiment"] = sentiment_result

            # Step 2: Summarize points
            self.logger.info(f"")
            self.logger.info(f"   ┌─ STEP 2: Summarizing Key Points")
            self.logger.info(f"   │  🔧 Tool: summarize_points")

            tool_start = time.time()
            summary_result = summarize_points.invoke({"text": request.research_notes})
            tool_time = (time.time() - tool_start) * 1000

            self.logger.info(f"   │  📤 Output: \"{summary_result[:80]}...\"")
            self.logger.info(f"   │  ⏱️  Time: {tool_time:.0f}ms")
            self.logger.info(f"   └─ ✓ Complete")

            tools_used.append(ToolUsage(
                tool_name="summarize_points",
                input_preview=f"Research notes ({len(request.research_notes)} chars)",
                output_preview=summary_result[:200],
                execution_time_ms=tool_time
            ))
            steps_performed.append(f"Summarized key points from research")
            tool_results["summary"] = summary_result

            # Step 3: Extract keywords
            self.logger.info(f"")
            self.logger.info(f"   ┌─ STEP 3: Extracting Keywords")
            self.logger.info(f"   │  🔧 Tool: extract_keywords")

            tool_start = time.time()
            keywords_result = extract_keywords.invoke({"text": request.research_notes})
            tool_time = (time.time() - tool_start) * 1000

            self.logger.info(f"   │  📤 Output: \"{keywords_result}\"")
            self.logger.info(f"   │  ⏱️  Time: {tool_time:.0f}ms")
            self.logger.info(f"   └─ ✓ Complete")

            tools_used.append(ToolUsage(
                tool_name="extract_keywords",
                input_preview=f"Research notes ({len(request.research_notes)} chars)",
                output_preview=keywords_result,
                execution_time_ms=tool_time
            ))
            steps_performed.append(f"Extracted keywords: {keywords_result}")
            tool_results["keywords"] = keywords_result

        elif self.agent_type == "writer":
            # Step 1: Summarize all inputs
            self.logger.info(f"")
            self.logger.info(f"   ┌─ STEP 1: Summarizing All Inputs")
            self.logger.info(f"   │  🔧 Tool: summarize_points")

            combined_input = f"{request.research_notes}\n{request.analysis}"
            tool_start = time.time()
            summary_result = summarize_points.invoke({"text": combined_input})
            tool_time = (time.time() - tool_start) * 1000

            self.logger.info(f"   │  📤 Output: \"{summary_result[:80]}...\"")
            self.logger.info(f"   │  ⏱️  Time: {tool_time:.0f}ms")
            self.logger.info(f"   └─ ✓ Complete")

            tools_used.append(ToolUsage(
                tool_name="summarize_points",
                input_preview=f"Combined inputs ({len(combined_input)} chars)",
                output_preview=summary_result[:200],
                execution_time_ms=tool_time
            ))
            steps_performed.append(f"Summarized all inputs for final composition")
            tool_results["summary"] = summary_result

        # Build prompt with tool results
        self.logger.info(f"")
        self.logger.info(f"   ┌─ STEP {len(tools_used)+1}: Generating Response with LLM")
        self.logger.info(f"   │  🤖 Model: {self.model_name}")
        steps_performed.append(f"Calling LLM ({self.model_name}) for final generation")

        if self.agent_type == "researcher":
            prompt = f"""Research this topic thoroughly:

TOPIC: {request.task}

KNOWLEDGE BASE RESULTS:
{tool_results.get('knowledge', 'No results')}

IDENTIFIED KEYWORDS:
{tool_results.get('keywords', 'None')}

Based on the above information and your knowledge, provide comprehensive research notes."""

        elif self.agent_type == "analyst":
            prompt = f"""Analyze the following research:

ORIGINAL TASK: {request.task}

RESEARCH NOTES:
{request.research_notes}

TOOL ANALYSIS RESULTS:
- {tool_results.get('sentiment', 'No sentiment analysis')}
- {tool_results.get('keywords', 'No keywords')}

KEY POINTS IDENTIFIED:
{tool_results.get('summary', 'No summary')}

Provide a structured analysis with key insights and recommendations."""

        elif self.agent_type == "writer":
            prompt = f"""Create a final response based on:

ORIGINAL TASK: {request.task}

RESEARCH NOTES:
{request.research_notes}

ANALYSIS:
{request.analysis}

SUMMARY OF KEY POINTS:
{tool_results.get('summary', 'No summary')}

Write a comprehensive, well-structured final response that addresses the original task."""

        else:
            prompt = f"Process this task: {request.task}"

        # Call LLM
        llm_start = time.time()
        self.logger.info(f"   │  📥 Prompt length: {len(prompt)} chars")

        loop = asyncio.get_event_loop()
        llm_output = await loop.run_in_executor(
            None,
            lambda: self._call_llm(prompt)
        )

        llm_time = (time.time() - llm_start) * 1000
        total_time = (time.time() - start_time) * 1000

        # Log thinking process if present (for models like DeepSeek-R1)
        if llm_output.thinking:
            self.logger.info(f"   │")
            self.logger.info(f"   │  💭 THINKING PROCESS:")
            # Show first few lines of thinking
            thinking_lines = llm_output.thinking.split('\n')[:5]
            for line in thinking_lines:
                if line.strip():
                    self.logger.info(f"   │     {line[:80]}{'...' if len(line) > 80 else ''}")
            if len(llm_output.thinking.split('\n')) > 5:
                self.logger.info(f"   │     ... ({len(llm_output.thinking)} chars total)")
            self.logger.info(f"   │")
            steps_performed.append(f"Model reasoning: {len(llm_output.thinking)} chars of thinking")

        self.logger.info(f"   │  📤 Answer length: {len(llm_output.answer)} chars")
        self.logger.info(f"   │  ⏱️  LLM time: {llm_time:.0f}ms")
        self.logger.info(f"   └─ ✓ Complete")
        steps_performed.append(f"Generated response ({len(llm_output.answer)} chars) in {llm_time:.0f}ms")

        # Final summary
        self.logger.info(f"")
        self.logger.info(f"📊 PROCESSING SUMMARY:")
        self.logger.info(f"   • Tools executed: {len(tools_used)}")
        for tu in tools_used:
            self.logger.info(f"     - {tu.tool_name}: {tu.execution_time_ms:.0f}ms")
        self.logger.info(f"   • LLM generation: {llm_time:.0f}ms")
        if llm_output.thinking:
            self.logger.info(f"   • Thinking: {len(llm_output.thinking)} chars")
        self.logger.info(f"   • Total time: {total_time:.0f}ms")

        # Estimate tokens (rough approximation)
        full_response = llm_output.thinking + llm_output.answer
        estimated_tokens = len(prompt.split()) + len(full_response.split())

        return TaskResponse(
            task_id=request.task_id,
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            status="completed",
            result=llm_output.answer,
            thinking=llm_output.thinking,
            next_agent=config.get("next_agent"),
            output_field=config.get("output_field", "result"),
            processing_time_ms=total_time,
            llm_tokens_used=estimated_tokens,
            hostname=self.hostname,
            timestamp=datetime.now().isoformat(),
            tools_used=tools_used,
            steps_performed=steps_performed
        )

    def _parse_thinking_response(self, content: str) -> ThinkingOutput:
        """
        Parse response from thinking models like DeepSeek-R1.

        DeepSeek-R1 outputs in format:
        <think>
        ... reasoning process ...
        </think>

        Final answer here
        """
        # Try to extract thinking tags
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, content, re.DOTALL)

        if think_match:
            thinking = think_match.group(1).strip()
            # Get everything after </think> as the answer
            answer = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
            return ThinkingOutput(thinking=thinking, answer=answer)
        else:
            # No thinking tags - treat entire response as answer
            return ThinkingOutput(thinking="", answer=content)

    def _call_llm(self, prompt: str) -> ThinkingOutput:
        """Call the LLM synchronously and parse thinking output."""
        messages = [
            SystemMessage(content=self.config.get("system_prompt", "")),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages)
        return self._parse_thinking_response(response.content)

    def run(self):
        """Run the server."""
        print(f"🚀 Starting {self.agent_type} agent on port {self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HTTP Agent Server")
    parser.add_argument(
        "--type", "-t",
        required=True,
        choices=["researcher", "analyst", "writer"],
        help="Type of agent"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run on"
    )
    parser.add_argument(
        "--llm-host",
        default="localhost",
        help="Ollama host"
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=11434,
        help="Ollama port"
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="LLM model (default: llama3.2, alternatives: deepseek-r1:70b, etc.)"
    )

    args = parser.parse_args()

    server = AgentServer(
        agent_type=args.type,
        port=args.port,
        llm_host=args.llm_host,
        llm_port=args.llm_port,
        model_name=args.model
    )
    server.run()


if __name__ == "__main__":
    main()
