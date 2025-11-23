# Distributed Multi-Agent System with HTTP Communication

A production-ready distributed multi-agent architecture using FastAPI, LangChain, and Ollama. Each agent runs as an independent HTTP service that can be deployed on separate machines, enabling horizontal scaling and fault isolation.

## Architecture Overview

```
                                DISTRIBUTED MULTI-AGENT PIPELINE
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                     │
    │                              ┌─────────────────────────────────────┐                │
    │                              │         ORCHESTRATOR                │                │
    │                              │   • Agent Registry                  │                │
    │                              │   • Load Balancing                  │                │
    │                              │   • Health Checking                 │                │
    │                              └──────────────┬──────────────────────┘                │
    │                                             │                                       │
    │              ┌──────────────────────────────┼──────────────────────────────┐        │
    │              │                              │                              │        │
    │              ▼                              ▼                              ▼        │
    │   ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐│
    │   │ RESEARCHER AGENTS   │      │  ANALYST AGENTS     │      │   WRITER AGENTS     ││
    │   │   Instance 1 :8001  │      │   Instance 1 :8002  │      │   Instance 1 :8003  ││
    │   │   Instance 2 :8011  │      │   Instance 2 :8012  │      │   Instance 2 :8013  ││
    │   │   Instance N :80X1  │      │   Instance N :80X2  │      │   Instance N :80X3  ││
    │   └─────────┬───────────┘      └─────────┬───────────┘      └─────────┬───────────┘│
    │             │                            │                            │            │
    │             └────────────────────────────┼────────────────────────────┘            │
    │                                          │                                         │
    │                                          ▼                                         │
    │                          ┌───────────────────────────────┐                         │
    │                          │        OLLAMA LLM SERVER      │                         │
    │                          │   (Shared or Per-Instance)    │                         │
    │                          │   Models: llama3.2, deepseek  │                         │
    │                          └───────────────────────────────┘                         │
    │                                                                                    │
    └────────────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
http_agents/
├── core/                       # Core framework
│   ├── __init__.py            # Module exports
│   ├── base_agent.py          # Abstract base agent class
│   ├── config.py              # Agent and LLM configuration
│   ├── models.py              # Pydantic models (request/response)
│   ├── registry.py            # Agent discovery and load balancing
│   └── tools.py               # Shared tools for all agents
│
├── agents/                     # Individual agent implementations
│   ├── __init__.py            # Module exports
│   ├── researcher.py          # Researcher agent
│   ├── analyst.py             # Analyst agent
│   └── writer.py              # Writer agent
│
├── orchestrator.py            # Multi-instance orchestrator
├── run_agents.sh              # Start multiple agent instances
├── run_demo.sh                # Quick demo script
└── README.md                  # This file
```

## Load Balancing Strategies

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    LOAD BALANCING STRATEGIES                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ROUND ROBIN (default)                                         │
    │  ├── Distributes requests evenly across instances              │
    │  └── Best for: Uniform workloads                               │
    │                                                                 │
    │  LEAST LOADED                                                  │
    │  ├── Routes to instance with fewest processed tasks            │
    │  └── Best for: Variable task complexity                        │
    │                                                                 │
    │  FASTEST                                                        │
    │  ├── Routes to instance with lowest response time              │
    │  └── Best for: Latency-sensitive workloads                     │
    │                                                                 │
    │  RANDOM                                                         │
    │  ├── Random instance selection                                  │
    │  └── Best for: Simple load distribution                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

## Agent Communication Flow

```
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │                           TASK PROCESSING PIPELINE                                │
    └──────────────────────────────────────────────────────────────────────────────────┘

         USER REQUEST
              │
              ▼
    ┌─────────────────┐
    │   ORCHESTRATOR  │
    │   • Registry    │
    │   • Load Balance│
    └────────┬────────┘
             │
             │ HTTP POST /process → select instance via strategy
             ▼
    ┌─────────────────┐     ┌─────────────────────────────────────────┐
    │   RESEARCHER    │     │  STEP 1: Research Phase                 │
    │   (Instance N)  │────▶│  • Search knowledge base                │
    │                 │     │  • Extract keywords                     │
    │   Tools:        │     │  • Generate research notes via LLM      │
    │   • KB Search   │     │                                         │
    │   • Keywords    │     │  Output: research_notes                 │
    └────────┬────────┘     └─────────────────────────────────────────┘
             │
             │ research_notes
             ▼
    ┌─────────────────┐     ┌─────────────────────────────────────────┐
    │    ANALYST      │     │  STEP 2: Analysis Phase                 │
    │   (Instance N)  │────▶│  • Analyze sentiment                    │
    │                 │     │  • Summarize key points                 │
    │   Tools:        │     │  • Extract insights via LLM             │
    │   • Sentiment   │     │                                         │
    │   • Summarize   │     │  Output: analysis                       │
    └────────┬────────┘     └─────────────────────────────────────────┘
             │
             │ research_notes + analysis
             ▼
    ┌─────────────────┐     ┌─────────────────────────────────────────┐
    │    WRITER       │     │  STEP 3: Writing Phase                  │
    │   (Instance N)  │────▶│  • Format output                        │
    │                 │     │  • Compose final response via LLM       │
    │   Tools:        │     │                                         │
    │   • Format      │     │  Output: final_response                 │
    │   • Summarize   │     │                                         │
    └────────┬────────┘     └─────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  FINAL RESULT   │
    └─────────────────┘
```

## Deployment Topology

### Single Machine with Multiple Instances

```
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                           SINGLE MACHINE                                   │
    │                                                                            │
    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                    │
    │   │ Researcher-1 │  │ Researcher-2 │  │ Researcher-3 │                    │
    │   │    :8001     │  │    :8011     │  │    :8021     │                    │
    │   └──────────────┘  └──────────────┘  └──────────────┘                    │
    │                                                                            │
    │   ┌──────────────┐  ┌──────────────┐                                      │
    │   │  Analyst-1   │  │  Analyst-2   │    ...more instances                 │
    │   │    :8002     │  │    :8012     │                                      │
    │   └──────────────┘  └──────────────┘                                      │
    │                                                                            │
    │   ┌──────────────┐                                                         │
    │   │   Writer-1   │                                                         │
    │   │    :8003     │                                                         │
    │   └──────────────┘                                                         │
    │                                                                            │
    │                          ┌──────────────┐                                  │
    │                          │    Ollama    │                                  │
    │                          │    :11434    │                                  │
    │                          └──────────────┘                                  │
    │                                                                            │
    └────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Machine (Production)

```
    ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
    │      HOST 1         │     │      HOST 2         │     │      HOST 3         │
    │   (Researchers)     │     │   (Analysts)        │     │   (Writers)         │
    │                     │     │                     │     │                     │
    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
    │ │ Researcher-1    │ │     │ │  Analyst-1      │ │     │ │   Writer-1      │ │
    │ │     :8001       │ │     │ │     :8002       │ │     │ │     :8003       │ │
    │ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
    │ │ Researcher-2    │ │     │ │  Analyst-2      │ │     │ │   Writer-2      │ │
    │ │     :8011       │ │     │ │     :8012       │ │     │ │     :8013       │ │
    │ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
    │                     │     │                     │     │                     │
    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
    │ │     Ollama      │ │     │ │     Ollama      │ │     │ │     Ollama      │ │
    │ │     :11434      │ │     │ │     :11434      │ │     │ │     :11434      │ │
    │ └─────────────────┘ │     │ └─────────────────┘ │     │ └─────────────────┘ │
    └──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘
               │                           │                           │
               └───────────────────────────┼───────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │      ORCHESTRATOR       │
                              │     (Control Plane)     │
                              │                         │
                              │  researcher-urls:       │
                              │  • host1:8001,host1:8011│
                              │  analyst-urls:          │
                              │  • host2:8002,host2:8012│
                              │  writer-urls:           │
                              │  • host3:8003,host3:8013│
                              └─────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with a model
ollama pull llama3.2
ollama serve
```

### Running the Demo

#### Option 1: Quick Demo Script

```bash
cd distributed/http_agents
chmod +x run_demo.sh
./run_demo.sh
```

With options:
```bash
./run_demo.sh --model llama3.2 --scale 2 --task "Explain quantum computing"
```

#### Option 2: Start Agents Script (for scaling)

```bash
chmod +x run_agents.sh
./run_agents.sh --scale 3 --model llama3.2
```

This starts 3 instances of each agent type.

#### Option 3: Manual Setup

**Terminal 1 - Researcher Agent:**
```bash
python -m agents.researcher --port 8001 --instance researcher-1
```

**Terminal 2 - Analyst Agent:**
```bash
python -m agents.analyst --port 8002 --instance analyst-1
```

**Terminal 3 - Writer Agent:**
```bash
python -m agents.writer --port 8003 --instance writer-1
```

**Terminal 4 - Orchestrator:**
```bash
python orchestrator.py --task "Explain the benefits of renewable energy"
```

### Multi-Instance Deployment

Start multiple instances and use load balancing:

```bash
# Start multiple researchers
python -m agents.researcher --port 8001 --instance researcher-1 &
python -m agents.researcher --port 8011 --instance researcher-2 &

# Start multiple analysts
python -m agents.analyst --port 8002 --instance analyst-1 &
python -m agents.analyst --port 8012 --instance analyst-2 &

# Start writers
python -m agents.writer --port 8003 --instance writer-1 &

# Run orchestrator with all instances
python orchestrator.py \
    --task "Explain quantum computing" \
    --researcher-urls "http://localhost:8001,http://localhost:8011" \
    --analyst-urls "http://localhost:8002,http://localhost:8012" \
    --writer-urls "http://localhost:8003" \
    --strategy least_loaded
```

## Features

### Agent Capabilities

| Agent | Role | Tools | Output |
|-------|------|-------|--------|
| **Researcher** | Gather information | `search_knowledge_base`, `extract_keywords` | Research notes |
| **Analyst** | Analyze & synthesize | `analyze_sentiment`, `summarize_points`, `extract_keywords` | Analysis report |
| **Writer** | Compose final output | `format_output`, `summarize_points` | Final response |

### Built-in Tools

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        TOOL REGISTRY                            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  search_knowledge_base(query)                                   │
    │  ├── Input:  Topic or question                                  │
    │  └── Output: Relevant knowledge snippets                        │
    │                                                                 │
    │  extract_keywords(text)                                         │
    │  ├── Input:  Any text content                                   │
    │  └── Output: List of key topics                                 │
    │                                                                 │
    │  analyze_sentiment(text)                                        │
    │  ├── Input:  Text to analyze                                    │
    │  └── Output: Sentiment (POSITIVE/NEGATIVE/NEUTRAL)              │
    │                                                                 │
    │  summarize_points(text)                                         │
    │  ├── Input:  Long text content                                  │
    │  └── Output: Bullet-point summary                               │
    │                                                                 │
    │  format_output(content, format_type)                            │
    │  ├── Input:  Content + format (paragraph/bullets/numbered)      │
    │  └── Output: Formatted text                                     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### Thinking Model Support (DeepSeek-R1)

The system supports thinking models that output their reasoning process:

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    THINKING MODEL OUTPUT                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   LLM Response:                                                 │
    │   ┌───────────────────────────────────────────────────────┐     │
    │   │ <think>                                               │     │
    │   │ Let me analyze this step by step...                   │     │
    │   │ First, I need to consider the key factors...          │     │
    │   │ </think>                                              │     │
    │   │                                                       │     │
    │   │ Here is my final answer based on my analysis...       │     │
    │   └───────────────────────────────────────────────────────┘     │
    │                           │                                     │
    │                           ▼                                     │
    │   ┌───────────────────────────────────────────────────────┐     │
    │   │ Parsed Output:                                        │     │
    │   │                                                       │     │
    │   │ thinking: "Let me analyze this step by step..."       │     │
    │   │ answer:   "Here is my final answer..."                │     │
    │   └───────────────────────────────────────────────────────┘     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

## API Reference

### Agent Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health status, uptime, tasks processed |
| `/process` | POST | Process a task |
| `/info` | GET | Agent configuration and metadata |
| `/stats` | GET | Performance statistics |

### Request Format

```json
{
  "task_id": "task-001",
  "task": "Explain renewable energy",
  "research_notes": "",
  "analysis": "",
  "context": {}
}
```

### Response Format

```json
{
  "task_id": "task-20251122-141530",
  "agent_type": "researcher",
  "agent_id": "researcher-8001",
  "instance_id": "researcher-1",
  "status": "completed",
  "result": "Research findings...",
  "thinking": "Step-by-step reasoning (for thinking models)",
  "next_agent": "analyst",
  "processing_time_ms": 2500.5,
  "llm_tokens_used": 450,
  "hostname": "vm-research-01",
  "timestamp": "2025-11-22T14:15:30.123456",
  "tools_used": [
    {
      "tool_name": "search_knowledge_base",
      "input_preview": "renewable energy",
      "output_preview": "[RENEWABLE ENERGY]: Solar, wind...",
      "execution_time_ms": 5.2
    }
  ],
  "steps_performed": [
    "Started processing",
    "Searched knowledge base",
    "Generated response (850 chars)"
  ]
}
```

## Configuration

### Agent CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--port`, `-p` | 8001-8003 | Port to run the agent on |
| `--instance`, `-i` | None | Instance identifier |
| `--llm-host` | localhost | Ollama server hostname |
| `--llm-port` | 11434 | Ollama server port |
| `--model` | llama3.2 | LLM model to use |
| `--registry` | None | Registry URL for service discovery |

### Orchestrator CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | Required | Task description |
| `--researcher-urls` | localhost:8001 | Comma-separated researcher URLs |
| `--analyst-urls` | localhost:8002 | Comma-separated analyst URLs |
| `--writer-urls` | localhost:8003 | Comma-separated writer URLs |
| `--strategy` | round_robin | Load balancing strategy |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | localhost | Ollama server host |
| `OLLAMA_PORT` | 11434 | Ollama server port |
| `OLLAMA_MODEL` | llama3.2 | Default model |
| `LLM_TEMPERATURE` | 0.7 | Model temperature |
| `LLM_NUM_CTX` | 4096 | Context window size |
| `LLM_NUM_PREDICT` | 1024 | Max output tokens |
| `LLM_NUM_GPU` | 99 | GPU layers (99 = all) |

## Performance Tuning

### GPU Optimization

```bash
# Use all GPU layers (for NVIDIA GPUs)
LLM_NUM_GPU=99 python -m agents.researcher --port 8001

# Enable Flash Attention
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

### Model Selection

| Model | Speed | Capability | VRAM |
|-------|-------|-----------|------|
| llama3.2 | Fast | Good | ~4GB |
| llama3.2:70b | Slow | Excellent | ~40GB |
| deepseek-r1 | Medium | Good + Thinking | ~8GB |
| deepseek-r1:70b | Slow | Excellent + Thinking | ~40GB |

## Extending the System

### Adding New Tools

Edit `core/tools.py`:

```python
@tool
def my_custom_tool(query: str) -> str:
    """Description of what this tool does."""
    return result

# Add to agent's tool list
AGENT_TOOLS["researcher"].append(my_custom_tool)
```

### Adding New Agent Types

1. Create agent in `agents/my_agent.py`:

```python
from core.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, port=8004, instance_id=None, llm_config=None):
        super().__init__(
            agent_type="my_agent",
            port=port,
            instance_id=instance_id,
            llm_config=llm_config
        )

    async def _process(self, request: TaskRequest) -> TaskResponse:
        # Your implementation
        pass
```

2. Add configuration in `core/config.py`:

```python
"my_agent": cls(
    agent_type="my_agent",
    system_prompt="You are a Custom Agent...",
    next_agent="writer",
    output_field="custom_output",
    tools=["my_tool"]
)
```

3. Add tools in `core/tools.py`:

```python
AGENT_TOOLS["my_agent"] = [my_custom_tool]
```

## Logging & Monitoring

The system provides color-coded logging:

```
[14:15:30.123] [INFO] Starting RESEARCHER agent on port 8001
[14:15:30.125] [INFO] LLM: localhost:11434/llama3.2

──────────────────────────────────────────────────────────────────────
[14:15:35.456] [STEP 1] RESEARCH PHASE
──────────────────────────────────────────────────────────────────────
[14:15:35.458] [RESEARCHER] Sending to researcher-1@http://localhost:8001
[14:15:35.459] [RESEARCHER]    Strategy: round_robin
[14:15:38.789] [RESEARCHER] ✅ Response received in 3331ms
[14:15:38.790] [RESEARCHER]    Instance: researcher-1
[14:15:38.790] [RESEARCHER]    Tools used: 2
[14:15:38.790] [RESEARCHER]      • search_knowledge_base: 5ms
[14:15:38.790] [RESEARCHER]      • extract_keywords: 3ms
```

## License

MIT License - See LICENSE file for details.

---

**Built with:**
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangChain](https://langchain.com/) - LLM orchestration
- [Ollama](https://ollama.ai/) - Local LLM inference
- [HTTPX](https://www.python-httpx.org/) - Async HTTP client
