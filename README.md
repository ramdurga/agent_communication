# Multi-Agent Communication Systems with LangGraph

A comprehensive learning resource and production-ready framework for building multi-agent AI systems using LangChain, LangGraph, and Ollama. From basic concepts to distributed systems with load balancing.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## What You'll Learn

This repository takes you from zero to production-ready multi-agent systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEARNING JOURNEY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FUNDAMENTALS                    PATTERNS                   PRODUCTION      │
│  ───────────                    ────────                   ──────────       │
│  • What is an Agent?            • Sequential Pipeline      • HTTP Agents    │
│  • LLM + Tools + Memory         • Supervisor/Orchestrator  • Load Balancing │
│  • ReAct Pattern                • Hierarchical Teams       • Multi-Instance │
│  • State Management             • Debate (Multi-Round)     • SQL Learning   │
│                                 • Reflection/Self-Critique • Auto-Correction│
│                                 • Human-in-the-Loop                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ramdurga/agent_communication.git
cd agent_communication

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama pull llama3.2
ollama serve

# Run the quickstart example
python tutorial/00_quickstart.py
```

## Project Structure

```
agent_communication/
├── tutorial/                      # Step-by-step tutorials (start here!)
│   ├── 00_quickstart.py          # 5-minute multi-agent demo
│   ├── 01_introduction.py        # What is an AI Agent?
│   ├── 02_agents_with_tools.py   # Adding tools to agents
│   ├── 03_multi_agent_communication.py  # LangGraph basics
│   ├── 04_advanced_patterns.py   # Debate, Reflection, Hierarchical
│   └── 05_practical_examples.py  # Real-world use cases
│
├── distributed/                   # Production distributed system
│   └── http_agents/              # HTTP-based agent services
│       ├── core/                 # Framework (base_agent, registry, tools)
│       ├── agents/               # Agent implementations
│       ├── orchestrator.py       # Multi-instance orchestrator
│       └── run_demo.sh           # Quick demo script
│
├── Agent_RL/                      # Reinforcement Learning for Agents
│   ├── advanced_sql_agent.py     # SQL agent with learning
│   ├── docker/                   # PostgreSQL setup
│   └── run_sql_agent.sh          # Tmux automation
│
└── requirements.txt
```

---

# Tutorial Series

## Tutorial 0: Quickstart (5 Minutes)

**File:** `tutorial/00_quickstart.py`

The minimal code needed to create a working multi-agent system:

```python
from langgraph.graph import StateGraph, END, START

# Step 1: Define shared state (communication channel)
class TeamState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    plan: str
    result: str

# Step 2: Create agents (nodes)
def create_planner(llm):
    def planner(state: TeamState) -> dict:
        response = llm.invoke([...])
        return {"plan": response.content}
    return planner

# Step 3: Build graph (workflow)
workflow = StateGraph(TeamState)
workflow.add_node("planner", create_planner(llm))
workflow.add_node("executor", create_executor(llm))
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", END)

# Step 4: Run it!
team = workflow.compile()
result = team.invoke({"task": "Write a poem about AI"})
```

**Key Concepts:**
- Shared state = how agents communicate
- Nodes = agent functions
- Edges = workflow paths

---

## Tutorial 1: Introduction to AI Agents

**File:** `tutorial/01_introduction.py`

Understand what makes an AI agent different from a simple chatbot:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANATOMY OF AN AI AGENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                               │
│   │    LLM      │  ← Brain: Reasoning & Decision Making         │
│   │  (Brain)    │                                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│   ┌──────┴──────┐                                               │
│   │   TOOLS     │  ← Capabilities: APIs, Search, Code Exec      │
│   │  (Hands)    │                                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│   ┌──────┴──────┐                                               │
│   │   MEMORY    │  ← Context: State across interactions         │
│   │  (Memory)   │                                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│   ┌──────┴──────┐                                               │
│   │  PROMPTS    │  ← Personality: Goals & Behavior              │
│   │  (Soul)     │                                               │
│   └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Topics Covered:**
- Simple chat vs. Agent (what's the difference?)
- The ReAct pattern (Reason + Act)
- Creating agents with system prompts
- Agent loops and decision making

---

## Tutorial 2: Agents with Tools

**File:** `tutorial/02_agents_with_tools.py`

Tools transform chatbots into powerful agents that can interact with the world:

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for information."""
    # Your search implementation
    return results

# Bind tools to agent
llm_with_tools = llm.bind_tools([calculator, search_knowledge])
```

**Topics Covered:**
- Creating custom tools with `@tool` decorator
- Tool schemas and documentation
- Binding tools to LLMs
- Tool execution and observation
- ReAct agent with tools

---

## Tutorial 3: Multi-Agent Communication

**File:** `tutorial/03_multi_agent_communication.py`

Learn how multiple agents work together using LangGraph:

```
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-AGENT PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   USER INPUT                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │ RESEARCHER  │────▶│  ANALYST    │────▶│   WRITER    │       │
│   │             │     │             │     │             │       │
│   │ Gathers     │     │ Synthesizes │     │ Composes    │       │
│   │ information │     │ insights    │     │ output      │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                 │
│   SHARED STATE (Communication Channel)                          │
│   ┌─────────────────────────────────────────────────────┐       │
│   │ task: "Write about AI"                              │       │
│   │ research_notes: "AI history, applications..."       │       │
│   │ analysis: "Key themes: automation, ethics..."       │       │
│   │ final_output: "The complete article..."             │       │
│   └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Topics Covered:**
- Why use multiple agents?
- LangGraph State, Nodes, and Edges
- Sequential pipelines
- Conditional routing
- Building research → analysis → writing pipeline

---

## Tutorial 4: Advanced Patterns

**File:** `tutorial/04_advanced_patterns.py`

Master sophisticated multi-agent architectures:

### 1. Hierarchical Pattern
```
         ┌─────────┐
         │ Manager │
         └────┬────┘
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Agent A│ │Agent B│ │Agent C│
└───────┘ └───────┘ └───────┘
```

### 2. Debate Pattern (5 Rounds)
```
┌──────────┐     ┌──────────┐
│   PRO    │◄───►│   CON    │  ← 5 rounds of debate
└────┬─────┘     └─────┬────┘
     │                 │
     └────────┬────────┘
              ▼
        ┌──────────┐
        │  JUDGE   │  ← Synthesizes all rounds
        └──────────┘
```

### 3. Reflection Pattern
```
┌──────────┐     ┌──────────┐
│Generator │────►│ Critic   │
└────┬─────┘     └─────┬────┘
     ▲                 │
     └─────────────────┘  ← Iterates until quality
```

### 4. Human-in-the-Loop
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Agent A  │────►│  HUMAN   │────►│ Agent B  │
└──────────┘     │ APPROVAL │     └──────────┘
                 └──────────┘
```

**When to Use Each Pattern:**
| Pattern | Use Case |
|---------|----------|
| Hierarchical | Complex tasks needing decomposition |
| Debate | Decisions requiring multiple perspectives |
| Reflection | Quality-critical outputs |
| Human-in-the-Loop | Safety-critical applications |

---

## Tutorial 5: Practical Examples

**File:** `tutorial/05_practical_examples.py`

Real-world multi-agent systems you can use today:

### 1. Code Review Team
```
Security Reviewer → Code Reviewer → Test Suggester → Summary
```
- Finds SQL injection, XSS vulnerabilities
- Reviews code quality and best practices
- Suggests unit and integration tests
- Provides actionable summary

### 2. Content Creation Pipeline
```
Outline Agent → Writer Agent → Editor Agent
```
- Creates structured outlines
- Writes engaging content
- Polishes and formats final output

### 3. Customer Support System
```
          ┌─────────────┐
          │ Classifier  │
          └──────┬──────┘
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│Technical│ │ Billing │ │ General │
└─────────┘ └─────────┘ └─────────┘
```
- Automatic query classification
- Specialized agent routing
- Sentiment-aware responses

---

# Distributed Multi-Agent System

**Directory:** `distributed/http_agents/`

Production-ready architecture for deploying agents as HTTP services:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DISTRIBUTED ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────────────┐                         │
│                         │      ORCHESTRATOR       │                         │
│                         │   • Agent Registry      │                         │
│                         │   • Load Balancing      │                         │
│                         │   • Health Checking     │                         │
│                         └───────────┬─────────────┘                         │
│                                     │                                       │
│           ┌─────────────────────────┼─────────────────────────┐             │
│           │                         │                         │             │
│           ▼                         ▼                         ▼             │
│   ┌───────────────┐       ┌───────────────┐       ┌───────────────┐         │
│   │  RESEARCHER   │       │   ANALYST     │       │    WRITER     │         │
│   │  Instance 1   │       │  Instance 1   │       │  Instance 1   │         │
│   │  Instance 2   │       │  Instance 2   │       │  Instance 2   │         │
│   │  Instance N   │       │  Instance N   │       │  Instance N   │         │
│   └───────────────┘       └───────────────┘       └───────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **HTTP-based communication** - Each agent is a FastAPI service
- **Multi-instance support** - Run multiple copies of each agent
- **Load balancing strategies** - Round Robin, Least Loaded, Fastest, Random
- **Health monitoring** - Automatic health checks and failover
- **Thinking model support** - Compatible with DeepSeek-R1 reasoning

## Quick Start

```bash
cd distributed/http_agents

# Run the demo (starts all agents + orchestrator)
./run_demo.sh

# Or with options
./run_demo.sh --model llama3.2 --scale 3 --task "Explain quantum computing"
```

## Load Balancing

| Strategy | Description | Best For |
|----------|-------------|----------|
| `round_robin` | Distributes evenly | Uniform workloads |
| `least_loaded` | Routes to least busy | Variable task complexity |
| `fastest` | Routes to quickest | Latency-sensitive |
| `random` | Random selection | Simple distribution |

---

# SQL Learning Agent

**Directory:** `Agent_RL/`

An intelligent agent that learns SQL patterns, auto-corrects errors, and persists knowledge:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQL LEARNING LOOP                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. USER QUERY: "SELECT * FROM employes"                        │
│                          │                                      │
│  2. EXECUTE → Error: relation "employes" does not exist         │
│                          │                                      │
│  3. AUTO-CORRECT                                                │
│     ├── Check learned corrections                               │
│     ├── Find similar: "employees"                               │
│     └── Retry corrected query                                   │
│                          │                                      │
│  4. SUCCESS → Return results                                    │
│                          │                                      │
│  5. LEARN → Store: employes → employees                         │
│                          │                                      │
│  6. PERSIST → Save to sql_knowledge.json                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Real PostgreSQL** with Docker
- **Auto-correction** of table names, syntax errors, ambiguous columns
- **Pattern learning** - Remembers successful queries
- **Schema awareness** - Understands table relationships
- **Interactive mode** - Natural language queries
- **Tmux multi-pane** - See database, agent, and psql together

## Quick Start

```bash
cd Agent_RL

# Start everything (database + agent + psql)
./run_sql_agent.sh

# Interactive commands
.tables          # Show tables
.ask employees with managers  # Natural language query
.stats           # Show learning statistics
```

## Database Schema

```
DEPARTMENTS ──┬──> EMPLOYEES ──> EMPLOYEES (self-ref: manager)
              │
CATEGORIES ───┼──> CATEGORIES (self-ref: subcategories)
              └──> PRODUCTS

CUSTOMERS ────┬──> ADDRESSES
              └──> ORDERS ──> ORDER_ITEMS ──> PRODUCTS
```

---

# Installation

## Requirements

- Python 3.8+
- Ollama (for local LLM inference)
- Docker (for SQL Learning Agent)
- tmux (optional, for multi-pane views)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull a model
# Visit: https://ollama.ai
ollama pull llama3.2
ollama serve
```

## Verify Installation

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Run quickstart
python tutorial/00_quickstart.py
```

---

# Key Concepts

## State: The Communication Channel

Agents communicate through shared state:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Accumulates
    task: str                    # Input task
    research_notes: str          # From Researcher
    analysis: str                # From Analyst
    final_output: str            # From Writer
```

## Conditional Routing

Route based on state:

```python
def route_support(state) -> Literal["technical", "billing", "general"]:
    if state["category"] == "technical":
        return "technical"
    elif state["category"] == "billing":
        return "billing"
    return "general"

workflow.add_conditional_edges("classifier", route_support, {...})
```

## Tools

Give agents capabilities:

```python
@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    return results

llm_with_tools = llm.bind_tools([search_database])
```

---

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**Built with LangChain, LangGraph, Ollama, and FastAPI**
