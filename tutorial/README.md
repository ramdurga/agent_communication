# Multi-Agent Communication Tutorial

A comprehensive tutorial series on building AI agents and multi-agent systems using LangChain and LangGraph.

## Prerequisites

```bash
# Install dependencies
pip install langchain langgraph langchain-core langchain-ollama

# Make sure Ollama is running with a model
ollama pull llama3.2
ollama serve
```

## Tutorial Structure

### 1. Introduction to AI Agents (`01_introduction.py`)
- What is an AI Agent?
- Agent vs Chatbot
- Basic agent with memory
- ReAct pattern fundamentals

### 2. Agents with Tools (`02_agents_with_tools.py`)
- Creating custom tools
- Binding tools to LLMs
- Tool execution loop
- Building a complete ToolAgent class

### 3. Multi-Agent Communication (`03_multi_agent_communication.py`)
- LangGraph basics (State, Nodes, Edges)
- Shared state as communication channel
- Sequential agent pipeline
- Supervised/orchestrated agents

### 4. Advanced Patterns (`04_advanced_patterns.py`)
- Hierarchical multi-agent systems
- Debate pattern (Pro vs Con)
- Reflection/Self-critique pattern
- Human-in-the-loop pattern

### 5. Practical Examples (`05_practical_examples.py`)
- Code Review Team
- Content Creation Pipeline
- Customer Support System

## Quick Start

```bash
# Run the basic introduction
python tutorial/01_introduction.py

# Run agents with tools
python tutorial/02_agents_with_tools.py

# Run multi-agent communication
python tutorial/03_multi_agent_communication.py

# Run advanced patterns
python tutorial/04_advanced_patterns.py

# Run practical examples
python tutorial/05_practical_examples.py
```

## Key Concepts

### 1. Agent Components
```
┌─────────────────────────────────────┐
│              AI AGENT               │
├─────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐          │
│  │   LLM   │  │  Tools  │          │
│  │ (Brain) │  │ (Hands) │          │
│  └─────────┘  └─────────┘          │
│  ┌─────────┐  ┌─────────┐          │
│  │ Memory  │  │ Prompt  │          │
│  │ (State) │  │(Identity)│         │
│  └─────────┘  └─────────┘          │
└─────────────────────────────────────┘
```

### 2. Multi-Agent Communication via Shared State
```python
class AgentState(TypedDict):
    # All agents read/write to this shared state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research: str      # Agent A writes, Agent B reads
    analysis: str      # Agent B writes, Agent C reads
    final_output: str  # Agent C writes
```

### 3. LangGraph Workflow
```
     ┌─────────┐
     │  START  │
     └────┬────┘
          │
          ▼
     ┌─────────┐
     │ Agent A │──── writes to state
     └────┬────┘
          │
          ▼
     ┌─────────┐
     │ Agent B │──── reads from state, writes results
     └────┬────┘
          │
          ▼
     ┌─────────┐
     │   END   │
     └─────────┘
```

## Common Patterns

| Pattern | Use Case | Structure |
|---------|----------|-----------|
| Sequential | Simple pipeline | A → B → C |
| Supervised | Dynamic routing | Manager → Workers |
| Hierarchical | Complex tasks | Manager → Specialists |
| Debate | Multiple perspectives | Pro ↔ Con → Judge |
| Reflection | Quality assurance | Generator ↔ Critic |
| Parallel | Speed optimization | Multiple agents at once |

## Using Different LLMs

The tutorials use Ollama by default. To use other LLMs:

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Local with Ollama
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")

# vLLM (high performance)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="meta-llama/Llama-3.1-8B-Instruct"
)
```

## Project Structure

```
tutorial/
├── README.md                    # This file
├── 01_introduction.py           # Basic agent concepts
├── 02_agents_with_tools.py      # Tool usage
├── 03_multi_agent_communication.py  # LangGraph basics
├── 04_advanced_patterns.py      # Advanced patterns
└── 05_practical_examples.py     # Real-world examples
```

## Next Steps

After completing these tutorials:

1. **Experiment**: Modify the examples to solve your own problems
2. **Add Tools**: Create custom tools for your use case
3. **Scale Up**: Try more complex multi-agent architectures
4. **Production**: Add error handling, logging, and monitoring
5. **Optimize**: Fine-tune prompts and agent interactions

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.com/)
