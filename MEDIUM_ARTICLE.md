# Building Multi-Agent AI Systems from Scratch: A Complete Guide with LangGraph

*From basic chatbots to production-ready distributed systems — learn how to build AI agents that collaborate, debate, and learn from mistakes.*

---

The future of AI isn't a single all-knowing model — it's teams of specialized agents working together. Just like human organizations, AI systems perform better when different "experts" handle different aspects of a problem.

In this comprehensive guide, I'll take you from zero to building production-ready multi-agent systems. We'll cover everything from basic concepts to deploying distributed agents with load balancing.

**What you'll build:**
- Multi-agent pipelines (Researcher → Analyst → Writer)
- Debate systems where agents argue different perspectives
- Self-improving agents with reflection patterns
- A SQL agent that learns from its mistakes
- Production HTTP services with load balancing

Let's dive in.

---

## What is an AI Agent?

Before we build multi-agent systems, let's understand what makes something an "agent" rather than just a chatbot.

An AI agent has four key components:

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

**A chatbot** responds to messages. **An agent** perceives, reasons, acts, and learns.

The key difference? Agents can:
1. **Use tools** to interact with the world (search, calculate, call APIs)
2. **Make decisions** about what to do next
3. **Iterate** until a goal is achieved
4. **Maintain state** across interactions

---

## Why Multiple Agents?

Single agents have limits. Multi-agent systems unlock new capabilities:

| Benefit | Description | Example |
|---------|-------------|---------|
| **Specialization** | Each agent masters one domain | Security expert + Code reviewer |
| **Separation of Concerns** | Different agents handle different tasks | Researcher → Analyst → Writer |
| **Checks & Balances** | Agents review each other's work | Proposer ↔ Critic |
| **Parallelization** | Multiple agents work simultaneously | Search multiple sources at once |

Think of it like a company: you don't have one person do everything. You have specialists who collaborate.

---

## Your First Multi-Agent System (5 Minutes)

Let's build a simple two-agent system using LangGraph. One agent plans, the other executes.

```python
import operator
from typing import Annotated, TypedDict, Sequence
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

# Step 1: Define shared state (this is how agents communicate!)
class TeamState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    plan: str
    result: str

# Step 2: Create the Planner agent
def create_planner(llm):
    def planner(state: TeamState) -> dict:
        response = llm.invoke([
            SystemMessage(content="You are a Planner. Create a brief 3-step plan."),
            HumanMessage(content=f"Create a plan for: {state['task']}")
        ])
        return {
            "messages": [AIMessage(content=f"[PLANNER]: {response.content}")],
            "plan": response.content,
        }
    return planner

# Step 3: Create the Executor agent
def create_executor(llm):
    def executor(state: TeamState) -> dict:
        response = llm.invoke([
            SystemMessage(content="You are an Executor. Follow the plan and provide results."),
            HumanMessage(content=f"Execute this plan:\n{state['plan']}")
        ])
        return {
            "messages": [AIMessage(content=f"[EXECUTOR]: {response.content}")],
            "result": response.content,
        }
    return executor

# Step 4: Build the graph (workflow)
def build_team():
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

    workflow = StateGraph(TeamState)
    workflow.add_node("planner", create_planner(llm))
    workflow.add_node("executor", create_executor(llm))

    # Define flow: START → planner → executor → END
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)

    return workflow.compile()

# Step 5: Run it!
team = build_team()
result = team.invoke({
    "messages": [],
    "task": "Write a haiku about programming",
    "plan": "",
    "result": ""
})

print(result["result"])
```

**Key insight:** The shared `TeamState` is how agents communicate. Each agent reads from it and writes back to it. It's like a shared whiteboard.

---

## The Core Pattern: State, Nodes, and Edges

Every LangGraph multi-agent system has three components:

### 1. State (The Communication Channel)

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Accumulates
    task: str                    # Input
    research_notes: str          # From Researcher
    analysis: str                # From Analyst
    final_output: str            # From Writer
```

The `Annotated[..., operator.add]` means messages accumulate rather than overwrite.

### 2. Nodes (The Agents)

Each node is a function that takes state and returns updates:

```python
def researcher(state: AgentState) -> dict:
    # Do research...
    return {"research_notes": "My findings..."}
```

### 3. Edges (The Workflow)

Edges define how agents connect:

```python
workflow.add_edge(START, "researcher")      # Start with researcher
workflow.add_edge("researcher", "analyst")   # Then analyst
workflow.add_edge("analyst", "writer")       # Then writer
workflow.add_edge("writer", END)             # Done
```

---

## Advanced Pattern 1: The Debate Pattern

Sometimes you need multiple perspectives. The debate pattern has agents argue different sides before a judge synthesizes:

```
┌──────────┐     ┌──────────┐
│   PRO    │◄───►│   CON    │  ← 5 rounds of debate
└────┬─────┘     └─────┬────┘
     │                 │
     └────────┬────────┘
              ▼
        ┌──────────┐
        │  JUDGE   │  ← Synthesizes all arguments
        └──────────┘
```

Here's how to implement multi-round debates:

```python
class DebateState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    pro_argument: str
    con_argument: str
    debate_history: List[dict]  # Track all rounds
    debate_round: int
    max_rounds: int
    final_verdict: str

def route_after_con(state: DebateState) -> Literal["pro", "judge"]:
    """After CON speaks, either continue or go to judge."""
    if state["debate_round"] > state["max_rounds"]:
        return "judge"
    return "pro"  # Continue debating

# Build the graph
workflow = StateGraph(DebateState)
workflow.add_node("pro", create_pro_agent(llm))
workflow.add_node("con", create_con_agent(llm))
workflow.add_node("judge", create_judge_agent(llm))

workflow.add_edge(START, "pro")
workflow.add_edge("pro", "con")
workflow.add_conditional_edges("con", route_after_con, {"pro": "pro", "judge": "judge"})
workflow.add_edge("judge", END)
```

The key is the **conditional edge** after CON — it loops back for more rounds or proceeds to the judge.

**When to use:** Decisions requiring multiple perspectives, reducing bias, complex deliberation.

---

## Advanced Pattern 2: Reflection (Self-Improvement)

The reflection pattern has a generator create content, then a critic reviews it, and the cycle repeats until quality is achieved:

```
┌──────────┐     ┌──────────┐
│Generator │────►│ Critic   │
└────┬─────┘     └─────┬────┘
     ▲                 │
     └─────────────────┘  ← Iterate until quality
```

```python
def create_critic(llm):
    def critic(state: ReflectionState) -> dict:
        if state["iteration"] >= 3:  # Max iterations
            return {"final_output": state["draft"]}

        response = llm.invoke([
            SystemMessage(content="You are a constructive critic."),
            HumanMessage(content=f"Review this draft:\n{state['draft']}")
        ])

        if "APPROVED" in response.content.upper():
            return {"final_output": state["draft"]}

        return {"critique": response.content}

    return critic
```

**When to use:** Quality-critical outputs, writing tasks, code generation.

---

## Advanced Pattern 3: Hierarchical Teams

For complex tasks, a manager agent breaks down work and delegates to specialists:

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

The manager:
1. Receives a complex task
2. Breaks it into subtasks
3. Assigns each to the right specialist
4. Synthesizes the results

**When to use:** Complex tasks requiring decomposition, when different subtasks need different expertise.

---

## Practical Example: Code Review Team

Let's build a real-world system — a code review pipeline with security, quality, and testing agents:

```
Security Reviewer → Code Reviewer → Test Suggester → Summary
```

```python
class CodeReviewState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    code: str
    language: str
    security_review: str
    code_review: str
    test_suggestions: str
    final_summary: str

def create_security_reviewer(llm):
    def reviewer(state: CodeReviewState) -> dict:
        prompt = f"""Review this code for security vulnerabilities:

{state['code']}

Check for: SQL injection, XSS, authentication issues, data exposure."""

        response = llm.invoke([
            SystemMessage(content="You are a security expert."),
            HumanMessage(content=prompt)
        ])

        return {"security_review": response.content}

    return reviewer
```

Feed it vulnerable code like this:

```python
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}'"  # SQL Injection!
    result = db.execute(query)
    return {"password": data.password, "ssn": data.ssn}  # Data exposure!
```

The security agent catches both vulnerabilities, the code reviewer suggests improvements, and the test agent recommends what to test.

---

## Building a Learning Agent: SQL Auto-Correction

Here's where it gets interesting. What if an agent could learn from its mistakes?

I built a SQL agent that:
1. Executes queries against PostgreSQL
2. When queries fail, it auto-corrects them
3. Remembers corrections for next time
4. Persists knowledge to disk

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

The agent maintains a knowledge base:

```json
{
  "corrections": {
    "table_not_found:employes": {
      "corrected_example": "SELECT * FROM employees",
      "count": 3
    }
  },
  "common_joins": {
    "orders:customers": "orders.customer_id = customers.id"
  },
  "stats": {
    "total_queries": 50,
    "successful": 45,
    "auto_corrected": 8
  }
}
```

After using it for a while, it becomes remarkably good at fixing typos and ambiguous queries automatically.

---

## Going Production: Distributed HTTP Agents

For production systems, you need:
- Horizontal scaling
- Fault isolation
- Load balancing
- Health monitoring

I built a distributed architecture where each agent runs as an independent HTTP service:

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
│   └───────────────┘       └───────────────┘       └───────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each agent is a FastAPI service:

```python
from fastapi import FastAPI
from core.base_agent import BaseAgent

class ResearcherAgent(BaseAgent):
    async def process(self, request: TaskRequest) -> TaskResponse:
        # Research logic here
        return TaskResponse(result=research_notes)

app = FastAPI()
agent = ResearcherAgent(port=8001)

@app.post("/process")
async def process_task(request: TaskRequest):
    return await agent.process(request)
```

The orchestrator handles load balancing with multiple strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `round_robin` | Distributes evenly | Uniform workloads |
| `least_loaded` | Routes to least busy instance | Variable complexity |
| `fastest` | Routes to quickest responder | Latency-sensitive |

---

## Choosing the Right Pattern

Here's a decision framework:

| Situation | Pattern |
|-----------|---------|
| Simple pipeline, fixed steps | Sequential |
| Complex task, needs breakdown | Hierarchical |
| Quality is critical | Reflection |
| Need multiple perspectives | Debate |
| Safety-critical decisions | Human-in-the-Loop |
| High throughput required | Distributed HTTP |

---

## Key Takeaways

1. **Agents ≠ Chatbots**: Agents have tools, make decisions, and iterate
2. **State is Communication**: Agents talk through shared state
3. **Patterns Matter**: Choose hierarchical, debate, or reflection based on your needs
4. **Learning is Powerful**: Agents that remember mistakes improve over time
5. **Production Needs Scale**: Use HTTP services with load balancing for real deployments

---

## Get the Code

The complete codebase with all examples is available on GitHub:

**Repository:** [github.com/ramdurga/agent_communication](https://github.com/ramdurga/agent_communication)

```bash
git clone https://github.com/ramdurga/agent_communication.git
cd agent_communication
pip install -r requirements.txt
python tutorial/00_quickstart.py
```

The repository includes:
- 6 progressive tutorials (00-05)
- Distributed HTTP agent framework
- SQL learning agent with Docker PostgreSQL
- All patterns: debate, reflection, hierarchical, human-in-the-loop

---

## What's Next?

Multi-agent systems are evolving rapidly. Some areas to explore:

- **Agent-to-agent protocols**: Standardized ways for agents to communicate
- **Memory systems**: Long-term memory across sessions
- **Tool learning**: Agents that learn to use new tools
- **Self-organizing teams**: Agents that decide their own structure

The future isn't one super-intelligent AI — it's collaborative AI systems where specialized agents work together, each contributing their expertise.

---

*If you found this helpful, follow me for more AI engineering content. Questions? Drop a comment below.*

---

**Tags:** #AI #LLM #LangChain #LangGraph #MultiAgent #MachineLearning #Python #Tutorial
