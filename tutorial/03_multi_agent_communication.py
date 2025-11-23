"""
================================================================================
TUTORIAL 3: Multi-Agent Communication with LangGraph
================================================================================

Multi-agent systems enable complex workflows by having specialized agents
work together. This tutorial covers:

1. Why use multiple agents?
2. LangGraph basics - State and Graphs
3. Communication patterns between agents
4. Building a multi-agent system step by step

Prerequisites:
- Understanding of basic agents (Tutorial 1)
- Understanding of tools (Tutorial 2)
- pip install langgraph langchain-ollama
"""

import operator
from typing import Annotated, TypedDict, Sequence, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START


def create_llm():
    """Create LLM instance."""
    return ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=0.7,
    )


# ============================================================================
# PART 1: Why Multiple Agents?
# ============================================================================

"""
WHY USE MULTIPLE AGENTS?
------------------------

1. SPECIALIZATION:
   - Each agent can be an expert in one domain
   - Better prompts, better results
   - Example: One agent for coding, another for testing

2. SEPARATION OF CONCERNS:
   - Different agents handle different parts of a task
   - Easier to debug and maintain
   - Example: Researcher → Analyst → Writer

3. PARALLELIZATION:
   - Multiple agents can work simultaneously
   - Faster completion of complex tasks
   - Example: Search multiple sources in parallel

4. CHECKS AND BALANCES:
   - Agents can review each other's work
   - Debate pattern for better decisions
   - Example: Proposer → Critic → Refiner
"""


# ============================================================================
# PART 2: LangGraph Basics - State
# ============================================================================

"""
LANGGRAPH CORE CONCEPTS:
------------------------

1. STATE: A shared data structure that all agents can read/write
2. NODES: Agents or functions that process the state
3. EDGES: Connections between nodes (workflow paths)
4. GRAPH: The complete workflow definition
"""


# Define the shared state - THIS IS THE KEY TO COMMUNICATION
class AgentState(TypedDict):
    """
    Shared state for multi-agent communication.

    Think of this as a "shared whiteboard" that all agents can:
    - READ from: Get information from other agents
    - WRITE to: Share their results with other agents

    The Annotated[..., operator.add] means messages accumulate (append)
    rather than overwrite.
    """
    # Conversation history - accumulates with each agent's contribution
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Task to be completed
    task: str

    # Results from each agent (this is how agents communicate!)
    research_notes: str
    analysis: str
    final_response: str

    # Control flow - which agent should act next
    next_agent: str


# ============================================================================
# PART 3: Creating Specialized Agents as Nodes
# ============================================================================

def create_researcher_node(llm):
    """
    RESEARCHER AGENT
    ----------------
    Responsibility: Gather information about the topic
    Reads: task
    Writes: research_notes, messages
    """

    def researcher(state: AgentState) -> dict:
        """Research node - gathers information."""

        system_prompt = """You are a Research Agent. Your job is to:
1. Understand the topic given to you
2. Provide comprehensive research notes
3. Include key facts, definitions, and context

Be thorough but concise. Focus on factual information."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research this topic: {state['task']}")
        ]

        response = llm.invoke(messages)

        # Return updates to state - THIS IS HOW WE COMMUNICATE
        return {
            "messages": [AIMessage(content=f"[RESEARCHER]: {response.content}")],
            "research_notes": response.content,
        }

    return researcher


def create_analyst_node(llm):
    """
    ANALYST AGENT
    -------------
    Responsibility: Analyze the research and extract insights
    Reads: task, research_notes (from Researcher)
    Writes: analysis, messages
    """

    def analyst(state: AgentState) -> dict:
        """Analyst node - analyzes research."""

        system_prompt = """You are an Analyst Agent. Your job is to:
1. Review research notes provided by the Research Agent
2. Identify key insights and patterns
3. Provide structured analysis

Focus on extracting actionable insights."""

        # READ from state - get researcher's output
        research = state.get('research_notes', 'No research available')

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Analyze this research:

TOPIC: {state['task']}

RESEARCH NOTES:
{research}

Provide your analysis.""")
        ]

        response = llm.invoke(messages)

        # WRITE to state - share analysis with other agents
        return {
            "messages": [AIMessage(content=f"[ANALYST]: {response.content}")],
            "analysis": response.content,
        }

    return analyst


def create_writer_node(llm):
    """
    WRITER AGENT
    ------------
    Responsibility: Compose the final response
    Reads: task, research_notes, analysis (from both previous agents)
    Writes: final_response, messages
    """

    def writer(state: AgentState) -> dict:
        """Writer node - creates final response."""

        system_prompt = """You are a Writer Agent. Your job is to:
1. Review all research and analysis from your colleagues
2. Compose a clear, well-structured final response
3. Ensure the response addresses the original task

Write in a professional, engaging manner."""

        # READ from state - get both agents' outputs
        research = state.get('research_notes', '')
        analysis = state.get('analysis', '')

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Create a final response based on:

ORIGINAL TASK: {state['task']}

RESEARCH NOTES:
{research}

ANALYSIS:
{analysis}

Write the final response.""")
        ]

        response = llm.invoke(messages)

        return {
            "messages": [AIMessage(content=f"[WRITER]: {response.content}")],
            "final_response": response.content,
        }

    return writer


# ============================================================================
# PART 4: Building the Graph (Workflow)
# ============================================================================

def build_sequential_graph(llm):
    """
    Build a SEQUENTIAL multi-agent graph.

    Flow: START → researcher → analyst → writer → END

    This is the simplest pattern - agents run one after another,
    each building on the previous agent's work.
    """

    # Create the graph with our state type
    workflow = StateGraph(AgentState)

    # Create agent nodes
    researcher = create_researcher_node(llm)
    analyst = create_analyst_node(llm)
    writer = create_writer_node(llm)

    # Add nodes to graph
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("writer", writer)

    # Define edges (the flow)
    workflow.add_edge(START, "researcher")     # Start with researcher
    workflow.add_edge("researcher", "analyst")  # Then analyst
    workflow.add_edge("analyst", "writer")      # Then writer
    workflow.add_edge("writer", END)            # Done

    # Compile the graph
    return workflow.compile()


def example_sequential_agents():
    """Run the sequential multi-agent system."""
    print("\n" + "="*60)
    print("Example: Sequential Multi-Agent System")
    print("="*60)

    llm = create_llm()
    graph = build_sequential_graph(llm)

    # Initial state
    initial_state = {
        "messages": [],
        "task": "Explain how artificial intelligence is changing healthcare",
        "research_notes": "",
        "analysis": "",
        "final_response": "",
        "next_agent": "",
    }

    print(f"\nTask: {initial_state['task']}")
    print("-" * 40)

    # Run the graph
    result = graph.invoke(initial_state)

    # Show the communication between agents
    print("\n📝 AGENT COMMUNICATION LOG:")
    print("-" * 40)
    for msg in result["messages"]:
        print(f"\n{msg.content[:500]}...\n")

    print("\n" + "="*40)
    print("📋 FINAL RESPONSE:")
    print("="*40)
    print(result["final_response"][:1000])

    return result


# ============================================================================
# PART 5: Conditional Routing (Dynamic Flow)
# ============================================================================

def create_router_node(llm):
    """
    ROUTER/SUPERVISOR AGENT
    -----------------------
    Responsibility: Decide which agent should act next
    This enables dynamic workflows based on the current state.
    """

    def router(state: AgentState) -> dict:
        """Decide which agent should act next."""

        # Simple rule-based routing
        if not state.get('research_notes'):
            return {"next_agent": "researcher"}
        elif not state.get('analysis'):
            return {"next_agent": "analyst"}
        elif not state.get('final_response'):
            return {"next_agent": "writer"}
        else:
            return {"next_agent": "end"}

    return router


def route_to_agent(state: AgentState) -> Literal["researcher", "analyst", "writer", "end"]:
    """Routing function for conditional edges."""
    return state.get("next_agent", "researcher")


def build_supervised_graph(llm):
    """
    Build a SUPERVISED multi-agent graph with dynamic routing.

    Flow:
    START → router → [researcher|analyst|writer] → router → ... → END

    The router decides which agent should act based on current state.
    """

    workflow = StateGraph(AgentState)

    # Create nodes
    router = create_router_node(llm)
    researcher = create_researcher_node(llm)
    analyst = create_analyst_node(llm)
    writer = create_writer_node(llm)

    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("writer", writer)

    # Start with router
    workflow.add_edge(START, "router")

    # Conditional routing from router
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "end": END
        }
    )

    # All agents go back to router after completing
    workflow.add_edge("researcher", "router")
    workflow.add_edge("analyst", "router")
    workflow.add_edge("writer", "router")

    return workflow.compile()


def example_supervised_agents():
    """Run the supervised multi-agent system."""
    print("\n" + "="*60)
    print("Example: Supervised Multi-Agent System")
    print("="*60)

    llm = create_llm()
    graph = build_supervised_graph(llm)

    initial_state = {
        "messages": [],
        "task": "What are the benefits and risks of electric vehicles?",
        "research_notes": "",
        "analysis": "",
        "final_response": "",
        "next_agent": "",
    }

    print(f"\nTask: {initial_state['task']}")
    print("-" * 40)

    result = graph.invoke(initial_state)

    print("\n📝 AGENT COMMUNICATION LOG:")
    for msg in result["messages"]:
        print(f"\n{msg.content[:300]}...")

    return result


# ============================================================================
# Key Concepts Summary
# ============================================================================

"""
KEY TAKEAWAYS - MULTI-AGENT COMMUNICATION:
------------------------------------------

1. SHARED STATE IS THE COMMUNICATION CHANNEL:
   - All agents read from and write to the same state
   - State fields are how agents pass information
   - Use TypedDict to define clear state structure

2. NODES ARE AGENTS:
   - Each node is a function that processes state
   - Nodes return updates to the state (not full state)
   - This is how information flows between agents

3. EDGES DEFINE WORKFLOW:
   - Static edges: Always go A → B
   - Conditional edges: Route based on state
   - Enables complex, dynamic workflows

4. PATTERNS:
   - Sequential: A → B → C (pipeline)
   - Supervised: Router decides next agent
   - Parallel: Multiple agents run simultaneously
   - Iterative: Agents can loop back

5. COMMUNICATION METHODS:
   a. Via State Fields:
      - Agent A writes to state['research']
      - Agent B reads from state['research']

   b. Via Messages:
      - Append to messages list
      - Full conversation history visible

   c. Via Control Flow:
      - Set 'next_agent' to control routing
      - Supervisor pattern

NEXT: Tutorial 4 - Advanced Patterns (Hierarchical, Debate, etc.)
"""


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TUTORIAL 3: Multi-Agent Communication")
    print("="*60)

    # Run sequential example
    #example_sequential_agents()

    # Uncomment to run supervised example:
    example_supervised_agents()
