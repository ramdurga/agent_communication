"""
================================================================================
QUICKSTART: Multi-Agent System in 5 Minutes
================================================================================

This is the minimal code needed to create a working multi-agent system.
Run this file to see agents communicating in action!

Requirements:
    pip install langchain langgraph langchain-ollama
    ollama pull llama3.2
    ollama serve
"""

import operator
from typing import Annotated, TypedDict, Sequence
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START


# ============================================================================
# STEP 1: Define Shared State (Communication Channel)
# ============================================================================

class TeamState(TypedDict):
    """Shared state - this is how agents communicate!"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    plan: str
    result: str


# ============================================================================
# STEP 2: Create Agent Functions (Nodes)
# ============================================================================

def create_planner(llm):
    """Planner agent - creates a plan for the task."""
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


def create_executor(llm):
    """Executor agent - executes the plan."""
    def executor(state: TeamState) -> dict:
        response = llm.invoke([
            SystemMessage(content="You are an Executor. Follow the plan and provide results."),
            HumanMessage(content=f"Execute this plan:\n{state['plan']}\n\nFor task: {state['task']}")
        ])
        return {
            "messages": [AIMessage(content=f"[EXECUTOR]: {response.content}")],
            "result": response.content,
        }
    return executor


# ============================================================================
# STEP 3: Build the Graph (Workflow)
# ============================================================================

def build_team():
    """Build the multi-agent team."""
    # Create LLM
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

    # Create graph
    workflow = StateGraph(TeamState)

    # Add nodes (agents)
    workflow.add_node("planner", create_planner(llm))
    workflow.add_node("executor", create_executor(llm))

    # Define flow: START → planner → executor → END
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)

    return workflow.compile()


# ============================================================================
# STEP 4: Run It!
# ============================================================================

def main():
    print("="*60)
    print("🤖 MULTI-AGENT QUICKSTART")
    print("="*60)

    # Build the team
    team = build_team()

    # Define initial state
    state = {
        "messages": [],
        "task": "Write a Python function to calculate fibonacci numbers",
        "plan": "",
        "result": "",
    }

    print(f"\n📋 Task: {state['task']}\n")
    print("-"*60)

    # Run the agents
    result = team.invoke(state)

    # Show the communication
    print("\n📝 AGENT COMMUNICATION:\n")
    for msg in result["messages"]:
        print(msg.content)
        print("-"*40)

    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
