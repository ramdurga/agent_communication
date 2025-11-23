"""
================================================================================
TUTORIAL 4: Advanced Multi-Agent Communication Patterns
================================================================================

This tutorial covers advanced patterns for agent communication:

1. Hierarchical Multi-Agent Systems
2. Debate Pattern (Agent Discussion)
3. Parallel Execution
4. Human-in-the-Loop
5. Agent Handoff Pattern
6. Reflection/Self-Critique Pattern

These patterns enable more sophisticated agent interactions.
"""

import operator
from typing import Annotated, TypedDict, Sequence, Literal, List
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
# PATTERN 1: Hierarchical Multi-Agent System
# ============================================================================

"""
HIERARCHICAL PATTERN:
--------------------
A manager agent coordinates specialist agents.

Structure:
         ┌─────────┐
         │ Manager │
         └────┬────┘
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│Agent A│ │Agent B│ │Agent C│
└───────┘ └───────┘ └───────┘

Use when:
- Complex tasks need to be broken down
- Different specialists for different subtasks
- Need central coordination
"""


class HierarchicalState(TypedDict):
    """State for hierarchical agent system."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    subtasks: List[str]
    current_subtask: str
    subtask_results: dict
    final_result: str
    iteration: int


def create_manager_agent(llm):
    """Manager agent that breaks down tasks and coordinates specialists."""

    def manager(state: HierarchicalState) -> dict:
        # First call: Break down the task into subtasks
        if not state.get('subtasks'):
            prompt = f"""You are a Manager Agent. Break down this task into 2-3 subtasks:

TASK: {state['task']}

List the subtasks, one per line. Be specific and actionable."""

            response = llm.invoke([
                SystemMessage(content="You break down complex tasks into subtasks."),
                HumanMessage(content=prompt)
            ])

            # Parse subtasks (simple split by newline)
            subtasks = [s.strip() for s in response.content.split('\n') if s.strip()]
            subtasks = subtasks[:3]  # Limit to 3

            return {
                "messages": [AIMessage(content=f"[MANAGER] Created subtasks: {subtasks}")],
                "subtasks": subtasks,
                "current_subtask": subtasks[0] if subtasks else "",
                "subtask_results": {},
                "iteration": 0,
            }

        # Subsequent calls: Check if all subtasks are done
        results = state.get('subtask_results', {})
        subtasks = state.get('subtasks', [])

        if len(results) >= len(subtasks):
            # All done - create final summary
            summary_prompt = f"""Summarize the results from all subtasks:

ORIGINAL TASK: {state['task']}

SUBTASK RESULTS:
{chr(10).join(f'- {k}: {v}' for k, v in results.items())}

Provide a cohesive final answer."""

            response = llm.invoke([
                SystemMessage(content="You synthesize results into a final answer."),
                HumanMessage(content=summary_prompt)
            ])

            return {
                "messages": [AIMessage(content=f"[MANAGER] Final: {response.content}")],
                "final_result": response.content,
            }

        # Find next incomplete subtask
        for subtask in subtasks:
            if subtask not in results:
                return {
                    "current_subtask": subtask,
                    "iteration": state.get('iteration', 0) + 1,
                }

        return {}

    return manager


def create_specialist_agent(llm):
    """Specialist agent that handles individual subtasks."""

    def specialist(state: HierarchicalState) -> dict:
        current = state.get('current_subtask', '')
        if not current:
            return {}

        prompt = f"""Complete this subtask:

MAIN TASK: {state['task']}
YOUR SUBTASK: {current}

Provide a focused, helpful response."""

        response = llm.invoke([
            SystemMessage(content="You are a specialist who completes subtasks efficiently."),
            HumanMessage(content=prompt)
        ])

        # Update results
        results = dict(state.get('subtask_results', {}))
        results[current] = response.content

        return {
            "messages": [AIMessage(content=f"[SPECIALIST] {current}: {response.content[:200]}...")],
            "subtask_results": results,
        }

    return specialist


def route_hierarchical(state: HierarchicalState) -> Literal["manager", "specialist", "end"]:
    """Route in hierarchical system."""
    if state.get('final_result'):
        return "end"
    elif state.get('current_subtask') and state.get('current_subtask') not in state.get('subtask_results', {}):
        return "specialist"
    else:
        return "manager"


def build_hierarchical_graph(llm):
    """Build hierarchical multi-agent graph."""
    workflow = StateGraph(HierarchicalState)

    workflow.add_node("manager", create_manager_agent(llm))
    workflow.add_node("specialist", create_specialist_agent(llm))

    workflow.add_edge(START, "manager")

    workflow.add_conditional_edges(
        "manager",
        route_hierarchical,
        {"manager": "manager", "specialist": "specialist", "end": END}
    )

    workflow.add_edge("specialist", "manager")

    return workflow.compile()


# ============================================================================
# PATTERN 2: Debate Pattern
# ============================================================================

"""
DEBATE PATTERN:
--------------
Agents argue different perspectives, then reach consensus.

Structure:
┌──────────┐     ┌──────────┐
│ Agent A  │◄───►│ Agent B  │
│(Pro/View1)│     │(Con/View2)│
└────┬─────┘     └─────┬────┘
     │                 │
     └────────┬────────┘
              ▼
        ┌──────────┐
        │  Judge   │
        │(Synthesize)│
        └──────────┘

Use when:
- Need to consider multiple perspectives
- Want to reduce bias
- Complex decisions need deliberation
"""


class DebateState(TypedDict):
    """State for debate pattern."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    topic: str
    pro_argument: str
    con_argument: str
    debate_history: List[dict]  # Track all rounds
    debate_round: int
    max_rounds: int
    final_verdict: str


def create_pro_agent(llm):
    """Agent that argues FOR the topic."""

    def pro_agent(state: DebateState) -> dict:
        round_num = state.get('debate_round', 1)
        history = state.get('debate_history', [])

        # Build context from previous rounds
        context = ""
        if state.get('con_argument'):
            context = f"\n\nThe opposing argument in round {round_num - 1} was:\n{state['con_argument']}\n\nCounter their points and strengthen your position."

        # Show debate history for context
        history_text = ""
        if history:
            history_text = "\n\nPREVIOUS ROUNDS:\n"
            for h in history[-3:]:  # Last 3 rounds
                history_text += f"Round {h['round']} - PRO: {h['pro'][:100]}...\n"
                history_text += f"Round {h['round']} - CON: {h['con'][:100]}...\n"

        prompt = f"""You are arguing IN FAVOR of this position:

TOPIC: {state['topic']}

ROUND: {round_num} of {state.get('max_rounds', 5)}
{history_text}
{context}

Make your strongest argument FOR this position. Build on your previous points and counter the opposition."""

        response = llm.invoke([
            SystemMessage(content="You argue in favor of positions. Be persuasive, logical, and build on previous arguments."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[PRO - Round {round_num}]: {response.content}")],
            "pro_argument": response.content,
        }

    return pro_agent


def create_con_agent(llm):
    """Agent that argues AGAINST the topic."""

    def con_agent(state: DebateState) -> dict:
        round_num = state.get('debate_round', 1)
        pro_arg = state.get('pro_argument', '')
        history = state.get('debate_history', [])

        # Build context from previous rounds
        history_text = ""
        if history:
            history_text = "\n\nPREVIOUS ROUNDS:\n"
            for h in history[-3:]:  # Last 3 rounds
                history_text += f"Round {h['round']} - PRO: {h['pro'][:100]}...\n"
                history_text += f"Round {h['round']} - CON: {h['con'][:100]}...\n"

        prompt = f"""You are arguing AGAINST this position:

TOPIC: {state['topic']}

ROUND: {round_num} of {state.get('max_rounds', 5)}
{history_text}

The argument in favor (Round {round_num}) was:
{pro_arg}

Make your strongest argument AGAINST this position. Counter the pro arguments and build on your previous points."""

        response = llm.invoke([
            SystemMessage(content="You argue against positions. Be critical, logical, and build on previous arguments."),
            HumanMessage(content=prompt)
        ])

        # Add current round to history
        new_history = list(history)
        new_history.append({
            'round': round_num,
            'pro': pro_arg,
            'con': response.content
        })

        return {
            "messages": [AIMessage(content=f"[CON - Round {round_num}]: {response.content}")],
            "con_argument": response.content,
            "debate_history": new_history,
            "debate_round": round_num + 1,
        }

    return con_agent


def create_judge_agent(llm):
    """Judge agent that synthesizes the debate after all rounds."""

    def judge(state: DebateState) -> dict:
        history = state.get('debate_history', [])
        total_rounds = len(history)

        # Build full debate summary
        debate_summary = ""
        for h in history:
            debate_summary += f"\n--- ROUND {h['round']} ---\n"
            debate_summary += f"PRO: {h['pro']}\n"
            debate_summary += f"CON: {h['con']}\n"

        prompt = f"""You are the judge of this debate that lasted {total_rounds} rounds.

TOPIC: {state['topic']}

FULL DEBATE TRANSCRIPT:
{debate_summary}

After reviewing all {total_rounds} rounds of arguments, provide a balanced verdict that:
1. Summarizes the key arguments from both sides across all rounds
2. Notes how the arguments evolved during the debate
3. Identifies the most compelling points from each side
4. Gives a final recommendation with clear reasoning"""

        response = llm.invoke([
            SystemMessage(content="You are an impartial judge who synthesizes multi-round debates fairly and thoroughly."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[JUDGE - After {total_rounds} rounds]: {response.content}")],
            "final_verdict": response.content,
        }

    return judge


def route_debate(state: DebateState) -> Literal["pro", "con", "judge", "end"]:
    """Route in debate system - loops through pro/con for max_rounds before judge."""
    if state.get('final_verdict'):
        return "end"

    current_round = state.get('debate_round', 1)
    max_rounds = state.get('max_rounds', 5)

    # After max_rounds, go to judge for final verdict
    if current_round > max_rounds:
        return "judge"

    # Route based on current state within a round
    # Each round: pro speaks, then con speaks, then next round
    return "pro"  # Start next round with pro


def route_after_pro(state: DebateState) -> Literal["con"]:
    """After pro, always go to con."""
    return "con"


def route_after_con(state: DebateState) -> Literal["pro", "judge"]:
    """After con, either continue debate or go to judge."""
    current_round = state.get('debate_round', 1)
    max_rounds = state.get('max_rounds', 5)

    if current_round > max_rounds:
        return "judge"
    return "pro"  # Continue to next round


def build_debate_graph(llm):
    """Build debate multi-agent graph with multiple rounds."""
    workflow = StateGraph(DebateState)

    workflow.add_node("pro", create_pro_agent(llm))
    workflow.add_node("con", create_con_agent(llm))
    workflow.add_node("judge", create_judge_agent(llm))

    # Start with pro agent
    workflow.add_edge(START, "pro")

    # Pro always goes to con
    workflow.add_edge("pro", "con")

    # Con either loops back to pro (more rounds) or goes to judge (done)
    workflow.add_conditional_edges(
        "con",
        route_after_con,
        {"pro": "pro", "judge": "judge"}
    )

    # Judge ends the debate
    workflow.add_edge("judge", END)

    return workflow.compile()


def example_debate():
    """Run the debate pattern example with 5 rounds."""
    print("\n" + "="*60)
    print("Example: Debate Pattern (5 Rounds)")
    print("="*60)

    llm = create_llm()
    graph = build_debate_graph(llm)

    initial_state = {
        "messages": [],
        "topic": "Remote work should be the default for all office jobs",
        "pro_argument": "",
        "con_argument": "",
        "debate_history": [],  # Track all rounds
        "debate_round": 1,
        "max_rounds": 5,  # Debate for 5 rounds before judge
        "final_verdict": "",
    }

    print(f"\nDebate Topic: {initial_state['topic']}")
    print(f"Number of Rounds: {initial_state['max_rounds']}")
    print("-" * 40)

    result = graph.invoke(initial_state)

    print("\n📝 DEBATE LOG:")
    for msg in result["messages"]:
        content = msg.content
        # Truncate very long messages for display
        if len(content) > 1000:
            content = content[:1000] + "..."
        print(f"\n{content}\n")
        print("-" * 30)

    print(f"\n✅ DEBATE COMPLETE: {len(result.get('debate_history', []))} rounds")

    return result


# ============================================================================
# PATTERN 3: Reflection/Self-Critique Pattern
# ============================================================================

"""
REFLECTION PATTERN:
------------------
An agent generates output, then a critic agent reviews and suggests improvements.

Structure:
┌──────────┐     ┌──────────┐
│Generator │────►│ Critic   │
└────┬─────┘     └─────┬────┘
     ▲                 │
     └─────────────────┘
         (iterate)

Use when:
- Quality is critical
- Self-improvement needed
- Want to catch errors
"""


class ReflectionState(TypedDict):
    """State for reflection pattern."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    draft: str
    critique: str
    final_output: str
    iteration: int


def create_generator_agent(llm):
    """Agent that generates content."""

    def generator(state: ReflectionState) -> dict:
        critique = state.get('critique', '')

        if critique:
            # Revision based on critique
            prompt = f"""Revise your previous response based on this feedback:

TASK: {state['task']}

YOUR PREVIOUS DRAFT:
{state.get('draft', '')}

CRITIQUE:
{critique}

Improve your response addressing the feedback."""
        else:
            # Initial generation
            prompt = f"""Complete this task:

TASK: {state['task']}

Provide your best response."""

        response = llm.invoke([
            SystemMessage(content="You generate high-quality content and improve based on feedback."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[GENERATOR v{state.get('iteration', 1)}]: {response.content}")],
            "draft": response.content,
            "iteration": state.get('iteration', 1) + 1,
        }

    return generator


def create_critic_agent(llm):
    """Agent that critiques and suggests improvements."""

    def critic(state: ReflectionState) -> dict:
        # After enough iterations, approve
        if state.get('iteration', 1) >= 3:
            return {
                "messages": [AIMessage(content="[CRITIC]: Approved! Quality is satisfactory.")],
                "final_output": state.get('draft', ''),
                "critique": "",  # Clear critique to signal completion
            }

        prompt = f"""Review this draft and provide constructive criticism:

TASK: {state['task']}

DRAFT:
{state.get('draft', '')}

Provide specific, actionable feedback for improvement. If the draft is excellent, say "APPROVED"."""

        response = llm.invoke([
            SystemMessage(content="You are a constructive critic who helps improve content."),
            HumanMessage(content=prompt)
        ])

        # Check if approved
        if "APPROVED" in response.content.upper():
            return {
                "messages": [AIMessage(content=f"[CRITIC]: {response.content}")],
                "final_output": state.get('draft', ''),
                "critique": "",
            }

        return {
            "messages": [AIMessage(content=f"[CRITIC]: {response.content}")],
            "critique": response.content,
        }

    return critic


def route_reflection(state: ReflectionState) -> Literal["generator", "critic", "end"]:
    """Route in reflection system."""
    if state.get('final_output'):
        return "end"
    if state.get('draft') and not state.get('critique'):
        return "critic"
    return "generator"


def build_reflection_graph(llm):
    """Build reflection multi-agent graph."""
    workflow = StateGraph(ReflectionState)

    workflow.add_node("generator", create_generator_agent(llm))
    workflow.add_node("critic", create_critic_agent(llm))

    workflow.add_edge(START, "generator")

    workflow.add_conditional_edges(
        "generator",
        lambda s: "critic" if not s.get('final_output') else "end",
        {"critic": "critic", "end": END}
    )

    workflow.add_conditional_edges(
        "critic",
        route_reflection,
        {"generator": "generator", "critic": "critic", "end": END}
    )

    return workflow.compile()


def example_reflection():
    """Run the reflection pattern example."""
    print("\n" + "="*60)
    print("Example: Reflection Pattern")
    print("="*60)

    llm = create_llm()
    graph = build_reflection_graph(llm)

    initial_state = {
        "messages": [],
        "task": "Write a short poem about artificial intelligence",
        "draft": "",
        "critique": "",
        "final_output": "",
        "iteration": 1,
    }

    print(f"\nTask: {initial_state['task']}")
    print("-" * 40)

    result = graph.invoke(initial_state)

    print("\n📝 REFLECTION LOG:")
    for msg in result["messages"]:
        print(f"\n{msg.content[:400]}...")
        print("-" * 30)

    print("\n✅ FINAL OUTPUT:")
    print(result.get("final_output", "No output"))

    return result


# ============================================================================
# PATTERN 4: Human-in-the-Loop
# ============================================================================

"""
HUMAN-IN-THE-LOOP PATTERN:
-------------------------
Agents pause for human approval at critical points.

Structure:
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Agent A  │────►│  HUMAN   │────►│ Agent B  │
└──────────┘     │ APPROVAL │     └──────────┘
                 └──────────┘

Use when:
- Critical decisions need human oversight
- Safety-critical applications
- Learning from human feedback
"""


class HumanLoopState(TypedDict):
    """State for human-in-the-loop pattern."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task: str
    proposal: str
    human_approved: bool
    human_feedback: str
    final_result: str


def create_proposal_agent(llm):
    """Agent that creates proposals for human review."""

    def proposer(state: HumanLoopState) -> dict:
        feedback = state.get('human_feedback', '')

        if feedback:
            prompt = f"""Revise your proposal based on human feedback:

TASK: {state['task']}
PREVIOUS PROPOSAL: {state.get('proposal', '')}
HUMAN FEEDBACK: {feedback}

Create an improved proposal."""
        else:
            prompt = f"""Create a proposal for this task:

TASK: {state['task']}

Outline what you plan to do and ask for approval."""

        response = llm.invoke([
            SystemMessage(content="You create proposals for human approval."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[PROPOSER]: {response.content}")],
            "proposal": response.content,
        }

    return proposer


def human_approval_node(state: HumanLoopState) -> dict:
    """
    Simulated human approval node.
    In production, this would actually wait for human input.
    """
    # Simulate human approval (in real app, this would be interactive)
    print("\n" + "="*40)
    print("🙋 HUMAN APPROVAL REQUIRED")
    print("="*40)
    print(f"Proposal:\n{state.get('proposal', '')[:300]}...")

    # For demo, auto-approve
    # In production: input() or web interface
    approved = True
    feedback = "Looks good, proceed!"

    return {
        "messages": [HumanMessage(content=f"[HUMAN]: {feedback}")],
        "human_approved": approved,
        "human_feedback": feedback,
    }


def create_executor_agent(llm):
    """Agent that executes approved proposals."""

    def executor(state: HumanLoopState) -> dict:
        prompt = f"""Execute this approved proposal:

TASK: {state['task']}
APPROVED PROPOSAL: {state.get('proposal', '')}
HUMAN FEEDBACK: {state.get('human_feedback', '')}

Provide the final result."""

        response = llm.invoke([
            SystemMessage(content="You execute approved proposals thoroughly."),
            HumanMessage(content=prompt)
        ])

        return {
            "messages": [AIMessage(content=f"[EXECUTOR]: {response.content}")],
            "final_result": response.content,
        }

    return executor


# ============================================================================
# Summary of All Patterns
# ============================================================================

"""
PATTERN SUMMARY:
===============

1. SEQUENTIAL: A → B → C
   - Simple pipeline
   - Each agent builds on previous

2. SUPERVISED/ORCHESTRATED: Manager coordinates agents
   - Dynamic routing
   - Manager decides next agent

3. HIERARCHICAL: Manager → Specialists
   - Task decomposition
   - Parallel subtask execution

4. DEBATE: Pro ↔ Con → Judge
   - Multiple perspectives
   - Reduces bias

5. REFLECTION: Generator ↔ Critic
   - Self-improvement
   - Quality assurance

6. HUMAN-IN-THE-LOOP: Agent → Human → Agent
   - Safety critical
   - Human oversight

7. PARALLEL: Multiple agents simultaneously
   - Speed optimization
   - Independent subtasks

CHOOSING A PATTERN:
- Simple task? → Sequential
- Complex task? → Hierarchical
- Need quality? → Reflection
- Need perspectives? → Debate
- Need safety? → Human-in-the-Loop
- Need speed? → Parallel
"""


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TUTORIAL 4: Advanced Multi-Agent Patterns")
    print("="*60)

    # Uncomment to run examples:
    example_debate()
    #example_reflection()
