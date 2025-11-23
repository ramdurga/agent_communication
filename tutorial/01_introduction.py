"""
================================================================================
TUTORIAL 1: Introduction to AI Agents
================================================================================

What is an AI Agent?
--------------------
An AI Agent is an autonomous software entity that:
1. Perceives its environment (receives inputs/context)
2. Makes decisions using an LLM (reasoning)
3. Takes actions (uses tools, generates outputs)
4. Learns from feedback (iterates based on results)

Key Components of an Agent:
---------------------------
1. LLM (Brain) - The language model that powers reasoning
2. Tools - Functions the agent can call to interact with the world
3. Memory - State that persists across interactions
4. Prompt/Instructions - Define the agent's behavior and goals

This tutorial series covers:
- Part 1: Introduction & Basic Agent (this file)
- Part 2: Agent with Tools
- Part 3: Multi-Agent Communication Patterns
- Part 4: Supervisor/Orchestrator Pattern
- Part 5: Advanced Patterns (Hierarchical, Debate, etc.)
"""

# ============================================================================
# PART 1: Creating a Basic Agent with LangChain
# ============================================================================

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_basic_llm():
    """
    Create a basic LLM instance using Ollama (local).

    You can swap this with any LangChain-compatible LLM:
    - ChatOpenAI (OpenAI API)
    - ChatAnthropic (Claude API)
    - ChatOllama (Local via Ollama)
    - And many more...
    """
    return ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=0.7,
    )


# ============================================================================
# Example 1: Simple Chat - Not yet an "Agent"
# ============================================================================

def example_simple_chat():
    """
    A simple chat is NOT an agent - it's just a single LLM call.
    There's no autonomy, no tools, no decision-making loop.
    """
    print("\n" + "="*60)
    print("Example 1: Simple Chat (Not an Agent)")
    print("="*60)

    llm = create_basic_llm()

    # Direct LLM call - no agency here
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is Python?")
    ]

    response = llm.invoke(messages)
    print(f"\nResponse: {response.content}")

    return response


# ============================================================================
# Example 2: Basic Agent with a Loop (ReAct Pattern)
# ============================================================================

def example_basic_agent():
    """
    A basic agent implements the ReAct (Reason + Act) pattern:
    1. Receive input
    2. Think about what to do
    3. Take an action (or respond)
    4. Observe the result
    5. Repeat until done

    This is the foundation of all agent architectures.
    """
    print("\n" + "="*60)
    print("Example 2: Basic Agent with Reasoning Loop")
    print("="*60)

    llm = create_basic_llm()

    # Agent's system prompt - defines its behavior
    system_prompt = """You are a helpful AI assistant that thinks step by step.

When given a task, you should:
1. THINK: Analyze what needs to be done
2. PLAN: Break down the task into steps
3. RESPOND: Provide your answer

Format your response as:
THINKING: [Your analysis]
PLAN: [Your step-by-step plan]
ANSWER: [Your final response]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Explain how to learn Python programming effectively.")
    ]

    response = llm.invoke(messages)
    print(f"\nAgent Response:\n{response.content}")

    return response


# ============================================================================
# Example 3: Agent with Memory (Conversation History)
# ============================================================================

class SimpleAgent:
    """
    A simple agent with memory - remembers conversation history.

    This is still basic but demonstrates key agent concepts:
    - State management (memory)
    - Persistent identity (system prompt)
    - Multi-turn interactions
    """

    def __init__(self, llm, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory: list = []  # Conversation history

    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        # Add user message to memory
        self.memory.append(HumanMessage(content=user_input))

        # Construct full message list
        messages = [
            SystemMessage(content=self.system_prompt),
            *self.memory
        ]

        # Get LLM response
        response = self.llm.invoke(messages)

        # Add assistant response to memory
        self.memory.append(AIMessage(content=response.content))

        return response.content

    def clear_memory(self):
        """Reset conversation history."""
        self.memory = []


def example_agent_with_memory():
    """Demonstrate an agent with conversation memory."""
    print("\n" + "="*60)
    print("Example 3: Agent with Memory")
    print("="*60)

    llm = create_basic_llm()

    agent = SimpleAgent(
        llm=llm,
        system_prompt="You are a Python tutor. Help users learn Python step by step. Remember what they've asked before."
    )

    # Multi-turn conversation
    conversations = [
        "What are variables in Python?",
        "Can you give me an example?",
        "What did we discuss first?"  # Tests memory
    ]

    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response[:500]}...")  # Truncate for readability


# ============================================================================
# Key Concepts Summary
# ============================================================================

"""
KEY TAKEAWAYS:
--------------

1. AGENT vs CHATBOT:
   - Chatbot: Single request-response, no autonomy
   - Agent: Can reason, plan, use tools, maintain state

2. CORE AGENT COMPONENTS:
   - LLM: The "brain" that does reasoning
   - System Prompt: Defines agent's persona and behavior
   - Memory: Stores conversation history and state
   - Tools: (covered in next tutorial)

3. ReAct PATTERN:
   - Reason: Think about the problem
   - Act: Take an action or provide response
   - Observe: See the results
   - Repeat: Continue until task is complete

4. MEMORY TYPES:
   - Short-term: Conversation history (what we implemented)
   - Long-term: Persistent storage (databases, vector stores)
   - Working memory: Current task context

NEXT: Tutorial 2 - Adding Tools to Agents
"""


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TUTORIAL 1: Introduction to AI Agents")
    print("="*60)

    # Uncomment to run examples:
    # example_simple_chat()
    # example_basic_agent()
    example_agent_with_memory()
