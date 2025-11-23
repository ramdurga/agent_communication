"""
================================================================================
TUTORIAL 2: Agents with Tools
================================================================================

Tools transform a simple chatbot into a powerful agent that can:
- Search the web
- Execute code
- Query databases
- Call APIs
- Interact with files
- And much more...

This tutorial covers:
1. What are Tools?
2. Creating custom tools
3. Binding tools to agents
4. Tool execution and observation
5. ReAct agent with tools
"""

from typing import Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, Tool
from langchain_core.prompts import ChatPromptTemplate


def create_llm():
    """Create LLM instance."""
    return ChatOllama(
        model="llama3.2",
        base_url="http://localhost:11434",
        temperature=0,  # Lower temperature for more deterministic tool use
    )


# ============================================================================
# PART 1: Creating Tools
# ============================================================================

# Method 1: Using the @tool decorator (Recommended)
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"

    Returns:
        The result of the calculation
    """
    try:
        # Safety: only allow basic math operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Args:
        city: Name of the city to get weather for

    Returns:
        Weather information for the city
    """
    # Simulated weather data (in real app, call weather API)
    weather_data = {
        "new york": "72°F, Partly Cloudy",
        "london": "59°F, Rainy",
        "tokyo": "68°F, Clear",
        "paris": "65°F, Cloudy",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}"


@tool
def search_knowledge(query: str) -> str:
    """
    Search the knowledge base for information.

    Args:
        query: The search query

    Returns:
        Relevant information from the knowledge base
    """
    # Simulated knowledge base
    knowledge = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "langchain": "LangChain is a framework for building applications with large language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-agent applications.",
        "agent": "An AI agent is an autonomous system that can perceive, reason, and act.",
    }

    results = []
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            results.append(value)

    return " ".join(results) if results else "No relevant information found."


# Method 2: Creating Tool from function
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


time_tool = Tool(
    name="get_current_time",
    description="Get the current date and time",
    func=get_current_time,
)


# ============================================================================
# PART 2: Binding Tools to LLM
# ============================================================================

def example_tool_binding():
    """
    Demonstrate binding tools to an LLM.

    When tools are bound, the LLM knows:
    1. What tools are available
    2. What each tool does (from docstrings)
    3. What arguments each tool needs
    4. When to use each tool
    """
    print("\n" + "="*60)
    print("Example: Tool Binding")
    print("="*60)

    llm = create_llm()

    # Define available tools
    tools = [calculator, get_weather, search_knowledge]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Now the LLM can decide to use tools
    messages = [
        SystemMessage(content="You are a helpful assistant with access to tools."),
        HumanMessage(content="What's 15 * 7?")
    ]

    response = llm_with_tools.invoke(messages)

    print(f"\nLLM Response: {response}")
    print(f"\nTool Calls: {response.tool_calls}")

    return response


# ============================================================================
# PART 3: Manual Tool Execution Loop
# ============================================================================

def example_manual_tool_loop():
    """
    Demonstrate manual tool execution - the core of how agents work.

    The loop:
    1. LLM receives input
    2. LLM decides which tool to call (or responds directly)
    3. We execute the tool
    4. We send the result back to LLM
    5. LLM formulates final response
    """
    print("\n" + "="*60)
    print("Example: Manual Tool Execution Loop")
    print("="*60)

    llm = create_llm()
    tools = [calculator, get_weather, search_knowledge]
    tools_dict = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    # Initial user request
    messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed."),
        HumanMessage(content="What's the weather in Tokyo and what is 100 / 4?")
    ]

    print(f"\nUser: {messages[-1].content}")

    # Agent loop
    max_iterations = 5
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")

        # Get LLM response
        print(f"Input message : {messages}")
        response = llm_with_tools.invoke(messages)
        print(f"LLM Response : {response}")
        messages.append(response)

        # Check if LLM wants to use tools
        if not response.tool_calls:
            print(f"\nFinal Response: {response.content}")
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\nTool Call: {tool_name}({tool_args})")

            # Execute the tool
            if tool_name in tools_dict:
                result = tools_dict[tool_name].invoke(tool_args)
                print(f"Tool Result: {result}")

                # Add tool result to messages
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

    return messages


# ============================================================================
# PART 4: Agent Class with Tools
# ============================================================================

class ToolAgent:
    """
    A complete agent implementation with tool support.

    Features:
    - Tool binding and execution
    - Conversation memory
    - Automatic tool loop
    - Error handling
    """

    def __init__(self, llm, tools: list, system_prompt: str = None):
        self.tools = tools
        self.tools_dict = {t.name: t for t in tools}
        self.llm = llm.bind_tools(tools)
        self.system_prompt = system_prompt or "You are a helpful assistant with access to tools."
        self.memory: list = []

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        """
        Run the agent with user input.

        Args:
            user_input: The user's request
            max_iterations: Maximum tool execution loops

        Returns:
            The agent's final response
        """
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            *self.memory,
            HumanMessage(content=user_input)
        ]

        # Add user message to memory
        self.memory.append(HumanMessage(content=user_input))

        # Agent loop
        for _ in range(max_iterations):
            response = self.llm.invoke(messages)
            print(f"LLM Response : {response}")
            messages.append(response)

            # No tool calls - we have the final answer
            if not response.tool_calls:
                self.memory.append(AIMessage(content=response.content))
                return response.content

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Execute tool
                try:
                    if tool_name in self.tools_dict:
                        result = self.tools_dict[tool_name].invoke(tool_args)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'"
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"

                # Add result to messages
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

        return "Max iterations reached without final response."


def example_tool_agent():
    """Demonstrate the ToolAgent class."""
    print("\n" + "="*60)
    print("Example: Complete Tool Agent")
    print("="*60)

    llm = create_llm()
    tools = [calculator, get_weather, search_knowledge]

    agent = ToolAgent(
        llm=llm,
        tools=tools,
        system_prompt="""You are a helpful assistant with access to these tools:
- calculator: For math calculations
- get_weather: For weather information
- search_knowledge: For searching information

Always use tools when they would help answer the question accurately."""
    )

    # Test queries
    queries = [
        "What is 25 * 4 + 10?",
        "What's the weather like in London?",
        "Tell me about LangChain",
    ]

    for query in queries:
        print(f"\n{'='*40}")
        print(f"User: {query}")
        response = agent.run(query)
        print(f"Agent: {response[:500]}")


# ============================================================================
# PART 5: Understanding Tool Schemas
# ============================================================================

def example_tool_schemas():
    """
    Show how tools are represented as schemas for the LLM.

    The LLM sees tools as structured schemas, not Python functions.
    This is how it knows what tools are available and how to use them.
    """
    print("\n" + "="*60)
    print("Example: Tool Schemas")
    print("="*60)

    tools = [calculator, get_weather, search_knowledge]

    for t in tools:
        print(f"\nTool: {t.name}")
        print(f"Description: {t.description}")
        print(f"Args Schema: {t.args_schema.schema() if t.args_schema else 'None'}")


# ============================================================================
# Key Concepts Summary
# ============================================================================

"""
KEY TAKEAWAYS:
--------------

1. TOOLS ARE THE AGENT'S HANDS:
   - Without tools: Agent can only generate text
   - With tools: Agent can interact with the world

2. TOOL ANATOMY:
   - Name: Identifier for the tool
   - Description: Tells LLM when to use it (VERY IMPORTANT!)
   - Arguments: What inputs the tool needs
   - Function: The actual code that runs

3. TOOL EXECUTION LOOP:
   a. User sends request
   b. LLM decides: respond directly OR use tool
   c. If tool: execute and send result back
   d. LLM formulates response with tool result
   e. Repeat until done

4. TOOL BINDING:
   - llm.bind_tools(tools) makes LLM aware of available tools
   - LLM returns tool_calls when it wants to use a tool
   - We must execute tools and send results back

5. BEST PRACTICES:
   - Write clear tool descriptions (LLM reads these!)
   - Handle tool errors gracefully
   - Limit max iterations to prevent infinite loops
   - Validate tool inputs

NEXT: Tutorial 3 - Multi-Agent Communication
"""


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TUTORIAL 2: Agents with Tools")
    print("="*60)

    # Show tool schemas first
    example_tool_schemas()

    # Uncomment to run other examples:
    example_tool_binding()
    example_manual_tool_loop()
    #example_tool_agent()
