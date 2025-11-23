"""
Multi-Agent Communication Example using LangChain and LangGraph

This example demonstrates how to create multiple agents that can communicate
with each other using LangGraph's state management and message passing.

The scenario: A team of agents working together to analyze and respond to queries
- Researcher Agent: Gathers information and facts
- Analyst Agent: Analyzes the information provided by the researcher
- Writer Agent: Composes the final response based on analysis
"""

import operator
from typing import Annotated, Sequence, TypedDict, Literal, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


# ============================================================================
# LLM Configuration - Multiple options for Local LLMs
# ============================================================================

def get_local_llm(provider: str = "ollama", model_name: str = None):
    """
    Get a local LLM instance. Supports multiple providers.

    Options for DGX/Local GPU:
    1. Ollama - Easy to use, supports many models
    2. vLLM - High performance, OpenAI-compatible API
    3. HuggingFace - Direct model loading with transformers
    4. LlamaCpp - Efficient CPU/GPU inference
    5. TGI (Text Generation Inference) - HuggingFace's production server
    """

    if provider == "ollama":
        # Option 1: Ollama (easiest setup)
        # Install: curl -fsSL https://ollama.com/install.sh | sh
        # Run: ollama pull llama3.1 && ollama serve
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or "llama3.1",  # or "mistral", "codellama", "mixtral"
            base_url="http://localhost:11434",
            temperature=0.7,
        )

    elif provider == "vllm":
        # Option 2: vLLM (high performance for DGX)
        # Install: pip install vllm
        # Run: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "meta-llama/Llama-3.2-8B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="not-needed",  # vLLM doesn't require API key
            temperature=0.7,
        )

    elif provider == "lmstudio":
        # Option 3: LM Studio (GUI-based, OpenAI-compatible)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "local-model",
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            temperature=0.7,
        )

    elif provider == "huggingface":
        # Option 4: Direct HuggingFace loading (uses GPU directly)
        # Best for DGX with multiple GPUs
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        model_id = model_name or "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",  # Automatically use available GPUs
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
        )
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=hf_pipeline)

    elif provider == "llamacpp":
        # Option 5: LlamaCpp (efficient, supports GGUF models)
        # Install: pip install llama-cpp-python
        from langchain_community.llms import LlamaCpp
        from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path=model_name or "/path/to/model.gguf",
            n_gpu_layers=-1,  # Use all GPU layers
            n_ctx=4096,
            callback_manager=callback_manager,
            verbose=True,
        )

    elif provider == "tgi":
        # Option 6: Text Generation Inference (HuggingFace's production server)
        # Run: docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference:latest --model-id meta-llama/Llama-3.1-8B-Instruct
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            endpoint_url="http://localhost:8080",
            task="text-generation",
            max_new_tokens=1024,
        )

    elif provider == "openai":
        # Fallback: OpenAI API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=0.7,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# State Definition - Shared state for inter-agent communication
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state that enables communication between agents.

    - messages: The conversation history (accumulates across agents)
    - research_notes: Information gathered by the researcher agent
    - analysis: Analysis produced by the analyst agent
    - final_response: The composed response from the writer agent
    - next_agent: Determines which agent should act next
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_notes: str
    analysis: str
    final_response: str
    next_agent: str
    task: str


# ============================================================================
# Tools that agents can use
# ============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    # Simulated knowledge base search
    knowledge = {
        "python": "Python is a high-level programming language known for readability and versatility.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
    }

    results = []
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            results.append(value)

    return " ".join(results) if results else "No specific information found in knowledge base."


@tool
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment and key themes in the provided text."""
    # Simulated sentiment analysis
    positive_words = ["good", "great", "excellent", "powerful", "versatile", "efficient"]
    negative_words = ["bad", "poor", "difficult", "complex", "confusing"]

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return "Sentiment: Positive. The content has an optimistic and favorable tone."
    elif negative_count > positive_count:
        return "Sentiment: Negative. The content expresses concerns or criticism."
    else:
        return "Sentiment: Neutral. The content is balanced and informative."


# ============================================================================
# Agent Definitions
# ============================================================================

def create_researcher_agent(llm: BaseChatModel):
    """Create the researcher agent that gathers information."""

    def researcher_node(state: AgentState) -> dict:
        """
        Researcher Agent: Gathers relevant information for the task.
        Communicates findings via the shared state.
        """
        system_prompt = """You are a Research Agent. Your role is to:
1. Understand the user's query
2. Gather relevant information and facts
3. Provide comprehensive research notes for the analyst

Be thorough but concise. Focus on factual information."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research the following topic: {state['task']}")
        ]

        # Use the search tool to gather information
        search_result = search_knowledge_base.invoke({"query": state['task']})

        # Generate research notes
        research_prompt = f"""Based on my search, I found: {search_result}

Please compile comprehensive research notes about: {state['task']}
Include any relevant context and background information."""

        messages.append(HumanMessage(content=research_prompt))
        response = llm.invoke(messages)

        # Communicate findings to other agents via state
        return {
            "messages": [AIMessage(content=f"[RESEARCHER]: {response.content}")],
            "research_notes": response.content,
            "next_agent": "analyst"
        }

    return researcher_node


def create_analyst_agent(llm: BaseChatModel):
    """Create the analyst agent that analyzes information."""

    def analyst_node(state: AgentState) -> dict:
        """
        Analyst Agent: Analyzes the research and provides insights.
        Receives information from researcher, passes analysis to writer.
        """
        system_prompt = """You are an Analyst Agent. Your role is to:
1. Review the research notes provided by the Research Agent
2. Identify key insights, patterns, and important points
3. Provide a structured analysis for the Writer Agent

Focus on extracting actionable insights and organizing information logically."""

        # Get sentiment analysis
        sentiment = analyze_sentiment.invoke({"text": state['research_notes']})

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Analyze the following research notes:

RESEARCH NOTES:
{state['research_notes']}

SENTIMENT ANALYSIS:
{sentiment}

ORIGINAL TASK:
{state['task']}

Provide a structured analysis with key insights and recommendations.""")
        ]

        response = llm.invoke(messages)

        return {
            "messages": [AIMessage(content=f"[ANALYST]: {response.content}")],
            "analysis": response.content,
            "next_agent": "writer"
        }

    return analyst_node


def create_writer_agent(llm: BaseChatModel):
    """Create the writer agent that composes the final response."""

    def writer_node(state: AgentState) -> dict:
        """
        Writer Agent: Composes the final response based on research and analysis.
        This agent synthesizes all information from previous agents.
        """
        system_prompt = """You are a Writer Agent. Your role is to:
1. Review the research notes and analysis from your colleague agents
2. Compose a clear, well-structured final response
3. Ensure the response directly addresses the user's original query

Write in a professional, engaging manner. Be concise but comprehensive."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Compose a final response based on the following:

ORIGINAL QUERY:
{state['task']}

RESEARCH NOTES (from Research Agent):
{state['research_notes']}

ANALYSIS (from Analyst Agent):
{state['analysis']}

Write a comprehensive yet concise response that addresses the user's query.""")
        ]

        response = llm.invoke(messages)

        return {
            "messages": [AIMessage(content=f"[WRITER]: {response.content}")],
            "final_response": response.content,
            "next_agent": "end"
        }

    return writer_node


# ============================================================================
# Supervisor Agent - Orchestrates communication between agents
# ============================================================================

def create_supervisor(llm: BaseChatModel):
    """Create a supervisor that decides which agent should act next."""

    def supervisor_node(state: AgentState) -> dict:
        """
        Supervisor: Orchestrates the flow between agents.
        Decides routing based on current state and task progress.
        """
        # Simple rule-based routing for this example
        # In a more complex scenario, this could use LLM for dynamic routing

        if not state.get('research_notes'):
            return {"next_agent": "researcher"}
        elif not state.get('analysis'):
            return {"next_agent": "analyst"}
        elif not state.get('final_response'):
            return {"next_agent": "writer"}
        else:
            return {"next_agent": "end"}

    return supervisor_node


# ============================================================================
# Router Function - Determines next step in the graph
# ============================================================================

def route_next_agent(state: AgentState) -> Literal["researcher", "analyst", "writer", "end"]:
    """Route to the next agent based on state."""
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "end":
        return "end"
    return next_agent


# ============================================================================
# Build the Multi-Agent Graph
# ============================================================================

def build_multi_agent_graph(llm: BaseChatModel) -> StateGraph:
    """
    Build the LangGraph workflow for multi-agent communication.

    Graph Structure:
    START -> supervisor -> researcher -> supervisor -> analyst -> supervisor -> writer -> END

    Communication Flow:
    1. Supervisor decides which agent should act
    2. Researcher gathers information, stores in shared state
    3. Analyst reads research from state, adds analysis
    4. Writer reads all information, produces final response
    """

    # Create the graph with our state schema
    workflow = StateGraph(AgentState)

    # Create agent nodes
    researcher = create_researcher_agent(llm)
    analyst = create_analyst_agent(llm)
    writer = create_writer_agent(llm)
    supervisor = create_supervisor(llm)

    # Add nodes to the graph
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("writer", writer)

    # Define the edges (communication paths)
    workflow.add_edge(START, "supervisor")

    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_next_agent,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "end": END
        }
    )

    # After each agent completes, go back to supervisor for routing
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")

    return workflow.compile()


# ============================================================================
# Alternative: Direct Agent-to-Agent Communication Pattern
# ============================================================================

def build_direct_communication_graph(llm: BaseChatModel) -> StateGraph:
    """
    Alternative pattern: Direct agent-to-agent communication without supervisor.

    Graph Structure:
    START -> researcher -> analyst -> writer -> END

    This is a simpler linear flow where agents communicate directly
    through the shared state without a supervisor.
    """

    workflow = StateGraph(AgentState)

    # Create agent nodes
    researcher = create_researcher_agent(llm)
    analyst = create_analyst_agent(llm)
    writer = create_writer_agent(llm)

    # Add nodes
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("writer", writer)

    # Linear communication flow
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()


# ============================================================================
# Message Broker Pattern - For more complex communication
# ============================================================================

class MessageBroker:
    """
    A message broker for managing inter-agent communication.
    Useful for more complex scenarios with async communication.
    """

    def __init__(self):
        self.message_queue: dict[str, list] = {}
        self.subscribers: dict[str, list] = {}

    def subscribe(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(agent_id)

    def publish(self, topic: str, message: dict):
        """Publish a message to a topic."""
        if topic not in self.message_queue:
            self.message_queue[topic] = []
        self.message_queue[topic].append(message)

    def get_messages(self, agent_id: str, topic: str) -> list:
        """Get messages for an agent from a topic."""
        return self.message_queue.get(topic, [])


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Run the multi-agent communication example."""

    # =========================================================================
    # Choose your local LLM provider - uncomment the one you want to use
    # =========================================================================

    # Option 1: Ollama (easiest - recommended for quick setup)
    # First run: ollama pull llama3.2 && ollama serve
    llm = get_local_llm(provider="ollama", model_name="llama3.2")

    # Option 2: vLLM (best performance on DGX)
    # First run: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4
    # llm = get_local_llm(provider="vllm", model_name="meta-llama/Llama-3.1-8B-Instruct")

    # Option 3: Direct HuggingFace (loads model directly on GPU)
    # llm = get_local_llm(provider="huggingface", model_name="meta-llama/Llama-3.1-8B-Instruct")

    # Option 4: TGI (HuggingFace Text Generation Inference)
    # First run: docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference:latest --model-id meta-llama/Llama-3.1-8B-Instruct
    # llm = get_local_llm(provider="tgi")

    # Option 5: OpenAI API (fallback)
    # llm = get_local_llm(provider="openai", model_name="gpt-4o-mini")

    print("=" * 60)
    print("Multi-Agent Communication Example")
    print("=" * 60)

    # Build the graph with supervisor pattern
    print("\n1. Building multi-agent graph with supervisor pattern...")
    graph = build_multi_agent_graph(llm)

    # Initial state
    initial_state: AgentState = {
        "messages": [],
        "research_notes": "",
        "analysis": "",
        "final_response": "",
        "next_agent": "",
        "task": "Explain how LangChain and LangGraph work together for building AI applications"
    }

    print("\n2. Starting agent communication...")
    print(f"   Task: {initial_state['task']}")
    print("-" * 60)

    # Run the graph
    result = graph.invoke(initial_state)

    # Display the communication flow
    print("\n3. Agent Communication Log:")
    print("-" * 60)
    for msg in result["messages"]:
        print(f"\n{msg.content}\n")

    print("=" * 60)
    print("FINAL RESPONSE:")
    print("=" * 60)
    print(result["final_response"])

    # Demonstrate direct communication pattern
    print("\n" + "=" * 60)
    print("Alternative: Direct Communication Pattern")
    print("=" * 60)

    direct_graph = build_direct_communication_graph(llm)

    direct_state: AgentState = {
        "messages": [],
        "research_notes": "",
        "analysis": "",
        "final_response": "",
        "next_agent": "",
        "task": "What is machine learning and how is it used in modern applications?"
    }

    print(f"\nTask: {direct_state['task']}")
    print("-" * 60)

    direct_result = direct_graph.invoke(direct_state)

    print("\nAgent Communication Log:")
    for msg in direct_result["messages"]:
        print(f"\n{msg.content}\n")


if __name__ == "__main__":
    main()
