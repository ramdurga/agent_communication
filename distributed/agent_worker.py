"""
================================================================================
Distributed Agent Worker
================================================================================

This is an agent that can run on ANY machine and communicate with other agents
via Redis. Each agent:
1. Connects to Redis
2. Listens for tasks assigned to its type
3. Processes tasks and publishes results
4. Other agents pick up from where this one left off

Run on different machines:
    python agent_worker.py --type researcher --redis-host 192.168.1.100
    python agent_worker.py --type analyst --redis-host 192.168.1.100
    python agent_worker.py --type writer --redis-host 192.168.1.100
"""

import argparse
import uuid
import time
import signal
import sys
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from shared_state import DistributedState, create_distributed_state, TaskState


class DistributedAgent:
    """
    A distributed agent that runs on any machine.

    It connects to Redis, claims tasks, processes them, and publishes results
    for other agents to continue the workflow.
    """

    def __init__(
        self,
        agent_type: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        llm_host: str = "localhost",
        llm_port: int = 11434,
        model_name: str = "llama3.2"
    ):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"

        # Connect to Redis
        self.state = create_distributed_state(redis_host, redis_port)

        # Create LLM connection
        self.llm = ChatOllama(
            model=model_name,
            base_url=f"http://{llm_host}:{llm_port}",
            temperature=0.7
        )

        # Agent-specific configuration
        self.config = self._get_agent_config()

        # Running flag
        self.running = False

        print(f"🤖 Agent initialized: {self.agent_id}")
        print(f"   Type: {self.agent_type}")
        print(f"   Redis: {redis_host}:{redis_port}")
        print(f"   LLM: {llm_host}:{llm_port}/{model_name}")

    def _get_agent_config(self) -> dict:
        """Get configuration based on agent type."""
        configs = {
            "researcher": {
                "system_prompt": """You are a Research Agent. Your role is to:
1. Understand the topic given to you
2. Gather comprehensive information and facts
3. Provide detailed research notes

Be thorough and factual. Include key definitions, context, and important details.""",
                "next_agent": "analyst",
                "output_field": "research_notes"
            },
            "analyst": {
                "system_prompt": """You are an Analyst Agent. Your role is to:
1. Review the research notes provided
2. Identify key insights and patterns
3. Provide structured analysis

Focus on extracting actionable insights and organizing information logically.""",
                "next_agent": "writer",
                "output_field": "analysis",
                "input_field": "research_notes"
            },
            "writer": {
                "system_prompt": """You are a Writer Agent. Your role is to:
1. Review all research and analysis
2. Compose a clear, well-structured final response
3. Ensure the response addresses the original task

Write in a professional, engaging manner. Be concise but comprehensive.""",
                "next_agent": None,  # End of pipeline
                "output_field": "final_response",
                "input_fields": ["research_notes", "analysis"]
            }
        }
        return configs.get(self.agent_type, {})

    def process_task(self, task_state: TaskState) -> str:
        """Process a task and return the result."""
        config = self.config

        # Build the prompt based on agent type
        if self.agent_type == "researcher":
            prompt = f"Research this topic thoroughly:\n\n{task_state.task}"

        elif self.agent_type == "analyst":
            prompt = f"""Analyze the following research:

ORIGINAL TASK: {task_state.task}

RESEARCH NOTES:
{task_state.research_notes}

Provide a structured analysis with key insights."""

        elif self.agent_type == "writer":
            prompt = f"""Create a final response based on:

ORIGINAL TASK: {task_state.task}

RESEARCH NOTES:
{task_state.research_notes}

ANALYSIS:
{task_state.analysis}

Write a comprehensive, well-structured response."""

        else:
            prompt = f"Process this task: {task_state.task}"

        # Call LLM
        messages = [
            SystemMessage(content=config.get("system_prompt", "You are a helpful assistant.")),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    def handle_task(self, task_id: str) -> None:
        """Handle a single task from the queue."""
        print(f"\n📋 Processing task: {task_id}")

        # Get task state
        task_state = self.state.get_task_state(task_id)
        if not task_state:
            print(f"❌ Task {task_id} not found")
            return

        # Update status
        self.state.update_task_state(
            task_id,
            status="in_progress",
            current_agent=self.agent_id
        )

        try:
            # Process the task
            print(f"⚙️  {self.agent_type} processing...")
            result = self.process_task(task_state)

            # Save result to appropriate field
            output_field = self.config.get("output_field", "result")
            updates = {output_field: result}

            # Add message to history
            self.state.add_message(
                task_id,
                self.agent_id,
                f"[{self.agent_type.upper()}]: {result[:200]}..."
            )

            # Check if there's a next agent
            next_agent = self.config.get("next_agent")

            if next_agent:
                # Pass to next agent
                updates["status"] = "in_progress"
                updates["current_agent"] = next_agent
                self.state.update_task_state(task_id, **updates)

                # Queue task for next agent
                self.state.enqueue_task(task_id, next_agent)
                print(f"✅ Passed to {next_agent}")
            else:
                # Task complete
                updates["status"] = "completed"
                self.state.update_task_state(task_id, **updates)
                self.state.enqueue_result(task_id)
                print(f"✅ Task completed!")

        except Exception as e:
            print(f"❌ Error processing task: {e}")
            self.state.update_task_state(
                task_id,
                status="failed",
                current_agent=self.agent_id
            )
            self.state.add_message(task_id, self.agent_id, f"ERROR: {str(e)}")

    def run(self) -> None:
        """Main loop - listen for and process tasks."""
        print(f"\n🚀 Starting agent {self.agent_id}")
        print(f"   Waiting for tasks of type: {self.agent_type}")
        print("   Press Ctrl+C to stop\n")

        # Register agent
        self.state.register_agent(self.agent_id, self.agent_type)

        self.running = True

        try:
            while self.running:
                # Update heartbeat
                self.state.heartbeat(self.agent_id)

                # Wait for a task (blocking with timeout)
                task_id = self.state.dequeue_task(self.agent_type, timeout=5)

                if task_id:
                    self.handle_task(task_id)
                else:
                    # No task, just continue waiting
                    print(".", end="", flush=True)

        except KeyboardInterrupt:
            print("\n\n⚠️  Shutting down...")

        finally:
            self.state.unregister_agent(self.agent_id, self.agent_type)
            print(f"👋 Agent {self.agent_id} stopped")

    def stop(self) -> None:
        """Stop the agent."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Run a distributed agent worker")
    parser.add_argument(
        "--type", "-t",
        required=True,
        choices=["researcher", "analyst", "writer"],
        help="Type of agent to run"
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis server host"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis server port"
    )
    parser.add_argument(
        "--llm-host",
        default="localhost",
        help="Ollama LLM host"
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=11434,
        help="Ollama LLM port"
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="LLM model name"
    )

    args = parser.parse_args()

    # Create and run agent
    agent = DistributedAgent(
        agent_type=args.type,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        llm_host=args.llm_host,
        llm_port=args.llm_port,
        model_name=args.model
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        agent.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the agent
    agent.run()


if __name__ == "__main__":
    main()
