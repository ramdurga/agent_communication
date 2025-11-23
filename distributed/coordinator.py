"""
================================================================================
Distributed Agent Coordinator
================================================================================

The coordinator is responsible for:
1. Accepting tasks from users/clients
2. Creating task state in Redis
3. Dispatching tasks to the first agent
4. Collecting final results
5. Monitoring overall progress

Run on the main/coordinator machine:
    python coordinator.py --redis-host localhost
"""

import argparse
import uuid
import time
from typing import Optional, Dict, List

from shared_state import DistributedState, create_distributed_state, TaskState


class AgentCoordinator:
    """
    Coordinates distributed agents across multiple machines.

    The coordinator:
    - Manages the overall workflow
    - Dispatches tasks to the first agent
    - Monitors progress
    - Collects final results
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.state = create_distributed_state(redis_host, redis_port)
        self.active_tasks: Dict[str, TaskState] = {}

        print(f"🎯 Coordinator initialized")
        print(f"   Redis: {redis_host}:{redis_port}")

    def check_agents(self) -> Dict[str, List[str]]:
        """Check which agents are available."""
        agent_types = ["researcher", "analyst", "writer"]
        available = {}

        for agent_type in agent_types:
            agents = self.state.get_available_agents(agent_type)
            available[agent_type] = agents
            status = "✓" if agents else "✗"
            print(f"   {status} {agent_type}: {len(agents)} agent(s)")

        return available

    def submit_task(self, task: str, task_id: str = None) -> str:
        """
        Submit a new task for processing.

        Returns the task_id for tracking.
        """
        if task_id is None:
            task_id = f"task-{uuid.uuid4().hex[:8]}"

        print(f"\n📝 Submitting task: {task_id}")
        print(f"   Task: {task[:100]}...")

        # Create task in Redis
        task_state = self.state.create_task(task_id, task)
        self.active_tasks[task_id] = task_state

        # Dispatch to first agent (researcher)
        self.state.enqueue_task(task_id, "researcher")
        print(f"   ✓ Dispatched to researcher agent")

        return task_id

    def get_task_status(self, task_id: str) -> Optional[TaskState]:
        """Get the current status of a task."""
        return self.state.get_task_state(task_id)

    def wait_for_result(self, task_id: str, timeout: int = 300) -> Optional[TaskState]:
        """
        Wait for a task to complete.

        Args:
            task_id: The task to wait for
            timeout: Maximum seconds to wait

        Returns:
            The completed TaskState or None if timeout
        """
        print(f"\n⏳ Waiting for task {task_id} to complete...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            task_state = self.state.get_task_state(task_id)

            if task_state:
                if task_state.status == "completed":
                    print(f"✅ Task completed!")
                    return task_state

                if task_state.status == "failed":
                    print(f"❌ Task failed!")
                    return task_state

                # Show progress
                current = task_state.current_agent or "waiting"
                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed}s] Status: {task_state.status}, Agent: {current}")

            time.sleep(2)

        print(f"⚠️  Timeout waiting for task")
        return None

    def get_result(self, task_id: str) -> Optional[str]:
        """Get the final result of a completed task."""
        task_state = self.state.get_task_state(task_id)
        if task_state and task_state.status == "completed":
            return task_state.final_response
        return None

    def get_all_messages(self, task_id: str) -> List[Dict]:
        """Get all messages from the task's history."""
        task_state = self.state.get_task_state(task_id)
        if task_state:
            return task_state.messages
        return []

    def run_interactive(self):
        """Run interactive mode for testing."""
        print("\n" + "="*60)
        print("🎯 Distributed Agent Coordinator - Interactive Mode")
        print("="*60)

        # Check available agents
        print("\n📡 Checking available agents...")
        available = self.check_agents()

        all_available = all(len(agents) > 0 for agents in available.values())
        if not all_available:
            print("\n⚠️  Warning: Not all agent types are available!")
            print("   Start agents with: python agent_worker.py --type <type>")

        print("\nCommands:")
        print("  task <text>  - Submit a new task")
        print("  status <id>  - Check task status")
        print("  result <id>  - Get task result")
        print("  agents       - Check available agents")
        print("  quit         - Exit")

        while True:
            try:
                cmd = input("\n> ").strip()

                if not cmd:
                    continue

                parts = cmd.split(maxsplit=1)
                command = parts[0].lower()

                if command == "quit" or command == "exit":
                    break

                elif command == "agents":
                    self.check_agents()

                elif command == "task" and len(parts) > 1:
                    task_text = parts[1]
                    task_id = self.submit_task(task_text)

                    # Wait for result
                    result = self.wait_for_result(task_id, timeout=120)
                    if result:
                        print("\n" + "="*40)
                        print("📋 FINAL RESULT:")
                        print("="*40)
                        print(result.final_response[:1000])

                elif command == "status" and len(parts) > 1:
                    task_id = parts[1]
                    status = self.get_task_status(task_id)
                    if status:
                        print(f"Task: {task_id}")
                        print(f"Status: {status.status}")
                        print(f"Current Agent: {status.current_agent}")
                        print(f"Messages: {len(status.messages)}")
                    else:
                        print(f"Task {task_id} not found")

                elif command == "result" and len(parts) > 1:
                    task_id = parts[1]
                    result = self.get_result(task_id)
                    if result:
                        print(result)
                    else:
                        print(f"No result for task {task_id}")

                else:
                    print("Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Distributed Agent Coordinator")
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
        "--task",
        help="Submit a single task and wait for result"
    )

    args = parser.parse_args()

    coordinator = AgentCoordinator(
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )

    if args.task:
        # Single task mode
        task_id = coordinator.submit_task(args.task)
        result = coordinator.wait_for_result(task_id)

        if result and result.status == "completed":
            print("\n" + "="*60)
            print("FINAL RESULT:")
            print("="*60)
            print(result.final_response)

            print("\n" + "="*60)
            print("AGENT COMMUNICATION LOG:")
            print("="*60)
            for msg in result.messages:
                print(f"\n[{msg['sender']}]: {msg['content'][:300]}...")
    else:
        # Interactive mode
        coordinator.run_interactive()


if __name__ == "__main__":
    main()
