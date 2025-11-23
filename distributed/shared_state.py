"""
================================================================================
Distributed Shared State using Redis
================================================================================

This module provides a Redis-based shared state that allows agents running
on different machines to communicate with each other.

Redis is used for:
1. Shared state storage (key-value)
2. Message passing (pub/sub)
3. Task queues (lists)
"""

import json
import time
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
import redis


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    content: str
    timestamp: float
    message_type: str = "agent_response"  # agent_response, task, status


@dataclass
class TaskState:
    """Shared state for a task being processed by multiple agents."""
    task_id: str
    task: str
    status: str  # pending, in_progress, completed, failed
    research_notes: str = ""
    analysis: str = ""
    final_response: str = ""
    current_agent: str = ""
    messages: List[Dict] = None
    created_at: float = 0
    updated_at: float = 0

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.created_at == 0:
            self.created_at = time.time()
        self.updated_at = time.time()


class DistributedState:
    """
    Redis-based distributed state manager.

    Allows agents on different machines to:
    - Read/write shared state
    - Subscribe to updates
    - Claim and process tasks
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_password: str = None):
        """Initialize connection to Redis."""
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        self.pubsub = self.redis.pubsub()

        # Key prefixes
        self.STATE_PREFIX = "agent_state:"
        self.TASK_QUEUE = "task_queue"
        self.RESULT_QUEUE = "result_queue"
        self.AGENT_CHANNEL = "agent_channel:"

    def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return self.redis.ping()
        except redis.ConnectionError:
            return False

    # =========================================================================
    # Task State Management
    # =========================================================================

    def create_task(self, task_id: str, task: str) -> TaskState:
        """Create a new task and store in Redis."""
        state = TaskState(
            task_id=task_id,
            task=task,
            status="pending"
        )
        self._save_state(state)
        return state

    def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """Get the current state of a task."""
        key = f"{self.STATE_PREFIX}{task_id}"
        data = self.redis.get(key)
        if data:
            return TaskState(**json.loads(data))
        return None

    def update_task_state(self, task_id: str, **updates) -> TaskState:
        """Update specific fields of a task state."""
        state = self.get_task_state(task_id)
        if not state:
            raise ValueError(f"Task {task_id} not found")

        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

        state.updated_at = time.time()
        self._save_state(state)
        return state

    def add_message(self, task_id: str, sender: str, content: str) -> None:
        """Add a message to the task's message history."""
        state = self.get_task_state(task_id)
        if state:
            msg = AgentMessage(
                sender=sender,
                content=content,
                timestamp=time.time()
            )
            state.messages.append(asdict(msg))
            self._save_state(state)

    def _save_state(self, state: TaskState) -> None:
        """Save task state to Redis."""
        key = f"{self.STATE_PREFIX}{state.task_id}"
        self.redis.set(key, json.dumps(asdict(state)))

    # =========================================================================
    # Task Queue (for distributing work)
    # =========================================================================

    def enqueue_task(self, task_id: str, target_agent: str) -> None:
        """Add a task to an agent's queue."""
        queue_key = f"{self.TASK_QUEUE}:{target_agent}"
        self.redis.rpush(queue_key, task_id)

        # Also publish notification
        self.publish(f"task_ready:{target_agent}", task_id)

    def dequeue_task(self, agent_type: str, timeout: int = 0) -> Optional[str]:
        """
        Get next task from agent's queue.
        Blocks until a task is available or timeout.
        """
        queue_key = f"{self.TASK_QUEUE}:{agent_type}"
        if timeout > 0:
            result = self.redis.blpop(queue_key, timeout=timeout)
            return result[1] if result else None
        else:
            return self.redis.lpop(queue_key)

    def enqueue_result(self, task_id: str) -> None:
        """Signal that a task has a new result ready."""
        self.redis.rpush(self.RESULT_QUEUE, task_id)
        self.publish("result_ready", task_id)

    # =========================================================================
    # Pub/Sub (for real-time notifications)
    # =========================================================================

    def publish(self, channel: str, message: str) -> None:
        """Publish a message to a channel."""
        self.redis.publish(channel, message)

    def subscribe(self, *channels) -> None:
        """Subscribe to channels for notifications."""
        self.pubsub.subscribe(*channels)

    def get_message(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get next message from subscribed channels."""
        return self.pubsub.get_message(timeout=timeout)

    def listen(self):
        """Generator that yields messages from subscribed channels."""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                yield message

    # =========================================================================
    # Agent Registration
    # =========================================================================

    def register_agent(self, agent_id: str, agent_type: str) -> None:
        """Register an agent as available."""
        key = f"agents:{agent_type}"
        self.redis.sadd(key, agent_id)
        self.redis.set(f"agent_heartbeat:{agent_id}", time.time(), ex=60)

    def heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat."""
        self.redis.set(f"agent_heartbeat:{agent_id}", time.time(), ex=60)

    def get_available_agents(self, agent_type: str) -> List[str]:
        """Get list of registered agents of a type."""
        key = f"agents:{agent_type}"
        return list(self.redis.smembers(key))

    def unregister_agent(self, agent_id: str, agent_type: str) -> None:
        """Unregister an agent."""
        key = f"agents:{agent_type}"
        self.redis.srem(key, agent_id)
        self.redis.delete(f"agent_heartbeat:{agent_id}")

    # =========================================================================
    # Locking (for exclusive access)
    # =========================================================================

    def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """Acquire a distributed lock."""
        return self.redis.set(
            f"lock:{lock_name}",
            "locked",
            nx=True,
            ex=timeout
        )

    def release_lock(self, lock_name: str) -> None:
        """Release a distributed lock."""
        self.redis.delete(f"lock:{lock_name}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_distributed_state(
    redis_host: str = "localhost",
    redis_port: int = 6379
) -> DistributedState:
    """Factory function to create distributed state."""
    state = DistributedState(redis_host=redis_host, redis_port=redis_port)
    if not state.ping():
        raise ConnectionError(f"Cannot connect to Redis at {redis_host}:{redis_port}")
    return state


if __name__ == "__main__":
    # Test the distributed state
    print("Testing Distributed State...")

    try:
        state = create_distributed_state()
        print("✓ Connected to Redis")

        # Create a task
        task = state.create_task("test-001", "Explain quantum computing")
        print(f"✓ Created task: {task.task_id}")

        # Update task
        state.update_task_state("test-001", status="in_progress", current_agent="researcher")
        print("✓ Updated task state")

        # Add message
        state.add_message("test-001", "researcher", "Starting research...")
        print("✓ Added message")

        # Read back
        result = state.get_task_state("test-001")
        print(f"✓ Read task: status={result.status}, agent={result.current_agent}")
        print(f"✓ Messages: {len(result.messages)}")

        print("\n✅ All tests passed!")

    except ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("Make sure Redis is running: docker run -d -p 6379:6379 redis:latest")
