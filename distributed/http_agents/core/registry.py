"""
Agent Registry - Discovers and manages multiple agent instances.
Supports load balancing across multiple instances of the same agent type.
"""
import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from .models import AgentRegistration, HealthResponse


@dataclass
class AgentInstance:
    """Represents a single agent instance."""
    agent_type: str
    url: str
    instance_id: str = ""
    hostname: str = ""
    port: int = 0
    model: str = ""
    status: str = "unknown"
    tasks_processed: int = 0
    last_health_check: float = 0
    healthy: bool = False
    response_time_ms: float = 0


@dataclass
class AgentRegistry:
    """
    Registry for discovering and managing agent instances.

    Supports:
    - Multiple instances per agent type
    - Health checking
    - Load balancing (round-robin, least-loaded, random)
    """
    instances: Dict[str, List[AgentInstance]] = field(default_factory=dict)
    _current_index: Dict[str, int] = field(default_factory=dict)

    def register(self, agent_type: str, url: str, instance_id: str = "") -> AgentInstance:
        """Register an agent instance."""
        instance = AgentInstance(
            agent_type=agent_type,
            url=url,
            instance_id=instance_id
        )

        if agent_type not in self.instances:
            self.instances[agent_type] = []
            self._current_index[agent_type] = 0

        # Check if already registered
        for existing in self.instances[agent_type]:
            if existing.url == url:
                return existing

        self.instances[agent_type].append(instance)
        return instance

    def register_defaults(
        self,
        researcher_urls: List[str] = None,
        analyst_urls: List[str] = None,
        writer_urls: List[str] = None
    ):
        """Register default agent URLs."""
        researcher_urls = researcher_urls or ["http://localhost:8001"]
        analyst_urls = analyst_urls or ["http://localhost:8002"]
        writer_urls = writer_urls or ["http://localhost:8003"]

        for url in researcher_urls:
            self.register("researcher", url)
        for url in analyst_urls:
            self.register("analyst", url)
        for url in writer_urls:
            self.register("writer", url)

    async def health_check_all(self) -> Dict[str, List[AgentInstance]]:
        """Check health of all registered agents."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for agent_type, instances in self.instances.items():
                for instance in instances:
                    await self._check_instance_health(client, instance)
        return self.instances

    async def _check_instance_health(self, client: httpx.AsyncClient, instance: AgentInstance):
        """Check health of a single instance."""
        try:
            start = time.time()
            response = await client.get(f"{instance.url}/health")
            response_time = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                instance.healthy = True
                instance.status = "healthy"
                instance.hostname = data.get("hostname", "")
                instance.instance_id = data.get("instance_id", "")
                instance.model = data.get("llm_model", "")
                instance.tasks_processed = data.get("tasks_processed", 0)
                instance.response_time_ms = response_time
            else:
                instance.healthy = False
                instance.status = f"unhealthy (status {response.status_code})"

            instance.last_health_check = time.time()

        except Exception as e:
            instance.healthy = False
            instance.status = f"unreachable: {str(e)[:50]}"
            instance.last_health_check = time.time()

    def get_instance(
        self,
        agent_type: str,
        strategy: str = "round_robin"
    ) -> Optional[AgentInstance]:
        """
        Get an agent instance using the specified load balancing strategy.

        Strategies:
        - round_robin: Rotate through instances
        - least_loaded: Pick instance with fewest tasks
        - random: Random selection
        - fastest: Pick instance with lowest response time
        """
        if agent_type not in self.instances:
            return None

        healthy = [i for i in self.instances[agent_type] if i.healthy]
        if not healthy:
            # Fall back to all instances if none are marked healthy
            healthy = self.instances[agent_type]

        if not healthy:
            return None

        if strategy == "round_robin":
            idx = self._current_index.get(agent_type, 0)
            instance = healthy[idx % len(healthy)]
            self._current_index[agent_type] = (idx + 1) % len(healthy)
            return instance

        elif strategy == "least_loaded":
            return min(healthy, key=lambda x: x.tasks_processed)

        elif strategy == "fastest":
            return min(healthy, key=lambda x: x.response_time_ms or float('inf'))

        elif strategy == "random":
            return random.choice(healthy)

        # Default to first available
        return healthy[0]

    def get_all_instances(self, agent_type: str) -> List[AgentInstance]:
        """Get all instances of a specific agent type."""
        return self.instances.get(agent_type, [])

    def get_healthy_instances(self, agent_type: str) -> List[AgentInstance]:
        """Get all healthy instances of a specific agent type."""
        return [i for i in self.instances.get(agent_type, []) if i.healthy]

    def summary(self) -> Dict:
        """Get a summary of all registered agents."""
        result = {}
        for agent_type, instances in self.instances.items():
            healthy = [i for i in instances if i.healthy]
            result[agent_type] = {
                "total": len(instances),
                "healthy": len(healthy),
                "instances": [
                    {
                        "url": i.url,
                        "instance_id": i.instance_id,
                        "healthy": i.healthy,
                        "tasks": i.tasks_processed
                    }
                    for i in instances
                ]
            }
        return result
