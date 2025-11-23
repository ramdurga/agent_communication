"""
================================================================================
HTTP-Based Agent Orchestrator with Multi-Instance Support
================================================================================

The orchestrator coordinates multiple HTTP-based agents running on different
machines. It supports multiple instances of each agent type with load balancing.

Usage:
    # Start multiple agent instances:
    python -m agents.researcher --port 8001 --instance researcher-1
    python -m agents.researcher --port 8011 --instance researcher-2
    python -m agents.analyst --port 8002 --instance analyst-1
    python -m agents.writer --port 8003 --instance writer-1

    # Run orchestrator:
    python orchestrator.py --task "Explain quantum computing"

    # With multiple instances:
    python orchestrator.py --task "..." \
        --researcher-urls http://localhost:8001,http://localhost:8011 \
        --analyst-urls http://localhost:8002 \
        --writer-urls http://localhost:8003 \
        --strategy round_robin
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from core.registry import AgentRegistry, AgentInstance


# =============================================================================
# Colored Output Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @staticmethod
    def colorize(text: str, color: str) -> str:
        return f"{color}{text}{Colors.RESET}"


def log_info(msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"{Colors.GREEN}[{timestamp}] [INFO]{Colors.RESET} {msg}")

def log_step(step: int, msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"{Colors.CYAN}[{timestamp}] [STEP {step}]{Colors.RESET} {msg}")

def log_agent(agent: str, msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    agent_colors = {
        "researcher": Colors.BLUE,
        "analyst": Colors.YELLOW,
        "writer": Colors.HEADER
    }
    color = agent_colors.get(agent, Colors.WHITE)
    print(f"{color}[{timestamp}] [{agent.upper()}]{Colors.RESET} {msg}")

def log_error(msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"{Colors.RED}[{timestamp}] [ERROR]{Colors.RESET} {msg}")

def log_success(msg: str):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"{Colors.GREEN}[{timestamp}] [SUCCESS]{Colors.RESET} {msg}")


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    task: str
    research_notes: str = ""
    analysis: str = ""
    final_response: str = ""
    messages: List[Dict] = field(default_factory=list)
    thinking: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"
    total_time_ms: float = 0
    agent_times: Dict[str, float] = field(default_factory=dict)
    instances_used: Dict[str, str] = field(default_factory=dict)


class HTTPOrchestrator:
    """
    Orchestrates distributed HTTP-based agents with multi-instance support.

    Features:
    - Multiple instances per agent type
    - Load balancing (round_robin, least_loaded, random, fastest)
    - Health checking and automatic failover
    - Detailed logging and metrics
    """

    def __init__(
        self,
        researcher_urls: List[str] = None,
        analyst_urls: List[str] = None,
        writer_urls: List[str] = None,
        load_balancing_strategy: str = "round_robin"
    ):
        """
        Initialize orchestrator with agent URLs.

        Args:
            researcher_urls: List of researcher agent URLs
            analyst_urls: List of analyst agent URLs
            writer_urls: List of writer agent URLs
            load_balancing_strategy: Strategy for selecting instances
                                    (round_robin, least_loaded, random, fastest)
        """
        self.registry = AgentRegistry()
        self.strategy = load_balancing_strategy

        # Register agent instances
        self.registry.register_defaults(
            researcher_urls=researcher_urls,
            analyst_urls=analyst_urls,
            writer_urls=writer_urls
        )

        # Workflow order
        self.workflow = ["researcher", "analyst", "writer"]

    async def check_agents(self) -> Dict[str, Dict]:
        """Check health of all registered agents."""
        log_info("📡 Checking agent health...")
        print()

        await self.registry.health_check_all()

        health_status = {}
        for agent_type in self.workflow:
            instances = self.registry.get_all_instances(agent_type)
            healthy = self.registry.get_healthy_instances(agent_type)

            health_status[agent_type] = {
                "total": len(instances),
                "healthy": len(healthy),
                "all_healthy": len(healthy) == len(instances) and len(instances) > 0
            }

            for instance in instances:
                status_icon = "✓" if instance.healthy else "✗"
                status_color = Colors.GREEN if instance.healthy else Colors.RED

                log_agent(agent_type, f"{status_color}{status_icon}{Colors.RESET} {instance.url}")
                if instance.healthy:
                    log_agent(agent_type, f"   Instance: {instance.instance_id or 'default'}")
                    log_agent(agent_type, f"   Hostname: {instance.hostname}")
                    log_agent(agent_type, f"   LLM Model: {instance.model}")
                    log_agent(agent_type, f"   Tasks Processed: {instance.tasks_processed}")
                    log_agent(agent_type, f"   Response Time: {instance.response_time_ms:.0f}ms")
                else:
                    log_agent(agent_type, f"   Status: {instance.status}")

            if len(instances) > 1:
                log_agent(agent_type, f"   📊 {len(healthy)}/{len(instances)} instances healthy")

        print()
        return health_status

    async def call_agent(
        self,
        agent_type: str,
        task_id: str,
        task: str,
        research_notes: str = "",
        analysis: str = ""
    ) -> Optional[Dict]:
        """
        Call an agent instance using load balancing.

        Returns the agent's response or None on failure.
        """
        instance = self.registry.get_instance(agent_type, self.strategy)
        if not instance:
            log_error(f"No available instances for {agent_type}")
            return None

        request_data = {
            "task_id": task_id,
            "task": task,
            "research_notes": research_notes,
            "analysis": analysis,
            "context": {}
        }

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                instance_label = f"{instance.instance_id or 'default'}@{instance.url}"
                log_agent(agent_type, f"📤 Sending to {instance_label}")
                log_agent(agent_type, f"   Task ID: {task_id}")
                log_agent(agent_type, f"   Strategy: {self.strategy}")
                log_agent(agent_type, f"   Payload size: {len(str(request_data))} bytes")

                start_time = time.time()
                response = await client.post(
                    f"{instance.url}/process",
                    json=request_data
                )
                elapsed_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    result = response.json()

                    # Update instance metrics
                    instance.tasks_processed += 1
                    instance.response_time_ms = elapsed_ms

                    log_agent(agent_type, f"✅ Response received in {elapsed_ms:.0f}ms")
                    log_agent(agent_type, f"   Instance: {result.get('instance_id', 'default')}")
                    log_agent(agent_type, f"   Hostname: {result.get('hostname', 'unknown')}")
                    log_agent(agent_type, f"   Output length: {len(result.get('result', ''))} chars")
                    log_agent(agent_type, f"   Processing time: {result.get('processing_time_ms', 0):.0f}ms")
                    log_agent(agent_type, f"   Est. tokens: {result.get('llm_tokens_used', 'N/A')}")

                    # Log thinking process
                    thinking = result.get('thinking', '')
                    if thinking:
                        log_agent(agent_type, f"   💭 Thinking: {len(thinking)} chars")
                        thinking_preview = thinking[:200].replace('\n', ' ')
                        log_agent(agent_type, f"      \"{thinking_preview}...\"")

                    # Log tool usage
                    tools_used = result.get('tools_used', [])
                    if tools_used:
                        log_agent(agent_type, f"   🛠️  Tools used: {len(tools_used)}")
                        for tool in tools_used:
                            log_agent(agent_type, f"      • {tool['tool_name']}: {tool['execution_time_ms']:.0f}ms")
                            output_preview = tool.get('output_preview', '')[:60]
                            log_agent(agent_type, f"        └─ Output: {output_preview}...")

                    # Log steps performed
                    steps = result.get('steps_performed', [])
                    if steps:
                        log_agent(agent_type, f"   📝 Steps performed: {len(steps)}")
                        for i, step in enumerate(steps, 1):
                            log_agent(agent_type, f"      {i}. {step[:70]}{'...' if len(step) > 70 else ''}")

                    log_agent(agent_type, f"   Next agent: {result.get('next_agent') or 'None (final)'}")

                    # Add instance info to result
                    result['_instance_url'] = instance.url
                    result['_instance_id'] = instance.instance_id

                    return result
                else:
                    log_error(f"{agent_type} @ {instance.url} failed with status {response.status_code}")
                    log_error(f"   Response: {response.text[:200]}")
                    instance.healthy = False
                    return None

        except httpx.TimeoutException:
            log_error(f"{agent_type} @ {instance.url} timed out after 600s")
            instance.healthy = False
            return None
        except Exception as e:
            log_error(f"{agent_type} @ {instance.url} error: {type(e).__name__}: {e}")
            instance.healthy = False
            return None

    async def run_task(self, task: str, task_id: str = None) -> TaskResult:
        """
        Run a complete task through all agents.

        Args:
            task: The task description
            task_id: Optional task ID

        Returns:
            TaskResult with all outputs
        """
        if task_id is None:
            task_id = f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        result = TaskResult(task_id=task_id, task=task)
        overall_start = time.time()

        print()
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
        log_info(f"🎯 STARTING DISTRIBUTED TASK PIPELINE")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
        log_info(f"Task ID: {task_id}")
        log_info(f"Task: {task[:80]}...")
        log_info(f"Pipeline: researcher → analyst → writer")
        log_info(f"Load Balancing: {self.strategy}")
        print()

        # Step 1: Researcher
        print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}")
        log_step(1, "📚 RESEARCH PHASE")
        print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}")

        step_start = time.time()
        researcher_result = await self.call_agent(
            "researcher",
            task_id,
            task
        )

        if researcher_result:
            result.research_notes = researcher_result.get("result", "")
            result.agent_times["researcher"] = researcher_result.get("processing_time_ms", 0)
            result.instances_used["researcher"] = researcher_result.get("_instance_url", "")
            if researcher_result.get("thinking"):
                result.thinking["researcher"] = researcher_result.get("thinking")
            result.messages.append({
                "agent": "researcher",
                "agent_id": researcher_result.get("agent_id"),
                "instance_id": researcher_result.get("instance_id"),
                "instance_url": researcher_result.get("_instance_url"),
                "hostname": researcher_result.get("hostname"),
                "processing_time_ms": researcher_result.get("processing_time_ms"),
                "content": result.research_notes,
                "thinking": researcher_result.get("thinking", "")
            })
            log_step(1, f"✅ Research complete ({(time.time()-step_start)*1000:.0f}ms total)")
        else:
            result.status = "failed"
            log_error("Research phase failed - aborting pipeline")
            return result

        print()

        # Step 2: Analyst
        print(f"{Colors.YELLOW}{'─'*70}{Colors.RESET}")
        log_step(2, "🔍 ANALYSIS PHASE")
        print(f"{Colors.YELLOW}{'─'*70}{Colors.RESET}")

        step_start = time.time()
        analyst_result = await self.call_agent(
            "analyst",
            task_id,
            task,
            research_notes=result.research_notes
        )

        if analyst_result:
            result.analysis = analyst_result.get("result", "")
            result.agent_times["analyst"] = analyst_result.get("processing_time_ms", 0)
            result.instances_used["analyst"] = analyst_result.get("_instance_url", "")
            if analyst_result.get("thinking"):
                result.thinking["analyst"] = analyst_result.get("thinking")
            result.messages.append({
                "agent": "analyst",
                "agent_id": analyst_result.get("agent_id"),
                "instance_id": analyst_result.get("instance_id"),
                "instance_url": analyst_result.get("_instance_url"),
                "hostname": analyst_result.get("hostname"),
                "processing_time_ms": analyst_result.get("processing_time_ms"),
                "content": result.analysis,
                "thinking": analyst_result.get("thinking", "")
            })
            log_step(2, f"✅ Analysis complete ({(time.time()-step_start)*1000:.0f}ms total)")
        else:
            result.status = "failed"
            log_error("Analysis phase failed - aborting pipeline")
            return result

        print()

        # Step 3: Writer
        print(f"{Colors.HEADER}{'─'*70}{Colors.RESET}")
        log_step(3, "✍️  WRITING PHASE")
        print(f"{Colors.HEADER}{'─'*70}{Colors.RESET}")

        step_start = time.time()
        writer_result = await self.call_agent(
            "writer",
            task_id,
            task,
            research_notes=result.research_notes,
            analysis=result.analysis
        )

        if writer_result:
            result.final_response = writer_result.get("result", "")
            result.agent_times["writer"] = writer_result.get("processing_time_ms", 0)
            result.instances_used["writer"] = writer_result.get("_instance_url", "")
            if writer_result.get("thinking"):
                result.thinking["writer"] = writer_result.get("thinking")
            result.messages.append({
                "agent": "writer",
                "agent_id": writer_result.get("agent_id"),
                "instance_id": writer_result.get("instance_id"),
                "instance_url": writer_result.get("_instance_url"),
                "hostname": writer_result.get("hostname"),
                "processing_time_ms": writer_result.get("processing_time_ms"),
                "content": result.final_response,
                "thinking": writer_result.get("thinking", "")
            })
            result.status = "completed"
            log_step(3, f"✅ Writing complete ({(time.time()-step_start)*1000:.0f}ms total)")
        else:
            result.status = "failed"
            log_error("Writing phase failed")
            return result

        result.total_time_ms = (time.time() - overall_start) * 1000

        print()
        print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
        log_success(f"🎉 PIPELINE COMPLETED SUCCESSFULLY")
        log_success(f"   Total time: {result.total_time_ms:.0f}ms ({result.total_time_ms/1000:.1f}s)")
        log_success(f"   Researcher: {result.agent_times.get('researcher', 0):.0f}ms @ {result.instances_used.get('researcher', 'N/A')}")
        log_success(f"   Analyst: {result.agent_times.get('analyst', 0):.0f}ms @ {result.instances_used.get('analyst', 'N/A')}")
        log_success(f"   Writer: {result.agent_times.get('writer', 0):.0f}ms @ {result.instances_used.get('writer', 'N/A')}")
        print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")

        return result

    async def run_parallel_tasks(self, tasks: List[str]) -> List[TaskResult]:
        """
        Run multiple tasks in parallel.

        With multi-instance support, different tasks can be
        distributed across different agent instances.
        """
        print(f"\n🚀 Running {len(tasks)} tasks in parallel...")
        print(f"   Load balancing: {self.strategy}")

        # Show available instances
        summary = self.registry.summary()
        for agent_type, info in summary.items():
            print(f"   {agent_type}: {info['healthy']}/{info['total']} instances available")

        results = await asyncio.gather(*[
            self.run_task(task, f"parallel-{i}")
            for i, task in enumerate(tasks)
        ])

        return results


def parse_urls(url_string: str) -> List[str]:
    """Parse comma-separated URLs."""
    if not url_string:
        return None
    return [url.strip() for url in url_string.split(",") if url.strip()]


async def main():
    parser = argparse.ArgumentParser(
        description="HTTP Agent Orchestrator with Multi-Instance Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single instance per agent type (default):
  python orchestrator.py --task "Explain quantum computing"

  # Multiple researcher instances with round-robin:
  python orchestrator.py --task "Research AI" \\
      --researcher-urls http://localhost:8001,http://localhost:8011 \\
      --strategy round_robin

  # Use least-loaded strategy for optimal distribution:
  python orchestrator.py --task "Complex research" \\
      --researcher-urls http://host1:8001,http://host2:8001 \\
      --analyst-urls http://host1:8002,http://host2:8002 \\
      --writer-urls http://host1:8003 \\
      --strategy least_loaded
        """
    )

    parser.add_argument(
        "--task",
        required=True,
        help="Task to process"
    )
    parser.add_argument(
        "--researcher-urls",
        default="http://localhost:8001",
        help="Comma-separated researcher agent URLs"
    )
    parser.add_argument(
        "--analyst-urls",
        default="http://localhost:8002",
        help="Comma-separated analyst agent URLs"
    )
    parser.add_argument(
        "--writer-urls",
        default="http://localhost:8003",
        help="Comma-separated writer agent URLs"
    )
    parser.add_argument(
        "--strategy",
        choices=["round_robin", "least_loaded", "random", "fastest"],
        default="round_robin",
        help="Load balancing strategy (default: round_robin)"
    )

    args = parser.parse_args()

    # Create orchestrator with multiple instance support
    orchestrator = HTTPOrchestrator(
        researcher_urls=parse_urls(args.researcher_urls),
        analyst_urls=parse_urls(args.analyst_urls),
        writer_urls=parse_urls(args.writer_urls),
        load_balancing_strategy=args.strategy
    )

    # Check agent health
    health = await orchestrator.check_agents()

    # Check if at least one instance per agent type is healthy
    all_available = all(
        info["healthy"] > 0
        for info in health.values()
    )

    if not all_available:
        print("\n⚠️  Not all agent types have available instances!")
        print("Please start at least one instance of each agent type:")
        print("  python -m agents.researcher --port 8001 --instance researcher-1")
        print("  python -m agents.analyst --port 8002 --instance analyst-1")
        print("  python -m agents.writer --port 8003 --instance writer-1")
        print("\nFor multiple instances:")
        print("  python -m agents.researcher --port 8011 --instance researcher-2")
        return

    # Run the task
    result = await orchestrator.run_task(args.task)

    # Display results
    print()
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}📋 TASK RESULT SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"Status: {Colors.GREEN if result.status == 'completed' else Colors.RED}{result.status}{Colors.RESET}")
    print(f"Task ID: {result.task_id}")
    print(f"Total Time: {result.total_time_ms:.0f}ms ({result.total_time_ms/1000:.1f}s)")
    print(f"Load Balancing: {args.strategy}")

    if result.status == "completed":
        print()
        print(f"{Colors.BLUE}{'─'*70}{Colors.RESET}")
        print(f"{Colors.BLUE}🔬 RESEARCH NOTES (excerpt):{Colors.RESET}")
        print(f"{Colors.BLUE}{'─'*70}{Colors.RESET}")
        print(result.research_notes[:600] + "..." if len(result.research_notes) > 600 else result.research_notes)

        print()
        print(f"{Colors.YELLOW}{'─'*70}{Colors.RESET}")
        print(f"{Colors.YELLOW}📊 ANALYSIS (excerpt):{Colors.RESET}")
        print(f"{Colors.YELLOW}{'─'*70}{Colors.RESET}")
        print(result.analysis[:600] + "..." if len(result.analysis) > 600 else result.analysis)

        print()
        print(f"{Colors.HEADER}{'─'*70}{Colors.RESET}")
        print(f"{Colors.HEADER}📝 FINAL RESPONSE:{Colors.RESET}")
        print(f"{Colors.HEADER}{'─'*70}{Colors.RESET}")
        print(result.final_response)

        print()
        print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}")
        print(f"{Colors.CYAN}🤖 AGENT COMMUNICATION LOG:{Colors.RESET}")
        print(f"{Colors.CYAN}{'─'*70}{Colors.RESET}")
        for msg in result.messages:
            agent_color = {
                "researcher": Colors.BLUE,
                "analyst": Colors.YELLOW,
                "writer": Colors.HEADER
            }.get(msg['agent'], Colors.WHITE)

            instance_info = f"{msg.get('instance_id', 'default')} @ {msg.get('instance_url', 'unknown')}"
            print(f"\n{agent_color}[{msg['agent'].upper()}]{Colors.RESET} ({instance_info})")
            print(f"   Hostname: {msg.get('hostname', 'unknown')}")
            print(f"   Processing time: {msg.get('processing_time_ms', 0):.0f}ms")
            if msg.get('thinking'):
                print(f"   💭 Thinking: {len(msg['thinking'])} chars")
                thinking_preview = msg['thinking'][:150].replace('\n', ' ')
                print(f"      \"{thinking_preview}...\"")
            print(f"   Output: {msg['content'][:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
