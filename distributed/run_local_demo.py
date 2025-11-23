"""
================================================================================
Local Demo: Run Distributed Agents in Separate Processes
================================================================================

This script demonstrates the distributed system by running all components
locally in separate processes. In production, each would run on a different VM.

Usage:
    python run_local_demo.py

This will:
1. Start Redis (if Docker is available)
2. Start 3 agent workers in separate processes
3. Start the coordinator
4. Submit a test task
5. Show the results
"""

import subprocess
import time
import sys
import os
import signal
from multiprocessing import Process
import threading


def check_redis():
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        return True
    except:
        return False


def start_redis_docker():
    """Start Redis using Docker."""
    print("🐳 Starting Redis via Docker...")
    try:
        # Check if container already exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=redis-agents", "--format", "{{.Names}}"],
            capture_output=True, text=True
        )

        if "redis-agents" in result.stdout:
            # Container exists, start it
            subprocess.run(["docker", "start", "redis-agents"], capture_output=True)
        else:
            # Create new container
            subprocess.run([
                "docker", "run", "-d",
                "--name", "redis-agents",
                "-p", "6379:6379",
                "redis:latest"
            ], capture_output=True)

        time.sleep(2)
        return check_redis()
    except Exception as e:
        print(f"   ❌ Docker error: {e}")
        return False


def run_agent_worker(agent_type: str):
    """Run an agent worker (called in subprocess)."""
    from agent_worker import DistributedAgent

    agent = DistributedAgent(
        agent_type=agent_type,
        redis_host="localhost",
        redis_port=6379,
        llm_host="localhost",
        llm_port=11434,
        model_name="llama3.2"
    )
    agent.run()


def run_coordinator_task(task: str):
    """Run coordinator and submit a task."""
    from coordinator import AgentCoordinator

    coordinator = AgentCoordinator(redis_host="localhost", redis_port=6379)

    print("\n" + "="*60)
    print("📡 Checking available agents...")
    print("="*60)

    # Wait for agents to register
    time.sleep(3)
    coordinator.check_agents()

    # Submit task
    task_id = coordinator.submit_task(task)

    # Wait for result
    result = coordinator.wait_for_result(task_id, timeout=180)

    if result and result.status == "completed":
        print("\n" + "="*60)
        print("✅ TASK COMPLETED SUCCESSFULLY!")
        print("="*60)

        print("\n📝 AGENT COMMUNICATION LOG:")
        print("-"*40)
        for msg in result.messages:
            sender = msg.get('sender', 'unknown')
            content = msg.get('content', '')[:300]
            print(f"\n[{sender}]:\n{content}...")

        print("\n" + "="*60)
        print("📋 FINAL RESULT:")
        print("="*60)
        print(result.final_response)

    else:
        print("\n❌ Task did not complete successfully")
        if result:
            print(f"   Status: {result.status}")

    return result


def main():
    print("="*60)
    print("🚀 DISTRIBUTED MULTI-AGENT DEMO")
    print("="*60)
    print("\nThis demo runs all agents in separate processes to simulate")
    print("a distributed system. In production, each would be on a different VM.\n")

    # Step 1: Check/Start Redis
    print("Step 1: Checking Redis...")
    if check_redis():
        print("   ✓ Redis is already running")
    else:
        print("   Redis not found. Attempting to start...")
        if start_redis_docker():
            print("   ✓ Redis started via Docker")
        else:
            print("   ❌ Could not start Redis!")
            print("   Please start Redis manually:")
            print("   docker run -d -p 6379:6379 redis:latest")
            sys.exit(1)

    # Step 2: Check Ollama
    print("\nStep 2: Checking Ollama LLM...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✓ Ollama is running")
        else:
            raise Exception("Ollama not responding")
    except:
        print("   ❌ Ollama not found!")
        print("   Please start Ollama: ollama serve")
        sys.exit(1)

    # Step 3: Start agent workers in separate processes
    print("\nStep 3: Starting agent workers...")

    agent_processes = []
    for agent_type in ["researcher", "analyst", "writer"]:
        p = Process(target=run_agent_worker, args=(agent_type,))
        p.daemon = True
        p.start()
        agent_processes.append(p)
        print(f"   ✓ Started {agent_type} agent (PID: {p.pid})")

    # Give agents time to start
    time.sleep(2)

    # Step 4: Run a test task
    print("\nStep 4: Running test task...")

    test_task = "Explain how machine learning is used in healthcare. Include examples of current applications and future possibilities."

    try:
        result = run_coordinator_task(test_task)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted!")
    finally:
        # Cleanup: stop agent processes
        print("\n\nCleaning up...")
        for p in agent_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        print("✓ All agents stopped")


if __name__ == "__main__":
    main()
