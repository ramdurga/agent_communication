# Distributed Multi-Agent System

This example demonstrates how to run agents on **separate machines/VMs** that communicate over a network.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      VM 1       │     │      VM 2       │     │      VM 3       │
│                 │     │                 │     │                 │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Researcher│  │     │  │  Analyst  │  │     │  │  Writer   │  │
│  │   Agent   │  │     │  │   Agent   │  │     │  │   Agent   │  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │     │  └─────┬─────┘  │
│        │        │     │        │        │     │        │        │
└────────┼────────┘     └────────┼────────┘     └────────┼────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Redis Server       │
                    │   (Message Broker +     │
                    │    Shared State)        │
                    │   192.168.1.100:6379    │
                    └─────────────────────────┘
```

## Components

1. **`redis_coordinator.py`** - Orchestrator that manages the workflow
2. **`agent_worker.py`** - Individual agent that runs on any machine
3. **`shared_state.py`** - Redis-based shared state management
4. **`run_coordinator.py`** - Start the coordinator
5. **`run_agent.py`** - Start an agent worker

## Setup

### 1. Install Dependencies
```bash
pip install redis langchain langchain-ollama
```

### 2. Start Redis (on coordinator machine)
```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or install Redis directly
sudo apt install redis-server
sudo systemctl start redis
```

### 3. Run the System

**On Coordinator Machine:**
```bash
python run_coordinator.py --redis-host localhost
```

**On Agent Machine 1 (Researcher):**
```bash
python run_agent.py --agent-type researcher --redis-host 192.168.1.100
```

**On Agent Machine 2 (Analyst):**
```bash
python run_agent.py --agent-type analyst --redis-host 192.168.1.100
```

**On Agent Machine 3 (Writer):**
```bash
python run_agent.py --agent-type writer --redis-host 192.168.1.100
```

## Communication Flow

1. Coordinator publishes task to Redis
2. Researcher agent picks up task, processes, publishes results
3. Analyst agent picks up research, analyzes, publishes results
4. Writer agent picks up analysis, writes final response
5. Coordinator collects final result

## Alternative: HTTP-Based Communication

See `http_agents/` for a REST API-based approach using FastAPI.
