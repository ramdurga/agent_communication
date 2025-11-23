#!/bin/bash
# =============================================================================
# Distributed Multi-Agent System - Start Script
# =============================================================================
#
# This script starts all agent instances for the distributed system.
# Each agent runs as a separate process and can be scaled independently.
#
# Usage:
#   ./run_agents.sh              # Start one instance of each agent
#   ./run_agents.sh --scale 2    # Start 2 instances of each agent
#   ./run_agents.sh --help       # Show help
# =============================================================================

set -e

# Default settings
SCALE=1
MODEL="llama3.2"
LLM_HOST="localhost"
LLM_PORT=11434

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║       Distributed Multi-Agent System with LLM Integration          ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    print_header
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --scale N       Start N instances of each agent type (default: 1)"
    echo "  --model MODEL   LLM model to use (default: llama3.2)"
    echo "  --llm-host HOST Ollama host (default: localhost)"
    echo "  --llm-port PORT Ollama port (default: 11434)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Start 1 instance of each agent"
    echo "  $0 --scale 3            # Start 3 instances of each agent"
    echo "  $0 --model deepseek-r1  # Use deepseek-r1 model"
    echo ""
    echo "Agent ports:"
    echo "  Researcher: 8001, 8011, 8021, ..."
    echo "  Analyst:    8002, 8012, 8022, ..."
    echo "  Writer:     8003, 8013, 8023, ..."
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --llm-host)
            LLM_HOST="$2"
            shift 2
            ;;
        --llm-port)
            LLM_PORT="$2"
            shift 2
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

print_header

echo -e "${GREEN}Configuration:${NC}"
echo "  Scale:     $SCALE instance(s) per agent type"
echo "  Model:     $MODEL"
echo "  LLM Host:  $LLM_HOST:$LLM_PORT"
echo ""

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama...${NC}"
if curl -s "http://$LLM_HOST:$LLM_PORT/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${RED}✗ Ollama is not responding at $LLM_HOST:$LLM_PORT${NC}"
    echo "  Please start Ollama: ollama serve"
    exit 1
fi

# Check if model is available
echo -e "${YELLOW}Checking model '$MODEL'...${NC}"
if curl -s "http://$LLM_HOST:$LLM_PORT/api/tags" | grep -q "\"$MODEL\""; then
    echo -e "${GREEN}✓ Model '$MODEL' is available${NC}"
else
    echo -e "${YELLOW}⚠ Model '$MODEL' not found. Pulling...${NC}"
    ollama pull "$MODEL"
fi

echo ""
echo -e "${CYAN}Starting agents...${NC}"
echo ""

# Store PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down agents...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "  Stopped PID $pid"
        fi
    done
    echo -e "${GREEN}All agents stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start agents
for i in $(seq 1 $SCALE); do
    # Calculate ports
    RESEARCHER_PORT=$((8001 + (i-1) * 10))
    ANALYST_PORT=$((8002 + (i-1) * 10))
    WRITER_PORT=$((8003 + (i-1) * 10))

    # Start Researcher
    echo -e "${BLUE}Starting Researcher instance $i on port $RESEARCHER_PORT${NC}"
    python -m agents.researcher \
        --port $RESEARCHER_PORT \
        --instance "researcher-$i" \
        --model "$MODEL" \
        --llm-host "$LLM_HOST" \
        --llm-port "$LLM_PORT" &
    PIDS+=($!)

    # Start Analyst
    echo -e "${YELLOW}Starting Analyst instance $i on port $ANALYST_PORT${NC}"
    python -m agents.analyst \
        --port $ANALYST_PORT \
        --instance "analyst-$i" \
        --model "$MODEL" \
        --llm-host "$LLM_HOST" \
        --llm-port "$LLM_PORT" &
    PIDS+=($!)

    # Start Writer
    echo -e "${PURPLE}Starting Writer instance $i on port $WRITER_PORT${NC}"
    python -m agents.writer \
        --port $WRITER_PORT \
        --instance "writer-$i" \
        --model "$MODEL" \
        --llm-host "$LLM_HOST" \
        --llm-port "$LLM_PORT" &
    PIDS+=($!)
done

echo ""
echo -e "${GREEN}All agents started!${NC}"
echo ""

# Build URL lists for orchestrator
RESEARCHER_URLS=""
ANALYST_URLS=""
WRITER_URLS=""

for i in $(seq 1 $SCALE); do
    RESEARCHER_PORT=$((8001 + (i-1) * 10))
    ANALYST_PORT=$((8002 + (i-1) * 10))
    WRITER_PORT=$((8003 + (i-1) * 10))

    if [ -z "$RESEARCHER_URLS" ]; then
        RESEARCHER_URLS="http://localhost:$RESEARCHER_PORT"
        ANALYST_URLS="http://localhost:$ANALYST_PORT"
        WRITER_URLS="http://localhost:$WRITER_PORT"
    else
        RESEARCHER_URLS="$RESEARCHER_URLS,http://localhost:$RESEARCHER_PORT"
        ANALYST_URLS="$ANALYST_URLS,http://localhost:$ANALYST_PORT"
        WRITER_URLS="$WRITER_URLS,http://localhost:$WRITER_PORT"
    fi
done

echo -e "${CYAN}To run a task, use:${NC}"
echo ""
echo "python orchestrator.py --task \"Your task here\" \\"
echo "    --researcher-urls \"$RESEARCHER_URLS\" \\"
echo "    --analyst-urls \"$ANALYST_URLS\" \\"
echo "    --writer-urls \"$WRITER_URLS\" \\"
echo "    --strategy round_robin"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all agents${NC}"
echo ""

# Wait for all agents
wait
