#!/bin/bash
# ============================================================================
# HTTP-Based Distributed Agents Demo
# ============================================================================
#
# This script starts all agents and runs a demo task.
# Updated for new modular agent structure.
#
# Usage:
#   chmod +x run_demo.sh
#   ./run_demo.sh
#   ./run_demo.sh --model deepseek-r1
#   ./run_demo.sh --scale 2
#
# ============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
MODEL="llama3.2"
SCALE=1
TASK="Explain the benefits of renewable energy"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL   LLM model to use (default: llama3.2)"
            echo "  --scale N       Number of instances per agent (default: 1)"
            echo "  --task TASK     Task to process"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              HTTP-Based Distributed Agents Demo                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Scale: $SCALE instance(s) per agent"
echo "  Task:  $TASK"
echo "  Directory: $SCRIPT_DIR"
echo ""

# Store PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping agents...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    # Also kill any tmux session we created
    tmux kill-session -t agents 2>/dev/null || true
    echo -e "${GREEN}Done!${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Check if Ollama is running
echo -e "${YELLOW}Checking Ollama...${NC}"
if curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${RED}✗ Ollama is not responding at localhost:11434${NC}"
    echo "  Please start Ollama: ollama serve"
    exit 1
fi

# Check if model is available
echo -e "${YELLOW}Checking model '$MODEL'...${NC}"
if curl -s "http://localhost:11434/api/tags" | grep -q "\"name\":\"$MODEL\""; then
    echo -e "${GREEN}✓ Model '$MODEL' is available${NC}"
else
    echo -e "${YELLOW}⚠ Model '$MODEL' not found. Pulling...${NC}"
    ollama pull "$MODEL"
fi
echo ""

# Check if tmux is available for better experience (only for scale=1)
if command -v tmux &> /dev/null && [ "$SCALE" -eq 1 ] && [ -z "$NO_TMUX" ]; then
    echo "Using tmux for multi-pane view..."
    echo "(Set NO_TMUX=1 to disable)"
    echo ""

    # Kill any existing session
    tmux kill-session -t agents 2>/dev/null || true

    # Create a new tmux session
    tmux new-session -d -s agents -c "$SCRIPT_DIR"

    # Split into 4 panes
    tmux split-window -h -c "$SCRIPT_DIR"
    tmux split-window -v -c "$SCRIPT_DIR"
    tmux select-pane -t 0
    tmux split-window -v -c "$SCRIPT_DIR"

    # Start agents in each pane
    tmux send-keys -t agents:0.0 "python -m agents.researcher --port 8001 --instance researcher-1 --model $MODEL" C-m
    tmux send-keys -t agents:0.1 "python -m agents.analyst --port 8002 --instance analyst-1 --model $MODEL" C-m
    tmux send-keys -t agents:0.2 "python -m agents.writer --port 8003 --instance writer-1 --model $MODEL" C-m

    # Wait for agents to start
    sleep 5

    # Run orchestrator in the last pane
    tmux send-keys -t agents:0.3 "python orchestrator.py --task '$TASK'" C-m

    # Attach to the session
    tmux attach-session -t agents

else
    echo "Starting agents in background..."
    echo ""

    # Build URL lists
    RESEARCHER_URLS=""
    ANALYST_URLS=""
    WRITER_URLS=""

    for i in $(seq 1 $SCALE); do
        RESEARCHER_PORT=$((8001 + (i-1) * 10))
        ANALYST_PORT=$((8002 + (i-1) * 10))
        WRITER_PORT=$((8003 + (i-1) * 10))

        echo -e "${CYAN}Starting instance set $i...${NC}"

        # Start Researcher
        echo "  Researcher on port $RESEARCHER_PORT"
        python -m agents.researcher \
            --port $RESEARCHER_PORT \
            --instance "researcher-$i" \
            --model "$MODEL" &
        PIDS+=($!)

        # Start Analyst
        echo "  Analyst on port $ANALYST_PORT"
        python -m agents.analyst \
            --port $ANALYST_PORT \
            --instance "analyst-$i" \
            --model "$MODEL" &
        PIDS+=($!)

        # Start Writer
        echo "  Writer on port $WRITER_PORT"
        python -m agents.writer \
            --port $WRITER_PORT \
            --instance "writer-$i" \
            --model "$MODEL" &
        PIDS+=($!)

        # Build URL lists
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

    # Wait for agents to start
    echo ""
    echo "Waiting for agents to initialize..."
    sleep 5

    # Run orchestrator
    echo ""
    echo -e "${GREEN}Running orchestrator...${NC}"
    echo ""

    python orchestrator.py \
        --task "$TASK" \
        --researcher-urls "$RESEARCHER_URLS" \
        --analyst-urls "$ANALYST_URLS" \
        --writer-urls "$WRITER_URLS" \
        --strategy round_robin

    echo ""
    echo -e "${GREEN}Demo complete!${NC}"
fi
