#!/bin/bash
# ============================================================================
# SQL Learning Agent - Complete Setup and Run Script
# ============================================================================
#
# This script:
# 1. Starts PostgreSQL in Docker
# 2. Waits for database to be ready
# 3. Opens tmux with multiple panes showing:
#    - Database logs
#    - Interactive SQL agent
#    - Database shell (psql)
#
# Usage:
#   ./run_sql_agent.sh           # Full setup with tmux
#   ./run_sql_agent.sh --simple  # Just start DB and agent (no tmux)
#   ./run_sql_agent.sh --stop    # Stop everything
#   ./run_sql_agent.sh --reset   # Reset database (delete data)
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║           SQL Learning Agent with PostgreSQL                       ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}Docker daemon is not running. Please start Docker.${NC}"
        exit 1
    fi
}

check_python_deps() {
    echo -e "${YELLOW}Checking Python dependencies...${NC}"
    pip install psycopg2-binary --quiet 2>/dev/null || pip3 install psycopg2-binary --quiet 2>/dev/null
    echo -e "${GREEN}✓ Dependencies ready${NC}"
}

start_database() {
    echo -e "${YELLOW}Starting PostgreSQL...${NC}"
    cd "$DOCKER_DIR"

    # Start the database
    docker-compose up -d

    # Wait for it to be ready
    echo -e "${YELLOW}Waiting for database to be ready...${NC}"
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U agent -d learning_db &> /dev/null; then
            echo -e "${GREEN}✓ Database is ready${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "${RED}Database failed to start${NC}"
    return 1
}

stop_database() {
    echo -e "${YELLOW}Stopping PostgreSQL...${NC}"
    cd "$DOCKER_DIR"
    docker-compose down
    echo -e "${GREEN}✓ Database stopped${NC}"
}

reset_database() {
    echo -e "${YELLOW}Resetting database (deleting all data)...${NC}"
    cd "$DOCKER_DIR"
    docker-compose down -v
    echo -e "${GREEN}✓ Database reset${NC}"
}

run_simple() {
    print_header
    check_docker
    check_python_deps
    start_database

    echo ""
    echo -e "${GREEN}Starting SQL Learning Agent...${NC}"
    echo -e "${CYAN}Type SQL queries or use commands like .tables, .ask, .stats${NC}"
    echo ""

    cd "$SCRIPT_DIR"
    python3 advanced_sql_agent.py
}

run_with_tmux() {
    print_header
    check_docker
    check_python_deps

    # Check tmux
    if ! command -v tmux &> /dev/null; then
        echo -e "${YELLOW}tmux not found, running in simple mode${NC}"
        run_simple
        return
    fi

    # Kill existing session if any
    tmux kill-session -t sql_agent 2>/dev/null || true

    # Start database first
    start_database

    # Create tmux session with multiple panes
    echo -e "${GREEN}Creating tmux session...${NC}"

    # Create new session with database logs pane
    tmux new-session -d -s sql_agent -n main -c "$SCRIPT_DIR"

    # Pane 0: Database logs
    tmux send-keys -t sql_agent:main.0 "cd $DOCKER_DIR && docker-compose logs -f postgres" C-m

    # Split horizontally for agent
    tmux split-window -h -t sql_agent:main -c "$SCRIPT_DIR"

    # Pane 1: SQL Agent
    tmux send-keys -t sql_agent:main.1 "sleep 2 && python3 advanced_sql_agent.py" C-m

    # Split pane 1 vertically for psql
    tmux split-window -v -t sql_agent:main.1 -c "$SCRIPT_DIR"

    # Pane 2: psql shell
    tmux send-keys -t sql_agent:main.2 "sleep 3 && cd $DOCKER_DIR && docker-compose exec postgres psql -U agent -d learning_db" C-m

    # Create status pane at bottom
    tmux split-window -v -t sql_agent:main.0 -p 20 -c "$SCRIPT_DIR"

    # Pane 3: Status/help
    tmux send-keys -t sql_agent:main.3 "echo '
╔══════════════════════════════════════════════════════════════════════════════╗
║ SQL Learning Agent - Pane Layout                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  [Pane 0] Database Logs      │  [Pane 1] SQL Learning Agent                 ║
║  Shows PostgreSQL activity   │  Interactive SQL with auto-learning          ║
║                              │  Commands: .tables .ask .stats .quit         ║
║  ────────────────────────────┤                                              ║
║  [Pane 3] This help pane     │  ────────────────────────────────────────────║
║                              │  [Pane 2] psql Shell                          ║
║                              │  Direct PostgreSQL access                     ║
║                                                                              ║
║  Navigation: Ctrl+B then arrow keys to switch panes                         ║
║  Exit: Ctrl+B then d to detach, or .quit in agent pane                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
'" C-m

    # Select the agent pane
    tmux select-pane -t sql_agent:main.1

    # Attach to session
    echo ""
    echo -e "${GREEN}Attaching to tmux session...${NC}"
    echo -e "${CYAN}Use Ctrl+B then arrow keys to navigate panes${NC}"
    echo -e "${CYAN}Use Ctrl+B then d to detach${NC}"
    echo ""

    tmux attach-session -t sql_agent
}

show_help() {
    print_header
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  (none)      Start with tmux multi-pane view"
    echo "  --simple    Start without tmux (single terminal)"
    echo "  --stop      Stop the database"
    echo "  --reset     Reset database (delete all data)"
    echo "  --help      Show this help"
    echo ""
    echo "In the SQL Agent, use:"
    echo "  .tables          Show available tables"
    echo "  .columns <table> Show columns for a table"
    echo "  .ask <question>  Get SQL suggestion"
    echo "  .stats           Show learning statistics"
    echo "  .save            Save knowledge"
    echo "  .quit            Exit"
    echo ""
    echo "Example queries to try:"
    echo "  SELECT * FROM employees LIMIT 5"
    echo "  SELECT * FROM v_employee_hierarchy"
    echo "  .ask show me orders with customer names"
}

# Main
case "${1:-}" in
    --simple)
        run_simple
        ;;
    --stop)
        stop_database
        ;;
    --reset)
        reset_database
        ;;
    --help|-h)
        show_help
        ;;
    *)
        run_with_tmux
        ;;
esac
