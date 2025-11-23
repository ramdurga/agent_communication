# Advanced SQL Learning Agent

An intelligent SQL agent that learns from experience, auto-corrects errors, and persists knowledge across sessions. Uses a real PostgreSQL database with complex parent-child relationships.

## Features

- **Real PostgreSQL database** with Docker
- **Auto-correction** of SQL errors
- **Learning from failures** - remembers what went wrong
- **Pattern recognition** - learns successful query patterns
- **Schema awareness** - understands table relationships
- **Interactive mode** - ask questions in natural language
- **Persistent knowledge** - saves learning to JSON
- **Tmux multi-pane view** - see everything at once

## Quick Start

```bash
# Start everything (database + agent + psql)
./run_sql_agent.sh

# Or simple mode (no tmux)
./run_sql_agent.sh --simple

# Stop database
./run_sql_agent.sh --stop

# Reset database (delete all data)
./run_sql_agent.sh --reset
```

## Database Schema

The sample database has realistic parent-child relationships:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATABASE SCHEMA                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DEPARTMENTS ──┬──> EMPLOYEES ──> EMPLOYEES (self-ref: manager) │
│                │                                                │
│  CATEGORIES ───┼──> CATEGORIES (self-ref: subcategories)        │
│                └──> PRODUCTS                                    │
│                                                                 │
│  CUSTOMERS ────┬──> ADDRESSES                                   │
│                └──> ORDERS ──> ORDER_ITEMS ──> PRODUCTS         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tables

| Table | Description | Relationships |
|-------|-------------|---------------|
| `departments` | Company departments | Parent of employees |
| `employees` | Employee records | Belongs to department, has manager (self-ref) |
| `categories` | Product categories | Has subcategories (self-ref) |
| `products` | Product catalog | Belongs to category |
| `customers` | Customer records | Has addresses, orders |
| `addresses` | Shipping/billing addresses | Belongs to customer |
| `orders` | Customer orders | Has order items |
| `order_items` | Order line items | Links order to product |

### Pre-built Views

- `v_employee_hierarchy` - Employees with their managers
- `v_order_summary` - Orders with customer names and totals
- `v_category_tree` - Category hierarchy with paths

## Interactive Commands

```
🔷 SQL> .tables          # Show available tables
🔷 SQL> .columns orders  # Show columns for 'orders' table
🔷 SQL> .ask employees with their managers  # Get SQL suggestion
🔷 SQL> .stats           # Show learning statistics
🔷 SQL> .schema          # Show known schema
🔷 SQL> .save            # Save knowledge
🔷 SQL> .quit            # Exit
```

## Auto-Correction Examples

The agent learns from mistakes and auto-corrects:

```
🔷 SQL> SELECT * FROM employes
  🔧 Auto-correcting: Fixed table name: employes
  ✅ Success (12 rows, 5.2ms)

🔷 SQL> SELECT * FROM orders JOIN customers ON customer_id = id
  🔧 Auto-correcting: Fixed ambiguous column: customer_id
  ✅ Success (13 rows, 8.1ms)

🔷 SQL> SLECT * FROM products
  🔧 Auto-correcting: Fixed syntax error
  ✅ Success (18 rows, 4.3ms)
```

## Learning Persistence

Knowledge is saved to `sql_knowledge_advanced.json`:

```json
{
  "schema": {
    "employees": [
      {"name": "id", "type": "integer"},
      {"name": "first_name", "type": "character varying"}
    ]
  },
  "patterns": {
    "SELECT_JOIN": {
      "success_count": 15,
      "failure_count": 2,
      "examples": ["SELECT * FROM orders JOIN customers..."]
    }
  },
  "corrections": {
    "table_not_found:employes": {
      "corrected_example": "SELECT * FROM employees",
      "count": 3
    }
  },
  "common_joins": {
    "orders:customers": "orders.customer_id = customers.id"
  },
  "stats": {
    "total_queries": 50,
    "successful": 45,
    "auto_corrected": 8
  }
}
```

## Example Queries to Try

### Basic Queries
```sql
SELECT * FROM employees LIMIT 5;
SELECT * FROM products WHERE price > 500;
SELECT COUNT(*) FROM orders WHERE status = 'delivered';
```

### Parent-Child Queries
```sql
-- Employees with their managers (self-referential)
SELECT * FROM v_employee_hierarchy;

-- Categories with subcategories
SELECT * FROM v_category_tree;

-- Orders with items and products
SELECT o.id, c.first_name, p.name, oi.quantity
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id;
```

### Aggregations
```sql
-- Sales by customer
SELECT c.first_name, c.last_name,
       COUNT(o.id) as orders,
       SUM(o.total_amount) as total
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.first_name, c.last_name
ORDER BY total DESC;

-- Products by category
SELECT cat.name, COUNT(p.id) as products, AVG(p.price) as avg_price
FROM categories cat
JOIN products p ON cat.id = p.category_id
GROUP BY cat.name;
```

## Tmux Pane Layout

When running with tmux, you get this view:

```
┌─────────────────────────────┬────────────────────────────────┐
│                             │                                │
│   Database Logs             │   SQL Learning Agent           │
│   (PostgreSQL output)       │   (Interactive mode)           │
│                             │                                │
│                             │   🔷 SQL> _                    │
│                             │                                │
├─────────────────────────────┼────────────────────────────────┤
│                             │                                │
│   Help/Status               │   psql Shell                   │
│                             │   (Direct database access)     │
│                             │                                │
└─────────────────────────────┴────────────────────────────────┘

Navigation: Ctrl+B then arrow keys
Detach: Ctrl+B then d
```

## Files

```
Agent_RL/
├── advanced_sql_agent.py      # Main agent code
├── run_sql_agent.sh           # Setup and run script
├── sql_knowledge_advanced.json # Saved learning (auto-generated)
├── docker/
│   ├── docker-compose.yml     # PostgreSQL container config
│   └── init.sql               # Database schema and sample data
└── README.md
```

## Requirements

- Docker
- Python 3.8+
- psycopg2-binary (auto-installed)
- tmux (optional, for multi-pane view)

## How Learning Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQL LEARNING LOOP                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. USER QUERY                                                  │
│     └── "SELECT * FROM employes"                                │
│                                                                 │
│  2. EXECUTE                                                     │
│     └── Error: relation "employes" does not exist               │
│                                                                 │
│  3. AUTO-CORRECT                                                │
│     ├── Check learned corrections                               │
│     ├── Find similar table name: "employees"                    │
│     └── Retry: "SELECT * FROM employees"                        │
│                                                                 │
│  4. SUCCESS                                                     │
│     └── Return results to user                                  │
│                                                                 │
│  5. LEARN                                                       │
│     ├── Store correction: employes -> employees                 │
│     ├── Update pattern success count                            │
│     └── Save to knowledge base                                  │
│                                                                 │
│  6. PERSIST                                                     │
│     └── Save to sql_knowledge_advanced.json                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Extending

### Add Custom Corrections

```python
# In advanced_sql_agent.py
def custom_correction(self, sql, error):
    if "my_special_error" in error:
        return sql.replace("wrong", "right"), "Fixed my special case"
    return None, "No fix"
```

### Add LLM Integration

```python
# Use with Ollama or other LLM
from langchain_ollama import ChatOllama

class LLMSQLAgent(AdvancedSQLAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = ChatOllama(model="llama3.2")

    def natural_language_query(self, question):
        schema = self.knowledge.get_schema_summary()
        prompt = f"""
        Schema: {schema}

        Convert to SQL: {question}

        Return only the SQL query.
        """
        response = self.llm.invoke(prompt)
        return response.content
```
