#!/usr/bin/env python3
"""
Advanced SQL Learning Agent
============================

A sophisticated SQL agent that:
1. Connects to a real PostgreSQL database
2. Learns from query successes and failures
3. Auto-corrects failed queries using learned patterns
4. Persists knowledge across sessions
5. Supports natural language queries (with LLM integration)

Requirements:
    pip install psycopg2-binary
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    import psycopg2
    from psycopg2 import sql as psql
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Installing psycopg2-binary...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'psycopg2-binary'])
    import psycopg2
    from psycopg2 import sql as psql
    from psycopg2.extras import RealDictCursor


@dataclass
class QueryResult:
    """Result of a SQL query execution."""
    success: bool
    sql: str
    rows: List[Dict] = field(default_factory=list)
    row_count: int = 0
    error: str = ""
    execution_time_ms: float = 0
    corrected: bool = False
    original_sql: str = ""


@dataclass
class SQLPattern:
    """A learned SQL pattern."""
    pattern_type: str  # SELECT, INSERT, UPDATE, DELETE, JOIN, etc.
    template: str
    success_count: int = 0
    failure_count: int = 0
    last_used: str = ""
    examples: List[str] = field(default_factory=list)
    common_errors: List[str] = field(default_factory=list)


class SQLKnowledgeBase:
    """
    Persistent knowledge base for SQL learning.
    Stores patterns, corrections, schema info, and query history.
    """

    def __init__(self, filepath: str = "sql_knowledge_advanced.json"):
        self.filepath = filepath
        self.knowledge = {
            "schema": {},           # table_name -> columns
            "patterns": {},         # pattern_type -> SQLPattern
            "corrections": {},      # error_pattern -> correction
            "successful_queries": [],
            "failed_queries": [],
            "stats": {
                "total_queries": 0,
                "successful": 0,
                "failed": 0,
                "auto_corrected": 0
            },
            "common_joins": {},     # table_pair -> join_condition
            "column_types": {},     # table.column -> type
            "last_updated": None
        }
        self.load()

    def load(self):
        """Load knowledge from file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.knowledge = json.load(f)
                print(f"📚 Loaded knowledge base ({self.knowledge['stats']['total_queries']} queries)")
            except Exception as e:
                print(f"⚠️ Could not load knowledge: {e}")

    def save(self):
        """Save knowledge to file."""
        self.knowledge["last_updated"] = datetime.now().isoformat()
        with open(self.filepath, 'w') as f:
            json.dump(self.knowledge, f, indent=2, default=str)

    def learn_schema(self, table: str, columns: List[Dict]):
        """Learn table schema."""
        self.knowledge["schema"][table] = columns
        for col in columns:
            key = f"{table}.{col['name']}"
            self.knowledge["column_types"][key] = col.get("type", "unknown")

    def learn_success(self, sql: str, pattern_type: str):
        """Learn from a successful query."""
        self.knowledge["stats"]["total_queries"] += 1
        self.knowledge["stats"]["successful"] += 1

        # Store in patterns
        if pattern_type not in self.knowledge["patterns"]:
            self.knowledge["patterns"][pattern_type] = {
                "success_count": 0,
                "failure_count": 0,
                "examples": [],
                "common_errors": []
            }

        pattern = self.knowledge["patterns"][pattern_type]
        pattern["success_count"] += 1
        pattern["last_used"] = datetime.now().isoformat()

        # Keep last 20 examples
        if sql not in pattern["examples"]:
            pattern["examples"].append(sql)
            pattern["examples"] = pattern["examples"][-20:]

        # Store in successful queries
        self.knowledge["successful_queries"].append({
            "sql": sql,
            "pattern": pattern_type,
            "timestamp": datetime.now().isoformat()
        })
        self.knowledge["successful_queries"] = self.knowledge["successful_queries"][-100:]

    def learn_failure(self, sql: str, error: str, pattern_type: str):
        """Learn from a failed query."""
        self.knowledge["stats"]["total_queries"] += 1
        self.knowledge["stats"]["failed"] += 1

        if pattern_type not in self.knowledge["patterns"]:
            self.knowledge["patterns"][pattern_type] = {
                "success_count": 0,
                "failure_count": 0,
                "examples": [],
                "common_errors": []
            }

        pattern = self.knowledge["patterns"][pattern_type]
        pattern["failure_count"] += 1

        # Track common errors
        error_key = self._extract_error_pattern(error)
        if error_key not in pattern["common_errors"]:
            pattern["common_errors"].append(error_key)
            pattern["common_errors"] = pattern["common_errors"][-10:]

        # Store in failed queries
        self.knowledge["failed_queries"].append({
            "sql": sql,
            "error": error,
            "pattern": pattern_type,
            "timestamp": datetime.now().isoformat()
        })
        self.knowledge["failed_queries"] = self.knowledge["failed_queries"][-50:]

    def learn_correction(self, error_pattern: str, original_sql: str, corrected_sql: str):
        """Learn a correction for an error pattern."""
        self.knowledge["stats"]["auto_corrected"] += 1
        self.knowledge["corrections"][error_pattern] = {
            "original_example": original_sql,
            "corrected_example": corrected_sql,
            "count": self.knowledge["corrections"].get(error_pattern, {}).get("count", 0) + 1
        }

    def learn_join(self, table1: str, table2: str, condition: str):
        """Learn a join relationship."""
        key = f"{table1}:{table2}"
        self.knowledge["common_joins"][key] = condition
        # Also store reverse
        key_rev = f"{table2}:{table1}"
        self.knowledge["common_joins"][key_rev] = condition

    def get_join_condition(self, table1: str, table2: str) -> Optional[str]:
        """Get learned join condition for two tables."""
        return self.knowledge["common_joins"].get(f"{table1}:{table2}")

    def get_correction(self, error: str) -> Optional[str]:
        """Get a learned correction for an error."""
        error_pattern = self._extract_error_pattern(error)
        correction = self.knowledge["corrections"].get(error_pattern)
        if correction:
            return correction.get("corrected_example")
        return None

    def get_similar_successful_query(self, failed_sql: str) -> Optional[str]:
        """Find a similar successful query."""
        failed_words = set(failed_sql.upper().split())

        best_match = None
        best_score = 0

        for query in self.knowledge["successful_queries"]:
            query_words = set(query["sql"].upper().split())
            score = len(failed_words & query_words) / max(len(failed_words | query_words), 1)
            if score > best_score and score > 0.3:
                best_score = score
                best_match = query["sql"]

        return best_match

    def _extract_error_pattern(self, error: str) -> str:
        """Extract a generalizable pattern from an error."""
        error = error.lower()

        if "column" in error and "does not exist" in error:
            match = re.search(r'column "(\w+)"', error)
            if match:
                return f"column_not_found:{match.group(1)}"
            return "column_not_found"

        if "relation" in error and "does not exist" in error:
            match = re.search(r'relation "(\w+)"', error)
            if match:
                return f"table_not_found:{match.group(1)}"
            return "table_not_found"

        if "syntax error" in error:
            return "syntax_error"

        if "permission denied" in error:
            return "permission_denied"

        if "duplicate key" in error:
            return "duplicate_key"

        if "violates" in error and "constraint" in error:
            return "constraint_violation"

        if "ambiguous" in error:
            return "ambiguous_column"

        return "unknown_error"

    def get_schema_summary(self) -> str:
        """Get a summary of known schema."""
        lines = ["Known Tables:"]
        for table, columns in self.knowledge["schema"].items():
            col_names = [c["name"] for c in columns[:5]]
            if len(columns) > 5:
                col_names.append("...")
            lines.append(f"  {table}: {', '.join(col_names)}")
        return "\n".join(lines)

    def print_stats(self):
        """Print learning statistics."""
        stats = self.knowledge["stats"]
        print("\n" + "=" * 50)
        print("📊 SQL LEARNING STATISTICS")
        print("=" * 50)
        print(f"Total Queries:    {stats['total_queries']}")
        print(f"Successful:       {stats['successful']} ({stats['successful']/max(stats['total_queries'],1)*100:.1f}%)")
        print(f"Failed:           {stats['failed']}")
        print(f"Auto-Corrected:   {stats['auto_corrected']}")
        print(f"Tables Learned:   {len(self.knowledge['schema'])}")
        print(f"Patterns Learned: {len(self.knowledge['patterns'])}")
        print(f"Joins Learned:    {len(self.knowledge['common_joins'])//2}")


class SQLAutoCorrector:
    """
    Automatically corrects common SQL errors using learned patterns.
    """

    def __init__(self, knowledge: SQLKnowledgeBase):
        self.knowledge = knowledge

    def correct(self, sql: str, error: str) -> Tuple[Optional[str], str]:
        """
        Try to correct a failed SQL query.

        Returns:
            (corrected_sql, correction_reason) or (None, reason_for_failure)
        """
        error_lower = error.lower()

        # Try learned corrections first
        learned = self.knowledge.get_correction(error)
        if learned:
            return learned, "Applied learned correction"

        # Column not found
        if "column" in error_lower and "does not exist" in error_lower:
            match = re.search(r'column "(\w+)"', error_lower)
            if match:
                bad_column = match.group(1)
                corrected = self._fix_column_name(sql, bad_column)
                if corrected:
                    return corrected, f"Fixed column name: {bad_column}"

        # Table not found
        if "relation" in error_lower and "does not exist" in error_lower:
            match = re.search(r'relation "(\w+)"', error_lower)
            if match:
                bad_table = match.group(1)
                corrected = self._fix_table_name(sql, bad_table)
                if corrected:
                    return corrected, f"Fixed table name: {bad_table}"

        # Ambiguous column - add table prefix
        if "ambiguous" in error_lower:
            match = re.search(r'column reference "(\w+)" is ambiguous', error_lower)
            if match:
                ambiguous_col = match.group(1)
                corrected = self._fix_ambiguous_column(sql, ambiguous_col)
                if corrected:
                    return corrected, f"Fixed ambiguous column: {ambiguous_col}"

        # Missing quotes in string
        if "syntax error" in error_lower:
            corrected = self._fix_syntax_errors(sql)
            if corrected and corrected != sql:
                return corrected, "Fixed syntax error"

        # Try to find similar successful query
        similar = self.knowledge.get_similar_successful_query(sql)
        if similar:
            return similar, "Suggested similar successful query"

        return None, "Could not auto-correct"

    def _fix_column_name(self, sql: str, bad_column: str) -> Optional[str]:
        """Try to fix a column name error."""
        # Look for similar column names in schema
        for table, columns in self.knowledge.knowledge["schema"].items():
            for col in columns:
                col_name = col["name"].lower()
                if self._similar(bad_column.lower(), col_name):
                    return sql.replace(bad_column, col["name"])
        return None

    def _fix_table_name(self, sql: str, bad_table: str) -> Optional[str]:
        """Try to fix a table name error."""
        for table in self.knowledge.knowledge["schema"].keys():
            if self._similar(bad_table.lower(), table.lower()):
                return re.sub(rf'\b{bad_table}\b', table, sql, flags=re.IGNORECASE)
        return None

    def _fix_ambiguous_column(self, sql: str, column: str) -> Optional[str]:
        """Add table prefix to ambiguous column."""
        # Find which tables have this column
        tables_with_col = []
        for table, columns in self.knowledge.knowledge["schema"].items():
            if any(c["name"].lower() == column.lower() for c in columns):
                tables_with_col.append(table)

        if len(tables_with_col) == 1:
            # Replace column with table.column
            pattern = rf'\b{column}\b(?!\s*\.)'
            return re.sub(pattern, f"{tables_with_col[0]}.{column}", sql, count=1)

        return None

    def _fix_syntax_errors(self, sql: str) -> Optional[str]:
        """Try to fix common syntax errors."""
        corrected = sql

        # Fix common typos
        typos = {
            "FORM": "FROM",
            "SLECT": "SELECT",
            "WEHRE": "WHERE",
            "GRUOP": "GROUP",
            "ODRER": "ORDER",
            "LIMT": "LIMIT",
        }
        for typo, fix in typos.items():
            corrected = re.sub(rf'\b{typo}\b', fix, corrected, flags=re.IGNORECASE)

        return corrected if corrected != sql else None

    def _similar(self, s1: str, s2: str) -> bool:
        """Check if two strings are similar (simple edit distance check)."""
        if s1 == s2:
            return True
        if abs(len(s1) - len(s2)) > 2:
            return False

        # Check if one is substring of other
        if s1 in s2 or s2 in s1:
            return True

        # Check character overlap
        common = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        return common >= min(len(s1), len(s2)) - 1


class AdvancedSQLAgent:
    """
    Advanced SQL Learning Agent with auto-correction capabilities.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "learning_db",
        user: str = "agent",
        password: str = "agent123",
        knowledge_file: str = "sql_knowledge_advanced.json"
    ):
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
        self.conn = None
        self.knowledge = SQLKnowledgeBase(knowledge_file)
        self.corrector = SQLAutoCorrector(self.knowledge)
        self.max_retries = 3

    def connect(self) -> bool:
        """Connect to the database."""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            print(f"✅ Connected to {self.conn_params['database']}@{self.conn_params['host']}")
            self._learn_schema()
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from database."""
        if self.conn:
            self.conn.close()
            self.knowledge.save()
            print("📁 Knowledge saved. Disconnected.")

    def _learn_schema(self):
        """Learn the database schema."""
        if not self.conn:
            return

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all tables
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                """)
                tables = [row['table_name'] for row in cur.fetchall()]

                # Get columns for each table
                for table in tables:
                    cur.execute("""
                        SELECT column_name as name, data_type as type, is_nullable
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                    """, (table,))
                    columns = [dict(row) for row in cur.fetchall()]
                    self.knowledge.learn_schema(table, columns)

                # Learn foreign key relationships (for joins)
                cur.execute("""
                    SELECT
                        tc.table_name,
                        kcu.column_name,
                        ccu.table_name AS foreign_table,
                        ccu.column_name AS foreign_column
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                """)
                for row in cur.fetchall():
                    join_cond = f"{row['table_name']}.{row['column_name']} = {row['foreign_table']}.{row['foreign_column']}"
                    self.knowledge.learn_join(row['table_name'], row['foreign_table'], join_cond)

                print(f"📖 Learned schema: {len(tables)} tables")

        except Exception as e:
            print(f"⚠️ Could not learn schema: {e}")

    def _classify_query(self, sql: str) -> str:
        """Classify the query type."""
        sql_upper = sql.strip().upper()

        if sql_upper.startswith("SELECT"):
            if " JOIN " in sql_upper:
                return "SELECT_JOIN"
            elif " GROUP BY " in sql_upper:
                return "SELECT_AGGREGATE"
            elif " WHERE " in sql_upper:
                return "SELECT_WHERE"
            else:
                return "SELECT_SIMPLE"
        elif sql_upper.startswith("INSERT"):
            return "INSERT"
        elif sql_upper.startswith("UPDATE"):
            return "UPDATE"
        elif sql_upper.startswith("DELETE"):
            return "DELETE"
        elif sql_upper.startswith("CREATE"):
            return "CREATE"
        else:
            return "OTHER"

    def execute(self, sql: str, auto_correct: bool = True) -> QueryResult:
        """
        Execute a SQL query with optional auto-correction.
        """
        if not self.conn:
            return QueryResult(success=False, sql=sql, error="Not connected to database")

        original_sql = sql
        attempts = 0
        last_error = ""

        while attempts < self.max_retries:
            attempts += 1

            start_time = time.time()
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql)

                    # Handle SELECT vs other queries
                    if sql.strip().upper().startswith("SELECT"):
                        rows = [dict(row) for row in cur.fetchall()]
                        row_count = len(rows)
                    else:
                        self.conn.commit()
                        rows = []
                        row_count = cur.rowcount

                    execution_time = (time.time() - start_time) * 1000

                    # Learn from success
                    pattern = self._classify_query(sql)
                    self.knowledge.learn_success(sql, pattern)

                    if sql != original_sql:
                        self.knowledge.learn_correction(
                            self.knowledge._extract_error_pattern(last_error),
                            original_sql,
                            sql
                        )

                    return QueryResult(
                        success=True,
                        sql=sql,
                        rows=rows,
                        row_count=row_count,
                        execution_time_ms=execution_time,
                        corrected=(sql != original_sql),
                        original_sql=original_sql if sql != original_sql else ""
                    )

            except Exception as e:
                self.conn.rollback()
                last_error = str(e)
                execution_time = (time.time() - start_time) * 1000

                if auto_correct and attempts < self.max_retries:
                    # Try to auto-correct
                    corrected, reason = self.corrector.correct(sql, last_error)
                    if corrected and corrected != sql:
                        print(f"  🔧 Auto-correcting (attempt {attempts}): {reason}")
                        sql = corrected
                        continue

                # Learn from failure
                pattern = self._classify_query(sql)
                self.knowledge.learn_failure(sql, last_error, pattern)

                return QueryResult(
                    success=False,
                    sql=sql,
                    error=last_error,
                    execution_time_ms=execution_time,
                    original_sql=original_sql if sql != original_sql else ""
                )

        return QueryResult(
            success=False,
            sql=sql,
            error=f"Max retries exceeded. Last error: {last_error}",
            original_sql=original_sql
        )

    def query(self, sql: str) -> QueryResult:
        """Execute a query and print results nicely."""
        print(f"\n📝 SQL: {sql}")

        result = self.execute(sql)

        if result.success:
            if result.corrected:
                print(f"  ⚡ Auto-corrected from: {result.original_sql}")
            print(f"  ✅ Success ({result.row_count} rows, {result.execution_time_ms:.1f}ms)")

            if result.rows:
                self._print_table(result.rows[:10])  # Show first 10 rows
                if result.row_count > 10:
                    print(f"  ... and {result.row_count - 10} more rows")
        else:
            print(f"  ❌ Error: {result.error}")

        return result

    def _print_table(self, rows: List[Dict]):
        """Print rows as a formatted table."""
        if not rows:
            return

        # Get column widths
        headers = list(rows[0].keys())
        widths = {h: len(str(h)) for h in headers}
        for row in rows:
            for h in headers:
                widths[h] = max(widths[h], len(str(row.get(h, ""))[:30]))

        # Print header
        header_line = " | ".join(str(h).ljust(widths[h])[:30] for h in headers)
        print(f"  {header_line}")
        print(f"  {'-' * len(header_line)}")

        # Print rows
        for row in rows:
            row_line = " | ".join(str(row.get(h, "")).ljust(widths[h])[:30] for h in headers)
            print(f"  {row_line}")

    def suggest_query(self, intent: str) -> str:
        """Suggest a query based on intent (simplified NL processing)."""
        intent_lower = intent.lower()

        # Pattern matching for common intents
        if "employee" in intent_lower:
            if "manager" in intent_lower or "hierarchy" in intent_lower:
                return "SELECT * FROM v_employee_hierarchy"
            elif "salary" in intent_lower or "highest" in intent_lower or "top" in intent_lower:
                return "SELECT * FROM employees ORDER BY salary DESC LIMIT 10"
            elif "department" in intent_lower:
                return """
                SELECT d.name as department, COUNT(*) as employee_count, AVG(e.salary) as avg_salary
                FROM employees e
                JOIN departments d ON e.department_id = d.id
                GROUP BY d.name
                """
            else:
                return "SELECT * FROM employees LIMIT 10"

        elif "order" in intent_lower:
            if "customer" in intent_lower:
                return """
                SELECT c.first_name, c.last_name, COUNT(o.id) as order_count, SUM(o.total_amount) as total_spent
                FROM customers c
                JOIN orders o ON c.id = o.customer_id
                GROUP BY c.id, c.first_name, c.last_name
                ORDER BY total_spent DESC
                """
            elif "item" in intent_lower or "product" in intent_lower:
                return """
                SELECT o.id as order_id, p.name as product, oi.quantity, oi.subtotal
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                JOIN products p ON oi.product_id = p.id
                ORDER BY o.id
                """
            else:
                return "SELECT * FROM v_order_summary LIMIT 10"

        elif "product" in intent_lower:
            if "category" in intent_lower:
                return """
                SELECT c.name as category, COUNT(p.id) as product_count, AVG(p.price) as avg_price
                FROM products p
                JOIN categories c ON p.category_id = c.id
                GROUP BY c.name
                """
            elif "stock" in intent_lower or "inventory" in intent_lower:
                return "SELECT name, price, stock_quantity FROM products ORDER BY stock_quantity"
            else:
                return "SELECT * FROM products LIMIT 10"

        elif "category" in intent_lower:
            if "tree" in intent_lower or "hierarchy" in intent_lower:
                return "SELECT * FROM v_category_tree"
            else:
                return "SELECT * FROM categories"

        elif "customer" in intent_lower:
            if "address" in intent_lower:
                return """
                SELECT c.first_name, c.last_name, a.address_type, a.street, a.city, a.state
                FROM customers c
                JOIN addresses a ON c.id = a.customer_id
                """
            else:
                return "SELECT * FROM customers LIMIT 10"

        elif "join" in intent_lower:
            # Show available joins
            joins = list(self.knowledge.knowledge["common_joins"].items())[:5]
            if joins:
                table1, cond = joins[0]
                tables = table1.split(":")
                return f"SELECT * FROM {tables[0]} JOIN {tables[1]} ON {cond} LIMIT 10"

        # Default: show tables
        return "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"

    def get_tables(self) -> List[str]:
        """Get list of available tables."""
        return list(self.knowledge.knowledge["schema"].keys())

    def get_columns(self, table: str) -> List[str]:
        """Get columns for a table."""
        cols = self.knowledge.knowledge["schema"].get(table, [])
        return [c["name"] for c in cols]

    def save(self):
        """Save knowledge base."""
        self.knowledge.save()
        print("💾 Knowledge saved.")


# For interactive use
def interactive_session(agent: AdvancedSQLAgent):
    """Run an interactive SQL session."""
    print("\n" + "=" * 60)
    print("🤖 SQL LEARNING AGENT - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  .tables          - Show available tables")
    print("  .columns <table> - Show columns for a table")
    print("  .ask <question>  - Get SQL suggestion for a question")
    print("  .stats           - Show learning statistics")
    print("  .schema          - Show known schema")
    print("  .save            - Save knowledge")
    print("  .quit            - Exit")
    print("\nOr enter any SQL query directly.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n🔷 SQL> ").strip()

            if not user_input:
                continue

            if user_input.lower() == ".quit":
                break

            elif user_input.lower() == ".tables":
                tables = agent.get_tables()
                print("\n📋 Available tables:")
                for t in tables:
                    cols = agent.get_columns(t)
                    print(f"  {t}: {len(cols)} columns")

            elif user_input.lower().startswith(".columns"):
                parts = user_input.split()
                if len(parts) > 1:
                    table = parts[1]
                    cols = agent.get_columns(table)
                    if cols:
                        print(f"\n📋 Columns in {table}:")
                        schema = agent.knowledge.knowledge["schema"].get(table, [])
                        for col in schema:
                            print(f"  {col['name']}: {col['type']}")
                    else:
                        print(f"Table '{table}' not found")
                else:
                    print("Usage: .columns <table_name>")

            elif user_input.lower().startswith(".ask"):
                question = user_input[4:].strip()
                if question:
                    suggested = agent.suggest_query(question)
                    print(f"\n💡 Suggested SQL:")
                    print(f"   {suggested}")
                    confirm = input("   Execute? (y/n): ").strip().lower()
                    if confirm == 'y':
                        agent.query(suggested)
                else:
                    print("Usage: .ask <your question>")

            elif user_input.lower() == ".stats":
                agent.knowledge.print_stats()

            elif user_input.lower() == ".schema":
                print("\n" + agent.knowledge.get_schema_summary())

            elif user_input.lower() == ".save":
                agent.save()

            else:
                # Execute as SQL
                agent.query(user_input)

        except KeyboardInterrupt:
            print("\n\nUse .quit to exit")
        except Exception as e:
            print(f"Error: {e}")

    agent.disconnect()


if __name__ == "__main__":
    import sys

    # Get connection params from environment or use defaults
    agent = AdvancedSQLAgent(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        database=os.environ.get("PGDATABASE", "learning_db"),
        user=os.environ.get("PGUSER", "agent"),
        password=os.environ.get("PGPASSWORD", "agent123"),
        knowledge_file="/home/ram/Agentic_projects/agent_communication/Agent_RL/sql_knowledge_advanced.json"
    )

    if agent.connect():
        if len(sys.argv) > 1:
            # Execute query from command line
            sql = " ".join(sys.argv[1:])
            agent.query(sql)
            agent.disconnect()
        else:
            # Interactive mode
            interactive_session(agent)
    else:
        print("Could not connect to database. Is PostgreSQL running?")
        print("Start it with: cd docker && docker-compose up -d")
