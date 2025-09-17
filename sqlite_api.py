import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class SQLiteAPIService:
    """Service to handle SQLite database operations"""
    
    def __init__(self, db_path: str = "local_database.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database and sample tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sample tables for testing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT,
                    salary REAL,
                    hire_date DATE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS departments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    budget REAL,
                    manager_id INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    department_id INTEGER,
                    start_date DATE,
                    end_date DATE,
                    status TEXT
                )
            ''')
            
            # Insert sample data if tables are empty
            cursor.execute("SELECT COUNT(*) FROM employees")
            if cursor.fetchone()[0] == 0:
                sample_employees = [
                    (1, 'John Doe', 'Engineering', 75000, '2022-01-15'),
                    (2, 'Jane Smith', 'Marketing', 65000, '2022-03-01'),
                    (3, 'Bob Johnson', 'Engineering', 80000, '2021-11-10'),
                    (4, 'Alice Brown', 'HR', 60000, '2023-02-20'),
                    (5, 'Charlie Davis', 'Engineering', 90000, '2020-08-05')
                ]
                cursor.executemany(
                    "INSERT INTO employees (id, name, department, salary, hire_date) VALUES (?, ?, ?, ?, ?)",
                    sample_employees
                )
                
                sample_departments = [
                    (1, 'Engineering', 500000, 5),
                    (2, 'Marketing', 200000, 2),
                    (3, 'HR', 150000, 4)
                ]
                cursor.executemany(
                    "INSERT INTO departments (id, name, budget, manager_id) VALUES (?, ?, ?, ?)",
                    sample_departments
                )
                
                sample_projects = [
                    (1, 'Web App Redesign', 1, '2024-01-01', '2024-06-30', 'In Progress'),
                    (2, 'Marketing Campaign', 2, '2024-02-01', '2024-04-30', 'Completed'),
                    (3, 'Employee Portal', 1, '2024-03-01', '2024-12-31', 'Planning')
                ]
                cursor.executemany(
                    "INSERT INTO projects (id, name, department_id, start_date, end_date, status) VALUES (?, ?, ?, ?, ?, ?)",
                    sample_projects
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error creating database: {e}")
    
    def list_tables(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """List all tables in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "success": True,
                "tables": tables,
                "count": len(tables)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tables": []
            }
    
    def describe_table(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the structure of a specific table"""
        table_name = params.get("table_name")
        if not table_name:
            return {
                "success": False,
                "error": "table_name parameter is required"
            }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            if not columns:
                return {
                    "success": False,
                    "error": f"Table '{table_name}' does not exist"
                }
            
            # Format column information
            column_info = []
            for col in columns:
                column_info.append({
                    "name": col[1],
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default_value": col[4],
                    "primary_key": bool(col[5])
                })
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "success": True,
                "table_name": table_name,
                "columns": column_info,
                "row_count": row_count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a SQL query and return results"""
        query = params.get("query", "").strip()
        if not query:
            return {
                "success": False,
                "error": "query parameter is required"
            }
        
        # Basic security check - prevent destructive operations in this demo
        destructive_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
        query_upper = query.upper()
        
        # Allow only SELECT queries for safety
        if not query_upper.startswith("SELECT"):
            return {
                "success": False,
                "error": "Only SELECT queries are allowed for security reasons"
            }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            conn.close()
            
            # Format results as list of dictionaries
            formatted_results = []
            for row in results:
                row_dict = {}
                for i, value in enumerate(row):
                    row_dict[column_names[i]] = value
                formatted_results.append(row_dict)
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "row_count": len(formatted_results),
                "columns": column_names
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Return available API tools"""
        return {
            "list_tables": {
                "description": "List all tables in the database",
                "parameters": {}
            },
            "describe_table": {
                "description": "Get detailed information about a specific table",
                "parameters": {
                    "table_name": "Name of the table to describe"
                }
            },
            "execute_query": {
                "description": "Execute a SELECT query on the database",
                "parameters": {
                    "query": "SQL SELECT query to execute"
                }
            }
        }
    
    async def execute_specific_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a specific tool by name"""
        tools = {
            "list_tables": self.list_tables,
            "describe_table": self.describe_table,
            "execute_query": self.execute_query
        }
        
        if tool_name not in tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
            }
        
        try:
            return tools[tool_name](params)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing {tool_name}: {str(e)}"
            }
