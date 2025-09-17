from typing import Dict, Any, List, Optional, Tuple
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import asyncio
import json
import re
from enum import Enum

# Import our SQLite API service (assuming it's in the same directory)
from sqlite_api_service import SQLiteAPIService

# Mock logger for demonstration
class Logger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")

logger = Logger()

# Mock config for demonstration
class MockConfig:
    def __init__(self):
        self.model = "gpt-4"
        self.azure_endpoint = "your-azure-endpoint"
        self.api_key = "your-api-key"
        self.api_version = "2024-02-01"

class MockConfigNodes:
    @staticmethod
    def get_llm_config(config_name):
        return MockConfig()

# Use mock config
ConfigNodes = MockConfigNodes()


class ExecutionMode(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"


class APIStep:
    """Represents a single API call step in execution plan"""
    def __init__(self, api_name: str, params: Dict[str, Any], depends_on: List[str] = None):
        self.api_name = api_name
        self.params = params
        self.depends_on = depends_on or []  # List of step IDs this depends on
        self.step_id = f"{api_name}_{hash(str(params))}"


class ExecutionPlan:
    """Represents the execution plan with dependencies"""
    def __init__(self):
        self.steps: Dict[str, APIStep] = {}
        self.execution_order: List[List[str]] = []  # Groups of steps that can run in parallel
    
    def add_step(self, step: APIStep):
        self.steps[step.step_id] = step
    
    def build_execution_order(self):
        """Build execution order respecting dependencies"""
        remaining_steps = set(self.steps.keys())
        completed_steps = set()
        self.execution_order = []
        
        while remaining_steps:
            # Find steps with no unmet dependencies
            ready_steps = []
            for step_id in remaining_steps:
                step = self.steps[step_id]
                if all(dep_id in completed_steps for dep_id in step.depends_on):
                    ready_steps.append(step_id)
            
            if not ready_steps:
                raise ValueError("Circular dependency detected in execution plan")
            
            self.execution_order.append(ready_steps)
            remaining_steps -= set(ready_steps)
            completed_steps.update(ready_steps)


class EnhancedLanggraphSQLiteService:
    """Enhanced service with SQLite API chaining capabilities"""

    def __init__(self, db_path: str = "local_database.db"):
        # Initialize LLM (you can replace with a simple mock for testing)
        try:
            self.llm_config = ConfigNodes.get_llm_config("azure_openai")
            self.llm = AzureChatOpenAI(
                azure_deployment=self.llm_config.model,
                azure_endpoint=self.llm_config.azure_endpoint,
                api_key=self.llm_config.api_key,
                api_version=self.llm_config.api_version,
                temperature=0,
            )
        except:
            # Fallback to mock LLM for testing
            self.llm = MockLLM()
            
        self.sqlite_service = SQLiteAPIService(db_path)
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create enhanced LangGraph workflow with API chaining"""

        class AgentState(Dict):
            messages: List[Any]
            analysis: str
            execution_plan: Optional[ExecutionPlan]
            execution_mode: ExecutionMode
            step_results: Dict[str, Any]  # Results keyed by step_id
            final_response: str

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("create_execution_plan", self._create_execution_plan)
        workflow.add_node("execute_plan", self._execute_plan)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("fallback_response", self._fallback_response)

        # Add edges
        workflow.add_edge("analyze_query", "create_execution_plan")
        workflow.add_conditional_edges(
            "create_execution_plan",
            lambda state: "execute_plan" if state.get("execution_plan") else "fallback_response",
            {"execute_plan": "execute_plan", "fallback_response": "fallback_response"},
        )
        workflow.add_edge("execute_plan", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        workflow.add_edge("fallback_response", END)

        workflow.set_entry_point("analyze_query")
        return workflow.compile()

    # -----------------------
    # Node implementations
    # -----------------------

    async def _analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analysis to detect API chaining needs"""
        query = state.get("messages", [])[-1]
        available_apis = self.sqlite_service.get_available_tools()

        prompt = f"""You are an advanced query analyzer for SQLite database operations.

Available APIs:
- list_tables: List all tables in the database
- describe_table: Get table structure and information
- execute_query: Execute SELECT queries

User query: {query}

Analyze this query and determine:
1. What database operations are needed?
2. Is this PARALLEL (independent operations) or SEQUENTIAL (one depends on another)?
3. What data flows between operations?

Example patterns:
- "Show me all tables" -> PARALLEL, single API
- "Describe the employees table" -> PARALLEL, single API  
- "List tables, then show me the structure of the first one" -> SEQUENTIAL
- "Get employee data and department data separately" -> PARALLEL, multiple APIs
"""

        try:
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are a database query analyzer."),
                    HumanMessage(content=prompt)
                ])
                analysis = response.content
            else:
                # Mock response for testing
                analysis = f"Simple query requiring database operation for: {query}"
                
            state["analysis"] = analysis
            logger.info(f"Query analysis: {analysis}")
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            state["analysis"] = f"Simple database query: {query}"
            
        return state

    async def _create_execution_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan with dependencies"""
        query = state.get("messages", [])[-1].lower()
        analysis = state.get("analysis", "")
        
        # Simple rule-based plan creation for common patterns
        execution_plan = ExecutionPlan()
        
        try:
            if "list tables" in query or "show tables" in query or "all tables" in query:
                step = APIStep("list_tables", {})
                execution_plan.add_step(step)
                
            elif "describe" in query or "structure" in query:
                # Extract table name
                table_name = self._extract_table_name(query)
                if table_name:
                    step = APIStep("describe_table", {"table_name": table_name})
                    execution_plan.add_step(step)
                else:
                    # First get tables, then describe
                    step1 = APIStep("list_tables", {})
                    step2 = APIStep("describe_table", {"table_name": "${list_tables.first_table}"}, ["list_tables"])
                    execution_plan.add_step(step1)
                    execution_plan.add_step(step2)
                    
            elif "select" in query or "get" in query or "find" in query or "show" in query:
                # Try to create a SELECT query
                sql_query = self._extract_or_build_query(query)
                step = APIStep("execute_query", {"query": sql_query})
                execution_plan.add_step(step)
                
            else:
                # Complex query - try to use LLM for planning
                plan_data = await self._llm_create_plan(query, analysis)
                if plan_data:
                    for step_data in plan_data.get("steps", []):
                        step = APIStep(
                            api_name=step_data["api_name"],
                            params=step_data["params"],
                            depends_on=step_data.get("depends_on", [])
                        )
                        execution_plan.add_step(step)
            
            if execution_plan.steps:
                execution_plan.build_execution_order()
                state["execution_plan"] = execution_plan
                state["execution_mode"] = ExecutionMode.SEQUENTIAL if len(execution_plan.execution_order) > 1 else ExecutionMode.PARALLEL
                logger.info(f"Created execution plan with {len(execution_plan.steps)} steps")
            else:
                state["execution_plan"] = None
                
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            state["execution_plan"] = None

        return state

    def _extract_table_name(self, query: str) -> Optional[str]:
        """Extract table name from query"""
        # Look for common patterns
        patterns = [
            r"describe\s+(\w+)",
            r"structure\s+of\s+(\w+)",
            r"(\w+)\s+table",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_or_build_query(self, query: str) -> str:
        """Extract SQL query or build a simple one"""
        # If it already looks like SQL, return it
        if "select" in query.lower() and "from" in query.lower():
            return query
        
        # Try to build a simple query
        if "employee" in query.lower():
            return "SELECT * FROM employees LIMIT 10"
        elif "department" in query.lower():
            return "SELECT * FROM departments LIMIT 10"
        elif "project" in query.lower():
            return "SELECT * FROM projects LIMIT 10"
        else:
            return "SELECT name FROM sqlite_master WHERE type='table'"

    async def _llm_create_plan(self, query: str, analysis: str) -> Optional[Dict]:
        """Use LLM to create complex execution plans"""
        try:
            prompt = f"""Create a JSON execution plan for this database query:

Query: {query}
Analysis: {analysis}

Available APIs:
- list_tables (no params)
- describe_table (table_name required)
- execute_query (query required)

Return JSON like:
{{
    "mode": "sequential",
    "steps": [
        {{"api_name": "list_tables", "params": {{}}, "depends_on": []}},
        {{"api_name": "describe_table", "params": {{"table_name": "${{list_tables.first_table}}"}}, "depends_on": ["list_tables"]}}
    ]
}}"""

            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are a database execution planner."),
                    HumanMessage(content=prompt)
                ])
                return self._parse_execution_plan(response.content)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM plan creation: {e}")
            return None

    def _parse_execution_plan(self, llm_response: str) -> Optional[Dict]:
        """Parse LLM response to extract execution plan"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            logger.error(f"Error parsing execution plan: {e}")
            return None

    async def _execute_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan with proper dependency handling"""
        execution_plan = state.get("execution_plan")
        if not execution_plan:
            return state

        step_results = {}
        
        # Execute steps in dependency order
        for step_group in execution_plan.execution_order:
            if len(step_group) == 1:
                # Single step - execute sequentially
                step_id = step_group[0]
                result = await self._execute_single_step(
                    execution_plan.steps[step_id], 
                    step_results
                )
                step_results[step_id] = result
                logger.info(f"Executed sequential step {step_id}")
                
            else:
                # Multiple steps - execute in parallel
                tasks = []
                for step_id in step_group:
                    task = self._execute_single_step(
                        execution_plan.steps[step_id], 
                        step_results
                    )
                    tasks.append((step_id, task))
                
                # Wait for all parallel tasks
                for step_id, task in tasks:
                    result = await task
                    step_results[step_id] = result
                    logger.info(f"Executed parallel step {step_id}")

        state["step_results"] = step_results
        return state

    async def _execute_single_step(self, step: APIStep, previous_results: Dict[str, Any]) -> Any:
        """Execute a single API step with parameter resolution"""
        try:
            # Resolve parameters with dependencies
            resolved_params = self._resolve_parameters(step.params, previous_results)
            
            # Execute the API call
            result = await self.sqlite_service.execute_specific_tool(
                step.api_name, 
                resolved_params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            return {"error": str(e)}

    def _resolve_parameters(self, params: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters that depend on previous step results"""
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${"):
                # Extract dependency reference: ${step_id.field_name}
                dependency_path = value[2:-1]  # Remove ${ and }
                
                if "." in dependency_path:
                    step_id, field_name = dependency_path.split(".", 1)
                else:
                    step_id = dependency_path
                    field_name = None
                
                # Get value from previous results
                if step_id in previous_results:
                    step_result = previous_results[step_id]
                    if field_name == "first_table" and isinstance(step_result, dict):
                        tables = step_result.get("tables", [])
                        resolved_value = tables[0] if tables else "employees"
                    elif field_name:
                        resolved_value = self._extract_nested_value(step_result, field_name)
                    else:
                        resolved_value = step_result
                    resolved[key] = resolved_value
                else:
                    logger.warning(f"Dependency {step_id} not found for parameter {key}")
                    resolved[key] = value
            else:
                resolved[key] = value
                
        return resolved

    def _extract_nested_value(self, data: Any, field_path: str) -> Any:
        """Extract nested value using dot notation"""
        try:
            fields = field_path.split(".")
            current = data
            for field in fields:
                if isinstance(current, dict):
                    current = current.get(field)
                else:
                    return None
            return current
        except:
            return None

    async def _synthesize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response from all step results"""
        query = state.get("messages", [])[-1]
        step_results = state.get("step_results", {})
        
        # Simple response synthesis
        response_parts = []
        response_parts.append(f"Query: {query}\n")
        
        for step_id, result in step_results.items():
            if isinstance(result, dict):
                if result.get("success"):
                    if "tables" in result:
                        response_parts.append(f"📋 Available tables: {', '.join(result['tables'])}")
                    elif "columns" in result:
                        table_name = result.get("table_name", "Unknown")
                        response_parts.append(f"📊 Table '{table_name}' structure:")
                        for col in result["columns"]:
                            response_parts.append(f"  • {col['name']} ({col['type']})")
                    elif "results" in result:
                        response_parts.append(f"📈 Query results ({result['row_count']} rows):")
                        for row in result["results"][:5]:  # Show first 5 rows
                            response_parts.append(f"  {row}")
                        if result["row_count"] > 5:
                            response_parts.append(f"  ... and {result['row_count'] - 5} more rows")
                else:
                    response_parts.append(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        final_response = "\n".join(response_parts)
        state["final_response"] = final_response
        logger.info("Synthesized response from database operations")
        return state

    async def _fallback_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback with suggestions"""
        query = state.get("messages", [])[-1]
        available_apis = list(self.sqlite_service.get_available_tools().keys())
        
        msg = f"""I couldn't create an execution plan for: '{query}'

Available database operations:
• List all tables in the database
• Describe table structure and information
• Execute SELECT queries on the data

Try queries like:
• "Show me all tables"
• "Describe the employees table"  
• "Get all employees"
• "SELECT * FROM employees WHERE department = 'Engineering'"
"""
        
        state["final_response"] = msg
        logger.info("Used fallback response")
        return state

    # -----------------------
    # Public API
    # -----------------------

    async def process_query_with_chaining(self, user_query: str) -> str:
        """Process query with API chaining support"""
        initial_state = {
            "messages": [user_query],
            "analysis": "",
            "execution_plan": None,
            "execution_mode": ExecutionMode.PARALLEL,
            "step_results": {},
            "final_response": "",
        }
        
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["final_response"]

    # Synchronous wrapper for easier testing
    def process_query(self, user_query: str) -> str:
        """Synchronous wrapper for process_query_with_chaining"""
        import asyncio
        return asyncio.run(self.process_query_with_chaining(user_query))


# Mock LLM for testing without real API
class MockLLM:
    async def ainvoke(self, messages):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        # Simple mock responses based on content
        content = str(messages[-1].content).lower()
        if "execution plan" in content:
            return MockResponse('{"mode": "sequential", "steps": [{"api_name": "list_tables", "params": {}, "depends_on": []}]}')
        else:
            return MockResponse("This query requires database operations to list tables and analyze data.")


# Example usage and testing
if __name__ == "__main__":
    # Create service instance
    service = EnhancedLanggraphSQLiteService()
    
    # Test queries
    test_queries = [
        "Show me all tables",
        "Describe the employees table",
        "Get all employees",
        "SELECT * FROM employees WHERE department = 'Engineering'",
        "List tables then describe the first one"
    ]
    
    print("=== Testing SQLite LangGraph Service ===\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        try:
            response = service.process_query(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)
