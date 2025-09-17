from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import asyncio
import json
import re
import time
from enum import Enum

# Import our SQLite API service
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
        self.depends_on = depends_on or []
        self.step_id = f"{api_name}_{hash(str(params))}"
        self.status = "pending"
        self.result = None
        self.duration_ms = None
        self.start_time = None


class ExecutionPlan:
    """Represents the execution plan with dependencies"""
    def __init__(self):
        self.steps: Dict[str, APIStep] = {}
        self.execution_order: List[List[str]] = []
    
    def add_step(self, step: APIStep):
        self.steps[step.step_id] = step
    
    def build_execution_order(self):
        """Build execution order respecting dependencies"""
        remaining_steps = set(self.steps.keys())
        completed_steps = set()
        self.execution_order = []
        
        while remaining_steps:
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
    """Enhanced service with FastAPI integration and orchestration visibility"""

    def __init__(self, db_path: str = "local_database.db"):
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
            self.llm = MockLLM()
            
        self.sqlite_service = SQLiteAPIService(db_path)
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create enhanced LangGraph workflow with orchestration tracking"""

        class AgentState(Dict):
            messages: List[Any]
            analysis: str
            execution_plan: Optional[ExecutionPlan]
            execution_mode: ExecutionMode
            step_results: Dict[str, Any]
            final_response: str
            orchestration_steps: List[Dict[str, Any]]  # For tracking

        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("create_execution_plan", self._create_execution_plan)
        workflow.add_node("execute_plan", self._execute_plan)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("fallback_response", self._fallback_response)

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

    async def _analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analysis with orchestration tracking"""
        query = state.get("messages", [])[-1]
        
        # Add orchestration step
        step = {
            "step_id": "analyze_query",
            "api_name": "query_analysis",
            "params": {"query": query},
            "status": "running",
            "result": None,
            "duration_ms": None
        }
        
        orchestration_steps = state.get("orchestration_steps", [])
        orchestration_steps.append(step)
        state["orchestration_steps"] = orchestration_steps
        
        start_time = time.time()
        
        try:
            # Analysis logic (simplified for demo)
            analysis = f"Query requires database operations: {query}"
            
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            # Update step
            step["status"] = "completed"
            step["result"] = {"analysis": analysis}
            step["duration_ms"] = duration
            
            state["analysis"] = analysis
            logger.info(f"Query analysis completed in {duration:.2f}ms")
            
        except Exception as e:
            step["status"] = "failed"
            step["result"] = {"error": str(e)}
            logger.error(f"Error in analysis: {e}")
            state["analysis"] = f"Simple database query: {query}"
            
        return state

    async def _create_execution_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan with orchestration tracking"""
        query = state.get("messages", [])[-1].lower()
        
        step = {
            "step_id": "create_plan",
            "api_name": "execution_planning",
            "params": {"query": query},
            "status": "running",
            "result": None,
            "duration_ms": None
        }
        
        orchestration_steps = state.get("orchestration_steps", [])
        orchestration_steps.append(step)
        
        start_time = time.time()
        
        try:
            execution_plan = ExecutionPlan()
            
            # Plan creation logic
            if "list tables" in query or "show tables" in query or "all tables" in query:
                api_step = APIStep("list_tables", {})
                execution_plan.add_step(api_step)
                
            elif "describe" in query:
                table_name = self._extract_table_name(query)
                if table_name:
                    api_step = APIStep("describe_table", {"table_name": table_name})
                    execution_plan.add_step(api_step)
                else:
                    step1 = APIStep("list_tables", {})
                    step2 = APIStep("describe_table", {"table_name": "${list_tables.first_table}"}, ["list_tables"])
                    execution_plan.add_step(step1)
                    execution_plan.add_step(step2)
                    
            elif "select" in query or "get" in query or "find" in query or "show" in query:
                sql_query = self._extract_or_build_query(query)
                api_step = APIStep("execute_query", {"query": sql_query})
                execution_plan.add_step(api_step)
            
            if execution_plan.steps:
                execution_plan.build_execution_order()
                state["execution_plan"] = execution_plan
                state["execution_mode"] = ExecutionMode.SEQUENTIAL if len(execution_plan.execution_order) > 1 else ExecutionMode.PARALLEL
                
                end_time = time.time()
                duration = (end_time - start_time) * 1000
                
                step["status"] = "completed"
                step["result"] = {
                    "steps_created": len(execution_plan.steps),
                    "execution_mode": state["execution_mode"].value
                }
                step["duration_ms"] = duration
                
                logger.info(f"Created execution plan with {len(execution_plan.steps)} steps")
            else:
                step["status"] = "failed"
                step["result"] = {"error": "No valid execution plan created"}
                state["execution_plan"] = None
                
        except Exception as e:
            step["status"] = "failed"
            step["result"] = {"error": str(e)}
            logger.error(f"Error creating execution plan: {e}")
            state["execution_plan"] = None

        return state

    async def _execute_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan with detailed orchestration tracking"""
        execution_plan = state.get("execution_plan")
        if not execution_plan:
            return state

        orchestration_steps = state.get("orchestration_steps", [])
        step_results = {}
        
        for step_group in execution_plan.execution_order:
            if len(step_group) == 1:
                # Sequential execution
                step_id = step_group[0]
                api_step = execution_plan.steps[step_id]
                
                # Create orchestration tracking
                orch_step = {
                    "step_id": step_id,
                    "api_name": api_step.api_name,
                    "params": api_step.params,
                    "status": "running",
                    "result": None,
                    "duration_ms": None
                }
                orchestration_steps.append(orch_step)
                
                start_time = time.time()
                
                try:
                    result = await self._execute_single_step(api_step, step_results)
                    step_results[step_id] = result
                    
                    end_time = time.time()
                    duration = (end_time - start_time) * 1000
                    
                    orch_step["status"] = "completed" if result.get("success") else "failed"
                    orch_step["result"] = result
                    orch_step["duration_ms"] = duration
                    
                    logger.info(f"Executed step {step_id} in {duration:.2f}ms")
                    
                except Exception as e:
                    orch_step["status"] = "failed"
                    orch_step["result"] = {"error": str(e)}
                    logger.error(f"Error executing step {step_id}: {e}")
                    
            else:
                # Parallel execution
                tasks = []
                for step_id in step_group:
                    api_step = execution_plan.steps[step_id]
                    
                    orch_step = {
                        "step_id": step_id,
                        "api_name": api_step.api_name,
                        "params": api_step.params,
                        "status": "running",
                        "result": None,
                        "duration_ms": None
                    }
                    orchestration_steps.append(orch_step)
                    
                    task = self._execute_single_step(api_step, step_results)
                    tasks.append((step_id, task, orch_step))
                
                # Execute in parallel
                for step_id, task, orch_step in tasks:
                    start_time = time.time()
                    try:
                        result = await task
                        step_results[step_id] = result
                        
                        end_time = time.time()
                        duration = (end_time - start_time) * 1000
                        
                        orch_step["status"] = "completed" if result.get("success") else "failed"
                        orch_step["result"] = result
                        orch_step["duration_ms"] = duration
                        
                    except Exception as e:
                        orch_step["status"] = "failed"
                        orch_step["result"] = {"error": str(e)}

        state["step_results"] = step_results
        state["orchestration_steps"] = orchestration_steps
        return state

    async def _execute_single_step(self, step: APIStep, previous_results: Dict[str, Any]) -> Any:
        """Execute single step with parameter resolution"""
        try:
            resolved_params = self._resolve_parameters(step.params, previous_results)
            result = await self.sqlite_service.execute_specific_tool(step.api_name, resolved_params)
            return result
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            return {"error": str(e)}

    def _resolve_parameters(self, params: Dict[str, Any], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters with dependencies"""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${"):
                dependency_path = value[2:-1]
                if "." in dependency_path:
                    step_id, field_name = dependency_path.split(".", 1)
                else:
                    step_id = dependency_path
                    field_name = None

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

    def _extract_table_name(self, query: str) -> Optional[str]:
        """Extract table name from query"""
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
        """Extract or build SQL query"""
        if "select" in query.lower() and "from" in query.lower():
            return query
        
        if "employee" in query.lower():
            return "SELECT * FROM employees LIMIT 10"
        elif "department" in query.lower():
            return "SELECT * FROM departments LIMIT 10"
        elif "project" in query.lower():
            return "SELECT * FROM projects LIMIT 10"
        else:
            return "SELECT name FROM sqlite_master WHERE type='table'"

    async def _synthesize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response with orchestration completion"""
        query = state.get("messages", [])[-1]
        step_results = state.get("step_results", {})
        
        response_parts = [f"Query: {query}\n"]
        
        for step_id, result in step_results.items():
            if isinstance(result, dict) and result.get("success"):
                if "tables" in result:
                    response_parts.append(f"📋 Available tables: {', '.join(result['tables'])}")
                elif "columns" in result:
                    table_name = result.get("table_name", "Unknown")
                    response_parts.append(f"📊 Table '{table_name}' structure:")
                    for col in result["columns"][:3]:  # Show first 3 columns
                        response_parts.append(f"  • {col['name']} ({col['type']})")
                elif "results" in result:
                    response_parts.append(f"📈 Query results ({result['row_count']} rows):")
                    for row in result["results"][:3]:
                        response_parts.append(f"  {row}")

        state["final_response"] = "\n".join(response_parts)
        return state

    async def _fallback_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback response"""
        query = state.get("messages", [])[-1]
        state["final_response"] = f"Could not process query: {query}. Try simpler database operations."
        return state

    # Public API methods
    async def process_query_with_chaining(self, user_query: str) -> str:
        """Process query and return simple response"""
        initial_state = {
            "messages": [user_query],
            "analysis": "",
            "execution_plan": None,
            "execution_mode": ExecutionMode.PARALLEL,
            "step_results": {},
            "final_response": "",
            "orchestration_steps": [],
        }
        
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["final_response"]

    async def process_query_with_orchestration_details(self, user_query: str) -> Dict[str, Any]:
        """Process query and return detailed orchestration information"""
        initial_state = {
            "messages": [user_query],
            "analysis": "",
            "execution_plan": None,
            "execution_mode": ExecutionMode.PARALLEL,
            "step_results": {},
            "final_response": "",
            "orchestration_steps": [],
        }
        
        final_state = await self.workflow.ainvoke(initial_state)
        
        return {
            "steps": final_state.get("orchestration_steps", []),
            "final_response": final_state.get("final_response", ""),
            "execution_mode": final_state.get("execution_mode", ExecutionMode.PARALLEL).value
        }

    async def process_query_with_streaming(self, user_query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Process query with streaming updates (for WebSocket)"""
        # This would stream each step as it completes
        # Simplified implementation
        result = await self.process_query_with_orchestration_details(user_query)
        
        for step in result["steps"]:
            yield step
            await asyncio.sleep(0.1)  # Simulate streaming delay

# Mock LLM
class MockLLM:
    async def ainvoke(self, messages):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse("Mock analysis response")
