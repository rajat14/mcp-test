from typing import Dict, Any, List, Optional, Tuple
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import asyncio
from enum import Enum

from src.services.mcp.mcp_service import MCPService
from src.core.instrumentation.logger.logger import Logger
from src.core.config.config_refs import ConfigNodes

logger = Logger()


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


class EnhancedLanggraphMCPService:
    """Enhanced service with API chaining capabilities"""

    def __init__(self):
        self.llm_config = ConfigNodes.get_llm_config("azure_openai")
        self.llm = AzureChatOpenAI(
            azure_deployment=self.llm_config.model,
            azure_endpoint=self.llm_config.azure_endpoint,
            api_key=self.llm_config.api_key,
            api_version=self.llm_config.api_version,
            temperature=0,
        )
        self.mcp_service = MCPService()
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
        available_apis = self.mcp_service.get_available_tools()

        prompt = [
            SystemMessage(content="""You are an advanced query analyzer that detects:
            1. Required APIs/tools
            2. Dependencies between API calls
            3. Data flow requirements
            
            Analyze if this is a simple query (single API) or complex query (multiple APIs with dependencies)."""),
            HumanMessage(content=f"""
            User query: {query}
            Available APIs: {list(available_apis.keys())}
            
            Determine:
            - What APIs are needed?
            - Is this PARALLEL (independent APIs) or SEQUENTIAL (one depends on another) or HYBRID?
            - What data flows between APIs?
            """),
        ]
        
        response = await self.llm.ainvoke(prompt)
        state["analysis"] = response.content
        logger.info(f"Enhanced query analysis: {response.content}")
        return state

    async def _create_execution_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan with dependencies"""
        query = state.get("messages", [])[-1]
        analysis = state.get("analysis", "")
        available_apis = self.mcp_service.get_available_tools()

        prompt = [
            SystemMessage(content="""You are an execution planner. Create a detailed plan for API execution.
            
            Return a JSON structure like:
            {
                "mode": "sequential|parallel|hybrid",
                "steps": [
                    {
                        "api_name": "weather_api",
                        "params": {"location": "extracted_from_query"},
                        "depends_on": [],
                        "output_mapping": {"temperature": "temp_value"}
                    },
                    {
                        "api_name": "recommendation_api", 
                        "params": {"temperature": "${weather_api.temperature}"},
                        "depends_on": ["weather_api"],
                        "output_mapping": {}
                    }
                ]
            }
            
            Use ${step_id.field_name} syntax for dependencies."""),
            HumanMessage(content=f"""
            Query: {query}
            Analysis: {analysis}
            Available APIs: {list(available_apis.keys())}
            
            Create execution plan:
            """),
        ]

        try:
            response = await self.llm.ainvoke(prompt)
            # Parse the JSON response (in real implementation, add proper JSON parsing)
            plan_data = self._parse_execution_plan(response.content)
            
            if plan_data:
                execution_plan = ExecutionPlan()
                
                for step_data in plan_data.get("steps", []):
                    step = APIStep(
                        api_name=step_data["api_name"],
                        params=step_data["params"],
                        depends_on=step_data.get("depends_on", [])
                    )
                    execution_plan.add_step(step)
                
                execution_plan.build_execution_order()
                
                state["execution_plan"] = execution_plan
                state["execution_mode"] = ExecutionMode(plan_data.get("mode", "parallel"))
                logger.info(f"Created execution plan with {len(execution_plan.steps)} steps")
            else:
                state["execution_plan"] = None
                
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            state["execution_plan"] = None

        return state

    def _parse_execution_plan(self, llm_response: str) -> Optional[Dict]:
        """Parse LLM response to extract execution plan (simplified)"""
        try:
            # In real implementation, use proper JSON extraction from LLM response
            # This is a placeholder for the parsing logic
            import json
            import re
            
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
                logger.info(f"Executed sequential step {step_id}: {result}")
                
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
                    logger.info(f"Executed parallel step {step_id}: {result}")

        state["step_results"] = step_results
        return state

    async def _execute_single_step(self, step: APIStep, previous_results: Dict[str, Any]) -> Any:
        """Execute a single API step with parameter resolution"""
        try:
            # Resolve parameters with dependencies
            resolved_params = self._resolve_parameters(step.params, previous_results)
            
            # Execute the API call
            result = await self.mcp_service.execute_specific_tool(
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
                step_id, field_name = dependency_path.split(".", 1)
                
                # Get value from previous results
                if step_id in previous_results:
                    step_result = previous_results[step_id]
                    resolved_value = self._extract_nested_value(step_result, field_name)
                    resolved[key] = resolved_value
                else:
                    logger.warning(f"Dependency {step_id} not found for parameter {key}")
                    resolved[key] = value
            else:
                resolved[key] = value
                
        return resolved

    def _extract_nested_value(self, data: Any, field_path: str) -> Any:
        """Extract nested value using dot notation (e.g., 'weather.temperature')"""
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
        execution_plan = state.get("execution_plan")

        prompt = [
            SystemMessage(content="""You are a response synthesizer for multi-API workflows.
            Present the information in a coherent way, showing how different APIs contributed to the final answer."""),
            HumanMessage(content=f"""
            User query: {query}
            API execution results: {step_results}
            Execution plan: {[step.api_name for step in execution_plan.steps.values()] if execution_plan else 'None'}
            
            Create a comprehensive response that:
            1. Answers the user's question
            2. Shows the data flow between APIs
            3. Mentions which APIs were used
            """),
        ]

        response = await self.llm.ainvoke(prompt)
        state["final_response"] = response.content
        logger.info(f"Synthesized response from {len(step_results)} API calls")
        return state

    async def _fallback_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback with suggestions"""
        query = state.get("messages", [])[-1]
        available_apis = list(self.mcp_service.get_available_tools().keys())
        
        msg = f"""I couldn't create an execution plan for: '{query}'
        
Available APIs: {', '.join(available_apis)}
        
Try rephrasing your query to be more specific about what information you need."""
        
        state["final_response"] = msg
        logger.info(f"Enhanced fallback response")
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


# Example usage scenarios:

"""
Example 1: Sequential API chaining
Query: "Get weather for New York and recommend activities based on temperature"

Execution Plan:
1. weather_api(location="New York") -> {temperature: 25, condition: "sunny"}
2. activity_api(temperature=25, condition="sunny") -> {activities: ["beach", "hiking"]}

Example 2: Hybrid execution
Query: "Compare weather in NYC and LA, then recommend best city for vacation"

Execution Plan:
1. [weather_api(location="NYC"), weather_api(location="LA")] - Parallel
2. comparison_api(nyc_weather=${step1_nyc}, la_weather=${step1_la}) - Sequential
3. recommendation_api(comparison=${step2}) - Sequential
"""
