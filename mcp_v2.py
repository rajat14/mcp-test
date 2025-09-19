# ====================================================================
# File: src/infra/support/mcp/mcp.py (MODIFIED)
# ====================================================================
from typing import Dict, Any, List
from mcp import Tool 

# New LangGraph Dependencies
from src.core.langgraph.mcp_graph import build_mcp_graph
from src.core.langgraph.graph_state import AgentState

# Existing Dependencies
from src.infra.support.mcp.tool_registry.tool_registry import ToolRegistry
from src.infra.support.mcp.tool_executor.tool_executor import ToolExecutor

class MCP:
    """
    The central manager. It loads tools and instantiates the ToolExecutor, 
    but uses the LangGraph application to orchestrate the query processing.
    """
    def __init__(self, default_prompt_path_for_lookups="mcp"):
        self.tools: Dict[str, Tool] = {}
        self.route_mappings = {}
        self.default_prompt_path_for_lookups = default_prompt_path_for_lookups
        
        # 1. Load tools and routes using the registry
        self.tools, self.route_mappings = ToolRegistry(self.tools, self.route_mappings).load_router_tools()
        
        # 2. Initialize the ToolExecutor instance once to reuse configurations
        self.executor_instance = ToolExecutor(
            self.default_prompt_path_for_lookups,
            self.tools,
            self.route_mappings
        )

    def get_available_tools(self) -> List[Tool]:
        """Get list of available tools (KEPT)."""
        return list(self.tools.values())

    def load_router_tools(self):
        # Helper method for re-loading (implementation remains within ToolRegistry)
        self.tools, self.route_mappings = ToolRegistry(self.tools, self.route_mappings).load_router_tools()
        return self.tools, self.route_mappings
        
    async def select_and_execute_tool(self, query: str) -> Dict[str, Any]:
        """Starts the LangGraph workflow to process the query."""
        
        # 1. Build the graph instance, passing the configured executor
        mcp_app = build_mcp_graph(self.executor_instance)

        # 2. Define the initial state for the graph
        initial_state = AgentState(
            query=query, 
            tools_available=self.get_available_tools(), 
            steps_history=[], 
            next_action=None, 
            final_response=None,
            error=None
        )

        # 3. Run the LangGraph application
        # The LangGraph controls the entire flow from planning to synthesis.
        final_state = await mcp_app.ainvoke(initial_state)

        # 4. Return the result derived from the final state
        return {
            "natural_response": final_state.get("final_response", "Orchestration failed to produce a final response."),
            "execution_plan": {"action": "langgraph_orchestrated"},
            "tools_used": [step["tool"] for step in final_state["steps_history"] if step.get("success")],
            "results": final_state["steps_history"] # Detailed history of steps
        }

    async def execute_specific_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a specific tool by name (KEPT, direct execution bypassing the graph)."""
        # Note: This is useful for internal direct calls or testing.
        return await self.executor_instance.execute_tool(tool_name, params)
