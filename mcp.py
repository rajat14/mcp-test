from typing import Dict, Any, List
from mcp import Tool
import httpx

from src.infra.support.mcp.tool_registry.tool_registry import ToolRegistry
from src.infra.support.mcp.tool_executor.tool_executor import ToolExecutor

class MCP:
    def __init__(self, default_prompt_path_for_lookups="mcp"):
        self.tools: Dict[str, Tool] = {}
        self.route_mappings = {}
        self.default_prompt_path_for_lookups = default_prompt_path_for_lookups
        self.tools, self.route_mappings = ToolRegistry(self.tools, self.route_mappings).load_router_tools()


    def get_available_tools(self) -> List[Tool]:
        """Get list of available tools"""
        return list(self.tools.values())

    def load_router_tools(self):
        tools = ToolRegistry(self.tools, self.route_mappings).load_router_tools()
        return tools

    def select_and_execute_tool(self, query: str):
        self.load_router_tools()
        natural_response = ToolExecutor(
            self.default_prompt_path_for_lookups,
            self.tools,
            self.route_mappings).select_and_execute_tool(query)
        return natural_response

    def execute_specific_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        self.load_router_tools()
        tool_response = ToolExecutor(
            self.default_prompt_path_for_lookups,
            self.tools,
            self.route_mappings).execute_tool(tool_name, params)
        return tool_response
