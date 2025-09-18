# backend/src/services/mcp/mcp_service.py
from typing import Dict, Any, List
from src.infra.support.mcp.mcp import MCP

class MCPService:
    def __init__(self):
        self.tools_manager = MCP()

    def get_available_tools(self) -> List:
        """List all available MCP tools"""
        tools = self.tools_manager.get_available_tools()
        return tools

    async def select_and_execute_tool(self, query) -> Dict[str, Any]:
        """Execute an MCP tool"""
        response = await self.tools_manager.select_and_execute_tool(query)
        return response

    # Add this method to your MCPService class
    async def execute_specific_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a specific tool by name"""
        available_tools = self.get_available_tools()

        for tool in available_tools:
            if tool.name == tool_name:
                selected_tool = tool
                break

        try:
            response = await self.tools_manager.execute_specific_tool(selected_tool.name, params)
            return response
        except Exception as e:
            raise f"The {tool_name} tool does not exist in the available tools"

