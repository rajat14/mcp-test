# ====================================================================
# File: src/services/mcp/mcp_service.py (MINIMAL MODIFICATION)
# ====================================================================
from typing import Dict, Any, List, Optional
# Placeholder imports
from src.domain.dto.mcp_dto import McpExecuteRequest, McpToolResponse 
# Core imports
from src.infra.support.mcp.mcp import MCP 

class MCPService:
    """
    The service layer facade. It initializes the MCP manager and exposes 
    methods used by the public-facing API router.
    """
    def __init__(self):
        # Initializes the MCP manager, which now contains the LangGraph logic
        self.tools_manager = MCP() 

    def get_available_tools(self) -> List[Any]:
        """List all available MCP tools."""
        # This is used by the GET /mcp/tools endpoint
        return self.tools_manager.get_available_tools()

    async def select_and_execute_tool(self, query: str) -> Dict[str, Any]:
        """
        Execute an MCP query. This method triggers the LangGraph orchestration 
        within the tools_manager.
        """
        # This is used by the POST /mcp/execute endpoint
        response = await self.tools_manager.select_and_execute_tool(query)
        return response

    async def execute_specific_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a specific tool by name (direct execution, bypassing the graph)."""
        available_tools = self.get_available_tools()
        selected_tool = next((tool for tool in available_tools if tool.name == tool_name), None)
        
        if selected_tool:
            # Calls the direct execution method on the MCP manager
            response = await self.tools_manager.execute_specific_tool(selected_tool.name, params)
            return response
        else:
            raise ValueError(f"The {tool_name} tool does not exist in the available tools")
