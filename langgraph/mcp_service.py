from typing import Dict, Any, List, Optional
from src.infra.support.mcp.mcp import MCP 

class MCPService:
    """
    It initializes the MCP manager and exposes methods used by the API router.
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
        Execute incoming query. This method triggers the LangGraph orchestration 
        within the tools_manager.
        """
        # This is used by the POST /mcp/execute endpoint
        response = await self.tools_manager.select_and_execute_tool(query)
        return response
