from fastapi import APIRouter
from src.services.mcp.mcp_service import MCPService
from src.domain.dto.mcp_dto import McpExecuteRequest, McpToolResponse

from src.core.instrumentation.decorators.transaction_data import initialize_transaction_data
from src.core.instrumentation.decorators.function_timer import timer_start, timer_end
from src.core.instrumentation.decorators.log_entry_exit import log_entry_exit

router = APIRouter(prefix="/mcp", tags=["MCP Tools"])
mcp_service = MCPService()

@router.post("/execute")
async def execute_mcp_query(request: McpExecuteRequest):
    """Execute MCP query with automatic tool selection"""
    result = await mcp_service.select_and_execute_tool(request.user_query)
    return result

@router.get("/tools",
            openapi_extra={
                "mcp_description": "Get all available MCP tools that can be used for queries.",
                "mcp_scenarios": [
                    "When you need to see what MCP tools are available",
                    "When you want to list all tools before selecting one to use"
                ]
            }
            )
@initialize_transaction_data()
@timer_start()
@timer_end()
@log_entry_exit(api_router=True)
def get_available_tools() -> McpToolResponse:
    """Get list of available MCP tools"""
    mcp_service = MCPService()
    tools = mcp_service.get_available_tools()
    return McpToolResponse(tools=list(tools))
