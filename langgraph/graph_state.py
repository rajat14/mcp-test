from typing import TypedDict, List, Dict, Any, Optional
from mcp import Tool 

class AgentState(TypedDict):
    """
    Defines the state passed between all nodes in the LangGraph workflow.
    """
    query: str
    tools_available: List[Tool]             # List of Tool objects loaded by MCP
    steps_history: List[Dict[str, Any]]     # Track of executed tool calls and their raw results
    next_action: Optional[Dict[str, Any]]   # The LLM's planned action (tool_name/params or final response)
    final_response: Optional[str]
    error: Optional[str]
