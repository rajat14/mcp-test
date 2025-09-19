# ====================================================================
# File: src/core/langgraph/graph_state.py (NEW)
# ====================================================================
from typing import TypedDict, List, Dict, Any, Optional
from mcp import Tool # Assuming 'mcp' module defines the Tool class

class AgentState(TypedDict):
    """
    Defines the state passed between all nodes in the LangGraph workflow.
    This replaces the monolithic data structures previously handled internally
    by the ToolExecutor's single function call.
    """
    query: str
    tools_available: List[Tool]      # List of Tool objects loaded by MCP
    steps_history: List[Dict[str, Any]] # Track of executed tool calls and their raw results
    next_action: Optional[Dict[str, Any]] # The LLM's planned action (tool_name/params or final response)
    final_response: Optional[str]
    error: Optional[str]

# ====================================================================
# File: src/core/langgraph/router.py (NEW)
# ====================================================================
from .graph_state import AgentState

def route_step(state: AgentState) -> str:
    """
    Decides the next node based on the output of the planner_node.
    This function implements the conditional logic of the graph.
    """
    action = state.get("next_action")
    
    # Safety check: if no action was planned, try planning again
    if not action:
        return "planner" 
        
    action_type = action.get("action")
    
    if action_type == "call_tool":
        # The LLM decided to execute an external API call.
        return "executor"
    elif action_type == "respond":
        # The LLM decided it has enough information or the question requires no tools.
        return "synthesizer" 
    else:
        # Unexpected or corrupted LLM output, forces a retry or error handling.
        # For simplicity, we loop back to the planner to try again.
        return "planner" 
