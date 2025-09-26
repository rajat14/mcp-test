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
        # Unexpected or corrupted LLM output, forces a retry or error handling. #can make it more robust
        # For simplicity, we loop back to the planner to try again.
        return "planner" 
