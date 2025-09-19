# ====================================================================
# File: src/core/langgraph/nodes.py (NEW)
# ====================================================================
import json
from typing import Dict, Any
from .graph_state import AgentState
from src.infra.support.mcp.tool_executor.tool_executor import ToolExecutor 
from src.services.inference.inference_service import InferenceService 
from src.core.instrumentation.logger.logger import Logger

logger = Logger()

# --- NODE 1: PLANNER ---
async def planner_node(state: AgentState, tool_executor: ToolExecutor) -> AgentState:
    """
    Uses the LLM to decide the next step: call a tool or synthesize the final response.
    """
    logger.info("Planner Node: Starting new planning cycle.")

    # Convert tool objects to the description format the LLM can understand
    tools_info = "\n".join([
        f"- {tool.name}: {tool.description} (params: {tool_executor._get_parameter_info(tool.inputSchema)})"
        for tool in state["tools_available"]
    ])
    
    # Use existing prompt library logic
    prompt = tool_executor.prompt_library.retrieve_prompt_via_defaults(
        prompt_name="azure_openai/create_execution_plan",
        placeholder_names_and_values={
            "tools_info": tools_info,
            "query": state["query"],
            "history": json.dumps(state["steps_history"])
        }
    )
    
    try:
        # Call the external LLM service
        response = InferenceService().query_via_config(llm_connector_name="azure_openai", prompt=prompt)
        
        # Parse the structured JSON output
        llm_output = json.loads(response.replace("```json", "").replace("```", "").strip())
        
        return {"next_action": llm_output}

    except Exception as e:
        logger.error(f"Planner Node Error: {e}")
        # If planning fails, force a response action with an error message
        return {"next_action": {"action": "respond", "response": f"Planning failed: {str(e)}"}}


# --- NODE 2: EXECUTOR ---
async def executor_node(state: AgentState, tool_executor: ToolExecutor) -> AgentState:
    """
    Executes the tool specified in 'next_action' using the core ToolExecutor logic.
    """
    action = state["next_action"]
    
    if action and action.get("action") == "call_tool":
        tool_name = action.get("tool")
        params = action.get("parameters", {})
        
        logger.info(f"Executor Node: Executing tool '{tool_name}'.")

        try:
            # Call the existing, robust execution logic (API building, HTTP call, etc.)
            result = await tool_executor.execute_tool(tool_name, params)
            
            # Check for execution errors (e.g., HTTP 404, connection failure)
            success = "error" not in result
            
            # Package result for history
            new_history_step = {
                "step": len(state["steps_history"]) + 1,
                "tool": tool_name, 
                "success": success, 
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Tool execution critical error for {tool_name}: {e}")
            new_history_step = {"tool": tool_name, "success": False, "error": str(e)}

        # Clear action and update history; the graph will loop back to the Planner
        return {
            "steps_history": state["steps_history"] + [new_history_step],
            "next_action": None 
        }
    
    return state # Should not happen if router is correct


# --- NODE 3: SYNTHESIZER ---
async def synthesizer_node(state: AgentState, tool_executor: ToolExecutor) -> AgentState:
    """
    Generates the final human-readable response using the synthesized LLM call.
    """
    logger.info("Synthesizer Node: Generating final response.")
    
    # Use the repurposed synthesis method from the ToolExecutor
    response = await tool_executor.synthesize_response(
        original_query=state["query"],
        results={"history": state["steps_history"]},
        execution_plan={"action": "synthesize_final"}
    )
    
    # The final action is taken. The graph will terminate (END).
    return {"final_response": response}
