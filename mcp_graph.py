# ====================================================================
# File: src/core/langgraph/mcp_graph.py (NEW)
# ====================================================================
from langgraph.graph import StateGraph, END
from .graph_state import AgentState
from .nodes import planner_node, executor_node, synthesizer_node
from .router import route_step
from src.infra.support.mcp.tool_executor.tool_executor import ToolExecutor

def build_mcp_graph(tool_executor_instance: ToolExecutor):
    """
    Builds and compiles the LangGraph workflow, setting up the nodes and edges
    for intelligent, stateful orchestration.
    """
    
    # 1. Initialize the StateGraph with the defined AgentState
    workflow = StateGraph(AgentState)

    # 2. Add nodes (wrap the functions to pass the tool_executor_instance)
    #    The lambda ensures the executor instance is available within the node's execution context.
    workflow.add_node("planner", lambda state: planner_node(state, tool_executor_instance))
    workflow.add_node("executor", lambda state: executor_node(state, tool_executor_instance))
    workflow.add_node("synthesizer", lambda state: synthesizer_node(state, tool_executor_instance))

    # 3. Define the entry point
    workflow.set_entry_point("planner")

    # 4. Define conditional routing after the planner
    #    The router decides: Execute tool, or skip straight to response synthesis.
    workflow.add_conditional_edges(
        "planner", 
        route_step, # Uses the logic from router.py
        {
            "executor": "executor",
            "synthesizer": "synthesizer",
            "planner": "planner", # Loop back to planner (e.g., if LLM output was ambiguous)
        }
    )

    # 5. Define the loop back for multi-step execution
    #    After execution is done, always return to the planner to decide the next action.
    workflow.add_edge("executor", "planner")

    # 6. Define the end point
    #    Once the final response is synthesized, the workflow ends.
    workflow.add_edge("synthesizer", END)

    # 7. Compile the workflow into a runnable application
    return workflow.compile()
