from typing import Literal
from graph_state import AgentState, ActionType
import logging

logger = logging.getLogger(__name__)

class IntelligentRouter:
    """
    Enhanced routing logic that makes intelligent decisions about next node
    based on current state, action type, and processing context
    """
    
    def __init__(self):
        self.route_history = []  # Track routing decisions for debugging
    
    def route_from_planner(self, state: AgentState) -> Literal["executor", "synthesizer"]:
        """Route from planner node based on current action and state"""
        
        current_action = state.get("current_action")
        
        # Log routing decision
        self._log_routing_decision("planner", current_action, state)
        
        if current_action in [ActionType.TOOL_CALL_PRIMARY, ActionType.TOOL_CALL_PREREQUISITE]:
            # Need to execute tool calls
            return "executor"
        
        elif current_action in [ActionType.RESPOND, ActionType.INSUFFICIENT_INFO]:
            # Ready to generate final response
            return "synthesizer"
        
        else:
            # Default fallback
            logger.warning(f"Unexpected action type from planner: {current_action}")
            return "synthesizer"
    
    def route_from_executor(self, state: AgentState) -> Literal["planner", "synthesizer"]:
        """Route from executor node based on execution results"""
        
        current_action = state.get("current_action")
        pending_calls = state.get("pending_tool_calls", [])
        error_context = state.get("error_context")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        # Log routing decision
        self._log_routing_decision("executor", current_action, state)
        
        # Check for terminal conditions first
        if current_action == ActionType.RESPOND:
            # Successful completion of primary tool call
            return "synthesizer"
        
        if current_action == ActionType.INSUFFICIENT_INFO:
            # Error condition or max retries reached
            return "synthesizer"
        
        if error_context and retry_count >= max_retries:
            # Failed after max retries
            state["current_action"] = ActionType.INSUFFICIENT_INFO
            return "synthesizer"
        
        # Check if we have more work to do
        if current_action == ActionType.TOOL_CALL_PREREQUISITE:
            if pending_calls:
                # More tools to execute
                return "executor"
            else:
                # No more pending calls, go back to planner to reassess
                return "planner"
        
        # For prerequisite tool completion, always go back to planner
        # to reassess parameter gaps and determine next steps
        return "planner"
    
    def should_continue_workflow(self, state: AgentState) -> bool:
        """Determine if the workflow should continue or terminate"""
        
        # Explicit termination signals
        if not state.get("should_continue", True):
            return False
        
        if state.get("processing_status") == "completed":
            return False
        
        if state.get("processing_status") == "failed":
            return False
        
        # Check for infinite loops
        if self._detect_infinite_loop(state):
            logger.warning("Potential infinite loop detected, terminating workflow")
            state["processing_status"] = "failed"
            state["error_context"] = {"error": "Infinite loop detected"}
            return False
        
        # Check for excessive retries
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if retry_count >= max_retries:
            logger.warning(f"Max retries ({max_retries}) exceeded")
            state["processing_status"] = "failed"
            state["current_action"] = ActionType.INSUFFICIENT_INFO
            return False
        
        # Continue if we have pending work
        current_action = state.get("current_action")
        
        if current_action in [ActionType.TOOL_CALL_PRIMARY, ActionType.TOOL_CALL_PREREQUISITE]:
            return True
        
        if current_action == ActionType.RESPOND and not state.get("final_response"):
            return True
        
        if current_action == ActionType.INSUFFICIENT_INFO and not state.get("final_response"):
            return True
        
        # Default to continue unless explicitly stopped
        return True
    
    def get_next_node_override(self, state: AgentState) -> str:
        """Check if state specifies a next node override"""
        
        next_node = state.get("next_node")
        if next_node:
            # Clear the override after using it
            state["next_node"] = None
            self._log_routing_decision("override", next_node, state)
            return next_node
        
        return None
    
    def _log_routing_decision(self, from_node: str, decision_context: any, state: AgentState):
        """Log routing decisions for debugging and monitoring"""
        
        routing_info = {
            "from_node": from_node,
            "decision_context": str(decision_context),
            "current_step": state.get("current_step", 0),
            "retry_count": state.get("retry_count", 0),
            "pending_calls": len(state.get("pending_tool_calls", [])),
            "completed_calls": len(state.get("completed_tool_calls", [])),
            "parameter_gaps": len(state.get("parameter_gaps", [])),
        }
        
        self.route_history.append(routing_info)
        logger.debug(f"Routing decision: {routing_info}")
        
        # Keep history bounded
        if len(self.route_history) > 100:
            self.route_history = self.route_history[-50:]  # Keep last 50 entries
    
    def _detect_infinite_loop(self, state: AgentState) -> bool:
        """Detect potential infinite loops in routing"""
        
        if len(self.route_history) < 10:
            return False
        
        # Check for repeated patterns in recent routing decisions
        recent_routes = [entry["from_node"] for entry in self.route_history[-10:]]
        
        # Simple pattern detection: same sequence repeating
        if len(set(recent_routes)) <= 2:  # Only 2 or fewer unique nodes in recent history
            # Check if we're stuck in planner -> executor -> planner loop without progress
            planner_count = recent_routes.count("planner")
            executor_count = recent_routes.count("executor")
            
            if planner_count >= 4 and executor_count >= 4:
                # Check if we're making progress (parameter gaps decreasing)
                current_gaps = len(state.get("parameter_gaps", []))
                
                # Look at parameter gaps from a few steps ago
                if len(self.route_history) >= 6:
                    # If gaps haven't decreased in several iterations, likely stuck
                    return current_gaps > 0 and state.get("retry_count", 0) >= 2
        
        return False
    
    def get_routing_summary(self) -> dict:
        """Get summary of routing decisions for monitoring/debugging"""
        
        if not self.route_history:
            return {"total_routes": 0}
        
        summary = {
            "total_routes": len(self.route_history),
            "node_counts": {},
            "recent_pattern": [entry["from_node"] for entry in self.route_history[-5:]],
            "max_retries_hit": any(entry["retry_count"] >= 3 for entry in self.route_history),
            "avg_pending_calls": sum(entry["pending_calls"] for entry in self.route_history) / len(self.route_history),
        }
        
        # Count visits to each node
        for entry in self.route_history:
            node = entry["from_node"]
            summary["node_counts"][node] = summary["node_counts"].get(node, 0) + 1
        
        return summary


def create_conditional_edges():
    """
    Create the conditional edge functions for the LangGraph workflow
    """
    
    router = IntelligentRouter()
    
    def route_planner(state: AgentState) -> Literal["executor", "synthesizer"]:
        """Conditional edge from planner node"""
        return router.route_from_planner(state)
    
    def route_executor(state: AgentState) -> Literal["planner", "synthesizer"]:
        """Conditional edge from executor node"""
        return router.route_from_executor(state)
    
    def should_continue(state: AgentState) -> Literal["planner", "__end__"]:
        """Determine if workflow should continue or end"""
        
        # Check for explicit next node override
        override = router.get_next_node_override(state)
        if override:
            return override
        
        # Check if workflow should continue
        if router.should_continue_workflow(state):
            return "planner"
        else:
            return "__end__"
    
    return {
        "route_planner": route_planner,
        "route_executor": route_executor, 
        "should_continue": should_continue,
        "router_instance": router  # For monitoring/debugging
    }
