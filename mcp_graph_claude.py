from typing import Dict, Any
from datetime import datetime
import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph_state import AgentState, ActionType
from nodes import EnhancedPlannerNode, ExecutorNode, SynthesizerNode
from router import create_conditional_edges
import logging

logger = logging.getLogger(__name__)

class MCPOrchestrationGraph:
    """
    Enhanced LangGraph workflow for intelligent tool orchestration
    with sophisticated state management and dependency-aware routing
    """
    
    def __init__(self, llm, tools: Dict[str, Any], tool_catalog: Dict[str, Dict[str, Any]]):
        """
        Initialize the orchestration graph
        
        Args:
            llm: Language model for decision making
            tools: Available tools for execution
            tool_catalog: Tool metadata with schemas and descriptions
        """
        self.llm = llm
        self.tools = tools
        self.tool_catalog = tool_catalog
        
        # Initialize nodes
        self.planner_node = EnhancedPlannerNode(llm, tool_catalog)
        self.executor_node = ExecutorNode(tools)
        self.synthesizer_node = SynthesizerNode(llm)
        
        # Create routing functions
        self.routing_functions = create_conditional_edges()
        self.router = self.routing_functions["router_instance"]
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Compile with memory for state persistence
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_wrapper)
        workflow.add_node("executor", self._executor_wrapper) 
        workflow.add_node("synthesizer", self._synthesizer_wrapper)
        
        # Add edges
        # Start always goes to planner for initial analysis
        workflow.add_edge(START, "planner")
        
        # Conditional edges from planner
        workflow.add_conditional_edges(
            "planner",
            self.routing_functions["route_planner"],
            {
                "executor": "executor",
                "synthesizer": "synthesizer"
            }
        )
        
        # Conditional edges from executor  
        workflow.add_conditional_edges(
            "executor", 
            self.routing_functions["route_executor"],
            {
                "planner": "planner", 
                "synthesizer": "synthesizer"
            }
        )
        
        # Synthesizer always ends workflow
        workflow.add_edge("synthesizer", END)
        
        return workflow
    
    def _planner_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for planner node with enhanced logging and state management"""
        
        try:
            # Update current node and timestamp
            state["current_node"] = "planner"
            state["last_updated"] = datetime.now().isoformat()
            
            # Increment step counter
            current_step = state.get("current_step", 0) + 1
            state["current_step"] = current_step
            
            logger.info(f"Step {current_step}: Entering planner node")
            logger.debug(f"Current state - Action: {state.get('current_action')}, "
                        f"Gaps: {state.get('parameter_gaps', [])}, "
                        f"Objective: {state.get('immediate_objective', 'Not set')}")
            
            # Execute planner logic
            state = self.planner_node(state)
            
            logger.info(f"Step {current_step}: Planner decided action: {state.get('current_action')}")
            
            return state
            
        except Exception as e:
            logger.error(f"Planner node error: {str(e)}")
            return self._handle_node_error(state, "planner", e)
    
    def _executor_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for executor node with enhanced logging and state management"""
        
        try:
            # Update current node and timestamp
            state["current_node"] = "executor"
            state["last_updated"] = datetime.now().isoformat()
            
            current_step = state.get("current_step", 0)
            pending_calls = state.get("pending_tool_calls", [])
            
            logger.info(f"Step {current_step}: Entering executor node")
            if pending_calls:
                tool_name = pending_calls[0].tool_name
                call_type = pending_calls[0].call_type
                logger.info(f"Step {current_step}: Executing {call_type} call to {tool_name}")
            
            # Execute tool logic
            state = self.executor_node(state)
            
            logger.info(f"Step {current_step}: Executor completed, next action: {state.get('current_action')}")
            
            return state
            
        except Exception as e:
            logger.error(f"Executor node error: {str(e)}")
            return self._handle_node_error(state, "executor", e)
    
    def _synthesizer_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for synthesizer node with enhanced logging and state management"""
        
        try:
            # Update current node and timestamp
            state["current_node"] = "synthesizer"
            state["last_updated"] = datetime.now().isoformat()
            
            current_step = state.get("current_step", 0)
            logger.info(f"Step {current_step}: Entering synthesizer node")
            
            # Execute synthesizer logic
            state = self.synthesizer_node(state)
            
            logger.info(f"Step {current_step}: Synthesizer completed workflow")
            
            return state
            
        except Exception as e:
            logger.error(f"Synthesizer node error: {str(e)}")
            return self._handle_node_error(state, "synthesizer", e)
    
    def _handle_node_error(self, state: AgentState, node_name: str, error: Exception) -> AgentState:
        """Handle errors that occur within nodes"""
        
        state["error_context"] = {
            "node": node_name,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        state["current_action"] = ActionType.INSUFFICIENT_INFO
        state["processing_status"] = "failed"
        
        # Increment retry count
        retry_count = state.get("retry_count", 0) + 1
        state["retry_count"] = retry_count
        
        logger.error(f"Node {node_name} failed (retry {retry_count}): {str(error)}")
        
        return state
    
    def create_initial_state(self, user_query: str, session_id: str = None) -> AgentState:
        """Create initial state for a new query"""
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        initial_state = AgentState(
            # Input and Output
            user_query=user_query,
            final_response="",
            
            # Current Processing Context
            current_node="",
            current_action=ActionType.TOOL_CALL_PREREQUISITE,  # Will be determined by planner
            processing_status="active",
            
            # Goal and Objective Tracking
            primary_goal=None,
            immediate_objective="Analyze user query and determine approach",
            
            # Tool and Parameter Management
            available_tools=self.tool_catalog,
            pending_tool_calls=[],
            completed_tool_calls=[],
            failed_tool_calls=[],
            
            # Parameter Resolution
            required_parameters={},
            resolved_parameters={},
            parameter_gaps=[],
            
            # Execution Chain
            execution_steps=[],
            current_step=0,
            
            # Context and Memory  
            conversation_context=[],
            intermediate_results={},
            dependency_graph={},
            
            # Decision Making Context
            reasoning_trace=[],
            alternative_paths=[],
            confidence_score=1.0,
            
            # Error Handling
            error_context=None,
            retry_count=0,
            max_retries=3,
            
            # Metadata
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            
            # Control Flow
            next_node=None,
            should_continue=True
        )
        
        return initial_state
    
    def invoke(self, user_query: str, session_id: str = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Invoke the orchestration workflow for a user query
        
        Args:
            user_query: The user's question or request
            session_id: Optional session ID for conversation continuity
            config: Optional configuration for the workflow
            
        Returns:
            Dictionary containing the final response and execution metadata
        """
        
        try:
            # Create initial state
            initial_state = self.create_initial_state(user_query, session_id)
            
            logger.info(f"Starting workflow for query: {user_query[:100]}...")
            logger.info(f"Session ID: {initial_state['session_id']}")
            
            # Set up config for checkpointing if session_id provided
            run_config = config or {}
            if session_id:
                run_config["configurable"] = {"thread_id": session_id}
            
            # Execute the workflow
            result = self.compiled_graph.invoke(initial_state, config=run_config)
            
            # Extract key information for response
            response_data = {
                "final_response": result.get("final_response", "I apologize, but I wasn't able to complete your request."),
                "session_id": result.get("session_id"),
                "processing_status": result.get("processing_status", "unknown"),
                "execution_summary": {
                    "total_steps": result.get("current_step", 0),
                    "tools_called": len(result.get("completed_tool_calls", [])),
                    "tools_failed": len(result.get("failed_tool_calls", [])),
                    "retry_count": result.get("retry_count", 0),
                    "reasoning_trace": result.get("reasoning_trace", [])
                },
                "router_summary": self.router.get_routing_summary()
            }
            
            logger.info(f"Workflow completed: {response_data['processing_status']}")
            logger.info(f"Steps: {response_data['execution_summary']['total_steps']}, "
                       f"Tools called: {response_data['execution_summary']['tools_called']}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "final_response": f"I apologize, but an error occurred while processing your request: {str(e)}",
                "session_id": session_id,
                "processing_status": "failed",
                "error": str(e)
            }
    
    def stream_invoke(self, user_query: str, session_id: str = None, config: Dict[str, Any] = None):
        """
        Stream the workflow execution for real-time updates
        
        Args:
            user_query: The user's question or request  
            session_id: Optional session ID for conversation continuity
            config: Optional configuration for the workflow
            
        Yields:
            State updates as the workflow progresses
        """
        
        try:
            # Create initial state
            initial_state = self.create_initial_state(user_query, session_id)
            
            # Set up config for checkpointing if session_id provided
            run_config = config or {}
            if session_id:
                run_config["configurable"] = {"thread_id": session_id}
            
            # Stream the workflow execution
            for state_update in self.compiled_graph.stream(initial_state, config=run_config):
                yield state_update
                
        except Exception as e:
            logger.error(f"Streaming workflow failed: {str(e)}")
            yield {
                "error": str(e),
                "final_response": f"An error occurred: {str(e)}",
                "processing_status": "failed"
            }
    
    def get_workflow_state(self, session_id: str) -> Dict[str, Any]:
        """Get the current state of a workflow session"""
        
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.compiled_graph.get_state(config)
            return state.values if state else {}
        except Exception as e:
            logger.error(f"Failed to get workflow state: {str(e)}")
            return {}


def create_mcp_graph(llm, tools: Dict[str, Any], tool_catalog: Dict[str, Dict[str, Any]]) -> MCPOrchestrationGraph:
    """
    Factory function to create the MCP orchestration graph
    
    Args:
        llm: Language model for decision making
        tools: Available tools for execution  
        tool_catalog: Tool metadata with schemas and descriptions
        
    Returns:
        Configured MCPOrchestrationGraph instance
    """
    
    return MCPOrchestrationGraph(llm, tools, tool_catalog)


# Example usage and testing
if __name__ == "__main__":
    # Example tool catalog structure
    example_tool_catalog = {
        "get_lineage": {
            "description": "Get column lineage information",
            "parameters": {
                "required": ["database_name", "schema_name", "table_name", "column_name"],
                "properties": {
                    "database_name": {"type": "string", "description": "Database name"},
                    "schema_name": {"type": "string", "description": "Schema name"}, 
                    "table_name": {"type": "string", "description": "Table name"},
                    "column_name": {"type": "string", "description": "Column name"}
                }
            }
        },
        "get_table_info": {
            "description": "Get table metadata and schema information",
            "parameters": {
                "required": ["table_name"],
                "properties": {
                    "table_name": {"type": "string", "description": "Table name"}
                }
            }
        },
        "search_columns": {
            "description": "Search for columns by name across databases", 
            "parameters": {
                "required": ["column_name"],
                "properties": {
                    "column_name": {"type": "string", "description": "Column name to search for"}
                }
            }
        }
    }
    
    print("MCP Orchestration Graph structure created successfully!")
    print("Available tool catalog:", list(example_tool_catalog.keys()))
