from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from graph_state import AgentState, Goal, Parameter, ToolCall, ExecutionStep, ActionType
import logging

logger = logging.getLogger(__name__)

class EnhancedPlannerNode:
    """
    Enhanced planner with goal decomposition, parameter gap analysis,
    context tracking, and dependency-aware decision making
    """
    
    def __init__(self, llm, tool_catalog: Dict[str, Dict[str, Any]]):
        self.llm = llm
        self.tool_catalog = tool_catalog
    
    def __call__(self, state: AgentState) -> AgentState:
        """Main planner logic with enhanced capabilities"""
        
        try:
            # Step 1: Goal Decomposition
            if not state.get("primary_goal"):
                state = self._analyze_primary_goal(state)
            
            # Step 2: Parameter Gap Analysis
            state = self._analyze_parameter_gaps(state)
            
            # Step 3: Context Tracking and Decision Making
            state = self._make_dependency_aware_decision(state)
            
            # Step 4: Update reasoning trace
            state = self._update_reasoning_trace(state)
            
        except Exception as e:
            logger.error(f"Planner error: {str(e)}")
            state["error_context"] = {"node": "planner", "error": str(e)}
            state["current_action"] = ActionType.INSUFFICIENT_INFO
        
        return state
    
    def _analyze_primary_goal(self, state: AgentState) -> AgentState:
        """Analyze user query to determine primary goal and tool requirements"""
        
        user_query = state["user_query"]
        
        # Create prompt for goal analysis
        goal_analysis_prompt = f"""
        Analyze this user query and determine the primary goal:
        Query: {user_query}
        
        Available tools: {json.dumps(self.tool_catalog, indent=2)}
        
        Determine:
        1. What is the user ultimately trying to achieve?
        2. Which tool would provide the final answer?
        3. What parameters are required for that tool?
        4. Can the primary tool be called directly with current information?
        
        Respond with JSON:
        {{
            "primary_goal_description": "string",
            "target_tool": "string", 
            "required_parameters": [
                {{
                    "name": "string",
                    "description": "string", 
                    "required": true/false,
                    "can_infer_from_query": true/false,
                    "possible_sources": ["tool1", "tool2"]
                }}
            ],
            "can_call_directly": true/false,
            "reasoning": "string"
        }}
        """
        
        response = self.llm.invoke(goal_analysis_prompt)
        goal_data = json.loads(response.content)
        
        # Create Goal object
        parameters = []
        for param_data in goal_data["required_parameters"]:
            param = Parameter(
                name=param_data["name"],
                description=param_data["description"],
                required=param_data["required"],
                possible_sources=param_data.get("possible_sources", [])
            )
            parameters.append(param)
        
        goal = Goal(
            description=goal_data["primary_goal_description"],
            target_tool=goal_data["target_tool"],
            required_parameters=parameters,
            reasoning_chain=[goal_data["reasoning"]]
        )
        
        state["primary_goal"] = goal
        state["immediate_objective"] = goal.description if goal_data["can_call_directly"] else "Gather required parameters"
        
        return state
    
    def _analyze_parameter_gaps(self, state: AgentState) -> AgentState:
        """Identify missing parameters and update parameter tracking"""
        
        goal = state["primary_goal"]
        user_query = state["user_query"]
        intermediate_results = state.get("intermediate_results", {})
        
        required_params = {}
        resolved_params = {}
        gaps = []
        
        # Analyze each required parameter
        for param in goal.required_parameters:
            required_params[param.name] = param
            
            # Check if we can extract from user query
            value = self._extract_parameter_from_query(user_query, param)
            if value:
                param.current_value = value
                resolved_params[param.name] = value
                param.obtained_from = "user_query"
                continue
            
            # Check if we have it from previous tool calls
            if param.name in intermediate_results:
                param.current_value = intermediate_results[param.name]
                resolved_params[param.name] = intermediate_results[param.name]
                param.obtained_from = "previous_tool_call"
                continue
            
            # Still missing this parameter
            if param.required:
                gaps.append(param.name)
        
        state["required_parameters"] = required_params
        state["resolved_parameters"] = resolved_params
        state["parameter_gaps"] = gaps
        
        return state
    
    def _make_dependency_aware_decision(self, state: AgentState) -> AgentState:
        """Make intelligent routing decision based on current context and goals"""
        
        goal = state["primary_goal"]
        gaps = state.get("parameter_gaps", [])
        
        # If no gaps, we can call primary tool
        if not gaps:
            state["current_action"] = ActionType.TOOL_CALL_PRIMARY
            tool_call = ToolCall(
                tool_name=goal.target_tool,
                parameters=state["resolved_parameters"],
                call_type=ActionType.TOOL_CALL_PRIMARY,
                reasoning="All required parameters available, calling primary tool",
                expected_output="Final answer to user query"
            )
            state["pending_tool_calls"] = [tool_call]
            return state
        
        # Find best prerequisite tool to fill gaps
        best_prerequisite = self._select_best_prerequisite_tool(state, gaps)
        
        if best_prerequisite:
            state["current_action"] = ActionType.TOOL_CALL_PREREQUISITE
            state["pending_tool_calls"] = [best_prerequisite]
            state["immediate_objective"] = f"Getting {best_prerequisite.expected_output} for primary goal"
        else:
            state["current_action"] = ActionType.INSUFFICIENT_INFO
            state["immediate_objective"] = "Need user clarification for missing parameters"
        
        return state
    
    def _select_best_prerequisite_tool(self, state: AgentState, gaps: List[str]) -> Optional[ToolCall]:
        """Select the best tool to fill parameter gaps"""
        
        goal = state["primary_goal"]
        
        # Find tools that can provide missing parameters
        candidate_tools = []
        
        for gap in gaps:
            param = state["required_parameters"][gap]
            for source_tool in param.possible_sources:
                if source_tool in self.tool_catalog:
                    tool_info = self.tool_catalog[source_tool]
                    
                    # Check if we have parameters needed for this source tool
                    can_call = self._can_call_tool(source_tool, tool_info, state)
                    
                    if can_call:
                        candidate_tools.append({
                            "tool_name": source_tool,
                            "tool_info": tool_info,
                            "fills_gaps": [gap],
                            "priority": self._calculate_tool_priority(source_tool, gaps, goal)
                        })
        
        if not candidate_tools:
            return None
        
        # Select best candidate
        best_candidate = max(candidate_tools, key=lambda x: x["priority"])
        
        # Create ToolCall for best candidate
        tool_call = ToolCall(
            tool_name=best_candidate["tool_name"],
            parameters=self._extract_tool_parameters(best_candidate["tool_info"], state),
            call_type=ActionType.TOOL_CALL_PREREQUISITE,
            reasoning=f"Need to call {best_candidate['tool_name']} to get {', '.join(best_candidate['fills_gaps'])}",
            expected_output=f"Parameters: {', '.join(best_candidate['fills_gaps'])}",
            prerequisite_for=goal.target_tool
        )
        
        return tool_call
    
    def _extract_parameter_from_query(self, query: str, param: Parameter) -> Any:
        """Extract parameter value from user query using LLM"""
        
        extraction_prompt = f"""
        Extract the value for parameter '{param.name}' from this user query.
        
        Query: {query}
        Parameter: {param.name}
        Description: {param.description}
        
        If the parameter value can be clearly identified, return just the value.
        If not clearly identifiable, return null.
        
        Examples:
        - Query: "Get lineage for sales_amount column" -> For column_name parameter: "sales_amount"  
        - Query: "Show me the lineage" -> For column_name parameter: null
        
        Response format: {{"value": "extracted_value_or_null"}}
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            result = json.loads(response.content)
            return result.get("value")
        except:
            return None
    
    def _can_call_tool(self, tool_name: str, tool_info: Dict[str, Any], state: AgentState) -> bool:
        """Check if we have all required parameters to call a tool"""
        
        required_params = tool_info.get("parameters", {}).get("required", [])
        available_data = {**state.get("resolved_parameters", {}), **state.get("intermediate_results", {})}
        
        for param in required_params:
            if param not in available_data:
                # Try to extract from user query
                if not self._extract_parameter_from_query(state["user_query"], Parameter(param, "", True)):
                    return False
        
        return True
    
    def _calculate_tool_priority(self, tool_name: str, gaps: List[str], goal: Goal) -> float:
        """Calculate priority score for tool selection"""
        
        # Simple priority calculation - could be enhanced
        base_priority = 1.0
        
        # Prefer tools that fill more gaps
        gaps_filled = len([gap for gap in gaps if tool_name in 
                          next((p.possible_sources for p in goal.required_parameters if p.name == gap), [])])
        
        return base_priority + (gaps_filled * 0.5)
    
    def _extract_tool_parameters(self, tool_info: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Extract parameters needed for tool call"""
        
        parameters = {}
        required_params = tool_info.get("parameters", {}).get("required", [])
        all_params = tool_info.get("parameters", {}).get("properties", {})
        
        available_data = {**state.get("resolved_parameters", {}), **state.get("intermediate_results", {})}
        
        for param_name in required_params:
            if param_name in available_data:
                parameters[param_name] = available_data[param_name]
            else:
                # Try to extract from user query
                param_info = all_params.get(param_name, {})
                param_obj = Parameter(param_name, param_info.get("description", ""), True)
                value = self._extract_parameter_from_query(state["user_query"], param_obj)
                if value:
                    parameters[param_name] = value
        
        return parameters
    
    def _update_reasoning_trace(self, state: AgentState) -> AgentState:
        """Update reasoning trace with current decision"""
        
        reasoning_trace = state.get("reasoning_trace", [])
        
        action = state["current_action"]
        immediate_obj = state["immediate_objective"]
        
        reasoning = f"Step {len(reasoning_trace) + 1}: Action={action}, Objective={immediate_obj}"
        
        if action == ActionType.TOOL_CALL_PREREQUISITE:
            pending_calls = state.get("pending_tool_calls", [])
            if pending_calls:
                reasoning += f", Calling={pending_calls[0].tool_name}"
        
        reasoning_trace.append(reasoning)
        state["reasoning_trace"] = reasoning_trace
        
        return state


class ExecutorNode:
    """Enhanced executor with better error handling and result processing"""
    
    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools
    
    def __call__(self, state: AgentState) -> AgentState:
        """Execute pending tool calls"""
        
        pending_calls = state.get("pending_tool_calls", [])
        if not pending_calls:
            state["current_action"] = ActionType.RESPOND
            return state
        
        tool_call = pending_calls[0]
        
        try:
            # Execute the tool
            result = self._execute_tool(tool_call)
            
            # Process the result
            state = self._process_tool_result(state, tool_call, result)
            
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            state = self._handle_tool_error(state, tool_call, str(e))
        
        return state
    
    def _execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute the actual tool call"""
        
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool = self.tools[tool_name]
        result = tool.invoke(parameters)
        
        return {"status": "success", "data": result, "tool_name": tool_name}
    
    def _process_tool_result(self, state: AgentState, tool_call: ToolCall, result: Dict[str, Any]) -> AgentState:
        """Process successful tool execution result"""
        
        # Move from pending to completed
        pending_calls = state.get("pending_tool_calls", [])
        completed_calls = state.get("completed_tool_calls", [])
        
        if pending_calls and pending_calls[0] == tool_call:
            pending_calls.pop(0)
        
        completed_calls.append(tool_call)
        
        # Store intermediate results
        intermediate_results = state.get("intermediate_results", {})
        
        # Extract relevant data from result for parameter resolution
        if tool_call.call_type == ActionType.TOOL_CALL_PREREQUISITE:
            # Parse result to extract parameter values
            extracted_params = self._extract_parameters_from_result(result, tool_call)
            intermediate_results.update(extracted_params)
        
        # Update state
        state["pending_tool_calls"] = pending_calls
        state["completed_tool_calls"] = completed_calls
        state["intermediate_results"] = intermediate_results
        
        # Determine next action
        if tool_call.call_type == ActionType.TOOL_CALL_PRIMARY:
            state["current_action"] = ActionType.RESPOND
            state["final_response"] = self._format_final_response(result)
        else:
            # Go back to planner for next step
            state["current_action"] = ActionType.TOOL_CALL_PREREQUISITE
        
        return state
    
    def _extract_parameters_from_result(self, result: Dict[str, Any], tool_call: ToolCall) -> Dict[str, Any]:
        """Extract parameter values from tool result"""
        
        # This would be tool-specific logic
        # For now, return the raw result data
        data = result.get("data", {})
        
        if isinstance(data, dict):
            return data
        else:
            return {"result": data}
    
    def _format_final_response(self, result: Dict[str, Any]) -> str:
        """Format the final response for the user"""
        
        data = result.get("data", {})
        
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return json.dumps(data, indent=2)
        else:
            return str(data)
    
    def _handle_tool_error(self, state: AgentState, tool_call: ToolCall, error: str) -> AgentState:
        """Handle tool execution errors"""
        
        pending_calls = state.get("pending_tool_calls", [])
        failed_calls = state.get("failed_tool_calls", [])
        
        if pending_calls and pending_calls[0] == tool_call:
            pending_calls.pop(0)
        
        failed_calls.append(tool_call)
        
        state["pending_tool_calls"] = pending_calls
        state["failed_tool_calls"] = failed_calls
        state["error_context"] = {"tool": tool_call.tool_name, "error": error}
        
        # Increment retry count
        retry_count = state.get("retry_count", 0) + 1
        state["retry_count"] = retry_count
        
        if retry_count >= state.get("max_retries", 3):
            state["current_action"] = ActionType.INSUFFICIENT_INFO
        else:
            # Go back to planner to try alternative approach
            state["current_action"] = ActionType.TOOL_CALL_PREREQUISITE
        
        return state


class SynthesizerNode:
    """Enhanced synthesizer with context-aware response generation"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate final response with context"""
        
        if state.get("final_response"):
            # Already have final response from executor
            return self._enhance_final_response(state)
        
        # Generate response based on current state
        return self._generate_contextual_response(state)
    
    def _enhance_final_response(self, state: AgentState) -> AgentState:
        """Enhance existing final response with context"""
        
        user_query = state["user_query"]
        final_response = state["final_response"]
        reasoning_trace = state.get("reasoning_trace", [])
        
        enhancement_prompt = f"""
        Enhance this response with appropriate context for the user.
        
        Original Query: {user_query}
        Raw Response: {final_response}
        Processing Steps: {reasoning_trace}
        
        Create a user-friendly response that:
        1. Directly answers the user's question
        2. Provides the requested information clearly
        3. Includes relevant context if helpful
        4. Maintains professional tone
        
        Enhanced Response:
        """
        
        try:
            response = self.llm.invoke(enhancement_prompt)
            state["final_response"] = response.content.strip()
        except Exception as e:
            logger.error(f"Response enhancement failed: {str(e)}")
            # Keep original response if enhancement fails
        
        state["processing_status"] = "completed"
        state["should_continue"] = False
        
        return state
    
    def _generate_contextual_response(self, state: AgentState) -> AgentState:
        """Generate response when no direct answer available"""
        
        user_query = state["user_query"]
        current_action = state.get("current_action")
        error_context = state.get("error_context")
        reasoning_trace = state.get("reasoning_trace", [])
        
        if current_action == ActionType.INSUFFICIENT_INFO:
            response = self._generate_clarification_request(state)
        else:
            response = self._generate_status_response(state)
        
        state["final_response"] = response
        state["processing_status"] = "completed"
        state["should_continue"] = False
        
        return state
    
    def _generate_clarification_request(self, state: AgentState) -> str:
        """Generate request for user clarification"""
        
        user_query = state["user_query"]
        gaps = state.get("parameter_gaps", [])
        error_context = state.get("error_context", {})
        
        clarification_prompt = f"""
        Generate a helpful clarification request for the user.
        
        User Query: {user_query}
        Missing Information: {gaps}
        Error Context: {error_context}
        
        Create a polite, specific request that helps the user provide the missing information needed to complete their request.
        """
        
        try:
            response = self.llm.invoke(clarification_prompt)
            return response.content.strip()
        except:
            return f"I need more information to help with your request: {user_query}. Could you please provide additional details?"
    
    def _generate_status_response(self, state: AgentState) -> str:
        """Generate status response for other scenarios"""
        
        return "I'm processing your request. Please wait while I gather the necessary information."
