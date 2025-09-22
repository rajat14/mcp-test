from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum

class ActionType(str, Enum):
    """Available action types for the enhanced planner"""
    TOOL_CALL_PRIMARY = "tool_call_primary"
    TOOL_CALL_PREREQUISITE = "tool_call_prerequisite" 
    RESPOND = "respond"
    INSUFFICIENT_INFO = "insufficient_info"

@dataclass
class ToolCall:
    """Represents a tool call with metadata"""
    tool_name: str
    parameters: Dict[str, Any]
    call_type: ActionType
    reasoning: str
    expected_output: str
    prerequisite_for: Optional[str] = None  # Which tool this is a prerequisite for

@dataclass
class Parameter:
    """Represents a required parameter and its sources"""
    name: str
    description: str
    required: bool
    current_value: Any = None
    possible_sources: List[str] = field(default_factory=list)  # Tools that can provide this
    obtained_from: Optional[str] = None  # Which tool actually provided it

@dataclass
class Goal:
    """Represents the primary objective and progress"""
    description: str
    target_tool: str
    required_parameters: List[Parameter]
    completion_percentage: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)

@dataclass
class ExecutionStep:
    """Represents a step in the execution chain"""
    step_number: int
    action_type: ActionType
    tool_call: Optional[ToolCall]
    result: Optional[Dict[str, Any]] = None
    status: Literal["pending", "executing", "completed", "failed"] = "pending"
    error_message: Optional[str] = None
    timestamp: Optional[str] = None

class AgentState(TypedDict):
    """
    Enhanced state for intelligent orchestration workflow
    
    This state maintains comprehensive context for complex multi-step
    tool orchestration scenarios
    """
    
    # Input and Output
    user_query: str
    final_response: str
    
    # Current Processing Context
    current_node: str
    current_action: ActionType
    processing_status: Literal["active", "waiting", "completed", "failed"]
    
    # Goal and Objective Tracking
    primary_goal: Optional[Goal]
    immediate_objective: str  # What we're trying to accomplish right now
    
    # Tool and Parameter Management  
    available_tools: Dict[str, Dict[str, Any]]  # Tool catalog with schemas
    pending_tool_calls: List[ToolCall]
    completed_tool_calls: List[ToolCall] 
    failed_tool_calls: List[ToolCall]
    
    # Parameter Resolution
    required_parameters: Dict[str, Parameter]  # Parameters needed for primary goal
    resolved_parameters: Dict[str, Any]  # Successfully obtained parameter values
    parameter_gaps: List[str]  # Still missing these parameters
    
    # Execution Chain
    execution_steps: List[ExecutionStep]
    current_step: int
    
    # Context and Memory
    conversation_context: List[Dict[str, str]]  # Previous messages if any
    intermediate_results: Dict[str, Any]  # Results from prerequisite calls
    dependency_graph: Dict[str, List[str]]  # Which tools depend on which others
    
    # Decision Making Context
    reasoning_trace: List[str]  # Why each decision was made
    alternative_paths: List[Dict[str, Any]]  # Backup plans if current path fails
    confidence_score: float  # Confidence in current approach (0-1)
    
    # Error Handling
    error_context: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int
    
    # Metadata
    session_id: str
    start_time: str
    last_updated: str
    
    # Control Flow
    next_node: Optional[str]  # Override for routing decisions
    should_continue: bool
