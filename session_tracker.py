import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from utils.logger import logger

@dataclass
class SessionMetrics:
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    query_count: int = 0
    tool_calls_count: int = 0
    server_switches_count: int = 0
    active_servers_used: list = None
    tool_usage: dict = None
    error_count: int = 0
    
    def __post_init__(self):
        if self.active_servers_used is None:
            self.active_servers_used = []
        if self.tool_usage is None:
            self.tool_usage = {}

class SessionTracker:
    def __init__(self):
        self.session_id = self._generate_session_id()
        self.session_dir = "session_metrics"
        self.metrics = SessionMetrics(
            session_id=self.session_id,
            start_time=datetime.now().isoformat()
        )
        self.logger = logger
        self._ensure_session_dir()
        
    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
    
    def _ensure_session_dir(self):
        """Ensure session metrics directory exists"""
        os.makedirs(self.session_dir, exist_ok=True)
    
    def _get_session_file_path(self) -> str:
        """Get the file path for current session metrics"""
        return os.path.join(self.session_dir, f"{self.session_id}.json")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str = "claude-sonnet-4") -> float:
        """Calculate estimated cost based on token usage"""
        # Pricing for Claude Sonnet 4 (example rates - adjust based on actual pricing)
        pricing = {
            "claude-sonnet-4": {
                "input": 0.000003,  # $3 per million input tokens
                "output": 0.000015  # $15 per million output tokens
            },
            "claude-3-5-sonnet": {
                "input": 0.000003,
                "output": 0.000015
            }
        }
        
        rates = pricing.get(model, pricing["claude-sonnet-4"])
        input_cost = input_tokens * rates["input"]
        output_cost = output_tokens * rates["output"]
        return input_cost + output_cost
    
    def track_query(self, query: str, server_name: str):
        """Track a new query"""
        self.metrics.query_count += 1
        if server_name not in self.metrics.active_servers_used:
            self.metrics.active_servers_used.append(server_name)
        self._save_metrics()
    
    def track_llm_usage(self, input_tokens: int, output_tokens: int, model: str = "claude-sonnet-4"):
        """Track LLM token usage and calculate cost"""
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_tokens = self.metrics.total_input_tokens + self.metrics.total_output_tokens
        
        session_cost = self._calculate_cost(input_tokens, output_tokens, model)
        self.metrics.estimated_cost += session_cost
        self._save_metrics()
    
    def track_tool_call(self, tool_name: str, server_name: str):
        """Track tool usage"""
        self.metrics.tool_calls_count += 1
        
        # Track tool usage by name
        if tool_name not in self.metrics.tool_usage:
            self.metrics.tool_usage[tool_name] = {"count": 0, "servers": []}
        
        self.metrics.tool_usage[tool_name]["count"] += 1
        if server_name not in self.metrics.tool_usage[tool_name]["servers"]:
            self.metrics.tool_usage[tool_name]["servers"].append(server_name)
        
        self._save_metrics()
    
    def track_server_switch(self, from_server: str, to_server: str):
        """Track server switches"""
        self.metrics.server_switches_count += 1
        if to_server not in self.metrics.active_servers_used:
            self.metrics.active_servers_used.append(to_server)
        self._save_metrics()
    
    def track_error(self, error_type: str, error_message: str):
        """Track errors"""
        self.metrics.error_count += 1
        self._save_metrics()
        self.logger.error(f"Session {self.session_id} error: {error_type} - {error_message}")
    
    def _save_metrics(self):
        """Save current metrics to file"""
        try:
            # Update end_time on each save
            self.metrics.end_time = datetime.now().isoformat()
            
            # Convert to dict and save
            metrics_dict = asdict(self.metrics)
            file_path = self._get_session_file_path()
            
            with open(file_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save session metrics: {str(e)}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current session metrics"""
        duration = None
        if self.metrics.end_time:
            start = datetime.fromisoformat(self.metrics.start_time)
            end = datetime.fromisoformat(self.metrics.end_time)
            duration = str(end - start)
        
        return {
            "session_id": self.metrics.session_id,
            "duration": duration,
            "queries": self.metrics.query_count,
            "tool_calls": self.metrics.tool_calls_count,
            "server_switches": self.metrics.server_switches_count,
            "total_tokens": self.metrics.total_tokens,
            "estimated_cost_usd": round(self.metrics.estimated_cost, 6),
            "servers_used": self.metrics.active_servers_used,
            "top_tools": sorted(
                [(tool, data["count"]) for tool, data in self.metrics.tool_usage.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "errors": self.metrics.error_count
        }
    
    def finalize_session(self):
        """Mark session as complete and save final metrics"""
        self.metrics.end_time = datetime.now().isoformat()
        self._save_metrics()
        self.logger.info(f"Session {self.session_id} finalized with metrics: {self.get_metrics_summary()}")
