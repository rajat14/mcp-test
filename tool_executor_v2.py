# ====================================================================
# File: src/infra/support/mcp/tool_executor/tool_executor.py (MODIFIED)
# ====================================================================
from typing import Dict, Any, List
import httpx
import json
import re

# Internal dependencies
from mcp import Tool # Assuming external definition of Tool class
from src.core.config.config_refs import ServerConfig
from src.core.instrumentation.logger.logger import Logger
from src.domain.agents.base_prompt_library import PromptLibrary
from src.infra.prompt_libraries.prompt_library_factory import PromptLibraryFactory
from src.services.inference.inference_service import InferenceService

logger = Logger()

class ToolExecutor:
    """
    Core execution logic. It handles API calls and response synthesis, 
    but relies on LangGraph for planning and orchestration.
    """
    def __init__(self, default_prompt_path_for_lookups: str, tools: Dict[str, Tool] = None, route_mappings: Dict = None):
        self.base_url = f"http://{ServerConfig.host}:{ServerConfig.port}{ServerConfig.api.endpoint_prefix}"
        self.tools = tools if tools is not None else {}
        self.route_mappings = route_mappings if route_mappings is not None else {}
        self.prompt_library: PromptLibrary = PromptLibraryFactory.get_prompt_library_via_config(
            default_prompt_path_for_lookups=default_prompt_path_for_lookups
        )
        self.logger = logger # Reference logger for use in LangGraph nodes

    # --- Core Tool Execution Logic (KEPT) ---

    async def execute_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """Execute a specific tool with given parameters."""
        logger.info(f"Executing tool: {tool_name} with params: {parameters}")

        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        if tool_name not in self.route_mappings:
            return {"error": f"No route mapping found for tool {tool_name}"}

        route_info = self.route_mappings[tool_name]
        
        try:
            path = route_info["path"]
            method = route_info["methods"][0].lower()
            
            # Extract parameters correctly
            query_params = parameters.get("query_params", {})
            request_body = parameters.get("request_body", {})
            
            # Handle path parameters (complex original logic)
            final_path = path
            path_param_pattern = r'\{([^}]+)\}'
            path_params = re.findall(path_param_pattern, path)
            
            for param_name in path_params:
                if param_name in request_body:
                    final_path = final_path.replace(f"{{{param_name}}}", str(request_body[param_name]))
            
            full_url = f"{self.base_url}{final_path}"
            
            async with httpx.AsyncClient(timeout=600.0) as client:
                if method == "post":
                    response = await client.post(full_url, json=request_body, params=query_params)
                elif method == "get":
                    response = await client.get(full_url, params=query_params)
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}

                response.raise_for_status()
                
                if response.headers.get("content-type", "").startswith("application/json"):
                    return {"result": response.json(), "status_code": response.status_code}
                else:
                    return {"result": response.text, "status_code": response.status_code}

        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {str(e)}")
            return {"error": f"Failed to execute tool: {str(e)}"}


    def _get_parameter_info(self, input_schema: Dict) -> Dict:
        """Extract readable parameter information from tool schema (KEPT)."""
        param_info = {}
        properties = input_schema.get("properties", {})
        if "query_params" in properties:
            param_info["query_params"] = list(properties["query_params"].get("properties", {}).keys())
        if "request_body" in properties:
            param_info["request_body"] = list(properties["request_body"].get("properties", {}).keys())
        return param_info

    # --- Synthesis Logic (REPURPOSED for LangGraph) ---

    async def synthesize_response(self, original_query: str, results: Dict[str, Any], execution_plan: Dict[str, Any]) -> str:
        """Generate natural language response from all tool results (used by Synthesizer Node)."""

        prompt = self.prompt_library.retrieve_prompt_via_defaults(
            prompt_name="azure_openai/generate_comprehensive_response",
            placeholder_names_and_values={
                "original_query": original_query,
                "execution_plan": json.dumps(execution_plan),
                "results": json.dumps(results)
            }
        )

        try:
            # Use the LLM to synthesize the final answer
            response = InferenceService().query_via_config(
                llm_connector_name="azure_openai",
                prompt=prompt
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating comprehensive response: {e}")
            return "I executed operations but failed to generate a comprehensive response."
