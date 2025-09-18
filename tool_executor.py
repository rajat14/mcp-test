from typing import Dict, Any, List
from mcp import Tool
import httpx

from src.core.config.config_refs import ServerConfig
from src.core.instrumentation.logger.logger import Logger
from src.domain.agents.base_prompt_library import PromptLibrary
from src.infra.prompt_libraries.prompt_library_factory import PromptLibraryFactory

logger = Logger()

class ToolExecutor:
    def __init__(self, default_prompt_path_for_lookups: str, tools: Dict[str, Tool] = None, route_mappings: Dict = None):
        self.base_url = f"http://{ServerConfig.host}:{ServerConfig.port}{ServerConfig.api.endpoint_prefix}"
        self.tools = tools
        self.route_mappings = route_mappings
        self.prompt_library: PromptLibrary = PromptLibraryFactory.get_prompt_library_via_config(
            default_prompt_path_for_lookups=default_prompt_path_for_lookups
        )

    async def select_and_execute_tool(self, query: str) -> Any:
        """Analyze query, execute multiple tools if needed, and respond in natural language"""

        available_tools = []
        for tool_name, tool in self.tools.items():
            # Extract actual parameter names from the schema
            param_info = self._get_parameter_info(tool.inputSchema)

            available_tools.append({
                "name": tool_name,
                "description": tool.description,
                "parameters": param_info  # Now includes actual field names
            })

        # DEBUG: Print what tools are available
        logger.info("=== AVAILABLE TOOLS FOR LLM ===")
        for tool in available_tools:
            logger.info(f"Name: {tool['name']}")
            logger.info(f"Description: {tool['description']}")
            logger.info(f"Parameters: {tool['parameters']}")
            logger.info("---")

        # Use LLM to create execution plan
        execution_plan = await self._create_execution_plan(query, available_tools)

        logger.info(f"=== EXECUTION PLAN ===")
        logger.info(execution_plan)

        # Execute the plan
        results = await self._execute_plan(execution_plan)

        # Generate natural response from all results
        natural_response = await self._generate_comprehensive_response(query, results, execution_plan)

        return {
            "natural_response": natural_response,
            "execution_plan": execution_plan,
            "tools_used": [step["tool"] for step in execution_plan.get("steps", [])],
            "results": results
        }

    async def execute_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """Execute a specific tool with given parameters"""
        logger.info(f"Executing tool: {tool_name}")
        logger.info(f"Parameters: {parameters}")

        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}

        # Use stored route mappings instead of parsing descriptions
        if not hasattr(self, 'route_mappings') or tool_name not in self.route_mappings:
            return {"error": f"No route mapping found for tool {tool_name}"}

        route_info = self.route_mappings[tool_name]
        logger.info(f"Route info: {route_info}")

        try:
            # Build the API call
            path = route_info["path"]
            method = route_info["methods"][0].lower()
            full_url = f"{self.base_url}{path}"

            logger.info(f"Making {method.upper()} request to: {full_url}")

            # Extract parameters correctly
            query_params = parameters.get("query_params", {})
            request_body = parameters.get("request_body", {})

            # Handle path parameters - extract them from request_body if they match path placeholders
            final_path = path
            remaining_body = request_body.copy()

            # Find path parameters in the URL (e.g., {workflow_name})
            import re
            path_param_pattern = r'\{([^}]+)\}'
            path_params = re.findall(path_param_pattern, path)

            for param_name in path_params:
                if param_name in request_body:
                    # Replace the path parameter and remove from request body
                    final_path = final_path.replace(f"{{{param_name}}}", str(request_body[param_name]))
                    remaining_body.pop(param_name, None)
                    logger.info(f"Substituted path parameter {param_name}: {request_body[param_name]}")

            full_url = f"{self.base_url}{final_path}"
            logger.info(f"Making {method.upper()} request to: {full_url}")
            logger.info(f"Remaining request body: {remaining_body}")

            import httpx
            async with httpx.AsyncClient(timeout=600.0) as client:
                if method == "post":
                    response = await client.post(
                        full_url,
                        json=request_body,
                        params=query_params
                    )
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
            logger.info(f"Error executing tool {tool_name}: {str(e)}")
            return {"error": f"Failed to execute tool: {str(e)}"}

    def _get_parameter_info(self, input_schema: Dict) -> Dict:
        """Extract readable parameter information from tool schema"""
        param_info = {}

        properties = input_schema.get("properties", {})

        # Extract query parameters
        if "query_params" in properties:
            query_props = properties["query_params"].get("properties", {})
            if query_props:
                param_info["query_params"] = list(query_props.keys())

        # Extract request body parameters
        if "request_body" in properties:
            body_props = properties["request_body"].get("properties", {})
            if body_props:
                param_info["request_body"] = list(body_props.keys())

        return param_info

    async def _create_execution_plan(self, query: str, available_tools: List[Dict]) -> Dict[str, Any]:
        """Create a multi-step execution plan using LLM"""

        tools_info = "\n".join([
            f"- {tool['name']}: {tool['description']} (params: {tool['parameters']})"
            for tool in available_tools
        ])

        prompt = self.prompt_library.retrieve_prompt_via_defaults(
            prompt_name="azure_openai/create_execution_plan",
            placeholder_names_and_values={
                "tools_info": tools_info,
                "query": query
            }
        )

        try:
            from src.services.inference.inference_service import InferenceService
            response = InferenceService().query_via_config(
                llm_connector_name="azure_openai",
                prompt=prompt
            )

            import json
            result = json.loads(response.replace("```json", "").replace("```", "").strip())
            return result

        except Exception as e:
            logger.info(f"Error creating execution plan: {e}")
            return {"action": "none", "reasoning": "Failed to analyze query", "steps": []}

    async def _enhance_parameters_with_context(self, parameters: Dict[str, Any],
                                               previous_results: Dict[str, Any],
                                               depends_on: List[int]) -> Dict[str, Any]:
        """Enhance parameters using results from previous steps"""
        enhanced_params = parameters.copy()

        # Special handling for SQL execution step
        if depends_on:
            for dep_step in depends_on:
                step_result = previous_results.get(f"step_{dep_step}", {})
                if step_result.get("success"):
                    result_data = step_result.get("result", {})

                    # If this is the Execute Sql step and previous step generated SQL
                    if "sql_query" in enhanced_params.get("request_body", {}):
                        # Extract SQL from previous step
                        if "result" in result_data and "sql" in result_data["result"]:
                            sql_query = result_data["result"]["sql"]
                            enhanced_params["request_body"]["sql_query"] = sql_query
                            logger.info(f"Enhanced SQL parameter: {sql_query}")

        return enhanced_params

    async def _generate_comprehensive_response(self, original_query: str,
                                               results: Dict[str, Any],
                                               execution_plan: Dict[str, Any]) -> str:
        """Generate natural language response from multiple tool results"""

        # Check if this was a brain-only response
        if results.get("method") == "llm_knowledge":
            return results.get("brain_response", "I tried to answer using my knowledge base.")

        prompt = self.prompt_library.retrieve_prompt_via_defaults(
            prompt_name="azure_openai/generate_comprehensive_response",
            placeholder_names_and_values={
                "original_query": original_query,
                "execution_plan": execution_plan,
                "results": results
            }
        )

        try:
            from src.services.inference.inference_service import InferenceService
            response = InferenceService().query_via_config(
                llm_connector_name="azure_openai",
                prompt=prompt
            )
            return response.strip()

        except Exception as e:
            logger.info(f"Error generating comprehensive response: {e}")

            # Fallback response
            successful_results = [r for r in results.values() if r.get("success")]
            if successful_results:
                return f"I executed {len(successful_results)} operations to answer your query. Here's what I found: {successful_results}"
            else:
                return "I tried to help with your request but encountered some issues. Please try rephrasing your question."


    async def _execute_plan(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned steps"""
        action = execution_plan.get("action")
        steps = execution_plan.get("steps", [])
        results = {}

        if action == "none":
            original_query = execution_plan.get("original_query", "")
            reasoning = execution_plan.get("reasoning", "")

            brain_response = await self._use_llm_knowledge(original_query, reasoning)
            return {
                "message": "Used AI knowledge base",
                "brain_response": brain_response,
                "method": "llm_knowledge"
            }

        elif action == "single":
            # Execute single tool
            if steps:
                step = steps[0]
                tool_name = step["tool"]
                parameters = step["parameters"]

                try:
                    result = await self.execute_tool(tool_name, parameters)
                    results[f"step_{step.get('step', 0)}"] = {
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result,
                        "success": True
                    }
                except Exception as e:
                    results[f"step_{step.get('step', 0)}"] = {
                        "tool": tool_name,
                        "parameters": parameters,
                        "error": str(e),
                        "success": False
                    }
        elif action == "sequential":
            # Execute tools in sequence
            for step in steps:
                tool_name = step["tool"]
                parameters = step["parameters"]
                depends_on = step.get("depends_on", [])

                # Check if dependencies are satisfied
                if depends_on:
                    dependency_failed = False
                    for dep_step in depends_on:
                        if not results.get(f"step_{dep_step}", {}).get("success", False):
                            dependency_failed = True
                            break

                    if dependency_failed:
                        results[f"step_{step.get('step', 0)}"] = {
                            "tool": tool_name,
                            "error": "Dependency failed",
                            "success": False
                        }
                        continue

                    # Use results from previous steps to enhance parameters
                    parameters = await self._enhance_parameters_with_context(parameters, results, depends_on)

                try:
                    result = await self.execute_tool(tool_name, parameters)
                    results[f"step_{step.get('step', 0)}"] = {
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result,
                        "success": True
                    }
                except Exception as e:
                    results[f"step_{step.get('step', 0)}"] = {
                        "tool": tool_name,
                        "parameters": parameters,
                        "error": str(e),
                        "success": False
                    }

        elif action == "parallel":
            # Execute tools in parallel
            import asyncio

            async def execute_step(step):
                tool_name = step["tool"]
                parameters = step["parameters"]

                try:
                    result = await self.execute_tool(tool_name, parameters)
                    return f"step_{step.get('step', 0)}", {
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result,
                        "success": True
                    }
                except Exception as e:
                    return f"step_{step.get('step', 0)}", {
                        "tool": tool_name,
                        "parameters": parameters,
                        "error": str(e),
                        "success": False
                    }

            # Execute all steps concurrently
            tasks = [execute_step(step) for step in steps]
            step_results = await asyncio.gather(*tasks)

            for step_key, step_result in step_results:
                results[step_key] = step_result

        return results


    async def _use_llm_knowledge(self, query: str, reasoning: str) -> str:
        """Use LLM's own knowledge when no tools are needed"""

        prompt = self.prompt_library.retrieve_prompt_via_defaults(
            prompt_name="azure_openai/use_llm_knowledge",
            placeholder_names_and_values={
                "query": query,
                "reasoning": reasoning
            }
        )

        try:
            from src.services.inference.inference_service import InferenceService
            response = InferenceService().query_via_config(
                llm_connector_name="azure_openai",
                prompt=prompt
            )
            return response.strip()

        except Exception as e:
            logger.info(f"Error using LLM brain: {e}")
            return "I understand your question, but I'm having trouble accessing my knowledge base right now. Could you try rephrasing your question?"
