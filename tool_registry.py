from typing import List, Dict
import importlib
from pathlib import Path
from mcp import Tool

from src.core.instrumentation.logger.logger import Logger

logger = Logger()

class ToolRegistry:
    """Base class for tools used in MCP"""
    def __init__(self, tools: Dict[str, Tool] = None, route_mappings: Dict = None):
        self.tools = tools
        self.route_mappings = route_mappings


    def load_router_tools(self):
        """Load all router endpoints as MCP tools"""
        current_dir = Path(__file__).parents[4] # backend/src/
        routers_path = current_dir / "web" / "routers"

        logger.info(f"Looking for routers in: {routers_path}")
        logger.info(f"Directory exists: {routers_path.exists()}")

        if not routers_path.exists():
            logger.info("Routers directory not found!")
            return self.tools, self.route_mappings

        router_files = list(routers_path.glob("*.py"))
        logger.info(f"Found router files: {[f.name for f in router_files]}")

        for router_file in router_files:
            if router_file.name.startswith("__"):
                continue

            module_name = f"src.web.routers.{router_file.stem}"
            logger.info(f"Attempting to import: {module_name}")

            try:
                module = importlib.import_module(module_name)
                logger.info(f"Successfully imported {module_name}")
                self._extract_tools_from_router(module, router_file.stem)
            except Exception as e:
                logger.info(f"Error processing {module_name}: {e}")

        logger.info(f"\n=== ALL LOADED TOOLS ===")
        for tool_name, tool in self.tools.items():
            logger.info(f"Tool: {tool_name}")
            logger.info(f"Description: {tool.description}")
            if hasattr(self, 'route_mappings') and tool_name in self.route_mappings:
                route_info = self.route_mappings[tool_name]
                logger.info(f"Route: {route_info['methods']} {route_info['path']}")
            logger.info("---")
        logger.info(f"=== TOTAL: {len(self.tools)} tools ===\n")

        return self.tools, self.route_mappings



    def _extract_tools_from_router(self, module, router_name: str):
        """Extract FastAPI endpoints as MCP tools"""
        logger.info(f"Extracting tools from {router_name}")

        if hasattr(module, 'router'):
            router = module.router
            logger.info(f"Router type: {type(router)}")
            logger.info(f"Router routes: {router.routes}")
            logger.info(f"Found router with {len(router.routes)} routes")
            logger.info(f"Found router with {len(router.routes)} routes")

            for route in router.routes:
                if hasattr(route, 'methods') and hasattr(route, 'path'):
                    methods = [m for m in route.methods if m != 'OPTIONS']
                    if not methods:
                        continue

                    _tool_name = f"{route.name or route.path.replace('/', '_').replace('{', '').replace('}', '')}"
                    _tool_name = _tool_name.replace('__', '_').strip('_')

                    parts = _tool_name.split('_')
                    tool_name = ' '.join(word.capitalize() for word in parts if word)

                    # Extract MCP descriptions
                    description, scenarios = self._extract_mcp_metadata(route)

                    # Get the actual parameter schema
                    param_schema = self._extract_parameters(route)

                    tool = Tool(
                        name=tool_name,
                        description=description or f"API endpoint: {'/'.join(methods)} {route.path}",
                        inputSchema=param_schema  # Use the schema directly
                    )

                    self.tools[tool_name] = tool

                    # Store route mapping
                    self.route_mappings[tool_name] = {
                        "router_name": router_name,
                        "path": route.path,
                        "methods": methods,
                        "endpoint": route.endpoint
                    }

                    logger.info(f"Added tool: {tool_name} -> {route.path}")
                    logger.info(f"Schema: {param_schema}")

    def _extract_mcp_metadata(self, route):
        """Extract MCP-specific metadata from route's openapi_extra"""
        description = None
        scenarios = None

        if hasattr(route, 'endpoint') and hasattr(route.endpoint, '__dict__'):
            # Check if the endpoint function has openapi_extra metadata
            endpoint_func = route.endpoint
            if hasattr(endpoint_func, 'openapi_extra'):
                extra = endpoint_func.openapi_extra
                description = extra.get('mcp_description')
                scenarios = extra.get('mcp_scenarios')

            # Also check the route's own openapi_extra
            if hasattr(route, 'openapi_extra') and route.openapi_extra:
                extra = route.openapi_extra
                description = description or extra.get('mcp_description')
                scenarios = scenarios or extra.get('mcp_scenarios')

        return description, scenarios

    def _extract_parameters(self, route):
        """Extract parameters from route for MCP tool schema"""
        schema = {
            "type": "object",
            "properties": {
                "query_params": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True
                },
                "request_body": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True
                }
            },
            "required": []
        }

        # Extract Pydantic model fields
        if hasattr(route, 'endpoint') and hasattr(route.endpoint, '__annotations__'):
            for param_name, param_type in route.endpoint.__annotations__.items():
                if hasattr(param_type, '__fields__'):  # It's a Pydantic model
                    for field_name, field_info in param_type.__fields__.items():
                        schema["properties"]["request_body"]["properties"][field_name] = {
                            "type": "string",
                            "description": f"Field {field_name} for {param_type.__name__}"
                        }
                        logger.info(f"Added field: {field_name} for {param_type.__name__}")

        return schema
