from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.services.mcp.mcp_service import MCPService
from src.core.instrumentation.logger.logger import Logger
from src.core.config.config_refs import ConfigNodes

logger = Logger()

class LanggraphMCPService:
    """Service that uses LangGraph to orchestrate MCP tools"""

    def __init__(self):
        self.llm_config = ConfigNodes.get_llm_config("azure_openai")
        self.llm = AzureChatOpenAI(
            azure_deployment=self.llm_config.model,
            azure_endpoint=self.llm_config.azure_endpoint,     # Your Azure endpoint
            api_key=self.llm_config.api_key,                   # Your Azure API key
            api_version=self.llm_config.api_version,                  # Azure API version
            temperature=0
        )
        self.mcp_service = MCPService()
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for MCP tool orchestration"""

        # Define the state structure
        class AgentState(Dict):
            messages: List[Any]
            next_action: str
            selected_tools: List[str]
            results: Dict[str, Any]
            final_response: str

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("select_tools", self._select_tools)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("synthesize_response", self._synthesize_response)

        # Add edges
        workflow.add_edge("analyze_query", "select_tools")
        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "synthesize_response")
        workflow.add_edge("synthesize_response", END)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        return workflow.compile()

    async def _analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user query to understand intent"""
        query = state.get("messages", [])[-1] if state.get("messages") else ""

        analysis_prompt = f"""
        Analyze this user query and determine what type of information or action is needed:
        Query: {query}
        
        Consider what MCP tools might be useful for this query.
        """

        response = await self.llm.ainvoke([SystemMessage(content=analysis_prompt)])

        state["next_action"] = "tool_selection"
        logger.info(f"Query analysis: {response.content}")

        return state

    async def _select_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate MCP tools based on query analysis"""
        available_tools = self.mcp_service.get_available_tools()
        query = state.get("messages", [])[-1] if state.get("messages") else ""

        tool_selection_prompt = f"""
        Available MCP tools: {list(available_tools)}
        
        User query: {query}
        
        Select the most appropriate tools (by name) to answer this query.
        Return only the tool names as a comma-separated list.
        """

        response = await self.llm.ainvoke([SystemMessage(content=tool_selection_prompt)])
        selected_tools = [tool.strip() for tool in response.content.split(",")]

        state["selected_tools"] = selected_tools
        logger.info(f"Selected tools: {selected_tools}")

        return state

    async def _execute_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute selected MCP tools"""
        selected_tools = state.get("selected_tools", [])
        query = state.get("messages", [])[-1] if state.get("messages") else ""
        results = {}

        for tool_name in selected_tools:
            try:
                # Execute each tool through MCP service
                result = await self.mcp_service.execute_specific_tool(tool_name, {"query": query})
                results[tool_name] = result
                logger.info(f"Executed {tool_name}: {result}")
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                results[tool_name] = {"error": str(e)}

        state["results"] = results
        return state

    async def _synthesize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final response from tool results"""
        results = state.get("results", {})
        query = state.get("messages", [])[-1] if state.get("messages") else ""

        synthesis_prompt = f"""
        User query: {query}
        
        Tool execution results:
        {results}
        
        Synthesize a comprehensive response to the user's query based on these results.
        If there were errors, handle them gracefully.
        """

        response = await self.llm.ainvoke([SystemMessage(content=synthesis_prompt)])

        state["final_response"] = response.content
        logger.info(f"Final synthesized response: {response.content}")

        return state

    async def process_query_with_langgraph(self, user_query: str) -> str:
        """Process user query through LangGraph workflow"""
        initial_state = {
            "messages": [user_query],
            "next_action": "",
            "selected_tools": [],
            "results": {},
            "final_response": ""
        }

        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["final_response"]
