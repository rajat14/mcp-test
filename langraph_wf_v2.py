from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from src.services.mcp.mcp_service import MCPService
from src.core.instrumentation.logger.logger import Logger
from src.core.config.config_refs import ConfigNodes

logger = Logger()


class LanggraphMCPService:
    """Service that uses LangGraph to orchestrate MCP tools with dynamic branching"""

    def __init__(self):
        self.llm_config = ConfigNodes.get_llm_config("azure_openai")
        self.llm = AzureChatOpenAI(
            azure_deployment=self.llm_config.model,
            azure_endpoint=self.llm_config.azure_endpoint,
            api_key=self.llm_config.api_key,
            api_version=self.llm_config.api_version,
            temperature=0,
        )
        self.mcp_service = MCPService()
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for MCP tool orchestration"""

        # Define state structure
        class AgentState(Dict):
            messages: List[Any]
            analysis: str
            selected_tools: List[str]
            tool_params: Dict[str, Any]
            results: Dict[str, Any]
            final_response: str

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("plan_or_fallback", self._plan_or_fallback)
        workflow.add_node("generate_params", self._generate_params)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("synthesize_response", self._synthesize_response)
        workflow.add_node("fallback_response", self._fallback_response)

        # Add edges
        workflow.add_edge("analyze_query", "plan_or_fallback")
        workflow.add_conditional_edges(
            "plan_or_fallback",
            lambda state: "generate_params" if state.get("selected_tools") else "fallback_response",
            {"generate_params": "generate_params", "fallback_response": "fallback_response"},
        )
        workflow.add_edge("generate_params", "execute_tools")
        workflow.add_edge("execute_tools", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        workflow.add_edge("fallback_response", END)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        return workflow.compile()

    # -----------------------
    # Node implementations
    # -----------------------

    async def _analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user query to extract intent/entities"""
        query = state.get("messages", [])[-1]

        prompt = [
            SystemMessage(content="You are an intent analyzer."),
            HumanMessage(content=f"User query: {query}\nExtract intent, entities, and context."),
        ]
        response = await self.llm.ainvoke(prompt)

        state["analysis"] = response.content
        logger.info(f"Query analysis: {response.content}")
        return state

    async def _plan_or_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Decide which tools (if any) to call"""
        query = state.get("messages", [])[-1]
        available_tools = self.mcp_service.get_available_tools()

        tool_prompt = [
            SystemMessage(content="You are a planner that selects tools."),
            HumanMessage(
                content=(
                    f"Available tools: {list(available_tools.keys())}\n"
                    f"User query: {query}\n"
                    "Select the most relevant tool names. "
                    "Return comma-separated names, or 'NONE' if none apply."
                )
            ),
        ]

        response = await self.llm.ainvoke(tool_prompt)
        selected = (
            []
            if "NONE" in response.content.upper()
            else [t.strip() for t in response.content.split(",") if t.strip()]
        )

        state["selected_tools"] = selected
        logger.info(f"Selected tools: {selected}")
        return state

    async def _generate_params(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate params for selected tools (schemas handled elsewhere)"""
        query = state.get("messages", [])[-1]
        selected = state.get("selected_tools", [])
        params = {}

        for tool in selected:
            params[tool] = {"query": query}  # Simplified: real module handles schema filling
            logger.info(f"Generated params for {tool}: {params[tool]}")

        state["tool_params"] = params
        return state

    async def _execute_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute selected tools"""
        results = {}
        for tool in state.get("selected_tools", []):
            try:
                params = state.get("tool_params", {}).get(tool, {})
                result = await self.mcp_service.execute_specific_tool(tool, params)
                results[tool] = result
                logger.info(f"Executed {tool}: {result}")
            except Exception as e:
                logger.error(f"Error executing {tool}: {e}")
                results[tool] = {"error": str(e)}

        state["results"] = results
        return state

    async def _synthesize_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response from tool results"""
        query = state.get("messages", [])[-1]
        results = state.get("results", {})

        prompt = [
            SystemMessage(content="You are a response synthesizer."),
            HumanMessage(
                content=(
                    f"User query: {query}\n"
                    f"Tool results: {results}\n"
                    "Provide a clear user-facing answer. Mention which tool provided which info."
                )
            ),
        ]

        response = await self.llm.ainvoke(prompt)
        state["final_response"] = response.content
        logger.info(f"Final synthesized response: {response.content}")
        return state

    async def _fallback_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback if no tool can answer the query"""
        query = state.get("messages", [])[-1]
        msg = f"Sorry, I don’t have a tool that can answer: '{query}'. Please rephrase or try another query."
        state["final_response"] = msg
        logger.info(f"Fallback response: {msg}")
        return state

    # -----------------------
    # Public API
    # -----------------------

    async def process_query_with_langgraph(self, user_query: str) -> str:
        """Process query through LangGraph workflow"""
        initial_state = {
            "messages": [user_query],
            "analysis": "",
            "selected_tools": [],
            "tool_params": {},
            "results": {},
            "final_response": "",
        }
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["final_response"]
