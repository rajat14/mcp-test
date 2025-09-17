import os
import json
import time
import uuid
import logging
import asyncio
from typing import Any, Dict, List, Optional


import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

from app.core.session_manager import SessionManager

session_mgr = SessionManager()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])

# ----------------------
# Azure OpenAI Config
# ----------------------
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT_NAME")  # deployment name you set in Azure
AZURE_TEMPERATURE = float(os.environ.get("AZURE_TEMPERATURE", "0.0"))
AZURE_MAX_TOKENS = int(os.environ.get("AZURE_MAX_TOKENS", "1024"))

if not (AZURE_ENDPOINT and AZURE_KEY and AZURE_DEPLOYMENT):
    raise RuntimeError("Azure OpenAI env vars missing: set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_DEPLOYMENT_NAME")

_azure_client = OpenAIClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_KEY))


# ----------------------
# LLM Client (Azure)
# ----------------------
class AzureLLMClient:
    def __init__(self, deployment: str = AZURE_DEPLOYMENT, temperature: float = AZURE_TEMPERATURE, max_tokens: int = AZURE_MAX_TOKENS):
        self.deployment = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, tool_registry_summary: str, conversation_context: str, user_query: str, few_shot_examples: str) -> str:
        system = (
            "You are an assistant that must choose which tools to call to answer a user's question.\n"
            "Rules:\n"
            "1) Only use tools from the registry.\n"
            "2) Output STRICTLY valid JSON: either {\"plan\":[...]} or {\"clarify\":{...}}\n"
            "3) Each plan step must have step_id, tool, params.\n"
            "4) If uncertain, return a clarify object.\n"
            "5) Never invent tools or output SQL.\n"
        )
        return (
            f"{system}\n\n"
            f"TOOLS:\n{tool_registry_summary}\n\n"
            f"CONTEXT:\n{conversation_context}\n\n"
            f"FEW_SHOT_EXAMPLES:\n{few_shot_examples}\n\n"
            f"USER QUERY:\n{user_query}\n\n"
            "Return only valid JSON."
        )

    def call(self, prompt: str) -> str:
        """Call Azure OpenAI ChatCompletion and return assistant text"""
        response = _azure_client.get_chat_completions(
            deployment_name=self.deployment,
            messages=[{"role": "system", "content": "Planner"},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0]
        if hasattr(choice, "message"):
            return choice.message.get("content")
        return choice.get("message", {}).get("content")


# ----------------------
# Models
# ----------------------
class QueryRequest(BaseModel):
    session_id: str
    query: str


class PlanStep(BaseModel):
    step_id: str
    tool: str
    params: Dict[str, Any]


class Plan(BaseModel):
    steps: List[PlanStep]


class ExecutionResult(BaseModel):
    step_id: str
    tool: str
    success: bool
    status_code: int
    output: Any = None
    error: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    trace_id: str
    plan: Optional[Plan] = None
    results: Optional[List[ExecutionResult]] = None
    response_text: str
    clarify: Optional[Dict[str, Any]] = None

class DryRunResponse(BaseModel):
    session_id: str
    trace_id: str
    raw_output: str
    plan: Optional[Dict[str, Any]] = None
    clarify: Optional[Dict[str, Any]] = None



# ----------------------
# Few-shot examples
# ----------------------
_FEW_SHOT = """
# Example 1
User: "List all tables"
JSON:
{"plan":[{"step_id":"s1","tool":"list_tables","params":{}}]}

# Example 2
User: "Schema for orders"
JSON:
{"plan":[{"step_id":"s1","tool":"get_schema","params":{"table":"orders"}}]}

# Example 3
User: "Where does transactions.amount_usd come from?"
JSON:
{"plan":[{"step_id":"s1","tool":"get_column_lineage","params":{"table":"transactions","column":"amount_usd"}}]}
"""


# ----------------------
# Helpers
# ----------------------
TOOLS_BASE_URL = os.environ.get("TOOLS_BASE_URL", "http://127.0.0.1:8000/tools")
HTTP_TIMEOUT = 8.0


async def fetch_tool_registry() -> Dict[str, Any]:
    url = f"{TOOLS_BASE_URL}/registry"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


def summarize_tool_registry(registry: Dict[str, Any]) -> str:
    lines = []
    for t in registry.get("tools", []):
        lines.append(f"{t.get('name')} | {t.get('description','')} | path={t.get('path')}")
    return "\n".join(lines)


async def validate_plan(plan_obj: Dict[str, Any], registry: Dict[str, Any]) -> Plan:
    tools_map = {t["name"]: t for t in registry.get("tools", [])}
    if "plan" not in plan_obj:
        raise ValueError("No 'plan' key in response")
    steps = []
    for s in plan_obj["plan"]:
        if s["tool"] not in tools_map:
            raise ValueError(f"Unknown tool {s['tool']}")
        steps.append(PlanStep(step_id=s["step_id"], tool=s["tool"], params=s["params"]))
    return Plan(steps=steps)


async def execute_plan(plan: Plan) -> List[ExecutionResult]:
    results = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for step in plan.steps:
            url = f"{TOOLS_BASE_URL}/{step.tool.replace('_','-')}"
            try:
                r = await client.get(url, params=step.params)
                if r.status_code >= 400:
                    results.append(ExecutionResult(step_id=step.step_id, tool=step.tool,
                                                   success=False, status_code=r.status_code,
                                                   error=r.text))
                else:
                    results.append(ExecutionResult(step_id=step.step_id, tool=step.tool,
                                                   success=True, status_code=r.status_code,
                                                   output=r.json()))
            except Exception as e:
                results.append(ExecutionResult(step_id=step.step_id, tool=step.tool,
                                               success=False, status_code=0,
                                               error=str(e)))
    return results


# ----------------------
# Endpoint
# ----------------------
@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    trace_id = f"orch-{uuid.uuid4().hex[:6]}"

    # Save user query to session history
    await session_mgr.add_message(req.session_id, "user", req.query)

    registry = await fetch_tool_registry()
    tool_summary = summarize_tool_registry(registry)

    # Build conversation context
    history_text = await session_mgr.get_history_text(req.session_id, n=6)
    context_summary = await session_mgr.get_context_summary(req.session_id)
    context = f"history:\n{history_text}\n{context_summary}"

    # Build prompt
    llm = AzureLLMClient()
    prompt = llm._build_prompt(tool_summary, conversation_context=context,  # add memory if you want
                               user_query=req.query, few_shot_examples=_FEW_SHOT)

    raw = llm.call(prompt).strip()
    try:
        plan_obj = json.loads(raw)
    except Exception:
        return QueryResponse(session_id=req.session_id, trace_id=trace_id,
                             response_text="I could not parse a valid plan.",
                             clarify={"raw": raw})

    if "clarify" in plan_obj:
        return QueryResponse(session_id=req.session_id, trace_id=trace_id,
                             response_text=plan_obj["clarify"]["question"],
                             clarify=plan_obj["clarify"])

    try:
        plan = await validate_plan(plan_obj, registry)
    except Exception as e:
        return QueryResponse(session_id=req.session_id, trace_id=trace_id,
                             response_text=f"Plan invalid: {e}",
                             clarify={"raw": raw})

    results = await execute_plan(plan)

    # Compose response text
    texts = []
    for r in results:
        if r.success:
            texts.append(f"{r.tool} succeeded")
        else:
            texts.append(f"{r.tool} failed: {r.error}")
    response_text = "\n".join(texts)

    # Update session with outputs + assistant reply
    await session_mgr.update_context(
        req.session_id,
        last_plan=plan.dict(),
        last_outputs=[r.dict() for r in results],
        trace_id=trace_id
    )
    await session_mgr.add_message(req.session_id, "assistant", response_text)

    return QueryResponse(
        session_id=req.session_id,
        trace_id=trace_id,
        plan=plan,
        results=results,
        response_text=response_text
    )

@router.post("/dryrun", response_model=DryRunResponse)
async def dryrun_endpoint(req: QueryRequest):
    trace_id = f"dry-{uuid.uuid4().hex[:6]}"

    # Record user query into history
    await session_mgr.add_message(req.session_id, "user", req.query)

    # Fetch registry + summarize
    registry = await fetch_tool_registry()
    tool_summary = summarize_tool_registry(registry)

    # Build conversation context
    history_text = await session_mgr.get_history_text(req.session_id, n=6)
    context_summary = await session_mgr.get_context_summary(req.session_id)
    context = f"history:\n{history_text}\n{context_summary}"

    # Build prompt
    llm = AzureLLMClient()
    prompt = llm._build_prompt(
        tool_summary,
        conversation_context=context,
        user_query=req.query,
        few_shot_examples=_FEW_SHOT
    )

    # Call Azure LLM
    raw = llm.call(prompt).strip()

    try:
        plan_obj = json.loads(raw)
    except Exception:
        # not valid JSON
        return DryRunResponse(
            session_id=req.session_id,
            trace_id=trace_id,
            raw_output=raw,
            plan=None
        )

    if "clarify" in plan_obj:
        return DryRunResponse(
            session_id=req.session_id,
            trace_id=trace_id,
            raw_output=raw,
            clarify=plan_obj["clarify"]
        )

    return DryRunResponse(
        session_id=req.session_id,
        trace_id=trace_id,
        raw_output=raw,
        plan=plan_obj
    )


