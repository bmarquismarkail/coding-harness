from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Mapping, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import AgentConfig
from .context import build_context
from .model import DualModelProvider, ModelProvider
from .zvec import ZvecGrepError, ZvecGrepSettings, ZvecGrepTool


class AgentState(TypedDict, total=False):
    task: str
    root: str
    context: str
    plan: dict[str, Any]
    zvec_query: str
    zvec_results: str
    subagent_results: Annotated[list[dict[str, Any]], operator.add]
    synthesis: dict[str, Any]


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]) if len(lines) >= 3 else candidate
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return {"text": text}
    return value if isinstance(value, dict) else {"value": value}


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def create_graph(
    config: AgentConfig,
    provider: ModelProvider | None = None,
    zvec_tool: ZvecGrepTool | None = None,
):
    provider = provider or DualModelProvider(config)
    zvec_tool = zvec_tool or ZvecGrepTool(
        ZvecGrepSettings(
            executable=config.zvec.executable,
            mode=config.zvec.mode,  # type: ignore[arg-type]
            query_limit=config.zvec.query_limit,
            timeout_seconds=config.zvec.timeout_seconds,
            embedding=config.zvec.embedding,
        )
    )

    def collect_context(state: AgentState) -> dict[str, Any]:
        return {"context": build_context(state["root"], config.context)}

    def collect_zvec_context(state: AgentState) -> dict[str, Any]:
        query = state.get("zvec_query")
        if not query:
            return {"zvec_results": ""}
        try:
            results = zvec_tool.query(state["root"], query)
        except ZvecGrepError as exc:
            raise RuntimeError(f"zvec-grep query failed: {exc}") from exc
        return {"zvec_results": results}

    def repository_evidence(state: AgentState) -> str:
        zvec_results = state.get("zvec_results", "")
        if not zvec_results:
            return state["context"]
        return (
            state["context"]
            + "\n\n--- ZVEC-GREP NAVIGATION RESULTS ---\n"
            + zvec_results
            + "\nTreat these as navigation hints and verify claims against source files."
        )

    def plan(state: AgentState) -> dict[str, Any]:
        response = provider.invoke(
            _messages(
                """You are the lead coding agent. Use repository evidence, not guesses. """
                """The repository may contain any language; prioritize Python and C/C++ when present. """
                "This is the PLANNING stage only. A separate final synthesis stage will deliver the fix. "
                "Any task requirements about the FINAL output apply only to that later synthesis. "
                "For this stage provide a short plan using only the following keys. "
                """Return JSON with keys summary, constraints, work_items, likely_files, and verification.""",
                f"TASK:\n{state['task']}\n\nREPOSITORY CONTEXT:\n{repository_evidence(state)}",
            ),
            role="main",
        )
        return {"plan": _json_object(response)}

    def route_to_specialists(state: AgentState) -> list[Send]:
        return [
            Send(
                "specialist",
                {
                    "task": state["task"],
                    "context": state["context"],
                    "plan": state.get("plan", {}),
                    "role_name": role,
                },
            )
            for role in config.specialist_roles[: config.parallelism]
        ]

    def specialist(state: AgentState) -> dict[str, Any]:
        role_name = str(state["role_name"])
        response = provider.invoke(
            _messages(
                f"""You are a parallel specialist in the role: {role_name}. """
                "This is the SPECIALIST REVIEW stage. A separate lead synthesis will deliver the final fix; "
                "task requirements about the final output apply to that later stage. "
                "Do not claim a file, symbol, test, or result that is absent from the supplied context. "
                "Return JSON with keys role, findings, evidence, risks, and recommendations. "
                "Evidence must cite a supplied path or say insufficient evidence.",
                f"TASK:\n{state['task']}\n\nLEAD PLAN:\n{json.dumps(state.get('plan', {}), indent=2)}"
                f"\n\nREPOSITORY CONTEXT:\n{repository_evidence(state)}",
            ),
            role="subagent",
        )
        result = _json_object(response)
        result.setdefault("role", role_name)
        return {"subagent_results": [result]}

    def synthesize(state: AgentState) -> dict[str, Any]:
        response = provider.invoke(
            _messages(
                """You are the lead coding agent producing the final handoff. """
                "Reconcile specialist evidence and distinguish confirmed facts from hypotheses. "
                "Return JSON with keys status, summary, implementation_steps, files_to_change, "
                "verification_commands, risks, and evidence_gaps. Never claim a command was run "
                "unless it appears in the repository context or an explicit tool result.",
                f"TASK:\n{state['task']}\n\nLEAD PLAN:\n{json.dumps(state.get('plan', {}), indent=2)}"
                f"\n\nSPECIALIST REPORTS:\n{json.dumps(state.get('subagent_results', []), indent=2)}"
                f"\n\nREPOSITORY CONTEXT:\n{repository_evidence(state)}",
            ),
            role="main",
        )
        return {"synthesis": _json_object(response)}

    builder = StateGraph(AgentState)
    builder.add_node("context", collect_context)
    builder.add_node("zvec", collect_zvec_context)
    builder.add_node("plan", plan)
    builder.add_node("specialist", specialist)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "context")
    builder.add_edge("context", "zvec")
    builder.add_edge("zvec", "plan")
    builder.add_conditional_edges("plan", route_to_specialists, ["specialist"])
    builder.add_edge("specialist", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


def run_agent(
    task: str,
    root: str,
    config: AgentConfig,
    *,
    provider: ModelProvider | None = None,
    zvec_query: str | None = None,
    zvec_tool: ZvecGrepTool | None = None,
) -> AgentState:
    if not task.strip():
        raise ValueError("task must not be empty")
    graph = create_graph(config, provider, zvec_tool)
    return graph.invoke({
        "task": task,
        "root": root,
        "zvec_query": zvec_query or "",
        "subagent_results": [],
    })
