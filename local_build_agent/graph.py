from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .detector import detect_build_commands, is_ambiguous
from .llm_resolver import LlmResolver
from .models import BuildCommand, RepoSnapshot
from .scanner import scan_repo


class AgentState(TypedDict, total=False):
    repo_path: str
    snapshot: RepoSnapshot
    mapping: dict[str, Any]
    commands: list[BuildCommand]
    llm_mode: str
    llm_resolver: LlmResolver | None


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("scan", _scan)
    graph.add_node("detect", _detect)
    graph.add_node("llm_resolve", _llm_resolve)

    graph.add_edge(START, "scan")
    graph.add_edge("scan", "detect")
    graph.add_conditional_edges(
        "detect",
        _route_after_detect,
        {
            "llm": "llm_resolve",
            "done": END,
        },
    )
    graph.add_edge("llm_resolve", END)
    return graph.compile()


def _scan(state: AgentState) -> AgentState:
    return {"snapshot": scan_repo(Path(state["repo_path"]))}


def _detect(state: AgentState) -> AgentState:
    snapshot = state["snapshot"]
    return {"commands": detect_build_commands(snapshot, state["mapping"])}


def _route_after_detect(state: AgentState) -> Literal["llm", "done"]:
    mode = state.get("llm_mode", "auto")
    if mode == "off":
        return "done"
    if mode == "always":
        return "llm"
    snapshot = state["snapshot"]
    commands = state.get("commands", [])
    return "llm" if is_ambiguous(commands, snapshot) and state.get("llm_resolver") else "done"


def _llm_resolve(state: AgentState) -> AgentState:
    resolver = state.get("llm_resolver")
    if resolver is None:
        return {}
    llm_commands = resolver.resolve(state["snapshot"], state.get("commands", []))
    return {"commands": llm_commands or state.get("commands", [])}

