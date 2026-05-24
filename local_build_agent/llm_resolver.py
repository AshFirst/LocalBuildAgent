from __future__ import annotations

import json
from typing import Any

from .llama_server import LlamaServerManager, chat_completion
from .models import BuildCommand, RepoSnapshot


class LlmResolver:
    def __init__(self, server: LlamaServerManager, mapping: dict[str, Any]) -> None:
        self.server = server
        self.mapping = mapping

    def resolve(self, snapshot: RepoSnapshot, current: list[BuildCommand]) -> list[BuildCommand]:
        self.server.ensure_running()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._payload(snapshot, current)},
        ]
        content = self._chat(messages)
        return _parse_llm_commands(content, self.mapping)

    def _payload(self, snapshot: RepoSnapshot, current: list[BuildCommand]) -> str:
        return json.dumps(
            {
                "files": _trim_files(snapshot.relative_files()),
                # Actual text of manifest files (pom.xml, package.json, Cargo.toml, etc.)
                # so the LLM can read declared dependencies, not just file names.
                # Manifests are budget-capped to keep the prompt within the model's context window.
                "manifest_contents": _trim_manifests(snapshot.manifests),
                "current_detections": [command.__dict__ for command in current],
                "allowed_build_systems": self.mapping["build_systems"],
                "allowed_compilers": self.mapping["compilers"],
            }
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        # Keep the LLM client dependency-light: llama-cpp-python exposes an
        # OpenAI-compatible server, and LangGraph handles the agent workflow.
        return chat_completion(self.server.base_url, self.server.model_path.name, messages)


_SYSTEM_PROMPT = """You identify build commands for a repository.
You are given the list of files and the actual text of key manifest files
(pom.xml, package.json, Cargo.toml, requirements.txt, etc.) so you can read
declared dependencies and understand the project in detail.
Return only JSON with this shape:
{"projects":[{"path":".","language":"Java","build_system":"Maven","command":"mvn package","confidence":0.8,"notes":["short note"]}]}
Use only commands from allowed_build_systems or allowed_compilers. Do not invent commands. The user will print commands only, never run them."""


def _trim_manifests(manifests: dict[str, str], budget_chars: int = 12_000) -> dict[str, str]:
    """Return a subset of manifests that fits within *budget_chars* total characters.

    Manifests are included in order of path depth (shallowest / root-level first) so the
    most informative files are kept when a large monorepo exceeds the budget.
    """
    # Sort by depth (number of path separators), then alphabetically for stability.
    sorted_items = sorted(manifests.items(), key=lambda kv: (kv[0].count("/"), kv[0]))
    result: dict[str, str] = {}
    used = 0
    for path, content in sorted_items:
        if used + len(content) > budget_chars:
            # Include a truncated version if we still have at least 200 chars of budget.
            remaining = budget_chars - used
            if remaining >= 200:
                result[path] = content[:remaining] + "\n... (truncated)"
            break
        result[path] = content
        used += len(content)
    return result


def _trim_files(files: tuple[str, ...], limit: int = 100) -> list[str]:
    """Return at most *limit* representative file paths.

    The file list is already supplemented by full manifest contents, so the LLM
    does not need to see every individual source file.  Keeping it small ensures
    the total prompt fits within the model's context window.
    Files are sorted alphabetically so the head contains shallow/root-level paths
    (where manifests live) which are the most useful for build detection.
    """
    if len(files) <= limit:
        return list(files)
    head = list(files[:80])
    tail = list(files[-20:])
    return head + ["... trimmed ..."] + tail


def _parse_llm_commands(content: str, mapping: dict[str, Any]) -> list[BuildCommand]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    allowed_commands = {
        item["command"]
        for group_name in ("build_systems", "compilers")
        for item in mapping[group_name].values()
        if "command" in item
    }
    allowed_commands.update(
        item[key]
        for item in mapping["build_systems"].values()
        for key in ("wrapper_command_windows", "wrapper_command_posix")
        if key in item
    )

    projects = data.get("projects", [])
    if not isinstance(projects, list):
        return []

    commands: list[BuildCommand] = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        command = str(project.get("command", ""))
        if command not in allowed_commands:
            continue
        notes_value = project.get("notes", [])
        notes = tuple(str(note) for note in notes_value) if isinstance(notes_value, list) else ()
        commands.append(
            BuildCommand(
                project_path=str(project.get("path", ".")),
                language=str(project.get("language", "Unknown")),
                build_system=str(project.get("build_system", "Unknown")),
                command=command,
                confidence=float(project.get("confidence", 0.5)),
                source="llm",
                notes=notes,
            )
        )
    return commands
