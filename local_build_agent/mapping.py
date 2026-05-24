from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent / "build_commands.json"


def load_mapping(path: Path | None = None) -> dict[str, Any]:
    mapping_path = path or DEFAULT_MAPPING_PATH
    with mapping_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gradle_command(project_dir: Path, mapping: dict[str, Any]) -> str:
    gradle = mapping["build_systems"]["gradle"]
    if os.name == "nt" and (project_dir / "gradlew.bat").exists():
        return gradle["wrapper_command_windows"]
    if os.name != "nt" and (project_dir / "gradlew").exists():
        return gradle["wrapper_command_posix"]
    return gradle["command"]

