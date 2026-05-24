from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .mapping import gradle_command
from .models import BuildCommand, RepoSnapshot


MANIFEST_PRIORITY = {
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
    "CMakeLists.txt": "cmake",
    "Makefile": "make",
    "makefile": "make",
    "pyproject.toml": "python-package",
    "setup.py": "python-package",
}

AGGREGATE_BUILD_SYSTEMS = {"cmake", "gradle", "maven"}

SOURCE_LANGS = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def detect_build_commands(snapshot: RepoSnapshot, mapping: dict[str, Any]) -> list[BuildCommand]:
    by_dir: dict[Path, set[str]] = defaultdict(set)
    for rel_file in snapshot.files:
        by_dir[rel_file.parent].add(rel_file.name)

    projects: list[BuildCommand] = []
    manifest_dirs: set[Path] = set()
    manifest_roots: dict[Path, str] = {}

    for rel_dir, names in sorted(by_dir.items(), key=lambda item: len(item[0].parts)):
        manifest_key = _manifest_key(names)
        if manifest_key and _covered_by_ancestor(rel_dir, manifest_key, manifest_roots):
            continue
        project_dir = snapshot.root / rel_dir
        command = _detect_manifest_project(rel_dir, project_dir, names, mapping)
        if command:
            projects.append(command)
            manifest_dirs.add(rel_dir)
            if manifest_key:
                manifest_roots[rel_dir] = manifest_key

    projects.extend(_detect_source_only_projects(snapshot, mapping, manifest_dirs))
    return _dedupe(projects)


def is_ambiguous(commands: list[BuildCommand], snapshot: RepoSnapshot) -> bool:
    if not commands:
        return True
    low_confidence = any(command.confidence < 0.75 for command in commands)
    source_count = sum(1 for path in snapshot.files if path.suffix.lower() in SOURCE_LANGS)
    return low_confidence and source_count > 0


def _detect_manifest_project(
    rel_dir: Path,
    project_dir: Path,
    names: set[str],
    mapping: dict[str, Any],
) -> BuildCommand | None:
    if "package.json" in names:
        return _detect_node_project(rel_dir, project_dir, names, mapping)

    if "tsconfig.json" in names:
        system = mapping["build_systems"]["typescript"]
        return _command(rel_dir, system["language"], system["name"], system["command"], 0.82, "rules")

    for manifest, build_system_key in MANIFEST_PRIORITY.items():
        if manifest not in names:
            continue
        system = mapping["build_systems"][build_system_key]
        command = gradle_command(project_dir, mapping) if build_system_key == "gradle" else system["command"]
        return _command(rel_dir, system["language"], system["name"], command, 0.95, "rules")

    return None


def _detect_node_project(
    rel_dir: Path,
    project_dir: Path,
    names: set[str],
    mapping: dict[str, Any],
) -> BuildCommand | None:
    package_path = project_dir / "package.json"
    scripts: dict[str, str] = {}
    try:
        package_data = json.loads(package_path.read_text(encoding="utf-8"))
        scripts = package_data.get("scripts", {}) if isinstance(package_data, dict) else {}
    except (OSError, json.JSONDecodeError):
        scripts = {}

    package_manager = _node_package_manager(names)
    build_key = {
        "pnpm": "pnpm-build",
        "yarn": "yarn-build",
        "bun": "bun-build",
        "npm": "npm-build",
    }[package_manager]
    system = mapping["build_systems"][build_key]

    if "build" in scripts:
        return _command(rel_dir, system["language"], system["name"], system["command"], 0.95, "rules")

    if "tsconfig.json" in names:
        tsc = mapping["build_systems"]["typescript"]
        return _command(
            rel_dir,
            tsc["language"],
            tsc["name"],
            tsc["command"],
            0.78,
            "rules",
            ("No package build script found; using the TypeScript compiler.",),
        )

    return None


def _manifest_key(names: set[str]) -> str | None:
    if "package.json" in names:
        return "node"
    if "tsconfig.json" in names:
        return "typescript"
    for manifest, build_system_key in MANIFEST_PRIORITY.items():
        if manifest in names:
            return build_system_key
    return None


def _covered_by_ancestor(rel_dir: Path, build_system_key: str, manifest_roots: dict[Path, str]) -> bool:
    if build_system_key not in AGGREGATE_BUILD_SYSTEMS:
        return False
    for ancestor_dir, ancestor_key in manifest_roots.items():
        if ancestor_key != build_system_key:
            continue
        if ancestor_dir == rel_dir:
            continue
        if ancestor_dir == Path("."):
            return True
        try:
            rel_dir.relative_to(ancestor_dir)
            return True
        except ValueError:
            continue
    return False


def _node_package_manager(names: set[str]) -> str:
    if "pnpm-lock.yaml" in names:
        return "pnpm"
    if "yarn.lock" in names:
        return "yarn"
    if "bun.lockb" in names or "bun.lock" in names:
        return "bun"
    return "npm"


def _detect_source_only_projects(
    snapshot: RepoSnapshot,
    mapping: dict[str, Any],
    manifest_dirs: set[Path],
) -> list[BuildCommand]:
    counts: Counter[str] = Counter()
    for rel_file in snapshot.files:
        if _is_under_manifest_project(rel_file, manifest_dirs):
            continue
        language = SOURCE_LANGS.get(rel_file.suffix.lower())
        if language:
            counts[language] += 1

    if not counts:
        return []

    projects: list[BuildCommand] = []
    compilers = mapping["compilers"]

    if counts["cpp"]:
        compiler = compilers["cpp"]
        projects.append(_command(Path("."), "C++", compiler["name"], compiler["command"], 0.72, "rules"))
    elif counts["c"]:
        compiler = compilers["c"]
        projects.append(_command(Path("."), "C", compiler["name"], compiler["command"], 0.72, "rules"))

    for language_key, display in (
        ("java", "Java"),
        ("kotlin", "Kotlin"),
        ("typescript", "TypeScript"),
        ("python", "Python"),
        ("javascript", "JavaScript"),
    ):
        if counts[language_key]:
            compiler = compilers[language_key]
            confidence = 0.68 if language_key in {"javascript", "python"} else 0.72
            projects.append(_command(Path("."), display, compiler["name"], compiler["command"], confidence, "rules"))

    return projects


def _is_under_manifest_project(rel_file: Path, manifest_dirs: set[Path]) -> bool:
    for manifest_dir in manifest_dirs:
        if manifest_dir == Path("."):
            return True
        try:
            rel_file.relative_to(manifest_dir)
            return True
        except ValueError:
            continue
    return False


def _command(
    rel_dir: Path,
    language: str,
    build_system: str,
    command: str,
    confidence: float,
    source: str,
    notes: tuple[str, ...] = (),
) -> BuildCommand:
    project_path = "." if str(rel_dir) in {"", "."} else rel_dir.as_posix()
    return BuildCommand(
        project_path=project_path,
        language=language,
        build_system=build_system,
        command=command,
        confidence=confidence,
        source=source,
        notes=notes,
    )


def _dedupe(commands: list[BuildCommand]) -> list[BuildCommand]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[BuildCommand] = []
    for command in commands:
        key = (command.project_path, command.build_system, command.command)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(command)
    return deduped
