from __future__ import annotations

from pathlib import Path

from .models import RepoSnapshot


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    "out",
}

# Manifest file names whose text content will be captured for the LLM.
MANIFEST_NAMES = {
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "CMakeLists.txt",
    "Makefile",
    "makefile",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "tsconfig.json",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "build.sbt",
    "settings.gradle",
    "settings.gradle.kts",
}

# File extensions that are also treated as manifests (e.g. .csproj, .sln for .NET).
MANIFEST_EXTENSIONS = {".csproj", ".sln"}

# Maximum bytes to read from each manifest file.
# Keeping this small (2 KB) means the total manifest payload stays manageable even
# for large monorepos that have many build files.
MANIFEST_SIZE_LIMIT = 2048


def scan_repo(root: Path, max_files: int = 5000) -> RepoSnapshot:
    root = root.resolve()
    files: list[Path] = []

    for path in root.rglob("*"):
        if len(files) >= max_files:
            break
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if path.is_file():
            files.append(relative)

    manifests: dict[str, str] = {}
    for rel_file in files:
        if rel_file.name in MANIFEST_NAMES or rel_file.suffix in MANIFEST_EXTENSIONS:
            abs_path = root / rel_file
            try:
                raw = abs_path.read_bytes()[:MANIFEST_SIZE_LIMIT]
                manifests[rel_file.as_posix()] = raw.decode("utf-8", errors="replace")
            except OSError:
                pass

    return RepoSnapshot(root=root, files=tuple(sorted(files)), manifests=manifests)
