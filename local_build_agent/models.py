from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BuildCommand:
    project_path: str
    language: str
    build_system: str
    command: str
    confidence: float
    source: str
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RepoSnapshot:
    root: Path
    files: tuple[Path, ...]
    # Key manifest files read during scanning: relative POSIX path → file text (truncated).
    # Populated by scanner.scan_repo(); consumed by llm_resolver to enrich LLM context.
    manifests: dict[str, str] = field(default_factory=dict)

    def relative_files(self) -> tuple[str, ...]:
        return tuple(path.as_posix() for path in self.files)
