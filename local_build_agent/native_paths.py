from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def nvidia_dll_dirs() -> list[Path]:
    if sys.platform != "win32":
        return []

    dirs: list[Path] = []
    for site_package in site.getsitepackages():
        nvidia_root = Path(site_package) / "nvidia"
        if not nvidia_root.exists():
            continue
        dirs.extend(path for path in nvidia_root.glob("*/bin") if path.exists())
    return _dedupe(dirs)


def prepare_native_dll_paths() -> None:
    dirs = nvidia_dll_dirs()
    if not dirs:
        return
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join([str(path) for path in dirs] + [existing_path])
    for path in dirs:
        try:
            os.add_dll_directory(str(path))
        except (AttributeError, OSError):
            continue


def env_with_native_dll_paths(env: dict[str, str] | None = None) -> dict[str, str]:
    next_env = dict(env or os.environ)
    dirs = nvidia_dll_dirs()
    if not dirs:
        return next_env
    existing_path = next_env.get("PATH", "")
    next_env["PATH"] = os.pathsep.join([str(path) for path in dirs] + [existing_path])
    return next_env


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped

