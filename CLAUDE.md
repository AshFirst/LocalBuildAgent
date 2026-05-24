# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**Local Build Command Agent** — a LangGraph CLI that accepts a GitHub URL (or local path), inspects the project structure, and prints the commands that *would* build the detected projects. It never runs compilers or package managers.

## Setup & Running

All commands assume the `.venv` is active (`\.venv\Scripts\Activate.ps1` on Windows).

**Install base deps (deterministic scanner only):**
```powershell
python -m pip install -r requirements.txt
```

**Install local LLM support — CUDA (Python 3.12 recommended):**
```powershell
python -m pip install -r requirements-llama-cuda-cu124.txt
```

**Install local LLM support — CPU fallback:**
```powershell
python -m pip install -r requirements-llama-cpu.txt
```

**Run the agent:**
```powershell
python agent.py https://github.com/org/repo          # deterministic first, LLM for ambiguous
python agent.py https://github.com/org/repo --llm always  # force LLM validation
python agent.py https://github.com/org/repo --llm off     # skip LLM entirely
python agent.py /path/to/local/repo                  # local path also accepted
```

**Key CLI flags:** `--model <path>`, `--port`, `--n-ctx`, `--gpu-layers`, `--server-timeout`, `--keep-clone`

## Running Tests

```powershell
python -m pytest tests/
```

Run a single test file:
```powershell
python -m pytest tests/test_detector.py
```

Run a single test:
```powershell
python -m pytest tests/test_detector.py::DetectorTests::test_detects_maven
```

Tests use `unittest.TestCase` but are discovered by pytest. The `fixture_repo` helper (defined in each test file) creates a temporary directory tree from a `{relative_path: content}` dict.

## Architecture

### LangGraph Pipeline (`local_build_agent/graph.py`)

Three-node DAG compiled from `StateGraph(AgentState)`:

```
START → scan → detect → [conditional] → llm_resolve → END
                      ↘ (done)        → END
```

- **`scan`** — calls `scanner.scan_repo()` → `RepoSnapshot`
- **`detect`** — calls `detector.detect_build_commands()` → `list[BuildCommand]`
- **`_route_after_detect`** — routes to `llm_resolve` if `llm_mode == "always"`, or if `llm_mode == "auto"` and `detector.is_ambiguous()` returns True
- **`llm_resolve`** — delegates to `LlmResolver.resolve()`, falls back to prior detections on empty result

### Deterministic Detector (`local_build_agent/detector.py`)

1. Groups scanned files by directory.
2. For each directory (shallowest first), looks for manifest files in priority order: `pom.xml` (Maven), `build.gradle`/`.kts` (Gradle), `CMakeLists.txt`, `Makefile`, `pyproject.toml`/`setup.py`, then `package.json` and `tsconfig.json`.
3. For **aggregate build systems** (CMake, Gradle, Maven), subdirectory manifests are suppressed if a parent already claimed that system.
4. For Node projects, detects the package manager from lockfile (`pnpm-lock.yaml`, `yarn.lock`, `bun.lockb`/`bun.lock`), then checks whether a `build` script exists in `package.json`.
5. Falls back to **source-only detection** (by file extension counts) for directories not covered by any manifest project — confidence is lower (0.68–0.72 vs. 0.95 for manifest-based).
6. `is_ambiguous()` returns True when any command has confidence < 0.75 AND there are source files present.

### LLM Path (`local_build_agent/llm_resolver.py` + `llama_server.py`)

- `LlamaServerManager` spawns `python -m llama_cpp.server` as a subprocess and polls `/v1/models` to detect readiness.
- GPU detection: runs `nvidia-smi`, then calls `llama_cpp.llama_supports_gpu_offload()`. Tries `gpu_layer_attempts` (default `[20, 12, 0]`) in sequence; falls back to CPU (`n_gpu_layers=0`) if CUDA is unavailable.
- On Windows, NVIDIA DLL directories are prepended to `PATH` and registered via `os.add_dll_directory()` (`native_paths.py`).
- `chat_completion()` in `llama_server.py` calls the OpenAI-compatible `/v1/chat/completions` endpoint with `temperature=0` and `response_format: json_object`.
- `LlmResolver` validates LLM output against the allowlist in `build_commands.json` — commands not in the allowlist are silently dropped.

### Data Flow Types (`local_build_agent/models.py`)

- `RepoSnapshot(root: Path, files: tuple[Path, ...])` — immutable snapshot of scanned relative paths.
- `BuildCommand` — frozen dataclass: `project_path`, `language`, `build_system`, `command`, `confidence`, `source` (`"rules"` or `"llm"`), `notes`.

### Build Command Allowlist (`build_commands.json`)

Defines two top-level keys: `"build_systems"` and `"compilers"`. Every command the LLM may emit must appear here. To add support for a new build system, add an entry to this file and update the detection logic in `detector.py`.

### Model Files

The two `qwen2.5-coder-7b-instruct-q4_k_m-000*-of-00002.gguf` shards must reside in the same directory. The CLI resolves the model path by looking in `cwd` first, then the package root. Override with `--model`.
