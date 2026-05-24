# How It Works

A detailed walkthrough of the agent's internals — from a GitHub URL to a printed build command.

---

## High-Level Flow

```
User input: GitHub URL
      │
      ▼
┌─────────────────┐
│   git_repo.py   │  shallow-clone the repo to a temp directory
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   scanner.py    │  walk every file (up to 5,000); read manifest file contents
└────────┬────────┘
         │  RepoSnapshot { files: [...], manifests: { "pom.xml": "...", ... } }
         ▼
┌─────────────────┐
│   detector.py   │  apply deterministic rules → list[BuildCommand]
└────────┬────────┘
         │
         ▼
    ambiguous?  ──── no ──▶  print results
         │
        yes (or --llm always)
         │
         ▼
┌─────────────────┐
│ llm_resolver.py │  call local Qwen model → refined list[BuildCommand]
└────────┬────────┘
         │
         ▼
      print results
```

---

## Stage 1 — Repository Cloning (`git_repo.py`)

The agent accepts either a GitHub HTTPS URL or a local directory path.

- **GitHub URL**: runs `git clone --depth 1 <url>` into a temporary directory. Shallow clone means only the latest commit is fetched — no history, much faster.
- **Local path**: used directly, no cloning.

After the agent finishes, the temporary directory is deleted automatically (unless `--keep-clone` is passed).

---

## Stage 2 — File Scanner (`scanner.py`)

```python
RepoSnapshot = { root: Path, files: tuple[Path, ...], manifests: dict[str, str] }
```

Two things happen here:

### 2a. File collection
Walks the entire repository with `Path.rglob("*")`, skipping noise directories:

```
.git  .venv  node_modules  dist  build  target  out  __pycache__  ...
```

Stops after 5,000 files to avoid memory issues with extremely large repos.

### 2b. Manifest content reading
After building the file list, the scanner looks for **manifest files by name**:

```
pom.xml  build.gradle  build.gradle.kts  CMakeLists.txt  Makefile
pyproject.toml  setup.py  package.json  tsconfig.json  requirements.txt
Cargo.toml  go.mod  Gemfile  build.sbt  settings.gradle  settings.gradle.kts
```

Plus any file with extension `.csproj` or `.sln` (.NET projects).

For each manifest found, it reads up to **2 KB** of text. This is enough to see declared dependencies without overwhelming the LLM context window. The contents are stored in `RepoSnapshot.manifests` as a `{ relative_path → text }` dictionary.

---

## Stage 3 — Deterministic Detector (`detector.py`)

This runs first, before any LLM call. It produces high-confidence results (0.95) for well-known project structures and falls back to lower-confidence guesses (0.68–0.72) for source-only repos.

### 3a. Manifest-based detection

Files are grouped by directory. For each directory, the detector checks for manifest files in priority order:

| Manifest | Build system | Command |
|---|---|---|
| `package.json` | npm / pnpm / yarn / bun | `npm run build` etc. |
| `tsconfig.json` | TypeScript compiler | `npx tsc -p tsconfig.json` |
| `pom.xml` | Maven | `mvn package` |
| `build.gradle` / `build.gradle.kts` | Gradle | `gradlew.bat build` (Windows wrapper), `./gradlew build` (POSIX), or `gradle build` |
| `CMakeLists.txt` | CMake | `cmake -S . -B build && cmake --build build` |
| `Makefile` | Make | `make` |
| `pyproject.toml` / `setup.py` | Python package | `python -m build` |

**Node projects** get special treatment: the detector reads `package.json` to check if a `"build"` script is defined. If not, it falls back to the TypeScript compiler. The package manager is inferred from the lockfile (`pnpm-lock.yaml`, `yarn.lock`, `bun.lockb`).

**Gradle projects** check for a wrapper script (`gradlew.bat` on Windows, `./gradlew` on POSIX) before falling back to bare `gradle build`.

### 3b. Monorepo deduplication

For **aggregate build systems** (CMake, Gradle, Maven), sub-module manifests are suppressed if a parent directory already claimed the same build system. This prevents reporting 10 separate `gradlew.bat build` entries for a Gradle multi-module project when one root command builds everything.

### 3c. Source-only fallback

If a directory has source files (`.c`, `.cpp`, `.java`, `.kt`, `.py`, `.ts`, etc.) but no manifest, the detector falls back to a direct compiler command with lower confidence (0.68–0.72).

### 3d. Ambiguity check

After detection, `is_ambiguous()` returns `True` if:
- Any detected command has confidence < 0.75, **and**
- There are source files in the repo

This determines whether to call the LLM in `--llm auto` mode.

---

## Stage 4 — LangGraph Orchestration (`graph.py`)

The three stages above are wired together as a **LangGraph StateGraph**:

```
START
  │
  ▼
[scan]      →  fills state["snapshot"]
  │
  ▼
[detect]    →  fills state["commands"]
  │
  ▼
_route_after_detect()
  ├── "done"  (llm_mode=off, or auto+confident)  →  END
  └── "llm"   (llm_mode=always, or auto+ambiguous)
        │
        ▼
      [llm_resolve]  →  updates state["commands"]
        │
        ▼
       END
```

`AgentState` is a `TypedDict` that flows through every node:

```python
{
    "repo_path":    str,
    "snapshot":     RepoSnapshot,
    "mapping":      dict,           # contents of build_commands.json
    "commands":     list[BuildCommand],
    "llm_mode":     "auto" | "always" | "off",
    "llm_resolver": LlmResolver | None,
}
```

---

## Stage 5 — LLM Resolver (`llm_resolver.py` + `llama_server.py`)

Only reached when `--llm always` or when `--llm auto` and the repo is ambiguous.

### 5a. Server management (`llama_server.py`)

`LlamaServerManager` spawns `python -m llama_cpp.server` as a child process.

**GPU detection sequence:**
1. Run `nvidia-smi` — if it returns a GPU name, NVIDIA hardware is present.
2. Call `llama_cpp.llama_supports_gpu_offload()` — if `False`, the installed build is CPU-only.
3. Try each value in `gpu_layer_attempts` (default `[16, 12, 8, 0]`):
   - Start the server with that many GPU layers
   - Poll `GET /v1/models` every second for up to 180 seconds
   - If the server responds → success, record the selection
   - If the process exits early (OOM crash) → try the next layer count

**Why a temp file for stderr?**  
llama.cpp prints several megabytes of verbose model-loader output during startup. If the parent process uses `subprocess.PIPE` for stderr and never reads it, the pipe buffer fills up (typically 64 KB on Windows) and the child process **blocks** before it can start the HTTP server. Using a temp file avoids this deadlock entirely.

**Port selection:**  
`find_available_port()` probes the preferred port (default 8080). If it's already in use, it asks the OS for any free port. This means multiple agent instances can run in parallel without conflicts.

**Windows DLL paths (`native_paths.py`):**  
CUDA DLLs (`cublas64_12.dll`, `cudart64_12.dll`, etc.) live inside the venv at `.venv/Lib/site-packages/nvidia/*/bin`. `env_with_native_dll_paths()` prepends these directories to `PATH` before spawning the server subprocess so Windows can find them.

### 5b. Prompt construction

The LLM receives a single JSON user message containing:

```json
{
  "files": ["Android/LeapAudioDemo/build.gradle.kts", "...", "... trimmed ..."],
  "manifest_contents": {
    "Android/LeapAudioDemo/build.gradle.kts": "plugins {\n    id 'com.android.application'...",
    "Android/LeapAudioDemo/settings.gradle.kts": "..."
  },
  "current_detections": [
    { "project_path": "Android/LeapAudioDemo", "command": "gradlew.bat build", ... }
  ],
  "allowed_build_systems": { ... },
  "allowed_compilers": { ... }
}
```

**File list cap**: at most 100 file paths (80 head + 20 tail, sorted alphabetically so root-level files come first). The manifests section already contains the detailed file contents, so the LLM doesn't need the full file listing.

**Manifest budget cap**: manifests are included in order of path depth (shallowest first) up to a total of 12,000 characters. For a large monorepo this means the root/top-level manifests are always included; deep submodule duplicates are dropped.

### 5c. LLM call

The request goes to the llama-cpp OpenAI-compatible endpoint:

```
POST http://127.0.0.1:<port>/v1/chat/completions
{
  "model": "qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf",
  "messages": [ { "role": "system", ... }, { "role": "user", ... } ],
  "temperature": 0,
  "response_format": { "type": "json_object" }
}
```

`temperature: 0` makes output deterministic. `json_object` mode forces the model to produce valid JSON.

### 5d. Response validation

The LLM returns a JSON object like:

```json
{
  "projects": [
    { "path": ".", "language": "Java", "build_system": "Maven",
      "command": "mvn package", "confidence": 0.9, "notes": [] }
  ]
}
```

**Allowlist enforcement**: every command in the LLM's response is checked against the set of commands in `build_commands.json`. Any command not in that set is silently dropped. This prevents the LLM from inventing commands that don't exist or are unsafe.

---

## The Mapping File (`build_commands.json`)

This is the **single source of truth** for all build commands. Both the deterministic detector and the LLM allowlist validator read from it.

```json
{
  "build_systems": {
    "maven":    { "language": "Java",            "name": "Maven",   "command": "mvn package" },
    "gradle":   { "language": "Java/Kotlin",     "name": "Gradle",  "command": "gradle build",
                  "wrapper_command_windows": "gradlew.bat build",
                  "wrapper_command_posix":   "./gradlew build" },
    "npm-build": { ... },
    ...
  },
  "compilers": {
    "cpp":  { "name": "g++",   "command": "g++ *.cpp -o app" },
    "java": { "name": "javac", "command": "javac *.java" },
    ...
  }
}
```

To add a new language (e.g. Go): add an entry here and add the manifest file name (`go.mod`) to `MANIFEST_PRIORITY` in `detector.py`.

---

## Data Models (`models.py`)

### `RepoSnapshot`
Immutable record produced by the scanner and consumed by both the detector and LLM resolver.

```python
@dataclass
class RepoSnapshot:
    root:      Path
    files:     tuple[Path, ...]           # all scanned relative paths
    manifests: dict[str, str]             # relative path → file text (≤ 2 KB each)
```

### `BuildCommand`
Frozen dataclass representing one detected project.

```python
@dataclass(frozen=True)
class BuildCommand:
    project_path: str      # e.g. "Android/LeapAudioDemo" or "."
    language:     str      # e.g. "Java/Kotlin"
    build_system: str      # e.g. "Gradle"
    command:      str      # e.g. "gradlew.bat build"
    confidence:   float    # 0.68–0.95
    source:       str      # "rules" or "llm"
    notes:        tuple[str, ...]
```

---

## Why Qwen2.5-Coder?

Qwen2.5-Coder is trained specifically on code and understands build systems, `pom.xml` dependency declarations, `package.json` scripts, and `Cargo.toml` workspace configurations natively. The Q4_K_M quantisation keeps the model at ~4.4 GB while retaining most of the quality of the full-precision model.
