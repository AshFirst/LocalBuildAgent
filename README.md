# Local Build Command Agent

A local AI agent that takes a GitHub repository URL, inspects the project structure, reads manifest files, and **prints** the commands needed to build the project. It never runs compilers or package managers — output only.

Built with **LangGraph** and a local **Qwen2.5-Coder-7B** model served via **llama-cpp-python**. Runs entirely offline after the one-time model download.

---

## Features

- Detects build systems from manifest files (Maven, Gradle, CMake, Make, npm/pnpm/yarn/bun, Python packages, TypeScript)
- Reads actual manifest file contents (pom.xml, package.json, Cargo.toml, etc.) and sends them to the LLM so it understands declared dependencies
- Handles monorepos — reports one build command per detected sub-project
- Auto-detects NVIDIA GPU and offloads transformer layers to VRAM; falls back to CPU if needed
- Works without the LLM for clear-cut repos (`--llm off`) — instant results

---

## Requirements

| Requirement | Notes |
|---|---|
| Python **3.12** | Prebuilt CUDA wheels are available for 3.12. Python 3.13 requires building llama-cpp-python from source. |
| Git | Must be on PATH (used for shallow cloning) |
| NVIDIA GPU (optional) | CUDA 12.x driver (≥ 525). CPU fallback works without a GPU. |

---

## Installation

### 1. Clone this repo

```powershell
git clone https://github.com/your-username/local-build-agent.git
cd local-build-agent
```

### 2. Download the model

Download both GGUF shards of **Qwen2.5-Coder-7B-Instruct Q4_K_M** from Hugging Face and place them in the project root:

- [`qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)
- `qwen2.5-coder-7b-instruct-q4_k_m-00002-of-00002.gguf`

Both shards must be in the same directory (the project root by default).

### 3. Create a Python 3.12 virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4. Install dependencies

**Base (LangGraph only — for deterministic mode):**
```powershell
pip install -r requirements.txt
```

**With GPU support (CUDA 12.4 prebuilt wheel — recommended):**
```powershell
pip install -r requirements-llama-cuda-cu124.txt
```

**CPU-only llama-cpp:**
```powershell
pip install -r requirements-llama-cpu.txt
```

---

## Usage

```powershell
# Activate the venv first
.\.venv\Scripts\Activate.ps1

# Deterministic rules only — instant, no model loaded
python agent.py https://github.com/org/repo --llm off

# Default — rules first, LLM only for ambiguous repos
python agent.py https://github.com/org/repo

# Always run LLM validation (GPU kicks in)
python agent.py https://github.com/org/repo --llm always

# Local repo path also works
python agent.py C:\path\to\local\repo --llm off
```

### Example output

```
Project: Android/LeapAudioDemo
Language: Java/Kotlin
Build system: Gradle
Command: gradlew.bat build
Confidence: 0.95
Source: llm

Project: iOS/LeapChatExample
Language: C/C++
Build system: Make
Command: make
Confidence: 0.95
Source: llm
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--llm` | `auto` | `auto` \| `always` \| `off` |
| `--gpu-layers` | `16 12 8 0` | GPU layer counts to try, in order |
| `--n-ctx` | `8192` | LLM context window size |
| `--model` | *(auto-detected)* | Path to the first GGUF shard |
| `--server-timeout` | `180` | Seconds to wait for llama-cpp server startup |
| `--keep-clone` | off | Keep the temporary git clone after running |

---

## Supported Languages & Build Systems

| Language | Detected via | Build command |
|---|---|---|
| Java | `pom.xml` | `mvn package` |
| Java / Kotlin | `build.gradle`, `build.gradle.kts` | `gradlew.bat build` / `./gradlew build` / `gradle build` |
| C / C++ | `CMakeLists.txt` | `cmake -S . -B build && cmake --build build` |
| C / C++ | `Makefile` | `make` |
| Python | `pyproject.toml`, `setup.py` | `python -m build` |
| JavaScript | `package.json` + lock file | `npm run build` / `pnpm build` / `yarn build` / `bun run build` |
| TypeScript | `tsconfig.json` | `npx tsc -p tsconfig.json` |
| C (source-only) | `.c` files | `gcc *.c -o app` |
| C++ (source-only) | `.cpp` files | `g++ *.cpp -o app` |
| Java (source-only) | `.java` files | `javac *.java` |
| Kotlin (source-only) | `.kt` files | `kotlinc *.kt -include-runtime -d app.jar` |

The `build_commands.json` file is the single source of truth for all mappings — edit it to add new languages or change commands.

---

## GPU Behaviour

The server manager tries GPU layer counts in order (`--gpu-layers`, default `16 12 8 0`):

1. Checks for `nvidia-smi` to confirm an NVIDIA GPU is present
2. Checks `llama_cpp.llama_supports_gpu_offload()` to confirm a CUDA build is installed
3. Tries each layer count — whichever starts the server successfully is used
4. Falls back to CPU (`n_gpu_layers=0`) as the last resort

For a **4 GB VRAM GPU** with `n_ctx=8192`, 16 layers offloads approximately 2.9 GB to the GPU. Increase the first value in `--gpu-layers` if you have more VRAM.

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

---

## Project Structure

```
├── agent.py                          Entry point
├── build_commands.json               Language → build command mapping
├── requirements.txt                  LangGraph
├── requirements-llama-cuda-cu124.txt llama-cpp-python CUDA 12.4 wheels
├── requirements-llama-cpu.txt        llama-cpp-python CPU-only
├── local_build_agent/
│   ├── cli.py            Argument parsing and orchestration
│   ├── graph.py          LangGraph DAG
│   ├── scanner.py        File walker + manifest reader
│   ├── detector.py       Deterministic rule engine
│   ├── llm_resolver.py   LLM prompt builder and response parser
│   ├── llama_server.py   llama-cpp subprocess manager
│   ├── models.py         BuildCommand and RepoSnapshot dataclasses
│   ├── mapping.py        Loads build_commands.json
│   ├── git_repo.py       Shallow-clones GitHub URLs
│   └── native_paths.py   NVIDIA DLL path fix for Windows
└── tests/
    ├── test_detector.py
    ├── test_graph.py
    └── test_llama_server.py
```
