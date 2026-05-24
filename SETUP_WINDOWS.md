# Windows Setup Notes

This project can run the deterministic scanner with only `requirements.txt`.
The local Qwen server needs `llama-cpp-python[server]`.

## Recommended CUDA Setup

Use Python 3.12 for the easiest CUDA path. The official `llama-cpp-python`
docs list prebuilt CUDA wheels for Python 3.10, 3.11, and 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -r requirements-llama-cuda-cu124.txt
```

The CUDA requirements include NVIDIA's runtime and cuBLAS Python wheels. The app
automatically prepends those DLL directories when launching the llama-cpp server.

Then run:

```powershell
.\.venv\Scripts\python agent.py https://github.com/org/repo --llm auto
```

## Current Machine Finding

This machine currently has:

- Python 3.13
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM
- CMake
- No `cl` compiler on PATH
- No `nvcc` CUDA compiler on PATH

Installing `llama-cpp-python` on Python 3.13 attempted a source install and
failed while unpacking/building. For GPU support on Python 3.13, install Visual
Studio Build Tools and the CUDA Toolkit, then build from source:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
python -m pip install --no-cache-dir "llama-cpp-python[server]"
```

## CPU Fallback

For CPU-only inference:

```powershell
python -m pip install -r requirements-llama-cpu.txt
python agent.py https://github.com/org/repo --llm auto
```

The app still auto-detects the GPU at runtime. If the installed llama backend is
CPU-only, it reports that and falls back to CPU.
