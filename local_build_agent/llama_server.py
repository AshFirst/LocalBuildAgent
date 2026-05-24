from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .native_paths import env_with_native_dll_paths, prepare_native_dll_paths


@dataclass(frozen=True)
class ServerSelection:
    backend: str
    n_gpu_layers: int
    message: str


class LlamaServerManager:
    def __init__(
        self,
        model_path: Path,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_ctx: int = 8192,
        gpu_layer_attempts: tuple[int, ...] = (16, 12, 8, 0),
        startup_timeout_seconds: int = 180,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        self.n_ctx = n_ctx
        self.gpu_layer_attempts = gpu_layer_attempts
        self.startup_timeout_seconds = startup_timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self.selection: ServerSelection | None = None
        self._stderr_log: Path | None = None  # temp file for server stderr

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def ensure_running(self) -> ServerSelection:
        if self._healthcheck():
            self.selection = ServerSelection("existing", -1, f"Using existing server at {self.base_url}")
            return self.selection

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._assert_llama_cpp_installed()

        attempts = self._planned_gpu_layers()
        errors: list[str] = []
        for n_gpu_layers in attempts:
            self._start(n_gpu_layers)
            if self._wait_until_ready(timeout_seconds=self.startup_timeout_seconds):
                backend = "cpu" if n_gpu_layers == 0 else "cuda-partial-offload"
                self.selection = ServerSelection(
                    backend=backend,
                    n_gpu_layers=n_gpu_layers,
                    message=f"LLM backend: {backend}, n_gpu_layers={n_gpu_layers}",
                )
                return self.selection
            errors.append(self._terminate_failed_process())

        detail = "\n".join(error for error in errors if error)
        raise RuntimeError(f"Could not start llama-cpp server.\n{detail}".strip())

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self._cleanup_stderr_log()

    def _planned_gpu_layers(self) -> tuple[int, ...]:
        if not _has_nvidia_gpu():
            return (0,)
        gpu_supported = _llama_cpp_gpu_support()
        if gpu_supported is False:
            return (0,)
        return self.gpu_layer_attempts

    def _start(self, n_gpu_layers: int) -> None:
        # Use a temp file for stderr instead of subprocess.PIPE.
        # The llama.cpp server emits megabytes of verbose model-loader output.
        # If stderr is a PIPE and the parent never reads it, the pipe buffer fills up
        # and the server process BLOCKS before it can start the HTTP listener.
        self._cleanup_stderr_log()
        stderr_fd, stderr_path = tempfile.mkstemp(suffix="-llama-server.log")
        self._stderr_log = Path(stderr_path)
        os.close(stderr_fd)  # Popen will re-open it

        command = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n_ctx",
            str(self.n_ctx),
            "--n_gpu_layers",
            str(n_gpu_layers),
        ]
        self.process = subprocess.Popen(
            command,
            env=env_with_native_dll_paths(os.environ),
            stdout=subprocess.DEVNULL,  # HTTP access logs not needed
            stderr=self._stderr_log.open("wb"),
        )

    def _wait_until_ready(self, timeout_seconds: int) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                return False
            if self._healthcheck():
                return True
            time.sleep(1)
        return False

    def _healthcheck(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def _terminate_failed_process(self) -> str:
        if not self.process:
            return ""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

        stderr = ""
        if self._stderr_log and self._stderr_log.exists():
            try:
                raw = self._stderr_log.read_bytes()
                # Show last 2 KB — skip the verbose model-loader preamble
                stderr = raw[-2048:].decode("utf-8", errors="replace")
            except OSError:
                pass
        self._cleanup_stderr_log()
        return stderr

    def _cleanup_stderr_log(self) -> None:
        if self._stderr_log:
            try:
                self._stderr_log.unlink(missing_ok=True)
            except OSError:
                pass
            self._stderr_log = None

    @staticmethod
    def _assert_llama_cpp_installed() -> None:
        try:
            prepare_native_dll_paths()
            import llama_cpp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc


def chat_completion(base_url: str, model: str, messages: list[dict[str, str]], timeout: int = 300) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _llama_cpp_gpu_support() -> bool | None:
    try:
        prepare_native_dll_paths()
        import llama_cpp
    except ImportError:
        return False

    support_fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if callable(support_fn):
        try:
            return bool(support_fn())
        except Exception:
            return None
    return None


def find_available_port(host: str, preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex((host, preferred_port)) != 0:
            return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
