from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .git_repo import cloned_repo
from .graph import build_graph
from .llama_server import LlamaServerManager, find_available_port
from .llm_resolver import LlmResolver
from .mapping import DEFAULT_MAPPING_PATH, load_mapping
from .models import BuildCommand


DEFAULT_MODEL_PATTERN = "qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    server: LlamaServerManager | None = None
    try:
        mapping = load_mapping(args.mapping)
        model_path = _resolve_model_path(args.model)
        port = find_available_port(args.host, args.port)
        server = LlamaServerManager(
            model_path=model_path,
            host=args.host,
            port=port,
            n_ctx=args.n_ctx,
            gpu_layer_attempts=tuple(args.gpu_layers),
            startup_timeout_seconds=args.server_timeout,
        )
        resolver = None if args.llm == "off" else LlmResolver(server, mapping)

        with cloned_repo(args.repo, keep_clone=args.keep_clone) as repo_path:
            app = build_graph()
            result = app.invoke(
                {
                    "repo_path": str(repo_path),
                    "mapping": mapping,
                    "llm_mode": args.llm,
                    "llm_resolver": resolver,
                }
            )
            if server.selection:
                print(server.selection.message)
            _print_commands(result.get("commands", []))
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if server:
            server.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print build commands for a GitHub repository.")
    parser.add_argument("repo", help="GitHub HTTPS URL or local repository path")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="Path to build command JSON")
    parser.add_argument("--model", type=Path, default=None, help="Path to the first GGUF shard")
    parser.add_argument("--host", default="127.0.0.1", help="llama-cpp server host")
    parser.add_argument("--port", type=int, default=8080, help="Preferred llama-cpp server port")
    parser.add_argument("--n-ctx", type=int, default=8192, help="llama.cpp context size")
    parser.add_argument("--server-timeout", type=int, default=180, help="Seconds to wait for llama-cpp startup")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        nargs="+",
        default=[16, 12, 8, 0],
        help=(
            "GPU offload layer attempts, tried in order when NVIDIA is detected. "
            "Default [16, 12, 8, 0] is tuned for a 4 GB VRAM GPU with n_ctx=8192. "
            "Increase the first value if you have more VRAM."
        ),
    )
    parser.add_argument(
        "--llm",
        choices=("auto", "always", "off"),
        default="auto",
        help="Use the local LLM for ambiguous repos, always, or never",
    )
    parser.add_argument("--keep-clone", action="store_true", help="Do not delete the temporary clone")
    return parser


def _resolve_model_path(model_arg: Path | None) -> Path:
    if model_arg:
        return model_arg.resolve()
    cwd_candidate = Path.cwd() / DEFAULT_MODEL_PATTERN
    if cwd_candidate.exists():
        return cwd_candidate
    package_candidate = Path(__file__).resolve().parent.parent / DEFAULT_MODEL_PATTERN
    return package_candidate.resolve()


def _print_commands(commands: list[BuildCommand]) -> None:
    if not commands:
        print("No build command detected.")
        return

    for index, command in enumerate(commands):
        if index:
            print()
        print(f"Project: {command.project_path}")
        print(f"Language: {command.language}")
        print(f"Build system: {command.build_system}")
        print(f"Command: {command.command}")
        print(f"Confidence: {command.confidence:.2f}")
        print(f"Source: {command.source}")
        for note in command.notes:
            print(f"Note: {note}")
