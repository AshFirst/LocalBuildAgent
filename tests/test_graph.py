from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_build_agent.graph import build_graph
from local_build_agent.mapping import load_mapping


class GraphTests(unittest.TestCase):
    def test_graph_detects_commands_without_llm(self) -> None:
        with fixture_repo({"package.json": json.dumps({"scripts": {"build": "vite build"}})}) as root:
            app = build_graph()
            result = app.invoke(
                {
                    "repo_path": str(root),
                    "mapping": load_mapping(),
                    "llm_mode": "off",
                    "llm_resolver": None,
                }
            )

        self.assertEqual(result["commands"][0].command, "npm run build")


class fixture_repo:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.temp_dir = TemporaryDirectory()

    def __enter__(self) -> Path:
        root = Path(self.temp_dir.name)
        for name, content in self.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return root

    def __exit__(self, exc_type, exc, tb) -> None:
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

