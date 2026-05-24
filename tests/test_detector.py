from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_build_agent.detector import detect_build_commands
from local_build_agent.mapping import load_mapping
from local_build_agent.scanner import scan_repo


class DetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = load_mapping()

    def test_detects_maven(self) -> None:
        with fixture_repo({"pom.xml": "<project></project>", "src/main/java/App.java": "class App {}"}) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        self.assertEqual(commands[0].build_system, "Maven")
        self.assertEqual(commands[0].command, "mvn package")

    def test_detects_node_package_manager_build_script(self) -> None:
        package = {"scripts": {"build": "tsc"}}
        with fixture_repo(
            {
                "package.json": json.dumps(package),
                "pnpm-lock.yaml": "",
                "src/index.ts": "export {};",
            }
        ) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        self.assertEqual(commands[0].build_system, "pnpm")
        self.assertEqual(commands[0].command, "pnpm build")

    def test_detects_typescript_without_build_script(self) -> None:
        with fixture_repo(
            {
                "package.json": json.dumps({"scripts": {}}),
                "tsconfig.json": "{}",
                "src/index.ts": "export {};",
            }
        ) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        self.assertEqual(commands[0].build_system, "TypeScript compiler")
        self.assertEqual(commands[0].command, "npx tsc -p tsconfig.json")

    def test_detects_cmake_before_source_fallback(self) -> None:
        with fixture_repo({"CMakeLists.txt": "cmake_minimum_required(VERSION 3.20)", "main.cpp": "int main(){}"}) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].build_system, "CMake")

    def test_detects_monorepo_projects(self) -> None:
        with fixture_repo(
            {
                "api/pom.xml": "<project></project>",
                "web/package.json": json.dumps({"scripts": {"build": "vite build"}}),
            }
        ) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        project_paths = {command.project_path for command in commands}
        self.assertEqual(project_paths, {"api", "web"})

    def test_skips_nested_gradle_modules(self) -> None:
        with fixture_repo(
            {
                "settings.gradle.kts": "pluginManagement {}",
                "build.gradle.kts": "plugins {}",
                "gradlew.bat": "",
                "app/build.gradle.kts": "plugins {}",
            }
        ) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        self.assertEqual([command.project_path for command in commands], ["."])
        self.assertEqual(commands[0].command, "gradlew.bat build")

    def test_skips_nested_cmake_components(self) -> None:
        with fixture_repo(
            {
                "CMakeLists.txt": "cmake_minimum_required(VERSION 3.20)",
                "src/CMakeLists.txt": "add_library(x x.cpp)",
                "src/x.cpp": "int x() { return 1; }",
            }
        ) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        self.assertEqual([command.project_path for command in commands], ["."])

    def test_skips_node_package_without_build_script(self) -> None:
        with fixture_repo({"package.json": json.dumps({"scripts": {"test": "node test.js"}})}) as root:
            commands = detect_build_commands(scan_repo(root), self.mapping)
        self.assertEqual(commands, [])


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
