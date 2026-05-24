from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from local_build_agent.llama_server import LlamaServerManager


class LlamaServerPlanningTests(unittest.TestCase):
    def test_cpu_plan_when_no_nvidia_gpu(self) -> None:
        manager = LlamaServerManager(Path("model.gguf"), gpu_layer_attempts=(20, 12, 0))
        with patch("local_build_agent.llama_server._has_nvidia_gpu", return_value=False):
            self.assertEqual(manager._planned_gpu_layers(), (0,))

    def test_gpu_plan_when_nvidia_and_cuda_support_available(self) -> None:
        manager = LlamaServerManager(Path("model.gguf"), gpu_layer_attempts=(20, 12, 0))
        with patch("local_build_agent.llama_server._has_nvidia_gpu", return_value=True), patch(
            "local_build_agent.llama_server._llama_cpp_gpu_support", return_value=True
        ):
            self.assertEqual(manager._planned_gpu_layers(), (20, 12, 0))

    def test_cpu_plan_when_llama_cpp_is_cpu_only(self) -> None:
        manager = LlamaServerManager(Path("model.gguf"), gpu_layer_attempts=(20, 12, 0))
        with patch("local_build_agent.llama_server._has_nvidia_gpu", return_value=True), patch(
            "local_build_agent.llama_server._llama_cpp_gpu_support", return_value=False
        ):
            self.assertEqual(manager._planned_gpu_layers(), (0,))


if __name__ == "__main__":
    unittest.main()

