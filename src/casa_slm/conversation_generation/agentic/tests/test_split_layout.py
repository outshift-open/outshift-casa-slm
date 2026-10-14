# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Test ASTRA split layouts and generation launchers."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_launcher_data_root(data_root: Path) -> None:
    data_specs = {
        "single_mcp_server": ("01_one_tool", "02_two_tools", "03_three_tools"),
        "cross_mcp_server": ("02_two_tools", "03_three_tools"),
    }
    for server_type, tool_dirs in data_specs.items():
        for tool_dir in tool_dirs:
            grouped_dir = data_root / "task_tool" / server_type / tool_dir / "ASTRA" / "03_grouped_filtered"
            grouped_dir.mkdir(parents=True, exist_ok=True)
            for split in ("test", "validation"):
                (grouped_dir / f"{split}_curated_grouped.json").write_text("[]\n", encoding="utf-8")
    (data_root / "mcp_servers" / "ASTRA").mkdir(parents=True, exist_ok=True)


def make_fake_screen_bin(bin_dir: Path) -> None:
    screen_path = bin_dir / "screen"
    screen_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    screen_path.chmod(screen_path.stat().st_mode | stat.S_IXUSR)


class SplitLayoutTests(unittest.TestCase):
    def test_single_split_aware_runner_is_used(self) -> None:
        self.assertTrue((ROOT / "run_experiments.sh").exists())
        self.assertTrue((ROOT / "run_cross_experiments.sh").exists())
        self.assertTrue((ROOT / "launch_generation_screens.sh").exists())
        self.assertFalse((ROOT / "run_experiments_test.sh").exists())
        self.assertFalse((ROOT / "run_experiments_validation.sh").exists())

    def test_runner_uses_split_input_and_output_paths(self) -> None:
        script = (ROOT / "run_experiments.sh").read_text(encoding="utf-8")

        self.assertIn('SPLIT="test"', script)
        self.assertIn('OUT_ROOT="./generated_agentic_conversations"', script)
        self.assertIn("--split)", script)
        self.assertIn('DATA_ROOT="${REPOSITORY_ROOT}/data/conversation"', script)
        self.assertIn('TASK_TOOL_ROOT="${DATA_ROOT}/task_tool/single_mcp_server"', script)
        self.assertIn('MCP_SERVERS_DIR="${DATA_ROOT}/mcp_servers"', script)
        self.assertIn(
            'data_file="${TASK_TOOL_ROOT}/${dir}/${bench}/03_grouped_filtered/${SPLIT}_curated_grouped.json"', script
        )
        self.assertIn('OUT_DIR="${OUT_ROOT}/${SPLIT}/${LABEL}"', script)
        self.assertIn('--task-data "$DATA_FILE"', script)
        self.assertIn("--resume", script)
        self.assertNotIn("--astra-data", script)
        self.assertIn('if [[ "$SPLIT" != "test" && "$SPLIT" != "validation" ]]', script)
        self.assertNotIn("synthetic_traces_regenerated", script)

    def test_cross_runner_uses_cross_input_and_output_paths(self) -> None:
        script = (ROOT / "run_cross_experiments.sh").read_text(encoding="utf-8")

        self.assertIn('SPLIT="test"', script)
        self.assertIn('OUT_ROOT="./generated_agentic_conversations"', script)
        self.assertIn('TOOLS="2 3"', script)
        self.assertIn('BENCH="ASTRA"', script)
        self.assertIn("--split)", script)
        self.assertIn('TOOL_DIR_MAP=([2]="02_two_tools" [3]="03_three_tools")', script)
        self.assertIn('LABEL_DIR_MAP=([2]="02_tools_cross" [3]="03_tools_cross")', script)
        self.assertIn("Invalid tool count: $t (must be 2 or 3)", script)
        self.assertIn('DATA_ROOT="${REPOSITORY_ROOT}/data/conversation"', script)
        self.assertIn('TASK_TOOL_ROOT="${DATA_ROOT}/task_tool/cross_mcp_server"', script)
        self.assertIn('MCP_SERVERS_DIR="${DATA_ROOT}/mcp_servers"', script)
        self.assertIn(
            'data_file="${TASK_TOOL_ROOT}/${dir}/${BENCH}/03_grouped_filtered/${SPLIT}_curated_grouped.json"', script
        )
        self.assertIn('EXPERIMENTS+=("${data_file}:${label}_${BENCH}:${BENCH}")', script)
        self.assertIn('OUT_DIR="${OUT_ROOT}/${SPLIT}/${LABEL}"', script)
        self.assertIn('--task-data "$DATA_FILE"', script)
        self.assertIn("--resume", script)
        self.assertNotIn("--astra-data", script)
        self.assertIn('if [[ "$SPLIT" != "test" && "$SPLIT" != "validation" ]]', script)
        self.assertNotIn("synthetic_traces_regenerated", script)

    def test_screen_launcher_uses_curated_inputs_and_mirrored_output_paths(self) -> None:
        script = (ROOT / "launch_generation_screens.sh").read_text(encoding="utf-8")

        self.assertIn('SERVER_TYPE="${1:-}"', script)
        self.assertIn('OUT_ROOT="./generated_agentic_conversations"', script)
        self.assertIn("single_mcp_server)", script)
        self.assertIn("cross_mcp_server)", script)
        self.assertIn('SPLITS="test validation"', script)
        self.assertIn('SINGLE_TOOLS="1 2 3"', script)
        self.assertIn('CROSS_TOOLS="2 3"', script)
        self.assertIn("${SPLIT}_curated_grouped.json", script)
        self.assertNotIn("${SPLIT}_grouped.json", script)
        self.assertIn(
            'OUT_DIR="${OUT_ROOT}/task_tool/${SERVER_TYPE}/${DIR}/ASTRA/03_grouped_filtered/${SPLIT}_curated_grouped"',
            script,
        )
        self.assertIn(
            'screen -dmS "$SCREEN_NAME" bash -lc \'exec "$1"\' screen-runner "$RUNNER_PATH"',
            script,
        )
        self.assertIn('"\\$python_bin" -u generate_dataset.py', script)
        self.assertIn("--resume", script)
        self.assertIn("--dry-run", script)
        self.assertIn("NON_RESTARTABLE_PATTERNS=", script)
        self.assertIn("Pool exhausted", script)
        self.assertIn("Failed to generate a valid", script)
        self.assertIn('grep -E "\\$NON_RESTARTABLE_PATTERNS" "\\$log_file"', script)
        self.assertNotIn("synthetic_traces_regenerated", script)

    def test_runner_rejects_unsupported_benchmark_before_loading_data(self) -> None:
        result = subprocess.run(
            ["bash", "run_experiments.sh", "--benchmarks", "OTHER"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHON_BIN": sys.executable},
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Invalid benchmark: OTHER (Agentic generation is ASTRA-only)", result.stderr)

    def test_screen_launcher_dry_run_prints_curated_mirrored_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            out_root = tmp_path / "generated_agentic_conversations"
            make_launcher_data_root(data_root)

            result = subprocess.run(
                [
                    "bash",
                    "launch_generation_screens.sh",
                    "single_mcp_server",
                    "--dry-run",
                    "--data-root",
                    str(data_root),
                    "--out-root",
                    str(out_root),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
                env={**os.environ, "PYTHON_BIN": sys.executable},
            )

            self.assertIn(
                f"data={data_root}/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped.json",
                result.stdout,
            )
            self.assertIn(
                f"output={out_root}/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped",
                result.stdout,
            )
            self.assertIn(
                f"data={data_root}/task_tool/single_mcp_server/03_three_tools/ASTRA/03_grouped_filtered/validation_curated_grouped.json",
                result.stdout,
            )
            self.assertNotIn("test_grouped.json", result.stdout)
            self.assertNotIn("/single_mcp_server/test/01_tool_ASTRA", result.stdout)
            input_paths = [
                line.removeprefix("data=") for line in result.stdout.splitlines() if line.startswith("data=")
            ]
            self.assertEqual(len(input_paths), 6)
            for input_path in input_paths:
                self.assertIn("ASTRA", Path(input_path).parts)

    def test_screen_launcher_writes_runner_without_expanding_runner_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            out_root = tmp_path / "generated_agentic_conversations"
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            make_launcher_data_root(data_root)
            make_fake_screen_bin(bin_dir)

            env = {**os.environ, "PYTHON_BIN": sys.executable}
            env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
            subprocess.run(
                [
                    "bash",
                    "launch_generation_screens.sh",
                    "single_mcp_server",
                    "--data-root",
                    str(data_root),
                    "--out-root",
                    str(out_root),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            runner = (
                out_root
                / "task_tool"
                / "single_mcp_server"
                / "01_one_tool"
                / "ASTRA"
                / "03_grouped_filtered"
                / "test_curated_grouped"
                / "screen_runner.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("NON_RESTARTABLE_PATTERNS='Pool exhausted|Failed to generate a valid'", runner)
            self.assertIn('grep -E "$NON_RESTARTABLE_PATTERNS"', runner)
            self.assertIn(f"python_bin={sys.executable}", runner)
            self.assertNotIn(".venv/bin/python", runner)
            self.assertIn('"$python_bin" -u generate_dataset.py', runner)
            self.assertIn("Warning: no HTTP(S) proxy variables are set", runner)
            self.assertIn("/run.log", runner)
            self.assertIn('tail -n 40 -F "$log_file" &', runner)
            self.assertIn("tail_pid=$!", runner)
            self.assertIn("trap handle_interrupt INT", runner)
            self.assertIn('kill "$pid"', runner)
            self.assertIn('exec "${SHELL:-/bin/bash}" -l', runner)

    def test_screen_launcher_defaults_find_bundled_inputs_from_another_directory(self) -> None:
        data_root = ROOT.parents[3] / "data" / "conversation"
        with tempfile.TemporaryDirectory() as tmp:
            for server_type, expected_count in (("single_mcp_server", 6), ("cross_mcp_server", 4)):
                with self.subTest(server_type=server_type):
                    result = subprocess.run(
                        ["bash", str(ROOT / "launch_generation_screens.sh"), server_type, "--dry-run"],
                        cwd=tmp,
                        check=True,
                        text=True,
                        capture_output=True,
                        env={**os.environ, "PYTHON_BIN": sys.executable},
                    )
                    inputs = [
                        Path(line.removeprefix("data="))
                        for line in result.stdout.splitlines()
                        if line.startswith("data=")
                    ]
                    self.assertEqual(len(inputs), expected_count)
                    for path in inputs:
                        self.assertTrue(path.is_relative_to(data_root / "task_tool" / server_type))
                        self.assertTrue(path.is_file())

    def test_generator_records_source_provenance_fields(self) -> None:
        source = (ROOT / "generate_dataset.py").read_text(encoding="utf-8")

        self.assertIn('"source_split": source_metadata["source_split"]', source)
        self.assertIn('"source_data_path": source_metadata["source_data_path"]', source)
        self.assertIn('"source_data_file": source_metadata["source_data_file"]', source)
        self.assertIn('"benchmark": source_metadata["benchmark"]', source)
        self.assertIn('"tool_count": source_metadata["tool_count"]', source)


if __name__ == "__main__":
    unittest.main()
