# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Matching simulator for task-tool evaluation.

Simulates correct, wrong, and null tool matches for evaluating
the tool selector component.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load and validate configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["mcp_servers", "num_correct_matches", "ratio_wrong_matches", "ratio_null_matches"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys in {config_path}: {missing}")

    return config


def load_mcp_tools(mcp_server_path: str, mcp_server_names: list[str]) -> dict[str, list[str]]:
    """Load all tools for each MCP server.

    Auto-detects JSON format:
      - MCP server format: {"name": ..., "tools": [...]} -> reads tool names from "tools" key
      - Flat list format: [{"name": ...}, ...] -> reads tool names directly
    """
    mcp_to_tools = {}
    base_path = Path(mcp_server_path)

    for server_name in mcp_server_names:
        file_path = base_path / f"{server_name.replace('-', '_')}.json"
        if not file_path.exists():
            file_path = base_path / f"{server_name}.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"MCP server file not found: tried '{server_name.replace('-', '_')}.json' "
                f"and '{server_name}.json' in {mcp_server_path}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            server_data = json.load(f)

        if isinstance(server_data, dict) and "tools" in server_data:
            mcp_to_tools[server_name] = [tool["name"] for tool in server_data["tools"]]
        elif isinstance(server_data, list):
            mcp_to_tools[server_name] = [tool["name"] for tool in server_data]
        else:
            raise ValueError(f"Unrecognized format in {file_path}. Expected {{'tools': [...]}} or a flat list.")

    return mcp_to_tools


def create_task_collection(input_path: str, mcp_server_names: list[str]) -> list[dict[str, Any]]:
    """Load generated tasks and create a flat list of tasks with their metadata."""
    tasks_file = Path(input_path)
    if not tasks_file.exists():
        raise FileNotFoundError(f"Generated tasks file not found: {tasks_file}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks_data = json.load(f)

    all_tasks = []
    skipped = 0
    for item in tasks_data:
        if any(server not in mcp_server_names for server in item["mcp_servers"]):
            skipped += 1
            continue
        for task in item.get("synthetic_tasks", []):
            all_tasks.append(
                {
                    "task": task,
                    "correct_tools": item["tool_names"],
                    "mcp_servers": item["mcp_servers"],
                }
            )

    if skipped:
        logger.info(f"  Skipped {skipped} tool groups (servers not in config.mcp_servers)")

    logger.info(f"Loaded {len(all_tasks)} tasks from {input_path}")
    if tasks_data:
        logger.info(f"  System prompt tag: {tasks_data[0].get('system_prompt', 'N/A')}")
        logger.info(f"  Tools per task: {tasks_data[0].get('tools_per_task', 'N/A')}")
        logger.info(f"  Conversation mode: {tasks_data[0].get('conversation', False)}")

    return all_tasks


def generate_matches(all_tasks: list[dict[str, Any]], config: dict, mcp_tools: dict[str, list[str]]) -> list[dict]:
    """Generate correct, wrong, and null matches with a flexible sampling strategy."""
    num_correct = config["num_correct_matches"]
    correct_tasks = random.sample(all_tasks, k=num_correct)
    num_wrong = int(num_correct * config["ratio_wrong_matches"])
    num_null = int(num_correct * config["ratio_null_matches"])

    generated_entries: list[dict] = []

    def _append_entry(
        task: str,
        requested_tools: list[str],
        mcp_servers: list[str],
        gt_tools: list[str],
        gt_mcp_servers: list[str],
        match_tag: str,
    ):
        generated_entries.append(
            {
                "input": {
                    "task": task,
                    "tools": requested_tools,
                    "mcp_servers": mcp_servers,
                },
                "groundtruth": {
                    "tools": gt_tools,
                    "mcp_servers": gt_mcp_servers,
                },
                "match_tag": match_tag,
            }
        )

    for task in correct_tasks:
        _append_entry(
            task["task"],
            task["correct_tools"],
            task["mcp_servers"],
            task["correct_tools"],
            task["mcp_servers"],
            "correct",
        )

    if num_wrong > 0:
        wrong_tasks_base = _oversample_tasks(all_tasks, num_wrong)

        for task in wrong_tasks_base:
            wrong_tools = []
            used_tools: set[str] = set()
            for tool_name, server in zip(task["correct_tools"], task["mcp_servers"]):
                possible = [t for t in mcp_tools[server] if t != tool_name and t not in used_tools]
                if not possible:
                    raise ValueError(
                        f"Not enough tools in MCP server '{server}' "
                        f"to create a wrong match for task '{task['task'][:80]}...'."
                    )
                chosen = random.choice(possible)
                wrong_tools.append(chosen)
                used_tools.add(chosen)

            if config.get("mixing_wrong_null_in_multitool", False) and len(task["correct_tools"]) > 1:
                wrong_tools = _mix_tools(wrong_tools, task["correct_tools"])

            _append_entry(
                task["task"],
                wrong_tools,
                task["mcp_servers"],
                task["correct_tools"],
                task["mcp_servers"],
                "wrong",
            )

    if num_null > 0:
        null_tasks_base = _oversample_tasks(all_tasks, num_null)
        mcp_list = list(mcp_tools.keys())

        for task in null_tasks_base:
            wrong_tools = []
            mixed_wrong_mcp = []

            task_servers = set(task["mcp_servers"])
            possible_other_mcps = [mcp for mcp in mcp_list if mcp not in task_servers and mcp_tools[mcp]]
            if not possible_other_mcps:
                raise ValueError(
                    f"Not enough other MCP servers to create a null match for task '{task['task'][:80]}...'. "
                    f"Task uses servers {task_servers} but no other servers are available."
                )

            used_null_tools: set[str] = set()
            for tool_name, server in zip(task["correct_tools"], task["mcp_servers"]):
                wrong_mcp = random.choice(possible_other_mcps)
                candidates = [t for t in mcp_tools[wrong_mcp] if t not in used_null_tools]
                if not candidates:
                    candidates = mcp_tools[wrong_mcp]
                wrong_tool = random.choice(candidates)
                wrong_tools.append(wrong_tool)
                used_null_tools.add(wrong_tool)
                mixed_wrong_mcp.append(wrong_mcp)

            if config.get("mixing_wrong_null_in_multitool", False) and len(task["correct_tools"]) > 1:
                bin_vec = _random_nonzero_bitvec(len(wrong_tools))
                wrong_tools = [
                    wrong_tools[i] if bin_vec[i] == 1 else task["correct_tools"][i] for i in range(len(wrong_tools))
                ]
                mixed_wrong_mcp = [
                    mixed_wrong_mcp[i] if bin_vec[i] == 1 else task["mcp_servers"][i] for i in range(len(wrong_tools))
                ]

            _append_entry(
                task["task"],
                wrong_tools,
                mixed_wrong_mcp,
                task["correct_tools"],
                task["mcp_servers"],
                "null",
            )

    random.shuffle(generated_entries)
    return generated_entries


def _oversample_tasks(all_tasks: list[dict], target_count: int) -> list[dict]:
    """Sample tasks to reach target_count, repeating and shuffling if needed."""
    if target_count <= len(all_tasks):
        return random.sample(all_tasks, k=target_count)

    coverage = target_count // len(all_tasks)
    remainder = target_count - (coverage * len(all_tasks))
    result = [task for _ in range(coverage) for task in all_tasks] + random.sample(all_tasks, k=remainder)
    random.shuffle(result)
    return result


def _random_nonzero_bitvec(length: int) -> list[int]:
    """Generate a random bit vector of given length with at least one 1."""
    vec = [0] * length
    while sum(vec) == 0:
        vec = [random.randint(0, 1) for _ in range(length)]
    return vec


def _mix_tools(wrong_tools: list[str], correct_tools: list[str]) -> list[str]:
    """Mix wrong and correct tools ensuring at least one wrong tool remains."""
    bin_vec = _random_nonzero_bitvec(len(wrong_tools))
    return [wrong_tools[i] if bin_vec[i] == 1 else correct_tools[i] for i in range(len(wrong_tools))]


def generate_tool_requests_data(
    config_path: str, mcp_server_path: str, input_path: str, output_path: str, seed: int = 42
) -> list[dict]:
    """Run the matching simulator to produce evaluation data.

    Args:
        config_path: Path to the configuration file.
        mcp_server_path: Path to the directory containing MCP server JSON files.
        input_path: Path to the input JSON file with synthetic tasks.
        output_path: Path to the output JSON file for simulated matches.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    logger.info(f"Random seed: {seed}")

    config = load_config(config_path)
    all_tasks = create_task_collection(input_path, config["mcp_servers"])

    if not all_tasks:
        raise ValueError("No tasks were loaded. Check that input file contains tasks for the configured MCP servers.")

    if len(all_tasks) < config["num_correct_matches"]:
        logger.warning(
            f"Not enough tasks ({len(all_tasks)}) for {config['num_correct_matches']} correct matches. "
            f"Capping to {len(all_tasks)}."
        )
        config["num_correct_matches"] = len(all_tasks)

    mcp_to_tools = load_mcp_tools(mcp_server_path, config["mcp_servers"])
    generated_data = generate_matches(all_tasks, config, mcp_to_tools)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, indent=2)

    num_correct = config["num_correct_matches"]
    num_wrong = int(num_correct * config["ratio_wrong_matches"])
    num_null = int(num_correct * config["ratio_null_matches"])

    logger.info(f"Generated {len(generated_data)} entries: correct={num_correct}, wrong={num_wrong}, null={num_null}")
    logger.info(f"MCP servers used: {', '.join(config['mcp_servers'])}")
    logger.info(f"Saved to: {output_path}")

    return generated_data


def main():
    """Main function to run the matching simulator."""
    parser = argparse.ArgumentParser(
        description="Matching simulator: generate correct/wrong/null tool matches for evaluation."
    )
    parser.add_argument(
        "--config", default="config_test.json", help="Path to configuration file (default: config_test.json)."
    )
    parser.add_argument(
        "--mcp-server-path", default="mcp_servers", help="Path to MCP server JSON files (default: mcp_servers)."
    )
    parser.add_argument("--input", default="data/generated_tasks.json", help="Path to input JSON with synthetic tasks.")
    parser.add_argument("--output", default="data/generated_data.json", help="Path to output JSON for evaluation data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()

    generate_tool_requests_data(
        config_path=args.config,
        mcp_server_path=args.mcp_server_path,
        input_path=args.input,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
