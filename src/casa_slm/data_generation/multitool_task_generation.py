# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Synthetic generation of tasks corresponding to a combination of (MCP) tool descriptions provided in JSONs."""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
SERVER_DIVERSITY_CHOICES = ("cross", "same", "any")

SYSTEM_PROMPT_TEMPLATE = """You are an expert scenario designer. Your specialty is creating realistic and detailed user task requests that require a tool or a combination of tools to execute them (you are given the tools, and you generate a corresponding task).

Your goal is therefore to generate a **single, high-quality user request**. This request must be a realistic command, and it must require the use of the specific tool(s) provided to you.

You are given the following information about the tool(s), as shown below:

---
[Tool Details]
---

### Your Instructions:

1.  **Be Specific and Realistic:** Do not use generic placeholders. Invent plausible details that look real and coherent. For example, use `'bug-fix/login-error'` instead of `'a branch name'`, `'PROJ-456'` instead of `'an issue key'`, and `'our Q3 marketing campaign'` instead of `'a project summary'`.

2.  **Map to Inputs:** The generated request must implicitly provide all the necessary values for the `required` parameters in the `Input Schema` of all the tools. You can also include values for optional parameters to make the task more realistic and detailed.

3.  **Natural Language Only:** The output must be a single fluid sentence or two, and it must be a realistic task. It should **not** be a list of parameters or a JSON object.

4.  **Focus on the User's Goal:** The request should describe what the user wants to *achieve*, not how the tool works. The tools are needed for the *solution* to the user's request.

### Output Format:

Your response must contain **ONLY** the generated request text and nothing else. Do not add any explanations, preambles, or markdown formatting."""


OBSCURE_SYSTEM_PROMPT_TEMPLATE = """You are an expert scenario designer. Your specialty is creating realistic and detailed user task requests that require a tool or a combination of tools to execute them (you are given the tools, and you generate a corresponding task).

Your goal is to generate a **single, high-quality user request**. This request must be a realistic command that would require the use of the specific tool(s) provided to you to complete it, but it should require them in a somewhat **obscure or implicit or indirect** way, **not** directly ask for the tools. This means that the connection between the task and the tools should **not** be immediately obvious.

You are given the following information about the tool(s), as shown below:

---
[Tool Details]
---

### Your Instructions:

1.  **Be Specific and Realistic:** Do not use generic placeholders. Invent plausible details that look real and coherent. For example, use `'bug-fix/login-error'` instead of `'a branch name'`, `'PROJ-456'` instead of `'an issue key'`, and `'our Q3 marketing campaign'` instead of `'a project summary'`.

2.  **Natural Language Only:** The output must be a single fluid sentence or two, and it must be a realistic task. It should **not** be a list of parameters or a JSON object.

3.  **Focus on the User's Goal:** The request should describe what the user wants to *achieve*, not how the tool works. The tools are needed for the *solution* to the user's request, but the user does **not** directly request them. Make the user's intent the primary focus, with the need for the tools being a secondary inference.

### Output Format:

Your response must contain **ONLY** the generated request text and nothing else. Do not add any explanations, preambles, or markdown formatting."""


def load_env() -> tuple[str, str]:
    """Read root credentials, giving exported environment variables precedence."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    config = {**dotenv_values(env_path), **os.environ}
    api_key = config.get("LLM_API_KEY")
    base_url = config.get("LLM_BASE_URL")

    if not api_key:
        print("Error: LLM_API_KEY not set in the environment or repository-root .env.")
        print("Create the repository-root .env from .env.sample with LLM_API_KEY and LLM_BASE_URL.")
        sys.exit(1)

    if not base_url:
        print("Warning: LLM_BASE_URL not set, using OpenAI default.")

    return api_key, base_url or "https://api.openai.com/v1"


def load_tools_from_file(file_path: str) -> list[dict]:
    """Load tools from a JSON file, handling both flat list and nested MCP server formats.

    Supports:
      - Flat list of tool dicts: [{"name": ..., "inputSchema": ...}, ...]
      - MCP server format: {"name": ..., "tools": [...], "resources": [...]}

    Returns copies of tool dicts (original data is not mutated).
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        server_name = Path(file_path).stem
        return [{**tool, "_mcp_server": server_name} for tool in data]
    elif isinstance(data, dict) and "tools" in data:
        server_name = data.get("name", file_path)
        return [{**tool, "_mcp_server": server_name} for tool in data["tools"]]
    else:
        raise ValueError(f"Unrecognized JSON format in {file_path}. Expected a list of tools or {{'tools': [...]}}.")


def format_tool_details(tools: list[dict], obscure: bool) -> str:
    """Formats the details of multiple tool dicts (list) for the system prompt."""
    details = []
    for tool in tools:
        tool_description = tool.get("description", "N/A")
        cut_sequence = "\n\n    Args:\n        "
        if cut_sequence in tool_description:
            tool_description = tool_description.split(cut_sequence)[0]

        tool_info = [f"**Tool Name:**\n`{tool.get('name', 'N/A')}`\n\n**Tool Description:**\n`{tool_description}`"]
        if not obscure:
            tool_info.append(
                f"\n\n**Input Schema:**\n```json\n{json.dumps(tool.get('inputSchema', {}), indent=2)}\n```"
            )
        details.append("".join(tool_info))
    return "\n\n---\n\n".join(details)


async def task_synthesizer(
    tools_data: list[dict],
    semaphore: asyncio.Semaphore,
    system_prompt_template: str,
    obscure: bool,
    model: str,
    client: AsyncOpenAI,
) -> dict | None:
    """Synthesizes tasks based on the description of multiple MCP tools, with a generative model."""
    tool_details = format_tool_details(tools_data, obscure)
    system_prompt = system_prompt_template.replace("[Tool Details]", tool_details)
    user_prompt = "Generate a single user request now."

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                )
                synthetic_task = response.choices[0].message.content
                tool_names = [tool.get("name") for tool in tools_data]
                return {
                    "tool_names": tool_names,
                    "synthetic_task": synthetic_task,
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    tool_names_str = ", ".join([tool.get("name", "N/A") for tool in tools_data])
                    print(f"Error processing tools {tool_names_str} after {MAX_RETRIES} attempts: {e}")
                    return None
    return None


def _sample_tool_group(
    tool: dict,
    tools_by_server: dict[str, list[dict]],
    available_servers: list[str],
    num_tools: int,
    server_diversity: str,
    rng=random,
) -> list[dict]:
    """Sample a tool group according to the requested MCP server diversity mode."""
    if server_diversity not in SERVER_DIVERSITY_CHOICES:
        raise ValueError(
            f"Invalid server_diversity={server_diversity!r}. Expected one of: {', '.join(SERVER_DIVERSITY_CHOICES)}"
        )

    if num_tools == 1:
        return [tool]

    anchor_server = tool.get("_mcp_server", "unknown")

    if server_diversity == "cross":
        if len(available_servers) < num_tools:
            raise ValueError(
                f"Not enough MCP servers ({len(available_servers)}) to enforce cross-server diversity "
                f"with num_tools={num_tools}. Need at least {num_tools} distinct servers."
            )
        other_servers = [s for s in available_servers if s != anchor_server]
        partner_servers = rng.sample(other_servers, num_tools - 1)
        random_tools = [rng.choice(tools_by_server[s]) for s in partner_servers]
        return [tool, *random_tools]

    if server_diversity == "same":
        candidates = [candidate for candidate in tools_by_server[anchor_server] if candidate is not tool]
        if len(candidates) < num_tools - 1:
            raise ValueError(
                f"Not enough tools in MCP server '{anchor_server}' to create a same-server group "
                f"with num_tools={num_tools}. Need at least {num_tools} tools in the anchor server."
            )
        return [tool, *rng.sample(candidates, num_tools - 1)]

    candidates = [
        candidate for server in available_servers for candidate in tools_by_server[server] if candidate is not tool
    ]
    if len(candidates) < num_tools - 1:
        raise ValueError(f"Not enough tools ({len(candidates) + 1}) to create a task with {num_tools} tools.")
    return [tool, *rng.sample(candidates, num_tools - 1)]


async def process_tools_files(
    input_paths: list[str],
    output_path: str,
    multiplier: int,
    num_tools: int,
    obscure: bool,
    model: str,
    concurrency: int,
    server_diversity: str = "cross",
):
    """Reads tools from JSON files, creates their tasks in parallel, and saves the results."""
    all_tools: list[dict] = []
    tool_to_server: dict[str, str] = {}
    for input_path in input_paths:
        tools = load_tools_from_file(input_path)
        for tool in tools:
            name = tool["name"]
            server = tool.get("_mcp_server", "unknown")
            if name in tool_to_server and tool_to_server[name] != server:
                print(
                    f"Warning: tool '{name}' exists in multiple servers ({tool_to_server[name]}, {server}). Using latest."
                )
            tool_to_server[name] = server
        all_tools.extend(tools)

    if len(all_tools) < num_tools:
        print(f"Error: Not enough tools ({len(all_tools)}) to create a task with {num_tools} tools.")
        return

    api_key, base_url = load_env()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"Loaded {len(all_tools)} tools from {len(input_paths)} file(s).")
    print(f"Scheduling {len(all_tools) * multiplier} LLM calls (multiplier={multiplier}, num_tools={num_tools})...")

    semaphore = asyncio.Semaphore(concurrency)
    system_prompt_template = OBSCURE_SYSTEM_PROMPT_TEMPLATE if obscure else SYSTEM_PROMPT_TEMPLATE
    system_prompt_tag = "obscure_base" if obscure else "base"

    tools_by_server: dict[str, list[dict]] = {}
    for tool in all_tools:
        tools_by_server.setdefault(tool.get("_mcp_server", "unknown"), []).append(tool)

    available_servers = list(tools_by_server.keys())
    task_inputs = []
    for idx, tool in enumerate(all_tools):
        for _ in range(multiplier):
            tool_group = _sample_tool_group(tool, tools_by_server, available_servers, num_tools, server_diversity)
            key = tuple(sorted(t["name"] for t in tool_group))
            task_inputs.append((key, tool_group))

    async def _run(tool_group):
        return await task_synthesizer(tool_group, semaphore, system_prompt_template, obscure, model, client)

    results = await tqdm.gather(*[_run(tg) for _, tg in task_inputs])

    grouped: dict[tuple, dict] = {}
    for (key, tool_group), res in zip(task_inputs, results):
        if res is None:
            continue
        if key not in grouped:
            grouped[key] = {"tool_names": res["tool_names"], "tasks": []}
        grouped[key]["tasks"].append(res["synthetic_task"])

    final_results = []
    for key, group in grouped.items():
        tool_names = sorted(group["tool_names"])
        final_results.append(
            {
                "tool_names": tool_names,
                "mcp_servers": [tool_to_server.get(t, "unknown") for t in tool_names],
                "synthetic_tasks": group["tasks"],
                "system_prompt": system_prompt_tag,
                "tools_per_task": num_tools,
                "SO_tasks_per_sample": len(group["tasks"]),
                "conversation": False,
            }
        )

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_path_obj.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    total_requested = len(all_tools) * multiplier
    total_successful = sum(len(g["synthetic_tasks"]) for g in final_results)
    print(f"\nCreated {total_successful} / {total_requested} requested synthetic tasks for {len(all_tools)} tools.")
    print(f"Output saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Generate synthetic tasks from MCP tool descriptions using an LLM.")
    parser.add_argument(
        "--input-files", nargs="+", required=True, help="Paths to input JSON files with MCP tool descriptions."
    )
    parser.add_argument("--output-file", required=True, help="Path to output JSON file for synthetic tasks.")
    parser.add_argument("--multiplier", type=int, default=1, help="Number of synthetic tasks to generate per tool.")
    parser.add_argument("--num-tools", type=int, default=1, help="Number of tools to combine per synthetic task.")
    parser.add_argument(
        "--no-obscure",
        action="store_true",
        help="Disable obscure mode: generate direct tasks (includes input schema in prompt).",
    )
    parser.add_argument(
        "--model", default="azure/gpt-5.2", help="Model to use for generation (default: azure/gpt-5.2)."
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests (default: 10).")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible tool combinations (default: None)."
    )
    parser.add_argument(
        "--server-diversity",
        choices=SERVER_DIVERSITY_CHOICES,
        default="cross",
        help=(
            "MCP server diversity mode for multi-tool groups: cross=distinct MCP servers (default), "
            "same=all tools from the anchor MCP server, any=no MCP server constraint."
        ),
    )
    return parser


async def main():
    """Main function to parse arguments and run the script."""
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    start_time = time.time()
    await process_tools_files(
        args.input_files,
        args.output_file,
        args.multiplier,
        args.num_tools,
        not args.no_obscure,
        args.model,
        args.concurrency,
        args.server_diversity,
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f}s ({elapsed_time / 60:.2f}min)")


if __name__ == "__main__":
    asyncio.run(main())
