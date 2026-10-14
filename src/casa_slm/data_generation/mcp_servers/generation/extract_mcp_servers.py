# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Extract MCP Tools from Servers."""

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Resource, Tool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class McpServer(BaseModel):
    """Portable export schema for a server's public tool and resource metadata."""

    name: str
    tools: list[Tool] = Field(default_factory=list)
    resources: list[Resource] = Field(default_factory=list)


async def extract_server(name: str, config: dict) -> None:
    """Extract description of a single MCP server and save to JSON file."""
    exit_stack = AsyncExitStack()

    try:
        # Get command path
        command = shutil.which("npx") if config["command"] == "npx" else config["command"]
        if command is None:
            raise ValueError(f"Command not found: {config['command']}")

        # Set up server parameters
        server_params = StdioServerParameters(
            command=command, args=config["args"], env={**os.environ, **config.get("env", {})}
        )

        # Connect to server
        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # Get tools
        tools = []
        try:
            list_tools_response = await session.list_tools()
            tools = list_tools_response.tools
        except Exception as e:
            print(e)

        # Get resources
        resources = []
        try:
            list_resources_response = await session.list_resources()
            resources = list_resources_response.resources
        except Exception as e:
            print(e)

        mcp_server = McpServer(name=name, tools=tools, resources=resources)

        # Use mode='json' to properly serialize Pydantic types like AnyUrl to JSON-compatible formats
        mcp_server_data = mcp_server.model_dump(mode="json")

        # Save to JSON file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.dirname(script_dir)

        # replace dash with underscore in name and create output path
        name = name.replace("-", "_")
        output_path = os.path.join(output_dir, f"{name}.json")

        with open(output_path, "w") as f:
            json.dump(mcp_server_data, f, indent=2)

        logging.info(f"Saved {len(tools)} MCP Server description from '{name}' to {output_path}")
        for tool in tools:
            logging.info(f"  - {tool.name}")

    except Exception as e:
        logging.error(f"Failed to generate MCP Server description from server '{name}': {e}")
        raise
    finally:
        await exit_stack.aclose()


async def main() -> None:
    """Load server config and extract tools from all servers."""
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "servers_config.json")

    with open(config_path, "r") as f:
        server_config = json.load(f)

    # Extract tools from each server
    for name, config in server_config["mcpServers"].items():
        try:
            await extract_server(name, config)
        except Exception as e:
            logging.error(f"Skipping server '{name}' due to error: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
