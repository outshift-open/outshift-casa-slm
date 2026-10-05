"""Tests for multi-tool task generation sampling modes."""

import pytest

from casa_slm.data_generation import multitool_task_generation as mtg


class FirstRng:
    """Deterministic random-like object for sampling tests."""

    def sample(self, population, k):
        return list(population[:k])

    def choice(self, population):
        return population[0]


def _tool(name: str, server: str) -> dict:
    return {"name": name, "description": f"{name} description", "_mcp_server": server}


@pytest.fixture
def tools_by_server():
    return {
        "server-a": [_tool("a0", "server-a"), _tool("a1", "server-a"), _tool("a2", "server-a")],
        "server-b": [_tool("b0", "server-b"), _tool("b1", "server-b")],
        "server-c": [_tool("c0", "server-c"), _tool("c1", "server-c")],
    }


def test_parser_defaults_server_diversity_to_cross():
    parser = mtg.build_parser()

    args = parser.parse_args(["--input-files", "mcp_servers/a.json", "--output-file", "data/generated_tasks.json"])

    assert args.server_diversity == "cross"


def test_parser_rejects_invalid_server_diversity():
    parser = mtg.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input-files",
                "mcp_servers/a.json",
                "--output-file",
                "data/generated_tasks.json",
                "--server-diversity",
                "bogus",
            ]
        )


def test_cross_mode_samples_distinct_partner_servers(tools_by_server):
    anchor = tools_by_server["server-a"][0]

    group = mtg._sample_tool_group(
        anchor,
        tools_by_server,
        list(tools_by_server.keys()),
        num_tools=3,
        server_diversity="cross",
        rng=FirstRng(),
    )

    assert group[0] is anchor
    assert len(group) == 3
    assert {tool["_mcp_server"] for tool in group} == {"server-a", "server-b", "server-c"}


def test_cross_mode_raises_when_there_are_too_few_servers(tools_by_server):
    anchor = tools_by_server["server-a"][0]

    with pytest.raises(ValueError, match="Not enough MCP servers"):
        mtg._sample_tool_group(
            anchor,
            {"server-a": tools_by_server["server-a"]},
            ["server-a"],
            num_tools=2,
            server_diversity="cross",
            rng=FirstRng(),
        )


def test_same_mode_samples_only_from_anchor_server(tools_by_server):
    anchor = tools_by_server["server-a"][0]

    group = mtg._sample_tool_group(
        anchor,
        tools_by_server,
        list(tools_by_server.keys()),
        num_tools=3,
        server_diversity="same",
        rng=FirstRng(),
    )

    assert group[0] is anchor
    assert len(group) == 3
    assert all(tool["_mcp_server"] == "server-a" for tool in group)
    assert len({id(tool) for tool in group}) == 3


def test_same_mode_raises_when_anchor_server_has_too_few_tools(tools_by_server):
    anchor = tools_by_server["server-b"][0]

    with pytest.raises(ValueError, match="Not enough tools in MCP server 'server-b'"):
        mtg._sample_tool_group(
            anchor,
            tools_by_server,
            list(tools_by_server.keys()),
            num_tools=3,
            server_diversity="same",
            rng=FirstRng(),
        )


def test_any_mode_samples_from_full_non_anchor_pool_without_server_constraint(tools_by_server):
    anchor = tools_by_server["server-a"][0]

    group = mtg._sample_tool_group(
        anchor,
        tools_by_server,
        list(tools_by_server.keys()),
        num_tools=3,
        server_diversity="any",
        rng=FirstRng(),
    )

    assert group[0] is anchor
    assert len(group) == 3
    assert len({id(tool) for tool in group}) == 3
    assert [tool["name"] for tool in group] == ["a0", "a1", "a2"]


@pytest.mark.parametrize("server_diversity", ["cross", "same", "any"])
def test_num_tools_one_returns_anchor_for_all_modes(tools_by_server, server_diversity):
    anchor = tools_by_server["server-a"][0]

    group = mtg._sample_tool_group(
        anchor,
        {"server-a": [anchor]},
        ["server-a"],
        num_tools=1,
        server_diversity=server_diversity,
        rng=FirstRng(),
    )

    assert group == [anchor]
