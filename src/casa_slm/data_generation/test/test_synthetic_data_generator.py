"""Tests for Stage 2 synthetic match generation."""

import random

from casa_slm.data_generation import synthetic_data_generator as sdg


def _task(task: str, tools: list[str], servers: list[str]) -> dict:
    return {
        "task": task,
        "correct_tools": tools,
        "mcp_servers": servers,
    }


def _entries_by_tag(entries: list[dict]) -> dict[str, list[dict]]:
    by_tag: dict[str, list[dict]] = {"correct": [], "wrong": [], "null": []}
    for entry in entries:
        by_tag[entry["match_tag"]].append(entry)
    return by_tag


def _base_tasks() -> list[dict]:
    return [
        _task("Review the checkout incident and release fix", ["a_get", "b_get"], ["server-a", "server-b"]),
        _task("Audit the quarterly metrics pipeline", ["a_list", "b_list"], ["server-a", "server-b"]),
    ]


def _mcp_tools() -> dict[str, list[str]]:
    return {
        "server-a": ["a_get", "a_list", "a_delete", "a_update"],
        "server-b": ["b_get", "b_list", "b_delete", "b_update"],
        "server-c": ["c_get", "c_list", "c_delete", "c_update"],
        "server-d": ["d_get", "d_list", "d_delete", "d_update"],
    }


def _config(*, wrong_ratio: float = 0.0, null_ratio: float = 0.0, mixing: bool = False) -> dict:
    return {
        "num_correct_matches": 1,
        "ratio_wrong_matches": wrong_ratio,
        "ratio_null_matches": null_ratio,
        "mixing_wrong_null_in_multitool": mixing,
    }


def test_wrong_matches_keep_groundtruth_servers_and_replace_each_tool():
    random.seed(42)

    entries = sdg.generate_matches(_base_tasks(), _config(wrong_ratio=1.0), _mcp_tools())

    wrong_entries = _entries_by_tag(entries)["wrong"]
    assert len(wrong_entries) == 1
    wrong_entry = wrong_entries[0]
    assert wrong_entry["input"]["mcp_servers"] == wrong_entry["groundtruth"]["mcp_servers"]
    for requested_tool, correct_tool, server in zip(
        wrong_entry["input"]["tools"],
        wrong_entry["groundtruth"]["tools"],
        wrong_entry["groundtruth"]["mcp_servers"],
    ):
        assert requested_tool != correct_tool
        assert requested_tool in _mcp_tools()[server]


def test_null_matches_use_only_servers_outside_the_groundtruth_server_set():
    random.seed(42)

    entries = sdg.generate_matches(_base_tasks(), _config(null_ratio=1.0), _mcp_tools())

    null_entries = _entries_by_tag(entries)["null"]
    assert len(null_entries) == 1
    null_entry = null_entries[0]
    groundtruth_servers = set(null_entry["groundtruth"]["mcp_servers"])
    for requested_tool, requested_server in zip(null_entry["input"]["tools"], null_entry["input"]["mcp_servers"]):
        assert requested_server not in groundtruth_servers
        assert requested_tool in _mcp_tools()[requested_server]


def test_mixed_wrong_matches_keep_at_least_one_wrong_tool(monkeypatch):
    random.seed(42)
    monkeypatch.setattr(sdg, "_random_nonzero_bitvec", lambda length: [0, 1])

    entries = sdg.generate_matches(_base_tasks(), _config(wrong_ratio=1.0, mixing=True), _mcp_tools())

    wrong_entry = _entries_by_tag(entries)["wrong"][0]
    requested_tools = wrong_entry["input"]["tools"]
    correct_tools = wrong_entry["groundtruth"]["tools"]
    assert any(requested == correct for requested, correct in zip(requested_tools, correct_tools))
    assert any(requested != correct for requested, correct in zip(requested_tools, correct_tools))


def test_mixed_null_matches_keep_at_least_one_outside_server(monkeypatch):
    random.seed(42)
    monkeypatch.setattr(sdg, "_random_nonzero_bitvec", lambda length: [0, 1])

    entries = sdg.generate_matches(_base_tasks(), _config(null_ratio=1.0, mixing=True), _mcp_tools())

    null_entry = _entries_by_tag(entries)["null"][0]
    requested_servers = null_entry["input"]["mcp_servers"]
    correct_servers = null_entry["groundtruth"]["mcp_servers"]
    assert any(requested == correct for requested, correct in zip(requested_servers, correct_servers))
    assert any(requested != correct for requested, correct in zip(requested_servers, correct_servers))


def test_oversample_tasks_is_deterministic_for_same_seed():
    tasks = _base_tasks()

    random.seed(2026)
    first_sample = [task["task"] for task in sdg._oversample_tasks(tasks, target_count=7)]
    random.seed(2026)
    second_sample = [task["task"] for task in sdg._oversample_tasks(tasks, target_count=7)]

    assert first_sample == second_sample
    assert len(first_sample) == 7
    assert set(first_sample) == {task["task"] for task in tasks}
