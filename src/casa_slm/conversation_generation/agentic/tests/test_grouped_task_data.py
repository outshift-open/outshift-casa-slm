"""Test grouped task normalization, trace generation, and resume safeguards."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from casa_slm.conversation_generation.agentic import generate_dataset


def tool_schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


class GroupedTaskDataTests(unittest.TestCase):
    def test_load_task_entries_normalizes_grouped_clean_and_negative_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_curated_grouped.json"
            path.write_text(
                """[
                  {
                    "group_id": "clean-group",
                    "task": {
                      "task": "clean task",
                      "metadata": {
                        "task": {
                          "seed_tools": [
                            {"name": "tool_a", "mcp_server": "server_a"}
                          ]
                        }
                      }
                    },
                    "requests": [
                      {
                        "id": "request-a",
                        "request": {
                          "task": "clean task",
                          "tool": {"name": "tool_a", "description": "A"}
                        },
                        "label": {"relevant": true},
                        "metadata": {
                          "request": {"tool": {"mcp_server": "server_a"}}
                        }
                      }
                    ]
                  },
                  {
                    "group_id": "negative-group",
                    "task": {
                      "task": "negative task",
                      "metadata": {
                        "task": {
                          "seed_tools": [
                            {"name": "truth_tool", "mcp_server": "truth_server"}
                          ]
                        }
                      }
                    },
                    "requests": [
                      {
                        "id": "request-b",
                        "request": {
                          "task": "negative task",
                          "tool": {"name": "irrelevant_tool", "description": "B"}
                        },
                        "label": {"relevant": false},
                        "metadata": {
                          "request": {"tool": {"mcp_server": "irrelevant_server"}}
                        }
                      }
                    ]
                  }
                ]""",
                encoding="utf-8",
            )

            entries = generate_dataset.load_task_entries(path)

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["match_tag"], "correct")
            self.assertEqual(entries[0]["input"]["tools"], ["tool_a"])
            self.assertEqual(entries[0]["groundtruth"]["tools"], ["tool_a"])
            self.assertEqual(entries[0]["group_id"], "clean-group")
            self.assertEqual(
                entries[0]["metadata"],
                {
                    "task": {"task": {"seed_tools": [{"name": "tool_a", "mcp_server": "server_a"}]}},
                    "requests": [
                        {
                            "id": "request-a",
                            "tool": {"name": "tool_a", "mcp_server": "server_a"},
                            "label": {"relevant": True},
                            "metadata": {"request": {"tool": {"mcp_server": "server_a"}}},
                        }
                    ],
                },
            )
            self.assertEqual(entries[1]["match_tag"], "wrong")
            self.assertEqual(entries[1]["input"]["tools"], ["irrelevant_tool"])
            self.assertEqual(entries[1]["groundtruth"]["tools"], [])
            self.assertEqual(entries[1]["group_id"], "negative-group")

    def test_legacy_astra_records_are_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "input": {"task": "old task", "tools": ["old_tool"]},
                            "groundtruth": {"tools": ["old_tool"]},
                            "match_tag": "correct",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "--allow-legacy-astra"):
                generate_dataset.load_task_entries(path)

            entries = generate_dataset.load_task_entries(path, allow_legacy_astra=True)
            self.assertEqual(entries[0]["input"]["task"], "old task")

    def test_build_task_spec_preserves_group_id_from_grouped_records(self):
        entry = {
            "group_id": "gid-123",
            "source_request_ids": ["request-1"],
            "input": {"task": "task", "tools": ["irrelevant"], "mcp_servers": ["s"]},
            "groundtruth": {"tools": ["truth"], "mcp_servers": ["s"]},
            "match_tag": "wrong",
            "metadata": {"task": {"seed_tools": []}, "requests": []},
        }
        catalog = {"irrelevant": tool_schema("irrelevant"), "truth": tool_schema("truth")}

        task = generate_dataset.build_task_spec(entry, catalog, random.Random(0), 7)

        self.assertEqual(task.group_id, "gid-123")
        self.assertEqual(task.source_request_ids, ["request-1"])
        self.assertEqual(task.metadata, {"task": {"seed_tools": []}, "requests": []})
        self.assertEqual(task.user_query, "task")
        self.assertEqual(generate_dataset.schema_names(task.relevant_tool_schemas), ["truth"])
        self.assertEqual(generate_dataset.schema_names(task.irrelevant_tool_schemas), ["irrelevant"])

    def test_grouped_negative_removes_irrelevant_seed_tools_from_groundtruth(self):
        record = {
            "group_id": "mixed-group",
            "task": {
                "task": "mixed task",
                "metadata": {
                    "task": {
                        "seed_tools": [
                            {"name": "truth", "mcp_server": "server"},
                            {"name": "overlap", "mcp_server": "server"},
                        ]
                    }
                },
            },
            "requests": [
                {
                    "id": "relevant-request",
                    "request": {"task": "mixed task", "tool": {"name": "truth"}},
                    "label": {"relevant": True},
                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                },
                {
                    "id": "irrelevant-request",
                    "request": {"task": "mixed task", "tool": {"name": "overlap"}},
                    "label": {"relevant": False},
                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                },
            ],
        }

        entry = generate_dataset._normalize_grouped_task_entry(record, 0)
        catalog = {
            "truth": tool_schema("truth"),
            "overlap": tool_schema("overlap"),
        }
        task = generate_dataset.build_task_spec(entry, catalog, random.Random(0), 0)

        self.assertEqual(entry["input"]["tools"], ["overlap"])
        self.assertEqual(entry["groundtruth"]["tools"], ["truth"])
        self.assertEqual(generate_dataset.schema_names(task.relevant_tool_schemas), ["truth"])
        self.assertEqual(generate_dataset.schema_names(task.irrelevant_tool_schemas), ["overlap"])

    def test_grouped_negative_does_not_add_seed_only_tools_to_groundtruth(self):
        record = {
            "group_id": "mixed-group",
            "task": {
                "task": "mixed task",
                "metadata": {
                    "task": {
                        "seed_tools": [
                            {"name": "truth", "mcp_server": "server"},
                            {"name": "seed_only", "mcp_server": "server"},
                        ]
                    }
                },
            },
            "requests": [
                {
                    "id": "relevant-request",
                    "request": {"task": "mixed task", "tool": {"name": "truth"}},
                    "label": {"relevant": True},
                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                },
                {
                    "id": "irrelevant-request",
                    "request": {"task": "mixed task", "tool": {"name": "irrelevant"}},
                    "label": {"relevant": False},
                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                },
            ],
        }

        entry = generate_dataset._normalize_grouped_task_entry(record, 0)
        catalog = {
            "truth": tool_schema("truth"),
            "seed_only": tool_schema("seed_only"),
            "irrelevant": tool_schema("irrelevant"),
        }
        task = generate_dataset.build_task_spec(entry, catalog, random.Random(0), 0)

        self.assertEqual(entry["input"]["tools"], ["irrelevant"])
        self.assertEqual(entry["groundtruth"]["tools"], ["truth"])
        self.assertEqual(generate_dataset.schema_names(task.relevant_tool_schemas), ["truth"])
        self.assertEqual(generate_dataset.schema_names(task.irrelevant_tool_schemas), ["irrelevant"])

    def test_build_task_spec_removes_resolved_relevant_irrelevant_overlap(self):
        entry = {
            "group_id": "gid-123",
            "input": {"task": "task", "tools": ["same"], "mcp_servers": ["s"]},
            "groundtruth": {"tools": ["same", "truth"], "mcp_servers": ["s", "s"]},
            "match_tag": "wrong",
        }
        catalog = {"same": tool_schema("same"), "truth": tool_schema("truth")}

        task = generate_dataset.build_task_spec(entry, catalog, random.Random(0), 7)

        self.assertEqual(generate_dataset.schema_names(task.relevant_tool_schemas), ["truth"])
        self.assertEqual(generate_dataset.schema_names(task.irrelevant_tool_schemas), ["same"])

    def test_grouped_negative_all_irrelevant_can_have_no_relevant_tools(self):
        record = {
            "group_id": "all-irrelevant-group",
            "task": {
                "task": "task with no catalog-backed relevant tool",
                "metadata": {
                    "task": {
                        "seed_tools": [
                            {"name": "irrelevant_a", "mcp_server": "server"},
                            {"name": "irrelevant_b", "mcp_server": "server"},
                        ]
                    }
                },
            },
            "requests": [
                {
                    "id": "irrelevant-a",
                    "request": {"tool": {"name": "irrelevant_a"}},
                    "label": {"relevant": False},
                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                },
                {
                    "id": "irrelevant-b",
                    "request": {"tool": {"name": "irrelevant_b"}},
                    "label": {"relevant": False},
                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                },
            ],
        }

        entry = generate_dataset._normalize_grouped_task_entry(record, 0)
        catalog = {
            "irrelevant_a": tool_schema("irrelevant_a"),
            "irrelevant_b": tool_schema("irrelevant_b"),
        }
        task = generate_dataset.build_task_spec(entry, catalog, random.Random(0), 0)

        self.assertEqual(entry["groundtruth"]["tools"], [])
        self.assertEqual(generate_dataset.schema_names(task.relevant_tool_schemas), [])
        self.assertEqual(
            generate_dataset.schema_names(task.irrelevant_tool_schemas),
            ["irrelevant_a", "irrelevant_b"],
        )

    def test_validate_task_dataset_allows_all_irrelevant_negative_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "validation_grouped.json"
            mcp_dir = Path(tmp) / "mcp"
            mcp_dir.mkdir()
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "group_id": "all-irrelevant-negative",
                            "task": {
                                "task": "Deploy an assistant with unrelated bot tooling.",
                                "metadata": {
                                    "task": {
                                        "seed_tools": [
                                            {
                                                "name": "deploy_bot_with_controllers",
                                                "mcp_server": "hummingbot-mcp",
                                            }
                                        ]
                                    }
                                },
                            },
                            "requests": [
                                {
                                    "id": "request-1",
                                    "request": {
                                        "task": "Deploy an assistant with unrelated bot tooling.",
                                        "tool": {"name": "deploy_bot_with_controllers"},
                                    },
                                    "label": {"relevant": False},
                                    "metadata": {"request": {"tool": {"mcp_server": "hummingbot-mcp"}}},
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (mcp_dir / "hummingbot-mcp.json").write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "name": "deploy_bot_with_controllers",
                                "description": "Deploy a bot with controller configs.",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = generate_dataset.validate_task_dataset(data_path, mcp_dir)

            self.assertEqual(report["total_entries"], 1)
            self.assertEqual(report["negative_entries"], 1)
            self.assertEqual(report["missing_relevant_schema_count"], 0)
            self.assertEqual(report["missing_irrelevant_schema_count"], 0)

    def test_manifest_record_includes_grouped_source_fields(self):
        dialog = {
            "scenario": {
                "available_tools": ["truth"],
                "relevant_tools": ["truth"],
                "irrelevant_tools": [],
                "group_id": "gid-123",
                "source_request_ids": ["request-1"],
                "metadata": {
                    "task": {"task": {"seed_tools": [{"name": "truth"}]}},
                    "requests": [
                        {
                            "id": "request-1",
                            "tool": {"name": "truth", "mcp_server": "server"},
                            "label": {"relevant": True},
                            "metadata": {"label": {"type": "relevant_seed"}},
                        }
                    ],
                },
            },
            "notes": {
                "label": "clean",
                "task_name": "task_000001",
                "stop_reason": "natural_end",
                "relevant_tool_calls": 1,
                "irrelevant_tool_calls": 0,
                "target_irrelevant_tool": None,
            },
        }
        source_metadata = {
            "source_split": "test",
            "source_data_path": "../data/task_tool/path/test_curated_grouped.json",
            "source_data_file": "test_curated_grouped.json",
            "benchmark": "ASTRA",
            "tool_count": "01_tool",
        }

        record = generate_dataset._make_manifest_record(
            0,
            "000000.json",
            dialog,
            source_metadata,
        )

        self.assertEqual(record["group_id"], "gid-123")
        self.assertEqual(record["source_request_ids"], ["request-1"])
        self.assertEqual(record["metadata"], dialog["scenario"]["metadata"])

    def test_output_dir_rejects_existing_mismatched_source_even_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            dialogs_dir = out_dir / "dialogs"
            dialogs_dir.mkdir(parents=True)
            (dialogs_dir / "000000.json").write_text("{}", encoding="utf-8")
            (out_dir / "stats.json").write_text(
                json.dumps(
                    {
                        "source_data_path": "old/source.json",
                        "mcp_servers_dir": "mcp",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "different source"):
                generate_dataset._validate_output_dir_for_write(
                    out_dir,
                    {"source_data_path": "new/source.json", "mcp_servers_dir": "mcp"},
                    resume=True,
                )

    def test_output_dir_rejects_resume_without_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            dialogs_dir = out_dir / "dialogs"
            dialogs_dir.mkdir(parents=True)
            (dialogs_dir / "000000.json").write_text("{}", encoding="utf-8")
            (out_dir / "stats.json").write_text(json.dumps({"accepted_total": 1}), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "source_data_path"):
                generate_dataset._validate_output_dir_for_write(
                    out_dir,
                    {"source_data_path": "source.json", "mcp_servers_dir": "mcp"},
                    resume=True,
                )

    def test_output_dir_rejects_resume_with_different_mcp_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            dialogs_dir = out_dir / "dialogs"
            dialogs_dir.mkdir(parents=True)
            (dialogs_dir / "000000.json").write_text("{}", encoding="utf-8")
            (out_dir / "stats.json").write_text(
                json.dumps(
                    {
                        "source_data_path": "source.json",
                        "mcp_servers_dir": "old-mcp",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "different MCP server directory"):
                generate_dataset._validate_output_dir_for_write(
                    out_dir,
                    {"source_data_path": "source.json", "mcp_servers_dir": "new-mcp"},
                    resume=True,
                )

    def test_output_dir_requires_resume_for_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            dialogs_dir = out_dir / "dialogs"
            dialogs_dir.mkdir(parents=True)
            (dialogs_dir / "000000.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "--resume"):
                generate_dataset._validate_output_dir_for_write(
                    out_dir,
                    {"source_data_path": "source.json"},
                    resume=False,
                )

    def test_source_metadata_understands_new_paper_data_paths(self):
        path = Path(
            "../data/task_tool/single_mcp_server/02_two_tools/ASTRA/03_grouped_filtered/test_curated_grouped.json"
        )

        metadata = generate_dataset._source_metadata(path, Path("../data/mcp_servers/ASTRA"))

        self.assertEqual(metadata["source_split"], "test")
        self.assertEqual(metadata["tool_count"], "02_tools")
        self.assertEqual(metadata["benchmark"], "ASTRA")

    def test_source_metadata_understands_cross_paper_data_paths(self):
        path = Path(
            "../data/task_tool/cross_mcp_server/03_three_tools/ASTRA/03_grouped_filtered/validation_curated_grouped.json"
        )

        metadata = generate_dataset._source_metadata(path, Path("../data/mcp_servers/ASTRA"))

        self.assertEqual(metadata["source_split"], "validation")
        self.assertEqual(metadata["tool_count"], "03_tools_cross")
        self.assertEqual(metadata["benchmark"], "ASTRA")

    def test_accept_dialog_requires_negative_assistant_turn_index(self):
        dialog = {
            "notes": {
                "label": "malicious_irrelevant_tool_call",
                "relevant_tool_calls": 1,
                "irrelevant_tool_calls": 1,
            }
        }

        self.assertFalse(generate_dataset.accept_dialog(dialog, target_negative=True))
        dialog["notes"]["assistant_turn_index"] = 1
        self.assertTrue(generate_dataset.accept_dialog(dialog, target_negative=True))

    def test_negative_episode_exposes_relevant_tools_plus_one_irrelevant_target(self):
        relevant_schemas = [tool_schema("relevant_a"), tool_schema("relevant_b")]
        irrelevant_schemas = [
            tool_schema("irrelevant_a"),
            tool_schema("irrelevant_b"),
            tool_schema("irrelevant_c"),
        ]
        task = generate_dataset.TaskSpec(
            name="task_000123",
            user_role="user",
            assistant_role="assistant",
            user_query="Do the grouped task",
            relevant_tool_schemas=relevant_schemas,
            irrelevant_tool_schemas=irrelevant_schemas,
            group_id="mixed-group",
        )
        captured_bound_tools = []

        class FakeLLM:
            def bind_tools(self, schemas):
                captured_bound_tools.extend(generate_dataset.schema_names(schemas))
                return self

        def fake_invoke(llm, messages, max_retries=3):
            system_prompt = messages[0].content
            target = system_prompt.split("Your hidden target tool is: ", 1)[1].split("\n", 1)[0]
            return type(
                "Response",
                (),
                {
                    "content": "",
                    "tool_calls": [{"name": target, "args": {}, "id": "call-irrelevant"}],
                },
            )()

        with mock.patch.object(generate_dataset, "build_llm", return_value=FakeLLM()):
            with mock.patch.object(generate_dataset, "_invoke_with_retry", side_effect=fake_invoke):
                dialog = generate_dataset.run_episode(
                    task=task,
                    force_negative=True,
                    episode_seed=5,
                    model_name="mock-model",
                )

        payload = generate_dataset._dialog_payload(dialog)
        self.assertIn("agentic_conversation", dialog)
        self.assertNotIn("scenario", dialog)

        target = payload["scenario"]["target_irrelevant_tool"]
        self.assertIn(target, ["irrelevant_a", "irrelevant_b", "irrelevant_c"])
        self.assertCountEqual(
            payload["scenario"]["available_tools"],
            ["relevant_a", "relevant_b", target],
        )
        self.assertCountEqual(
            captured_bound_tools,
            ["relevant_a", "relevant_b", target],
        )
        self.assertCountEqual(payload["scenario"]["relevant_tools"], ["relevant_a", "relevant_b"])
        self.assertEqual(payload["scenario"]["irrelevant_tools"], [target])
        self.assertEqual(payload["notes"]["label"], "malicious_irrelevant_tool_call")
        self.assertEqual(payload["notes"]["tool_call_trace"][0]["relevance"], "irrelevant")

    def test_invoke_with_retry_treats_timed_out_message_as_transient(self):
        class FlakyLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    raise Exception("Request timed out.")
                return "ok"

        llm = FlakyLLM()

        with mock.patch.object(generate_dataset.time, "sleep"):
            result = generate_dataset._invoke_with_retry(llm, [], max_retries=1)

        self.assertEqual(result, "ok")
        self.assertEqual(llm.calls, 2)

    def test_validate_task_dataset_reports_resolvable_new_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "test_curated_grouped.json"
            mcp_dir = Path(tmp) / "mcp"
            mcp_dir.mkdir()
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "group_id": "clean",
                            "task": {
                                "task": "clean task",
                                "metadata": {"task": {"seed_tools": [{"name": "truth", "mcp_server": "server"}]}},
                            },
                            "requests": [
                                {
                                    "request": {"tool": {"name": "truth"}},
                                    "label": {"relevant": True},
                                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                                }
                            ],
                        },
                        {
                            "group_id": "negative",
                            "task": {
                                "task": "negative task",
                                "metadata": {"task": {"seed_tools": [{"name": "truth", "mcp_server": "server"}]}},
                            },
                            "requests": [
                                {
                                    "request": {"tool": {"name": "irrelevant"}},
                                    "label": {"relevant": False},
                                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                                }
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (mcp_dir / "server.json").write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "name": "truth",
                                "description": "truth",
                                "inputSchema": {"type": "object", "properties": {}},
                            },
                            {
                                "name": "irrelevant",
                                "description": "irrelevant",
                                "inputSchema": {"type": "object", "properties": {}},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = generate_dataset.validate_task_dataset(data_path, mcp_dir)

            self.assertEqual(report["total_entries"], 2)
            self.assertEqual(report["clean_entries"], 1)
            self.assertEqual(report["negative_entries"], 1)
            self.assertEqual(report["missing_relevant_schema_count"], 0)
            self.assertEqual(report["missing_irrelevant_schema_count"], 0)

    def test_build_dataset_writes_grouped_outputs_with_mocked_episode_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "test_curated_grouped.json"
            mcp_dir = tmp_path / "mcp"
            out_dir = tmp_path / "out"
            mcp_dir.mkdir()
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "group_id": "clean-group",
                            "task": {
                                "task": "clean task",
                                "metadata": {"task": {"seed_tools": [{"name": "truth", "mcp_server": "server"}]}},
                            },
                            "requests": [
                                {
                                    "id": "clean-request",
                                    "request": {"tool": {"name": "truth"}},
                                    "label": {"relevant": True},
                                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                                }
                            ],
                        },
                        {
                            "group_id": "negative-group",
                            "task": {
                                "task": "negative task",
                                "metadata": {"task": {"seed_tools": [{"name": "truth", "mcp_server": "server"}]}},
                            },
                            "requests": [
                                {
                                    "id": "negative-request",
                                    "request": {"tool": {"name": "irrelevant"}},
                                    "label": {"relevant": False},
                                    "metadata": {"request": {"tool": {"mcp_server": "server"}}},
                                }
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (mcp_dir / "server.json").write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "name": "truth",
                                "description": "truth",
                                "inputSchema": {"type": "object", "properties": {}},
                            },
                            {
                                "name": "irrelevant",
                                "description": "irrelevant",
                                "inputSchema": {"type": "object", "properties": {}},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_episode(task, force_negative, episode_seed, model_name, use_fallback=False):
                relevant_names = generate_dataset.schema_names(task.relevant_tool_schemas)
                irrelevant_names = generate_dataset.schema_names(task.irrelevant_tool_schemas)
                label = "malicious_irrelevant_tool_call" if force_negative else "clean"
                return {
                    "seed": episode_seed,
                    "complete": True,
                    "scenario": {
                        "task_name": task.name,
                        "group_id": task.group_id,
                        "source_request_ids": task.source_request_ids or [],
                        "user_query": task.user_query,
                        "available_tools": relevant_names + irrelevant_names[:1],
                        "relevant_tools": relevant_names,
                        "irrelevant_tools": irrelevant_names[:1] if force_negative else [],
                        "negative_mode": force_negative,
                        "target_irrelevant_tool": irrelevant_names[0] if force_negative else None,
                    },
                    "turns": [
                        {"speaker": "USER", "text": task.user_query},
                        {"speaker": "AGENT", "text": "done"},
                    ],
                    "events": [],
                    "notes": {
                        "label": label,
                        "stop_reason": "mocked",
                        "task_name": task.name,
                        "relevant_tool_calls": 0 if force_negative else 1,
                        "irrelevant_tool_calls": 1 if force_negative else 0,
                        "target_irrelevant_tool": irrelevant_names[0] if force_negative else None,
                        "assistant_turn_index": 1 if force_negative else None,
                        "tool_call_trace": [],
                    },
                }

            with contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(generate_dataset, "run_episode", side_effect=fake_run_episode):
                    generate_dataset.build_dataset(
                        out_dir=out_dir,
                        seed=13,
                        max_attempts_per_example=1,
                        model_name="mock-model",
                        task_data_path=data_path,
                        mcp_servers_dir=mcp_dir,
                    )

            dialogs = sorted((out_dir / "dialogs").glob("*.json"))
            manifest_rows = [
                json.loads(line) for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))

            self.assertEqual(len(dialogs), 2)
            written_dialog = json.loads(dialogs[0].read_text(encoding="utf-8"))
            self.assertEqual(list(written_dialog), ["agentic_conversation"])
            self.assertIn("scenario", written_dialog["agentic_conversation"])
            self.assertNotIn("scenario", written_dialog)
            self.assertEqual(len(manifest_rows), 2)
            self.assertEqual(stats["accepted_total"], 2)
            self.assertEqual(stats["source_data_file"], "test_curated_grouped.json")
            self.assertEqual(
                sorted(row["group_id"] for row in manifest_rows),
                ["clean-group", "negative-group"],
            )

    def test_parse_args_uses_task_data_only(self):
        parser = generate_dataset.build_arg_parser()

        args = parser.parse_args(
            [
                "--out",
                "/tmp/out",
                "--task-data",
                "../data/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped.json",
                "--mcp-servers-dir",
                "../data/mcp_servers/ASTRA",
            ]
        )

        self.assertEqual(args.task_data.name, "test_curated_grouped.json")
        self.assertFalse(args.allow_legacy_astra)

        help_text = parser.format_help()
        self.assertIn("--task-data", help_text)
        self.assertNotIn("--astra-data", help_text)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--out",
                        "/tmp/out",
                        "--astra-data",
                        "../data/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped.json",
                        "--mcp-servers-dir",
                        "../data/mcp_servers/ASTRA",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
