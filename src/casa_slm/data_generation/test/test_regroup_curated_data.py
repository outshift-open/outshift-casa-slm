# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for regrouping curated per-tool rows into task-level examples."""

from casa_slm.data_generation import regroup_curated_data as rcd


def _row(
    *,
    group_id: str,
    label_type: str,
    task: str,
    input_tool: str,
    input_server: str,
    seed_tool: str,
    seed_server: str,
) -> dict:
    return {
        "request": {
            "task": task,
            "tool": {"name": input_tool},
        },
        "metadata": {
            "label": {"type": label_type},
            "request": {
                "task": {
                    "group_id": group_id,
                    "seed_tool": {"name": seed_tool, "mcp_server": seed_server},
                },
                "tool": {"mcp_server": input_server},
            },
        },
    }


def test_regroups_rows_by_group_id_and_preserves_input_order():
    rows = [
        _row(
            group_id="group-1",
            label_type="irrelevant",
            task="Review the failed CI run and fetch the paper PDF",
            input_tool="search_orgs",
            input_server="github-official",
            seed_tool="rerun_failed_jobs",
            seed_server="github-official",
        ),
        _row(
            group_id="group-1",
            label_type="irrelevant",
            task="Review the failed CI run and fetch the paper PDF",
            input_tool="download_medrxiv",
            input_server="paper-search",
            seed_tool="download_iacr",
            seed_server="paper-search",
        ),
    ]

    regrouped = rcd.regroup_records(rows)

    assert regrouped == [
        {
            "input": {
                "task": "Review the failed CI run and fetch the paper PDF",
                "tools": ["search_orgs", "download_medrxiv"],
                "mcp_servers": ["github-official", "paper-search"],
            },
            "groundtruth": {
                "tools": ["rerun_failed_jobs", "download_iacr"],
                "mcp_servers": ["github-official", "paper-search"],
            },
            "match_tag": "wrong",
            "group_id": "group-1",
        }
    ]


def test_maps_relevant_and_external_labels_to_match_tags():
    rows = [
        _row(
            group_id="correct-group",
            label_type="relevant_seed",
            task="List stock statistic fields",
            input_tool="list_stock_stat_fields",
            input_server="nasdaq-data-link",
            seed_tool="list_stock_stat_fields",
            seed_server="nasdaq-data-link",
        ),
        _row(
            group_id="null-group",
            label_type="irrelevant_external",
            task="Assign a GitHub issue and fetch a DOI",
            input_tool="get_recent_price_history",
            input_server="nasdaq-data-link",
            seed_tool="assign_copilot_to_issue",
            seed_server="github-official",
        ),
    ]

    regrouped = rcd.regroup_records(rows)

    assert [record["match_tag"] for record in regrouped] == ["correct", "null"]


def test_drops_groups_that_do_not_have_the_expected_tool_count():
    rows = [
        _row(
            group_id="complete-group",
            label_type="relevant_seed",
            task="Fetch a paper and inspect a GitHub issue",
            input_tool="get_crossref_paper_by_doi",
            input_server="paper-search",
            seed_tool="get_crossref_paper_by_doi",
            seed_server="paper-search",
        ),
        _row(
            group_id="complete-group",
            label_type="relevant_seed",
            task="Fetch a paper and inspect a GitHub issue",
            input_tool="get_issue",
            input_server="github-official",
            seed_tool="get_issue",
            seed_server="github-official",
        ),
        _row(
            group_id="incomplete-group",
            label_type="relevant_seed",
            task="Only one curated row survived filtering",
            input_tool="get_crossref_paper_by_doi",
            input_server="paper-search",
            seed_tool="get_crossref_paper_by_doi",
            seed_server="paper-search",
        ),
    ]

    regrouped = rcd.regroup_records(rows, expected_tool_count=2)

    assert [record["group_id"] for record in regrouped] == ["complete-group"]
