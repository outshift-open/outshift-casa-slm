"""Regroup curated per-tool rows into task-level multi-tool examples."""

import argparse
import json
from pathlib import Path
from typing import Any

MATCH_TAG_BY_LABEL_TYPE = {
    "relevant_seed": "correct",
    "irrelevant": "wrong",
    "irrelevant_external": "null",
}


def regroup_records(
    rows: list[dict[str, Any]],
    *,
    expected_tool_count: int | None = None,
) -> list[dict[str, Any]]:
    """Regroup curated rows by their shared task group ID.

    Args:
        rows: Curated per-tool rows loaded from one JSON file.
        expected_tool_count: Required number of retained rows per group. When
            set, groups with fewer or more rows are dropped.

    Returns:
        Task-level records where each group contains the retained tools for the
        shared task and match type.
    """
    groups: dict[str, dict[str, Any]] = {}

    for row in rows:
        group_id = row["metadata"]["request"]["task"]["group_id"]
        label_type = row["metadata"]["label"]["type"]
        match_tag = _match_tag(label_type)
        task = row["request"]["task"]
        input_tool = row["request"]["tool"]["name"]
        input_server = row["metadata"]["request"]["tool"]["mcp_server"]
        seed_tool = row["metadata"]["request"]["task"]["seed_tool"]["name"]
        seed_server = row["metadata"]["request"]["task"]["seed_tool"]["mcp_server"]

        if group_id not in groups:
            groups[group_id] = {
                "input": {
                    "task": task,
                    "tools": [],
                    "mcp_servers": [],
                },
                "groundtruth": {
                    "tools": [],
                    "mcp_servers": [],
                },
                "match_tag": match_tag,
                "group_id": group_id,
            }
        else:
            _validate_same_group(groups[group_id], task, match_tag, group_id)

        groups[group_id]["input"]["tools"].append(input_tool)
        groups[group_id]["input"]["mcp_servers"].append(input_server)
        groups[group_id]["groundtruth"]["tools"].append(seed_tool)
        groups[group_id]["groundtruth"]["mcp_servers"].append(seed_server)

    records = list(groups.values())
    if expected_tool_count is None:
        return records
    return [record for record in records if len(record["input"]["tools"]) == expected_tool_count]


def process_file(input_path: Path, output_path: Path) -> int:
    """Regroup one curated JSON file and write it to the output path.

    Args:
        input_path: Source curated JSON file.
        output_path: Destination regrouped JSON file.

    Returns:
        Number of regrouped records written.
    """
    with input_path.open(encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        msg = f"Expected a JSON array in {input_path}"
        raise ValueError(msg)

    regrouped = regroup_records(rows, expected_tool_count=_expected_tool_count(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(regrouped, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return len(regrouped)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/curated"),
        help="Directory containing the four top-level curated JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/curated_regrouped"),
        help="Directory where regrouped JSON files are written.",
    )
    return parser


def main() -> None:
    """Run the curated-data regrouping script."""
    args = build_parser().parse_args()
    input_files = sorted(args.input_dir.glob("*_curated.json"))
    if not input_files:
        msg = f"No *_curated.json files found in {args.input_dir}"
        raise SystemExit(msg)

    for input_path in input_files:
        output_path = args.output_dir / input_path.name
        count = process_file(input_path, output_path)
        print(f"Wrote {count} records to {output_path}")


def _match_tag(label_type: str) -> str:
    try:
        return MATCH_TAG_BY_LABEL_TYPE[label_type]
    except KeyError as exc:
        msg = f"Unknown curated label type: {label_type}"
        raise ValueError(msg) from exc


def _expected_tool_count(input_path: Path) -> int:
    if "two_tools" in input_path.name:
        return 2
    if "three_tools" in input_path.name:
        return 3
    msg = f"Cannot infer expected tool count from filename: {input_path.name}"
    raise ValueError(msg)


def _validate_same_group(record: dict[str, Any], task: str, match_tag: str, group_id: str) -> None:
    if record["input"]["task"] != task:
        msg = f"Group {group_id} contains multiple task texts"
        raise ValueError(msg)
    if record["match_tag"] != match_tag:
        msg = f"Group {group_id} contains multiple match tags"
        raise ValueError(msg)


if __name__ == "__main__":
    main()
