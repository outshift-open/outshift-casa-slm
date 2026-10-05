"""Offline tests for paired GRPO dataset construction."""

import json
from pathlib import Path

import pytest

from casa_slm.model_finetuning import build_paired_balanced_dataset as builder


def _example(example_id: str, task: str, relevant: bool) -> dict:
    """Create a minimal curated task-tool example."""
    return {
        "id": example_id,
        "request": {"task": task, "tool": {"name": example_id, "description": "Test tool"}},
        "label": {"relevant": relevant},
        "metadata": {},
    }


def test_pairs_and_supplements_are_balanced_and_deterministic(tmp_path: Path) -> None:
    """Pair opposite labels per task, then balance unmatched rows."""
    source = tmp_path / "train.json"
    examples = [
        _example("a-pos", "Task A", True),
        _example("a-neg", "Task A", False),
        _example("b-pos", "Task B", True),
        _example("c-neg", "Task C", False),
        _example("d-pos", "Task D", True),
    ]
    source.write_text(json.dumps(examples), encoding="utf-8")
    loaded = builder._load_examples([source], group_key="task", group_scope="source")
    rows, summary = builder._build_dataset(
        loaded,
        seed=3407,
        group_scope="source",
        supplement_mode="balanced",
        supplement_strategy="source",
        max_supplement_per_class=None,
        shuffle_output=True,
    )
    repeated, repeated_summary = builder._build_dataset(
        loaded,
        seed=3407,
        group_scope="source",
        supplement_mode="balanced",
        supplement_strategy="source",
        max_supplement_per_class=None,
        shuffle_output=True,
    )

    assert (rows, summary) == (repeated, repeated_summary)
    assert summary["matched_pairs"] == 1
    assert summary["output_counts"] == {"positive": 2, "negative": 2, "total": 4}
    pair = [row for row in rows if row["metadata"][builder.BUILDER_METADATA_KEY]["status"] == "matched_pair"]
    assert {row["id"] for row in pair} == {"a-pos", "a-neg"}
    assert len({row["metadata"][builder.BUILDER_METADATA_KEY]["pair_id"] for row in pair}) == 1
    assert [row["id"] for row in examples] == ["a-pos", "a-neg", "b-pos", "c-neg", "d-pos"]
    assert all(builder.BUILDER_METADATA_KEY not in row["metadata"] for row in examples)


def test_source_provenance_is_relative_to_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the working directory does not change source metadata."""
    source = tmp_path / "data" / "train.json"
    source.parent.mkdir()
    source.write_text(json.dumps([_example("x", "Task X", True)]), encoding="utf-8")
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(source.parent)
    loaded = builder._load_examples([source], group_key="task", group_scope="source")
    assert loaded[0].source_file == "data/train.json"


def test_builder_rejects_missing_input_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report missing source datasets without creating partial outputs."""
    output = tmp_path / "paired.json"
    parse_args = builder.parse_args
    monkeypatch.setattr(
        builder,
        "parse_args",
        lambda: parse_args(["--input", str(tmp_path / "missing.json"), "--output", str(output)]),
    )
    with pytest.raises(FileNotFoundError, match="Missing curated training inputs"):
        builder.main()
    assert not output.exists()


def test_default_cli_paths_work_outside_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build default inputs and outputs in the data directory regardless of cwd."""
    data_dir = tmp_path / "data" / "optimization_and_finetuning"
    data_dir.mkdir(parents=True)
    examples = [_example("a-pos", "Task A", True), _example("a-neg", "Task A", False)]
    for name in builder.DEFAULT_TRAIN_PATHS:
        (data_dir / name).write_text(json.dumps(examples), encoding="utf-8")
    monkeypatch.setattr(builder, "DATA_DIR", data_dir)
    monkeypatch.setattr(builder, "REPOSITORY_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    parse_args = builder.parse_args
    monkeypatch.setattr(builder, "parse_args", lambda: parse_args([]))

    builder.main()

    rows = json.loads((data_dir / "paired_balanced_train_seed3407.json").read_text(encoding="utf-8"))
    summary = json.loads((data_dir / "paired_balanced_train_seed3407.summary.json").read_text(encoding="utf-8"))
    assert summary["matched_pairs"] == len(builder.DEFAULT_TRAIN_PATHS)
    assert len(rows) == 2 * len(builder.DEFAULT_TRAIN_PATHS)
    assert summary["output_counts"]["positive"] == summary["output_counts"]["negative"]
    assert not (elsewhere / "data" / "paired_balanced_train_seed3407.json").exists()
