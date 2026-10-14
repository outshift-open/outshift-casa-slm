# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dataset loading."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from casa_slm.prompt_optimizer.config import EvaluationConfig
from casa_slm.prompt_optimizer.dataset import load_dataset

type Row = dict[str, object]


def _write_json(path: Path, rows: list[Row]) -> None:
    """Write JSON records to disk for dataset loading tests."""
    path.write_text(json.dumps(rows), encoding="utf-8")


def _row(prompt: str, relevant: bool) -> Row:
    """Build one labeled dataset row."""
    return {"prompt": prompt, "label": {"relevant": relevant}}


def _data_config(
    tmp_path: Path,
    name: str,
    limits: list[int],
    *,
    train: list[Row],
    validation: list[Row],
    test: list[Row],
) -> dict[str, object]:
    """Write one named dataset and return its configuration."""
    paths: dict[str, Path] = {}
    for split, rows in (("train", train), ("validation", validation), ("test", test)):
        paths[split] = tmp_path / f"{name}_{split}.json"
        _write_json(paths[split], rows)

    return {
        "name": name,
        "train": [str(paths["train"])],
        "validation": [str(paths["validation"])],
        "test": [str(paths["test"])],
        "limits": limits,
    }


def _evaluation_config(datasets: list[dict[str, object]], *, shuffle: bool = False) -> EvaluationConfig:
    """Build a minimal evaluation config for dataset loading tests."""
    return EvaluationConfig.model_validate(
        {
            "experiment": {
                "name": "sample",
                "shuffle": shuffle,
                "dataset": datasets,
            },
            "job": {},
            "models": {
                "optimizer_model": {"name": "optimizer", "end_point": "azure/optimizer"},
                "task_model": {
                    "name": "task",
                    "hf_model_id": "org/task-model",
                    "path": "models/task-model",
                },
            },
            "vllm": {},
        }
    )


def _default_data_config(tmp_path: Path, limits: list[int]) -> dict[str, object]:
    """Write the default balanced dataset used by loader tests."""
    return _data_config(
        tmp_path,
        "demo",
        limits,
        train=[_row("a", True), _row("b", False)],
        validation=[_row("c", True), _row("d", False)],
        test=[_row("e", True), _row("f", False), _row("g", True), _row("h", False)],
    )


def test_load_dataset_rejects_limit_larger_than_split_size(tmp_path: Path) -> None:
    """Reject limits that exceed the number of rows in a split."""
    evaluation_config = _evaluation_config([_default_data_config(tmp_path, [3, 2, 4])])

    with pytest.raises(ValueError, match=r"Limit for split 'train'.*exceeds available rows"):
        load_dataset(evaluation_config)


def test_load_dataset_allows_limit_equal_to_split_size(tmp_path: Path) -> None:
    """Allow limits that match the number of rows in a split exactly."""
    evaluation_config = _evaluation_config([_default_data_config(tmp_path, [2, 2, 4])])

    dataset = load_dataset(evaluation_config)

    assert len(dataset["train"]) == 2
    assert len(dataset["validation"]) == 2
    assert len(dataset["test"]) == 4


def test_load_dataset_uses_experiment_shuffle_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the experiment shuffle flag when no explicit override is provided."""
    evaluation_config = _evaluation_config([_default_data_config(tmp_path, [0, 0, 0])], shuffle=True)

    sample_calls: list[float] = []

    def fake_sample(self, *, frac: float) -> object:
        sample_calls.append(frac)
        return self.iloc[::-1]

    monkeypatch.setattr("pandas.core.frame.DataFrame.sample", fake_sample)

    dataset = load_dataset(evaluation_config)

    assert sample_calls == [1, 1, 1]
    assert dataset["train"]["prompt"].tolist() == ["b", "a"]
    assert dataset["validation"]["prompt"].tolist() == ["d", "c"]
    assert dataset["test"]["prompt"].tolist() == ["h", "g", "f", "e"]


def test_load_dataset_returns_exact_balanced_limit_without_shuffling(tmp_path: Path) -> None:
    """Select the exact balanced limit from the full split in source order."""
    data_config = _data_config(
        tmp_path,
        "demo",
        [4, 0, 0],
        train=[_row("a", True), _row("b", True), _row("c", True), _row("d", False), _row("e", False)],
        validation=[_row("f", True), _row("g", False)],
        test=[_row("h", True), _row("i", False)],
    )

    dataset = load_dataset(_evaluation_config([data_config]))

    assert dataset["train"]["prompt"].tolist() == ["a", "b", "d", "e"]
    assert dataset["train"]["label"].map(lambda label: label["relevant"]).value_counts().to_dict() == {
        False: 2,
        True: 2,
    }


def test_load_dataset_balances_across_all_files_in_named_dataset(tmp_path: Path) -> None:
    """Pool a named dataset's files before applying the balanced limit."""
    data_config = _data_config(
        tmp_path,
        "demo",
        [4, 0, 0],
        train=[_row("a", True), _row("b", True), _row("c", True)],
        validation=[_row("f", True), _row("g", False)],
        test=[_row("h", True), _row("i", False)],
    )
    second_train_path = tmp_path / "demo_train_second.json"
    _write_json(second_train_path, [_row("d", False), _row("e", False)])
    train_paths = data_config["train"]
    assert isinstance(train_paths, list)
    train_paths.append(str(second_train_path))

    dataset = load_dataset(_evaluation_config([data_config]))

    assert dataset["train"]["prompt"].tolist() == ["a", "b", "d", "e"]
    assert dataset["train"]["prompt"].is_unique


def test_load_dataset_returns_exact_balanced_limit_after_shuffling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shuffle once and select the exact balanced limit in shuffled order."""
    data_config = _data_config(
        tmp_path,
        "demo",
        [4, 0, 0],
        train=[_row("a", True), _row("b", True), _row("c", False), _row("d", False), _row("e", False)],
        validation=[_row("f", True), _row("g", False)],
        test=[_row("h", True), _row("i", False)],
    )

    def reverse_sample(self: pd.DataFrame, *, frac: float) -> pd.DataFrame:
        assert frac == 1
        return self.iloc[::-1]

    monkeypatch.setattr("pandas.core.frame.DataFrame.sample", reverse_sample)

    dataset = load_dataset(_evaluation_config([data_config], shuffle=True))

    assert dataset["train"]["label"].map(lambda label: label["relevant"]).value_counts().to_dict() == {
        False: 2,
        True: 2,
    }
    assert dataset["train"]["prompt"].tolist() == ["e", "d", "b", "a"]


def test_load_dataset_rejects_limits_across_multiple_named_datasets(tmp_path: Path) -> None:
    """Reject per-entry limits that do not define one combined target."""
    first = _data_config(
        tmp_path,
        "first",
        [2, 0, 0],
        train=[_row("a", True), _row("b", True)],
        validation=[_row("c", True), _row("d", False)],
        test=[_row("e", True), _row("f", False)],
    )
    second = _data_config(
        tmp_path,
        "second",
        [2, 0, 0],
        train=[_row("g", False), _row("h", False)],
        validation=[_row("i", True), _row("j", False)],
        test=[_row("k", True), _row("l", False)],
    )

    with pytest.raises(ValidationError, match="Nonzero dataset limits require exactly one named dataset"):
        _evaluation_config([first, second])


def test_load_dataset_assigns_odd_extra_quota_to_first_label(tmp_path: Path) -> None:
    """Give the first encountered label the extra row for an odd limit."""
    data_config = _data_config(
        tmp_path,
        "demo",
        [3, 0, 0],
        train=[_row("a", True), _row("b", False), _row("c", True)],
        validation=[_row("d", True), _row("e", False)],
        test=[_row("f", True), _row("g", False)],
    )

    dataset = load_dataset(_evaluation_config([data_config]))

    assert dataset["train"]["prompt"].tolist() == ["a", "b", "c"]


def test_load_dataset_rejects_even_class_shortage(tmp_path: Path) -> None:
    """Reject a limit when one label cannot satisfy its exact quota."""
    data_config = _data_config(
        tmp_path,
        "demo",
        [4, 0, 0],
        train=[_row("a", True), _row("b", True), _row("c", True), _row("d", False)],
        validation=[_row("e", True), _row("f", False)],
        test=[_row("g", True), _row("h", False)],
    )

    with pytest.raises(ValueError, match=r"label False requires 2 rows but only 1"):
        load_dataset(_evaluation_config([data_config]))


def test_load_dataset_rejects_limited_split_when_one_class_is_absent(tmp_path: Path) -> None:
    """Reject a balanced limit when one required label is absent."""
    data_config = _data_config(
        tmp_path,
        "demo",
        [2, 0, 0],
        train=[_row("a", True), _row("b", True)],
        validation=[_row("c", True), _row("d", False)],
        test=[_row("e", True), _row("f", False)],
    )

    with pytest.raises(ValueError, match=r"label False requires 1 rows but only 0"):
        load_dataset(_evaluation_config([data_config]))


def test_load_dataset_combines_multiple_unlimited_named_datasets(tmp_path: Path) -> None:
    """Keep every row when multiple named datasets are all unlimited."""
    first = _data_config(
        tmp_path,
        "first",
        [0, 0, 0],
        train=[_row("a", True), _row("b", True)],
        validation=[_row("c", True), _row("d", False)],
        test=[_row("e", True), _row("f", False)],
    )
    second = _data_config(
        tmp_path,
        "unlimited",
        [0, 0, 0],
        train=[_row("g", False), _row("h", False), _row("i", False), _row("j", False)],
        validation=[_row("k", True), _row("l", False)],
        test=[_row("m", True), _row("n", False)],
    )

    dataset = load_dataset(_evaluation_config([first, second]))

    assert dataset["train"]["prompt"].tolist() == ["a", "b", "g", "h", "i", "j"]


@pytest.mark.parametrize(
    "invalid_row",
    [
        {"prompt": "a"},
        {"prompt": "a", "label": {}},
        {"prompt": "a", "label": {"relevant": 1}},
    ],
)
def test_load_dataset_rejects_invalid_relevance_labels(tmp_path: Path, invalid_row: Row) -> None:
    """Reject rows without a Boolean label.relevant value when balancing."""
    valid_schema_rows: list[Row] = [{"prompt": "b"}] if "label" not in invalid_row else [_row("b", True)]
    data_config = _data_config(
        tmp_path,
        "demo",
        [1, 0, 0],
        train=[invalid_row],
        validation=valid_schema_rows,
        test=valid_schema_rows,
    )

    with pytest.raises(ValueError, match=r"Cannot balance split 'train'"):
        load_dataset(_evaluation_config([data_config]))


def test_load_dataset_leaves_unlimited_unlabeled_splits_unchanged(tmp_path: Path) -> None:
    """Do not require relevance labels when no split limit activates balancing."""
    data_config = _data_config(
        tmp_path,
        "demo",
        [0, 0, 0],
        train=[{"prompt": "a"}, {"prompt": "b"}],
        validation=[{"prompt": "c"}],
        test=[{"prompt": "d"}],
    )

    dataset = load_dataset(_evaluation_config([data_config]))

    assert dataset["train"]["prompt"].tolist() == ["a", "b"]
