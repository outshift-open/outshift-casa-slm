"""Tests for dataset validation and token-count reporting."""

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

import casa_slm.prompt_optimizer.scripts.check_dataset as check_dataset
from casa_slm.prompt_optimizer.config import EvaluationConfig


class FakeTokenizer:
    """Return configured token counts while recording formatted prompts."""

    def __init__(self, counts_by_task: dict[str, int]) -> None:
        """Initialize task-to-count mappings and prompt call records."""
        self.counts_by_task = counts_by_task
        self.calls: list[tuple[list[dict[str, Any]], dict[str, bool]]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        """Return a deterministic token sequence for the prompt's task field."""
        self.calls.append(
            (
                conversation,
                {
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "return_dict": return_dict,
                },
            )
        )
        user_content = conversation[-1]["content"]
        assert isinstance(user_content, str)
        for task, count in self.counts_by_task.items():
            if f"[[ ## task ## ]]\n{task}\n" in user_content:
                return list(range(count))
        raise ValueError("No fake token count configured for prompt task.")


def _row(task: str, *, valid: bool = True) -> dict[str, object]:
    """Build one task-tool dataset row."""
    row: dict[str, object] = {
        "request": {
            "task": task,
            "tool": {"name": "issue_tracker", "description": "Create an issue"},
        },
        "label": {"relevant": True},
        "metadata": {"label": {"reasoning": f"secret gold reasoning for {task}"}},
    }
    if not valid:
        row.pop("request")
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dataset rows as JSON records."""
    path.write_text(json.dumps(rows), encoding="utf-8")


def _config(tmp_path: Path, split_paths: dict[str, list[Path]]) -> EvaluationConfig:
    """Build a complete checker configuration for temporary dataset files."""
    return EvaluationConfig.model_validate(
        {
            "experiment": {
                "name": "checker-test",
                "dataset": [
                    {
                        "name": "combined",
                        "train": [str(path) for path in split_paths["train"]],
                        "validation": [str(path) for path in split_paths["validation"]],
                        "test": [str(path) for path in split_paths["test"]],
                        "limits": [1, 1, 1],
                    }
                ],
            },
            "job": {},
            "models": {
                "optimizer_model": {"name": "optimizer", "end_point": "azure/optimizer"},
                "task_model": {
                    "name": "baseline",
                    "hf_model_id": "org/task-model",
                    "path": tmp_path / "models",
                },
            },
            "vllm": {},
            "HF_TOKEN": "test-token",
        }
    )


def test_prompt_token_counter_formats_only_task_model_inputs() -> None:
    """Count the DSPy chat prompt without exposing gold dataset fields."""
    tokenizer = FakeTokenizer({"create issue": 17})
    counter = check_dataset.PromptTokenCounter(tokenizer)
    example = check_dataset.dataframe_to_examples(pd.DataFrame([_row("create issue")]))[0]

    count = counter.count(example)

    assert count == 17
    messages, options = tokenizer.calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "create issue" in messages[-1]["content"]
    assert "issue_tracker" in messages[-1]["content"]
    assert "secret gold reasoning" not in str(messages)
    assert options == {"tokenize": True, "add_generation_prompt": True, "return_dict": False}


def test_check_file_collects_counts_and_reports_bad_rows(tmp_path: Path) -> None:
    """Continue after malformed records while counting every valid row."""
    dataset_path = tmp_path / "mixed.json"
    _write_rows(dataset_path, [_row("first"), _row("broken", valid=False), _row("third")])
    counter = check_dataset.PromptTokenCounter(FakeTokenizer({"first": 10, "third": 30}))

    result = check_dataset.check_file(dataset_path, counter)

    assert result.total_rows == 3
    assert result.token_counts == [10, 30]
    assert len(result.errors) == 1
    assert "[RECORD ERROR] row position 1" in result.errors[0]


def test_check_file_reports_load_errors(tmp_path: Path) -> None:
    """Return a load error and no statistics for unreadable JSON."""
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text("not json", encoding="utf-8")

    result = check_dataset.check_file(dataset_path, None)

    assert result.total_rows == 0
    assert result.token_counts == []
    assert len(result.errors) == 1
    assert result.errors[0].startswith("  [LOAD ERROR]")


def test_calculate_token_statistics_uses_observed_higher_values() -> None:
    """Return observed integer lengths for both requested percentiles."""
    statistics = check_dataset.calculate_token_statistics([10, 20, 30, 40])

    assert statistics == check_dataset.TokenStatistics(p50=30, p95=40, maximum=40)


def test_calculate_token_statistics_rejects_empty_input() -> None:
    """Reject an empty population instead of producing NaN statistics."""
    with pytest.raises(ValueError, match="empty sequence"):
        check_dataset.calculate_token_statistics([])


def test_load_task_tokenizer_prefers_local_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use an already-downloaded tokenizer without a Hugging Face request."""
    config = _config(tmp_path, {"train": [], "validation": [], "test": []})
    local_model_path = tmp_path / "models" / "baseline"
    local_model_path.mkdir(parents=True)
    (local_model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tokenizer = Mock()
    from_pretrained = Mock(return_value=tokenizer)
    monkeypatch.setattr(check_dataset.AutoTokenizer, "from_pretrained", from_pretrained)

    result = check_dataset.load_task_tokenizer(config.models.task_model, config.HF_TOKEN)

    assert result is tokenizer
    from_pretrained.assert_called_once_with(local_model_path, token="test-token", use_fast=True)


def test_load_task_tokenizer_falls_back_to_hugging_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the model ID when no local tokenizer checkpoint exists."""
    config = _config(tmp_path, {"train": [], "validation": [], "test": []})
    from_pretrained = Mock(return_value=Mock())
    monkeypatch.setattr(check_dataset.AutoTokenizer, "from_pretrained", from_pretrained)

    check_dataset.load_task_tokenizer(config.models.task_model, "")

    from_pretrained.assert_called_once_with("org/task-model", token=None, use_fast=True)


def test_main_aggregates_all_files_regardless_of_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report one distribution per named dataset across every split and file."""
    train_a = tmp_path / "train-a.json"
    train_b = tmp_path / "train-b.json"
    validation = tmp_path / "validation.json"
    test = tmp_path / "test.json"
    _write_rows(train_a, [_row("ten")])
    _write_rows(train_b, [_row("twenty")])
    _write_rows(validation, [_row("thirty")])
    _write_rows(test, [_row("forty")])
    config = _config(
        tmp_path,
        {
            "train": [train_a, train_b],
            "validation": [validation],
            "test": [test],
        },
    )
    tokenizer = FakeTokenizer({"ten": 10, "twenty": 20, "thirty": 30, "forty": 40})
    monkeypatch.setattr(check_dataset, "load_config", Mock(return_value=config))
    monkeypatch.setattr(check_dataset, "load_task_tokenizer", Mock(return_value=tokenizer))
    caplog.set_level(logging.INFO)

    check_dataset.main()

    assert "TOKENS  dataset='combined'  valid_rows=4  total_rows=4 - Counts:  p50=30  p95=40  max=40" in caplog.text
    assert "All dataset files passed validation." in caplog.text


def test_main_still_validates_files_when_tokenizer_loading_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Surface tokenizer failure without skipping the existing validation pass."""
    dataset_path = tmp_path / "valid.json"
    _write_rows(dataset_path, [_row("valid")])
    config = _config(
        tmp_path,
        {"train": [dataset_path], "validation": [dataset_path], "test": [dataset_path]},
    )
    monkeypatch.setattr(check_dataset, "load_config", Mock(return_value=config))
    monkeypatch.setattr(check_dataset, "load_task_tokenizer", Mock(side_effect=RuntimeError("offline")))
    caplog.set_level(logging.INFO)

    with pytest.raises(SystemExit, match="1"):
        check_dataset.main()

    assert "TOKENIZER ERROR" in caplog.text
    assert "OK      dataset='combined'" in caplog.text
    assert "Dataset check finished with errors." in caplog.text
