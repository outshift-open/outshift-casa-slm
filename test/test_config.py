# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for project configuration loading."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

import casa_slm.prompt_optimizer.config as config_module


@pytest.fixture(autouse=True)
def isolate_host_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep config fixtures independent of environment variables on the host."""
    monkeypatch.setattr(os, "environ", {})


def _config_data() -> dict[str, object]:
    """Return a complete, valid configuration dictionary."""
    return {
        "experiment": {
            "name": "sample-evaluation",
            "dataset": [
                {
                    "name": "primary",
                    "train": ["data/train-a.jsonl", "data/train-b.jsonl"],
                    "validation": ["data/validation.jsonl"],
                    "test": ["data/test.jsonl"],
                    "limits": [0, 0, 0],
                },
                {
                    "name": "supplemental",
                    "train": ["data/supplemental-train.jsonl"],
                    "validation": ["data/supplemental-validation.jsonl"],
                    "test": ["data/supplemental-test.jsonl"],
                },
            ],
        },
        "job": {"machine": "A100", "gpus": 2},
        "models": {
            "optimizer_model": {"name": "optimizer", "end_point": "azure/optimizer"},
            "task_model": {
                "name": "task",
                "hf_model_id": "org/task-model",
                "path": "models/task-model",
            },
        },
        "vllm": {
            "port": 1234,
            "max_model_len": 4096,
            "max_batch_size": 16,
            "batch_timeout": 0.1,
            "gpu_memory_utilization": 0.8,
            "enforce_eager": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 8192,
        },
    }


def _write_config(path: Path, data: object) -> None:
    """Write YAML configuration data to a file."""
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_config_parses_nested_values_and_environment_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load nested settings and supply missing credentials from the environment."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _config_data())
    dotenv_loader = Mock()
    monkeypatch.setattr(config_module, "load_dotenv", dotenv_loader)
    monkeypatch.setenv("HF_TOKEN", "environment-hf-token")
    monkeypatch.setenv("LLM_API_KEY", "environment-api-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://environment.example.test")

    config = config_module.load_config(config_path)

    assert config.experiment.name == "sample-evaluation"
    assert config.experiment.shuffle is False
    assert len(config.experiment.dataset) == 2
    primary_dataset, supplemental_dataset = config.experiment.dataset
    assert primary_dataset.name == "primary"
    assert primary_dataset.train == ["data/train-a.jsonl", "data/train-b.jsonl"]
    assert primary_dataset.validation == ["data/validation.jsonl"]
    assert primary_dataset.test == ["data/test.jsonl"]
    assert primary_dataset.limits == [0, 0, 0]
    assert supplemental_dataset.name == "supplemental"
    assert supplemental_dataset.limits == [0, 0, 0]
    assert config.job.machine == "A100"
    assert config.job.gpus == 2
    assert config.gepa.auto == "light"
    assert config.gepa.reflection_minibatch_size == 35
    assert config.gepa.candidate_selection_strategy == "current_best"
    assert config.gepa.skip_perfect_score is False
    assert config.gepa.use_merge is False
    assert config.gepa.num_threads == 32
    assert config.gepa.seed == 0
    assert config.gepa.task_max_tokens == 1024
    assert config.gepa.reflection_max_tokens == 32000
    assert config.gepa.output_dir == Path("outputs")
    assert config.models.optimizer_model.end_point == "azure/optimizer"
    assert config.models.task_model.path == Path("models/task-model")
    assert config.vllm.port == 1234
    assert config.vllm.max_model_len == 4096
    assert config.vllm.enforce_eager is True
    assert config.vllm.enable_chunked_prefill is True
    assert config.HF_TOKEN == "environment-hf-token"
    assert config.LLM_API_KEY == "environment-api-key"
    assert config.LLM_BASE_URL == "https://environment.example.test"
    dotenv_loader.assert_called_once_with()


def test_load_config_prefers_credentials_in_yaml_over_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep credential values specified directly in the configuration file."""
    config_path = tmp_path / "config.yaml"
    data = _config_data()
    data.update(
        {
            "HF_TOKEN": "yaml-hf-token",
            "LLM_API_KEY": "yaml-api-key",
            "LLM_BASE_URL": "https://yaml.example.test",
        },
    )
    _write_config(config_path, data)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())
    monkeypatch.setenv("HF_TOKEN", "environment-hf-token")
    monkeypatch.setenv("LLM_API_KEY", "environment-api-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://environment.example.test")

    config = config_module.load_config(config_path)

    assert config.HF_TOKEN == "yaml-hf-token"
    assert config.LLM_API_KEY == "yaml-api-key"
    assert config.LLM_BASE_URL == "https://yaml.example.test"


def test_load_config_uses_empty_credential_defaults_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use empty strings when optional credentials are absent everywhere."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _config_data())
    monkeypatch.setattr(config_module, "load_dotenv", Mock())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    config = config_module.load_config(config_path)

    assert config.HF_TOKEN == ""
    assert config.LLM_API_KEY == ""
    assert config.LLM_BASE_URL == ""


def test_load_config_uses_default_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read eval_config.yaml in the current directory when no path is supplied."""
    _write_config(tmp_path / "eval_config.yaml", _config_data())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    config = config_module.load_config()

    assert config.experiment.name == "sample-evaluation"


def test_eval_config_snapshot_matches_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load a stable eval config snapshot for schema coverage."""
    config_path = Path(__file__).resolve().parent / "fixtures" / "eval_config.snapshot.yaml"
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    config = config_module.load_config(config_path)

    assert config.experiment.shuffle is False

    one_tool, two_tools, three_tools = config.experiment.dataset
    assert one_tool.name == "one_tool"
    assert one_tool.train == ["./data/one_tool_astra_train_curated.json"]
    assert one_tool.validation == ["./data/one_tool_astra_val_curated.json"]
    assert one_tool.test == ["./data/one_tool_astra_test_curated.json"]
    assert one_tool.limits == [0, 0, 0]
    assert two_tools.name == "two_tools"
    assert two_tools.train == [
        "./data/two_tools_astra_train_curated.json",
    ]
    assert two_tools.validation == [
        "./data/two_tools_astra_val_curated.json",
    ]
    assert two_tools.test == ["./data/two_tools_astra_test_curated.json"]
    assert two_tools.limits == [0, 0, 0]
    assert three_tools.name == "three_tools"
    assert three_tools.train == [
        "./data/three_tools_astra_train_curated.json",
    ]
    assert three_tools.validation == [
        "./data/three_tools_astra_val_curated.json",
    ]
    assert three_tools.test == [
        "./data/three_tools_astra_test_curated.json",
    ]
    assert three_tools.limits == [0, 0, 0]
    assert config.models.optimizer_model.end_point == "azure/gpt-5.4"
    assert config.models.task_model.hf_model_id == "google/gemma-3-4b-it"
    assert config.models.task_model.path == Path("data/models")


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """Reject YAML documents that cannot represent a configuration mapping."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, ["not", "a", "mapping"])

    with pytest.raises(AssertionError, match="Config file must be a dictionary"):
        config_module.load_config(config_path)


def test_load_config_rejects_invalid_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface Pydantic validation errors for incomplete configuration data."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, {"experiment": {"name": "missing-fields"}})
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    with pytest.raises(ValidationError):
        config_module.load_config(config_path)


def test_load_config_parses_gepa_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load explicit GEPA optimization settings."""
    config_path = tmp_path / "config.yaml"
    data = _config_data()
    data["gepa"] = {
        "auto": "heavy",
        "reflection_minibatch_size": 7,
        "candidate_selection_strategy": "pareto",
        "skip_perfect_score": True,
        "use_merge": True,
        "num_threads": 8,
        "seed": 42,
        "task_max_tokens": 256,
        "reflection_max_tokens": 8192,
        "output_dir": "artifacts",
    }
    _write_config(config_path, data)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    config = config_module.load_config(config_path)

    assert config.gepa.auto == "heavy"
    assert config.gepa.reflection_minibatch_size == 7
    assert config.gepa.candidate_selection_strategy == "pareto"
    assert config.gepa.skip_perfect_score is True
    assert config.gepa.use_merge is True
    assert config.gepa.num_threads == 8
    assert config.gepa.seed == 42
    assert config.gepa.task_max_tokens == 256
    assert config.gepa.reflection_max_tokens == 8192
    assert config.gepa.output_dir == Path("artifacts")


@pytest.mark.parametrize(
    "field",
    ["reflection_minibatch_size", "num_threads", "task_max_tokens", "reflection_max_tokens"],
)
def test_load_config_rejects_non_positive_gepa_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Require positive GEPA sizes, concurrency, and token limits."""
    config_path = tmp_path / "config.yaml"
    data = _config_data()
    data["gepa"] = {field: 0}
    _write_config(config_path, data)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    with pytest.raises(ValidationError, match=field):
        config_module.load_config(config_path)


def test_load_config_rejects_unknown_gepa_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Restrict GEPA automatic budgets to DSPy's supported values."""
    config_path = tmp_path / "config.yaml"
    data = _config_data()
    data["gepa"] = {"auto": "unlimited"}
    _write_config(config_path, data)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    with pytest.raises(ValidationError, match=r"gepa\.auto"):
        config_module.load_config(config_path)


def test_load_config_rejects_unknown_candidate_selection_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restrict candidate selection to strategies supported by both workflows."""
    config_path = tmp_path / "config.yaml"
    data = _config_data()
    data["gepa"] = {"candidate_selection_strategy": "epsilon_greedy"}
    _write_config(config_path, data)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    with pytest.raises(ValidationError, match="candidate_selection_strategy"):
        config_module.load_config(config_path)


def test_load_config_rejects_nonzero_limits_for_multiple_named_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject ambiguous per-entry limits when combining named datasets."""
    config_path = tmp_path / "config.yaml"
    data = _config_data()
    experiment = data["experiment"]
    assert isinstance(experiment, dict)
    datasets = experiment["dataset"]
    assert isinstance(datasets, list)
    primary = datasets[0]
    assert isinstance(primary, dict)
    primary["limits"] = [100, 50, 0]
    _write_config(config_path, data)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    with pytest.raises(ValidationError, match="Nonzero dataset limits require exactly one named dataset"):
        config_module.load_config(config_path)


@pytest.mark.parametrize("limits", [[10, 20], [10, 20, 30, 40]])
def test_load_config_rejects_dataset_limits_with_wrong_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limits: list[int],
) -> None:
    """Require one limit for each train, validation, and test split."""
    config_path = tmp_path / "config.yaml"
    data = _config_data()
    dataset = data["experiment"]
    assert isinstance(dataset, dict)
    datasets = dataset["dataset"]
    assert isinstance(datasets, list)
    datasets[0]["limits"] = limits
    _write_config(config_path, data)
    monkeypatch.setattr(config_module, "load_dotenv", Mock())

    with pytest.raises(ValidationError, match=r"at least 3 items|at most 3 items"):
        config_module.load_config(config_path)
