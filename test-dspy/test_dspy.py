# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSPy GEPA optimization workflow."""

import json
from importlib.metadata import metadata, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import dspy
import pandas as pd
import pytest

import casa_slm.prompt_optimizer.scripts.dspy as dspy_script
from casa_slm.prompt_optimizer.config import EvaluationConfig


def _dataframe(
    *,
    task: str = "Create a tracked issue for the incident",
    relevant: bool = True,
) -> pd.DataFrame:
    """Return one valid task-tool matching dataset row."""
    return pd.DataFrame(
        [
            {
                "request": {
                    "task": task,
                    "tool": {"name": "jira_create_issue", "description": "Create a Jira issue"},
                },
                "label": {"relevant": relevant},
                "metadata": {"label": {"reasoning": "The tool creates the requested tracked issue."}},
            }
        ]
    )


def _config(output_dir: Path) -> EvaluationConfig:
    """Return a complete optimization configuration for orchestration tests."""
    return EvaluationConfig.model_validate(
        {
            "experiment": {
                "name": "test-experiment",
                "dataset": [
                    {"name": "test", "train": ["train.json"], "validation": ["val.json"], "test": ["test.json"]}
                ],
            },
            "job": {"machine": "L4", "gpus": 1},
            "gepa": {"output_dir": output_dir, "num_threads": 4},
            "models": {
                "optimizer_model": {"name": "reflection", "end_point": "azure/reflection"},
                "task_model": {"name": "baseline", "hf_model_id": "org/model", "path": "models"},
            },
            "vllm": {
                "port": 8123,
                "max_model_len": 2048,
                "max_batch_size": 8,
                "gpu_memory_utilization": 0.8,
            },
            "LLM_API_KEY": "secret",
            "LLM_BASE_URL": "https://llm.example.test",
        }
    )


def test_installed_dspy_uses_compatible_gepa_release() -> None:
    """Use DSPy 3.3 with the GEPA release it requires."""
    assert dspy.__version__.startswith("3.3.")
    requirements = metadata("dspy").get_all("Requires-Dist", [])
    gepa_requirement = next(requirement for requirement in requirements if requirement.startswith("gepa"))
    assert f"=={version('gepa')}" in gepa_requirement
    assert hasattr(dspy, "GEPA")


def test_task_tool_matcher_signature_uses_initial_prompt_and_typed_fields() -> None:
    """Define the prompt and JSON fields as an optimizable DSPy signature."""
    signature = dspy_script.task_tool_matcher_signature()

    assert signature.instructions == dspy_script.initial_prompt
    assert list(signature.input_fields) == ["task", "tool_name", "tool_description"]
    assert list(signature.output_fields) == ["reasoning", "appropriate"]
    assert signature.output_fields["appropriate"].annotation is bool


def test_dataframe_to_examples_maps_nested_dataset_fields() -> None:
    """Convert nested rows and mark only runtime values as DSPy inputs."""
    example = dspy_script.dataframe_to_examples(_dataframe())[0]

    assert example.task == "Create a tracked issue for the incident"
    assert example.tool_name == "jira_create_issue"
    assert example.tool_description == "Create a Jira issue"
    assert example.appropriate is True
    assert example.reference_reasoning == "The tool creates the requested tracked issue."
    assert example.inputs().toDict() == {
        "task": "Create a tracked issue for the incident",
        "tool_name": "jira_create_issue",
        "tool_description": "Create a Jira issue",
    }


def test_dataframe_to_examples_reports_malformed_row_index() -> None:
    """Reject malformed nested data with a useful row location."""
    dataframe = _dataframe().rename(columns={"request": "invalid"})

    with pytest.raises(ValueError, match=r"index 0.*request"):
        dspy_script.dataframe_to_examples(dataframe)


def test_gepa_metric_returns_correctness_and_reference_feedback() -> None:
    """Provide GEPA with an exact score and actionable classification feedback."""
    gold = dspy_script.dataframe_to_examples(_dataframe())[0]

    correct = dspy_script.gepa_metric(gold, dspy.Prediction(reasoning="Useful", appropriate=True))
    traced_correct = dspy_script.gepa_metric(
        gold,
        dspy.Prediction(reasoning="Useful", appropriate=True),
        program_trace=[("matcher", {"task": gold.task}, {"appropriate": True})],
    )
    incorrect = dspy_script.gepa_metric(gold, dspy.Prediction(reasoning="Unrelated", appropriate=False))
    malformed = dspy_script.gepa_metric(gold, dspy.Prediction(appropriate=True))

    assert correct.score == 1.0
    assert "Correct" in correct.feedback
    assert traced_correct.score == correct.score
    assert traced_correct.feedback == correct.feedback
    assert incorrect.score == 0.0
    assert "Expected appropriate=True" in incorrect.feedback
    assert gold.reference_reasoning in incorrect.feedback
    assert malformed.score == 0.0
    assert "Invalid output" in malformed.feedback


def test_evaluate_program_calculates_metrics_and_format_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summarize binary metrics from DSPy's per-example evaluation results."""
    positive = dspy_script.dataframe_to_examples(_dataframe(relevant=True))[0]
    negative = dspy_script.dataframe_to_examples(_dataframe(relevant=False))[0]
    evaluation = SimpleNamespace(
        results=[
            (positive, dspy.Prediction(reasoning="yes", appropriate=True), 1.0),
            (negative, dspy.Prediction(reasoning="no", appropriate=True), 0.0),
            (positive, dspy.Prediction(appropriate=True), 0.0),
        ]
    )
    evaluate = Mock(return_value=evaluation)
    evaluate_factory = Mock(return_value=evaluate)
    monkeypatch.setattr(dspy_script.dspy, "Evaluate", evaluate_factory)

    metrics = dspy_script.evaluate_program(Mock(), [positive, negative, positive], num_threads=3)

    assert metrics == {
        "total": 3,
        "accuracy": pytest.approx(1 / 3),
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 1,
        "format_failures": 1,
    }
    assert evaluate_factory.call_args.kwargs["num_threads"] == 3
    evaluate.assert_called_once()


def test_run_optimization_wires_models_gepa_evaluation_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the workflow with mocked model boundaries and persist its outputs."""
    config = _config(tmp_path / "outputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    training_dataframe = _dataframe(task="Training task")
    validation_dataframe = _dataframe(task="Validation task")
    test_dataframe = _dataframe(task="Held-out test task")
    load_dataset = Mock(
        return_value={
            "train": training_dataframe,
            "validation": validation_dataframe,
            "test": test_dataframe,
        }
    )
    monkeypatch.setattr(dspy_script, "load_dataset", load_dataset)
    monkeypatch.setattr(dspy_script, "create_run_directory", Mock(return_value=run_dir))

    server = Mock()
    server_factory = Mock(return_value=server)
    monkeypatch.setattr(dspy_script, "VLLMServer", server_factory)

    lm_calls: list[dict[str, Any]] = []
    lm_objects: list[object] = []

    def fake_lm(**kwargs: Any) -> object:
        lm_calls.append(kwargs)
        lm = object()
        lm_objects.append(lm)
        return lm

    monkeypatch.setattr(dspy_script.dspy, "LM", fake_lm)
    configure = Mock()
    monkeypatch.setattr(dspy_script.dspy, "configure", configure)
    adapter = object()
    monkeypatch.setattr(dspy_script.dspy, "JSONAdapter", Mock(return_value=adapter))

    class FakeProgram:
        def __init__(self, instructions: str) -> None:
            self.matcher = SimpleNamespace(signature=SimpleNamespace(instructions=instructions))

        def save(self, path: str) -> None:
            Path(path).write_text("compiled", encoding="utf-8")

    baseline_program = FakeProgram(dspy_script.initial_prompt)
    optimized_program = FakeProgram("Use the optimized decision boundary.")
    monkeypatch.setattr(dspy_script, "TaskToolMatcher", Mock(return_value=baseline_program))

    gepa = Mock()
    gepa.compile.return_value = optimized_program
    gepa_factory = Mock(return_value=gepa)
    monkeypatch.setattr(dspy_script.dspy, "GEPA", gepa_factory)
    baseline_metrics = {"accuracy": 0.5}
    optimized_metrics = {"accuracy": 0.75}
    baseline_test_metrics = {"accuracy": 0.25}
    optimized_test_metrics = {"accuracy": 1.0}
    evaluate = Mock(
        side_effect=[
            baseline_metrics,
            optimized_metrics,
            baseline_test_metrics,
            optimized_test_metrics,
        ]
    )
    monkeypatch.setattr(dspy_script, "evaluate_program", evaluate)

    result = dspy_script.run_optimization(config)

    assert result == run_dir
    load_dataset.assert_called_once_with(config)
    server.start.assert_called_once_with()
    server.stop.assert_called_once_with()
    assert server_factory.call_args.kwargs["log_path"] == run_dir / "optimization.log"
    assert lm_calls == [
        {
            "model": "openai/baseline",
            "api_key": "empty",
            "api_base": "http://127.0.0.1:8123/v1",
            "max_tokens": 1024,
        },
        {
            "model": "openai/azure/reflection",
            "api_key": "secret",
            "api_base": "https://llm.example.test",
            "max_tokens": 32000,
        },
    ]
    configure.assert_called_once_with(lm=lm_objects[0], adapter=adapter)
    assert gepa_factory.call_args.kwargs["auto"] == "light"
    assert gepa_factory.call_args.kwargs["reflection_minibatch_size"] == 35
    assert gepa_factory.call_args.kwargs["candidate_selection_strategy"] == "current_best"
    assert gepa_factory.call_args.kwargs["skip_perfect_score"] is False
    assert gepa_factory.call_args.kwargs["use_merge"] is False
    assert gepa_factory.call_args.kwargs["reflection_lm"] is lm_objects[1]
    compile_kwargs = gepa.compile.call_args.kwargs
    assert len(compile_kwargs["trainset"]) == 1
    assert len(compile_kwargs["valset"]) == 1
    assert compile_kwargs["trainset"][0].task == "Training task"
    assert compile_kwargs["valset"][0].task == "Validation task"
    assert evaluate.call_args_list[0].args[0] is baseline_program
    assert evaluate.call_args_list[0].args[1][0].task == "Validation task"
    assert evaluate.call_args_list[1].args[0] is optimized_program
    assert evaluate.call_args_list[1].args[1][0].task == "Validation task"
    assert evaluate.call_args_list[2].args[0] is baseline_program
    assert evaluate.call_args_list[2].args[1][0].task == "Held-out test task"
    assert evaluate.call_args_list[3].args[0] is optimized_program
    assert evaluate.call_args_list[3].args[1][0].task == "Held-out test task"
    assert (run_dir / "optimized_program.json").read_text(encoding="utf-8") == "compiled"
    assert (run_dir / "optimized_prompt.txt").read_text(encoding="utf-8") == "Use the optimized decision boundary.\n"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["accuracy_delta"] == 0.25
    assert metrics["dataset"] == {"train_size": 1, "validation_size": 1, "test_size": 1}
    assert metrics["test"] == {
        "baseline": baseline_test_metrics,
        "optimized": optimized_test_metrics,
        "accuracy_delta": 0.75,
    }


def test_run_optimization_stops_vllm_when_evaluation_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the child server when work inside the optimization lifecycle raises."""
    config = _config(tmp_path / "outputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataframe = _dataframe()
    load_dataset = Mock(return_value={"train": dataframe, "validation": dataframe, "test": dataframe})
    monkeypatch.setattr(dspy_script, "load_dataset", load_dataset)
    server = Mock()
    monkeypatch.setattr(dspy_script, "VLLMServer", Mock(return_value=server))
    monkeypatch.setattr(dspy_script.dspy, "LM", Mock(return_value=object()))
    monkeypatch.setattr(dspy_script.dspy, "configure", Mock())
    monkeypatch.setattr(dspy_script.dspy, "JSONAdapter", Mock(return_value=object()))
    monkeypatch.setattr(dspy_script, "TaskToolMatcher", Mock(return_value=Mock()))
    monkeypatch.setattr(dspy_script, "evaluate_program", Mock(side_effect=RuntimeError("evaluation failed")))

    with pytest.raises(RuntimeError, match="evaluation failed"):
        dspy_script.run_optimization(config)

    load_dataset.assert_called_once_with(config)
    server.stop.assert_called_once_with()


@pytest.mark.parametrize("test_evaluation", ["baseline", "optimized"])
def test_run_optimization_stops_vllm_when_test_evaluation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_evaluation: str,
) -> None:
    """Stop the child server when either held-out evaluation raises."""
    config = _config(tmp_path / "outputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataframe = _dataframe()
    monkeypatch.setattr(
        dspy_script,
        "load_dataset",
        Mock(return_value={"train": dataframe, "validation": dataframe, "test": dataframe}),
    )
    monkeypatch.setattr(dspy_script, "create_run_directory", Mock(return_value=run_dir))
    server = Mock()
    monkeypatch.setattr(dspy_script, "VLLMServer", Mock(return_value=server))
    monkeypatch.setattr(dspy_script.dspy, "LM", Mock(return_value=object()))
    monkeypatch.setattr(dspy_script.dspy, "configure", Mock())
    monkeypatch.setattr(dspy_script.dspy, "JSONAdapter", Mock(return_value=object()))
    baseline_program = Mock()
    optimized_program = Mock()
    monkeypatch.setattr(dspy_script, "TaskToolMatcher", Mock(return_value=baseline_program))
    optimizer = Mock()
    optimizer.compile.return_value = optimized_program
    monkeypatch.setattr(dspy_script.dspy, "GEPA", Mock(return_value=optimizer))
    validation_metrics = {"accuracy": 0.5}
    side_effects: list[object] = [validation_metrics, validation_metrics]
    if test_evaluation == "baseline":
        side_effects.append(RuntimeError("baseline test evaluation failed"))
    else:
        side_effects.extend(
            [
                validation_metrics,
                RuntimeError("optimized test evaluation failed"),
            ]
        )
    monkeypatch.setattr(dspy_script, "evaluate_program", Mock(side_effect=side_effects))

    with pytest.raises(RuntimeError, match="test evaluation failed"):
        dspy_script.run_optimization(config)

    server.stop.assert_called_once_with()
    assert not (run_dir / "metrics.json").exists()


@pytest.mark.parametrize(
    ("empty_split", "error_split"),
    [("train", "training"), ("validation", "validation"), ("test", "test")],
)
def test_run_optimization_rejects_empty_splits_before_gpu_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_split: str,
    error_split: str,
) -> None:
    """Reject empty optimization and evaluation inputs before creating resources."""
    config = _config(tmp_path / "outputs")
    dataset = {
        "train": _dataframe(),
        "validation": _dataframe(),
        "test": _dataframe(),
    }
    dataset[empty_split] = pd.DataFrame()
    monkeypatch.setattr(dspy_script, "load_dataset", Mock(return_value=dataset))
    create_run_directory = Mock()
    server_factory = Mock()
    monkeypatch.setattr(dspy_script, "create_run_directory", create_run_directory)
    monkeypatch.setattr(dspy_script, "VLLMServer", server_factory)

    with pytest.raises(ValueError, match=f"combined {error_split} split is empty"):
        dspy_script.run_optimization(config)

    create_run_directory.assert_not_called()
    server_factory.assert_not_called()


@pytest.mark.parametrize("field", ["LLM_API_KEY", "LLM_BASE_URL"])
def test_run_optimization_requires_reflection_credentials_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Fail before dataset or GPU work when reflection credentials are missing."""
    config = _config(tmp_path / "outputs")
    config = config.model_copy(update={field: ""})
    load_dataset = Mock()
    monkeypatch.setattr(dspy_script, "load_dataset", load_dataset)

    with pytest.raises(ValueError, match=field):
        dspy_script.run_optimization(config)

    load_dataset.assert_not_called()
