# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for the direct GEPA optimization workflow."""

import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import Mock

import gepa
import pandas as pd
import pytest

import casa_slm.prompt_optimizer.scripts.gepa as gepa_script
from casa_slm.prompt_optimizer.config import EvaluationConfig


def _dataframe(
    *,
    task: str = "Create a tracked issue for the incident",
    relevant: bool = True,
    description: object = "Create a Jira issue",
) -> pd.DataFrame:
    """Return one valid task-tool matching dataset row."""
    return pd.DataFrame(
        [
            {
                "request": {
                    "task": task,
                    "tool": {
                        "name": "jira_create_issue",
                        "description": description,
                    },
                },
                "label": {"relevant": relevant},
                "metadata": {
                    "label": {
                        "reasoning": "The tool creates the requested tracked issue.",
                    }
                },
            }
        ]
    )


def _example(
    *,
    task: str = "Create a tracked issue for the incident",
    appropriate: bool = True,
) -> gepa_script.TaskToolExample:
    """Return one direct GEPA example."""
    return gepa_script.TaskToolExample(
        task=task,
        tool_name="jira_create_issue",
        tool_description="Create a Jira issue",
        appropriate=appropriate,
        reference_reasoning="The tool creates the requested tracked issue.",
    )


def _outcome(
    appropriate: bool | None,
    *,
    reasoning: str = "The tool can help.",
) -> gepa_script.EvaluationOutcome:
    """Return one parsed or malformed task-model outcome."""
    if appropriate is None:
        return gepa_script.EvaluationOutcome(
            output=None,
            raw_response="not json",
            format_error="Invalid task-model JSON",
        )
    output = gepa_script.TaskToolMatcherOutput(
        reasoning=reasoning,
        appropriate=appropriate,
    )
    return gepa_script.EvaluationOutcome(
        output=output,
        raw_response=output.model_dump_json(),
    )


def _completion(content: str | None) -> SimpleNamespace:
    """Return a LiteLLM-like completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _config(output_dir: Path) -> EvaluationConfig:
    """Return a complete optimization configuration for orchestration tests."""
    return EvaluationConfig.model_validate(
        {
            "experiment": {
                "name": "test-experiment",
                "dataset": [
                    {
                        "name": "test",
                        "train": ["train.json"],
                        "validation": ["val.json"],
                        "test": ["test.json"],
                    }
                ],
            },
            "job": {"machine": "L4", "gpus": 1},
            "gepa": {
                "output_dir": output_dir,
                "auto": "light",
                "reflection_minibatch_size": 4,
                "num_threads": 3,
                "seed": 17,
                "task_max_tokens": 123,
                "reflection_max_tokens": 456,
            },
            "models": {
                "optimizer_model": {
                    "name": "reflection",
                    "end_point": "azure/reflection",
                },
                "task_model": {
                    "name": "baseline",
                    "hf_model_id": "org/model",
                    "path": "models",
                },
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


class _FakeGEPAResult:
    """Small JSON-safe direct GEPA result for orchestration tests."""

    best_candidate: ClassVar[dict[str, str]] = {"preamble": "Use the optimized decision boundary."}
    best_idx = 1
    num_candidates = 2
    total_metric_calls = 42

    def to_dict(self) -> dict[str, object]:
        """Return the serialized optimization result."""
        return {
            "candidates": [
                {"preamble": gepa_script.initial_preamble},
                self.best_candidate,
            ],
            "parents": [[None], [0]],
            "val_aggregate_scores": [0.5, 0.75],
            "best_idx": self.best_idx,
            "total_metric_calls": self.total_metric_calls,
        }


def _patch_run_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[EvaluationConfig, Path, Mock, Mock, Mock, Mock, object]:
    """Patch external orchestration boundaries for one optimization run."""
    config = _config(tmp_path / "outputs")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    training_dataframe = _dataframe(task="Training task")
    validation_dataframe = _dataframe(task="Validation task")
    test_dataframe = _dataframe(task="Held-out test task")
    monkeypatch.setattr(
        gepa_script,
        "load_dataset",
        Mock(
            return_value={
                "train": training_dataframe,
                "validation": validation_dataframe,
                "test": test_dataframe,
            }
        ),
    )
    monkeypatch.setattr(
        gepa_script,
        "create_run_directory",
        Mock(return_value=run_dir),
    )
    monkeypatch.setattr(gepa_script, "configure_logging", Mock())
    monkeypatch.setattr(
        gepa_script,
        "capture_console_output",
        Mock(return_value=contextlib.nullcontext()),
    )

    server = Mock()
    server_factory = Mock(return_value=server)
    monkeypatch.setattr(gepa_script, "VLLMServer", server_factory)

    evaluator = object()
    evaluator_factory = Mock(return_value=evaluator)
    monkeypatch.setattr(gepa_script, "TaskModelEvaluator", evaluator_factory)

    optimize = Mock(return_value=_FakeGEPAResult())
    monkeypatch.setattr(gepa_script, "optimize_anything", optimize)
    return (
        config,
        run_dir,
        server,
        server_factory,
        evaluator_factory,
        optimize,
        evaluator,
    )


def test_gepa_cpu_environment_has_direct_gepa_without_dspy() -> None:
    """Run the direct workflow in the GEPA environment without DSPy."""
    assert hasattr(gepa, "optimize_anything")
    assert importlib.util.find_spec("dspy") is None


def test_dataframe_to_examples_maps_nested_dataset_fields() -> None:
    """Convert nested rows into direct GEPA examples."""
    example = gepa_script.dataframe_to_examples(_dataframe())[0]

    assert example == _example()


@pytest.mark.parametrize("description", [None, "", "   ", 42])
def test_dataframe_to_examples_supplies_missing_description(
    description: object,
) -> None:
    """Fall back to the tool name when its description is unusable."""
    example = gepa_script.dataframe_to_examples(
        _dataframe(description=description),
    )[0]

    assert example.tool_description == ("(No description. Rely on tool name: 'jira_create_issue')")


def test_dataframe_to_examples_reports_malformed_row_index() -> None:
    """Reject malformed nested data with a useful row location."""
    dataframe = _dataframe().rename(columns={"request": "invalid"})

    with pytest.raises(ValueError, match=r"index 0.*request"):
        gepa_script.dataframe_to_examples(dataframe)


def test_task_model_evaluator_sends_structured_vllm_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the candidate, example, and strict output schema through LiteLLM."""
    completion = Mock(return_value=_completion('{"reasoning":"The tool creates an issue.","appropriate":true}'))
    monkeypatch.setattr(gepa_script.litellm, "completion", completion)
    evaluator = gepa_script.TaskModelEvaluator(
        model_name="openai/baseline",
        api_base="http://127.0.0.1:8123/v1",
        max_tokens=123,
    )

    outcome = evaluator.evaluate(
        {"preamble": "Classify the requested tool."},
        _example(),
    )

    assert outcome.output == gepa_script.TaskToolMatcherOutput(
        reasoning="The tool creates an issue.",
        appropriate=True,
    )
    kwargs = completion.call_args.kwargs
    assert kwargs["model"] == "openai/baseline"
    assert kwargs["api_key"] == "empty"
    assert kwargs["api_base"] == "http://127.0.0.1:8123/v1"
    assert kwargs["max_tokens"] == 123
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": (
                "Classify the requested tool.\n\n"
                "### User Task\n"
                "Create a tracked issue for the incident\n\n"
                "### Requested Tool Name\n"
                "jira_create_issue\n\n"
                "### Requested Tool Description\n"
                "Create a Jira issue"
            ),
        }
    ]
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["reasoning", "appropriate"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["appropriate"]["type"] == "boolean"


def test_render_user_prompt_preserves_multiline_unicode_and_braces() -> None:
    """Insert example text once without interpreting its formatting characters."""
    example = gepa_script.TaskToolExample(
        task="Review {incident}\nfor Zürich",
        tool_name="lookup_{asset}",
        tool_description="Find {asset}\n第二行",
        appropriate=True,
        reference_reasoning="The tool can find the asset.",
    )

    prompt = gepa_script._render_user_prompt(
        {"preamble": "Classify {carefully}."},
        example,
    )

    assert prompt == (
        "Classify {carefully}.\n\n"
        "### User Task\n"
        "Review {incident}\n"
        "for Zürich\n\n"
        "### Requested Tool Name\n"
        "lookup_{asset}\n\n"
        "### Requested Tool Description\n"
        "Find {asset}\n"
        "第二行"
    )


@pytest.mark.parametrize(
    ("content", "error_pattern"),
    [
        ("not json", "Invalid task-model JSON"),
        ('{"reasoning":"","appropriate":true}', "Invalid task-model JSON"),
        ('{"reasoning":"yes"}', "Invalid task-model JSON"),
        ("", "empty response"),
        (None, "empty response"),
    ],
)
def test_task_model_evaluator_reports_format_failures(
    monkeypatch: pytest.MonkeyPatch,
    content: str | None,
    error_pattern: str,
) -> None:
    """Turn malformed model responses into explicit format failures."""
    monkeypatch.setattr(
        gepa_script.litellm,
        "completion",
        Mock(return_value=_completion(content)),
    )
    evaluator = gepa_script.TaskModelEvaluator(
        model_name="openai/baseline",
        api_base="http://127.0.0.1:8123/v1",
        max_tokens=123,
    )

    outcome = evaluator.evaluate(
        {"preamble": "Classify."},
        _example(),
    )

    assert outcome.output is None
    assert outcome.format_error is not None
    assert error_pattern.lower() in outcome.format_error.lower()


def test_task_model_evaluator_propagates_model_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not silently score task-model transport failures as bad prompts."""
    monkeypatch.setattr(
        gepa_script.litellm,
        "completion",
        Mock(side_effect=RuntimeError("vLLM unavailable")),
    )
    evaluator = gepa_script.TaskModelEvaluator(
        model_name="openai/baseline",
        api_base="http://127.0.0.1:8123/v1",
        max_tokens=123,
    )

    with pytest.raises(RuntimeError, match="vLLM unavailable"):
        evaluator.evaluate({"preamble": "Classify."}, _example())


def test_score_outcome_returns_correctness_and_actionable_feedback() -> None:
    """Provide GEPA with exact scores and rich classification diagnostics."""
    example = _example()

    correct_score, correct_info = gepa_script.score_outcome(
        example,
        _outcome(True),
    )
    incorrect_score, incorrect_info = gepa_script.score_outcome(
        example,
        _outcome(False, reasoning="The tool looks unrelated."),
    )
    malformed_score, malformed_info = gepa_script.score_outcome(
        example,
        _outcome(None),
    )

    assert correct_score == 1.0
    assert correct_info["feedback"] == "Correct: appropriate=True."
    assert incorrect_score == 0.0
    assert "Expected appropriate=True" in incorrect_info["feedback"]
    assert example.reference_reasoning in incorrect_info["feedback"]
    assert incorrect_info["predicted_appropriate"] is False
    assert malformed_score == 0.0
    assert "Invalid output" in malformed_info["feedback"]
    assert malformed_info["predicted_appropriate"] is None
    assert malformed_info["format_error"] == "Invalid task-model JSON"


def test_calculate_metrics_counts_confusion_matrix_and_format_failures() -> None:
    """Summarize binary results and make every format failure an error."""
    examples = [
        _example(task="true-positive", appropriate=True),
        _example(task="false-positive", appropriate=False),
        _example(task="true-negative", appropriate=False),
        _example(task="false-negative", appropriate=True),
        _example(task="format-failure", appropriate=False),
    ]
    outcomes = [
        _outcome(True),
        _outcome(True),
        _outcome(False),
        _outcome(False),
        _outcome(None),
    ]

    metrics = gepa_script.calculate_metrics(examples, outcomes)

    assert metrics == {
        "total": 5,
        "accuracy": 0.4,
        "precision": pytest.approx(1 / 3),
        "recall": 0.5,
        "f1": 0.4,
        "true_positive": 1,
        "false_positive": 2,
        "true_negative": 1,
        "false_negative": 1,
        "format_failures": 1,
    }


def test_calculate_metrics_handles_empty_input_and_length_mismatch() -> None:
    """Return zero-valued metrics safely and reject misaligned inputs."""
    assert gepa_script.calculate_metrics([], []) == {
        "total": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "format_failures": 0,
    }

    with pytest.raises(ValueError, match="same length"):
        gepa_script.calculate_metrics([_example()], [])


def test_evaluate_preamble_preserves_example_order() -> None:
    """Associate concurrent outcomes with examples in their input order."""
    examples = [
        _example(task="positive", appropriate=True),
        _example(task="negative", appropriate=False),
    ]
    evaluator = Mock()

    def evaluate(
        candidate: dict[str, str],
        example: gepa_script.TaskToolExample,
    ) -> gepa_script.EvaluationOutcome:
        assert candidate == {"preamble": "candidate prompt"}
        return _outcome(example.task == "positive")

    evaluator.evaluate.side_effect = evaluate

    metrics = gepa_script.evaluate_preamble(
        "candidate prompt",
        examples,
        evaluator,
        num_threads=2,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1


@pytest.mark.parametrize(
    ("auto", "validation_size", "expected"),
    [
        ("light", 10, 420),
        ("medium", 10, 740),
        ("heavy", 10, 1105),
        ("light", 50, 580),
        ("medium", 50, 940),
        ("heavy", 50, 1385),
    ],
)
def test_auto_metric_budget_matches_dspy(
    auto: gepa_script.AutoBudget,
    validation_size: int,
    expected: int,
) -> None:
    """Reproduce DSPy's one-component automatic GEPA budget."""
    assert gepa_script.auto_metric_budget(auto, validation_size, 35) == expected


def test_auto_metric_budget_rejects_negative_validation_size() -> None:
    """Reject an impossible validation-set size."""
    with pytest.raises(ValueError, match="non-negative"):
        gepa_script.auto_metric_budget("light", -1, 35)


def test_auto_metric_budget_uses_configured_reflection_minibatch_size() -> None:
    """Charge each reflection trial for its configured number of examples."""
    assert gepa_script.auto_metric_budget("light", 10, 4) == 110


def test_auto_metric_budget_rejects_non_positive_reflection_minibatch_size() -> None:
    """Reject a reflection batch that cannot evaluate any examples."""
    with pytest.raises(ValueError, match="reflection_minibatch_size must be positive"):
        gepa_script.auto_metric_budget("light", 10, 0)


def test_gepa_config_applies_search_policy_overrides(tmp_path: Path) -> None:
    """Apply configurable selection, perfect-score, and merge behavior."""
    config = _config(tmp_path / "outputs")
    config = config.model_copy(
        update={
            "gepa": config.gepa.model_copy(
                update={
                    "candidate_selection_strategy": "pareto",
                    "skip_perfect_score": True,
                    "use_merge": True,
                }
            )
        }
    )

    direct_config = gepa_script._gepa_config(config, tmp_path / "gepa", max_metric_calls=100)

    assert direct_config.engine.candidate_selection_strategy == "pareto"
    assert direct_config.reflection.skip_perfect_score is True
    assert isinstance(direct_config.merge, gepa_script.MergeConfig)


def test_run_optimization_wires_direct_gepa_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the workflow with mocked model boundaries and persist its outputs."""
    (
        config,
        run_dir,
        server,
        server_factory,
        evaluator_factory,
        optimize,
        evaluator,
    ) = _patch_run_boundaries(tmp_path, monkeypatch)
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
    monkeypatch.setattr(gepa_script, "evaluate_preamble", evaluate)

    result = gepa_script.run_optimization(config)

    assert result == run_dir
    server.start.assert_called_once_with()
    server.stop.assert_called_once_with()
    assert server_factory.call_args.kwargs == {
        "model_dir": "models/baseline",
        "served_model_name": "baseline",
        "port": 8123,
        "max_model_len": 2048,
        "max_batch_size": 8,
        "gpu_memory_utilization": 0.8,
        "apply_wsl_workaround": False,
        "log_path": run_dir / "optimization.log",
    }
    evaluator_factory.assert_called_once_with(
        model_name="openai/baseline",
        api_base="http://127.0.0.1:8123/v1",
        max_tokens=123,
    )

    assert evaluate.call_args_list[0].args == (
        gepa_script.initial_preamble,
        gepa_script.dataframe_to_examples(_dataframe(task="Validation task")),
        evaluator,
    )
    assert evaluate.call_args_list[0].kwargs == {"num_threads": 3}
    assert evaluate.call_args_list[1].args[0] == ("Use the optimized decision boundary.")
    assert evaluate.call_args_list[1].args[1][0].task == "Validation task"
    assert evaluate.call_args_list[2].args[0] == gepa_script.initial_preamble
    assert evaluate.call_args_list[2].args[1][0].task == "Held-out test task"
    assert evaluate.call_args_list[3].args[0] == "Use the optimized decision boundary."
    assert evaluate.call_args_list[3].args[1][0].task == "Held-out test task"

    optimize_kwargs = optimize.call_args.kwargs
    assert optimize_kwargs["seed_candidate"] == {
        "preamble": gepa_script.initial_preamble,
    }
    assert optimize_kwargs["evaluator"] is evaluator
    assert optimize_kwargs["dataset"] == gepa_script.dataframe_to_examples(_dataframe(task="Training task"))
    assert optimize_kwargs["valset"] == gepa_script.dataframe_to_examples(_dataframe(task="Validation task"))
    assert optimize_kwargs["objective"] == gepa_script.optimization_objective
    assert optimize_kwargs["background"] == gepa_script.optimization_background

    direct_config = optimize_kwargs["config"]
    assert direct_config.engine.run_dir == str(run_dir / "gepa")
    assert direct_config.engine.seed == 17
    assert direct_config.engine.max_metric_calls == 74
    assert direct_config.engine.candidate_selection_strategy == "current_best"
    assert direct_config.engine.max_workers == 3
    assert direct_config.engine.parallel is True
    assert direct_config.engine.raise_on_exception is True
    assert direct_config.reflection.skip_perfect_score is False
    assert direct_config.reflection.perfect_score == 1.0
    assert direct_config.reflection.reflection_minibatch_size == 4
    assert direct_config.reflection.reflection_lm == "openai/azure/reflection"
    assert direct_config.reflection.reflection_lm_kwargs == {
        "api_key": "secret",
        "api_base": "https://llm.example.test",
        "max_tokens": 456,
    }
    assert direct_config.merge is None

    assert (run_dir / "optimized_prompt.txt").read_text(encoding="utf-8") == (
        f"Use the optimized decision boundary.\n\n{gepa_script.user_prompt_template}\n"
    )
    result_artifact = json.loads((run_dir / "gepa_result.json").read_text(encoding="utf-8"))
    assert result_artifact["best_idx"] == 1
    assert len(result_artifact["candidates"]) == 2
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["accuracy_delta"] == 0.25
    assert metrics["dataset"] == {
        "train_size": 1,
        "validation_size": 1,
        "test_size": 1,
    }
    assert metrics["test"] == {
        "baseline": baseline_test_metrics,
        "optimized": optimized_test_metrics,
        "accuracy_delta": 0.75,
    }
    assert metrics["gepa"]["resolved_max_metric_calls"] == 74
    assert metrics["gepa"]["candidate_count"] == 2
    assert metrics["gepa"]["total_metric_calls"] == 42


@pytest.mark.parametrize(
    ("configured_model", "litellm_model"),
    [
        ("azure/gpt-5.4", "openai/azure/gpt-5.4"),
        (
            "bedrock/global.anthropic.claude-sonnet-4-6",
            "openai/bedrock/global.anthropic.claude-sonnet-4-6",
        ),
        ("openai/azure/gpt-5.4", "openai/azure/gpt-5.4"),
        ("  azure/gpt-5.4  ", "openai/azure/gpt-5.4"),
    ],
)
def test_openai_proxy_model_name_preserves_logical_model(
    configured_model: str,
    litellm_model: str,
) -> None:
    """Keep the full proxy model name after LiteLLM provider parsing."""
    assert gepa_script._openai_proxy_model_name(configured_model) == litellm_model


def test_openai_proxy_model_name_rejects_empty_model() -> None:
    """Reject an unusable reflection model before calling LiteLLM."""
    with pytest.raises(ValueError, match="must not be empty"):
        gepa_script._openai_proxy_model_name("   ")


@pytest.mark.parametrize(
    "failure_stage",
    ["baseline", "optimization", "optimized", "test_baseline", "test_optimized", "artifacts"],
)
def test_run_optimization_stops_vllm_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Stop the child server when any in-lifecycle stage raises."""
    (
        config,
        _run_dir,
        server,
        _server_factory,
        _evaluator_factory,
        optimize,
        _evaluator,
    ) = _patch_run_boundaries(tmp_path, monkeypatch)
    metrics = {"accuracy": 0.5}
    if failure_stage == "baseline":
        evaluate = Mock(side_effect=RuntimeError("baseline failed"))
    elif failure_stage == "optimized":
        evaluate = Mock(side_effect=[metrics, RuntimeError("optimized failed")])
    elif failure_stage == "test_baseline":
        evaluate = Mock(side_effect=[metrics, metrics, RuntimeError("test baseline failed")])
    elif failure_stage == "test_optimized":
        evaluate = Mock(side_effect=[metrics, metrics, metrics, RuntimeError("test optimized failed")])
    else:
        evaluate = Mock(return_value=metrics)
    monkeypatch.setattr(gepa_script, "evaluate_preamble", evaluate)

    if failure_stage == "optimization":
        optimize.side_effect = RuntimeError("optimization failed")
    if failure_stage == "artifacts":
        monkeypatch.setattr(
            gepa_script,
            "_write_artifacts",
            Mock(side_effect=RuntimeError("artifacts failed")),
        )

    with pytest.raises(RuntimeError, match="failed"):
        gepa_script.run_optimization(config)

    server.stop.assert_called_once_with()
    if failure_stage in {"test_baseline", "test_optimized"}:
        assert not (_run_dir / "metrics.json").exists()


@pytest.mark.parametrize("field", ["LLM_API_KEY", "LLM_BASE_URL"])
def test_run_optimization_requires_reflection_credentials_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Fail before dataset or GPU work when credentials are missing."""
    config = _config(tmp_path / "outputs").model_copy(update={field: ""})
    load_dataset = Mock()
    server_factory = Mock()
    monkeypatch.setattr(gepa_script, "load_dataset", load_dataset)
    monkeypatch.setattr(gepa_script, "VLLMServer", server_factory)

    with pytest.raises(ValueError, match=field):
        gepa_script.run_optimization(config)

    load_dataset.assert_not_called()
    server_factory.assert_not_called()


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
    """Reject empty optimization inputs before creating GPU resources."""
    config = _config(tmp_path / "outputs")
    dataset: dict[str, pd.DataFrame] = {
        "train": _dataframe(),
        "validation": _dataframe(),
        "test": _dataframe(),
    }
    dataset[empty_split] = pd.DataFrame()
    monkeypatch.setattr(
        gepa_script,
        "load_dataset",
        Mock(return_value=dataset),
    )
    create_run_directory = Mock()
    server_factory = Mock()
    monkeypatch.setattr(gepa_script, "create_run_directory", create_run_directory)
    monkeypatch.setattr(gepa_script, "VLLMServer", server_factory)

    with pytest.raises(
        ValueError,
        match=f"combined {error_split} split is empty",
    ):
        gepa_script.run_optimization(config)

    create_run_directory.assert_not_called()
    server_factory.assert_not_called()


def test_optimized_preamble_requires_named_non_empty_component() -> None:
    """Reject malformed optimization results before artifact reporting."""
    for candidate in ("prompt", {}, {"preamble": ""}):
        with pytest.raises(RuntimeError, match=r"candidate|preamble"):
            gepa_script._optimized_preamble(candidate)


def test_litellm_response_shape_is_validated() -> None:
    """Raise on malformed client responses instead of scoring them."""
    for response in (object(), SimpleNamespace(choices=[])):
        with pytest.raises(
            RuntimeError,
            match="invalid completion response",
        ):
            gepa_script._response_content(response)

    response = _completion(content=cast(Any, ["not", "text"]))
    with pytest.raises(RuntimeError, match="non-text"):
        gepa_script._response_content(response)
