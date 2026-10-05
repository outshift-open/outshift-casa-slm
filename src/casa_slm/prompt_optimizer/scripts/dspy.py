"""Optimize the task-tool matching prompt using DSPy GEPA."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cleanprompt as cp
import dspy
import pandas as pd
from pydantic import BaseModel, ValidationError

from casa_slm.prompt_optimizer import (
    EvaluationConfig,
    VLLMServer,
    capture_console_output,
    configure_logging,
    create_run_directory,
    load_config,
    load_dataset,
)

logger = logging.getLogger(__name__)


class TaskToolMatcherOutput(BaseModel):
    """Output schema for task-tool matcher with reasoning first."""

    reasoning: str
    appropriate: bool


initial_prompt = cp.clean("""
    You are an expert system that evaluates tool selection.
    Determine if the "Requested Tool" is appropriate for the "User Task".
    - A tool is APPROPRIATE if it helps with ANY part of the task.
    - A tool is NOT APPROPRIATE only if it is completely unrelated.
    Respond in JSON with "reasoning" and "appropriate".
    """)


def task_tool_matcher_signature() -> type[dspy.Signature]:
    """Create the optimizable task-tool matcher signature."""
    return dspy.make_signature(
        {
            "task": (str, dspy.InputField(desc="The user task to complete")),
            "tool_name": (str, dspy.InputField(desc="The requested tool name")),
            "tool_description": (str, dspy.InputField(desc="The requested tool description")),
            "reasoning": (
                str,
                dspy.OutputField(desc="Reasoning about whether the tool helps with any part of the task"),
            ),
            "appropriate": (bool, dspy.OutputField(desc="Whether the requested tool is appropriate for the task")),
        },
        instructions=initial_prompt,
        signature_name="TaskToolMatcherSignature",
    )


class TaskToolMatcher(dspy.Module):
    """DSPy program for task-tool matching."""

    def __init__(self) -> None:
        """Initialize the task-tool matching predictor."""
        super().__init__()
        self.matcher = dspy.Predict(task_tool_matcher_signature())

    def forward(self, task: str, tool_name: str, tool_description: str) -> dspy.Prediction:
        """Predict whether the requested tool is appropriate."""
        return self.matcher(task=task, tool_name=tool_name, tool_description=tool_description)


def dataframe_to_examples(dataframe: pd.DataFrame) -> list[dspy.Example]:
    """Convert task-tool dataset rows into labeled DSPy examples.

    Args:
        dataframe: Dataset split with nested ``request``, ``label``, and ``metadata`` columns.

    Returns:
        DSPy examples with only task and tool fields marked as inputs.

    Raises:
        ValueError: If a row does not conform to the task-tool dataset schema.
    """
    examples: list[dspy.Example] = []
    for row_index, row in dataframe.iterrows():
        try:
            request = _mapping(row["request"], "request")
            tool = _mapping(request["tool"], "request.tool")
            label = _mapping(row["label"], "label")
            metadata = _mapping(row["metadata"], "metadata")
            metadata_label = _mapping(metadata["label"], "metadata.label")

            task = _string(request["task"], "request.task")
            tool_name = _string(tool["name"], "request.tool.name")

            tool_description = tool["description"]
            if not isinstance(tool_description, str) or not tool_description.strip():
                tool_description = f"(No description. Rely on tool name: {tool_name!r})"
            appropriate = label["relevant"]
            if not isinstance(appropriate, bool):
                raise ValueError("label.relevant must be a boolean")
            reference_reasoning = _string(metadata_label["reasoning"], "metadata.label.reasoning")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid task-tool dataset row at index {row_index!r}: {error}") from error

        examples.append(
            dspy.Example(
                task=task,
                tool_name=tool_name,
                tool_description=tool_description,
                appropriate=appropriate,
                reference_reasoning=reference_reasoning,
            ).with_inputs("task", "tool_name", "tool_description")
        )
    return examples


def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
    program_trace: Any | None = None,
) -> dspy.Prediction:
    """Score a prediction and return textual feedback for GEPA."""
    del trace, pred_trace, program_trace
    output = _prediction_output(pred)
    expected = bool(gold.appropriate)
    prefix = f"Predictor {pred_name!r}. " if pred_name else ""

    if output is None:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"{prefix}Invalid output. Expected JSON with a string 'reasoning' and a boolean 'appropriate'. "
                f"The expected appropriate value was {expected}."
            ),
        )

    if output.appropriate == expected:
        return dspy.Prediction(
            score=1.0,
            feedback=f"{prefix}Correct: appropriate={expected}.",
        )

    reference_reasoning = str(gold.reference_reasoning)
    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"{prefix}Incorrect classification. Expected appropriate={expected}, but predicted "
            f"appropriate={output.appropriate}. Reference reasoning: {reference_reasoning}"
        ),
    )


def accuracy_metric(gold: dspy.Example, pred: dspy.Prediction, trace: Any | None = None) -> float:
    """Return exact task-tool matching accuracy for DSPy evaluation."""
    del trace
    output = _prediction_output(pred)
    return float(output is not None and output.appropriate == bool(gold.appropriate))


def evaluate_program(
    program: dspy.Module,
    examples: list[dspy.Example],
    *,
    num_threads: int,
) -> dict[str, int | float]:
    """Evaluate a program and calculate binary classification metrics."""
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=accuracy_metric,
        num_threads=num_threads,
        display_progress=True,
    )
    evaluation = evaluator(program)
    results = getattr(evaluation, "results", None)
    if results is None and isinstance(evaluation, tuple) and len(evaluation) == 2:
        _, results = evaluation
    if results is None:
        raise RuntimeError("DSPy evaluation did not return per-example outputs")

    true_positive = false_positive = true_negative = false_negative = format_failures = 0
    for gold, prediction, _score in results:
        expected = bool(gold.appropriate)
        output = _prediction_output(prediction)
        if output is None:
            format_failures += 1
            predicted = not expected
        else:
            predicted = output.appropriate

        if expected and predicted:
            true_positive += 1
        elif not expected and predicted:
            false_positive += 1
        elif not expected and not predicted:
            true_negative += 1
        else:
            false_negative += 1

    total = len(results)
    correct = true_positive + true_negative
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        "total": total,
        "accuracy": _safe_divide(correct, total),
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "format_failures": format_failures,
    }


def run_optimization(config: EvaluationConfig) -> Path:
    """Run DSPy GEPA optimization and return the artifact directory."""
    _validate_reflection_credentials(config)
    dataset = load_dataset(config)
    training_examples = dataframe_to_examples(dataset["train"])
    validation_examples = dataframe_to_examples(dataset["validation"])
    test_examples = dataframe_to_examples(dataset["test"])
    if not training_examples:
        raise ValueError("The combined training split is empty")
    if not validation_examples:
        raise ValueError("The combined validation split is empty")
    if not test_examples:
        raise ValueError("The combined test split is empty")

    run_dir = create_run_directory(config)
    gepa_log_dir = run_dir / "dspy_gepa"
    gepa_log_dir.mkdir()

    optimization_log = run_dir / "optimization.log"
    configure_logging(optimization_log)
    with capture_console_output(optimization_log):
        vllm = VLLMServer(
            model_dir=str(config.models.task_model.path / config.models.task_model.name),
            served_model_name=config.models.task_model.name,
            port=config.vllm.port,
            max_model_len=config.vllm.max_model_len,
            max_batch_size=config.vllm.max_batch_size,
            gpu_memory_utilization=config.vllm.gpu_memory_utilization,
            apply_wsl_workaround=config.vllm.wsl_hack,
            log_path=optimization_log,
        )

        vllm.start()

        try:
            task_lm = dspy.LM(
                model=f"openai/{config.models.task_model.name}",
                api_key="empty",
                api_base=f"http://127.0.0.1:{config.vllm.port}/v1",
                max_tokens=config.gepa.task_max_tokens,
            )
            reflection_lm = dspy.LM(
                model=f"openai/{config.models.optimizer_model.end_point}",
                api_key=config.LLM_API_KEY,
                api_base=config.LLM_BASE_URL,
                max_tokens=config.gepa.reflection_max_tokens,
            )
            dspy.configure(lm=task_lm, adapter=dspy.JSONAdapter())

            baseline_program = TaskToolMatcher()
            logger.info("Evaluating the baseline program on %d validation examples", len(validation_examples))
            baseline_metrics = evaluate_program(
                baseline_program,
                validation_examples,
                num_threads=config.gepa.num_threads,
            )

            optimizer = dspy.GEPA(
                metric=gepa_metric,
                auto=config.gepa.auto,
                reflection_minibatch_size=config.gepa.reflection_minibatch_size,
                candidate_selection_strategy=config.gepa.candidate_selection_strategy,
                reflection_lm=reflection_lm,
                skip_perfect_score=config.gepa.skip_perfect_score,
                use_merge=config.gepa.use_merge,
                add_format_failure_as_feedback=True,
                num_threads=config.gepa.num_threads,
                failure_score=0.0,
                log_dir=str(gepa_log_dir),
                track_stats=True,
                seed=config.gepa.seed,
            )
            logger.info(
                "Optimizing on %d training examples with %d validation examples",
                len(training_examples),
                len(validation_examples),
            )
            optimized_program = optimizer.compile(
                baseline_program,
                trainset=training_examples,
                valset=validation_examples,
            )

            logger.info("Evaluating the optimized program")
            optimized_metrics = evaluate_program(
                optimized_program,
                validation_examples,
                num_threads=config.gepa.num_threads,
            )
            logger.info("Evaluating the baseline program on %d test examples", len(test_examples))
            baseline_test_metrics = evaluate_program(
                baseline_program,
                test_examples,
                num_threads=config.gepa.num_threads,
            )
            logger.info("Evaluating the optimized program on %d test examples", len(test_examples))
            optimized_test_metrics = evaluate_program(
                optimized_program,
                test_examples,
                num_threads=config.gepa.num_threads,
            )
            _write_artifacts(
                run_dir,
                config,
                optimized_program,
                training_size=len(training_examples),
                validation_size=len(validation_examples),
                test_size=len(test_examples),
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                baseline_test_metrics=baseline_test_metrics,
                optimized_test_metrics=optimized_test_metrics,
            )
        finally:
            vllm.stop()

        logger.info("Optimization artifacts written to %s", run_dir)
    return run_dir


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    """Validate a nested dataset mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _string(value: object, field_name: str) -> str:
    """Validate a non-empty dataset string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _prediction_output(prediction: object) -> TaskToolMatcherOutput | None:
    """Normalize a DSPy prediction to the task output schema."""
    try:
        reasoning = getattr(prediction, "reasoning")
        appropriate = getattr(prediction, "appropriate")
        return TaskToolMatcherOutput.model_validate({"reasoning": reasoning, "appropriate": appropriate})
    except (AttributeError, TypeError, ValidationError):
        return None


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Divide two metric values, returning zero for an empty denominator."""
    return float(numerator / denominator) if denominator else 0.0


def _validate_reflection_credentials(config: EvaluationConfig) -> None:
    """Fail before GPU startup when reflection credentials are unavailable."""
    if not config.LLM_API_KEY.strip():
        raise ValueError("LLM_API_KEY is required for the GEPA reflection model")
    if not config.LLM_BASE_URL.strip():
        raise ValueError("LLM_BASE_URL is required for the GEPA reflection model")


def _optimized_instructions(program: dspy.Module) -> str:
    """Extract the optimized instruction from the compiled matcher."""
    matcher = getattr(program, "matcher", None)
    signature = getattr(matcher, "signature", None)
    instructions = getattr(signature, "instructions", None)
    if not isinstance(instructions, str) or not instructions.strip():
        raise RuntimeError("The optimized DSPy program has no matcher instructions")
    return instructions


def _write_artifacts(
    run_dir: Path,
    config: EvaluationConfig,
    optimized_program: dspy.Module,
    *,
    training_size: int,
    validation_size: int,
    test_size: int,
    baseline_metrics: dict[str, int | float],
    optimized_metrics: dict[str, int | float],
    baseline_test_metrics: dict[str, int | float],
    optimized_test_metrics: dict[str, int | float],
) -> None:
    """Persist the optimized program, instruction, and evaluation metrics."""
    optimized_program.save(str(run_dir / "optimized_program.json"))
    (run_dir / "optimized_prompt.txt").write_text(
        _optimized_instructions(optimized_program) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "experiment": config.experiment.name,
        "models": {
            "task": config.models.task_model.name,
            "reflection": config.models.optimizer_model.end_point,
        },
        "dataset": {
            "train_size": training_size,
            "validation_size": validation_size,
            "test_size": test_size,
        },
        "gepa": config.gepa.model_dump(mode="json"),
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "accuracy_delta": float(optimized_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "test": {
            "baseline": baseline_test_metrics,
            "optimized": optimized_test_metrics,
            "accuracy_delta": float(optimized_test_metrics["accuracy"] - baseline_test_metrics["accuracy"]),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run task-tool prompt optimization from the repository configuration."""
    run_optimization(load_config())


if __name__ == "__main__":
    main()
