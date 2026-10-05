"""Optimize the task-tool matching prompt using GEPA directly."""

import json
import logging
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cleanprompt as cp
import litellm
import pandas as pd
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    MergeConfig,
    ReflectionConfig,
    optimize_anything,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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

type AutoBudget = Literal["light", "medium", "heavy"]
type Metrics = dict[str, int | float]
type SideInfo = dict[str, Any]

_AUTO_CANDIDATE_COUNTS: dict[AutoBudget, int] = {
    "light": 6,
    "medium": 12,
    "heavy": 18,
}
_AUTO_FULL_EVAL_STEPS = 5


@dataclass(frozen=True, slots=True)
class TaskToolExample:
    """One task-tool classification example."""

    task: str
    tool_name: str
    tool_description: str
    appropriate: bool
    reference_reasoning: str


class TaskToolMatcherOutput(BaseModel):
    """Output schema for task-tool matcher with reasoning first."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(min_length=1)
    appropriate: bool


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Parsed task-model response and any structured-output failure."""

    output: TaskToolMatcherOutput | None
    raw_response: str
    format_error: str | None = None


initial_preamble = cp.clean("""
    You are an expert who evaluates tool selection.
    Determine if the "Requested Tool" is appropriate for the "User Task".
    - A tool is APPROPRIATE if it helps with ANY part of the task.
    - A tool is NOT APPROPRIATE only if it is completely unrelated.
    Respond in JSON with "reasoning" and "appropriate".
    """)

user_prompt_template = cp.clean("""
    ### User Task
    {user_task}

    ### Requested Tool Name
    {tool_name}

    ### Requested Tool Description
    {tool_description}
    """)

optimization_objective = cp.clean("""
    Optimize the instructional preamble of a single user prompt to maximize exact task-tool
    classification accuracy while always returning valid JSON with a non-empty reasoning string
    and a boolean appropriate value.
    """)

optimization_background = cp.clean("""
    The task model receives exactly one user-role message and no system message.
    The application appends fixed, labeled User Task, Requested Tool Name, and Requested Tool
    Description fields after the optimized preamble.
    Optimize only the preamble, without adding data placeholders or relying on system-role
    instructions.
    The classification boundary is deliberately permissive: a tool is appropriate if it can
    help with any part of the user task, and inappropriate only when it is completely unrelated.
    Preserve the required reasoning-first JSON output contract.
    """)


class TaskModelEvaluator:
    """Evaluate prompt candidates with the task model served by vLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        api_base: str,
        max_tokens: int,
    ) -> None:
        """Initialize the task-model connection.

        Args:
            model_name: LiteLLM model identifier exposed by vLLM.
            api_base: OpenAI-compatible vLLM API base URL.
            max_tokens: Maximum number of output tokens per classification.
        """
        self.model_name = model_name
        self.api_base = api_base
        self.max_tokens = max_tokens

    def evaluate(self, candidate: Mapping[str, str], example: TaskToolExample) -> EvaluationOutcome:
        """Run one candidate prompt against one dataset example."""
        response = litellm.completion(
            model=self.model_name,
            api_key="empty",
            api_base=self.api_base,
            messages=[{"role": "user", "content": _render_user_prompt(candidate, example)}],
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format=_response_format(),
        )
        raw_response = _response_content(response)
        if not raw_response.strip():
            return EvaluationOutcome(
                output=None,
                raw_response=raw_response,
                format_error="The task model returned an empty response.",
            )

        try:
            output = TaskToolMatcherOutput.model_validate_json(raw_response)
        except ValidationError as error:
            return EvaluationOutcome(
                output=None,
                raw_response=raw_response,
                format_error=f"Invalid task-model JSON: {error}",
            )
        return EvaluationOutcome(output=output, raw_response=raw_response)

    def __call__(
        self,
        candidate: Mapping[str, str],
        example: TaskToolExample,
    ) -> tuple[float, SideInfo]:
        """Return GEPA's numeric score and Actionable Side Information."""
        return score_outcome(example, self.evaluate(candidate, example))


def dataframe_to_examples(dataframe: pd.DataFrame) -> list[TaskToolExample]:
    """Convert nested task-tool dataset rows into direct GEPA examples.

    Args:
        dataframe: Dataset split with nested ``request``, ``label``, and
            ``metadata`` columns.

    Returns:
        Validated task-tool examples.

    Raises:
        ValueError: If a row does not conform to the task-tool dataset schema.
    """
    examples: list[TaskToolExample] = []
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
            TaskToolExample(
                task=task,
                tool_name=tool_name,
                tool_description=tool_description,
                appropriate=appropriate,
                reference_reasoning=reference_reasoning,
            )
        )
    return examples


def score_outcome(example: TaskToolExample, outcome: EvaluationOutcome) -> tuple[float, SideInfo]:
    """Score one task-model outcome and provide diagnostic feedback to GEPA."""
    output = outcome.output
    side_info: SideInfo = {
        "task": example.task,
        "requested_tool": {
            "name": example.tool_name,
            "description": example.tool_description,
        },
        "expected_appropriate": example.appropriate,
        "predicted_appropriate": output.appropriate if output is not None else None,
        "reference_reasoning": example.reference_reasoning,
        "model_reasoning": output.reasoning if output is not None else None,
        "raw_response": outcome.raw_response,
        "format_error": outcome.format_error,
    }

    if output is None:
        side_info["feedback"] = (
            "Invalid output. Return JSON with a non-empty string 'reasoning' and a boolean "
            f"'appropriate'. The expected appropriate value was {example.appropriate}."
        )
        return 0.0, side_info

    if output.appropriate == example.appropriate:
        side_info["feedback"] = f"Correct: appropriate={example.appropriate}."
        return 1.0, side_info

    side_info["feedback"] = (
        f"Incorrect classification. Expected appropriate={example.appropriate}, but predicted "
        f"appropriate={output.appropriate}. Reference reasoning: {example.reference_reasoning}"
    )
    return 0.0, side_info


def evaluate_preamble(
    preamble: str,
    examples: list[TaskToolExample],
    evaluator: TaskModelEvaluator,
    *,
    num_threads: int,
) -> Metrics:
    """Evaluate one preamble and calculate binary classification metrics."""
    candidate = {"preamble": preamble}

    def evaluate_example(example: TaskToolExample) -> EvaluationOutcome:
        return evaluator.evaluate(candidate, example)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        outcomes = list(executor.map(evaluate_example, examples))
    return calculate_metrics(examples, outcomes)


def calculate_metrics(
    examples: list[TaskToolExample],
    outcomes: list[EvaluationOutcome],
) -> Metrics:
    """Calculate binary metrics while counting malformed outputs as errors."""
    if len(examples) != len(outcomes):
        raise ValueError("Examples and outcomes must have the same length")

    true_positive = false_positive = true_negative = false_negative = format_failures = 0
    for example, outcome in zip(examples, outcomes, strict=True):
        expected = example.appropriate
        if outcome.output is None:
            format_failures += 1
            predicted = not expected
        else:
            predicted = outcome.output.appropriate

        if expected and predicted:
            true_positive += 1
        elif not expected and predicted:
            false_positive += 1
        elif not expected and not predicted:
            true_negative += 1
        else:
            false_negative += 1

    total = len(outcomes)
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


def auto_metric_budget(auto: AutoBudget, validation_size: int, reflection_minibatch_size: int) -> int:
    """Reproduce DSPy GEPA's automatic budget for one prompt component."""
    if validation_size < 0:
        raise ValueError("validation_size must be non-negative")
    if reflection_minibatch_size <= 0:
        raise ValueError("reflection_minibatch_size must be positive")

    candidate_count = _AUTO_CANDIDATE_COUNTS[auto]
    num_trials = int(max(4 * math.log2(candidate_count), 1.5 * candidate_count))

    total = validation_size
    total += candidate_count * 5
    total += num_trials * reflection_minibatch_size
    if num_trials == 0:
        return total

    periodic_full_evaluations = (num_trials + 1) // _AUTO_FULL_EVAL_STEPS + 1
    extra_final_evaluation = int(num_trials < _AUTO_FULL_EVAL_STEPS)
    total += (periodic_full_evaluations + extra_final_evaluation) * validation_size
    return total


def run_optimization(config: EvaluationConfig) -> Path:
    """Run direct GEPA optimization and return the artifact directory."""
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
    gepa_log_dir = run_dir / "gepa"
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
            evaluator = TaskModelEvaluator(
                model_name=f"openai/{config.models.task_model.name}",
                api_base=f"http://127.0.0.1:{config.vllm.port}/v1",
                max_tokens=config.gepa.task_max_tokens,
            )
            logger.info("Evaluating the baseline prompt on %d validation examples", len(validation_examples))
            baseline_metrics = evaluate_preamble(
                initial_preamble,
                validation_examples,
                evaluator,
                num_threads=config.gepa.num_threads,
            )

            max_metric_calls = auto_metric_budget(
                config.gepa.auto,
                len(validation_examples),
                config.gepa.reflection_minibatch_size,
            )
            gepa_config = _gepa_config(config, gepa_log_dir, max_metric_calls=max_metric_calls)
            logger.info(
                "Optimizing on %d training examples with %d validation examples and a budget of %d metric calls",
                len(training_examples),
                len(validation_examples),
                max_metric_calls,
            )
            result = optimize_anything(
                seed_candidate={"preamble": initial_preamble},
                evaluator=evaluator,
                dataset=training_examples,
                valset=validation_examples,
                objective=optimization_objective,
                background=optimization_background,
                config=gepa_config,
            )
            optimized_preamble = _optimized_preamble(result.best_candidate)

            logger.info("Evaluating the optimized prompt")
            optimized_metrics = evaluate_preamble(
                optimized_preamble,
                validation_examples,
                evaluator,
                num_threads=config.gepa.num_threads,
            )
            logger.info("Evaluating the baseline prompt on %d test examples", len(test_examples))
            baseline_test_metrics = evaluate_preamble(
                initial_preamble,
                test_examples,
                evaluator,
                num_threads=config.gepa.num_threads,
            )
            logger.info("Evaluating the optimized prompt on %d test examples", len(test_examples))
            optimized_test_metrics = evaluate_preamble(
                optimized_preamble,
                test_examples,
                evaluator,
                num_threads=config.gepa.num_threads,
            )
            _write_artifacts(
                run_dir,
                config,
                result,
                optimized_preamble,
                training_size=len(training_examples),
                validation_size=len(validation_examples),
                test_size=len(test_examples),
                max_metric_calls=max_metric_calls,
                baseline_metrics=baseline_metrics,
                optimized_metrics=optimized_metrics,
                baseline_test_metrics=baseline_test_metrics,
                optimized_test_metrics=optimized_test_metrics,
            )
        finally:
            vllm.stop()

        logger.info("Optimization artifacts written to %s", run_dir)
    return run_dir


def _gepa_config(
    config: EvaluationConfig,
    gepa_log_dir: Path,
    *,
    max_metric_calls: int,
) -> GEPAConfig:
    """Build the direct GEPA configuration from project settings."""
    return GEPAConfig(
        engine=EngineConfig(
            run_dir=str(gepa_log_dir),
            seed=config.gepa.seed,
            display_progress_bar=True,
            raise_on_exception=True,
            track_best_outputs=True,
            max_metric_calls=max_metric_calls,
            candidate_selection_strategy=config.gepa.candidate_selection_strategy,
            parallel=True,
            max_workers=config.gepa.num_threads,
        ),
        reflection=ReflectionConfig(
            skip_perfect_score=config.gepa.skip_perfect_score,
            perfect_score=1.0,
            reflection_minibatch_size=config.gepa.reflection_minibatch_size,
            reflection_lm=_openai_proxy_model_name(
                config.models.optimizer_model.end_point,
            ),
            reflection_lm_kwargs={
                "api_key": config.LLM_API_KEY,
                "api_base": config.LLM_BASE_URL,
                "max_tokens": config.gepa.reflection_max_tokens,
            },
        ),
        merge=MergeConfig() if config.gepa.use_merge else None,
    )


def _openai_proxy_model_name(model: str) -> str:
    """Qualify a proxy model for LiteLLM without losing its provider prefix."""
    model = model.strip()
    if not model:
        raise ValueError("The optimizer model endpoint must not be empty")
    if model.startswith("openai/"):
        return model
    return f"openai/{model}"


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


def _candidate_preamble(candidate: Mapping[str, str]) -> str:
    """Extract and validate the optimizable user-prompt preamble."""
    preamble = candidate.get("preamble")
    if not isinstance(preamble, str) or not preamble.strip():
        raise ValueError("GEPA candidate must contain a non-empty 'preamble'")
    return preamble


def _render_user_prompt(candidate: Mapping[str, str], example: TaskToolExample) -> str:
    """Render one complete user prompt without reformatting inserted values."""
    preamble = _candidate_preamble(candidate)
    populated_template = user_prompt_template.format(
        user_task=example.task,
        tool_name=example.tool_name,
        tool_description=example.tool_description,
    )
    return _combine_user_prompt(preamble, populated_template)


def _combine_user_prompt(preamble: str, data_template: str) -> str:
    """Join a preamble and data template with one blank line."""
    return f"{preamble.rstrip()}\n\n{data_template}"


def _response_format() -> dict[str, Any]:
    """Return the OpenAI-compatible strict JSON schema request."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "task_tool_matcher_output",
            "strict": True,
            "schema": TaskToolMatcherOutput.model_json_schema(),
        },
    }


def _response_content(response: object) -> str:
    """Extract text content from a LiteLLM completion response."""
    try:
        choices = getattr(response, "choices")
        message = choices[0].message
        content = message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError("LiteLLM returned an invalid completion response") from error
    if content is None:
        return ""
    if not isinstance(content, str):
        raise RuntimeError("LiteLLM returned non-text completion content")
    return content


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Divide two metric values, returning zero for an empty denominator."""
    return float(numerator / denominator) if denominator else 0.0


def _validate_reflection_credentials(config: EvaluationConfig) -> None:
    """Fail before GPU startup when reflection credentials are unavailable."""
    if not config.LLM_API_KEY.strip():
        raise ValueError("LLM_API_KEY is required for the GEPA reflection model")
    if not config.LLM_BASE_URL.strip():
        raise ValueError("LLM_BASE_URL is required for the GEPA reflection model")


def _optimized_preamble(candidate: object) -> str:
    """Extract the optimized preamble from GEPA's best candidate."""
    if not isinstance(candidate, Mapping):
        raise RuntimeError("GEPA returned a best candidate that is not a mapping")
    preamble = candidate.get("preamble")
    if not isinstance(preamble, str) or not preamble.strip():
        raise RuntimeError("GEPA's best candidate has no preamble")
    return preamble


def _write_artifacts(
    run_dir: Path,
    config: EvaluationConfig,
    result: Any,
    optimized_preamble: str,
    *,
    training_size: int,
    validation_size: int,
    test_size: int,
    max_metric_calls: int,
    baseline_metrics: Metrics,
    optimized_metrics: Metrics,
    baseline_test_metrics: Metrics,
    optimized_test_metrics: Metrics,
) -> None:
    """Persist the direct GEPA result, optimized prompt, and metrics."""
    reusable_prompt = _combine_user_prompt(optimized_preamble, user_prompt_template)
    (run_dir / "optimized_prompt.txt").write_text(reusable_prompt + "\n", encoding="utf-8")
    (run_dir / "gepa_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
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
        "gepa": {
            **config.gepa.model_dump(mode="json"),
            "resolved_max_metric_calls": max_metric_calls,
            "best_candidate_index": int(result.best_idx),
            "candidate_count": int(result.num_candidates),
            "total_metric_calls": result.total_metric_calls,
        },
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
