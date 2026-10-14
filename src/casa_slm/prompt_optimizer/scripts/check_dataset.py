# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Validate dataset files and report task-model input token statistics."""

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import dspy
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from casa_slm.prompt_optimizer import TaskModelConfig, load_config
from casa_slm.prompt_optimizer.scripts.dspy import dataframe_to_examples, task_tool_matcher_signature

logger = logging.getLogger(__name__)


class ChatTokenizer(Protocol):
    """Tokenizer operations needed to count a formatted chat prompt."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> Sequence[int]:
        """Apply the model chat template and return token IDs."""
        ...


@dataclass
class FileCheckResult:
    """Validation and token-count results for one dataset file."""

    errors: list[str]
    token_counts: list[int]
    total_rows: int


@dataclass(frozen=True)
class TokenStatistics:
    """Token-count distribution for a non-empty dataset."""

    p50: int
    p95: int
    maximum: int


class PromptTokenCounter:
    """Count tokens in the baseline DSPy task-model prompt for an example."""

    def __init__(self, tokenizer: ChatTokenizer) -> None:
        """Initialize the counter with the configured task-model tokenizer."""
        self._tokenizer = tokenizer
        self._adapter = dspy.ChatAdapter()
        self._signature = task_tool_matcher_signature()

    def count(self, example: dspy.Example) -> int:
        """Return the number of input tokens represented by one dataset row."""
        messages = self._adapter.format(
            signature=self._signature,
            demos=[],
            inputs=dict(example.inputs()),
        )
        token_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        return len(token_ids)


def load_task_tokenizer(task_model: TaskModelConfig, hf_token: str) -> ChatTokenizer:
    """Load the configured task-model tokenizer from disk or Hugging Face.

    Args:
        task_model: Task-model name, Hugging Face ID, and local cache path.
        hf_token: Optional Hugging Face access token.

    Returns:
        The tokenizer used by the configured task model.
    """
    local_model_path = task_model.path / task_model.name
    tokenizer_source: str | Path
    if (local_model_path / "tokenizer_config.json").is_file():
        tokenizer_source = local_model_path
    else:
        tokenizer_source = task_model.hf_model_id

    return cast(
        ChatTokenizer,
        AutoTokenizer.from_pretrained(
            tokenizer_source,
            token=hf_token or None,
            use_fast=True,
        ),
    )


def calculate_token_statistics(token_counts: Sequence[int]) -> TokenStatistics:
    """Calculate integer p50, p95, and maximum token counts.

    The ``higher`` percentile method returns observed row lengths rather than
    interpolated fractional token counts.

    Args:
        token_counts: Token counts for at least one valid dataset row.

    Raises:
        ValueError: If no token counts are supplied.

    Returns:
        Distribution statistics for the supplied counts.
    """
    if not token_counts:
        raise ValueError("Cannot calculate token statistics for an empty sequence.")

    return TokenStatistics(
        p50=int(np.percentile(token_counts, 50, method="higher")),
        p95=int(np.percentile(token_counts, 95, method="higher")),
        maximum=max(token_counts),
    )


def check_file(json_path: Path, token_counter: PromptTokenCounter | None) -> FileCheckResult:
    """Validate a single JSON dataset file and count tokens for valid rows.

    Loads the file as a DataFrame and passes it through ``dataframe_to_examples``
    row-by-row so that broken records are identified precisely.

    Args:
        json_path: Absolute or relative path to the JSON dataset file.
        token_counter: Baseline prompt token counter, or ``None`` when tokenizer
            initialization failed and only validation can be performed.

    Returns:
        Validation errors, successful token counts, and the total row count.
    """
    errors: list[str] = []
    token_counts: list[int] = []

    try:
        dataframe = pd.read_json(json_path)
    except Exception as exc:
        errors.append(f"  [LOAD ERROR] Cannot read file: {exc}")
        return FileCheckResult(errors=errors, token_counts=token_counts, total_rows=0)

    for idx in range(len(dataframe)):
        row_df = dataframe.iloc[[idx]]
        try:
            example = dataframe_to_examples(row_df)[0]
        except ValueError as exc:
            # dataframe_to_examples reports the DataFrame index, not the
            # positional row number, so we surface both for clarity.
            df_index = dataframe.index[idx]
            errors.append(f"  [RECORD ERROR] row position {idx} (DataFrame index {df_index}): {exc}")
            continue

        if token_counter is not None:
            try:
                token_counts.append(token_counter.count(example))
            except Exception as exc:
                df_index = dataframe.index[idx]
                errors.append(
                    f"  [TOKEN ERROR] row position {idx} (DataFrame index {df_index}): Cannot calculate tokens: {exc}"
                )

    return FileCheckResult(errors=errors, token_counts=token_counts, total_rows=len(dataframe))


def main() -> None:
    """Validate every dataset file and report per-dataset token statistics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    config = load_config()
    any_error = False
    token_counter: PromptTokenCounter | None
    try:
        tokenizer = load_task_tokenizer(config.models.task_model, config.HF_TOKEN)
        token_counter = PromptTokenCounter(tokenizer)
    except Exception as exc:
        any_error = True
        token_counter = None
        logger.error(
            "TOKENIZER ERROR  model=%r  hf_model_id=%r: %s",
            config.models.task_model.name,
            config.models.task_model.hf_model_id,
            exc,
        )

    for data_config in config.experiment.dataset:
        dataset_token_counts: list[int] = []
        dataset_total_rows = 0
        split_files: dict[str, list[str]] = {
            "train": data_config.train,
            "validation": data_config.validation,
            "test": data_config.test,
        }
        for split_name, file_paths in split_files.items():
            for file_path in file_paths:
                path = Path(file_path)
                result = check_file(path, token_counter)
                dataset_token_counts.extend(result.token_counts)
                dataset_total_rows += result.total_rows
                if result.errors:
                    any_error = True
                    logger.error(
                        "FAILED  dataset=%r  split=%s  file=%s  (%d error(s))",
                        data_config.name,
                        split_name,
                        path,
                        len(result.errors),
                    )
                    for message in result.errors:
                        logger.error(message)
                else:
                    logger.info(
                        "OK      dataset=%r  split=%s  file=%s",
                        data_config.name,
                        split_name,
                        path,
                    )

        if dataset_token_counts:
            statistics = calculate_token_statistics(dataset_token_counts)
            logger.info(
                "TOKENS  dataset=%r  valid_rows=%d  total_rows=%d - Counts:  p50=%d  p95=%d  max=%d",
                data_config.name,
                len(dataset_token_counts),
                dataset_total_rows,
                statistics.p50,
                statistics.p95,
                statistics.maximum,
            )
        elif token_counter is not None:
            logger.warning(
                "TOKENS  dataset=%r  valid_rows=0  total_rows=%d  no statistics available",
                data_config.name,
                dataset_total_rows,
            )

    if any_error:
        logger.error("Dataset check finished with errors.")
        sys.exit(1)
    else:
        logger.info("All dataset files passed validation.")


if __name__ == "__main__":
    main()
