# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Read dataset from disk and prepare it for training and evaluation."""

from collections.abc import Mapping

import pandas as pd

from .config import EvaluationConfig

__all__ = ["Dataset", "load_dataset"]


type Dataset = dict[str, pd.DataFrame]
type DatasetDict = dict[str, Dataset]


def _sample_balanced_relevance_labels(
    frame: pd.DataFrame,
    split_name: str,
    sample_size: int,
) -> pd.DataFrame:
    """Return an exact-size balanced sample while preserving row order.

    Args:
        frame: Candidate rows for the final split, already shuffled if requested.
        split_name: Split name used to provide context in validation errors.
        sample_size: Exact number of unique rows to return.

    Returns:
        Exactly ``sample_size`` rows with equal label counts for an even size, or
        counts differing by one for an odd size.

    Raises:
        ValueError: If a row does not contain a Boolean ``label.relevant`` value.
    """
    if "label" not in frame.columns:
        raise ValueError(f"Cannot balance split '{split_name}': rows must contain a 'label' column.")

    relevant_labels: list[bool] = []
    for row_number, label in enumerate(frame["label"]):
        if not isinstance(label, Mapping) or not isinstance(label.get("relevant"), bool):
            raise ValueError(
                f"Cannot balance split '{split_name}': row {row_number} must contain a Boolean 'label.relevant' value."
            )
        relevant_labels.append(label["relevant"])

    quotas = {False: sample_size // 2, True: sample_size // 2}
    if sample_size % 2 == 1:
        quotas[relevant_labels[0]] += 1

    label_counts = {False: relevant_labels.count(False), True: relevant_labels.count(True)}
    shortages = [
        f"label {label} requires {quotas[label]} rows but only {label_counts[label]} are available"
        for label in (False, True)
        if label_counts[label] < quotas[label]
    ]
    if shortages:
        raise ValueError(
            f"Cannot sample {sample_size} balanced rows from split '{split_name}': " + "; ".join(shortages)
        )

    selected_counts = {False: 0, True: 0}
    selected_rows: list[bool] = []
    for relevant in relevant_labels:
        selected = selected_counts[relevant] < quotas[relevant]
        selected_rows.append(selected)
        if selected:
            selected_counts[relevant] += 1

    return frame.loc[selected_rows].reset_index(drop=True)


def load_dataset(evaluation_config: EvaluationConfig, *, shuffle: bool | None = None) -> Dataset:
    """Load dataset based on the experiment and evaluation configuration.

    Args:
        evaluation_config (EvaluationConfig): The evaluation configuration containing the experiment and dataset information.
        shuffle (bool | None, optional): Whether to shuffle the dataset splits.
            Defaults to the experiment configuration value when omitted.

    Returns:
        Dataset: The concatenated dataset containing train, validation, and test
            splits. A split with a positive limit contains exactly that many rows,
            balanced across Boolean ``label.relevant`` values.
    """
    if shuffle is None:
        shuffle = evaluation_config.experiment.shuffle

    datasets: DatasetDict = {}
    for data in evaluation_config.experiment.dataset:
        train = pd.concat([pd.read_json(file) for file in data.train], ignore_index=True)
        validation = pd.concat([pd.read_json(file) for file in data.validation], ignore_index=True)
        test = pd.concat([pd.read_json(file) for file in data.test], ignore_index=True)

        # Check that all splits have the same schema
        if not (set(train.columns) == set(validation.columns) == set(test.columns)):
            raise ValueError(
                f"Schema mismatch in dataset '{data.name}': train, validation, and test splits have different columns."
            )

        for split_name, split_frame, limit in (
            ("train", train, data.limits[0]),
            ("validation", validation, data.limits[1]),
            ("test", test, data.limits[2]),
        ):
            if limit > len(split_frame):
                raise ValueError(
                    f"Limit for split '{split_name}' in dataset '{data.name}' "
                    f"({limit}) exceeds available rows ({len(split_frame)})."
                )

        datasets[data.name] = {
            "train": train,
            "validation": validation,
            "test": test,
        }

    concatenated: Dataset = {}
    for split_index, split in enumerate(["train", "validation", "test"]):
        concatenated[split] = pd.concat([datasets[data][split] for data in datasets], ignore_index=True)
        if shuffle:
            concatenated[split] = concatenated[split].sample(frac=1).reset_index(drop=True)
        limits = [data.limits[split_index] for data in evaluation_config.experiment.dataset]
        positive_limits = [limit for limit in limits if limit > 0]
        if positive_limits:
            sample_size = positive_limits[0]
            concatenated[split] = _sample_balanced_relevance_labels(
                concatenated[split],
                split,
                sample_size,
            )

    return concatenated
