# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Utility helpers for binary-classification evaluation."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from scipy.stats import DegenerateDataWarning, bootstrap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
) -> dict[str, float]:
    """Compute point-estimate evaluation metrics on the full dataset.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Dictionary mapping metric names to their float values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "f1_score": float(f1_score(y_true, y_pred, average="binary")),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    }


def bootstrap_classification_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    failed_parses: int = 0,
    iterations: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap point estimates and CIs for binary-classification metrics.

    All six metrics are derived from a single confusion matrix per resample
    computed vectorized over the bootstrap axis, so one ``scipy.stats.bootstrap``
    call yields jointly consistent intervals. The CI method is BCa, which
    corrects for skew/bias and is preferred over the plain percentile method
    for proportions bounded in [0, 1].

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        failed_parses: Number of parse failures to count as deterministic
            incorrect predictions for end-to-end accuracy.
        iterations: Number of bootstrap resampling iterations.
        confidence_level: Two-sided confidence level (e.g. 0.95).
        seed: Seed for ``numpy.random.default_rng``.

    Returns:
        Dictionary mapping each metric name to a dict with keys ``"value"``,
        ``"low"`` and ``"high"``.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_pred = np.asarray(y_pred).astype(np.int8)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    if failed_parses < 0:
        raise ValueError("failed_parses must be >= 0")
    if y_true.size == 0:
        if failed_parses == 0:
            raise ValueError("Cannot evaluate an empty dataset")
        empty = {"value": 0.0, "low": 0.0, "high": 0.0}
        return {
            name: dict(empty)
            for name in (
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate",
                "false_negative_rate",
                "end_to_end_accuracy",
            )
        }

    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "false_negative_rate",
    )

    def _metrics(yt: np.ndarray, yp: np.ndarray, axis: int = -1) -> np.ndarray:
        tp = np.sum((yt == 1) & (yp == 1), axis=axis)
        fp = np.sum((yt == 0) & (yp == 1), axis=axis)
        tn = np.sum((yt == 0) & (yp == 0), axis=axis)
        fn = np.sum((yt == 1) & (yp == 0), axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            denom_f1 = 2 * tp + fp + fn
            f1 = np.where(denom_f1 > 0, 2 * tp / denom_f1, 0.0)
            fpr = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)
            fnr = np.where((fn + tp) > 0, fn / (fn + tp), 0.0)

        return np.stack([accuracy, precision, recall, f1, fpr, fnr], axis=0)

    point = _metrics(y_true, y_pred)

    with _ignore_degenerate_bca_warnings():
        res = bootstrap(
            (y_true, y_pred),
            _metrics,
            n_resamples=iterations,
            confidence_level=confidence_level,
            method="bca",
            paired=True,
            vectorized=True,
            random_state=np.random.default_rng(seed),
        )
    low, high = _finite_interval(res, confidence_level)

    return {
        name: {
            "value": float(point[i]),
            "low": float(low[i]),
            "high": float(high[i]),
        }
        for i, name in enumerate(metric_names)
    } | {
        "end_to_end_accuracy": _bootstrap_end_to_end_accuracy(
            y_true,
            y_pred,
            failed_parses=failed_parses,
            iterations=iterations,
            confidence_level=confidence_level,
            seed=seed,
        )
    }


def _bootstrap_end_to_end_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    failed_parses: int,
    iterations: int,
    confidence_level: float,
    seed: int | None,
) -> dict[str, float]:
    """Bootstrap CI for end-to-end accuracy including failed parses.

    Failed parses are represented as fixed zero-correctness examples.
    """
    if failed_parses < 0:
        raise ValueError("failed_parses must be >= 0")

    successful_correct = (y_true == y_pred).astype(float)
    failed_correct = np.zeros(failed_parses, dtype=float)
    correctness = np.concatenate([successful_correct, failed_correct])

    if correctness.size == 0:
        return {"value": 0.0, "low": 0.0, "high": 0.0}

    point = float(np.mean(correctness))
    with _ignore_degenerate_bca_warnings():
        res = bootstrap(
            (correctness,),
            np.mean,
            n_resamples=iterations,
            confidence_level=confidence_level,
            method="bca",
            random_state=np.random.default_rng(seed),
        )
    low, high = _finite_interval(res, confidence_level)
    return {"value": point, "low": float(low), "high": float(high)}


def _finite_interval(result, confidence_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Use percentile bounds where a degenerate BCa interval is undefined."""
    low = np.asarray(result.confidence_interval.low)
    high = np.asarray(result.confidence_interval.high)
    if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
        return low, high
    alpha = (1 - confidence_level) / 2
    fallback_low, fallback_high = np.quantile(result.bootstrap_distribution, [alpha, 1 - alpha], axis=-1)
    return np.where(np.isfinite(low), low, fallback_low), np.where(np.isfinite(high), high, fallback_high)


@contextmanager
def _ignore_degenerate_bca_warnings() -> Iterator[None]:
    """Suppress expected BCa warnings when percentile bounds will be used."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DegenerateDataWarning)
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in .*divide",
            category=RuntimeWarning,
            module=r"scipy\.stats\._resampling",
        )
        yield
