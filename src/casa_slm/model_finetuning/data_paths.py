# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bundled curated splits used by fine-tuning and evaluation."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "optimization_and_finetuning"
SFT_TRAIN = ("one_tool_astra_train_curated.json",)
SFT_VALIDATION = ("one_tool_astra_val_curated.json",)
GRPO_TRAIN = ("paired_balanced_train_seed3407.json",)
GRPO_VALIDATION = (
    "three_tools_astra_val_curated.json",
    "two_tools_astra_val_curated.json",
    "cross_mcp_astra_two_tools_val_curated.json",
    "cross_mcp_astra_three_tools_val_curated.json",
)
EVALUATION_TEST = (
    "one_tool_astra_test_curated.json",
    "two_tools_astra_test_curated.json",
    "three_tools_astra_test_curated.json",
    "cross_mcp_astra_two_tools_test_curated.json",
    "cross_mcp_astra_three_tools_test_curated.json",
)
EVALUATION_VALIDATION = ("one_tool_astra_val_curated.json", *GRPO_VALIDATION)


def paths(names: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve bundled data filenames independently of the current directory."""
    return tuple(str(DATA_DIR / name) for name in names)
