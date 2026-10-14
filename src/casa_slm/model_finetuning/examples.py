# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Task-tool conversation formatting shared by SFT and its search."""

import json

from casa_slm.model_finetuning.prompt import load_prompt, render_prompt


def build_messages(example: dict, *, prompt_template: str | None = None) -> list[dict]:
    """Build a user prompt and labeled assistant response for one curated row."""
    request, label, metadata = example["request"], example["label"], example["metadata"]
    user_content = render_prompt(
        prompt_template or load_prompt(),
        request["task"],
        request["tool"]["name"],
        request["tool"]["description"],
    )
    assistant_content = json.dumps({"reasoning": metadata["label"]["reasoning"], "appropriate": label["relevant"]})
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
