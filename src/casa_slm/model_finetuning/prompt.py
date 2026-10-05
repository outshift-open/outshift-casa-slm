"""Prompt templates shared by the paper's fine-tuning and evaluation runs."""

import re
from pathlib import Path
from typing import Literal

ModelProfile = Literal["1b", "4b"]

MODEL_IDS: dict[ModelProfile, str] = {
    "1b": "unsloth/gemma-3-1b-it",
    "4b": "unsloth/gemma-3-4b-it",
}
PROMPT_FILENAMES: dict[ModelProfile, str] = {
    "1b": "gemma_3_1b.txt",
    "4b": "gemma_3_4b.txt",
}
_PLACEHOLDER = re.compile(r"\{(user_task|tool_name|tool_description)\}")


def load_prompt(profile: ModelProfile = "1b", prompt_file: Path | None = None, checkpoint: Path | None = None) -> str:
    """Load an override, a checkpoint's saved prompt, or a bundled profile prompt."""
    if prompt_file is not None:
        path = prompt_file
    elif checkpoint is not None and (checkpoint.parent / "prompt.txt").is_file():
        path = checkpoint.parent / "prompt.txt"
    elif checkpoint is not None and (checkpoint / "prompt.txt").is_file():
        path = checkpoint / "prompt.txt"
    else:
        path = Path(__file__).parent / "prompts" / PROMPT_FILENAMES[profile]
    template = path.read_text(encoding="utf-8").strip()
    fields = set(_PLACEHOLDER.findall(template))
    if fields != {"user_task", "tool_name", "tool_description"}:
        raise ValueError(f"Prompt {path} must contain all three task/tool placeholders")
    return template


def render_prompt(template: str, task: str, tool_name: str, tool_description: str) -> str:
    """Render known fields in one pass, preserving JSON braces and user text."""
    values = {"user_task": task, "tool_name": tool_name, "tool_description": tool_description}
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template).strip()


def save_run_prompt(run_dir: Path, template: str) -> Path:
    """Persist the exact template used by a training run."""
    path = run_dir / "prompt.txt"
    if path.exists() and path.read_text(encoding="utf-8").strip() != template:
        raise ValueError(f"Refusing to change the saved prompt in {run_dir}")
    path.write_text(template + "\n", encoding="utf-8")
    return path
