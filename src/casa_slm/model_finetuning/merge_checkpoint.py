# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Merge a single LoRA checkpoint into a standalone 16-bit model.

Given a checkpoint directory produced during training (e.g.
``outputs/grpo/GRPO-20260709-182208/checkpoint-600``), this loads the base model
referenced by the checkpoint's ``adapter_config.json``, applies the LoRA adapter,
and saves the merged model under ``outputs/models/`` using the same naming/format as the
training scripts (``save_merged_model``). The originating run name is preserved and
the model is tagged as the best-by-F1 checkpoint, e.g.
``outputs/models/GRPO-20260709-182208-best-checkpoint-f1-B16``.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from shutil import copy2

from dotenv import load_dotenv

if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    from casa_slm.model_finetuning.cli_args import merge_parser

    merge_parser().parse_args()

load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

import unsloth  # noqa: F401 # isort: skip -- must be imported before transformers/trl
from unsloth import FastModel

from casa_slm.model_finetuning.cli_args import merge_parser
from casa_slm.model_finetuning.training_common import save_merged_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def resolve_run_name(checkpoint_dir: Path) -> str:
    """Build the output run name from a ``.../<run_name>/checkpoint-<step>`` path.

    The parent directory name is the training run name; the merged model is
    tagged as the best-by-F1 checkpoint of that run, e.g.
    ``GRPO-20260709-182208-best-checkpoint-f1``.
    """
    match = re.search(r"checkpoint-(\d+)$", checkpoint_dir.name)
    if match is None:
        raise ValueError(f"Expected a 'checkpoint-<step>' directory, got: {checkpoint_dir.name}")
    run_name = checkpoint_dir.parent.name
    return f"{run_name}-best-checkpoint-f1"


def read_base_model(checkpoint_dir: Path) -> tuple[str, int]:
    """Return ``(base_model_name_or_path, lora_rank)`` from the adapter config."""
    with (checkpoint_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["base_model_name_or_path"], int(cfg.get("r", 16))


def merge_checkpoint(checkpoint_dir: str | Path, max_seq_length: int = 1280) -> str:
    """Load a LoRA checkpoint, merge it into the base model, and save to ``outputs/models/``."""
    ckpt = Path(checkpoint_dir)
    if not (ckpt / "adapter_config.json").exists():
        raise FileNotFoundError(f"No 'adapter_config.json' found in checkpoint directory: {ckpt}")

    base_model_name, lora_rank = read_base_model(ckpt)
    run_name = resolve_run_name(ckpt)
    output_dir = f"outputs/models/{run_name}-B16"

    logger.info("Checkpoint: %s", ckpt)
    logger.info("Base model: %s (lora_r=%d)", base_model_name, lora_rank)
    logger.info("Output dir: %s", output_dir)

    # Loading the checkpoint directory directly makes Unsloth restore the base
    # model referenced in adapter_config.json and attach the saved LoRA adapter.
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(ckpt),
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    save_merged_model(model, tokenizer, run_name)
    source_prompt = ckpt.parent / "prompt.txt"
    if source_prompt.is_file():
        copy2(source_prompt, Path(output_dir) / "prompt.txt")
    logger.info("Saved merged model to %s", output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the checkpoint-merge script."""
    return merge_parser().parse_args()


def main() -> None:
    """Entry point for merging a checkpoint from the command line."""
    args = parse_args()
    merge_checkpoint(args.checkpoint_dir, max_seq_length=args.max_seq_length)


if __name__ == "__main__":
    main()
