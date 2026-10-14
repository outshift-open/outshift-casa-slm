# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Evaluate every LoRA checkpoint in a training run folder.

Serves the shared base model once with all checkpoint adapters registered, then
evaluates each checkpoint sequentially and writes a single metrics-only summary
file into ``<input_folder>/eval_val``. Checkpoint evaluation always uses the validation
split. Only per-checkpoint metrics are kept (no per-item results).
This reuses the evaluation primitives from :mod:`casa_slm.evaluation.run` so the per-item
logic and metrics stay identical to single-model runs.
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from casa_slm.evaluation.run import (
    DEFAULT_CONCURRENCY,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_SEED,
    FULL_MULTI_TOOL_FILES,
    VLLM_HOST,
    VLLM_PORT,
    compute_metrics_report,
    load_data,
    read_adapter_config,
    resolve_checkpoint,
    run_evaluation,
    vllm_server,
)
from casa_slm.model_finetuning.prompt import ModelProfile, load_prompt

logger = logging.getLogger(__name__)

# Split is always the validation set; output subdirectory under the input folder.
SPLIT = "validation"
SPLIT_DIR_NAME = "eval_val"


def _checkpoint_step(path: Path) -> int:
    """Return the integer step encoded in a ``checkpoint-<n>`` directory name."""
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoints(input_folder: Path) -> list[Path]:
    """Return LoRA checkpoint dirs (those with an adapter config), step-ordered."""
    checkpoints = [
        child for child in input_folder.iterdir() if child.is_dir() and (child / "adapter_config.json").exists()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint directories with 'adapter_config.json' found in {input_folder}")
    return sorted(checkpoints, key=_checkpoint_step)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multi-checkpoint evaluation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_folder",
        help="Training run folder containing 'checkpoint-*' subdirectories (e.g. 'outputs/grpo/GRPO-20260709-182208').",
    )
    parser.add_argument("--profile", choices=("1b", "4b"), default="1b")
    parser.add_argument("--prompt-file", help="Override the training run's saved prompt template.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of concurrent in-flight requests to the vLLM server.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampling seed sent with each request for run-to-run repeatability.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="GPU memory utilization fraction for vLLM server (0.0-1.0).",
    )
    return parser.parse_args()


def main(
    input_folder: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    seed: int = DEFAULT_SEED,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    profile: ModelProfile = "1b",
    prompt_file: str | None = None,
) -> None:
    """Evaluate every checkpoint in ``input_folder`` and save per-checkpoint results."""
    folder = Path(input_folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Input folder not found: {folder}")

    checkpoints = find_checkpoints(folder)
    prompt_template = load_prompt(profile, Path(prompt_file) if prompt_file else None, checkpoints[0])
    input_files = FULL_MULTI_TOOL_FILES[SPLIT]
    data = load_data(SPLIT)

    run_start = datetime.now()

    # All checkpoints in a run share one base model; serve it once and register
    # every adapter so we avoid restarting the vLLM server per checkpoint.
    base_models: set[str] = set()
    max_rank = 0
    request_models: dict[Path, str] = {}
    lora_modules: list[str] = []
    for ckpt in checkpoints:
        resolve_checkpoint(ckpt)
        base_model, lora_rank = read_adapter_config(ckpt)
        base_models.add(base_model)
        max_rank = max(max_rank, lora_rank)
        request_model = f"{ckpt.parent.name}-{ckpt.name}"
        request_models[ckpt] = request_model
        lora_modules.append(f"{request_model}={ckpt.resolve()}")

    if len(base_models) > 1:
        raise ValueError(f"Checkpoints reference multiple base models: {sorted(base_models)}")
    serve_model = base_models.pop()

    serve_extra_args = [
        "--enable-lora",
        "--max-lora-rank",
        str(max_rank),
        "--lora-modules",
        *lora_modules,
    ]

    output_dir = folder / SPLIT_DIR_NAME
    print(
        f"Evaluating {len(checkpoints)} checkpoints from '{folder}' on {len(data)} examples "
        f"(split '{SPLIT}', base model '{serve_model}'). Results -> {output_dir}"
    )

    per_checkpoint: list[dict] = []
    with vllm_server(serve_model, gpu_memory_utilization=gpu_memory_utilization, extra_args=serve_extra_args):
        openai_client = OpenAI(base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="token-abc123")
        for i, ckpt in enumerate(checkpoints, start=1):
            request_model = request_models[ckpt]
            print(f"\n=== [{i}/{len(checkpoints)}] Evaluating {request_model} ===")
            start_time = datetime.now()
            y_true, y_pred, failed_parses, _ = run_evaluation(
                data,
                request_model=request_model,
                openai_client=openai_client,
                concurrency=concurrency,
                seed=seed,
                prompt_template=prompt_template,
            )
            print(f"Done {request_model}. Time elapsed: {datetime.now() - start_time}")
            metrics = compute_metrics_report(
                y_true,
                y_pred,
                failed_parses=failed_parses,
                request_model=request_model,
                split=SPLIT,
                checkpoint_dir=ckpt,
            )
            per_checkpoint.append(
                {
                    "model": request_model,
                    "checkpoint": str(ckpt),
                    "step": _checkpoint_step(ckpt),
                    "failed_parses": failed_parses,
                    "metrics": metrics,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{folder.name}_metrics_{timestamp}.json"
    summary = {
        "config": {
            "input_folder": str(folder),
            "input_files": input_files,
            "split": SPLIT,
            "base_model": serve_model,
            "prompt_template": prompt_template,
            "total_examples": len(data),
            "num_checkpoints": len(checkpoints),
        },
        "checkpoints": per_checkpoint,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    total_seconds = int((datetime.now() - run_start).total_seconds())
    hours, minutes = divmod(total_seconds // 60, 60)

    print(f"\nAll {len(checkpoints)} checkpoints evaluated. Metrics summary saved to {output_file}")
    print(f"Total execution time: {hours}h {minutes}m")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("casa_slm.evaluation.run").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    args = parse_args()
    main(
        input_folder=args.input_folder,
        concurrency=args.concurrency,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        profile=args.profile,
        prompt_file=args.prompt_file,
    )
