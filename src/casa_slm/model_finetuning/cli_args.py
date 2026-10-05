"""CPU-safe argument parsers for the GPU fine-tuning commands."""

import argparse


def sft_parser() -> argparse.ArgumentParser:
    """Build the supervised fine-tuning parser."""
    parser = argparse.ArgumentParser(description="Response-only SFT for task-tool matching.")
    parser.add_argument("--profile", choices=("1b", "4b"), default="1b")
    parser.add_argument("--prompt-file", help="Full prompt template, such as a GEPA optimized_prompt.txt.")
    parser.add_argument("--resume-from-checkpoint", help="Resume inside an existing SFT checkpoint's run directory.")
    return parser


def search_sft_parser() -> argparse.ArgumentParser:
    """Build the Optuna SFT search parser."""
    parser = argparse.ArgumentParser(description="Optuna search for response-only SFT.")
    parser.add_argument("--output-root", default="outputs/optuna_instr_ft_2")
    parser.add_argument("--storage", default="sqlite:///outputs/optuna_instr_ft_2.db")
    parser.add_argument("--study-name", default="instr-ft-hpo-2")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--eval-sample-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--profile", choices=("1b", "4b"), default="4b")
    parser.add_argument("--prompt-file", help="Full prompt template, such as a GEPA optimized_prompt.txt.")
    return parser


def grpo_parser() -> argparse.ArgumentParser:
    """Build the GRPO training parser."""
    parser = argparse.ArgumentParser(description="GRPO training for task-tool matching.")
    parser.add_argument("--model-path", required=True, help="Selected merged SFT model directory.")
    parser.add_argument("--profile", choices=("1b", "4b"), default="1b")
    parser.add_argument("--prompt-file", help="Full prompt template, such as a GEPA optimized_prompt.txt.")
    parser.add_argument("--resume-from-checkpoint", help="Resume inside an existing GRPO checkpoint's run directory.")
    return parser


def merge_parser() -> argparse.ArgumentParser:
    """Build the LoRA checkpoint merge parser."""
    parser = argparse.ArgumentParser(description="Merge a selected LoRA checkpoint into a 16-bit model.")
    parser.add_argument("checkpoint_dir", help="A checkpoint-<step> directory containing adapter_config.json.")
    parser.add_argument("--max-seq-length", type=int, default=1280)
    return parser
