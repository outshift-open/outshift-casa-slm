# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""GRPO training script for tool-relevance evaluation."""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from dotenv import load_dotenv

if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    from casa_slm.model_finetuning.cli_args import grpo_parser

    grpo_parser().parse_args()

# Reduce CUDA allocator fragmentation; must be set before torch/unsloth import.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

import unsloth  # noqa: F401 # isort: skip -- must be imported before trl
from datasets import concatenate_datasets
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from casa_slm.model_finetuning.cli_args import grpo_parser
from casa_slm.model_finetuning.grpo_config import GRPOTrainingConfig
from casa_slm.model_finetuning.grpo_helpers import (
    BalancedRepeatSampler,
    _compute_group_reward_std_metrics,
    make_correctness_reward_func,
    make_formatting_reward_func,
    make_reasoning_length_penalty_func,
    make_reasoning_quality_reward_func,
)
from casa_slm.model_finetuning.prompt import load_prompt, save_run_prompt
from casa_slm.model_finetuning.training_common import (
    build_model_and_tokenizer,
    build_run_name,
    build_user_prompt,
    load_json_dataset,
    save_merged_model,
    strip_bos,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_prompt(example: dict, tokenizer, *, prompt_template: str | None = None) -> str:
    """Build a chat-formatted prompt from a single example."""
    user_content = build_user_prompt(
        example["request"]["task"],
        example["request"]["tool"]["name"],
        example["request"]["tool"]["description"],
        prompt_template=prompt_template,
    )
    messages = [{"role": "user", "content": user_content}]
    return strip_bos(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
        tokenizer,
    )


def format_for_grpo(examples: dict, tokenizer, *, prompt_template: str | None = None) -> dict:
    """Convert a batch of examples into GRPO prompt/answer pairs."""
    prompts = []
    answers = []
    for i in range(len(examples["request"])):
        example = {k: examples[k][i] for k in examples}
        prompts.append(build_prompt(example, tokenizer, prompt_template=prompt_template))
        answers.append(example["label"]["relevant"])
    return {"prompt": prompts, "answer": answers}


def _prompt_token_length(prompt: str, text_tokenizer) -> int:
    """Count prompt tokens exactly as text passed to GRPO generation."""
    return len(text_tokenizer.encode(prompt, add_special_tokens=False))


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile for compact prompt-length logging."""
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * q)))
    return sorted_values[idx]


def _drop_overlong_prompts(dataset, text_tokenizer, max_prompt_length: int, split_name: str):
    """Drop examples whose final formatted prompt exceeds max_prompt_length."""
    if len(dataset) == 0:
        logger.info("[prompt_length:%s] empty split", split_name)
        return dataset

    lengths = [_prompt_token_length(prompt, text_tokenizer) for prompt in dataset["prompt"]]
    sorted_lengths = sorted(lengths)
    keep_indices = [i for i, n_tokens in enumerate(lengths) if n_tokens <= max_prompt_length]
    dropped_lengths = [n_tokens for n_tokens in lengths if n_tokens > max_prompt_length]

    logger.info(
        "[prompt_length:%s] n=%d min=%d median=%d p95=%d p99=%d max=%d cap=%d",
        split_name,
        len(lengths),
        sorted_lengths[0],
        _percentile(sorted_lengths, 0.50),
        _percentile(sorted_lengths, 0.95),
        _percentile(sorted_lengths, 0.99),
        sorted_lengths[-1],
        max_prompt_length,
    )

    if not dropped_lengths:
        logger.info("[prompt_length:%s] dropped=0", split_name)
        return dataset

    logger.warning(
        "[prompt_length:%s] dropping %d/%d prompts over cap; longest_over_cap=%s",
        split_name,
        len(dropped_lengths),
        len(lengths),
        sorted(dropped_lengths, reverse=True)[:5],
    )
    return dataset.select(keep_indices)


def _count_binary_labels(dataset) -> tuple[int, int]:
    """Return (n_pos, n_neg) for a dataset with boolean ``answer`` labels."""
    n_pos = int(sum(dataset["answer"]))
    n_neg = len(dataset) - n_pos
    return n_pos, n_neg


def balanced_downsample_dataset(dataset, seed: int, split_name: str):
    """Keep all negatives and randomly sample an equal number of positives.

    Operates on a formatted dataset with a boolean ``answer`` column. All
    negative examples are kept; positives are randomly subsampled (without
    replacement) down to the negative count, yielding a 50/50 split. When
    positives are already the minority all of them are kept and the result
    stays imbalanced (logged as a warning).
    """
    n_pos, n_neg = _count_binary_labels(dataset)
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "[data_strategy:balanced_downsample:%s] single-class split (pos=%d neg=%d); leaving unchanged",
            split_name,
            n_pos,
            n_neg,
        )
        return dataset

    answers = dataset["answer"]
    neg_indices = [i for i, label in enumerate(answers) if not label]
    pos_indices = [i for i, label in enumerate(answers) if label]

    k = min(n_pos, n_neg)
    if n_pos < n_neg:
        logger.warning(
            "[data_strategy:balanced_downsample:%s] positives are the minority "
            "(pos=%d neg=%d); keeping all positives, result stays imbalanced",
            split_name,
            n_pos,
            n_neg,
        )
        sampled_pos = pos_indices
    else:
        sampled_pos = random.Random(seed).sample(pos_indices, k=k)

    keep_indices = neg_indices + sampled_pos
    random.Random(seed).shuffle(keep_indices)
    logger.info(
        "[data_strategy:balanced_downsample:%s] pos %d->%d neg %d->%d total %d->%d",
        split_name,
        n_pos,
        len(sampled_pos),
        n_neg,
        len(neg_indices),
        len(dataset),
        len(keep_indices),
    )
    return dataset.select(keep_indices)


class _CustomGRPOTrainer(GRPOTrainer):
    """GRPOTrainer variant with optional balanced batch sampling and reward-std diagnostics."""

    def __init__(self, *args, balanced_batch_sampling: bool = False, **kwargs):
        self._balanced_batch_sampling = balanced_batch_sampling
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        if not self._balanced_batch_sampling:
            return super()._get_train_sampler(dataset)

        if dataset is None:
            dataset = self.train_dataset
        if "answer" not in dataset.column_names:
            logger.warning(
                "balanced_batch_sampling requested but 'answer' column is missing; "
                "falling back to the default RepeatSampler."
            )
            return super()._get_train_sampler(dataset)

        labels = [bool(a) for a in dataset["answer"]]
        batch_size = self.args.generation_batch_size // self.num_generations
        logger.info(
            "[balanced_batch_sampling] enabled: batch_size=%d (pos=%d neg=%d)",
            batch_size,
            sum(labels),
            len(labels) - sum(labels),
        )
        return BalancedRepeatSampler(
            data_source=dataset,
            labels=labels,
            mini_repeat_count=self.num_generations,
            batch_size=batch_size,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # TRL's built-in frac_reward_zero_std follows the active scale_rewards
        # denominator. Keep a raw per-prompt-group diagnostic regardless of
        # whether advantages are scaled by group std or global batch std.
        try:
            rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
            group_std_mean, frac_group_zero_std = _compute_group_reward_std_metrics(
                rewards,
                self.num_generations,
            )
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["reward_group_std"].append(group_std_mean)
            self._metrics[mode]["frac_reward_group_zero_std"].append(frac_group_zero_std)
        except Exception:
            logger.warning("Failed to compute per-group reward std metrics", exc_info=True)

        return rewards_per_func


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


class _LogMetricsCallback(TrainerCallback):
    """Forward the Trainer's per-step log dict through our logger.

    Trainer.log() writes its metric dict via ``transformers.trainer`` (which
    has propagate=False), so the dict never reaches our root file handler.
    This callback re-emits it on our module logger, which goes to both the
    console and the run's .log file.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        compact = {k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()}
        logger.info("step=%s %s", state.global_step, json.dumps(compact, default=str))


def _dump_effective_grpo_config(trainer: GRPOTrainer) -> None:
    """Log and persist the live GRPOConfig used by the trainer.

    Unsloth's patched ``GRPOConfig`` rewrites several fields during
    construction (e.g. ``beta``, ``num_iterations``, batch sizing). Dump the
    actual ``trainer.args`` so silent overrides are visible.
    """
    args = trainer.args
    try:
        effective: dict = args.to_dict()
    except Exception:
        effective = asdict(args)  # type: ignore[arg-type]

    # Diff vs TRL defaults. Read field defaults from the TRL dataclass without
    # instantiating it: instantiating a second ``GRPOConfig`` re-runs
    # ``TrainingArguments.__post_init__`` and mutates the shared accelerate
    # state singletons, breaking ``trainer.train()``.
    overrides: dict = {}
    try:
        import dataclasses as _dc

        from trl import GRPOConfig as _TRLGRPOConfig

        for field in _dc.fields(_TRLGRPOConfig):
            if field.name not in effective:
                continue
            if field.default is not _dc.MISSING:
                default_val = field.default
            elif field.default_factory is not _dc.MISSING:  # type: ignore[misc]
                try:
                    default_val = field.default_factory()  # type: ignore[misc]
                except Exception:
                    continue
            else:
                continue
            ours = effective[field.name]
            if ours != default_val:
                overrides[field.name] = {"ours": ours, "trl_default": default_val}
    except Exception:
        logger.warning("Could not compute GRPOConfig diff vs TRL defaults", exc_info=True)

    logger.info(
        "Effective GRPOConfig (post-Unsloth patches):\n%s",
        json.dumps(effective, indent=2, default=str, sort_keys=True),
    )
    logger.info(
        "GRPOConfig overrides vs TRL defaults:\n%s",
        json.dumps(overrides, indent=2, default=str, sort_keys=True),
    )
    kl_keys = (
        "beta",
        "num_iterations",
        "loss_type",
        "scale_rewards",
        "importance_sampling_level",
        "mask_truncated_completions",
        "epsilon",
        "epsilon_high",
    )
    logger.info(
        "GRPO KL/objective summary: %s",
        ", ".join(f"{k}={effective.get(k)!r}" for k in kl_keys),
    )

    assert trainer.args.output_dir is not None
    out_path = os.path.join(trainer.args.output_dir, "effective_config.json")
    try:
        with open(out_path, "w") as f:
            json.dump(effective, f, indent=2, default=str, sort_keys=True)
        logger.info("Wrote effective GRPOConfig to %s", out_path)
    except Exception:
        logger.warning("Failed to write effective config to %s", out_path, exc_info=True)


def main(cfg: GRPOTrainingConfig | None = None) -> None:
    """Run GRPO training."""
    if cfg is None:
        cfg = GRPOTrainingConfig()
    if not cfg.model_name:
        raise ValueError("GRPO requires --model-path pointing to the selected merged SFT model")

    prompt_template = load_prompt(
        cfg.profile,
        Path(cfg.prompt_file) if cfg.prompt_file else None,
        Path(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else Path(cfg.model_name),
    )

    if cfg.resume_from_checkpoint:
        # Continue inside the checkpoint's existing run dir so checkpoint
        # numbering and TensorBoard event files stay contiguous.
        run_dir = os.path.dirname(os.path.abspath(cfg.resume_from_checkpoint))
        run_name = os.path.basename(run_dir)
    else:
        run_name = build_run_name(cfg.exp_id)
        run_dir = os.path.join(cfg.output_dir, run_name)

    # Set up logging: console + file, both at INFO level. Track the handlers
    # we add so they can be removed on exit; otherwise repeated main() calls
    # (Optuna, notebooks) duplicate log lines per run.
    os.makedirs(run_dir, exist_ok=True)
    save_run_prompt(Path(run_dir), prompt_template)
    log_path = os.path.join(run_dir, "run.log")
    log_fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_fmt)

    handlers_to_remove: list[logging.Handler] = [file_handler]
    logging.root.addHandler(file_handler)
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logging.root.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_fmt)
        logging.root.addHandler(console_handler)
        handlers_to_remove.append(console_handler)
    logging.root.setLevel(logging.INFO)

    try:
        _run(cfg, run_name, run_dir, log_path, prompt_template)
    finally:
        for handler in handlers_to_remove:
            logging.root.removeHandler(handler)
            handler.close()


def _run(cfg: GRPOTrainingConfig, run_name: str, run_dir: str, log_path: str, prompt_template: str) -> None:
    """Body of ``main`` extracted so logging handlers can be cleaned up."""
    logger.info("Starting GRPO run: %s", run_name)
    logger.info("Run dir: %s", run_dir)
    logger.info("Log file: %s", log_path)
    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2))

    model, tokenizer = build_model_and_tokenizer(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        seed=cfg.seed,
        use_gradient_checkpointing="unsloth",
        padding_side="left",
    )

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    logger.info(
        "Tokenizer padding_side=%r pad_token_id=%s",
        text_tokenizer.padding_side,
        text_tokenizer.pad_token_id,
    )

    grpo_format = lambda examples: format_for_grpo(examples, tokenizer, prompt_template=prompt_template)
    columns_to_keep = {"prompt", "answer"}

    def _format_split(dataset):
        return dataset.map(
            grpo_format,
            batched=True,
            remove_columns=[c for c in dataset.column_names if c not in columns_to_keep],
        )

    if not cfg.drop_overlong_prompts:
        logger.warning(
            "Overlong prompt filtering is disabled; generation may exceed max_prompt_length in patched trainers."
        )

    # Build the val split by simply concatenating all val files (no balancing).
    val_dataset = _format_split(concatenate_datasets([load_json_dataset(p) for p in cfg.val_path]))
    if cfg.drop_overlong_prompts:
        val_dataset = _drop_overlong_prompts(val_dataset, text_tokenizer, cfg.max_prompt_length, "val")

    if cfg.data_strategy == "concat_all":
        # Concatenate all train splits and shuffle within the GRPOTrainer.
        train_dataset = _format_split(concatenate_datasets([load_json_dataset(p) for p in cfg.train_path]))
        if cfg.drop_overlong_prompts:
            train_dataset = _drop_overlong_prompts(train_dataset, text_tokenizer, cfg.max_prompt_length, "train")
    elif cfg.data_strategy == "balanced_downsample":
        # Balance each train file independently (keep all negatives, sample an
        # equal number of positives) after overlong filtering, then concat.
        train_parts = []
        for path in cfg.train_path:
            split_name = os.path.splitext(os.path.basename(path))[0]
            part = _format_split(load_json_dataset(path))
            if cfg.drop_overlong_prompts:
                part = _drop_overlong_prompts(part, text_tokenizer, cfg.max_prompt_length, f"train:{split_name}")
            part = balanced_downsample_dataset(part, cfg.seed, split_name)
            train_parts.append(part)
        train_dataset = concatenate_datasets(train_parts)
    else:
        raise ValueError(
            f"Unknown data_strategy={cfg.data_strategy!r}. Expected 'concat_all' or 'balanced_downsample'."
        )

    n_pos_train, n_neg_train = _count_binary_labels(train_dataset)
    logger.info("[label_balance:train] (pos=%d neg=%d)", n_pos_train, n_neg_train)

    logger.info("Train size: %d", len(train_dataset))
    logger.info("Val size: %d", len(val_dataset))
    logger.info("Columns: %s", train_dataset.column_names)

    if (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps) % cfg.num_generations != 0:
        raise ValueError(
            "GRPO training requires per_device_train_batch_size * gradient_accumulation_steps "
            f"to be divisible by num_generations. Got "
            f"{cfg.per_device_train_batch_size} * {cfg.gradient_accumulation_steps} and "
            f"num_generations={cfg.num_generations}."
        )

    if cfg.per_device_eval_batch_size % cfg.num_generations != 0:
        raise ValueError(
            "GRPO evaluation requires per_device_eval_batch_size to be divisible by "
            f"num_generations. Got per_device_eval_batch_size={cfg.per_device_eval_batch_size} "
            f"and num_generations={cfg.num_generations}."
        )

    if cfg.generation_batch_size % cfg.per_device_train_batch_size != 0:
        raise ValueError(
            "GRPO generation_batch_size must be divisible by per_device_train_batch_size "
            f"for single-process chunking. Got generation_batch_size={cfg.generation_batch_size} "
            f"and per_device_train_batch_size={cfg.per_device_train_batch_size}."
        )

    if cfg.generation_batch_size % cfg.num_generations != 0:
        raise ValueError(
            "GRPO generation_batch_size must be divisible by num_generations. "
            f"Got generation_batch_size={cfg.generation_batch_size} and "
            f"num_generations={cfg.num_generations}."
        )

    reward_funcs: list[str | Callable] = [
        make_formatting_reward_func(cfg.reward),
        make_correctness_reward_func(cfg.reward),
        make_reasoning_length_penalty_func(cfg.reward, tokenizer),
    ]
    reward_weights: list[float] = [1.0, 1.0, 1.0]
    if cfg.use_llm_judge:
        # Optional and expensive. Keep this disabled for normal local-reward runs.
        reward_funcs.append(make_reasoning_quality_reward_func(cfg.reward))
        reward_weights.append(1.0)

    training_args = GRPOConfig(
        learning_rate=cfg.learning_rate,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        dataloader_num_workers=4,
        num_train_epochs=cfg.n_epochs,
        max_steps=cfg.max_steps,
        importance_sampling_level=cfg.importance_sampling_level,
        mask_truncated_completions=cfg.mask_truncated_completions,
        loss_type=cfg.loss_type,
        scale_rewards=cfg.scale_rewards,
        beta=cfg.beta,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
        report_to=cfg.report_to,
        run_name=run_name,
        output_dir=run_dir,
        logging_dir=run_dir,
        reward_weights=reward_weights,
        generation_batch_size=cfg.generation_batch_size,
        torch_empty_cache_steps=cfg.torch_empty_cache_steps,
    )

    trainer = _CustomGRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[_LogMetricsCallback()],
        balanced_batch_sampling=cfg.balanced_batch_sampling,
    )

    # Re-assert eval batching after trainer init. Unsloth's patched
    # GRPOTrainer.__init__ may shrink these to fit memory heuristics. The eval
    # dataloader is built lazily, so mutating ``args.per_device_eval_batch_size``
    # here is picked up via the live ``args.eval_batch_size`` property.
    # ``num_generations`` is cached into ``trainer.num_generations`` during
    # init, so we must also patch the cached attribute -- mutating only
    # ``args.num_generations`` would leave generation/sampling at the patched
    # value while logs misleadingly show our intended one.
    grpo_args = trainer.args
    assert isinstance(grpo_args, GRPOConfig)
    grpo_args.per_device_eval_batch_size = cfg.per_device_eval_batch_size
    grpo_args.num_generations = cfg.num_generations
    trainer.num_generations = cfg.num_generations

    logger.info(
        "GRPO batch config: train=%d eval=%d grad_accum=%d generations=%d generation_batch=%s steps_per_generation=%s",
        cfg.per_device_train_batch_size,
        grpo_args.per_device_eval_batch_size,
        cfg.gradient_accumulation_steps,
        grpo_args.num_generations,
        getattr(grpo_args, "generation_batch_size", cfg.generation_batch_size),
        getattr(grpo_args, "steps_per_generation", None),
    )

    _dump_effective_grpo_config(trainer)

    t0 = time.perf_counter()
    if cfg.resume_from_checkpoint:
        logger.info("Resuming training from checkpoint: %s", cfg.resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    else:
        trainer.train()
    train_elapsed = time.perf_counter() - t0

    save_merged_model(model, tokenizer, run_name, prompt_template=prompt_template)
    total_elapsed = time.perf_counter() - t0

    logger.info(
        "Training completed in %.1fs (%.1f min). Total with save: %.1fs (%.1f min).",
        train_elapsed,
        train_elapsed / 60,
        total_elapsed,
        total_elapsed / 60,
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the GRPO run."""
    return grpo_parser().parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = GRPOTrainingConfig(model_name=args.model_path, profile=args.profile, prompt_file=args.prompt_file)
    if args.resume_from_checkpoint is not None:
        cfg = replace(cfg, resume_from_checkpoint=args.resume_from_checkpoint)
    main(cfg)
