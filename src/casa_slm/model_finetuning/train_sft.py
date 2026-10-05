"""Instruction fine-tuning training script."""

import argparse
import json
import logging
import os
import sys
import time
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    from casa_slm.model_finetuning.cli_args import sft_parser

    sft_parser().parse_args()

load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

import unsloth  # noqa: F401 # isort: skip -- must be imported before trl
from datasets import Dataset, concatenate_datasets
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from casa_slm.model_finetuning.cli_args import sft_parser
from casa_slm.model_finetuning.data_paths import SFT_TRAIN, SFT_VALIDATION, paths
from casa_slm.model_finetuning.examples import build_messages
from casa_slm.model_finetuning.prompt import MODEL_IDS, load_prompt, save_run_prompt
from casa_slm.model_finetuning.training_common import (
    build_model_and_tokenizer,
    build_run_name,
    evaluate_metrics_on_validation,
    load_json_dataset,
    save_merged_model,
    strip_bos,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEQ_LENGTH = 1024
EXP_ID = "sft-1tool"
SEED = 3407
VAL_PER_FILE = None  # cap per source file; set to None to disable
VAL_SEED = SEED
ENABLE_GENERATED_VALIDATION_METRICS = False
OUTPUT_DIR = "outputs/sft"
TRAIN_NUM_PROC = 2

LORA_R = 32
LORA_ALPHA = LORA_R * 2


def _model_slug(base_model: str) -> str:
    """Create a filesystem-safe model slug for run naming."""
    return os.path.basename(base_model.rstrip("/"))


class _LogMetricsCallback(TrainerCallback):
    """Forward Trainer metrics to this module logger for file persistence."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        compact = {k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()}
        logger.info("step=%s %s", state.global_step, json.dumps(compact, default=str))


def _build_static_config(resume_from_checkpoint: str | None, base_model: str) -> dict:
    """Snapshot module-level constants so each run logs its full configuration."""
    return {
        "exp_id": EXP_ID,
        "base_model": base_model,
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
        },
        "val_per_file": VAL_PER_FILE,
        "val_seed": VAL_SEED,
        "enable_generated_validation_metrics": ENABLE_GENERATED_VALIDATION_METRICS,
        "output_dir": OUTPUT_DIR,
        "train_num_proc": TRAIN_NUM_PROC,
        "resume_from_checkpoint": resume_from_checkpoint,
        "train_path": list(train_path),
        "val_path": list(val_path),
    }


def _dump_effective_training_args(trainer: SFTTrainer, run_dir: str) -> None:
    """Log and persist the live TrainingArguments used by the trainer.

    Unsloth/TRL patches can silently rewrite fields (batch sizing, optimizer
    flags). Dumping ``trainer.args`` makes the actual values visible in the
    run log and on disk for reproducibility.
    """
    try:
        effective = trainer.args.to_dict()
    except Exception:
        logger.warning("Could not serialize TrainingArguments", exc_info=True)
        return

    logger.info(
        "Effective TrainingArguments (post-Unsloth/TRL patches):\n%s",
        json.dumps(effective, indent=2, default=str, sort_keys=True),
    )
    out_path = os.path.join(run_dir, "effective_config.json")
    try:
        with open(out_path, "w") as f:
            json.dump(effective, f, indent=2, default=str, sort_keys=True)
        logger.info("Wrote effective TrainingArguments to %s", out_path)
    except Exception:
        logger.warning("Failed to write effective config to %s", out_path, exc_info=True)


def _write_tensorboard_scalars(run_dir: str, metrics: dict[str, float], step: int) -> None:
    """Write generated-validation metrics to TensorBoard."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(log_dir=run_dir) as writer:
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)
        logger.info("Wrote generated validation metrics to TensorBoard at step %d", step)
    except Exception:
        logger.warning("Failed to write generated validation metrics to TensorBoard", exc_info=True)


class _GeneratedValidationMetricsCallback(TrainerCallback):
    """Generate validation metrics at every Trainer eval step."""

    def __init__(
        self,
        *,
        tokenizer,
        run_dir: str,
        val_path: tuple[str, ...],
        max_seq_length: int,
        seed: int,
        prompt_template: str,
        batch_size: int = 16,
    ) -> None:
        self.tokenizer = tokenizer
        self.run_dir = run_dir
        self.val_path = val_path
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.prompt_template = prompt_template
        self.batch_size = batch_size

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            logger.warning("Skipping generated validation metrics because callback did not receive a model")
            return

        validation_metrics = evaluate_metrics_on_validation(
            model,
            self.tokenizer,
            self.max_seq_length,
            val_path=self.val_path,
            batch_size=self.batch_size,
            seed=self.seed,
            prompt_template=self.prompt_template,
        )
        tensorboard_metrics = {
            "eval/f1": validation_metrics["f1_score"],
            "eval/fpr": validation_metrics["false_positive_rate"],
            "eval/fnr": validation_metrics["false_negative_rate"],
            "eval/parse_rate": validation_metrics["parse_rate"],
        }
        logger.info(
            "Generated validation metrics at step=%s: %s",
            state.global_step,
            json.dumps(tensorboard_metrics, indent=2, sort_keys=True),
        )
        _write_tensorboard_scalars(self.run_dir, tensorboard_metrics, state.global_step)
        self._append_metrics(state, validation_metrics)

    def _append_metrics(self, state, validation_metrics: dict[str, float]) -> None:
        out_path = os.path.join(self.run_dir, "validation_generation_metrics.jsonl")
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "metrics": validation_metrics,
        }
        try:
            with open(out_path, "a") as f:
                json.dump(record, f, default=str, sort_keys=True)
                f.write("\n")
        except Exception:
            logger.warning("Failed to append generated validation metrics to %s", out_path, exc_info=True)


# Data
train_path: tuple[str, ...] = paths(SFT_TRAIN)
val_path: tuple[str, ...] = paths(SFT_VALIDATION)


def _load_and_tokenize(paths: tuple[str, ...], tokenize_fn) -> Dataset:
    """Load and tokenize one or more source files."""
    if not paths:
        raise ValueError("train_path must include at least one dataset file")
    raw = concatenate_datasets([load_json_dataset(p) for p in paths])
    return raw.map(
        tokenize_fn,
        batched=True,
        num_proc=TRAIN_NUM_PROC,
        remove_columns=raw.column_names,
    )


def build_train_dataset(
    *,
    tokenize_fn,
) -> Dataset:
    """Build the train dataset by concatenating all files in train_path."""
    train_dataset = _load_and_tokenize(train_path, tokenize_fn)
    logger.info("Tokenized train dataset size: %d", len(train_dataset))
    return train_dataset


def format_and_tokenize(examples: dict, *, tokenizer, prompt_template: str | None = None) -> dict:
    """Format examples as chat and tokenize for training."""
    all_input_ids = []
    all_attention_mask = []

    for i in range(len(examples["request"])):
        example = {k: examples[k][i] for k in examples}
        messages = build_messages(example, prompt_template=prompt_template)

        text = strip_bos(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ),
            tokenizer,
        )

        # Tokenize explicitly to avoid Unsloth's aggressive num_proc override in SFTTrainer.
        # Gemma3 uses a Processor API; pass text= explicitly.
        enc = tokenizer(
            text=text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_attention_mask=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # Gemma3 processor may return a batch dimension for text inputs
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask}


def main() -> None:
    """Run the instruction fine-tuning pipeline."""
    args = parse_args()
    resume_from_checkpoint = args.resume_from_checkpoint
    base_model = MODEL_IDS[args.profile]
    prompt_template = load_prompt(
        args.profile,
        Path(args.prompt_file) if args.prompt_file else None,
        Path(resume_from_checkpoint) if resume_from_checkpoint else None,
    )
    os.environ["PYTHONHASHSEED"] = str(SEED)
    set_seed(SEED)

    t_start = time.monotonic()
    if resume_from_checkpoint:
        # Continue inside the checkpoint's existing run dir so checkpoint
        # numbering and TensorBoard event files stay contiguous.
        run_dir = os.path.dirname(os.path.abspath(resume_from_checkpoint))
        run_name = os.path.basename(run_dir)
    else:
        run_name = build_run_name(f"{EXP_ID}--{_model_slug(base_model)}")
        run_dir = os.path.join(OUTPUT_DIR, run_name)
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
        logger.info("Starting instruction FT run: %s", run_name)
        logger.info("Run dir: %s", run_dir)
        logger.info("Log file: %s", log_path)
        logger.info("Reproducibility seed: %d", SEED)
        logger.info("Training data files: %s", json.dumps(list(train_path), indent=2))
        logger.info("Validation data files: %s", json.dumps(list(val_path), indent=2))
        logger.info("Generated validation metrics enabled: %s", ENABLE_GENERATED_VALIDATION_METRICS)
        logger.info(
            "Static config: %s",
            json.dumps(_build_static_config(resume_from_checkpoint, base_model), indent=2, default=str),
        )

        model, tokenizer = build_model_and_tokenizer(base_model, MAX_SEQ_LENGTH, lora_r=LORA_R, lora_alpha=LORA_ALPHA)

        tokenize_fn = partial(format_and_tokenize, tokenizer=tokenizer, prompt_template=prompt_template)

        val_parts = []
        for p in val_path:
            ds = load_json_dataset(p)
            if VAL_PER_FILE is not None and len(ds) > VAL_PER_FILE:
                ds = ds.shuffle(seed=VAL_SEED).select(range(VAL_PER_FILE))
            val_parts.append(ds)
        val_ds = concatenate_datasets(val_parts)

        train_dataset = build_train_dataset(tokenize_fn=tokenize_fn)
        val_dataset = val_ds.map(tokenize_fn, batched=True, num_proc=TRAIN_NUM_PROC, remove_columns=val_ds.column_names)

        logger.info("Train size: %d", len(train_dataset))
        logger.info("Val size: %d", len(val_dataset))

        callbacks: list[TrainerCallback] = [_LogMetricsCallback()]
        if ENABLE_GENERATED_VALIDATION_METRICS:
            callbacks.append(
                _GeneratedValidationMetricsCallback(
                    tokenizer=tokenizer,
                    run_dir=run_dir,
                    val_path=val_path,
                    max_seq_length=MAX_SEQ_LENGTH,
                    seed=SEED,
                    prompt_template=prompt_template,
                )
            )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            args=SFTConfig(
                max_length=MAX_SEQ_LENGTH,
                packing=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                per_device_train_batch_size=8,
                gradient_accumulation_steps=8,
                num_train_epochs=3,
                learning_rate=0.00019556165907728997,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                bf16=True,
                optim="adamw_torch",
                weight_decay=0.01,
                seed=SEED,
                data_seed=SEED,
                output_dir=run_dir,
                logging_dir=run_dir,
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=20,
                save_strategy="steps",
                save_steps=20,
                save_total_limit=-1,
                load_best_model_at_end=True,
                run_name=run_name,
                report_to="tensorboard",
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )

        _dump_effective_training_args(trainer, run_dir)

        if resume_from_checkpoint:
            logger.info("Resuming training from checkpoint: %s", resume_from_checkpoint)
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()

        save_merged_model(model, tokenizer, run_name, prompt_template=prompt_template)

        elapsed = time.monotonic() - t_start
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("Total time: %dh %dm %ds", hours, minutes, seconds)
    finally:
        for handler in handlers_to_remove:
            logging.root.removeHandler(handler)
            handler.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    return sft_parser().parse_args()


if __name__ == "__main__":
    main()
