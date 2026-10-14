# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Optuna hyperparameter search for the response-only SFT pipeline."""

import argparse
import gc
import json
import sys
from functools import partial
from pathlib import Path

if __name__ == "__main__" and {"-h", "--help"}.intersection(sys.argv[1:]):
    from casa_slm.model_finetuning.cli_args import search_sft_parser

    search_sft_parser().parse_args()

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

import unsloth  # noqa: F401 # isort: skip -- must be imported before trl
import optuna
import torch
from datasets import concatenate_datasets
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from casa_slm.model_finetuning.cli_args import search_sft_parser
from casa_slm.model_finetuning.examples import build_messages
from casa_slm.model_finetuning.prompt import MODEL_IDS, load_prompt, save_run_prompt
from casa_slm.model_finetuning.train_sft import (
    MAX_SEQ_LENGTH,
    train_path,
    val_path,
)
from casa_slm.model_finetuning.training_common import (
    build_model_and_tokenizer,
    build_run_name,
    evaluate_f1_on_validation,
    load_json_dataset,
    strip_bos,
)


def format_and_tokenize_hpo(examples: dict, *, tokenizer, max_seq_length: int, prompt_template: str) -> dict:
    """Tokenise a batch of examples for HPO training."""
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

        enc = tokenizer(
            text=text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
            return_attention_mask=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_mask}


def prepare_datasets(tokenizer, max_seq_length: int, prompt_template: str):
    """Tokenise and return the train and validation datasets.

    Uses the same ``train_path``/``val_path`` files as ``train_instr_ft`` so the
    HPO search optimises on the exact training and validation distributions.
    """
    tokenize_fn = partial(
        format_and_tokenize_hpo, tokenizer=tokenizer, max_seq_length=max_seq_length, prompt_template=prompt_template
    )

    train_ds = concatenate_datasets([load_json_dataset(p) for p in train_path])
    val_ds = concatenate_datasets([load_json_dataset(p) for p in val_path])

    train_dataset = train_ds.map(tokenize_fn, batched=True, num_proc=2, remove_columns=train_ds.column_names)
    val_dataset = val_ds.map(tokenize_fn, batched=True, num_proc=2, remove_columns=val_ds.column_names)

    return train_dataset, val_dataset


def objective(trial: optuna.Trial, args: argparse.Namespace, datasets: tuple) -> float:
    """Train a single trial and return its validation F1 score."""
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha_mult = trial.suggest_categorical("lora_alpha_mult", [1, 2])
    lora_alpha = lora_r * lora_alpha_mult
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
    epochs = trial.suggest_int("num_train_epochs", 3, 5)

    # Fixed training batch config
    batch_size = 8
    grad_accum = 8

    model, tokenizer = build_model_and_tokenizer(
        args.base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        seed=args.seed,
    )
    train_dataset, val_dataset = datasets
    trial_dir = Path(args.output_root) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    save_run_prompt(trial_dir, args.prompt_template)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            max_length=MAX_SEQ_LENGTH,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            output_dir=str(trial_dir),
            seed=args.seed,
            bf16=True,
            optim="adamw_torch",
            report_to="none",
            logging_steps=10,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="no",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    try:
        trainer.train()
        f1 = evaluate_f1_on_validation(
            model,
            tokenizer,
            MAX_SEQ_LENGTH,
            val_path=val_path,
            sample_fraction=args.eval_sample_fraction,
            seed=args.seed,
            prompt_template=args.prompt_template,
        )
        trial.set_user_attr("validation_f1", f1)
        return f1
    finally:
        del trainer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def retrain_best_trial(
    params: dict, datasets: tuple, seed: int, output_root: str, base_model: str, prompt_template: str
) -> None:
    """Retrain the best trial's model with full validation and save it."""
    run_name = build_run_name("OptunaBest")
    lora_r = params["lora_r"]
    lora_alpha = lora_r * params["lora_alpha_mult"]
    model, tokenizer = build_model_and_tokenizer(
        base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        seed=seed,
    )
    train_dataset, val_dataset = datasets
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_prompt(run_dir, prompt_template)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            max_length=MAX_SEQ_LENGTH,
            packing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            output_dir=str(run_dir),
            seed=seed,
            bf16=True,
            optim="adamw_torch",
            report_to="tensorboard",
            run_name=run_name,
            logging_steps=10,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            num_train_epochs=params["num_train_epochs"],
            learning_rate=params["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    trainer.train()

    f1 = evaluate_f1_on_validation(
        model, tokenizer, MAX_SEQ_LENGTH, val_path=val_path, seed=seed, prompt_template=prompt_template
    )
    print(f"Retrained model F1 (full validation): {f1:.4f}")

    model_dir = f"{output_root}/ft-{run_name}-B16"
    model.save_pretrained_merged(model_dir, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(model_dir)
    save_run_prompt(Path(model_dir), prompt_template)
    print(f"Model saved to {model_dir}")


def main():
    """Run the Optuna hyperparameter search."""
    args = search_sft_parser().parse_args()
    args.base_model = MODEL_IDS[args.profile]
    args.prompt_template = load_prompt(args.profile, Path(args.prompt_file) if args.prompt_file else None)

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Pre-tokenize datasets once (tokenizer is the same across all trials)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    datasets = prepare_datasets(tokenizer, MAX_SEQ_LENGTH, args.prompt_template)
    del tokenizer

    # Note: to set up no pruning, pruner=optuna.pruners.NopPruner(), otherwise the default MedianPruner is used.
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )
    study.optimize(lambda trial: objective(trial, args, datasets), n_trials=args.n_trials)

    print("best value:", study.best_value)
    print(json.dumps(study.best_trial.params, indent=2))

    retrain_best_trial(
        study.best_trial.params,
        datasets,
        seed=args.seed,
        output_root=args.output_root,
        base_model=args.base_model,
        prompt_template=args.prompt_template,
    )


if __name__ == "__main__":
    main()
