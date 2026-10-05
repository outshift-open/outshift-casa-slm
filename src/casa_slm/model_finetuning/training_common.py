"""Shared utilities for training scripts."""

import json
from collections.abc import Sequence
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, interleave_datasets

from casa_slm.model_finetuning.data_paths import SFT_VALIDATION, paths
from casa_slm.model_finetuning.prompt import load_prompt, render_prompt, save_run_prompt


def build_user_prompt(
    task: str,
    tool_name: str,
    tool_description: str,
    *,
    prompt_template: str | None = None,
) -> str:
    """Render the selected paper prompt for a task-tool pair."""
    return render_prompt(prompt_template or load_prompt(), task, tool_name, tool_description)


def load_json_dataset(path: str) -> Dataset:
    """Load a JSON file as a HuggingFace Dataset."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return Dataset.from_list(json.load(handle))


DATASET_MIX_STRATEGIES = ("baseline_concat", "balanced_interleave")


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    """Normalize positive interleave weights so they sum to one."""
    total = sum(weights)
    if total <= 0:
        raise ValueError("Interleave weights must sum to a positive value")
    return [w / total for w in weights]


def _build_length_capped_interleave(
    datasets: list[Dataset],
    *,
    probabilities: Sequence[float],
    target_size: int,
    seed: int,
    stopping_strategy: str,
) -> Dataset:
    """Interleave datasets by probability and pin the result to ``target_size``."""
    probs = _normalize_weights(probabilities)
    mixed = interleave_datasets(
        datasets,
        probabilities=probs,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if len(mixed) >= target_size:
        return mixed.select(range(target_size))

    # Repeat deterministically if the interleaved set is shorter than requested.
    repeats = ceil(target_size / len(mixed))
    expanded = concatenate_datasets([mixed] * repeats)
    return expanded.select(range(target_size))


def mix_datasets(
    source_datasets: Sequence[Dataset | None],
    *,
    strategy: str = "baseline_concat",
    probabilities: Sequence[float] | None = None,
    seed: int = 3407,
    stopping_strategy: str = "all_exhausted",
) -> Dataset:
    """Combine tokenized source datasets according to a deterministic mix strategy.

    Empty or ``None`` sources are ignored. Supported strategies:

    - ``baseline_concat``: concatenate all sources in order.
    - ``balanced_interleave``: interleave sources by ``probabilities`` (uniform
      when omitted) and cap the result to the combined source length.
    """
    sources = [ds for ds in source_datasets if ds is not None and len(ds) > 0]
    if not sources:
        raise ValueError("No training data found; all source datasets are empty")

    if strategy == "baseline_concat":
        return concatenate_datasets(sources)

    if strategy == "balanced_interleave":
        weights = list(probabilities) if probabilities is not None else [1.0] * len(sources)
        if len(weights) != len(sources):
            raise ValueError(
                f"balanced_interleave expects one probability per non-empty source; "
                f"got {len(weights)} weights for {len(sources)} sources"
            )
        return _build_length_capped_interleave(
            sources,
            probabilities=weights,
            target_size=sum(len(ds) for ds in sources),
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    raise ValueError(f"Unknown dataset mix strategy={strategy!r}; expected one of: {', '.join(DATASET_MIX_STRATEGIES)}")


def build_run_name(exp_id: str) -> str:
    """Build a timestamped run name from an experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{exp_id}-{timestamp}"


def strip_bos(text: str, tokenizer) -> str:
    """Remove the BOS token prefix from text if present."""
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and text.startswith(bos_token):
        text = text[len(bos_token) :]
    return text


def build_model_and_tokenizer(
    model_name: str = "unsloth/gemma-3-4b-it",
    max_seq_length: int = 1024,
    *,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    seed: int = 3407,
    use_gradient_checkpointing: bool | str = False,
    padding_side: str = "right",
):
    """Load a pretrained model with LoRA adapters and configure the tokenizer.

    ``use_gradient_checkpointing`` is forwarded to Unsloth's ``get_peft_model``.
    Defaults to ``False`` to preserve SFT behavior; pass ``"unsloth"`` to enable
    activation checkpointing for memory-bound runs (e.g. GRPO).

    ``padding_side`` defaults to ``"right"`` (correct for SFT completion-only
    loss). Pass ``"left"`` for GRPO so prompt batching during rollout generation
    does not feed PAD tokens to the model.
    """
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=seed,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma 3 returns a processor wrapping an inner text tokenizer; set padding
    # on both so HF/TRL collators see a consistent value regardless of which
    # object they read.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    text_tokenizer.padding_side = padding_side
    if text_tokenizer is not tokenizer and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = padding_side

    return model, tokenizer


def print_token_stats(texts: Sequence[str], tokenizer, label: str) -> None:
    """Print token length statistics for a list of texts."""
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    lengths = np.array([len(tok.encode(t, add_special_tokens=False)) for t in texts])
    print(f"\n{'=' * 50}")
    print(f"Token stats — {label}  (n={len(lengths)})")
    print(f"  min:    {lengths.min()}")
    print(f"  max:    {lengths.max()}")
    print(f"  mean:   {lengths.mean():.1f}")
    print(f"  std:    {lengths.std():.1f}")
    print(f"  median: {np.median(lengths):.0f}")
    print(f"  p90:    {np.percentile(lengths, 90):.0f}")
    print(f"  p95:    {np.percentile(lengths, 95):.0f}")
    print(f"  p99:    {np.percentile(lengths, 99):.0f}")
    print(f"{'=' * 50}\n")


def save_merged_model(model, tokenizer, run_name: str, *, prompt_template: str | None = None) -> None:
    """Save the model merged to 16-bit format."""
    output_dir = f"outputs/models/{run_name}-B16"
    model.save_pretrained_merged(
        output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    tokenizer.save_pretrained(output_dir)
    if prompt_template is not None:
        save_run_prompt(Path(output_dir), prompt_template)


def extract_prediction(text: str) -> bool | None:
    """Parse a JSON response and return the ``appropriate`` boolean, or ``None``."""
    try:
        data = json.loads(text)
        value = data.get("appropriate")
        return value if isinstance(value, bool) else None
    except Exception:
        return None


def _load_validation_dataset(val_path: str | Sequence[str]) -> Dataset:
    """Load one or more validation JSON files."""
    if isinstance(val_path, str):
        return load_json_dataset(val_path)
    return concatenate_datasets([load_json_dataset(path) for path in val_path])


def evaluate_metrics_on_validation(
    model,
    tokenizer,
    max_seq_length: int,
    *,
    val_path: str | Sequence[str] = paths(SFT_VALIDATION),
    sample_fraction: float = 1.0,
    batch_size: int = 16,
    seed: int = 3407,
    prompt_template: str | None = None,
) -> dict[str, float]:
    """Generate validation predictions and return classification metrics."""
    import random

    import torch
    from tqdm import tqdm

    from casa_slm.evaluation.metrics import get_metrics

    val_ds = _load_validation_dataset(val_path)

    if sample_fraction < 1.0:
        rng = random.Random(seed)
        n = max(1, int(len(val_ds) * sample_fraction))
        indices = rng.sample(range(len(val_ds)), n)
        val_ds = val_ds.select(indices)

    prompts: list[str] = []
    labels: list[bool] = []
    for ex in val_ds:
        prompt = build_user_prompt(
            ex["request"]["task"],
            ex["request"]["tool"]["name"],
            ex["request"]["tool"]["description"],
            prompt_template=prompt_template,
        )
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(chat_text)
        labels.append(bool(ex["label"]["relevant"]))

    # Use the underlying text tokenizer to avoid the unsloth-patched
    # Gemma 3 processor's BatchFeature issue with padded batched text.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    original_padding_side = text_tokenizer.padding_side
    text_tokenizer.padding_side = "left"
    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token_id = text_tokenizer.eos_token_id

    y_true: list[bool] = []
    y_pred: list[bool] = []
    was_training = getattr(model, "training", False)

    try:
        model.eval()
        with torch.inference_mode():
            for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating validation metrics"):
                batch_prompts = prompts[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                inputs = text_tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                )

                for j, (output, prompt_ids) in enumerate(
                    zip(outputs, inputs["input_ids"]),
                ):
                    new_tokens = output[prompt_ids.shape[0] :]
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    pred = extract_prediction(text)
                    if pred is not None:
                        y_true.append(batch_labels[j])
                        y_pred.append(pred)
    finally:
        text_tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()

    total_examples = len(labels)
    parsed_examples = len(y_true)
    if not y_true:
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        }
    else:
        metrics = get_metrics(y_true, y_pred)

    metrics["parsed_examples"] = float(parsed_examples)
    metrics["total_examples"] = float(total_examples)
    metrics["parse_rate"] = float(parsed_examples / total_examples) if total_examples else 0.0
    return metrics


def evaluate_f1_on_validation(
    model,
    tokenizer,
    max_seq_length: int,
    *,
    val_path: str | Sequence[str] = paths(SFT_VALIDATION),
    sample_fraction: float = 1.0,
    batch_size: int = 16,
    seed: int = 3407,
    prompt_template: str | None = None,
) -> float:
    """Generate predictions on the validation set and return the F1 score."""
    return evaluate_metrics_on_validation(
        model,
        tokenizer,
        max_seq_length,
        val_path=val_path,
        sample_fraction=sample_fraction,
        batch_size=batch_size,
        seed=seed,
        prompt_template=prompt_template,
    )["f1_score"]
