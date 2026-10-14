# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclasses for the GRPO training pipeline.

These are intentionally free of heavy imports (unsloth/torch/trl) so that an
orchestrator process (e.g. the Optuna driver in ``optimize_grpo``) can import
the config and build the hyperparameter search space without initializing a
CUDA context.
"""

from dataclasses import dataclass
from typing import Literal

from casa_slm.model_finetuning.data_paths import GRPO_TRAIN, GRPO_VALIDATION, paths


@dataclass(frozen=True)
class RewardConfig:
    """Reward values used by the GRPO reward functions."""

    # Formatting. Fenced output is parseable and schema-correct, so it is
    # rewarded equally with strict JSON: penalizing fences produced a constant
    # signal in early runs (every completion fenced), giving zero gradient
    # while costing reward. Wrong-schema and invalid-JSON are functionally
    # broken for downstream consumers and are penalized equally hard.
    strict_json_reward: float = 0.5
    fenced_json_reward: float = 0.5
    wrong_schema_penalty: float = -1.5
    invalid_json_penalty: float = -1.5

    # Reward for correctness. Two options are documented below; the active
    # values implement the symmetric (F1) option.
    #
    # Option A -- Asymmetric (security-posture) reward:
    #   The security posture (false positives are the highest-risk error:
    #   granting a bad tool is worse than asking the user to retry) is encoded
    #   entirely through the FP/FN penalty asymmetry. TP == TN intentionally:
    #   any conservative bias the trained policy shows should come from the FP
    #   penalty, not from a baked-in "deny is better" prior on correct outcomes
    #   (which would encourage over-refusal). unparseable is set strictly worse
    #   than FN and, after formatting penalty, worse than FP so the model cannot
    #   game the reward by emitting malformed output when uncertain.
    #   Values: TP=1.0, TN=1.0, FP=-6.0, FN=-2.0, unparseable=-5.0
    #
    # Option B -- Symmetric reward, tuned to optimize F1 (ACTIVE):
    #   F1 weights precision and recall equally, so false positives and false
    #   negatives are treated as equally costly errors (no security-posture
    #   asymmetry). TP == TN and FP == FN intentionally: correct outcomes are
    #   rewarded equally and both error types are penalized equally, so the
    #   policy is pushed toward maximizing correct predictions of either class
    #   rather than developing a class bias. unparseable is set strictly worse
    #   than the FP/FN penalty so the model cannot game the reward by emitting
    #   malformed output when uncertain.
    true_positive_reward: float = 1.0
    true_negative_reward: float = 1.0
    false_positive_penalty: float = -1.0
    false_negative_penalty: float = -1.0
    unparseable_correctness_reward: float = -2.0

    # Keep rationales concise. This is a penalty only, not a positive
    # reasoning-quality reward.
    reasoning_max_tokens: int = 160
    reasoning_too_long_penalty: float = -0.3

    # Reasoning quality (LLM judge) -not used for now-
    # Judge uses a 0-4 rubric (see _JUDGE_SYSTEM_PROMPT) to avoid the saturation seen
    # with the previous 0-2 scale, where nearly every completion received the max score.
    judge_model: str = "bedrock/global.anthropic.claude-sonnet-4-6"  # "azure/gpt-5.4"
    judge_max_tokens: int = 256
    judge_max_workers: int = 8
    judge_max_score: int = 4
    reasoning_quality_max_reward: float = 1.0
    reasoning_reward_if_incorrect: float = 0.0


@dataclass(frozen=True)
class GRPOTrainingConfig:
    """Unified configuration for GRPO training."""

    # Model
    model_name: str = ""
    profile: Literal["1b", "4b"] = "1b"
    prompt_file: str | None = None
    max_prompt_length: int = 1024
    max_completion_length: int = 192
    max_seq_length: int = 1280  # >= max_prompt_length (1024) + max_completion_length (192)
    exp_id: str = "GRPO"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    seed: int = 3407

    # Training hyperparameters
    learning_rate: float = 1e-05
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 4
    num_generations: int = 16
    # Explicit cap on the generation batch. 128 keeps the same optimizer-level
    # batch implied by train_batch * grad_accum, but splits rollout generation
    # into smaller chunks so Gemma3 MLP temporaries do not sit on the H100
    # memory ceiling.
    generation_batch_size: int = 128
    temperature: float = 1.1
    scale_rewards: str = "batch"
    # KL coefficient. TRL's documented default is 0.0 (no ref model, no KL
    # term, no KL logging), but Unsloth's patched GRPOConfig silently defaults
    # to 0.001. Set explicitly here to keep the value visible in our config
    # and match the Unsloth default we've been training with.
    beta: float = 0.001
    mask_truncated_completions: bool = True  # recommended to False because of divergence issues
    loss_type: str = "dr_grpo"
    importance_sampling_level: str = "sequence"
    n_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch_fused"
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    bf16: bool = True
    tf32: bool = True

    # Output / logging
    output_dir: str = "outputs/grpo"
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = -1
    eval_strategy: str = "no"
    eval_steps: int = -1
    report_to: str = "tensorboard"
    torch_empty_cache_steps: int = 100

    # Data
    train_path: tuple[str, ...] = paths(GRPO_TRAIN)
    val_path: tuple[str, ...] = paths(GRPO_VALIDATION)
    # Dataset-construction strategy for the train split.
    #   "concat_all"          -> concatenate all train files unchanged (default).
    #   "balanced_downsample" -> per source file, keep ALL negatives and randomly
    #                            sample an equal number of positives, then concat.
    # Balancing is applied per file (after overlong-prompt filtering) so each
    # split's class ratio is corrected independently before concatenation. The
    # val split is always built with "concat_all".
    data_strategy: str = "concat_all"
    drop_overlong_prompts: bool = True
    # When True, generation batches are drawn class-balanced: each generation
    # batch of ``generation_batch_size // num_generations`` distinct prompts is
    # filled half with positive-label and half with negative-label prompts (see
    # ``BalancedRepeatSampler``). When one class is exhausted near the end of an
    # epoch the trailing batch is allowed to be imbalanced rather than dropping
    # data. Ordering is reshuffled each epoch (seeded). Off by default.
    balanced_batch_sampling: bool = True

    # Reward
    reward: RewardConfig = RewardConfig()
    use_llm_judge: bool = False

    # Resume. Path to a saved ``checkpoint-XXXX`` directory to resume from. When
    # set, training continues inside the checkpoint's existing run directory so
    # checkpoint numbering and TensorBoard logs stay contiguous.
    resume_from_checkpoint: str | None = None
