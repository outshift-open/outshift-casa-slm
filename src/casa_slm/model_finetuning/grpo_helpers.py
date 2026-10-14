# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU-testable GRPO sampler and reward functions."""

import json
import logging
import os
from collections.abc import Callable, Sized
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import torch
from openai import OpenAI
from pydantic import BaseModel
from torch.utils.data import Sampler

from casa_slm.model_finetuning.grpo_config import RewardConfig

logger = logging.getLogger(__name__)


def _compute_group_reward_std_metrics(rewards, num_generations: int) -> tuple[float, float]:
    """Return mean per-group reward std and fraction of zero-std groups."""
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive. Got {num_generations}.")
    if rewards.numel() == 0:
        raise ValueError("Cannot compute group reward std metrics for an empty reward tensor.")
    if rewards.numel() % num_generations != 0:
        raise ValueError(
            "Reward tensor length must be divisible by num_generations. "
            f"Got length={rewards.numel()} num_generations={num_generations}."
        )

    group_std = rewards.view(-1, num_generations).std(dim=1)
    zero_std = group_std.isclose(group_std.new_zeros(group_std.shape))
    return group_std.mean().item(), zero_std.float().mean().item()


class BalancedRepeatSampler(Sampler):
    """RepeatSampler variant that yields class-balanced generation batches.

    Mirrors TRL's ``RepeatSampler`` (mini-repeat each prompt ``num_generations``
    times, repeat each generation batch ``num_iterations * steps_per_generation``
    times) but constructs each generation batch so that, per ``batch_size``
    distinct prompts, half carry a positive label and half a negative label.

    Positives and negatives are shuffled independently each epoch (seeded), then
    drawn ``batch_size // 2`` per class into each batch. When one class runs out
    the remaining slots in a full-size batch are filled from the other class, so
    the trailing batch may be imbalanced rather than dropping data. Only a final
    remainder smaller than ``batch_size`` is dropped, matching the GRPO
    requirement that every generation batch be full so prompt groups stay
    well-formed.
    """

    def __init__(
        self,
        data_source: Sized,
        labels: list[bool],
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        """Initialize the balanced sampler from a dataset and aligned labels."""
        if len(labels) != len(data_source):
            raise ValueError(f"labels length ({len(labels)}) must match dataset length ({len(data_source)}).")
        self.data_source = data_source
        self.labels = labels
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        self.pos_indices = [i for i, label in enumerate(labels) if label]
        self.neg_indices = [i for i, label in enumerate(labels) if not label]

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def _build_chunks(self) -> list[list[int]]:
        if self.shuffle:
            pos = [
                self.pos_indices[i] for i in torch.randperm(len(self.pos_indices), generator=self.generator).tolist()
            ]
            neg = [
                self.neg_indices[i] for i in torch.randperm(len(self.neg_indices), generator=self.generator).tolist()
            ]
        else:
            pos = list(self.pos_indices)
            neg = list(self.neg_indices)

        half = self.batch_size // 2
        chunks: list[list[int]] = []
        pi = ni = 0
        while (len(pos) - pi) + (len(neg) - ni) >= self.batch_size:
            take_pos = min(half, len(pos) - pi)
            take_neg = min(half, len(neg) - ni)
            chunk = pos[pi : pi + take_pos] + neg[ni : ni + take_neg]
            pi += take_pos
            ni += take_neg

            # Fill any deficit (odd batch_size, or a class exhausted) from
            # whichever class still has prompts so the batch stays full size.
            deficit = self.batch_size - len(chunk)
            if deficit > 0:
                extra_pos = min(deficit, len(pos) - pi)
                chunk += pos[pi : pi + extra_pos]
                pi += extra_pos
                deficit -= extra_pos
                extra_neg = min(deficit, len(neg) - ni)
                chunk += neg[ni : ni + extra_neg]
                ni += extra_neg

            # Shuffle within the batch so positives/negatives are interleaved
            # rather than blocked by class order.
            if self.shuffle:
                perm = torch.randperm(len(chunk), generator=self.generator).tolist()
                chunk = [chunk[i] for i in perm]
            chunks.append(chunk)
        return chunks

    def __iter__(self):
        """Yield prompt indices grouped into class-balanced generation batches."""
        for chunk in self._build_chunks():
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        """Return the total number of indices yielded across one full pass."""
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


class ReasoningJudgment(BaseModel):
    """Structured output from the LLM judge."""

    score: int
    justification: str


class ParseResult(NamedTuple):
    """Result of parsing a completion's JSON output."""

    data: dict
    clean: bool  # True when raw output parsed without stripping/fence removal


def parse_completion_json(completion: str) -> ParseResult | None:
    """Parse a completion and return the JSON object with a cleanliness flag."""
    text = completion.strip()
    clean = True
    # Strip markdown code fences the model may wrap around JSON.
    if text.startswith("```"):
        clean = False
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return ParseResult(parsed, clean) if isinstance(parsed, dict) else None


def has_valid_output_schema(result: ParseResult) -> bool:
    """Return whether the parsed JSON matches the expected response schema."""
    return isinstance(result.data.get("reasoning"), str) and isinstance(result.data.get("appropriate"), bool)


def has_exact_output_schema(result: ParseResult) -> bool:
    """Return whether the parsed result has exactly the expected output schema."""
    data = result.data
    return (
        set(data.keys()) == {"reasoning", "appropriate"}
        and isinstance(data["reasoning"], str)
        and bool(data["reasoning"].strip())
        and isinstance(data["appropriate"], bool)
    )


def make_formatting_reward_func(reward_cfg: RewardConfig) -> Callable:
    """Create a reward function that scores completions based on JSON formatting."""

    def formatting_reward_func(completions: list[str], **kwargs) -> list[float]:
        scores = []
        for completion in completions:
            result = parse_completion_json(completion)

            if result is None:
                scores.append(reward_cfg.invalid_json_penalty)
                continue

            if not has_exact_output_schema(result):
                scores.append(reward_cfg.wrong_schema_penalty)
                continue

            if result.clean:
                scores.append(reward_cfg.strict_json_reward)
            else:
                scores.append(reward_cfg.fenced_json_reward)

        return scores

    return formatting_reward_func


def make_correctness_reward_func(reward_cfg: RewardConfig) -> Callable:
    """Reward correct predictions with asymmetric security costs."""

    def correctness_reward_func(
        prompts: list[str],
        completions: list[str],
        answer: list[bool],
        **kwargs,
    ) -> list[float]:
        scores = []
        for completion, ground_truth in zip(completions, answer):
            result = parse_completion_json(completion)
            if result is None or not has_valid_output_schema(result):
                scores.append(reward_cfg.unparseable_correctness_reward)
                continue

            predicted = result.data["appropriate"]
            if predicted and ground_truth:
                scores.append(reward_cfg.true_positive_reward)
            elif not predicted and not ground_truth:
                scores.append(reward_cfg.true_negative_reward)
            elif predicted and not ground_truth:
                scores.append(reward_cfg.false_positive_penalty)
            else:
                scores.append(reward_cfg.false_negative_penalty)
        return scores

    return correctness_reward_func


def make_reasoning_length_penalty_func(reward_cfg: RewardConfig, tokenizer) -> Callable:
    """Penalize reasoning fields that exceed the token budget."""
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    def reasoning_length_penalty_func(
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        scores = []
        for completion in completions:
            result = parse_completion_json(completion)
            if result is None or not has_valid_output_schema(result):
                scores.append(0.0)
                continue

            # Generated reasoning can contain lone Unicode surrogates (or NULs)
            # that json.loads accepts but the Rust fast tokenizer rejects with
            # "TextEncodeInput must be Union[...]". Sanitize to a tokenizer-safe
            # string and fall back to no penalty if encoding still fails, so one
            # malformed completion can't crash a multi-hour training run.
            reasoning = result.data["reasoning"].encode("utf-8", "ignore").decode("utf-8")
            try:
                n_tokens = len(text_tokenizer.encode(reasoning, add_special_tokens=False))
            except Exception:
                logger.warning("reasoning_length_penalty: tokenizer.encode failed; skipping penalty", exc_info=True)
                scores.append(0.0)
                continue

            if n_tokens > reward_cfg.reasoning_max_tokens:
                scores.append(reward_cfg.reasoning_too_long_penalty)
            else:
                scores.append(0.0)

        return scores

    return reasoning_length_penalty_func


_JUDGE_SYSTEM_PROMPT = """\
Score the quality of the model's explanation for the gold label.

The gold label is provided. Do not reward verbosity, restated task text, or generic
claims. Default to a score of 2 unless concrete evidence in the reasoning supports
a higher score.

Score 0-4:
0 - Empty, irrelevant, contradictory, or supports the wrong label.
1 - Vague tautology or restates the task/tool with no real analysis.
2 - Supports the correct label but is generic; does not clearly tie the task to the
    tool's specific capabilities. (Default.)
3 - Supports the correct label and references either the tool's name or its
    description meaningfully, but is missing one of those anchors or the explicit
    task-fit argument.
4 - Supports the correct label, names the tool, references its description, and
    explicitly explains why the task does or does not fit. Reserved for clearly
    specific reasoning. Do NOT award 4 for merely restating the task.

Respond with JSON:
{"score": <0-4>, "justification": "<brief>"}.
"""


def _build_judge_input_message(
    input_prompt: str,
    reasoning: str,
    predicted: bool,
    ground_truth: bool,
) -> str:
    return (
        f"Input Prompt: {input_prompt}\n"
        f"Gold Label: {ground_truth}\n"
        f"Model Prediction: {predicted}\n"
        f"Model Reasoning: {reasoning}"
    )


def make_reasoning_quality_reward_func(reward_cfg: RewardConfig) -> Callable:
    """Reward high-quality reasoning using an LLM judge."""
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ.get("API_BASE"),
    )

    def _judge_single(idx: int, messages: list[dict[str, str]]) -> float:
        try:
            response = client.responses.parse(
                model=reward_cfg.judge_model,
                input=messages,  # type: ignore[arg-type]
                max_output_tokens=reward_cfg.judge_max_tokens,
                text_format=ReasoningJudgment,
            )
            judgment = response.output_parsed
            if judgment is None:
                logger.debug("[judge %d] no parsed response -> 0.0", idx)
                return 0.0
            clamped = max(0, min(reward_cfg.judge_max_score, judgment.score))
            reward = clamped / reward_cfg.judge_max_score * reward_cfg.reasoning_quality_max_reward
            logger.debug(
                "[judge %d] score=%d reward=%.2f | %s",
                idx,
                judgment.score,
                reward,
                judgment.justification[:100],
            )
            return reward
        except Exception:
            logger.warning("LLM judge call failed", exc_info=True)
            return 0.0

    def reasoning_quality_reward_func(
        prompts: list[str],
        completions: list[str],
        answer: list[bool],
        **kwargs,
    ) -> list[float]:
        scores = [0.0] * len(completions)
        tasks: list[tuple[int, list[dict]]] = []

        for i, (prompt, completion, ground_truth) in enumerate(zip(prompts, completions, answer)):
            result = parse_completion_json(completion)
            if result is None or not has_valid_output_schema(result):
                continue

            reasoning: str = result.data["reasoning"]
            predicted: bool = result.data["appropriate"]

            # Hard gate: no reasoning reward for wrong answers
            if predicted != ground_truth:
                scores[i] = reward_cfg.reasoning_reward_if_incorrect
                continue

            messages = [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_judge_input_message(
                        input_prompt=prompt,
                        reasoning=reasoning,
                        predicted=predicted,
                        ground_truth=ground_truth,
                    ),
                },
            ]
            tasks.append((i, messages))

        with ThreadPoolExecutor(max_workers=min(reward_cfg.judge_max_workers, len(tasks) or 1)) as pool:
            futures = {pool.submit(_judge_single, idx, msgs): idx for idx, msgs in tasks}
            for future in futures:
                scores[futures[future]] = future.result()
        return scores

    return reasoning_quality_reward_func
