# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Project configuration options."""

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from .lightning_utils import get_lightning_model_base_folder, is_running_in_lightning

__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "GepaConfig",
    "JobConfig",
    "ModelsConfig",
    "OptimizerModelConfig",
    "TaskModelConfig",
    "VLLMConfig",
    "load_config",
]


class DataConfig(BaseModel):
    """Configuration for a dataset."""

    name: str = Field(..., description="Dataset name")
    train: list[str] = Field(..., description="List of training dataset files")
    validation: list[str] = Field(..., description="List of validation dataset files")
    test: list[str] = Field(..., description="List of test dataset files")
    limits: list[int] = Field(
        default_factory=lambda: [0, 0, 0],
        min_length=3,
        max_length=3,
        description="Number of samples to use for train, validation, and test datasets (0 means use all samples)",
    )


class ExperimentConfig(BaseModel):
    """Configuration for an experiment."""

    name: str = Field(..., description="Experiment name")
    shuffle: bool = Field(False, description="Whether to shuffle dataset splits when loading data")
    dataset: list[DataConfig] = Field(..., description="List of datasets for the experiment")

    @model_validator(mode="after")
    def reject_ambiguous_multi_dataset_limits(self) -> "ExperimentConfig":
        """Reject per-dataset limits that cannot define one combined target."""
        if len(self.dataset) > 1 and any(any(limit > 0 for limit in data.limits) for data in self.dataset):
            raise ValueError(
                "Nonzero dataset limits require exactly one named dataset configuration; "
                "combine its split files under one dataset entry or set every limit to zero."
            )
        return self


class JobConfig(BaseModel):
    """Configuration for a job."""

    machine: str = Field("L4", description="Machine type to run the job on")
    gpus: int = Field(1, description="Number of GPUs to use for the job")


class GepaConfig(BaseModel):
    """Configuration for DSPy GEPA prompt optimization."""

    auto: Literal["light", "medium", "heavy"] = Field("light", description="Automatic GEPA optimization budget")
    reflection_minibatch_size: int = Field(35, gt=0, description="Examples used for each reflection step")
    candidate_selection_strategy: Literal["pareto", "current_best"] = Field(
        "current_best",
        description="Strategy used to select a prompt candidate for reflection",
    )
    skip_perfect_score: bool = Field(False, description="Skip reflection when every minibatch score is perfect")
    use_merge: bool = Field(False, description="Enable GEPA candidate merging")
    num_threads: int = Field(32, gt=0, description="Concurrent task-model evaluations")
    seed: int = Field(0, description="Random seed for GEPA")
    task_max_tokens: int = Field(1024, gt=0, description="Maximum output tokens for the task model")
    reflection_max_tokens: int = Field(32000, gt=0, description="Maximum output tokens for the reflection model")
    output_dir: Path = Field(Path("./outputs"), description="Root directory for optimization artifacts")


class VLLMConfig(BaseModel):
    """Configuration for vLLM."""

    port: int = Field(8080, description="Port for vLLM server")
    max_model_len: int = Field(2048, description="Maximum model length for vLLM")
    max_batch_size: int = Field(32, description="Maximum batch size for vLLM")
    batch_timeout: float = Field(0.05, description="Batch timeout in seconds for vLLM")
    gpu_memory_utilization: float = Field(0.90, description="Fraction of GPU memory to utilize for vLLM")
    enforce_eager: bool = Field(
        False,
        description="Whether to enforce eager execution for vLLM (useful for T4 GPUs)",
    )
    enable_chunked_prefill: bool = Field(
        False,
        description="Whether to enable chunked prefill for vLLM (see vLLM docs)",
    )
    max_num_batched_tokens: int = Field(2048, description="Maximum number of batched tokens for vLLM")
    wsl_hack: bool = Field(False, description="Whether to apply WSL2 hack for vLLM server startup issues")


class OptimizerModelConfig(BaseModel):
    """Configuration for an optimizer model."""

    name: str = Field(..., description="Optimizer model name")
    end_point: str = Field(..., description="API Endpoint for the optimizer model")


class TaskModelConfig(BaseModel):
    """Configuration for a task model."""

    name: str = Field(..., description="Task model name")
    hf_model_id: str = Field(..., description="Hugging Face model ID")
    path: Path = Field(..., description="Path to the model checkpoint")


class ModelsConfig(BaseModel):
    """Configuration for a model."""

    optimizer_model: OptimizerModelConfig = Field(..., description="Optimizer model configuration")
    task_model: TaskModelConfig = Field(..., description="Task model configuration")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    experiment: ExperimentConfig = Field(..., description="Experiment configuration")
    job: JobConfig = Field(..., description="Job configuration")
    gepa: GepaConfig = GepaConfig(
        auto="light",
        reflection_minibatch_size=35,
        candidate_selection_strategy="current_best",
        skip_perfect_score=False,
        use_merge=False,
        num_threads=32,
        seed=0,
        task_max_tokens=1024,
        reflection_max_tokens=32000,
        output_dir=Path("./outputs"),
    )
    models: ModelsConfig = Field(..., description="List of models to evaluate")
    vllm: VLLMConfig = Field(..., description="vLLM configuration for the model")
    HF_TOKEN: str = Field("", description="Hugging Face token for gated models")
    LLM_API_KEY: str = Field("", description="API key for LLM service")
    LLM_BASE_URL: str = Field("", description="Base URL for LLM service")


def load_config(config_path: Path = Path("./eval_config.yaml")) -> EvaluationConfig:
    """Load evaluation configuration from a YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    assert isinstance(config_dict, dict), "Config file must be a dictionary"

    load_dotenv()  # Load environment variables from .env file

    config_dict["HF_TOKEN"] = config_dict.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    config_dict["LLM_API_KEY"] = config_dict.get("LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
    config_dict["LLM_BASE_URL"] = config_dict.get("LLM_BASE_URL", os.getenv("LLM_BASE_URL", ""))

    # when running in Lightning, the path to the model repo is fixed.
    if is_running_in_lightning():
        config_dict["models"]["task_model"]["path"] = str(get_lightning_model_base_folder())

    return EvaluationConfig.model_validate(config_dict)
