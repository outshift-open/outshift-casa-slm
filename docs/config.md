# Configuration

## Overview

Configuration is managed through two files:
- `eval_config.yaml` - Main evaluation settings
- `.env` - Environment variables and API credentials

## Setup

1. Clone the repository
2. Rename `.env.sample` to `.env`
3. Fill in the required credentials in `.env`

## Environment Variables (`.env`)

Required variables:
- `HF_TOKEN` - Hugging Face token for accessing gated models
- `LLM_API_KEY` - API key for LLM service
- `LLM_BASE_URL` - Base URL for LLM service (default: `https://llm-proxy.prod.outshift.ai/`)

## Evaluation Configuration (`eval_config.yaml`)

### Experiment
- `name` - Experiment name
- `dataset` - List of named dataset configurations.
  Each configuration groups one or more files for the `train`, `validation`, and `test` splits.
  `limits` is optional and contains exactly three sample limits in train/validation/test order; use `0` to include every sample in that split.
- A positive limit selects exactly that many unique rows from the named dataset's pooled split files, balanced across the Boolean `label.relevant` value.
- Even limits use equal label counts.
  Odd limits differ by one, with the first label encountered after optional shuffling receiving the extra row.
- Limited experiments must use exactly one named dataset configuration.
  Group all source files under that entry.
  Multiple named entries are supported when every limit is zero.
- Loading fails when a label does not have enough rows to satisfy the requested balanced quota.

```yaml
experiment:
  name: my-eval-run
  dataset:
    - name: combined
      train:
        - data/primary-train-part-1.jsonl
        - data/primary-train-part-2.jsonl
        - data/supplemental-train.jsonl
      validation:
        - data/primary-validation.jsonl
        - data/supplemental-validation.jsonl
      test:
        - data/primary-test.jsonl
        - data/supplemental-test.jsonl
      limits: [1000, 200, 0]
```

### Job
- `machine` - Machine type (L4, T4, A100, A100_80GB)
- `gpus` - Number of GPUs to use

### Models
- `optimizer_model` - Configuration for the optimizer model (name, endpoint).
  The endpoint is sent as the model ID through the OpenAI-compatible
  `LLM_BASE_URL`; for example, `azure/gpt-5.4` remains that exact proxy model ID.
- `task_model` - Configuration for the task model (name, HF model ID, path). Its
  `name` is used as the stable vLLM served-model name; for example, `baseline`
  is requested through LiteLLM as `openai/baseline`.

### GEPA
- `auto` - Automatic GEPA budget: `light`, `medium`, or `heavy`.
- `reflection_minibatch_size` - Training examples included in each reflection step.
- `candidate_selection_strategy` - Candidate-selection policy shared by the direct and DSPy workflows: `current_best` or `pareto`.
- `skip_perfect_score` - Skip reflection when all examples in a reflection batch receive a perfect score.
- `use_merge` - Enable candidate merging.
  Keep this disabled while optimizing one prompt component.
- `num_threads` - Concurrent task-model evaluations.
  Keep this at or below the vLLM batch capacity.
- `seed` - Random seed used by GEPA.
- `task_max_tokens` and `reflection_max_tokens` - Output token limits for the
  task and reflection models.
- `output_dir` - Root directory for timestamped optimization artifacts.

```yaml
gepa:
  auto: light
  reflection_minibatch_size: 35
  candidate_selection_strategy: current_best
  skip_perfect_score: false
  use_merge: false
  num_threads: 32
  seed: 0
  task_max_tokens: 1024
  reflection_max_tokens: 32000
  output_dir: ./outputs
```

### vLLM
Performance tuning parameters:
- `max_model_len` - Maximum model sequence length
- `max_batch_size` - Maximum batch size
- `gpu_memory_utilization` - GPU memory fraction (0-1)
- `enforce_eager` - Enable for T4 GPUs
- `enable_chunked_prefill` - Enable for better long sequence performance. See [ref](https://docs.vllm.ai/en/v0.8.2/performance/optimization.html#chunked-prefill).
- `max_num_batched_tokens` - Use together with enable_chunked_prefill for better performance on long sequences
- `wsl_hack` - Set true to workaround a networking bug in WSL2 in mirrored mode.

```yaml
llm:
  port: 8080
  max_model_len: 4096
  max_batch_size: 16
  batch_timeout: 0.05
  gpu_memory_utilization: 0.90
  enforce_eager: false
```
