# Pixi Tasks

Pixi is the supported interface for installing environments, running checks and tests, validating experiment inputs, and launching prompt optimization.
Task definitions in `pyproject.toml` are the source of truth.
Run `pixi task list` to inspect the tasks available on the current machine, or `pixi task list --json` to obtain machine-readable task metadata.

See [Pixi Environments](pixi-environments.md) for the feature composition and purpose of each environment.

## Running Tasks

Run a task with `pixi run <task>`.
Pixi selects the task's configured default environment or the only environment containing it.
Use `pixi run -e <environment> <task>` when a task exists in more than one environment without a configured default, or when overriding the environment to select CPU or GPU dependencies explicitly.
Arguments following `--` are passed to the underlying command.
For example, `pixi run check-llm -- --config other-config.yaml --timeout 60` uses another configuration file and a longer request timeout.

## Setup and Quality Tasks

| Task | Recommended command | Purpose and notes |
| --- | --- | --- |
| `install` | `pixi run install` | Installs or refreshes all seven environments with `pixi install --all` and may update `pixi.lock` when the manifest has changed. |
| `run-once` | `pixi run run-once` | Runs in `dev`, initializes Git only outside an existing worktree, and installs the pre-commit hooks. |
| `fix` | `pixi run fix` | Runs all pre-commit hooks in `dev` against every tracked file and may rewrite files through formatters and fixers. |
| `check` | `pixi run check` | Runs Ruff, mypy, YAML validation, large-file checks, case-conflict checks, and ShellCheck in `dev` without intentionally rewriting files. |

## Test Tasks

| Task | Recommended command | Purpose and notes |
| --- | --- | --- |
| `test` | `pixi run test` | Runs the complete CPU test matrix, including the paper-component tests. |
| `test-dev` | `pixi run -e dev test-dev` | Runs the general `test/` suite in the CPU development environment, while an unqualified call resolves through the GPU-backed `default` environment. |
| `test-dspy` | `pixi run -e dspy-cpu test-dspy` | Runs `test-dspy/` with CPU PyTorch and DSPy; use `pixi run -e dspy test-dspy` for the GPU dependency set because an unqualified call is ambiguous. |
| `test-gepa` | `pixi run -e gepa-cpu test-gepa` | Runs `test-gepa/` with CPU PyTorch and standalone GEPA; use `pixi run -e gepa test-gepa` for the GPU dependency set because an unqualified call is ambiguous. |
| `test-paper-components` | `pixi run test-paper-components` | Runs offline fine-tuning and evaluation tests. |

## Diagnostics and Utility Tasks

| Task | Recommended command | Purpose and notes |
| --- | --- | --- |
| `cuda-check` | `pixi run cuda-check` | Prints the installed Torch version and `torch.cuda.is_available()` from the GPU-backed `default` environment but still exits successfully when the printed value is `False`. |
| `check-dataset` | `pixi run -e dspy-cpu check-dataset` | Validates configured dataset records and reports prompt-token statistics using the configured paths and tokenizer; use `-e dspy` for the GPU dependency set because an unqualified call is ambiguous. |
| `check-llm` | `pixi run check-llm` | Sends potentially billable test completions through the OpenAI SDK and LiteLLM using the configured optimizer model, base URL, and API key, so it requires network access. |
| `download-model` | `pixi run download-model` | Writes the configured task model to its cache by trying LitModels before Hugging Face and then uploading a successful Hugging Face download to LitModels, which requires network access and may require both services' credentials. |
| `show-config` | `pixi run show-config` | Loads `eval_config.yaml` plus `.env` and prints the configuration, including `HF_TOKEN` and `LLM_API_KEY` verbatim, so its output must never be pasted, published, or retained in logs or reports. |

## Experiment Tasks

| Task | Recommended command | Purpose and notes |
| --- | --- | --- |
| `dspy-optimize` | `pixi run dspy-optimize` | Runs task-tool matcher prompt optimization with DSPy GEPA in `dspy`, requiring configured datasets, task-model files, CUDA, vLLM, and reflection-model credentials while writing timestamped artifacts below the configured output directory. |
| `gepa-optimize` | `pixi run gepa-optimize` | Runs direct GEPA task-tool matcher optimization in `gepa` with the same dataset, model, CUDA, vLLM, credential, and artifact requirements as the DSPy workflow. |

## Generation Tasks

These tasks default to the `generation` environment and share the root `.env`.
Data tasks run from `src/casa_slm/data_generation`; conversation generation and
test tasks run from the repository root. Relative input/output paths follow each
task's working directory. See [Generation workflows](generation.md) for examples.

| Task | Purpose |
| --- | --- |
| `generate-tasks` | Generate synthetic tasks using the configured LLM endpoint. |
| `generate-matches` | Simulate correct, wrong, and null matches offline. |
| `regroup-data` | Regroup curated rows into complete task examples. |
| `extract-mcp-servers` | Extract live tool schemas; requires Docker and relevant service credentials. |
| `generate-conversations` | Generate ASTRA conversations; `--dry-run` validates inputs without LLM calls. |
| `test-data-generation` | Run the data-generation test suite offline. |
| `test-conversation-generation` | Run the conversation-generation test suite offline. |

## Fine-tuning and Evaluation Tasks

Run training and merging tasks in `paper-gpu`; run evaluation tasks in
`paper-eval`. See [Fine-tuning and evaluation](finetuning-evaluation.md) for
their input and output paths.

| Task | Purpose |
| --- | --- |
| `finetune-sft` | Train the response-only SFT model; 1B is the default profile. |
| `search-sft` | Run Optuna SFT search; 4B is the default profile. |
| `finetune-grpo` | Train GRPO from a selected merged SFT model supplied with `--model-path`. |
| `merge-finetune-checkpoint` | Merge a selected LoRA checkpoint and copy its run prompt. |
| `evaluate-model` | Evaluate a base, merged, or adapter model on validation or test data. |
| `evaluate-checkpoints` | Evaluate a training run's checkpoints on validation and save metrics. |

## Operational Notes

Most application tasks load `eval_config.yaml` from the repository root and supplement its credential fields from `.env`.
The `.env` file is ignored by Git and must remain untracked.
Model downloads, optimizer-model checks, and full optimization runs contact external services.
Optimization outputs and downloaded models can be large and are written beneath ignored `outputs/` and `data/models/` paths by default. Curated inputs under `data/` are no longer ignored by Git.
