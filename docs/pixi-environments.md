# Pixi Environments

The project uses separate Pixi environments to keep CPU and GPU dependencies isolated and to keep the DSPy and direct GEPA dependency sets independent.

## Shared Features

- `dev` provides development tools such as pytest, Ruff, mypy, pre-commit, and Jupyter.
- `cpu` provides CPU-only PyTorch and Transformers packages for development and testing without a CUDA-capable GPU.
- `gpu` provides CUDA, CUDA-enabled PyTorch, Transformers, and vLLM for GPU-backed model serving and production experiments.
- `dspy` installs DSPy and the exact GEPA dependency selected by DSPy.
- `gepa` installs standalone GEPA without DSPy for the direct GEPA workflow.
- `generation` provides generation tasks on Linux without CPU/GPU model-serving features.
- `paper-gpu` isolates Unsloth training from the existing GPU environment.
- `paper-eval` adds evaluation metrics dependencies to the existing GPU/vLLM environment.

## Environments

| Environment | Features | Purpose |
| --- | --- | --- |
| `default` | `dev`, `gpu` | General GPU-capable development environment with vLLM, but without the DSPy or standalone GEPA experiment packages. |
| `dev` | `dev`, `cpu` | General CPU-only development environment for common tests, static checks, configuration tools, and model downloads. |
| `dspy` | `dev`, `gpu`, `dspy` | Production environment for DSPy-based prompt optimization using a GPU and vLLM. |
| `dspy-cpu` | `dev`, `cpu`, `dspy` | CPU-only environment for DSPy unit tests and related utilities that do not start vLLM. |
| `gepa` | `dev`, `gpu`, `gepa` | Production environment for direct GEPA prompt optimization using a GPU and vLLM. |
| `gepa-cpu` | `dev`, `cpu`, `gepa` | CPU-only environment for direct GEPA unit tests and development without vLLM. |
| `generation` | `dev`, `generation` | API-based data and conversation generation, dry runs, analysis, and generator tests without PyTorch or CUDA. |
| `paper-gpu` | `dev`, `paper-gpu` | SFT, Optuna search, GRPO, and checkpoint merging on CUDA. |
| `paper-eval` | `dev`, `gpu`, `paper-eval` | Paper model and checkpoint evaluation with vLLM and SciPy. |

Run a task with `pixi run <task>`; Pixi selects its configured default environment or the only environment containing it. Use `pixi run -e <environment> <command>` to select an environment explicitly for an ambiguous task or a direct command, or open an interactive shell with `pixi shell -e <environment>`.

The aggregate `pixi run test` task runs the general, DSPy, and direct GEPA test suites in `dev`, `dspy-cpu`, and `gepa-cpu`, respectively, both generation suites in `generation`, and offline paper-component tests in `dev`.

All environments currently target Linux x86-64, matching the original project.
Use `pixi install -e generation --locked` for just the generation workflows.

See the [Pixi task guide](pixi-tasks.md) for the complete task catalog, environment-selection guidance, prerequisites, and side effects.
