[![pytest](https://github.com/outshift-open/outshift-casa-slm/actions/workflows/pytest.yml/badge.svg)](https://github.com/outshift-open/outshift-casa-slm/actions/workflows/pytest.yml) [![pre-commit](https://github.com/outshift-open/outshift-casa-slm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/outshift-open/outshift-casa-slm/actions/workflows/pre-commit.yml)

# CASA-SLM paper

This repository generates task to tool data, optimizes a matcher prompt with
GEPA, fine tunes Gemma with supervised fine tuning (SFT) and group relative
policy optimization (GRPO), and evaluates held out examples. The code uses
Pixi environments so CPU preparation and GPU training share a lockfile.

## Reproduce the main workflow

To manage Python environment we use [pixi](https://pixi.prefix.dev/). Make sure it is installed in your system!
See the [Pixi environments guide](docs/pixi-environments.md) for the purpose and dependency composition of each available environment.

We have defined `pixi` tasks for each step of the workflow. Running the corresponding command will install the needed environment as well. However, make sure that the needed LLM API keys are configured before running the tasks:

```bash
cp -n .env.sample .env
```

Set `LLM_API_KEY` and `LLM_BASE_URL` for synthetic generation and GEPA's
reflection model, and `HF_TOKEN` for gated Gemma downloads. GEPA, SFT, GRPO, and model evaluation require a CUDA machine; the `generation` environment can run on CPU. The repository includes
the curated inputs under [`data/`](data/README.md), so you can start from GEPA
without regenerating or reviewing the data.

| Stage | Pixi task and minimal command | Output |
| --- | --- | --- |
| Generate tasks | `pixi run generate-tasks -- --input-files src/casa_slm/data_generation/mcp_servers/github_official.json src/casa_slm/data_generation/mcp_servers/nasdaq_data_link.json --output-file outputs/generated_tasks.json --num-tools 2 --seed 42` | Synthetic task JSON |
| Generate matches | `pixi run generate-matches -- --config src/casa_slm/data_generation/config_test.json --mcp-server-path src/casa_slm/data_generation/mcp_servers --input outputs/generated_tasks.json --output outputs/generated_matches.json --seed 42` | Candidate task to tool labels |
| Optimize prompt | `pixi run gepa-optimize` | GEPA prompt under `outputs/` |
| Train SFT | `pixi run finetune-sft -- --profile 1b` | Checkpoints under `outputs/sft/` and merged model under `outputs/models/` |
| Train GRPO | `pixi run finetune-grpo -- --model-path outputs/models/<SFT-run-name>-B16` | GRPO checkpoints and merged model |
| Evaluate | `pixi run evaluate-model -- --eval-model outputs/models/<GRPO-run-name>-B16 --split test` | Metrics and item records under `outputs/evaluation/test/` |

The generation example produces raw data for one two tool experiment. Paper
curation is a separate review step; the bundled curated JSON files provide its
reproducible input to optimization and training. `eval_config.yaml` defines
GEPA's train, validation, and test files and model settings. Finetuning and
evaluation select their datasets in
[`data_paths.py`](src/casa_slm/model_finetuning/data_paths.py). SFT and its
hyperparameter search use one-tool ASTRA train/validation data. GRPO reads the
single `paired_balanced_train_seed3407.json`, built from two-tool and three-tool
ASTRA training files and their cross-MCP variants. Its validation split uses
the corresponding original curated validation files. Model evaluation uses
one-tool, two-tool, and three-tool ASTRA splits, including cross-MCP variants.
These components use ASTRA data. Run the preparation task before GRPO if
the paired-balanced dataset has not been built; add `-- --dry-run` to preview
counts without writing files.

To train with an optimized prompt, pass
`--prompt-file outputs/<GEPA-run>/optimized_prompt.txt` to SFT. The exact GEPA
artifact path depends on the run. Training saves `prompt.txt` with each model;
GRPO loads it from the selected SFT model, and evaluation loads it from the
model or checkpoint. `--prompt-file` overrides the saved prompt. These
components also provide bundled prompts, so GEPA is optional for finetuning
and evaluation.

For checkpoint selection, run
`pixi run evaluate-checkpoints -- outputs/sft/<run-name>` on
validation data, then merge a selected adapter with
`pixi run merge-finetune-checkpoint -- outputs/sft/<run-name>/checkpoint-<step>`.
Use its merged model path for GRPO. The example above uses the SFT run's final
merged model.

## Method in brief

GEPA searches over prompt candidates using model feedback and validation scores.
SFT minimizes token cross entropy over assistant response tokens, masking the
user prompt: for reference response tokens, its loss is
`-mean(log p(token | preceding tokens, task, tool))`. GRPO samples a group of
responses per prompt. For each response, it subtracts the group's mean reward
and divides by the batch reward standard deviation (`scale_rewards="batch"`)
to form an advantage. A clipped policy update favors positive advantages while
limiting abrupt policy changes, with a KL penalty (`beta=0.001`) to discourage
drift from the reference model. Here the reward combines output format, answer
correctness, and a length penalty. Evaluation reports classification metrics on parsed
Boolean predictions and an end to end accuracy that counts parse failures as
incorrect. See the [fine tuning and evaluation guide](docs/finetuning-evaluation.md)
for profiles, checkpoint selection, and confidence intervals.

The training objective descriptions follow the [GEPA paper](https://arxiv.org/abs/2507.19457)
and the pinned [TRL 0.24 SFT](https://huggingface.co/docs/trl/v0.24.0/sft_trainer)
and [GRPO](https://huggingface.co/docs/trl/v0.24.0/grpo_trainer) guides.

## Extra documentation and guides
- [Data layout](data/README.md)
- [Generation workflow](docs/generation.md)
- [GEPA configuration](docs/config.md)
- [Pixi environments and tasks](docs/pixi-environments.md)
- [Fine tuning and evaluation](docs/finetuning-evaluation.md)

## Unit tests
Use the following to run all unit tests. Unit test don't require  GPU.

```bash
pixi run test
```

## License

Distributed under the Apache 2.0 License. See [LICENSE](./LICENSE) for more information.
