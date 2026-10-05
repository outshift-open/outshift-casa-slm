# Fine-tuning and evaluation

The `model_finetuning` and `evaluation` packages contain the SFT, GRPO, and
model evaluation workflows for the paper. Their curated JSON inputs are
under `data/optimization_and_finetuning`; commands resolve these files
from the repository root, so they work from any current directory. Training checkpoints,
merged models, and evaluation reports are written below the ignored
`outputs/` directory.

Install the CUDA environments on Linux with `pixi install -e paper-gpu --locked`
and `pixi install -e paper-eval --locked`. The first isolates Unsloth training;
the second uses the repository's vLLM serving stack. Run commands from the
repository root. `--help` on each command lists additional settings.
The commands load `.env` from the current directory before accessing Hugging
Face Hub; an already exported `HF_TOKEN` takes precedence.

## Prompts and model profiles

The `1b` profile is the default for SFT, GRPO, and evaluation. The SFT search
defaults to `4b`, as in the source experiment. Bundled text templates preserve
the two source prompt variants. Each accepts `{user_task}`, `{tool_name}`, and
`{tool_description}`. Use `--prompt-file PATH` to supply another full template,
including an `optimized_prompt.txt` copied from a GEPA run. No prompt optimizer
code or runtime artifact is required by these components.

Each training run saves its exact template as `prompt.txt`. Merged models also
carry that file. Checkpoint and merged-model evaluation reads it automatically;
`--prompt-file` takes precedence. When evaluating a base model, select its
profile or provide a prompt file explicitly.

## Paired-balanced training data

The dataset builder runs on CPU:

```bash
pixi run build-paired-balanced-dataset -- --dry-run
pixi run build-paired-balanced-dataset
```

By default, it combines the two-tool and three-tool ASTRA training files,
including their cross-MCP variants. It writes the dataset and a summary to
`data/optimization_and_finetuning/paired_balanced_train_seed3407.json` and
`paired_balanced_train_seed3407.summary.json` in the same directory.
Use repeated `--input PATH` arguments to select inputs, `--output PATH` to
choose a destination, and `--overwrite` to replace existing outputs.
Explicit paths are relative to the current directory; default paths resolve
from the repository root.

For each task group within a source file, the builder selects
`k = min(positive_count, negative_count)` examples from each class to form
`k` pairs. It then samples equal numbers of unmatched positives and negatives
per source file. Thus the default dataset has equal class counts, with
deterministic sampling controlled by `--seed` (default `3407`). Use
`--supplement-mode none` to retain only matched pairs.

The default GRPO configuration reads this single combined dataset, including
the cross-MCP examples. Its validation split uses the original two-tool and
three-tool ASTRA validation files and their cross-MCP variants.

## Training workflow

```bash
pixi run finetune-sft -- --profile 1b
pixi run evaluate-checkpoints -- outputs/sft/<run-name>
pixi run merge-finetune-checkpoint -- outputs/sft/<run-name>/checkpoint-<step>
pixi run finetune-grpo -- --model-path outputs/models/<merged-sft-model>
pixi run evaluate-checkpoints -- outputs/grpo/<run-name>
```

Use `--profile 4b` on SFT and GRPO when running the 4B experiment. Pass the
selected SFT merged model path to GRPO; the code does not guess a checkpoint.
To search SFT hyperparameters first, run
`pixi run search-sft -- --profile 4b`.
Resume SFT or GRPO with `--resume-from-checkpoint` and the same
profile and model path. The saved prompt must match any explicit override.

## Evaluation

```bash
pixi run evaluate-model -- --profile 1b --split test
pixi run evaluate-model -- --eval-model outputs/models/<merged-model> --split test
pixi run evaluate-model -- --checkpoint outputs/grpo/<run-name>/checkpoint-<step> --split test
```

`evaluate-checkpoints` always uses the validation split and writes a metrics
summary inside the training run. `evaluate-model` defaults to the test split
and writes item-level results and failed-parse records under
`outputs/evaluation/<split>`. Classification metrics use parsed Boolean
predictions. Failed parses count as incorrect in end-to-end accuracy and are
also reported separately. Confidence intervals use 10,000 paired BCa bootstrap
samples, matching the source evaluation. If a BCa interval is undefined for a
constant metric, its bounds use percentiles from the same bootstrap samples.
Server and transport errors stop the run instead of being counted as malformed
predictions.

Run the offline dataset-builder tests with `pixi run test-paper-components`.
