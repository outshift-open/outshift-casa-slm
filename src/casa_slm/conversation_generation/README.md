# Running Multi-Agent Conversations for Task-Tool Matching Evaluation

This repository contains the input data and generation code for studying
irrelevant tool calls in multi-turn LLM agent conversations.

The current paper-facing work is centered on the ASTRA benchmark. The generator
builds labeled conversations in which an assistant either stays on task or makes
a hidden-objective irrelevant tool call while trying to appear helpful. These
traces are intended for training, evaluating, and auditing detectors of
off-task tool use in agentic systems.

## Repository Layout

```text
src/casa_slm/conversation_generation/
|-- agentic/                 # ASTRA generator and tests
`-- README.md                # This overview

data/conversation/           # Shared inputs at the repository root
|-- mcp_servers/             # ASTRA tool schemas
`-- task_tool/               # Curated single-MCP and cross-MCP task/tool groupings
```

Use [agentic/README.md](agentic/README.md) as the detailed operator manual for setup, generation,
screen launchers and output format.

## What Is Generated

The Agentic generator produces two labels:

| Label | Meaning |
| --- | --- |
| `clean` | The assistant uses at least one task-relevant tool and no irrelevant tools. |
| `malicious_irrelevant_tool_call` | The assistant is given a hidden objective to call one irrelevant tool while still trying to satisfy the user's request. |

Negative examples are generated directly by the assistant model under a hidden
malicious system prompt; they are not post-hoc edited into otherwise clean
dialogues. Tool outputs are simulated by a separate LLM role so each conversation
contains natural tool-call and tool-result turns.

Current generated dialog JSON files wrap new conversation payloads under:

```json
{
  "agentic_conversation": {
    "scenario": {},
    "turns": [],
    "events": [],
    "notes": {}
  }
}
```

Reader code also accepts older migrated traces where those fields live at the
top level.

## Data Sources

The shared input data lives under `data/conversation/` at the repository root.
Download the bundled JSON contents with `git lfs pull` after cloning.

```text
data/conversation/task_tool/
  single_mcp_server/
    01_one_tool/ASTRA/03_grouped_filtered/
    02_two_tools/ASTRA/03_grouped_filtered/
    03_three_tools/ASTRA/03_grouped_filtered/
  cross_mcp_server/
    02_two_tools/ASTRA/03_grouped_filtered/
    03_three_tools/ASTRA/03_grouped_filtered/

data/conversation/mcp_servers/
  ASTRA/
```

Agentic generation and validation use ASTRA task data and MCP tool catalogs.

## Generated Outputs

New screen-launched runs write their output under:

```text
agentic/generated_agentic_conversations/task_tool/
```

The launchers cover ten ASTRA configurations:

| Surface | Tool counts | Splits |
| --- | --- | --- |
| Single MCP | 1, 2, 3 tools | test, validation |
| Cross MCP | 2, 3 tools | test, validation |

Generated dialogs, logs, and statistics are local artifacts and are ignored by
default. Start new batches with a fresh `--out-root`, such as
`./generated_agentic_conversations/local`. The generator refuses to resume a
historical run when its recorded source paths differ from this checkout.

## Quick Start

The shared CPU-only `generation` environment targets Linux x86-64 (`linux-64`).
Use a Linux host or VM for these Pixi commands. Git LFS is required for the
bundled JSON inputs; detached launches additionally require GNU Screen.

From the CASA-SLM repository root, install dependencies and create the configuration:

```bash
pixi install -e generation
git lfs pull
cp -n .env.sample .env
```

Edit the repository-root `.env` with credentials for your endpoint:

```text
LLM_API_KEY="your-key"
LLM_BASE_URL="https://your-endpoint"
```

The generator loads this root `.env` regardless of the current working
directory. Exported environment variables take precedence. No nested `.env`
or separate virtual environment is needed. The compatibility setup script
`agentic/setup_repo.sh` installs this same Pixi environment.

You can also run the packaged command from the repository root:

```bash
pixi run -e generation python -m casa_slm.conversation_generation.agentic.generate_dataset --help
```

The default generator model is `azure/gpt-5.2`, passed through
`langchain-openai`'s `ChatOpenAI`. Your endpoint must serve this model identifier;
otherwise pass a supported identifier with `--model`.

From the repository root, enter the environment and generator directory for the
validation commands below:

```bash
pixi shell -e generation
cd src/casa_slm/conversation_generation/agentic
```

## Validate Without LLM Calls

Run these from `agentic/`:

```bash
python generate_dataset.py --help
python -m pytest tests
python generate_dataset.py --dry-run \
  --out ./generated_agentic_conversations/dry-run \
  --task-data ../../../../data/conversation/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped.json \
  --mcp-servers-dir ../../../../data/conversation/mcp_servers/ASTRA
```

The dry run checks input schema loading, grouped task normalization, and MCP tool
resolution without calling an LLM.

## Run Generation

From the repository root, enter `pixi shell -e generation` first.
Single run:

```bash
cd src/casa_slm/conversation_generation/agentic
python generate_dataset.py \
  --out ./generated_agentic_conversations/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped \
  --seed 13 \
  --model azure/gpt-5.2 \
  --task-data ../../../../data/conversation/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped.json \
  --mcp-servers-dir ../../../../data/conversation/mcp_servers/ASTRA \
  --resume
```

Detached screen launches (commands below start from the repository root):

```bash
cd src/casa_slm/conversation_generation/agentic
bash launch_generation_screens.sh single_mcp_server
bash launch_generation_screens.sh cross_mcp_server
```

Sequential watchdog-backed batches (commands below start from the repository root):

```bash
cd src/casa_slm/conversation_generation/agentic
bash run_experiments.sh --split test
bash run_experiments.sh --split validation
bash run_cross_experiments.sh --split test
bash run_cross_experiments.sh --split validation
```

Screen-launched output directories contain `dialogs/`, `manifest.jsonl`,
`stats.json`, `run.log`, and a generated `screen_runner.sh`. The screen runner
exists so a detached `screen` session can restart transient failures and resume
matching output directories while preserving the exact command used for the run.
Launchers resolve the absolute Python interpreter from the root Pixi `generation`
environment, so detached sessions retain the correct interpreter. An explicit
`PYTHON_BIN=/absolute/path/to/python` override is supported for an already installed
environment. Default inputs resolve to repository-root `data/conversation/`.
Relative `--out-root` paths and the screen launcher's relative `--data-root`
override resolve from `agentic/`.

## Development Notes

- Keep generation work ASTRA-only unless a newer task explicitly expands scope.
- Keep generated local experiments ignored until they are intentionally selected
  for publication.
- Do not rely on top-level dialog fields only; new traces use the
  `agentic_conversation` wrapper.
- Use `--dry-run` before expensive generation to catch missing task data or MCP
  schema mismatches.
- Use each completed run's `stats.json` for accepted and discarded counts.
