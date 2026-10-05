# Generation workflows

Both pipelines are copied into `src/casa_slm` as ordinary files, with no
submodules, nested repositories, or dependency on the source repositories.
The root `pyproject.toml` and `pixi.lock` manage their environment.

## Setup

From the repository root on Linux x86-64:

```bash
pixi install -e generation --locked
cp -n .env.sample .env
```

Configure `LLM_API_KEY` and `LLM_BASE_URL` in the root `.env`. Exported variables
take precedence. Offline matching and conversation dry runs need no LLM credentials.
Full generation sends requests to the configured LLM endpoint.

## Synthetic task and match generation

Data-generation tasks run from the repository root. Input and output arguments
are relative to that directory. For example:

```bash
pixi run generate-tasks -- \
  --input-files src/casa_slm/data_generation/mcp_servers/github_official.json src/casa_slm/data_generation/mcp_servers/nasdaq_data_link.json \
  --output-file outputs/generated_tasks.json --num-tools 2 --multiplier 1 \
  --model azure/gpt-5.2 --seed 42
pixi run generate-matches -- \
  --config src/casa_slm/data_generation/config_test.json \
  --mcp-server-path src/casa_slm/data_generation/mcp_servers \
  --input outputs/generated_tasks.json --output outputs/generated_matches.json --seed 42
```

See the [data-generation guide](../src/casa_slm/data_generation/README.md) for
sampling, match labels, curation, and optional live MCP extraction.

## Conversation validation and generation

The conversation task runs from the repository root. This checks a bundled
ASTRA dataset and catalog without LLM calls:

```bash
pixi run generate-conversations -- --dry-run \
  --out outputs/conversation-dry-run \
  --task-data data/conversation/task_tool/single_mcp_server/01_one_tool/ASTRA/03_grouped_filtered/test_curated_grouped.json \
  --mcp-servers-dir data/conversation/mcp_servers/ASTRA
```

Remove `--dry-run` and use a new output directory to generate conversations.
Use `--resume` only for an existing run with matching input provenance.
The [conversation guide](../src/casa_slm/conversation_generation/README.md) and
[operator manual](../src/casa_slm/conversation_generation/agentic/README.md)
cover batch launchers, retries, and outputs.

For direct Python commands, enter `pixi shell -e generation` first. Both
`python -m casa_slm.data_generation.multitool_task_generation` and
`python -m casa_slm.conversation_generation.agentic.generate_dataset` are
importable modules; relative CLI paths are resolved from the current directory.

## Tests

```bash
pixi run test-data-generation
pixi run test-conversation-generation
```

Both tasks run without external LLM calls and are included in `pixi run test`.

## Source snapshots

| Destination | Source | Snapshot commit |
| --- | --- | --- |
| `data_generation` | [ioc-governance-casa-slm-data-gen](https://github.com/cisco-eti/ioc-governance-casa-slm-data-gen) | `c5856a1ec96d3bf5ff8adeb15e1633e01decd609` |
| `conversation_generation` | [ioc-governance-casa-slm-conversation-gen](https://github.com/cisco-eti/ioc-governance-casa-slm-conversation-gen) | `c2d60032feebab4992645d94570c208bcfa28d8a` |

Standalone environment setup is superseded by the root Pixi configuration.
Bundled JSON datasets use the repository's Git LFS configuration; run `git lfs pull`
after cloning to download their contents. Generated conversations and analysis
artifacts are not bundled; new experimental outputs should remain local unless
explicitly selected for publication.
