# Synthetic Data Generation for Task-Tool Matching Evaluation

Generates synthetic evaluation datasets for the **tool selector** component of [ZTA (Zero Trust for Multi-Agent Systems)](https://github.com/cisco-eti/identity-auth-server). The tool selector determines which MCP tools are required to fulfill a given user task — this pipeline produces labeled data to evaluate that capability.

## Prerequisites

- Linux x86-64 (`linux-64`, the platform configured for the `generation` environment)
- Python 3.13 (installed by Pixi)
- Git LFS (for bundled JSON datasets)
- [Pixi](https://pixi.sh/) (use the repository-root environment)
- Docker (only for MCP tool extraction)

## Setup

Run all commands from the `casa-slm-paper` repository root. The shared `generation`
Pixi environment runs on CPU and needs no GPU runtime.

```bash
# Install dependencies
pixi install -e generation
git lfs pull

# Configure environment
cp -n .env.sample .env
# Edit .env with your API key and base URL
```

Set `LLM_BASE_URL` to your OpenAI-compatible endpoint (such as a LiteLLM proxy)
and `LLM_API_KEY` to a valid key for that endpoint in the repository-root `.env`.
Exported environment variables override that file. If no base URL is provided,
the public OpenAI endpoint is used. The default model identifier `azure/gpt-5.2`
must be supported by the selected endpoint; otherwise set `--model` to one it
serves. Stage 1 uses the OpenAI Python client (`AsyncOpenAI`). Stage 2 and
regrouping do not need credentials.

The copied standalone `pyproject.toml`, `uv.lock`, and `.env.example` have been
replaced by the root `pyproject.toml`, `pixi.lock`, and `.env.sample`. Bundled JSON
datasets use the repository's Git LFS rules; run `git lfs pull` after cloning.

## Pipeline Overview

```
  mcp_servers/*.json         .env              config_*.json
  (352 tools, 12 servers)    (API key)         (match params)
         │                     │                     │
         ▼                     ▼                     │
  ┌────────────────────────────────────────┐         │
  │  STAGE 1: Task Generation              │         │
  │                                        │         │
  │  1. Load & parse tool JSONs            │         │
  │  2. Combine tools (N per group,        │         │
  │     configurable server diversity)     │         │
  │  3. LLM generates user tasks           │         │
  │     (async, bounded concurrency)       │         │
  │  4. Group tasks by tool combination    │         │
  └───────────────────┬────────────────────┘         │
                      │                              │
                      ▼                              │
  ┌────────────────────────────────────────┐         │
  │  data/cross_mcp/original_generated/    │         │
  │  generated_tasks_*.json                │         │
  │  (synthetic user requests)             │         │
  └───────────────────┬────────────────────┘         │
                      │                              │
                      ▼                              ▼
  ┌──────────────────────────────────────────────────┐
  │  STAGE 2: Matching Simulator                     │
  │                                                  │
  │  1. Load tasks & MCP tool registry               │
  │  2. Simulate CORRECT matches                     │
  │     (right tools, right servers)                 │
  │  3. Simulate WRONG matches                       │
  │     (same servers, different tools)              │
  │  4. Simulate NULL matches                        │
  │     (servers outside task's set)                 │
  └─────────────────────┬────────────────────────────┘
                        │
                        ▼
  ┌──────────────────────────────────────────────────┐
  │  data/cross_mcp/original_generated/              │
  │  generated_data_*.json                           │
  │  (raw labeled evaluation dataset)                │
  │                                                  │
  │  Each entry: {input, groundtruth, match_tag}     │
  └─────────────────────┬────────────────────────────┘
                        │
                        ▼
  ┌──────────────────────────────────────────────────┐
  │  CURATION + REGROUPING                           │
  │                                                  │
  │  1. Keep unanimous curated per-tool rows         │
  │  2. Regroup by group_id                          │
  │  3. Drop incomplete groups                       │
  └─────────────────────┬────────────────────────────┘
                        │
                        ▼
  ┌──────────────────────────────────────────────────┐
  │  data/cross_mcp/curated_regrouped/               │
  │  *_curated.json                                  │
  │  (complete task-level curated examples)          │
  └──────────────────────────────────────────────────┘
```

## Usage

The `generate-tasks` and `generate-matches` tasks run from the repository root.
Their relative input/output paths are resolved from that directory. For custom
locations, use absolute paths. The scripts also
support module execution, for example
`pixi run -e generation python -m casa_slm.data_generation.multitool_task_generation --help`;
with direct module execution, relative paths follow your current working directory.

The command examples below write local results under the ignored `outputs/`
directory. The committed raw test/validation artifacts are stored under
`data/cross_mcp/original_generated/`.

### Stage 1: Generate Synthetic Tasks

Creates realistic user requests that require specific MCP tools to fulfill.

```bash
pixi run -e generation generate-tasks \
  --input-files src/casa_slm/data_generation/mcp_servers/github_official.json src/casa_slm/data_generation/mcp_servers/nasdaq_data_link.json \
  --output-file outputs/generated_tasks.json \
  --multiplier 3 \
  --num-tools 2 \
  --server-diversity cross
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-files` | *(required)* | One or more MCP tool JSON files |
| `--output-file` | *(required)* | Output path for generated tasks |
| `--multiplier` | 1 | Stage 1 supply: LLM task-generation attempts per anchor tool |
| `--num-tools` | 1 | Tools to combine per task |
| `--server-diversity` | `cross` | MCP server sampling mode for multi-tool groups: `cross`, `same`, or `any` |
| `--no-obscure` | false | Disable obscure mode: generate direct tasks (includes input schema in prompt) |
| `--model` | `azure/gpt-5.2` | Model identifier sent through the OpenAI Python client to `LLM_BASE_URL` |
| `--concurrency` | 10 | Max concurrent API requests |
| `--seed` | None | Random seed for reproducible tool combinations |

**Server diversity modes:** When `--num-tools N` is specified, `--server-diversity` controls how partner tools are sampled:

- **`cross`** (default): each tool in a group comes from a distinct MCP server (2 tools → 2 servers, 3 tools → 3 servers). This requires at least N distinct servers in the input files.
- **`same`**: all tools in a group come from the anchor tool's MCP server. This requires each anchor server to have at least N tools.
- **`any`**: partner tools are sampled from the full non-anchor tool pool with no MCP server constraint.

**Stage 1 supply vs Stage 2 demand:** `--multiplier` controls how many synthetic task texts Stage 1 tries to create before matching simulation. `num_correct_matches` in the selected config file controls how many correct samples Stage 2 tries to draw from that generated task pool. Stage 2 cannot draw more correct samples than Stage 1 produced, so if `num_correct_matches` is larger than the available task texts, the simulator caps it to the available count. For example, the 117 tools in the Stage 1 command above with `--multiplier 3` can supply up to 351 task texts; if Stage 2 asks for 500 correct matches, it will cap to the number actually generated (at most 351). Requests that fail after retries reduce this supply. Successful texts are grouped by tool combination; duplicate texts are not removed.

**Prompt modes:**

- **Obscure** (default): LLM sees tool name + description only. Generates indirect tasks where the connection to tools is non-obvious.
- **Standard** (`--no-obscure`): LLM sees tool name + description + full input schema. Generates direct, explicit tasks.

### Stage 2: Matching Simulator

Simulates correct, wrong, and null tool matches against synthetic tasks to create a labeled evaluation dataset.

```bash
pixi run -e generation generate-matches \
  --config src/casa_slm/data_generation/config_test.json \
  --mcp-server-path src/casa_slm/data_generation/mcp_servers \
  --input outputs/generated_tasks.json \
  --output outputs/generated_data.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config_test.json` | Generation configuration |
| `--mcp-server-path` | `mcp_servers` | Directory with MCP server JSONs |
| `--input` | `data/generated_tasks.json` | Stage 1 output |
| `--output` | `data/generated_data.json` | Stage 2 raw evaluation dataset path |
| `--seed` | 42 | Random seed for reproducibility |

The CLI defaults for `--config` and `--mcp-server-path` are relative to the
working directory. Keep the explicit repository-relative paths above when
running from the repository root. The selected config must include the task
servers and at least one additional server when generating null matches.

### MCP Tool Extraction

To regenerate tool definition files from live MCP servers:

```bash
# Requires Docker running
pixi run -e generation extract-mcp-servers
```

Connects to each server defined in `mcp_servers/generation/servers_config.json`, queries tool schemas via the MCP protocol, and saves them as JSON files in `mcp_servers/`. Before live extraction, replace the service-specific placeholders in that configuration, or remove the corresponding `env` entries to inherit exported values. See the [extraction setup](mcp_servers/README.md).

The extraction script uses a local typed export schema and the root environment's MCP SDK. The pre-extracted JSON files can be used without running Docker or connecting to live servers.

## Configuration Reference

### Split Configs

```json
{
  "mcp_servers": ["github-official", "nasdaq-data-link", "sonarqube", "paper-search"],
  "num_correct_matches": 500,
  "ratio_wrong_matches": 0.8,
  "ratio_null_matches": 0.2,
  "mixing_wrong_null_in_multitool": false
}
```

The repository includes `config_test.json` and `config_val.json` for the test and validation splits used by the generated datasets.

| Field | Type | Description |
|-------|------|-------------|
| `mcp_servers` | `list[str]` | MCP servers to include (must have corresponding JSON in `mcp_servers/`) |
| `num_correct_matches` | `int` | Stage 2 demand: number of correctly-matched samples to draw from the Stage 1 task pool, capped by available task texts |
| `ratio_wrong_matches` | `float` | Multiplier for wrong matches (same servers, different tools) |
| `ratio_null_matches` | `float` | Multiplier for null matches (tools from servers outside the task's server set) |
| `mixing_wrong_null_in_multitool` | `bool` | For multi-tool tasks: randomly mix some correct tools into wrong/null entries |

### Match Types Explained

| Type | Input tools | Input servers | Meaning |
|------|-------------|---------------|---------|
| **correct** | Same as groundtruth | Same as groundtruth | Perfect match — tools are right |
| **wrong** | Different tools | Same servers | Right servers, wrong tools (in theory harder negative) |
| **null** | Tools from other servers | Servers outside task's set | Completely unrelated tools (in theory easier negative) |

The table describes the default `mixing_wrong_null_in_multitool: false` behavior.
When enabled, negative multi-tool entries may retain some correct tools.
The Stage 2 output is shuffled — match types are interleaved randomly (seeded).

### Curated And Regrouped Outputs

The repository also includes curated per-tool rows and regrouped task-level
records. Curation is a separate step; the regrouping command below consumes the
bundled curated rows and does not run labelers:

```bash
pixi run -e generation regroup-data \
  --input-dir ../../../data/cross_mcp/curated \
  --output-dir ../../../outputs/curated_regrouped
```

The regrouping task runs from `src/casa_slm/data_generation`; the example paths
above point back to the repository root. Bundled `data/cross_mcp/curated/` stores
the retained per-tool rows after unanimous labeler agreement.
`data/cross_mcp/curated_regrouped/` stores complete task-level examples regrouped by `group_id`.
Incomplete post-curation groups are dropped, so every retained regrouped record has exactly
2 or 3 tools.

| Dataset | Curated rows | Complete regrouped records | Incomplete groups dropped |
|---------|-------------:|---------------------------:|--------------------------:|
| `cross_mcp_astra_two_tools_test_curated.json` | 1,669 | 744 | 181 |
| `cross_mcp_astra_two_tools_validation_curated.json` | 2,039 | 910 | 219 |
| `cross_mcp_astra_three_tools_test_curated.json` | 2,497 | 664 | 278 |
| `cross_mcp_astra_three_tools_validation_curated.json` | 3,084 | 837 | 318 |

## Output Formats

### Stage 1 Output (`generated_tasks*.json`)

```json
[
  {
    "tool_names": ["jira_get_issue", "get_pull_request"],
    "mcp_servers": ["atlassian", "github-official"],
    "synthetic_tasks": ["Retrieve details of Jira issue DEVOPS-458 and PR #24..."],
    "system_prompt": "obscure_base",
    "tools_per_task": 2,
    "SO_tasks_per_sample": 1,
    "conversation": false
  }
]
```

### Stage 2 Output (`generated_data*.json`)

```json
[
  {
    "input": {
      "task": "Retrieve the details of...",
      "tools": ["jira_get_issue", "get_pull_request"],
      "mcp_servers": ["atlassian", "github-official"]
    },
    "groundtruth": {
      "tools": ["jira_get_issue", "get_pull_request"],
      "mcp_servers": ["atlassian", "github-official"]
    },
    "match_tag": "correct"
  }
]
```

## Available MCP Servers

| Server | Tools | File |
|--------|-------|------|
| GitHub Official | 85 | `github_official.json` |
| Grafana | 43 | `grafana.json` |
| Atlassian (Jira/Confluence) | 37 | `atlassian.json` |
| Nasdaq Data Link | 32 | `nasdaq_data_link.json` |
| Azure | 27 | `azure.json` |
| Paper Search | 23 | `paper_search.json` |
| Stripe | 22 | `stripe.json` |
| MongoDB | 21 | `mongodb.json` |
| Notion | 19 | `notion.json` |
| SonarQube | 18 | `sonarqube.json` |
| Hummingbot | 15 | `hummingbot_mcp.json` |
| Wikipedia | 10 | `wikipedia_mcp.json` |

**Total: 352 tools across 12 MCP servers**

## Project Structure

```
multitool_task_generation.py        # Stage 1: LLM-based task synthesis (async)
synthetic_data_generator.py         # Stage 2: Matching simulator
config_test.json / config_val.json  # Generation parameters for the test/validation splits
regroup_curated_data.py             # Regroups curated rows and drops incomplete groups
mcp_servers/                        # MCP tool definition JSONs (one per server)
  generation/
    extract_mcp_servers.py          # Tool extraction from live Docker servers
    servers_config.json             # Server connection config
```

Bundled task/data artifacts live separately under `data/cross_mcp/` at the
repository root, in `original_generated/`, `curated/`, and `curated_regrouped/`.

## Verification

From the repository root:

```bash
pixi run -e generation test-data-generation
```

Tests run without LLM credentials or live MCP servers.
