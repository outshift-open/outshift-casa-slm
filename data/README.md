# Research data

Run paths in this directory from the repository root. The files are ordinary Git
files; no Git LFS pull is needed. New run outputs belong in `outputs/`.

| Directory | Contents | Used by |
| --- | --- | --- |
| `optimization_and_finetuning/` | Curated ASTRA splits | GEPA, SFT, evaluation |
| `cross_mcp/original_generated/` | Synthetic two and three tool tasks and match candidates | Data generation |
| `cross_mcp/curated/` | Reviewed per tool ASTRA rows, with discarded sidecars | Regrouping |
| `cross_mcp/curated_regrouped/` | Complete grouped cross MCP tasks | Conversation generation |
| `conversation/mcp_servers/` | ASTRA MCP catalogs | Conversation generation |
| `conversation/task_tool/` | Single and cross MCP task groups and intermediate stages | Conversation generation |

## Optimization and fine tuning splits

`optimization_and_finetuning/one_tool_astra_train_curated.json` (4,404 rows)
is the SFT train split. Its validation counterpart has 1,136 rows. GEPA uses the split
files listed in `eval_config.yaml`.

The default evaluator uses five ASTRA test files: one, two, and three tool
single MCP splits, plus two and three tool cross MCP splits (8,691 rows total).
Validation uses the matching five validation files.

The files are derived from the [ASTRA dataset](https://github.com/outshift-open/ASTRA/tree/main/data)
and the synthetic generation and curation process described in the
[generation guide](../docs/generation.md). See the
[fine tuning and evaluation guide](../docs/finetuning-evaluation.md) for the
training sequence and [root README](../README.md) for the short reproduction path.
