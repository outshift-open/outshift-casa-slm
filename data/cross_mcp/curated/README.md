# Cross-MCP curation outcomes

This folder contains the curated samples for the **cross-MCP** task-tool datasets.
This is a first low hanging fruit curation without human in the loop.

## Procedure

Each raw sample was independently labeled by **5 LLM labelers**:

- `o3-pro`
- `claude-opus-4.6`
- `claude-sonnet-4.6`
- `gpt-5.2`
- `gpt-5.4`

A sample is **kept** in the curated dataset only when **all 5 labelers agree
unanimously**. Its curated label is the unanimous vote:

- If it matches the seed ground-truth label → kept as-is.
- If it differs from the seed ground-truth label → relabeled.

Samples without unanimous agreement (or with fewer than 5 labelers) are
**dropped**.

Class labels: `True` = *relevant*, `False` = *irrelevant*.

## Per-dataset outcomes

### `two_tools_validation`

| Stage                            | Count | relevant (True) | irrelevant (False) |
|----------------------------------|------:|----------------:|-------------------:|
| **Initial (raw)**                |  2326 |            1164 |               1162 |
| Kept as-is (unanimous = GT)      |  1922 |            1137 |                785 |
| Relabeled (unanimous ≠ GT)       |   117 |               2 |                115 |
| Dropped (non-unanimous + partial)|   287 |              25 |                262 |
| **Curated (final)**              |  2039 |            1252 |                787 |

Relabeling direction: 115 False→True, 2 True→False.

### `two_tools_test`

| Stage                            | Count | relevant (True) | irrelevant (False) |
|----------------------------------|------:|----------------:|-------------------:|
| **Initial (raw)**                |  1894 |             948 |                946 |
| Kept as-is (unanimous = GT)      |  1582 |             918 |                664 |
| Relabeled (unanimous ≠ GT)       |    87 |               0 |                 87 |
| Dropped (non-unanimous + partial)|   225 |              30 |                195 |
| **Curated (final)**              |  1669 |            1005 |                664 |

Relabeling direction: 87 False→True, 0 True→False.

### `three_tools_validation`

| Stage                            | Count | relevant (True) | irrelevant (False) |
|----------------------------------|------:|----------------:|-------------------:|
| **Initial (raw)**                |  3489 |            1746 |               1743 |
| Kept as-is (unanimous = GT)      |  2797 |            1643 |               1154 |
| Relabeled (unanimous ≠ GT)       |   287 |              60 |                227 |
| Dropped (non-unanimous + partial)|   405 |              43 |                362 |
| **Curated (final)**              |  3084 |            1870 |               1214 |

Relabeling direction: 227 False→True, 60 True→False.

### `three_tools_test`

| Stage                            | Count | relevant (True) | irrelevant (False) |
|----------------------------------|------:|----------------:|-------------------:|
| **Initial (raw)**                |  2841 |            1422 |               1419 |
| Kept as-is (unanimous = GT)      |  2215 |            1233 |                982 |
| Relabeled (unanimous ≠ GT)       |   282 |             138 |                144 |
| Dropped (non-unanimous + partial)|   344 |              51 |                293 |
| **Curated (final)**              |  2497 |            1377 |               1120 |

Relabeling direction: 144 False→True, 138 True→False.

## Totals across all four datasets

| Stage                            | Count | relevant (True) | irrelevant (False) |
|----------------------------------|------:|----------------:|-------------------:|
| **Initial (raw)**                | 10550 |            5280 |               5270 |
| Kept as-is (unanimous = GT)      |  8516 |            4931 |               3585 |
| Relabeled (unanimous ≠ GT)       |   773 |             200 |                573 |
| Dropped (non-unanimous + partial)|  1261 |             149 |               1112 |
| **Curated (final)**              |  9289 |            5504 |               3785 |

The "per-class" columns for the *Relabeled* row report the **original** (seed
GT) class; the *Curated (final)* row reports the **post-curation** label.

## Regrouped Complete Examples

The curated files are per-tool rows. `regroup_curated_data.py` groups retained
rows by `metadata.request.task.group_id` and writes complete task-level examples
to [data/curated_regrouped/](../curated_regrouped/).

Incomplete post-curation groups are dropped: a two-tool dataset must retain both
tool rows, and a three-tool dataset must retain all three tool rows.

| Dataset | Curated rows | Groups | Complete regrouped records | Incomplete groups dropped |
|---------|-------------:|-------:|---------------------------:|--------------------------:|
| `cross_mcp_astra_two_tools_test_curated.json` | 1,669 | 925 | 744 | 181 |
| `cross_mcp_astra_two_tools_validation_curated.json` | 2,039 | 1,129 | 910 | 219 |
| `cross_mcp_astra_three_tools_test_curated.json` | 2,497 | 942 | 664 | 278 |
| `cross_mcp_astra_three_tools_validation_curated.json` | 3,084 | 1,155 | 837 | 318 |

## Notes

- This is a first low-hanging-fruit curation: requiring unanimous agreement across all labelers discards the controversial samples, which are often the most informative for training. Properly curating those borderline cases requires a human in the loop, which we are deferring for now and will revisit when resources allow. The samples dropped at this stage are preserved under [discarded/](discarded/) for future review.
- The `metadata.label.reasoning` fields are derived from the
  `claude-sonnet-4.6` per-sample reasoning, with `sample.label.relevant` replaced by the
  unanimous curated label and `metadata.label.agreement_rate` set to `5.0`.
- Raw sources:
  - [data/original_generated/generated_data_val_2tool.json](../original_generated/generated_data_val_2tool.json)
  - [data/original_generated/generated_data_test_2tool.json](../original_generated/generated_data_test_2tool.json)
  - [data/original_generated/generated_data_val_3tool.json](../original_generated/generated_data_val_3tool.json)
  - [data/original_generated/generated_data_test_3tool.json](../original_generated/generated_data_test_3tool.json)
