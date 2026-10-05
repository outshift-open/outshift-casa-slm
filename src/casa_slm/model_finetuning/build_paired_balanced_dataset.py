"""Build a fixed paired-balanced training dataset from curated tool data.

The builder first keeps one-to-one positive/negative matches inside each task
group, then optionally adds a balanced supplement from examples that lost their
sibling during curation. The output is a normal JSON list with the original
schema plus metadata that records how each example was selected.
"""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from casa_slm.model_finetuning.data_paths import DATA_DIR

REPOSITORY_ROOT = DATA_DIR.parents[1]
DEFAULT_TRAIN_PATHS = (
    "two_tools_astra_train_curated.json",
    "three_tools_astra_train_curated.json",
    "cross_mcp_astra_two_tools_train_curated.json",
    "cross_mcp_astra_three_tools_train_curated.json",
)

BUILDER_METADATA_KEY = "paired_balanced_builder"


@dataclass(frozen=True)
class LoadedExample:
    """A curated example with source and grouping metadata."""

    example: dict[str, Any]
    source_file: str
    original_index: int
    label: bool
    group_key: str
    group_key_hash: str
    group_key_kind: str

    @property
    def stable_id(self) -> str:
        """Return a stable identifier for deterministic sorting and summaries."""
        example_id = self.example.get("id")
        if isinstance(example_id, str) and example_id:
            return example_id
        return f"{self.source_file}:{self.original_index}"


@dataclass(frozen=True)
class Candidate:
    """An unpaired supplement candidate and why it is unpaired."""

    item: LoadedExample
    reason: str


def _hash_text(text: str, *, length: int = 16) -> str:
    """Return a short stable hash for a grouping value."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _normalize_task(task: str) -> str:
    """Normalize task text enough for grouping without changing semantics."""
    return " ".join(task.split())


def _get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Read a nested value from a dictionary, returning None if any key is absent."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_label(example: dict[str, Any], path: Path, index: int) -> bool:
    """Extract the curated binary label and fail loudly on malformed data."""
    value = _get_nested(example, ("label", "relevant"))
    if not isinstance(value, bool):
        raise ValueError(f"{path}:{index} has no boolean label.relevant")
    return value


def _extract_group_key(
    example: dict[str, Any],
    *,
    group_key: str,
    group_scope: str,
    source_file: str,
    path: Path,
    index: int,
) -> tuple[str, str, str]:
    """Extract the grouping key used to find positive/negative siblings."""
    task = _get_nested(example, ("request", "task"))
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{path}:{index} has no non-empty request.task")

    group_id = _get_nested(example, ("metadata", "request", "task", "group_id"))
    if group_key == "group_id":
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"{path}:{index} has no metadata.request.task.group_id; use --group-key task or auto")
        key_kind = "group_id"
        raw_key = group_id.strip()
    elif group_key == "auto" and isinstance(group_id, str) and group_id.strip():
        key_kind = "group_id"
        raw_key = group_id.strip()
    else:
        key_kind = "task"
        raw_key = _normalize_task(task)

    scoped_key = raw_key if group_scope == "global" else f"{source_file}\0{raw_key}"
    return scoped_key, _hash_text(scoped_key), key_kind


def _load_examples(paths: list[Path], *, group_key: str, group_scope: str) -> list[LoadedExample]:
    """Load and annotate examples from the input JSON files."""
    loaded: list[LoadedExample] = []
    for path in paths:
        path = path.resolve()
        try:
            source_file = str(path.relative_to(REPOSITORY_ROOT))
        except ValueError:
            source_file = str(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")

        for index, example in enumerate(data):
            if not isinstance(example, dict):
                raise ValueError(f"{path}:{index} is not a JSON object")
            label = _extract_label(example, path, index)
            scoped_key, key_hash, key_kind = _extract_group_key(
                example,
                group_key=group_key,
                group_scope=group_scope,
                source_file=source_file,
                path=path,
                index=index,
            )
            loaded.append(
                LoadedExample(
                    example=example,
                    source_file=source_file,
                    original_index=index,
                    label=label,
                    group_key=scoped_key,
                    group_key_hash=key_hash,
                    group_key_kind=key_kind,
                )
            )
    return loaded


def _sort_key(item: LoadedExample) -> tuple[str, int, str]:
    """Sort examples deterministically."""
    return item.source_file, item.original_index, item.stable_id


def _sample_items(rng: random.Random, items: list[LoadedExample], k: int) -> list[LoadedExample]:
    """Sample items deterministically with stable ordering as the input to RNG."""
    pool = sorted(items, key=_sort_key)
    if k < len(pool):
        pool = rng.sample(pool, k)
    rng.shuffle(pool)
    return pool


def _annotate(
    item: LoadedExample,
    *,
    status: str,
    group_scope: str,
    pair_id: str | None = None,
    counterpart_id: str | None = None,
    supplement_reason: str | None = None,
) -> dict[str, Any]:
    """Copy an example and record deterministic builder metadata."""
    example = copy.deepcopy(item.example)
    metadata = example.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        example["metadata"] = metadata

    builder_metadata: dict[str, Any] = {
        "version": 1,
        "status": status,
        "source_file": item.source_file,
        "original_index": item.original_index,
        "original_id": item.stable_id,
        "group_key_kind": item.group_key_kind,
        "group_key_hash": item.group_key_hash,
        "group_scope": group_scope,
    }
    if pair_id is not None:
        builder_metadata["pair_id"] = pair_id
    if counterpart_id is not None:
        builder_metadata["counterpart_id"] = counterpart_id
    if supplement_reason is not None:
        builder_metadata["supplement_reason"] = supplement_reason

    metadata[BUILDER_METADATA_KEY] = builder_metadata
    return example


def _sample_supplement(
    rng: random.Random,
    candidates: list[Candidate],
    *,
    strategy: str,
    max_per_class: int | None,
) -> list[Candidate]:
    """Sample a label-balanced supplement from unpaired candidates."""
    if not candidates:
        return []

    buckets: dict[str, list[Candidate]] = defaultdict(list)
    if strategy == "global":
        for candidate in candidates:
            buckets["__global__"].append(candidate)
    elif strategy == "source":
        for candidate in candidates:
            buckets[candidate.item.source_file].append(candidate)
    else:
        raise ValueError(f"Unknown supplement strategy: {strategy}")

    selected: list[Candidate] = []
    for _, bucket in sorted(buckets.items()):
        positives = [candidate for candidate in bucket if candidate.item.label]
        negatives = [candidate for candidate in bucket if not candidate.item.label]
        k = min(len(positives), len(negatives))
        if max_per_class is not None:
            k = min(k, max_per_class)
        if k == 0:
            continue
        selected.extend(
            Candidate(item=item, reason=reason) for item, reason in _sample_candidate_pairs(rng, positives, k)
        )
        selected.extend(
            Candidate(item=item, reason=reason) for item, reason in _sample_candidate_pairs(rng, negatives, k)
        )
    return selected


def _sample_candidate_pairs(
    rng: random.Random,
    candidates: list[Candidate],
    k: int,
) -> list[tuple[LoadedExample, str]]:
    """Sample candidates and keep their unpaired reason."""
    ordered = sorted(candidates, key=lambda candidate: _sort_key(candidate.item))
    if k < len(ordered):
        ordered = rng.sample(ordered, k)
    rng.shuffle(ordered)
    return [(candidate.item, candidate.reason) for candidate in ordered]


def _build_dataset(
    loaded: list[LoadedExample],
    *,
    seed: int,
    group_scope: str,
    supplement_mode: str,
    supplement_strategy: str,
    max_supplement_per_class: int | None,
    shuffle_output: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the paired-balanced dataset and return examples plus summary."""
    rng = random.Random(seed)
    groups: dict[str, list[LoadedExample]] = defaultdict(list)
    for item in loaded:
        groups[item.group_key].append(item)

    output: list[dict[str, Any]] = []
    unpaired_candidates: list[Candidate] = []
    group_size_counts: Counter[int] = Counter()
    group_status_counts: Counter[str] = Counter()
    matched_pairs = 0

    for group_key in sorted(groups):
        group = sorted(groups[group_key], key=_sort_key)
        group_size_counts[len(group)] += 1
        positives = [item for item in group if item.label]
        negatives = [item for item in group if not item.label]

        if positives and negatives:
            group_status_counts["mixed_label_groups"] += 1
            k = min(len(positives), len(negatives))
            selected_positives = _sample_items(rng, positives, k)
            selected_negatives = _sample_items(rng, negatives, k)
            selected_keys = {(_sort_key(item), item.label) for item in selected_positives + selected_negatives}

            for pair_index, (positive, negative) in enumerate(zip(selected_positives, selected_negatives, strict=True)):
                pair_id = f"{positive.group_key_hash}-{pair_index:04d}"
                output.append(
                    _annotate(
                        positive,
                        status="matched_pair",
                        group_scope=group_scope,
                        pair_id=pair_id,
                        counterpart_id=negative.stable_id,
                    )
                )
                output.append(
                    _annotate(
                        negative,
                        status="matched_pair",
                        group_scope=group_scope,
                        pair_id=pair_id,
                        counterpart_id=positive.stable_id,
                    )
                )
                matched_pairs += 1

            for item in positives + negatives:
                if (_sort_key(item), item.label) not in selected_keys:
                    unpaired_candidates.append(Candidate(item=item, reason="matched_group_majority_leftover"))
        elif positives:
            group_status_counts["positive_only_groups"] += 1
            unpaired_candidates.extend(
                Candidate(item=item, reason="single_class_group_after_curation") for item in positives
            )
        elif negatives:
            group_status_counts["negative_only_groups"] += 1
            unpaired_candidates.extend(
                Candidate(item=item, reason="single_class_group_after_curation") for item in negatives
            )

    if supplement_mode == "balanced":
        supplement = _sample_supplement(
            rng,
            unpaired_candidates,
            strategy=supplement_strategy,
            max_per_class=max_supplement_per_class,
        )
    elif supplement_mode == "none":
        supplement = []
    elif supplement_mode == "all":
        supplement = list(unpaired_candidates)
    else:
        raise ValueError(f"Unknown supplement mode: {supplement_mode}")

    for candidate in supplement:
        output.append(
            _annotate(
                candidate.item,
                status="unpaired_supplement",
                group_scope=group_scope,
                supplement_reason=candidate.reason,
            )
        )

    if shuffle_output:
        rng.shuffle(output)

    summary = _summarize(
        loaded,
        output,
        unpaired_candidates,
        supplement,
        matched_pairs=matched_pairs,
        group_size_counts=group_size_counts,
        group_status_counts=group_status_counts,
    )
    return output, summary


def _label_counts_from_examples(examples: list[dict[str, Any]]) -> dict[str, int]:
    """Count positive and negative labels in output examples."""
    labels = [bool(_get_nested(example, ("label", "relevant"))) for example in examples]
    positives = sum(labels)
    return {"positive": positives, "negative": len(labels) - positives, "total": len(labels)}


def _label_counts_from_loaded(examples: list[LoadedExample]) -> dict[str, int]:
    """Count positive and negative labels in loaded examples."""
    positives = sum(item.label for item in examples)
    return {"positive": positives, "negative": len(examples) - positives, "total": len(examples)}


def _label_counts_from_candidates(candidates: list[Candidate]) -> dict[str, int]:
    """Count positive and negative labels in supplement candidates."""
    positives = sum(candidate.item.label for candidate in candidates)
    return {"positive": positives, "negative": len(candidates) - positives, "total": len(candidates)}


def _summarize(
    loaded: list[LoadedExample],
    output: list[dict[str, Any]],
    unpaired_candidates: list[Candidate],
    supplement: list[Candidate],
    *,
    matched_pairs: int,
    group_size_counts: Counter[int],
    group_status_counts: Counter[str],
) -> dict[str, Any]:
    """Build a JSON-serializable summary for auditing."""
    output_status_counts: Counter[str] = Counter()
    output_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in output:
        builder = _get_nested(example, ("metadata", BUILDER_METADATA_KEY))
        if not isinstance(builder, dict):
            continue
        status = str(builder.get("status", "unknown"))
        source_file = str(builder.get("source_file", "unknown"))
        label = bool(_get_nested(example, ("label", "relevant")))
        output_status_counts[status] += 1
        output_source_counts[source_file]["total"] += 1
        output_source_counts[source_file]["positive" if label else "negative"] += 1

    return {
        "input_counts": _label_counts_from_loaded(loaded),
        "output_counts": _label_counts_from_examples(output),
        "matched_pairs": matched_pairs,
        "unpaired_candidate_counts": _label_counts_from_candidates(unpaired_candidates),
        "supplement_counts": _label_counts_from_candidates(supplement),
        "group_status_counts": dict(sorted(group_status_counts.items())),
        "group_size_counts": {str(size): count for size, count in sorted(group_size_counts.items())},
        "output_status_counts": dict(sorted(output_status_counts.items())),
        "output_counts_by_source": {
            source: dict(sorted(counts.items())) for source, counts in sorted(output_source_counts.items())
        },
    }


def _write_json(path: Path, data: Any, *, overwrite: bool) -> None:
    """Write JSON with overwrite protection."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_paths",
        action="append",
        default=None,
        help="Input curated JSON file. Repeat to include multiple files. Defaults to curated ASTRA training inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "paired_balanced_train_seed3407.json",
        help="Output JSON dataset path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to OUTPUT with '.summary.json' suffix.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Deterministic sampling seed.")
    parser.add_argument(
        "--group-key",
        choices=("task", "group_id", "auto"),
        default="task",
        help="Key used to find sibling examples. 'task' is safest across current curated files.",
    )
    parser.add_argument(
        "--group-scope",
        choices=("source", "global"),
        default="source",
        help="Whether identical group keys can match across different input files.",
    )
    parser.add_argument(
        "--supplement-mode",
        choices=("balanced", "none", "all"),
        default="balanced",
        help="How to handle examples that are not selected into one-to-one matched pairs.",
    )
    parser.add_argument(
        "--supplement-strategy",
        choices=("source", "global"),
        default="source",
        help="When supplement-mode=balanced, balance unpaired examples per source file or globally.",
    )
    parser.add_argument(
        "--max-supplement-per-class",
        type=int,
        default=None,
        help="Optional cap per class for each supplement sampling bucket.",
    )
    parser.add_argument(
        "--no-shuffle-output",
        action="store_true",
        help="Keep matched examples and supplement examples in deterministic grouped order.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Print the summary without writing files.")
    return parser.parse_args(argv)


def main() -> None:
    """Run the paired-balanced dataset builder."""
    args = parse_args()
    input_paths = (
        [Path(path) for path in args.input_paths]
        if args.input_paths is not None
        else [DATA_DIR / name for name in DEFAULT_TRAIN_PATHS]
    )
    summary_output = args.summary_output or args.output.with_name(f"{args.output.stem}.summary.json")

    missing = [path for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing curated training inputs: {', '.join(map(str, missing))}")
    output_paths = (args.output.resolve(), summary_output.resolve())
    if output_paths[0] == output_paths[1]:
        raise ValueError("Dataset and summary output paths must differ")
    if any(path in output_paths for path in (input_path.resolve() for input_path in input_paths)):
        raise ValueError("An output path must not overwrite an input dataset")
    if not args.dry_run and not args.overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {', '.join(map(str, existing))}; pass --overwrite")

    loaded = _load_examples(input_paths, group_key=args.group_key, group_scope=args.group_scope)
    output, summary = _build_dataset(
        loaded,
        seed=args.seed,
        group_scope=args.group_scope,
        supplement_mode=args.supplement_mode,
        supplement_strategy=args.supplement_strategy,
        max_supplement_per_class=args.max_supplement_per_class,
        shuffle_output=not args.no_shuffle_output,
    )

    summary = {
        "config": {
            "input_paths": [str(path) for path in input_paths],
            "output": str(args.output),
            "summary_output": str(summary_output),
            "seed": args.seed,
            "group_key": args.group_key,
            "group_scope": args.group_scope,
            "supplement_mode": args.supplement_mode,
            "supplement_strategy": args.supplement_strategy,
            "max_supplement_per_class": args.max_supplement_per_class,
            "shuffle_output": not args.no_shuffle_output,
        },
        **summary,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return

    _write_json(args.output, output, overwrite=args.overwrite)
    _write_json(summary_output, summary, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
