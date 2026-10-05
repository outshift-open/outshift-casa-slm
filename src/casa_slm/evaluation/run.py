"""Evaluation pipeline for task-tool matching models."""

import argparse
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, OpenAI
from pydantic import BaseModel
from sklearn.metrics import classification_report
from tqdm import tqdm

from casa_slm.evaluation.metrics import bootstrap_classification_metrics
from casa_slm.model_finetuning.data_paths import EVALUATION_TEST, EVALUATION_VALIDATION, paths
from casa_slm.model_finetuning.prompt import MODEL_IDS, ModelProfile, load_prompt, render_prompt

logger = logging.getLogger(__name__)


# -- Structured Output Schema ------------------------------------------------
class TaskToolMatcherOutput(BaseModel):
    """Output schema for task-tool matcher with reasoning first."""

    reasoning: str
    appropriate: bool


# -- Prompt builder ----------------------------------------------------------


def build_user_prompt(item: dict, *, prompt_template: str | None = None) -> str:
    """Build the user prompt from an evaluation item."""
    return render_prompt(
        prompt_template or load_prompt(), item["task"], item["tool"]["name"], item["tool"]["description"]
    )


# -- Config ------------------------------------------------------------------

DEFAULT_EVAL_MODEL = "unsloth/gemma-3-1b-it"
# DEFAULT_EVAL_MODEL = "unsloth/gemma-3-4b-it"
MAX_OUTPUT_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 45
DEFAULT_CONCURRENCY = 32
DEFAULT_SEED = 42
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
VLLM_HOST = "localhost"
VLLM_PORT = 8000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

FULL_MULTI_TOOL_FILES: dict[str, list[str]] = {
    "test": list(paths(EVALUATION_TEST)),
    "validation": list(paths(EVALUATION_VALIDATION)),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-model", help="Model name or merged model path to evaluate; defaults to the profile model."
    )
    parser.add_argument("--profile", choices=("1b", "4b"), default="1b")
    parser.add_argument("--prompt-file", help="Prompt template copied from a GEPA run or another source.")
    parser.add_argument(
        "--split",
        choices=["test", "validation"],
        default="test",
        help="Which hardcoded set of data files to evaluate on (see 'FULL_MULTI_TOOL_FILES').",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of concurrent in-flight requests to the vLLM server.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampling seed sent with each request for run-to-run repeatability.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a LoRA checkpoint directory to evaluate, i.e. a folder "
            "containing 'adapter_config.json' (e.g. "
            "'outputs/grpo/GRPO-20260624-174505/checkpoint-394'). The base "
            "model and LoRA rank are read from the adapter config, so "
            "'--eval-model' is ignored."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="GPU memory utilization fraction for vLLM server (0.0-1.0).",
    )
    return parser.parse_args()


def load_data(split: str = "test") -> list[dict]:
    """Load evaluation examples for the given ``split``.

    Examples are loaded from the hardcoded ``FULL_MULTI_TOOL_FILES[split]``
    collection of curated data files and each item is tagged with a ``dataset``
    field (the source file stem).
    """
    items: list[dict] = []
    for file_path in FULL_MULTI_TOOL_FILES[split]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            file_items = json.load(f)
        for item in file_items:
            item["dataset"] = path.stem
        items.extend(file_items)
    return items


def resolve_checkpoint(path: str | Path) -> Path:
    """Validate and return a LoRA checkpoint directory.

    The directory must contain ``adapter_config.json``.
    """
    ckpt = Path(path)
    if not (ckpt / "adapter_config.json").exists():
        raise FileNotFoundError(f"No 'adapter_config.json' found in checkpoint directory: {ckpt}")
    return ckpt


def read_adapter_config(checkpoint_dir: Path) -> tuple[str, int]:
    """Return ``(base_model_name_or_path, lora_rank)`` from a checkpoint's adapter config."""
    with (checkpoint_dir / "adapter_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["base_model_name_or_path"], int(cfg.get("r", 16))


def _wait_for_server(host: str, port: int, timeout: float = 300, proc: subprocess.Popen | None = None) -> None:
    """Block until the vLLM server accepts connections or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"vLLM server did not start within {timeout}s")


@contextmanager
def vllm_server(
    model: str,
    host: str = VLLM_HOST,
    port: int = VLLM_PORT,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    extra_args: list[str] | None = None,
):
    """Start a vLLM server as a subprocess, yield when ready, and tear it down on exit."""
    try:
        with socket.create_connection((host, port), timeout=1):
            raise RuntimeError(f"Cannot start vLLM: {host}:{port} is already in use")
    except OSError:
        pass

    # The server downloads the model in a child process, so load credentials
    # before spawning it. An already exported token keeps precedence.
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Starting vLLM server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        _wait_for_server(host, port, proc=proc)
        logger.info("vLLM server ready on %s:%s", host, port)
        yield
    finally:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait()
        except ProcessLookupError:
            logger.warning("vLLM server process already terminated")
        logger.info("vLLM server stopped")


def evaluate_item(
    item: dict, *, eval_model: str, openai_client: OpenAI, seed: int, prompt_template: str | None = None
) -> tuple[dict, bool, bool | str, bool]:
    """Evaluate a single item against the model."""
    ground_truth = item["label"]["relevant"]
    user_input = build_user_prompt(item["request"], prompt_template=prompt_template)

    model_reasoning = "PARSE_ERROR"
    prediction: bool | str = "PARSE_ERROR"
    success = False
    raw_response = None
    parse_error: str | None = None

    try:
        raw_response = openai_client.responses.parse(
            model=eval_model,
            input=[{"role": "user", "content": user_input}],
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text_format=TaskToolMatcherOutput,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            extra_body={"top_k": DEFAULT_TOP_K, "seed": seed},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        structured_response = raw_response.output_parsed
        if structured_response is None:
            raise ValueError("Model returned no parsed response")
        model_reasoning = structured_response.reasoning
        prediction = structured_response.appropriate
        success = True
    except APIError:
        # A server or transport failure leaves the evaluation incomplete; it is
        # not a malformed model prediction and must not become a zero score.
        raise
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
        logger.error("Error on item %s: %s", item.get("id", "N/A"), e)
        logger.error("Raw response: %s", raw_response)

    raw_response_text: str | None = None
    if not success:
        if raw_response is not None:
            raw_response_text = str(raw_response)
        elif parse_error is not None:
            raw_response_text = f"NO_RAW_RESPONSE ({parse_error})"

    result = {
        "id": item.get("id", "N/A"),
        "task": item["request"]["task"],
        "requested_tool": item["request"]["tool"]["name"],
        "tool_description": item["request"]["tool"]["description"],
        "curated_groundtruth": ground_truth,
        "model_reasoning": model_reasoning,
        "model_prediction": prediction,
        "raw_response": raw_response_text,
        "match_tag": item["metadata"]["label"]["type"],
        "agreement_rate": item["metadata"]["label"]["agreement_rate"],
        "selected_reasoning": item["metadata"]["label"]["reasoning"],
    }
    if "dataset" in item:
        result["dataset"] = item["dataset"]
    return result, ground_truth, prediction, success


def run_evaluation(
    data: list[dict],
    *,
    request_model: str,
    openai_client: OpenAI,
    concurrency: int,
    seed: int,
    prompt_template: str | None = None,
) -> tuple[list[bool], list[bool], int, list[dict]]:
    """Evaluate ``data`` concurrently and return ordered results.

    Returns ``(y_true, y_pred, failed_parses, verbose_results)`` with the arrays
    built in input order so downstream metrics/CIs are deterministic regardless
    of request completion order.
    """
    # Buffer results by original input index so that downstream ordering is
    # deterministic regardless of completion order. The bootstrap CIs use a
    # fixed-seed RNG over array positions, so completion-order-dependent lists
    # would yield slightly different CI bounds run-to-run.
    buffer: list[tuple[dict, bool, bool | str, bool] | None] = [None] * len(data)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                evaluate_item,
                item,
                eval_model=request_model,
                openai_client=openai_client,
                seed=seed,
                prompt_template=prompt_template,
            ): i
            for i, item in enumerate(data)
        }
        for future in tqdm(as_completed(futures), total=len(data), desc=f"Evaluating {request_model}"):
            index = futures[future]
            try:
                buffer[index] = future.result()
            except APIError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    y_true: list[bool] = []
    y_pred: list[bool] = []
    failed_parses = 0
    verbose_results: list[dict] = []
    for entry in buffer:
        assert entry is not None
        result, ground_truth, prediction, success = entry
        if success:
            assert isinstance(prediction, bool)
            y_true.append(ground_truth)
            y_pred.append(prediction)
        else:
            failed_parses += 1
            logger.warning(
                "Failed parse for item %s. Raw response: %s",
                result["id"],
                result["raw_response"],
            )
        verbose_results.append(result)
    return y_true, y_pred, failed_parses, verbose_results


def compute_metrics_report(
    y_true: list[bool],
    y_pred: list[bool],
    *,
    failed_parses: int,
    request_model: str,
    split: str,
    checkpoint_dir: Path | None,
) -> dict:
    """Compute bootstrap metrics and log a formatted report.

    Returns the metrics dict from :func:`bootstrap_classification_metrics`.
    """
    metrics = bootstrap_classification_metrics(y_true, y_pred, failed_parses=failed_parses)
    print(f"Failed parses: {failed_parses}")
    print(f"Successful parses: {len(y_true)}")

    logger.info("\n" + "=" * 40)
    logger.info("METRICS REPORT: %s", request_model)
    logger.info("Split: %s", split)
    logger.info("Checkpoint: %s", str(checkpoint_dir) if checkpoint_dir else None)
    logger.info("=" * 40)

    def _fmt(stat: dict[str, float]) -> str:
        value = stat["value"]
        return f"{value:.4f} (+{stat['high'] - value:.4f} / -{value - stat['low']:.4f})"

    logger.info("Accuracy:            %s", _fmt(metrics["accuracy"]))
    logger.info("Precision:           %s", _fmt(metrics["precision"]))
    logger.info("Recall:              %s", _fmt(metrics["recall"]))
    logger.info("F1 Score:            %s", _fmt(metrics["f1_score"]))
    logger.info("End-to-End Accuracy: %s", _fmt(metrics["end_to_end_accuracy"]))
    logger.info("False Positive Rate: %s", _fmt(metrics["false_positive_rate"]))
    logger.info("False Negative Rate: %s", _fmt(metrics["false_negative_rate"]))

    logger.info("\nDetailed Classification Report:")
    if y_true:
        logger.info(
            "\n%s",
            classification_report(
                y_true, y_pred, labels=[False, True], target_names=["Not Appropriate", "Appropriate"], zero_division=0
            ),
        )
    else:
        logger.info("No successfully parsed predictions to classify")
    return metrics


def compute_metrics_and_save(
    *,
    y_true: list[bool],
    y_pred: list[bool],
    failed_parses: int,
    verbose_results: list[dict],
    request_model: str,
    split: str,
    input_files: list[str],
    total_examples: int,
    serve_model: str,
    checkpoint_dir: Path | None,
    output_dir: Path,
    prompt_template: str | None = None,
) -> Path:
    """Compute bootstrap metrics, log a report, and persist results to disk.

    Returns the path to the written results file.
    """
    metrics = compute_metrics_report(
        y_true,
        y_pred,
        failed_parses=failed_parses,
        request_model=request_model,
        split=split,
        checkpoint_dir=checkpoint_dir,
    )

    safe_model_name = re.sub(r"[:/]", "_", request_model)
    safe_model_name = re.sub(r"^models_", "", safe_model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{safe_model_name}_{timestamp}.json"
    failed_parses_file = output_dir / f"{safe_model_name}_{timestamp}_failed_parses.json"

    sys_prompt = build_user_prompt(
        item={
            "task": "TASK_PLACEHOLDER",
            "tool": {"name": "TOOL_NAME", "description": "TOOL_DESCRIPTION"},
        },
        prompt_template=prompt_template,
    )

    full_output = {
        "config": {
            "input_files": input_files,
            "split": split,
            "output_file": str(output_file),
            "failed_parses_output_file": str(failed_parses_file),
            "model": request_model,
            "checkpoint": str(checkpoint_dir) if checkpoint_dir else None,
            "base_model": serve_model if checkpoint_dir else None,
            "total_examples": total_examples,
            "failed_parses": failed_parses,
        },
        "sys_prompt": sys_prompt,
        "metrics": metrics,
        "results": verbose_results,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=4)

    failed_parse_records = [
        {
            "id": result.get("id"),
            "dataset": result.get("dataset"),
            "task": result.get("task"),
            "requested_tool": result.get("requested_tool"),
            "raw_response": result.get("raw_response"),
        }
        for result in verbose_results
        if result.get("model_prediction") == "PARSE_ERROR"
    ]
    with failed_parses_file.open("w", encoding="utf-8") as f:
        json.dump(failed_parse_records, f, indent=4)

    logger.info("Results saved to %s", output_file)
    logger.info("Failed parse raw responses saved to %s", failed_parses_file)
    return output_file


def main(
    eval_model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    seed: int = DEFAULT_SEED,
    checkpoint: str | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    split: str = "test",
    profile: ModelProfile = "1b",
    prompt_file: str | None = None,
) -> None:
    """Run the evaluation flow for the provided model."""
    # -- Data Loading --------------------------------------------------------
    input_files = FULL_MULTI_TOOL_FILES[split]
    data = load_data(split)

    # -- Resolve serving target ----------------------------------------------
    # Without a checkpoint, ``eval_model`` is served directly. With a LoRA
    # checkpoint, vLLM serves the adapter's base model and applies the adapter
    # at inference; the adapter (named after the checkpoint) is the model used
    # in requests and for output naming.
    serve_extra_args: list[str] | None = None
    checkpoint_dir: Path | None = None
    if checkpoint:
        checkpoint_dir = resolve_checkpoint(checkpoint)
        base_model, lora_rank = read_adapter_config(checkpoint_dir)
        serve_model = base_model
        request_model = f"{checkpoint_dir.parent.name}-{checkpoint_dir.name}"
        serve_extra_args = [
            "--enable-lora",
            "--max-lora-rank",
            str(lora_rank),
            "--lora-modules",
            f"{request_model}={checkpoint_dir.resolve()}",
        ]
        print(
            f"Starting evaluation on {len(data)} examples (split '{split}') using checkpoint "
            f"'{checkpoint_dir}' (base model '{base_model}') ..."
        )
    else:
        serve_model = eval_model
        request_model = eval_model
        print(f"Starting evaluation on {len(data)} examples (split '{split}') using model '{request_model}' ...")

    prompt_template = load_prompt(
        profile,
        Path(prompt_file) if prompt_file else None,
        checkpoint_dir or Path(eval_model),
    )

    # -- Start vLLM server & run evaluation ----------------------------------
    with vllm_server(serve_model, gpu_memory_utilization=gpu_memory_utilization, extra_args=serve_extra_args):
        openai_client = OpenAI(base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="token-abc123")

        start_time = datetime.now()
        y_true, y_pred, failed_parses, verbose_results = run_evaluation(
            data,
            request_model=request_model,
            openai_client=openai_client,
            concurrency=concurrency,
            seed=seed,
            prompt_template=prompt_template,
        )
        elapsed = datetime.now() - start_time
        print(f"\nEvaluation complete. Time elapsed: {elapsed}")

    # -- Metrics & Save ------------------------------------------------------
    compute_metrics_and_save(
        y_true=y_true,
        y_pred=y_pred,
        failed_parses=failed_parses,
        verbose_results=verbose_results,
        request_model=request_model,
        split=split,
        input_files=input_files,
        total_examples=len(data),
        serve_model=serve_model,
        checkpoint_dir=checkpoint_dir,
        output_dir=Path("outputs/evaluation") / split,
        prompt_template=prompt_template,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)
    args = parse_args()
    main(
        eval_model=args.eval_model or MODEL_IDS[args.profile],
        concurrency=args.concurrency,
        seed=args.seed,
        checkpoint=args.checkpoint,
        gpu_memory_utilization=args.gpu_memory_utilization,
        split=args.split,
        profile=args.profile,
        prompt_file=args.prompt_file,
    )
