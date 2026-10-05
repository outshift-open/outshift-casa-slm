"""Check optimizer-model access through OpenAI and LiteLLM clients."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import litellm
from openai import OpenAI

from casa_slm.prompt_optimizer import EvaluationConfig, load_config

_DEFAULT_CONFIG_PATH = Path("./eval_config.yaml")
_DEFAULT_MAX_TOKENS = 128
_DEFAULT_TIMEOUT_SECONDS = 30.0
_CHECK_PROMPT = "This is a configuration check. Reply with exactly: optimizer model configuration OK"

type Check = tuple[str, Callable[[], str]]


def check_with_openai(
    config: EvaluationConfig,
    *,
    max_tokens: int,
    timeout: float,
) -> str:
    """Call the optimizer model directly through the OpenAI SDK."""
    client = OpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        max_retries=0,
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=config.models.optimizer_model.end_point,
        messages=[{"role": "user", "content": _CHECK_PROMPT}],
        max_completion_tokens=max_tokens,
    )
    return _response_content(response, "OpenAI SDK")


def check_with_litellm(
    config: EvaluationConfig,
    *,
    max_tokens: int,
    timeout: float,
) -> str:
    """Call the optimizer model through LiteLLM's OpenAI provider."""
    response = litellm.completion(
        model=_litellm_proxy_model_name(
            config.models.optimizer_model.end_point,
        ),
        api_key=config.LLM_API_KEY,
        api_base=config.LLM_BASE_URL,
        messages=[{"role": "user", "content": _CHECK_PROMPT}],
        max_tokens=max_tokens,
        num_retries=0,
        timeout=timeout,
    )
    return _response_content(response, "LiteLLM")


def run_checks(
    config: EvaluationConfig,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Run both client checks and report whether they all succeeded."""
    _validate_config(config)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    model = config.models.optimizer_model.end_point
    print(f"Checking optimizer model {model!r} at {config.LLM_BASE_URL!r}.")
    checks: list[Check] = [
        (
            "OpenAI SDK",
            lambda: check_with_openai(
                config,
                max_tokens=max_tokens,
                timeout=timeout,
            ),
        ),
        (
            "LiteLLM",
            lambda: check_with_litellm(
                config,
                max_tokens=max_tokens,
                timeout=timeout,
            ),
        ),
    ]

    failures = 0
    for name, check in checks:
        print(f"\n[{name}] Sending request...")
        try:
            content = check()
        except Exception as error:
            failures += 1
            print(
                f"[{name}] FAILED: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        else:
            print(f"[{name}] PASSED")
            print(f"[{name}] Response: {content}")

    if failures:
        print(
            f"\nConfiguration check failed for {failures} of {len(checks)} clients.",
            file=sys.stderr,
        )
        return False

    print("\nConfiguration check passed for both clients.")
    return True


def _validate_config(config: EvaluationConfig) -> None:
    """Validate fields required by both optimizer-model clients."""
    if not config.LLM_API_KEY.strip():
        raise ValueError(
            "LLM_API_KEY is missing; set it in the configuration or .env file",
        )
    if not config.LLM_BASE_URL.strip():
        raise ValueError(
            "LLM_BASE_URL is missing; set it in the configuration or .env file",
        )
    if not config.models.optimizer_model.end_point.strip():
        raise ValueError("models.optimizer_model.end_point must not be empty")


def _litellm_proxy_model_name(model: str) -> str:
    """Select LiteLLM's OpenAI provider while preserving the proxy model."""
    model = model.strip()
    if not model:
        raise ValueError("The optimizer model endpoint must not be empty")
    if model.startswith("openai/"):
        return model
    return f"openai/{model}"


def _response_content(response: object, client_name: str) -> str:
    """Extract a non-empty text response from either client."""
    try:
        choices = getattr(response, "choices")
        content: Any = choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"{client_name} returned an invalid completion response",
        ) from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{client_name} returned an empty response")
    return content.strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Validate optimizer-model configuration through the OpenAI SDK and LiteLLM."),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Configuration YAML path (default: eval_config.yaml).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULT_MAX_TOKENS,
        help=f"Maximum response tokens per check (default: {_DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help=(f"Request timeout in seconds (default: {_DEFAULT_TIMEOUT_SECONDS:g})."),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load configuration, run both checks, and return a process exit code."""
    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        succeeded = run_checks(
            config,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except Exception as error:
        print(
            f"Configuration check could not start: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
