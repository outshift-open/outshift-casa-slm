"""Tests for the optimizer-model configuration checker."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import casa_slm.prompt_optimizer.scripts.check_llm as check_llm
from casa_slm.prompt_optimizer.config import EvaluationConfig


def _config() -> EvaluationConfig:
    """Return a complete configuration for client tests."""
    return EvaluationConfig.model_validate(
        {
            "experiment": {
                "name": "test-experiment",
                "dataset": [
                    {
                        "name": "test",
                        "train": ["train.json"],
                        "validation": ["validation.json"],
                        "test": ["test.json"],
                    }
                ],
            },
            "job": {"machine": "L4", "gpus": 1},
            "models": {
                "optimizer_model": {
                    "name": "reflection",
                    "end_point": "azure/gpt-5.4",
                },
                "task_model": {
                    "name": "baseline",
                    "hf_model_id": "org/model",
                    "path": "models",
                },
            },
            "vllm": {
                "port": 8123,
                "max_model_len": 2048,
                "max_batch_size": 8,
                "gpu_memory_utilization": 0.8,
            },
            "LLM_API_KEY": "secret",
            "LLM_BASE_URL": "https://llm.example.test/v1",
        }
    )


def _completion(content: str | None) -> SimpleNamespace:
    """Return an OpenAI-compatible completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_check_with_openai_uses_configured_proxy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the logical proxy model unchanged through the OpenAI SDK."""
    create = Mock(return_value=_completion("direct OK"))
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(check_llm, "OpenAI", client_factory)

    content = check_llm.check_with_openai(
        _config(),
        max_tokens=64,
        timeout=12.5,
    )

    assert content == "direct OK"
    assert client_factory.call_args.kwargs == {
        "api_key": "secret",
        "base_url": "https://llm.example.test/v1",
        "max_retries": 0,
        "timeout": 12.5,
    }
    assert create.call_args.kwargs["model"] == "azure/gpt-5.4"
    assert create.call_args.kwargs["max_completion_tokens"] == 64
    assert create.call_args.kwargs["messages"] == [{"role": "user", "content": check_llm._CHECK_PROMPT}]


def test_check_with_litellm_preserves_proxy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select LiteLLM's OpenAI provider without dropping `azure/`."""
    completion = Mock(return_value=_completion("litellm OK"))
    monkeypatch.setattr(check_llm.litellm, "completion", completion)

    content = check_llm.check_with_litellm(
        _config(),
        max_tokens=64,
        timeout=12.5,
    )

    assert content == "litellm OK"
    assert completion.call_args.kwargs == {
        "model": "openai/azure/gpt-5.4",
        "api_key": "secret",
        "api_base": "https://llm.example.test/v1",
        "messages": [{"role": "user", "content": check_llm._CHECK_PROMPT}],
        "max_tokens": 64,
        "num_retries": 0,
        "timeout": 12.5,
    }


def test_run_checks_attempts_both_clients_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report each client independently and fail if either request fails."""
    openai_check = Mock(side_effect=RuntimeError("direct failed"))
    litellm_check = Mock(return_value="litellm OK")
    monkeypatch.setattr(check_llm, "check_with_openai", openai_check)
    monkeypatch.setattr(check_llm, "check_with_litellm", litellm_check)

    succeeded = check_llm.run_checks(
        _config(),
        max_tokens=64,
        timeout=12.5,
    )

    assert succeeded is False
    openai_check.assert_called_once()
    litellm_check.assert_called_once()
    captured = capsys.readouterr()
    assert "[OpenAI SDK] FAILED: RuntimeError: direct failed" in captured.err
    assert "[LiteLLM] PASSED" in captured.out
    assert "failed for 1 of 2 clients" in captured.err


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("LLM_API_KEY", "LLM_API_KEY is missing"),
        ("LLM_BASE_URL", "LLM_BASE_URL is missing"),
    ],
)
def test_run_checks_rejects_missing_credentials(
    field: str,
    message: str,
) -> None:
    """Reject missing `.env` values before making requests."""
    config = _config().model_copy(update={field: ""})

    with pytest.raises(ValueError, match=message):
        check_llm.run_checks(config)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("azure/gpt-5.4", "openai/azure/gpt-5.4"),
        ("openai/azure/gpt-5.4", "openai/azure/gpt-5.4"),
        (
            "bedrock/global.anthropic.claude-sonnet-4-6",
            "openai/bedrock/global.anthropic.claude-sonnet-4-6",
        ),
    ],
)
def test_litellm_proxy_model_name(
    model: str,
    expected: str,
) -> None:
    """Preserve the proxy's full logical model identifier."""
    assert check_llm._litellm_proxy_model_name(model) == expected


def test_main_loads_requested_config_and_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pass CLI options to the loader and return a nonzero failed-check code."""
    config_path = tmp_path / "custom.yaml"
    load_config = Mock(return_value=_config())
    run_checks = Mock(return_value=False)
    monkeypatch.setattr(check_llm, "load_config", load_config)
    monkeypatch.setattr(check_llm, "run_checks", run_checks)

    exit_code = check_llm.main(
        [
            "--config",
            str(config_path),
            "--max-tokens",
            "32",
            "--timeout",
            "7.5",
        ]
    )

    assert exit_code == 1
    load_config.assert_called_once_with(config_path)
    run_checks.assert_called_once_with(
        load_config.return_value,
        max_tokens=32,
        timeout=7.5,
    )
