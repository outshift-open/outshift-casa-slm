"""Regression tests for integration into the shared CASA repository."""

import pytest
from mcp.types import Resource, Tool

from casa_slm.data_generation import multitool_task_generation as mtg
from casa_slm.data_generation.mcp_servers.generation.extract_mcp_servers import McpServer


@pytest.fixture
def root_env(tmp_path, monkeypatch):
    """Create a repository-shaped fixture with isolated credential sources."""
    module = tmp_path / "src" / "casa_slm" / "data_generation" / "multitool_task_generation.py"
    module.parent.mkdir(parents=True)
    monkeypatch.setattr(mtg, "__file__", str(module))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.chdir(module.parent)
    return tmp_path / ".env"


def test_load_env_reads_repository_root_and_ignores_nested_dotenv(root_env):
    """Generation tasks use shared root credentials even from the pipeline folder."""
    root_env.write_text("LLM_API_KEY=root-key\nLLM_BASE_URL=https://root.example/v1\n")
    nested_env = root_env.parent / "src" / "casa_slm" / "data_generation" / ".env"
    nested_env.write_text("LLM_API_KEY=nested-key\nLLM_BASE_URL=https://nested.example/v1\n")

    assert mtg.load_env() == ("root-key", "https://root.example/v1")


def test_exported_credentials_override_root_dotenv(root_env, monkeypatch):
    """Shell and CI credentials take precedence over local configuration."""
    root_env.write_text("LLM_API_KEY=file-key\nLLM_BASE_URL=https://file.example/v1\n")
    monkeypatch.setenv("LLM_API_KEY", "exported-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://exported.example/v1")

    assert mtg.load_env() == ("exported-key", "https://exported.example/v1")


def test_exported_credentials_work_without_dotenv(root_env, monkeypatch):
    """Installed or automated runs need no local credentials file."""
    monkeypatch.setenv("LLM_API_KEY", "exported-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://exported.example/v1")

    assert mtg.load_env() == ("exported-key", "https://exported.example/v1")


def test_missing_key_reports_shared_setting(root_env, capsys):
    """A missing key fails before any API request with an actionable error."""
    with pytest.raises(SystemExit, match="1"):
        mtg.load_env()

    assert "LLM_API_KEY" in capsys.readouterr().out


def test_unset_endpoint_uses_public_default(root_env, monkeypatch):
    """The default URL is explicit, avoiding SDK-specific environment fallbacks."""
    monkeypatch.setenv("LLM_API_KEY", "exported-key")

    assert mtg.load_env() == ("exported-key", "https://api.openai.com/v1")


def test_mcp_export_serializes_sdk_models_without_external_zta_package():
    """Local exports keep the nested schema consumed by both generators."""
    server = McpServer(
        name="example-server",
        tools=[Tool(name="example-tool", inputSchema={"type": "object"})],
        resources=[Resource(name="example-resource", uri="https://example.com/resource")],
    )

    exported = server.model_dump(mode="json")

    assert exported["name"] == "example-server"
    assert exported["tools"][0]["inputSchema"] == {"type": "object"}
    assert exported["resources"][0]["uri"] == "https://example.com/resource"
