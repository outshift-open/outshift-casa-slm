#!/usr/bin/env bash
set -euo pipefail

AGENTIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$AGENTIC_DIR/../../../.." && pwd)"
pixi install --manifest-path "$REPOSITORY_ROOT/pyproject.toml" -e generation
echo "Generation dependencies installed through the repository Pixi environment."
echo "Set LLM_API_KEY and LLM_BASE_URL in $REPOSITORY_ROOT/.env or export them."
