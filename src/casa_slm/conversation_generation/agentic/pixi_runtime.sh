#!/usr/bin/env bash
# Shared by the launchers so paths and detached interpreters are deterministic.
AGENTIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$AGENTIC_DIR/../../../.." && pwd)"
cd "$AGENTIC_DIR" || exit 1

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
else
    PYTHON_BIN="$(pixi run --manifest-path "$REPOSITORY_ROOT/pyproject.toml" -e generation python -c 'import sys; print(sys.executable)')"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "No executable Python interpreter found for the generation environment." >&2
    exit 1
fi
export PYTHON_BIN
