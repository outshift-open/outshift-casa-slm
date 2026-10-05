"""Common utility functions."""

import contextlib
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, TextIO, cast

from .config import EvaluationConfig

__all__ = [
    "TeeStream",
    "capture_console_output",
    "configure_logging",
    "create_run_directory",
]
_ANSI_ESCAPE_SEQUENCE = re.compile(r"\x1B(?:\[[0-?]*[ -/]*[@-~]|\][^\x1B\x07]*(?:\x07|\x1B\\)|[@-_])")


def create_run_directory(config: EvaluationConfig) -> Path:
    """Create a unique UTC-stamped artifact directory."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.gepa.output_dir / config.experiment.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_logging(log_path: Path) -> None:
    """Configure logging so every logging record is written to ``log_path``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(asctime)s:%(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


@contextlib.contextmanager
def capture_console_output(log_path: Path) -> Iterator[None]:
    """Copy standard output and error to the optimization log during a run."""
    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = cast(TextIO, TeeStream(stdout, log_file))
        sys.stderr = cast(TextIO, TeeStream(stderr, log_file))
        try:
            yield
        finally:
            sys.stdout = stdout
            sys.stderr = stderr


class TeeStream:
    """Write console output to its original stream and the optimization log."""

    def __init__(self, console_stream: TextIO, log_file: TextIO) -> None:
        """Initialize the stream proxy.

        Args:
            console_stream: Original standard stream shown to the user.
            log_file: Optimization log that receives a copy of the output.
        """
        self._console_stream = console_stream
        self._log_file = log_file

    def write(self, message: str) -> int:
        """Write a message to the console and log file."""
        self._console_stream.write(message)
        self._log_file.write(_ANSI_ESCAPE_SEQUENCE.sub("", message))
        return len(message)

    def flush(self) -> None:
        """Flush both the console and log file."""
        self._console_stream.flush()
        self._log_file.flush()

    def __getattr__(self, name: str) -> object:
        """Delegate unsupported stream operations to the original console stream."""
        return getattr(self._console_stream, name)
