# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the casa_slm.prompt_optimizer.utils module."""

import logging
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from casa_slm.prompt_optimizer.utils import TeeStream, capture_console_output, configure_logging


def test_configure_logging_replaces_existing_root_handlers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the optimization log handler to be installed even after prior setup."""
    basic_config = Mock()
    monkeypatch.setattr(logging, "basicConfig", basic_config)

    configure_logging(tmp_path / "optimization.log")

    kwargs = basic_config.call_args.kwargs
    assert kwargs["force"] is True
    assert isinstance(kwargs["handlers"][0], logging.FileHandler)
    for handler in kwargs["handlers"]:
        handler.close()


def test_capture_console_output_copies_standard_streams_to_optimization_log(tmp_path: Path) -> None:
    """Persist progress displays and other non-logging console output."""
    log_path = tmp_path / "optimization.log"

    with capture_console_output(log_path):
        print("standard output")
        print("standard error", file=sys.stderr)

    assert log_path.read_text(encoding="utf-8") == "standard output\nstandard error\n"


def test_tee_stream_keeps_ansi_colors_in_console_and_removes_them_from_log() -> None:
    """Keep terminal styling while writing plain text to the optimization log."""
    console = StringIO()
    log_file = StringIO()
    stream = TeeStream(console, log_file)
    message = "\x1b[31mcolored output\x1b[0m\n"

    assert stream.write(message) == len(message)

    assert console.getvalue() == message
    assert log_file.getvalue() == "colored output\n"
