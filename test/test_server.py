# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the self-contained vLLM server controller."""

import asyncio
import builtins
import importlib
import logging
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest

import casa_slm.prompt_optimizer.server as server_module
from casa_slm.prompt_optimizer.server import VLLMServer


class _FakeConnection:
    """Small in-memory stand-in for a one-way multiprocessing connection."""

    def __init__(self, events: list[tuple[str, str | None]] | None = None) -> None:
        self.events = list(events or [])
        self.sent: list[tuple[str, str | None]] = []
        self.closed = False

    def poll(self) -> bool:
        """Return whether a queued event can be received."""
        return bool(self.events)

    def recv(self) -> tuple[str, str | None]:
        """Receive the next queued event."""
        return self.events.pop(0)

    def send(self, event: tuple[str, str | None]) -> None:
        """Record an event sent by the child entry point."""
        self.sent.append(event)

    def close(self) -> None:
        """Record that the connection was closed."""
        self.closed = True


class _FakeProcess:
    """Controllable stand-in for a spawned multiprocessing process."""

    def __init__(self, *, alive_after_start: bool = True, pid: int = 1234) -> None:
        self.alive_after_start = alive_after_start
        self.alive = False
        self.pid = pid
        self.exitcode: int | None = None
        self.started = False
        self.join_timeouts: list[float | None] = []
        self.closed = False
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self) -> None:
        """Mark the fake process as started."""
        self.started = True
        self.alive = self.alive_after_start
        if not self.alive:
            self.exitcode = 1

    def is_alive(self) -> bool:
        """Return the configured liveness state."""
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        """Record a join without changing liveness."""
        self.join_timeouts.append(timeout)

    def terminate(self) -> None:
        """Simulate terminating the process directly."""
        self.terminate_calls += 1
        self.alive = False
        self.exitcode = -signal.SIGTERM

    def kill(self) -> None:
        """Simulate killing the process directly."""
        self.kill_calls += 1
        self.alive = False
        self.exitcode = -signal.SIGKILL

    def close(self) -> None:
        """Record closure of multiprocessing resources."""
        self.closed = True


class _FakeContext:
    """Provide predetermined pipes and processes to the controller."""

    def __init__(
        self,
        processes: list[_FakeProcess],
        connections: list[tuple[_FakeConnection, _FakeConnection]],
    ) -> None:
        self.processes = processes
        self.connections = connections
        self.process_calls: list[dict[str, Any]] = []

    def Pipe(self, *, duplex: bool) -> tuple[_FakeConnection, _FakeConnection]:  # noqa: N802
        """Return the next parent and child connection pair."""
        assert duplex is False
        return self.connections.pop(0)

    def Process(self, **kwargs: Any) -> _FakeProcess:  # noqa: N802
        """Return the next configured child process."""
        self.process_calls.append(kwargs)
        return self.processes.pop(0)


def _make_server(*, startup_timeout: float = 600.0) -> VLLMServer:
    """Create a controller with representative valid settings."""
    return VLLMServer(
        "models/example",
        port=8080,
        max_model_len=2048,
        max_batch_size=32,
        gpu_memory_utilization=0.9,
        startup_timeout=startup_timeout,
    )


def _install_context(
    monkeypatch: pytest.MonkeyPatch,
    processes: list[_FakeProcess],
    parent_events: list[list[tuple[str, str | None]]],
) -> tuple[_FakeContext, list[_FakeConnection]]:
    """Install a fake spawn context and return its child pipe endpoints."""
    child_connections: list[_FakeConnection] = []
    connections: list[tuple[_FakeConnection, _FakeConnection]] = []
    for events in parent_events:
        child_connection = _FakeConnection()
        child_connections.append(child_connection)
        connections.append((_FakeConnection(events), child_connection))
    context = _FakeContext(processes, connections)
    monkeypatch.setattr(server_module.multiprocessing, "get_context", lambda method: context)
    return context, child_connections


def _stop_on_signal(process: _FakeProcess) -> Callable[[int, int], None]:
    """Create a process-group signal fake that stops on SIGTERM or SIGKILL."""

    def killpg(_process_group_id: int, signal_number: int) -> None:
        if signal_number == 0:
            if process.alive:
                return
            raise ProcessLookupError
        process.alive = False
        process.exitcode = -signal_number

    return killpg


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_dir": " "}, "model_dir"),
        ({"served_model_name": " "}, "served_model_name"),
        ({"port": 0}, "port"),
        ({"max_model_len": 0}, "max_model_len"),
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization"),
        ({"startup_timeout": 0.0}, "startup_timeout"),
    ],
)
def test_constructor_rejects_invalid_values(overrides: dict[str, Any], message: str) -> None:
    """Reject invalid configuration before creating any process."""
    arguments: dict[str, Any] = {
        "model_dir": "models/example",
        "port": 8080,
        "max_model_len": 2048,
        "max_batch_size": 32,
        "gpu_memory_utilization": 0.9,
        "startup_timeout": 600.0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        VLLMServer(**arguments)


def test_parent_module_does_not_import_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep vLLM imports out of the CPU-side parent module."""
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError(f"unexpected parent-side import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(server_module)


def test_capture_process_output_copies_child_streams_to_log(tmp_path: Path) -> None:
    """Persist vLLM child-process console output alongside the parent output."""
    log_path = tmp_path / "optimization.log"

    with server_module._capture_process_output(str(log_path)):
        print("child standard output")
        print("child standard error", file=sys.stderr)

    assert log_path.read_text(encoding="utf-8") == "child standard output\nchild standard error\n"


def test_start_waits_for_bound_and_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return from start only after the bound server passes its health check."""
    process = _FakeProcess()
    context, child_connections = _install_context(monkeypatch, [process], [[("bound", None)]])
    controller = _make_server()
    health_check = Mock(return_value=True)
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", health_check)

    controller.start()

    assert process.started
    assert child_connections[0].closed
    assert health_check.call_count == 1
    assert context.process_calls[0]["daemon"] is False
    assert context.process_calls[0]["target"] is server_module._run_vllm_server
    assert context.process_calls[0]["args"][:6] == ("models/example", "example", 8080, 2048, 32, 0.9)


def test_start_logs_launch_progress_and_duration(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log the launch configuration, readiness transition, and elapsed startup time."""
    process = _FakeProcess()
    _install_context(monkeypatch, [process], [[("bound", None)]])
    controller = _make_server()
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", lambda _timeout: True)

    with caplog.at_level(logging.INFO, logger=server_module.__name__):
        controller.start()

    messages = caplog.messages
    assert any(
        "Starting vLLM server for model models/example as example on port 8080" in message for message in messages
    )
    assert any("vLLM server bound port 8080" in message for message in messages)
    assert any("vLLM server is ready at http://127.0.0.1:8080 after" in message for message in messages)


def test_is_healthy_requires_a_live_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return false without a live child and otherwise delegate to the health endpoint."""
    controller = _make_server()
    health_check = Mock(return_value=True)
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", health_check)

    assert not controller.is_healthy()
    assert health_check.call_count == 0

    process = _FakeProcess()
    process.alive = True
    controller._process = process  # type: ignore[assignment]

    assert controller.is_healthy()
    health_check.assert_called_once_with(1.0)

    process.alive = False
    assert not controller.is_healthy()
    assert health_check.call_count == 1


def test_duplicate_start_fails_without_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a second start while the first child remains alive."""
    process = _FakeProcess()
    context, _ = _install_context(monkeypatch, [process], [[("bound", None)]])
    controller = _make_server()
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", lambda _timeout: True)
    controller.start()

    with pytest.raises(RuntimeError, match="already running"):
        controller.start()

    assert len(context.process_calls) == 1


def test_startup_error_is_relayed_and_cleaned_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise the child startup error and terminate the failed process."""
    process = _FakeProcess()
    _install_context(monkeypatch, [process], [[("error", "model failed")]])
    monkeypatch.setattr(server_module.os, "killpg", _stop_on_signal(process))
    controller = _make_server()

    with pytest.raises(RuntimeError, match="model failed"):
        controller.start()

    assert controller._process is None
    assert process.closed


def test_early_child_exit_becomes_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a child that exits before binding or becoming healthy."""
    process = _FakeProcess(alive_after_start=False)
    _install_context(monkeypatch, [process], [[]])
    controller = _make_server()

    with pytest.raises(RuntimeError, match="exited during startup with code 1"):
        controller.start()

    assert controller._process is None


def test_startup_timeout_terminates_partial_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean up a bound child if readiness exceeds the configured timeout."""
    process = _FakeProcess()
    _install_context(monkeypatch, [process], [[("bound", None)]])
    monkeypatch.setattr(server_module.time, "monotonic", Mock(side_effect=[0.0, 1.0]))
    monkeypatch.setattr(server_module.os, "killpg", _stop_on_signal(process))
    controller = _make_server(startup_timeout=0.5)

    with pytest.raises(TimeoutError, match=r"0\.5 seconds"):
        controller.start()

    assert controller._process is None


def test_stop_escalates_to_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Escalate from SIGTERM when the process does not exit during its grace period."""
    process = _FakeProcess()
    _install_context(monkeypatch, [process], [[("bound", None)]])
    controller = _make_server()
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", lambda _timeout: True)
    controller.start()
    received_signals: list[int] = []

    def killpg(_process_group_id: int, signal_number: int) -> None:
        received_signals.append(signal_number)
        if signal_number == signal.SIGKILL:
            process.alive = False
            process.exitcode = -signal.SIGKILL

    monkeypatch.setattr(server_module.os, "killpg", killpg)
    controller.stop()

    assert received_signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.join_timeouts[-2:] == [10.0, 5.0]
    assert process.closed
    assert controller._process is None


def test_stop_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow stopping before start and repeatedly after shutdown."""
    process = _FakeProcess()
    _install_context(monkeypatch, [process], [[("bound", None)]])
    controller = _make_server()
    controller.stop()
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", lambda _timeout: True)
    controller.start()
    monkeypatch.setattr(server_module.os, "killpg", _stop_on_signal(process))

    controller.stop()
    controller.stop()

    assert controller._process is None


def test_instance_can_restart_after_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a fresh child when start is called after a completed stop."""
    first_process = _FakeProcess(pid=1234)
    second_process = _FakeProcess(pid=5678)
    context, _ = _install_context(
        monkeypatch,
        [first_process, second_process],
        [[("bound", None)], [("bound", None)]],
    )
    controller = _make_server()
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", lambda _timeout: True)
    controller.start()
    monkeypatch.setattr(server_module.os, "killpg", _stop_on_signal(first_process))
    controller.stop()

    controller.start()

    assert len(context.process_calls) == 2
    assert controller._process is second_process


def test_instance_can_restart_after_unexpected_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reap an unexpectedly exited child before creating its replacement."""
    first_process = _FakeProcess(pid=1234)
    second_process = _FakeProcess(pid=5678)
    context, _ = _install_context(
        monkeypatch,
        [first_process, second_process],
        [[("bound", None)], [("bound", None)]],
    )
    controller = _make_server()
    monkeypatch.setattr(controller, "_health_endpoint_is_healthy", lambda _timeout: True)
    controller.start()
    first_process.alive = False
    first_process.exitcode = 1

    controller.start()

    assert first_process.closed
    assert len(context.process_calls) == 2
    assert controller._process is second_process


def test_child_entry_point_maps_vllm_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map controller settings to the corresponding vLLM server arguments."""
    parsed_arguments: list[str] = []
    parsed_namespace = object()
    worker_arguments: list[Any] = []

    class FakeParser:
        """Capture synthetic command-line arguments passed to vLLM's parser."""

        def __init__(self, *, description: str) -> None:
            assert description

        def parse_args(self, arguments: list[str]) -> object:
            parsed_arguments.extend(arguments)
            return parsed_namespace

    async def run_server_worker(listen_address: str, server_socket: object, args: object) -> None:
        worker_arguments.extend((listen_address, server_socket, args))

    api_server = ModuleType("vllm.entrypoints.openai.api_server")
    api_server.run_server_worker = run_server_worker  # type: ignore[attr-defined]
    api_server.setup_server = lambda args: ("http://0.0.0.0:9000", "socket")  # type: ignore[attr-defined]
    cli_args = ModuleType("vllm.entrypoints.openai.cli_args")
    cli_args.make_arg_parser = lambda parser: parser  # type: ignore[attr-defined]
    cli_args.validate_parsed_serve_args = lambda args: None  # type: ignore[attr-defined]
    argparse_utils = ModuleType("vllm.utils.argparse_utils")
    argparse_utils.FlexibleArgumentParser = FakeParser  # type: ignore[attr-defined]
    uvloop = ModuleType("uvloop")
    uvloop.run = asyncio.run  # type: ignore[attr-defined]

    fake_modules = {
        "uvloop": uvloop,
        "vllm": ModuleType("vllm"),
        "vllm.entrypoints": ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.openai": ModuleType("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.api_server": api_server,
        "vllm.entrypoints.openai.cli_args": cli_args,
        "vllm.utils": ModuleType("vllm.utils"),
        "vllm.utils.argparse_utils": argparse_utils,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(server_module.os, "setsid", Mock())
    connection = _FakeConnection()
    server_module._run_vllm_server("model-id", "baseline", 9000, 4096, 17, 0.75, connection)  # type: ignore[arg-type]

    assert parsed_arguments == [
        "--model",
        "model-id",
        "--served-model-name",
        "baseline",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "17",
        "--gpu-memory-utilization",
        "0.75",
    ]
    assert connection.sent == [("bound", None)]
    assert connection.closed
    assert worker_arguments == ["http://0.0.0.0:9000", "socket", parsed_namespace]
