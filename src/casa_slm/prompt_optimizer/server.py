"""Manage an OpenAI-compatible vLLM server in a child process."""

import contextlib
import logging
import multiprocessing
import os
import platform
import signal
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Iterator
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal, TextIO, cast

__all__ = [
    "VLLMServer",
]

type _StartupMessage = tuple[Literal["bound", "error"], str | None]

_HEALTH_POLL_INTERVAL_SECONDS = 0.1
_HEALTH_REQUEST_TIMEOUT_SECONDS = 1.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_KILL_TIMEOUT_SECONDS = 5.0

_LOGGER = logging.getLogger(__name__)


class _TeeStream:
    """Write a child process stream to the console and an optional log file."""

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
        return self._log_file.write(message)

    def flush(self) -> None:
        """Flush both the console and log file."""
        self._console_stream.flush()
        self._log_file.flush()

    def __getattr__(self, name: str) -> object:
        """Delegate unsupported stream operations to the original console stream."""
        return getattr(self._console_stream, name)


@contextlib.contextmanager
def _capture_process_output(log_path: str | None) -> Iterator[None]:
    """Copy a vLLM process's console output to the optimization log."""
    if log_path is None:
        yield
        return

    with Path(log_path).open("a", encoding="utf-8") as log_file:
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = cast(TextIO, _TeeStream(stdout, log_file))
        sys.stderr = cast(TextIO, _TeeStream(stderr, log_file))
        try:
            yield
        finally:
            sys.stdout = stdout
            sys.stderr = stderr


def _run_vllm_server(
    model_dir: str,
    served_model_name: str,
    port: int,
    max_model_len: int,
    max_batch_size: int,
    gpu_memory_utilization: float,
    startup_connection: Connection,
    log_path: str | None = None,
) -> None:
    """Run vLLM while copying its console output to the optimization log."""
    with _capture_process_output(log_path):
        _run_vllm_server_impl(
            model_dir,
            served_model_name,
            port,
            max_model_len,
            max_batch_size,
            gpu_memory_utilization,
            startup_connection,
        )


def _run_vllm_server_impl(
    model_dir: str,
    served_model_name: str,
    port: int,
    max_model_len: int,
    max_batch_size: int,
    gpu_memory_utilization: float,
    startup_connection: Connection,
) -> None:
    """Run vLLM's OpenAI-compatible API server in a new process group."""
    try:
        os.setsid()
        _LOGGER.info(
            "Launching vLLM API process for model %s as %s on port %d",
            model_dir,
            served_model_name,
            port,
        )

        import uvloop
        from vllm.entrypoints.openai.api_server import run_server_worker, setup_server
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description="vLLM OpenAI-compatible API server")
        parser = make_arg_parser(parser)
        args = parser.parse_args(
            [
                "--model",
                model_dir,
                "--served-model-name",
                served_model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--max-model-len",
                str(max_model_len),
                "--max-num-seqs",
                str(max_batch_size),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
            ]
        )
        validate_parsed_serve_args(args)

        listen_address, server_socket = setup_server(args)
        _LOGGER.info("vLLM API process bound to %s; initializing the engine", listen_address)
        startup_connection.send(("bound", None))
        uvloop.run(run_server_worker(listen_address, server_socket, args))
    except BaseException as error:
        _LOGGER.exception("vLLM API process failed")
        message = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            startup_connection.send(("error", message))
        raise
    finally:
        startup_connection.close()


def _is_running_in_wsl() -> bool:
    """Detects if the Python script is running under WSL (WSL1 or WSL2).

    Returns:
        True if running under WSL, False otherwise.
    """
    # 1. Check WSL-specific environment variables (Fast-path)
    # WSL automatically sets 'WSL_DISTRO_NAME' and 'WSL_INTEROP' in the environment.
    if any(var in os.environ for var in ("WSL_DISTRO_NAME", "WSL_INTEROP")):
        return True

    # 2. Check kernel release metadata via platform.release()
    # On WSL, this typically contains "microsoft" or "wsl" (e.g., "5.15.146.1-microsoft-standard-WSL2")
    release_info = platform.release().lower()
    if "microsoft" in release_info or "wsl" in release_info:
        return True

    # 3. Check /proc/version as a fallback (Linux-only)
    if platform.system() == "Linux":
        try:
            with open("/proc/version", "r") as f:
                if "wsl" in f.read().lower():
                    return True
        except (FileNotFoundError, PermissionError):
            pass

    return False


class VLLMServer:
    """Control one OpenAI-compatible vLLM server process."""

    def __init__(
        self,
        model_dir: str,
        *,
        served_model_name: str | None = None,
        port: int = 8080,
        max_model_len: int,
        max_batch_size: int,
        gpu_memory_utilization: float,
        startup_timeout: float = 600.0,
        apply_wsl_workaround: bool = False,
        log_path: Path | None = None,
    ) -> None:
        """Initialize a vLLM server controller.

        Args:
            model_dir: Local model directory or Hugging Face model identifier.
            served_model_name: Model ID exposed by the OpenAI-compatible API. Defaults
                to the final component of ``model_dir``.
            port: TCP port exposed by the OpenAI-compatible HTTP server.
            max_model_len: Maximum sequence length accepted by vLLM.
            max_batch_size: Maximum number of sequences scheduled concurrently.
            gpu_memory_utilization: Fraction of GPU memory available to vLLM.
            startup_timeout: Seconds to wait for the engine health check.
            apply_wsl_workaround: Whether to apply a WSL-specific workaround for port binding issues.
            log_path: File that receives a copy of the vLLM child process's console output.

        Raises:
            ValueError: If a configuration value is invalid.
        """
        if not model_dir.strip():
            raise ValueError("model_dir must not be empty")
        if served_model_name is not None and not served_model_name.strip():
            raise ValueError("served_model_name must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in the interval (0, 1]")
        if not startup_timeout > 0:
            raise ValueError("startup_timeout must be positive")

        self.port = port
        self.model_dir = model_dir
        self.served_model_name = served_model_name or Path(model_dir).name
        self.max_model_len = max_model_len
        self.max_batch_size = max_batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.startup_timeout = startup_timeout
        self.apply_wsl_workaround = apply_wsl_workaround
        self.log_path = log_path

        self._process: BaseProcess | None = None
        self._process_group_id: int | None = None
        self._startup_connection: Connection | None = None

    def start(self) -> None:
        """Start vLLM and wait until its engine reports healthy.

        Raises:
            RuntimeError: If the server is already running or startup fails.
            TimeoutError: If vLLM does not become healthy before the timeout.
        """
        self._reap_dead_process()
        if self._process is not None:
            raise RuntimeError("vLLM server is already running")

        _LOGGER.info(
            "Starting vLLM server for model %s as %s on port %d "
            "(max_model_len=%d, max_batch_size=%d, gpu_memory_utilization=%.2f)",
            self.model_dir,
            self.served_model_name,
            self.port,
            self.max_model_len,
            self.max_batch_size,
            self.gpu_memory_utilization,
        )
        launch_started_at = time.monotonic()
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_run_vllm_server,
            args=(
                self.model_dir,
                self.served_model_name,
                self.port,
                self.max_model_len,
                self.max_batch_size,
                self.gpu_memory_utilization,
                child_connection,
                str(self.log_path) if self.log_path is not None else None,
            ),
            name=f"vllm-server-{self.port}",
            daemon=False,
        )

        try:
            process.start()
        except BaseException:
            _LOGGER.exception("Failed to spawn the vLLM server process")
            child_connection.close()
            parent_connection.close()
            with contextlib.suppress(ValueError):
                process.close()
            raise

        child_connection.close()
        self._process = process
        self._process_group_id = process.pid
        self._startup_connection = parent_connection
        _LOGGER.info("Spawned vLLM server process with PID %d", process.pid)
        try:
            startup_duration = self._wait_until_ready(launch_started_at)
        except BaseException:
            _LOGGER.exception("vLLM server failed to become ready; stopping the child process")
            self._terminate_process()
            raise
        finally:
            self._close_startup_connection()

        _LOGGER.info(
            "vLLM server is ready at http://127.0.0.1:%d after %.2f seconds",
            self.port,
            startup_duration,
        )

    def stop(self) -> None:
        """Stop the vLLM process group, escalating to SIGKILL if necessary."""
        _LOGGER.info("Stopping vLLM server on port %d", self.port)
        self._terminate_process()

    def is_healthy(self) -> bool:
        """Return whether the managed vLLM server is running and healthy.

        Returns:
            True when the managed child is alive and its engine health endpoint
            responds with HTTP 200; otherwise, False.
        """
        process = self._process
        if process is None or not process.is_alive():
            return False
        return self._health_endpoint_is_healthy(_HEALTH_REQUEST_TIMEOUT_SECONDS)

    def _wait_until_ready(self, launch_started_at: float) -> float:
        """Wait for the child to bind its port and pass its health check."""
        process = self._require_process()
        connection = self._require_startup_connection()
        deadline = launch_started_at + self.startup_timeout
        port_bound = False
        connection_open = True

        while True:
            if connection_open:
                try:
                    while connection.poll():
                        event, detail = connection.recv()
                        if event == "error":
                            raise RuntimeError(f"vLLM server failed during startup:\n{detail}")
                        if event != "bound":
                            raise RuntimeError(f"vLLM server sent an unknown startup event: {event!r}")
                        port_bound = True
                        _LOGGER.info("vLLM server bound port %d; waiting for engine health", self.port)
                except EOFError:
                    connection_open = False

            if not process.is_alive():
                process.join()
                _LOGGER.error("vLLM server process exited during startup with code %s", process.exitcode)
                raise RuntimeError(f"vLLM server exited during startup with code {process.exitcode}")

            if _is_running_in_wsl() and self.apply_wsl_workaround:
                self._apply_wsl_workaround()

            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"vLLM server did not become healthy within {self.startup_timeout:g} seconds")

            if port_bound and self._health_endpoint_is_healthy(deadline - now):
                return now - launch_started_at

            time.sleep(min(_HEALTH_POLL_INTERVAL_SECONDS, deadline - now))

    def _health_endpoint_is_healthy(self, timeout: float) -> bool:
        """Return whether the vLLM engine health endpoint responds successfully."""
        timeout = min(_HEALTH_REQUEST_TIMEOUT_SECONDS, timeout)
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status == 200
        except (TimeoutError, urllib.error.URLError, OSError):
            return False

    def _reap_dead_process(self) -> None:
        """Release resources held by an exited child so the server can restart."""
        if self._process is None or self._process.is_alive():
            return
        self._process.join()
        self._clear_process()

    def _terminate_process(self) -> None:
        """Terminate the managed process and release its local resources."""
        process = self._process
        if process is None:
            self._close_startup_connection()
            return

        if not process.is_alive():
            process.join()
            self._clear_process()
            return

        self._signal_process_group(signal.SIGTERM)
        process.join(_SHUTDOWN_TIMEOUT_SECONDS)

        if process.is_alive() or self._process_group_exists():
            self._signal_process_group(signal.SIGKILL)
            process.join(_KILL_TIMEOUT_SECONDS)

        if process.is_alive():
            raise RuntimeError("vLLM server process could not be killed")

        self._clear_process()

    def _signal_process_group(self, signal_number: signal.Signals) -> None:
        """Signal the process group, falling back to the managed process."""
        process = self._require_process()
        if self._process_group_id is not None:
            try:
                os.killpg(self._process_group_id, signal_number)
                return
            except ProcessLookupError:
                pass

        if not process.is_alive():
            return
        if signal_number == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()

    def _process_group_exists(self) -> bool:
        """Return whether any processes remain in the managed process group."""
        if self._process_group_id is None:
            return False
        try:
            os.killpg(self._process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _clear_process(self) -> None:
        """Close process resources and reset lifecycle state."""
        process = self._process
        self._process = None
        self._process_group_id = None
        self._close_startup_connection()
        if process is not None:
            with contextlib.suppress(ValueError):
                process.close()

    def _close_startup_connection(self) -> None:
        """Close and clear the parent side of the startup pipe."""
        if self._startup_connection is not None:
            self._startup_connection.close()
            self._startup_connection = None

    def _require_process(self) -> BaseProcess:
        """Return the managed process or fail on an invalid internal state."""
        if self._process is None:
            raise RuntimeError("vLLM server process is not initialized")
        return self._process

    def _require_startup_connection(self) -> Connection:
        """Return the startup pipe or fail on an invalid internal state."""
        if self._startup_connection is None:
            raise RuntimeError("vLLM server startup connection is not initialized")
        return self._startup_connection

    def _apply_wsl_workaround(self) -> None:
        """Workaround for WSL port binding issues by starting a dummy HTTP server."""
        import socketserver
        from http.server import SimpleHTTPRequestHandler

        import psutil

        # Notice that it might take a while before vLLM is ready.
        # Delay applying the workaround until the port is bound.
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN and conn.laddr.port == self.port:
                _LOGGER.info(f"WSL hack: register/repair the port {self.port} state")
                try:
                    with socketserver.TCPServer(("", self.port), SimpleHTTPRequestHandler) as httpd:
                        _LOGGER.info("WSL hack: server started at port %d", self.port)
                        httpd.serve_forever()
                except Exception:
                    _LOGGER.info("WSL hack: Exception caught. Port should be ready to use.")
                    self.apply_wsl_workaround = False
