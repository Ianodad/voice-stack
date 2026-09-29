"""
Manage `mlx_lm.server` as a subprocess, serving the LLM over a local HTTP
API that OpenAILLMService talks to (see PLAN.md: "LLM out of process").

The subprocess runs in its own session (start_new_session=True) so it is not
in our process group; we are responsible for terminating it ourselves, in a
`finally` block covering normal exit, Ctrl-C, and crashes -- an orphaned
mlx_lm.server holds ~20GB of GPU memory.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "mlx_lm_server.log"


class MLXLMServer:
    """Starts, polls readiness for, and stops an `mlx_lm.server` subprocess."""

    def __init__(self, model_id: str, host: str = "127.0.0.1", port: int = 8080):
        self._model_id = model_id
        self._host = host
        self._port = port
        self._process: subprocess.Popen | None = None
        self._log_file = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    def _spawn_locked(self) -> None:
        if self._closed:
            raise RuntimeError("mlx_lm.server manager is stopped; refusing to spawn")
        self._close_log()
        log_file = open(LOG_PATH, "w")
        self._log_file = log_file
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "server",
                "--model",
                self._model_id,
                "--host",
                self._host,
                "--port",
                str(self._port),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def start(self, timeout: float = 60.0) -> None:
        with self._lock:
            self._spawn_locked()
        self._wait_ready(timeout)

    def restart(self, timeout: float = 60.0) -> None:
        """Atomically terminate + respawn; refuses if stop() was requested."""
        with self._lock:
            if self._closed:
                raise RuntimeError("mlx_lm.server manager is stopped; refusing to restart")
            self._terminate_locked()
            self._spawn_locked()
        self._wait_ready(timeout)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/models"
        while time.monotonic() < deadline:
            proc = self._process
            if self._closed or proc is None:
                raise RuntimeError("mlx_lm.server stopped while starting")
            if proc.poll() is not None:
                raise RuntimeError(
                    f"mlx_lm.server exited early (code={proc.returncode}); "
                    f"see {LOG_PATH}. A stale mlx_lm.server may already be holding "
                    f"port {self._port} -- try `pkill -f mlx_lm.server` and retry."
                )
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        with self._lock:
            self._terminate_locked()
        raise TimeoutError(
            f"mlx_lm.server did not become ready within {timeout}s; see {LOG_PATH}. "
            f"A stale mlx_lm.server may already be holding port {self._port} -- try "
            f"`pkill -f mlx_lm.server` and retry."
        )

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            finally:
                self._log_file = None

    def _terminate_locked(self, grace_period: float = 5.0) -> None:
        try:
            if self._process is None:
                return
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=grace_period)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=grace_period)
            self._process = None
        finally:
            self._close_log()

    def stop(self, grace_period: float = 5.0) -> None:
        """Permanently stop: terminate the subprocess (kill after a grace
        period) and refuse any later start()/restart(). Serialised with an
        in-flight spawn via the lock."""
        with self._lock:
            self._closed = True
            self._terminate_locked(grace_period)
